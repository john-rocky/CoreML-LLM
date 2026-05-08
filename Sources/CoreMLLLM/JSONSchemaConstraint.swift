// Token-level decoding constraint that forces the Qwen3-VL Fashion FT
// to emit a structurally valid v3 fashion JSON. Hooks into the decode
// loop's argmax output: the constraint observes each emitted token and
// either passes it through (when it agrees with the schema) or replaces
// it with a schema-required token (when it would break structure).
//
// The CoreML head model bakes argmax into its graph and only exposes
// the chosen `next_token` (Int32), not raw logits. True logit masking
// would require reconverting the head; force-emit is the closest we
// can get without that. Trade-off: when the model emits a forbidden
// token, we substitute a state-deterministic default rather than the
// model's "second-best", because we can't see the rest of its
// distribution.
//
// Schema (v3 fashion, compact training format — `separators=(",", ":")`):
//
//     {"items":[
//        {"category":"<E>","description":"<S>",
//         "scores":{"color":<N>,"silhouette":<N>,"material":<N>,
//                   "design":<N>,"item_type":<N>},
//         "item_dress_score":<N>},
//        ...],
//      "overall_dress_ratio":<N>,
//      "coordinate_silhouette":{"type":"<E>","style_score":<N>,
//                               "rationale":"<S>"},
//      "target_ratio":0.7,
//      "verdict":"<S>",
//      "advice":"<S>"}
//
// E = enum string in quotes, S = free string in quotes, N = number 0-1.

import Foundation
import Tokenizers


/// Token-level decoding constraint applied just after model argmax.
public protocol TokenConstraint: AnyObject {
    /// Returns the token to actually emit. The model's argmax is the
    /// "input"; the constraint may pass it through, or substitute a
    /// schema-required token. The returned token is what feeds back
    /// into the next decode step (so KV state advances on the emitted
    /// token, not the model's preferred one).
    func nextToken(modelArgmax: Int32) -> Int32

    /// Reset state at the start of a new generate() call.
    func reset()

    /// True once the schema document is fully closed (no more output
    /// expected). Generators may use this to break out of the decode
    /// loop early.
    var isDone: Bool { get }
}


/// Hand-coded tokenization tables for the v3 fashion schema. Built
/// once at constraint init from a real tokenizer. Tests construct
/// directly with hand-crafted token IDs (avoids loading the real
/// vocabulary in unit tests).
///
/// Naming convention for the structural sequences:
///
///   * `enter_X`  — emitted BEFORE entering free state X. Includes the
///     opening `"` for free-string fields, or the opening `:`/colon
///     for free-number fields. Followed by free generation.
///   * `exitFreeString_X` — emitted AFTER a free-string state ends.
///     Starts with the closing `"` because free-string states leave
///     the closing quote to the constraint (the model emitted a
///     quote-containing token to signal close, but we substitute it).
///   * `exitEnum_X` — emitted AFTER an enum value ends. Does NOT
///     start with `"` because the enum value's own token sequence
///     already includes its closing quote.
///   * `exitFreeNumber_X` — emitted AFTER a free-number state ends.
///     Starts with `,` or `}` (number terminator).
public struct FashionTokens {
    // Single-byte ASCII tokens (verified single-token in Qwen3 BPE).
    public let openBrace: Int32       // {
    public let closeBrace: Int32      // }
    public let openBracket: Int32     // [
    public let closeBracket: Int32    // ]
    public let comma: Int32           // ,
    public let quote: Int32           // "
    /// Token for the literal digit "0". Used to force-emit a leading
    /// zero when the model would produce an empty / leading-dot
    /// number (`"color":,...` → `"color":0,...`), keeping the JSON
    /// numerically parseable.
    public let digitZero: Int32
    /// Tokens for digits "0".."9" only (no dots, no merges). Used to
    /// detect whether a number's body contains at least one digit:
    /// without one, JSONSerialization rejects the number ("0." is
    /// invalid, ".5" is invalid, empty is invalid).
    public let digitTokens: Set<Int32>

    /// Tokens whose decoded text is purely "0".."9" / "." (any
    /// concatenation thereof, non-empty). Free-number state accepts
    /// these as content; anything else terminates the number.
    public let numberContinuationTokens: Set<Int32>

    /// Tokens whose decoded text contains '"'. Free-string state
    /// treats one of these as the closing-quote signal. Includes
    /// the bare-quote token plus BPE merges like `","`, `",`, `":"`,
    /// `"}`, etc.
    public let quoteContainingTokens: Set<Int32>

    // Forced structural sequences. See struct comment for naming
    // convention; each is the literal substring emitted between
    // schema slots, encoded by the tokenizer at init.
    public let enter_documentItems: [Int32]      // {"items":[
    public let enter_itemFromArray: [Int32]      // {"category":"
    public let exitEnum_categoryToDescription: [Int32]  // ,"description":"
    public let exitFreeString_descriptionToScores: [Int32]  // ","scores":{"color":
    public let enter_silhouette: [Int32]         // ,"silhouette":
    public let enter_material: [Int32]           // ,"material":
    public let enter_design: [Int32]             // ,"design":
    public let enter_itemType: [Int32]           // ,"item_type":
    public let exitFreeNumber_itemTypeToItemDressScore: [Int32]  // },"item_dress_score":
    public let exitFreeNumber_itemDressScoreToItemClose: [Int32] // }
    public let exitItemArray_toOverall: [Int32]  // ,"overall_dress_ratio":  (NO leading `]` — `]` is emitted as the model's choice)
    public let enter_coordType: [Int32]          // ,"coordinate_silhouette":{"type":"
    public let exitEnum_typeToStyleScore: [Int32]  // ,"style_score":
    public let enter_rationale: [Int32]          // ,"rationale":"
    public let exitFreeString_rationaleToVerdict: [Int32]  // ","target_ratio":0.7,"verdict":"
    public let exitFreeString_verdictToAdvice: [Int32]   // ","advice":"
    public let exitFreeString_adviceCloseDocument: [Int32]  // "}

    /// Each entry: (display name, full token sequence INCLUDING the
    /// closing `"`). Model picks one by emitting the first token of an
    /// option; the rest is force-emitted. Closing `"` lives in the
    /// option so the post-enum sequence can start with `,`.
    public let categoryEnumOptions: [(name: String, tokens: [Int32])]
    public let typeEnumOptions: [(name: String, tokens: [Int32])]

    /// EOS token to emit once schema is closed.
    public let eosToken: Int32

    public init(
        openBrace: Int32, closeBrace: Int32,
        openBracket: Int32, closeBracket: Int32,
        comma: Int32, quote: Int32,
        digitZero: Int32,
        digitTokens: Set<Int32>,
        numberContinuationTokens: Set<Int32>,
        quoteContainingTokens: Set<Int32>,
        enter_documentItems: [Int32],
        enter_itemFromArray: [Int32],
        exitEnum_categoryToDescription: [Int32],
        exitFreeString_descriptionToScores: [Int32],
        enter_silhouette: [Int32],
        enter_material: [Int32],
        enter_design: [Int32],
        enter_itemType: [Int32],
        exitFreeNumber_itemTypeToItemDressScore: [Int32],
        exitFreeNumber_itemDressScoreToItemClose: [Int32],
        exitItemArray_toOverall: [Int32],
        enter_coordType: [Int32],
        exitEnum_typeToStyleScore: [Int32],
        enter_rationale: [Int32],
        exitFreeString_rationaleToVerdict: [Int32],
        exitFreeString_verdictToAdvice: [Int32],
        exitFreeString_adviceCloseDocument: [Int32],
        categoryEnumOptions: [(name: String, tokens: [Int32])],
        typeEnumOptions: [(name: String, tokens: [Int32])],
        eosToken: Int32
    ) {
        self.openBrace = openBrace
        self.closeBrace = closeBrace
        self.openBracket = openBracket
        self.closeBracket = closeBracket
        self.comma = comma
        self.quote = quote
        self.digitZero = digitZero
        self.digitTokens = digitTokens
        self.numberContinuationTokens = numberContinuationTokens
        self.quoteContainingTokens = quoteContainingTokens
        self.enter_documentItems = enter_documentItems
        self.enter_itemFromArray = enter_itemFromArray
        self.exitEnum_categoryToDescription = exitEnum_categoryToDescription
        self.exitFreeString_descriptionToScores =
            exitFreeString_descriptionToScores
        self.enter_silhouette = enter_silhouette
        self.enter_material = enter_material
        self.enter_design = enter_design
        self.enter_itemType = enter_itemType
        self.exitFreeNumber_itemTypeToItemDressScore =
            exitFreeNumber_itemTypeToItemDressScore
        self.exitFreeNumber_itemDressScoreToItemClose =
            exitFreeNumber_itemDressScoreToItemClose
        self.exitItemArray_toOverall = exitItemArray_toOverall
        self.enter_coordType = enter_coordType
        self.exitEnum_typeToStyleScore = exitEnum_typeToStyleScore
        self.enter_rationale = enter_rationale
        self.exitFreeString_rationaleToVerdict =
            exitFreeString_rationaleToVerdict
        self.exitFreeString_verdictToAdvice = exitFreeString_verdictToAdvice
        self.exitFreeString_adviceCloseDocument =
            exitFreeString_adviceCloseDocument
        self.categoryEnumOptions = categoryEnumOptions
        self.typeEnumOptions = typeEnumOptions
        self.eosToken = eosToken
    }

    /// Build by probing a real tokenizer. Encodes each schema-literal
    /// substring once and scans the full vocabulary to compute
    /// number-continuation and quote-containing token sets. Vocabulary
    /// scan is O(vocab) at init only, ~150k tokens for Qwen3 — well
    /// under a second on M-series.
    public static func probing(_ tok: any Tokenizer) -> FashionTokens {
        func enc(_ s: String) -> [Int32] {
            tok.encode(text: s).map { Int32($0) }
        }
        func single(_ s: String) -> Int32 {
            let ids = enc(s)
            precondition(ids.count == 1,
                "expected '\(s)' to tokenize as a single token, got \(ids)")
            return ids[0]
        }

        // Single-char punctuation (Qwen3 maps these to single tokens).
        let openBrace = single("{")
        let closeBrace = single("}")
        let openBracket = single("[")
        let closeBracket = single("]")
        let comma = single(",")
        let quote = single("\"")

        // Single-token digits: encode each "0".."9" and assert it's
        // a single token (true on Qwen3). digitZero is the canonical
        // "0" we force-emit to keep numbers JSON-valid.
        let digitZero = single("0")
        var digits: Set<Int32> = [digitZero]
        for d in 1...9 {
            digits.insert(single(String(d)))
        }

        // Vocab scan. Probe IDs 0..upper. `upper` is the larger of
        // 151643 (Qwen3 vocab) and eosTokenId+1, in case a future
        // tokenizer extends the range.
        let upper = max(151_643, (tok.eosTokenId ?? 0) + 1)
        var numberSet: Set<Int32> = []
        var quoteSet: Set<Int32> = []
        let digitDot: Set<Character> = [
            "0","1","2","3","4","5","6","7","8","9","."
        ]
        for id in 0..<upper {
            let s = tok.decode(tokens: [id])
            if s.isEmpty { continue }
            if s.contains("\"") { quoteSet.insert(Int32(id)) }
            if s.allSatisfy({ digitDot.contains($0) }) {
                numberSet.insert(Int32(id))
            }
        }

        // Enum option sequences INCLUDE closing `"` so the post-enum
        // forced sequence can start with `,`.
        let categoryNames = ["top", "bottom", "outer",
                              "shoes", "accessory", "headwear"]
        let typeNames = ["I", "A", "Y", "off"]
        let categoryEnum = categoryNames.map { name in
            (name: name, tokens: enc(name + "\""))
        }
        let typeEnum = typeNames.map { name in
            (name: name, tokens: enc(name + "\""))
        }

        return FashionTokens(
            openBrace: openBrace, closeBrace: closeBrace,
            openBracket: openBracket, closeBracket: closeBracket,
            comma: comma, quote: quote,
            digitZero: digitZero,
            digitTokens: digits,
            numberContinuationTokens: numberSet,
            quoteContainingTokens: quoteSet,
            enter_documentItems: enc("{\"items\":["),
            enter_itemFromArray: enc("{\"category\":\""),
            // Post-enum: enum value already closed with `"`, so this
            // starts with `,`.
            exitEnum_categoryToDescription: enc(",\"description\":\""),
            // Post-free-string: starts with closing `"`.
            exitFreeString_descriptionToScores:
                enc("\",\"scores\":{\"color\":"),
            enter_silhouette: enc(",\"silhouette\":"),
            enter_material: enc(",\"material\":"),
            enter_design: enc(",\"design\":"),
            enter_itemType: enc(",\"item_type\":"),
            // Post-free-number for item_type: starts with `}` to close
            // the scores object, then comma+key for item_dress_score.
            exitFreeNumber_itemTypeToItemDressScore:
                enc("},\"item_dress_score\":"),
            // Post-free-number for item_dress_score: just `}` to close
            // the item object. After this, model picks `,` (continue)
            // or `]` (end array).
            exitFreeNumber_itemDressScoreToItemClose: enc("}"),
            // Items array close: model emitted `]` as its choice; this
            // is everything AFTER the `]` (so starts with `,`).
            exitItemArray_toOverall: enc(",\"overall_dress_ratio\":"),
            enter_coordType:
                enc(",\"coordinate_silhouette\":{\"type\":\""),
            // Post-enum: enum value already closed with `"`, so this
            // starts with `,`.
            exitEnum_typeToStyleScore: enc(",\"style_score\":"),
            enter_rationale: enc(",\"rationale\":\""),
            // Post-free-string: starts with closing `"`. Includes the
            // pinned `target_ratio:0.7` constant since both endpoints
            // are deterministic. Closing `}` of coord_silhouette is
            // included.
            exitFreeString_rationaleToVerdict:
                enc("\"},\"target_ratio\":0.7,\"verdict\":\""),
            exitFreeString_verdictToAdvice: enc("\",\"advice\":\""),
            exitFreeString_adviceCloseDocument: enc("\"}"),
            categoryEnumOptions: categoryEnum,
            typeEnumOptions: typeEnum,
            eosToken: Int32(tok.eosTokenId ?? 151645))
    }
}


/// Force-emit constraint for the v3 fashion schema. Walks the JSON
/// document as a deterministic state machine. Three classes of
/// segments:
///
///   * Forced sequences (punctuation, key strings, structural
///     transitions) — emitted token-by-token regardless of model
///     argmax. The model's KV state still advances on each emitted
///     token, so we don't need to skip the chunk_head call.
///   * Free segments (description / verdict / advice / rationale
///     strings, all numbers) — model's argmax passes through
///     unmodified UNTIL it would close the segment. Closing
///     detection is by token-decoded-text content (quote presence
///     for strings, non-numeric for numbers).
///   * Choice segments (category enum, coordinate_silhouette type
///     enum, items array continue/end) — model picks one of the
///     allowed alternatives by emitting its first token; remainder
///     is force-emitted. If model emits a token outside the
///     alternatives, the first option is picked as deterministic
///     fallback.
public final class FashionV3Constraint: TokenConstraint {

    // MARK: - State

    /// Current position in the schema. Each phase says "what the next
    /// model argmax means" when `pendingForced` is empty. While
    /// `pendingForced` is non-empty, the phase represents what we'll
    /// be in once the queue drains.
    enum Phase: Equatable {
        case start                    // forced sequence currently draining
        case insideItemPickCategory   // ENUM pick
        case insideItemDescription    // FREE_STRING
        case insideItemScoreColor     // FREE_NUMBER
        case insideItemScoreSilhouette
        case insideItemScoreMaterial
        case insideItemScoreDesign
        case insideItemScoreItemType
        case insideItemDressScore     // FREE_NUMBER
        case afterItemClose           // CHOICE: continue items or close
        case overallDressRatio        // FREE_NUMBER
        case coordType                // ENUM pick
        case coordStyleScore          // FREE_NUMBER
        case coordRationale           // FREE_STRING
        case verdictString            // FREE_STRING
        case adviceString             // FREE_STRING
        case done
    }

    private let tokens: FashionTokens
    private(set) var phase: Phase = .start
    private var pendingForced: [Int32] = []
    public private(set) var isDone: Bool = false

    /// Whether the current free-number state has emitted at least one
    /// digit. Used by `freeNumber` to force-emit "0" when the model
    /// would otherwise produce an empty / leading-dot value
    /// (`"color":,...`), which JSONSerialization rejects.
    /// Reset to false whenever a free-number phase ends or starts;
    /// see `freeNumber` for the entry/exit transitions.
    private var emittedDigitInCurrentNumber: Bool = false

    /// True when the constraint has drained all pending forced tokens
    /// and is waiting on the model's argmax to drive the next
    /// emission. Tests use this to dispatch by phase only on
    /// meaningful steps; in production the value is irrelevant
    /// because every step feeds the real argmax (force-emitted steps
    /// just ignore it).
    var isAwaitingModel: Bool { pendingForced.isEmpty }

    public init(tokens: FashionTokens) {
        self.tokens = tokens
        // Prime: opening structural sequence is the first thing to
        // emit. Fill pendingForced now so step 0 returns the first
        // structural token regardless of what the model argmaxes.
        self.pendingForced = tokens.enter_documentItems
        self.phase = .start
    }

    // MARK: - Public API

    public func reset() {
        phase = .start
        pendingForced = tokens.enter_documentItems
        isDone = false
        emittedDigitInCurrentNumber = false
    }

    public func nextToken(modelArgmax: Int32) -> Int32 {
        if isDone { return tokens.eosToken }

        // Drain any queued forced tokens. When the queue empties on
        // this step, advance the phase via `advanceAfterDrain` so the
        // NEXT call evaluates the right state.
        if !pendingForced.isEmpty {
            let t = pendingForced.removeFirst()
            if pendingForced.isEmpty {
                advanceAfterDrain()
            }
            return t
        }

        // Queue empty → phase decides what to do with model's argmax.
        switch phase {
        case .start:
            // Unreachable in normal flow (init always primes
            // pendingForced). Recover by re-priming.
            pendingForced = tokens.enter_documentItems
            return nextToken(modelArgmax: modelArgmax)

        case .insideItemPickCategory:
            return pickEnum(model: modelArgmax,
                             options: tokens.categoryEnumOptions,
                             postEnum: tokens.exitEnum_categoryToDescription,
                             nextPhase: .insideItemDescription)

        case .insideItemDescription:
            return freeString(model: modelArgmax,
                               postSequence:
                                tokens.exitFreeString_descriptionToScores,
                               nextPhase: .insideItemScoreColor)

        case .insideItemScoreColor:
            return freeNumber(model: modelArgmax,
                               postSequence: tokens.enter_silhouette,
                               nextPhase: .insideItemScoreSilhouette)

        case .insideItemScoreSilhouette:
            return freeNumber(model: modelArgmax,
                               postSequence: tokens.enter_material,
                               nextPhase: .insideItemScoreMaterial)

        case .insideItemScoreMaterial:
            return freeNumber(model: modelArgmax,
                               postSequence: tokens.enter_design,
                               nextPhase: .insideItemScoreDesign)

        case .insideItemScoreDesign:
            return freeNumber(model: modelArgmax,
                               postSequence: tokens.enter_itemType,
                               nextPhase: .insideItemScoreItemType)

        case .insideItemScoreItemType:
            return freeNumber(model: modelArgmax,
                               postSequence:
                                tokens.exitFreeNumber_itemTypeToItemDressScore,
                               nextPhase: .insideItemDressScore)

        case .insideItemDressScore:
            return freeNumber(model: modelArgmax,
                               postSequence:
                                tokens.exitFreeNumber_itemDressScoreToItemClose,
                               nextPhase: .afterItemClose)

        case .afterItemClose:
            return decideItemContinuation(model: modelArgmax)

        case .overallDressRatio:
            return freeNumber(model: modelArgmax,
                               postSequence: tokens.enter_coordType,
                               nextPhase: .coordType)

        case .coordType:
            return pickEnum(model: modelArgmax,
                             options: tokens.typeEnumOptions,
                             postEnum: tokens.exitEnum_typeToStyleScore,
                             nextPhase: .coordStyleScore)

        case .coordStyleScore:
            return freeNumber(model: modelArgmax,
                               postSequence: tokens.enter_rationale,
                               nextPhase: .coordRationale)

        case .coordRationale:
            return freeString(model: modelArgmax,
                               postSequence:
                                tokens.exitFreeString_rationaleToVerdict,
                               nextPhase: .verdictString)

        case .verdictString:
            return freeString(model: modelArgmax,
                               postSequence:
                                tokens.exitFreeString_verdictToAdvice,
                               nextPhase: .adviceString)

        case .adviceString:
            return freeString(model: modelArgmax,
                               postSequence:
                                tokens.exitFreeString_adviceCloseDocument,
                               nextPhase: .done)

        case .done:
            isDone = true
            return tokens.eosToken
        }
    }

    // MARK: - State-transition helpers

    /// Called when `pendingForced` has just drained its last token.
    /// Most phase transitions set both `pendingForced` and `phase`
    /// together via the `freeString`/`freeNumber`/`pickEnum`/
    /// `decideItemContinuation` methods, so this hook only needs to
    /// handle the entry sequences that don't have a corresponding
    /// "consume model" phase: the document opener and each new
    /// item's opener.
    private func advanceAfterDrain() {
        switch phase {
        case .start:
            // Drained `{"items":[`. Queue the first item's opener
            // (`{"category":"`). Once that drains, model picks the
            // category enum.
            pendingForced = tokens.enter_itemFromArray
            phase = .insideItemPickCategory
        case .done:
            // Drained `"}` — document is fully closed.
            isDone = true
        default:
            // No-op. The phase is set correctly by whatever queued
            // this forced sequence; once it drains, the next call
            // evaluates that phase.
            break
        }
    }

    /// Free-string segment: pass model argmax through until it emits a
    /// quote-containing token (close-quote signal). On close, ignore
    /// model's emitted token and substitute the post-string forced
    /// sequence (which starts with the closing `"`). Substitution
    /// preserves structural validity — the BPE-merged forced sequence
    /// like `","scores":{"color":` overwrites whatever the model
    /// intended to emit after the quote.
    private func freeString(model: Int32,
                              postSequence: [Int32],
                              nextPhase: Phase) -> Int32 {
        if tokens.quoteContainingTokens.contains(model) {
            // Model wants to close. Substitute with our forced
            // post-string transition, which begins with the closing
            // quote.
            let head = postSequence.first ?? tokens.quote
            pendingForced = Array(postSequence.dropFirst())
            phase = nextPhase
            // When closing the document's last free-string (advice),
            // the post-sequence is just `"}` — a single token. Once
            // that token is emitted (i.e., now), the document is done.
            // Set isDone immediately so the decode loop can break
            // before wasting another stepPredict call.
            if phase == .done && pendingForced.isEmpty {
                isDone = true
            }
            return head
        }
        // Content token — pass through, stay in current phase.
        return model
    }

    /// Free-number segment: pass model argmax through if it's pure
    /// digits/dots; otherwise treat as terminator and substitute the
    /// post-number forced sequence (which starts with `,` or `}`).
    ///
    /// Digit-protection: JSONSerialization rejects empty numbers
    /// (`"color":,...`) and leading-dot floats (`"color":.5,...`).
    /// If the model tries to terminate or emit a dot before any
    /// digit lands in this number's body, we force-emit `0` first
    /// and STAY in the number phase. The model then sees `0` as the
    /// previous token and continues naturally — most often with a
    /// `,` (now harmless: we have a digit) or another digit.
    private func freeNumber(model: Int32,
                              postSequence: [Int32],
                              nextPhase: Phase) -> Int32 {
        // Track digit emission within the current number's body.
        if tokens.digitTokens.contains(model) {
            emittedDigitInCurrentNumber = true
            return model
        }
        if tokens.numberContinuationTokens.contains(model) {
            // Non-digit numeric content (a bare `.` or BPE-merged
            // dot-token). If we've already seen a digit, accept
            // (`0.5` is fine). If not, force-emit `0` first; stay
            // in the number phase so the model's dot/digit follows
            // naturally on the next call.
            if emittedDigitInCurrentNumber {
                return model
            }
            emittedDigitInCurrentNumber = true
            return tokens.digitZero
        }
        // Non-numeric token = terminator. If no digit has landed,
        // force-emit `0` first and stay in the number phase. The
        // next call will see the model's argmax for "after 0",
        // which is normally a digit / dot / terminator — all of
        // which freeNumber handles correctly with emittedDigitIn-
        // CurrentNumber now true.
        if !emittedDigitInCurrentNumber {
            emittedDigitInCurrentNumber = true
            return tokens.digitZero
        }
        // Normal terminator: queue the post-number forced sequence,
        // advance to the next phase, reset the digit flag for the
        // upcoming number (or non-number) state.
        let head = postSequence.first ?? tokens.comma
        pendingForced = Array(postSequence.dropFirst())
        phase = nextPhase
        emittedDigitInCurrentNumber = false
        return head
    }

    /// Enum pick: each option is a token sequence INCLUDING its
    /// closing `"`. If model's argmax matches the first token of any
    /// option, commit to that option (queue the rest plus the
    /// post-enum sequence). Otherwise fallback to the first option
    /// for determinism.
    private func pickEnum(model: Int32,
                            options: [(name: String, tokens: [Int32])],
                            postEnum: [Int32],
                            nextPhase: Phase) -> Int32 {
        let chosen: [Int32]
        if let opt = options.first(where: { $0.tokens.first == model }) {
            chosen = opt.tokens
        } else {
            chosen = options[0].tokens
        }
        // Queue: rest of enum value + post-enum forced sequence.
        // When this queue drains, we'll be in `nextPhase` ready to
        // consume the next model argmax.
        var queue = Array(chosen.dropFirst())
        queue.append(contentsOf: postEnum)
        pendingForced = queue
        phase = nextPhase
        return chosen[0]
    }

    /// At `.afterItemClose` (item's `}` was just emitted): model picks
    /// `,` (continue items) or `]` (end array). We honor whichever
    /// the model wants, with `]` as deterministic fallback so the
    /// document eventually terminates.
    private func decideItemContinuation(model: Int32) -> Int32 {
        if model == tokens.comma {
            // Continue: emit `,` (model's choice) and queue the next
            // item's `{"category":"` opener.
            pendingForced = tokens.enter_itemFromArray
            phase = .insideItemPickCategory
            return tokens.comma
        }
        // End items array — emit `]` and queue post-array sequence
        // `,"overall_dress_ratio":`. After draining, free-number for
        // overall_dress_ratio.
        pendingForced = tokens.exitItemArray_toOverall
        phase = .overallDressRatio
        return tokens.closeBracket
    }
}
