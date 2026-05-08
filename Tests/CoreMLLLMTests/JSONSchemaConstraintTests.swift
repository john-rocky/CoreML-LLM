import XCTest
import Tokenizers
@testable import CoreMLLLM

/// Unit tests for the v3 fashion JSON schema constraint. Uses the real
/// Qwen3-VL-2B-Instruct tokenizer (cached at `~/.cache/huggingface/`)
/// because the constraint's behavior hinges on BPE merges like `","`,
/// `":"`, `},"` — a hand-rolled fake tokenizer would either match
/// these exactly (and add no signal) or diverge (and miss bugs).
///
/// Tokenizer load + vocab scan runs once in `setUp()` and the cached
/// `FashionTokens` is reused across tests.
///
/// Tests may be skipped if the tokenizer cache is missing AND there's
/// no network — the swift-transformers `AutoTokenizer.from(pretrained:)`
/// fetches on first run, so first execution after `~/.cache/huggingface`
/// is wiped takes a few seconds.
final class JSONSchemaConstraintTests: XCTestCase {

    nonisolated(unsafe) private static var cachedTokens: FashionTokens?
    nonisolated(unsafe) private static var cachedTokenizer: (any Tokenizer)?

    override class func setUp() {
        super.setUp()
        // Probe once for the whole test class. Vocab scan over ~150k
        // tokens is the expensive bit — repeating it per test would
        // dominate runtime.
        let sema = DispatchSemaphore(value: 0)
        Task {
            do {
                let tok = try await AutoTokenizer.from(
                    pretrained: "Qwen/Qwen3-VL-2B-Instruct")
                cachedTokenizer = tok
                cachedTokens = FashionTokens.probing(tok)
            } catch {
                print("[JSONSchemaConstraintTests] tokenizer load failed: \(error)")
            }
            sema.signal()
        }
        sema.wait()
    }

    private func tokens() throws -> FashionTokens {
        guard let t = Self.cachedTokens else {
            throw XCTSkip("Qwen3-VL tokenizer not available; "
                          + "ensure ~/.cache/huggingface is populated or "
                          + "network is reachable on first run.")
        }
        return t
    }

    private func tokenizer() throws -> any Tokenizer {
        guard let t = Self.cachedTokenizer else {
            throw XCTSkip("Qwen3-VL tokenizer not available")
        }
        return t
    }

    // MARK: - Phase-aware driver

    /// Drive the constraint forward until done (or step cap), feeding
    /// reasonable model argmax values per phase. The driver dispatches
    /// only when the constraint is awaiting model input; while it's
    /// draining a forced sequence, we feed an inert sentinel that the
    /// constraint ignores. Returns the full emitted token sequence.
    @discardableResult
    private func driveToCompletion(_ c: FashionV3Constraint,
                                     toks: FashionTokens,
                                     tok: any Tokenizer,
                                     stopAtPhase: FashionV3Constraint.Phase? = nil,
                                     stepCap: Int = 2048) -> [Int32] {
        let zero: Int32 = 15
        let closeMerge = tok.encode(text: "\",\"").map { Int32($0) }[0]
        let plainQuoteClose = toks.quote
        var emittedDigitInCurrentNumber = false
        var prevPhase = c.phase
        var emitted: [Int32] = []
        var safety = 0
        while !c.isDone && safety < stepCap {
            if let stop = stopAtPhase, c.phase == stop, c.isAwaitingModel {
                break
            }
            // Reset the digit-flag whenever phase advances.
            if c.phase != prevPhase {
                emittedDigitInCurrentNumber = false
                prevPhase = c.phase
            }
            let arg: Int32
            if !c.isAwaitingModel {
                arg = 0  // ignored; constraint drains its queue
            } else {
                switch c.phase {
                case .start, .done:
                    arg = 0
                case .insideItemPickCategory:
                    arg = toks.categoryEnumOptions[0].tokens[0]
                case .insideItemDescription, .coordRationale,
                     .verdictString:
                    arg = closeMerge
                case .adviceString:
                    // Last free-string of the document. Use the bare
                    // close-quote so the post-sequence `"}` gets fully
                    // force-emitted (closeMerge would still trigger
                    // close, but bare `"` exercises the simpler path).
                    arg = plainQuoteClose
                case .insideItemScoreColor, .insideItemScoreSilhouette,
                     .insideItemScoreMaterial, .insideItemScoreDesign,
                     .insideItemScoreItemType, .insideItemDressScore,
                     .overallDressRatio, .coordStyleScore:
                    if emittedDigitInCurrentNumber {
                        arg = toks.comma
                        emittedDigitInCurrentNumber = false
                    } else {
                        arg = zero
                        emittedDigitInCurrentNumber = true
                    }
                case .afterItemClose:
                    arg = toks.closeBracket
                case .coordType:
                    arg = toks.typeEnumOptions[0].tokens[0]
                }
            }
            emitted.append(c.nextToken(modelArgmax: arg))
            safety += 1
        }
        return emitted
    }

    // MARK: - Tests

    /// Constraint emits the full opening `{"items":[{"category":"`
    /// before consuming any model argmax for content. The first ~8
    /// emitted tokens are pure structural force-emit.
    func testStructuralOpeningIsForceEmitted() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        // Feed garbage — constraint ignores model argmax until the
        // opener fully drains.
        var emitted: [Int32] = []
        let openingLen = toks.enter_documentItems.count
            + toks.enter_itemFromArray.count
        for _ in 0..<openingLen {
            emitted.append(c.nextToken(modelArgmax: 99999))
        }

        let text = tok.decode(tokens: emitted.map { Int($0) })
        XCTAssertEqual(text, "{\"items\":[{\"category\":\"",
            "expected force-emitted opening, got: \(text)")
        XCTAssertTrue(c.isAwaitingModel,
            "should now be awaiting model argmax for category enum pick")
        XCTAssertEqual(c.phase, .insideItemPickCategory)
    }

    /// After the opening, model picks a category. If model emits the
    /// first token of "shoes" (multi-token enum value), the constraint
    /// commits to that option and force-emits the rest INCLUDING the
    /// closing quote.
    func testCategoryEnumPickShoes() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        // Drain opening.
        let openingLen = toks.enter_documentItems.count
            + toks.enter_itemFromArray.count
        for _ in 0..<openingLen { _ = c.nextToken(modelArgmax: 0) }

        let shoesTokens = tok.encode(text: "shoes\"").map { Int32($0) }
        let firstShoesToken = shoesTokens.first!

        var emitted: [Int32] = [c.nextToken(modelArgmax: firstShoesToken)]
        XCTAssertEqual(emitted[0], firstShoesToken,
            "should accept first token of an enum option")

        // Drain rest of `shoes"` + the post-enum forced sequence.
        let drainCount = (shoesTokens.count - 1)
            + toks.exitEnum_categoryToDescription.count
        for _ in 0..<drainCount {
            emitted.append(c.nextToken(modelArgmax: 0))
        }
        let text = tok.decode(tokens: emitted.map { Int($0) })
        XCTAssertEqual(text, "shoes\",\"description\":\"",
            "expected `shoes\",\"description\":\"`, got: \(text)")
    }

    /// Model emits a totally invalid first-token for the category
    /// (e.g., a digit). The constraint substitutes with the FIRST
    /// enum option ("top") for determinism.
    func testCategoryEnumFallbackOnInvalidPick() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)

        let openingLen = toks.enter_documentItems.count
            + toks.enter_itemFromArray.count
        for _ in 0..<openingLen { _ = c.nextToken(modelArgmax: 0) }

        // Model emits "0" (digit), not a valid category.
        let returned = c.nextToken(modelArgmax: 15)
        let topTokens = toks.categoryEnumOptions[0].tokens
        XCTAssertEqual(returned, topTokens[0],
            "expected fallback to first enum option ('top'), got \(returned)")
    }

    /// Free-string state passes content tokens through unchanged.
    func testFreeStringPassesContentThrough() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        // Walk to .insideItemDescription via valid opening + "top".
        let openingLen = toks.enter_documentItems.count
            + toks.enter_itemFromArray.count
        for _ in 0..<openingLen { _ = c.nextToken(modelArgmax: 0) }
        for arg in toks.categoryEnumOptions[0].tokens {
            _ = c.nextToken(modelArgmax: arg)
        }
        for _ in 0..<toks.exitEnum_categoryToDescription.count {
            _ = c.nextToken(modelArgmax: 0)
        }
        XCTAssertEqual(c.phase, .insideItemDescription)
        XCTAssertTrue(c.isAwaitingModel)

        // Feed Japanese content tokens; each should pass through.
        let jpContent = tok.encode(text: "ホワイトデニム")
            .map { Int32($0) }
        for arg in jpContent {
            XCTAssertFalse(toks.quoteContainingTokens.contains(arg),
                "test prerequisite: '\(arg)' must not contain '\"'")
            let returned = c.nextToken(modelArgmax: arg)
            XCTAssertEqual(returned, arg,
                "free-string state should pass content through unchanged")
        }
        XCTAssertEqual(c.phase, .insideItemDescription)
    }

    /// Free-string state closes when model emits a quote-containing
    /// BPE-merged token (e.g., `","`). After close, the post-string
    /// forced sequence drains starting from the closing `"`.
    func testFreeStringCloseTransitions() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        // Walk to .insideItemDescription.
        let openingLen = toks.enter_documentItems.count
            + toks.enter_itemFromArray.count
        for _ in 0..<openingLen { _ = c.nextToken(modelArgmax: 0) }
        for arg in toks.categoryEnumOptions[0].tokens {
            _ = c.nextToken(modelArgmax: arg)
        }
        for _ in 0..<toks.exitEnum_categoryToDescription.count {
            _ = c.nextToken(modelArgmax: 0)
        }

        // Emit one content token then close with `","`.
        let jp = tok.encode(text: "白").map { Int32($0) }
        for arg in jp { _ = c.nextToken(modelArgmax: arg) }

        let closeMerged = tok.encode(text: "\",\"").map { Int32($0) }
        XCTAssertEqual(closeMerged.count, 1,
            "expected `\",\"` to be a single Qwen3 BPE token")
        XCTAssertTrue(toks.quoteContainingTokens.contains(closeMerged[0]),
            "BPE-merged `\",\"` should be in quoteContainingTokens")

        var emitted: [Int32] = [c.nextToken(modelArgmax: closeMerged[0])]
        let postLen = toks.exitFreeString_descriptionToScores.count
        for _ in 0..<(postLen - 1) {
            emitted.append(c.nextToken(modelArgmax: 0))
        }

        let text = tok.decode(tokens: emitted.map { Int($0) })
        XCTAssertEqual(text, "\",\"scores\":{\"color\":",
            "expected post-string forced sequence")
        XCTAssertEqual(c.phase, .insideItemScoreColor)
    }

    /// Free-number state accepts digits + dots, terminates on
    /// non-numeric tokens with substitution.
    func testFreeNumberAcceptsDigitsAndTerminates() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        // Walk to .insideItemScoreColor.
        let openingLen = toks.enter_documentItems.count
            + toks.enter_itemFromArray.count
        for _ in 0..<openingLen { _ = c.nextToken(modelArgmax: 0) }
        for arg in toks.categoryEnumOptions[0].tokens {
            _ = c.nextToken(modelArgmax: arg)
        }
        for _ in 0..<toks.exitEnum_categoryToDescription.count {
            _ = c.nextToken(modelArgmax: 0)
        }
        let closeMerged = tok.encode(text: "\",\"").map { Int32($0) }[0]
        _ = c.nextToken(modelArgmax: closeMerged)
        for _ in 0..<(toks.exitFreeString_descriptionToScores.count - 1) {
            _ = c.nextToken(modelArgmax: 0)
        }
        XCTAssertEqual(c.phase, .insideItemScoreColor)

        // Feed "0.85" — digit, dot, digit, digit. All should pass.
        let numTokens = tok.encode(text: "0.85").map { Int32($0) }
        for n in numTokens {
            let returned = c.nextToken(modelArgmax: n)
            XCTAssertEqual(returned, n,
                "free-number should pass digit/dot through")
        }
        XCTAssertEqual(c.phase, .insideItemScoreColor,
            "still in number phase while emitting digits")

        // Terminate with `,`. Constraint substitutes with first token
        // of `,"silhouette":`.
        let returned = c.nextToken(modelArgmax: toks.comma)
        XCTAssertEqual(returned, toks.enter_silhouette[0],
            "terminator should substitute with next forced sequence head")
        XCTAssertEqual(c.phase, .insideItemScoreSilhouette)
    }

    /// Coordinate silhouette type pick: enum mechanism works at the
    /// second enum site. Driving past the items array, we reach
    /// `.coordType` and pick "I".
    func testCoordTypeEnumPick() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        // Drive forward until just before .coordType becomes awaiting.
        // We can't easily stop at "right before" — drive until
        // .coordStyleScore (just past coord type) and verify the
        // emitted text contains `"type":"I"`.
        let emitted = driveToCompletion(c, toks: toks, tok: tok,
                                          stopAtPhase: .coordStyleScore)
        let text = tok.decode(tokens: emitted.map { Int($0) })
        XCTAssertTrue(text.contains("\"type\":\"I\","),
            "expected coord type 'I' in driven output, got: \(text)")
    }

    /// Empty-number protection: model emits a non-digit terminator
    /// before any digit lands in the number's body. Constraint must
    /// force-emit `0` so the JSON `"color":,...` (invalid) becomes
    /// `"color":0,...` (valid). The model's terminator is honored on
    /// the next step.
    func testEmptyNumberForcesLeadingZero() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        // Walk to .insideItemScoreColor as in
        // testFreeNumberAcceptsDigitsAndTerminates.
        let openingLen = toks.enter_documentItems.count
            + toks.enter_itemFromArray.count
        for _ in 0..<openingLen { _ = c.nextToken(modelArgmax: 0) }
        for arg in toks.categoryEnumOptions[0].tokens {
            _ = c.nextToken(modelArgmax: arg)
        }
        for _ in 0..<toks.exitEnum_categoryToDescription.count {
            _ = c.nextToken(modelArgmax: 0)
        }
        let closeMerged = tok.encode(text: "\",\"").map { Int32($0) }[0]
        _ = c.nextToken(modelArgmax: closeMerged)
        for _ in 0..<(toks.exitFreeString_descriptionToScores.count - 1) {
            _ = c.nextToken(modelArgmax: 0)
        }
        XCTAssertEqual(c.phase, .insideItemScoreColor)

        // Model emits `,` (terminator) immediately — no digit yet.
        // Constraint should force-emit `0` and STAY in the number
        // phase, deferring the terminator handling.
        let zeroEmit = c.nextToken(modelArgmax: toks.comma)
        XCTAssertEqual(zeroEmit, toks.digitZero,
            "expected forced `0` when model tries to terminate empty number")
        XCTAssertEqual(c.phase, .insideItemScoreColor,
            "should still be in color number phase after forced `0`")

        // Model emits `,` again — this time the number has content,
        // so terminator path runs and we transition to silhouette.
        let term = c.nextToken(modelArgmax: toks.comma)
        XCTAssertEqual(term, toks.enter_silhouette[0],
            "second `,` after the forced `0` should terminate normally")
        XCTAssertEqual(c.phase, .insideItemScoreSilhouette)
    }

    /// Leading-dot protection: model emits `.` before any digit.
    /// Constraint inserts `0` first so the value starts as `0.`,
    /// which combined with the model's subsequent digits yields a
    /// valid `0.X` number (not the JSON-invalid `.X`).
    func testLeadingDotForcesLeadingZero() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        let openingLen = toks.enter_documentItems.count
            + toks.enter_itemFromArray.count
        for _ in 0..<openingLen { _ = c.nextToken(modelArgmax: 0) }
        for arg in toks.categoryEnumOptions[0].tokens {
            _ = c.nextToken(modelArgmax: arg)
        }
        for _ in 0..<toks.exitEnum_categoryToDescription.count {
            _ = c.nextToken(modelArgmax: 0)
        }
        let closeMerged = tok.encode(text: "\",\"").map { Int32($0) }[0]
        _ = c.nextToken(modelArgmax: closeMerged)
        for _ in 0..<(toks.exitFreeString_descriptionToScores.count - 1) {
            _ = c.nextToken(modelArgmax: 0)
        }
        XCTAssertEqual(c.phase, .insideItemScoreColor)

        // Model emits `.` first — should be substituted with `0`.
        let dotToken = tok.encode(text: ".").map { Int32($0) }[0]
        XCTAssertTrue(toks.numberContinuationTokens.contains(dotToken),
            "test prereq: bare `.` must be in numberContinuationTokens")
        let returned = c.nextToken(modelArgmax: dotToken)
        XCTAssertEqual(returned, toks.digitZero,
            "leading `.` should be substituted with forced `0`")
        XCTAssertEqual(c.phase, .insideItemScoreColor)
    }

    /// Items array continuation: model emits `,` → another item is
    /// queued. Then closing with `]` advances to overall_dress_ratio.
    func testItemsContinuationThenEnd() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        // Drive to .afterItemClose (1 item complete).
        _ = driveToCompletion(c, toks: toks, tok: tok,
                                stopAtPhase: .afterItemClose)
        XCTAssertEqual(c.phase, .afterItemClose)

        // Continue with comma.
        let cont = c.nextToken(modelArgmax: toks.comma)
        XCTAssertEqual(cont, toks.comma)
        // Constraint queued the next item's `{"category":"` opener.
        // Once that drains, we're back at .insideItemPickCategory.
        XCTAssertEqual(c.phase, .insideItemPickCategory)
        XCTAssertFalse(c.isAwaitingModel,
            "opener for second item should be queued")

        // Drive again to .afterItemClose (now 2 items).
        _ = driveToCompletion(c, toks: toks, tok: tok,
                                stopAtPhase: .afterItemClose)
        XCTAssertEqual(c.phase, .afterItemClose)

        // End array.
        let end = c.nextToken(modelArgmax: toks.closeBracket)
        XCTAssertEqual(end, toks.closeBracket)
        XCTAssertEqual(c.phase, .overallDressRatio)
    }

    /// `decideItemContinuation` falls back to closing the array (`]`)
    /// when the model emits anything other than `,`. Defensive — if
    /// the model goes off-schema at the choice point, we always
    /// terminate the document rather than loop forever.
    func testItemsContinuationFallbackToEnd() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        _ = driveToCompletion(c, toks: toks, tok: tok,
                                stopAtPhase: .afterItemClose)

        // Model emits a digit (out-of-schema) at the choice point.
        let returned = c.nextToken(modelArgmax: 15)
        XCTAssertEqual(returned, toks.closeBracket,
            "fallback should be `]` to terminate the array")
        XCTAssertEqual(c.phase, .overallDressRatio)
    }

    /// Reset between generate() calls fully clears state — second
    /// run starts from .start with the document opener queued.
    func testResetBetweenCalls() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        _ = driveToCompletion(c, toks: toks, tok: tok,
                                stopAtPhase: .afterItemClose)
        XCTAssertEqual(c.phase, .afterItemClose)

        c.reset()
        XCTAssertEqual(c.phase, .start)
        XCTAssertFalse(c.isDone)
        XCTAssertFalse(c.isAwaitingModel,
            "reset should re-prime the document opener queue")

        // First emission after reset is the first token of the
        // document opener — Qwen3 BPE merges `{"` into a single token,
        // so the first emission is NOT `{` alone but the merged
        // `{"` token at enter_documentItems[0].
        let first = c.nextToken(modelArgmax: 99999)
        XCTAssertEqual(first, toks.enter_documentItems[0],
            "post-reset first emission should be the document opener's "
            + "first token (likely `{\"`, not bare `{` — Qwen3 BPE merges "
            + "`{\"` into one token in JSON contexts)")
    }

    /// Full end-to-end document: constraint produces a structurally
    /// valid v3 fashion JSON. Asserts JSON parseability + schema
    /// keys are all present.
    func testFullValidDocumentEndToEnd() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        let emitted = driveToCompletion(c, toks: toks, tok: tok)
        XCTAssertTrue(c.isDone, "document should complete")

        let text = tok.decode(tokens: emitted.map { Int($0) })

        // Structural: starts with `{"items":[`, ends with `"}`.
        XCTAssertTrue(text.hasPrefix("{\"items\":[{\"category\":\""),
            "should start with items array opening, prefix: "
            + "\(String(text.prefix(40)))")
        XCTAssertTrue(text.hasSuffix("\"}"),
            "should end with `\"}`, suffix: \(String(text.suffix(20)))")
        // All 6 top-level keys.
        for key in ["items", "overall_dress_ratio", "coordinate_silhouette",
                    "target_ratio", "verdict", "advice"] {
            XCTAssertTrue(text.contains("\"\(key)\":"),
                "should contain top-level key '\(key)'")
        }
        // All 5 score axes.
        for axis in ["color", "silhouette", "material", "design", "item_type"] {
            XCTAssertTrue(text.contains("\"\(axis)\":"),
                "should contain score axis '\(axis)'")
        }
        // item_dress_score + coordinate_silhouette sub-keys.
        XCTAssertTrue(text.contains("\"item_dress_score\":"))
        XCTAssertTrue(text.contains("\"type\":\"I\""))
        XCTAssertTrue(text.contains("\"style_score\":"))
        XCTAssertTrue(text.contains("\"rationale\":"))
        // Pinned target_ratio.
        XCTAssertTrue(text.contains("\"target_ratio\":0.7"),
            "target_ratio should be pinned to 0.7")

        // Must parse as JSON.
        let data = text.data(using: .utf8)!
        XCTAssertNoThrow(try JSONSerialization.jsonObject(with: data),
            "constrained output must parse as valid JSON. Got: \(text)")
    }

    /// `isDone` flips to true the moment the closing `"}` is emitted,
    /// so the decode loop can break before wasting another stepPredict.
    func testIsDoneFlipsImmediatelyOnClose() throws {
        let toks = try tokens()
        let c = FashionV3Constraint(tokens: toks)
        let tok = try tokenizer()

        let emitted = driveToCompletion(c, toks: toks, tok: tok)
        XCTAssertTrue(c.isDone)
        // The last emitted token should be `"}` (token 9207 in Qwen3).
        let last = emitted.last!
        XCTAssertTrue(toks.quoteContainingTokens.contains(last),
            "last token before isDone should be `\"}` (quote-containing)")
    }

    /// Guard against unintended growth in the structural sequences.
    /// If someone accidentally adds a trailing space or changes
    /// tokenization, this test catches the drift.
    func testForcedSequenceRoundTrip() throws {
        let toks = try tokens()
        let tok = try tokenizer()
        let cases: [(name: String, expected: String, tokens: [Int32])] = [
            ("enter_documentItems", "{\"items\":[", toks.enter_documentItems),
            ("enter_itemFromArray", "{\"category\":\"", toks.enter_itemFromArray),
            ("exitEnum_categoryToDescription",
                ",\"description\":\"", toks.exitEnum_categoryToDescription),
            ("exitFreeString_descriptionToScores",
                "\",\"scores\":{\"color\":",
                toks.exitFreeString_descriptionToScores),
            ("enter_silhouette", ",\"silhouette\":", toks.enter_silhouette),
            ("enter_material", ",\"material\":", toks.enter_material),
            ("enter_design", ",\"design\":", toks.enter_design),
            ("enter_itemType", ",\"item_type\":", toks.enter_itemType),
            ("exitFreeNumber_itemTypeToItemDressScore",
                "},\"item_dress_score\":",
                toks.exitFreeNumber_itemTypeToItemDressScore),
            ("exitFreeNumber_itemDressScoreToItemClose", "}",
                toks.exitFreeNumber_itemDressScoreToItemClose),
            ("exitItemArray_toOverall",
                ",\"overall_dress_ratio\":", toks.exitItemArray_toOverall),
            ("enter_coordType",
                ",\"coordinate_silhouette\":{\"type\":\"",
                toks.enter_coordType),
            ("exitEnum_typeToStyleScore",
                ",\"style_score\":", toks.exitEnum_typeToStyleScore),
            ("enter_rationale", ",\"rationale\":\"", toks.enter_rationale),
            ("exitFreeString_rationaleToVerdict",
                "\"},\"target_ratio\":0.7,\"verdict\":\"",
                toks.exitFreeString_rationaleToVerdict),
            ("exitFreeString_verdictToAdvice",
                "\",\"advice\":\"", toks.exitFreeString_verdictToAdvice),
            ("exitFreeString_adviceCloseDocument", "\"}",
                toks.exitFreeString_adviceCloseDocument),
        ]
        for (name, expected, ids) in cases {
            let decoded = tok.decode(tokens: ids.map { Int($0) })
            XCTAssertEqual(decoded, expected,
                "\(name) round-trip mismatch — expected `\(expected)`, got `\(decoded)`")
        }
    }
}
