//
//  PromptLookupLoop.swift
//  CoreMLLLM
//
//  Route B Task 1 — runtime wiring for Prompt Lookup Decoding (PLD).
//
//  Wraps the pure `PromptLookupDraft.propose(...)` algorithm in a
//  speculative-decoding loop that reuses the target's existing Q=K verify
//  chunks (`ChunkedEngine.verifyCandidates`). No model assets, no extra
//  ANE/GPU dispatch — drafts come from a cheap n-gram lookup over the
//  running token history.
//
//  Acceptance contract (mirrors `MtpSpeculativeEngine`):
//    - For each burst, propose up to `maxDraftLen` tokens via PLD.
//    - Pad to the target's `verifyK` (verify chunks require exactly K
//      tokens per call) and run `target.verifyCandidates`.
//    - Walk proposals against the target's argmax; accept the matching
//      prefix, then emit one correction token at the first miss.
//    - Emitted = [tTokNext, matched proposals..., correction].
//    - Commit advances `currentPosition` by `matchCount + 1`
//      (tTokNext + matched). The correction's KV will be (re)written on
//      the next burst.
//
//  At `temperature = 0` (the only sampling mode currently), the output
//  stream is the same as the serial `predictStep` path — that is the
//  invariant of speculative decoding with verify, and the merge-blocking
//  test for this task.
//

import Foundation

/// Speculative-decoding loop driven by Prompt Lookup Decoding.
///
/// Owns no model assets and does no ANE work of its own; it only orchestrates
/// the pure n-gram drafter against a `SpeculativeTarget` (typically
/// `ChunkedEngine`).
public final class PromptLookupLoop {

    /// Suffix length used for the n-gram match (typical 2 or 3).
    public let ngramSize: Int

    /// Cap on draft length per burst, after clamping to `verifyK - 1`
    /// (one verify slot is reserved for `tTokNext`).
    public let maxDraftLen: Int

    /// Target verify chunk capacity (Q=K). Verify calls require exactly
    /// this many tokens. Drafts are padded up to this size with
    /// `padTokenID` before being sent through verify.
    public let verifyK: Int

    /// Token id used to fill verify slots beyond the actual proposals.
    /// Padded slots are never accepted/yielded — their argmax is ignored.
    public var padTokenID: Int32 = 0

    /// EMA of per-burst acceptance ratio in [0, 1]. Starts optimistic so
    /// the first few bursts are not gated out.
    private(set) public var rollingAcceptance: Double = 1.0
    private let rollingAlpha: Double = 0.10

    /// Disable speculation when rolling acceptance drops below this.
    /// PLD has bursty acceptance (huge wins on quoted prompts, near-zero
    /// on free-form chat), so the floor is low — we mainly want to skip
    /// sustained zero-accept stretches.
    public var fallbackThreshold: Double = 0.05

    // MARK: - Metrics

    public private(set) var totalRounds: Int = 0
    public private(set) var totalAccepted: Int = 0
    public private(set) var totalEmitted: Int = 0

    /// Lifetime acceptance rate = totalAccepted / max(1, totalProposed).
    public var acceptanceRate: Double {
        let proposed = totalRounds * maxDraftLen
        return proposed == 0 ? 0 : Double(totalAccepted) / Double(proposed)
    }

    /// Whether to attempt speculation for the next burst.
    public var shouldSpeculate: Bool {
        verifyK > 0 && maxDraftLen > 0 && rollingAcceptance >= fallbackThreshold
    }

    // MARK: - Init

    /// - Parameters:
    ///   - verifyK: capacity of the target's verify chunks. If 0, the loop
    ///     is permanently disabled (`shouldSpeculate == false`).
    ///   - ngramSize: suffix length to match against history (default 3).
    ///   - maxDraftLen: cap on draft length per burst; clamped to
    ///     `verifyK - 1` since one slot is reserved for `tTokNext`.
    public init(verifyK: Int, ngramSize: Int = 3, maxDraftLen: Int = 4) {
        self.verifyK = max(verifyK, 0)
        self.ngramSize = max(ngramSize, 1)
        let cap = max(self.verifyK - 1, 0)
        self.maxDraftLen = max(min(maxDraftLen, cap), 0)
    }

    // MARK: - Reset

    public func reset() {
        rollingAcceptance = 1.0
        totalRounds = 0
        totalAccepted = 0
        totalEmitted = 0
    }

    // MARK: - Speculative burst

    /// Execute one PLD burst.
    ///
    /// - Parameters:
    ///   - target: speculative target (ChunkedEngine).
    ///   - history: full token sequence committed so far (prompt + emitted).
    ///     Should NOT include `nextID`; it is treated as `tTokNext` — the
    ///     argmax from the previous decode step that has not yet been
    ///     committed.
    ///   - nextID: target's argmax from the previous decode step. Updated
    ///     in-place to the last emitted token (correction or last accepted
    ///     draft) for the next burst's seed.
    ///   - startPosition: KV position at which `nextID` will be committed
    ///     (i.e. the engine's `currentPosition`).
    ///
    /// - Returns: tokens to emit to the caller. On a clean skip (no usable
    ///   n-gram match, or speculation gated off) returns `[]` and the
    ///   caller should fall back to a regular `predictStep`.
    public func drawBurst(
        target: SpeculativeTarget,
        history: [Int32],
        nextID: inout Int32,
        startPosition: Int
    ) throws -> [Int32] {
        guard shouldSpeculate else { return [] }

        // Lookup uses committed history + tTokNext as the suffix to match.
        var lookupHistory = history
        lookupHistory.append(nextID)

        let proposals = PromptLookupDraft.propose(
            history: lookupHistory,
            ngramSize: ngramSize,
            maxDraftLen: maxDraftLen)
        guard !proposals.isEmpty else {
            // No n-gram match in history — caller will use the serial path.
            // Don't penalise rollingAcceptance for a no-proposal burst.
            return []
        }

        // Build verify input of exactly `verifyK` tokens. Verify position k
        // computes the target's argmax for "what comes after the first k+1
        // input tokens". So:
        //   verify[0] = nextID                  → argmax compared to proposals[0]
        //   verify[1] = proposals[0]            → argmax compared to proposals[1]
        //   ...
        //   verify[P] = proposals[P-1]          → argmax used as bonus token
        //
        // Where P = proposals.count and verify[P+1..verifyK-1] are pad.
        let P = proposals.count
        var verifyTokens = [Int32](repeating: padTokenID, count: verifyK)
        verifyTokens[0] = nextID
        for k in 0..<P {
            let slot = k + 1
            if slot < verifyK { verifyTokens[slot] = proposals[k] }
        }

        let targetArgmax = try target.verifyCandidates(verifyTokens, K: verifyK)
        guard targetArgmax.count == verifyK else { return [] }

        // Walk: accept while proposals[k] == targetArgmax[k], stop at miss.
        var matchCount = 0
        for k in 0..<P {
            if proposals[k] == targetArgmax[k] {
                matchCount += 1
            } else {
                break
            }
        }

        // Emitted = [tTokNext, matched proposals..., correction-or-bonus]
        var emitted: [Int32] = []
        emitted.reserveCapacity(matchCount + 2)
        emitted.append(nextID)
        for k in 0..<matchCount { emitted.append(proposals[k]) }
        if matchCount < P {
            // Mismatch within proposed range — emit target's argmax as
            // correction, do not commit any further drafts.
            emitted.append(targetArgmax[matchCount])
        } else if matchCount + 1 < verifyK {
            // All proposed tokens matched AND the next verify slot was fed
            // a real proposal (the last proposals[P-1]) — its argmax is a
            // legitimate bonus token.
            //   slot = matchCount + 1 = P + 1; that slot was fed
            //   proposals[matchCount-1] = proposals[P-1] iff matchCount ==
            //   P, which is the branch we're in.
            emitted.append(targetArgmax[matchCount])
        }
        // (If matchCount + 1 == verifyK there is no further argmax to
        // consult — the model only computed verifyK positions.)

        // Position advance: tTokNext + matched proposals are now committed
        // in the KV cache. The correction/bonus stays as the carry seed for the
        // next burst and is NOT committed here — its KV slot will be written
        // by the next verify pass.
        //
        // 11c protocol: pass the ACTUAL accepted tokens (not stub zeros).
        // ChunkedEngine.commitAccepted matches each position against the verify
        // input to decide which slices to commit vs which need a fresh T=1.
        let committed = matchCount + 1
        _ = startPosition  // documented for caller clarity; engine uses its own currentPosition
        let committedTokens = Array(emitted.prefix(committed))
        try target.commitAccepted(committedTokens)

        // Metrics
        totalRounds += 1
        totalAccepted += matchCount
        totalEmitted += emitted.count
        let rate = Double(matchCount) / Double(max(maxDraftLen, 1))
        rollingAcceptance = rollingAlpha * rate + (1 - rollingAlpha) * rollingAcceptance

        // Carry the last emitted token as the next burst's tTokNext.
        nextID = emitted.last!
        return emitted
    }
}
