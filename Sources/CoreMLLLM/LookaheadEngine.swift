//
//  LookaheadEngine.swift
//  CoreMLLLM
//
//  Linear LookAhead / Jacobi speculative decoding for our single-sequence
//  verify_qK path. Adapts llama.cpp's lookahead algorithm to the
//  ANE-friendly constraints of our chunked target:
//
//    * Single KV sequence (no per-slot seq IDs, no multi-ray tree).
//    * `verify_qK` takes exactly K tokens sharing one linear position
//      range and emits per-slot argmax.
//    * Argmax-only sampling on device (temperature = 0).
//
//  Per-cycle mechanics:
//
//    1. Build K draft tokens as [nextID, g[0], g[1], …, g[K-2]], where
//       g[…] comes from (in priority order):
//         a. n-gram lookup (`PromptLookupDraft` / `SuffixSpeculativeEngine`)
//            — works when a matching suffix was seen earlier in the run
//            or the cross-session trie.
//         b. Jacobi guess buffer — the tail of the previous cycle's
//            verify argmax. Stale (conditioned on wrong context) but a
//            meaningful warm-start vs random tokens, especially for
//            repetitive or formulaic output.
//    2. Call `engine.verifyCandidates` → targetArgmax[0..K-1].
//    3. Accept greedy prefix where `useProps[k] == targetArgmax[k]`.
//    4. Commit `[nextID, accepted drafts…]` via `engine.commitAccepted`
//       (11c protocol — verify does not mutate persistent KV).
//    5. Refresh the Jacobi buffer from targetArgmax tail so the next
//       cycle has warmed guesses even without an n-gram hit.
//
//  Contract with the caller:
//
//    * `nextID` on entry is the token to commit at `engine.currentPosition`.
//    * `nextID` on exit is the carry (target correction / bonus) — it is
//       NOT in the returned `emitted` array; the outer loop holds it as
//       the seed for the next speculative cycle.
//    * Returned array is all committed tokens in order, so callers may
//       yield them directly to the user.
//
//  Bit-exactness: at temperature = 0, emitted tokens match serial
//  `predictStep` output exactly. This is enforced by committing only
//  prefix-matches of target's own argmax — rejected drafts never reach
//  the output stream.
//

import CoreML
import Foundation

/// LookAhead / Jacobi speculative engine. Drafts K tokens per cycle via
/// n-gram lookup and Jacobi warm-start, verifies them in one ANE
/// dispatch, and commits the accepted prefix.
public final class LookaheadEngine {

    // MARK: - Configuration

    let engine: ChunkedEngine
    let K: Int

    /// Optional suffix-trie drafter for cross-session n-gram lookup. When
    /// `nil` the engine falls back to in-run PLD + Jacobi only.
    public var suffix: SuffixSpeculativeEngine?

    /// N-gram sizes to try for the in-run PromptLookupDraft. Larger n
    /// first (more specific match); smaller n as fallback.
    public var pldNgramSizes: [Int] = [3, 2]

    /// Minimum in-run history length before we start using n-gram
    /// lookup at all. Below this threshold, only the Jacobi buffer
    /// drives draft choice — which avoids wasting verify on arbitrary
    /// filler tokens that happen to n-gram-match the tiny prompt.
    public var minHistoryForNgram: Int = 16

    // MARK: - Metrics

    private(set) public var totalCycles: Int = 0
    private(set) public var totalEmitted: Int = 0
    private(set) public var totalAcceptedDraftSlots: Int = 0
    private(set) public var ngramPicks: Int = 0
    private(set) public var jacobiPicks: Int = 0

    /// Cumulative accept rate over draft slots (slots/total-draft-slots).
    public var acceptanceRate: Double {
        let denom = totalCycles * (K - 1)
        return denom == 0 ? 0 : Double(totalAcceptedDraftSlots) / Double(denom)
    }

    /// Average tokens emitted per speculative cycle (includes seed, not
    /// the carry — so 1.0 means pure serial, K.0 means full all-accept).
    public var tokensPerCycle: Double {
        totalCycles == 0 ? 0 : Double(totalEmitted) / Double(totalCycles)
    }

    public var shouldSpeculate: Bool { true }

    // MARK: - State

    private(set) public var history: [Int32] = []

    /// Jacobi guess buffer — size K-1. Initialised once at first use with
    /// safe filler (repeats of a common token) and refreshed each cycle
    /// from the target's argmax tail. Never exposed to the output.
    private var jacobiGuesses: [Int32] = []

    private var isBootstrapped = false

    // MARK: - Init

    init(engine: ChunkedEngine) {
        precondition(engine.hasVerify,
                     "LookaheadEngine requires verify chunks (K >= 2)")
        precondition(engine.verifyK >= 2,
                     "LookaheadEngine needs K >= 2 (one seed slot + one draft slot)")
        self.engine = engine
        self.K = engine.verifyK
    }

    // MARK: - Lifecycle

    public func reset() {
        history.removeAll()
        jacobiGuesses.removeAll()
        isBootstrapped = false
        totalCycles = 0
        totalEmitted = 0
        totalAcceptedDraftSlots = 0
        ngramPicks = 0
        jacobiPicks = 0
        suffix?.reset()
    }

    public func setPrefillHistory(_ tokens: [Int32]) {
        history = tokens
        suffix?.setPrefillHistory(tokens)
    }

    // MARK: - Speculative step

    /// One speculative cycle. See file header for semantics.
    public func speculateStep(nextID: inout Int32) throws -> [Int32] {
        if !isBootstrapped { return try bootstrapStep(nextID: &nextID) }

        let pos = engine.currentPosition
        let compareLen = K - 1  // verify slot 0 is nextID; drafts fill 1..K-1

        // 1. Build draft. draftSource records where the guesses came
        //    from for metrics; it does not affect correctness.
        let (drafts, draftSource, draftMs) = buildDraftAt(nextID: nextID, compareLen: compareLen)

        precondition(drafts.count == compareLen,
                     "buildDraftAt must return exactly compareLen tokens")

        var verifyTokens = [Int32](repeating: 0, count: K)
        verifyTokens[0] = nextID
        for (i, t) in drafts.enumerated() { verifyTokens[i + 1] = t }

        // 2. Verify. Returns argmax at positions pos..pos+K-1.
        let (targetArgmax, verifyMs) = try SpecProfile.time {
            try engine.verifyCandidates(tokens: verifyTokens, startPosition: pos)
        }

        // 3. Greedy accept on the draft slots. targetArgmax[k] is the
        //    target's argmax at position pos+k+1, conditioned on verify
        //    slots 0..k (== [nextID, drafts[0..k-1]]). If drafts[k]
        //    matches, the next slot's argmax is still a valid prediction.
        var matchCount = 0
        for k in 0..<compareLen {
            if drafts[k] == targetArgmax[k] {
                matchCount += 1
            } else {
                break
            }
        }

        // 4. Commit: [nextID, drafts[0..matchCount-1]] are confirmed at
        //    positions pos..pos+matchCount. `commitAccepted` writes the
        //    corresponding KV slices into the persistent cache.
        var emitted: [Int32] = [nextID]
        emitted.reserveCapacity(matchCount + 1)
        for k in 0..<matchCount { emitted.append(drafts[k]) }

        let committedTokens = Array(emitted.prefix(matchCount + 1))
        let (_, commitMs) = try SpecProfile.time {
            try engine.commitAccepted(committedTokens)
        }

        // Track in-run history and feed the suffix trie (cross-session
        // knowledge base — learns from every trajectory regardless of
        // which source picked this burst).
        history.append(contentsOf: committedTokens)
        suffix?.applyCommit(tokens: committedTokens)

        // 5. Carry: target's argmax at slot `matchCount`.
        //    - On miss (matchCount < compareLen): correction for the
        //      rejected draft position. Becomes next cycle's seed.
        //    - On all-accept (matchCount == compareLen): bonus argmax
        //      at pos+K-1 — the token that would follow drafts[K-2].
        //      This lets the next cycle skip decoding position pos+K-1.
        let carry = targetArgmax[matchCount]

        // 6. Refresh Jacobi buffer for next cycle.
        //
        //    Next cycle will verify [carry, g[0], …, g[K-2]] at position
        //    pos + matchCount + 1. We have two sources for g[…]:
        //
        //    * targetArgmax[matchCount+1 .. K-1] — predicted under the
        //      assumption that draft[matchCount] was correct. It wasn't
        //      (that's why we stopped), so these predictions are biased;
        //      but for the Jacobi fixed-point iteration, they still
        //      converge toward the true continuation over a few cycles.
        //    * Previous jacobiGuesses — stale but previously converged.
        //
        //    We blend: use the valid tail of targetArgmax first, then
        //    fall back to the old Jacobi buffer's tail to fill up to K-1.
        refreshJacobiGuesses(targetArgmax: targetArgmax, matchCount: matchCount)

        // Metrics
        totalCycles += 1
        totalEmitted += emitted.count
        totalAcceptedDraftSlots += matchCount
        switch draftSource {
        case .ngram: ngramPicks += 1
        case .jacobi: jacobiPicks += 1
        }

        SpecProfile.logBurst(
            engine: "lookahead",
            cycle: totalCycles,
            draftMs: draftMs,
            verifyMs: verifyMs,
            commitMs: commitMs,
            accepted: matchCount,
            compareLen: compareLen,
            emitted: emitted.count,
            rolling: acceptanceRate)

        nextID = carry
        return emitted
    }

    // MARK: - Private helpers

    private enum DraftSource { case ngram, jacobi }

    /// Build the `compareLen`-long draft for this cycle. Preference:
    /// n-gram lookup when history is long enough and a match exists;
    /// otherwise the Jacobi guess buffer (possibly padded).
    private func buildDraftAt(nextID: Int32, compareLen: Int)
        -> (drafts: [Int32], source: DraftSource, ms: Double)
    {
        var lookupHist = history
        lookupHist.append(nextID)

        let (ngramDraft, ms) = SpecProfile.time { () -> [Int32] in
            guard lookupHist.count >= minHistoryForNgram else { return [] }
            // Try in-run PLD at each ngram size, longest first.
            for n in pldNgramSizes {
                let p = PromptLookupDraft.propose(
                    history: lookupHist, ngramSize: n, maxDraftLen: compareLen)
                if !p.isEmpty { return p }
            }
            // Cross-session suffix trie.
            if let sfx = suffix {
                let s = sfx.drawBurst(context: lookupHist, K: compareLen)
                if !s.isEmpty { return s }
            }
            return []
        }

        if !ngramDraft.isEmpty {
            // Pad short n-gram hits with Jacobi tail to fill K-1 slots —
            // verify runs at fixed K regardless of draft length, so an
            // unused slot would be wasted.
            let padded = padToCompareLen(ngramDraft, compareLen: compareLen)
            return (padded, .ngram, ms)
        }

        let guesses = currentJacobiGuesses(compareLen: compareLen)
        return (guesses, .jacobi, ms)
    }

    /// Return a compareLen-length draft from the Jacobi buffer,
    /// initialising the buffer lazily with common filler if empty.
    private func currentJacobiGuesses(compareLen: Int) -> [Int32] {
        if jacobiGuesses.count < compareLen {
            // Lazy init. Token 100 (" the" in many BPE vocabs) is a
            // benign filler — ANE cost is token-value-independent, so
            // the exact choice only affects accept rate before the
            // buffer warms up from the first verify's argmax.
            let filler: Int32 = 100
            jacobiGuesses = [Int32](repeating: filler, count: compareLen)
        }
        return Array(jacobiGuesses.prefix(compareLen))
    }

    /// Refresh `jacobiGuesses` from this cycle's target argmax.
    private func refreshJacobiGuesses(targetArgmax: [Int32], matchCount: Int) {
        let compareLen = K - 1
        var fresh = [Int32](); fresh.reserveCapacity(compareLen)

        // Valid-ish tail of targetArgmax (positions beyond the accepted
        // prefix, up to K-1). Positions matchCount+1..K-1 give up to
        // (K-1) - (matchCount+1) + 1 = K-1-matchCount guesses.
        if matchCount + 1 <= K - 1 {
            for i in (matchCount + 1)..<K {
                fresh.append(targetArgmax[i])
            }
        }

        // Fill the rest from the previous Jacobi buffer's tail. When the
        // buffer is empty (first cycle), fall back to filler.
        while fresh.count < compareLen {
            let idx = fresh.count
            if idx < jacobiGuesses.count {
                fresh.append(jacobiGuesses[idx])
            } else {
                fresh.append(100)
            }
        }
        jacobiGuesses = Array(fresh.prefix(compareLen))
    }

    private func padToCompareLen(_ draft: [Int32], compareLen: Int) -> [Int32] {
        if draft.count >= compareLen { return Array(draft.prefix(compareLen)) }
        var out = draft
        let guesses = currentJacobiGuesses(compareLen: compareLen)
        for i in draft.count..<compareLen { out.append(guesses[i]) }
        return out
    }

    /// First call: run a normal decode to populate KV and seed the
    /// next cycle's nextID. Same pattern as MtpSpeculativeEngine.
    private func bootstrapStep(nextID: inout Int32) throws -> [Int32] {
        let emittedTok = nextID
        let (newNext, targetStepMs) = try SpecProfile.time {
            try engine.predictStep(
                tokenID: Int(nextID), position: engine.currentPosition)
        }
        engine.currentPosition += 1
        history.append(emittedTok)
        suffix?.applyCommit(tokens: [emittedTok])
        nextID = Int32(newNext)
        isBootstrapped = true
        SpecProfile.logBootstrap(
            engine: "lookahead", replayCount: 0,
            replayMs: 0, targetStepMs: targetStepMs)
        return [emittedTok]
    }
}
