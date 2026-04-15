//
//  CrossVocabSpeculativeEngine.swift
//  CoreMLLLM
//
//  Route B / Task 3 — cross-vocabulary speculative decoding loop.
//
//  Mirrors the shape of `MtpSpeculativeEngine` so the same seam in
//  `CoreMLLLM.generate` can host either drafter. Differences from the
//  MTP engine:
//    * The drafter (Qwen 2.5 0.5B) is a fully independent model with its
//      own KV state, not a lightweight head reading Gemma's KV cache.
//    * Acceptance rates are lower (~40% expected vs 55-65% for trained
//      heads), so the rolling-accept gate matters more.
//    * Bootstrap must align Qwen's state with Gemma's committed prefix
//      before the first speculation cycle.
//

import CoreML
import Foundation

public final class CrossVocabSpeculativeEngine {

    let engine: ChunkedEngine
    let drafter: CrossVocabDraft
    let K: Int

    // Rolling-accept gate ------------------------------------------------

    /// EMA of per-cycle acceptance ratio (accepted / K). 1.0 at init so
    /// the first cycle always runs.
    private(set) public var rollingAcceptance: Double = 1.0
    private let rollingAlpha: Double = 0.1
    /// Disable speculation when rolling acceptance drops below this.
    public var fallbackThreshold: Double = 0.20

    // Bootstrap state ----------------------------------------------------

    /// Prefill tokens (Gemma ids) the engine consumed before decode.
    /// The drafter must replay the same prefix to align Qwen state.
    /// Set once before the first `speculateStep` by the generate loop.
    private var isBootstrapped = false

    // Metrics ------------------------------------------------------------

    private(set) var totalCycles = 0
    private(set) var totalAccepted = 0
    private(set) var totalEmitted = 0
    private(set) var totalMissCycles = 0  // unmappable seed

    public var acceptanceRate: Double {
        totalCycles == 0 ? 0 : Double(totalAccepted) / Double(totalCycles * K)
    }
    public var tokensPerCycle: Double {
        totalCycles == 0 ? 0 : Double(totalEmitted) / Double(totalCycles)
    }

    // Prefill history (Gemma ids) for bootstrap replay.
    private var prefillHistory: [Int32] = []

    // MARK: - Init

    init(engine: ChunkedEngine, drafter: CrossVocabDraft) {
        self.engine = engine
        self.drafter = drafter
        self.K = drafter.K
        precondition(engine.hasVerify,
                     "CrossVocab speculation requires verify chunks")
        precondition(engine.verifyK == drafter.K,
                     "Drafter K=\(drafter.K) must match verify K=\(engine.verifyK)")
    }

    /// Record the Gemma prompt tokens the target consumed during prefill.
    /// Called once before the first speculateStep. The drafter replays
    /// them lazily on bootstrap.
    public func setPrefillHistory(_ tokens: [Int32]) {
        prefillHistory = tokens
    }

    // MARK: - Public entry

    public var shouldSpeculate: Bool {
        rollingAcceptance >= fallbackThreshold
    }

    public func reset() {
        drafter.reset()
        isBootstrapped = false
        rollingAcceptance = 1.0
        totalCycles = 0
        totalAccepted = 0
        totalEmitted = 0
        totalMissCycles = 0
        prefillHistory.removeAll()
    }

    /// Execute one speculative cycle. Mirrors `MtpSpeculativeEngine.speculateStep`.
    /// Updates `nextID` in place with the last emitted (uncommitted) token
    /// and returns all tokens emitted this cycle.
    public func speculateStep(nextID: inout Int32) throws -> [Int32] {
        if !isBootstrapped {
            return try bootstrap(nextID: &nextID)
        }

        let pos = engine.currentPosition
        assert(drafter.committedPosition == pos,
               "drafter out of sync: drafter=\(drafter.committedPosition) target=\(pos)")

        // 1. Draft K tokens.
        let burst = try drafter.draftBurst(seed: nextID)
        if burst.drafts.count < K {
            // Unmappable seed or intermediate miss — skip this cycle cleanly.
            // Rewind drafter to P and fall back to a single target decode.
            drafter.committedPosition = pos  // abort burst; stale KV invisible
            totalMissCycles += 1
            // Penalise rolling acceptance a little so repeated misses disable us.
            rollingAcceptance = rollingAlpha * 0.0 + (1 - rollingAlpha) * rollingAcceptance
            return try fallbackSingleStep(nextID: &nextID)
        }

        // 2. Verify on target.
        var verifyTokens = [nextID]
        verifyTokens.append(contentsOf: burst.drafts.dropLast())
        let targetArgmax = try engine.verifyCandidates(
            tokens: verifyTokens, startPosition: pos)

        // 3. Accept prefix.
        var emitted: [Int32] = [nextID]
        emitted.reserveCapacity(K + 1)
        var matchCount = 0
        for k in 0..<K {
            if burst.drafts[k] == targetArgmax[k] {
                emitted.append(burst.drafts[k])
                matchCount += 1
            } else {
                emitted.append(targetArgmax[k])
                break
            }
        }

        // 4. Advance target position and re-anchor drafter.
        let committed = matchCount + 1
        engine.currentPosition = pos + committed
        try drafter.applyCommit(matchCount: matchCount, burst: burst)

        // 5. Update metrics + rolling accept.
        totalCycles += 1
        totalAccepted += matchCount
        totalEmitted += emitted.count
        let rate = Double(matchCount) / Double(K)
        rollingAcceptance = rollingAlpha * rate
            + (1 - rollingAlpha) * rollingAcceptance

        // 6. nextID = last emitted; it will be consumed by the next cycle's
        //    verify (target side) OR next single-step (fallback).
        nextID = emitted.last!
        return emitted
    }

    // MARK: - Private

    /// First spec call. Replays the prefill-derived Gemma prefix through
    /// Qwen so its KV state lines up with the target's currentPosition,
    /// then hands off to the normal speculative path by running one
    /// target predictStep (the same way MtpSpeculativeEngine bootstraps).
    private func bootstrap(nextID: inout Int32) throws -> [Int32] {
        // Replay committed Gemma prefix through Qwen. The target's
        // currentPosition at this point equals the prompt length, and
        // nextID is what the target emitted at that position. We want
        // Qwen's state to cover positions [0..targetPos-1].
        let targetPos = engine.currentPosition
        let history = prefillHistory
        // Defensive: if history is shorter than targetPos (unusual,
        // e.g. prefill ran fewer steps), only replay what we have and
        // let the drafter drift — acceptance will drop and the gate
        // will disable us.
        let replayCount = min(history.count, targetPos)
        for i in 0..<replayCount {
            // Unmappable tokens leave stale Qwen state; acceptance gate
            // will handle it. We still advance committedPosition so it
            // tracks the target — write-through means a later correct
            // write at the same slot overwrites the stale entry.
            let gid = history[i]
            let qid = drafter.vocabMap.gemma(gid)
            if qid >= 0 {
                _ = try drafter.consume(gemmaToken: gid)
            } else {
                // Silent miss: just advance the counter. Pretend the
                // slot will be repopulated later — in practice many
                // Gemma-only tokens (markers, etc.) never reappear in
                // an attention query against that slot.
                drafter.committedPosition += 1
            }
        }
        // If the target ran more positions than we have history for,
        // fast-forward the drafter counter.
        if targetPos > drafter.committedPosition {
            drafter.committedPosition = targetPos
        }
        isBootstrapped = true

        // Emit the current nextID and do one target decode step to
        // advance nextID — matching MtpSpeculativeEngine.bootstrapStep.
        let emitted = nextID
        let newNext = try engine.predictStep(
            tokenID: Int(nextID), position: engine.currentPosition)
        engine.currentPosition += 1
        // Bring drafter to the new currentPosition by consuming the
        // token we just committed on the target.
        _ = try drafter.consume(gemmaToken: nextID)
        nextID = Int32(newNext)
        return [emitted]
    }

    /// Fall back to a single target decode step when the drafter can't
    /// propose (e.g. unmappable seed). Keeps Gemma and Qwen in sync.
    private func fallbackSingleStep(nextID: inout Int32) throws -> [Int32] {
        let emitted = nextID
        let newNext = try engine.predictStep(
            tokenID: Int(nextID), position: engine.currentPosition)
        engine.currentPosition += 1
        _ = try drafter.consume(gemmaToken: nextID)
        nextID = Int32(newNext)
        return [emitted]
    }
}
