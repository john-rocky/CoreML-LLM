//
//  DrafterUnion.swift
//  CoreMLLLM
//
//  Phase B Task 1 — Union-of-drafters orchestrator.
//
//  Per A5 decision (`docs/PHASE_A5_DECISION.md`), no single drafter wins all
//  four workload categories. The union runs three sources per burst:
//
//    * Cross-vocab Qwen 2.5 0.5B  — chat + qa anchor (E[tok/burst] ≈ 2.31, 3.17)
//    * Prompt-Lookup, n = 3       — code anchor (E[tok/burst] ≈ 2.94)
//    * Prompt-Lookup, n = 2       — summary anchor (E[tok/burst] ≈ 3.26)
//
//  Selection policy (priority, since temp=0): pick the drafter whose
//  proposal is longest; break ties in priority order cv > pl-n3 > pl-n2.
//  One target verify call per burst regardless of how many drafters ran.
//
//  Bit-exactness invariant: at temperature = 0 the emitted token stream
//  must equal the serial `predictStep` path. This is the merge-blocking
//  test (run on Mac before iPhone trip).
//

import Foundation

public final class DrafterUnion {

    public enum Source: String, Hashable {
        case crossVocab = "cv"
        case pldN3      = "pl3"
        case pldN2      = "pl2"
        case fallback   = "single"
    }

    let engine: ChunkedEngine
    let crossVocab: CrossVocabDraft?
    let K: Int

    private static let priority: [Source: Int] = [.crossVocab: 0, .pldN3: 1, .pldN2: 2]

    /// Per-source rolling-accept gates. Cross-vocab uses a higher floor
    /// because its drafter cost is non-trivial (Qwen forward); prompt-lookup
    /// is a CPU n-gram scan so the floor is near zero.
    public var crossVocabThreshold: Double = 0.20
    public var pldThreshold: Double = 0.05

    /// Hard-disable the cross-vocab source even when it's loaded. Skips
    /// both the bootstrap replay through Qwen and the per-burst draftBurst.
    /// Used by the Mac bit-exact verifier to sidestep the staging Qwen's
    /// ctx=512 vs config.contextLength=2048 mismatch.
    public var crossVocabDisabled: Bool = false

    private(set) public var rollingCV: Double  = 1.0
    private(set) public var rollingPL3: Double = 1.0
    private(set) public var rollingPL2: Double = 1.0
    private let rollingAlpha: Double = 0.10

    private var isBootstrapped = false
    private var prefillHistory: [Int32] = []
    private(set) public var history: [Int32] = []

    private(set) public var totalCycles: Int = 0
    private(set) public var totalEmitted: Int = 0
    private(set) public var totalAcceptedDraftSlots: Int = 0
    private(set) public var picks: [Source: Int] = [:]

    public var acceptanceRate: Double {
        let denom = totalCycles * K
        return denom == 0 ? 0 : Double(totalAcceptedDraftSlots) / Double(denom)
    }

    public var tokensPerCycle: Double {
        totalCycles == 0 ? 0 : Double(totalEmitted) / Double(totalCycles)
    }

    /// Always returns true; per-source gates filter inside `speculateStep`
    /// and a `fallbackSingleStep` keeps progress when no drafter proposes.
    public var shouldSpeculate: Bool { true }

    init(engine: ChunkedEngine, crossVocab: CrossVocabDraft?, K: Int) {
        precondition(engine.hasVerify, "DrafterUnion requires verify chunks")
        precondition(engine.verifyK == K, "K mismatch (engine=\(engine.verifyK), K=\(K))")
        self.engine = engine
        self.crossVocab = crossVocab
        self.K = K
        for s: Source in [.crossVocab, .pldN3, .pldN2, .fallback] { picks[s] = 0 }
    }

    public func reset() {
        isBootstrapped = false
        prefillHistory.removeAll()
        history.removeAll()
        crossVocab?.reset()
        rollingCV = 1.0; rollingPL3 = 1.0; rollingPL2 = 1.0
        totalCycles = 0; totalEmitted = 0; totalAcceptedDraftSlots = 0
        for k in picks.keys { picks[k] = 0 }
    }

    public func setPrefillHistory(_ tokens: [Int32]) {
        prefillHistory = tokens
        history = tokens
    }

    public func speculateStep(nextID: inout Int32) throws -> [Int32] {
        if !isBootstrapped { return try bootstrap(nextID: &nextID) }

        let pos = engine.currentPosition

        // UNION_DEBUG_CV: snapshot CV state before any per-burst work. We
        // report this for every burst (even ones where CV wasn't proposed)
        // so drift across "CV skipped" intervals is visible.
        let cvPosBefore: Int = cvActive?.committedPosition ?? -1
        let rCVSnapshot = rollingCV
        let rPL3Snapshot = rollingPL3
        let rPL2Snapshot = rollingPL2
        let cvGatePassed = (rollingCV >= crossVocabThreshold)

        // 1. Collect per-source proposals.
        var lookupHist = history
        lookupHist.append(nextID)

        let (plProps3, pl3Ms): ([Int32], Double) = SpecProfile.time {
            (rollingPL3 >= pldThreshold)
                ? PromptLookupDraft.propose(history: lookupHist, ngramSize: 3, maxDraftLen: K - 1)
                : []
        }
        let (plProps2, pl2Ms): ([Int32], Double) = SpecProfile.time {
            (rollingPL2 >= pldThreshold)
                ? PromptLookupDraft.propose(history: lookupHist, ngramSize: 2, maxDraftLen: K - 1)
                : []
        }

        var cvBurst: DraftBurst? = nil
        var cvMs: Double = 0
        if let cv = crossVocab, !crossVocabDisabled, cvGatePassed {
            // Side-effect: writes K Qwen KV slots speculatively. If CV is
            // not picked, we rewind below.
            (cvBurst, cvMs) = try SpecProfile.time { try cv.draftBurst(seed: nextID) }
        }
        let cvProps: [Int32] = cvBurst?.drafts ?? []
        let cvPosAfterPropose: Int = cvActive?.committedPosition ?? -1

        // 2. Selection: longest first, then priority cv > pl-n3 > pl-n2.
        var candidates: [(Source, [Int32])] = []
        if !cvProps.isEmpty  { candidates.append((.crossVocab, cvProps)) }
        if !plProps3.isEmpty { candidates.append((.pldN3, plProps3)) }
        if !plProps2.isEmpty { candidates.append((.pldN2, plProps2)) }

        guard !candidates.isEmpty else {
            picks[.fallback, default: 0] += 1
            let emittedFallback = try fallbackSingleStep(nextID: &nextID, cvBurst: cvBurst)
            let cvPosAfterFallback = cvActive?.committedPosition ?? -1
            SpecProfile.logUnionDebugCV(
                cycle: totalCycles, source: Source.fallback.rawValue,
                rollingCV: rCVSnapshot, rollingPL3: rPL3Snapshot, rollingPL2: rPL2Snapshot,
                cvProposed: !cvProps.isEmpty,
                cvPosBefore: cvPosBefore,
                cvPosAfterPropose: cvPosAfterPropose,
                cvPosAfterRewind: cvPosAfterPropose, // fallback path doesn't explicitly rewind
                cvPosAfterCommit: cvPosAfterFallback,
                enginePosBefore: pos,
                enginePosAfterCommit: engine.currentPosition,
                matchCount: 0, compareLen: 0)
            return emittedFallback
        }

        candidates.sort { a, b in
            if a.1.count != b.1.count { return a.1.count > b.1.count }
            return Self.priority[a.0]! < Self.priority[b.0]!
        }
        let (source, props) = candidates[0]

        // 3. Build verify input. Verify chunks take exactly K tokens.
        //    Slot 0 = nextID; slots 1..K-1 = first K-1 proposals; pad if short.
        //    Cap proposals to K-1 across all sources (including CV) so every
        //    accepted token's KV is actually written by verify — comparing
        //    a K-th draft (not in the input) would commit a position whose
        //    KV slot was never populated, leaving a hole the next burst
        //    silently reads.
        let useProps = Array(props.prefix(K - 1))
        let compareLen = useProps.count
        _ = source  // (kept for the per-source rolling-accept update below)

        var verifyTokens = [Int32](repeating: 0, count: K)
        verifyTokens[0] = nextID
        for (i, t) in useProps.enumerated() { verifyTokens[i + 1] = t }

        let (targetArgmax, verifyMs) = try SpecProfile.time {
            try engine.verifyCandidates(
                tokens: verifyTokens, startPosition: pos)
        }

        // 4. Walk acceptance. `matches` records per-position agreement
        //    across the full compareLen — not just the accepted prefix —
        //    so downstream profiling can see post-miss tail matches
        //    (useful for C0 tolerance analysis per PHASE_B_V4 findings).
        var matchCount = 0
        var matches = [Bool](repeating: false, count: compareLen)
        var prefixStillMatching = true
        for k in 0..<compareLen {
            let hit = (useProps[k] == targetArgmax[k])
            matches[k] = hit
            if prefixStillMatching {
                if hit { matchCount += 1 } else { prefixStillMatching = false }
            }
        }

        // 5. Emitted = [nextID, matched...] — committed tokens, all yielded.
        //    Carry = correction (on miss) OR bonus argmax (on all-match) —
        //    NOT yielded here; becomes next burst's seed. This avoids the
        //    classic speculative-decode double-emit where the previous
        //    burst's last token is also the new burst's first.
        var emitted: [Int32] = [nextID]
        emitted.reserveCapacity(matchCount + 1)
        for k in 0..<matchCount { emitted.append(useProps[k]) }

        let carry: Int32
        if matchCount < compareLen {
            carry = targetArgmax[matchCount]
        } else if matchCount < K {
            carry = targetArgmax[matchCount]
        } else {
            // matchCount == compareLen == K; impossible with the K-1 cap
            // above, but keep the carry well-defined.
            carry = targetArgmax[K - 1]
        }

        // 6. Commit on target. Under the 11c protocol, verify does NOT write
        //    KV to the persistent cache; commitAccepted is what writes the
        //    accepted-prefix slices. Bumping currentPosition directly (as the
        //    pre-11c code did) leaves the cache stale and corrupts decode.
        let committed = matchCount + 1
        let committedTokens = Array(emitted.prefix(committed))

        // 7. Sync history (committed tokens only).
        for k in 0..<committed { history.append(emitted[k]) }

        // 8. Commit KV via the engine, plus sync cross-vocab Qwen state.
        //    If CV was the source, applyCommit rewinds/extends correctly.
        //    If CV ran but wasn't picked, its speculative writes don't match
        //    the committed prefix — rewind and replay the actual committed
        //    tokens through Qwen.
        var cvPosAfterRewind: Int = cvActive?.committedPosition ?? -1
        let (_, commitMs) = try SpecProfile.time {
            try engine.commitAccepted(committedTokens)
            if let cv = cvActive, let burst = cvBurst {
                if source == .crossVocab {
                    try cv.applyCommit(matchCount: matchCount, burst: burst)
                } else {
                    cv.committedPosition = pos
                    cvPosAfterRewind = cv.committedPosition
                    for k in 0..<committed {
                        let gid = emitted[k]
                        let qid = cv.vocabMap.gemma(gid)
                        if qid >= 0 {
                            _ = try cv.consume(gemmaToken: gid)
                        } else {
                            cv.committedPosition += 1
                        }
                    }
                }
            }
        }
        let cvPosAfterCommit: Int = cvActive?.committedPosition ?? -1

        // 9. Update rolling-accept of the chosen source. Sources we ran
        //    but didn't pick get no signal — keep their EMA stable.
        let denom = max(compareLen, 1)
        let rate = Double(matchCount) / Double(denom)
        switch source {
        case .crossVocab: rollingCV  = rollingAlpha * rate + (1 - rollingAlpha) * rollingCV
        case .pldN3:      rollingPL3 = rollingAlpha * rate + (1 - rollingAlpha) * rollingPL3
        case .pldN2:      rollingPL2 = rollingAlpha * rate + (1 - rollingAlpha) * rollingPL2
        case .fallback:   break
        }

        picks[source, default: 0] += 1
        totalCycles += 1
        totalEmitted += emitted.count
        totalAcceptedDraftSlots += matchCount

        SpecProfile.logUnionBurst(
            cycle: totalCycles, source: source.rawValue,
            perSourceMs: ["cv": cvMs, "pl3": pl3Ms, "pl2": pl2Ms],
            verifyMs: verifyMs, commitMs: commitMs,
            accepted: matchCount, compareLen: compareLen,
            emitted: emitted.count, matches: matches)

        SpecProfile.logUnionDebugCV(
            cycle: totalCycles, source: source.rawValue,
            rollingCV: rCVSnapshot, rollingPL3: rPL3Snapshot, rollingPL2: rPL2Snapshot,
            cvProposed: !cvProps.isEmpty,
            cvPosBefore: cvPosBefore,
            cvPosAfterPropose: cvPosAfterPropose,
            cvPosAfterRewind: cvPosAfterRewind,
            cvPosAfterCommit: cvPosAfterCommit,
            enginePosBefore: pos,
            enginePosAfterCommit: pos + committed,
            matchCount: matchCount, compareLen: compareLen)

        nextID = carry
        return emitted
    }

    // MARK: - Private

    private var cvActive: CrossVocabDraft? {
        crossVocabDisabled ? nil : crossVocab
    }

    private func bootstrap(nextID: inout Int32) throws -> [Int32] {
        var replayCount = 0
        let (_, replayMs) = try SpecProfile.time {
            if let cv = cvActive {
                let targetPos = engine.currentPosition
                replayCount = min(prefillHistory.count, targetPos)
                for i in 0..<replayCount {
                    let gid = prefillHistory[i]
                    let qid = cv.vocabMap.gemma(gid)
                    if qid >= 0 {
                        _ = try cv.consume(gemmaToken: gid)
                    } else {
                        cv.committedPosition += 1
                    }
                }
                if targetPos > cv.committedPosition {
                    cv.committedPosition = targetPos
                }
            }
        }
        isBootstrapped = true

        let emittedTok = nextID
        let (newNext, targetStepMs) = try SpecProfile.time {
            try engine.predictStep(
                tokenID: Int(nextID), position: engine.currentPosition)
        }
        engine.currentPosition += 1
        history.append(emittedTok)
        if let cv = cvActive {
            _ = try cv.consume(gemmaToken: nextID)
        }
        nextID = Int32(newNext)
        SpecProfile.logBootstrap(
            engine: "union", replayCount: replayCount,
            replayMs: replayMs, targetStepMs: targetStepMs)
        return [emittedTok]
    }

    private func fallbackSingleStep(nextID: inout Int32,
                                    cvBurst: DraftBurst?) throws -> [Int32] {
        let pos = engine.currentPosition
        if let cv = cvActive, cvBurst != nil {
            cv.committedPosition = pos
        }
        let emittedTok = nextID
        let (newNext, targetStepMs) = try SpecProfile.time {
            try engine.predictStep(
                tokenID: Int(nextID), position: engine.currentPosition)
        }
        engine.currentPosition += 1
        history.append(emittedTok)
        if let cv = cvActive {
            let qid = cv.vocabMap.gemma(nextID)
            if qid >= 0 {
                _ = try cv.consume(gemmaToken: nextID)
            } else {
                cv.committedPosition += 1
            }
        }
        nextID = Int32(newNext)
        SpecProfile.logFallback(
            engine: "union", cycle: totalCycles, targetStepMs: targetStepMs)
        return [emittedTok]
    }
}
