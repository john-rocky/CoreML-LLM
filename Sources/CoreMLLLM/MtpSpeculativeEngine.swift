//
//  MtpSpeculativeEngine.swift
//  CoreMLLLM
//
//  Orchestrates the MTP speculative decoding loop:
//    draft K tokens → verify → accept/reject → commit → extract carry state
//
//  Design matches Google's LiteRT runtime (K=3 fixed, greedy argmax):
//    - Write-through KV: verify writes all K positions, rejected entries
//      masked by causal mask and overwritten on next cycle.
//    - Commit = position advance only (no decode re-runs).
//    - Carry state from lastVerifyHiddenStates bootstraps next MTP cycle.
//

import CoreML
import Foundation

/// MTP speculative decoding engine — drafts K tokens with the MTP drafter,
/// verifies them against the target (ChunkedEngine), and commits accepted tokens.
public final class MtpSpeculativeEngine {

    let engine: ChunkedEngine
    let drafter: MtpDraftSource
    let K: Int

    // Carry state for MTP drafting (L34 hidden from verify, or zero on bootstrap)
    private var carryState: MLMultiArray?
    private var isBootstrapped = false

    // MARK: - Metrics

    private(set) var totalRounds = 0
    private(set) var totalAccepted = 0
    private(set) var totalEmitted = 0

    /// acc0 = num_accepted / (num_rounds * K)
    var acceptanceRate: Double {
        totalRounds == 0 ? 0 : Double(totalAccepted) / Double(totalRounds * K)
    }

    /// Average tokens emitted per round (includes tTokNext + bonus/correction)
    var tokensPerRound: Double {
        totalRounds == 0 ? 0 : Double(totalEmitted) / Double(totalRounds)
    }

    // MARK: - Init

    init(engine: ChunkedEngine, drafter: MtpDraftSource) {
        self.engine = engine
        self.drafter = drafter
        self.K = drafter.K
        precondition(engine.hasVerify, "MTP speculation requires verify chunks")
        precondition(engine.verifyK == drafter.K,
                     "Drafter K=\(drafter.K) must match verify K=\(engine.verifyK)")
    }

    // MARK: - Speculative step

    /// Execute one speculative cycle.
    ///
    /// On the first call, runs a normal decode to bootstrap kv13/kv14 state.
    /// Subsequent calls draft K tokens, verify, and commit the accepted prefix.
    ///
    /// - Parameter nextID: the token predicted from the last decode/verify.
    ///   Updated in-place to the next token for the following cycle.
    /// - Returns: tokens to emit to the output stream.
    func speculateStep(nextID: inout Int32) throws -> [Int32] {
        // Bootstrap: first call does a normal decode to populate kv13/kv14
        if !isBootstrapped {
            return try bootstrapStep(nextID: &nextID)
        }

        let pos = engine.currentPosition
        let hidden = engine.config.hiddenSize

        // Initialize carry state with zeros if needed (first real speculative cycle)
        if carryState == nil {
            carryState = try MLMultiArray(
                shape: [1, 1, NSNumber(value: hidden)], dataType: .float16)
            memset(carryState!.dataPointer, 0, hidden * MemoryLayout<UInt16>.stride)
        }

        guard let kv13K = engine.lastKV13K,
              let kv13VRaw = engine.lastKV13V,
              let kv14K = engine.lastKV14K,
              let kv14VRaw = engine.lastKV14V
        else {
            throw SpeculativeError.verifyFailed("kv13/kv14 not available for MTP drafter")
        }
        // The MTP drafter (per build_mtp_drafter.py metadata) expects V
        // transposed as (1, 1, head_dim, seq). ChunkedEngine stores V as
        // (1, 1, seq, head_dim), so we transpose the last two dims once per cycle.
        let kv13V = try Self.transposeLastTwoDims(kv13VRaw)
        let kv14V = try Self.transposeLastTwoDims(kv14VRaw)

        // Build mask ONCE per cycle at the last committed position.
        // The drafter reads target KV (positions 0..pos-1), so mask allows 0..pos-1.
        let maskPos = pos - 1
        let maskSwa = try engine.makeDrafterSWAMask(position: max(maskPos, 0))
        let maskFull = try engine.makeDrafterFullMask(position: max(maskPos, 0))

        // Draft K tokens with per-step RoPE updates.
        var proposals = [Int32]()
        proposals.reserveCapacity(K)
        var embedToken = try engine.lookupRawEmbed(nextID)
        var projAct = carryState!

        let (_, draftMs) = try SpecProfile.time {
            for k in 0..<K {
                let draftPos = pos + k
                let cosSwa = try Self.reshapeRoPEForDrafter(
                    try engine.lookupCosSWA(position: draftPos))
                let sinSwa = try Self.reshapeRoPEForDrafter(
                    try engine.lookupSinSWA(position: draftPos))
                let cosFull = try Self.reshapeRoPEForDrafter(
                    try engine.lookupCosFull(position: draftPos))
                let sinFull = try Self.reshapeRoPEForDrafter(
                    try engine.lookupSinFull(position: draftPos))

                let (tokenId, projActOut) = try drafter.draftOne(
                    embedToken: embedToken,
                    projAct: projAct,
                    kv13K: kv13K, kv13V: kv13V,
                    kv14K: kv14K, kv14V: kv14V,
                    cosSwa: cosSwa, sinSwa: sinSwa,
                    cosFull: cosFull, sinFull: sinFull,
                    maskSwa: maskSwa, maskFull: maskFull)

                proposals.append(tokenId)
                embedToken = try engine.lookupRawEmbed(tokenId)
                projAct = projActOut
            }
        }

        // Verify [nextID, proposals[0..K-2]] at currentPosition.
        // Cap compared proposals to K-1 so every accepted token's KV is
        // actually written by verify — comparing proposals[K-1] against
        // targetArgmax[K-1] was a bug: targetArgmax[K-1] is the argmax at
        // position pos+K (after proposals[K-2] as input), not a verification
        // of proposals[K-1] (which was never fed to verify). On all-accept
        // this also leaves slot pos+K without a KV write, the "all-accept
        // KV hole" that lets the next burst read garbage. Back-port of the
        // DrafterUnion fix.
        let useProps = Array(proposals.prefix(K - 1))
        let compareLen = useProps.count
        var verifyTokens = [Int32](repeating: 0, count: K)
        verifyTokens[0] = nextID
        for (i, t) in useProps.enumerated() { verifyTokens[i + 1] = t }
        let (targetArgmax, verifyMs) = try SpecProfile.time {
            try engine.verifyCandidates(
                tokens: verifyTokens, startPosition: pos)
        }

        // Accept/reject: greedy comparison of the draft positions only.
        var matchCount = 0
        for k in 0..<compareLen {
            if useProps[k] == targetArgmax[k] {
                matchCount += 1
            } else {
                break
            }
        }

        // Emitted = [nextID, matched...] — do NOT include the carry here.
        // The carry (correction on miss, bonus on all-match) stays as the
        // next burst's seed. Previously we emitted it AND used it as
        // nextID, causing the same token to be re-committed at two
        // consecutive positions — visible in output as duplicated words
        // ("some some context context about about"). Back-port of the
        // DrafterUnion fix.
        var emitted: [Int32] = [nextID]
        emitted.reserveCapacity(matchCount + 1)
        for k in 0..<matchCount { emitted.append(useProps[k]) }

        let carry: Int32
        if matchCount < compareLen {
            carry = targetArgmax[matchCount]  // correction
        } else if matchCount < K {
            carry = targetArgmax[matchCount]  // bonus (= argmax at slot K-1)
        } else {
            carry = targetArgmax[K - 1]  // defensive; unreachable given K-1 cap
        }

        // Commit: under the 11c protocol, verify does NOT write KV to the
        // persistent cache. commitAccepted is what writes the accepted-prefix
        // slices. Bumping currentPosition directly leaves the cache stale.
        let committed = matchCount + 1
        let committedTokens = Array(emitted.prefix(committed))
        try engine.commitAccepted(committedTokens)

        // Extract carry state from verify hidden states.
        // lastVerifyHiddenStates: (1, K, hidden) — matchCount indexes the
        // slot whose argmax is the carry (same as hiddenIdx used before,
        // just without clamping to K-1 since matchCount ≤ K-1 now).
        carryState = sliceVerifyHidden(at: matchCount, hidden: hidden)

        // Update metrics
        totalRounds += 1
        totalAccepted += matchCount
        totalEmitted += emitted.count

        SpecProfile.logBurst(
            engine: "mtp", cycle: totalRounds,
            draftMs: draftMs, verifyMs: verifyMs, commitMs: 0,
            accepted: matchCount, compareLen: compareLen,
            emitted: emitted.count, rolling: drafter.rollingAcceptance)

        // Carry becomes next burst's seed — NOT yielded from this cycle.
        nextID = carry

        return emitted
    }

    /// Whether speculation should be used for the next cycle.
    var shouldSpeculate: Bool {
        drafter.shouldSpeculate
    }

    /// Reset state for new conversation.
    func reset() {
        carryState = nil
        isBootstrapped = false
        totalRounds = 0
        totalAccepted = 0
        totalEmitted = 0
    }

    // MARK: - Private

    /// First call: run a normal decode to populate kv13/kv14 and warm up.
    private func bootstrapStep(nextID: inout Int32) throws -> [Int32] {
        let emitted = nextID
        let (newNext, targetStepMs) = try SpecProfile.time {
            try engine.predictStep(
                tokenID: Int(nextID), position: engine.currentPosition)
        }
        engine.currentPosition += 1
        nextID = Int32(newNext)
        isBootstrapped = true
        SpecProfile.logBootstrap(
            engine: "mtp", replayCount: 0,
            replayMs: 0, targetStepMs: targetStepMs)
        return [emitted]
    }

    /// Slice hidden state at index `k` from lastVerifyHiddenStates (1, K, hidden).
    private func sliceVerifyHidden(at k: Int, hidden: Int) -> MLMultiArray? {
        guard let hs = engine.lastVerifyHiddenStates else { return nil }
        guard let result = try? MLMultiArray(
            shape: [1, 1, NSNumber(value: hidden)], dataType: .float16) else { return nil }
        let src = hs.dataPointer.bindMemory(to: UInt16.self, capacity: hs.count)
        let dst = result.dataPointer.bindMemory(to: UInt16.self, capacity: hidden)
        memcpy(dst, src.advanced(by: k * hidden), hidden * MemoryLayout<UInt16>.stride)
        return result
    }

    /// Transpose the last two dimensions of a rank-4 MLMultiArray [1, 1, A, B] → [1, 1, B, A].
    /// Used for V caches: ChunkedEngine stores V as (1, 1, seq, hd) but the
    /// MTP drafter expects (1, 1, hd, seq) (Google TFLite pre-transposed layout).
    static func transposeLastTwoDims(_ a: MLMultiArray) throws -> MLMultiArray {
        let shape = a.shape.map { $0.intValue }
        guard shape.count == 4, shape[0] == 1, shape[1] == 1 else {
            throw SpeculativeError.verifyFailed(
                "transposeLastTwoDims: unexpected shape \(a.shape)")
        }
        let A = shape[2]
        let B = shape[3]
        let out = try MLMultiArray(
            shape: [1, 1, NSNumber(value: B), NSNumber(value: A)], dataType: .float16)
        let src = a.dataPointer.bindMemory(to: UInt16.self, capacity: A * B)
        let dst = out.dataPointer.bindMemory(to: UInt16.self, capacity: A * B)
        // src layout: src[i, j] = src[i*B + j], want dst[j, i] = dst[j*A + i]
        for i in 0..<A {
            for j in 0..<B {
                dst[j * A + i] = src[i * B + j]
            }
        }
        return out
    }

    /// Reshape RoPE cos/sin table for drafter I/O.
    /// ChunkedEngine.lookupRoPE returns shape `[1, 1, 1, dim]` (LLaMA-style
    /// duplicated halves). The MTP drafter (built by conversion/build_mtp_drafter.py)
    /// expects `[1, dim/2]` with only the unique half.
    /// Since `cos_full[:half] == cos_full[half:]` by construction, we copy
    /// the first half and reshape to rank 2.
    static func reshapeRoPEForDrafter(_ a: MLMultiArray) throws -> MLMultiArray {
        // Expect shape [1, 1, 1, dim]; output [1, dim/2]
        let dim = a.shape.last?.intValue ?? 0
        guard dim > 0 && dim % 2 == 0 else {
            throw SpeculativeError.verifyFailed(
                "reshapeRoPEForDrafter: unexpected dim \(dim) in shape \(a.shape)")
        }
        let half = dim / 2
        let out = try MLMultiArray(shape: [1, NSNumber(value: half)], dataType: .float16)
        let src = a.dataPointer.bindMemory(to: UInt16.self, capacity: dim)
        let dst = out.dataPointer.bindMemory(to: UInt16.self, capacity: half)
        memcpy(dst, src, half * MemoryLayout<UInt16>.stride)
        return out
    }
}
