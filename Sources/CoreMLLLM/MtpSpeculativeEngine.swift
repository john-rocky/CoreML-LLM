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
              let kv13V = engine.lastKV13V,
              let kv14K = engine.lastKV14K,
              let kv14V = engine.lastKV14V
        else {
            throw SpeculativeError.verifyFailed("kv13/kv14 not available for MTP drafter")
        }

        // Build mask ONCE per cycle at the last committed position.
        // The drafter reads target KV (positions 0..pos-1), so mask allows 0..pos-1.
        let maskPos = pos - 1
        let maskSwa = try engine.makeDrafterSWAMask(position: max(maskPos, 0))
        let maskFull = try engine.makeDrafterFullMask(position: max(maskPos, 0))

        // Draft K tokens with per-step RoPE updates
        var proposals = [Int32]()
        proposals.reserveCapacity(K)
        var embedToken = try engine.lookupRawEmbed(nextID)
        var projAct = carryState!

        for k in 0..<K {
            let draftPos = pos + k
            let cosSwa = try engine.lookupCosSWA(position: draftPos)
            let sinSwa = try engine.lookupSinSWA(position: draftPos)
            let cosFull = try engine.lookupCosFull(position: draftPos)
            let sinFull = try engine.lookupSinFull(position: draftPos)

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

        // Verify [nextID, proposals[0..K-2]] at currentPosition
        var verifyTokens = [nextID]
        verifyTokens.append(contentsOf: proposals.dropLast())
        let targetArgmax = try engine.verifyCandidates(
            tokens: verifyTokens, startPosition: pos)

        // Accept/reject: greedy comparison
        var emitted = [Int32]()
        emitted.reserveCapacity(K + 1)
        emitted.append(nextID) // always emit tTokNext
        var matchCount = 0
        for k in 0..<K {
            if proposals[k] == targetArgmax[k] {
                emitted.append(proposals[k])
                matchCount += 1
            } else {
                emitted.append(targetArgmax[k]) // correction/bonus
                break
            }
        }

        // Commit: advance position by the verified prefix only.
        // The correction/bonus token is NOT committed — its KV will be written
        // in the next cycle's verify pass (overwriting any stale entry).
        let committed = matchCount + 1 // tTokNext + matched drafts
        engine.currentPosition = pos + committed

        // Extract carry state from verify hidden states.
        // lastVerifyHiddenStates: (1, K, hidden) — use the last committed index.
        let hiddenIdx = min(matchCount, K - 1)
        carryState = sliceVerifyHidden(at: hiddenIdx, hidden: hidden)

        // Update metrics
        totalRounds += 1
        totalAccepted += matchCount
        totalEmitted += emitted.count

        // nextID = last emitted token (correction or bonus, not yet committed)
        nextID = emitted.last!

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
        let newNext = try engine.predictStep(
            tokenID: Int(nextID), position: engine.currentPosition)
        engine.currentPosition += 1
        nextID = Int32(newNext)
        isBootstrapped = true
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
}
