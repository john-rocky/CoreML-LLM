//
//  MtpModuleStackEngine.swift
//  CoreMLLLM
//
//  Path C speculative engine: K=2 self-trained MTP modules (DeepSeek V3 style).
//
//  Per cycle:
//    1. module_0(L34_carry, embed(nextID)) → (h_0, d_0)      [predicts t+1]
//    2. module_1(h_0,        embed(d_0))    → (h_1, d_1)      [predicts t+2]
//    3. verify [nextID, d_0, d_1] through target (K=3)
//    4. accept/reject: compare d_0 vs argmax[0], d_1 vs argmax[1]
//       — if both match, argmax[2] is a FREE bonus token (trunk's own
//         prediction after seeing the accepted pair; no extra compute).
//    5. commit matched positions, update L34 carry from verify's hidden[0]
//    6. each module maintains own KV cache (W=128) across cycles
//
//  Differs from MtpSpeculativeEngine (MTP Path A, parked) in:
//    - Two CoreML models (one per module) instead of one autoregressive drafter
//    - Modules have own KV caches (vs Path A drafter reads target's kv13/kv14)
//    - carry state = L34 hidden (vs Path A's projected_activations)
//    - Drafts = 2 (vs Path A's K=3); verify position 2 used as bonus
//

import CoreML
import Foundation

/// Path C speculative decoding driver — sequential modules, frozen target.
public final class MtpModuleStackEngine: SpeculativeDrafterEngine {

    // MARK: - Models

    let engine: ChunkedEngine
    private let module0: MLModel
    private let module1: MLModel

    /// Number of drafter modules (K=2 for v1).
    public let K: Int = 2

    /// KV cache window per module (must match build_mtp_coreml.py default).
    private let W: Int = 128
    private let headDim: Int = 256
    private let numKvHeads: Int = 1

    // MARK: - State

    /// L34 hidden for next cycle's module_0 input. From verify's hidden_states[0]
    /// after each successful verify pass. Zero on bootstrap.
    private var carryL34: MLMultiArray?

    /// Per-module KV caches, persistent across speculation cycles.
    private var m0KvK: MLMultiArray
    private var m0KvV: MLMultiArray
    private var m1KvK: MLMultiArray
    private var m1KvV: MLMultiArray

    /// Per-module position counters (for RoPE and update_idx slot).
    private var m0Pos: Int = 0
    private var m1Pos: Int = 0

    private var isBootstrapped = false

    // MARK: - Metrics

    private(set) var totalRounds = 0
    private(set) var totalAccepted = 0
    private(set) var totalEmitted = 0

    public var acceptanceRate: Double {
        totalRounds == 0 ? 0 : Double(totalAccepted) / Double(totalRounds * K)
    }
    public var tokensPerRound: Double {
        totalRounds == 0 ? 0 : Double(totalEmitted) / Double(totalRounds)
    }

    /// Always speculate (for simplicity in v1 — no rolling acceptance gate).
    public var shouldSpeculate: Bool { true }

    // MARK: - Init

    init(engine: ChunkedEngine, module0URL: URL, module1URL: URL,
         configuration: MLModelConfiguration? = nil) throws {
        let cfg = configuration ?? {
            let c = MLModelConfiguration()
            c.computeUnits = .cpuAndNeuralEngine
            return c
        }()
        self.engine = engine
        self.module0 = try MLModel(contentsOf: module0URL, configuration: cfg)
        self.module1 = try MLModel(contentsOf: module1URL, configuration: cfg)

        precondition(engine.hasVerify,
                     "Path C speculation requires verify chunks (engine.verifyK=3)")
        precondition(engine.verifyK == 3,
                     "Target verify was built for K=3; K=2 drafts + verify[2] as bonus")

        // Allocate zero-initialized KV caches for both modules.
        // Inlined (not a nested func) because the nested func would capture
        // self.numKvHeads etc., which Swift considers a use-before-init.
        let kvNKV = 1    // matches self.numKvHeads
        let kvW = 128    // matches self.W
        let kvHD = 256   // matches self.headDim
        let kvShape: [NSNumber] = [
            1, NSNumber(value: kvNKV),
            NSNumber(value: kvW), NSNumber(value: kvHD),
        ]
        let kvBytes = kvNKV * kvW * kvHD * MemoryLayout<UInt16>.stride
        self.m0KvK = try MLMultiArray(shape: kvShape, dataType: .float16)
        memset(self.m0KvK.dataPointer, 0, kvBytes)
        self.m0KvV = try MLMultiArray(shape: kvShape, dataType: .float16)
        memset(self.m0KvV.dataPointer, 0, kvBytes)
        self.m1KvK = try MLMultiArray(shape: kvShape, dataType: .float16)
        memset(self.m1KvK.dataPointer, 0, kvBytes)
        self.m1KvV = try MLMultiArray(shape: kvShape, dataType: .float16)
        memset(self.m1KvV.dataPointer, 0, kvBytes)
    }

    // MARK: - Public entry

    /// Execute one speculative cycle. On first call runs a normal decode
    /// (bootstrap to warm target KV). Subsequent calls produce K+1 tokens
    /// per cycle on average (1 + matched drafts + bonus if all match).
    public func speculateStep(nextID: inout Int32) throws -> [Int32] {
        if !isBootstrapped {
            return try bootstrapStep(nextID: &nextID)
        }

        let pos = engine.currentPosition
        let hidden = engine.config.hiddenSize

        // Initialize L34 carry with zeros on first real speculative cycle.
        if carryL34 == nil {
            carryL34 = try MLMultiArray(
                shape: [1, 1, NSNumber(value: hidden)], dataType: .float16)
            memset(carryL34!.dataPointer, 0,
                   hidden * MemoryLayout<UInt16>.stride)
        }

        // -----------------------------
        // Module 0: predict token at pos P+1
        // -----------------------------
        let embedNextID = try engine.lookupRawEmbed(nextID)
        let (d0Token, h0Out, m0Kk, m0Kv) = try runModule(
            model: module0,
            hiddenPrev: carryL34!,
            embedToken: embedNextID,
            modulePos: m0Pos,
            kvKin: m0KvK, kvVin: m0KvV)

        m0KvK = m0Kk
        m0KvV = m0Kv
        m0Pos += 1

        // -----------------------------
        // Module 1: predict token at pos P+2, given module_0's output
        // -----------------------------
        let embedD0 = try engine.lookupRawEmbed(d0Token)
        let (d1Token, _, m1Kk, m1Kv) = try runModule(
            model: module1,
            hiddenPrev: h0Out,
            embedToken: embedD0,
            modulePos: m1Pos,
            kvKin: m1KvK, kvVin: m1KvV)

        m1KvK = m1Kk
        m1KvV = m1Kv
        m1Pos += 1

        // -----------------------------
        // Verify [nextID, d_0, d_1] at positions [P, P+1, P+2]
        // -----------------------------
        let verifyTokens: [Int32] = [nextID, d0Token, d1Token]
        let targetArgmax = try engine.verifyCandidates(
            tokens: verifyTokens, startPosition: pos)
        // targetArgmax[0] verifies d_0, [1] verifies d_1, [2] is bonus

        // -----------------------------
        // Accept/reject
        // -----------------------------
        var emitted: [Int32] = [nextID]
        emitted.reserveCapacity(K + 2)  // nextID + K drafts + 1 bonus
        var matchCount = 0

        // Compare draft k against targetArgmax[k]
        if d0Token == targetArgmax[0] {
            emitted.append(d0Token)
            matchCount += 1
            if d1Token == targetArgmax[1] {
                emitted.append(d1Token)
                matchCount += 1
                // Both drafts matched — trunk's argmax[2] is a free bonus token.
                // It's trunk's genuine prediction after seeing [nextID, d_0, d_1],
                // which equals trunk's natural next continuation (not a draft guess).
                emitted.append(targetArgmax[2])
                // Do NOT count bonus in matchCount (it's not a draft match).
            } else {
                emitted.append(targetArgmax[1])  // correction at pos P+2
            }
        } else {
            emitted.append(targetArgmax[0])  // correction at pos P+1
        }

        // -----------------------------
        // Commit
        // -----------------------------
        // Verify wrote KV at K+1 = 3 positions (P, P+1, P+2). Commit the
        // verified prefix (positions with correct KV):
        //   - 0 drafts match: positions [P] verified → commit 1
        //   - 1 draft match:  positions [P, P+1] verified → commit 2
        //   - 2 drafts match: positions [P, P+1, P+2] verified → commit 3
        // The correction or bonus at position P+committed is NOT committed;
        // it becomes nextID for the next cycle (its KV written then).
        let committed = matchCount + 1
        engine.currentPosition = pos + committed

        // -----------------------------
        // Extract L34 carry for next cycle
        // -----------------------------
        // verify's hidden_states has shape (1, K_verify=3, hidden). We want
        // the hidden state at the LAST committed position in the verify output.
        let hiddenIdx = min(matchCount, 2)  // verify has 3 positions (0..2)
        carryL34 = sliceVerifyHidden(at: hiddenIdx, hidden: hidden)

        // -----------------------------
        // Metrics + update nextID
        // -----------------------------
        totalRounds += 1
        totalAccepted += matchCount
        totalEmitted += emitted.count

        nextID = emitted.last!
        return emitted
    }

    public func reset() {
        carryL34 = nil
        m0Pos = 0
        m1Pos = 0
        isBootstrapped = false
        totalRounds = 0
        totalAccepted = 0
        totalEmitted = 0
        // Zero the KV caches
        for arr in [m0KvK, m0KvV, m1KvK, m1KvV] {
            memset(arr.dataPointer, 0, arr.count * MemoryLayout<UInt16>.stride)
        }
    }

    // MARK: - Private helpers

    private func bootstrapStep(nextID: inout Int32) throws -> [Int32] {
        let emitted = nextID
        let newNext = try engine.predictStep(
            tokenID: Int(nextID), position: engine.currentPosition)
        engine.currentPosition += 1
        nextID = Int32(newNext)
        isBootstrapped = true
        return [emitted]
    }

    /// Run one module's forward pass. Returns (top-1 token, hidden_out, KV_out_k, KV_out_v).
    private func runModule(
        model: MLModel,
        hiddenPrev: MLMultiArray,
        embedToken: MLMultiArray,
        modulePos: Int,
        kvKin: MLMultiArray,
        kvVin: MLMultiArray
    ) throws -> (Int32, MLMultiArray, MLMultiArray, MLMultiArray) {
        // Build RoPE: cos/sin for SWA head_dim=256, at absolute position modulePos.
        // These are (1, 128) tensors — we slice from the target's RoPE tables
        // (same precomputed tables as target chunks, first half).
        let cos = try sliceAndReshape(engine.lookupCosSWA(position: modulePos), halfDim: 128)
        let sin = try sliceAndReshape(engine.lookupSinSWA(position: modulePos), halfDim: 128)

        // Causal mask over W=128 slots. Position modulePos can attend to
        // slots [0 .. modulePos mod W] inclusive (right-aligned within window).
        let mask = try makeModuleMask(modulePos: modulePos)

        // update_idx: one-hot at slot (modulePos % W).
        let updateIdx = try makeUpdateIdx(slot: modulePos % W)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_prev": hiddenPrev,
            "embed_token": embedToken,
            "kv_k_in": kvKin,
            "kv_v_in": kvVin,
            "cos": cos,
            "sin": sin,
            "mask": mask,
            "update_idx": updateIdx,
        ])
        let out = try model.prediction(from: input)

        guard let topIds = out.featureValue(for: "top_k_indices")?.multiArrayValue,
              let hOut = out.featureValue(for: "hidden_out")?.multiArrayValue,
              let kOut = out.featureValue(for: "kv_k_out")?.multiArrayValue,
              let vOut = out.featureValue(for: "kv_v_out")?.multiArrayValue
        else {
            throw SpeculativeError.verifyFailed("MTP module output missing expected features")
        }

        let tokenId: Int32 = topIds.dataPointer
            .bindMemory(to: Int32.self, capacity: 1).pointee
        return (tokenId, hOut, kOut, vOut)
    }

    /// Slice first `halfDim` values from target's RoPE (duplicated-halves convention)
    /// and reshape to (1, halfDim).
    private func sliceAndReshape(_ src: MLMultiArray, halfDim: Int) throws -> MLMultiArray {
        let result = try MLMultiArray(
            shape: [1, NSNumber(value: halfDim)], dataType: .float16)
        let srcPtr = src.dataPointer.bindMemory(to: UInt16.self, capacity: src.count)
        let dstPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: halfDim)
        memcpy(dstPtr, srcPtr, halfDim * MemoryLayout<UInt16>.stride)
        return result
    }

    /// Causal mask for module attention. Shape (1, 1, 1, W).
    /// Position modulePos can attend to the most recent min(modulePos+1, W) slots.
    private func makeModuleMask(modulePos: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: W)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: W)
        // All slots are valid once modulePos >= W (circular buffer full).
        // Before that, only slots [0 .. modulePos] are valid (left-aligned).
        let validCount = min(modulePos + 1, W)
        for i in 0..<W {
            mp[i] = i < validCount ? 0 : 0xFC00  // 0 or -inf fp16
        }
        return mask
    }

    /// One-hot update index. Shape (1, 1, W, 1). fp16 = 1.0 at slot, 0 elsewhere.
    private func makeUpdateIdx(slot: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: W), 1], dataType: .float16)
        let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: W)
        memset(ptr, 0, W * MemoryLayout<UInt16>.stride)
        ptr[slot] = 0x3C00  // 1.0 fp16
        return arr
    }

    /// Slice hidden state at index `k` from lastVerifyHiddenStates (1, K_verify, hidden).
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
