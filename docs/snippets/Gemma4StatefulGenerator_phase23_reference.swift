// Gemma 4 E2B stateful generator — MLState + slice_update path.
// Mirrors Qwen3VL2BStatefulGenerator.swift's structure adapted for Gemma 4's
// dual-attention (sliding + full) topology and per-layer embeddings.
//
// Artifacts on disk (per build_gemma4_e2b_stateful_chunks.py output):
//   Documents/Models/gemma4-e2b-stateful/gemma4_e2b_stateful_chunks/
//     embed_tokens_q8.bin / embed_tokens_scales.bin
//     embed_tokens_per_layer_q8.bin / embed_tokens_per_layer_scales.bin
//     per_layer_projection.bin / per_layer_norm_weight.bin
//     cos_sliding.npy / sin_sliding.npy / cos_full.npy / sin_full.npy
//     model_config.json
//     chunk_1.mlpackage  (own KV state, computes PLE)
//     chunk_2.mlpackage  (own KV state, emits kv13/kv14)
//     chunk_3.mlpackage  (stateless, reads kv13/14)
//     chunk_4.mlpackage  (stateless, reads kv13/14, lm_head + argmax)
//
// Per-chunk inputs (T=1 decode):
//   chunk_1: hidden_states (1,1,1536), causal_mask_full (1,1,1,ctx),
//            causal_mask_sliding (1,1,1,W), per_layer_raw (1,1,N*PLD),
//            cos_s/sin_s (1,1,1,head_dim), cos_f/sin_f (1,1,1,global_head_dim),
//            current_pos (1,) int32, ring_pos (1,) int32
//            states: kv_cache_sliding, kv_cache_full
//   chunk_2: same minus per_layer_raw, plus per_layer_combined input
//            states: kv_cache_sliding, kv_cache_full (own, separate from chunk_1)
//            outputs: hidden_states_out, kv13_k, kv13_v, kv14_k, kv14_v
//   chunk_3: stateless, takes hidden_states + masks + RoPE + per_layer_combined
//            + kv13/14 from chunk_2
//   chunk_4: same as chunk_3, outputs token_id (int32), token_logit (fp32),
//            hidden_normed (fp16)
//
// Phase 1 status: scaffold only. stepPredict and generate are stubs marked
// [TODO]. The plan is to fill stepPredict next, then generate (with cross-
// turn KV reuse), then prewarm + audit. See docs/SESSION_2026_04_25_*.md.

import Accelerate
import CoreML
import Foundation


@Observable
final class Gemma4StatefulGenerator {
    struct Config {
        let ctx: Int                    // state_length / context length (e.g. 512)
        let slidingWindow: Int          // 512 for E2B
        let vocab: Int                  // 262_144
        let hiddenSize: Int             // 1536
        let numLayers: Int              // 35 (E2B)
        let numKVHeads: Int             // 1
        let headDim: Int                // 256 (sliding)
        let globalHeadDim: Int          // 512 (full)
        let perLayerInputDim: Int       // 256 (PLD)
        let slidingRopeTheta: Float     // 10_000
        let fullRopeTheta: Float        // 1_000_000
        let fullPartialRotaryFactor: Float // 0.25
        // Gemma 4 embed scaling (baked into each EmbeddingLookup's scale arg).
        // Inputs: int8 weight × per-row fp16 scale × (this) → fp16 output.
        let embedScale: Float           // sqrt(hiddenSize) for E2B = 39.1918...
        let perLayerEmbedScale: Float   // sqrt(perLayerInputDim) = 16.0
        let computeUnits: MLComputeUnits

        static let e2b = Config(
            ctx: 512, slidingWindow: 512, vocab: 262_144,
            hiddenSize: 1536, numLayers: 35,
            numKVHeads: 1, headDim: 256, globalHeadDim: 512,
            perLayerInputDim: 256,
            slidingRopeTheta: 10_000, fullRopeTheta: 1_000_000,
            fullPartialRotaryFactor: 0.25,
            embedScale: 39.191835884530853,   // sqrt(1536)
            perLayerEmbedScale: 16.0,          // sqrt(256)
            computeUnits: .cpuAndNeuralEngine)
    }

    // MARK: - Public state

    var status = "Idle"
    var running = false
    var outputText = ""
    var stats = ""
    var auditText = ""

    private var cfg = Config.e2b

    // Models — chunks 1/2 carry MLState; chunks 3/4 are stateless.
    private var chunk1: MLModel?
    private var chunk2: MLModel?
    private var chunk3: MLModel?
    private var chunk4: MLModel?

    // Embed sidecars — reuse existing EmbeddingLookup from Sources/CoreMLLLM
    // for the int8 × fp16-scale dequant path (vDSP/vImage vectorized).
    private var embedTokens: EmbeddingLookup?
    private var embedPerLayer: EmbeddingLookup?

    // Reusable per-step buffers.
    private var reusableHidden: MLMultiArray!
    private var reusablePerLayerRaw: MLMultiArray!
    private var reusableCausalMaskFull: MLMultiArray!
    private var reusableCausalMaskSliding: MLMultiArray!
    private var reusableCosS: MLMultiArray!
    private var reusableSinS: MLMultiArray!
    private var reusableCosF: MLMultiArray!
    private var reusableSinF: MLMultiArray!
    private var reusableCurrentPos: MLMultiArray!
    private var reusableRingPos: MLMultiArray!

    private var fvHidden: MLFeatureValue!
    private var fvPerLayerRaw: MLFeatureValue!
    private var fvCausalMaskFull: MLFeatureValue!
    private var fvCausalMaskSliding: MLFeatureValue!
    private var fvCosS: MLFeatureValue!
    private var fvSinS: MLFeatureValue!
    private var fvCosF: MLFeatureValue!
    private var fvSinF: MLFeatureValue!
    private var fvCurrentPos: MLFeatureValue!
    private var fvRingPos: MLFeatureValue!

    // Precomputed RoPE tables (built in init).
    private var cosSlidingTable: [Float] = []  // (maxLen, head_dim)
    private var sinSlidingTable: [Float] = []
    private var cosFullTable: [Float] = []     // (maxLen, global_head_dim)
    private var sinFullTable: [Float] = []

    // Cross-turn KV reuse — same pattern as Qwen3VL2BStatefulGenerator.
    // Both chunk_1 and chunk_2 carry state; we persist both.
    private var persistedState1: MLState?
    private var persistedState2: MLState?
    private var persistedInputIds: [Int32] = []
    private var persistedPosition: Int = 0

    // Per-step timing diagnostics (filled by stepPredict).
    private var c1Ms: Double = 0
    private var c2Ms: Double = 0
    private var c3Ms: Double = 0
    private var c4Ms: Double = 0
    private var embedMs: Double = 0
    private var ropeFillMs: Double = 0
    private var timedSteps: Int = 0

    init(cfg: Config = .e2b) {
        self.cfg = cfg
        buildRopeTables()
        allocBuffers()
    }

    deinit {
        // EmbeddingLookup uses Data(.mappedIfSafe) — releases on deinit.
    }

    // MARK: - RoPE tables

    private func buildRopeTables() {
        let maxLen = cfg.ctx * 2
        // Sliding RoPE: full rotation, theta=10_000, head_dim=256
        let halfS = cfg.headDim / 2
        var invFreqS = [Float](repeating: 0, count: halfS)
        for i in 0..<halfS {
            invFreqS[i] = 1.0 / powf(cfg.slidingRopeTheta,
                                     Float(2 * i) / Float(cfg.headDim))
        }
        cosSlidingTable = [Float](repeating: 0, count: maxLen * cfg.headDim)
        sinSlidingTable = [Float](repeating: 0, count: maxLen * cfg.headDim)
        for t in 0..<maxLen {
            for i in 0..<halfS {
                let a = Float(t) * invFreqS[i]
                let c = cosf(a), s = sinf(a)
                cosSlidingTable[t * cfg.headDim + i] = c
                cosSlidingTable[t * cfg.headDim + i + halfS] = c
                sinSlidingTable[t * cfg.headDim + i] = s
                sinSlidingTable[t * cfg.headDim + i + halfS] = s
            }
        }
        // Full RoPE: proportional, theta=1M, head_dim=512.
        // partial_rotary_factor=0.25 → only first 128 dims are rotated; rest 0/1
        // (handled by the model graph, we still emit the full 512-wide table).
        let halfF = cfg.globalHeadDim / 2
        var invFreqF = [Float](repeating: 0, count: halfF)
        for i in 0..<halfF {
            invFreqF[i] = 1.0 / powf(cfg.fullRopeTheta,
                                     Float(2 * i) / Float(cfg.globalHeadDim))
        }
        cosFullTable = [Float](repeating: 0, count: maxLen * cfg.globalHeadDim)
        sinFullTable = [Float](repeating: 0, count: maxLen * cfg.globalHeadDim)
        for t in 0..<maxLen {
            for i in 0..<halfF {
                let a = Float(t) * invFreqF[i]
                let c = cosf(a), s = sinf(a)
                cosFullTable[t * cfg.globalHeadDim + i] = c
                cosFullTable[t * cfg.globalHeadDim + i + halfF] = c
                sinFullTable[t * cfg.globalHeadDim + i] = s
                sinFullTable[t * cfg.globalHeadDim + i + halfF] = s
            }
        }
    }

    // MARK: - Buffer allocation

    private func allocBuffers() {
        let plRaw: [NSNumber] = [
            1, 1, NSNumber(value: cfg.numLayers * cfg.perLayerInputDim)
        ]
        reusableHidden = try! MLMultiArray(
            shape: [1, 1, NSNumber(value: cfg.hiddenSize)], dataType: .float16)
        reusablePerLayerRaw = try! MLMultiArray(shape: plRaw, dataType: .float16)
        reusableCausalMaskFull = try! MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: cfg.ctx)], dataType: .float16)
        reusableCausalMaskSliding = try! MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: cfg.slidingWindow)], dataType: .float16)
        reusableCosS = try! MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: cfg.headDim)], dataType: .float16)
        reusableSinS = try! MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: cfg.headDim)], dataType: .float16)
        reusableCosF = try! MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: cfg.globalHeadDim)], dataType: .float16)
        reusableSinF = try! MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: cfg.globalHeadDim)], dataType: .float16)
        reusableCurrentPos = try! MLMultiArray(shape: [1], dataType: .int32)
        reusableRingPos = try! MLMultiArray(shape: [1], dataType: .int32)

        fvHidden = MLFeatureValue(multiArray: reusableHidden)
        fvPerLayerRaw = MLFeatureValue(multiArray: reusablePerLayerRaw)
        fvCausalMaskFull = MLFeatureValue(multiArray: reusableCausalMaskFull)
        fvCausalMaskSliding = MLFeatureValue(multiArray: reusableCausalMaskSliding)
        fvCosS = MLFeatureValue(multiArray: reusableCosS)
        fvSinS = MLFeatureValue(multiArray: reusableSinS)
        fvCosF = MLFeatureValue(multiArray: reusableCosF)
        fvSinF = MLFeatureValue(multiArray: reusableSinF)
        fvCurrentPos = MLFeatureValue(multiArray: reusableCurrentPos)
        fvRingPos = MLFeatureValue(multiArray: reusableRingPos)
    }

    // MARK: - Embed lookup helpers

    private func loadEmbeddings(folder: URL) throws {
        embedTokens = try EmbeddingLookup(
            dataURL: folder.appendingPathComponent("embed_tokens_q8.bin"),
            scalesURL: folder.appendingPathComponent("embed_tokens_scales.bin"),
            vocabSize: cfg.vocab, dim: cfg.hiddenSize, scale: cfg.embedScale)
        embedPerLayer = try EmbeddingLookup(
            dataURL: folder.appendingPathComponent("embed_tokens_per_layer_q8.bin"),
            scalesURL: folder.appendingPathComponent("embed_tokens_per_layer_scales.bin"),
            vocabSize: cfg.vocab, dim: cfg.numLayers * cfg.perLayerInputDim,
            scale: cfg.perLayerEmbedScale)
    }

    /// Look up token embedding into reusableHidden.
    private func embedLookup(token: Int32) {
        guard let e = embedTokens else { return }
        let raw = e.lookupRaw(Int(token))
        let dst = reusableHidden.dataPointer.bindMemory(
            to: UInt16.self, capacity: cfg.hiddenSize)
        raw.withUnsafeBufferPointer { src in
            memcpy(dst, src.baseAddress, cfg.hiddenSize * 2)
        }
    }

    /// Look up per-layer raw embedding into reusablePerLayerRaw.
    private func perLayerEmbedLookup(token: Int32) {
        guard let e = embedPerLayer else { return }
        let total = cfg.numLayers * cfg.perLayerInputDim
        let raw = e.lookupRaw(Int(token))
        let dst = reusablePerLayerRaw.dataPointer.bindMemory(
            to: UInt16.self, capacity: total)
        raw.withUnsafeBufferPointer { src in
            memcpy(dst, src.baseAddress, total * 2)
        }
    }

    // MARK: - RoPE / mask / position fill

    private func fillRopeAt(_ position: Int) {
        let p = min(max(position, 0), cfg.ctx * 2 - 1)
        // Sliding (head_dim = 256)
        let cosSDst = reusableCosS.dataPointer.bindMemory(to: UInt16.self, capacity: cfg.headDim)
        let sinSDst = reusableSinS.dataPointer.bindMemory(to: UInt16.self, capacity: cfg.headDim)
        for i in 0..<cfg.headDim {
            cosSDst[i] = Float16(cosSlidingTable[p * cfg.headDim + i]).bitPattern
            sinSDst[i] = Float16(sinSlidingTable[p * cfg.headDim + i]).bitPattern
        }
        // Full (global_head_dim = 512)
        let cosFDst = reusableCosF.dataPointer.bindMemory(to: UInt16.self, capacity: cfg.globalHeadDim)
        let sinFDst = reusableSinF.dataPointer.bindMemory(to: UInt16.self, capacity: cfg.globalHeadDim)
        for i in 0..<cfg.globalHeadDim {
            cosFDst[i] = Float16(cosFullTable[p * cfg.globalHeadDim + i]).bitPattern
            sinFDst[i] = Float16(sinFullTable[p * cfg.globalHeadDim + i]).bitPattern
        }
    }

    private func fillCausalMasks(_ position: Int) {
        // Apple recommends -1e4 over -inf for fp16 mask composition.
        let neg1e4 = Float16(-10_000.0).bitPattern
        // Full mask: length=ctx, 0 for slots ≤ position, -1e4 otherwise.
        let mFull = reusableCausalMaskFull.dataPointer.bindMemory(
            to: UInt16.self, capacity: cfg.ctx)
        let pFull = min(max(position, 0), cfg.ctx - 1)
        for i in 0..<cfg.ctx {
            mFull[i] = (i <= pFull) ? 0 : neg1e4
        }
        // Sliding mask: length=W. With ring_pos = position % W, sliding cache
        // holds the last W tokens. Per the model graph, the mask is laid out
        // ringbuffer-aligned: positions ≤ current within the W window are
        // attendable, beyond are -1e4. For the first W steps (position < W),
        // future slots in the ring haven't been written yet — mask them out.
        let mSld = reusableCausalMaskSliding.dataPointer.bindMemory(
            to: UInt16.self, capacity: cfg.slidingWindow)
        let pSld = min(position, cfg.slidingWindow - 1)
        // For early decode: only positions [0, position] are valid.
        // Once position ≥ W, all W slots are valid (ring fully written).
        if position < cfg.slidingWindow {
            for i in 0..<cfg.slidingWindow {
                mSld[i] = (i <= pSld) ? 0 : neg1e4
            }
        } else {
            for i in 0..<cfg.slidingWindow {
                mSld[i] = 0
            }
        }
    }

    private func setPositions(_ position: Int) {
        let cp = reusableCurrentPos.dataPointer
            .bindMemory(to: Int32.self, capacity: 1)
        let rp = reusableRingPos.dataPointer
            .bindMemory(to: Int32.self, capacity: 1)
        cp[0] = Int32(position)
        rp[0] = Int32(position % cfg.slidingWindow)
    }

    // MARK: - Resolve model directory

    var modelFolderOverride: URL?

    private func resolveURLs()
        -> (chunk1: URL, chunk2: URL, chunk3: URL, chunk4: URL,
            sidecars: URL)?
    {
        let subdir = "gemma4_e2b_stateful_chunks"
        let fm = FileManager.default
        func resolveOne(_ dir: URL, _ base: String) -> URL? {
            let mlc = dir.appendingPathComponent("\(base).mlmodelc")
            if fm.fileExists(atPath: mlc.path) { return mlc }
            let pkg = dir.appendingPathComponent("\(base).mlpackage")
            if fm.fileExists(atPath: pkg.path) {
                return try? MLModel.compileModel(at: pkg)
            }
            return nil
        }
        func resolve(_ base: URL)
            -> (chunk1: URL, chunk2: URL, chunk3: URL, chunk4: URL,
                sidecars: URL)?
        {
            let dir = base.appendingPathComponent(subdir)
            let q8 = dir.appendingPathComponent("embed_tokens_q8.bin")
            guard fm.fileExists(atPath: q8.path) else { return nil }
            guard
                let c1 = resolveOne(dir, "chunk_1"),
                let c2 = resolveOne(dir, "chunk_2"),
                let c3 = resolveOne(dir, "chunk_3"),
                let c4 = resolveOne(dir, "chunk_4")
            else { return nil }
            return (c1, c2, c3, c4, dir)
        }
        if let folder = modelFolderOverride, let r = resolve(folder) { return r }
        if let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask,
                                  appropriateFor: nil, create: false),
           let r = resolve(docs) { return r }
        let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask,
                                appropriateFor: nil, create: false)
        return docs.flatMap {
            resolve($0.appendingPathComponent("Models/gemma4-e2b-stateful"))
        }
    }

    // MARK: - Load

    func resetPersistedState() {
        persistedState1 = nil
        persistedState2 = nil
        persistedInputIds = []
        persistedPosition = 0
    }

    func load() throws {
        guard let r = resolveURLs() else {
            throw NSError(domain: "Gemma4Stateful", code: 40,
                userInfo: [NSLocalizedDescriptionKey:
                    "gemma4_e2b_stateful_chunks/{embed_tokens_*.bin, chunk_1..4} "
                    + "not found in Documents/ or Documents/Models/gemma4-e2b-stateful/"])
        }
        resetPersistedState()
        try loadEmbeddings(folder: r.sidecars)
        let mcfg = MLModelConfiguration()
        mcfg.computeUnits = cfg.computeUnits
        chunk1 = try MLModel(contentsOf: r.chunk1, configuration: mcfg)
        chunk2 = try MLModel(contentsOf: r.chunk2, configuration: mcfg)
        chunk3 = try MLModel(contentsOf: r.chunk3, configuration: mcfg)
        chunk4 = try MLModel(contentsOf: r.chunk4, configuration: mcfg)
        status = "Loaded: chunk_1..4 + sidecars, units=\(cfg.computeUnits.rawValue)"
    }

    // MARK: - Feature providers

    /// Provider for chunk_1 (own state, takes per_layer_raw).
    private final class Chunk1Provider: NSObject, MLFeatureProvider {
        let h, mF, mS, plr, cs, ss, cf, sf, cp, rp: MLFeatureValue
        let featureNames: Set<String> = [
            "hidden_states", "causal_mask_full", "causal_mask_sliding",
            "per_layer_raw", "cos_s", "sin_s", "cos_f", "sin_f",
            "current_pos", "ring_pos",
        ]
        init(h: MLFeatureValue, mF: MLFeatureValue, mS: MLFeatureValue,
             plr: MLFeatureValue, cs: MLFeatureValue, ss: MLFeatureValue,
             cf: MLFeatureValue, sf: MLFeatureValue,
             cp: MLFeatureValue, rp: MLFeatureValue) {
            self.h = h; self.mF = mF; self.mS = mS; self.plr = plr
            self.cs = cs; self.ss = ss; self.cf = cf; self.sf = sf
            self.cp = cp; self.rp = rp; super.init()
        }
        func featureValue(for n: String) -> MLFeatureValue? {
            switch n {
            case "hidden_states":         return h
            case "causal_mask_full":      return mF
            case "causal_mask_sliding":   return mS
            case "per_layer_raw":         return plr
            case "cos_s":                 return cs
            case "sin_s":                 return ss
            case "cos_f":                 return cf
            case "sin_f":                 return sf
            case "current_pos":           return cp
            case "ring_pos":              return rp
            default: return nil
            }
        }
    }

    /// Provider for chunk_2 (own state, takes per_layer_combined instead of raw).
    private final class Chunk2Provider: NSObject, MLFeatureProvider {
        let h, mF, mS, plc, cs, ss, cf, sf, cp, rp: MLFeatureValue
        let featureNames: Set<String> = [
            "hidden_states", "causal_mask_full", "causal_mask_sliding",
            "per_layer_combined", "cos_s", "sin_s", "cos_f", "sin_f",
            "current_pos", "ring_pos",
        ]
        init(h: MLFeatureValue, mF: MLFeatureValue, mS: MLFeatureValue,
             plc: MLFeatureValue, cs: MLFeatureValue, ss: MLFeatureValue,
             cf: MLFeatureValue, sf: MLFeatureValue,
             cp: MLFeatureValue, rp: MLFeatureValue) {
            self.h = h; self.mF = mF; self.mS = mS; self.plc = plc
            self.cs = cs; self.ss = ss; self.cf = cf; self.sf = sf
            self.cp = cp; self.rp = rp; super.init()
        }
        func featureValue(for n: String) -> MLFeatureValue? {
            switch n {
            case "hidden_states":         return h
            case "causal_mask_full":      return mF
            case "causal_mask_sliding":   return mS
            case "per_layer_combined":    return plc
            case "cos_s":                 return cs
            case "sin_s":                 return ss
            case "cos_f":                 return cf
            case "sin_f":                 return sf
            case "current_pos":           return cp
            case "ring_pos":              return rp
            default: return nil
            }
        }
    }

    /// Provider for chunk_3 / chunk_4 (stateless, takes kv13/14).
    private final class SharedChunkProvider: NSObject, MLFeatureProvider {
        let h, mF, mS, plc, cs, ss, cf, sf: MLFeatureValue
        let kv13K, kv13V, kv14K, kv14V: MLFeatureValue
        let featureNames: Set<String> = [
            "hidden_states", "causal_mask_full", "causal_mask_sliding",
            "per_layer_combined", "cos_s", "sin_s", "cos_f", "sin_f",
            "kv13_k", "kv13_v", "kv14_k", "kv14_v",
        ]
        init(h: MLFeatureValue, mF: MLFeatureValue, mS: MLFeatureValue,
             plc: MLFeatureValue, cs: MLFeatureValue, ss: MLFeatureValue,
             cf: MLFeatureValue, sf: MLFeatureValue,
             kv13K: MLFeatureValue, kv13V: MLFeatureValue,
             kv14K: MLFeatureValue, kv14V: MLFeatureValue) {
            self.h = h; self.mF = mF; self.mS = mS; self.plc = plc
            self.cs = cs; self.ss = ss; self.cf = cf; self.sf = sf
            self.kv13K = kv13K; self.kv13V = kv13V
            self.kv14K = kv14K; self.kv14V = kv14V
            super.init()
        }
        func featureValue(for n: String) -> MLFeatureValue? {
            switch n {
            case "hidden_states":         return h
            case "causal_mask_full":      return mF
            case "causal_mask_sliding":   return mS
            case "per_layer_combined":    return plc
            case "cos_s":                 return cs
            case "sin_s":                 return ss
            case "cos_f":                 return cf
            case "sin_f":                 return sf
            case "kv13_k":                return kv13K
            case "kv13_v":                return kv13V
            case "kv14_k":                return kv14K
            case "kv14_v":                return kv14V
            default: return nil
            }
        }
    }

    // MARK: - Step (TODO: implement)

    /// Run one decode step: token → embed → chunk_1..4 → next token id.
    /// state1 / state2 are MLState handles for chunk_1 / chunk_2.
    func stepPredict(token: Int32, position: Int,
                     state1: MLState, state2: MLState,
                     collectTimings: Bool = false) async throws -> Int32 {
        guard let c1 = chunk1, let c2 = chunk2,
              let c3 = chunk3, let c4 = chunk4 else {
            throw NSError(domain: "Gemma4Stateful", code: 60,
                userInfo: [NSLocalizedDescriptionKey: "not loaded"])
        }
        let opts = MLPredictionOptions()
        let t0 = CFAbsoluteTimeGetCurrent()

        // 1. Embed + per-layer raw embed lookups
        embedLookup(token: token)
        perLayerEmbedLookup(token: token)
        let tEmbed = CFAbsoluteTimeGetCurrent()

        // 2. Fill RoPE (sliding + full), masks (full + sliding), positions
        fillRopeAt(position)
        fillCausalMasks(position)
        setPositions(position)
        let tRope = CFAbsoluteTimeGetCurrent()

        // 3. chunk_1: hidden_states_out, per_layer_combined_out
        let p1 = Chunk1Provider(
            h: fvHidden!, mF: fvCausalMaskFull!, mS: fvCausalMaskSliding!,
            plr: fvPerLayerRaw!, cs: fvCosS!, ss: fvSinS!,
            cf: fvCosF!, sf: fvSinF!,
            cp: fvCurrentPos!, rp: fvRingPos!)
        let tC1 = CFAbsoluteTimeGetCurrent()
        let out1 = try await c1.prediction(from: p1, using: state1, options: opts)
        if collectTimings { c1Ms += (CFAbsoluteTimeGetCurrent() - tC1) * 1000 }
        guard let fvH1 = out1.featureValue(for: "hidden_states_out"),
              let fvPLC = out1.featureValue(for: "per_layer_combined_out")
        else {
            throw NSError(domain: "Gemma4Stateful", code: 70,
                userInfo: [NSLocalizedDescriptionKey:
                    "chunk_1 missing hidden_states_out / per_layer_combined_out"])
        }

        // 4. chunk_2: hidden_states_out, kv13_k/v, kv14_k/v
        let p2 = Chunk2Provider(
            h: fvH1, mF: fvCausalMaskFull!, mS: fvCausalMaskSliding!,
            plc: fvPLC, cs: fvCosS!, ss: fvSinS!,
            cf: fvCosF!, sf: fvSinF!,
            cp: fvCurrentPos!, rp: fvRingPos!)
        let tC2 = CFAbsoluteTimeGetCurrent()
        let out2 = try await c2.prediction(from: p2, using: state2, options: opts)
        if collectTimings { c2Ms += (CFAbsoluteTimeGetCurrent() - tC2) * 1000 }
        guard let fvH2 = out2.featureValue(for: "hidden_states_out"),
              let fvKV13K = out2.featureValue(for: "kv13_k"),
              let fvKV13V = out2.featureValue(for: "kv13_v"),
              let fvKV14K = out2.featureValue(for: "kv14_k"),
              let fvKV14V = out2.featureValue(for: "kv14_v")
        else {
            throw NSError(domain: "Gemma4Stateful", code: 71,
                userInfo: [NSLocalizedDescriptionKey:
                    "chunk_2 missing required outputs"])
        }

        // 5. chunk_3: stateless, reads kv13/14
        let p3 = SharedChunkProvider(
            h: fvH2, mF: fvCausalMaskFull!, mS: fvCausalMaskSliding!,
            plc: fvPLC, cs: fvCosS!, ss: fvSinS!,
            cf: fvCosF!, sf: fvSinF!,
            kv13K: fvKV13K, kv13V: fvKV13V,
            kv14K: fvKV14K, kv14V: fvKV14V)
        let tC3 = CFAbsoluteTimeGetCurrent()
        let out3 = try await c3.prediction(from: p3, options: opts)
        if collectTimings { c3Ms += (CFAbsoluteTimeGetCurrent() - tC3) * 1000 }
        guard let fvH3 = out3.featureValue(for: "hidden_states_out") else {
            throw NSError(domain: "Gemma4Stateful", code: 72,
                userInfo: [NSLocalizedDescriptionKey:
                    "chunk_3 missing hidden_states_out"])
        }

        // 6. chunk_4: stateless, reads kv13/14, lm_head + argmax → token_id
        let p4 = SharedChunkProvider(
            h: fvH3, mF: fvCausalMaskFull!, mS: fvCausalMaskSliding!,
            plc: fvPLC, cs: fvCosS!, ss: fvSinS!,
            cf: fvCosF!, sf: fvSinF!,
            kv13K: fvKV13K, kv13V: fvKV13V,
            kv14K: fvKV14K, kv14V: fvKV14V)
        let tC4 = CFAbsoluteTimeGetCurrent()
        let out4 = try await c4.prediction(from: p4, options: opts)
        if collectTimings {
            c4Ms += (CFAbsoluteTimeGetCurrent() - tC4) * 1000
            embedMs += (tEmbed - t0) * 1000
            ropeFillMs += (tRope - tEmbed) * 1000
            timedSteps += 1
        }
        guard let fvTok = out4.featureValue(for: "token_id"),
              let arr = fvTok.multiArrayValue else {
            throw NSError(domain: "Gemma4Stateful", code: 73,
                userInfo: [NSLocalizedDescriptionKey:
                    "chunk_4 missing token_id"])
        }
        return arr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
    }

    // MARK: - Generate (TODO: implement)

    /// Full generation with cross-turn KV reuse via persistedState1 / 2.
    /// Mirrors Qwen3VL2BStatefulGenerator.generate but with two stateful
    /// chunks (chunk_1 + chunk_2). T=1 only — multifunction prefill_bN is
    /// a follow-up (matches build_gemma4_e2b_stateful_chunks.py docstring).
    func generate(inputIds: [Int32], maxNewTokens: Int = 64,
                  eosTokenIds: Set<Int32> = [],
                  onToken: ((Int32) -> Void)? = nil) async throws -> [Int32] {
        guard let c1 = chunk1, let c2 = chunk2 else {
            throw NSError(domain: "Gemma4Stateful", code: 80,
                userInfo: [NSLocalizedDescriptionKey: "not loaded"])
        }

        // Cross-turn resume: persistedInputIds must be a STRICT prefix of
        // inputIds, both states must be non-nil. Otherwise drop and restart.
        var resumeAt = 0
        let canResume = persistedState1 != nil && persistedState2 != nil
            && !persistedInputIds.isEmpty
        if canResume {
            let cap = min(persistedInputIds.count, inputIds.count)
            var l = 0
            while l < cap && persistedInputIds[l] == inputIds[l] { l += 1 }
            if l == persistedInputIds.count && l < inputIds.count && l > 0 {
                resumeAt = l
            }
        }

        let state1: MLState
        let state2: MLState
        if resumeAt > 0, let s1 = persistedState1, let s2 = persistedState2 {
            state1 = s1; state2 = s2
            print("[Gemma4Stateful] RESUME L=\(resumeAt) "
                  + "(persisted=\(persistedInputIds.count), "
                  + "new=\(inputIds.count))")
            // Clear bookkeeping during in-flight; restored on success below.
            persistedInputIds = []
            persistedPosition = 0
        } else {
            state1 = c1.makeState()
            state2 = c2.makeState()
            persistedState1 = state1
            persistedState2 = state2
            persistedInputIds = []
            persistedPosition = 0
        }

        var position = resumeAt
        var lastToken: Int32 = 0

        c1Ms = 0; c2Ms = 0; c3Ms = 0; c4Ms = 0
        embedMs = 0; ropeFillMs = 0; timedSteps = 0

        // Prefill — T=1 per the build-script contract. The last prefill step
        // emits the first decode token, so the decode loop starts with that.
        let t0 = CFAbsoluteTimeGetCurrent()
        var prefillPredicted: Int32 = 0
        for i in resumeAt..<inputIds.count {
            let tok = inputIds[i]
            prefillPredicted = try await stepPredict(
                token: tok, position: position,
                state1: state1, state2: state2,
                collectTimings: false)
            lastToken = tok
            position += 1
        }
        let prefillEnd = CFAbsoluteTimeGetCurrent()

        // Decode — emit prefillPredicted as the first decode token, then loop.
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var decoded: [Int32] = []
        if maxNewTokens > 0 && inputIds.count > resumeAt {
            decoded.append(prefillPredicted)
            onToken?(prefillPredicted)
            lastToken = prefillPredicted
        }
        while decoded.count < maxNewTokens {
            if eosTokenIds.contains(lastToken) { break }
            if position >= cfg.ctx { break }
            let next = try await stepPredict(
                token: lastToken, position: position,
                state1: state1, state2: state2,
                collectTimings: true)
            decoded.append(next)
            onToken?(next)
            lastToken = next
            position += 1
        }
        let t1 = CFAbsoluteTimeGetCurrent()

        // Persist consumed tokens. The state has consumed prompt[resumeAt:]
        // + decoded[:-1] (the last decoded token's "feed" step never ran).
        let consumed = decoded.dropLast()
        var newPersisted = inputIds
        newPersisted.append(contentsOf: consumed)
        persistedInputIds = newPersisted
        persistedPosition = newPersisted.count

        let prefillMs = (prefillEnd - t0) * 1000
        let decodeMs = (t1 - decodeStart) * 1000
        let decodeTokPerS = Double(decoded.count) / max(t1 - decodeStart, 1e-3)
        let n = max(timedSteps, 1)
        let resumeTag = resumeAt > 0 ? " [resumed L=\(resumeAt)]" : ""
        let prefillTokCount = max(inputIds.count - resumeAt, 1)
        let prefillTokPerS = Double(prefillTokCount)
            / max(prefillEnd - t0, 1e-3)
        stats = String(format:
            "prefill %d tok in %.1fms (%.1f tok/s)%@ | decode %d tok in %.1fms (%.1f tok/s)\n\n"
            + "per-step breakdown (decode):\n"
            + "  chunk_1: %.2f ms/step\n"
            + "  chunk_2: %.2f ms/step\n"
            + "  chunk_3: %.2f ms/step\n"
            + "  chunk_4: %.2f ms/step\n"
            + "  embed+rope fill: %.2f ms/step",
            inputIds.count - resumeAt, prefillMs, prefillTokPerS,
            resumeTag as NSString,
            decoded.count, decodeMs, decodeTokPerS,
            c1Ms / Double(n), c2Ms / Double(n),
            c3Ms / Double(n), c4Ms / Double(n),
            (embedMs + ropeFillMs) / Double(n))
        return decoded
    }

    // MARK: - Prewarm

    /// Run one dummy prediction through every chunk on throwaway states so
    /// the ANE compiles its dispatch cache before the user types anything.
    /// First generate() in a fresh process pays multi-second compile per
    /// chunk; prewarm front-loads it into the post-load wait the user
    /// already expects.
    func prewarm() async throws {
        guard let c1 = chunk1, let c2 = chunk2,
              let c3 = chunk3, let c4 = chunk4 else { return }

        // Zero all reusable buffers; correctness doesn't matter for compile-
        // cache population, only that every shape variant is exercised.
        memset(reusableHidden.dataPointer, 0, reusableHidden.count * 2)
        memset(reusablePerLayerRaw.dataPointer, 0, reusablePerLayerRaw.count * 2)
        // Mask: 0 everywhere is fine for warm (softmax over zeros is uniform).
        memset(reusableCausalMaskFull.dataPointer, 0, reusableCausalMaskFull.count * 2)
        memset(reusableCausalMaskSliding.dataPointer, 0, reusableCausalMaskSliding.count * 2)
        memset(reusableCosS.dataPointer, 0, reusableCosS.count * 2)
        memset(reusableSinS.dataPointer, 0, reusableSinS.count * 2)
        memset(reusableCosF.dataPointer, 0, reusableCosF.count * 2)
        memset(reusableSinF.dataPointer, 0, reusableSinF.count * 2)
        setPositions(0)

        let warm1 = c1.makeState()
        let warm2 = c2.makeState()
        let opts = MLPredictionOptions()

        let p1 = Chunk1Provider(
            h: fvHidden!, mF: fvCausalMaskFull!, mS: fvCausalMaskSliding!,
            plr: fvPerLayerRaw!, cs: fvCosS!, ss: fvSinS!,
            cf: fvCosF!, sf: fvSinF!,
            cp: fvCurrentPos!, rp: fvRingPos!)
        let out1 = try await c1.prediction(from: p1, using: warm1, options: opts)
        guard let fvH1 = out1.featureValue(for: "hidden_states_out"),
              let fvPLC = out1.featureValue(for: "per_layer_combined_out")
        else {
            throw NSError(domain: "Gemma4Stateful", code: 90,
                userInfo: [NSLocalizedDescriptionKey: "prewarm chunk_1 failed"])
        }

        let p2 = Chunk2Provider(
            h: fvH1, mF: fvCausalMaskFull!, mS: fvCausalMaskSliding!,
            plc: fvPLC, cs: fvCosS!, ss: fvSinS!,
            cf: fvCosF!, sf: fvSinF!,
            cp: fvCurrentPos!, rp: fvRingPos!)
        let out2 = try await c2.prediction(from: p2, using: warm2, options: opts)
        guard let fvH2 = out2.featureValue(for: "hidden_states_out"),
              let fvKV13K = out2.featureValue(for: "kv13_k"),
              let fvKV13V = out2.featureValue(for: "kv13_v"),
              let fvKV14K = out2.featureValue(for: "kv14_k"),
              let fvKV14V = out2.featureValue(for: "kv14_v") else {
            throw NSError(domain: "Gemma4Stateful", code: 91,
                userInfo: [NSLocalizedDescriptionKey: "prewarm chunk_2 failed"])
        }

        let p3 = SharedChunkProvider(
            h: fvH2, mF: fvCausalMaskFull!, mS: fvCausalMaskSliding!,
            plc: fvPLC, cs: fvCosS!, ss: fvSinS!,
            cf: fvCosF!, sf: fvSinF!,
            kv13K: fvKV13K, kv13V: fvKV13V, kv14K: fvKV14K, kv14V: fvKV14V)
        let out3 = try await c3.prediction(from: p3, options: opts)
        guard let fvH3 = out3.featureValue(for: "hidden_states_out") else {
            throw NSError(domain: "Gemma4Stateful", code: 92,
                userInfo: [NSLocalizedDescriptionKey: "prewarm chunk_3 failed"])
        }

        let p4 = SharedChunkProvider(
            h: fvH3, mF: fvCausalMaskFull!, mS: fvCausalMaskSliding!,
            plc: fvPLC, cs: fvCosS!, ss: fvSinS!,
            cf: fvCosF!, sf: fvSinF!,
            kv13K: fvKV13K, kv13V: fvKV13V, kv14K: fvKV14K, kv14V: fvKV14V)
        _ = try await c4.prediction(from: p4, options: opts)
        // throwaway states drop on return; persistedState{1,2} untouched.
    }
}
