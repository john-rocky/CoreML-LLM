// Gemma4StatefulEngine — Phase 1 runtime for the MLState + slice_update
// KV variant of Gemma 4. Independent of the legacy ChunkedEngine path
// (which keeps shipping for backward compatibility).
//
// Loads 4 chunk mlpackages produced by
// conversion/build_gemma4_e2b_stateful_chunks.py:
//
//   chunk_1.mlpackage  own KV state (kv_cache_sliding + kv_cache_full),
//                      computes per_layer_combined from raw
//   chunk_2.mlpackage  own KV state, emits kv13_*/kv14_* producer aliases
//   chunk_3.mlpackage  stateless, reads kv13/14 (KV-shared layers)
//   chunk_4.mlpackage  stateless, reads kv13/14 + lm_head + argmax
//
// Sidecars (embed_tokens_q8.bin, embed_tokens_per_layer_q8.bin,
// per_layer_projection.bin, cos_sliding.npy / sin_sliding.npy /
// cos_full.npy / sin_full.npy, model_config.json) are reused unchanged
// from the existing v1.4.0 build pipeline. The engine reads them via
// the same EmbeddingLookup helper ChunkedEngine uses.
//
// Sliding KV is a ring buffer: writes go to slot
// `ring_pos = current_pos % W` (Swift precomputes; the chunk graphs
// take it as a separate int32 input). The matching causal mask is
// LEFT-aligned: first (pos+1) slots valid for pos<W, all W slots
// valid for pos>=W. (The recurrent shift build was right-aligned.)
//
// Phase 1 scope:
//  - per-generate fresh MLState (no cross-turn reuse — that ports the
//    v1.6.0 LCP-match logic, slated for Phase 2)
//  - T=1 prefill loop and T=1 decode loop (no multifunction prefill_bN
//    yet — slow but correct, mirrors Qwen3-VL Phase 1 → v1.5.0)
//  - text-only (vision/audio stays on the existing CoreMLLLM path)

import Accelerate
import CoreML
import Foundation

@available(iOS 18.0, macOS 15.0, *)
public final class Gemma4StatefulEngine {
    // MARK: - Public surface

    public struct Config {
        public let computeUnits: MLComputeUnits
        public init(computeUnits: MLComputeUnits = .cpuAndNeuralEngine) {
            self.computeUnits = computeUnits
        }
    }

    public private(set) var modelConfig: ModelConfig?
    public var lastDecodeTokensPerSecond: Double = 0

    // MARK: - Storage

    private let cfg: Config
    private var modelDir: URL?

    private var chunk1: MLModel?
    private var chunk2: MLModel?
    private var chunk3: MLModel?
    private var chunk4: MLModel?

    private var embedTokens: EmbeddingLookup?
    private var embedTokensPerLayer: EmbeddingLookup?

    // RoPE sidecar buffers (mmap-backed). Same layout as ChunkedEngine.
    private var cosSlidingTable: Data?
    private var sinSlidingTable: Data?
    private var cosFullTable: Data?
    private var sinFullTable: Data?

    // Reusable per-step scratch (T=1).
    private var maskFull: MLMultiArray!
    private var maskSliding: MLMultiArray!
    private var fvMaskFull: MLFeatureValue!
    private var fvMaskSliding: MLFeatureValue!
    private var posScratch: MLMultiArray!
    private var ringScratch: MLMultiArray!
    private var fvPos: MLFeatureValue!
    private var fvRing: MLFeatureValue!

    // Cross-turn KV reuse (Phase 2a). When persistedInputIds is a strict
    // prefix of the next generate's inputIds AND both states are non-nil,
    // skip prefill of the matching prefix and resume from the LCP boundary.
    // Mirrors the Qwen3-VL v1.6.0 pattern. State handles bind to specific
    // MLModel instances — load() drops them so a model reload doesn't
    // dangle. LLMRunner is expected to call resetPersistedState() on chat
    // clear, image change, or any prompt-prefix invariant break.
    private var persistedState1: MLState?
    private var persistedState2: MLState?
    private var persistedInputIds: [Int32] = []
    private var persistedPosition: Int = 0

    public init(config: Config = Config()) {
        self.cfg = config
    }

    // MARK: - Load

    /// Drop the cross-turn KV cache. Call when chat history clears, the
    /// vision/audio prefix changes, or anything else that breaks the
    /// "persisted prefix is a prefix of the next prompt" invariant.
    public func resetPersistedState() {
        persistedState1 = nil
        persistedState2 = nil
        persistedInputIds = []
        persistedPosition = 0
    }

    public func load(modelDirectory: URL) async throws {
        // State handles bind to specific MLModel instances. Any persisted
        // state from a prior load points at models we're about to release;
        // drop it before we lose the binding.
        resetPersistedState()
        modelDir = modelDirectory
        let mc = try ModelConfig.load(from: modelDirectory)
        modelConfig = mc

        let mcfg = MLModelConfiguration()
        mcfg.computeUnits = cfg.computeUnits

        func openChunk(_ name: String) throws -> MLModel {
            let mlc = modelDirectory.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: mlc.path) {
                return try MLModel(contentsOf: mlc, configuration: mcfg)
            }
            let pkg = modelDirectory.appendingPathComponent("\(name).mlpackage")
            if FileManager.default.fileExists(atPath: pkg.path) {
                let compiled = try MLModel.compileModel(at: pkg)
                return try MLModel(contentsOf: compiled, configuration: mcfg)
            }
            throw CoreMLLLMError.modelNotFound("\(name).mlmodelc/.mlpackage not found in \(modelDirectory.path)")
        }
        chunk1 = try openChunk("chunk_1")
        chunk2 = try openChunk("chunk_2")
        chunk3 = try openChunk("chunk_3")
        chunk4 = try openChunk("chunk_4")

        embedTokens = try EmbeddingLookup(
            dataURL: modelDirectory.appendingPathComponent("embed_tokens_q8.bin"),
            scalesURL: modelDirectory.appendingPathComponent("embed_tokens_scales.bin"),
            vocabSize: mc.vocabSize, dim: mc.hiddenSize, scale: mc.embedScale)
        embedTokensPerLayer = try EmbeddingLookup(
            dataURL: modelDirectory.appendingPathComponent("embed_tokens_per_layer_q8.bin"),
            scalesURL: modelDirectory.appendingPathComponent("embed_tokens_per_layer_scales.bin"),
            vocabSize: mc.vocabSize,
            dim: mc.numLayers * mc.perLayerDim,
            scale: mc.perLayerEmbedScale)

        cosSlidingTable = try? Data(
            contentsOf: modelDirectory.appendingPathComponent("cos_sliding.npy"),
            options: .mappedIfSafe)
        sinSlidingTable = try? Data(
            contentsOf: modelDirectory.appendingPathComponent("sin_sliding.npy"),
            options: .mappedIfSafe)
        cosFullTable = try? Data(
            contentsOf: modelDirectory.appendingPathComponent("cos_full.npy"),
            options: .mappedIfSafe)
        sinFullTable = try? Data(
            contentsOf: modelDirectory.appendingPathComponent("sin_full.npy"),
            options: .mappedIfSafe)

        // Allocate reusable mask + position scratch.
        let ctx = mc.contextLength
        let W = mc.slidingWindow
        maskFull = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: ctx)], dataType: .float16)
        maskSliding = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: W)], dataType: .float16)
        posScratch = try MLMultiArray(shape: [1], dataType: .int32)
        ringScratch = try MLMultiArray(shape: [1], dataType: .int32)
        fvMaskFull = MLFeatureValue(multiArray: maskFull)
        fvMaskSliding = MLFeatureValue(multiArray: maskSliding)
        fvPos = MLFeatureValue(multiArray: posScratch)
        fvRing = MLFeatureValue(multiArray: ringScratch)
    }

    // MARK: - Mask + position helpers

    /// Right-aligned full causal mask: [0, pos] valid (0), (pos, ctx) masked (-inf).
    private func fillFullCausalMask(position: Int) {
        let ctx = modelConfig!.contextLength
        let dst = maskFull.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        let neg = Float16(-65504).bitPattern  // safe -inf for fp16
        let p = min(max(position, 0), ctx - 1)
        for i in 0..<ctx { dst[i] = i <= p ? 0 : neg }
    }

    /// LEFT-aligned sliding causal mask for ring-buffer KV:
    ///   pos <  W: first (pos+1) slots valid, rest masked
    ///   pos >= W: all W slots valid (ring is full)
    /// Different from the recurrent shift build which was right-aligned.
    private func fillSlidingCausalMask(position: Int) {
        let W = modelConfig!.slidingWindow
        let dst = maskSliding.dataPointer.bindMemory(to: UInt16.self, capacity: W)
        let neg = Float16(-65504).bitPattern
        let valid = min(position + 1, W)
        for i in 0..<W { dst[i] = i < valid ? 0 : neg }
    }

    private func setPos(_ pos: Int) {
        posScratch.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(pos)
    }

    private func setRing(_ pos: Int) {
        let W = modelConfig!.slidingWindow
        ringScratch.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(pos % W)
    }

    /// Single-position RoPE row from the baked sidecar table.
    /// Layout matches ChunkedEngine.lookupRoPE: header + (pos × dim × fp16).
    private func lookupRoPE(table: Data?, position: Int, dim: Int) throws -> MLMultiArray {
        let result = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: dim)], dataType: .float16)
        let dst = result.dataPointer.bindMemory(to: UInt16.self, capacity: dim)
        guard let table else {
            memset(dst, 0, dim * MemoryLayout<UInt16>.stride); return result
        }
        var headerSize = 128
        table.withUnsafeBytes { raw in
            let b = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            headerSize = 10 + (Int(b[8]) | (Int(b[9]) << 8))
        }
        let rowBytes = dim * MemoryLayout<UInt16>.stride
        let offset = headerSize + position * rowBytes
        guard offset + rowBytes <= table.count else {
            memset(dst, 0, rowBytes); return result
        }
        _ = table.withUnsafeBytes { raw in
            memcpy(dst, raw.baseAddress!.advanced(by: offset), rowBytes)
        }
        return result
    }

    // MARK: - Per-step plumbing

    /// Provider that re-uses pre-built MLFeatureValue slots so we don't
    /// rebuild a Dictionary per step (a small but real per-token cost).
    private final class FeatureProvider: NSObject, MLFeatureProvider {
        let map: [String: MLFeatureValue]
        let featureNames: Set<String>
        init(_ map: [String: MLFeatureValue]) {
            self.map = map
            self.featureNames = Set(map.keys)
        }
        func featureValue(for name: String) -> MLFeatureValue? { map[name] }
    }

    /// Run one T=1 step through chunks 1→2→3→4. State buffers are
    /// updated in-place by the chunk graphs (slice_update).
    /// Returns the next token id from chunk_4's argmax.
    private func step(token: Int32, position: Int,
                       states: (s1: MLState, s2: MLState),
                       opts: MLPredictionOptions) async throws -> Int32 {
        guard let mc = modelConfig else {
            throw CoreMLLLMError.modelNotFound("not loaded")
        }
        // 1) Embed lookups (full hidden + per-layer raw).
        let hidden = try embedTokens!.lookup(
            Int(token), shape: [1, 1, NSNumber(value: mc.hiddenSize)])
        let perLayerRaw = try embedTokensPerLayer!.lookup(
            Int(token),
            shape: [1, 1, NSNumber(value: mc.numLayers * mc.perLayerDim)])

        // 2) Position-dependent scratch (mask + RoPE + indices).
        fillFullCausalMask(position: position)
        fillSlidingCausalMask(position: position)
        setPos(position)
        setRing(position)
        // ModelConfig doesn't carry head_dim/global_head_dim; Gemma 4
        // E2B and E4B both ship at sliding=256 / full=512 (matching the
        // existing ChunkedEngine hardcode at lookupRoPE call sites).
        let cosS = try lookupRoPE(table: cosSlidingTable, position: position, dim: 256)
        let sinS = try lookupRoPE(table: sinSlidingTable, position: position, dim: 256)
        let cosF = try lookupRoPE(table: cosFullTable,    position: position, dim: 512)
        let sinF = try lookupRoPE(table: sinFullTable,    position: position, dim: 512)

        // 3) Chunk 1 (own state, computes per_layer_combined).
        let p1 = FeatureProvider([
            "hidden_states":      MLFeatureValue(multiArray: hidden),
            "causal_mask_full":   fvMaskFull,
            "causal_mask_sliding": fvMaskSliding,
            "per_layer_raw":      MLFeatureValue(multiArray: perLayerRaw),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "current_pos": fvPos,
            "ring_pos":    fvRing,
        ])
        let out1 = try await chunk1!.prediction(
            from: p1, using: states.s1, options: opts)
        guard let h1 = out1.featureValue(for: "hidden_states_out"),
              let plc = out1.featureValue(for: "per_layer_combined_out")
        else { throw CoreMLLLMError.modelNotFound("chunk_1 missing outputs") }

        // 4) Chunk 2 (own state, emits kv13/kv14 alias outputs).
        let p2 = FeatureProvider([
            "hidden_states":      h1,
            "causal_mask_full":   fvMaskFull,
            "causal_mask_sliding": fvMaskSliding,
            "per_layer_combined": plc,
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "current_pos": fvPos,
            "ring_pos":    fvRing,
        ])
        let out2 = try await chunk2!.prediction(
            from: p2, using: states.s2, options: opts)
        guard let h2 = out2.featureValue(for: "hidden_states_out"),
              let kv13k = out2.featureValue(for: "kv13_k"),
              let kv13v = out2.featureValue(for: "kv13_v"),
              let kv14k = out2.featureValue(for: "kv14_k"),
              let kv14v = out2.featureValue(for: "kv14_v")
        else { throw CoreMLLLMError.modelNotFound("chunk_2 missing outputs") }

        // 5) Chunks 3 + 4 — stateless, reuse the kv13/kv14 alias inputs.
        let kvShared: [String: MLFeatureValue] = [
            "kv13_k": kv13k, "kv13_v": kv13v,
            "kv14_k": kv14k, "kv14_v": kv14v,
        ]
        var sharedInputs: [String: MLFeatureValue] = [
            "causal_mask_full":   fvMaskFull,
            "causal_mask_sliding": fvMaskSliding,
            "per_layer_combined": plc,
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
        ]
        sharedInputs.merge(kvShared) { _, new in new }

        var p3map = sharedInputs
        p3map["hidden_states"] = h2
        let out3 = try await chunk3!.prediction(
            from: FeatureProvider(p3map), options: opts)
        guard let h3 = out3.featureValue(for: "hidden_states_out") else {
            throw CoreMLLLMError.modelNotFound("chunk_3 missing hidden_states_out")
        }

        var p4map = sharedInputs
        p4map["hidden_states"] = h3
        let out4 = try await chunk4!.prediction(
            from: FeatureProvider(p4map), options: opts)
        guard let tokFV = out4.featureValue(for: "token_id"),
              let tokArr = tokFV.multiArrayValue
        else { throw CoreMLLLMError.modelNotFound("chunk_4 no token_id") }
        return tokArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
    }

    // MARK: - Generate (T=1 prefill + T=1 decode, cross-turn KV reuse)

    /// Runs the prompt through chunks (T=1 prefill, slow but correct),
    /// then continues decoding up to maxNewTokens.
    ///
    /// Phase 2a: cross-turn KV reuse. If `persistedInputIds` is a strict
    /// prefix of `inputIds` and both states are still bound, skip prefill
    /// of the matching prefix and resume at the LCP boundary. Otherwise
    /// allocate fresh states and start from scratch.
    public func generate(inputIds: [Int32], maxNewTokens: Int = 64,
                          eosTokenIds: Set<Int32> = [],
                          onToken: ((Int32) -> Void)? = nil
    ) async throws -> [Int32] {
        guard let chunk1, chunk2 != nil, chunk3 != nil, chunk4 != nil else {
            throw CoreMLLLMError.modelNotFound("Gemma4StatefulEngine: not loaded")
        }
        guard let mc = modelConfig else {
            throw CoreMLLLMError.modelNotFound("Gemma4StatefulEngine: no config")
        }
        if inputIds.isEmpty { return [] }
        if inputIds.count >= mc.contextLength {
            throw CoreMLLLMError.modelNotFound(
                "prompt (\(inputIds.count) tokens) >= ctx (\(mc.contextLength))")
        }

        // Cross-turn resume detection. We require persistedInputIds to be a
        // STRICT prefix of inputIds (and non-empty) — partial overlaps would
        // mean the persisted state has tokens the new prompt doesn't, and
        // MLState's slice_update can't rewind. In that case, drop persisted
        // state and start fresh.
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
            // If we throw mid-prefill, persistedInputIds = [] prevents a
            // future generate from trying to resume past where the state
            // actually got advanced to.
            persistedInputIds = []
            persistedPosition = 0
        } else {
            // State must come from THIS chunk1/chunk2 instance — handles
            // bind to specific MLModels. Stateless chunks (3, 4) need none.
            state1 = chunk1.makeState()
            state2 = chunk2!.makeState()
            persistedState1 = state1
            persistedState2 = state2
            persistedInputIds = []
            persistedPosition = 0
        }
        let opts = MLPredictionOptions()

        var position = resumeAt
        var lastToken: Int32 = inputIds[max(resumeAt - 1, 0)]
        var prefillPredicted: Int32 = 0

        let t0 = CFAbsoluteTimeGetCurrent()
        // T=1 prefill from the resume boundary — multifunction prefill_bN
        // is Phase 2b.
        for j in resumeAt..<inputIds.count {
            let tok = inputIds[j]
            prefillPredicted = try await step(
                token: tok, position: position,
                states: (state1, state2), opts: opts)
            lastToken = tok
            position += 1
        }
        let prefillEnd = CFAbsoluteTimeGetCurrent()

        // Decode. The prefill's last step already produced the first
        // post-prompt token (prefillPredicted) — emit it, then continue.
        // (Skip the auto-emit when resumeAt == inputIds.count — there was
        // nothing new to prefill, the next user-visible token must come
        // from a real decode step.)
        var decoded: [Int32] = []
        if maxNewTokens > 0 && inputIds.count > resumeAt {
            decoded.append(prefillPredicted)
            onToken?(prefillPredicted)
            lastToken = prefillPredicted
        }
        while decoded.count < maxNewTokens {
            if eosTokenIds.contains(lastToken) { break }
            if position >= mc.contextLength { break }
            let next = try await step(
                token: lastToken, position: position,
                states: (state1, state2), opts: opts)
            decoded.append(next)
            onToken?(next)
            lastToken = next
            position += 1
        }
        let t1 = CFAbsoluteTimeGetCurrent()

        // Persist consumed tokens. The state has consumed
        //   prompt[0..<inputIds.count]  +  decoded[..<decoded.count-1]
        // (the last decoded token's "feed" step never ran). Off-by-one
        // vs the displayed assistant text in the max-tokens-hit case
        // costs at most 1 token of re-prefill on the next turn.
        let consumed = decoded.dropLast()
        var newPersisted = inputIds
        newPersisted.append(contentsOf: consumed)
        persistedInputIds = newPersisted
        persistedPosition = newPersisted.count

        let prefillTokCount = inputIds.count - resumeAt
        let prefillMs = (prefillEnd - t0) * 1000
        let decodeMs = (t1 - prefillEnd) * 1000
        if decodeMs > 0 && decoded.count > 1 {
            lastDecodeTokensPerSecond = Double(decoded.count - 1) / (decodeMs / 1000)
        }
        let resumeTag = resumeAt > 0 ? " [resumed L=\(resumeAt)]" : ""
        print("[Gemma4Stateful] prefill \(prefillTokCount) tok in "
              + String(format: "%.0fms (%.1f tok/s)%@ | decode %d tok in %.0fms (%.1f tok/s)",
                        prefillMs,
                        Double(max(prefillTokCount, 1))
                            / max(prefillMs / 1000, 1e-3),
                        resumeTag,
                        decoded.count, decodeMs, lastDecodeTokensPerSecond))
        return decoded
    }
}
