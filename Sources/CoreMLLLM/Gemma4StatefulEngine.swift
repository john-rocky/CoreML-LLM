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
// Phase 2b scope:
//  - cross-turn KV reuse via LCP-match (Phase 2a, b5fef64)
//  - multifunction `prefill_bN` (Phase 2b) — when the loaded mlpackages
//    carry a `prefill_b<T>` function on every chunk, prefill batches T
//    tokens per forward and falls back to T=1 for the tail (or when a
//    sliding-cache wrap would split the write)
//  - T=1 decode loop (multifunction decode is not pursued — decode is
//    bandwidth-bound, batching gives no win)
//  - text-only (vision/audio stays on the existing CoreMLLLM path)

import Accelerate
import CoreGraphics
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

    // 1-chunk mode (single mlpackage all-in-one):
    //   model.mlpackage = entire 35-layer model + lm_head + argmax,
    //   one unified MLState (kv_cache_unified). Auto-detected at
    //   load() when model.{mlmodelc,mlpackage} is present.
    private var is1Chunk: Bool = false
    private var model1Chunk: MLModel?
    private var model1ChunkPrefill: MLModel?
    private var model1ChunkState: MLState?

    // 3-chunk mode (default when chunk_4 not present in the bundle):
    //   chunk_1 (L0-7)  +  chunk_2 merged (L8-24)  +  chunk_3 (L25-34 + head)
    // 4-chunk mode (legacy / 4-chunk Gemma4 stateful default):
    //   chunk_1 + chunk_2 + chunk_3 + chunk_4 (head). The 3-chunk's
    //   chunk_3 is structurally the same module as the 4-chunk's
    //   chunk_4 — both KV-shared L25-34 + lm_head + argmax. Auto-
    //   detected at load() by checking if chunk_4.{mlmodelc,mlpackage}
    //   exists in the model directory.
    private var is3Chunk: Bool = false

    // Multifunction prefill_bN variants. Non-nil only when the loaded
    // mlpackages were built with `--prefill-batches`. All four chunks
    // must carry the same N for batched dispatch to work — load() drops
    // any partial set. T=1 fallback always available via chunk{1..4}.
    private var prefillT: Int = 1
    private var chunk1Prefill: MLModel?
    private var chunk2Prefill: MLModel?
    private var chunk3Prefill: MLModel?
    private var chunk4Prefill: MLModel?

    /// True when the loaded bundle has functional `prefill_bN` chunks.
    public var hasMultifunctionPrefill: Bool {
        if is1Chunk { return model1ChunkPrefill != nil }
        let core = chunk1Prefill != nil && chunk2Prefill != nil
            && chunk3Prefill != nil
        return is3Chunk ? core : (core && chunk4Prefill != nil)
    }

    // Batched-prefill scratch (allocated lazily once T is known).
    private var batchHidden: MLMultiArray?
    private var batchPerLayerRaw: MLMultiArray?
    private var batchMaskFull: MLMultiArray?
    private var batchMaskSliding: MLMultiArray?
    private var batchCosS: MLMultiArray?
    private var batchSinS: MLMultiArray?
    private var batchCosF: MLMultiArray?
    private var batchSinF: MLMultiArray?

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

    // MARK: - Multimodal (Stage 6)
    //
    // Gemma 4 reserves three placeholder token IDs that the engine swaps
    // for vision/audio encoder output rows during prefill. Image-pad
    // (256 per still image) and video-pad (64 per frame) share the
    // imageFeatures buffer; audio-pad uses a separate audioFeatures
    // buffer. Per-layer-raw is forced to zero at these positions so the
    // chunks compute per_layer_combined entirely from the spliced hidden
    // state — matches the legacy ChunkedEngine pattern. (Gemma 4 does
    // NOT use DeepStack; the handoff doc's DS premise was a Qwen3-VL
    // confusion. See docs/SESSION_2026_04_27_STAGE6_MULTIMODAL.md.)
    private static let IMAGE_TOKEN_ID: Int32 = 258880
    private static let AUDIO_TOKEN_ID: Int32 = 258881
    private static let VIDEO_TOKEN_ID: Int32 = 258884

    // Still-image vision encoder. Default GPU; LLM_VISION_FORCE_ANE=1
    // opts into vision.ane.* for benchmarking (mirrors legacy CoreMLLLM).
    private var visionModelURL: URL?
    private var visionConfig: MLModelConfiguration?
    private var visionModel: MLModel?
    private var visionUsesANEBuild: Bool = false

    // Video-grade vision encoder (64 tokens/frame). GPU only.
    private var videoVisionModelURL: URL?
    private var videoVisionConfig: MLModelConfiguration?
    private var videoVisionModel: MLModel?

    // Audio (Conformer 12-layer). GPU. Sidecars are loaded eagerly at
    // engine load(); the encoder MLModel is loaded lazily on first call.
    private var audioModelURL: URL?
    private var audioConfig: MLModelConfiguration?
    private var audioModel: MLModel?
    private var melFilterbank: [Float]?
    private var audioProjection: AudioProcessor.ProjectionWeights?
    private var audioMelFrames: Int = 200
    private var audioNumTokensConfig: Int = 188
    private var audioMelFloor: Float = 0.001

    /// True when vision.{mlmodelc,mlpackage,ane.*} was found at load.
    public var hasVision: Bool { visionModelURL != nil }
    /// True when vision_video.{mlmodelc,mlpackage} was found at load.
    public var hasVideoVision: Bool { videoVisionModelURL != nil }
    /// True when audio.{mlmodelc,mlpackage} was found at load.
    public var hasAudio: Bool { audioModelURL != nil }
    /// Audio token count from sidecar config (default 188).
    public var defaultAudioNumTokens: Int { audioNumTokensConfig }

    // Per-call multimodal state. Bound by generate(...) on entry, cleared
    // via defer on exit so a subsequent text-only generate doesn't read
    // stale features. step()/prefillStep() consult these to decide
    // whether to splice an encoder row over the embed_tokens lookup.
    private var mmImageFeatures: MLMultiArray?
    private var mmImageNumTokens: Int = 0
    private var mmAudioFeatures: MLMultiArray?
    private var mmAudioNumTokens: Int = 0
    private var mmImageIdx: Int = 0
    private var mmAudioIdx: Int = 0
    // Vision-aware mask group ids — index by ABSOLUTE prompt position;
    // -1 = text/audio (causal-only), 0/1/2/... = contiguous run id of
    // image-pad / video-pad tokens (bidirectional within run).
    private var mmVisionGroupIds: [Int]?
    // Reusable PLR=0 scratch (T=1 path) — vision/audio positions get
    // per-layer-raw zeroed because the chunks recompute per_layer_combined
    // from hidden alone for those tokens.
    private var prlZerosT1: MLMultiArray?

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
        model1ChunkState = nil
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
        // 1-chunk mode probe: model.{mlmodelc,mlpackage} present AND no
        // chunked bundle alongside? Stage 6: prefer the chunked bundle
        // when both are present — sideload upgrades from a prior
        // 1-chunk push leave model.mlmodelc behind in the same dir, and
        // the production ship variant is the 3-chunk merged bundle.
        // Without this guard, devicectl push of a chunked bundle on
        // top of a stale 1-chunk install silently shadow-loads the
        // wrong graph.
        let model1ChunkMlc = modelDirectory.appendingPathComponent("model.mlmodelc")
        let model1ChunkPkg = modelDirectory.appendingPathComponent("model.mlpackage")
        let has1ChunkModel = FileManager.default.fileExists(atPath: model1ChunkMlc.path)
            || FileManager.default.fileExists(atPath: model1ChunkPkg.path)
        let chunk1Mlc = modelDirectory.appendingPathComponent("chunk_1.mlmodelc")
        let chunk1Pkg = modelDirectory.appendingPathComponent("chunk_1.mlpackage")
        let hasChunkedBundle = FileManager.default.fileExists(atPath: chunk1Mlc.path)
            || FileManager.default.fileExists(atPath: chunk1Pkg.path)
        if has1ChunkModel && !hasChunkedBundle {
            is1Chunk = true
            // Prefer .mlpackage over a possibly-stale .mlmodelc — the
            // 35-layer single graph hits Mac↔iPhone ANE incompatibility
            // when compiled on Mac (E5RT "must re-compile the E5 bundle"
            // observed). Forcing iPhone-side compile via .mlpackage
            // avoids the Mac ANEF artifact mismatch.
            let pkg = modelDirectory.appendingPathComponent("model.mlpackage")
            let compiledURL: URL
            if FileManager.default.fileExists(atPath: pkg.path) {
                print("[Gemma4Stateful] 1-chunk mode (compiling model.mlpackage on device)")
                compiledURL = try await MLModel.compileModel(at: pkg)
            } else {
                print("[Gemma4Stateful] 1-chunk mode (loading model.mlmodelc)")
                compiledURL = modelDirectory.appendingPathComponent("model.mlmodelc")
            }
            model1Chunk = try MLModel(contentsOf: compiledURL, configuration: mcfg)
            // Probe prefill_b<N> by re-loading the same compiled URL
            // with a per-function configuration.
            for T in [16, 8, 4] {
                let pcfg = MLModelConfiguration()
                pcfg.computeUnits = cfg.computeUnits
                pcfg.functionName = "prefill_b\(T)"
                if let pm = try? MLModel(contentsOf: compiledURL, configuration: pcfg) {
                    model1ChunkPrefill = pm
                    prefillT = T
                    print("[Gemma4Stateful] multifunction prefill_b\(T) loaded (1-chunk)")
                    break
                }
            }
            // EmbeddingLookup + RoPE sidecars are still needed for input prep.
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
            loadMultimodalEncoders(modelDirectory: modelDirectory)
            return
        }

        chunk1 = try openChunk("chunk_1")
        chunk2 = try openChunk("chunk_2")
        chunk3 = try openChunk("chunk_3")

        // 3-chunk mode detection. Two signals (either is sufficient):
        //  1. chunk_4 absent in the bundle (clean install of 3-chunk).
        //  2. chunk_3's output schema contains `token_id` — meaning
        //     chunk_3 IS the final lm_head + argmax chunk (3-chunk
        //     merged final). This catches the case where chunk_4 from
        //     a prior 4-chunk install was not deleted but chunks 1-3
        //     were overwritten with the 3-chunk variant.
        let chunk4Mlc = modelDirectory.appendingPathComponent("chunk_4.mlmodelc")
        let chunk4Pkg = modelDirectory.appendingPathComponent("chunk_4.mlpackage")
        let has4 = FileManager.default.fileExists(atPath: chunk4Mlc.path)
            || FileManager.default.fileExists(atPath: chunk4Pkg.path)
        let chunk3HasLmHead: Bool = {
            guard let c3 = chunk3 else { return false }
            let outs = c3.modelDescription.outputDescriptionsByName.keys
            return outs.contains("token_id")
        }()
        is3Chunk = !has4 || chunk3HasLmHead
        if is3Chunk {
            let reason = !has4 ? "chunk_4 absent" : "chunk_3 has token_id output"
            print("[Gemma4Stateful] 3-chunk mode (\(reason) — chunk_3 = merged final)")
            chunk4 = nil
        } else {
            chunk4 = try openChunk("chunk_4")
        }

        // Probe each chunk for a `prefill_b<N>` function. Mirrors the
        // Qwen3-VL stateful generator pattern (Phase 2b multifunction).
        // Try N candidates in descending order of preference; first
        // success that loads on ALL four chunks wins.
        chunk1Prefill = nil
        chunk2Prefill = nil
        chunk3Prefill = nil
        chunk4Prefill = nil
        prefillT = 1
        let candidates = [16, 8, 4]
        func openChunkPrefill(_ name: String, T: Int) -> MLModel? {
            let pcfg = MLModelConfiguration()
            pcfg.computeUnits = cfg.computeUnits
            pcfg.functionName = "prefill_b\(T)"
            let mlc = modelDirectory.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: mlc.path) {
                return try? MLModel(contentsOf: mlc, configuration: pcfg)
            }
            let pkg = modelDirectory.appendingPathComponent("\(name).mlpackage")
            if FileManager.default.fileExists(atPath: pkg.path) {
                if let compiled = try? MLModel.compileModel(at: pkg) {
                    return try? MLModel(contentsOf: compiled, configuration: pcfg)
                }
            }
            return nil
        }
        for T in candidates {
            guard let p1 = openChunkPrefill("chunk_1", T: T),
                  let p2 = openChunkPrefill("chunk_2", T: T),
                  let p3 = openChunkPrefill("chunk_3", T: T)
            else { continue }
            if !is3Chunk {
                guard let p4 = openChunkPrefill("chunk_4", T: T) else { continue }
                chunk4Prefill = p4
            }
            chunk1Prefill = p1; chunk2Prefill = p2; chunk3Prefill = p3
            prefillT = T
            print("[Gemma4Stateful] multifunction prefill_b\(T) loaded "
                  + "(\(is3Chunk ? "3-chunk" : "4-chunk"))")
            break
        }

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
        loadMultimodalEncoders(modelDirectory: modelDirectory)
    }

    // MARK: - Multimodal encoder loading (Stage 6)

    /// Probe optional vision / video / audio encoders and load configs +
    /// sidecars. The MLModel instances themselves are loaded lazily on
    /// first use so text-only chats don't pay the encoder compile cost.
    /// Vision encoder gets a background prewarm so the first user image
    /// prompt doesn't pay the ~30 s graph compile on the critical path.
    private func loadMultimodalEncoders(modelDirectory: URL) {
        let forceANE = ProcessInfo.processInfo.environment["LLM_VISION_FORCE_ANE"] == "1"
        let visionANEv2Compiled = modelDirectory.appendingPathComponent("vision.ane.v2.mlmodelc")
        let visionANECompiled = modelDirectory.appendingPathComponent("vision.ane.mlmodelc")
        let visionANEPkg = modelDirectory.appendingPathComponent("vision.ane.mlpackage")
        let visionCompiled = modelDirectory.appendingPathComponent("vision.mlmodelc")
        let visionPkg = modelDirectory.appendingPathComponent("vision.mlpackage")
        if forceANE, FileManager.default.fileExists(atPath: visionANEv2Compiled.path) {
            visionModelURL = visionANEv2Compiled; visionUsesANEBuild = true
        } else if forceANE, FileManager.default.fileExists(atPath: visionANECompiled.path) {
            visionModelURL = visionANECompiled; visionUsesANEBuild = true
        } else if forceANE, FileManager.default.fileExists(atPath: visionANEPkg.path) {
            visionModelURL = visionANEPkg; visionUsesANEBuild = true
        } else if FileManager.default.fileExists(atPath: visionCompiled.path) {
            visionModelURL = visionCompiled
        } else if FileManager.default.fileExists(atPath: visionPkg.path) {
            visionModelURL = visionPkg
        } else if FileManager.default.fileExists(atPath: visionANEv2Compiled.path) {
            visionModelURL = visionANEv2Compiled; visionUsesANEBuild = true
        } else if FileManager.default.fileExists(atPath: visionANECompiled.path) {
            visionModelURL = visionANECompiled; visionUsesANEBuild = true
        } else if FileManager.default.fileExists(atPath: visionANEPkg.path) {
            visionModelURL = visionANEPkg; visionUsesANEBuild = true
        }
        if let url = visionModelURL {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = visionUsesANEBuild ? .cpuAndNeuralEngine : .cpuAndGPU
            visionConfig = cfg
            print("[Gemma4Stateful/Vision] selected \(url.lastPathComponent) → \(visionUsesANEBuild ? "ANE" : "GPU")")
            prewarmVisionInBackground()
        }

        let videoVisionCompiled = modelDirectory.appendingPathComponent("vision_video.mlmodelc")
        let videoVisionPkg = modelDirectory.appendingPathComponent("vision_video.mlpackage")
        if FileManager.default.fileExists(atPath: videoVisionCompiled.path) {
            videoVisionModelURL = videoVisionCompiled
        } else if FileManager.default.fileExists(atPath: videoVisionPkg.path) {
            videoVisionModelURL = videoVisionPkg
        }
        if videoVisionModelURL != nil {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndGPU
            videoVisionConfig = cfg
        }

        let audioCompiled = modelDirectory.appendingPathComponent("audio.mlmodelc")
        let audioPkg = modelDirectory.appendingPathComponent("audio.mlpackage")
        if FileManager.default.fileExists(atPath: audioCompiled.path) {
            audioModelURL = audioCompiled
        } else if FileManager.default.fileExists(atPath: audioPkg.path) {
            audioModelURL = audioPkg
        }
        if audioModelURL != nil {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndGPU
            audioConfig = cfg

            let melURL = modelDirectory.appendingPathComponent("mel_filterbank.bin")
            if FileManager.default.fileExists(atPath: melURL.path) {
                melFilterbank = try? AudioProcessor.loadMelFilterbank(from: melURL)
            }
            let projURL = modelDirectory.appendingPathComponent("output_proj_weight.npy")
            if FileManager.default.fileExists(atPath: projURL.path) {
                audioProjection = try? AudioProcessor.ProjectionWeights.load(from: modelDirectory)
            }
            let audioConfURL = modelDirectory.appendingPathComponent("audio_config.json")
            if let data = try? Data(contentsOf: audioConfURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                audioMelFrames = json["mel_frames"] as? Int ?? 200
                audioNumTokensConfig = json["num_tokens"] as? Int ?? 188
                if let mf = json["log_offset"] as? Double {
                    audioMelFloor = Float(mf)
                } else if let mf = json["mel_floor"] as? Double {
                    audioMelFloor = Float(mf)
                }
            }
        }

        if hasVision || hasAudio || hasVideoVision {
            print("[Gemma4Stateful] multimodal encoders: vision=\(hasVision) " +
                  "video=\(hasVideoVision) audio=\(hasAudio)")
        }
    }

    /// Background prewarm of the GPU vision encoder. ANE builds compile
    /// fast (no warmup needed); the GPU graph compile is ~30 s on first
    /// call, which would otherwise land on TTFT for the first image
    /// prompt. Submits a dummy 48×48 patch grid so the live forward is
    /// hot. Runs on .utility queue so text-only chats aren't blocked.
    private func prewarmVisionInBackground() {
        guard let url = visionModelURL, let cfg = visionConfig,
              !visionUsesANEBuild else { return }
        DispatchQueue.global(qos: .utility).async { [weak self] in
            do {
                let t0 = CFAbsoluteTimeGetCurrent()
                let m = try MLModel(contentsOf: url, configuration: cfg)
                self?.visionModel = m
                let pd = 16 * 16 * 3
                let total = 2520
                let pv = try MLMultiArray(
                    shape: [1, NSNumber(value: total), NSNumber(value: pd)],
                    dataType: .float32)
                let pid = try MLMultiArray(
                    shape: [1, NSNumber(value: total), 2], dataType: .int32)
                let pidp = pid.dataPointer.bindMemory(
                    to: Int32.self, capacity: total * 2)
                var k = 0
                for py in 0..<48 {
                    for px in 0..<48 {
                        pidp[k * 2] = Int32(px)
                        pidp[k * 2 + 1] = Int32(py)
                        k += 1
                    }
                }
                for i in (48 * 48)..<total {
                    pidp[i * 2] = -1
                    pidp[i * 2 + 1] = -1
                }
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "pixel_values": MLFeatureValue(multiArray: pv),
                    "pixel_position_ids": MLFeatureValue(multiArray: pid),
                ])
                _ = try? m.prediction(from: input)
                let dt = CFAbsoluteTimeGetCurrent() - t0
                print("[Gemma4Stateful/Vision] prewarm done in \(String(format: "%.1f", dt))s")
            } catch {
                print("[Gemma4Stateful/Vision] prewarm failed: \(error)")
            }
        }
    }

    // MARK: - Multimodal feature extraction (Stage 6)

    /// Encode `image` → (1, 256, hidden) features ready to splice at
    /// IMAGE_TOKEN_ID positions. Lazy-loads the vision model on first
    /// call (background prewarm usually has it loaded already).
    public func processImage(_ image: CGImage) throws -> MLMultiArray {
        if visionModel == nil, let url = visionModelURL, let cfg = visionConfig {
            do {
                visionModel = try MLModel(contentsOf: url, configuration: cfg)
                print("[Gemma4Stateful/Vision] model loaded (\(url.lastPathComponent))")
            } catch {
                print("[Gemma4Stateful/Vision] load failed at \(url.path): \(error)")
                throw error
            }
        }
        guard let vm = visionModel else { throw CoreMLLLMError.visionNotAvailable }
        let arr = visionUsesANEBuild
            ? try ImageProcessor.processANE(image, with: vm)
            : try ImageProcessor.process(image, with: vm)
        print("[Gemma4Stateful/Vision] encoded shape=\(arr.shape.map { $0.intValue })")
        return arr
    }

    /// Encode a single video frame at 64 tokens via vision_video.mlmodelc
    /// when present. Throws if no video encoder is loaded — caller is
    /// expected to fall back to 2×2 pooling of `processImage`'s output.
    public func processVideoFrame(_ image: CGImage) throws -> MLMultiArray {
        if videoVisionModel == nil, let url = videoVisionModelURL, let cfg = videoVisionConfig {
            do {
                videoVisionModel = try MLModel(contentsOf: url, configuration: cfg)
                print("[Gemma4Stateful/VideoVision] model loaded (\(url.lastPathComponent))")
            } catch {
                print("[Gemma4Stateful/VideoVision] load failed at \(url.path): \(error)")
                throw error
            }
        }
        guard let vm = videoVisionModel else { throw CoreMLLLMError.visionNotAvailable }
        return try ImageProcessor.processVideoFrame(image, with: vm)
    }

    /// Encode 16 kHz mono PCM → (audio features, actual token count).
    /// Token count is bounded by the encoder's static input size and the
    /// real audio length, matching HF Gemma4AudioFeatureExtractor.
    public func processAudio(_ samples: [Float]) throws -> (MLMultiArray, Int) {
        if audioModel == nil, let url = audioModelURL, let cfg = audioConfig {
            do {
                audioModel = try MLModel(contentsOf: url, configuration: cfg)
                print("[Gemma4Stateful/Audio] model loaded (\(url.lastPathComponent))")
            } catch {
                print("[Gemma4Stateful/Audio] load failed at \(url.path): \(error)")
                throw error
            }
        }
        guard let am = audioModel else { throw CoreMLLLMError.audioNotAvailable }
        guard let mel = melFilterbank else {
            print("[Gemma4Stateful/Audio] melFilterbank=nil (mel_filterbank.bin missing)")
            throw CoreMLLLMError.audioNotAvailable
        }

        // pad-left + unfold → mel frames; 2× Conv2d stride 2 → tokens.
        let padLeft = 160
        let paddedLen = padLeft + samples.count
        let unfoldSize = 321
        let actualMelFrames = max(0, (paddedLen - unfoldSize) / 160 + 1)
        let afterConv1 = (actualMelFrames + 1) / 2
        let actualTokens = min((afterConv1 + 1) / 2, audioNumTokensConfig)

        let features = try AudioProcessor.process(
            samples, with: am, melFilterbank: mel,
            targetFrames: audioMelFrames, projection: audioProjection,
            melFloor: audioMelFloor)
        return (features, actualTokens)
    }

    // MARK: - Multimodal mask + splice helpers (Stage 6)

    /// Compute vision-group ids over the prompt. Each contiguous run of
    /// image-pad / video-pad tokens forms one bidirectional group; text
    /// and audio positions get -1 (causal-only). Mirrors HF Gemma 4
    /// `token_type_ids_mask_function`. Audio doesn't form a vision
    /// group because Gemma 4 audio attention is causal even within an
    /// audio span (different from vision).
    private func computeVisionGroupIds(inputIds: [Int32]) -> [Int] {
        var ids = [Int](repeating: -1, count: inputIds.count)
        var current = -1
        var prev = false
        for i in 0..<inputIds.count {
            let isVision = inputIds[i] == Self.IMAGE_TOKEN_ID
                || inputIds[i] == Self.VIDEO_TOKEN_ID
            if isVision {
                if !prev { current += 1 }
                ids[i] = current
            }
            prev = isVision
        }
        return ids
    }

    /// Reusable per-layer-raw=0 scratch for T=1 multimodal positions.
    private func prlZerosT1Buffer() throws -> MLMultiArray {
        if let buf = prlZerosT1 { return buf }
        guard let mc = modelConfig else {
            throw CoreMLLLMError.modelNotFound("no config")
        }
        let dim = mc.numLayers * mc.perLayerDim
        let arr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dim)], dataType: .float16)
        memset(arr.dataPointer, 0, dim * MemoryLayout<UInt16>.stride)
        prlZerosT1 = arr
        return arr
    }

    /// Splice an encoder row over the embed lookup for IMAGE/VIDEO/AUDIO
    /// placeholder tokens. Returns nil for text tokens (caller falls back
    /// to embed_tokens.lookup). Advances the matching multimodal counter.
    private func multimodalSpliceT1(token: Int32) -> MLMultiArray? {
        guard let mc = modelConfig else { return nil }
        if (token == Self.IMAGE_TOKEN_ID || token == Self.VIDEO_TOKEN_ID),
           let img = mmImageFeatures, mmImageIdx < mmImageNumTokens {
            let row = ImageProcessor.sliceFeature(
                img, at: mmImageIdx, hiddenSize: mc.hiddenSize)
            mmImageIdx += 1
            return row
        }
        if token == Self.AUDIO_TOKEN_ID,
           let aud = mmAudioFeatures, mmAudioIdx < mmAudioNumTokens {
            let row = AudioProcessor.sliceFeature(
                aud, at: mmAudioIdx, hiddenSize: mc.hiddenSize)
            mmAudioIdx += 1
            return row
        }
        return nil
    }

    /// Vision-aware T=1 full causal mask. With T=1 only the current
    /// position has fresh K/V in state — future same-group positions
    /// haven't been computed yet, so unmasking them would let stale
    /// state slots leak into softmax. The vision-aware path therefore
    /// reduces to strict causal at T=1 (kept as a separate function
    /// for symmetry with the T=N path and so multifunction-disabled
    /// fallback turns still take the same code path).
    private func fillFullCausalMaskVisionAware(position p: Int, groupIds _: [Int]) {
        fillFullCausalMask(position: p)
    }

    private func fillSlidingCausalMaskVisionAware(position p: Int, groupIds _: [Int]) {
        fillSlidingCausalMask(position: p)
    }

    /// Vision-aware T-row prefill masks. Same shape as fillBatchMasks
    /// but unmasks within-vision-group attention so image-pad spans
    /// attend bidirectionally. Optimization E: keeps the multifunction
    /// prefill_b8 win on vision turns (else we'd fall back to T=1, ≈8×
    /// slower for the 256-token image span).
    ///
    /// IMPORTANT: bidirectional unmask is bounded to within-batch
    /// positions [startPos, startPos+T-1]. K/V at positions beyond the
    /// current batch is STALE (zeros after a state reset, or worse: an
    /// older prompt's K/V) — unmasking them lets stale slots leak into
    /// softmax with non-trivial weights and corrupts attention. The
    /// legacy 4-chunk prefill (prefillN=1024) didn't hit this because
    /// the whole image group fit in one batch; stateful T=8 splits
    /// each image group across ~32 batches.
    private func fillBatchMasksVisionAware(startPos: Int, T: Int, groupIds: [Int]) {
        let ctx = modelConfig!.contextLength
        let W = modelConfig!.slidingWindow
        let neg = Float16(-65504).bitPattern
        let mf = batchMaskFull!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * ctx)
        let ms = batchMaskSliding!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * W)
        let batchEnd = startPos + T - 1
        for t in 0..<T {
            let p = startPos + t
            let pGroup = (p < groupIds.count) ? groupIds[p] : -1
            let pClamped = min(max(p, 0), ctx - 1)
            // Full mask
            for i in 0..<ctx {
                let causal = i <= pClamped
                let sameGroup = pGroup >= 0
                    && i >= startPos && i <= batchEnd
                    && i < groupIds.count
                    && groupIds[i] == pGroup
                mf[t * ctx + i] = (causal || sameGroup) ? 0 : neg
            }
            // Sliding mask. For p < W slot i corresponds to position i;
            // for p >= W ring rotation makes the mapping non-trivial,
            // but the vision groups (≤256 tok) always fit inside W=512
            // and the within-batch bidirectional bound applies the
            // same way regardless.
            if p < W {
                let valid = min(p + 1, W)
                for i in 0..<W {
                    let causal = i < valid
                    let sameGroup = pGroup >= 0
                        && i >= startPos && i <= batchEnd
                        && i < groupIds.count
                        && groupIds[i] == pGroup
                    ms[t * W + i] = (causal || sameGroup) ? 0 : neg
                }
            } else {
                // All W slots already valid for strict-causal; same-
                // group within-batch positions are subset.
                for i in 0..<W { ms[t * W + i] = 0 }
            }
        }
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
    // MARK: - 1-chunk single-prediction helpers

    /// Single-mlpackage step. Inputs identical to chunk_1's signature
    /// (hidden + per_layer_raw + masks + RoPE + positions); output is
    /// chunk_3's final token_id directly.
    private func step1Chunk(token: Int32, position: Int,
                              state: MLState,
                              opts: MLPredictionOptions) async throws -> Int32 {
        guard let mc = modelConfig, let m = model1Chunk else {
            throw CoreMLLLMError.modelNotFound("1-chunk model not loaded")
        }
        let hidden = try embedTokens!.lookup(
            Int(token), shape: [1, 1, NSNumber(value: mc.hiddenSize)])
        let perLayerRaw = try embedTokensPerLayer!.lookup(
            Int(token),
            shape: [1, 1, NSNumber(value: mc.numLayers * mc.perLayerDim)])
        fillFullCausalMask(position: position)
        fillSlidingCausalMask(position: position)
        setPos(position)
        setRing(position)
        let cosS = try lookupRoPE(table: cosSlidingTable, position: position, dim: 256)
        let sinS = try lookupRoPE(table: sinSlidingTable, position: position, dim: 256)
        let cosF = try lookupRoPE(table: cosFullTable,    position: position, dim: 512)
        let sinF = try lookupRoPE(table: sinFullTable,    position: position, dim: 512)

        let p = FeatureProvider([
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
        let out = try await m.prediction(from: p, using: state, options: opts)
        guard let tokFV = out.featureValue(for: "token_id"),
              let tokArr = tokFV.multiArrayValue
        else { throw CoreMLLLMError.modelNotFound("1-chunk no token_id") }
        return tokArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
    }

    /// 1-chunk batched prefill (T tokens through model.prefill_b<T>).
    private func prefillStep1Chunk(inputIds: [Int32], startBatch: Int,
                                     position: Int, T: Int, state: MLState,
                                     opts: MLPredictionOptions) async throws -> Int32 {
        guard let mc = modelConfig, let m = model1ChunkPrefill,
              let embed = embedTokens, let perLayer = embedTokensPerLayer
        else { throw CoreMLLLMError.modelNotFound("1-chunk prefill not loaded") }
        try ensureBatchScratch(T: T)
        let H = mc.hiddenSize
        let PL = mc.numLayers * mc.perLayerDim
        let hPtr = batchHidden!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * H)
        let plPtr = batchPerLayerRaw!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * PL)
        for t in 0..<T {
            let tok = Int(inputIds[startBatch + t])
            let row = try embed.lookup(tok, shape: [1, 1, NSNumber(value: H)])
            memcpy(hPtr.advanced(by: t * H),
                   row.dataPointer, H * MemoryLayout<UInt16>.stride)
            let plRow = try perLayer.lookup(
                tok, shape: [1, 1, NSNumber(value: PL)])
            memcpy(plPtr.advanced(by: t * PL),
                   plRow.dataPointer, PL * MemoryLayout<UInt16>.stride)
        }
        fillBatchMasks(startPos: position, T: T)
        fillBatchRoPE(table: cosSlidingTable, dst: batchCosS!,
                      startPos: position, T: T, dim: 256)
        fillBatchRoPE(table: sinSlidingTable, dst: batchSinS!,
                      startPos: position, T: T, dim: 256)
        fillBatchRoPE(table: cosFullTable, dst: batchCosF!,
                      startPos: position, T: T, dim: 512)
        fillBatchRoPE(table: sinFullTable, dst: batchSinF!,
                      startPos: position, T: T, dim: 512)
        setPos(position)
        setRing(position)
        let p = FeatureProvider([
            "hidden_states":      MLFeatureValue(multiArray: batchHidden!),
            "causal_mask_full":   MLFeatureValue(multiArray: batchMaskFull!),
            "causal_mask_sliding": MLFeatureValue(multiArray: batchMaskSliding!),
            "per_layer_raw":      MLFeatureValue(multiArray: batchPerLayerRaw!),
            "cos_s": MLFeatureValue(multiArray: batchCosS!),
            "sin_s": MLFeatureValue(multiArray: batchSinS!),
            "cos_f": MLFeatureValue(multiArray: batchCosF!),
            "sin_f": MLFeatureValue(multiArray: batchSinF!),
            "current_pos": fvPos,
            "ring_pos":    fvRing,
        ])
        let out = try await m.prediction(from: p, using: state, options: opts)
        guard let tokFV = out.featureValue(for: "token_id"),
              let tokArr = tokFV.multiArrayValue
        else { throw CoreMLLLMError.modelNotFound("1-chunk prefill no token_id") }
        return tokArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
    }

    // MARK: - 4-chunk / 3-chunk step

    private func step(token: Int32, position: Int,
                       states: (s1: MLState, s2: MLState),
                       opts: MLPredictionOptions) async throws -> Int32 {
        guard let mc = modelConfig else {
            throw CoreMLLLMError.modelNotFound("not loaded")
        }
        // 1) Embed lookups (full hidden + per-layer raw). Multimodal
        //    placeholder tokens splice an encoder feature row instead
        //    and force per_layer_raw=0 so the chunks compute
        //    per_layer_combined entirely from the spliced hidden.
        let hidden: MLMultiArray
        let perLayerRaw: MLMultiArray
        if let mmRow = multimodalSpliceT1(token: token) {
            hidden = mmRow
            perLayerRaw = try prlZerosT1Buffer()
        } else {
            hidden = try embedTokens!.lookup(
                Int(token), shape: [1, 1, NSNumber(value: mc.hiddenSize)])
            perLayerRaw = try embedTokensPerLayer!.lookup(
                Int(token),
                shape: [1, 1, NSNumber(value: mc.numLayers * mc.perLayerDim)])
        }

        // 2) Position-dependent scratch (mask + RoPE + indices). Vision-
        //    aware mask when the prompt contains image/video groups so
        //    in-group positions attend bidirectionally — matches HF
        //    Gemma 4 `token_type_ids_mask_function` and the legacy
        //    ChunkedEngine vision prefill.
        if let groupIds = mmVisionGroupIds {
            fillFullCausalMaskVisionAware(position: position, groupIds: groupIds)
            fillSlidingCausalMaskVisionAware(position: position, groupIds: groupIds)
        } else {
            fillFullCausalMask(position: position)
            fillSlidingCausalMask(position: position)
        }
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
        if is3Chunk {
            // 3-chunk mode: chunk_3 IS the final lm_head + argmax chunk.
            guard let tokFV = out3.featureValue(for: "token_id"),
                  let tokArr = tokFV.multiArrayValue
            else { throw CoreMLLLMError.modelNotFound("chunk_3 (3-chunk final) no token_id") }
            return tokArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
        }

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

    // MARK: - Batched-prefill plumbing (Phase 2b multifunction)

    /// Allocate per-T scratch (hidden / per_layer_raw / mask / RoPE).
    /// Re-entrant — cheaper than re-allocating each call.
    private func ensureBatchScratch(T: Int) throws {
        guard let mc = modelConfig else { return }
        let H = mc.hiddenSize
        let PL = mc.numLayers * mc.perLayerDim
        let ctx = mc.contextLength
        let W = mc.slidingWindow
        let hdS = 256
        let hdF = 512
        if batchHidden == nil || batchHidden!.shape[1].intValue != T {
            batchHidden = try MLMultiArray(
                shape: [1, NSNumber(value: T), NSNumber(value: H)],
                dataType: .float16)
            batchPerLayerRaw = try MLMultiArray(
                shape: [1, NSNumber(value: T), NSNumber(value: PL)],
                dataType: .float16)
            batchMaskFull = try MLMultiArray(
                shape: [1, 1, NSNumber(value: T), NSNumber(value: ctx)],
                dataType: .float16)
            batchMaskSliding = try MLMultiArray(
                shape: [1, 1, NSNumber(value: T), NSNumber(value: W)],
                dataType: .float16)
            batchCosS = try MLMultiArray(
                shape: [1, 1, NSNumber(value: T), NSNumber(value: hdS)],
                dataType: .float16)
            batchSinS = try MLMultiArray(
                shape: [1, 1, NSNumber(value: T), NSNumber(value: hdS)],
                dataType: .float16)
            batchCosF = try MLMultiArray(
                shape: [1, 1, NSNumber(value: T), NSNumber(value: hdF)],
                dataType: .float16)
            batchSinF = try MLMultiArray(
                shape: [1, 1, NSNumber(value: T), NSNumber(value: hdF)],
                dataType: .float16)
        }
    }

    /// Fill T-row causal masks for the prefill window starting at `startPos`.
    private func fillBatchMasks(startPos: Int, T: Int) {
        let ctx = modelConfig!.contextLength
        let W = modelConfig!.slidingWindow
        let neg = Float16(-65504).bitPattern
        let mf = batchMaskFull!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * ctx)
        let ms = batchMaskSliding!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * W)
        for t in 0..<T {
            let p = startPos + t
            // Full mask row: positions [0, p] valid.
            let full = min(max(p, 0), ctx - 1)
            for i in 0..<ctx { mf[t * ctx + i] = i <= full ? 0 : neg }
            // Sliding LEFT-aligned: first (p+1) slots valid for p<W,
            // all W for p>=W. (Same rule as the T=1 path.)
            let valid = min(p + 1, W)
            for i in 0..<W { ms[t * W + i] = i < valid ? 0 : neg }
        }
    }

    /// Copy T RoPE rows from the sidecar table into a (1,1,T,dim) buffer.
    private func fillBatchRoPE(table: Data?, dst: MLMultiArray,
                                startPos: Int, T: Int, dim: Int) {
        let p = dst.dataPointer.bindMemory(to: UInt16.self, capacity: T * dim)
        guard let table else { memset(p, 0, T * dim * 2); return }
        var headerSize = 128
        table.withUnsafeBytes { raw in
            let b = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            headerSize = 10 + (Int(b[8]) | (Int(b[9]) << 8))
        }
        let rowBytes = dim * MemoryLayout<UInt16>.stride
        for t in 0..<T {
            let pos = startPos + t
            let offset = headerSize + pos * rowBytes
            if offset + rowBytes <= table.count {
                _ = table.withUnsafeBytes { raw in
                    memcpy(p.advanced(by: t * dim),
                           raw.baseAddress!.advanced(by: offset), rowBytes)
                }
            } else {
                memset(p.advanced(by: t * dim), 0, rowBytes)
            }
        }
    }

    /// Run one T=N prefill window through the four chunk prefill_bN
    /// functions. Consumes `inputIds[startPos ..< startPos+T]` at
    /// sequence positions `[position, position+T)`. State buffers
    /// advance T slots. Returns the chunk_4 argmax for the LAST batch
    /// position (the prefill's auto-emitted next token).
    private func prefillStep(inputIds: [Int32], startBatch: Int, position: Int,
                              T: Int, states: (s1: MLState, s2: MLState),
                              opts: MLPredictionOptions) async throws -> Int32 {
        guard let mc = modelConfig,
              let c1 = chunk1Prefill, let c2 = chunk2Prefill,
              let c3 = chunk3Prefill,
              let embed = embedTokens, let perLayer = embedTokensPerLayer
        else { throw CoreMLLLMError.modelNotFound("prefill chunks not loaded") }
        if !is3Chunk && chunk4Prefill == nil {
            throw CoreMLLLMError.modelNotFound("prefill chunk_4 missing in 4-chunk mode")
        }

        try ensureBatchScratch(T: T)
        let H = mc.hiddenSize
        let PL = mc.numLayers * mc.perLayerDim

        // 1) Hidden + per_layer_raw: pack T rows into the batch buffer.
        //    Multimodal placeholder tokens splice an encoder row over
        //    the embed lookup and zero per_layer_raw at that position.
        let hPtr = batchHidden!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * H)
        let plPtr = batchPerLayerRaw!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * PL)
        let imgRowPtr = mmImageFeatures?.dataPointer.bindMemory(
            to: UInt16.self, capacity: mmImageFeatures?.count ?? 0)
        let audRowPtr = mmAudioFeatures?.dataPointer.bindMemory(
            to: UInt16.self, capacity: mmAudioFeatures?.count ?? 0)
        for t in 0..<T {
            let tokInt32 = inputIds[startBatch + t]
            let tok = Int(tokInt32)
            if let imgPtr = imgRowPtr,
               (tokInt32 == Self.IMAGE_TOKEN_ID || tokInt32 == Self.VIDEO_TOKEN_ID),
               mmImageIdx < mmImageNumTokens {
                memcpy(hPtr.advanced(by: t * H),
                       imgPtr.advanced(by: mmImageIdx * H),
                       H * MemoryLayout<UInt16>.stride)
                memset(plPtr.advanced(by: t * PL), 0,
                       PL * MemoryLayout<UInt16>.stride)
                mmImageIdx += 1
            } else if let audPtr = audRowPtr,
                      tokInt32 == Self.AUDIO_TOKEN_ID,
                      mmAudioIdx < mmAudioNumTokens {
                memcpy(hPtr.advanced(by: t * H),
                       audPtr.advanced(by: mmAudioIdx * H),
                       H * MemoryLayout<UInt16>.stride)
                memset(plPtr.advanced(by: t * PL), 0,
                       PL * MemoryLayout<UInt16>.stride)
                mmAudioIdx += 1
            } else {
                let row = try embed.lookup(tok, shape: [1, 1, NSNumber(value: H)])
                memcpy(hPtr.advanced(by: t * H),
                       row.dataPointer, H * MemoryLayout<UInt16>.stride)
                let plRow = try perLayer.lookup(
                    tok, shape: [1, 1, NSNumber(value: PL)])
                memcpy(plPtr.advanced(by: t * PL),
                       plRow.dataPointer, PL * MemoryLayout<UInt16>.stride)
            }
        }

        // 2) Position-dependent scratch. Optimization E: vision-aware
        //    mask preserves the multifunction prefill_b8 win on vision
        //    turns (else we'd fall back to T=1, ≈8× slower for the
        //    256-token image span).
        if let groupIds = mmVisionGroupIds {
            fillBatchMasksVisionAware(startPos: position, T: T, groupIds: groupIds)
        } else {
            fillBatchMasks(startPos: position, T: T)
        }
        fillBatchRoPE(table: cosSlidingTable, dst: batchCosS!,
                      startPos: position, T: T, dim: 256)
        fillBatchRoPE(table: sinSlidingTable, dst: batchSinS!,
                      startPos: position, T: T, dim: 256)
        fillBatchRoPE(table: cosFullTable, dst: batchCosF!,
                      startPos: position, T: T, dim: 512)
        fillBatchRoPE(table: sinFullTable, dst: batchSinF!,
                      startPos: position, T: T, dim: 512)
        setPos(position)
        setRing(position)

        let fvHidden = MLFeatureValue(multiArray: batchHidden!)
        let fvPLR = MLFeatureValue(multiArray: batchPerLayerRaw!)
        let fvMF = MLFeatureValue(multiArray: batchMaskFull!)
        let fvMS = MLFeatureValue(multiArray: batchMaskSliding!)
        let fvCS = MLFeatureValue(multiArray: batchCosS!)
        let fvSS = MLFeatureValue(multiArray: batchSinS!)
        let fvCF = MLFeatureValue(multiArray: batchCosF!)
        let fvSF = MLFeatureValue(multiArray: batchSinF!)

        // 3) Chunk 1
        let p1 = FeatureProvider([
            "hidden_states":      fvHidden,
            "causal_mask_full":   fvMF,
            "causal_mask_sliding": fvMS,
            "per_layer_raw":      fvPLR,
            "cos_s": fvCS, "sin_s": fvSS,
            "cos_f": fvCF, "sin_f": fvSF,
            "current_pos": fvPos, "ring_pos": fvRing,
        ])
        let out1 = try await c1.prediction(from: p1, using: states.s1, options: opts)
        guard let h1 = out1.featureValue(for: "hidden_states_out"),
              let plc = out1.featureValue(for: "per_layer_combined_out")
        else { throw CoreMLLLMError.modelNotFound("prefill chunk_1 missing outputs") }

        // 4) Chunk 2
        let p2 = FeatureProvider([
            "hidden_states":      h1,
            "causal_mask_full":   fvMF,
            "causal_mask_sliding": fvMS,
            "per_layer_combined": plc,
            "cos_s": fvCS, "sin_s": fvSS,
            "cos_f": fvCF, "sin_f": fvSF,
            "current_pos": fvPos, "ring_pos": fvRing,
        ])
        let out2 = try await c2.prediction(from: p2, using: states.s2, options: opts)
        guard let h2 = out2.featureValue(for: "hidden_states_out"),
              let kv13k = out2.featureValue(for: "kv13_k"),
              let kv13v = out2.featureValue(for: "kv13_v"),
              let kv14k = out2.featureValue(for: "kv14_k"),
              let kv14v = out2.featureValue(for: "kv14_v")
        else { throw CoreMLLLMError.modelNotFound("prefill chunk_2 missing outputs") }

        // 5) Chunks 3 + 4
        var shared: [String: MLFeatureValue] = [
            "causal_mask_full":   fvMF,
            "causal_mask_sliding": fvMS,
            "per_layer_combined": plc,
            "cos_s": fvCS, "sin_s": fvSS,
            "cos_f": fvCF, "sin_f": fvSF,
            "kv13_k": kv13k, "kv13_v": kv13v,
            "kv14_k": kv14k, "kv14_v": kv14v,
        ]
        var p3map = shared
        p3map["hidden_states"] = h2
        let out3 = try await c3.prediction(
            from: FeatureProvider(p3map), options: opts)
        if is3Chunk {
            // 3-chunk: chunk_3 is the final lm_head + argmax chunk.
            guard let tokFV = out3.featureValue(for: "token_id"),
                  let tokArr = tokFV.multiArrayValue
            else { throw CoreMLLLMError.modelNotFound("prefill chunk_3 (3-chunk final) no token_id") }
            return tokArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
        }
        guard let h3 = out3.featureValue(for: "hidden_states_out") else {
            throw CoreMLLLMError.modelNotFound("prefill chunk_3 missing output")
        }
        var p4map = shared
        p4map["hidden_states"] = h3
        let out4 = try await chunk4Prefill!.prediction(
            from: FeatureProvider(p4map), options: opts)
        guard let tokFV = out4.featureValue(for: "token_id"),
              let tokArr = tokFV.multiArrayValue
        else { throw CoreMLLLMError.modelNotFound("prefill chunk_4 no token_id") }
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
    public func generate(inputIds: [Int32],
                          imageFeatures: MLMultiArray? = nil,
                          imageNumTokens: Int = 0,
                          audioFeatures: MLMultiArray? = nil,
                          audioNumTokens: Int = 0,
                          maxNewTokens: Int = 64,
                          eosTokenIds: Set<Int32> = [],
                          onToken: ((Int32) -> Void)? = nil
    ) async throws -> [Int32] {
        // Bind multimodal state for the duration of this call. step()
        // and prefillStep() consult these to decide whether to splice
        // an encoder row over the embed lookup. defer clears them so a
        // subsequent text-only generate doesn't read stale features.
        mmImageFeatures = imageFeatures
        mmImageNumTokens = imageNumTokens
        mmAudioFeatures = audioFeatures
        mmAudioNumTokens = audioNumTokens
        mmImageIdx = 0
        mmAudioIdx = 0
        let hasMultimodal = imageFeatures != nil || audioFeatures != nil
        mmVisionGroupIds = hasMultimodal ? computeVisionGroupIds(inputIds: inputIds) : nil
        if hasMultimodal {
            let imgShape = imageFeatures?.shape.map { $0.intValue } ?? []
            let audShape = audioFeatures?.shape.map { $0.intValue } ?? []
            let imgPadCount = inputIds.lazy.filter {
                $0 == Self.IMAGE_TOKEN_ID || $0 == Self.VIDEO_TOKEN_ID
            }.count
            let audPadCount = inputIds.lazy.filter { $0 == Self.AUDIO_TOKEN_ID }.count
            let groupCount = (mmVisionGroupIds ?? []).reduce(into: -1) { acc, g in
                if g > acc { acc = g }
            } + 1
            print("[Gemma4Stateful/MM] entry: " +
                  "img feats=\(imgShape) numTok=\(imageNumTokens) padInPrompt=\(imgPadCount) | " +
                  "aud feats=\(audShape) numTok=\(audioNumTokens) padInPrompt=\(audPadCount) | " +
                  "vision groups=\(groupCount)")
        }
        defer {
            if hasMultimodal {
                print("[Gemma4Stateful/MM] exit: " +
                      "imageIdx=\(mmImageIdx)/\(mmImageNumTokens) " +
                      "audioIdx=\(mmAudioIdx)/\(mmAudioNumTokens)")
            }
            mmImageFeatures = nil
            mmAudioFeatures = nil
            mmImageNumTokens = 0
            mmAudioNumTokens = 0
            mmImageIdx = 0
            mmAudioIdx = 0
            mmVisionGroupIds = nil
        }
        if is1Chunk {
            return try await generate1Chunk(
                inputIds: inputIds, maxNewTokens: maxNewTokens,
                eosTokenIds: eosTokenIds, onToken: onToken)
        }
        // 3-chunk mode: chunk_4 is intentionally nil. Require chunks 1/2/3
        // always; require chunk_4 only in 4-chunk mode.
        guard let chunk1, chunk2 != nil, chunk3 != nil,
              (is3Chunk || chunk4 != nil) else {
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

        // Advance multimodal counters past the resumed prefix so the
        // first new token splices the correct encoder row. Caller is
        // responsible for resetPersistedState() when image/audio
        // changes (LLMRunner does this on attachment fingerprint
        // mismatch); the engine assumes consistent features here.
        if resumeAt > 0 && hasMultimodal {
            for j in 0..<resumeAt {
                let t = inputIds[j]
                if t == Self.IMAGE_TOKEN_ID || t == Self.VIDEO_TOKEN_ID {
                    mmImageIdx += 1
                } else if t == Self.AUDIO_TOKEN_ID {
                    mmAudioIdx += 1
                }
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
        // Phase 2b — batched prefill with multifunction `prefill_bN` when
        // available. Constraint: ring_pos+T <= W (no mid-batch wrap on the
        // sliding cache). For position < W we never wrap; once position
        // crosses W we fall back to T=1 for any window that would wrap.
        var i = resumeAt
        var batchedSteps = 0
        var t1Steps = 0
        let W = mc.slidingWindow
        if hasMultifunctionPrefill {
            let T = prefillT
            while i + T <= inputIds.count {
                let ringStart = position % W
                if ringStart + T > W { break }   // wrap — fall back to T=1
                prefillPredicted = try await prefillStep(
                    inputIds: inputIds, startBatch: i, position: position,
                    T: T, states: (state1, state2), opts: opts)
                lastToken = inputIds[i + T - 1]
                position += T
                i += T
                batchedSteps += 1
            }
        }
        for j in i..<inputIds.count {
            let tok = inputIds[j]
            prefillPredicted = try await step(
                token: tok, position: position,
                states: (state1, state2), opts: opts)
            lastToken = tok
            position += 1
            t1Steps += 1
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
        let mfTag = hasMultifunctionPrefill
            ? " [batched=\(batchedSteps)x\(prefillT) t1=\(t1Steps)]"
            : ""
        print("[Gemma4Stateful] prefill \(prefillTokCount) tok in "
              + String(format: "%.0fms (%.1f tok/s)%@%@ | decode %d tok in %.0fms (%.1f tok/s)",
                        prefillMs,
                        Double(max(prefillTokCount, 1))
                            / max(prefillMs / 1000, 1e-3),
                        resumeTag, mfTag,
                        decoded.count, decodeMs, lastDecodeTokensPerSecond))
        return decoded
    }

    // MARK: - 1-chunk generate (single mlpackage / single MLState)

    private func generate1Chunk(inputIds: [Int32], maxNewTokens: Int,
                                  eosTokenIds: Set<Int32>,
                                  onToken: ((Int32) -> Void)?
    ) async throws -> [Int32] {
        guard let m1 = model1Chunk else {
            throw CoreMLLLMError.modelNotFound("1-chunk model not loaded")
        }
        guard let mc = modelConfig else {
            throw CoreMLLLMError.modelNotFound("no config")
        }
        if inputIds.isEmpty { return [] }
        if inputIds.count >= mc.contextLength {
            throw CoreMLLLMError.modelNotFound(
                "prompt (\(inputIds.count) tok) >= ctx (\(mc.contextLength))")
        }

        // Cross-turn resume: in 1-chunk mode the persisted state lives in
        // model1ChunkState. Persisted prefix must be a STRICT prefix.
        var resumeAt = 0
        if let _ = model1ChunkState, !persistedInputIds.isEmpty {
            let cap = min(persistedInputIds.count, inputIds.count)
            var l = 0
            while l < cap && persistedInputIds[l] == inputIds[l] { l += 1 }
            if l == persistedInputIds.count && l < inputIds.count && l > 0 {
                resumeAt = l
            }
        }

        let state: MLState
        if resumeAt > 0, let s = model1ChunkState {
            state = s
            print("[Gemma4Stateful 1c] RESUME L=\(resumeAt) "
                  + "(persisted=\(persistedInputIds.count), new=\(inputIds.count))")
            persistedInputIds = []
            persistedPosition = 0
        } else {
            state = m1.makeState()
            model1ChunkState = state
            persistedInputIds = []
            persistedPosition = 0
        }
        let opts = MLPredictionOptions()

        var position = resumeAt
        var lastToken: Int32 = inputIds[max(resumeAt - 1, 0)]
        var prefillPredicted: Int32 = 0
        var batchedSteps = 0
        var t1Steps = 0
        let W = mc.slidingWindow

        let t0 = CFAbsoluteTimeGetCurrent()
        var i = resumeAt
        if hasMultifunctionPrefill {
            let T = prefillT
            while i + T <= inputIds.count {
                let ringStart = position % W
                if ringStart + T > W { break }
                prefillPredicted = try await prefillStep1Chunk(
                    inputIds: inputIds, startBatch: i, position: position,
                    T: T, state: state, opts: opts)
                lastToken = inputIds[i + T - 1]
                position += T
                i += T
                batchedSteps += 1
            }
        }
        for j in i..<inputIds.count {
            let tok = inputIds[j]
            prefillPredicted = try await step1Chunk(
                token: tok, position: position, state: state, opts: opts)
            lastToken = tok
            position += 1
            t1Steps += 1
        }
        let prefillEnd = CFAbsoluteTimeGetCurrent()

        var decoded: [Int32] = []
        if maxNewTokens > 0 && inputIds.count > resumeAt {
            decoded.append(prefillPredicted)
            onToken?(prefillPredicted)
            lastToken = prefillPredicted
        }
        while decoded.count < maxNewTokens {
            if eosTokenIds.contains(lastToken) { break }
            if position >= mc.contextLength { break }
            let next = try await step1Chunk(
                token: lastToken, position: position, state: state, opts: opts)
            decoded.append(next)
            onToken?(next)
            lastToken = next
            position += 1
        }
        let t1 = CFAbsoluteTimeGetCurrent()

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
        let mfTag = hasMultifunctionPrefill
            ? " [batched=\(batchedSteps)x\(prefillT) t1=\(t1Steps)]"
            : ""
        print("[Gemma4Stateful 1c] prefill \(prefillTokCount) tok in "
              + String(format: "%.0fms (%.1f tok/s)%@%@ | decode %d tok in %.0fms (%.1f tok/s)",
                        prefillMs,
                        Double(max(prefillTokCount, 1))
                            / max(prefillMs / 1000, 1e-3),
                        resumeTag, mfTag,
                        decoded.count, decodeMs, lastDecodeTokensPerSecond))
        return decoded
    }
}
