// Gemma4StatefulMultimodalEngine — Stage 8 runtime that pairs the
// 3-chunk merged stateful Linear decode path with single-function
// T=288 prefill chunks and the Stage 6 multimodal feature splice.
//
// Sibling of (not subclass of) Gemma4StatefulEngine. The legacy class
// keeps shipping the multifunction prefill_b8 path bit-identical for
// users who don't want multimodal; this class layers on:
//
//   - 3 separate single-function prefill mlpackages (T=288):
//       prefill_T288/chunk_1_prefill_T288.mlmodelc
//       prefill_T288/chunk_2_3way_prefill_T288.mlmodelc
//       prefill_T288/chunk_3_prefill_T288.mlmodelc
//     iPhone ANE 18 rejects multifunction T>1 + dual MLState. Probe
//     2 verified single-function T=288 compiles in 7.3 s on A19 Pro.
//
//   - 4 MLStates (decode_s1/s2 + prefill_s1/s2). After each prefill
//     pass we memcpy kv_cache_sliding + kv_cache_full from prefill
//     state into decode state via the NS_REFINED_FOR_SWIFT
//     `withMultiArray(for:handler:)` bridge.
//
//   - vision (256/image), video (64/frame), audio (~188/2 sec)
//     encoder splice at IMAGE/VIDEO/AUDIO_TOKEN_ID positions.
//     Vision-aware mask preserves bidirectional within-image
//     attention during prefill.
//
// Both Gemma 4 E2B (35 layers) and E4B (42 layers) drive through this
// engine — chunk topology comes from the loaded mlpackages, layer
// counts come from model_config.json. The engine itself is
// dimension-agnostic.

import Accelerate
import CoreGraphics
import CoreML
import Foundation

@available(iOS 18.0, macOS 15.0, *)
public final class Gemma4StatefulMultimodalEngine {
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

    // Decode chunks. Two layouts auto-detected at load():
    //  - 3-chunk merged: chunk_1 (own) + chunk_2 (own + KV-shared
    //    internal) + chunk_3 (KV-shared + lm_head). chunk_3 emits
    //    token_id. Used for E2B (chunk_2 fits ANE budget).
    //  - 4-chunk split: chunk_1 (own) + chunk_2 (own only) + chunk_3
    //    (KV-shared, no lm_head) + chunk_4 (KV-shared + lm_head).
    //    chunk_4 emits token_id. Used for E4B because the merged
    //    21-layer chunk_2 trips iPhone ANE 18 MIL→EIR translation
    //    (`std::bad_cast`); splitting keeps each subgraph small enough.
    private var decodeChunk1: MLModel?
    private var decodeChunk2: MLModel?
    private var decodeChunk3: MLModel?
    private var decodeChunk4: MLModel?
    private var is4Chunk: Bool = false

    // T=288 single-function prefill chunks (separate mlpackages).
    private static let kPrefillT: Int = 288
    private var prefillChunk1: MLModel?
    private var prefillChunk2: MLModel?
    private var prefillChunk3: MLModel?
    private var prefillChunk4: MLModel?

    /// True when both decode and prefill chunk sets loaded successfully.
    public var hasT288Prefill: Bool {
        let core = prefillChunk1 != nil && prefillChunk2 != nil
            && prefillChunk3 != nil
        return is4Chunk ? (core && prefillChunk4 != nil) : core
    }

    // Per-chunk MLStates. Decode + prefill paths each have their own
    // since they bind to different MLModel instances. chunk_3 is
    // stateless (KV-shared from chunk_2's kv13/kv14 outputs) for both
    // decode and prefill.
    private var decodeState1: MLState?
    private var decodeState2: MLState?
    private var prefillState1: MLState?
    private var prefillState2: MLState?

    // Sidecars (same as Gemma4StatefulEngine).
    private var embedTokens: EmbeddingLookup?
    private var embedTokensPerLayer: EmbeddingLookup?
    private var cosSlidingTable: Data?
    private var sinSlidingTable: Data?
    private var cosFullTable: Data?
    private var sinFullTable: Data?

    // T=1 decode scratch.
    private var maskFull: MLMultiArray!
    private var maskSliding: MLMultiArray!
    private var fvMaskFull: MLFeatureValue!
    private var fvMaskSliding: MLFeatureValue!
    private var posScratch: MLMultiArray!
    private var ringScratch: MLMultiArray!
    private var fvPos: MLFeatureValue!
    private var fvRing: MLFeatureValue!

    // T=288 prefill scratch (allocated once at load).
    private var batchHidden: MLMultiArray?
    private var batchPerLayerRaw: MLMultiArray?
    private var batchMaskFull: MLMultiArray?
    private var batchMaskSliding: MLMultiArray?
    private var batchCosS: MLMultiArray?
    private var batchSinS: MLMultiArray?
    private var batchCosF: MLMultiArray?
    private var batchSinF: MLMultiArray?

    // Cross-turn KV reuse — only on decode states (prefill states are
    // reusable scratch and get overwritten each generate()). The LCP
    // match invariant: persistedInputIds is a strict prefix of the
    // next prompt's inputIds. LLMRunner is responsible for calling
    // resetPersistedState() on chat clear or attachment change.
    private var persistedInputIds: [Int32] = []
    private var persistedPosition: Int = 0

    // MARK: - Multimodal (Stage 6 helpers, ported)

    private static let IMAGE_TOKEN_ID: Int32 = 258880
    private static let AUDIO_TOKEN_ID: Int32 = 258881
    private static let VIDEO_TOKEN_ID: Int32 = 258884

    private var visionModelURL: URL?
    private var visionConfig: MLModelConfiguration?
    private var visionModel: MLModel?
    private var visionUsesANEBuild: Bool = false

    private var videoVisionModelURL: URL?
    private var videoVisionConfig: MLModelConfiguration?
    private var videoVisionModel: MLModel?

    private var audioModelURL: URL?
    private var audioConfig: MLModelConfiguration?
    private var audioModel: MLModel?
    private var melFilterbank: [Float]?
    private var audioProjection: AudioProcessor.ProjectionWeights?
    private var audioMelFrames: Int = 200
    private var audioNumTokensConfig: Int = 188
    private var audioMelFloor: Float = 0.001

    public var hasVision: Bool { visionModelURL != nil }
    public var hasVideoVision: Bool { videoVisionModelURL != nil }
    public var hasAudio: Bool { audioModelURL != nil }
    public var defaultAudioNumTokens: Int { audioNumTokensConfig }

    // Per-call multimodal binding.
    private var mmImageFeatures: MLMultiArray?
    private var mmImageNumTokens: Int = 0
    private var mmAudioFeatures: MLMultiArray?
    private var mmAudioNumTokens: Int = 0
    private var mmImageIdx: Int = 0
    private var mmAudioIdx: Int = 0
    private var mmVisionGroupIds: [Int]?

    // Reusable PLR=0 scratch for T=1 multimodal positions.
    private var prlZerosT1: MLMultiArray?

    // MARK: - Init / Load

    public init(config: Config = Config()) {
        self.cfg = config
    }

    /// Drop the cross-turn KV cache. Call when chat history clears,
    /// the vision/audio prefix changes, or any other prompt-prefix
    /// invariant breaks.
    public func resetPersistedState() {
        decodeState1 = nil
        decodeState2 = nil
        prefillState1 = nil
        prefillState2 = nil
        persistedInputIds = []
        persistedPosition = 0
    }

    public func load(modelDirectory: URL) async throws {
        resetPersistedState()
        modelDir = modelDirectory
        let mc = try ModelConfig.load(from: modelDirectory)
        modelConfig = mc

        let mcfg = MLModelConfiguration()
        mcfg.computeUnits = cfg.computeUnits

        decodeChunk1 = try openChunk("chunk_1", in: modelDirectory, cfg: mcfg)
        decodeChunk2 = try openChunk("chunk_2", in: modelDirectory, cfg: mcfg)
        decodeChunk3 = try openChunk("chunk_3", in: modelDirectory, cfg: mcfg)

        // 4-chunk vs 3-chunk detection: chunk_4 present → 4-chunk;
        // chunk_3.token_id output → 3-chunk merged final.
        let chunk4Mlc = modelDirectory.appendingPathComponent("chunk_4.mlmodelc")
        let chunk4Pkg = modelDirectory.appendingPathComponent("chunk_4.mlpackage")
        let has4 = FileManager.default.fileExists(atPath: chunk4Mlc.path)
            || FileManager.default.fileExists(atPath: chunk4Pkg.path)
        if has4 {
            decodeChunk4 = try openChunk("chunk_4", in: modelDirectory, cfg: mcfg)
            is4Chunk = true
            print("[Gemma4MM] 4-chunk decode layout (chunk_2 own / chunk_3 shared / chunk_4 final)")
        } else {
            print("[Gemma4MM] 3-chunk merged decode layout (chunk_3 = final)")
        }

        // T=288 prefill chunks live under prefill_T288/ in the bundle
        // layout. Failing to find them is fatal — this engine has no
        // T=1 prefill fallback (the legacy engine handles that path).
        let pfDir = modelDirectory.appendingPathComponent("prefill_T288")
        prefillChunk1 = try openChunk(
            "chunk_1_prefill_T288", in: pfDir, cfg: mcfg)
        if is4Chunk {
            prefillChunk2 = try openChunk(
                "chunk_2_prefill_T288", in: pfDir, cfg: mcfg)
            prefillChunk3 = try openChunk(
                "chunk_3_prefill_T288", in: pfDir, cfg: mcfg)
            prefillChunk4 = try openChunk(
                "chunk_4_prefill_T288", in: pfDir, cfg: mcfg)
        } else {
            prefillChunk2 = try openChunk(
                "chunk_2_3way_prefill_T288", in: pfDir, cfg: mcfg)
            prefillChunk3 = try openChunk(
                "chunk_3_prefill_T288", in: pfDir, cfg: mcfg)
        }
        print("[Gemma4MM] T=\(Self.kPrefillT) prefill chunks loaded "
              + "(\(is4Chunk ? "4-chunk" : "3-chunk merged"))")

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

        try ensureBatchScratch(T: Self.kPrefillT)

        loadMultimodalEncoders(modelDirectory: modelDirectory)
    }

    private func openChunk(_ name: String, in dir: URL,
                           cfg: MLModelConfiguration) throws -> MLModel {
        let mlc = dir.appendingPathComponent("\(name).mlmodelc")
        let pkg = dir.appendingPathComponent("\(name).mlpackage")

        let url: URL
        if FileManager.default.fileExists(atPath: mlc.path) {
            url = mlc
        } else if FileManager.default.fileExists(atPath: pkg.path) {
            url = try MLModel.compileModel(at: pkg)
        } else {
            throw CoreMLLLMError.modelNotFound(
                "\(name).mlmodelc/.mlpackage not found in \(dir.path)")
        }

        // Try the requested compute units first. iPhone ANE 18 has been
        // observed to fail MIL→EIR translation on some merged chunks
        // (`std::bad_cast` in `_ANECompiler::ANECCompile()`); fall back to
        // CPU+GPU so the engine still loads. Gates per-chunk via env
        // var `LLM_GEMMA4MM_FORCE_GPU=<chunk_name>[,<chunk_name>...]`.
        let envForceGPU = ProcessInfo.processInfo
            .environment["LLM_GEMMA4MM_FORCE_GPU"] ?? ""
        let forced = envForceGPU.split(separator: ",")
            .map { String($0).trimmingCharacters(in: .whitespaces) }
        if forced.contains(name) {
            print("[Gemma4MM] \(name) — LLM_GEMMA4MM_FORCE_GPU forces cpuAndGPU")
            let gpu = MLModelConfiguration()
            gpu.computeUnits = .cpuAndGPU
            return try MLModel(contentsOf: url, configuration: gpu)
        }
        do {
            return try MLModel(contentsOf: url, configuration: cfg)
        } catch {
            print("[Gemma4MM] \(name) load failed on \(cfg.computeUnits.rawValue): \(error). Retrying on cpuAndGPU.")
            let gpu = MLModelConfiguration()
            gpu.computeUnits = .cpuAndGPU
            return try MLModel(contentsOf: url, configuration: gpu)
        }
    }

    // MARK: - Multimodal encoder loading (ported from Stage 6 02ac583)

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
            print("[Gemma4MM/Vision] selected \(url.lastPathComponent) → \(visionUsesANEBuild ? "ANE" : "GPU")")
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
            print("[Gemma4MM] multimodal encoders: vision=\(hasVision) " +
                  "video=\(hasVideoVision) audio=\(hasAudio)")
        }
    }

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
                print("[Gemma4MM/Vision] prewarm done in \(String(format: "%.1f", dt))s")
            } catch {
                print("[Gemma4MM/Vision] prewarm failed: \(error)")
            }
        }
    }

    // MARK: - Multimodal feature extraction

    public func processImage(_ image: CGImage) throws -> MLMultiArray {
        if visionModel == nil, let url = visionModelURL, let cfg = visionConfig {
            visionModel = try MLModel(contentsOf: url, configuration: cfg)
        }
        guard let vm = visionModel else { throw CoreMLLLMError.visionNotAvailable }
        return visionUsesANEBuild
            ? try ImageProcessor.processANE(image, with: vm)
            : try ImageProcessor.process(image, with: vm)
    }

    public func processVideoFrame(_ image: CGImage) throws -> MLMultiArray {
        if videoVisionModel == nil, let url = videoVisionModelURL, let cfg = videoVisionConfig {
            videoVisionModel = try MLModel(contentsOf: url, configuration: cfg)
        }
        guard let vm = videoVisionModel else { throw CoreMLLLMError.visionNotAvailable }
        return try ImageProcessor.processVideoFrame(image, with: vm)
    }

    public func processAudio(_ samples: [Float]) throws -> (MLMultiArray, Int) {
        if audioModel == nil, let url = audioModelURL, let cfg = audioConfig {
            audioModel = try MLModel(contentsOf: url, configuration: cfg)
        }
        guard let am = audioModel else { throw CoreMLLLMError.audioNotAvailable }
        guard let mel = melFilterbank else { throw CoreMLLLMError.audioNotAvailable }

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

    // MARK: - Multimodal mask + splice helpers

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

    // MARK: - Mask + position helpers (T=1 decode)

    private func fillFullCausalMask(position: Int) {
        let ctx = modelConfig!.contextLength
        let dst = maskFull.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        let neg = Float16(-65504).bitPattern
        let p = min(max(position, 0), ctx - 1)
        for i in 0..<ctx { dst[i] = i <= p ? 0 : neg }
    }

    private func fillSlidingCausalMask(position: Int) {
        let W = modelConfig!.slidingWindow
        let dst = maskSliding.dataPointer.bindMemory(to: UInt16.self, capacity: W)
        let neg = Float16(-65504).bitPattern
        let valid = min(position + 1, W)
        for i in 0..<W { dst[i] = i < valid ? 0 : neg }
    }

    private func fillFullCausalMaskVisionAware(position p: Int, groupIds: [Int]) {
        let ctx = modelConfig!.contextLength
        let neg = Float16(-65504).bitPattern
        let mp = maskFull.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        let pGroup = (p < groupIds.count) ? groupIds[p] : -1
        let pClamped = min(max(p, 0), ctx - 1)
        for i in 0..<ctx {
            let causal = i <= pClamped
            let sameGroup = pGroup >= 0 && i < groupIds.count && groupIds[i] == pGroup
            mp[i] = (causal || sameGroup) ? 0 : neg
        }
    }

    private func fillSlidingCausalMaskVisionAware(position p: Int, groupIds: [Int]) {
        let W = modelConfig!.slidingWindow
        let neg = Float16(-65504).bitPattern
        let mp = maskSliding.dataPointer.bindMemory(to: UInt16.self, capacity: W)
        if p >= W {
            for i in 0..<W { mp[i] = 0 }
            return
        }
        let pGroup = (p < groupIds.count) ? groupIds[p] : -1
        let valid = min(p + 1, W)
        for i in 0..<W {
            let causal = i < valid
            let sameGroup = pGroup >= 0 && i < groupIds.count && groupIds[i] == pGroup
            mp[i] = (causal || sameGroup) ? 0 : neg
        }
    }

    private func setPos(_ pos: Int) {
        posScratch.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(pos)
    }

    private func setRing(_ pos: Int) {
        let W = modelConfig!.slidingWindow
        ringScratch.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(pos % W)
    }

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

    // MARK: - Batched (T=288) prefill scratch

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

    /// Fill T-row causal masks for a contiguous batch starting at
    /// `startPos`. Padded rows (t >= validCount) are filled as
    /// duplicates of row validCount-1 so the auto-emit at row T-1 is
    /// the model's prediction for the LAST valid prompt token (=
    /// first post-prompt token).
    private func fillBatchMasks(startPos: Int, T: Int, validCount: Int,
                                 groupIds: [Int]?) {
        let ctx = modelConfig!.contextLength
        let W = modelConfig!.slidingWindow
        let neg = Float16(-65504).bitPattern
        let mf = batchMaskFull!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * ctx)
        let ms = batchMaskSliding!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * W)
        let effectiveT = max(validCount, 1)
        for t in 0..<T {
            // Padded rows duplicate row validCount-1.
            let row = min(t, effectiveT - 1)
            let p = startPos + row
            let pGroup = (groupIds != nil && p < groupIds!.count) ? groupIds![p] : -1
            let pClamped = min(max(p, 0), ctx - 1)
            for i in 0..<ctx {
                let causal = i <= pClamped
                let sameGroup = pGroup >= 0 && i < (groupIds?.count ?? 0)
                    && groupIds![i] == pGroup
                mf[t * ctx + i] = (causal || sameGroup) ? 0 : neg
            }
            if p < W {
                let valid = min(p + 1, W)
                for i in 0..<W {
                    let causal = i < valid
                    let sameGroup = pGroup >= 0 && i < (groupIds?.count ?? 0)
                        && groupIds![i] == pGroup
                    ms[t * W + i] = (causal || sameGroup) ? 0 : neg
                }
            } else {
                for i in 0..<W { ms[t * W + i] = 0 }
            }
        }
    }

    /// Padded rows (t >= validCount) duplicate row validCount-1 so the
    /// chunk graph sees a coherent batch where the auto-emit at row
    /// T-1 corresponds to the LAST valid prompt position.
    private func fillBatchRoPE(table: Data?, dst: MLMultiArray,
                                startPos: Int, T: Int, validCount: Int,
                                dim: Int) {
        let p = dst.dataPointer.bindMemory(to: UInt16.self, capacity: T * dim)
        guard let table else { memset(p, 0, T * dim * 2); return }
        var headerSize = 128
        table.withUnsafeBytes { raw in
            let b = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            headerSize = 10 + (Int(b[8]) | (Int(b[9]) << 8))
        }
        let rowBytes = dim * MemoryLayout<UInt16>.stride
        let effectiveT = max(validCount, 1)
        for t in 0..<T {
            let row = min(t, effectiveT - 1)
            let pos = startPos + row
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

    // MARK: - State bridge (prefill MLState → decode MLState)

    /// Copies kv_cache_sliding (and kv_cache_full when present) from
    /// `src` into `dst`. The withMultiArray closure scope is the only
    /// legal window to access the buffer pointer; nested closures
    /// keep both pointers live for the memcpy.
    private func bridgeKVState(from src: MLState, to dst: MLState) {
        for name in ["kv_cache_sliding", "kv_cache_full"] {
            src.withMultiArray(for: name) { srcArr in
                dst.withMultiArray(for: name) { dstArr in
                    guard srcArr.count == dstArr.count else { return }
                    memcpy(dstArr.dataPointer, srcArr.dataPointer,
                           srcArr.count * MemoryLayout<UInt16>.stride)
                }
            }
        }
    }

    // MARK: - Reusable feature provider

    private final class FeatureProvider: NSObject, MLFeatureProvider {
        let map: [String: MLFeatureValue]
        let featureNames: Set<String>
        init(_ map: [String: MLFeatureValue]) {
            self.map = map
            self.featureNames = Set(map.keys)
        }
        func featureValue(for name: String) -> MLFeatureValue? { map[name] }
    }

    // MARK: - T=1 decode step (3 chunks)

    private func decodeStep(token: Int32, position: Int,
                              opts: MLPredictionOptions) async throws -> Int32 {
        guard let mc = modelConfig,
              let c1 = decodeChunk1, let c2 = decodeChunk2, let c3 = decodeChunk3,
              let s1 = decodeState1, let s2 = decodeState2 else {
            throw CoreMLLLMError.modelNotFound("decode chunks/states not loaded")
        }

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

        if let groupIds = mmVisionGroupIds {
            fillFullCausalMaskVisionAware(position: position, groupIds: groupIds)
            fillSlidingCausalMaskVisionAware(position: position, groupIds: groupIds)
        } else {
            fillFullCausalMask(position: position)
            fillSlidingCausalMask(position: position)
        }
        setPos(position)
        setRing(position)

        let cosS = try lookupRoPE(table: cosSlidingTable, position: position, dim: 256)
        let sinS = try lookupRoPE(table: sinSlidingTable, position: position, dim: 256)
        let cosF = try lookupRoPE(table: cosFullTable,    position: position, dim: 512)
        let sinF = try lookupRoPE(table: sinFullTable,    position: position, dim: 512)

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
        let out1 = try await c1.prediction(from: p1, using: s1, options: opts)
        guard let h1 = out1.featureValue(for: "hidden_states_out"),
              let plc = out1.featureValue(for: "per_layer_combined_out")
        else { throw CoreMLLLMError.modelNotFound("decode chunk_1 missing outputs") }

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
        let out2 = try await c2.prediction(from: p2, using: s2, options: opts)
        guard let h2 = out2.featureValue(for: "hidden_states_out"),
              let kv13k = out2.featureValue(for: "kv13_k"),
              let kv13v = out2.featureValue(for: "kv13_v"),
              let kv14k = out2.featureValue(for: "kv14_k"),
              let kv14v = out2.featureValue(for: "kv14_v")
        else { throw CoreMLLLMError.modelNotFound("decode chunk_2 missing outputs") }

        var sharedInputs: [String: MLFeatureValue] = [
            "causal_mask_full":   fvMaskFull,
            "causal_mask_sliding": fvMaskSliding,
            "per_layer_combined": plc,
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "kv13_k": kv13k, "kv13_v": kv13v,
            "kv14_k": kv14k, "kv14_v": kv14v,
        ]
        var p3map = sharedInputs
        p3map["hidden_states"] = h2
        let out3 = try await c3.prediction(from: FeatureProvider(p3map), options: opts)
        if !is4Chunk {
            guard let tokFV = out3.featureValue(for: "token_id"),
                  let tokArr = tokFV.multiArrayValue
            else { throw CoreMLLLMError.modelNotFound("decode chunk_3 (3-chunk final) no token_id") }
            return tokArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
        }
        // 4-chunk: chunk_3 = KV-shared no lm_head; chunk_4 = KV-shared + lm_head.
        guard let h3 = out3.featureValue(for: "hidden_states_out") else {
            throw CoreMLLLMError.modelNotFound("decode chunk_3 missing hidden_states_out")
        }
        guard let c4 = decodeChunk4 else {
            throw CoreMLLLMError.modelNotFound("decode chunk_4 not loaded")
        }
        var p4map = sharedInputs
        p4map["hidden_states"] = h3
        let out4 = try await c4.prediction(from: FeatureProvider(p4map), options: opts)
        guard let tokFV = out4.featureValue(for: "token_id"),
              let tokArr = tokFV.multiArrayValue
        else { throw CoreMLLLMError.modelNotFound("decode chunk_4 no token_id") }
        return tokArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
    }

    // MARK: - T=288 single-function prefill pass

    /// One prefill pass over inputIds[startBatch ..< startBatch+validCount]
    /// at sequence positions [position, position+validCount). Padded to
    /// T=288 with -inf source masks. Returns the next token (chunk_3
    /// argmax for batch row validCount-1).
    private func prefillStepT288(inputIds: [Int32], startBatch: Int,
                                  position: Int, validCount: Int,
                                  opts: MLPredictionOptions) async throws -> Int32 {
        guard let mc = modelConfig,
              let c1 = prefillChunk1, let c2 = prefillChunk2, let c3 = prefillChunk3,
              let s1 = prefillState1, let s2 = prefillState2,
              let embed = embedTokens, let perLayer = embedTokensPerLayer
        else { throw CoreMLLLMError.modelNotFound("prefill T=288 not loaded") }

        let T = Self.kPrefillT
        precondition(validCount > 0 && validCount <= T,
                     "validCount=\(validCount) out of (0, \(T)]")
        try ensureBatchScratch(T: T)
        let H = mc.hiddenSize
        let PL = mc.numLayers * mc.perLayerDim

        let hPtr = batchHidden!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * H)
        let plPtr = batchPerLayerRaw!.dataPointer.bindMemory(
            to: UInt16.self, capacity: T * PL)
        let imgRowPtr = mmImageFeatures?.dataPointer.bindMemory(
            to: UInt16.self, capacity: mmImageFeatures?.count ?? 0)
        let audRowPtr = mmAudioFeatures?.dataPointer.bindMemory(
            to: UInt16.self, capacity: mmAudioFeatures?.count ?? 0)

        // Pack valid rows (real tokens) + zero-pad the tail.
        for t in 0..<validCount {
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
        // Pad rows [validCount..T-1] by duplicating row validCount-1.
        // Same hidden + per_layer_raw — combined with mask/RoPE that
        // pin padded rows to position validCount-1, the chunk graph
        // computes row T-1's output identical to row validCount-1's,
        // making the chunk_3 argmax at row T-1 a valid prediction
        // of the first post-prompt token. Multimodal counters do NOT
        // advance for padded rows (they already advanced for the
        // validCount real-token rows above).
        if validCount < T && validCount > 0 {
            let srcRowH = hPtr.advanced(by: (validCount - 1) * H)
            let srcRowPLR = plPtr.advanced(by: (validCount - 1) * PL)
            for t in validCount..<T {
                memcpy(hPtr.advanced(by: t * H), srcRowH,
                       H * MemoryLayout<UInt16>.stride)
                memcpy(plPtr.advanced(by: t * PL), srcRowPLR,
                       PL * MemoryLayout<UInt16>.stride)
            }
        }

        fillBatchMasks(startPos: position, T: T,
                       validCount: validCount, groupIds: mmVisionGroupIds)
        fillBatchRoPE(table: cosSlidingTable, dst: batchCosS!,
                      startPos: position, T: T, validCount: validCount, dim: 256)
        fillBatchRoPE(table: sinSlidingTable, dst: batchSinS!,
                      startPos: position, T: T, validCount: validCount, dim: 256)
        fillBatchRoPE(table: cosFullTable, dst: batchCosF!,
                      startPos: position, T: T, validCount: validCount, dim: 512)
        fillBatchRoPE(table: sinFullTable, dst: batchSinF!,
                      startPos: position, T: T, validCount: validCount, dim: 512)
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

        let p1 = FeatureProvider([
            "hidden_states":      fvHidden,
            "causal_mask_full":   fvMF,
            "causal_mask_sliding": fvMS,
            "per_layer_raw":      fvPLR,
            "cos_s": fvCS, "sin_s": fvSS,
            "cos_f": fvCF, "sin_f": fvSF,
            "current_pos": fvPos, "ring_pos": fvRing,
        ])
        let out1 = try await c1.prediction(from: p1, using: s1, options: opts)
        guard let h1 = out1.featureValue(for: "hidden_states_out"),
              let plc = out1.featureValue(for: "per_layer_combined_out")
        else { throw CoreMLLLMError.modelNotFound("prefill T=288 chunk_1 missing outputs") }

        let p2 = FeatureProvider([
            "hidden_states":      h1,
            "causal_mask_full":   fvMF,
            "causal_mask_sliding": fvMS,
            "per_layer_combined": plc,
            "cos_s": fvCS, "sin_s": fvSS,
            "cos_f": fvCF, "sin_f": fvSF,
            "current_pos": fvPos, "ring_pos": fvRing,
        ])
        let out2 = try await c2.prediction(from: p2, using: s2, options: opts)
        guard let h2 = out2.featureValue(for: "hidden_states_out"),
              let kv13k = out2.featureValue(for: "kv13_k"),
              let kv13v = out2.featureValue(for: "kv13_v"),
              let kv14k = out2.featureValue(for: "kv14_k"),
              let kv14v = out2.featureValue(for: "kv14_v")
        else { throw CoreMLLLMError.modelNotFound("prefill T=288 chunk_2 missing outputs") }

        var sharedInputs: [String: MLFeatureValue] = [
            "causal_mask_full":   fvMF,
            "causal_mask_sliding": fvMS,
            "per_layer_combined": plc,
            "cos_s": fvCS, "sin_s": fvSS,
            "cos_f": fvCF, "sin_f": fvSF,
            "kv13_k": kv13k, "kv13_v": kv13v,
            "kv14_k": kv14k, "kv14_v": kv14v,
        ]
        var p3map = sharedInputs
        p3map["hidden_states"] = h2
        let out3 = try await c3.prediction(from: FeatureProvider(p3map), options: opts)
        if !is4Chunk {
            guard let tokFV = out3.featureValue(for: "token_id"),
                  let tokArr = tokFV.multiArrayValue
            else { throw CoreMLLLMError.modelNotFound("prefill T=288 chunk_3 (3-chunk final) no token_id") }
            return tokArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
        }
        // 4-chunk: chunk_3 emits hidden_states_out only; chunk_4 emits token_id.
        guard let h3 = out3.featureValue(for: "hidden_states_out") else {
            throw CoreMLLLMError.modelNotFound("prefill T=288 chunk_3 missing hidden_states_out")
        }
        guard let c4 = prefillChunk4 else {
            throw CoreMLLLMError.modelNotFound("prefill T=288 chunk_4 not loaded")
        }
        var p4map = sharedInputs
        p4map["hidden_states"] = h3
        let out4 = try await c4.prediction(from: FeatureProvider(p4map), options: opts)
        guard let tokFV = out4.featureValue(for: "token_id"),
              let tokArr = tokFV.multiArrayValue
        else { throw CoreMLLLMError.modelNotFound("prefill T=288 chunk_4 no token_id") }
        // chunk_4 emits argmax at batch row T-1. When validCount < T
        // we replicate row validCount-1 across padded rows, so row T-1
        // is functionally identical to row validCount-1 and the
        // argmax is the valid first-post-prompt-token prediction.
        return tokArr.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0]
    }

    // MARK: - Generate (T=288 prefill + bridge + T=1 decode)

    /// Run the prompt through T=288 prefill passes, bridge KV state
    /// into the decode chunks, then T=1 decode up to maxNewTokens.
    /// imageFeatures / audioFeatures are pre-encoded by the caller
    /// (typically LLMRunner) via processImage / processAudio.
    public func generate(inputIds: [Int32],
                          imageFeatures: MLMultiArray? = nil,
                          imageNumTokens: Int = 0,
                          audioFeatures: MLMultiArray? = nil,
                          audioNumTokens: Int = 0,
                          maxNewTokens: Int = 512,
                          eosTokenIds: Set<Int32> = [],
                          onToken: ((Int32) -> Void)? = nil
    ) async throws -> [Int32] {
        guard let mc = modelConfig else {
            throw CoreMLLLMError.modelNotFound("Gemma4MM: no config")
        }
        guard let c1 = decodeChunk1, let c2 = decodeChunk2,
              decodeChunk3 != nil,
              let pc1 = prefillChunk1, let pc2 = prefillChunk2,
              prefillChunk3 != nil
        else { throw CoreMLLLMError.modelNotFound("Gemma4MM: not loaded") }
        if inputIds.isEmpty { return [] }
        if inputIds.count >= mc.contextLength {
            throw CoreMLLLMError.modelNotFound(
                "prompt (\(inputIds.count) tokens) >= ctx (\(mc.contextLength))")
        }

        // Bind multimodal state for the duration.
        mmImageFeatures = imageFeatures
        mmImageNumTokens = imageNumTokens
        mmAudioFeatures = audioFeatures
        mmAudioNumTokens = audioNumTokens
        mmImageIdx = 0
        mmAudioIdx = 0
        let hasMultimodal = imageFeatures != nil || audioFeatures != nil
        mmVisionGroupIds = hasMultimodal ? computeVisionGroupIds(inputIds: inputIds) : nil
        defer {
            mmImageFeatures = nil
            mmAudioFeatures = nil
            mmImageNumTokens = 0
            mmAudioNumTokens = 0
            mmImageIdx = 0
            mmAudioIdx = 0
            mmVisionGroupIds = nil
        }

        // Cross-turn resume: persisted decode state is reusable; if
        // persistedInputIds is a strict prefix of inputIds, skip the
        // prefix and only T=288-prefill the suffix. Prefill states
        // are scratch — always rebuilt for the suffix.
        var resumeAt = 0
        let canResume = decodeState1 != nil && decodeState2 != nil
            && !persistedInputIds.isEmpty
        if canResume {
            let cap = min(persistedInputIds.count, inputIds.count)
            var l = 0
            while l < cap && persistedInputIds[l] == inputIds[l] { l += 1 }
            if l == persistedInputIds.count && l < inputIds.count && l > 0 {
                resumeAt = l
            }
        }

        // Advance multimodal counters past resumed prefix.
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

        if resumeAt == 0 {
            // Fresh decode states.
            decodeState1 = c1.makeState()
            decodeState2 = c2.makeState()
            persistedInputIds = []
            persistedPosition = 0
        } else {
            print("[Gemma4MM] RESUME L=\(resumeAt) " +
                  "(persisted=\(persistedInputIds.count), new=\(inputIds.count))")
            persistedInputIds = []
            persistedPosition = 0
        }
        let opts = MLPredictionOptions()

        let suffixCount = inputIds.count - resumeAt
        var position = resumeAt
        var lastToken: Int32 = inputIds[max(resumeAt - 1, 0)]
        var prefillPredicted: Int32 = 0
        var passes = 0

        let t0 = CFAbsoluteTimeGetCurrent()

        if suffixCount > 0 {
            // Always-fresh prefill states for this generate call.
            prefillState1 = pc1.makeState()
            prefillState2 = pc2.makeState()

            let T = Self.kPrefillT
            var i = resumeAt
            while i < inputIds.count {
                let remaining = inputIds.count - i
                let validCount = min(remaining, T)
                prefillPredicted = try await prefillStepT288(
                    inputIds: inputIds, startBatch: i, position: position,
                    validCount: validCount, opts: opts)
                position += validCount
                lastToken = inputIds[i + validCount - 1]
                i += validCount
                passes += 1
            }

            // Bridge prefill KV → decode KV (full buffer memcpy each).
            if let ps1 = prefillState1, let ds1 = decodeState1 {
                bridgeKVState(from: ps1, to: ds1)
            }
            if let ps2 = prefillState2, let ds2 = decodeState2 {
                bridgeKVState(from: ps2, to: ds2)
            }
            // Drop prefill states — they're rebuilt next generate().
            prefillState1 = nil
            prefillState2 = nil
        }
        let prefillEnd = CFAbsoluteTimeGetCurrent()

        // The last prefill pass always auto-emits a valid next token
        // (full batch — padded rows duplicate row validCount-1, so
        // row T-1's argmax is the first-post-prompt-token prediction).
        var decoded: [Int32] = []
        if maxNewTokens > 0 && suffixCount > 0 {
            decoded.append(prefillPredicted)
            onToken?(prefillPredicted)
            lastToken = prefillPredicted
        }
        while decoded.count < maxNewTokens {
            if eosTokenIds.contains(lastToken) { break }
            if position >= mc.contextLength { break }
            let next = try await decodeStep(
                token: lastToken, position: position, opts: opts)
            decoded.append(next)
            onToken?(next)
            lastToken = next
            position += 1
        }
        let t1 = CFAbsoluteTimeGetCurrent()

        // Persist consumed tokens for next-turn LCP match.
        let consumed = decoded.dropLast()
        var newPersisted = inputIds
        newPersisted.append(contentsOf: consumed)
        persistedInputIds = newPersisted
        persistedPosition = newPersisted.count

        let prefillMs = (prefillEnd - t0) * 1000
        let decodeMs = (t1 - prefillEnd) * 1000
        if decodeMs > 0 && decoded.count > 1 {
            lastDecodeTokensPerSecond = Double(decoded.count - 1) / (decodeMs / 1000)
        }
        let resumeTag = resumeAt > 0 ? " [resumed L=\(resumeAt)]" : ""
        print("[Gemma4MM] prefill \(suffixCount) tok in " +
              String(format: "%.0fms (%.1f tok/s)%@ [T=288 passes=%d] | decode %d tok in %.0fms (%.1f tok/s)",
                      prefillMs,
                      Double(max(suffixCount, 1)) / max(prefillMs / 1000, 1e-3),
                      resumeTag, passes,
                      decoded.count, decodeMs, lastDecodeTokensPerSecond))
        return decoded
    }
}
