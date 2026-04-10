import Accelerate
import CoreML
import Foundation
import Tokenizers
import UIKit

/// Manages CoreML LLM model loading and inference.
/// Supports monolithic model or chunked model (for large models on iPhone).
@Observable
final class LLMRunner {
    var isLoaded = false
    var isGenerating = false
    var loadingStatus = "Not loaded"
    var tokensPerSecond: Double = 0
    var modelName = ""
    var hasVision = false

    // Monolithic model
    private var model: MLModel?
    private var state: MLState?

    // Chunked model (SWA 4-chunk, ANE-optimized)
    private var chunk1: MLModel?
    private var chunk2: MLModel?
    private var chunk3: MLModel?
    private var chunk4: MLModel?
    private var chunk1State: MLState?  // unused
    private var chunk2State: MLState?  // unused
    private var isChunked = false

    // Prefill chunks (seq=N batch prefill; optional, decode-only if missing).
    // N=512 lets a single CoreML call cover multimodal prompts (~296 tokens:
    // 280 image placeholders + ~16 text) and most text-only prompts.
    private var prefillChunk1: MLModel?
    private var prefillChunk2: MLModel?
    private var prefillChunk3: MLModel?
    private var prefillChunk4: MLModel?
    private let prefillN = 512
    // SWA KV buffers: separate sliding (W=512) and full (ctx) per chunk
    private var kSliding1: MLMultiArray?  // (7, 1, W, max_hd)
    private var vSliding1: MLMultiArray?
    private var kFull1: MLMultiArray?     // (1, 1, ctx, max_hd)
    private var vFull1: MLMultiArray?
    private var kSliding2: MLMultiArray?  // (5, 1, W, max_hd)
    private var vSliding2: MLMultiArray?
    private var kFull2: MLMultiArray?     // (2, 1, ctx, max_hd)
    private var vFull2: MLMultiArray?
    private var slidingWindow = 512

    // Vision (lazy-loaded to save memory)
    private var visionModel: MLModel?
    private var visionModelURL: URL?
    private var visionConfig: MLModelConfiguration?

    // Root folder of the currently loaded model (for MLComputePlan inspection).
    private var modelFolderURL: URL?

    // External embeddings (for chunked model without embedded vocab tables)
    private var embedTokens: EmbeddingLookup?
    private var embedPerLayer: EmbeddingLookup?
    private var perLayerProjWeight: Data?  // (8960, 1536) float16
    private var perLayerProjF32: [Float]? // (8960, 1536) float32 for Accelerate
    private var perLayerNormWeight: Data?  // (256,) float32

    // Pre-computed RoPE tables (for chunked model)
    private var cosSlidingTable: Data?  // (max_len, 256) float16
    private var sinSlidingTable: Data?
    private var cosFullTable: Data?     // (max_len, 512) float16
    private var sinFullTable: Data?

    // Shared
    private var tokenizer: (any Tokenizer)?
    private var contextLength = 512
    private var hiddenSize = 1536
    private var perLayerDim = 256
    private var architecture = "gemma4"
    private var useExternalPLE = false
    private var currentPosition = 0

    // Profiling
    private var profileEmbed: Double = 0
    private var profilePLE: Double = 0
    private var profilePredict: Double = 0
    private var profileCount: Int = 0
    private var embedScale: Float = 39.19
    private var perLayerProjScale: Float = 0.0255
    private var perLayerInputScale: Float = 0.707
    private var perLayerEmbedScale: Float = 16.0

    // MARK: - Loading

    func loadModel(from url: URL) async throws {
        let folder = url.deletingLastPathComponent()
        modelFolderURL = folder

        // Config
        loadingStatus = "Reading config..."
        let configURL = folder.appendingPathComponent("model_config.json")
        if let data = try? Data(contentsOf: configURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            contextLength = json["context_length"] as? Int ?? 512
            architecture = json["architecture"] as? String ?? "gemma4"
            hiddenSize = json["hidden_size"] as? Int ?? 1536
            perLayerDim = json["per_layer_dim"] as? Int ?? 256
            modelName = json["model_name"] as? String ?? "Model"
            if let es = json["embed_scale"] as? Double { embedScale = Float(es) }
            if let ps = json["per_layer_model_projection_scale"] as? Double { perLayerProjScale = Float(ps) }
            if let is_ = json["per_layer_input_scale"] as? Double { perLayerInputScale = Float(is_) }
            if let es2 = json["per_layer_embed_scale"] as? Double { perLayerEmbedScale = Float(es2) }
            if let sw = json["sliding_window"] as? Int { slidingWindow = sw }
        }

        let mlConfig = MLModelConfiguration()
        // Verified on Mac: model fully ANE-compatible, 34 tok/s with forced ANE.
        // On iPhone, .cpuAndNeuralEngine forces ANE and falls back to CPU (not GPU)
        // for unsupported ops. GPU is excluded to guarantee ANE usage.
        mlConfig.computeUnits = .cpuAndNeuralEngine

        let chunk1URL = findModel(in: folder, name: "chunk1")
        if chunk1URL != nil {
            try await loadChunked(folder: folder, config: mlConfig)
        } else {
            try await loadMonolithic(url: url, folder: folder, config: mlConfig)
        }

        // Vision model: defer loading to save memory (loaded on first image)
        // Vision uses .cpuAndGPU because it's not ANE-optimized (ANE compile fails).
        if findModel(in: folder, name: "vision") != nil {
            hasVision = true
            visionModelURL = findModel(in: folder, name: "vision")
            let visionCfg = MLModelConfiguration()
            visionCfg.computeUnits = .cpuAndGPU
            visionConfig = visionCfg
        }

        // Tokenizer
        loadingStatus = "Loading tokenizer..."
        let tokDir = folder.appendingPathComponent("hf_model")
        tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)

        isLoaded = true
        currentPosition = 0
        loadingStatus = "Ready"
    }

    private func loadMonolithic(url: URL, folder: URL, config: MLModelConfiguration) async throws {
        let modelcURL = folder.appendingPathComponent("model.mlmodelc")
        if FileManager.default.fileExists(atPath: modelcURL.path) {
            loadingStatus = "Loading model..."
            model = try MLModel(contentsOf: modelcURL, configuration: config)
        } else {
            loadingStatus = "Compiling model..."
            let compiled = try await MLModel.compileModel(at: url)
            model = try MLModel(contentsOf: compiled, configuration: config)
        }
        state = model?.makeState()
        isChunked = false

        // Check if model uses external embeddings (lite model has per_layer_combined input)
        let hasExternalPLE = model?.modelDescription.inputDescriptionsByName["per_layer_combined"] != nil
        if hasExternalPLE {
            loadingStatus = "Loading external embeddings..."
            let vocabSize = 262144
            let nlayers = 35
            let etURL = folder.appendingPathComponent("embed_tokens_q8.bin")
            let etScalesURL = folder.appendingPathComponent("embed_tokens_scales.bin")
            let eplURL = folder.appendingPathComponent("embed_tokens_per_layer_q8.bin")
            let eplScalesURL = folder.appendingPathComponent("embed_tokens_per_layer_scales.bin")

            if FileManager.default.fileExists(atPath: etURL.path) {
                embedTokens = try EmbeddingLookup(dataURL: etURL, scalesURL: etScalesURL,
                                                   vocabSize: vocabSize, dim: hiddenSize, scale: embedScale)
                embedPerLayer = try EmbeddingLookup(dataURL: eplURL, scalesURL: eplScalesURL,
                                                     vocabSize: vocabSize, dim: nlayers * perLayerDim, scale: perLayerEmbedScale)
            }

            perLayerProjWeight = try? Data(contentsOf: folder.appendingPathComponent("per_layer_projection.bin"), options: .mappedIfSafe)
            perLayerNormWeight = try? Data(contentsOf: folder.appendingPathComponent("per_layer_norm_weight.bin"), options: .mappedIfSafe)

            // Pre-convert projection to float32 for Accelerate BLAS
            if let projData = perLayerProjWeight {
                loadingStatus = "Converting projection..."
                let count = nlayers * perLayerDim * hiddenSize
                var f32 = [Float](repeating: 0, count: count)
                projData.withUnsafeBytes { raw in
                    let f16Ptr = raw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                    for i in 0..<count {
                        f32[i] = Float(Float16(bitPattern: f16Ptr[i]))
                    }
                }
                perLayerProjF32 = f32
                perLayerProjWeight = nil
            }

            useExternalPLE = true
        }
    }

    private func loadChunked(folder: URL, config mlConfig: MLModelConfiguration) async throws {
        // Stateless 4-chunk lite (SPLIT layout — separate decode/prefill mlpackages):
        // chunk1: layers 0-7 + embedding, KV cache I/O (8 slots)
        // chunk2: layers 8-14 (contains KV sources 13/14), KV cache I/O (7 slots) + kv13/14 output
        // chunk3: layers 15-24 (all shared), no KV cache, uses kv13/14
        // chunk4: layers 25-34 (all shared) + norm + lm_head, no KV cache, uses kv13/14
        //
        // NOTE: We previously used multifunction mlpackages (one file per chunk with
        // both decode and prefill functions) for 50% download savings, but iPhone
        // ANE compiler rejects the multifunction layout with:
        //   "MIL->EIR translation: std::bad_cast"
        // Reverting to split layout (separate mlmodelc for decode and prefill)
        // until the multifunction iPhone compile issue is root-caused.
        // Load all 4 decode chunks. The first call per chunk on a given
        // device triggers ANE compilation + caching, which can take 30s-2min.
        // Print progress so we can see exactly where it hangs if it does.
        func loadOne(_ name: String, _ progress: String) throws -> MLModel {
            print("[Load] \(progress) starting: \(name)")
            let t0 = CFAbsoluteTimeGetCurrent()
            let url = findModel(in: folder, name: name)!
            let m = try MLModel(contentsOf: url, configuration: mlConfig)
            let dt = CFAbsoluteTimeGetCurrent() - t0
            print("[Load] \(progress) done in \(String(format: "%.1f", dt))s: \(name)")
            return m
        }

        loadingStatus = "Loading decode chunk 1/4 (first run = ANE compile, can take 1-2 min)..."
        chunk1 = try loadOne("chunk1", "decode 1/4")
        loadingStatus = "Loading decode chunk 2/4..."
        chunk2 = try loadOne("chunk2", "decode 2/4")
        loadingStatus = "Loading decode chunk 3/4..."
        chunk3 = try loadOne("chunk3", "decode 3/4")
        loadingStatus = "Loading decode chunk 4/4..."
        chunk4 = try loadOne("chunk4", "decode 4/4")

        // Load prefill chunks as separate files (graceful fallback if missing).
        if findModel(in: folder, name: "prefill_chunk1") != nil {
            loadingStatus = "Loading prefill chunk 1/4..."
            prefillChunk1 = try? loadOne("prefill_chunk1", "prefill 1/4")
            loadingStatus = "Loading prefill chunk 2/4..."
            prefillChunk2 = try? loadOne("prefill_chunk2", "prefill 2/4")
            loadingStatus = "Loading prefill chunk 3/4..."
            prefillChunk3 = try? loadOne("prefill_chunk3", "prefill 3/4")
            loadingStatus = "Loading prefill chunk 4/4..."
            prefillChunk4 = try? loadOne("prefill_chunk4", "prefill 4/4")
        }
        print("[Load] All chunks loaded successfully")

        // Allocate persistent SWA KV cache buffers
        loadingStatus = "Allocating KV cache..."
        try allocateSWAKV(maxHd: 512)

        // Load external embeddings
        loadingStatus = "Loading external embeddings..."
        let vocabSize = 262144
        let nlayers = 35
        let etURL = folder.appendingPathComponent("embed_tokens_q8.bin")
        let etScalesURL = folder.appendingPathComponent("embed_tokens_scales.bin")
        let eplURL = folder.appendingPathComponent("embed_tokens_per_layer_q8.bin")
        let eplScalesURL = folder.appendingPathComponent("embed_tokens_per_layer_scales.bin")

        if FileManager.default.fileExists(atPath: etURL.path) {
            embedTokens = try EmbeddingLookup(dataURL: etURL, scalesURL: etScalesURL,
                                               vocabSize: vocabSize, dim: hiddenSize, scale: embedScale)
            embedPerLayer = try EmbeddingLookup(dataURL: eplURL, scalesURL: eplScalesURL,
                                                 vocabSize: vocabSize, dim: nlayers * perLayerDim, scale: perLayerEmbedScale)
        }

        perLayerProjWeight = try? Data(contentsOf: folder.appendingPathComponent("per_layer_projection.bin"), options: .mappedIfSafe)
        perLayerNormWeight = try? Data(contentsOf: folder.appendingPathComponent("per_layer_norm_weight.bin"), options: .mappedIfSafe)

        // Pre-convert projection to float32 for Accelerate BLAS
        if let projData = perLayerProjWeight {
            loadingStatus = "Converting projection..."
            let count = nlayers * perLayerDim * hiddenSize
            var f32 = [Float](repeating: 0, count: count)
            projData.withUnsafeBytes { raw in
                let f16Ptr = raw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                for i in 0..<count {
                    f32[i] = Float(Float16(bitPattern: f16Ptr[i]))
                }
            }
            perLayerProjF32 = f32
            perLayerProjWeight = nil
        }

        // Pre-computed RoPE tables (cos/sin, per position)
        loadingStatus = "Loading RoPE tables..."
        cosSlidingTable = try? Data(contentsOf: folder.appendingPathComponent("cos_sliding.npy"), options: .mappedIfSafe)
        sinSlidingTable = try? Data(contentsOf: folder.appendingPathComponent("sin_sliding.npy"), options: .mappedIfSafe)
        cosFullTable = try? Data(contentsOf: folder.appendingPathComponent("cos_full.npy"), options: .mappedIfSafe)
        sinFullTable = try? Data(contentsOf: folder.appendingPathComponent("sin_full.npy"), options: .mappedIfSafe)

        useExternalPLE = true
        isChunked = true
    }

    /// Allocate SWA KV buffers: separate sliding (W=512) and full (ctx) caches.
    private func allocateSWAKV(maxHd: Int) throws {
        let ctx = contextLength
        let W = slidingWindow
        func zeros(slots: Int, seqLen: Int) throws -> MLMultiArray {
            let arr = try MLMultiArray(
                shape: [NSNumber(value: slots), 1, NSNumber(value: seqLen), NSNumber(value: maxHd)],
                dataType: .float16
            )
            memset(arr.dataPointer, 0, slots * seqLen * maxHd * MemoryLayout<UInt16>.stride)
            return arr
        }
        // Chunk1: 7 sliding + 1 full
        kSliding1 = try zeros(slots: 7, seqLen: W)
        vSliding1 = try zeros(slots: 7, seqLen: W)
        kFull1 = try zeros(slots: 1, seqLen: ctx)
        vFull1 = try zeros(slots: 1, seqLen: ctx)
        // Chunk2: 5 sliding + 2 full
        kSliding2 = try zeros(slots: 5, seqLen: W)
        vSliding2 = try zeros(slots: 5, seqLen: W)
        kFull2 = try zeros(slots: 2, seqLen: ctx)
        vFull2 = try zeros(slots: 2, seqLen: ctx)
    }

    /// Run a dummy prediction to force model compilation and catch ANE errors early.
    private func dummyPredict(model: MLModel, state: MLState) throws {
        let ctx = contextLength
        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
        ids[[0, 0] as [NSNumber]] = NSNumber(value: Int32(2))
        let pos = try MLMultiArray(shape: [1], dataType: .int32)
        pos[0] = NSNumber(value: Int32(0))
        let mask = try makeCausalMask(position: 0, contextLength: ctx)
        let umask = try makeUpdateMask(position: 0, contextLength: ctx)

        var dict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: ids),
            "position_ids": MLFeatureValue(multiArray: pos),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
        ]
        let inputNames = model.modelDescription.inputDescriptionsByName
        if inputNames["per_layer_combined"] != nil {
            let plc = try MLMultiArray(shape: [1, 1, NSNumber(value: 35 * perLayerDim)], dataType: .float16)
            memset(plc.dataPointer, 0, 35 * perLayerDim * MemoryLayout<UInt16>.stride)
            dict["per_layer_combined"] = MLFeatureValue(multiArray: plc)
        }
        if inputNames["image_embedding"] != nil {
            let img = try MLMultiArray(shape: [1, 1, NSNumber(value: hiddenSize)], dataType: .float16)
            memset(img.dataPointer, 0, hiddenSize * MemoryLayout<UInt16>.stride)
            dict["image_embedding"] = MLFeatureValue(multiArray: img)
        }
        _ = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict), using: state)
    }

    private func findModel(in folder: URL, name: String) -> URL? {
        let compiled = folder.appendingPathComponent("\(name).mlmodelc")
        if FileManager.default.fileExists(atPath: compiled.path) { return compiled }
        let pkg = folder.appendingPathComponent("\(name).mlpackage")
        if FileManager.default.fileExists(atPath: pkg.path) { return pkg }
        return nil
    }

    // MARK: - Generation

    func generate(messages: [ChatMessage], image: CGImage? = nil) async throws -> AsyncStream<String> {
        guard tokenizer != nil, (model != nil || chunk1 != nil) else {
            throw NSError(domain: "LLMRunner", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }

        isGenerating = true
        let prompt = buildPrompt(messages: messages, hasImage: image != nil)
        let tokenIDs = tokenizer!.encode(text: prompt)

        var imageFeatures: MLMultiArray?
        if let image {
            // Lazy-load vision model on first use
            if visionModel == nil, let url = visionModelURL, let cfg = visionConfig {
                loadingStatus = "Loading vision model..."
                visionModel = try MLModel(contentsOf: url, configuration: cfg)
            }
            if let vm = visionModel {
                imageFeatures = try processImage(image, with: vm)
            }
        }

        resetConversation()

        return AsyncStream { continuation in
            Task {
                defer { self.isGenerating = false }
                do {
                    let IMAGE_TOKEN_ID = 258880
                    var imageIdx = 0
                    var nextID = 0

                    // Prefill with progress — wrap each step in autoreleasepool
                    // to drain CoreML MLMultiArray allocations (prevents memory growth)
                    self.loadingStatus = "Prefill 0/\(tokenIDs.count)..."
                    let ctxLimit = self.contextLength
                    if tokenIDs.count >= ctxLimit {
                        continuation.yield("[Error: prompt too long (\(tokenIDs.count) >= \(ctxLimit)). Try a shorter question or smaller image.]")
                        continuation.finish()
                        return
                    }

                    // Hybrid path: batched prefill for first up to N tokens,
                    // then per-token decode for the rest. Prefill bug from earlier
                    // (q_norm pre-scaling overflow) is fixed in v0.3.
                    let havePrefill = self.isChunked
                        && self.prefillChunk1 != nil
                        && self.prefillChunk2 != nil
                        && self.prefillChunk3 != nil
                        && self.prefillChunk4 != nil
                    let prefillLen = min(tokenIDs.count, self.prefillN)
                    let useHybrid = havePrefill && prefillLen > 0

                    if useHybrid {
                        self.loadingStatus = "Prefill (batch \(prefillLen)/\(tokenIDs.count))..."
                        try autoreleasepool {
                            let batch = Array(tokenIDs[0..<prefillLen])
                            nextID = try self.runPrefill(tokenIDs: batch, imageFeatures: imageFeatures)
                        }
                        // Advance imageIdx by image tokens consumed in the prefill batch.
                        imageIdx = tokenIDs[0..<prefillLen].filter { $0 == IMAGE_TOKEN_ID }.count
                        self.currentPosition = prefillLen

                        // Per-token decode for remaining prompt tokens (if any).
                        if prefillLen < tokenIDs.count {
                            for step in prefillLen..<tokenIDs.count {
                                let tid = tokenIDs[step]
                                try autoreleasepool {
                                    if tid == IMAGE_TOKEN_ID, let feats = imageFeatures, imageIdx < 256 {
                                        let imgEmb = self.sliceFeature(feats, at: imageIdx)
                                        nextID = try self.predictStep(tokenID: 0, position: step, imageEmbedding: imgEmb)
                                        imageIdx += 1
                                    } else {
                                        nextID = try self.predictStep(tokenID: tid, position: step)
                                    }
                                }
                                self.currentPosition = step + 1
                                self.loadingStatus = "Prefill \(step + 1)/\(tokenIDs.count)..."
                            }
                        }
                    } else {
                        for (step, tid) in tokenIDs.enumerated() {
                            try autoreleasepool {
                                if tid == IMAGE_TOKEN_ID, let feats = imageFeatures, imageIdx < 256 {
                                    let imgEmb = self.sliceFeature(feats, at: imageIdx)
                                    nextID = try self.predictStep(tokenID: 0, position: step, imageEmbedding: imgEmb)
                                    imageIdx += 1
                                } else {
                                    nextID = try self.predictStep(tokenID: tid, position: step)
                                }
                            }
                            self.currentPosition = step + 1
                            self.loadingStatus = "Prefill \(step + 1)/\(tokenIDs.count)..."
                        }
                    }
                    self.loadingStatus = "Generating..."

                    let startTime = CFAbsoluteTimeGetCurrent()
                    var tokenCount = 0
                    let eosIDs: Set<Int> = [1, 106, 151645]

                    for _ in 0..<256 {
                        if eosIDs.contains(nextID) { break }
                        // Stop if we hit the context length
                        if self.currentPosition >= ctxLimit { break }
                        let text = self.tokenizer!.decode(tokens: [nextID])
                        continuation.yield(text)
                        tokenCount += 1
                        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                        if elapsed > 0 { self.tokensPerSecond = Double(tokenCount) / elapsed }
                        try autoreleasepool {
                            nextID = try self.predictStep(tokenID: nextID, position: self.currentPosition)
                        }
                        self.currentPosition += 1
                    }
                    self.loadingStatus = "Ready"
                } catch {
                    self.loadingStatus = "Error: \(error.localizedDescription)"
                    continuation.yield("[Error: \(error.localizedDescription)]")
                }
                continuation.finish()
            }
        }
    }

    func resetConversation() {
        if isChunked {
            // Zero all SWA buffers
            for buf in [kSliding1, vSliding1, kFull1, vFull1, kSliding2, vSliding2, kFull2, vFull2] {
                if let b = buf {
                    memset(b.dataPointer, 0, b.count * MemoryLayout<UInt16>.stride)
                }
            }
        } else {
            state = model?.makeState()
        }
        currentPosition = 0
    }

    // MARK: - Prediction (dispatches to monolithic or chunked)

    private func predictStep(tokenID: Int, position: Int, imageEmbedding: MLMultiArray? = nil) throws -> Int {
        if isChunked {
            return try predictChunked(tokenID: tokenID, position: position, imageEmbedding: imageEmbedding)
        } else {
            return try predictMonolithic(tokenID: tokenID, position: position, imageEmbedding: imageEmbedding)
        }
    }

    // MARK: - Monolithic Prediction

    private func predictMonolithic(tokenID: Int, position: Int, imageEmbedding: MLMultiArray? = nil) throws -> Int {
        guard let model, let state else {
            throw NSError(domain: "LLMRunner", code: 0,
                          userInfo: [NSLocalizedDescriptionKey: "Model or state not initialized"])
        }
        let ctx = contextLength

        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
        ids[[0, 0] as [NSNumber]] = NSNumber(value: Int32(tokenID))
        let pos = try MLMultiArray(shape: [1], dataType: .int32)
        pos[0] = NSNumber(value: Int32(position))
        let mask = try makeCausalMask(position: position, contextLength: ctx)
        let umask = try makeUpdateMask(position: position, contextLength: ctx)

        var dict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: ids),
            "position_ids": MLFeatureValue(multiArray: pos),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
        ]

        // External PLE: compute per_layer_combined and pass as input
        if useExternalPLE {
            let t0 = CFAbsoluteTimeGetCurrent()
            let emb = try embedTokens!.lookup(tokenID, shape: [1, 1, NSNumber(value: hiddenSize)])
            let t1 = CFAbsoluteTimeGetCurrent()
            let plc = try computePerLayerCombined(tokenID: tokenID, embedding: emb)
            let t2 = CFAbsoluteTimeGetCurrent()
            profileEmbed += (t1 - t0)
            profilePLE += (t2 - t1)
            dict["per_layer_combined"] = MLFeatureValue(multiArray: plc)
        }

        // Only pass image_embedding if the model accepts it
        let inputNames = model.modelDescription.inputDescriptionsByName
        if inputNames["image_embedding"] != nil {
            let hs = hiddenSize
            let imgEmb: MLMultiArray
            if let imageEmbedding {
                imgEmb = imageEmbedding
            } else {
                imgEmb = try MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16)
                memset(imgEmb.dataPointer, 0, hs * MemoryLayout<UInt16>.stride)
            }
            dict["image_embedding"] = MLFeatureValue(multiArray: imgEmb)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: dict)
        let tp = CFAbsoluteTimeGetCurrent()
        let output = try model.prediction(from: input, using: state)
        profilePredict += (CFAbsoluteTimeGetCurrent() - tp)
        profileCount += 1

        if profileCount % 10 == 0 {
            let n = Double(profileCount)
            let eMs = profileEmbed/n * 1000
            let pMs = profilePLE/n * 1000
            let prMs = profilePredict/n * 1000
            let total = eMs + pMs + prMs
            print(String(format: "[Profile] emb=%.1fms ple=%.1fms predict=%.1fms total=%.1fms (%.1f tok/s)",
                         eMs, pMs, prMs, total, 1000.0/total))
        }

        return output.featureValue(for: "token_id")!.multiArrayValue![0].intValue
    }

    // MARK: - Chunked Prediction

    private func predictChunked(tokenID: Int, position: Int, imageEmbedding: MLMultiArray? = nil) throws -> Int {
        // SWA 4-chunk architecture:
        // chunk1: layers 0-7 (7 sliding + 1 full), explicit KV I/O for both types
        // chunk2: layers 8-14 (5 sliding + 2 full), outputs kv13 (W-sized) kv14 (ctx-sized)
        // chunk3: layers 15-24 (all shared), reads kv13/kv14
        // chunk4: layers 25-34 (all shared) + norm + lm_head
        guard let chunk1, let chunk2, let chunk3, let chunk4,
              let embedTokens,
              let ks1 = kSliding1, let vs1 = vSliding1,
              let kf1 = kFull1, let vf1 = vFull1,
              let ks2 = kSliding2, let vs2 = vSliding2,
              let kf2 = kFull2, let vf2 = vFull2 else {
            var missing = [String]()
            if self.chunk1 == nil { missing.append("chunk1") }
            if self.chunk2 == nil { missing.append("chunk2") }
            if self.chunk3 == nil { missing.append("chunk3") }
            if self.chunk4 == nil { missing.append("chunk4") }
            if self.embedTokens == nil { missing.append("embed_tokens") }
            if self.kSliding1 == nil { missing.append("KV1") }
            if self.kSliding2 == nil { missing.append("KV2") }
            throw NSError(domain: "LLMRunner", code: 0,
                          userInfo: [NSLocalizedDescriptionKey: "Missing: \(missing.joined(separator: ", "))"])
        }
        let ctx = contextLength
        let W = slidingWindow

        let t0 = CFAbsoluteTimeGetCurrent()
        let hiddenIn: MLMultiArray
        let plRaw: MLMultiArray
        if let imageEmbedding {
            // Image token: use vision features, ZERO per-layer raw.
            hiddenIn = imageEmbedding
            let totalDim = 35 * perLayerDim
            plRaw = try MLMultiArray(shape: [1, 1, NSNumber(value: totalDim)], dataType: .float16)
            memset(plRaw.dataPointer, 0, totalDim * MemoryLayout<UInt16>.stride)
        } else {
            hiddenIn = try embedTokens.lookup(tokenID, shape: [1, 1, NSNumber(value: hiddenSize)])
            plRaw = try lookupPerLayerRaw(tokenID: tokenID)
        }
        let t1 = CFAbsoluteTimeGetCurrent()
        profileEmbed += (t1 - t0)

        let maskFull = try makeCausalMask(position: position, contextLength: ctx)
        let maskSliding = try makeSlidingCausalMask(position: position, W: W)
        let umask = try makeUpdateMask(position: position, contextLength: ctx)

        let cosS = try lookupRoPE(table: cosSlidingTable, position: position, dim: 256)
        let sinS = try lookupRoPE(table: sinSlidingTable, position: position, dim: 256)
        let cosF = try lookupRoPE(table: cosFullTable, position: position, dim: 512)
        let sinF = try lookupRoPE(table: sinFullTable, position: position, dim: 512)

        let tp = CFAbsoluteTimeGetCurrent()

        // ---- Chunk 1: 7 sliding + 1 full KV. Computes PLE inside (ANE matmul). ----
        let input1 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenIn),
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "update_mask": MLFeatureValue(multiArray: umask),
            "per_layer_raw": MLFeatureValue(multiArray: plRaw),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "K_sliding_in": MLFeatureValue(multiArray: ks1),
            "V_sliding_in": MLFeatureValue(multiArray: vs1),
            "K_full_in": MLFeatureValue(multiArray: kf1),
            "V_full_in": MLFeatureValue(multiArray: vf1),
        ])
        let out1 = try chunk1.prediction(from: input1)
        let h1 = out1.featureValue(for: "hidden_states_out")!.multiArrayValue!
        let plc = out1.featureValue(for: "per_layer_combined_out")!.multiArrayValue!  // chunk1 computes PLC
        let newKs1 = out1.featureValue(for: "K_sliding_out")!.multiArrayValue!
        let newVs1 = out1.featureValue(for: "V_sliding_out")!.multiArrayValue!
        let newKf1 = out1.featureValue(for: "K_full_out")!.multiArrayValue!
        let newVf1 = out1.featureValue(for: "V_full_out")!.multiArrayValue!
        memcpy(ks1.dataPointer, newKs1.dataPointer, ks1.count * MemoryLayout<UInt16>.stride)
        memcpy(vs1.dataPointer, newVs1.dataPointer, vs1.count * MemoryLayout<UInt16>.stride)
        memcpy(kf1.dataPointer, newKf1.dataPointer, kf1.count * MemoryLayout<UInt16>.stride)
        memcpy(vf1.dataPointer, newVf1.dataPointer, vf1.count * MemoryLayout<UInt16>.stride)

        // ---- Chunk 2: 5 sliding + 2 full KV, emits kv13/14 ----
        let input2 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h1),
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "update_mask": MLFeatureValue(multiArray: umask),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "K_sliding_in": MLFeatureValue(multiArray: ks2),
            "V_sliding_in": MLFeatureValue(multiArray: vs2),
            "K_full_in": MLFeatureValue(multiArray: kf2),
            "V_full_in": MLFeatureValue(multiArray: vf2),
        ])
        let out2 = try chunk2.prediction(from: input2)
        let h2 = out2.featureValue(for: "hidden_states_out")!.multiArrayValue!
        let newKs2 = out2.featureValue(for: "K_sliding_out")!.multiArrayValue!
        let newVs2 = out2.featureValue(for: "V_sliding_out")!.multiArrayValue!
        let newKf2 = out2.featureValue(for: "K_full_out")!.multiArrayValue!
        let newVf2 = out2.featureValue(for: "V_full_out")!.multiArrayValue!
        memcpy(ks2.dataPointer, newKs2.dataPointer, ks2.count * MemoryLayout<UInt16>.stride)
        memcpy(vs2.dataPointer, newVs2.dataPointer, vs2.count * MemoryLayout<UInt16>.stride)
        memcpy(kf2.dataPointer, newKf2.dataPointer, kf2.count * MemoryLayout<UInt16>.stride)
        memcpy(vf2.dataPointer, newVf2.dataPointer, vf2.count * MemoryLayout<UInt16>.stride)
        let kv13_k = out2.featureValue(for: "kv13_k")!.multiArrayValue!
        let kv13_v = out2.featureValue(for: "kv13_v")!.multiArrayValue!
        let kv14_k = out2.featureValue(for: "kv14_k")!.multiArrayValue!
        let kv14_v = out2.featureValue(for: "kv14_v")!.multiArrayValue!

        // ---- Chunk 3: shared KV, reads kv13 (W-sized) and kv14 (ctx) ----
        let input3 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h2),
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "update_mask": MLFeatureValue(multiArray: umask),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "kv13_k": MLFeatureValue(multiArray: kv13_k),
            "kv13_v": MLFeatureValue(multiArray: kv13_v),
            "kv14_k": MLFeatureValue(multiArray: kv14_k),
            "kv14_v": MLFeatureValue(multiArray: kv14_v),
        ])
        let out3 = try chunk3.prediction(from: input3)
        let h3 = out3.featureValue(for: "hidden_states_out")!.multiArrayValue!

        // ---- Chunk 4: shared KV + norm + lm_head ----
        let input4 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h3),
            "causal_mask_full": MLFeatureValue(multiArray: maskFull),
            "causal_mask_sliding": MLFeatureValue(multiArray: maskSliding),
            "update_mask": MLFeatureValue(multiArray: umask),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "kv13_k": MLFeatureValue(multiArray: kv13_k),
            "kv13_v": MLFeatureValue(multiArray: kv13_v),
            "kv14_k": MLFeatureValue(multiArray: kv14_k),
            "kv14_v": MLFeatureValue(multiArray: kv14_v),
        ])
        let out4 = try chunk4.prediction(from: input4)

        profilePredict += (CFAbsoluteTimeGetCurrent() - tp)
        profileCount += 1
        if profileCount % 10 == 0 {
            let n = Double(profileCount)
            print(String(format: "[Profile] emb=%.1fms ple=%.1fms predict=%.1fms (%.1f tok/s)",
                         profileEmbed/n * 1000, profilePLE/n * 1000, profilePredict/n * 1000,
                         1000.0 / ((profileEmbed + profilePLE + profilePredict) / n * 1000)))
        }
        return out4.featureValue(for: "token_id")!.multiArrayValue![0].intValue
    }

    // MARK: - Batched Prefill (seq=N)

    /// Run a single prefill pass over up to N=prefillN tokens, writing K/V for
    /// positions [0, realLen) into the persistent SWA decode caches. After this
    /// call, `currentPosition` should be set to `realLen` by the caller and decode
    /// continues from the returned token id.
    ///
    /// If `imageFeatures` is non-nil, tokens matching IMAGE_TOKEN_ID are replaced
    /// with the corresponding vision encoder feature vector at the appropriate
    /// position in the hidden_states batch.
    private func runPrefill(tokenIDs: [Int], imageFeatures: MLMultiArray? = nil) throws -> Int {
        guard let p1 = prefillChunk1, let p2 = prefillChunk2,
              let p3 = prefillChunk3, let p4 = prefillChunk4,
              let embedTokens,
              let ks1 = kSliding1, let vs1 = vSliding1,
              let kf1 = kFull1, let vf1 = vFull1,
              let ks2 = kSliding2, let vs2 = vSliding2,
              let kf2 = kFull2, let vf2 = vFull2 else {
            throw NSError(domain: "LLMRunner", code: 10,
                          userInfo: [NSLocalizedDescriptionKey: "Prefill chunks or caches not loaded"])
        }
        let N = prefillN
        let realLen = tokenIDs.count
        precondition(realLen > 0 && realLen <= N, "prefill needs 1..N tokens")

        // Reset KV caches — prefill assumes starting position 0.
        for buf in [kSliding1, vSliding1, kFull1, vFull1, kSliding2, vSliding2, kFull2, vFull2] {
            if let b = buf {
                memset(b.dataPointer, 0, b.count * MemoryLayout<UInt16>.stride)
            }
        }

        let tStart = CFAbsoluteTimeGetCurrent()

        // Build inputs common to all chunks.
        let hiddenIn = try buildPrefillHiddenStates(tokenIDs: tokenIDs, N: N,
                                                     embedTokens: embedTokens,
                                                     imageFeatures: imageFeatures)
        let plRaw = try buildPrefillPerLayerRaw(tokenIDs: tokenIDs, N: N)
        let causal = try makePrefillCausalMask(N: N)
        let cosS = try buildPrefillRoPE(table: cosSlidingTable, N: N, dim: 256)
        let sinS = try buildPrefillRoPE(table: sinSlidingTable, N: N, dim: 256)
        let cosF = try buildPrefillRoPE(table: cosFullTable, N: N, dim: 512)
        let sinF = try buildPrefillRoPE(table: sinFullTable, N: N, dim: 512)
        let lastMask = try makeLastPositionMask(N: N, realLen: realLen)

        // ---- Prefill Chunk 1 (L0-7): computes PLE inside. ----
        let input1 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenIn),
            "causal_mask": MLFeatureValue(multiArray: causal),
            "per_layer_raw": MLFeatureValue(multiArray: plRaw),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
        ])
        let out1 = try p1.prediction(from: input1)
        let h1 = out1.featureValue(for: "hidden_states_out")!.multiArrayValue!
        let plc = out1.featureValue(for: "per_layer_combined_out")!.multiArrayValue!

        // chunk1 slot → kSliding1 mapping: [L0..3, L4=full, L5..7]
        // Sliding-slot order (sliding_map): L0→0, L1→1, L2→2, L3→3, L5→4, L6→5, L7→6
        // Full-slot order: L4→0
        try writeSlidingFromPrefill(src: out1, name: "K0", slotKV: ks1, slot: 0, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "V0", slotKV: vs1, slot: 0, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "K1", slotKV: ks1, slot: 1, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "V1", slotKV: vs1, slot: 1, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "K2", slotKV: ks1, slot: 2, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "V2", slotKV: vs1, slot: 2, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "K3", slotKV: ks1, slot: 3, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "V3", slotKV: vs1, slot: 3, realLen: realLen, hd: 256)
        try writeFullFromPrefill(src: out1, name: "K4", slotKV: kf1, slot: 0, realLen: realLen, hd: 512)
        try writeFullFromPrefill(src: out1, name: "V4", slotKV: vf1, slot: 0, realLen: realLen, hd: 512)
        try writeSlidingFromPrefill(src: out1, name: "K5", slotKV: ks1, slot: 4, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "V5", slotKV: vs1, slot: 4, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "K6", slotKV: ks1, slot: 5, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "V6", slotKV: vs1, slot: 5, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "K7", slotKV: ks1, slot: 6, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out1, name: "V7", slotKV: vs1, slot: 6, realLen: realLen, hd: 256)

        // ---- Prefill Chunk 2 (L8-14): outputs K0..K4 + kv13/kv14. ----
        let input2 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h1),
            "causal_mask": MLFeatureValue(multiArray: causal),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
        ])
        let out2 = try p2.prediction(from: input2)
        let h2 = out2.featureValue(for: "hidden_states_out")!.multiArrayValue!

        // chunk2 slot → kSliding2 mapping: [L8, L9=full, L10, L11, L12, L13=sliding, L14=full]
        // Sliding slots: L8→0, L10→1, L11→2, L12→3, L13→4
        // Full slots: L9→0, L14→1
        try writeSlidingFromPrefill(src: out2, name: "K0", slotKV: ks2, slot: 0, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out2, name: "V0", slotKV: vs2, slot: 0, realLen: realLen, hd: 256)
        try writeFullFromPrefill(src: out2, name: "K1", slotKV: kf2, slot: 0, realLen: realLen, hd: 512)
        try writeFullFromPrefill(src: out2, name: "V1", slotKV: vf2, slot: 0, realLen: realLen, hd: 512)
        try writeSlidingFromPrefill(src: out2, name: "K2", slotKV: ks2, slot: 1, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out2, name: "V2", slotKV: vs2, slot: 1, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out2, name: "K3", slotKV: ks2, slot: 2, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out2, name: "V3", slotKV: vs2, slot: 2, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out2, name: "K4", slotKV: ks2, slot: 3, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out2, name: "V4", slotKV: vs2, slot: 3, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out2, name: "kv13_k", slotKV: ks2, slot: 4, realLen: realLen, hd: 256)
        try writeSlidingFromPrefill(src: out2, name: "kv13_v", slotKV: vs2, slot: 4, realLen: realLen, hd: 256)
        try writeFullFromPrefill(src: out2, name: "kv14_k", slotKV: kf2, slot: 1, realLen: realLen, hd: 512)
        try writeFullFromPrefill(src: out2, name: "kv14_v", slotKV: vf2, slot: 1, realLen: realLen, hd: 512)

        // kv13/14 are needed as-is by chunks 3/4.
        let kv13_k = out2.featureValue(for: "kv13_k")!.multiArrayValue!
        let kv13_v = out2.featureValue(for: "kv13_v")!.multiArrayValue!
        let kv14_k = out2.featureValue(for: "kv14_k")!.multiArrayValue!
        let kv14_v = out2.featureValue(for: "kv14_v")!.multiArrayValue!

        // ---- Prefill Chunk 3 (L15-24, KV-shared). ----
        let input3 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h2),
            "causal_mask": MLFeatureValue(multiArray: causal),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "kv13_k": MLFeatureValue(multiArray: kv13_k),
            "kv13_v": MLFeatureValue(multiArray: kv13_v),
            "kv14_k": MLFeatureValue(multiArray: kv14_k),
            "kv14_v": MLFeatureValue(multiArray: kv14_v),
        ])
        let out3 = try p3.prediction(from: input3)
        let h3 = out3.featureValue(for: "hidden_states_out")!.multiArrayValue!

        // ---- Prefill Chunk 4 (L25-34 + norm + lm_head). ----
        let input4 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h3),
            "causal_mask": MLFeatureValue(multiArray: causal),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "kv13_k": MLFeatureValue(multiArray: kv13_k),
            "kv13_v": MLFeatureValue(multiArray: kv13_v),
            "kv14_k": MLFeatureValue(multiArray: kv14_k),
            "kv14_v": MLFeatureValue(multiArray: kv14_v),
            "last_position_mask": MLFeatureValue(multiArray: lastMask),
        ])
        let out4 = try p4.prediction(from: input4)
        let tokenID = out4.featureValue(for: "token_id")!.multiArrayValue![0].intValue

        let elapsedMs = (CFAbsoluteTimeGetCurrent() - tStart) * 1000
        print(String(format: "[Prefill] %d tokens in %.1fms (%.0f tok/s effective)",
                     realLen, elapsedMs, Double(realLen) / (elapsedMs / 1000)))
        return tokenID
    }

    // MARK: - Prefill helpers

    /// Build (1, N, hidden) batched input: real token embeddings for [0, realLen),
    /// zeros for the padding tail. When imageFeatures is supplied, IMAGE_TOKEN_ID
    /// occurrences are replaced in place with the vision encoder features.
    private func buildPrefillHiddenStates(tokenIDs: [Int], N: Int,
                                           embedTokens: EmbeddingLookup,
                                           imageFeatures: MLMultiArray? = nil) throws -> MLMultiArray {
        let IMAGE_TOKEN_ID = 258880
        let arr = try MLMultiArray(shape: [1, NSNumber(value: N), NSNumber(value: hiddenSize)], dataType: .float16)
        memset(arr.dataPointer, 0, N * hiddenSize * MemoryLayout<UInt16>.stride)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: N * hiddenSize)
        let featPtr = imageFeatures?.dataPointer.bindMemory(
            to: UInt16.self, capacity: imageFeatures!.count)
        var imageIdx = 0
        for (i, tid) in tokenIDs.enumerated() {
            if tid == IMAGE_TOKEN_ID, let fp = featPtr, imageIdx < 256 {
                // Vision encoder output is (1, 280, hidden). Copy slice `imageIdx`.
                memcpy(dst.advanced(by: i * hiddenSize),
                       fp.advanced(by: imageIdx * hiddenSize),
                       hiddenSize * MemoryLayout<UInt16>.stride)
                imageIdx += 1
            } else {
                let emb = try embedTokens.lookup(tid, shape: [1, 1, NSNumber(value: hiddenSize)])
                let src = emb.dataPointer.bindMemory(to: UInt16.self, capacity: hiddenSize)
                memcpy(dst.advanced(by: i * hiddenSize), src,
                       hiddenSize * MemoryLayout<UInt16>.stride)
            }
        }
        return arr
    }

    /// Build (1, N, 35*per_layer_dim) per-token raw per-layer embedding.
    /// Image positions get zero PLE — their per_layer contribution comes from
    /// the projection of vision features inside chunk1, not from token lookup.
    private func buildPrefillPerLayerRaw(tokenIDs: [Int], N: Int) throws -> MLMultiArray {
        let IMAGE_TOKEN_ID = 258880
        guard let embedPerLayer else { throw NSError(domain: "LLMRunner", code: 11) }
        let totalDim = 35 * perLayerDim
        let arr = try MLMultiArray(shape: [1, NSNumber(value: N), NSNumber(value: totalDim)], dataType: .float16)
        memset(arr.dataPointer, 0, N * totalDim * MemoryLayout<UInt16>.stride)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: N * totalDim)
        for (i, tid) in tokenIDs.enumerated() {
            if tid == IMAGE_TOKEN_ID { continue }  // leave as zero
            let raw = embedPerLayer.lookupRaw(tid)
            for j in 0..<totalDim {
                dst[i * totalDim + j] = raw[j]
            }
        }
        return arr
    }

    /// (1, 1, N, N) lower-triangular causal mask (0 on/below diag, -inf above).
    private func makePrefillCausalMask(N: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(shape: [1, 1, NSNumber(value: N), NSNumber(value: N)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: N * N)
        for i in 0..<N {
            for j in 0..<N {
                mp[i * N + j] = j <= i ? 0 : 0xFC00
            }
        }
        return mask
    }

    /// (1, 1, N, dim) RoPE table slice for positions [0, N).
    private func buildPrefillRoPE(table: Data?, N: Int, dim: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, 1, NSNumber(value: N), NSNumber(value: dim)], dataType: .float16)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: N * dim)
        guard let table else {
            memset(dst, 0, N * dim * MemoryLayout<UInt16>.stride)
            return arr
        }
        // numpy .npy header
        var headerSize = 128
        table.withUnsafeBytes { raw in
            let bytes = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            let hlen = Int(bytes[8]) | (Int(bytes[9]) << 8)
            headerSize = 10 + hlen
        }
        let rowBytes = dim * MemoryLayout<UInt16>.stride
        table.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            for p in 0..<N {
                let off = headerSize + p * rowBytes
                if off + rowBytes <= table.count {
                    memcpy(dst.advanced(by: p * dim), base.advanced(by: off), rowBytes)
                } else {
                    memset(dst.advanced(by: p * dim), 0, rowBytes)
                }
            }
        }
        return arr
    }

    /// (1, N, 1) mask with 1.0 at real last position (realLen-1), 0.0 elsewhere.
    private func makeLastPositionMask(N: Int, realLen: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1, NSNumber(value: N), 1], dataType: .float16)
        let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: N)
        memset(p, 0, N * MemoryLayout<UInt16>.stride)
        p[realLen - 1] = 0x3C00  // fp16 1.0
        return arr
    }

    /// Write prefill output K or V (shape 1,1,N,hd) into a sliding cache buffer
    /// (shape num_slots,1,W,max_hd) at positions [W-realLen, W) within `slot`.
    private func writeSlidingFromPrefill(src out: MLFeatureProvider, name: String,
                                          slotKV: MLMultiArray, slot: Int,
                                          realLen: Int, hd: Int) throws {
        guard let srcArr = out.featureValue(for: name)?.multiArrayValue else {
            throw NSError(domain: "LLMRunner", code: 12,
                          userInfo: [NSLocalizedDescriptionKey: "Missing prefill output \(name)"])
        }
        let W = slidingWindow
        let shape = slotKV.shape.map { $0.intValue }  // (slots, 1, W, max_hd)
        let maxHd = shape[3]
        precondition(shape[2] == W, "slot buffer W mismatch")
        let slotStride = 1 * W * maxHd
        let rowStride = maxHd  // per cache position
        let dst = slotKV.dataPointer.bindMemory(to: UInt16.self, capacity: slotKV.count)
        let src = srcArr.dataPointer.bindMemory(to: UInt16.self, capacity: srcArr.count)
        // src layout: (1, 1, N, hd). We copy positions [0, realLen) into slot
        // positions [W-realLen, W) to match the decode sliding cache "newest at end" convention.
        let startCachePos = W - realLen
        for p in 0..<realLen {
            let srcOff = p * hd  // (0,0,p,0) — src is (1,1,N,hd)
            let dstOff = slot * slotStride + (startCachePos + p) * rowStride
            // Copy hd values; remainder of maxHd is already zero.
            for j in 0..<hd { dst[dstOff + j] = src[srcOff + j] }
        }
    }

    /// Write prefill output K or V (shape 1,1,N,hd) into a full cache buffer
    /// (shape num_slots,1,ctx,max_hd) at positions [0, realLen) within `slot`.
    private func writeFullFromPrefill(src out: MLFeatureProvider, name: String,
                                       slotKV: MLMultiArray, slot: Int,
                                       realLen: Int, hd: Int) throws {
        guard let srcArr = out.featureValue(for: name)?.multiArrayValue else {
            throw NSError(domain: "LLMRunner", code: 13,
                          userInfo: [NSLocalizedDescriptionKey: "Missing prefill output \(name)"])
        }
        let shape = slotKV.shape.map { $0.intValue }  // (slots, 1, ctx, max_hd)
        let ctx = shape[2]
        let maxHd = shape[3]
        let slotStride = 1 * ctx * maxHd
        let rowStride = maxHd
        let dst = slotKV.dataPointer.bindMemory(to: UInt16.self, capacity: slotKV.count)
        let src = srcArr.dataPointer.bindMemory(to: UInt16.self, capacity: srcArr.count)
        for p in 0..<realLen {
            let srcOff = p * hd
            let dstOff = slot * slotStride + p * rowStride
            for j in 0..<hd { dst[dstOff + j] = src[srcOff + j] }
        }
    }

    /// Look up cos/sin values for a position from a numpy .npy file.
    private func lookupRoPE(table: Data?, position: Int, dim: Int) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: dim)], dataType: .float16)
        let dstPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: dim)

        guard let table else {
            memset(dstPtr, 0, dim * MemoryLayout<UInt16>.stride)
            return result
        }

        // numpy .npy format: header then raw data
        var headerSize = 128  // typical
        table.withUnsafeBytes { raw in
            let bytes = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            let hlen = Int(bytes[8]) | (Int(bytes[9]) << 8)
            headerSize = 10 + hlen
        }

        let rowBytes = dim * MemoryLayout<UInt16>.stride
        let offset = headerSize + position * rowBytes

        guard offset + rowBytes <= table.count else {
            memset(dstPtr, 0, rowBytes)
            return result
        }

        table.withUnsafeBytes { raw in
            let srcPtr = raw.baseAddress!.advanced(by: offset)
            memcpy(dstPtr, srcPtr, rowBytes)
        }

        return result
    }

    // MARK: - Per-Layer Computation

    private func computePerLayerCombined(tokenID: Int, embedding: MLMultiArray) throws -> MLMultiArray {
        guard let embedPerLayer, let perLayerProjF32 else {
            var missing = [String]()
            if self.embedPerLayer == nil { missing.append("embed_per_layer") }
            if self.perLayerProjF32 == nil { missing.append("per_layer_projection") }
            throw NSError(domain: "LLMRunner", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Missing files: \(missing.joined(separator: ", "))"])
        }
        let nlayers = 35, pld = perLayerDim
        let totalDim = nlayers * pld  // 8960
        let result = try MLMultiArray(shape: [1, 1, NSNumber(value: totalDim)], dataType: .float16)
        let resultPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: totalDim)

        // Raw per-layer embedding
        let raw = embedPerLayer.lookupRaw(tokenID)

        // Convert embedding to float32
        let embPtr = embedding.dataPointer.bindMemory(to: UInt16.self, capacity: hiddenSize)
        var embF32 = [Float](repeating: 0, count: hiddenSize)
        var embF16 = [Float16](repeating: 0, count: hiddenSize)
        for i in 0..<hiddenSize { embF16[i] = Float16(bitPattern: embPtr[i]) }
        vDSP.convertElements(of: embF16, to: &embF32)

        // Step 1: Matrix-vector multiply using Accelerate BLAS
        // proj = projWeight (8960×1536) × embedding (1536) × projScale
        var proj = [Float](repeating: 0, count: totalDim)
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    Int32(totalDim), Int32(hiddenSize),
                    perLayerProjScale,         // alpha = projScale
                    perLayerProjF32, Int32(hiddenSize),  // A, lda
                    embF32, 1,                 // x, incx
                    0.0,                       // beta
                    &proj, 1)                  // y, incy

        // Step 2: Apply RMSNorm to each per_layer_dim slice
        if let normData = perLayerNormWeight {
            normData.withUnsafeBytes { normRaw in
                let normW = normRaw.baseAddress!.assumingMemoryBound(to: Float.self)
                let eps: Float = 1e-6
                for li in 0..<nlayers {
                    let s = li * pld
                    var sumSq: Float = 0
                    vDSP_svesq(&proj + s, 1, &sumSq, vDSP_Length(pld))
                    let invRms = 1.0 / sqrtf(sumSq / Float(pld) + eps)
                    for j in 0..<pld {
                        proj[s + j] = proj[s + j] * invRms * normW[j]
                    }
                }
            }
        }

        // Step 3: Combine (normed_proj + raw) * inputScale
        for i in 0..<totalDim {
            let rawVal = float16ToFloat(raw[i])
            let combined = (proj[i] + rawVal) * perLayerInputScale
            resultPtr[i] = floatToFloat16(combined)
        }

        return result
    }

    private func float16ToFloat(_ bits: UInt16) -> Float {
        let sign: UInt32 = UInt32(bits >> 15) << 31
        let exp = UInt32((bits >> 10) & 0x1F)
        let frac = UInt32(bits & 0x3FF)
        if exp == 0 { return exp == 0 && frac == 0 ? Float(bitPattern: sign) : 0 }
        if exp == 31 { return Float.infinity }
        return Float(bitPattern: sign | ((exp + 112) << 23) | (frac << 13))
    }

    private func floatToFloat16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exp = Int((bits >> 23) & 0xFF) - 127 + 15
        let frac = UInt16((bits >> 13) & 0x3FF)
        if exp <= 0 { return sign }
        if exp >= 31 { return sign | 0x7C00 }
        return sign | UInt16(exp) << 10 | frac
    }

    // MARK: - Helpers

    private func makeCausalMask(position: Int, contextLength: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: contextLength)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: contextLength)
        for i in 0..<contextLength { mp[i] = i <= position ? 0 : 0xFC00 }
        return mask
    }

    /// Lookup per-layer raw embedding (already scaled by per_layer_embed_scale).
    /// Shape: (1, 1, num_layers * per_layer_dim).
    private func lookupPerLayerRaw(tokenID: Int) throws -> MLMultiArray {
        guard let embedPerLayer else { throw NSError(domain: "LLMRunner", code: 5) }
        let totalDim = 35 * perLayerDim
        let result = try MLMultiArray(shape: [1, 1, NSNumber(value: totalDim)], dataType: .float16)
        let raw = embedPerLayer.lookupRaw(tokenID)
        let dst = result.dataPointer.bindMemory(to: UInt16.self, capacity: totalDim)
        for i in 0..<totalDim { dst[i] = raw[i] }
        return result
    }

    /// Sliding window causal mask: shape (1, 1, 1, W).
    /// Last min(position+1, W) slots are valid (cache end holds newest).
    private func makeSlidingCausalMask(position: Int, W: Int) throws -> MLMultiArray {
        let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: W)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: W)
        let valid = min(position + 1, W)
        let start = W - valid
        for i in 0..<W { mp[i] = i >= start ? 0 : 0xFC00 }
        return mask
    }

    private func makeUpdateMask(position: Int, contextLength: Int) throws -> MLMultiArray {
        let umask = try MLMultiArray(shape: [1, 1, NSNumber(value: contextLength), 1], dataType: .float16)
        let up = umask.dataPointer.bindMemory(to: UInt16.self, capacity: contextLength)
        memset(up, 0, contextLength * MemoryLayout<UInt16>.stride)
        // Clamp to last valid position if overflow (generation will stop shortly)
        let clamped = min(position, contextLength - 1)
        up[clamped] = 0x3C00
        return umask
    }

    /// Preprocess image to match HuggingFace Gemma3nImageProcessor.
    ///
    /// Algorithm: aspect-ratio-preserving resize such that H×W ≤ 645,120 pixels
    /// (= max_patches(2520) × patch_size²(256)), with each side rounded down to
    /// a multiple of 48 (= pooling_kernel(3) × patch_size(16)). A square input
    /// becomes 768×768 → 48×48=2304 patches → 256 soft tokens (padded to 280
    /// inside the vision encoder).
    ///
    /// The vision model always outputs (1, 280, 1536) regardless of input grid,
    /// so the text prompt always inserts 280 `<|image|>` placeholders.
    ///
    /// Pixel layout in each 768-d row: patch_h × patch_w × channels (row-major),
    /// channels-last. Normalization: /255, no mean/std. Fp32 dtype.
    private func processImage(_ image: CGImage, with visionModel: MLModel) throws -> MLMultiArray {
        let ps = 16
        let total = 2520    // max patches the vision encoder accepts
        let pd = ps * ps * 3  // 768, flattened patch row

        // 1. Compute aspect-ratio-preserving target size (each side multiple of 48).
        let origH = Double(image.height)
        let origW = Double(image.width)
        let targetPx = Double(total * ps * ps)  // 645_120
        let factor = sqrt(targetPx / (origH * origW))
        let sideMult = 48  // pooling_kernel * patch_size
        var tH = Int(floor(factor * origH / Double(sideMult))) * sideMult
        var tW = Int(floor(factor * origW / Double(sideMult))) * sideMult
        if tH < sideMult { tH = sideMult }
        if tW < sideMult { tW = sideMult }
        let Hp = tH / ps  // patches in height
        let Wp = tW / ps  // patches in width
        let realPatches = Hp * Wp  // ≤ 2520

        // 2. Draw into an (tW, tH) RGBA canvas — Core Graphics handles bicubic resize.
        var pixels = [UInt8](repeating: 0, count: tW * tH * 4)
        let bitmap = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        let ctx = CGContext(data: &pixels, width: tW, height: tH, bitsPerComponent: 8,
                            bytesPerRow: tW * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: bitmap.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: tW, height: tH))

        // 3. Emit pixel_values (B, 2520, 768) fp32 and pixel_position_ids (B, 2520, 2) int32.
        let pv = try MLMultiArray(shape: [1, NSNumber(value: total), NSNumber(value: pd)], dataType: .float32)
        let pid = try MLMultiArray(shape: [1, NSNumber(value: total), 2], dataType: .int32)
        let pvp = pv.dataPointer.bindMemory(to: Float.self, capacity: total * pd)
        let pidp = pid.dataPointer.bindMemory(to: Int32.self, capacity: total * 2)
        memset(pvp, 0, total * pd * MemoryLayout<Float>.stride)

        var pi = 0
        for py in 0..<Hp {
            for px in 0..<Wp {
                var o = pi * pd
                for dy in 0..<ps {
                    for dx in 0..<ps {
                        let srcIdx = ((py * ps + dy) * tW + (px * ps + dx)) * 4
                        pvp[o]   = Float(pixels[srcIdx])   / 255
                        pvp[o+1] = Float(pixels[srcIdx+1]) / 255
                        pvp[o+2] = Float(pixels[srcIdx+2]) / 255
                        o += 3
                    }
                }
                // Meshgrid order (x, y) = (px, py) — matches HF's indexing="xy".
                pidp[pi * 2]     = Int32(px)
                pidp[pi * 2 + 1] = Int32(py)
                pi += 1
            }
        }
        // Zero-pad pixel_values (done by memset above), mark position_ids as -1 for padding.
        for i in realPatches..<total {
            pidp[i * 2]     = -1
            pidp[i * 2 + 1] = -1
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pv),
            "pixel_position_ids": MLFeatureValue(multiArray: pid),
        ])
        return try visionModel.prediction(from: input).featureValue(for: "image_features")!.multiArrayValue!
    }

    private func sliceFeature(_ features: MLMultiArray, at index: Int) -> MLMultiArray {
        let hs = hiddenSize
        let r = try! MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16)
        let s = features.dataPointer.bindMemory(to: UInt16.self, capacity: features.count)
        let d = r.dataPointer.bindMemory(to: UInt16.self, capacity: hs)
        memcpy(d, s.advanced(by: index * hs), hs * MemoryLayout<UInt16>.stride)
        return r
    }

    private func buildPrompt(messages: [ChatMessage], hasImage: Bool) -> String {
        if architecture.hasPrefix("qwen") {
            var p = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            for m in messages {
                if m.role == .user { p += "<|im_start|>user\n\(m.content)<|im_end|>\n" }
                else if m.role == .assistant { p += "<|im_start|>assistant\n\(m.content)<|im_end|>\n" }
            }
            return p + "<|im_start|>assistant\n"
        }
        var p = "<bos>"
        // HuggingFace Gemma3nProcessor wraps image tokens with BOI/EOI markers.
        // For a square 768x768 input: 48x48=2304 patches → 2304/9 = 256 actual
        // soft tokens. The vision encoder OUTPUT tensor is always (1, 280, hidden)
        // (max_soft_tokens=280) but the prompt should use only the REAL count
        // because tokens 256..279 are zero padding inside the encoder output.
        // Inserting 280 placeholders feeds the model 24 zero "image" hidden
        // states which it tries to interpret → garbled output.
        // For a single square image: <|image> + <|image|> * 256 + <image|>
        let imageBlock = "<|image>" + String(repeating: "<|image|>", count: 256) + "<image|>"
        for m in messages {
            if m.role == .user {
                if hasImage {
                    p += "<|turn>user\n\(imageBlock)\n\(m.content)<turn|>\n"
                } else { p += "<|turn>user\n\(m.content)<turn|>\n" }
            } else if m.role == .assistant { p += "<|turn>model\n\(m.content)<turn|>\n" }
        }
        return p + "<|turn>model\n"
    }

    // MARK: - Battery / sustained-throughput benchmark
    //
    // Runs continuous generation rounds for a fixed wall-clock duration,
    // recording:
    //   - start / end battery SoC (UIDevice.batteryLevel)
    //   - start / end thermal state (ProcessInfo.thermalState)
    //   - total tokens produced, average tok/s
    //
    // Each round uses a fixed long prompt and resets KV cache, so results
    // are directly comparable across runs. Caller should put the device
    // in airplane mode, unplugged, screen on, before starting.

    struct BenchmarkProgress {
        var elapsed: TimeInterval
        var totalTokens: Int
        var round: Int
        var avgTokPerSec: Double
        var batteryStart: Float     // 0..1, -1 if unknown
        var batteryNow: Float
        var thermal: ProcessInfo.ThermalState
    }

    struct BenchmarkResult {
        var duration: TimeInterval
        var totalTokens: Int
        var rounds: Int
        var avgTokPerSec: Double
        var batteryStart: Float     // 0..1
        var batteryEnd: Float
        var thermalStart: ProcessInfo.ThermalState
        var thermalEnd: ProcessInfo.ThermalState
        var abortedThermal: Bool = false

        var batteryDelta: Float { batteryStart - batteryEnd }  // positive = drained
        var drainedPercent: Double { Double(batteryDelta) * 100.0 }
        var drainedPerMinute: Double { duration > 0 ? drainedPercent / (duration / 60.0) : 0 }
        var tokensPerPercent: Double { drainedPercent > 0 ? Double(totalTokens) / drainedPercent : 0 }
    }

    /// Fixed prompt that tends to produce long, stable outputs.
    private static let benchmarkPrompt =
        "Write a very long, detailed article about the history of artificial intelligence from the 1950s through today. Cover: early symbolic AI and Alan Turing, the first and second AI winters, the rise of neural networks, deep learning breakthroughs like AlexNet and ResNet, the attention mechanism and transformers, the scaling era with GPT-2/3/4, reinforcement learning milestones, and the current era of multimodal foundation models running on-device. Be verbose and thorough."

    @MainActor
    func runBenchmark(
        duration: TimeInterval,
        onProgress: @escaping (BenchmarkProgress) -> Void
    ) async throws -> BenchmarkResult {
        // Enable battery monitoring (no-op if already enabled)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let startBat = UIDevice.current.batteryLevel
        let startThermal = ProcessInfo.processInfo.thermalState
        let startTime = Date()

        var totalTokens = 0
        var round = 0
        var abortedThermal = false
        let prompt = ChatMessage(role: .user, content: Self.benchmarkPrompt)

        // Safety: if the device reaches .serious thermal state (iOS has
        // already started aggressive throttling and the case is probably
        // >40 °C), we stop immediately. .critical means stop no matter what.
        func isThermalUnsafe() -> Bool {
            let s = ProcessInfo.processInfo.thermalState
            return s == .serious || s == .critical
        }

        while Date().timeIntervalSince(startTime) < duration {
            if isThermalUnsafe() { abortedThermal = true; break }
            round += 1
            // AsyncStream of decoded chunks — count by fragment, close to
            // token count (decode() can occasionally emit multi-token chunks
            // for merged BPE pieces, but for greedy decoding it's ~1:1).
            let stream = try await generate(messages: [prompt], image: nil)
            for await _ in stream {
                totalTokens += 1
                let elapsed = Date().timeIntervalSince(startTime)
                if totalTokens % 20 == 0 {
                    let prog = BenchmarkProgress(
                        elapsed: elapsed,
                        totalTokens: totalTokens,
                        round: round,
                        avgTokPerSec: elapsed > 0 ? Double(totalTokens) / elapsed : 0,
                        batteryStart: startBat,
                        batteryNow: UIDevice.current.batteryLevel,
                        thermal: ProcessInfo.processInfo.thermalState
                    )
                    onProgress(prog)
                }
                if elapsed >= duration { break }
                if isThermalUnsafe() { abortedThermal = true; break }
            }
            if abortedThermal { break }
            if Date().timeIntervalSince(startTime) >= duration { break }
        }

        let endTime = Date()
        let endBat = UIDevice.current.batteryLevel
        let endThermal = ProcessInfo.processInfo.thermalState
        let dur = endTime.timeIntervalSince(startTime)
        return BenchmarkResult(
            duration: dur,
            totalTokens: totalTokens,
            rounds: round,
            avgTokPerSec: dur > 0 ? Double(totalTokens) / dur : 0,
            batteryStart: startBat,
            batteryEnd: endBat,
            thermalStart: startThermal,
            thermalEnd: endThermal,
            abortedThermal: abortedThermal
        )
    }

    static func thermalString(_ s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "?"
        }
    }

    // MARK: - ANE placement verification
    //
    // Uses MLComputePlan (iOS 17+) to ask the CoreML runtime which compute
    // device it will actually dispatch each op to — Neural Engine, GPU, or CPU.
    // This is the ground truth for "is this model running on ANE?", and is
    // much more reliable than watching Instruments energy graphs.
    //
    // We run it against every chunk that's on disk (decode + prefill + vision)
    // and aggregate the counts. Result looks like:
    //
    //   chunk1       : 412/418 ANE (98%)  6 CPU
    //   chunk2       : 389/395 ANE (98%)  6 CPU
    //   ...
    //   TOTAL        : 3210/3280 ANE (97%)  70 CPU

    @available(iOS 17.0, *)
    func verifyANEPlacement() async -> String {
        guard let folder = modelFolderURL else {
            return "No model folder (load a model first)."
        }

        // Match how we load them so the plan reflects runtime decisions.
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine

        let visionCfg = MLModelConfiguration()
        visionCfg.computeUnits = .cpuAndGPU

        struct Entry {
            let label: String
            let url: URL
            let cfg: MLModelConfiguration
        }
        var entries: [Entry] = []
        let chunkNames = [
            "chunk1", "chunk2", "chunk3", "chunk4",
            "prefill_chunk1", "prefill_chunk2", "prefill_chunk3", "prefill_chunk4",
        ]
        for name in chunkNames {
            if let u = findModel(in: folder, name: name) {
                entries.append(Entry(label: name, url: u, cfg: cfg))
            }
        }
        if let vurl = visionModelURL {
            entries.append(Entry(label: "vision", url: vurl, cfg: visionCfg))
        }

        if entries.isEmpty {
            return "No chunks found under \(folder.lastPathComponent)."
        }

        // MLComputePlan.deviceUsage(for:) returns nil for "virtual" ops that
        // don't dispatch at runtime — const, constexpr_affine_dequantize,
        // constexpr_lut_to_dense (INT4 palette expansion), metadata-only
        // reshape/transpose etc. These are resolved at compile time and
        // shouldn't appear in the denominator. We report:
        //   "X/Y ANE" where Y = dispatched ops (ane + gpu + cpu).
        // The total op count is shown separately as a sanity check.
        var lines: [String] = []
        lines.append("MLComputePlan placement (dispatched ops only;")
        lines.append("virtual/constexpr ops excluded from %):")
        var tAll = 0, aAll = 0, gAll = 0, cAll = 0
        for e in entries {
            do {
                let plan = try await MLComputePlan.load(contentsOf: e.url, configuration: e.cfg)
                let (total, ane, gpu, cpu) = countOps(plan: plan)
                tAll += total; aAll += ane; gAll += gpu; cAll += cpu
                let dispatched = ane + gpu + cpu
                let pct = dispatched > 0 ? Int((Double(ane) / Double(dispatched) * 100.0).rounded()) : 0
                let label = e.label.padding(toLength: 16, withPad: " ", startingAt: 0)
                lines.append("  \(label) \(ane)/\(dispatched) ANE (\(pct)%)  GPU=\(gpu) CPU=\(cpu)  [\(total) total ops]")
            } catch {
                lines.append("  \(e.label): failed — \(error.localizedDescription)")
            }
        }
        let dispatchedAll = aAll + gAll + cAll
        let pctAll = dispatchedAll > 0 ? Int((Double(aAll) / Double(dispatchedAll) * 100.0).rounded()) : 0
        lines.append("  ----")
        lines.append("  TOTAL            \(aAll)/\(dispatchedAll) ANE (\(pctAll)%)  GPU=\(gAll) CPU=\(cAll)  [\(tAll) total ops]")
        return lines.joined(separator: "\n")
    }

    @available(iOS 17.0, *)
    private func countOps(plan: MLComputePlan) -> (total: Int, ane: Int, gpu: Int, cpu: Int) {
        guard case let .program(program) = plan.modelStructure,
              let main = program.functions["main"] else {
            return (0, 0, 0, 0)
        }
        var total = 0, ane = 0, gpu = 0, cpu = 0
        var stack: [MLModelStructure.Program.Block] = [main.block]
        while let block = stack.popLast() {
            for op in block.operations {
                total += 1
                if let usage = plan.deviceUsage(for: op) {
                    switch usage.preferred {
                    case .neuralEngine: ane += 1
                    case .gpu:          gpu += 1
                    case .cpu:          cpu += 1
                    @unknown default:   break
                    }
                }
                for nested in op.blocks {
                    stack.append(nested)
                }
            }
        }
        return (total, ane, gpu, cpu)
    }
}
