import Accelerate
import CoreML
import Foundation
import Tokenizers

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

    // Chunked model (stateless 4-chunk lite, ANE-optimized)
    private var chunk1: MLModel?
    private var chunk2: MLModel?
    private var chunk3: MLModel?
    private var chunk4: MLModel?
    private var chunk1State: MLState?  // unused in stateless mode
    private var chunk2State: MLState?  // unused in stateless mode
    private var isChunked = false
    // Persistent KV cache buffers for stateless chunks (zero-copy across calls)
    private var statelessKV1_K: MLMultiArray?  // (8, 1, ctx, max_hd)
    private var statelessKV1_V: MLMultiArray?
    private var statelessKV2_K: MLMultiArray?  // (7, 1, ctx, max_hd)
    private var statelessKV2_V: MLMultiArray?

    // Vision (lazy-loaded to save memory)
    private var visionModel: MLModel?
    private var visionModelURL: URL?
    private var visionConfig: MLModelConfiguration?

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
        if findModel(in: folder, name: "vision") != nil {
            hasVision = true
            visionModelURL = findModel(in: folder, name: "vision")
            visionConfig = mlConfig
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
        // Stateless 4-chunk lite:
        // chunk1: layers 0-7 + embedding, KV cache I/O (8 slots)
        // chunk2: layers 8-14 (contains KV sources 13/14), KV cache I/O (7 slots) + kv13/14 output
        // chunk3: layers 15-24 (all shared), no KV cache, uses kv13/14
        // chunk4: layers 25-34 (all shared) + norm + lm_head, no KV cache, uses kv13/14
        loadingStatus = "Loading chunk 1/4..."
        chunk1 = try MLModel(contentsOf: findModel(in: folder, name: "chunk1")!, configuration: mlConfig)

        loadingStatus = "Loading chunk 2/4..."
        chunk2 = try MLModel(contentsOf: findModel(in: folder, name: "chunk2")!, configuration: mlConfig)

        loadingStatus = "Loading chunk 3/4..."
        chunk3 = try MLModel(contentsOf: findModel(in: folder, name: "chunk3")!, configuration: mlConfig)

        loadingStatus = "Loading chunk 4/4..."
        chunk4 = try MLModel(contentsOf: findModel(in: folder, name: "chunk4")!, configuration: mlConfig)

        // Allocate persistent KV cache buffers (zero-filled)
        loadingStatus = "Allocating KV cache..."
        let maxHd = 512  // global_head_dim
        try allocateStatelessKV(maxHd: maxHd)

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

    /// Allocate zero-filled KV cache buffers for stateless chunks.
    private func allocateStatelessKV(maxHd: Int) throws {
        let ctx = contextLength
        func zeros(_ slots: Int) throws -> MLMultiArray {
            let arr = try MLMultiArray(
                shape: [NSNumber(value: slots), 1, NSNumber(value: ctx), NSNumber(value: maxHd)],
                dataType: .float16
            )
            memset(arr.dataPointer, 0, slots * ctx * maxHd * MemoryLayout<UInt16>.stride)
            return arr
        }
        statelessKV1_K = try zeros(8)  // chunk1: 8 layers
        statelessKV1_V = try zeros(8)
        statelessKV2_K = try zeros(7)  // chunk2: 7 layers
        statelessKV2_V = try zeros(7)
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
                    self.loadingStatus = "Generating..."

                    let startTime = CFAbsoluteTimeGetCurrent()
                    var tokenCount = 0
                    let eosIDs: Set<Int> = [1, 106, 151645]

                    for _ in 0..<256 {
                        if eosIDs.contains(nextID) { break }
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
            // Zero out persistent KV cache buffers
            if let k1 = statelessKV1_K { memset(k1.dataPointer, 0, k1.count * MemoryLayout<UInt16>.stride) }
            if let v1 = statelessKV1_V { memset(v1.dataPointer, 0, v1.count * MemoryLayout<UInt16>.stride) }
            if let k2 = statelessKV2_K { memset(k2.dataPointer, 0, k2.count * MemoryLayout<UInt16>.stride) }
            if let v2 = statelessKV2_V { memset(v2.dataPointer, 0, v2.count * MemoryLayout<UInt16>.stride) }
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
        // Stateless 4-chunk lite architecture:
        // chunk1: layers 0-7 + embedding, explicit KV I/O (8 slots)
        // chunk2: layers 8-14 (KV sources 13/14), explicit KV I/O (7 slots), outputs kv13/14
        // chunk3: layers 15-24 (all shared), no KV cache, takes kv13/14
        // chunk4: layers 25-34 (all shared) + norm + lm_head, takes kv13/14
        guard let chunk1, let chunk2, let chunk3, let chunk4,
              let embedTokens,
              let k1 = statelessKV1_K, let v1 = statelessKV1_V,
              let k2 = statelessKV2_K, let v2 = statelessKV2_V else {
            var missing = [String]()
            if self.chunk1 == nil { missing.append("chunk1") }
            if self.chunk2 == nil { missing.append("chunk2") }
            if self.chunk3 == nil { missing.append("chunk3") }
            if self.chunk4 == nil { missing.append("chunk4") }
            if self.embedTokens == nil { missing.append("embed_tokens") }
            if self.statelessKV1_K == nil { missing.append("KV1") }
            if self.statelessKV2_K == nil { missing.append("KV2") }
            throw NSError(domain: "LLMRunner", code: 0,
                          userInfo: [NSLocalizedDescriptionKey: "Missing: \(missing.joined(separator: ", "))"])
        }
        let ctx = contextLength

        let t0 = CFAbsoluteTimeGetCurrent()
        // External embedding (text by default). Image tokens: use imageEmbedding.
        let textEmb = try embedTokens.lookup(tokenID, shape: [1, 1, NSNumber(value: hiddenSize)])
        let hiddenIn: MLMultiArray
        if let imageEmbedding {
            hiddenIn = imageEmbedding
        } else {
            hiddenIn = textEmb
        }
        let t1 = CFAbsoluteTimeGetCurrent()
        let plc = try computePerLayerCombined(tokenID: tokenID, embedding: textEmb)
        let t2 = CFAbsoluteTimeGetCurrent()
        profileEmbed += (t1 - t0)
        profilePLE += (t2 - t1)

        let mask = try makeCausalMask(position: position, contextLength: ctx)
        let umask = try makeUpdateMask(position: position, contextLength: ctx)

        // Pre-computed RoPE for this position (from npy tables)
        let cosS = try lookupRoPE(table: cosSlidingTable, position: position, dim: 256)
        let sinS = try lookupRoPE(table: sinSlidingTable, position: position, dim: 256)
        let cosF = try lookupRoPE(table: cosFullTable, position: position, dim: 512)
        let sinF = try lookupRoPE(table: sinFullTable, position: position, dim: 512)

        let tp = CFAbsoluteTimeGetCurrent()

        // ---- Chunk 1: layers 0-7 ----
        let input1 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenIn),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "K_in": MLFeatureValue(multiArray: k1),
            "V_in": MLFeatureValue(multiArray: v1),
        ])
        let out1 = try chunk1.prediction(from: input1)
        let h1 = out1.featureValue(for: "hidden_states_out")!.multiArrayValue!
        let newK1 = out1.featureValue(for: "K_out")!.multiArrayValue!
        let newV1 = out1.featureValue(for: "V_out")!.multiArrayValue!
        memcpy(k1.dataPointer, newK1.dataPointer, k1.count * MemoryLayout<UInt16>.stride)
        memcpy(v1.dataPointer, newV1.dataPointer, v1.count * MemoryLayout<UInt16>.stride)

        // ---- Chunk 2: layers 8-14, emits kv13/14 ----
        let input2 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h1),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
            "per_layer_combined": MLFeatureValue(multiArray: plc),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "K_in": MLFeatureValue(multiArray: k2),
            "V_in": MLFeatureValue(multiArray: v2),
        ])
        let out2 = try chunk2.prediction(from: input2)
        let h2 = out2.featureValue(for: "hidden_states_out")!.multiArrayValue!
        let newK2 = out2.featureValue(for: "K_out")!.multiArrayValue!
        let newV2 = out2.featureValue(for: "V_out")!.multiArrayValue!
        memcpy(k2.dataPointer, newK2.dataPointer, k2.count * MemoryLayout<UInt16>.stride)
        memcpy(v2.dataPointer, newV2.dataPointer, v2.count * MemoryLayout<UInt16>.stride)
        let kv13_k = out2.featureValue(for: "kv13_k")!.multiArrayValue!
        let kv13_v = out2.featureValue(for: "kv13_v")!.multiArrayValue!
        let kv14_k = out2.featureValue(for: "kv14_k")!.multiArrayValue!
        let kv14_v = out2.featureValue(for: "kv14_v")!.multiArrayValue!

        // ---- Chunk 3: layers 15-24 (shared KV) ----
        let input3 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h2),
            "causal_mask": MLFeatureValue(multiArray: mask),
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

        // ---- Chunk 4: layers 25-34 + norm + lm_head ----
        let input4 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: h3),
            "causal_mask": MLFeatureValue(multiArray: mask),
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

    private func makeUpdateMask(position: Int, contextLength: Int) throws -> MLMultiArray {
        let umask = try MLMultiArray(shape: [1, 1, NSNumber(value: contextLength), 1], dataType: .float16)
        let up = umask.dataPointer.bindMemory(to: UInt16.self, capacity: contextLength)
        memset(up, 0, contextLength * MemoryLayout<UInt16>.stride)
        up[position] = 0x3C00
        return umask
    }

    private func processImage(_ image: CGImage, with visionModel: MLModel) throws -> MLMultiArray {
        let ps = 16, total = 2520, pd = 768, sz = 896
        var pixels = [UInt8](repeating: 0, count: sz * sz * 4)
        let ctx = CGContext(data: &pixels, width: sz, height: sz, bitsPerComponent: 8,
                            bytesPerRow: sz * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: sz, height: sz))

        let pv = try MLMultiArray(shape: [1, NSNumber(value: total), NSNumber(value: pd)], dataType: .float32)
        let pid = try MLMultiArray(shape: [1, NSNumber(value: total), 2], dataType: .int32)
        let pvp = pv.dataPointer.bindMemory(to: Float.self, capacity: total * pd)
        let pidp = pid.dataPointer.bindMemory(to: Int32.self, capacity: total * 2)

        var pi = 0; let pps = sz / ps
        for py in 0..<pps { for px in 0..<pps {
            guard pi < total else { break }
            var o = pi * pd
            for dy in 0..<ps { for dx in 0..<ps {
                let po = ((py * ps + dy) * sz + (px * ps + dx)) * 4
                pvp[o] = Float(pixels[po])/255; pvp[o+1] = Float(pixels[po+1])/255; pvp[o+2] = Float(pixels[po+2])/255; o += 3
            }}
            pidp[pi*2] = Int32(px); pidp[pi*2+1] = Int32(py); pi += 1
        }}
        for i in pi..<total { pidp[i*2] = -1; pidp[i*2+1] = -1 }

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
        for m in messages {
            if m.role == .user {
                if hasImage {
                    p += "<|turn>user\n\n\n\(String(repeating: "<|image|>", count: 256))\n\n\(m.content)<turn|>\n"
                } else { p += "<|turn>user\n\(m.content)<turn|>\n" }
            } else if m.role == .assistant { p += "<|turn>model\n\(m.content)<turn|>\n" }
        }
        return p + "<|turn>model\n"
    }
}
