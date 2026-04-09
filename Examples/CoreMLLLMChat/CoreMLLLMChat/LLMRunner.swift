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

    // Chunked model (for large models that don't fit in single compilation)
    private var chunk1: MLModel?
    private var chunk2: MLModel?
    private var chunk3: MLModel?
    private var chunk1State: MLState?
    private var chunk2State: MLState?
    private var isChunked = false

    // Vision
    private var visionModel: MLModel?

    // External embeddings (for chunked model without embedded vocab tables)
    private var embedTokens: EmbeddingLookup?
    private var embedPerLayer: EmbeddingLookup?
    private var perLayerProjWeight: Data?  // (8960, 1536) float16
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
    private var currentPosition = 0
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
        mlConfig.computeUnits = .cpuAndGPU

        // Detect: chunked or monolithic?
        let chunk1URL = findModel(in: folder, name: "chunk1")
        if let c1url = chunk1URL {
            // Chunked model
            try await loadChunked(folder: folder, config: mlConfig)
        } else {
            // Monolithic model
            try await loadMonolithic(url: url, folder: folder, config: mlConfig)
        }

        // Vision model
        if let vurl = findModel(in: folder, name: "vision") {
            loadingStatus = "Loading vision..."
            visionModel = try MLModel(contentsOf: vurl, configuration: mlConfig)
            hasVision = true
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
    }

    private func loadChunked(folder: URL, config mlConfig: MLModelConfiguration) async throws {
        loadingStatus = "Loading chunk 1/3..."
        chunk1 = try MLModel(contentsOf: findModel(in: folder, name: "chunk1")!, configuration: mlConfig)
        chunk1State = chunk1?.makeState()

        loadingStatus = "Loading chunk 2/3..."
        chunk2 = try MLModel(contentsOf: findModel(in: folder, name: "chunk2")!, configuration: mlConfig)
        chunk2State = chunk2?.makeState()

        loadingStatus = "Loading chunk 3/3..."
        chunk3 = try MLModel(contentsOf: findModel(in: folder, name: "chunk3")!, configuration: mlConfig)

        // Load external embeddings (memory-mapped)
        loadingStatus = "Loading embeddings..."
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

        // RoPE tables (numpy .npy files → raw float16 data, skip 128-byte npy header)
        loadingStatus = "Loading RoPE tables..."
        cosSlidingTable = try? Data(contentsOf: folder.appendingPathComponent("cos_sliding.npy"), options: .mappedIfSafe)
        sinSlidingTable = try? Data(contentsOf: folder.appendingPathComponent("sin_sliding.npy"), options: .mappedIfSafe)
        cosFullTable = try? Data(contentsOf: folder.appendingPathComponent("cos_full.npy"), options: .mappedIfSafe)
        sinFullTable = try? Data(contentsOf: folder.appendingPathComponent("sin_full.npy"), options: .mappedIfSafe)

        isChunked = true
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
        if let image, let vm = visionModel {
            imageFeatures = try processImage(image, with: vm)
        }

        resetConversation()

        return AsyncStream { continuation in
            Task {
                defer { self.isGenerating = false }
                do {
                    let IMAGE_TOKEN_ID = 258880
                    var imageIdx = 0
                    var nextID = 0

                    for (step, tid) in tokenIDs.enumerated() {
                        if tid == IMAGE_TOKEN_ID, let feats = imageFeatures, imageIdx < 256 {
                            let imgEmb = self.sliceFeature(feats, at: imageIdx)
                            nextID = try self.predictStep(tokenID: 0, position: step, imageEmbedding: imgEmb)
                            imageIdx += 1
                        } else {
                            nextID = try self.predictStep(tokenID: tid, position: step)
                        }
                        self.currentPosition = step + 1
                    }

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
                        nextID = try self.predictStep(tokenID: nextID, position: self.currentPosition)
                        self.currentPosition += 1
                    }
                } catch {}
                continuation.finish()
            }
        }
    }

    func resetConversation() {
        if isChunked {
            chunk1State = chunk1?.makeState()
            chunk2State = chunk2?.makeState()
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
        guard let model, let state else { throw NSError(domain: "", code: 0) }
        let ctx = contextLength, hs = hiddenSize

        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
        ids[[0, 0] as [NSNumber]] = NSNumber(value: Int32(tokenID))
        let pos = try MLMultiArray(shape: [1], dataType: .int32)
        pos[0] = NSNumber(value: Int32(position))
        let mask = try makeCausalMask(position: position, contextLength: ctx)
        let umask = try makeUpdateMask(position: position, contextLength: ctx)

        let imgEmb: MLMultiArray
        if let imageEmbedding {
            imgEmb = imageEmbedding
        } else {
            imgEmb = try MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16)
            memset(imgEmb.dataPointer, 0, hs * MemoryLayout<UInt16>.stride)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
            "position_ids": MLFeatureValue(multiArray: pos),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
            "image_embedding": MLFeatureValue(multiArray: imgEmb),
        ])
        let output = try model.prediction(from: input, using: state)
        return output.featureValue(for: "token_id")!.multiArrayValue![0].intValue
    }

    // MARK: - Chunked Prediction

    private func predictChunked(tokenID: Int, position: Int, imageEmbedding: MLMultiArray? = nil) throws -> Int {
        guard let chunk1, let chunk2, let chunk3,
              let chunk1State, let chunk2State,
              let embedTokens, let embedPerLayer else { throw NSError(domain: "", code: 0) }
        let ctx = contextLength, hs = hiddenSize

        let mask = try makeCausalMask(position: position, contextLength: ctx)
        let umask = try makeUpdateMask(position: position, contextLength: ctx)

        // Compute cos/sin for current position from RoPE tables
        let cosS = try lookupRoPE(table: cosSlidingTable, position: position, dim: 256)
        let sinS = try lookupRoPE(table: sinSlidingTable, position: position, dim: 256)
        let cosF = try lookupRoPE(table: cosFullTable, position: position, dim: 512)
        let sinF = try lookupRoPE(table: sinFullTable, position: position, dim: 512)

        // Compute embeddings externally
        let hiddenStatesIn: MLMultiArray
        if let imageEmbedding {
            hiddenStatesIn = imageEmbedding
        } else {
            hiddenStatesIn = try embedTokens.lookup(tokenID, shape: [1, 1, NSNumber(value: hs)])
        }

        let perLayerCombined = try computePerLayerCombined(tokenID: tokenID, embedding: hiddenStatesIn)

        // Chunk 1: layers 0-11 (cos/sin instead of position_ids)
        let input1 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenStatesIn),
            "per_layer_combined": MLFeatureValue(multiArray: perLayerCombined),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
        ])
        let out1 = try chunk1.prediction(from: input1, using: chunk1State)
        let hiddenStates = out1.featureValue(for: "hidden_states_out")!.multiArrayValue!

        // Chunk 2: layers 12-23 → hidden_states + kv13/kv14
        let input2 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenStates),
            "per_layer_combined": MLFeatureValue(multiArray: perLayerCombined),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
        ])
        let out2 = try chunk2.prediction(from: input2, using: chunk2State)
        let hiddenStates2 = out2.featureValue(for: "hidden_states_out")!.multiArrayValue!
        let kv13_k = out2.featureValue(for: "kv13_k")!.multiArrayValue!
        let kv13_v = out2.featureValue(for: "kv13_v")!.multiArrayValue!
        let kv14_k = out2.featureValue(for: "kv14_k")!.multiArrayValue!
        let kv14_v = out2.featureValue(for: "kv14_v")!.multiArrayValue!

        // Chunk 3: layers 24-34 + norm + lm_head → token_id
        let input3 = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_states": MLFeatureValue(multiArray: hiddenStates2),
            "per_layer_combined": MLFeatureValue(multiArray: perLayerCombined),
            "cos_s": MLFeatureValue(multiArray: cosS),
            "sin_s": MLFeatureValue(multiArray: sinS),
            "cos_f": MLFeatureValue(multiArray: cosF),
            "sin_f": MLFeatureValue(multiArray: sinF),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "kv13_k": MLFeatureValue(multiArray: kv13_k),
            "kv13_v": MLFeatureValue(multiArray: kv13_v),
            "kv14_k": MLFeatureValue(multiArray: kv14_k),
            "kv14_v": MLFeatureValue(multiArray: kv14_v),
        ])
        let out3 = try chunk3.prediction(from: input3)
        return out3.featureValue(for: "token_id")!.multiArrayValue![0].intValue
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
        guard let embedPerLayer, let perLayerProjWeight else {
            throw NSError(domain: "LLMRunner", code: 2)
        }
        let nlayers = 35, pld = perLayerDim
        let totalDim = nlayers * pld  // 8960
        let result = try MLMultiArray(shape: [1, 1, NSNumber(value: totalDim)], dataType: .float16)
        let resultPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: totalDim)

        // Raw per-layer embedding
        let raw = embedPerLayer.lookupRaw(tokenID)

        // Projection: per_layer_model_projection(embedding) * scale
        // projection weight: (8960, 1536) float16
        let embPtr = embedding.dataPointer.bindMemory(to: UInt16.self, capacity: hiddenSize)

        // Step 1: Compute projection = embedding @ projWeight^T * projScale
        var proj = [Float](repeating: 0, count: totalDim)
        perLayerProjWeight.withUnsafeBytes { rawBuf in
            let projWPtr = rawBuf.baseAddress!.assumingMemoryBound(to: UInt16.self)
            for i in 0..<totalDim {
                var sum: Float = 0
                let rowStart = i * hiddenSize
                for j in 0..<hiddenSize {
                    let e = float16ToFloat(embPtr[j])
                    let w = float16ToFloat(projWPtr[rowStart + j])
                    sum += e * w
                }
                proj[i] = sum * perLayerProjScale
            }
        }

        // Step 2: Apply RMSNorm to each per_layer_dim slice of projection
        if let normData = perLayerNormWeight {
            normData.withUnsafeBytes { normRaw in
                let normW = normRaw.baseAddress!.assumingMemoryBound(to: Float.self)
                let eps: Float = 1e-6
                for li in 0..<nlayers {
                    let s = li * pld
                    var meanSq: Float = 0
                    for j in 0..<pld {
                        meanSq += proj[s + j] * proj[s + j]
                    }
                    meanSq = meanSq / Float(pld) + eps
                    let invRms = 1.0 / sqrtf(meanSq)
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
