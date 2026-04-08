import CoreML
import Foundation
import Tokenizers

/// Manages CoreML LLM model loading and inference with multimodal support.
@Observable
final class LLMRunner {
    var isLoaded = false
    var isGenerating = false
    var loadingStatus = "Not loaded"
    var tokensPerSecond: Double = 0
    var modelName = ""

    private var model: MLModel?
    private var visionModel: MLModel?
    private var state: MLState?
    private var contextLength = 512
    private var currentPosition = 0
    private var architecture = "gemma4"
    private var hiddenSize = 1536

    private var tokenizer: (any Tokenizer)?

    func loadModel(from url: URL) async throws {
        let folder = url.deletingLastPathComponent()

        // Load config
        loadingStatus = "Reading config..."
        let configURL = folder.appendingPathComponent("model_config.json")
        if let data = try? Data(contentsOf: configURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            contextLength = json["context_length"] as? Int ?? 512
            architecture = json["architecture"] as? String ?? "gemma4"
            hiddenSize = json["hidden_size"] as? Int ?? 1536
            modelName = json["model_name"] as? String ?? "Model"
        }

        // Compile and load main model
        loadingStatus = "Compiling model..."
        let compiledURL = try await MLModel.compileModel(at: url)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: compiledURL, configuration: config)
        state = model?.makeState()

        // Load vision model if available
        let visionURL = folder.appendingPathComponent("vision.mlpackage")
        if FileManager.default.fileExists(atPath: visionURL.path) {
            loadingStatus = "Compiling vision model..."
            let compiledVision = try await MLModel.compileModel(at: visionURL)
            visionModel = try MLModel(contentsOf: compiledVision, configuration: config)
        }

        // Load tokenizer
        loadingStatus = "Loading tokenizer..."
        let tokenizerRepo = folder.appendingPathComponent("hf_model")
        if FileManager.default.fileExists(atPath: tokenizerRepo.appendingPathComponent("tokenizer.json").path) {
            tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerRepo)
        }

        isLoaded = true
        currentPosition = 0
        loadingStatus = "Ready"
    }

    var hasVision: Bool { visionModel != nil }

    // MARK: - Text Generation

    func generate(messages: [ChatMessage], image: CGImage? = nil) async throws -> AsyncStream<String> {
        guard let model, let state, let tokenizer else {
            throw LLMError.modelNotLoaded
        }

        isGenerating = true

        return AsyncStream { continuation in
            Task {
                defer { self.isGenerating = false }

                do {
                    let prompt = self.buildPrompt(messages: messages, hasImage: image != nil)
                    let tokenIDs = tokenizer.encode(text: prompt)

                    // Reset state
                    self.state = model.makeState()
                    self.currentPosition = 0

                    // Process image if provided
                    var imageFeatures: MLMultiArray?
                    var imageTokenCount = 0
                    if let image, let visionModel = self.visionModel {
                        imageFeatures = try self.processImage(image, with: visionModel)
                        imageTokenCount = 256  // Fixed for Gemma 4
                    }

                    let IMAGE_TOKEN_ID = 258880
                    let PAD_ID = 0
                    var imageIdx = 0

                    // Prefill
                    var nextID = 0
                    for (step, tid) in tokenIDs.enumerated() {
                        if tid == IMAGE_TOKEN_ID, let feats = imageFeatures, imageIdx < imageTokenCount {
                            let imgEmb = self.sliceImageFeature(feats, at: imageIdx)
                            nextID = try self.predict(tokenID: PAD_ID, position: step, imageEmbedding: imgEmb).tokenID
                            imageIdx += 1
                        } else {
                            nextID = try self.predict(tokenID: tid, position: step).tokenID
                        }
                        self.currentPosition = step + 1
                    }

                    // Decode
                    let startTime = CFAbsoluteTimeGetCurrent()
                    var tokenCount = 0
                    let eosTokenIDs: Set<Int> = [1, 106, 151645]

                    for _ in 0..<256 {
                        if eosTokenIDs.contains(nextID) { break }

                        let text = tokenizer.decode(tokens: [nextID])
                        continuation.yield(text)
                        tokenCount += 1

                        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                        if elapsed > 0 { self.tokensPerSecond = Double(tokenCount) / elapsed }

                        let result = try self.predict(tokenID: nextID, position: self.currentPosition)
                        nextID = result.tokenID
                        self.currentPosition += 1
                    }

                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
        }
    }

    func resetConversation() {
        guard let model else { return }
        state = model.makeState()
        currentPosition = 0
    }

    // MARK: - Prediction

    private func predict(tokenID: Int, position: Int, imageEmbedding: MLMultiArray? = nil) throws -> (tokenID: Int, logit: Float) {
        guard let model, let state else { throw LLMError.modelNotLoaded }

        let ctx = contextLength
        let inputIDs = try MLMultiArray(shape: [1, 1], dataType: .int32)
        inputIDs[[0, 0] as [NSNumber]] = NSNumber(value: Int32(tokenID))

        let positionIDs = try MLMultiArray(shape: [1], dataType: .int32)
        positionIDs[0] = NSNumber(value: Int32(position))

        let causalMask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: ctx)], dataType: .float16)
        let maskPtr = causalMask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        let negInf: UInt16 = 0xFC00
        for i in 0..<ctx { maskPtr[i] = i <= position ? 0 : negInf }

        let updateMask = try MLMultiArray(shape: [1, 1, NSNumber(value: ctx), 1], dataType: .float16)
        let updatePtr = updateMask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        memset(updatePtr, 0, ctx * MemoryLayout<UInt16>.stride)
        updatePtr[position] = 0x3C00

        var inputDict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "position_ids": MLFeatureValue(multiArray: positionIDs),
            "causal_mask": MLFeatureValue(multiArray: causalMask),
            "update_mask": MLFeatureValue(multiArray: updateMask),
        ]

        // Add image embedding if the model supports it
        let imgEmb = imageEmbedding ?? (try MLMultiArray(shape: [1, 1, NSNumber(value: hiddenSize)], dataType: .float16))
        if imageEmbedding == nil {
            let ptr = imgEmb.dataPointer.bindMemory(to: UInt16.self, capacity: hiddenSize)
            memset(ptr, 0, hiddenSize * MemoryLayout<UInt16>.stride)
        }
        inputDict["image_embedding"] = MLFeatureValue(multiArray: imgEmb)

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try model.prediction(from: input, using: state)

        guard let tokenIDArr = output.featureValue(for: "token_id")?.multiArrayValue,
              let tokenLogit = output.featureValue(for: "token_logit")?.multiArrayValue else {
            throw LLMError.predictionFailed
        }

        return (tokenID: tokenIDArr[0].intValue, logit: tokenLogit[0].floatValue)
    }

    // MARK: - Vision Processing

    private func processImage(_ image: CGImage, with visionModel: MLModel) throws -> MLMultiArray {
        // Gemma 4 vision: image → 2520 patches of 768 dims
        // Resize image to standard grid, create patches + position IDs
        let patchSize = 16
        let targetPatches = 2520

        // Resize to fit patch grid
        let (pixelValues, positionIDs) = createImagePatches(image, patchSize: patchSize, totalPatches: targetPatches)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pixelValues),
            "pixel_position_ids": MLFeatureValue(multiArray: positionIDs),
        ])

        let output = try visionModel.prediction(from: input)
        guard let features = output.featureValue(for: "image_features")?.multiArrayValue else {
            throw LLMError.predictionFailed
        }

        return features  // (1, 280, 1536)
    }

    private func sliceImageFeature(_ features: MLMultiArray, at index: Int) -> MLMultiArray {
        // Extract (1, 1, hidden_size) from (1, 280, hidden_size) at index
        let hs = hiddenSize
        let result = try! MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16)
        let srcPtr = features.dataPointer.bindMemory(to: UInt16.self, capacity: features.count)
        let dstPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: hs)
        memcpy(dstPtr, srcPtr.advanced(by: index * hs), hs * MemoryLayout<UInt16>.stride)
        return result
    }

    private func createImagePatches(_ image: CGImage, patchSize: Int, totalPatches: Int) -> (MLMultiArray, MLMultiArray) {
        // Resize image to standard size and create patches
        let targetSize = 896  // Standard Gemma 4 processing size
        let w = targetSize
        let h = targetSize

        // Render image to pixel buffer
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let context = CGContext(data: &pixels, width: w, height: h, bitsPerComponent: 8,
                               bytesPerRow: w * 4, space: colorSpace,
                               bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

        let patchDim = 3 * patchSize * patchSize  // 768
        let patchesPerSide = w / patchSize  // 56
        let numPatches = patchesPerSide * patchesPerSide  // 3136 > 2520

        let pixelValues = try! MLMultiArray(shape: [1, NSNumber(value: totalPatches), NSNumber(value: patchDim)], dataType: .float32)
        let positionIDs = try! MLMultiArray(shape: [1, NSNumber(value: totalPatches), 2], dataType: .int32)

        let pvPtr = pixelValues.dataPointer.bindMemory(to: Float.self, capacity: totalPatches * patchDim)
        let pidPtr = positionIDs.dataPointer.bindMemory(to: Int32.self, capacity: totalPatches * 2)

        // Fill patches
        var patchIdx = 0
        for py in 0..<min(patchesPerSide, totalPatches / patchesPerSide) {
            for px in 0..<patchesPerSide {
                if patchIdx >= totalPatches { break }
                // Extract patch pixels
                var offset = patchIdx * patchDim
                for dy in 0..<patchSize {
                    for dx in 0..<patchSize {
                        let ix = px * patchSize + dx
                        let iy = py * patchSize + dy
                        let pixelOffset = (iy * w + ix) * 4
                        // RGB, rescaled to [0, 1]
                        pvPtr[offset] = Float(pixels[pixelOffset]) / 255.0
                        pvPtr[offset + 1] = Float(pixels[pixelOffset + 1]) / 255.0
                        pvPtr[offset + 2] = Float(pixels[pixelOffset + 2]) / 255.0
                        offset += 3
                    }
                }
                pidPtr[patchIdx * 2] = Int32(px)
                pidPtr[patchIdx * 2 + 1] = Int32(py)
                patchIdx += 1
            }
        }

        // Fill remaining patches with padding (-1)
        for i in patchIdx..<totalPatches {
            pidPtr[i * 2] = -1
            pidPtr[i * 2 + 1] = -1
        }

        return (pixelValues, positionIDs)
    }

    // MARK: - Prompt Building

    private func buildPrompt(messages: [ChatMessage], hasImage: Bool) -> String {
        if architecture.hasPrefix("qwen") {
            return buildQwenPrompt(messages: messages)
        } else {
            return buildGemmaPrompt(messages: messages, hasImage: hasImage)
        }
    }

    private func buildGemmaPrompt(messages: [ChatMessage], hasImage: Bool) -> String {
        var prompt = "<bos>"
        for msg in messages {
            switch msg.role {
            case .user:
                if hasImage {
                    // Insert image tokens before text
                    let imageTokens = String(repeating: "<|image|>", count: 256)
                    prompt += "<|turn>user\n\n\n\(imageTokens)\n\n\(msg.content)<turn|>\n"
                } else {
                    prompt += "<|turn>user\n\(msg.content)<turn|>\n"
                }
            case .assistant:
                prompt += "<|turn>model\n\(msg.content)<turn|>\n"
            case .system:
                prompt += "<|turn>system\n\(msg.content)<turn|>\n"
            }
        }
        prompt += "<|turn>model\n"
        return prompt
    }

    private func buildQwenPrompt(messages: [ChatMessage]) -> String {
        var prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        for msg in messages {
            switch msg.role {
            case .user:
                prompt += "<|im_start|>user\n\(msg.content)<|im_end|>\n"
            case .assistant:
                prompt += "<|im_start|>assistant\n\(msg.content)<|im_end|>\n"
            case .system: break
            }
        }
        prompt += "<|im_start|>assistant\n"
        return prompt
    }
}

enum LLMError: LocalizedError {
    case modelNotLoaded
    case predictionFailed
    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Model not loaded"
        case .predictionFailed: return "Prediction failed"
        }
    }
}
