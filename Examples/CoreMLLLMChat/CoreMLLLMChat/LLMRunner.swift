import CoreML
import Foundation

/// Manages CoreML LLM model loading and inference.
/// Handles the low-level CoreML prediction loop with KV cache state.
@Observable
final class LLMRunner {
    var isLoaded = false
    var isGenerating = false
    var loadingStatus = "Not loaded"
    var tokensPerSecond: Double = 0

    private var model: MLModel?
    private var state: MLState?
    private var contextLength = 512
    private var currentPosition = 0

    // Tokenizer (simplified — in production, use swift-transformers)
    private var tokenizer: SimpleTokenizer?
    private var architecture = "gemma4"

    func loadModel(from url: URL) async throws {
        loadingStatus = "Compiling model..."

        let compiledURL = try await MLModel.compileModel(at: url)

        loadingStatus = "Loading model..."
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        let mlModel = try MLModel(contentsOf: compiledURL, configuration: config)
        self.model = mlModel
        self.state = mlModel.makeState()

        // Load model config to get context length
        let configURL = url.deletingLastPathComponent().appendingPathComponent("model_config.json")
        if let data = try? Data(contentsOf: configURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let ctx = json["context_length"] as? Int {
                self.contextLength = ctx
            }
            if let arch = json["architecture"] as? String {
                self.architecture = arch
            }
        }

        // Load tokenizer
        let tokenizerURL = url.deletingLastPathComponent().appendingPathComponent("hf_model")
        self.tokenizer = SimpleTokenizer(modelPath: tokenizerURL)

        isLoaded = true
        currentPosition = 0
        loadingStatus = "Ready"
    }

    func generate(messages: [ChatMessage]) async throws -> AsyncStream<String> {
        guard let model, let state, let tokenizer else {
            throw LLMError.modelNotLoaded
        }

        isGenerating = true

        return AsyncStream { continuation in
            Task {
                defer {
                    self.isGenerating = false
                }

                do {
                    // Build prompt from messages
                    let prompt = self.buildPrompt(messages: messages)
                    let tokenIDs = tokenizer.encode(prompt)

                    // Reset state for new conversation
                    self.state = model.makeState()
                    self.currentPosition = 0

                    // Prefill
                    var nextID = 0
                    for (step, tid) in tokenIDs.enumerated() {
                        let result = try self.predict(tokenID: tid, position: step)
                        nextID = result.tokenID
                        self.currentPosition = step + 1
                    }

                    // Decode
                    let startTime = CFAbsoluteTimeGetCurrent()
                    var tokenCount = 0
                    let eosTokenIDs: Set<Int> = [1, 106, 151645]  // Gemma + Qwen EOS

                    for _ in 0..<256 {
                        if eosTokenIDs.contains(nextID) { break }

                        let text = tokenizer.decode([nextID])
                        continuation.yield(text)
                        tokenCount += 1

                        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                        if elapsed > 0 {
                            self.tokensPerSecond = Double(tokenCount) / elapsed
                        }

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

    // MARK: - Private

    private func predict(tokenID: Int, position: Int) throws -> (tokenID: Int, logit: Float) {
        guard let model, let state else {
            throw LLMError.modelNotLoaded
        }

        let ctx = contextLength

        let inputIDs = try MLMultiArray(shape: [1, 1], dataType: .int32)
        inputIDs[[0, 0] as [NSNumber]] = NSNumber(value: Int32(tokenID))

        let positionIDs = try MLMultiArray(shape: [1], dataType: .int32)
        positionIDs[0] = NSNumber(value: Int32(position))

        let causalMask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: ctx)], dataType: .float16
        )
        let maskPtr = causalMask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        let negInf: UInt16 = 0xFC00
        for i in 0..<ctx {
            maskPtr[i] = i <= position ? 0 : negInf
        }

        let updateMask = try MLMultiArray(
            shape: [1, 1, NSNumber(value: ctx), 1], dataType: .float16
        )
        let updatePtr = updateMask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        memset(updatePtr, 0, ctx * MemoryLayout<UInt16>.stride)
        updatePtr[position] = 0x3C00  // 1.0 in float16

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "position_ids": MLFeatureValue(multiArray: positionIDs),
            "causal_mask": MLFeatureValue(multiArray: causalMask),
            "update_mask": MLFeatureValue(multiArray: updateMask),
        ])

        let output = try model.prediction(from: input, using: state)

        guard let tokenIDArr = output.featureValue(for: "token_id")?.multiArrayValue,
              let tokenLogit = output.featureValue(for: "token_logit")?.multiArrayValue else {
            throw LLMError.predictionFailed
        }

        return (tokenID: tokenIDArr[0].intValue, logit: tokenLogit[0].floatValue)
    }

    private func buildPrompt(messages: [ChatMessage]) -> String {
        if architecture.hasPrefix("qwen") {
            return buildQwenPrompt(messages: messages)
        } else {
            return buildGemmaPrompt(messages: messages)
        }
    }

    private func buildGemmaPrompt(messages: [ChatMessage]) -> String {
        var prompt = "<bos>"
        for msg in messages {
            switch msg.role {
            case .user:
                prompt += "<|turn>user\n\(msg.content)<turn|>\n"
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
            case .system:
                break
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
