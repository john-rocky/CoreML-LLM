import CoreML
import Foundation
import Transformers

/// Main entry point for CoreML LLM inference on Apple devices.
///
/// Usage:
/// ```swift
/// let llm = try await CoreMLLLM.load(from: modelDirectory)
/// let response = try await llm.generate("Hello, world!")
/// print(response)
/// ```
public final class CoreMLLLM: @unchecked Sendable {
    private let model: LLMModel
    private let engine: InferenceEngine

    /// Performance info from the last generation run.
    public private(set) var lastBenchmark: Benchmark?

    private init(model: LLMModel) {
        self.model = model
        self.engine = InferenceEngine(model: model)
    }

    // MARK: - Loading

    /// Load a model from a directory containing mlpackage files and model_config.json.
    ///
    /// - Parameters:
    ///   - directory: URL to the model directory
    ///   - computeUnits: CoreML compute units (.cpuAndNeuralEngine for ANE+CPU)
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> CoreMLLLM {
        let model = try await LLMModel.load(from: directory, computeUnits: computeUnits)
        return CoreMLLLM(model: model)
    }

    // MARK: - Generation

    /// Generate text from a prompt string.
    ///
    /// - Parameters:
    ///   - prompt: Input text prompt
    ///   - maxTokens: Maximum number of tokens to generate (default: 256)
    ///   - onToken: Optional callback with each generated token string.
    ///              Return `false` to stop generation early.
    /// - Returns: The generated text
    public func generate(
        _ prompt: String,
        maxTokens: Int = 256,
        onToken: ((String) -> Bool)? = nil
    ) async throws -> String {
        let tokenIDs = model.tokenizer.encode(text: prompt)

        if tokenIDs.isEmpty {
            throw CoreMLLLMError.emptyPrompt
        }

        let result = try engine.generate(
            promptTokens: tokenIDs,
            maxNewTokens: maxTokens,
            onToken: onToken.map { callback in
                { tokenID in
                    let text = self.model.tokenizer.decode(tokens: [tokenID])
                    return callback(text)
                }
            }
        )

        lastBenchmark = result.benchmark
        return model.tokenizer.decode(tokens: result.tokens)
    }

    /// Generate text from chat messages using the model's chat template.
    ///
    /// - Parameters:
    ///   - messages: Array of message dictionaries with "role" and "content" keys
    ///   - maxTokens: Maximum number of tokens to generate
    ///   - onToken: Optional callback for streaming
    /// - Returns: The assistant's response text
    public func chat(
        messages: [[String: String]],
        maxTokens: Int = 256,
        onToken: ((String) -> Bool)? = nil
    ) async throws -> String {
        // Apply chat template via swift-transformers
        let prompt = try model.tokenizer.applyChatTemplate(messages: messages)
        return try await generate(prompt, maxTokens: maxTokens, onToken: onToken)
    }

    /// Reset the KV cache state for a new conversation.
    public func reset() {
        model.resetState()
    }

    /// Model configuration info.
    public var configuration: LLMModelConfiguration {
        model.config
    }
}
