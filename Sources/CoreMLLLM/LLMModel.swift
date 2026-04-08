import CoreML
import Foundation
import Transformers

/// Manages the CoreML model and tokenizer for LLM inference.
///
/// Supports monolithic model (single .mlpackage with stateful KV cache).
/// Inputs: input_ids, position_ids, causal_mask, update_mask
/// Outputs: token_id, token_logit
/// State: kv_cache_0
final class LLMModel: @unchecked Sendable {
    let config: LLMModelConfiguration
    let tokenizer: any Tokenizer
    let model: MLModel
    var state: MLState

    init(
        config: LLMModelConfiguration,
        tokenizer: any Tokenizer,
        model: MLModel,
        state: MLState
    ) {
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.state = state
    }

    /// Load a model from a directory containing model.mlpackage and model_config.json.
    static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> LLMModel {
        let configURL = directory.appendingPathComponent("model_config.json")
        let config = try LLMModelConfiguration.load(from: configURL)

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        // Load model
        let modelFile = config.parts.model ?? "model.mlpackage"
        let modelURL = directory.appendingPathComponent(modelFile)

        let compiledURL = try await MLModel.compileModel(at: modelURL)
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)

        // Create initial state for KV cache
        let state = mlModel.makeState()

        // Load tokenizer from HuggingFace
        let tokenizer = try await AutoTokenizer.from(pretrained: config.tokenizerRepo)

        return LLMModel(
            config: config,
            tokenizer: tokenizer,
            model: mlModel,
            state: state
        )
    }

    // MARK: - Inference

    /// Run a single decode step: token_id -> next token prediction.
    func predict(tokenID: Int, position: Int) throws -> (tokenID: Int, logit: Float) {
        let ctx = config.contextLength

        // Input arrays
        let inputIDs = try MLMultiArray(shape: [1, 1], dataType: .int32)
        inputIDs[[0, 0] as [NSNumber]] = NSNumber(value: Int32(tokenID))

        let positionIDs = try MLMultiArray(shape: [1], dataType: .int32)
        positionIDs[0] = NSNumber(value: Int32(position))

        // Causal mask: -inf for positions > current, 0 for positions <= current
        let causalMask = try MLMultiArray(
            shape: [1, 1, 1, NSNumber(value: ctx)], dataType: .float16
        )
        let maskPtr = causalMask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        let negInf: UInt16 = 0xFC00
        for i in 0..<ctx {
            maskPtr[i] = i <= position ? 0 : negInf
        }

        // Update mask: 1.0 at current position, 0.0 elsewhere
        let updateMask = try MLMultiArray(
            shape: [1, 1, NSNumber(value: ctx), 1], dataType: .float16
        )
        let updatePtr = updateMask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        memset(updatePtr, 0, ctx * MemoryLayout<UInt16>.stride)
        updatePtr[position] = 0x3C00 // 1.0 in float16

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "position_ids": MLFeatureValue(multiArray: positionIDs),
            "causal_mask": MLFeatureValue(multiArray: causalMask),
            "update_mask": MLFeatureValue(multiArray: updateMask),
        ])

        let output = try model.prediction(from: input, using: state)

        guard let tokenIDArr = output.featureValue(for: "token_id")?.multiArrayValue,
              let tokenLogit = output.featureValue(for: "token_logit")?.multiArrayValue else {
            throw CoreMLLLMError.missingOutput("token_id/token_logit")
        }

        return (tokenID: tokenIDArr[0].intValue, logit: tokenLogit[0].floatValue)
    }

    /// Reset KV cache state for a new conversation.
    func resetState() {
        state = model.makeState()
    }
}
