import CoreML
import Foundation

/// Orchestrates the prefill + decode loop for text generation.
final class InferenceEngine: @unchecked Sendable {
    private let model: LLMModel

    init(model: LLMModel) {
        self.model = model
    }

    /// Generate tokens from a list of prompt token IDs.
    func generate(
        promptTokens: [Int],
        maxNewTokens: Int,
        onToken: ((Int) -> Bool)? = nil
    ) throws -> (tokens: [Int], benchmark: Benchmark) {
        let overallStart = CFAbsoluteTimeGetCurrent()
        var generatedTokens: [Int] = []

        // --- Prefill: process all prompt tokens ---
        let prefillStart = CFAbsoluteTimeGetCurrent()
        var nextID = 0

        for (step, tokenID) in promptTokens.enumerated() {
            let result = try model.predict(tokenID: tokenID, position: step)
            nextID = result.tokenID
        }

        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart

        // --- Decode: generate new tokens ---
        let decodeStart = CFAbsoluteTimeGetCurrent()
        let eosTokenID = model.config.eosTokenId

        for step in 0..<maxNewTokens {
            if nextID == eosTokenID {
                break
            }

            generatedTokens.append(nextID)

            if let onToken, !onToken(nextID) {
                break
            }

            let pos = promptTokens.count + step
            let result = try model.predict(tokenID: nextID, position: pos)
            nextID = result.tokenID
        }

        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        let totalTime = CFAbsoluteTimeGetCurrent() - overallStart

        let benchmark = Benchmark(
            tokenCount: generatedTokens.count,
            prefillTime: prefillTime,
            decodeTime: decodeTime,
            totalTime: totalTime
        )

        return (tokens: generatedTokens, benchmark: benchmark)
    }
}
