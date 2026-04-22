import CoreML
import Foundation
import Tokenizers

/// Minimal runtime for FunctionGemma-270M (Gemma 3 decoder fine-tuned for
/// function calling).
///
/// Intentionally narrow API — just load + generate. LocalAIKit (or any other
/// wrapper) is expected to compose this with orchestration (RAG, chat
/// sessions, multi-model pipelines). Nothing here depends on the Gemma 4
/// stack; the two live side by side in the same Swift target but share no
/// runtime state.
///
/// I/O contract of the underlying mlpackage (from
/// `conversion/build_functiongemma_bundle.py`):
///   input_ids      (1, 1)    int32
///   position_ids   (1,)      int32
///   causal_mask    (1,1,1,ctx) fp16 — additive, −1e4 outside window
///   update_mask    (1,1,ctx,1) fp16 — 1.0 at current position
///   → token_id     (1,)      int32  (in-model argmax)
///   → token_logit  (1,)      fp16
///   state kv_cache_0        fp16
public final class FunctionGemma {

    public struct Config: Sendable {
        public let contextLength: Int
        public let bosTokenId: Int
        public let eosTokenIds: [Int]
        public let functionCallStart: String
        public let functionCallEnd: String

        public init(contextLength: Int = 2048,
                    bosTokenId: Int = 2,
                    eosTokenIds: [Int] = [1, 50],
                    functionCallStart: String = "<start_function_call>",
                    functionCallEnd: String = "<end_function_call>") {
            self.contextLength = contextLength
            self.bosTokenId = bosTokenId
            self.eosTokenIds = eosTokenIds
            self.functionCallStart = functionCallStart
            self.functionCallEnd = functionCallEnd
        }

        static func load(from directory: URL) -> Config {
            let url = directory.appendingPathComponent("model_config.json")
            guard let data = try? Data(contentsOf: url),
                  let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return Config() }
            let ctx = j["context_length"] as? Int ?? 2048
            let bos = j["bos_token_id"] as? Int ?? 2
            let eos: [Int]
            if let list = j["eos_token_id"] as? [Int] { eos = list }
            else if let one = j["eos_token_id"] as? Int { eos = [one] }
            else { eos = [1, 50] }
            let markers = j["function_call_markers"] as? [String: String] ?? [:]
            return Config(
                contextLength: ctx,
                bosTokenId: bos,
                eosTokenIds: eos,
                functionCallStart: markers["start"] ?? "<start_function_call>",
                functionCallEnd: markers["end"] ?? "<end_function_call>")
        }
    }

    public let config: Config
    private let model: MLModel
    private let tokenizer: Tokenizer
    private var state: MLState

    private init(model: MLModel, tokenizer: Tokenizer, config: Config) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.state = model.makeState()
    }

    /// Load a FunctionGemma bundle from disk. Directory layout (produced by
    /// `build_functiongemma_bundle.py`):
    ///   <bundle>/model.mlpackage            or  model.mlmodelc
    ///   <bundle>/model_config.json
    ///   <bundle>/hf_model/tokenizer.json  (+ tokenizer_config.json, chat_template.jinja, ...)
    public static func load(
        bundleURL: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> FunctionGemma {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        let compiled = bundleURL.appendingPathComponent("model.mlmodelc")
        let pkg = bundleURL.appendingPathComponent("model.mlpackage")
        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            modelURL = compiled
        } else if FileManager.default.fileExists(atPath: pkg.path) {
            modelURL = try await MLModel.compileModel(at: pkg)
        } else {
            throw CoreMLLLMError.modelNotFound(
                "no model.mlmodelc or model.mlpackage under \(bundleURL.path)")
        }
        let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)

        let hfDir = bundleURL.appendingPathComponent("hf_model")
        let tokenizer = try await AutoTokenizer.from(modelFolder: hfDir)

        return FunctionGemma(model: model, tokenizer: tokenizer,
                             config: Config.load(from: bundleURL))
    }

    /// One-call convenience for wrapper libraries: download the bundle from
    /// HuggingFace if not already on disk, then load. Returns a ready-to-use
    /// FunctionGemma instance.
    ///
    /// `modelsDir` is the parent directory; the bundle lives at
    /// `<modelsDir>/functiongemma-270m/`. Reuses an existing local bundle
    /// when present (no re-download).
    public static func downloadAndLoad(
        modelsDir: URL,
        hfToken: String? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        onProgress: ((Gemma3BundleDownloader.Progress) -> Void)? = nil
    ) async throws -> FunctionGemma {
        let bundleURL = try await Gemma3BundleDownloader.download(
            .functionGemma270m, into: modelsDir,
            hfToken: hfToken, onProgress: onProgress)
        return try await load(bundleURL: bundleURL, computeUnits: computeUnits)
    }

    /// Reset the stateful KV cache so the next `generate` starts from an
    /// empty context. Call between unrelated prompts.
    public func resetState() {
        state = model.makeState()
    }

    // MARK: - Generation

    /// Generate up to `maxNewTokens` tokens after the prompt. Decoding is
    /// greedy (argmax baked into the model). Stops on any `eos_token_id` or
    /// when `onToken` returns false.
    public func generate(
        prompt: String,
        maxNewTokens: Int = 256,
        resetState: Bool = true,
        onToken: ((String) -> Bool)? = nil
    ) throws -> String {
        if resetState { self.resetState() }

        let promptIds = tokenizer.encode(text: prompt)
        guard !promptIds.isEmpty else { return "" }

        let ctx = config.contextLength
        let eos = Set(config.eosTokenIds)
        var position = 0
        var generatedText = ""

        // Prefill: feed every prompt token through the stateful decoder. The
        // argmax at the LAST prefill token is the first generated token, so
        // capture it as `nextId` and emit it before entering the main loop.
        // (One-token-at-a-time prefill — simpler to ship than a batched
        // prefill chunk. For 270M on ANE it's fast enough; benchmark later.)
        var nextId: Int32 = 0
        for tok in promptIds {
            if position >= ctx { throw CoreMLLLMError.predictionFailed }
            nextId = try predict(tokenId: Int32(tok), position: position)
            position += 1
        }

        // Generation loop — nextId at this point is the prediction for the
        // slot AFTER the last prompt token, which is exactly the first token
        // we want to emit.
        for _ in 0..<maxNewTokens {
            if eos.contains(Int(nextId)) { break }
            let piece = tokenizer.decode(tokens: [Int(nextId)])
            generatedText += piece
            if let cb = onToken, cb(piece) == false { break }
            if position >= ctx { break }

            let fedId = nextId
            nextId = try predict(tokenId: fedId, position: position)
            position += 1
        }

        return generatedText
    }

    /// Run `generate` and return the first `<start_function_call>…<end_function_call>`
    /// payload as a raw string, plus the full generated text. Returns `nil`
    /// for the payload if no function call was emitted.
    public func generateFunctionCall(
        prompt: String,
        maxNewTokens: Int = 256
    ) throws -> (text: String, functionCall: String?) {
        let text = try generate(prompt: prompt, maxNewTokens: maxNewTokens)
        let start = config.functionCallStart
        let end = config.functionCallEnd
        if let s = text.range(of: start), let e = text.range(of: end, range: s.upperBound..<text.endIndex) {
            return (text, String(text[s.upperBound..<e.lowerBound]))
        }
        return (text, nil)
    }

    /// Streaming variant: returns each generated piece as an AsyncStream
    /// element. Useful for SwiftUI views that want to update incrementally.
    /// The stream completes on EOS, on `maxNewTokens`, or if the underlying
    /// prediction throws (in which case the stream finishes without a final
    /// element — wrap in a `do/try await` to surface errors).
    public func stream(
        prompt: String,
        maxNewTokens: Int = 256,
        resetState: Bool = true
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task.detached { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    _ = try self.generate(
                        prompt: prompt,
                        maxNewTokens: maxNewTokens,
                        resetState: resetState,
                        onToken: { piece in
                            continuation.yield(piece)
                            return true
                        })
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Internal prediction

    /// Feed one token at `position`; return the argmax for the NEXT position.
    /// The KV cache is updated in-place (MLState).
    private func predict(tokenId: Int32, position: Int) throws -> Int32 {
        let ctx = config.contextLength

        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
        ids[[0, 0] as [NSNumber]] = NSNumber(value: tokenId)

        let posArr = try MLMultiArray(shape: [1], dataType: .int32)
        posArr[0] = NSNumber(value: Int32(position))

        // Additive causal mask: 0 for positions ≤ position, −1e4 otherwise.
        // fp16 bits: 0.0 = 0x0000, ≈ −10000 = 0xF0E2.
        let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: ctx)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        let negInf: UInt16 = 0xF0E2
        for i in 0..<ctx { mp[i] = i <= position ? 0 : negInf }

        // One-hot write mask — 1.0 at `position`, 0 elsewhere.
        let umask = try MLMultiArray(shape: [1, 1, NSNumber(value: ctx), 1], dataType: .float16)
        let up = umask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        memset(up, 0, ctx * MemoryLayout<UInt16>.stride)
        up[min(position, ctx - 1)] = 0x3C00  // 1.0 in fp16

        let dict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: ids),
            "position_ids": MLFeatureValue(multiArray: posArr),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
        ]
        let output = try model.prediction(
            from: MLDictionaryFeatureProvider(dictionary: dict),
            using: state)
        let out = output.featureValue(for: "token_id")!.multiArrayValue!
        return Int32(truncating: out[0])
    }
}
