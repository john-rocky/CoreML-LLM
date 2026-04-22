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
    private let prefillModel: MLModel?
    private let prefillT: Int
    private let tokenizer: Tokenizer
    private var state: MLState
    private var prefillState: MLState?

    private init(model: MLModel, prefillModel: MLModel?, prefillT: Int,
                 tokenizer: Tokenizer, config: Config) {
        self.model = model
        self.prefillModel = prefillModel
        self.prefillT = prefillT
        self.tokenizer = tokenizer
        self.config = config
        self.state = model.makeState()
        self.prefillState = prefillModel?.makeState()
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

        // Optional prefill mlpackage (batched T=32 or other). If present, it
        // shares weights with the decode model but has its own MLState; we
        // copy the state buffer after prefill finishes so decode picks up
        // where prefill left off.
        var prefillModel: MLModel? = nil
        var prefillT = 1
        for t in [64, 32, 16, 8] {
            let pcompiled = bundleURL.appendingPathComponent("prefill_t\(t).mlmodelc")
            let ppkg = bundleURL.appendingPathComponent("prefill_t\(t).mlpackage")
            let pURL: URL? = FileManager.default.fileExists(atPath: pcompiled.path) ? pcompiled
                : FileManager.default.fileExists(atPath: ppkg.path) ? ppkg : nil
            if let pURL {
                let compiledURL: URL
                if pURL.pathExtension == "mlpackage" {
                    compiledURL = try await MLModel.compileModel(at: pURL)
                } else {
                    compiledURL = pURL
                }
                prefillModel = try MLModel(contentsOf: compiledURL, configuration: mlConfig)
                prefillT = t
                break
            }
        }

        let hfDir = bundleURL.appendingPathComponent("hf_model")
        let tokenizer = try await AutoTokenizer.from(modelFolder: hfDir)

        return FunctionGemma(
            model: model, prefillModel: prefillModel, prefillT: prefillT,
            tokenizer: tokenizer, config: Config.load(from: bundleURL))
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
        prefillState = prefillModel?.makeState()
    }

    // MARK: - Generation

    /// Chat-templated generation — the recommended entry point.
    ///
    /// FunctionGemma is fine-tuned on its chat format; feeding raw text
    /// through `generate(prompt:)` gives garbage output (e.g. asking for
    /// "primary colors" produces a list of metals). This overload applies
    /// the bundled chat template before decoding, optionally declaring the
    /// tools the model may emit calls for.
    ///
    /// ```swift
    /// let tools: [[String: Any]] = [[
    ///   "type": "function",
    ///   "function": [
    ///     "name": "toggle_flashlight",
    ///     "description": "Turn the flashlight on or off",
    ///     "parameters": ["type": "object", "properties": [:], "required": []]
    ///   ]
    /// ]]
    /// let text = try fg.generate(
    ///   messages: [["role": "user", "content": "Turn on the flashlight"]],
    ///   tools: tools)
    /// // → "<start_function_call>call:toggle_flashlight{}<end_function_call>"
    /// ```
    public func generate(
        messages: [[String: Any]],
        tools: [[String: Any]]? = nil,
        maxNewTokens: Int = 256,
        resetState: Bool = true,
        onToken: ((String) -> Bool)? = nil
    ) throws -> String {
        let ids = try tokenizer.applyChatTemplate(messages: messages, tools: tools)
        return try generate(promptIds: ids, maxNewTokens: maxNewTokens,
                            resetState: resetState, onToken: onToken)
    }

    /// Generate up to `maxNewTokens` tokens after the raw `prompt`. Decoding
    /// is greedy (argmax baked into the model). Stops on any `eos_token_id`
    /// or when `onToken` returns false.
    ///
    /// **Prefer `generate(messages:tools:…)`** — FunctionGemma's training
    /// assumes prompts go through its chat template; the raw-text overload
    /// produces reasonable continuation for completion-style inputs but
    /// pathological output for instruction-style prompts.
    public func generate(
        prompt: String,
        maxNewTokens: Int = 256,
        resetState: Bool = true,
        onToken: ((String) -> Bool)? = nil
    ) throws -> String {
        let promptIds = tokenizer.encode(text: prompt)
        return try generate(promptIds: promptIds, maxNewTokens: maxNewTokens,
                            resetState: resetState, onToken: onToken)
    }

    private func generate(
        promptIds: [Int],
        maxNewTokens: Int,
        resetState: Bool,
        onToken: ((String) -> Bool)?
    ) throws -> String {
        if resetState { self.resetState() }

        guard !promptIds.isEmpty else { return "" }

        let ctx = config.contextLength
        let eos = Set(config.eosTokenIds)
        var position = 0
        var generatedText = ""
        var nextId: Int32 = 0
        var usedPrefill = false

        // 1) Batched prefill if available (T=prefillT tokens per forward).
        if prefillModel != nil, prefillT > 1 {
            while position + prefillT <= promptIds.count,
                  position + prefillT <= ctx {
                let chunk = Array(promptIds[position..<position + prefillT])
                nextId = try prefillBatch(tokens: chunk, startPosition: position)
                position += prefillT
                usedPrefill = true
            }
        }

        // 2) Bridge prefill state → decode state (one memcpy of the KV buffer).
        if usedPrefill {
            try bridgePrefillStateIntoDecode()
        }

        // 3) Any remaining tokens (< T) go through the single-token decode path.
        while position < promptIds.count {
            if position >= ctx { throw CoreMLLLMError.predictionFailed }
            nextId = try predict(tokenId: Int32(promptIds[position]), position: position)
            position += 1
        }

        // 4) Generation loop — nextId is already the first generated token.
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

    /// Run the prefill model once over `prefillT` prompt tokens starting at
    /// `startPosition`. Returns the argmax of the position right after the
    /// last prefilled token (ready to be consumed as the first generated
    /// token if this is the final prefill chunk).
    private func prefillBatch(tokens: [Int], startPosition: Int) throws -> Int32 {
        guard let prefillModel, let prefillState else {
            throw CoreMLLLMError.predictionFailed
        }
        let T = prefillT
        let ctx = config.contextLength
        assert(tokens.count == T, "prefillBatch expects exactly prefillT tokens")

        let ids = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        let ip = ids.dataPointer.bindMemory(to: Int32.self, capacity: T)
        for i in 0..<T { ip[i] = Int32(tokens[i]) }

        let pos = try MLMultiArray(shape: [NSNumber(value: T)], dataType: .int32)
        let pp = pos.dataPointer.bindMemory(to: Int32.self, capacity: T)
        for i in 0..<T { pp[i] = Int32(startPosition + i) }

        // Causal mask (1, 1, T, ctx): row t sees keys 0..startPosition+t.
        let mask = try MLMultiArray(
            shape: [1, 1, NSNumber(value: T), NSNumber(value: ctx)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: T * ctx)
        let negInf: UInt16 = 0xF0E2  // ≈ -10000 in fp16
        for t in 0..<T {
            let writePos = startPosition + t
            let row = mp.advanced(by: t * ctx)
            for c in 0..<ctx { row[c] = c <= writePos ? 0 : negInf }
        }

        // Update mask (1, 1, ctx, T): column t has a single 1.0 at row startPosition+t.
        let umask = try MLMultiArray(
            shape: [1, 1, NSNumber(value: ctx), NSNumber(value: T)], dataType: .float16)
        let up = umask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx * T)
        memset(up, 0, ctx * T * MemoryLayout<UInt16>.stride)
        for t in 0..<T {
            let c = startPosition + t
            if c < ctx { up[c * T + t] = 0x3C00 }  // 1.0 in fp16
        }

        let dict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: ids),
            "position_ids": MLFeatureValue(multiArray: pos),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
        ]
        let output = try prefillModel.prediction(
            from: MLDictionaryFeatureProvider(dictionary: dict),
            using: prefillState)
        let out = output.featureValue(for: "token_id")!.multiArrayValue!
        return Int32(truncating: out[0])
    }

    /// Copy the prefill model's KV cache (MLState buffer) byte-wise into
    /// the decode model's KV cache. Both states are shaped identically
    /// (2·L × kv_heads × ctx × head_dim, fp16) because they're produced by
    /// two traces of the same underlying Gemma3Model.
    ///
    /// Uses the NS_REFINED_FOR_SWIFT `withMultiArray(for:handler:)` bridge —
    /// the handler closure is the only legal scope to access the underlying
    /// buffer (Apple's contract — the backing address can differ between
    /// calls).
    private func bridgePrefillStateIntoDecode() throws {
        guard let prefillState else { return }
        prefillState.withMultiArray(for: "kv_cache_0") { src in
            state.withMultiArray(for: "kv_cache_0") { dst in
                guard src.count == dst.count else { return }
                memcpy(dst.dataPointer, src.dataPointer, src.count * 2)
            }
        }
    }

    /// Run `generate(messages:tools:)` and return the first
    /// `<start_function_call>…<end_function_call>` payload as a raw string,
    /// plus the full generated text. Returns `nil` for the payload if no
    /// function call was emitted.
    public func generateFunctionCall(
        messages: [[String: Any]],
        tools: [[String: Any]]? = nil,
        maxNewTokens: Int = 256
    ) throws -> (text: String, functionCall: String?) {
        let text = try generate(messages: messages, tools: tools,
                                maxNewTokens: maxNewTokens)
        return (text, extractFunctionCall(from: text))
    }

    /// Convenience wrapper that takes a single user utterance.
    public func generateFunctionCall(
        userPrompt: String,
        tools: [[String: Any]]? = nil,
        maxNewTokens: Int = 256
    ) throws -> (text: String, functionCall: String?) {
        try generateFunctionCall(
            messages: [["role": "user", "content": userPrompt]],
            tools: tools, maxNewTokens: maxNewTokens)
    }

    /// Extract the first `<start_function_call>…<end_function_call>` payload
    /// from `text`, if present.
    public func extractFunctionCall(from text: String) -> String? {
        let start = config.functionCallStart
        let end = config.functionCallEnd
        guard let s = text.range(of: start),
              let e = text.range(of: end, range: s.upperBound..<text.endIndex)
        else { return nil }
        return String(text[s.upperBound..<e.lowerBound])
    }

    /// Streaming variant for chat-templated generation — the recommended
    /// entry point for SwiftUI views.
    public func stream(
        messages: [[String: Any]],
        tools: [[String: Any]]? = nil,
        maxNewTokens: Int = 256,
        resetState: Bool = true
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task.detached { [weak self] in
                guard let self else { continuation.finish(); return }
                do {
                    _ = try self.generate(
                        messages: messages,
                        tools: tools,
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

    /// Streaming variant for raw-text prompts. Prefer
    /// `stream(messages:tools:)` — see the note on `generate(prompt:)`.
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
