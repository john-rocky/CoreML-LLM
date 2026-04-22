import CoreML
import Foundation
import Tokenizers

/// Minimal runtime for EmbeddingGemma-300M (Gemma 3 bidirectional encoder +
/// mean pool + two dense + L2 normalize).
///
/// Intentionally narrow API — just load + encode. LocalAIKit (or any other
/// wrapper) is expected to handle retrieval orchestration, batching across
/// a document corpus, cosine ranking, etc. No dependency on the Gemma 4
/// stack or the FunctionGemma runner.
///
/// I/O contract of the underlying mlpackage (from
/// `conversion/build_embeddinggemma_bundle.py`):
///   input_ids       (1, L)  int32
///   attention_mask  (1, L)  fp16  (1.0 valid, 0.0 pad)
///   → embedding     (1, 768) fp16 — L2 unit norm
public final class EmbeddingGemma {

    public struct Config: Sendable {
        public let maxSeqLen: Int
        public let embedDim: Int
        public let matryoshkaDims: [Int]
        public let taskPrefixes: [String: String]

        public init(maxSeqLen: Int = 512,
                    embedDim: Int = 768,
                    matryoshkaDims: [Int] = [768, 512, 256, 128],
                    taskPrefixes: [String: String] = [:]) {
            self.maxSeqLen = maxSeqLen
            self.embedDim = embedDim
            self.matryoshkaDims = matryoshkaDims
            self.taskPrefixes = taskPrefixes
        }

        static func load(from directory: URL) -> Config {
            let url = directory.appendingPathComponent("model_config.json")
            guard let data = try? Data(contentsOf: url),
                  let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return Config() }
            return Config(
                maxSeqLen: j["max_seq_len"] as? Int ?? 512,
                embedDim: j["embed_dim"] as? Int ?? 768,
                matryoshkaDims: j["matryoshka_dims"] as? [Int] ?? [768, 512, 256, 128],
                taskPrefixes: j["task_prefixes"] as? [String: String] ?? [:])
        }
    }

    /// EmbeddingGemma's published task-prefix taxonomy (HF model card). Prepend
    /// the relevant prefix to the input text before tokenization. Values come
    /// from model_config.json; these keys are convenience aliases.
    public enum Task: String, Sendable {
        case retrievalQuery = "retrieval_query"
        case retrievalDocument = "retrieval_document"
        case classification
        case clustering
        case similarity
        case codeRetrieval = "code_retrieval"
        case questionAnswering = "question_answering"
        case factVerification = "fact_verification"
    }

    public let config: Config
    private let model: MLModel
    private let tokenizer: Tokenizer

    private init(model: MLModel, tokenizer: Tokenizer, config: Config) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
    }

    /// One-call convenience: download from HuggingFace if needed, then load.
    /// Wrapper-friendly entry point — pass an app-support directory and let
    /// the package handle caching, then the returned instance is ready for
    /// `encode(text:)`.
    public static func downloadAndLoad(
        modelsDir: URL,
        hfToken: String? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        onProgress: ((Gemma3BundleDownloader.Progress) -> Void)? = nil
    ) async throws -> EmbeddingGemma {
        let bundleURL = try await Gemma3BundleDownloader.download(
            .embeddingGemma300m, into: modelsDir,
            hfToken: hfToken, onProgress: onProgress)
        return try await load(bundleURL: bundleURL, computeUnits: computeUnits)
    }

    /// Load an EmbeddingGemma bundle. Directory layout:
    ///   <bundle>/encoder.mlpackage  or  encoder.mlmodelc
    ///   <bundle>/model_config.json
    ///   <bundle>/hf_model/tokenizer.json  (+ tokenizer_config.json, ...)
    public static func load(
        bundleURL: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> EmbeddingGemma {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        let compiled = bundleURL.appendingPathComponent("encoder.mlmodelc")
        let pkg = bundleURL.appendingPathComponent("encoder.mlpackage")
        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            modelURL = compiled
        } else if FileManager.default.fileExists(atPath: pkg.path) {
            modelURL = try await MLModel.compileModel(at: pkg)
        } else {
            throw CoreMLLLMError.modelNotFound(
                "no encoder.mlmodelc or encoder.mlpackage under \(bundleURL.path)")
        }
        let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)

        let hfDir = bundleURL.appendingPathComponent("hf_model")
        let tokenizer = try await AutoTokenizer.from(modelFolder: hfDir)

        return EmbeddingGemma(model: model, tokenizer: tokenizer,
                              config: Config.load(from: bundleURL))
    }

    // MARK: - Encoding

    /// Encode `text` into a unit-norm vector. If `task` is provided, its
    /// task prefix (per the HF model card / model_config.json) is prepended
    /// to the input before tokenization.
    ///
    /// `dim` may be `nil` (full 768-d output), or one of the Matryoshka
    /// truncation sizes (512, 256, 128). Non-default dims trigger a
    /// post-hoc truncate + L2-renormalize step.
    public func encode(
        text: String,
        task: Task? = nil,
        dim: Int? = nil
    ) throws -> [Float] {
        let prefixed: String
        if let task, let p = config.taskPrefixes[task.rawValue] {
            prefixed = p + text
        } else {
            prefixed = text
        }

        var ids = tokenizer.encode(text: prefixed)
        let L = config.maxSeqLen
        if ids.count > L { ids = Array(ids.prefix(L)) }

        let inputIds = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .int32)
        let ip = inputIds.dataPointer.bindMemory(to: Int32.self, capacity: L)
        for i in 0..<L { ip[i] = i < ids.count ? Int32(ids[i]) : 0 }

        let attn = try MLMultiArray(shape: [1, NSNumber(value: L)], dataType: .float16)
        let ap = attn.dataPointer.bindMemory(to: UInt16.self, capacity: L)
        let one: UInt16 = 0x3C00  // 1.0 in fp16
        for i in 0..<L { ap[i] = i < ids.count ? one : 0 }

        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attn),
        ]))

        guard let arr = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }

        // Read (1, embed_dim) fp16 → [Float]. arr.dataType is .float16;
        // Float16 is a native type on iOS 14+ / macOS 11+, so the bit layout
        // matches and we can read it directly.
        let D = arr.count
        var vec = [Float](repeating: 0, count: D)
        let src = arr.dataPointer.bindMemory(to: Float16.self, capacity: D)
        for i in 0..<D { vec[i] = Float(src[i]) }

        guard let targetDim = dim, targetDim < D else { return vec }

        // Matryoshka truncate + renormalize.
        var truncated = Array(vec.prefix(targetDim))
        var normSq: Float = 0
        for v in truncated { normSq += v * v }
        let norm = sqrtf(max(normSq, 1e-20))
        for i in 0..<truncated.count { truncated[i] /= norm }
        return truncated
    }
}

