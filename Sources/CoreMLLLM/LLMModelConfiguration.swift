import Foundation

/// Model configuration loaded from model_config.json.
/// Written by the Python conversion pipeline.
public struct LLMModelConfiguration: Codable, Sendable {
    public let modelName: String
    public let architecture: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let vocabSize: Int
    public let contextLength: Int
    public let rmsNormEps: Float
    public let bosTokenId: Int
    public let eosTokenId: Int
    public let quantization: String
    public let computeUnits: String
    public let parts: ModelParts
    public let tokenizerRepo: String

    public struct ModelParts: Codable, Sendable {
        /// Monolithic model file (embed+transformer+lmhead in one)
        public let model: String?
        // Legacy 3-part split (kept for compatibility)
        public let embed: String?
        public let transformer: String?
        public let lmhead: String?

        public var isMonolithic: Bool { model != nil }
    }

    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case architecture
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabSize = "vocab_size"
        case contextLength = "context_length"
        case rmsNormEps = "rms_norm_eps"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case quantization
        case computeUnits = "compute_units"
        case parts
        case tokenizerRepo = "tokenizer_repo"
    }

    /// Load configuration from a JSON file.
    public static func load(from url: URL) throws -> LLMModelConfiguration {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(LLMModelConfiguration.self, from: data)
    }
}
