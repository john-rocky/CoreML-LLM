import Foundation

/// Model configuration loaded from model_config.json.
public struct ModelConfig: Sendable {
    public let modelName: String
    public let architecture: String
    public let hiddenSize: Int
    public let contextLength: Int
    public let vocabSize: Int
    public let bosTokenId: Int
    public let eosTokenId: Int

    // Gemma 4 specific (defaults match Gemma 4 E2B).
    public let perLayerDim: Int
    public let numLayers: Int
    public let slidingWindow: Int
    public let embedScale: Float
    public let perLayerProjScale: Float
    public let perLayerInputScale: Float
    public let perLayerEmbedScale: Float

    /// Per-layer attention type: "sliding_attention" or "full_attention".
    /// Loaded from model_config.json; falls back to E2B's "every 5th is full"
    /// schedule if the field is absent (old bundles predating E4B support).
    public let layerTypes: [String]
    /// Number of terminal layers that share KV (read from a producer layer
    /// rather than compute their own). E2B: 20, E4B: 18.
    public let numKvSharedLayers: Int

    public static func load(from directory: URL) throws -> ModelConfig {
        let url = directory.appendingPathComponent("model_config.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CoreMLLLMError.configNotFound
        }
        let numLayers = json["num_layers"] as? Int ?? 35
        let defaultLayerTypes = (0..<numLayers).map { i -> String in
            (i + 1) % 5 == 0 ? "full_attention" : "sliding_attention"
        }
        return ModelConfig(
            modelName: json["model_name"] as? String ?? "Model",
            architecture: json["architecture"] as? String ?? "gemma4",
            hiddenSize: json["hidden_size"] as? Int ?? 1536,
            contextLength: json["context_length"] as? Int ?? 2048,
            vocabSize: json["vocab_size"] as? Int ?? 262144,
            bosTokenId: json["bos_token_id"] as? Int ?? 2,
            eosTokenId: json["eos_token_id"] as? Int ?? 1,
            perLayerDim: json["per_layer_dim"] as? Int ?? 256,
            numLayers: numLayers,
            slidingWindow: json["sliding_window"] as? Int ?? 512,
            embedScale: Float(json["embed_scale"] as? Double ?? 39.19),
            perLayerProjScale: Float(json["per_layer_model_projection_scale"] as? Double ?? 0.0255),
            perLayerInputScale: Float(json["per_layer_input_scale"] as? Double ?? 0.707),
            perLayerEmbedScale: Float(json["per_layer_embed_scale"] as? Double ?? 16.0),
            layerTypes: (json["layer_types"] as? [String]) ?? defaultLayerTypes,
            numKvSharedLayers: json["num_kv_shared_layers"] as? Int ?? 20
        )
    }

    // MARK: - Derived layer / chunk helpers (mirror Python Gemma4Config).

    public func isFullAttention(_ layerIdx: Int) -> Bool {
        layerTypes[layerIdx] == "full_attention"
    }

    public func isKvShared(_ layerIdx: Int) -> Bool {
        layerIdx >= (numLayers - numKvSharedLayers)
    }

    /// Last non-shared sliding_attention layer (source of kv13_* aliases).
    public var kvSlidingProducer: Int {
        for i in stride(from: numLayers - 1, through: 0, by: -1) {
            if !isKvShared(i) && layerTypes[i] == "sliding_attention" { return i }
        }
        return -1
    }

    /// Last non-shared full_attention layer (source of kv14_* aliases).
    public var kvFullProducer: Int {
        for i in stride(from: numLayers - 1, through: 0, by: -1) {
            if !isKvShared(i) && layerTypes[i] == "full_attention" { return i }
        }
        return -1
    }

    /// 4 decode/prefill chunk (start, end) ranges. Mirrors
    /// `gemma4_swa_chunks.compute_chunk_boundaries(config)`.
    public var chunkBoundaries: [(start: Int, end: Int)] {
        let n = numLayers
        let ownEnd = kvFullProducer + 1
        let c1End = (ownEnd + 1) / 2
        let c3End = ownEnd + (n - ownEnd) / 2
        return [(0, c1End), (c1End, ownEnd), (ownEnd, c3End), (c3End, n)]
    }
}
