import XCTest
@testable import CoreMLLLM

final class CoreMLLLMTests: XCTestCase {
    func testConfigurationDecoding() throws {
        let json = """
        {
            "model_name": "qwen2.5-0.5b",
            "architecture": "qwen2",
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 151936,
            "context_length": 2048,
            "rms_norm_eps": 1e-6,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "quantization": "int4",
            "compute_units": "ALL",
            "parts": {
                "embed": "embed.mlpackage",
                "transformer": "transformer.mlpackage",
                "lmhead": "lmhead.mlpackage"
            },
            "tokenizer_repo": "Qwen/Qwen2.5-0.5B-Instruct"
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(LLMModelConfiguration.self, from: json)

        XCTAssertEqual(config.modelName, "qwen2.5-0.5b")
        XCTAssertEqual(config.hiddenSize, 896)
        XCTAssertEqual(config.numAttentionHeads, 14)
        XCTAssertEqual(config.numKeyValueHeads, 2)
        XCTAssertEqual(config.headDim, 64)
        XCTAssertEqual(config.contextLength, 2048)
        XCTAssertEqual(config.parts.embed, "embed.mlpackage")
        XCTAssertEqual(config.tokenizerRepo, "Qwen/Qwen2.5-0.5B-Instruct")
    }
}
