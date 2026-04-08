# CoreML-LLM

Run LLMs on Apple devices with CoreML, optimized for Apple Neural Engine + GPU.

## Features

- **ANE-optimized**: RMSNorm, Conv2d linear layers, in-model argmax — all tuned for Neural Engine
- **Stateful KV cache**: Uses Apple's MLState API (iOS 18+) for efficient autoregressive generation
- **3-part model splitting**: Embed / Transformer / LM Head — fits iOS memory constraints
- **Simple API**: 4 methods — `load()`, `generate()`, `chat()`, `reset()`
- **Swift Package**: Uses [swift-transformers](https://github.com/huggingface/swift-transformers) for tokenization and chat templates
- **Int4 quantization**: Block-wise palettization for minimal quality loss

## Supported Models

| Model | Parameters | Status |
|-------|-----------|--------|
| Qwen2.5-0.5B-Instruct | 0.5B | Available |
| Qwen2.5-1.5B-Instruct | 1.5B | Planned |
| Qwen3-0.6B | 0.6B | Planned |
| SmolLM2-1.7B | 1.7B | Planned |

## Quick Start

### 1. Convert a Model

```bash
cd conversion
pip install -r requirements.txt
python convert.py --model qwen2.5-0.5b --output ./output/
```

### 2. Swift Package

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/john-rocky/CoreML-LLM", from: "0.1.0"),
]
```

### 3. Use in Your App

```swift
import CoreMLLLM

// Load model
let llm = try await CoreMLLLM.load(from: modelDirectory)

// Generate text
let response = try await llm.generate("Explain quantum computing in one sentence.")
print(response)

// Stream tokens
let response = try await llm.generate("Hello!") { token in
    print(token, terminator: "")
    return true // return false to stop
}

// Chat
let response = try await llm.chat(messages: [
    ["role": "user", "content": "What is Swift?"]
])

// Reset for new conversation
llm.reset()

// Check performance
if let bench = llm.lastBenchmark {
    print(bench.summary) // "42 tokens in 1.23s (decode: 34.1 tok/s)"
}
```

## Architecture

```
Python Conversion Pipeline          Swift Inference Engine
┌─────────────────────┐            ┌──────────────────────┐
│ HuggingFace Model   │            │ CoreMLLLM            │
│         │           │            │   ├── load()         │
│    ANE Optimize     │            │   ├── generate()     │
│   (RMSNorm, Conv2d) │            │   ├── chat()         │
│         │           │            │   └── reset()        │
│   torch.jit.trace   │            │         │            │
│         │           │  ┌──────┐  │   LLMModel           │
│   ct.convert +      │──│.mlpkg│──│   (3-part loader)    │
│   StateType (KV)    │  └──────┘  │         │            │
│         │           │            │   InferenceEngine    │
│   Int4 Quantize     │            │   (prefill + decode) │
└─────────────────────┘            └──────────────────────┘
```

## ANE Optimizations

Techniques adapted from [ANEMLL](https://github.com/Anemll/Anemll):

- **RMSNorm**: `cat([x, -x])` -> `LayerNorm` -> slice. Uses ANE's optimized LayerNorm kernel.
- **Conv2d Linear**: All `nn.Linear` -> `nn.Conv2d(kernel_size=1)`. ANE processes Conv2d natively.
- **In-Model Argmax**: Computes argmax inside the CoreML graph. Avoids transferring 150K+ logits from ANE to CPU.
- **Stateful KV Cache**: Single unified buffer via `MLState`. 13x faster than I/O approach (Apple benchmark).

## Requirements

- **Conversion**: Python 3.10+, coremltools 8+, PyTorch 2.2+
- **Inference**: iOS 18+ / macOS 15+, Xcode 16+

## License

MIT
