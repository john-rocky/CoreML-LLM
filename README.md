# CoreML-LLM

**On-device multimodal AI** for Apple devices — text and image understanding powered by CoreML with Apple Neural Engine + GPU optimization.

Run **Gemma 4** vision-language model entirely on iPhone: point your camera at anything, ask questions about it, get answers — no server, no internet required.

<!-- TODO: Add demo GIF/video here after device testing -->

## Pre-converted Models (Download & Run)

| Model | Size | Multimodal | Download |
|-------|------|------------|----------|
| **Gemma 4 E2B** | 2.7 GB | **Image + Text** | [HuggingFace](https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml) |
| Qwen2.5-0.5B | 302 MB | Text only | [HuggingFace](https://huggingface.co/mlboydaisuke/qwen2.5-0.5b-coreml) |

The iOS app downloads these automatically. Or convert your own from any HuggingFace model.

### Verified Results

| Prompt | Gemma 4 E2B | Qwen2.5 0.5B |
|--------|-------------|--------------|
| "What is the capital of France?" | "The capital of France is **Paris**." ✅ | "The capital of France is Paris." ✅ |
| 🖼️ Red square on white background | "**solid red square** with a white background" ✅ | — |

## Quick Start

### 1. Convert a Model

```bash
cd conversion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install scikit-learn

# Qwen2.5-0.5B (~2 min)
python convert.py --model qwen2.5-0.5b --output ./output/qwen2.5-0.5b

# Gemma 4 E2B (~15 min, 10GB download)
python convert.py --model gemma4-e2b --output ./output/gemma4-e2b

# List available models
python convert.py --list
```

### 2. Verify

```python
import coremltools as ct
import numpy as np

model = ct.models.MLModel('./output/qwen2.5-0.5b/model.mlpackage')
state = model.make_state()

# Single prediction
out = model.predict({
    'input_ids': np.array([[818]], dtype=np.int32),
    'position_ids': np.array([0], dtype=np.int32),
    'causal_mask': np.zeros((1,1,1,2048), dtype=np.float16),
    'update_mask': np.array([[[[1]]+[[0]]*2047]], dtype=np.float16).reshape(1,1,2048,1),
}, state=state)

print(out['token_id'])  # Next token prediction
```

### 3. iOS App

```bash
open Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj
```

Set your development team → Build to device (iOS 18+) → "Get Model" → Download → Chat.

For Gemma 4 multimodal: tap 📷 to attach a photo, then ask about it.

## How It Works

```
                         CoreML Models                    iPhone/Mac
                    ┌─────────────────────┐
  Image ──────────► │  Vision Encoder     │──► Image Features
                    │  (322 MB)           │         │
                    └─────────────────────┘         │
                    ┌─────────────────────┐         ▼
  Text  ──────────► │  Decoder + KV Cache │──► Token Predictions
                    │  (302 MB - 2.4 GB)  │    (streaming)
                    └─────────────────────┘
```

### ANE Optimizations

| Technique | What | Why |
|-----------|------|-----|
| ANERMSNorm | `cat([x,-x])` → LayerNorm → slice | ANE has optimized LayerNorm kernel; standard RMSNorm is slow |
| Conv2d Linear | `nn.Linear` → `nn.Conv2d(1)` | ANE processes Conv2d natively |
| In-Model Argmax | Argmax inside CoreML graph | Avoids transferring 150K+ logits from ANE to CPU |
| Stateful KV Cache | `MLState` API (iOS 18+) | 13x faster than passing KV as input/output tensors |
| Mask-Based Update | `cache*(1-mask) + val*mask` | Trace-compatible cache writes (no dynamic indexing) |

### Why Not MLX?

| | CoreML-LLM (this) | MLX Swift |
|---|---|---|
| Hardware | **ANE + GPU** | GPU only |
| Power | **~2W** | ~20W |
| Memory (8B) | **~500 MB** | ~8 GB |
| Speed | Moderate | 2-5x faster |
| Use case | **iPhone, always-on, battery** | Mac, max throughput |

## Adding New Models

See [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) for a step-by-step guide.

Key architecture considerations documented from real debugging:
- Attention scale: some models use `1.0` (QK norm), not `1/sqrt(d)` — getting this wrong produces coherent but completely incorrect text
- KV sharing, v_norm, per-layer embeddings, dual RoPE
- Full precision debugging methodology

See [docs/CONVERSION.md](docs/CONVERSION.md) for the full conversion reference.

## Project Structure

```
CoreML-LLM/
├── Package.swift                          # Swift Package
├── Sources/CoreMLLLM/                     # Swift inference library
│   ├── CoreMLLLM.swift                    #   Public API
│   ├── LLMModel.swift                     #   CoreML model management
│   ├── InferenceEngine.swift              #   Prefill + decode loop
│   └── ...
├── conversion/                            # Python conversion pipeline
│   ├── convert.py                         #   CLI entry point
│   ├── ane_ops.py                         #   ANE-optimized operations
│   ├── base_model.py                      #   Abstract transformer
│   ├── exporter.py                        #   CoreML export + quantize
│   └── models/
│       ├── qwen2.py                       #   Qwen2/2.5
│       ├── gemma4.py                      #   Gemma 4 E2B
│       ├── gemma4_wrapper.py              #   Gemma 4 monolithic wrapper
│       └── ...
├── Examples/CoreMLLLMChat/                # iOS sample app
│   └── CoreMLLLMChat.xcodeproj
└── docs/
    ├── CONVERSION.md                      # Conversion guide + pitfalls
    └── ADDING_MODELS.md                   # How to add new models
```

## Requirements

- **Conversion**: Python 3.10-3.12, coremltools 8+, PyTorch 2.2+
- **Inference**: iOS 18+ / macOS 15+
- **Sample App**: Xcode 16+

## License

MIT
