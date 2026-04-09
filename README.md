# CoreML-LLM

**On-device LLMs on Apple Neural Engine** — Run **Gemma 4** on iPhone with CoreML, ANE-optimized, no server.

Unlike MLX Swift (GPU-only), CoreML-LLM targets the **Apple Neural Engine**: ~10x lower power for always-on, battery-friendly inference.

> **v0.2.0** — Sliding Window Attention, batched prefill (seq=64), PLE on ANE, 2048 context. See [What's new](#whats-new-in-v020).

<video src="https://github.com/john-rocky/CoreML-LLM/releases/download/v0.2.0/CoreMLLLM.mp4" width="360" controls></video>

## Performance (Gemma 4 E2B, iPhone 15 Pro)

| | v0.1.0 | **v0.2.0** |
|---|---:|---:|
| Context length | 512 | **2048** |
| Decode speed | ~11 tok/s | ~11.5 tok/s |
| Prefill (33 tokens) | ~2970 ms (per-token) | **188 ms** (batched, 15.8×) |
| Prefill throughput | ~11 tok/s | **~175 tok/s effective** |
| Memory footprint | ~250 MB | ~250 MB |
| Compute unit | ANE | ANE |

Mac (M-series, ANE): ~25 tok/s decode. Power draw ~2 W vs ~20 W for MLX GPU path on the same model.

## Pre-converted Models

| Model | Size | Multimodal | Download |
|-------|------|------------|----------|
| **Gemma 4 E2B** | 2.7 GB | Image + Text* | [HuggingFace](https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml) |
| Qwen2.5-0.5B | 302 MB | Text only | [HuggingFace](https://huggingface.co/mlboydaisuke/qwen2.5-0.5b-coreml) |

<sub>* Gemma 4 image understanding is present but preprocessing accuracy is being finalized for v0.3. Text inference is production-ready.</sub>

The iOS sample app downloads models automatically. You can also convert your own.

## Quick Start

### iOS App

```bash
open Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj
```

Set your development team → build to an iOS 18+ device → **Get Model** → Download → Chat.

The app uses `.cpuAndNeuralEngine` to force ANE execution.

### Convert a Model

```bash
cd conversion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install scikit-learn

# Qwen2.5-0.5B (~2 min)
python convert.py --model qwen2.5-0.5b --output ./output/qwen2.5-0.5b

# Gemma 4 E2B (~15 min, 10 GB download)
python convert.py --model gemma4-e2b --output ./output/gemma4-e2b

# List available models
python convert.py --list
```

### Swift Package

```swift
dependencies: [
    .package(url: "https://github.com/john-rocky/CoreML-LLM", from: "0.2.0"),
]
```

```swift
import CoreMLLLM

let llm = try await CoreMLLLM.load(from: modelDirectory)
let answer = try await llm.generate("What is the capital of France?")
// → "The capital of France is **Paris**."

for await token in try await llm.stream("Tell me a story") {
    print(token, terminator: "")
}
```

## What's new in v0.2.0

### Sliding Window Attention (SWA)
Gemma 4 ships with 28 sliding-window layers (W=512) and 7 full-attention layers. v0.2.0 exploits this natively: 28/35 layers are now O(W) instead of O(context), so decode speed stays flat as context grows from 512 → 2048.

- Sliding KV cache: `(num_slots, 1, W, max_hd)` with shift-based update.
- Full KV cache: `(num_slots, 1, ctx, max_hd)` with mask-based update.
- Two causal masks per step: `(1, 1, 1, W)` sliding + `(1, 1, 1, ctx)` full.

### Batched Prefill (seq=64)
A separate set of prefill chunks processes up to 64 tokens in a single CoreML call, then writes K/V back into the persistent decode caches. TTFT on a 33-token prompt drops from ~3 s to 188 ms on iPhone 15 Pro.

Implementation: 4 prefill chunks mirror the 4 decode chunks (L0-7 / L8-14 / L15-24 / L25-34). Prefill chunk 4 selects the real last token via a masked sum (`hidden_states × last_position_mask`), avoiding dynamic indexing (ANE-friendly).

Fallback: prompts >64 tokens or with images use the existing per-token decode path.

### PLE on ANE
Gemma 4's Per-Layer Embedding (8960×1536 projection + 35 RMSNorm slices) was 8 ms/token on CPU BLAS in v0.1.0. Moved inside the CoreML graph (Conv2d + LayerNorm) → **1.8 ms/token** on ANE.

### Context length 2048
Stateless KV cache with explicit I/O (no `MLState`) to avoid per-layer state registration overhead. Combined with SWA, context extension is nearly free.

## Architecture

```
      Prompt  ─┐
               ▼
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │ Prefill ch1  │─► │ Prefill ch2  │─► │ Prefill ch3  │─► │ Prefill ch4  │─► first token
        │ L0-7  + PLE  │   │ L8-14, kv13/ │   │ L15-24 shared│   │ L25-34 + LM  │
        └──────────────┘   │  kv14 out    │   └──────────────┘   └──────────────┘
               │           └──────────────┘          ▲                  ▲
               │                  │                  │                  │
               │                  └────────────┬─────┴──────────────────┘
               │                                          kv13_k/v, kv14_k/v
               ▼                                              (shared)
          writes K/V to persistent SWA caches
                      │
                      ▼  (decode loop, 1 token per step)
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │  Decode ch1  │─► │  Decode ch2  │─► │  Decode ch3  │─► │  Decode ch4  │─► next token
        └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### ANE Optimizations

| Technique | What | Why |
|-----------|------|-----|
| ANERMSNorm | `cat([x,-x])` → LayerNorm → slice | ANE has optimized LayerNorm; bare RMSNorm is slow |
| Conv2d Linear | `nn.Linear` → `nn.Conv2d(1)` | ANE executes Conv2d ~3× faster than MatMul |
| In-Model Argmax | Argmax inside the CoreML graph | Avoids shipping 256K logits from ANE to CPU |
| Manual softmax | `max/sub/exp/sum/div` with explicit fp16 casts | Prevents PyTorch's silent fp16→fp32 upcast in `torch.exp` |
| Pre-computed RoPE | cos/sin as model inputs, looked up in Swift | Eliminates `gather`/`greater_equal` (int ops → CPU) |
| Explicit KV I/O | Plain tensor inputs/outputs, no `MLState` | Avoids int64 state indices that break ANE placement |
| Sliding Window | Shift-based cache for 28/35 layers | O(W) per step instead of O(ctx) |
| Batched Prefill | One CoreML call for 64 tokens | 15× faster TTFT vs per-token |
| PLE in graph | Conv2d projection + per-layer norm | 8 ms → 1.8 ms/token |

### Why Not MLX?

| | CoreML-LLM (this) | MLX Swift |
|---|---|---|
| Hardware | **ANE + CPU fallback** | GPU only |
| Power (iPhone) | **~2 W** | ~20 W |
| Always-on friendly | **Yes** | No (thermal) |
| Peak speed | ~11 tok/s (E2B) | ~30 tok/s (E2B) |
| Use case | **Always-on, battery, background** | Max throughput on plugged-in Mac |

ANE and GPU are complementary. CoreML-LLM is the answer when you want an LLM running continuously on a phone without melting it.

## Adding New Models

See [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) for a step-by-step guide and [docs/CONVERSION.md](docs/CONVERSION.md) for the full conversion reference.

Known gotchas documented from real debugging:
- Attention scale: some models use `1.0` (QK-norm), not `1/√d` — wrong value produces coherent but incorrect text
- KV sharing, `v_norm`, per-layer embeddings, dual (sliding/full) RoPE
- PyTorch auto-promotes fp16 → fp32 inside `torch.exp`; force explicit `.to(fp16)` after softmax ops
- Any int64/int32 op in the graph (gather, greater_equal, etc.) will pull the whole block onto CPU — pre-compute constants on the host side

## Project Structure

```
CoreML-LLM/
├── Package.swift
├── Sources/CoreMLLLM/
│   ├── CoreMLLLM.swift                  # Public API
│   ├── ModelConfig.swift
│   └── ImageProcessor.swift
├── conversion/                          # Python conversion pipeline
│   ├── convert.py                       # CLI entry point
│   ├── ane_ops.py                       # ANE-optimized ops (RMSNorm, softmax, RoPE)
│   ├── base_model.py
│   ├── exporter.py                      # CoreML export + INT4 palettization
│   └── models/
│       ├── qwen2.py
│       ├── gemma4.py                    # Gemma 4 E2B base model
│       ├── gemma4_swa_chunks.py         # 4-chunk decode (SWA)
│       ├── gemma4_prefill_chunks.py     # 4-chunk prefill (seq=64)
│       └── gemma4_vision.py
├── Examples/CoreMLLLMChat/              # iOS sample app
│   └── CoreMLLLMChat.xcodeproj
└── docs/
    ├── CONVERSION.md
    └── ADDING_MODELS.md
```

## Requirements

- **Conversion**: Python 3.10-3.12, coremltools 8+, PyTorch 2.2+
- **Inference**: iOS 18+ / macOS 15+ (MLState-free stateless path)
- **Sample app**: Xcode 16+

## License

MIT for the CoreML-LLM code. Gemma 4 weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
