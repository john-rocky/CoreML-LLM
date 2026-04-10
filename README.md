# CoreML-LLM

**On-device LLMs on Apple Neural Engine** — Run **Gemma 4** on iPhone with CoreML, ANE-optimized, no server.

CoreML-LLM targets the **Apple Neural Engine** rather than the GPU, making it a good fit for always-on, battery-friendly inference. [MLX Swift](https://github.com/ml-explore/mlx-swift) is the best choice when you want maximum throughput from the GPU; CoreML-LLM is the answer when you want the LLM to live on the ANE so the GPU stays free.

> **v0.3.0** — Correct prefill (fp16 overflow fix), ~2.5× faster decode, N=512 batched prefill, 99.78% ops on ANE verified via `MLComputePlan`. See [What's new](#whats-new-in-v030).

<video src="https://github.com/user-attachments/assets/4f749080-eef1-4728-a2e9-4784afb44e80" width="360"></video>

## Performance (Gemma 4 E2B, iPhone 15 Pro)

| | v0.1.0 | v0.2.0 | **v0.3.0** |
|---|---:|---:|---:|
| Context length | 512 | 2048 | **2048** |
| Decode speed | ~11 tok/s | ~11 tok/s | **~28 tok/s** |
| Prefill (40 tokens) | ~3.6 s | ~220 ms | **~415 ms (96 tok/s eff.)** |
| Batched prefill window | — | 64 tokens | **512 tokens** |
| ANE placement (dispatched ops) | — | — | **99.78%** |
| Compute unit | ANE | ANE | ANE |

Ground-truth ANE placement measured on iPhone 15 Pro via `MLComputePlan` (7,294 of 7,310 dispatched LLM ops on ANE; the remaining 16 CPU ops are the tail argmax in chunk4 / prefill_chunk4). Vision encoder runs on GPU by design.

> **Power draw:** we don't publish a specific wattage yet. iOS's public `batteryLevel` API is too coarse (~5% granularity) for a clean short-run measurement. What we can say: the device stays at `ProcessInfo.thermalState == .fair` through 10 minutes of sustained generation, so the draw is clearly modest — but a specific number will wait until we have a USB-C power meter or 24h Settings → Battery data.

## Pre-converted Models

| Model | Size | Multimodal | Download |
|-------|------|------------|----------|
| **Gemma 4 E2B** | 2.7 GB | Image + Text* | [HuggingFace](https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml) |
| Qwen2.5-0.5B | 302 MB | Text only | [HuggingFace](https://huggingface.co/mlboydaisuke/qwen2.5-0.5b-coreml) |

<sub>* Gemma 4 image understanding is present but image preprocessing / vision-encoder alignment is still being tuned. Text inference is production-ready.</sub>

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
    .package(url: "https://github.com/john-rocky/CoreML-LLM", from: "0.3.0"),
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

## What's new in v0.3.0

### Prefill correctness fix (fp16 overflow)
v0.2.0's batched prefill produced wrong hidden states on iPhone: the `q_norm` weights were pre-scaled by `sqrt(head_dim)` in order to enable fused `scaled_dot_product_attention` (which always divides by `sqrt(d)`). In fp16 that pre-scaling overflowed inside `Q @ K^T` at prefill N ≥ 64, and the model emitted `<turn|>` immediately after the prompt.

Fix: reverted to manual attention (`matmul → add → softmax → matmul`) with scale=1.0, matching Gemma 4's effective attention scale after `q_norm` / `k_norm` unit-normalise Q and K. Manual attention loses SDPA fusion but keeps the graph on the ANE and — surprisingly — is also measurably faster on iPhone (likely because the previous path was partially falling back to CPU when the overflow tripped an ANE compile constraint).

Verified on Mac against the HuggingFace reference implementation, and on iPhone against `MLComputePlan`.

### Decode ~2.5× faster
As a side effect of the prefill fix and the associated chunk rebuild, decode went from ~11 tok/s to **~28 tok/s** on iPhone 15 Pro, sustained over a 10-minute benchmark run.

### Prefill window 64 → 512
The batched prefill path now covers the first 512 tokens in a single CoreML call (was 64). Multimodal prompts (≈ 280 image placeholders + surrounding text) now fit in a single prefill pass instead of falling back to per-token decode.

### ANE placement verification (`MLComputePlan`)
New "ANE?" debug button in the sample app calls `MLComputePlan.load(contentsOf:)` for every loaded chunk and walks `MLModelStructure.Program.Block.operations`, bucketing each op by its preferred `MLComputeDevice`. On iPhone 15 Pro the LLM chunks report **7,294 / 7,310 dispatched ops on the ANE (99.78%)**; the 16 CPU ops are the tail argmax in `chunk4` / `prefill_chunk4`.

The denominator excludes `constexpr_affine_dequantize` / `constexpr_lut_to_dense` (INT4 palette expansion) and other compile-time ops that `deviceUsage(for:)` correctly reports as `nil` — those don't dispatch at runtime and shouldn't appear in "X% on ANE".

### Battery benchmark mode
New "Bench" menu in the sample app runs sustained generation for 5 / 10 / 30 minutes against a fixed prompt, recording `UIDevice.batteryLevel` and `ProcessInfo.thermalState` start / end. Aborts automatically if thermal state reaches `.serious` so the device doesn't cook. Honest note: iOS's public battery API is too coarse to turn a 10-minute run into a precise wattage, which is why we don't publish a W number yet.

### Multimodal token count fix
The sample app was inserting 280 `<|image|>` placeholders per image (matching the vision encoder's output tensor shape). For square 768×768 inputs only the first 256 soft tokens are real — the encoder zero-pads the remaining 24. Feeding those 24 zero hidden states to the LLM was causing it to say "I can't describe this image" even when the rest of the pipeline worked. Fixed: the prompt now uses 256 placeholders.

## Carried over from v0.2.0

- **Sliding Window Attention** — 28 sliding-window layers (W=512) and 7 full-attention layers, so decode stays flat as context grows from 512 to 2048.
- **Per-Layer Embedding on ANE** — Gemma 4's 8960×1536 projection + 35 RMSNorm slices moved inside the CoreML graph (Conv2d + LayerNorm), ~1.8 ms/token on ANE instead of ~8 ms on CPU BLAS.
- **Context length 2048** — stateless KV cache with explicit I/O (no `MLState`) to avoid per-layer state registration overhead.

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
| Batched Prefill | One CoreML call for up to 512 tokens | Order-of-magnitude faster TTFT vs per-token |
| PLE in graph | Conv2d projection + per-layer norm | 8 ms → 1.8 ms/token |

### Why not MLX?

MLX Swift targets the Apple GPU (Metal) and is excellent when you want maximum throughput on a plugged-in Mac. CoreML-LLM targets the Apple Neural Engine instead — which matters when:

- You want the GPU to stay free for rendering, games, or other ML work
- You want the LLM to coexist with foreground apps without competing for the same silicon
- You want the model placed on the most power-efficient compute unit available on Apple devices

The two are complementary, not competing. If you're on a Mac and want to burn through a 70B model as fast as possible, use MLX. If you want a 2B model quietly running on ANE inside an iPhone app, this library is aimed at that case.

We have verified the ANE placement on iPhone with `MLComputePlan` (99.78% of dispatched ops). We have not yet published a head-to-head power comparison against MLX on iPhone — previous versions of this README quoted `~2 W` and `~20 W` numbers that were not measured on-device, and those have been removed. A proper comparison will land in a follow-up once we have instrumented measurement.

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
