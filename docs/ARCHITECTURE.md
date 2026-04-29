# Architecture

```
      Prompt ─┐
              ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Prefill ch1  │─►│ Prefill ch2  │─►│ Prefill ch3  │─►│ Prefill ch4  │─► first token
    │ L0-7 + PLE   │  │ L8-14, kv13/ │  │ L15-24 shared│  │ L25-34 + LM  │
    └──────────────┘  │  kv14 out    │  └──────────────┘  └──────────────┘
            │         └──────────────┘         ▲                 ▲
            │                │                 │                 │
            │                └────────────┬────┴─────────────────┘
            │                             │ kv13_k/v, kv14_k/v (shared)
            ▼                             ▼
    writes K/V to persistent SWA caches
            │
            ▼  (decode loop, 1 token per step)
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Decode ch1  │─►│  Decode ch2  │─►│  Decode ch3  │─►│  Decode ch4  │─► next token
    └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

As of v1.7.0 the Gemma 4 E2B picker default is the **3-chunk decode** variant (`gemma4e2b3way`) — `chunk1` + `chunk2_3way` (L8-24 merged) + `chunk3_3way` (L25-34 + lm_head). 3 ANE dispatches per decode step instead of 4, **+8.2 %** on iPhone A19 Pro. The 4-chunk legacy entry stays in the picker as `Gemma 4 E2B (4-chunk legacy)` for back-compat with users who already downloaded the older bundle. Prefill graphs stay 4-chunk (T=1024) so multimodal vision-aware bidirectional mask is preserved unchanged. The picker also has a **Download Options** toggle: turning off "Include multimodal" drops the vision/video/audio encoders + sidecars (~1 GB) for a text-only install. See [THREE_CHUNK_MAC_BENCH.md](THREE_CHUNK_MAC_BENCH.md).

## ANE optimizations

| Technique | What | Why |
|---|---|---|
| ANERMSNorm | `cat([x,-x])` → LayerNorm → slice | ANE has optimized LayerNorm; bare RMSNorm is slow |
| Conv2d-Linear | `nn.Linear` → `nn.Conv2d(kernel_size=1)` | ANE executes Conv2d ~3× faster than matmul |
| In-graph argmax | Argmax inside the CoreML graph | Avoids shipping 256K logits from ANE to CPU |
| Manual softmax | `max/sub/exp/sum/div` with explicit fp16 casts | Prevents PyTorch fp16→fp32 upcast in `torch.exp` |
| Pre-computed RoPE | cos/sin as model inputs, looked up in Swift | Eliminates `gather` / `greater_equal` (int ops → CPU) |
| Explicit KV I/O | Plain tensor inputs/outputs, no `MLState` | Avoids int64 state indices that break ANE placement |
| Sliding window | Shift-based cache for 28/35 layers | O(W) per step instead of O(ctx) |
| Batched prefill | One CoreML call per 512-token chunk | Order-of-magnitude faster TTFT vs per-token |
| PLE in-graph | Conv2d projection + per-layer norm | 8 ms → 1.8 ms/token |
| 3-chunk decode (v1.4) | Merge chunk2+chunk3 into one 17-layer block | −1 ANE dispatch, +8.2 % tok/s |

## Why not MLX?

MLX Swift targets the Apple GPU (Metal). Great on a plugged-in Mac pushing a 70B. This library targets the ANE, which matters when:

- The GPU should stay free for rendering, games, or other ML work
- The LLM must coexist with foreground apps without competing for the same silicon
- You want the most power-efficient compute unit on Apple silicon

The two are complementary — run MLX on desktop, run CoreML-LLM inside an iPhone app.

## Project structure

```
Sources/CoreMLLLM/          Swift Package (`import CoreMLLLM`)
  CoreMLLLM.swift            Public API — load, generate, stream
  ChunkedEngine.swift        SWA decode + prefill engine (3/4-chunk)
  FunctionGemma.swift        Function-calling specialist
  EmbeddingGemma.swift       Sentence-embedding specialist
  ModelDownloader.swift      Background download, pause/resume
  ImageProcessor.swift       Vision preprocessing (image + video)
  AudioProcessor.swift       Mel + Conformer
  …

Examples/CoreMLLLMChat/     iOS sample app (chat + multimodal)
Examples/Gemma3Demo/        Standalone sample (FunctionGemma + EmbeddingGemma)
conversion/                 Python conversion pipeline
  convert.py                   CLI entry point
  build_gemma4_bundle.py       One-shot Gemma 4 bundle builder
  build_gemma4_3way.py         3-chunk decode variant (v1.4)
  build_functiongemma_bundle.py
  build_embeddinggemma_bundle.py
  models/                      Per-architecture PyTorch traces
docs/                       Design docs, benchmarks, decision log
```
