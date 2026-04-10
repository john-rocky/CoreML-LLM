# Mobile LLM Inference Research Notes

Research findings informing CoreML-LLM development. Last updated 2026-04-10.

## Competitive Landscape (iPhone)

| Engine | Hardware | Speed (2B) | Sustained | ANE? |
|---|---|---|---|---|
| **CoreML-LLM** | **ANE** | **28 tok/s** | **stable (thermal fair)** | **99.78%** |
| LiteRT (Google) | GPU | ~30 tok/s | unknown | 0% |
| MLX Swift | GPU | ~22 tok/s sustained | **44% drop from peak** | 0% |
| llama.cpp | GPU/Metal | ~25 tok/s | throttles | 0% |
| MLC LLM | GPU | ~10 tok/s | **thermal shutdown after 5 iter** | 0% |

Key insight: **GPU-based inference on iPhone suffers severe thermal throttling.** MLX loses 44% throughput within 2 iterations. MLC LLM shuts down after 5. Our ANE approach maintains stable throughput because ANE draws ~2W vs GPU ~20W.

## Apple Intelligence Architecture (Reference Implementation)

Apple's on-device model (3.18B params):
- **48M draft model** (1.5% of base) for speculative decoding → **2-4x speedup**
- 60-80% acceptance rate
- **2-bit QAT** with learnable weight clipping → half the model size
- **KV cache sharing**: Block 2 (37.5% of layers) reuses KV from Block 1 → 37.5% memory reduction
- **8-bit KV cache quantization**

Sources: [AFM Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025)

## Speculative Decoding on Mobile

### Self-Speculative / LayerSkip (Meta)
- Uses early layers as draft "model" within the same network
- 1.34-2.16x speedup with **zero additional memory**
- Shared KV cache between draft and verify phases
- Best option for our chunked architecture

### EAGLE-3
- Acceptance rate: 70-80% (stable across context lengths)
- 3.0-6.5x speedup over vanilla autoregressive
- Draft "head" is <5% of base model parameters

### Retrieval-Based (sd.npu)
- No draft model needed — uses context/history n-grams for drafting
- 1.06-3.81x end-to-end speedup
- Particularly effective for RAG/summarization tasks

## Energy Efficiency

| | Power | Speed | J/tok |
|---|---|---|---|
| ANE (ours) | ~2W | 28 tok/s | ~0.07 |
| GPU (MLX) | ~20W | 30 tok/s | ~0.67 |
| Ratio | | | **~10x more efficient** |

ANE: 6.6 TFLOPS/watt, ~80x more efficient per FLOP than data center GPUs.

## Key Optimizations to Explore

1. **Self-speculative decoding (LayerSkip)** — exit after chunk1-2 for draft, verify with chunk1-4. Zero extra memory.
2. **2-bit QAT quantization** — Apple proves 2-bit works. 2.7 GB → 1.4 GB.
3. **IOSurface-backed MLMultiArray** — reduces memory copy overhead for KV cache.
4. **Delta compilation** — update weights without full recompilation (8.5x faster). Enables LoRA hot-swap.
5. **KV cache sharing** — share KV between SWA/global attention blocks (Apple's Block 1/Block 2 pattern).
6. **Retrieval-based drafting** — for RAG use cases, draft from context (free performance).
7. **Energy-per-token benchmarking** — market the 10x efficiency advantage.

## Notable Projects

- **ANEMLL** — OSS ANE LLM library. 47-62 tok/s on 1B, ~9 tok/s on 8B. Uses IOSurface buffers.
- **Orion** — Reverse-engineered ANE: 16-core, 32MB SRAM, 19 TFLOPS FP16, 27-op graph IR.
- **Cactus** — Cross-platform mobile LLM. 136 tok/s on 450M model.
- **HeteroLLM** — Splits work across CPU+GPU+NPU by tensor shape. 5.87x energy savings.
