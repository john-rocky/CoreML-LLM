# Mobile LLM runtime comparison — source-verified

**Date:** 2026-04-22
**Scope:** llama.cpp, MLX (+swift), MLC-LLM, ExecuTorch, ANEMLL, LiteRT-LM, our stack (CoreML-LLM).

## 0. TL;DR

| Concern | Winner | Runner-up |
|---|---|---|
| Direct CoreML + ANE template | **ExecuTorch** | **ANEMLL** |
| Metal kernel quality | **llama.cpp** | whisper.cpp |
| Gemma 4 E2B architecture correctness | **llama.cpp** (gemma4-iswa.cpp) | HF transformers |
| W4A8 static-graph conversion | **Our stack** (W4A8 not elsewhere) | ExecuTorch (IntxWeightOnlyConfig) |
| MTP / speculative end-to-end | **LiteRT-LM** | llama.cpp (5+ SD variants) |
| Mobile iOS Swift API ergonomics | **mlx-swift-examples** | llama.swiftui |
| Dispatch / encoding overhead | **llama.cpp** (multi-threaded encode) | MLX |
| Whisper-style encoder CoreML offload | **whisper.cpp** (only one) | — |
| Production ANE Gemma support | **ANEMLL** (Gemma 3 only) | — |
| World's fastest Gemma 4 E2B on iOS | **LiteRT-LM** 56 tok/s (Metal+MTP) | our stack 31 tok/s |

## 1. Comparison matrix — architectural concerns

### 1.1 Primary execution backend on Apple Silicon

| Runtime | Decode | Prefill | Notes |
|---|---|---|---|
| **Our stack** | ANE | ANE (or GPU opt-in via `GPU_PREFILL=1`) | 100% ANE decode is our signature |
| ANEMLL | ANE | ANE | Same philosophy |
| LiteRT-LM | Metal GPU | Metal GPU | No ANE code exists |
| llama.cpp | Metal GPU | Metal GPU | No ANE; pure Metal shaders |
| ExecuTorch CoreML | ANE (declared) | ANE | CoreML backend with `compute_unit=CPU_AND_NE` |
| MLC-LLM | Metal GPU (TVM Relax) | Metal GPU | No ANE; uses TVM codegen |
| mlx-swift | Metal GPU (MLX) | Metal GPU | No ANE; explicit refutation |
| whisper.cpp | Metal decode + **CoreML encoder** | CoreML | Only runtime that offloads encoder to CoreML for ANE |

### 1.2 KV cache management

| Runtime | Layout | Update | Sliding window |
|---|---|---|---|
| **Our stack** | IOSurface MLMultiArray, shape-specialized per chunk | `copyBack` memcpy in Swift | Explicit per-position masking |
| ANEMLL | IOSurface CVPixelBuffer + ring-16 / ping-pong-2 | In-place via backing | Rotate model variants for cross-window |
| LiteRT-LM | `LitertKVCache` dual-bank | in-place/out-of-place via backend | `DeleteTokensFromKvCache` (stream delete) |
| llama.cpp | `llama_kv_cache_unified` + `llama_kv_cache_iswa` (dual for ISWA) | ggml tensor ops | `is_masked_swa()` in hparams |
| ExecuTorch | `StaticKVCache` (shift-pointer or smart-mask) | explicit update | static-attention pattern |
| MLC-LLM | `PagedKVCache.create_generic` (MLA/MHA/SWA) | TVM-rewritten | Paged, rewrites in pass |
| mlx-swift | tuple `(K, V)`, concat each step | stateless | none |
| whisper.cpp | split self-attn + cross-attn (cross read-only) | ggml | N/A (fixed length) |

**Best for Gemma 4 E2B:** llama.cpp's `llama_kv_cache_iswa` — exact ISWA dual-bank that matches Gemma 4's alternating pattern.

### 1.3 Quantization

| Runtime | Schemes | ANE compatible | Gemma tested |
|---|---|---|---|
| **Our stack** | INT4 palettize (per_grouped_ch, g=32) + W4A8 experimental | Yes (INT4) | Yes (Gemma 4) |
| ANEMLL | LUT 4/6/8-bit only | Yes (LUT) | Yes (Gemma 3) |
| LiteRT-LM | W4A8 baked in `.litertlm` | Yes (proprietary pipeline) | Yes (Gemma 4 E2B) |
| llama.cpp | Q4_K, Q3_K, q4_0, q8_0, q5_0, iq2_xxs, iq3_xxs, iq4_nl, mxfp4 | N/A (Metal only) | Yes |
| ExecuTorch | Per-group INT4 (g=32), PT2E INT8 per-channel, `IntxWeightOnlyConfig` | Yes | Partial (Llama) |
| MLC-LLM | `GroupQuantize` (INT3/4/8), `BlockScaleQuantize` | N/A (Metal) | Yes (Gemma 3) |
| mlx-swift | q4_0, q8_0 via `get_quantized_kernel_wrapped` | N/A (Metal) | Via registry preset (Gemma 3n) |
| whisper.cpp | q4_0, q5_0, q8_0 | Partial (encoder) | N/A (Whisper) |

**Our advantage:** W4A8 with activation quantization is rare. Elsewhere most are weight-only (LUT) or block quant.

### 1.4 Speculative decoding

| Runtime | Variants |
|---|---|
| **Our stack** | DrafterUnion (CV Qwen + PLD n=2/3 + suffix trie) + EAGLE-3 (opt-in) + MTP drafter build + Medusa build + Flash + WFA |
| LiteRT-LM | MTP (linear 4-step greedy exact-match, wired in) |
| llama.cpp | Dual-model, simple, PLD (lookup), LookAhead, EAGLE3 stub, n-gram variants (SIMPLE, MAP_K, MAP_K4V, MOD, CACHE) |
| MLC-LLM | `speculative_mode = "small_draft" / "eagle" / "medusa"` |
| ExecuTorch | None in iOS path |
| mlx-swift | None |
| mlx-examples | Python speculative decoding script |
| ANEMLL | None |
| whisper.cpp | N/A |

**Our stack has the widest SD menu**, but all are fighting the oracle-live acc gap.

### 1.5 Prefill / decode separation

| Runtime | Pattern |
|---|---|
| **Our stack** | 4 chunks, separate decode_q1 / verify_qK entry points per chunk |
| ANEMLL | Separate `prefillModel` / `inferModel` per chunk |
| LiteRT-LM | Same compiled model, separate signatures (`kPrefillSignatureRunner` / `kDecodeSignatureRunner`) |
| llama.cpp | Single graph, batch dim switches mode |
| ExecuTorch | Static attention, fixed buffer, no explicit split |
| MLC-LLM | Paged cache + dynamic dispatch |

### 1.6 Sampling

| Runtime | Sampler |
|---|---|
| **Our stack** | Argmax only (T=0) |
| LiteRT-LM | TopK on GPU via `TopKMetalSampler` |
| llama.cpp | Full chain: bias, top-k, top-p, temperature, DRY, XTC, grammar |
| ANEMLL | Argmax |
| ExecuTorch | Temperature + top-p |
| MLC-LLM | `AttachGPUSamplingFunc` (Metal) |
| mlx-swift | Built-in samplers |

**Implication:** Our stack cannot do temperature sampling today. Adding a Metal top-k sampler (LiteRT pattern) is low-cost if ever needed.

## 2. Source template rankings

### 2.1 For a **new model conversion** (future Gemma 5, Qwen 3, etc.)

| Rank | Template | Why |
|---|---|---|
| 1 | **ExecuTorch `examples/apple/coreml/llama/export_static_llm_coreml.py`** | Per-layer graph break + CoreML partition + NamedDataStore weight sharing. Most structured. |
| 2 | **ANEMLL `anemll/ane_converter/gemma3_converter.py`** | Already-ANE-optimized (LUT quant, sliding rotation models). Gemma-specific. |
| 3 | Our `conversion/build_gemma4_bundle.py` | Our current pipeline, Gemma-4-specific. |

**Action:** Study ExecuTorch pattern and consider a refactor for future conversions.

### 2.2 For a **Metal port** of existing Gemma 4 E2B

| Rank | Template | Why |
|---|---|---|
| 1 | **llama.cpp `src/models/gemma4-iswa.cpp` + Metal shaders** | Production Gemma 4 path with ISWA, KV-sharing, fused GeGLU. |
| 2 | MLX STEEL attention kernels | Alternative Metal implementation. |
| 3 | whisper.cpp Metal | Encoder pattern + `GGML_METAL_EMBED_LIBRARY` build trick. |

### 2.3 For **speculative decoding validation / future MTP**

| Rank | Template | Why |
|---|---|---|
| 1 | **LiteRT-LM MTP** | Actually ships on Gemma 4 E2B mobile. Linear 4-step greedy exact-match. |
| 2 | **L-MTP GitHub** (not cloned yet) | Training code for multi-token prediction heads. |
| 3 | llama.cpp `examples/lookup` (PLD) | Retrieval drafter; narrow applicability. |

### 2.4 For **ANE decoder improvements** on existing stack

| Rank | Template | Why |
|---|---|---|
| 1 | **ANEMLL ring/ping-pong + IOSurface CVPixelBuffer** | Stability + mild throughput. |
| 2 | **coremltools iOS 18 native SDPA + MLState** | Less graph complexity, maybe less copyBack overhead. |
| 3 | **maderix Private API (`_ANEInMemoryModel*`)** | Last resort — bypasses CoreML compiler. High effort, uncertain ROI. |

## 3. Per-runtime key files index

### llama.cpp (`/Users/majimadaisuke/Downloads/workspace/repo-review/llama.cpp`)
- Gemma 4 model: `src/models/gemma4-iswa.cpp:1-300`
- ISWA dual KV: `src/llama-kv-cache-iswa.cpp:1-50`, `src/llama-hparams.h:316-343`
- FlashAttention Metal: `ggml/src/ggml-metal/ggml-metal.metal:5628-6512` (head-dim templates)
- Fused GeGLU: `src/llama-graph.cpp:1227-1231`
- Multi-threaded encode: `ggml/src/ggml-metal/ggml-metal-context.m:676-721`
- Speculative: `examples/speculative/speculative.cpp`, `examples/lookup/`, `examples/lookahead/`
- iOS build: `build-xcframework.sh`
- SwiftUI example: `examples/llama.swiftui/llama.cpp.swift/LibLlama.swift`

### MLX core (`/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/mlx`)
- Backend architecture: `mlx/backend/metal/eval.cpp`, `device.h`
- STEEL attention: `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h:74-150`
- RMSNorm: `mlx/backend/metal/normalization.cpp:13-93`
- Quantized matmul: `mlx/backend/metal/quantized.cpp:19-51`
- Compile-time fusion: `mlx/backend/metal/compiled.cpp:16-255`

### MLX Swift (`/Users/majimadaisuke/Downloads/workspace/repo-review/mlx-swift`)
- No ANE integration (confirmed)
- Lazy eval transparent

### MLX Swift Examples (`/Users/majimadaisuke/Downloads/workspace/repo-review/mlx-swift-examples`)
- LLM registry: `Applications/MLXChatExample/Services/MLXService.swift:23-38`
- Async generation stream via `AsyncStream<Generation>`
- Gemma 3n E2B preset: `gemma3n_E2B_it_lm_4bit`

### ExecuTorch (`/Users/majimadaisuke/Downloads/workspace/repo-review/executorch`)
- **Primary template:** `examples/apple/coreml/llama/export_static_llm_coreml.py`
- CoreML delegate: `backends/apple/coreml/compiler/coreml_preprocess.py:50-95`
- Static KV cache: `examples/models/llama/static_attention.py:33-120`
- Embedding quant: `kernels/quantized/quantized.yaml:37-71`

### MLC-LLM (`/Users/majimadaisuke/Downloads/workspace/repo-review/mlc-llm`)
- Gemma3 model: `python/mlc_llm/model/gemma3/gemma3_model.py`
- Paged KV cache: `python/mlc_llm/nn/kv_cache.py:17-94`
- Group quant: `python/mlc_llm/quantization/group_quantization.py:28-80`
- Speculative: `python/mlc_llm/interface/serve.py` (mode = small_draft / eagle / medusa)

### ANEMLL (`/Users/majimadaisuke/Downloads/workspace/repo-review/Anemll`)
- Gemma 3 model: `anemll/models/gemma3_model.py:416-712`
- Converter: `anemll/ane_converter/gemma3_converter.py`
- Chunk calc: `anemll/utils/calc_chunk_split.py:245-248`
- Runtime: `anemll-swift-cli/Sources/AnemllCore/InferenceManager.swift`
- FFN chunk: `anemll-swift-cli/Sources/AnemllCore/FFNChunk.swift:6-34`

### LiteRT-LM (`/Users/majimadaisuke/Downloads/workspace/LiteRT-LM`)
- Backend factory: `runtime/executor/llm_litert_compiled_model_executor_factory.cc:63-180`
- MTP drafter: `runtime/executor/llm_litert_mtp_drafter.cc:420-467`
- Metal sampler: `runtime/components/top_k_metal_sampler.h:92-183`
- KV cache: `runtime/executor/litert/kv_cache.h:35-100`
- Gemma 4 data processor: `runtime/conversation/model_data_processor/gemma4_data_processor.{h,cc}`

### whisper.cpp (`/Users/majimadaisuke/Downloads/workspace/repo-review/whisper.cpp`)
- CoreML encoder wrapper: `src/coreml/whisper-encoder.mm`
- Metal kernels (shared with llama.cpp)
- SwiftUI example: `examples/whisper.swiftui/LibWhisper.swift`

### maderix (`/Users/majimadaisuke/Downloads/workspace/repo-review/maderix-ANE`)
- Private API bridge: `bridge/ane_bridge.m:33-156`
- INT8 benchmark: `ane_int8_bench.m:220-264`
- Qwen3-0.6B training: `training/training_dynamic/models/qwen3_06b.h`
- Compile budget: `training/stories_config.h:26` (MAX_COMPILES=100)

### Apple ml-ane-transformers (`/Users/majimadaisuke/Downloads/workspace/repo-review/ml-ane-transformers`)
- LayerNormANE: `ane_transformers/reference/layer_norm.py:10-79`
- MultiHeadAttention (einsum channel-first): `ane_transformers/reference/multihead_attention.py:64-120`
- FFN: `ane_transformers/reference/ffn.py:14-21`

### coremltools (`/Users/majimadaisuke/Downloads/workspace/repo-review/coremltools`)
- Pass pipeline: `coremltools/converters/mil/mil/pass_pipeline.py` (123 passes)
- RMSNorm auto-conversion: `converters/mil/frontend/torch/ops.py:3107-3171`
- iOS 18 SDPA: `mil/ops/defs/iOS18/transformers.py:18-167`
- iOS 18 states: `mil/ops/defs/iOS18/states.py`
- Palettize: `optimize/coreml/_post_training_quantization.py:188-250`

## 4. If we had to pick ONE runtime to study for a specific concern

- **"How does Gemma 4 E2B really work?"** → llama.cpp `src/models/gemma4-iswa.cpp`
- **"How do I write a Metal kernel for X?"** → llama.cpp `ggml/src/ggml-metal/ggml-metal.metal`
- **"How do I ship LLM on iOS via CoreML?"** → ExecuTorch `examples/apple/coreml/llama/`
- **"How do I stream tokens in Swift?"** → mlx-swift-examples `Applications/MLXChatExample/`
- **"How do I beat LiteRT 56 tok/s on Gemma 4 E2B?"** → LiteRT-LM source (spoiler: you need MTP, Metal GPU, or both)
- **"How is Gemma 3 done on ANE?"** → ANEMLL (Gemma 4 not yet)
- **"What ANE Private APIs exist and how to call them?"** → maderix/ANE `bridge/ane_bridge.m`
- **"What does Apple recommend for ANE-optimized transformers?"** → ml-ane-transformers `ane_transformers/reference/`
- **"Which coremltools features am I underusing?"** → `COREMLTOOLS_AND_IOS18.md` (this knowledge base)
