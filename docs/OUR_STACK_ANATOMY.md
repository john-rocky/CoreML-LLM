# Our stack — source-verified anatomy

**Date:** 2026-04-22
**Scope:** Swift runtime (`Sources/`) + Python conversion (`conversion/`). Direct source read, not docs-summary.

## TL;DR

Reality is often different from what prior docs imply. Specifically:

- **EAGLE-3 is default-OFF** (`LLM_EAGLE3_ENABLE=1` required, 22% accept)
- **DrafterUnion is the active speculative path** (CV Qwen + PLD n=2/3 + suffix trie)
- **Cross-vocab Qwen is default-OFF** (1.8 tok/s on iPhone, too slow)
- **Sampling is argmax only**, no top-k/top-p on-device
- **W4A8 is experimental**; INT4 palettization (per_grouped_channel, group_size=32) is default
- **SDPA fusion was tried and reverted** due to numerical divergence

---

## 1. Swift runtime (`Sources/CoreMLLLM/`)

### 1.1 Key files

| File | Size | Role |
|---|---|---|
| `CoreMLLLM.swift` | 71 KB, ~1600L | Top-level API, actor-safe (@unchecked Sendable), wrap ChunkedEngine + SpeculativeLoop + drafters |
| `ChunkedEngine.swift` | 122 KB, 3400+L | Decode loop, KV cache management, 4-chunk dispatch, prewarm, profiling |
| `SpeculativeLoop.swift` | 13 KB | EAGLE-3 accept/reject logic with tolerance |
| `DrafterUnion.swift` | 17 KB | Phase B orchestrator — union of 3 drafter sources |
| `PrefixKVCache.swift` | 8.1 KB | Persistent KV snapshots (SHA256 keyed, LRU, for system-prompt caching) |
| `PrefixCache.swift` | 8.9 KB | Token → embedding memoization |
| `PromptLookupLoop.swift` | 9.2 KB | PLD drafter implementation |
| `ModelConfig.swift` | 2.0 KB | Chunk count, slot counts, IDs |

### 1.2 Decode call flow (per token)

Per `predictStep(tokenID, position)` at `ChunkedEngine.swift:777-949`:

1. **Embed lookup** — INT8 quantized external tables via `EmbeddingLookup`, CPU + Accelerate.
2. **Per-layer-raw (PLE)** — CPU-side projection via Conv2d-equivalent, produces `(1, 1, 35×256) = (1, 1, 8960)` tensor.
3. **Causal + sliding masks** — per-position, full (ctx-length) and sliding (512-wide) constructed.
4. **RoPE tables** — position-based cos/sin, two sets: sliding (head_dim=256) and full (head_dim=512).
5. **Chunk 1 (L0-7)** — (hidden, 4 masks, 8 KV in, per-layer raw, 4 RoPE). Emits hidden + per-layer-combined. `copyBack` memcpy updates `K_sliding1, V_sliding1, K_full1, V_full1`. `ChunkedEngine.swift:830-833`.
6. **Chunk 2 (L8-14)** — (h1, masks, PL-combined, 4 KV in). Emits h2 + kv13_k/kv13_v/kv14_k/kv14_v **aliases** (actual producers depend on config: E2B L13/L14, E4B L22/L23). `copyBack` updates sliding2/full2.
7. **Chunk 3 (L15-24)** — Reads kv13/kv14 shared anchors, **does not write**. Q-only layers.
8. **Chunk 4 (L25-34)** — Same Q-only pattern + lm_head + argmax. Returns `token_id` only (not 262K logits).

### 1.3 KV cache layout

From `ChunkedEngine.swift:535-605`, `PrefixKVCache.swift:17-31`:

- Shape: `(slots, nkv, seqLen, maxHd=512)` MLMultiArray.
- Chunk 1: `(7 sliding slots, 1 nkv)`, `(1 full slot, 1 nkv)` (E2B).
- Chunk 2: `(5 sliding, 1 nkv)`, `(2 full, 1 nkv)` (E2B).
- Chunk 3 / 4: read-only shared KV13/KV14 from Chunk 2, no update.
- E4B variant: shapes differ (described in `kvShape()` at `ChunkedEngine.swift:573-583`).
- **IOSurface backing**: `CVPixelBufferCreate` with `kCVPixelBufferIOSurfacePropertiesKey + kCVPixelBufferMetalCompatibilityKey`, remapped to MLMultiArray. Fallback to standard MLMultiArray if IOSurface alloc fails (`ChunkedEngine.swift:546-568`).

### 1.4 Prefix KV snapshot protocol

`PrefixKVCache.swift`:
- Key = SHA256(prefix_tokens + ctx + model_id + version).
- Storage: binary blobs of 8 buffers (k_sliding1, v_sliding1, k_full1, v_full1, k_sliding2, v_sliding2, k_full2, v_full2) + KV13/KV14 shared anchors.
- Eviction: LRU by file modification date, `maxEntries=64`.
- Purpose: TTFT speedup on repeated system prompts.

### 1.5 Prewarm pattern

`ChunkedEngine.swift:607-622, 626-647`:
- **Early prewarm (after chunk load, before aux models):** 4 dummy decode steps, KV reset after.
- **Final prewarm (after all aux models load, including EAGLE-3 fusion/draft + Qwen):** 8 additional decode + verify steps.
- **Why two phases:** loading Qwen or EAGLE-3 evicts ANE compiler caches; re-warm needed.

### 1.6 Speculative paths — what's active, what's dormant

Per `CoreMLLLM.swift:87-90, 203-293`:

| Drafter | Default | Trigger | Accept rate | Notes |
|---|---|---|---|---|
| None (serial T=1 decode) | **ACTIVE default** | - | 100% (no drafts) | Baseline fallback |
| DrafterUnion | **ACTIVE if components loaded** | aux files present | Composite | Merges CV Qwen + PLD n=2/3 + suffix-trie; picks longest proposal, tie-break cross-vocab > pldN3 > suffix > pldN2 (`DrafterUnion.swift:46-48`) |
| EAGLE-3 | **OFF** | `LLM_EAGLE3_ENABLE=1` | 22% | 234 MB ANE. Opt-in dead weight. |
| Cross-vocab Qwen | **OFF** | aux files + env flag | N/A (too slow) | 1.8 tok/s on iPhone 17 Pro, ~10× slower than projection drafter |

### 1.7 What's NOT used

- **No `MLPredictionOptions`** explicit tuning. Defaults + IOSurface-backed MLMultiArray only.
- **No `outputBackings`** dict. Direct `model.prediction(from:)` calls.
- **No ring buffer / ping-pong pool** for output buffers (unlike ANEMLL).
- **No top-k / top-p on-device.** Argmax only.
- **No Private API symbols** — no dlopen, no `_ANE*` access.

### 1.8 Profiling

`ChunkedEngine.swift:751-775`:
- Accumulated timers: `profileEmbed`, `profilePredict`, `profileMask`, `profileC1/C2/C3/C4`, `profileANEWait`, `profileCopyBack`.
- Output every 10 steps (or `LLM_PROFILE_EVERY_STEP=1`).
- Format: `[ANE/CPU] ANE_wait=Xms copyBack=Yms cpu_active=Zms (ZZ% CPU)`.

### 1.9 Test / bench harnesses

- **accept-rate-bench** (`Sources/accept-rate-bench/Bench.swift:1-150`): oracle replay (temp=0), modes oracle/argmax/chain. Measures per-drafter histogram.
- **union-bitexact** (`Sources/union-bitexact/main.swift:1-63`): merge-blocking test. Runs serial decode vs DrafterUnion, both must produce identical token stream. Exit non-zero on divergence.
- **CoreMLLLMSmoke** (`Sources/CoreMLLLMSmoke/main.swift:1-67`): CLI smoke test, Mac only.
- **VideoTest** (`Sources/VideoTest/main.swift:1-135`): frame extraction + vision + LLM chain.

---

## 2. Python conversion pipeline (`conversion/`)

### 2.1 Model config (confirmed from source)

`conversion/config.py:21-57` registers:
- `gemma4-e2b` → `google/gemma-4-E2B-it`
- `gemma4-e4b` → `google/gemma-4-E4B-it` (42 layers, hidden=2560, 2 KV heads)
- qwen2.5-0.5b / 1.5b / 3b

`conversion/models/gemma4.py:Gemma4Config`:

```python
hidden_size = 1536               # E2B
num_hidden_layers = 35
num_attention_heads = 8
num_key_value_heads = 1          # GQA 8:1
head_dim = 256                   # sliding layers
global_head_dim = 512            # full-attention layers (double!)
intermediate_size = 6144
use_double_wide_mlp = True       # FFN doubles to 12288 for KV-shared layers
vocab_size = 262144
hidden_activation = "gelu_pytorch_tanh"
sliding_window = 512
final_logit_softcapping = 30.0   # tanh-based, factor 30
num_kv_shared_layers = 20        # last 20 layers (L15-34) share KV from L13/L14
hidden_size_per_layer_input = 256
```

Layer types: every 5th layer (`(i+1) % 5 == 0`) is full_attention, else sliding_attention.
- L0-3 sliding, L4 full, L5-8 sliding, L9 full, L10-13 sliding, L14 full, L15-18 sliding, L19 full, L20-23 sliding, L24 full, L25-28 sliding, L29 full, L30-33 sliding, L34 full.

### 2.2 Chunk boundaries

From `conversion/models/gemma4_swa_chunks.py:compute_chunk_boundaries` + `collect_eagle_hidden_states_w4a8.py:45-50`:

| Chunk | Layers | Sliding | Full | Notes |
|---|---|---|---|---|
| 1 | L0-7 | 7 | 1 (L4) | KV producers |
| 2 | L8-14 | 5 | 2 (L9, L14) | 8K bottleneck; emits KV13/KV14 for downstream |
| 3 | L15-24 | 5 | 2 (L19, L24) | Q-only; reads kv13/kv14 |
| 4 | L25-34 | 5 | 2 (L29, L34) | Q-only; lm_head + argmax |

### 2.3 ANE op rewrites

`conversion/ane_ops.py`:

| Torch op | ANE-friendly rewrite | File:line | Reason |
|---|---|---|---|
| `nn.Linear` | `nn.Conv2d(kernel_size=1)` | `ane_ops.py:57-96` | 3× ANE throughput |
| `RMSNorm` | `LayerNorm([x, -x])[:hidden]` | `ane_ops.py:25-54` | ANE has optimized LayerNorm; `rsqrt` not accelerated |
| `Concat` (drafters) | Additive decomposition `Linear_A(x) + Linear_B(y)` | `build_eagle3.py:130-131`, `build_mtp_drafter.py:123-124` | Concat is slow on ANE |
| `Argmax` (final) | `InModelArgmax` embedded in graph | `ane_ops.py:114-135` | Avoid shipping 262K float logits |
| Manual attention | matmul + `ane_softmax` (NOT SDPA) | `models/gemma4_swa_chunks.py:140-142` | SDPA decomposition diverges numerically |

### 2.4 Quantization paths

**Default (all `build_*.py`):** INT4 palettization, `per_grouped_channel`, `group_size=32`.

**Embedding:** INT8 per-row symmetric, external .bin + .scales. Dequant: `fp16 = int8 * (scale/127) * embedScale` (`EmbeddingLookup.swift:43`). Key bug fix: including `per_layer_embed_scale` — omission caused 0% accept rate on iPhone EAGLE-3.

**W8A8 (experimental):** `build_w8a8_proper.py:186-196`. INT8 weight linear (symmetric per-channel) + INT8 activation via `coremltools.optimize.coreml.linear_quantize_activations(mlmodel, cfg, cal_data)`. Calibrated on 32 samples from running deployed chunks (realistic distribution).

**Not quantized:** RoPE tables, per_layer_projection, causal masks, KV cache — all fp16.

### 2.5 Build scripts inventory

`conversion/`:

| Script | Output | Purpose |
|---|---|---|
| `build_gemma4_bundle.py` | Full iPhone bundle: 4 chunks + embeddings + RoPE + per-layer proj + config + hf_model/ | Top-level entry |
| `build_verify_chunks.py` | Multifunction `.mlpackage` with `decode_q1` (N=1) and optional `verify_qK` (K=3) | 4-chunk decode + optional verify |
| `build_eagle3_chunks.py` | Target chunks with fusion taps at L8, L17, L34 | For EAGLE-3 hidden-state collection |
| `build_eagle3.py` | Trained EAGLE-3 drafter → CoreML (concat-free) | Drafter side |
| `build_mtp_drafter.py` | 4-layer MTP transformer, top-k(8), logit softcap | Alternative drafter |
| `build_speculative.py` | chunk4 rebuild + verify_chunk{1..4} + medusa (3 heads shared lm_head) | Medusa stack |
| `build_flash.py` | Flash Decoding, K-dim chunked (1024), online softmax | Alt attention |
| `build_wfa.py` | Windowed Full Attn, FW=2048, shift-based KV | Alt KV strategy |
| `build_prefill_gpu.py` | GPU-targeted prefill (A19 Pro tensor cores) | Prefill optim |
| `build_w8a8_proper.py` / `build_w8a8.py` | W8A8 quantized chunks | Experimental |
| `apply_vocab_pruning.py` | Sliced embeddings + lm_head + vocab_remap.json | Vocab pruning path |
| `build_qwen_gemma_vocab_map.py` | Bidirectional Qwen↔Gemma vocab alignment | Cross-vocab drafter |
| `collect_eagle_hidden_states_w4a8.py` | Memmap + manifest of hidden states from deployed W4A8 | EAGLE training data |
| `compare_bf16_vs_w4a8.py` | Argmax agreement, per-position drift | Validation |
| `benchmark_chunks.py` / `benchmark_prefill.py` / `benchmark_wfa.py` | Per-chunk + total latency | Local benchmark |
| `cascading_runtime.py` | Experimental inline gather-idx instead of mask-based KV updates | Not deployed |

### 2.6 Critical bugs / lessons encoded in code

`collect_eagle_hidden_states_w4a8.py:88-94`:
> PER-LAYER-RAW (PLE) Scale Bug Fix — earlier version omitted ×16 scale, producing OOD per_layer_combined vs deployment. Single largest contributor to 0% accept rate on iPhone.

`models/gemma4_swa_chunks.py:136-138`:
> SDPA fusion produces slightly different results from manual attention, causing wrong token predictions. Keeping manual attention for correctness.

`build_eagle3.py:143-144` vs `build_mtp_drafter.py:143`:
- EAGLE-3 returns all 262K logits (for training telemetry)
- MTP returns top-8 only (efficiency prune)
- Inconsistent; suggests drafter-quality vs efficiency trade still being tuned.

---

## 3. Relationship of our stack to external references

| Concern | Our stack | ANEMLL | ExecuTorch | llama.cpp |
|---|---|---|---|---|
| Quantization | INT4 palettize (per_grouped_ch, g=32) + W4A8 exp | LUT 4/6/8 | Per-group INT4 / INT8 | Q4_K block quant |
| Chunking | 4 chunks, layer-aligned | balanced-by-param | Per-block graph break | No chunk; single graph |
| KV layout | IOSurface MLMultiArray, shape-specialized | ring-16 / ping-pong-2 | StaticKVCache shift/mask | Standard tensor |
| Attention | Manual matmul + ane_softmax | Manual + explicit causal mask (ANE) | Static-attention | FlashAttention kernels |
| Speculative | DrafterUnion (active) + EAGLE-3 (opt-in, dead) | None | None | Dual / simple / PLD / lookahead |
| Sampling | Argmax only | Argmax | Temp + top-p | Full sampler chain |
| Prefill/decode | Separate chunks | Separate graphs | Separate signatures | Single graph w/ batch |
| Private API | None | None | None | None |

Our stack is **closest to ANEMLL** in philosophy (CoreML-converted, ANE-primary, static graphs) but diverges in quantization (we have W4A8 in flight, they're LUT-only) and speculation (we actively explore drafters, they don't).

---

## 4. Things we should probably do (source-grounded)

1. **Fix EAGLE-3 opt-in default or remove** — if 22% accept with 234 MB ANE cost, it's net-negative even when enabled. Either fix training (L-MTP style) or drop from codebase.
2. **Ring/ping-pong output buffers** — ANEMLL has clear pattern (`InferenceManager.swift:55-60`). We could A/B.
3. **iOS 18 stateful read_state/write_state** — potentially replaces our `copyBack` memcpy path. Test.
4. **Retry SDPA fusion on iOS 18 native op** — the "wrong predictions" comment is from before iOS 18 SDPA. Worth re-testing.
5. **Top-k sampling on-device (GPU)** — if we ever want non-greedy, Metal compute path like LiteRT's `TopKMetalSampler` is cheap.
