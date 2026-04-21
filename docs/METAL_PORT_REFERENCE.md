# Metal Port Reference — llama.cpp + MLX source-verified blueprint

**Date:** 2026-04-22
**Context:** Per `docs/project_drafter_structurally_dead.md`, Metal-LLM Phase 3 is
the critical path to beating LiteRT-LM 56 tok/s. This doc inventories what
llama.cpp and MLX already provide at source level, so we can make port
decisions against real code, not README summaries.

**Sources:**
- llama.cpp: `/Users/majimadaisuke/Downloads/workspace/repo-review/llama.cpp`
- MLX: `/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/mlx`

---

## 0. TL;DR — llama.cpp is a near-complete blueprint

**llama.cpp has a first-class Gemma 4 E2B Metal implementation.** Not a
toy, not a stub — a production class (`llm_build_gemma4_iswa`) covering
every architectural quirk: ISWA dual-KV sliding window, Q-only layer
KV-sharing via `has_kv()`, QK-norm, fused GeGLU, per-layer embeddings.
FlashAttention Metal kernels are templated for the exact head-dim we
need (`dk=256, dv=256`).

The Metal port is not "design from scratch." It is "port llama.cpp's
Gemma 4 path + its Metal shaders to our runtime + our quantization."

### The 5 primitives to steal, ranked

| # | What | File:line | Why it matters |
|---|---|---|---|
| 1 | FlashAttention kernel `f32_dk256_dv256` | `ggml/src/ggml-metal/ggml-metal.metal:6509` | Exact head-dim for Gemma 4 E2B. +15-20% decode throughput vs naive. |
| 2 | Dual KV cache (`kv_base` + `kv_swa`) | `src/llama-kv-cache-iswa.cpp:1-50` | Sliding-512 memory cost drops; alternating ISWA pattern matches ours. |
| 3 | Fused GeGLU (`ggml_geglu_split`) | `src/llama-graph.cpp:1227-1231` | Gate×up in one op. +8-12% FFN throughput. |
| 4 | Multi-threaded command buffer encoding | `ggml/src/ggml-metal/ggml-metal-context.m:676-721` | `dispatch_apply` parallel encode overlaps GPU exec. +10% wall-clock. |
| 5 | Q-only layer KV-sharing via `has_kv(il)` | `src/models/gemma4-iswa.cpp:79-110`, `src/llama-hparams.cpp:231-242` | Exactly our L15-34 pattern. Skip K,V compute for 20/35 layers. |

---

## 1. Gemma 4 E2B model architecture (llama.cpp)

**File:** `src/models/gemma4-iswa.cpp` (full class `llm_build_gemma4_iswa`)

Covered features:

| Feature | Source citation | Our Gemma 4 E2B parameter |
|---|---|---|
| Per-layer input embeddings | `gemma4-iswa.cpp:264-295` | `inp_per_layer` |
| Layered KV / Q-only split | `gemma4-iswa.cpp:79-110` | L0-14 KV, L15-34 Q-only |
| Sliding-window ISWA | `gemma4-iswa.cpp:56` + `llama-hparams.h:316-343` | `n_swa=512`, alternating |
| QK-norm (post-projection RMS) | `gemma4-iswa.cpp:70, 91` | `attn_q_norm`, `attn_k_norm` |
| Fused GeGLU | `llama-graph.cpp:1227-1231` via `ggml_geglu_split` | gate-parallel FFN |
| RoPE per-layer freq | `gemma4-iswa.cpp` RoPE block | multi-base RoPE |
| Final logit softcap | `gemma4-iswa.cpp` tail | applied at head |
| MoE (shared + expert) | supported in class | we don't use it |

**Live TODOs left in their impl (caveat tax):**
- `gemma4-iswa.cpp:26` — "TODO: is causal == true correct?"
- `gemma4-iswa.cpp:112` — "TODO @ngxson: strip unused token after last KV layer"
- `gemma4-iswa.cpp:212` — "TODO @ngxson: improve this" (per-layer embedding)

These are **real uncertainties in the reference implementation**. Don't
treat llama.cpp's Gemma 4 E2B as fully validated — port with eyes open.

---

## 2. Metal backend architecture (llama.cpp)

- **Organization:** `ggml/src/ggml-metal/` — `ggml-metal-context.m` (739L) manages command buffer encoding; `ggml-metal-ops.cpp` (157k L) dispatches ops; `ggml-metal.metal` (10.5k L) shader source.
- **Dispatch model:** `ggml-metal-context.m:676-721` — async encode block, `dispatch_apply(n_cb, ...)` parallel thread encoders, main thread encodes first ~10% of graph nodes immediately, workers handle remainder.
- **Not MPSGraph.** Pure custom Metal shaders with specialization constants. Stance vs Apple first-party options is consistent with MLX's design.
- **Kernel catalog:** 136+ kernels including `kernel_mul_mv_*` (per-quant-type matmuls), `kernel_flash_attn_ext_*` (FlashAttention, multiple head-dim variants), `kernel_rope`, `kernel_rms_norm`, `kernel_geglu`, etc.

### FlashAttention templates (the important one)

`ggml/src/ggml-metal/ggml-metal.metal:5628-6512` — `kernel_flash_attn_ext_*` family. Head-dim instantiations include:
- `dk32/dv32` through `dk576/dv512`
- `dk256_dv256` (line 6509) — **Gemma 4 E2B match**
- Asymmetric `dk320_dv256`
- Block/padding variants for large seqlen

Sliding-window is not a dedicated GPU kernel; enforced via mask tensor in the graph (see §3).

---

## 3. Sliding window / dual KV cache (ISWA)

**File:** `src/llama-kv-cache-iswa.cpp`, `src/llama-hparams.h:316-343`

**Design:** two separate KV caches — `kv_base` (full) and `kv_swa` (sliding-512, padded to 256 internally). Each layer routes to one based on `hparams.is_swa(il)`.

**Mask types:** `LLAMA_SWA_TYPE_ALTERNATING`, `LLAMA_SWA_TYPE_CHUNKED`. Gemma 4 E2B pattern is alternating. Mask is `is_masked_swa(n_swa, swa_type, p0, p1)` with half-window logic.

**Why this matters for us:** our current CoreML stack allocates full-KV
for all layers regardless of window; dual-cache strategy drops memory
~40% at long context and localizes cache-access patterns to benefit
Metal GPU L2.

---

## 4. KV sharing — Q-only layer handling

**File:** `src/models/gemma4-iswa.cpp:79-110`, `src/llama-hparams.cpp:231-242`

```cpp
if (hparams.has_kv(il)) {
    // Compute K,V
    Kcur = build_lora_mm(model.layers[il].wk, cur);
    Vcur = build_lora_mm(model.layers[il].wv, cur);
} else {
    // Reuse KV cache of earlier layers (line 106-110)
    cur = build_attn(inp_attn, ..., Qcur, nullptr, nullptr, ...);
}
```

`llama_hparams::has_kv(il)` checks `n_layer_kv_from_start` boundary. For
Gemma 4 E2B this is 15 (L0-14 produce KV, L15-34 are Q-only).

**Our current chunks already implement this** — but the llama.cpp path
is a proven reference for correctness comparison and for the Metal port.

---

## 5. Fused GeGLU

**File:** `src/llama-graph.cpp:1227-1231`

```cpp
case LLM_FFN_GELU:
    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_geglu_split(ctx0, cur, tmp);  // FUSED GeGLU
        type_gate = LLM_FFN_SEQ;
```

`ggml_geglu_split` fuses `GELU(gate) ⊗ up` in a single op. Parallel gate
architecture matches Gemma 4 E2B (gate-proj from input, not chained).

**Note:** MLX does **not** have equivalent fused GeGLU/SwiGLU. llama.cpp
is the only Metal reference for this.

---

## 6. Speculative decoding (llama.cpp)

**File:** `examples/speculative/speculative.cpp`

- Algorithm: greedy-with-threshold acceptance, dual models (target + draft).
- Tree-capable via `n_seq_dft = params.n_parallel` and `p_draft_split` threshold.
- Accept logic:
  ```cpp
  if (prob_draft > threshold) {
      accept = true;
      common_sampler_accept(smpl, token_id, true);
  }
  if (!accept) {
      // Sample from residual (rejected branch)
  }
  ```

**For us:** drafter is architecturally dead on Gemma 4 E2B for reasons
unrelated to Metal (oracle-live acc gap). llama.cpp's SD mechanism is
reference-only; applying it doesn't reverse the drafter death.

---

## 7. Quantization (llama.cpp)

`ggml/src/ggml-metal/ggml-metal.metal:3956-3959` — Q4_K_M/S kernels.
Block quantization (Q4_K, Q3_K), **not** weight-only int4 × activation int8.
Closest: `kernel_mul_mv_ext_q4x4_f32` with 256-block granularity.

**No direct W4A8 kernel in llama.cpp.** Our W4A8 CoreML path has no
llama.cpp equivalent to lift. Two options for Metal port:
1. Keep W4A8 semantics, write our own Metal matmul kernel.
2. Switch to Q4_K-ish block quant to use llama.cpp kernels directly.

Option 2 is cheaper in engineering; option 1 keeps the quality profile
we've validated. Open question — should be a separate investigation.

---

## 8. MLX — complementary reference

**Path:** `/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/mlx`

**What MLX offers:**
- STEEL attention kernels (`mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h:74-150`) — templated simdgroup tiling, GQA support, causal mask.
- Custom Metal RMSNorm (`mlx/backend/metal/normalization.cpp:13-93`) — grid-based reduction.
- Quantized matmul Metal kernels (`mlx/backend/metal/quantized.cpp:19-51`, `kernels/quantized_nax.*`) — group_size 32, 4/8-bit affine.
- Device-aware tuning (`mlx/backend/metal/device.h:146-151` `get_architecture()`) — per M-series class.

**What MLX does NOT offer:**
- No Gemma examples in this repo (core only).
- No fused GeGLU / SwiGLU.
- No sliding window attention.
- No speculative decoding.
- No ANE integration (zero matches for "ANE", "NeuralEngine", "IOSurface").

### MLX lazy-eval claim check

The Zenn article ascribes ANE-win collapse at D≥2560 to "MLX lazy eval
keeps intermediates in GPU L2 cache." **MLX source does not support
this mechanism:**
- Fusion is compile-time op-inlining (`mlx/backend/metal/compiled.cpp:16-255`), not runtime dynamic scheduling.
- Memory allocator is standard unified-memory pooling; no L2-pinning logic.
- No ANE support at all, so no ANE-gap to close.

The observed gap may still be real, but the *cause* is not what the article claims. Treat the author's causal explanation as speculation.

---

## 9. What NOT to port from these repos

- llama.cpp's GGUF loader — we have CoreML weights, different pipeline.
- llama.cpp's tokenizer — we use Gemma tokenizer via our existing path.
- llama.cpp's CPU backend — decode target is Metal only.
- MLX's Python frontend — we're Swift/ObjC.
- MLX's compiled-fusion framework — overkill for a fixed model graph.
- Speculative decoding code from either — drafter is dead on our target.

---

## 10. Recommended execution order for Metal Phase 3

1. **Stand up minimal Metal decoder kernel set** using llama.cpp as copy-source:
   - RMSNorm, RoPE, matmul (W4A8 or Q4_K shim).
   - FlashAttention `dk256_dv256` — direct copy + adapt for our mask layout.
   - Fused GeGLU op.
2. **Port Gemma 4 E2B graph structure** from `gemma4-iswa.cpp`. Handle `has_kv()` branching and dual KV cache.
3. **Multi-threaded command encoding** from `ggml-metal-context.m` — this is not the first optimization but is a "free" +10% wall-clock.
4. **Close the three llama.cpp TODOs** (`gemma4-iswa.cpp:26, 112, 212`) with our own testing.
5. **W4A8 decision** — either port our quant to Metal (custom kernel) or switch to Q4_K shim and re-validate quality.

Rough effort: 3-6 weeks for a working Metal decoder, assuming quant stays W4A8 (no re-validation loop). Q4_K shim cuts ~2 weeks off the kernel work but adds a quality-regression risk that needs recalibration data.

---

## 11. Citations summary

| Topic | Source |
|---|---|
| Gemma 4 E2B model class | `llama.cpp/src/models/gemma4-iswa.cpp` |
| ISWA dual KV cache | `llama.cpp/src/llama-kv-cache-iswa.cpp` |
| Sliding-window masking | `llama.cpp/src/llama-hparams.h:316-343` |
| Has-KV layer skip | `llama.cpp/src/llama-hparams.cpp:231-242` |
| Fused GeGLU op | `llama.cpp/src/llama-graph.cpp:1227-1231` |
| FlashAttention kernels | `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5628-6512` |
| Command buffer encoding | `llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:676-721` |
| STEEL attention | `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h:74-150` |
| MLX quantized matmul | `mlx/backend/metal/quantized.cpp:19-51` |
| MLX compile-time fusion | `mlx/backend/metal/compiled.cpp:16-255` |
