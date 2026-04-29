# Metal + ANE kernel patterns — reusable catalog

**Date:** 2026-04-22
**Sources (source-read):** llama.cpp, whisper.cpp, MLX, Apple ml-ane-transformers, maderix/ANE.

## Organization

- §1: Attention kernels (FlashAttention + variants)
- §2: Normalization kernels (RMSNorm, LayerNorm)
- §3: Quantized matmul
- §4: Activation / FFN fusion
- §5: RoPE variants
- §6: Softmax + numerical stability
- §7: Threadgroup & simdgroup patterns
- §8: Command buffer / dispatch
- §9: Memory layout & IOSurface
- §10: Apple's ANE optimization principles

---

## 1. Attention kernels

### 1.1 FlashAttention on Metal (llama.cpp) — most directly portable

**File:** `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5628-6512`.

- `kernel_flash_attn_ext_*` family.
- Head-dim templates: `dk32`, `dk40`, `dk48`, `dk64`, `dk72`, `dk80`, `dk96`, `dk112`, `dk128`, `dk192`, `dk256`, `dk320`, `dk512`, `dk576` (asymmetric dv also supported).
- **Matching for Gemma 4 E2B:** `dk256_dv256` (line 6509) for sliding layers, `dk512_dv512` for full-attention layers.
- **Asymmetric:** `dk320_dv256` shows how to template uneven Q vs V dim.
- Block/padding variants for large seqlen: `kernel_flash_attn_ext_blk` (line 5477-5530) pre-scans mask to tag blocks as (masked=0, partial=1, all-zero=2); skip full computation on masked blocks.

### 1.2 Whisper.cpp's FlashAttention with dequant-on-load

**File:** `whisper.cpp/ggml/src/ggml-metal/ggml-metal.metal:6325-6403`.

Unified `kernel_flash_attn_ext` template with 16 K/V quantization types: f16/f32 for Q, q4_0/q4_1/q5_0/q5_1/q8_0 for K, V. Inline dequant inside the attention loop, no separate kernel pass. **Pattern:** `template<typename q_t, typename k_t, typename v_t, ...>` with per-type dequant functions.

**Value for us:** Saves KV cache memory (KV in INT8/INT4 while compute in FP16). If we ship a Metal path and want to shrink 8K-context memory, this is the pattern.

### 1.3 MLX STEEL attention kernels

**File:** `mlx/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h:74-150`.

- Simdgroup tile-based (BQ=64, BK=32, BD varies).
- GQA support via `num_q_heads / num_kv_heads` ratio (line 108).
- Causal mask at line 179-187.
- **Sink token** support (prepended always-attendable token; used in some long-context models).

**Value for us:** MLX STEEL is an alternative Metal implementation. Higher-level than llama.cpp's raw kernel. Use as cross-reference for correctness.

### 1.4 Attention in Apple's ml-ane-transformers

**File:** `ane_transformers/reference/multihead_attention.py:64-120`.

- `einsum('bchq,bkhc->bkhq')` — channel-first notation.
- Transposes K only (line 87: `transpose(1,3)`) — minimizes reshapes.
- Additive mask **before** softmax (line 100-105).
- Additive mask value: **-1e4** (not -inf) — FP16 safety.

**Value for us:** ANE-specific. Confirms the `[1, C, 1, S]` channel-first layout. Confirms -1e4 masking convention.

### 1.5 maderix's decomposed attention (private API path)

**File:** `maderix-ANE/training/test_ane_causal_attn.m:82-125`.

- Non-causal SDPA on ANE as baseline.
- Causal path: Q@Kᵀ on ANE → CPU mask+softmax → scores@V on ANE.

**Value for us:** DO NOT follow. Fallback pattern for Private API hand-rolled MIL. On CoreML-converted path (ANEMLL), explicit causal mask input works on ANE without decomposition.

---

## 2. Normalization kernels

### 2.1 RMSNorm on ANE — the concat trick (ANEMLL)

**File:** `Anemll/anemll/models/gemma3_model.py:264, 308`.

```python
doubled = torch.cat([x, -x], dim=-1)   # zero-mean by construction
normed = F.layer_norm(doubled, ...)    # reuse ANE LayerNorm kernel
normed = normed[..., :hidden_size]     # drop mirror half
return normed * (1.0 + self.weight)    # Gemma scaling
```

**Value for us:** Already in our `ane_ops.py`. Reuses ANE's optimized LayerNorm instead of manual `rsqrt` chain. `rsqrt` SIGSEGVs on some ANE targets per Zenn.

### 2.2 RMSNorm auto-conversion in coremltools (with overflow prevention)

**File:** `coremltools/converters/mil/frontend/torch/ops.py:3107-3171`.

coremltools 2024+ converts `torch.nn.RMSNorm` with **max-value scaling** to prevent FP16 overflow:
```
max_val = reduce_max(abs(x))
x_scaled = x / max_val
rms = rsqrt(mean(x_scaled²) + eps)
out = x_scaled * rms * max_val * weight
```

Comment in source: *"Apple Neural Engine (ANE) does not have native RMSNorm support"* — the max-scaling trick is to prevent overflow during the `x²` step.

**Value for us:** If we move away from the concat([x,-x]) trick and use PyTorch `nn.RMSNorm`, coremltools handles ANE-safety automatically. Worth A/B test against our current pattern.

### 2.3 Apple's LayerNormANE

**File:** `ane_transformers/reference/layer_norm.py:10-79`.

- `clip_mag` clamps to `[-clip_mag, clip_mag]` before squaring (line 62-63).
- Reduces over channels only (dim=1), not full reduction.
- Output: `bias + scale` order (inverted vs torch.nn.LayerNorm for ANE fusion).
- eps=1e-7 (not 1e-12) — FP16 numerical stability.

### 2.4 RMSNorm on Metal

**File:** `llama.cpp/ggml-metal.metal:2816-2983` (`kernel_norm_fuse_impl<type, N_ops>`).

Template parameter `N_ops`:
- `N_ops=1`: norm only.
- `N_ops=2`: norm + multiply (for scale).
- `N_ops=3`: norm + multiply + add.

Fused in single kernel — reduces dispatches.

**Value for us:** For Metal port, this is the direct reference for Gemma 4's sandwich norms (4 per layer).

### 2.5 MLX RMSNorm

**File:** `mlx/mlx/backend/metal/normalization.cpp:13-93`.

Grid-based reduction. Simpler than llama.cpp's fused variant but less op coverage.

---

## 3. Quantized matmul Metal kernels

### 3.1 Block-quantization layouts (llama.cpp)

- **q4_0:** 32-val block, 2 bytes scale+zero (f16), 16 bytes nibbles. 0.5 byte/val. `struct block_q4_0 { f16 d, m; u8 qs[16]; }`.
- **q4_K:** 256-val block, 192 bytes (K-means super-block, 6 scales per block).
- **q8_0:** 32-val block, 4 bytes scale. 1.125 byte/val.
- **q5_0 / q5_1:** 32-val block with extra bits for 5-bit. Same footprint as q4 + 1 bit.
- **mxfp4 / iq2_xxs / iq3_xxs / iq4_nl:** non-uniform quant schemes, various block sizes.

### 3.2 Dequant-on-load (not pre-pass)

**Pattern:** Matmul kernels read quantized block header (scale, zero), unpack nibbles inline, multiply-accumulate in FP. No separate "dequant all weights then matmul" pass.

**Citation:** `llama.cpp/ggml-metal.metal:1614-1649` (`kernel_mul_mat_q_f32`).

**Value for us:** The gold pattern for a W4-equivalent Metal matmul. Our W4A8 CoreML path has W4 weights dequantized in MIL via `constexpr_affine_dequantize`; a native Metal port should preserve on-the-fly dequant.

### 3.3 MLX quantized matmul

**File:** `mlx/mlx/backend/metal/quantized.cpp:19-51`, `kernels/quantized_nax.metal`.

- `get_quantized_kernel_wrapped()` dispatches based on `group_size, bits, type`.
- Supports q4_0, q8_0, affine mode.
- Device-aware batch limits (`qmv_batch_limit` different for M3 gen 13 vs M4 gen 14).

### 3.4 Quantization in ExecuTorch

**File:** `executorch/kernels/quantized/quantized.yaml:37-71`.

`embedding_byte`, `embedding_2bit`, `embedding_4bit` — per-channel embedding quantization with runtime dequant. ExecuTorch's approach: pre-quant embedding lookup, dequant inline.

**Value for us:** Our embedding is INT8 per-row. ExecuTorch goes further (4-bit / 2-bit). If embedding memory pressure is ever an issue (currently fine), consider 4-bit.

---

## 4. Activation / FFN fusion

### 4.1 Fused GeGLU on Metal (llama.cpp)

**File:** `llama.cpp/src/llama-graph.cpp:1227-1231`.

```cpp
case LLM_FFN_GELU:
    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_geglu_split(ctx0, cur, tmp);  // single op
```

Fuses `GELU(gate) ⊗ up` into one kernel. Matches Gemma's parallel-gate architecture.

**Value for us:** Direct port candidate. Our CoreML path splits gate/up/activation/multiply into separate MIL ops.

### 4.2 MLX does NOT fuse GeGLU/SwiGLU

**File:** `mlx/mlx/backend/metal/` — no fused variant found. Adjacent ops via compile-time fusion (`compiled.cpp:16-255`), not a dedicated GeGLU kernel.

**Value for us:** MLX is weaker on GeGLU fusion than llama.cpp.

### 4.3 ml-ane-transformers FFN

**File:** `ane_transformers/reference/ffn.py:14-21`.

```
Conv2d(embed → ffn, 1) → ReLU → Dropout → Conv2d(ffn → embed, 1)
```

All-Conv2d (no nn.Linear). ReLU (not GeGLU; pre-Gemma).

**Value:** Confirms Conv2d(1×1) for projections. For GeGLU, adapt with gate branch.

---

## 5. RoPE variants

**File:** `llama.cpp/ggml-metal.metal:4143-4413`.

- `kernel_rope_norm` — standard Roformer / LLaMA RoPE.
- `kernel_rope_neox` — GPT-NeoX halves convention.
- `kernel_rope_multi` — multiple freq scales (Gemma 4 uses this concept for per-layer-type RoPE).
- `kernel_rope_vision` — 2D RoPE for vision models.

Per-token rotation inline in kernel; no separate lookup.

**Value for us:** Gemma 4's dual RoPE (10K for sliding, 1M for full) maps to `kernel_rope_multi` or two separate `rope_norm` passes with different freq tables. Our current CoreML path builds cos/sin tables on CPU per decode step; a Metal port could do on-the-fly rotation in kernel.

---

## 6. Softmax + numerical stability

**File:** `llama.cpp/ggml-metal.metal:1562-1588`.

Two-pass softmax with max-subtract:
1. Pass 1: reduce max over row (simdgroup reduction, then threadgroup barrier).
2. Pass 2: compute `exp(x - max)`, reduce sum.
3. Pass 3: divide.

**Numerical stability:** max-subtract prevents `exp()` overflow. Critical for FP16 attention scores on long sequences.

**Additive mask convention:** Use `-1e4` (Apple pattern), NOT `-inf`. `-inf` propagates to NaN through `exp(-inf + finite) = 0 * something`.

---

## 7. Threadgroup & simdgroup patterns

### 7.1 Simdgroup reductions

**File:** `whisper.cpp/ggml-metal.metal:6512-6520`.

Metal primitives:
- `simd_min(x)`, `simd_max(x)` — horizontal tree-reductions across 32-wide groups.
- `simd_sum(x)` — for softmax denominator.
- `simd_shuffle(x, lane)` — exchange values within simdgroup.

**Pattern:** Use simdgroup reductions first (no barrier cost), then threadgroup-level atomic/barrier for cross-simdgroup aggregation.

### 7.2 Function constants for specialization

**File:** `whisper.cpp/ggml-metal.metal:2816, 5532-5546`.

```metal
[[function_constant(FC_FLASH_ATTN_EXT_HAS_MASK)]]
constant bool has_mask;
```

Parameterizes kernel behavior without runtime cost (branches removed at compile time). Runtime specialization via `MTL::FunctionConstantValues`.

**Value for us:** Essential for supporting both sliding and full-attention with one kernel source (different head_dims via constants).

### 7.3 Tile sizes for Metal matmul

**File:** `whisper.cpp/ggml-metal.metal:3249-3350`.

- `FC_mul_mv_nsg` — simdgroup count.
- `FC_mul_mv_nxpsg` — threads per simdgroup (usually 32).
- Per-arch tile tuning via `get_architecture_gen()`.

---

## 8. Command buffer / dispatch

### 8.1 Multi-threaded command encoding (llama.cpp)

**File:** `llama.cpp/ggml/src/ggml-metal/ggml-metal-context.m:676-721`.

```c
dispatch_apply(n_cb, ...) {
    // Per-thread encoder: encode subset of graph nodes into its own command buffer
}
```

Main thread encodes first ~10% immediately, workers handle remainder. **Free +10% wall-clock on any non-trivial graph.**

### 8.2 Encoder pipeline (whisper.cpp)

Metal shader library **embedded in binary** (`GGML_METAL_EMBED_LIBRARY=ON`). No runtime `.metallib` lookup, faster first load.

### 8.3 MTL::CommandQueue per stream (MLX)

**File:** `mlx/mlx/backend/metal/eval.cpp:30-75`.

One `MTL::CommandQueue` per stream index. Streams are application-managed; no cross-stream sync by default.

---

## 9. Memory layout & IOSurface

### 9.1 IOSurface zero-copy (ANEMLL)

**File:** `Anemll/anemll-swift-cli/Sources/AnemllCore/InferenceManager.swift:330-335, 903-915`.

`MLMultiArray` backed by `CVPixelBuffer` with `kCVPixelBufferIOSurfacePropertiesKey + kCVPixelBufferMetalCompatibilityKey`. Zero CPU copies when ANE reads/writes.

**Our stack has this** (`ChunkedEngine.swift:546-568`).

### 9.2 Buffer pooling — ring + ping-pong (ANEMLL)

**File:** `Anemll/anemll-swift-cli/Sources/AnemllCore/InferenceManager.swift:55-60, 99-100`.

- Monolithic models: ring buffer depth 16.
- Chunked models: ping-pong depth 2.
- Prevents ANE async write collisions.
- Serial `DispatchQueue` gates prediction.

**We don't have this.** Potential +stability, maybe +throughput on multi-chunk decode.

### 9.3 Channel-first `[1, C, 1, S]`

Apple ANE requirement. BC1S layout. All `ml-ane-transformers` reference code assumes this. Our conversion enforces implicitly via `Conv2d(1×1)` substitutions.

---

## 10. Apple's ANE optimization principles (ml-ane-transformers)

Three numbered principles from the repo:

1. **Channel-First Layout** (BC1S): channels map to ANE execution parallelism. All internals use `[B, C, 1, S]`.
2. **Chunking Large Intermediates**: break very-large intermediates into tiles. Relevant for attention with large seq length.
3. **Minimizing Memory Copies**: Conv2d over Linear; additive masking not concat-based; fuse norm + scale; etc.

Unwritten but consistent across repo:
4. Additive mask with `-1e4`, not `-inf`, for FP16 safety.
5. Position embeddings **passed as separate input** to decoder, not computed inside layers.
6. No dynamic shapes — all statically traced.
7. Dropout → Identity for inference (eval mode).
8. `LayerNormANE`'s `bias + scale` order inverted vs PyTorch for ANE fusion.

---

## 11. Direct-copy priority list for a Metal port of Gemma 4 E2B decode

Ranked by impact × easiness:

| # | Pattern | Source | Effort |
|---|---|---|---|
| 1 | `kernel_flash_attn_ext_f32_dk256_dv256` (sliding layers) | `llama.cpp/ggml-metal.metal:6509` | 1 day (copy + adapt) |
| 2 | `kernel_flash_attn_ext_f32_dk512_dv512` (full-attention layers) | same repo, same family | 1 day |
| 3 | `ggml_geglu_split` fused GeGLU | `llama.cpp/src/llama-graph.cpp:1227-1231` | 0.5 day |
| 4 | `kernel_norm_fuse_impl<type, 3>` (RMSNorm + scale + add) | `llama.cpp/ggml-metal.metal:2816-2983` | 0.5 day |
| 5 | Multi-threaded `dispatch_apply` encoder | `llama.cpp/ggml-metal-context.m:676-721` | 1 day |
| 6 | Dequant-on-load matmul for W4 (or Q4_K shim) | `llama.cpp/ggml-metal.metal:1614-1649` | 3-5 days (custom W4A8 or Q4_K port) |
| 7 | RoPE multi-freq kernel | `llama.cpp/ggml-metal.metal:4143-4413` | 1 day |
| 8 | Softmax with max-subtract + `-1e4` mask | `llama.cpp/ggml-metal.metal:1562-1588` | 0.5 day |
| 9 | `GGML_METAL_EMBED_LIBRARY` in build | `whisper.cpp/build-xcframework.sh` | 0.5 day |
| 10 | Function constants for head-dim specialization | `whisper.cpp/ggml-metal.metal:5532-5546` | 0.5 day |

Total effort bundled: ~10-14 days for a minimal Metal decoder. This matches our `METAL_PORT_REFERENCE.md` estimate of 3-6 weeks when wiring + integration + testing are added.
