# coremltools internals + iOS 18 opportunities

**Date:** 2026-04-22
**Source:** `/Users/majimadaisuke/Downloads/workspace/repo-review/coremltools/` + WWDC24 session 10159

## 0. TL;DR — what we're probably leaving on the table

1. **iOS 18 native SDPA MIL op** — we still use manual `ane_softmax`. Our comment says SDPA caused wrong tokens; verify that was a pre-iOS-18 issue.
2. **iOS 18 stateful `read_state` / `write_state`** — could replace our explicit `copyBack` memcpy.
3. **iOS 18 per-block INT4** (block_size=32) — granularity finer than our default; ~500 MB savings on E4B bundle.
4. **Joint compression (prune + palettize)** — we don't combine static sparsity with quantization. Could stack.
5. **`@_multifunction_unsupported` in PTQ** — known blocker for our `verify_qK` multi-function packages. Need pre-export quantization workaround.

## 1. MIL pass pipeline (DEFAULT)

**File:** `coremltools/converters/mil/mil/pass_pipeline.py`

**123 passes in strict order** (lines 24-124). Relevant groupings:

### 1.1 Fusion passes (early, pre-quant)

- `fuse_conv_bias`, `fuse_linear_bias` — merge bias constants.
- `fuse_conv_batchnorm`.
- `common::fuse_layernorm_or_instancenorm` (`/mil/passes/defs/optimize_normalization.py:22-58`) — detects 5 patterns of `reduce_mean → sub → square → mean → rsqrt → mul/add` and fuses into single `layer_norm` op.
- `fuse_gelu_tanh_approximation`, `fuse_gelu_exact`, `fuse_gelu_sigmoid_approximation` (`/optimize_activation.py`) — erf/tanh patterns to single Gelu op.
- `reduce_transposes` — eliminates redundant transposes via shape inference.

### 1.2 Quantization passes (three-stage)

- Stage 1 (canonicalization): `int_op_canonicalization`, `nullify_redundant_quantization_zero_point`.
- Stage 2 (lowering): `dequantize_quantize_pair_elimination`, `distributive_quantized_binary_op_scale_normalization`.
- Stage 3 (constexpr): `dequantize_to_constexpr`, `canonicalize_quantized_lut_pattern`.

### 1.3 State handling (late)

- `canonicalize_inplace_pattern` — reorders `read_state → op → coreml_update_state`.
- `prefer_state_in_downstream` — replaces functional op outputs with state reads.

Runs at the **end** of DEFAULT pipeline (lines 120-122).

### 1.4 Cleanup (runs 5+ times)

- Dead code elimination, const elimination, op elimination — after each fusion pass that removes ops.

## 2. RMSNorm auto-conversion with ANE overflow prevention

**File:** `coremltools/converters/mil/frontend/torch/ops.py:3107-3171`.

Since 2024+, `torch.nn.RMSNorm` auto-converts to an ANE-safe pattern:

```python
max_val = reduce_max(abs(x), keepdims=True)
x_scaled = x / max_val
rms = rsqrt(reduce_mean(x_scaled * x_scaled) + eps)
out = x_scaled * rms * max_val * weight
```

Author comment: *"Apple Neural Engine (ANE) does not have native RMSNorm support."*

**Error bound:** < 0.1% in practice.
**Trade:** minimal numerical drift for ANE placement.

**Implication for us:** Our `concat([x,-x]) → LayerNorm → slice` trick is one valid path; coremltools' auto-converted RMSNorm is another. A/B test worth doing.

## 3. Quantization methods available

### 3.1 Palettization (LUT quantization)

**File:** `coremltools/optimize/coreml/_post_training_quantization.py:188-250`.

```python
from coremltools.optimize.coreml import OpPalettizerConfig, palettize_weights
config = OpPalettizerConfig(nbits=4, mode="kmeans")
mlmodel = palettize_weights(mlmodel, config)
```

- LUT of 2^nbits entries (4-bit = 16, 8-bit = 256).
- Modes: `"uniform"` (linear histogram), `"kmeans"` (clustering), `"unique"`.
- Granularities: `PER_TENSOR`, `PER_GROUPED_CHANNEL`, `PER_CHANNEL`, `PER_BLOCK`.
- Output MIL op: `constexpr_lut_to_dense`.

Our default: `PER_GROUPED_CHANNEL`, `group_size=32`, `nbits=4` (INT4 LUT).

### 3.2 Linear quantization

**File:** `_post_training_quantization.py:56-97`.

- Modes: `"linear"` (asymmetric) / `"linear_symmetric"` (symmetric, recommended for weights).
- INT4 / INT8 support.
- Output MIL op: `constexpr_affine_dequantize`.

Our W8A8 path uses this (`build_w8a8_proper.py:186-196`).

### 3.3 Joint compression (prune + palettize)

`_post_training_quantization.py:228` — `joint_compression=True` param.

**Sequence:**
1. First prune (structured or unstructured sparsity).
2. Then palettize on non-zero values.
3. Result: sparse + low-precision. Output MIL op: `constexpr_lut_to_sparse + constexpr_sparse_to_dense`.

**Potential savings for us:** 10× compression (sparse 40-50% × 4-bit LUT). Quality cost needs testing.

### 3.4 Torch-side K-Means (sensitivity-aware)

**File:** `coremltools/optimize/torch/palettization/sensitive_k_means.py`.

Pre-export quantization. Deprioritizes outliers. Supports FSDP training.

**Relevant for us:** Future V6-6 SpinQuant + V7-1 COMPACT pipeline might benefit.

## 4. iOS 18 features

### 4.1 Native SDPA MIL op

**File:** `coremltools/mil/ops/defs/iOS18/transformers.py:18-167`.

```python
scaled_dot_product_attention(Q, K, V, attn_mask=None, is_causal=False)
```

- Q `[B, ..., L, E]`, K/V `[B, ..., S, E]` → `[B, ..., L, EV]`.
- Value inference: `Q @ K.T / sqrt(E) → softmax → @ V`.
- Mask: bool (converted to -inf) or float (additive).
- **Does NOT support** dropout_p or `is_causal=True` kwarg (use mask instead).

**For us:**
- Convert `torch.nn.functional.scaled_dot_product_attention` and coremltools auto-detects iOS 18 target → native SDPA op.
- Our current comment says SDPA caused wrong predictions — but that was likely on iOS 17 path (manual decomposition).
- **Worth retesting on iOS 18 native op before dismissing.**

### 4.2 Stateful models — `read_state` / `write_state`

**File:** `coremltools/mil/ops/defs/iOS18/states.py`.

iOS 18+ supports KV cache as a graph-level state:

```python
k_cache = read_state("k_cache")        # read from MLState
k_cache_new = concat([k_cache, new_k], axis=seq_dim)
coreml_update_state(k_cache_new, "k_cache")  # write back
```

MIL pipeline reorders: `read_state → op → coreml_update_state` via `canonicalize_inplace_pattern` pass.

**Potential replacement for our `copyBack` path:** Currently we memcpy from chunk output to KV MLMultiArray in Swift. If we express KV as state, CoreML handles in-graph update, eliminating CPU-side copy.

**Caveat:** MLState is iOS 18+ only; iOS 17 compatibility requires dual-path.

### 4.3 Per-block 4-bit palettization

Per WWDC24 session 10159:
- iOS 18 adds block-granularity INT4 palettization (block_size=32).
- Better for large layers (lm_head, embedding tables benefit most).
- Old iOS 17 path: per-tensor or per-channel.

**Potential savings:** Our E4B bundle is 5.5 GB. Per-block INT4 on lm_head alone could cut ~500 MB.

### 4.4 MLProgram format preferred

iOS 18+ prefers ML Programs (stateless compute graphs) over legacy stateful MLModels. MLState is the bridge for stateful needs.

## 5. Torch frontend → MIL

### 5.1 TorchExport support (2024+)

Alternative frontend to TorchScript. Faster, more reliable, handles `torch.nn.functional` ops directly.

**Our pipeline:** uses TorchScript path (via `convert()` tracing). Worth evaluating TorchExport for simpler graph construction.

### 5.2 Ops supported

- Transformer ops: Linear, Conv2d, Conv1d, RMSNorm (auto-converted), LayerNorm, BatchNorm.
- Attention: manual matmul + softmax (iOS 17 and earlier) or native SDPA (iOS 18).
- Quantization ops: `torch.ops.aten.quantize_per_tensor`, `dequantize_tensor` → `constexpr_affine_quantize` / `dequantize`.
- GELU variants: tanh approximation, exact erf, sigmoid approx.

### 5.3 Shape inference — symbolic shapes supported

`Symbol` type for unknown dims. `is_symbolic()` checks.

**Our pipeline:** uses fixed shapes (context_length=512 in config). No dynamic sequence length in CoreML graph — shape buckets managed externally.

## 6. Hard gotchas

### 6.1 `@_multifunction_unsupported` for PTQ

**File:** `coremltools/optimize/coreml/_post_training_quantization.py:47-55`.

PTQ functions (palettize_weights, linear_quantize_weights, joint compression, pruning) do **NOT** support multifunction models.

**Problem for us:** `build_verify_chunks.py` produces multifunction packages (decode_q1 + verify_qK). PTQ on these fails.

**Workarounds:**
- Apply quantization **before** assembling multifunction (on single-function artifacts, then combine).
- OR use pre-export torch-side quantization (`torch.ao.quantization` or coremltools torch-side PTQ).
- OR split verify_qK into separate artifact (adds runtime load).

We need to check which our pipeline does.

### 6.2 compute_units is implicit, not explicit branching

coremltools does **not** branch on `compute_units='ane'` in the converter. ANE placement is implicit:
- Converter chooses MIL ops and layouts.
- Runtime (Core ML framework) decides ANE vs CPU/GPU based on op support + layout.

**Implication:** To force ANE, make ops ANE-compatible (Conv2d, LayerNorm, no dynamic shapes). Runtime then places them on ANE. `compute_units='.cpuAndNeuralEngine'` is a hint, not a guarantee.

### 6.3 Python inference path

`coremltools.models.MLModel.predict()`:
- CPU-only on macOS.
- **No ANE access from Python.**
- Used only for validation before device deploy.

Our `compare_bf16_vs_w4a8.py` runs in Python — it's comparing CPU-dequantized W4A8 vs BF16 CPU, not actual ANE fp16 output. Device-side validation is needed for final accept-rate numbers.

## 7. Actionable roadmap items

### 7.1 Re-test iOS 18 native SDPA (1-2 days)

- Build Gemma 4 E2B target with `torch.nn.functional.scaled_dot_product_attention` + iOS 18 minimum target.
- Run on device, check token argmax agreement vs current manual attention.
- If it matches: adopt (simpler MIL graph, fewer ops, native kernel).
- If it diverges: we have source-level evidence why, update our `models/gemma4_swa_chunks.py:136-138` comment with specifics.

### 7.2 iOS 18 MLState KV cache migration (3-5 days)

- Convert one chunk with explicit `read_state` / `write_state` for KV.
- Benchmark Swift-side: does `copyBack` time go to 0?
- Validate correctness against current CPU-copy path.
- If wins: cascade migration. If loses (e.g., read/write overhead exceeds copyBack): document why.

### 7.3 Per-block INT4 palettization for lm_head + embed (1-2 days)

- Apply `PER_BLOCK` granularity (block_size=32) to `lm_head` weight in chunk4.
- Measure accuracy delta (argmax agreement, perplexity).
- Measure size reduction on E2B + E4B bundles.
- Decide: adopt if quality cost < 0.1 PPL.

### 7.4 Joint prune + palettize POC (1 week)

- After COMPACT (V7-1) lands, try `palettize_weights(..., joint_compression=True)` on the pruned model.
- Measure: quality retention vs total size.
- Risk: training needed to re-learn weights after dual compression.

### 7.5 Multifunction PTQ workaround (2 days)

- Find out whether our current `build_w8a8_proper.py` or `build_verify_chunks.py` already handles multifunction specially.
- If yes: document. If no: design a split-then-combine pipeline that quantizes per-function then assembles.

## 8. Citations (all in `coremltools/`)

- `converters/mil/mil/pass_pipeline.py` (pipeline definition, 123 passes)
- `converters/mil/mil/passes/defs/optimize_normalization.py:22-58` (LayerNorm fusion)
- `converters/mil/mil/passes/defs/optimize_activation.py` (GELU fusion)
- `converters/mil/mil/passes/defs/optimize_state.py:18-210` (MLState canonicalization)
- `converters/mil/frontend/torch/ops.py:3107-3171` (RMSNorm with max-val scaling)
- `converters/mil/frontend/torch/quantization_ops.py` (torch quant ops)
- `mil/ops/defs/iOS18/transformers.py:18-167` (native SDPA)
- `mil/ops/defs/iOS18/states.py` (read_state / write_state)
- `optimize/coreml/_post_training_quantization.py:47-55, 56-97, 188-250` (PTQ core)
- `optimize/torch/palettization/sensitive_k_means.py` (pre-export k-means)
