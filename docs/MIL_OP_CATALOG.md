# MIL Operation Catalog — coremltools 9.0, ANE placement map

Date: 2026-04-16
Sources: `coremltools` tag `9.0` / HEAD `8147ec1` cloned at `/tmp/ct9`,
Apple *Deploying Transformers on the Apple Neural Engine* (2022), Orion
(arXiv 2603.06728), and the empirical `docs/MLPACKAGE_STRUCTURE_AUDIT.md` /
`docs/ANE_CONVERSION_RECIPE_2026.md` in this repo.

The goal of this document is to enumerate every MIL op available in
coremltools 9.0, mark which ones have a dedicated ANE kernel on A17/A18/A19,
and call out the ops the Gemma 4 E2B pipeline **should adopt** or **should
drop**. Use it alongside `docs/MIL_PASSES_ADDITIONAL.md` (which covers
*passes* rather than *ops*).

---

## 1. Executive summary — the top five missing wins

| # | Missing op | Replaces | Gain | File to edit |
|---|---|---|---|---|
| 1 | `scaled_dot_product_attention` (iOS18) | `matmul → add_mask → [ane_softmax=5 ops] → matmul` per attention site | Fuses 8–12 ops → 1 native ANE kernel. Unlocks `scaled_dot_product_attention_sliced_q` pass (ref: `coremltools/converters/mil/mil/passes/defs/transformer.py:20`). 20–35 % prefill win once seq ≥ 256. | `conversion/models/gemma4_swa_*.py`, `conversion/ane_ops.py:216` (keep `F.scaled_dot_product_attention` path active; the rewrite in `gemma4_swa_cascading.py:218` is the template) |
| 2 | `softmax` (iOS15, ct canonical) | `ane_softmax` — `reduce_max + sub + exp + reduce_sum + real_div` (5 ops × ≈10 sites × 11 layers/chunk) | 35 softmax call sites per decoder chunk become 1 op each. Proven ANE-native: `mtp_drafter.mlpackage` already uses the fused op (audit §2.4). | `conversion/ane_ops.py:216`: switch `ane_softmax` default to `ane_fused_softmax` (already defined at `:244`) |
| 3 | `slice_update` (iOS18) | `concat + slice_by_index` KV-append triad | One native write-in-place op instead of copy-concat-slice; removes ≈2 ops per layer × 35 layers × 2 caches = 140 ops model-wide. Also the canonical shape for ANE in-place KV (`docs/ANE_CONVERSION_RECIPE_2026.md` §3). | `conversion/models/gemma4_swa_chunks.py` KV append block |
| 4 | `constexpr_blockwise_shift_scale` (iOS18) | `constexpr_lut_to_dense` (iOS16 path) for INT4 block quant | The iOS18 op is the **native** hardware path for `per_block(block_size=32)` linear-symmetric quant; Apple's WWDC24 tutorial emits it. Audit §2 shows we currently emit **iOS16** `constexpr_lut_to_dense` — i.e. ct converted our INT4 palette as a LUT, not as block-shift-scale. The kernels are different on A18/A19 ANE. | `conversion/convert.py` / `build_verify_chunks.py`: set `minimum_deployment_target=ct.target.iOS18` (or iOS26) **before** calling `linear_quantize_weights(mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32)` |
| 5 | `read_state` + `coreml_update_state` (iOS18) | Multi-Array KV I/O (declared `"stateless": true`) | Apple measured 13× (Llama 3.1) by switching from I/O KV to stateful KV. Our production path is still I/O; stateful code exists in `stateful_chunk2.mlpkg` example but isn't shipped. See `docs/D5_STATEFUL_KV_IOS26.md`. | `build_merged_chunks.py` wire `states=[ct.StateType(...)]` |

Items 1–3 compound: enabling `softmax` is a prerequisite for `scaled_dot_product_attention` (the converter pattern-matches `matmul → softmax → matmul`; it will not match `matmul → decomposed_softmax → matmul`), and `scaled_dot_product_attention` is a prerequisite for the `scaled_dot_product_attention_sliced_q` pass.

---

## 2. Complete coremltools 9.0 MIL op catalog

Columns: **Op** (canonical MIL name) · **min iOS** (first opset the op registered for) · **ANE** (Y = dedicated kernel, ≈ = via conv/layernorm emulation, N = CPU/GPU fallback, ? = unverified) · **Notes / file:line in `/tmp/ct9/coremltools/converters/mil/mil/ops/defs/`**.

### 2.1 Activation (`iOS15/activation.py`, `iOS17/activation.py`)

| Op | min iOS | ANE | Notes |
|---|---|---|---|
| `clamped_relu` | 15 | Y | `iOS15/activation.py:57` |
| `elu` | 15 | Y | `iOS15/activation.py:86` |
| `gelu` | 15 | Y | `iOS15/activation.py:113` — `mode ∈ {EXACT, TANH_APPROXIMATION, SIGMOID_APPROXIMATION}`. Gemma 4 uses `TANH_APPROXIMATION`; PyTorch `F.gelu(approximate="tanh")` lowers to this mode. |
| `leaky_relu` | 15 | Y | `iOS15/activation.py:189` |
| `linear_activation` | 15 | Y | scale + bias; fused with conv |
| `prelu` | 15 | Y | `iOS15/activation.py:242` |
| `relu` / `relu6` | 15 | Y | fused with conv via `fuse_activation_into_conv` |
| `scaled_tanh` | 15 | Y | |
| `sigmoid` / `sigmoid_hard` | 15 | Y | native; often fused with prior mul |
| **`silu`** | 15 | Y | `iOS15/activation.py:430`. Native on ANE; `x * sigmoid(x)` is a single op. *(Gemma 4 uses GELU-tanh, so this does not apply to our model; relevant for Qwen / Mistral.)* |
| **`softmax`** | 15 | Y | `iOS15/activation.py:527`. One op. Fused with preceding `matmul` + `add_mask` into ANE SDPA block. **Currently not emitted by decoder chunks** (decomposed) — §4. |
| `softplus` / `softplus_parametric` | 15 | Y | |
| `softsign` | 15 | Y | |
| `thresholded_relu` | 15 | Y | |
| iOS17 updated variants | 17 | Y | fp16 type domain tightened |

### 2.2 Linear / matmul / einsum (`iOS15/linear.py`, `iOS17/linear.py`)

| Op | min iOS | ANE | Notes |
|---|---|---|---|
| `linear` | 15 | ≈ | `iOS15/linear.py:25`. ANE executes via conv1x1; Apple recommends emitting as `conv` directly (`docs/ANE_CONVERSION_RECIPE_2026.md` §2). Current build emits 0 `linear` in chunks, 1 in drafter (top-k head). |
| `matmul` | 15 | ≈ | `iOS15/linear.py:98`. Orion paper measures 3× overhead vs equivalent conv. Used today only for `Q @ K^T`. |
| `einsum` | 15 | ≈ | `iOS15/linear.py:235`. The converter lowers recognised patterns (`'bchq,bkhc->bkhq'`) to matmul; others decompose to `reduce_sum`/`mul`. Unreliable on ANE — audit §2.1 shows 0 `einsum` in our graphs because everything was pre-factored. |

### 2.3 Convolution (`iOS15/conv.py`, `iOS17/conv.py`)

| Op | min iOS | ANE | Notes |
|---|---|---|---|
| `conv` | 15 | **Y (native, dedicated hardware)** | `iOS15/conv.py:19`. This is the one op with a hardware-native MAC array on every ANE generation. 1×1 is the LLM projection canonical form (63–73 `conv` per chunk). |
| `conv_transpose` | 15 | Y | `iOS15/conv.py:265`. Not used by Gemma 4. |

### 2.4 Normalization (`iOS15/normalization.py`, `iOS17/normalization.py`)

| Op | min iOS | ANE | Notes |
|---|---|---|---|
| `batch_norm` | 15 | Y | folded into conv by `fuse_conv_batchnorm` |
| `instance_norm` | 15 | Y | |
| `l2_norm` | 15 | Y | |
| `layer_norm` | 15 | **Y (native)** | `iOS15/normalization.py:208`. Axes must be contiguous channels; `gamma/beta` optional. Fused via `fuse_layernorm_or_instancenorm` (`coremltools/converters/mil/mil/passes/defs/optimize_normalization.py:22`). Currently used 49× per stateful chunk. |
| `local_response_norm` | 15 | Y | |

There is **no** `rms_norm` primitive in MIL 9.0. RMSNorm is either (a) decomposed to `reduce_mean(square) → add(eps) → rsqrt → mul → mul(gamma)` (5 ops, ANE-native primitives), or (b) emitted as `layer_norm` via the Apple cat-trick (`F.layer_norm(cat([x,-x]))`) which is what `conversion/ane_ops.py:61` does. The decomposed form has fewer ops post-fusion and should be preferred — see §5.1.

### 2.5 Elementwise binary (`iOS15/elementwise_binary.py`)

All 17 elementwise binary ops derive from one base class (`elementwise_binary.py:24`) and are ANE-native when both operands are fp16 and shapes broadcast cleanly on the C or S axis. ANE prefers broadcasts that do **not** require transposing either operand.

| Op | ANE | Notes |
|---|---|---|
| `add`, `sub`, `mul`, `real_div` | Y | trivially fused with conv output when possible |
| `floor_div`, `mod`, `pow` | Y (fp16) / N (int) | int-path falls to CPU |
| `maximum`, `minimum` | Y | |
| `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal` | Y (result = bool, stays on ANE if downstream is `select`) | |
| `logical_and`, `logical_or`, `logical_xor` | N on bool; Y if cast to fp16 mask | |

### 2.6 Elementwise unary (`iOS15/elementwise_unary.py`, `iOS17/elementwise_unary.py`)

All ANE-native when input is fp16. Trig ops (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atanh`, `sinh`, `cosh`) are implemented as LUT on ANE — they are slow if placed in the hot path per-token but fine for RoPE tables built once.

| Op | min iOS | ANE | Notes |
|---|---|---|---|
| `abs` | 15 | Y | |
| `acos`, `asin`, `atan`, `atanh`, `cos`, `cosh`, `sin`, `sinh`, `tan` | 15 | Y | LUT — fine for offline RoPE precompute, bad for per-step |
| `ceil`, `floor`, `round`, `sign`, `square` | 15 | Y | |
| `clip` | 15/17 | Y | `iOS15/elementwise_unary.py:216`; iOS17 refines type domain |
| `erf` | 15 | Y | used by exact `gelu`; prefer tanh-approx |
| `exp`, `exp2` | 15 | Y | present in decomposed softmax; native |
| `inverse` | 15/17 | Y | |
| `log` | 15/17 | Y | |
| `logical_not` | 15 | Y (via select) | |
| **`rsqrt`** | 15/17 | **Y** | `iOS15/elementwise_unary.py:557`. Core of RMSNorm. |
| `sqrt` | 15 | Y | |
| `tanh` | 15 | Y | Used by logit softcap (`tanh(x/30) * 30`) in chunk4 tail |
| `threshold` | 15 | Y | |
| **`cast`** | 15/17 | Y (fp16↔fp32) / N (int↔fp) | `iOS15/elementwise_unary.py:819`. Adjacent-cast elimination is in default pipeline (`cast_optimization` at `optimize_repeat_ops.py:341`). |

### 2.7 Reduction (`iOS15/reduction.py`, `iOS17/reduction.py`)

| Op | min iOS | ANE | Notes |
|---|---|---|---|
| `reduce_argmax` / `reduce_argmin` | 15/17 | Mostly N (CPU tail) | `iOS15/reduction.py:126,161`. Used at sampler tail (chunk4) for greedy. |
| `reduce_max`, `reduce_min`, `reduce_sum`, `reduce_mean`, `reduce_prod` | 15 | **Y** | axis-aligned reductions are ANE-native |
| `reduce_l1_norm`, `reduce_l2_norm`, `reduce_sum_square` | 15 | Y | |
| `reduce_log_sum`, `reduce_log_sum_exp` | 15 | Y | `reduce_log_sum_exp` is a potential **SDPA alternative** for numerical-stable masked-max; unused today |

### 2.8 Tensor operations (`iOS15/tensor_operation.py`, `iOS17/tensor_operation.py`)

| Op | min iOS | ANE | Notes |
|---|---|---|---|
| `band_part` | 15 | Y if shapes static | `:29`. Causal mask via `band_part` is an alternative to materialising the mask on the Swift side; the op itself is ANE-native. |
| `cumsum` | 15 | Mostly CPU (sequential op) | `:98` |
| `fill` | 15 | Y (via const) | `:174` |
| `non_maximum_suppression` | 15/17 | N (CPU) | vision only |
| `non_zero` | 15 | N (CPU, dynamic shape) | |
| `one_hot` | 15 | N (CPU) | `:349`. **Avoid** — currently unused (audit §2 confirms 0 in decoder). |
| **`pad`** | 15 | Y | `:435`. Used 10–14× per chunk for seq alignment. |
| **`range_1d`** | 15 | Y if end-start known | `:557`. Useful for static position indices; underused. |
| **`tile`** | 15 | Y | `:632`. Used 18× in stateful chunk for GQA head-repeat. |
| `argsort` | 15 | N (CPU) | |
| `topk` | 15/17 | Partial — small-k ANE, large-k CPU | `:763`. Drafter uses `topk(k=8)` — probably CPU tail, < 1 ms. |
| `flatten2d` | 15 | Y (layout) | |
| `shape` | 15 | N (CPU, returns int tensor) | rarely used if graph is static-shape |
| **`concat`** | 15 | Y | `:944`. 73× in stateful chunk. |
| **`split`** | 15 | Y | `:1105`. 63× in stateful chunk (gate/up unpack). |
| `stack` | 15 | Y | |
| `identity` | 15 | Y (no-op) | |

### 2.9 Tensor transformation (`iOS15/tensor_transformation.py`, `iOS16/tensor_transformation.py`, `iOS17`, `iOS18/tensor_transformation.py`)

| Op | min iOS | ANE | Notes |
|---|---|---|---|
| `depth_to_space` / `space_to_depth` / `space_to_batch` / `batch_to_space` / `pixel_shuffle` / `pixel_unshuffle` | 15/16 | Y (layout) | vision ops; unused here |
| **`expand_dims`** | 15 | Y | 42–49× per chunk to reconcile 3D/4D boundaries |
| **`reshape`** (static) | 15 | Y | `:161`. 84–98× per chunk; **all static** (audit §3). |
| `reshape_like` | 16 | Y | `iOS16/tensor_transformation.py:19`. Copies a shape from another var — useful when you want to reuse a run-time shape without re-deriving it. Currently unused. |
| `reverse` / `reverse_sequence` | 15 | Y | |
| **`slice_by_index`** | 15 | Y (when begin/end are const) | `:457`. 41–88× per chunk. Dynamic begin/end path routes through CPU. |
| `slice_by_size` | 15 | Y | static size variant |
| **`slice_update`** | **18** | Y | `iOS18/tensor_transformation.py:22`. New. **In-place slice write** — exactly what KV-append wants. Not used in our decoders yet. |
| `sliding_windows` | 15 | ? | 1D conv-like reshape; rarely used |
| **`squeeze`** | 15 | Y | 28–31× per chunk |
| **`transpose`** | 15 | Y | `:948`. 126–162× per chunk — highest op-count after `const`. See `docs/D4_NCHW_FEASIBILITY.md`. |

### 2.10 Scatter / gather (`iOS15/scatter_gather.py`, `iOS16/scatter_gather.py`, `iOS17/scatter_gather.py`)

All of these route through CPU on iOS ≤ 26 when indices are dynamic — Apple's ANE compiler does not have a gather ALU.

| Op | min iOS | ANE | Notes |
|---|---|---|---|
| `gather` | 15/16/17 | N | `iOS16/scatter_gather.py:18` adds `batch_dims`, uint16 indices — still CPU |
| `gather_along_axis` | 15/17 | N | `iOS15/scatter_gather.py:234`. 1 use in chunk4 decode tail. |
| `gather_nd` | 15/16/17 | N | |
| `scatter` / `scatter_along_axis` / `scatter_nd` | 15/17 | N | **0 uses in our graphs** — all KV writes go through concat+slice, which is ANE. Keep it that way. |

### 2.11 Recurrent (`iOS15/recurrent.py`, `iOS17`, `iOS18/recurrent.py`)

`gru`, `lstm`, `rnn` — GPU-only on iOS 18+. Not used by Gemma 4.

### 2.12 Control flow (`iOS15/control_flow.py`)

`cond`, `while_loop`, `make_list`, `list_read`, `list_write`, `list_gather`, `list_scatter`, `select`, `const`, `identity`. `cond`/`while_loop` force a CPU boundary for the branch selector. We don't use them.

`const` appears 1458 times in stateful_chunk2 — most of those are the per-layer weight tensors feeding `constexpr_lut_to_dense`.

`select` is the elementwise `where`: ANE-native for fp16 inputs.

### 2.13 Image / resizing / pool / random / classify

`upsample_nearest_neighbor`, `upsample_bilinear`, `resize_nearest_neighbor`, `resize_bilinear`, `resize` (iOS17), `crop_resize`, `crop`, `affine`, `resample`, `avg_pool`, `max_pool`, `l2_pool`, `random_*`, `classify` — not used in LLM path. Pool ops are ANE-native, image ops fall to GPU for non-integer scales.

### 2.14 iOS16 additions (`iOS16/`)

| Op | File | Notes |
|---|---|---|
| `constexpr_affine_dequantize` | `iOS16/constexpr_ops.py:17` | Older quant path (pre-block) |
| `constexpr_cast` | `:133` | fp16↔fp32 on const |
| `constexpr_lut_to_dense` | `:176` | **What our build emits today** for INT4 palette. ANE-native but LUT-style, not block-shift-scale. |
| `constexpr_sparse_to_dense` | `:277` | |
| `reshape_like` | `iOS16/tensor_transformation.py:19` | see §2.9 |
| `pixel_unshuffle` | `:144` | |
| `fill_like` | `iOS16/tensor_operation.py:21` | |
| `gather` (w/ batch_dims) | `iOS16/scatter_gather.py:18` | still CPU |

### 2.15 iOS17 additions (`iOS17/`)

Updated type domains for `activation.py`, `conv.py`, `elementwise_unary.py`, `linear.py`, `normalization.py`, `reduction.py`, `recurrent.py`, `scatter_gather.py`, `tensor_operation.py`, `tensor_transformation.py`.

New ops in iOS17:
- `quantize` / `dequantize` (`iOS17/quantization_ops.py:70,164`) — **runtime** quantisation. Distinct from `constexpr_*` (compile-time). Placement: ANE for int8↔fp16 per-tensor, GPU/CPU for per-channel w/ dynamic scales. Not applicable to our pipeline unless we move to INT8 activations (rejected per MEMORY).
- `resize` (`iOS17/image_resizing.py:223`) — unified image resize op.

### 2.16 iOS18 additions (`iOS18/`)

The iOS18 opset is small but contains the four ops that matter most for LLM decoding.

| Op | File | ANE | Notes |
|---|---|---|---|
| **`scaled_dot_product_attention`** | `iOS18/transformers.py:18` | **Y (native fused)** | Attributes: `query`, `key`, `value`, optional `attn_mask` (bool or fp). Validates batch dims, broadcasting, etc. No `dropout_p` / `is_causal` params supported — pass a precomputed mask. |
| **`read_state`** | `iOS18/states.py:14` | Y | Input: `state<tensor>`. Output: wrapped tensor. Paired with `coreml_update_state` (below). |
| **`slice_update`** | `iOS18/tensor_transformation.py:22` | Y | In-place `x[begin:end:stride] = update`. Same semantics as `slice_by_index` but writes. |
| **`constexpr_blockwise_shift_scale`** | `iOS18/compression.py:24` | Y | Block-wise int4/int8/uint4/uint8 dequantize. `output = scale * (data - offset)`. This is the hardware-native path for `per_block(block_size=32)` quant. |
| `constexpr_lut_to_dense` (iOS18 version) | `:168` | Y | Updated to support n_bit in [1,2,3,4,6,8], multi-LUT heads |
| `constexpr_lut_to_sparse` | `:461` | Y | Joint palettization+pruning |
| `constexpr_sparse_blockwise_shift_scale` | `:637` | Y | Sparse + blockwise; both features together |
| `constexpr_sparse_to_dense` (iOS18) | `:389` | Y | Refreshed semantics |
| `gru` (iOS18 update) | `iOS18/recurrent.py` | Y | minor update |
| `coreml_update_state` | `coreml_dialect/ops.py:13` | Y | Dialect op paired with `read_state`; inserts the in-place write into the state buffer. |

### 2.17 iOS26 / coremltools 9.0

`coremltools/converters/mil/_deployment_compatibility.py:27` declares `AvailableTarget.iOS26`. No *new op defs* are registered under an `iOS26/` directory yet in this checkout (HEAD `8147ec1`); iOS 26 effectively inherits the iOS 18 op set. The 9.0 additions are at the **converter and runtime** layer:

- Int8 I/O dtypes for `ct.TensorType` (`dtype=np.int8`).
- Python 3.13 / PyTorch 2.7 front-end.
- State read/write APIs on `MLState` (Swift side) for warm-start and speculative revert.
- New MIL passes (not ops) — see `docs/MIL_PASSES_ADDITIONAL.md`.

If Apple adds iOS 26 ops later in the `9.x` branch (e.g. the FP8 or BF16 ops the WWDC 2026 watch doc tracks — `docs/APPLE_2026_ROADMAP_WATCH.md`), they would land at `coremltools/converters/mil/mil/ops/defs/iOS26/`.

---

## 3. Ops the Gemma 4 pipeline currently emits

Inventoried from `docs/MLPACKAGE_STRUCTURE_AUDIT.md` §2 across four packages (`stateful_chunk2`, `iphone_8k/chunk1`, `chunk4`, `mtp_drafter`). Ranked by appearance frequency per chunk.

| Op | Count / chunk | ANE | Notes |
|---|---|---|---|
| `const` | 1458–1727 | — | weights + constants |
| `constexpr_lut_to_dense` (iOS16) | 63–73 | Y | quantised weights, see §4.1 |
| `mul`, `add`, `sub` | 100–200 ea | Y | |
| `transpose` | 126–162 | Y | layout — highest reducible count (§5) |
| `reshape` (static) | 84–98 | Y | |
| `conv` | 63–73 | Y (native) | Q,K,V,O,gate+up(fused),down — 5 per layer × 14 layers ≈ 70 |
| `slice_by_index` | 41–88 | Y | |
| `layer_norm` | 49–57 | Y (native) | 4 RMSNorms/layer + QK-norm + post-embed + pre-logit |
| `concat` | 25–73 | Y | KV append, head-concat |
| `split` | 21–63 | Y | gate/up split after fused conv |
| `expand_dims` / `squeeze` | 42+28 ... 49+31 | Y | 3D↔4D boundaries |
| `reduce_max` + `sub` + `exp` + `reduce_sum` + `real_div` | 7 sites × 5 ops × 14 layers → ~35 | Y | **decomposed softmax** (should be one `softmax`) |
| `matmul` | 14–18 | ≈ | attention Q@K^T only; A@V is `conv` |
| `gelu` (tanh approx) | 14–16 | Y | GLU |
| `tile` | 18 | Y | GQA K/V broadcast |
| `pad` | 10–14 | Y | seq alignment |
| `tanh` | 1 (chunk4) | Y | logit softcap |
| `reduce_argmax` | 1 (chunk4) | N (CPU tail) | greedy sampler |
| `gather_along_axis` | 1 (chunk4 decode) | N (CPU tail) | single-element token gather |
| `topk` | 1 (drafter) | Partial | k=8 |
| `cast` | 0–1 | Y/N | drafter has 1 int32 cast at tail |
| `softmax` (native) | **0** (decoders) / 4 (drafter) | Y | see §4 |
| `scaled_dot_product_attention` | **0** | — | missing |
| `slice_update` | **0** | — | missing |
| `constexpr_blockwise_shift_scale` | **0** | — | missing (§4.1) |
| `read_state` / `coreml_update_state` | 14/14 (stateful_chunk2 only) | Y | production is stateless |

**Non-ANE ops in the current graph**: `reduce_argmax` × 1 and `gather_along_axis` × 1 at the chunk4 sampler tail; `topk` × 1 at the drafter head. Combined CPU share is < 1 % of token latency per the audit. Not worth hunting.

---

## 4. Ops worth adopting (with concrete call sites)

### 4.1 `softmax` — replace `ane_softmax` by default (`conversion/ane_ops.py:216`)

`ane_ops.py:216` has `ane_softmax` (decomposed) and `ane_fused_softmax` (wraps `F.softmax` with explicit fp16 casts). The docstring at `:227` already notes that for ct 9.0 / iOS 26 the fused path is ANE-native. The MTP drafter uses the fused form and passes on-device. Switch every decoder call site (`gemma4_prefill_chunks.py:130`, `gemma4_lite_chunks.py:94`, `gemma4_swa_chunks.py:142`, `gemma4_stateless_chunks.py:112`) from `ane_softmax` → `ane_fused_softmax`. Gate the rollout with `test_merged_parity.py` because the safety note at `:252` flags a numerical corner when `scale != 1/sqrt(d)` — for Gemma 4 we already run scale=1.0 in attention so fp16 overflow behaviour is unchanged, but verify.

### 4.2 `scaled_dot_product_attention` — use in the stateless builder

Template exists: `conversion/models/gemma4_swa_cascading.py:218` uses `F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)`. The converter's torch frontend lowers `F.scaled_dot_product_attention` with an explicit `attn_mask` directly onto the MIL `scaled_dot_product_attention` op (`iOS18/transformers.py:18`) when `minimum_deployment_target >= ct.target.iOS18`. Port the same line into `gemma4_swa_chunks.py` / `gemma4_lite_chunks.py` in place of the `matmul + softmax + matmul` triple.

Prerequisite: enable `softmax` first (4.1) — the pattern-matcher for SDPA needs a single `softmax` op to detect.

After this, enable the sliced-Q pass per `docs/ANE_CONVERSION_RECIPE_2026.md` §5.1 and `transformer.py:20`. Defaults: `min_seq_length=1280`, `seq_length_divider=16`. For our 512-long prefill you'd drop `min_seq_length` to 256.

### 4.3 `slice_update` — KV append

Current KV write uses `concat(old_K, new_K)` → `slice_by_index` to the valid range. iOS18 offers a single-op in-place write: `mb.slice_update(x=k_cache, update=new_k, begin=[0,0,pos,0], end=[1,1,pos+q,d])`. The PyTorch front-end lowers `k_cache[..., pos:pos+q, :] = new_k` to `slice_update` when the index is static. Refactor `gemma4_swa_chunks.py` KV append block (grep for `torch.cat([K,` around line 544).

### 4.4 `constexpr_blockwise_shift_scale` — verify / force

Audit §3 says we ship INT4 per-grouped-channel (g=32) palette. The MIL op that landed is `iOS16` `constexpr_lut_to_dense`. The 2026-blessed path is `iOS18` `constexpr_blockwise_shift_scale` (linear-symmetric int4, same bytes, different kernel). To force the iOS18 op:

1. Bump `minimum_deployment_target=ct.target.iOS18` (or iOS26) in the convert call.
2. Use `cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32)` rather than the palettizer config.
3. Re-inspect the emitted spec with `/tmp/audit_mlpkg.py` — success = the `constexpr_lut_to_dense` ops become `constexpr_blockwise_shift_scale`. Kernel change on A18/A19 ANE is measurable per Apple's WWDC24 talk (Mistral 7B ~2× over LUT).

### 4.5 `read_state` / `coreml_update_state`

Stateful KV is a larger refactor (`docs/D5_STATEFUL_KV_IOS26.md`), but the ops already exist in the MIL graph for the `stateful_chunk2.mlpkg` example. The Swift runner work is separate.

### 4.6 `reduce_log_sum_exp` — fallback for very long seq

`iOS15/reduction.py:305`. If we ever hit fp16 overflow in the `exp(QK^T)` term at long context (> 8192), swap the softmax decomposition for `reduce_log_sum_exp` (single op, numerically stable). Not needed today but nice to have in the toolbox.

### 4.7 `range_1d` — bake RoPE position indexing

`iOS15/tensor_operation.py:557`. Currently `cos_f/sin_f/cos_s/sin_s` arrive as Multi-Array inputs every step. Precompute with `range_1d(end=seq_len) → gather → mul cos_const` inside the graph, indexed by a single int scalar `pos`. Saves ~0.1 ms/step by avoiding four `MLMultiArray` binds. Marginal; tracked under `MLPACKAGE_STRUCTURE_AUDIT` §4.7.

---

## 5. Ops to eliminate / replace

### 5.1 `ANERMSNorm` cat-trick → direct RMSNorm decomposition

`conversion/ane_ops.py:61` uses `F.layer_norm(cat([x,-x]))` so the converter matches `fuse_layernorm_or_instancenorm` and emits `layer_norm` as one op. The trick works but pays for a 2× channel expand + split. A cleaner 2026 recipe is:

```python
denom = (x.pow(2).mean(dim=1, keepdims=True) + eps).rsqrt()
return x * denom * gamma
```

which emits `mul → reduce_mean → add → rsqrt → mul → mul` (6 primitives, all ANE-native, all chain-fused by `optimize_normalization.py`). Net: no channel doubling, same op count post-fusion, fewer bytes to dequantize. Benchmark on-device before committing — the cat-trick was chosen for a reason in an older ct build.

### 5.2 Decomposed softmax — drop it (see 4.1)

### 5.3 `F.layer_norm` on a doubled `per_layer` tensor

`gemma4_swa_merged1.py:83` / `gemma4_swa_merged2.py:84` run `F.layer_norm(doubled, (2 * per_layer_dim,))`. Same cat-trick smell as 5.1. Replace with a direct RMSNorm along the per-layer channel axis.

### 5.4 `gather_along_axis` in chunk4 decode tail

One-element gather of token logits. Already tiny (< 0.05 ms). If you touch the sampler, consider `reduce_argmax` only and skip the gather — but not worth chasing standalone.

### 5.5 `einsum` — keep an eye on the torch frontend

If anyone rewrites attention with `torch.einsum`, verify the MIL converter lowered it to `matmul` not to a `reduce_sum` blob. Current `audit` count is 0 `einsum` in all four packages, so we're safe, but the ml-ane-transformers reference uses `einsum` which can trap.

---

## 6. New ops in iOS 26 / coremltools 9.0

At HEAD of the `9.0` tag there is **no new `iOS26/` op directory**; iOS 26 inherits the iOS 18 op set. The 9.0 additions are at the adjacent layers:

- **Front-end**: `ct.TensorType(dtype=np.int8)` — unlocks INT8 input/output Multi-Arrays. Useful for the tied-embedding chunk (chunk4 LM head) where we can pass INT8 token ids without a CPU-side dequant.
- **Runtime**: Swift `MLState.read(ofType:)` / `MLState.write(_:forType:)` for warm-starting KV.
- **Passes (not ops)**: `const_deduplication`, `materialize_symbolic_shape_program`, `prefer_state_in_downstream`, `canonicalize_inplace_pattern` — see `docs/MIL_PASSES_ADDITIONAL.md`.
- **Type domain tightening**: several iOS17/18 op defs widen `U` (index dtype) to include `int8`/`int16`, allowing smaller KV-index tensors.

If Apple ships true FP8 or BF16 ops on iOS 26.x (per `docs/APPLE_2026_ROADMAP_WATCH.md`) they would register under an `iOS26/` directory; watch this checkout when it moves past `8147ec1`.

---

## 7. References

- `/tmp/ct9` — coremltools 9.0 source at HEAD `8147ec1` (cloned from `https://github.com/apple/coremltools`, tag `9.0`).
- `/tmp/ct9/coremltools/converters/mil/mil/ops/defs/iOS{15,16,17,18}/` — MIL op registry.
- `/tmp/ct9/coremltools/converters/mil/mil/passes/defs/transformer.py:20` — `scaled_dot_product_attention_sliced_q` pass.
- `/tmp/ct9/coremltools/converters/mil/mil/passes/defs/optimize_normalization.py:22` — `fuse_layernorm_or_instancenorm` pass.
- `/tmp/ct9/coremltools/converters/mil/mil/passes/defs/optimize_repeat_ops.py:341` — `cast_optimization`.
- `/tmp/ct9/coremltools/converters/mil/_deployment_compatibility.py:27` — iOS26 target registration.
- `docs/MLPACKAGE_STRUCTURE_AUDIT.md` — emitted op counts for the four audited packages.
- `docs/CONVERSION_AUDIT_2026_04_15.md` — Python converter audit (companion).
- `docs/MIL_PASSES_ADDITIONAL.md` — MIL passes (companion).
- `docs/ANE_CONVERSION_RECIPE_2026.md` §2–§5 — ANE placement conventions.
- `docs/GPU_WHY_FAST.md` — ANE dispatch-floor measurements; motivates op-count reduction.
- `docs/D4_NCHW_FEASIBILITY.md` — transpose elimination via NCHW end-to-end.
- Apple ML Research, *Deploying Transformers on the Apple Neural Engine* (2022).
- Apple ML Research, *On-Device Llama 3.1 with Core ML* (2024).
- Apple ML Research, *Apple Intelligence Foundation Models, Tech Report 2025*, arXiv 2507.13575.
- Orion (arXiv 2603.06728) — op-level ANE / GPU / CPU placement measurements.
- coremltools 9.0 release notes — https://github.com/apple/coremltools/releases/tag/9.0.
- `conversion/ane_ops.py:216,244` — `ane_softmax` / `ane_fused_softmax`.
- `conversion/models/gemma4_swa_cascading.py:218` — live `F.scaled_dot_product_attention` call site (template for 4.2).
- `conversion/models/gemma4_decoder.py:175` — fallback `torch.softmax` path (control for A/B).
