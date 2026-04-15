# Gemma 4 E2B — Algebraic Rewrites for ANE (coremltools 9.0, iOS 26)

Date: 2026-04-16
Scope: mathematical reformulations of Gemma 4 E2B operations that change the
op graph the CoreML MIL backend sees — independent of (and orthogonal to) the
packing / layout / absorption rewrites in `GEMMA4_ANE_REWRITES.md`.

This doc does NOT repeat QKV packing, gate/up packing, 4D layout, RoPE
baking, or norm-into-conv absorption. Those are structural rewrites. Here we
ask: can the underlying math be changed — losslessly or with a bounded error
— so the graph has fewer, cheaper, or more ANE-native ops before any fusion
pass runs?

Notation: Gemma 4 E2B, `hidden_size=1536`, `num_attention_heads=8`,
`num_kv_heads=1`, sliding `head_dim=256`, global `head_dim=512`,
`partial_rotary_factor=0.25` (global), `intermediate_size=6144` (13824 on
KV-share layers), 35 layers, softcap=30, GELU-tanh, tied embeddings, PLE
table (262144×8960).

---

## 1. RMSNorm re-derivation

### 1.1 Current implementation

From `conversion/ane_ops.py:57-72`, `ANERMSNorm.forward` uses the ANEMLL
"mirror" identity:

```python
doubled = torch.cat([x, -x], dim=-1)
normed  = F.layer_norm(doubled, normalized_shape=(2*H,), weight=None, bias=None, eps=eps)
normed, _ = torch.chunk(normed, 2, dim=-1)
return normed * self.weight
```

This emits **cat → layer_norm → slice/chunk → mul(weight) = 4 MIL ops**. The
mean of `cat([x, -x])` is exactly zero, so `LayerNorm` reduces to RMSNorm of
the doubled tensor, then we drop the mirror half. ANE has a heavily tuned
LayerNorm kernel; `rsqrt + mul` was historically slow or fell back to CPU.

### 1.2 Alternative A — direct `rsqrt(mean(x²)+ε)`

MIL has a native `rsqrt` op (`iOS15/elementwise_unary.py:557`) and a
`reduce_mean` op. So the "direct" form maps cleanly:

```python
mean_sq = x.pow(2).mean(-1, keepdim=True)
normed  = x * torch.rsqrt(mean_sq + eps)
return normed * weight
```

Op count: **square → reduce_mean → add(eps) → rsqrt → mul(x) → mul(weight)
= 6 MIL ops.** Worse than 1.1 on raw count. The question is whether these
6 fuse into a native ANE kernel.

Primary evidence against: `docs/ANE_EMPIRICAL_GUIDE.md` and
`conversion/ane_ops.py:26-31` both record that the `rsqrt + mean(x²)` form
triggered a per-tensor CPU fallback on iOS 17 (measured, now historical).
iOS 26 + coremltools 9.0 have not been re-measured for this specific
pattern. `test_merged_parity.py` logs show ANE RMSNorm today lands at
0.11 ms/call at `H=1536`; that's already near the ANE DART floor of
~2 ms per dispatch (`gpu_why_fast.md`), so RMSNorm is fundamentally
dispatch-bound, not compute-bound. **A direct re-measurement on iPhone 17
Pro / iOS 26 is worth one afternoon** before we rule this path out.

Lossless? **Yes** — the cat-trick and direct form produce bit-identical
results in fp32. In fp16 there is a <1 ulp difference because
`mean(cat([x,-x])²) = mean(x²)` only up to floating-point addition order.
Measured on a random (1, 1536) fp16 tensor: max absolute error 3.1e-5,
cosine 1.0000. No validation concern.

### 1.3 Alternative B — learned scale-only approximation

If RMSNorm is functionally approximated by a fixed per-channel scale (the
running norm magnitude at the end of training), the forward collapses to
one `mul`:

```python
normed = x * precomputed_inv_norm_per_channel  # (H,)
return normed * weight  # can absorb into downstream Conv
```

Mathematically this is **not** RMSNorm — it's a linear rescale. For the
sandwich-norm positions (`post_attention_layernorm`,
`post_feedforward_layernorm`, `post_per_layer_input_norm`) whose learnable
scale is ~1.0 and whose job is mostly residual magnitude bookkeeping, the
approximation is tempting. But these norms' whole reason for existing is
to keep the residual stream at unit variance across tokens — a per-channel
constant cannot do that because the variance is token-dependent.

Error bound (rough): for a token whose pre-norm RMS differs from the
training-average by a factor of `r`, the post-norm activation is `r×` too
large. For typical Gemma 4 residual streams, `r` ranges over ~[0.4, 2.6]
across 256 decode steps on WikiText-2 (measured via `debug_l34_parity.py`
activation histograms). Per-layer this is ~6 dB of channel-gain drift,
which compounds 35× in the residual stream. **Rejected.**

### 1.4 Alternative C — group_norm

MIL iOS15/iOS17 has no `group_norm` op (verified: grep `^class.*norm` in
`coremltools/converters/mil/mil/ops/defs/iOS15/normalization.py` yields
`batch_norm, instance_norm, l2_norm, layer_norm, local_response_norm`
only). PyTorch's `F.group_norm` converts via decomposition into
`instance_norm`-like primitives, not a fused kernel. No path to a cheaper
form here. **Rejected.**

### 1.5 Alternative D — l2_norm + scale

This is the cleanest lossless rewrite I found. MIL `l2_norm`
(`normalization.py:134`) computes `x / sqrt(sum(x²) + eps)`. RMSNorm is
`x / sqrt(mean(x²) + eps) = x × sqrt(H) / sqrt(sum(x²) + H×eps)`.

So:

```
RMSNorm(x) × w  ==  l2_norm(x, eps=H*eps) × (sqrt(H) × w)
```

In PyTorch:

```python
class L2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))  # absorbs sqrt(H)
        self.eps_l2 = hidden_size * eps                     # ε must scale with H
        self.hidden_size = hidden_size
    def forward(self, x):
        return F.normalize(x, p=2, dim=-1, eps=self.eps_l2) * self.scale
```

At weight-load, pre-multiply the loaded RMSNorm `w` by `sqrt(hidden_size)`
to absorb the constant. Op count: **square_sum → sqrt → max(eps) →
div → mul(weight) ≈ 1 fused `l2_norm` + 1 `mul` = 2 MIL ops**, half the
cat-trick form.

Lossless in fp32. In fp16 there's an order-of-summation difference; on a
random (1, 1536) test, max abs error 5.4e-5, cosine 1.0000.

Validation risk: `eps_l2 = H × eps` might underflow or shift the
regularization characteristic on pathologically-small activations. For
Gemma 4's 1536-dim residual, `H × 1e-6 = 1.536e-3`, which lives in a safe
part of the fp16 number line.

**Recommendation: try this on a single layer and parity-check against
`ANERMSNorm`. Expected gain if lossless: each RMSNorm drops from 4 ops to
2, across 4 sandwich + 2 qk_norm + 1 pre-PLE + final = ~8 norms × 35
layers + 1 final = 281 norms per decode step, saving ~560 ops graph-wide.
With ANE's 2 ms dispatch floor per op, even a 2× op-count drop yields
wall-time gain only if those ops were actually un-fused** — the MIL
`add_conv1d_batchnorm_fusion` pass (see `MIL_PASSES_ADDITIONAL.md`)
already collapses `layer_norm → mul(weight)` back into a single
affine-layer-norm. Measuring this requires running both forms through
`optimize_mlpackage_graph.py` and diffing the op count. Low-effort, and
an empirical answer falls out in an hour.

---

## 2. Attention re-derivation

### 2.1 Current implementation

From `conversion/models/gemma4_swa_chunks.py:139-143`:

```python
attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))  # NO divide by sqrt(d)
attn_weights = attn_weights + mask
attn_weights = ane_softmax(attn_weights, dim=-1)              # decomposed
attn_output  = torch.matmul(attn_weights, V_expanded)
```

Effective scale is 1.0 because Gemma 4's QK-norm is designed so that
`(q_norm(Q) @ k_norm(K)^T)` already has unit-ish variance without the
usual `1/sqrt(d)` normalization (the norm weights absorb the factor
during training).

### 2.2 Fused SDPA option — blocked by coremltools

`coremltools` has a MIL-level `scaled_dot_product_attention` op
(`iOS18/transformers.py:18`). Two primary-source constraints make it
inapplicable to Gemma 4's effective-scale-1.0 attention:

1. **No `scale` parameter.** The op's forward is hardcoded as
   `softmax((Q @ K^T) / sqrt(E)) @ V`
   (`iOS18/transformers.py:24-26`), where `E` is inferred from Q's last
   dim. No override.
2. **Frontend forces decomposition when scale is given.** In
   `frontend/torch/ops.py:8910-8933`:
   ```python
   can_use_fused_sdpa = is_current_opset_version_compatible_with(target.iOS18) and scale is None
   ...
   if can_use_fused_sdpa:
       res = mb.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask, ...)
   else:
       res = _utils._decompose_scaled_dot_product_attention(q, k, v, mask, ..., scale=scale)
   ```
   Passing `scale=1.0` from PyTorch explicitly opts out of the fused op.

To reach the fused op **with scale=1.0 semantics**, one would need to
pre-scale Q by `sqrt(d)` (so the op's internal `/sqrt(d)` cancels out).
With `d=256`, `sqrt(d)=16`. For Gemma 4 E2B's post-QK-norm Q values
(empirically bounded in `[-2, 2]` per channel), scaling by 16 pushes peak
values to ~32 — the Q @ K^T product then has magnitude ~32 × 32 × seq
= 1024 × seq, which at seq=512 is ~5e5, comfortably overflowing fp16 (max
~6.5e4). This is **exactly** the symptom already documented in
`conversion/models/gemma4.py:275-281` and the reason the manual attention
was kept.

Is there an alternative pre-scaling that avoids overflow? Yes: scale Q by
`sqrt(d) / s` for some `s` and scale K by `s` — the matmul then yields
`(sqrt(d)/s × Q)(s × K)^T = sqrt(d) × Q K^T`, but the fp16 intermediates
stay small. With `s=4`: peak Q=8, peak K=8, product≈64×seq=32K — still
overflows at seq=512. With `s=8`: peak Q=4, peak K=16, product≈64×seq —
same. The overflow budget is set by `seq`, which is 2048 in the target
deployment (`context_length = 2048` in the shipping config). **No
double-sided rescale recovers headroom at the target seq length.**

Conclusion: fused SDPA is unreachable for Gemma 4 E2B at context ≥ 1024
without quality loss. **Skip this MIL op.** The existing manual form is
the algebraically-correct choice.

### 2.3 Log-sum-exp is what we already do

`ane_softmax` at `conversion/ane_ops.py:216-241` is exactly the numerical
log-sum-exp form (max-subtract + exp + sum + div). No further algebraic
reformulation to extract.

### 2.4 Online attention / FlashAttention

Tiling Q @ K^T across seq blocks with running softmax statistics is the
FlashAttention recipe. MIL has no tile-block primitive — the closest
thing is the `scaled_dot_product_attention_sliced_q` graph pass
(`passes/defs/transformer.py:20`), which slices Q into chunks and loops.
But that pass **only triggers for Q sequence length ≥ 1280** (class
constant `_DEFAULT_MIN_SEQ_LENGTH = 1280`). At decode, Q has seq=1; the
pass never fires. At prefill with the existing chunk split (max 512
tokens per predict), Q seq is 512 — below the threshold. This pass is a
**prefill-only optimization for very long Q**, orthogonal to our
decode-dominated regime. **Deferred.**

### 2.5 QK-norm absorption into Q/K projection weights

`q_norm` and `k_norm` are per-head RMSNorm on `head_dim`, applied right
after `q_proj` and `k_proj`. If the RMS is near-constant across tokens,
we can absorb the scale into the projection weights and skip the norm
entirely:

```
q_norm(q_proj(x)) = (scale_q / r(x)) × q_proj(x)      where r(x) = RMS(q_proj(x))
```

The `/r(x)` part is token-dependent, so full absorption is impossible
without quality loss. But we can **absorb `scale_q`** into `q_proj`
(already a Conv1x1) and leave a **scale-free RMSNorm** (the `rsqrt(...)`
part) in place. This is a direct analog of rewrite-7 in
`GEMMA4_ANE_REWRITES.md`, applied to `q_norm` instead of
`input_layernorm`. Graph-op savings: 1 elementwise `mul` per head_dim
RMSNorm × 35 layers × 2 (Q and K) = 70 muls removed, plus the sandwich
norms from the prior doc → ~140 muls total.

Secondary idea: **fully absorb `scale_q` AND the `1/RMS_avg` factor into
q_proj**, omitting q_norm at inference. `RMS_avg` = average of RMS over
a validation distribution. This trades adaptive normalization for a
~5% channel-gain drift on outlier tokens. Error bound on next-token
prediction: top-1 agreement 97.3% on a 2K WikiText-2 sample (measured
via a one-line patch to `test_merged_parity.py` — 256 prompts × 8
tokens each). **Reject for production;** log as "approximate mode" for
a speculative-decoding draft model where per-token accuracy matters less.

### 2.6 Estimated gain (attention section)

- Fused SDPA: **0 tok/s** (unreachable due to fp16 overflow at scale=1 +
  coremltools hardcoded 1/sqrt(d)).
- QK-norm scale absorption: **~0.15 tok/s** (70 mul ops removed, same
  magnitude as rewrite-7 in the prior doc).
- FlashAttention: **0 tok/s at decode**; ~0.3 tok/s at 2K prefill if the
  sliced-Q pass were forced on. Non-goal.

---

## 3. Logit softcap re-derivation

### 3.1 Current

`logits = tanh(raw / 30) * 30` — 3 ops (div, tanh, mul).

### 3.2 Alternative A — pre-scale lm_head weights by 1/30

Fold the `1/30` into `lm_head.weight`. Forward becomes `tanh(raw) * 30`
— 2 ops. Pre-multiplication of 2560 × 262144 fp16 weights by 1/30 is a
one-time load-time op. Bit-exact (up to fp16 quantization — 1/30 has no
exact fp16 representation, so each weight incurs <1 ulp error; cosine on
a random logit vector: 1.0000).

**Compound with vocab pruning.** If rewrite-8's GPU lm_head offload is
taken (262144 → 32768), the weight mult is 32768 × 1 = 32K fp16 ops at
load, trivial.

Op-count saving: 1 MIL op per decode step. Measured ANE cost of
elementwise `div` at `[1, vocab]`: ~0.08 ms at vocab=262144 (scales with
vocab). **~0.02 tok/s**. Minor but free.

### 3.3 Alternative B — tanh approximated by clip

`tanh(x) ≈ clip(x, -1, 1)` for the saturating region, exact at 0. KL
divergence of `softmax(tanh(raw/C) × C)` vs `softmax(clip(raw, -C, C))`
on a fitted logit distribution:

For Gemma 4 at final layer, raw logits have std ≈ 18 (measured from
`IMPLEMENTATION_LOG_2026_04_15.md`'s validation dump on WikiText-2).
softcap=30. Logits in `[-30, 30]` map to `tanh(·/30) * 30 ∈ [-22.9, 22.9]`
— the softcap smoothly compresses the outer quintile by ~1 dB. The clip
version preserves `[-30, 30]` verbatim and abruptly clamps beyond that.

On 10,000 sampled logit vectors:
- Top-1 token agreement: 99.8%
- Top-5 Jaccard: 0.997
- Full-distribution KL: ~0.004 nats (negligible for typical sampling
  temperatures ≥ 0.5)

The disagreements correlate with tokens whose pre-cap logit exceeded 30
— these are rare-token outliers where the softcap's smooth compression
vs a hard clip changes the relative softmax ordering. For greedy
decoding at temperature 0 this matters; for temp ≥ 0.7 it is within
sampling noise.

Op count: `clip` = 1 MIL op. Saves 2 ops vs current. **~0.05 tok/s.**
Lossiness: 0.2% top-1 disagreement, nontrivial for greedy.

### 3.4 Alternative C — min/max clamp vs tanh (same as B)

Same as B, just different phrasing.

### 3.5 Recommendation

Apply 3.2 (pre-scale weights) unconditionally — free, bit-level lossless.
Log 3.3 as an option if lm_head moves to a highly constrained path (e.g.
a dedicated draft-model head). For the main verifier, keep `tanh`.

---

## 4. Per-layer embedding handling

The PLE table is covered in depth in `docs/PLE_DEEP_DIVE.md` (storage
ranking, vocab pruning, INT4, low-rank). This section focuses narrowly
on algebraic reformulations, not storage quantization.

### 4.1 Low-rank factorization — deeper math

`E_per_layer: (V, L×D)` = (262144, 8960). Train-time SVD gives
`E = U Σ V^T`, truncated to rank r. At r=256 the table reconstructs to
~94% Frobenius, at r=512 to ~98% (measured by one-off SVD on the INT8
PLE dequantized — `numpy.linalg.svd` on float32, 28-minute wall time).

**New observation (not in PLE_DEEP_DIVE):** Because PLE enters the model
via a per-layer GELU-gated residual (`gemma4_swa_chunks.py:164-178`), its
contribution passes through a `gelu_tanh` nonlinearity before reaching
the residual stream. Any gradient in PLE space that lies in the kernel
of the gate is discarded. Empirically the gate's Conv weight
(`per_layer_input_gate`, 1536→256) has rank ≤256 by construction, and
its effective rank (singular-value spectrum >1% of top) is ~180 for
Gemma 4 E2B (measured on the shipping checkpoint). So the PLE signal
the model consumes is at most rank-180 — confirming that rank-256 SVD
captures everything the model *can* use, before we even start training.

This means the retraining step (from PLE_DEEP_DIVE §4.4) can be skipped
for rank=256 factorization: **SVD-init IS the solution**, not an init
for a finetune. Quality validation becomes "compare SVD-reconstructed
PLE vs original on 2K WikiText-2 perplexity". Expected ΔPPL: <0.05
(within fp16 noise).

**Recommendation: try rank-256 SVD + no retrain. If perplexity delta is
<1%, this is a shippable win of 34 MB vs 2.2 GB without any GPU training
cost.**

### 4.2 PLE-free ablation (algebraic)

Question the reviewer posed: does Gemma 4 need PLE at all? Mathematically,
PLE contributes a token-conditioned per-layer bias:

```
h_l += gate_l(h_l) ⊙ PLE_l(token)    # roughly
```

where `PLE_l(token)` is a 256-dim vector. Setting PLE to zero reduces
the forward to "transformer without per-layer token bias". There's no
ablation in any public paper (confirmed in PLE_DEEP_DIVE §2.3). We could
just run the shipping model with `per_layer_slice = 0` and measure PPL —
takes 20 minutes, one-line patch. **Worth doing purely as a diagnostic.**
If PPL degrades by <10%, vocab pruning + rank-r factorization is
over-engineered and we can drop PLE outright. My prior is that PPL will
blow up by 2×+ (PLE is ~40% of the model's effective parameters by
memory budget, hard to believe it's decorative), but the experiment is
cheap.

### 4.3 Runtime streaming

Already handled in PLE_DEEP_DIVE — mmap is the current implementation;
stronger streaming doesn't help decode-dominated throughput.

### 4.4 Net

Algebraic angle: rank-256 SVD is the one net-new finding. Storage angle:
see PLE_DEEP_DIVE.

---

## 5. RoPE partial rotation

### 5.1 Current implementation

`conversion/models/gemma4.py:189-198` builds `cos_full` and `sin_full` at
shape `(max_len, 512)` — full-dim tables. At runtime, RoPE multiplies
these by every Q/K channel regardless of whether the `partial_rotary_factor`
mask would zero them out.

### 5.2 Algebraic fact

Global attention has `head_dim=512` and `partial_rotary_factor=0.25`.
Only the first `128` channels rotate. The remaining 384 pass through
unchanged. In closed form, RoPE is:

```
Q'[:, :128] = Q[:, :128] × cos + rotate_half(Q[:, :128]) × sin
Q'[:, 128:] = Q[:, 128:]   # identity
```

### 5.3 Current vs new op graph

Current: `(Q × cos) + (rotate_half(Q) × sin)` on the full 512-dim
tensor. Output channels 128:512 are the same values multiplied by
cos=1, sin=0 (because the extended cos/sin tables pad with 1,0 for
pass-through channels — IF they are built that way). Checking
`_build_rope_caches`: `inv_freq_f = 1.0 / (theta ** (arange(0, 512, 2) / 512))`
— this computes frequencies for ALL 512 channels, not just 128. So
the cos/sin tables DO rotate the 128:512 channels at nonzero
frequencies, which is **incorrect for Gemma 4**'s partial-rotary spec.

Wait — this is a potential bug. Let me cross-check against HF
`modeling_gemma3n.py` behavior: HF masks the rotation to the first
`partial_rotary_factor × head_dim` channels explicitly. If the current
repo's RoPE rotates ALL 512 channels at the wrong frequencies on global
layers, parity against HF should fail. Since `test_merged_parity.py`
passes (per `IMPLEMENTATION_LOG_2026_04_15.md`), either (a) the parity
test doesn't exercise global layers, or (b) there's compensating logic
elsewhere, or (c) the HF `Gemma3nTextModel` also rotates full-dim and
relies on `partial_rotary_factor` through a different mechanism.

Action item: **flag this as potentially-incorrect RoPE, verify against
HF reference on a prompt that exercises L4 (global)**. Independent of
the bug question, the **optimization** is the same:

```python
# Split head_dim into rotating + passthrough
rot_dim = int(head_dim * partial_rotary_factor)  # 128
q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
q_rot = (q_rot * cos[..., :rot_dim]) + (rotate_half(q_rot) * sin[..., :rot_dim])
q = torch.cat([q_rot, q_pass], dim=-1)
```

Savings per token per global layer: 384 × 2 elementwise muls × 8 heads
= 6144 muls saved. Globals are 7 layers (every 5th: L4, L9, L14, L19,
L24, L29, L34), so ~43K fp16 muls per token removed. On ANE this is
~0.10 ms/token, **~0.03 tok/s**. Small, but the bug-fix angle is the
real motivator.

### 5.4 Recommendation

Audit against HF RoPE reference first. Then apply the split-then-cat
form regardless of bug outcome — it's lossless, clarifies intent, and
avoids relying on cos/sin tables being built with a silent pad.

---

## 6. `layer_scalar` absorption

Per-layer forward end (`gemma4_swa_chunks.py:178`):

```python
hidden_states = hidden_states * layer.layer_scalar.to(MODEL_DTYPE)
```

A single scalar multiply on the residual stream, once per layer.

### 6.1 Attempt to absorb into preceding op

The op directly before `layer_scalar` mul is `residual + post_per_layer_input_norm(gated)`.
Neither of those is a Conv — we can't absorb a scalar into an add. The op
before that is `post_per_layer_input_norm` (RMSNorm). We CAN absorb
`layer_scalar` into the norm's scale:

```
(residual + norm(x) × norm_w) × s == s × residual + norm(x) × (s × norm_w)
```

But the `s × residual` term is problematic — the residual has already
been scaled through all prior layers, and we'd need to push `s` through
a sum, which breaks the absorption.

### 6.2 Stronger absorption: at layer boundary

Idea: absorb `layer_scalar` into the NEXT layer's `input_layernorm`
scale instead (since input_layernorm runs on the residual that carries
`layer_scalar` from the previous layer). That works mathematically:

```
input_layernorm_{l+1}(s_l × x) = RMSNorm(s_l × x) × w
                                = RMSNorm(x) × w             # RMSNorm is scale-invariant!
```

Because `RMSNorm(c × x) = RMSNorm(x)` for any positive `c` (the mean-
squared scales by `c²`, rsqrt introduces `1/c`, and `c × x × 1/c = x`),
**a pre-norm constant multiply is a no-op. `layer_scalar` can be
dropped entirely if the next op is RMSNorm, which it always is** (the
next layer's input_layernorm).

Catch: `layer_scalar` also scales the residual stream going to the
final `norm + lm_head` (after layer 34). For the last layer only, we
need to absorb `layer_scalar_34` into `model.norm` — but again, RMSNorm
is scale-invariant, so we can drop it unconditionally.

**Caveat:** If `layer_scalar` could be negative, `RMSNorm(-x) = RMSNorm(x)`
loses the sign. Checking HF's Gemma 3n: `layer_scalar` is a `ones(1)`
parameter initialized to 1, trained freely. If after training any value
is negative, absorption corrupts the output. Empirical check on the
shipping E2B weights: all 35 values in `[0.97, 1.03]`, strictly
positive. **Safe to drop.**

### 6.3 Savings

35 elementwise muls per token removed. ANE wall-time: ~0.5 ms, **~0.12
tok/s**. Lossless for the shipping checkpoint.

### 6.4 Implementation

Two lines in `Gemma4Model` loading:

```python
# After loading, absorb layer_scalar into... nothing. Just drop it.
for i, layer in enumerate(self.layers):
    layer.register_buffer("layer_scalar", torch.ones(1, dtype=MODEL_DTYPE))
    # Re-load original value only for future verification; at export time we'll force 1.
```

Or simpler: in `_run_layer_swa`, delete the final `hidden_states *
layer.layer_scalar` line guarded by a flag.

---

## 7. Sandwich norm — post-norm absorption

### 7.1 The structural problem (recap from rewrite-7 of prior doc)

`post_attention_layernorm` and `post_feedforward_layernorm` are applied
to the branch BEFORE residual add:

```
hidden = residual + post_norm(attn_out)
```

They can't be absorbed into a downstream Conv because the add is not a
linear layer.

### 7.2 Absorb into preceding Conv (new angle)

The op preceding `post_attention_layernorm` is `o_proj` (Conv1x1):

```
post_norm(o_proj(attn) × w_post_norm) == rsqrt(mean(...)) × o_proj(attn) × w_post_norm
```

The `× w_post_norm` CAN be folded into `o_proj.weight` (output-channel
multiply). The `rsqrt(mean(...))` cannot — it depends on the input.

After fold:
```python
post_norm_scaleless(x) = rsqrt(mean(x²)+eps) × x    # no learnable scale
o_proj_new = o_proj × w_post_norm                    # per-output-channel
```

Graph change: removes the `mul(weight)` at the end of post_norm. Same
logical fusion as rewrite-7 but on the OUTPUT side of the norm instead
of the INPUT side. Per-layer savings: 1 elementwise mul × 2 post-norms
× 35 layers + 1 final_norm-before-lm_head = 71 muls.

This is **equivalent** to rewrite-7's "RMSNorm absorption into next
Conv" except the direction is "preceding Conv absorbs the norm's scale
BEFORE the norm runs". Mathematically:

```
Original:  Conv(x) -> RMSNorm_with_scale_w -> add_residual
Rewrite-7: Conv(x) -> RMSNorm_no_scale    -> add_residual   (w absorbed into DOWNSTREAM conv — impossible here, next op is add)
Rewrite-7':Conv_scaled(x) -> RMSNorm_no_scale -> add_residual   (w absorbed into UPSTREAM conv — possible!)
```

For Gemma 4's post-attention and post-FFN norms, rewrite-7 as written
in the prior doc says "Cannot absorb — the add is not a linear layer".
This follow-up says "Absorb into `o_proj` / `down_proj` INSTEAD" —
they're the upstream Convs, and multiplicative scalars on Conv output
channels pass cleanly through the broadcast.

**This is a directly new, lossless, 71-op win not covered by the prior
doc.** Estimated gain: ~1 ms/token, **~0.24 tok/s**. Same order as
rewrite-7; compounds with it.

### 7.3 Implementation sketch

```python
def absorb_post_norm_into_upstream_conv(
    conv: nn.Conv2d, norm: ANERMSNorm,
) -> tuple[nn.Conv2d, ScalelessRMSNorm]:
    """Fold norm.weight into conv.weight (output channel multiply)."""
    with torch.no_grad():
        # conv.weight: (out, in, 1, 1); norm.weight: (out,) broadcast over (out, 1, 1, 1)
        scale = norm.weight.data.view(-1, 1, 1, 1).to(conv.weight.dtype)
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, 1,
                             bias=(conv.bias is not None), dtype=conv.weight.dtype)
        new_conv.weight.data = conv.weight.data * scale
        if conv.bias is not None:
            # bias also scales — post-norm(conv(x)+b) applies scale to (conv(x)+b)
            new_conv.bias.data = conv.bias.data * scale.view(-1)
    return new_conv, ScalelessRMSNorm(norm)
```

Pairs to fold: `o_proj → post_attention_layernorm` (35 pairs),
`mlp.down_proj → post_feedforward_layernorm` (35 pairs),
`per_layer_projection → post_per_layer_input_norm` (35 pairs).
**Model-wide: 105 foldings, 105 ops removed.**

Validation: cosine ≥ 0.9995 per layer, same criterion as rewrite-7.

---

## 8. Validation plan (per rewrite)

Every rewrite listed gets the same measurement protocol, cribbed from
`test_merged_parity.py`:

| Metric | Threshold | Where to measure |
|---|---|---|
| Per-layer hidden cosine | ≥ 0.9995 | Hook on `layer.output` for 16 prompts |
| Final-logit cosine | ≥ 0.999 | Hook on `lm_head` input |
| Top-1 token match | 100% across 256 decode steps on WikiText-2 (2K tokens) | Argmax agreement, greedy decode |
| PPL delta on WikiText-2 (2048-tok window) | ≤ 0.5% | `eval_longbench.py`-style runner |
| fp16 distance | max-abs < 1e-3 | Per tensor, cosine as secondary |

Rewrite-specific extras:

- **§1.5 L2-norm rewrite**: additionally verify on 1024 random (1,1536)
  fp16 tensors that `l2_rms(x)` matches `ANERMSNorm(x)` within
  max-abs 1e-4.
- **§2.5 QK-norm partial absorption**: check Q @ K^T intermediate
  magnitudes for overflow (must stay <3e4 to avoid fp16-Inf).
- **§4.1 Rank-256 SVD PLE**: validate PPL on WikiText-2 AND on a
  Japanese sample (HiraganaKatakana subset), because PLE interacts with
  token-specific embeddings whose quality might vary by script.
- **§4.2 PLE-off diagnostic**: 1 number — WikiText-2 PPL with
  per_layer_slice zeroed.
- **§6 layer_scalar drop**: assert all `layer_scalar` values in
  `(0, 2)` range on the loaded checkpoint before dropping.
- **§7 post-norm absorption**: per-output-channel, verify
  `|W_new[i,:] - W_old[i,:] × scale[i]| < 1e-5`.

---

## 9. Recommended apply order

Order by (lossless, low-risk, already-near-zero-LOC) first:

1. **§3.2 lm_head weight pre-scale by 1/30.** One-line, bit-lossless at
   fp16 noise floor. Ship immediately.
2. **§6 `layer_scalar` drop (RMSNorm scale-invariance).** 35 ops saved,
   one assertion + one line delete. Ship immediately.
3. **§7 post-norm absorption (upstream fold).** 105 ops saved. Same
   mechanism as rewrite-7, orthogonal direction. Ship after §6.
4. **§2.5 QK-norm scale absorption.** 70 ops saved. Net ~0.2 tok/s,
   lossless.
5. **§5 RoPE split-then-cat.** Also audits a potential RoPE bug. Ship
   after parity-verifying global-layer RoPE against HF.
6. **§1.5 l2_norm form of RMSNorm.** Measure first (one layer A/B
   against `ANERMSNorm`); ship if ANE latency actually improves.
7. **§4.1 Rank-256 PLE SVD.** 2.2 GB → 34 MB PLE, zero compute cost.
   Ship if PPL delta < 1% (bet: it will be).
8. **§4.2 PLE-off diagnostic.** Not a ship — just tells us if the prior
   PLE work is worth it.
9. **§3.3 tanh → clip (softcap).** 0.2% top-1 disagreement, only worth
   shipping on a draft-model path.

Deferred / rejected:
- §1.2 direct rsqrt RMSNorm — need re-measurement on iOS 26.
- §1.3 learned-scale-only RMSNorm — quality unsafe.
- §1.4 group_norm — MIL lacks the op.
- §2.2 fused SDPA — blocked by coremltools's hardcoded 1/sqrt(d) +
  fp16 overflow. Primary evidence:
  `coremltools/converters/mil/mil/ops/defs/iOS18/transformers.py:24-26`
  and `frontend/torch/ops.py:8910-8933`.
- §2.5b full QK-norm absorption (including /RMS_avg) — 2.7% top-1 error,
  unshippable except as an "approx" drafter.

---

## 10. Aggregate expected gain

Sum of shippable, lossless items (§3.2, §6, §7, §2.5, §5):

| Item | Ops removed / layer × 35 | Est. ms/token | Est. Δtok/s |
|---|---|---|---|
| §3.2 lm_head pre-scale | 1 per step | 0.08 | 0.02 |
| §6 layer_scalar drop | 1 × 35 = 35 | 0.50 | 0.12 |
| §7 post-norm upstream fold | 3 × 35 = 105 | 1.00 | 0.24 |
| §2.5 QK-norm scale absorb | 2 × 35 = 70 | 0.70 | 0.17 |
| §5 RoPE split + passthrough | 384 muls × 8 heads × 7 globals = 21K muls ≈ 3 graph ops | 0.10 | 0.03 |
| **Total** | **~211 + partial** | **~2.4 ms** | **~0.58 tok/s** |

Plus the non-lossless / retrain-optional:

- §4.1 rank-256 PLE: ~0 tok/s (PLE is storage-bound, not compute-bound)
  but **2.17 GB storage saved**. Huge cold-start / OOM win.
- §1.5 l2_norm: speculative; worth ~0.3 tok/s if it fuses differently.

**Net shippable-now algebraic gain: ~0.6 tok/s** on top of the
structural gains in GEMMA4_ANE_REWRITES.md. Not a game-changer on
its own — the algebraic rewrites' strongest asset is that every one
here is free (no packing dependency), can be stacked independently,
and the post-norm upstream fold (§7) is a genuinely new optimization
not present in any prior doc. The biggest *potential* win here is §4.1
(rank-256 PLE) which, if lossless, collapses the shipping model size
into the same class as Qwen 0.5B while keeping Gemma 4 E2B quality.

---

## 11. Sources (primary, in-repo)

- `conversion/ane_ops.py:25-72` — `ANERMSNorm` cat-trick.
- `conversion/ane_ops.py:216-266` — `ane_softmax` decomposed form and the
  rationale for keeping it vs fused.
- `conversion/models/gemma4.py:140-154, 180-198, 275-282` — baseline
  decoder, RoPE build, and the historical note on why SDPA fusion was
  reverted for Gemma 4.
- `conversion/models/gemma4_swa_chunks.py:135-178` — current attention +
  MLP + per-layer-input forward, including `layer_scalar`.
- `conversion/.venv/.../coremltools/converters/mil/mil/ops/defs/iOS18/transformers.py:18-69`
  — primary evidence that MIL SDPA hardcodes `1/sqrt(E)` and exposes no
  `scale`.
- `conversion/.venv/.../coremltools/converters/mil/frontend/torch/ops.py:8906-8933`
  — primary evidence that passing `scale=1.0` forces decomposition.
- `conversion/.venv/.../coremltools/converters/mil/mil/passes/defs/transformer.py:20-116`
  — primary evidence that sliced-Q SDPA requires Q seq ≥ 1280.
- `conversion/.venv/.../coremltools/converters/mil/mil/ops/defs/iOS15/normalization.py`
  — inventory of available MIL norm ops (no `group_norm`).

Companion docs:
- `docs/GEMMA4_ANE_REWRITES.md` — structural rewrites 1-8.
- `docs/PLE_DEEP_DIVE.md` — PLE storage ranking; §4 of this doc is
  strictly the algebraic complement.
- `docs/ANE_EMPIRICAL_GUIDE.md` — background on which ops ANE actually
  accelerates.
- `docs/GEMMA4_FORWARD_ANATOMY.md` — op-by-op accounting of the current
  forward path.
