# Parity Test Protocol for Pre-Conversion Optimizations

**Date:** 2026-04-15
**Branch:** `feat/pre-conv-optimizations`
**Scope:** Verification procedure for each optimization applied on this branch.

---

## 1. Acceptance bars

Each optimization must pass all three before shipping:

| Metric | Threshold | Tool |
|---|---|---|
| Per-layer hidden-state cosine similarity vs reference | ≥ 0.9995 | custom script below |
| Perplexity delta on WikiText-2 (256 samples) | ≤ 0.5 % | `conversion/eval_longbench.py` style harness |
| Top-1 token match rate over 256 decode steps from a fixed seed prompt | = 100 % | `conversion/test_merged_parity.py` |

The top-1 bar is strict because speculative-decoding acceptance is binary per
token; any divergence shows up as acceptance loss downstream even if cosine
looks fine in aggregate.

---

## 2. Reference mlpackage

Keep the pre-optimization 4-chunk mlpackage as the golden reference. Do not
re-convert it during this work. All diffs are measured against this single
fixed artifact.

```
output/gemma4-e2b/reference_8k/chunk{1..4}.mlpackage   (do not touch)
output/gemma4-e2b/candidate_8k/chunk{1..4}.mlpackage   (new, per-optimization)
```

---

## 3. Per-optimization plan

Apply one optimization at a time. Convert, test, commit or revert. Do not
stack until each has been confirmed.

### O1. iOS26 deployment target

**Risk:** Low. Unlocks newer MIL passes but behavior-compatible.

**Verify:**
```bash
python conversion/build_verify_chunks.py --output /tmp/cand_ios26 --ctx 8192
python conversion/test_merged_parity.py \
    --ref output/gemma4-e2b/reference_8k \
    --candidate /tmp/cand_ios26
```

Expected: cosine = 1.0000 (no numerical change, just spec version).

### O2. optimize_mlpackage_graph DEFAULT_PASSES

**Risk:** Low-medium. Pass pipeline changes op count but preserves semantics.
Size-guard catches INT4 blow-up; topological reorder is safe on a DAG.

**Verify:**
```bash
# Convert with --optimize flag
python conversion/build_verify_chunks.py --output /tmp/cand_opt --ctx 8192 --optimize
python conversion/test_merged_parity.py \
    --ref output/gemma4-e2b/reference_8k \
    --candidate /tmp/cand_opt
```

Expected:
- Op count reduction 10–30 % (reported by optimizer).
- cosine ≥ 0.9999 (allow small float16 re-association).
- Top-1 match = 100 %.

**Abort criteria:** any pass reports a size inflation (size-guard triggers);
any post-optimization op type that wasn't in the pre-optimization op type set
(check with `optimize_mlpackage_graph.py --passes ""` to just dump op counts).

### O3. `--include-opt-in` (fuse_matmul_weight_bias)

**Risk:** Medium. Can decompress INT4 constexpr to FP16 (size blow-up). The
size-guard in `optimize_mlpackage_graph.py` catches this, but run a size
check manually too.

**Verify:**
```bash
# First: baseline optimized size
du -sh /tmp/cand_opt

# Then: with opt-in passes
python conversion/optimize_mlpackage_graph.py \
    --input /tmp/cand_opt/chunk1.mlpackage \
    --output /tmp/cand_opt_plus/chunk1.mlpackage \
    --include-opt-in --verify-equivalence

du -sh /tmp/cand_opt_plus
```

Expected: size within 5 % of baseline (else abort — bias fusion decompressed
INT4 weights).

### O4. Fused QKV module (`FusedQKV` drop-in)

**Risk:** Medium. Changes weight layout (concat along out_channels). Load
path must pack q/k/v weights correctly; palettization quality may shift
slightly because the scale distribution across q/k/v differs.

**Verify (after wiring into Gemma4DecoderLayer — see below):**
```python
# test_fused_qkv_parity.py (quick sanity)
from conversion.models.gemma4 import Gemma4Model
from conversion.models.gemma4_fused_modules import fuse_layer_projections
import torch

model = Gemma4Model.from_pretrained(HF_DIR).eval()
x = torch.randn(1, model.config.hidden_size, 1, 1, dtype=torch.float16)
# Reference split forward
layer0 = model.layers[0]
q_ref = layer0.self_attn["q_proj"](x)
k_ref = layer0.self_attn["k_proj"](x)
v_ref = layer0.self_attn["v_proj"](x)
# Fused forward
fuse_layer_projections(layer0)
q, k, v = layer0.self_attn["qkv_fused"](x)
assert torch.allclose(q, q_ref, atol=1e-4)
assert torch.allclose(k, k_ref, atol=1e-4)
assert torch.allclose(v, v_ref, atol=1e-4)
print("fuse QKV: PASS")
```

Then run full chunk parity after chunk forward code is updated to call
`qkv_fused` instead of split.

### O5. Fused Gate/Up module (`FusedGateUp` drop-in)

**Risk:** Same shape as O4. Remember Gemma 4 uses **GELU-tanh**, not SiLU —
the activation stays in the layer's forward, not in the module.

**Verify:** analogous script to O4. Confirm `gate`, `up` outputs match the
split pair within fp16 tolerance.

### O6. RMSNorm scale absorption

**Risk:** High. Sandwich norm has four RMSNorms per layer, and not all of
them precede a Conv. Only absorb the pair `(norm, conv)` when the norm
feeds EXACTLY the conv's input with no intermediate op.

**Current safe candidates on Gemma 4 E2B**:
- `input_layernorm` -> q/k/v projections. BUT: q, k, v are three parallel
  consumers of the same input, so absorbing means multiplying the scale
  into all three conv weights. This is correct but changes palettization
  distribution across three projections that used to share a scale. Test
  perplexity carefully.
- `pre_feedforward_layernorm` -> gate/up projections. Same consideration.

**Not absorbable (do NOT try):**
- `post_attention_layernorm` — feeds a residual add, not a conv.
- `post_feedforward_layernorm` — feeds a residual add.
- `q_norm`, `k_norm` — operate on per-head query/key, not the residual
  stream; the follow-on op is a matmul against K/Q inside attention, not a
  simple Conv.

**Verify:** apply to ONE layer only. Convert just chunk1. Run top-1 parity
on a 64-token decode from a fixed prompt. If it diverges, revert — do not
continue to other layers.

### O7. Softmax fuse (ane_softmax -> ane_fused_softmax)

**Risk:** Medium in attention, low elsewhere. Attention uses effective
scale=1.0 to avoid fp16 overflow in Q@K^T (gemma4.py:277). Fused softmax
changes nothing about the Q@K^T intermediate so the overflow concern is
unchanged, but the fp16 casts inside `ane_softmax` (decomposed form) were
added as belt-and-suspenders. Measure before assuming safe.

**Verify:** swap `ane_softmax` -> `ane_fused_softmax` in ONE call site,
convert, parity test. Then next call site. Attention softmax last.

---

## 4. Regression bench

After each optimization passes parity, run the decode benchmark on iPhone
17 Pro before moving on:

```bash
# On-device via Xcode scheme
# Record: tok/s steady-state, TTFT, peak memory, thermal state
```

If tok/s drops despite op-count reduction, the optimization hurt ANE scheduling
(e.g., moved an op to a slower lane). Revert and leave a note.

---

## 5. Rollback procedure

Each optimization lives in a single commit. To roll back:

```bash
git log --oneline feat/pre-conv-optimizations
git revert <commit-hash>
```

If a whole-branch revert is needed:

```bash
git checkout device-bench
git branch -D feat/pre-conv-optimizations   # only when confirmed
```

Reference mlpackages are never touched, so rollback is always safe.

---

## 6. Result log

Append measurements here as optimizations land:

| Date | Optimization | Op count | Top-1 match | Decode tok/s | Notes |
|---|---|---|---|---|---|
| 2026-04-15 | (baseline) | — | — | 15 | reference |
| (pending) | O1 iOS26 target | | | | |
| (pending) | O2 DEFAULT_PASSES | | | | |
| (pending) | O3 opt-in passes | | | | |
| (pending) | O4 Fused QKV (1 layer) | | | | |
| (pending) | O4 Fused QKV (all) | | | | |
| (pending) | O5 Fused Gate/Up (1 layer) | | | | |
| (pending) | O5 Fused Gate/Up (all) | | | | |
| (pending) | O6 RMSNorm absorb (input_ln) | | | | |
| (pending) | O6 RMSNorm absorb (pre_ffn_ln) | | | | |
| (pending) | O7 Softmax fuse (non-attn) | | | | |
| (pending) | O7 Softmax fuse (attn) | | | | |
