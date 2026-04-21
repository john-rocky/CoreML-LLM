# D4 — Full NCHW end-to-end rewrite feasibility audit

Author: audit session 2026-04-15. Companion to `docs/MLPACKAGE_STRUCTURE_AUDIT.md`
(which motivates this investigation via its §4.3 finding that the decoder
chunk graph carries 4.2 layout ops per compute op). All file:line citations
are against the working tree at commit `dfd8c0a`.

---

## Executive verdict — **GO-AFTER-D1**

The rewrite is technically feasible and the ceiling is real (~100 layout ops
per chunk-function × 4 chunks ≈ 400 ops removable, matching the audit's
10–20 % decode-tok/s ceiling). However, **it is not the next move.** Three
cheaper converter-level wins must land first, because each of them either
(a) removes layout ops on its own, (b) changes the residual-stream shape and
would be redone under NCHW anyway, or (c) is what surfaces the ANE silent-
fallback risk that a 35-layer or 20-layer NCHW chunk would hit even harder.
Specifically: the fused-`softmax` swap (MLPACKAGE_STRUCTURE §4.1 — ~200 ops
removed model-wide, zero shape churn), the `scaled_dot_product_attention_
sliced_q` pass enablement (§4.2 — wants NCHW-consistent Q/K/V shapes and
fuses more when it gets them), and the chunk4 lm_head palettize (§4.8 —
orthogonal but blocks shipping merged1). Only once those land is the residual-
loop rewrite worth the three engineering weeks and the parity re-validation
bill. **Recommended posture: commit to the rewrite as a Track-D deliverable,
but gate start on D1/D2/D5 landing. Do it behind a feature flag with chunk1
A/B first (smallest blast radius). Do not attempt in-place on the 35-layer
MergedChunk1 — it is already a silent-fallback candidate today.**

---

## 1. Current layout op census

Counts are raw grep matches of
`permute|transpose|unsqueeze|squeeze|view|reshape` in each file, classified
manually from reading the context of every hit. "Removable" is an op that
disappears under a pure NCHW hidden-state pipeline (i.e. hidden states live
as `(B, C, 1, T)` from embed lookup to lm_head). "Architectural" stays
because the op's *semantic* consumer demands a different rank or axis
ordering (e.g. `q_norm` wants per-head normalization over a final hidden
dim). "Boundary" is at the model's outer input/output surface and therefore
unaffected by an internal layout change.

| File | Total hits | Removable | Architectural | Boundary / KV-bookkeeping |
|---|---:|---:|---:|---:|
| `conversion/models/gemma4_swa_chunks.py` | 62 | **36** | **14** | 12 |
| `conversion/models/gemma4_swa_merged2.py` | 15 | **8** | 2 | 5 |
| `conversion/models/gemma4_swa_merged1.py` | 15 | **8** | 2 | 5 |
| `conversion/models/gemma4_prefill_chunks.py` | 25 | **14** | 6 | 5 |
| `conversion/ane_ops.py` | 12 | **4** | 5 | 3 |
| **Total** | **129** | **70** | **29** | **30** |

Mapping to the MIL-graph op count (chunk1 verify: 162 transpose + 98
reshape = 260 layout ops): each Python op often lowers to one transpose
plus one reshape, so 70 removable Python ops ≈ 130–140 removable MIL ops,
consistent with the audit's "~100 layout ops per chunk-function" prediction.

### 1.1 Classification details (chunks.py, the largest contributor)

Removable (eliminated by keeping hidden state as `(1, C, 1, T)`):

- Residual-stream wrap/unwrap into Conv2d: lines 67, 145–148, 155, 160, 169,
  175, 238, 241, 443–444, 492, 601–602, 609, 614, 623, 626, 629, 669, 671,
  827–828 — **20+ hits, all the same `permute(0,2,1).unsqueeze(2)` / inverse
  dance identified in MLPACKAGE_STRUCTURE §4.3**.
- Q/K/V pack-for-attention reshapes that exist *only* because we re-3D-ified
  the hidden state before reaching them: 70–71, 84–86, 496–501, 517–524,
  65–72 (prefill), 86–91 (prefill).
- LM-head reshape 443–444, 827–828 — these are boundary-adjacent but the
  boundary can equally well be `(1, vocab, 1, T)` with a final squeeze.

Architectural (stay even under NCHW):

- `q_norm` / `k_norm` per-head-per-token reshape to `(T*H, D)` or
  `(1, H, T, D)` for attention matmul: 70–71, 86, 499–501, 522–524,
  70–72 (prefill), 89–91 (prefill). **These are unavoidable**: the
  semantics of q_norm require normalizing over the per-head dim `D`, which
  is not the outer channel axis. Even in pure NCHW you'd still have to
  reshape `(1, H*D, 1, T)` → `(1, H, D, T)` at the QKV boundary. This is
  the "layout op that doesn't hurt" mentioned in the task spec.
- `matmul(q, K^T)` inside attention (lines 140, 593, 128 in prefill) uses
  `transpose(-1, -2)`. The MIL graph already recognises this as part of
  the attention pattern and does not charge a separate op for it; it is
  fused into the matmul. Not a candidate for removal regardless of layout.
- `repeat_kv_ane` reshape/unsqueeze (`ane_ops.py` 281, 283): GQA head
  replication. Layout-independent.

Boundary / KV-bookkeeping:

- `K_full_in[fi].unsqueeze(0)` / `.squeeze(0)` at KV slot indexing (lines
  277–284, 297–301, 345–352, 366–370). These exist because the KV MultiArray
  surface is `(N_slots, 1, T_cache, max_hd)` and we select one slot per
  layer. Under NCHW the slot dim stays as dim-0 and the bookkeeping is the
  same. Not removable without changing the Swift KV layout, which is a
  separate project (Track K).

### 1.2 ane_ops.py classification

`Conv2dLinear.forward` (lines 168, 171) is the canonical `(B,S,C) →
(B,C,1,S)` wrapper. It is **never called on the hot path**: every caller in
`gemma4_swa_chunks.py` / `gemma4_swa_merged*.py` / `gemma4_prefill_chunks.py`
manually does the permute immediately before invoking `Conv2dLinear`, and
uses the underlying `self.conv(x)` call or `forward_conv`. So the
auto-converter wrapper exists only for unit tests and debug paths.
**Removing it would not affect any production MIL graph.** Confirmed by
grepping call sites: every chunk file invokes the Conv2d directly as
`layer.self_attn["q_proj"](x)` where `x` is already 4D.

`ANERMSNorm.forward` (lines 60, 61, 69) does `cat([x, -x], dim=-1)` →
`F.layer_norm(..., normalized_shape=(2*hidden,))` → `chunk(..., dim=-1)`.
This is the single most important obstacle to the rewrite — see §3.

---

## 2. Anchor points where 3D ↔ 4D transitions happen

These are the *recurring* boundaries. Cited against
`conversion/models/gemma4_swa_chunks.py` (the reference file; other chunk
files have structurally identical anchor sequences).

| # | What | File:line | Direction | Per-layer count |
|---|---|---|---|---:|
| A1 | Input to attention: `h.permute(0,2,1).unsqueeze(2)` after `input_layernorm` | gemma4_swa_chunks.py:67 | 3D→4D | 1 |
| A2 | Attention output back to 3D for residual add: `.squeeze(2).permute(0,2,1)` | gemma4_swa_chunks.py:148 | 4D→3D | 1 |
| A3 | Input to MLP: `h.permute(0,2,1).unsqueeze(2)` after `pre_feedforward_layernorm` | gemma4_swa_chunks.py:155 | 3D→4D | 1 |
| A4 | MLP output back to 3D: `mlp_out.squeeze(2).permute(0,2,1)` | gemma4_swa_chunks.py:160 | 4D→3D | 1 |
| A5 | Per-layer-input path: 3D→4D→3D around gate/projection | gemma4_swa_chunks.py:169, 172, 175 | both | 3 |
| A6 | QKV head-split: `.view(1, H, D, 1).permute(0, 1, 3, 2)` | gemma4_swa_chunks.py:70, 84–85 | 4D→4D (re-axis) | 3 |
| A7 | q_norm / k_norm: `reshape(1, H, D)` → `view(1, H, 1, D)` | gemma4_swa_chunks.py:71, 86 | 4D→3D→4D | 2 |
| A8 | Attention output head-merge: `.permute(0,2,1,3).contiguous().view(1,1,-1)` | gemma4_swa_chunks.py:145 | 4D→3D | 1 |
| A9 | o_proj re-wrap: `.permute(0,2,1).unsqueeze(2)` on 3D input | gemma4_swa_chunks.py:147 | 3D→4D | 1 |

**Per-layer layout-op budget under current scheme:** A1+A2+A3+A4+A5(×3)+
A7(×2)+A8+A9 = **11 layout ops per decode layer** that become either
unnecessary or mergeable if the hidden state never leaves `(1,C,1,T)`.
A6 stays (architectural). Over 35 layers that is **~385 layout ops** per
decode pass — matches the MIL-level count.

**Anchor A2 is the residual add.** This is the single spot where the 4D→3D
round-trip exists *solely* because `post_attention_layernorm` (an
`ANERMSNorm`) operates on the last dim of a rank-3 tensor. Fixing A2 fixes
A3, which fixes A4, and so on — they are not independent anchors but a
chain driven by the RMSNorm layout fight.

---

## 3. Key obstacle: RMSNorm on the channel dimension

`ANERMSNorm.forward` (`conversion/ane_ops.py:57-72`):

```python
doubled = torch.cat([x, -x], dim=-1)
normed = F.layer_norm(doubled, normalized_shape=(2 * self.hidden_size,), ...)
normed, _ = torch.chunk(normed, 2, dim=-1)
return normed * self.weight   # self.weight shape (hidden_size,)
```

Three structural facts make this hard to NCHW-ify:

1. `F.layer_norm` with `normalized_shape=(2*hidden,)` operates on the
   **last** dimension. If hidden state is `(1, C, 1, T)` with C=hidden, the
   norm axis we need is dim-1, not dim-(-1).
2. The `cat([x, -x], dim=-1)` doubles the normalized dim. This is a
   deliberate ANE trick to make `layer_norm` behave as RMSNorm (zero-mean
   input ⇒ mean-subtract is a no-op ⇒ `layer_norm` = `rmsnorm`). Under NCHW
   the `dim=-1` cat would double the T axis, which is wrong.
3. `self.weight` has shape `(hidden_size,)`. In 3D land it broadcasts
   to the last axis for free; in NCHW land it needs `(1, C, 1, 1)` view.

### 3.1 Three candidate replacements

**Option α — channel-dim layer_norm.** `F.layer_norm` supports
`normalized_shape=(C,)` when the last dims of the input match. With
`(1, C, 1, T)` input and `normalized_shape=(C,)` you would need to transpose
to `(1, 1, T, C)` first, negating the win. **No, this is a dead end.**

**Option β — instance_norm on channel dim via group_norm(groups=1).**
`F.group_norm(x, num_groups=1)` normalizes each `(C, H, W)` slice per batch,
which is exactly RMSNorm over C for NCHW with `H*W` as the token axis.
Mean-centering is harmless for the cat-trick (we can skip the cat because
group_norm already subtracts the mean). coremltools lowers `group_norm`
to a custom MIL `group_norm` op in opset `CoreML8` — **ANE support is
partial**: confirmed for standard groups in iOS 18 vision models, unknown
for rank-4 with H=1. **Risk: untested on ANE at hidden=1536 channels.**
The doubled-cat trick isn't needed under group_norm because mean subtraction
is built-in, so we lose the `layer_norm` fast path but gain a 2× narrower
axis (hidden instead of 2*hidden). Net: probably wins, but needs an on-device
ANE-residency probe before committing.

**Option γ — manual RMSNorm via reduce_sum + rsqrt + mul.**
Compute `rsqrt(mean(x**2) + eps)` with `reduce_sum(axis=1, keepdims=True)`.
Four primitive ops (mul, reduce_sum, add, rsqrt) + one mul for the scale =
5 MIL ops per norm. coremltools 9 lowers all of these to ANE kernels in fp16
without ceremony. **Downside:** loses the ANE-native `layer_norm` fused
kernel that the audit confirmed is in use (49–57 `layer_norm` ops per
chunk). Whether the 5-op manual sequence is faster than 1 fused
`layer_norm` on ANE is **device-dependent**. On A17 Pro the fused kernel is
reported ≈ 2× a comparable manual sequence (source: Apple ANEMLL team via
ANE_OPTIMIZATION_SURVEY §2). **Likely loses 10–15 % of the layer norm
budget**, which in decoder chunks is ~8 % of total runtime — net cost
~1.2 %. Still cheaper than the 10–20 % layout-op savings if the math
works.

**Option δ — keep RMSNorm in 3D land, transpose around it.** Accept one
`transpose(1, C↔last)` before each norm and one after. This reduces A1/A2
anchors to pure transposes (no unsqueeze), which the MIL compiler folds more
aggressively than unsqueeze+permute chains. Keeps the `layer_norm` fast
path. **Probably the best trade**: saves ~60 % of layout ops while
preserving the ANE-native LayerNorm. Test this first.

### 3.2 Weight-loading consequence

`ANERMSNorm.weight` is shape `(hidden,)` regardless of option. HF weights
load into this 1D parameter. No state_dict surgery needed for options α/β/δ.
For γ the weight shape is the same and only the forward changes. Fully
back-compatible.

---

## 4. Attention reformulation

Current flow (gemma4_swa_chunks.py:67 → 148):

```
hidden_states (1, 1, C) → permute+unsqueeze → (1, C, 1, 1)            [A1]
q_proj: conv 1×1 → (1, H*D, 1, 1)
view+permute → (1, H, 1, D)                                            [A6]
q_norm: reshape (1,H,D) → norm → view (1,H,1,D)                        [A7]
rope applied to (1, H, 1, D)
matmul Q @ K^T:  (1, H, 1, D) @ (1, H, D, S) → (1, H, 1, S)
softmax
matmul A @ V:    (1, H, 1, S) @ (1, H, S, D) → (1, H, 1, D)
permute+view → (1, 1, H*D)                                             [A8]
permute+unsqueeze → (1, H*D, 1, 1)                                     [A9]
o_proj conv → (1, C, 1, 1) → squeeze+permute → (1, 1, C)              [A2]
```

A6/A7 are architectural (q_norm needs per-head-dim).
A1/A2/A8/A9 are all removable if the hidden state stays 4D.

### 4.1 Proposed unified NCHW attention

```python
# Hidden state lives as (1, C, 1, T) throughout. C=hidden, T=q_len.
# input_layernorm now normalizes dim=1 (channel):
h = channel_rmsnorm(hidden)        # (1, C, 1, T), no reshape

# QKV proj (already a 1×1 conv, already NCHW):
q4 = q_proj(h)                     # (1, H*D, 1, T)
k4 = k_proj(h)                     # (1, Hkv*D, 1, T)
v4 = v_proj(h)                     # (1, Hkv*D, 1, T)

# Split into heads — ARCHITECTURAL, unavoidable:
q = q4.view(1, H, D, T)            # (1, H, D, T)  — still NCHW-ish
k = k4.view(1, Hkv, D, T)
v = v4.view(1, Hkv, D, T)

# q_norm / k_norm on D (last-but-one):
# Option: permute to (1, H, T, D) once, norm, keep.
q = q.permute(0, 1, 3, 2)          # (1, H, T, D)
k = k.permute(0, 1, 3, 2)

q = q_norm(q)                      # normalize over D (last dim) — native
k = k_norm(k)

# RoPE (operates on last dim D already) — unchanged.

# Attention (4D already):
attn = softmax((q @ k.transpose(-1,-2)) + mask, dim=-1) @ v_broadcast
# attn shape: (1, H, T, D)

# Merge heads back to NCHW channel: permute+view back to (1, H*D, 1, T)
attn = attn.permute(0, 1, 3, 2).reshape(1, H*D, 1, T)

# o_proj is a 1×1 conv → (1, C, 1, T). Back in the residual stream.
h = residual + o_proj(attn)        # (1, C, 1, T), no reshape to 3D
```

**Layout-op delta per attention block:**

| | current | NCHW |
|---|---:|---:|
| 3D↔4D wraps around linears | 4 (A1, A2, A9 + implicit o_proj wrap) | **0** |
| head split/merge | 2 (view, permute) + 2 (view, permute) | 2 + 1 (a final reshape absorbed in o_proj input) |
| q_norm / k_norm reshape fight | 2 | **0** (permute once, stay) |
| **total** | **10** | **3** |

Savings: **7 layout ops per attention block × 35 layers = 245 ops per
decode pass**. Plus ~3 ops per MLP block (A3, A4, and the pre_feedforward
norm wrap) × 35 = 105. Plus ~3 per per-layer-input path × 35 = 105. Total
≈ 450, closely matching the MIL-graph removable ceiling.

### 4.2 Numerics

Q @ K^T is the numerical hot spot. Gemma 4 uses effective scale=1.0 after
q_norm/k_norm absorbs the head-dim scaling, so the Q@K^T output magnitude
is already in fp16 range without upcast (confirmed
`gemma4_swa_chunks.py:135-142` comment). The NCHW reformulation does not
change the arithmetic order — same matmul, same operands, same cast sites.
**Bit-identical output expected.** Risk is zero from the math side. All
risk is in whether the MIL compiler still picks an ANE kernel for the
reshaped attention; see §7.

---

## 5. Scope / LOC estimate

### 5.1 Python (converter) surgery

| File | Existing LOC | Estimated touch | Notes |
|---|---:|---:|---|
| `conversion/ane_ops.py` | 332 | **+60 / −10** | Add `ANERMSNorm4D` variant (channel-dim norm via option δ transpose-pair or option γ manual). Keep old class for legacy models. |
| `conversion/models/gemma4_swa_chunks.py` | 834 | **+150 / −200** | Full rewrite of `_run_layer_swa` and `_run_layer_verify`. About 50% of the body is layout gymnastics that just vanishes. Net shrink. |
| `conversion/models/gemma4_swa_merged2.py` | 195 | **+30 / −40** | Mostly the PLE path (lines 74-88) and lm_head tail (lines 188-195) restructuring. |
| `conversion/models/gemma4_swa_merged1.py` | 163 | **+30 / −40** | Same class of change as merged2. |
| `conversion/models/gemma4_prefill_chunks.py` | 349 | **+80 / −120** | Prefill has the same pattern but with a real seq_len, so the head-split reshapes are bigger but structurally identical. |
| `conversion/models/gemma4.py` (the base Gemma4Model) | ~450 | **+20 / −10** | `Gemma4DecoderLayer.__init__` currently builds 4 `ANERMSNorm` per layer; need to switch to the 4D variant or wrap. |
| **Subtotal** | ~2300 | **~370 lines of diff** | |

### 5.2 Builder / spec changes

| File | LOC | Touch | Notes |
|---|---:|---:|---|
| `conversion/build_merged_chunks.py` | 325 | **+10 / −5** | `ct.TensorType(name="hidden_states", shape=(1,1,hidden))` → `(1, hidden, 1, 1)` on input and `hidden_states_out` on output. All KV shapes unchanged. |
| `conversion/build_verify_chunks.py` | 524 | **+15 / −5** | Same shape flip; verify output for `normed` (drafter carry-state) also changes shape. |
| `conversion/build_prefill_gpu.py` | unseen | **+10 / −5** | Prefill hidden_states needs the same flip. |
| **Subtotal** | ~900 | **~40 lines** | |

### 5.3 Test harness

`conversion/test_merged_parity.py` (297 LOC) — parity test compares Python
forward vs CoreML forward. Shapes flow end-to-end. Changes needed:
- Input fixture `hidden_states` creation (1–3 spots).
- Output name lookup for `hidden_states_out` (unchanged semantically, new
  shape assertion).
- Tolerance stays at fp16 default (1e-3 rel / 1e-3 abs) because the
  arithmetic is bit-identical. **~20 LOC**.

### 5.4 Swift runtime

`Sources/CoreMLLLM/ChunkedEngine.swift` (1873 LOC):
- Line 143: `MLMultiArray(shape: [1, 1, hiddenSize], …)` → `[1, hiddenSize, 1, 1]`.
- Lines 747, 761, 773, 785, 803, 812, 815, 821 and analogous sites: the
  MultiArray is passed through — shape-agnostic as long as the buffer is
  `hiddenSize * 2` bytes. The bind uses `MLFeatureValue(multiArray:)` so
  no axis math. **Actual Swift changes: ~10 spots × 3 lines = 30 LOC.**
- Drafter carry-state `normed` hand-off: drafter input expects the hidden
  state in its chunk4 output layout. If drafter is not NCHW-rewritten, we
  need one `MLMultiArray` reshape on the Swift side. ~10 LOC. **Note**:
  alternatively, leave chunk4's final `normed` output as 3D — add one
  permute+squeeze at the output boundary only. Recommended.

Total Swift: **~40 LOC**.

### 5.5 Effort summary

| Component | LOC delta | Hours | Days |
|---|---:|---:|---:|
| Python converter (chunks + ane_ops) | ~370 | 20 | 2.5 |
| Builder / spec | ~40 | 3 | 0.5 |
| Parity test | ~20 | 2 | 0.3 |
| Swift runtime | ~40 | 3 | 0.5 |
| Parity re-validation (incl. on-device probe) | — | 16 | 2 |
| ANE silent-fallback audit (ComputePlanAudit) | — | 8 | 1 |
| Debug & numerical diffs from unexpected MIL-pass choices | — | 20 | 2.5 |
| **Total** | **~470 LOC** | **~72 h** | **~9 working days** |

Call it **2 engineering weeks plus 1 week slack** = 3 calendar weeks.

---

## 6. Incremental validation plan

The task spec asks whether we can A/B a single chunk. Yes, and we should.

### 6.1 Start with SWAChunk1 decode

Reasons to pick chunk1 first:
- It has the PLE path, the most layout-gymnastics-heavy non-attention
  block — high signal if layout savings are real.
- It is the first chunk in the chain so its input is literally the Swift-
  side embedding lookup; we control the producer.
- 8 layers (L0–7) is well within the ANE stability ceiling.
- Its output handoff is hidden_states + KV + per_layer_combined, and
  per_layer_combined layout is internal to the rewrite — no cross-chunk
  wire-format change required if we keep output hidden_states as 3D
  (`squeeze+permute` at the boundary only) on this first rewrite.

### 6.2 A/B procedure

1. **Build** `chunk1_nchw.mlpackage` alongside the current `chunk1.mlpackage`.
2. **Parity test**: `test_merged_parity.py` with a `--variant=nchw` flag,
   fp16 cosine ≥ 0.9999 on 16 random seeds × 256 prompt tokens.
3. **MIL graph diff**: run the audit script on both, confirm
   `transpose + reshape` count drops by ≥ 40 % (ceiling).
4. **ComputePlanAudit on-device**: verify 0 % non-ANE ops for the new
   chunk. **Kill switch** if any `group_norm` or `layer_norm` falls to CPU.
5. **Tok/s bench** in the existing chat app: 50 prompts × 256 tokens, fresh
   device boot, compare steady-state tok/s.
6. **Decision gate**: ≥ 3 % decode-tok/s gain on chunk1 alone projects to
   ≥ 10 % on the full stack (linearity hypothesis, rough). If chunk1 does
   not clear 3 %, the whole rewrite is in doubt — abandon and revisit the
   RMSNorm choice or the layout fight.

### 6.3 Rollout

- Chunk2 rewrite — 2 days after chunk1 lands.
- Chunk3 + chunk4 — 2 days together (they are pure shared-KV layers, less
  risk).
- Merged1 (35-layer) — **only if** ComputePlanAudit passes on 4-chunk NCHW.
  If merged1 already falls off ANE in the current 3D pipeline, adding NCHW
  layout changes on top will not save it.

---

## 7. Risk register

### R1 (HIGH) — RMSNorm ANE fast path loss

If we swap `cat+layer_norm` for `group_norm` (option β) or manual
`rsqrt(mean(x**2))` (option γ), coremltools may emit a MIL op that the ANE
compiler does not map to its fused LayerNorm kernel. This could regress
the **~49 `layer_norm` ops per chunk** from 1-cycle-each ANE dispatches to
CPU-fallback or multi-cycle ANE sequences.

**Mitigation**: start with option δ (transpose pair around preserved
`layer_norm`). Profile before committing to β/γ.

### R2 (HIGH) — Silent ANE placement regression

`EXPERIMENTS.md` documents a stability ceiling around 15 layers per
function. Merged1 is already past it. Adding NCHW rewrites that introduce
new MIL op subgraphs the compiler has not seen before can tip a marginal
placement decision. Symptom: model compiles, loads, runs slower than
before.

**Mitigation**: ComputePlanAudit (`Sources/CoreMLLLM/ComputePlanAudit.swift`
exists) must be wired into the post-build pipeline. Refuse to ship a
variant where any op lands off ANE unless `test_merged_parity.py` on-device
tok/s beats the baseline by ≥ 3 %.

### R3 (MEDIUM) — Numerical drift from attention reformulation

The reformulation described in §4.1 is mathematically identical (same
matmul, same scale=1.0, same softmax axis). **However**, coremltools may
choose different fp16 intermediate rounding under the `scaled_dot_product_
attention_sliced_q` pass when the Q/K/V shapes change. We have an existing
parity-validated baseline on 50 prompts × 256 tokens; any numerical drift
above 1e-3 cosine breaks that validation and requires re-collection of
the reference set.

**Mitigation**: keep the SDPA sliced-Q pass *off* during NCHW rollout, turn
it on in a follow-up patch.

### R4 (MEDIUM) — Drafter layout contract

`mtp_drafter.mlpackage` consumes kv13/kv14 and carry-state `normed` from
chunk4. If chunk4's `normed` output shape changes from `(1, 1, hidden)` to
`(1, hidden, 1, 1)`, the drafter either has to match or the Swift runtime
has to reshape. MLPACKAGE_STRUCTURE §4.4 already flagged a V-layout
mismatch between chunk4 and drafter (`kv14_v 1x1x8192x512` vs
`1x1x512x8192`); adding another mismatch compounds the contract risk.

**Mitigation**: **preserve the outer boundary shape** on hidden_states.
The rewrite stays internal to the chunk. One `squeeze+permute` at the
output of each chunk before wire-format — not counted in the savings above
but it's a 2-op-per-chunk tax, negligible (< 0.2 % of total).

### R5 (LOW) — q_norm/k_norm behaviour under `(1,H,T,D)` vs `(T,H,D)`

Current code reshapes to `(T, H, D)` to run `ANERMSNorm` per-token. If we
normalize directly on `(1, H, T, D)`'s last dim, the output is identical
(`normalized_shape=(D,)` reduces over D for each of the `H*T`
positions, which is exactly per-token, per-head normalization). **No math
change.** Low risk because of the `F.layer_norm` contract.

### R6 (LOW) — Parity-test reference churn

`test_merged_parity.py` loads saved reference tensors. Their shape is part
of the checked-in fixture. We need to either re-record under NCHW or add
a permute adapter in the test. Preference: adapter. 20 LOC, no new
fixtures.

### R7 (LOW) — Build pipeline integration

`build_verify_chunks.py` and `build_merged_chunks.py` emit TensorType specs
by reading shapes from the traced module's example inputs. Changing the
example shape mostly propagates for free, but `build_prefill_gpu.py` has
hardcoded shape literals (need to audit all three). ~30 min per builder.

### R8 (LOW) — `vision_video.mlpackage` and audio package unaffected

Confirmed by §1 scope: the rewrite is decoder-chunk only. Vision-video
parity (per MEMORY.md `gemma4_video_phase2`) stays intact because vision
outputs feed Swift `hidden_states` which already has the reshape on the
consumer side.

---

## 8. What we are *not* doing

- **Drafter rewrite**: out of scope. Keep drafter 3D, eat one Swift-side
  `reshape` per spec cycle. Impact: ≤ 0.1 ms.
- **KV layout change**: out of scope. KV cache stays
  `(N_slots, 1, T_cache, max_hd)`. The `K_full_in[fi].unsqueeze(0)` slot
  indexing remains. That's a separate effort (Track K) with its own
  Swift-side cost.
- **Prefill gpu pipeline**: defer. Prefill is run rarely; a separate pass
  can NCHW-ify it later without blocking decode.
- **Stateful mlpackage path** (`Examples/CoreMLLLMChat/CoreMLLLMChat/
  stateful_chunk2.mlpkg`): legacy, targeted for removal per
  MLPACKAGE_STRUCTURE §6-7 recommendation. Do not port.

---

## 9. Expected payoff

From MLPACKAGE_STRUCTURE §4.3 the cited upper bound is 10–20 % decode-tok/s.
Breakdown:

- 245 attention layout ops removed × ~40 µs avg (ANE transpose cost on
  small tensors) = **~10 ms per decode pass**.
- 105 MLP layout ops × 40 µs = ~4 ms.
- 105 per-layer-input ops × 40 µs = ~4 ms.
- Total removable: **~18 ms per decode pass.**

At current baseline ~30 tok/s (33 ms/tok), removing 18 ms of layout work
would give 15 ms/tok → 66 tok/s theoretical ceiling. **Realistic expected
gain: 40-50 %**, because the compiler already fuses some adjacent transpose
pairs (confirmed `adjacent-transpose=0` in the audit). That is well above
the LiteRT-LM 56.5 tok/s target referenced in MEMORY.md's project_direction.

But this gain assumes no ANE fallback (R1, R2). If either triggers, the
gain collapses to zero or negative. **That is why the verdict is
GO-AFTER-D1**: land the cheaper, layout-neutral wins first, get the MIL
graph into a state where the fused `softmax` pattern lights up the ANE
attention kernel, *then* do the layout rewrite on the hottest chunk and
let on-device numbers decide whether to continue.

---

## 10. Go/no-go checklist before starting D4

- [ ] D1 (softmax fused) landed and parity-green.
- [ ] D2 (SDPA sliced-Q pass) landed or explicitly deferred with
      justification.
- [ ] D5 (chunk4 lm_head palettize) landed — chunk4 ships at ≤ 250 MB.
- [ ] ComputePlanAudit wired as a build-time gate, not just on-demand.
- [ ] Baseline tok/s measurement recorded on fresh device boot, cold
      cache, as the regression floor.
- [ ] `test_merged_parity.py` reference re-captured after D1/D2/D5 so the
      post-D4 comparison is apples-to-apples.

When all six check, start D4 at chunk1. Follow §6.2 A/B procedure. Ship
chunk-by-chunk, not as an atomic flip.

---

### Word count (including code blocks): ~3200
