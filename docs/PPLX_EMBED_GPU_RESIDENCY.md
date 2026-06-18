# pplx-embed encoder: why GPU residency is low under `CPU_AND_GPU`

**TL;DR.** The reviewer's hypothesis — that the ANE-oriented graph (Conv2d‑1×1,
cat/chunk RMSNorm, **fp32** residual/attention/norm) forces ops onto the CPU under
`CPU_AND_GPU` — is **refuted**. Two independent tests prove it:

1. An **fp16-only** encoder (residual + attention + norms all fp16) produces a
   *byte-identical* MLProgram op→device tally to the fp32 encoder, with identical
   latency. coremltools' default conversion pipeline already lowers the whole fp32
   trace to fp16 in MIL, so there are **no fp32 compute ops left to push to the CPU**.
2. A **fully GPU‑native** encoder rebuilt from scratch (`nn.Linear`/matmul instead
   of Conv2d‑1×1, native `x*rsqrt(mean(x²))·w` RMSNorm instead of cat/chunk, plain
   `(B,S,H)` layout, no permutes/tile) gets **12% GPU** — statistically the same as
   the shipped encoder's **8.5%**. Removing every ANE‑ism did *not* move the needle.

The real cause is a property of CoreML's **static GPU planner**: under
`CPU_AND_GPU` it places only **weight‑backed matmul-family ops** (`conv`/`linear`,
plus `silu`, `gather`) on the GPU and routes **all elementwise, reduction, softmax,
layout, and the attention matmul** to the CPU. At **B=1** the resulting CPU↔GPU
handoffs cost more than the GPU saves: **`CPU_AND_GPU` is *slower* than `CPU_ONLY`**
(0.70–0.83×). This is not a fixable graph/implementation issue — it is how the
backend partitions a single‑sequence transformer.

Environment: Apple M4 Max, macOS 26, coremltools 9, torch 2.11. Fixed‑shape
`pooled_fp16`, K=8 residual rescale, B=1, L∈{256,512}. Static placement via
`MLComputePlan` on the compiled `.mlmodelc`; timing is `MLModel.predict` median.

---

## 1. Op / dtype breakdown (L=256, B=1, `CPU_AND_GPU`)

Every non-const compute op in the MLProgram is **FLOAT16** (confirmed by reading the
MIL spec proto: 1976 fp16 outputs, 2 int32, 1 bool). The dtype hypothesis fails at
the proto level — there is no fp32 to assign anywhere.

### Shipped ANE-tuned encoder (Conv2d‑1×1, cat/chunk RMSNorm) — **8.5% GPU**

| op type        | count | device      | note |
|----------------|------:|-------------|------|
| mul            | 452   | **CPU** 100% | RoPE, RMSNorm scale, masking |
| transpose      | 252   | **CPU** 100% | (B,C,1,S) layout shuffles |
| conv (1×1)     | 196   | **GPU 71%** / CPU 29% | q/k/v/gate/up→GPU; o/down (out=1024)→CPU |
| reshape        | 169   | **CPU** 100% | |
| concat         | 169   | **CPU** 100% | cat([x,−x]) RMSNorm + rotate_half |
| split          | 169   | **CPU** 100% | chunk() RMSNorm + rotate_half |
| add            | 141   | **CPU** 100% | residual adds, mask add |
| layer_norm     | 113   | **CPU** 100% | the RMSNorm kernel |
| expand_dims    | 113   | **CPU** 100% | |
| tile           | 57    | **CPU** 100% | GQA k/v expansion |
| matmul         | 56    | **CPU** 100% | **attention scores + ctx (no weight const)** |
| softmax        | 28    | **CPU** 100% | attention |
| silu           | 28    | **GPU** 100% | MLP activation |
| gather         | 1     | **GPU** 100% | embedding lookup |

### GPU‑native rebuild (Linear/matmul, native RMSNorm, (B,S,H)) — **12% GPU**

| op type      | count | device | note |
|--------------|------:|--------|------|
| linear       | 196   | **GPU** 100% | all projections now GPU |
| silu         | 28    | **GPU** 100% | |
| gather       | 1     | **GPU** 100% | |
| mul / add    | 452 / 254 | CPU 100% | RoPE, RMSNorm scale, residual |
| reshape      | 169   | CPU 100% | |
| pow / reduce_mean / rsqrt | 113 each | CPU 100% | native RMSNorm internals |
| transpose    | 112   | CPU 100% | |
| split / concat | 56 each | CPU 100% | rotate_half |
| matmul / softmax | 56 / 28 | CPU 100% | **attention still on CPU** |
| tile / expand_dims | 57 / 57 | CPU 100% | GQA expansion |

**Observation:** switching Conv2d→Linear moved `o_proj`/`down_proj` onto the GPU
(196 vs 140 GPU ops), but the *entire elementwise/reduction/attention mass stayed on
the CPU* — including the attention `matmul`+`softmax`, which the planner never puts
on the GPU because they have no constant weight to anchor a GPU kernel. Net GPU share
rose only 8.5%→12%, and latency did not improve.

---

## 2. Root cause

CoreML's `CPU_AND_GPU` **static** partitioner is conservative for single-sequence
(B=1) transformers. It assigns to the GPU only the ops whose dominant operand is a
**constant weight** (the projection `conv`/`linear`, the `silu` fused after them, and
`gather`). Everything data-dependent — elementwise (`mul`/`add`), the RMSNorm
reductions, the **attention `matmul`/`softmax`**, and all layout ops (`transpose`/
`reshape`/`concat`/`split`/`tile`/`expand_dims`) — is left on the CPU. The graph then
ping‑pongs CPU→GPU→CPU around each projection.

This is independent of the encoder's ANE tuning:
- **dtype** is not the lever (everything is fp16 post-lowering; fp16/fp32 paths are
  identical ops),
- the **Conv2d‑1×1 + cat/chunk RMSNorm + (B,C,1,S) layout** is not the lever (a
  textbook Linear/native‑RMSNorm/(B,S,H) graph partitions the same way).

### Is the static plan misleading? No — confirmed by timing.

If the GPU were secretly carrying work, `CPU_AND_GPU` would beat `CPU_ONLY`. It does
not — it is **slower**, because the handoff overhead for the few GPU ops exceeds
their benefit at B=1:

| variant (L=256, B=1) | CPU_ONLY | CPU_AND_GPU | CPU_AND_NE | GPU speedup vs CPU_ONLY |
|----------------------|---------:|------------:|-----------:|------------------------:|
| ANE-tuned (fp32 resid) | 82.3 ms | 98.6 ms | **36.1 ms** | **0.83× (slower)** |
| ANE-tuned (fp16 resid) | — | 98.6 ms | 36.0 ms | — (identical to fp32) |
| GPU-native rebuild     | 68.0 ms | 96.5 ms | **28.6 ms** | **0.70× (slower)** |

At L=512 the picture is the same: GPU share rises to 14.2% (more matmul work) but
`CPU_AND_GPU` (220 ms) is still ~2.2× slower than `CPU_AND_NE` (100 ms).

The only regime where the GPU helps (per `docs/PPLX_EMBED_BATCHING.md`) is **small L
with batch B≫1** (L=128, B=16: ~1.4×), where the projection matmuls grow enough to
amortize the handoff. At B=1 there is nothing to amortize.

---

## 3. Mitigation — before/after

| metric (L=256, B=1)                | shipped (ANE fp32) | fp16 path | GPU-native | verdict |
|------------------------------------|-------------------:|----------:|-----------:|---------|
| GPU residency (static, CPU_AND_GPU)| 8.5%               | 8.5%      | 12.0%      | ~no change |
| CPU_AND_GPU latency                | 98.6 ms            | 98.6 ms   | 96.5 ms    | ~no change |
| CPU_AND_NE latency (for context)   | 36.1 ms            | 36.0 ms   | 28.6 ms    | best path unchanged |
| fidelity (cosine vs HF fp32 oracle)| 0.99993            | 0.99993   | 0.99997    | all PASS (gate 0.99) |

- **fp16 residual path:** fidelity holds (0.99993, identical to fp32), but it buys
  **zero** GPU residency or latency. Not worth shipping as a GPU lever.
- **GPU-native rebuild:** raises static GPU share modestly (8.5%→12%) and is even a
  touch faster on `CPU_AND_NE` (28.6 vs 36.1 ms — interesting as an ANE micro-opt,
  not the question here), but `CPU_AND_GPU` is unchanged and still slower than CPU.

There is **no graph change that materially raises GPU residency or makes the GPU
path win at B=1.** The bottleneck is the planner's CPU/GPU partition, not the ops we
emit.

### Does it help the dynamic RangeDim GPU model?

No. The >max-bucket flexible RangeDim model is GPU-only because **flexible shapes
force ANE fallback**, not because a GPU‑tuned graph would run well. Its ~10× slowness
vs a fixed ANE bucket is the same CPU‑heavy `CPU_AND_GPU` partition shown here plus
RangeDim overhead. A GPU-native graph would not fix it; only a fixed shape (→ANE)
does. The right lever for >max-bucket inputs is **more/larger fixed ANE buckets**, or
**chunk-and-pool** to stay within a bucket — not a GPU-tuned encoder.

---

## 4. Verdict & recommendation

**Inherent CoreML `CPU_AND_GPU` backend behavior for single-sequence transformers —
not a fixable implementation issue in our encoder.** Evidence:

1. fp16 and fp32 encoders compile to identical op→device plans (dtype is not the
   lever; everything is fp16 post-lowering).
2. A clean GPU-native graph (no Conv2d‑1×1, no cat/chunk RMSNorm, no fp32) lands at
   the same ~8–12% GPU share — removing the ANE-isms changes nothing.
3. The static plan is **accurate**, not misleading: `CPU_AND_GPU` is *slower* than
   `CPU_ONLY` (0.70–0.83×), so the GPU genuinely is not carrying meaningful work at
   B=1. The CPU/GPU handoff dominates.

**Recommendations:**

- **Keep the shipped encoder as-is (fp32 residual, ANE-tuned).** It is correct,
  fidelity-safe, and the ANE path (`CPU_AND_NE`, 99.8% ANE, 36 ms) is by far the
  fastest. Do **not** add an fp16 path as a "GPU lever" — it does nothing for the
  GPU. (An fp16 residual path was prototyped during this investigation and is
  fidelity-safe, but provides no GPU benefit, so it was not kept.)
- **Do not invest in a GPU-tuned encoder variant.** It will not beat the ANE bucket
  and will not even beat `CPU_ONLY` at B=1.
- **For the >max-bucket catch-all,** prefer adding a larger fixed ANE bucket or
  chunk-and-pool over the dynamic GPU model. If the GPU model must stay, the only
  knob that helps is **batching at small L** (≤128, B≫1, ~1.4×), which the catch-all
  use case (single long doc) does not exercise.
- **One incidental finding worth a follow-up:** the GPU-native rebuild ran the *ANE*
  path slightly faster (28.6 vs 36.1 ms at L=256). That is an ANE micro-optimization
  question (native RMSNorm vs cat/chunk on this chip/OS), orthogonal to GPU residency
  — flagged, not pursued here.

  **Resolved (follow-up).** `conversion/experiment_ane_rmsnorm.py` isolated the RMSNorm
  (changing *only* the 5 encoder norm sites, `norm_impl=native` vs `ane_cat`, holding
  Conv2d-1×1 and layout fixed): native `rsqrt(mean(x²))·w` RMSNorm is **12.7% faster at
  L=256 and 21.5% faster at L=512** on `CPU_AND_NE` (M4 Max / macOS 26 / coremltools 9),
  at **identical 99.81% ANE residency** and cosine **0.99998** vs the fp32 oracle. So the
  RMSNorm alone accounts for essentially all of the GPU-native rebuild's ANE speedup —
  the cat([x,−x])→LayerNorm trick (chosen years ago because the ANE lacked a fast native
  rsqrt) is now a *de-optimization* on this stack. **`norm_impl=native` is now the
  pplx-embed encoder default.** A shared rollout to the other decoder families' shared
  `ane_ops.ANERMSNorm` is a separate flagged follow-up (`docs/ANE_RMSNORM_FOLLOWUP.md`).

---

## Method

For each variant (shipped ANE-tuned encoder; an fp16-residual prototype; a from-scratch
GPU-native encoder using `nn.Linear`/native RMSNorm/`(B,S,H)` layout sharing the same
weights): build a fixed-shape `pooled_fp16` model (`build_pplx_embed_bundle.py`), compile
to `.mlmodelc`, tally every non-const MLProgram op by `preferred_compute_device` **and**
output dtype via `MLComputePlan` (the op×dtype×device breakdown above), then time
`MLModel.predict` (median) under `CPU_ONLY` / `CPU_AND_GPU` / `CPU_AND_NE` and check
cosine fidelity vs the fp32 `Reference` oracle. The op-device tally is read from the
compiled model's MIL spec proto.
