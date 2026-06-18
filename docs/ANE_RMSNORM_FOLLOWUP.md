# Follow-up: native RMSNorm for the shared `ane_ops.ANERMSNorm` (all decoder families)

**Status:** proposed, **not implemented**. Needs broad re-validation before any change.

## What we found (pplx-embed only)

`conversion/ane_ops.ANERMSNorm` implements RMSNorm via the `cat([x, −x]) → LayerNorm →
chunk` identity, chosen years ago because "the ANE has a highly optimized LayerNorm
kernel" and lacked a fast native `rsqrt`. On the pplx-embed bidirectional Qwen3 encoder
we A/B'd that against a native `x * rsqrt(mean(x²) + eps) * w` RMSNorm
(`conversion/models/qwen3_encoder.Qwen3RMSNorm`, selectable via `norm_impl`), changing
**only** the 5 norm sites and holding Conv2d-1×1 projections and tensor layout fixed
(`conversion/experiment_ane_rmsnorm.py`):

| L   | ane_cat (cat/chunk) | native (rsqrt) | speedup | ANE residency | cosine vs fp32 |
|-----|--------------------:|---------------:|--------:|--------------:|---------------:|
| 256 | 35.98 ms            | 31.92 ms       | **+12.7%** | 99.81% (both) | 0.99998 |
| 512 | 100.40 ms           | 82.61 ms       | **+21.5%** | 99.81% (both) | 0.99998 |

Environment: Apple M4 Max, macOS 26, coremltools 9, torch 2.11, B=1, K=8 residual
rescale, `pooled_fp16`. Native RMSNorm stays **fully ANE-resident** (the planner runs
`pow`/`reduce_mean`/`rsqrt` on the ANE here) and is fidelity-neutral. The cat/chunk trick
is now a *de-optimization* on this chip/OS/coremltools combination.

## Why this is only applied to pplx-embed so far

`ane_ops.ANERMSNorm` is shared by ~10 decoder families (Gemma3/4, LFM2, Qwen3.5,
Qwen3-VL, `base_model`). pplx-embed is a **bidirectional encoder, B=1, fp32 residual,
fixed full-attention** — a different regime from the **stateful causal decoders** (KV
cache, T=1 decode + T=32 prefill, sliding/full sandwich norms, the `(1+w)` gain
convention). The win may or may not carry over; a per-op kernel choice that helps a
1×L encoder pass need not help a chunked decode step.

## Proposed work (separate PR)

1. Add a `norm_impl` (or `native_rmsnorm=True`) switch to `ane_ops.ANERMSNorm`
   **without changing its default**, mirroring `Qwen3RMSNorm` (store the same 1-D weight;
   keep the `plus_one_gain` convention in `ane_norm_from_hf`).
2. A/B per family with that family's existing latency harness (e.g.
   `probe_e2e_linear_latency.py`-style), measuring decode **and** prefill, ANE residency,
   and end-to-end output parity — *not* just a single encoder pass.
3. Flip the shared default to native **only** for families where it is faster *and*
   residency/parity hold; leave the others on `ane_cat`.

Do **not** flip the shared default globally off the pplx-embed result alone.
