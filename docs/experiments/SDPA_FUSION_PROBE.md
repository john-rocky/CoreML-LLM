# SDPA fusion re-test — dead (2026-04-24)

## Question

`docs/SURVIVING_HYPOTHESES.md` §B2 flagged that coremltools 8.3+ shipped a
`scaled_dot_product_attention_sliced_q` pass with a reported ~34% ANE
speedup for attention-heavy graphs. The prior attempt (2025, iOS 18,
ctools pre-8) had failed under Gemma 4's post-QK-norm `scale=1.0`: the
fused op produced numerically different outputs and broke token parity.
Worth retrying on ctools 9 / iOS 26 with the `scale=1.0` edge handled
properly.

## Method

`conversion/probe_sdpa_fusion.py` minimal (no-weights) attention block,
SWA shape (Q=1, KV=512, H=8, D=256, mask=0). Two variants traced and
converted separately with `ct.target.iOS18`, `precision.FLOAT16`:

- **manual** — the current `_run_layer_swa` body:
  `matmul → add(mask) → ane_softmax(decomposed) → matmul`
- **sdpa** — `F.scaled_dot_product_attention(q, k, v, attn_mask, scale=1.0)`

Each resulting mlpackage is compiled to mlmodelc and walked via
`MLComputePlan` to census op types.

## Result

PyTorch parity is essentially identical (cos = 1.0001, max_abs = 7.6e-6 in
fp16) — SDPA and manual agree within roundoff.

MIL op census:

| Variant | ops | top types |
|---|---:|---|
| manual | 16 | const:8, matmul:2, add:1, reduce_max:1, sub:1, exp:1, reduce_sum:1, real_div:1 |
| **sdpa**   | **9**  | const:5, matmul:2, add:1, **softmax:1** |

**Neither path produces a fused `scaled_dot_product_attention` op.**
ctools 9 lowers both forms to a plain `matmul → add → softmax-or-decomposed → matmul`
chain. The SDPA variant does use the native `ios18.softmax` instead of the
decomposed `reduce_max/sub/exp/reduce_sum/real_div` that `ane_softmax`
produces — but this is the only material difference, and `ane_softmax`
exists specifically because the native op was avoided on earlier OS/chip
combinations.

## Verdict — skip

The `34% ANE speedup` claim depends on the *fused* op materializing. On
ctools 9.0 iOS 18 target for both SWA and full-attention layers, the
fused op does not materialize for either manual or SDPA input. We can't
claim the speedup without the fused op.

Switching `ane_softmax` → native `softmax` is a separate question
(potentially 0–5% via shorter op chain, real risk of ANE fallback on the
native softmax). If we want to pursue that, it should be benched directly
(`ane_softmax` → `F.softmax` swap, full-chunk rebuild, on-device audit) —
not routed through F.scaled_dot_product_attention.

**Moving to KV-share Q-batching (next ANE-only lever) instead.**

## Reproducing

```
python conversion/probe_sdpa_fusion.py
```

Writes `output/sdpa_probe/{manual,sdpa}.{mlpackage,mlmodelc}` and prints
the op census side-by-side.
