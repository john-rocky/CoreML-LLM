# native softmax vs decomposed ane_softmax — Mac bench: no delta (2026-04-24)

## Question

The shipped `_run_layer_swa` uses a decomposed softmax
(`max → sub → exp → sum → div`, 5 ops) instead of the native
`torch.nn.functional.softmax` op. The decomposed form exists because
earlier iOS/chip combinations rejected native softmax on ANE; ctools 9
+ iOS 26 might no longer need the workaround. Swap in a native-softmax
variant of the merged 17-layer `chunk2_3way` and measure.

Expected upside (from SDPA fusion probe): 4 MIL ops saved per attention
layer × 17 layers ≈ 68-119 fewer total ops. Whether those ops map to
measurable ANE time was unknown.

## Method

`USE_NATIVE_SOFTMAX=1 python conversion/build_gemma4_3way.py --only chunk2`
builds a drop-in replacement for `chunk2_3way.mlmodelc` that emits
native softmax. Swapped into the live bundle, ran
`coreml-llm-smoke` at steady-state. Reverted to the decomposed
variant after measurement.

Op census confirmed the expected shape:

- Decomposed: 4435 ops (sub:18, reduce_max:17, exp:17, reduce_sum:17, real_div:17, ...)
- Native softmax: 4316 ops (softmax:17, sub:1, ...)
- **Δ = 119 ops removed** by the swap

## Result — Mac (Apple Silicon)

| Variant | steady tok/s | c1 | c2 | c3 | c4 | sum |
|---|---:|---:|---:|---:|---:|---:|
| Decomposed `ane_softmax` | 35.26 | 5.2 | 12.7 | 0.0 | 10.5 | 28.4 ms |
| **Native softmax**       | **35.19** | 5.2 | 12.7 | 0.0 | 10.5 | 28.4 ms |

**Zero delta**, within measurement noise (±0.1 tok/s). The 119 op
reduction apparently costs the same ANE time as the fused `softmax`
op it was replaced with — Apple's ANE either (a) fuses the decomposed
form under the hood to the same kernel, or (b) the softmax is
compute-trivial relative to the surrounding matmul/layer_norm work.

Either way, **no speed win on Mac**. Per the Orion paper's dispatch-
dominated-decode model, the iPhone A19 Pro is extremely unlikely to
show a different signal — the ANE bottleneck is per-chunk
round-trip, not per-op compute inside a chunk. Upside is bounded; the
only divergent outcome is a regression if native softmax falls off
ANE on device.

## Verdict — skip on shipping path

Reverted the swap. Moving to HF re-upload and ship PR #131's 34.2
tok/s 3-chunk as the current ANE-only ceiling.

## Artifact

`conversion/ane_ops.py` gains a `USE_NATIVE_SOFTMAX=1` env toggle so
the A/B is one env var + one converter run, and can be re-probed cheaply
on future OS/chip combinations without code surgery.
