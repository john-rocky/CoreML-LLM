# ANE route for MoE — exhausted across 7 measured designs

Date: 2026-05-15
Branch: `feat/mtp-iphone-perf`
Companion to `docs/QWEN_MOE_MLX_BREAKTHROUGH_2026_05_15.md`.

## Why this document exists

The session twice declared the Qwen MoE pivot dead too quickly. The
user pushed back hard — don't give up from a single perspective,
question the premise, find the way through. So the ANE route was then
pursued **exhaustively**: 7 distinct designs, every one measured on the
real Swift production path (`Sources/moe-dispatch-probe`,
IOSurface-backed buffers, warm models, 200 timed iterations) — no
Python `predict()` proxies, no extrapolation of the per-dispatch cost.

This is the record so the ANE route is not re-litigated without new
silicon or a new model class.

## The 7 ANE designs and their measured results

All on Mac M-series, INT4 where applicable, extrapolated to a 24-layer
Qwen1.5-MoE-A2.7B decode.

| # | Design | Measured (ANE) | Full-decode extrapolation |
|---|---|---|---|
| 1 | `gather` op (60 experts constant, runtime gather top-4) | 46 ms/layer | dead — ANE rejects runtime gather, falls to CPU emulation |
| 2 | per-expert mlpackage (call 4 separate) | 0.31 ms/expert | ~33 tok/s |
| 3 | **multifunction (60 functions, call 4 by functionName)** | **0.69 ms/layer** | **~32 tok/s — ANE ceiling** |
| 4 | weights-as-input, dense+routed fused into 6-layer chunk | 27 ms/6L-chunk | ~9 tok/s — 415 MB/chunk input transfer dominates |
| 5 | MLState-backed expert weights (2 state buffers) | 2.8 ms/layer | ~15 tok/s — dynamic-weight matmul slow even from state |
| 6 | expert co-firing grouping (cut dispatch count 4→2) | — | FLAT: co-fire ratio 0.76× of random; load-balancing loss decorrelates experts by design |
| 7 | switch to a smaller-expert-count MoE | — | no silver bullet: no small MoE has 2 active experts; OLMoE-1B-7B is the best feasible but its 8×16=128 routed dispatches ≈ Qwen's 96+24 |

## The structural conclusion

ANE is a static-dataflow accelerator with ~0.15-0.3 ms fixed
per-dispatch overhead and **no efficient runtime gather**. MoE wants
per-token dynamic selection of K-of-N experts. The two are
fundamentally at odds:

* To avoid computing all N experts, you must gather (skip the
  unselected) — ANE can't (design 1).
* So routed experts become separate dispatches — and ANE's per-dispatch
  overhead × the irreducible per-layer dispatch count is the wall
  (designs 2, 3 — the best, ~32 tok/s).
* Making the routed-expert graph static via weights-as-input (design 4)
  or MLState (design 5) keeps it on ANE but the dynamic-weight matmul
  and the weight-movement cost are both slow — ANE wants weights as
  pre-baked constants.
* The dispatch count can't be reduced by grouping (design 6 — experts
  are trained to be decorrelated) or by model choice (design 7 — no
  small 2-active MoE exists).

Best ANE result for Qwen-class MoE: **~32 tok/s** (multifunction),
which is ~parity with the current Gemma 4 E2B CoreML/ANE deployment
(~35 tok/s Mac), not the 1.5× target.

**ANE cannot deliver a MoE speed win.** Not because the investigation
was shallow — because the MoE compute pattern structurally conflicts
with the ANE programming model. This holds until either (a) the ANE
driver gains efficient runtime gather, or (b) a MoE model with a
radically lower active-expert-per-layer count appears.

## The contrast: MLX

The same model, same Mac, via MLX: **219 tok/s** at 3-bit (see
`docs/QWEN_MOE_MLX_BREAKTHROUGH_2026_05_15.md`). MLX is an array
framework with JIT Metal kernels — it fuses the MoE gather + batched
expert matmul into one efficient kernel. It is not bound by the
static-graph + no-gather constraints that cap ANE.

So the honest split:
* **ANE route: exhausted at ~32 tok/s.** Pursued thoroughly per the
  directive; the negative is structural and now 7-ways confirmed.
* **MLX route: alive at 219 tok/s Mac.** The deployment path forward,
  but it abandons ANE's power efficiency and the blocker becomes
  iPhone memory + an unmeasured iPhone tok/s.

The decision between "ANE parity (power-efficient, ~32 tok/s)" and
"MLX speed (GPU, ~219 Mac / iPhone TBD)" is a product call. The
engineering facts for both sides are now measured, not guessed.

## Artifacts

* `conversion/phase_b_redux_build.py` — builds all 7 designs' mlpackages
  (single-expert, layer-gather, multifunction, dense-backbone,
  fused-moe, mlstate-expert; fp16 + INT4).
* `conversion/ane2_qwen_moe_cofiring.py` — expert co-firing analysis
  (verdict FLAT).
* `Sources/moe-dispatch-probe/main.swift` — the Swift harness measuring
  real dispatch latency for every design. Reusable for any future
  small-kernel routing scheme on a static-graph runtime.
