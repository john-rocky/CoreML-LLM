# Qwen1.5-MoE-A2.7B via MLX — the CoreML wall was a CoreML problem

Date: 2026-05-15 (same session as the β-1 and CoreML-MoE negative results)
Branch: `feat/mtp-iphone-perf`
Supersedes the pessimistic conclusion in
`docs/PHASE_BETA1_AND_QWEN_MOE_BOTH_DEAD_2026_05_15.md` for the Qwen
MoE half (the Gemma 3n β-1 half stands).

## TL;DR

The Qwen MoE pivot was declared dead twice in this session — both times
from a flawed CoreML-specific framing. Pushing past that:

* **CoreML/ANE genuinely cannot run MoE well** — the Swift
  `moe-dispatch-probe` harness confirmed it properly (not Python
  predict()): ANE per-kernel dispatch overhead ~0.15-0.3 ms ×
  irreducible per-layer dynamic dispatch count ≈ 30 tok/s ceiling.
  ANE also can't do runtime gather (35-46 ms). That part is real.
* **But MLX runs the exact same model at 201 tok/s on Mac.** The
  CoreML mlpackage-per-component model was the wall, not the model.
  MLX fuses the MoE forward into efficient Metal kernels.
* **3-bit re-quant: 219 tok/s Mac, 6.4 GB peak memory** — faster than
  4-bit and within iPhone 17 Pro's memory budget.

The deployment path is **mlx-swift, not CoreML**.

## Measurements

All on Mac (M-series), `mlx_lm generate`, steady-state generation.

| Model | tok/s (gen) | peak mem | notes |
|---|---:|---:|---|
| Qwen1.5-MoE-A2.7B-Chat-4bit | 201 / 200 / 202 | 8.8 GB | code / narrative / factual — rock steady |
| Qwen1.5-MoE-A2.7B 3-bit (3.5 bpw) | 219 / 218 | 6.4 GB | re-quant via dequant→requant; quality coherent |

3-bit is *faster* (less weight-read bandwidth per token) and cuts peak
memory by 2.4 GB. Output stayed coherent on a code prompt (correct
iterative Fibonacci) and a 3-paragraph essay.

For comparison, the CoreML probe extrapolations (Swift harness, real
IOSurface-backed dispatch) topped out at ~30 tok/s — a 6.7× gap vs the
MLX reality. Full CoreML probe data: `conversion/phase_b_redux_build.py`
+ `Sources/moe-dispatch-probe/main.swift`.

## Why CoreML lost and MLX won

CoreML's deployment model is "compile a static graph into an mlpackage,
dispatch it." MoE wants per-token dynamic expert selection. Every way
of expressing that in CoreML costs a dispatch (per-expert mlpackage,
multifunction function-call) or falls off ANE entirely (runtime
gather). ~24-170 dispatches/token × ~0.15-0.3 ms overhead = the wall.

MLX is an array framework with JIT-compiled Metal kernels. The MoE
gather + batched expert matmul is one fused kernel. No per-component
dispatch overhead. It just runs the math.

## Open blockers (must measure on-device — do NOT extrapolate)

1. **iPhone tok/s** — no published Qwen-MoE-on-iPhone-MLX benchmark
   exists; we would be first. Reference points: Qwen3 1.7B 4-bit =
   39.5 tok/s on iPhone 17 Pro via MLX. iPhone memory bandwidth is
   ~7-10× below M4-class Mac. A naive bandwidth-derate of the 219 Mac
   number lands somewhere in the 25-40 tok/s range — but that is an
   *inference, not a measurement*, and MoE's sparse compute may beat
   the naive derate.

2. **3-bit quality** — the 3-bit here was made via dequant(4-bit)→
   requant(3-bit), which is slightly lossier than a clean 3-bit from
   fp16. A production build should re-quant from the true fp16 weights
   (28 GB download) and run a proper quality eval vs the 4-bit and vs
   Gemma 4.

3. **On-device memory** — 6.4 GB peak measured on Mac. iPhone foreground
   jetsam ceiling with the Increased Memory Limit entitlement is
   ~6-8 GB (community estimate, not Apple-published). 6.4 GB is inside
   that band but with little headroom — needs on-device confirmation.

4. **mlx-swift integration** — mlx-swift officially supports iOS 17+
   with example apps (LLMEval, MLXChatExample). No code written yet for
   our use case.

## Next steps

Mac-side (no device needed):
* Clean 3-bit re-quant from true fp16 (download Qwen1.5-MoE-A2.7B-Chat
  fp16, `mlx_lm.convert --q-bits 3`), proper quality eval.
* Scaffold a minimal mlx-swift iOS app target (or adapt MLXChatExample)
  that loads the 3-bit model and reports tok/s + peak memory.

Device-side (needs iPhone 17 Pro):
* Deploy the mlx-swift app, measure real tok/s + confirm memory fits
  under jetsam.
* GO/NOGO: if iPhone ≥ ~45 tok/s and memory holds, this is the 1.5×
  win and becomes the new production model. If iPhone ~30 tok/s, it's
  parity — decide whether the (likely better) quality of a 14.3B-total
  MoE justifies a same-speed model swap.

## Artifacts

* `conversion/phase_b_redux_build.py` — builds single-expert /
  layer-gather / dense-backbone / multifunction mlpackages, fp16 + INT4,
  for CoreML dispatch characterisation.
* `Sources/moe-dispatch-probe/main.swift` — Swift harness measuring
  real CoreML dispatch latency (the proper test that confirmed CoreML
  can't do MoE; reusable for any future small-kernel routing scheme).
* `/tmp/qwen_moe_3bit/` — the 3-bit MLX model (5.9 GB, not committed).
* `/tmp/qwen_moe_fp16/` — dequantized fp16 (27 GB, regenerable, not
  committed; delete on disk pressure).
