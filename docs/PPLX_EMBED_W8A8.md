# W8A8 (int8 weights + int8 ACTIVATIONS) viability — milestone B4

**Question.** Weight-only quant is a proven dead end for this encoder (int8 linear ~0.42
cosine, int4 palettize ~0.905) *and* buys only 4–8% latency because the forward is
activation/compute-bound (fp16 attention), not weight-bandwidth-bound. The only real
bandwidth lever is **activation** quantization. So: can W8A8 reach acceptable fidelity, or
does it hit the documented **~cos 0.57 wall** on this attention family?

**Model.** `perplexity-ai/pplx-embed-v1-0.6b` — 28-layer bidirectional Qwen3-0.6B encoder,
fp16, head_dim 128, GQA 16/8, SwiGLU, QK-norm, RoPE θ=1e6. Built via
`conversion/models/qwen3_encoder.py` (`PplxEmbedModel(cfg, output_mode="pooled_fp16")`),
measured against `conversion/pplx_embed_reference.py` (`Reference.embed` fp32 oracle,
int8-tanh output, cosine).

---

## Approach (reproducible: `conversion/experiment_w8a8.py`)

1. Build an fp16 `pooled_fp16` encoder at a **small bucket** (L=128 default) so the pooled
   vector is Python-readable on macOS26 (native int8 output is *not* Python-readable;
   pooled_fp16 is). `compute_units=ALL`, `minimum_deployment_target=macOS26`.
2. Calibrate activation ranges on a 14-text multilingual corpus (en/es/fr/de/ja/zh + short
   fragments), tokenized and right-padded to the bucket, via
   `cto.experimental.linear_quantize_activations`.
3. Quantize weights int8 (`linear_symmetric`, `weight_threshold=512`) on top → W8A8.
4. Predict on 12 held-out multilingual eval texts, apply `int8_tanh_quant` to the CoreML
   pooled output, and compute cosine vs `Reference.embed`. Report mean/min.

**Activation-quant mode is the crux** (parametrized `--mode asymmetric|symmetric`):

- The pad-mask add uses `Qwen3Encoder.NEG_INF = -1e4`; CoreML lowers this toward the fp16
  floor (−65504). A **symmetric** activation quantizer sets `scale ≈ 1e4/127 ≈ 79`, so real
  attention scores (±10) round to ≈0 → after 28 layers the output collapses. This is the
  mechanism behind the documented wall when symmetric quant is used.
- **Asymmetric** (`mode="linear"`) lets the range span `[−1e4, +score]`; when that span
  overflows fp16 the computed scale goes `inf`, coremltools' `isinf` guard fires, and the
  op is **skipped (left in fp16)** — exactly the desired behaviour for the mask add, while
  every other (small-range) activation quantizes correctly. This is the only mode with a
  chance of beating the wall.

Two coremltools-9 patches are required for the activation-quant pass to run at all (both in
the script, written upstream-native):
- `_cast` const-fold extracts a Python scalar before `int()/bool()` (numpy≥2 (1,)-array fix).
- `insert_prefix_quantize_dequantize_pair.transform_op` skips ops whose input `x` is
  non-float (int32 mask/embedding path) — MIL `quantize` requires float input.

Contingencies swept by `--all`: asymmetric vs symmetric; rescale K ∈ {0 (none), 8, 16}
(the fp16 residual rescale interacts with activation ranges — K shrinks the residual stream
K×, changing what the activation quantizer sees).

---

## Reference points (measured previously, this repo / knowledge base)

| config | cos vs fp32 ref (int8 output) | source |
|---|---|---|
| fp16 baseline | **0.999** (min 0.99912) | `weight-quant-is-a-dead-end.md` |
| int8 `linear_quantize_weights` (weight-only) | **0.42** mean (min 0.006) | same |
| int4 `palettize_weights` (g=32, weight-only) | **0.905** | same |
| documented **A8 / W8A8 wall** on this attention family | **~0.57** | `contingency-fixes.md`, plan |
| fidelity gate | **0.990** | `fidelity-gates.md` |

---

## Results (W8A8 — MEASURED, L=128, 12 multilingual eval texts)

Run: `uv run python conversion/experiment_w8a8.py --all --bucket 128`. Each variant: 14-text
calibration via `cto.experimental.linear_quantize_activations`, then int8 weight quant.
fp16 baseline is the same graph with no quant (sanity: it reproduces the ~0.999 fp16 number).

| variant | activation mode | rescale K | fp16 mean | **W8A8 mean cos** | W8A8 min | beats 0.57 wall? | ≥0.990 gate? |
|---|---|---|---|---|---|---|---|
| `w8a8-asymmetric-k8-L128`  | asymmetric | 8  | 0.9999 | **0.0157** | −0.0219 | ❌ NO | ❌ |
| `w8a8-symmetric-k8-L128`   | symmetric  | 8  | 0.9999 | **0.0020** | −0.0459 | ❌ NO | ❌ |
| `w8a8-asymmetric-k0-L128`  | asymmetric | 0  | 0.9996 | **0.0191** | −0.0078 | ❌ NO | ❌ |
| `w8a8-asymmetric-k16-L128` | asymmetric | 16 | 0.9995 | **0.0201** | −0.0105 | ❌ NO | ❌ |

**All four W8A8 variants collapse to cosine ≈ 0** (statistically orthogonal to the reference —
the embedding carries no signal). This is *worse* than the documented ~0.57 wall and far worse
than the weight-only int8 number (0.42). Activation int8 is even more destructive than weight
int8 on this encoder.

Key observations:
- The collapse is **independent of all the contingencies**: asymmetric vs symmetric makes no
  meaningful difference (both ≈0), and the fp16 residual rescale K (0 / 8 / 16) does not move
  it. So the failure is not the mask-sentinel scale blow-up alone, nor the rescale interaction —
  it is that per-tensor int8 activation quant across this 28-layer bidirectional graph destroys
  the representation outright.
- The `linear_quantize_activations` pass emits **fp16 overflow / NaN-scale / invalid-cast
  RuntimeWarnings** during `insert_prefix_quantize_dequantize_pair` (the −1e4 mask sentinel and
  other large-range activations overflow the int8 affine `zero_point` computation). Some ops are
  skipped (left fp16) as designed, but enough activations are quantized to wreck the signal.
- The fp16 baseline on the identical graph is 0.9995–0.9999, so the build/measure harness is
  correct — the loss is entirely from the activation quantization.

### ANE residency (MEASURED — W8A8 *does* stay on ANE)

`uv run python conversion/experiment_w8a8.py --audit /tmp/w8a8-experiment/w8a8-asymmetric-k8-L128.mlpackage`
(compiles via `coremltools.models.utils.compile_model` + `MLComputePlan`):

```
total ops: 3531
  ANE: 3322  (94.1%)
  unknown: 199  (5.6%)   constexpr_blockwise_shift_scale  (int8 weight-dequant consts, not compute dispatch)
  CPU: 10   (0.3%)       greater_equal/add/select/gather  (int8-tanh + pooling tail)
```

So residency is **not** the blocker — the W8A8 model compiles and runs **94% ANE-resident**
(the only non-ANE compute is the tiny pooling/tanh tail, identical to fp16). The int8 weights
lower to `constexpr_blockwise_shift_scale` consts. The model is perfectly deployable on ANE; it
just produces garbage.

---

## VERDICT: NOT VIABLE (post-training). Needs rotation pre-conditioning or QAT.

Naive post-training W8A8 is **dead on this encoder** — it does not approach the 0.990 gate, does
not beat the ~0.57 wall, and in fact collapses all the way to **cos ≈ 0** (worse than weight-only
int8's 0.42). Neither asymmetric activation quant nor any rescale-K setting rescues it. ANE
residency is fine (94%), so the failure is purely numerical fidelity, not a fallback problem.

**Why it collapses this hard** (vs the documented 0.57): this is a 28-layer *bidirectional*
encoder where every layer's input passes through QK-norm + RoPE and the residual stream is held
in fp32 specifically because activations are wide / outlier-heavy. Per-tensor uniform int8
activation quant sets one scale per tensor from the max, so the heavy-tailed bulk rounds toward
zero; compounded over 28 layers the signal is annihilated. The mask sentinel (−1e4) makes at
least one attention-input tensor's range pathological (the overflow warnings), and the
asymmetric "skip on inf" trick only saves *that* op — every other quantized activation still
crushes. Uniform int8 simply cannot represent this activation distribution.

**Path to viability** (matches `docs/QUANTIZATION_SURVEY.md`): the bandwidth win requires
activation quant, and activation quant requires **outlier pre-conditioning**:
- **SpinQuant / QuaRot** — fold a learned (SpinQuant) or Hadamard (QuaRot) rotation into the
  weights at *zero* runtime cost; it spreads activation outliers across channels so int8
  activations become representable. This is the highest-ROI next step.
- **SmoothQuant** — migrate per-channel activation scale into the weights pre-quant.
- Failing those, full **QAT**.

Until one of those is in place, the shipping configuration remains **fp16 + buckets** (the
weight-quant lesson's conclusion stands, now extended: *activation* quant is also a post-training
dead end without rotation/QAT).

> Honest negative result: the wall is not just confirmed, it is *deeper* than documented for this
> port — post-training W8A8 lands at ≈0, not 0.57. The 0.57 figure in the knowledge base likely
> reflects a partial / A8-only or differently-scoped experiment; full per-tensor W8A8 here is ≈0.

---

## Mitigation feasibility (researched 2026-06) + the latency reality

Recovery candidates and — critically — whether they map to the ANE's fixed op set:

| approach | recovery (LLM literature) | ANE-deployable? | effort |
|---|---|---|---|
| **SmoothQuant** — per-channel scale migrated activation→weight | W8A8 "negligible loss" on LLMs; "alone insufficient" for total collapse | yes (folds into weights, no runtime ops) | low |
| **Rotation (QuaRot/SpinQuant)** | 4-bit ~99% zero-shot; 8-bit "negligible" (extrapolated) | **partial** — R1/R2 fold offline, but the down-proj/value-path **online Hadamards have no adjacent linear to absorb** → extra runtime ops the ANE may reject/spill; **no public QuaRot/SpinQuant-on-ANE precedent** | high |
| **QAT + distillation** — distil from the fp32 teacher on **unlabeled** text (no labels/contrastive pipeline) | 8-bit "almost lossless"; total collapse likely needs **full** QAT, not LoRA | yes (weights only; deployed graph stays standard int8 matmul) | highest |

QAT-distillation is the only path with **both** strong recovery and clean ANE deployment.

### But the premise is wrong — int8 activations barely help here (MEASURED)

W8A8 exists to buy ANE bandwidth via int8 *activations*. Measured (L=128, cpuAndNE):

| precision | median latency |
|---|---|
| fp16 (pooled) | 14.0 ms |
| W8A8 (int8 act) | 12.7 ms (**~9% faster**) |

The ANE is fp16-native; int8-activation matmul is only marginally faster, and the attention
score matmuls (activation×activation) that dominate at large L are not int8-accelerated at all.
With weight quant's ~4–8%, the **whole quantization latency upside is ~10%** — not the 2× the
bandwidth intuition suggests. So even a *perfect* fidelity recovery (weeks of QAT, or a rotation
reimplementation that may not map to ANE) would buy ~10% latency. **Not worth it.** Ship fp16 +
buckets (0.999, 99.8% ANE, 101 ms at L=512); revisit only if a future ANE accelerates int8
compute, or if memory (not latency) becomes the binding constraint.

## Files

- `conversion/experiment_w8a8.py` — builds + measures W8A8 fidelity, parametrized
  (`--mode`, `--rescale-k`, `--bucket`, `--all`), saves `.mlpackage` artifacts for ANE audit.
- `docs/PPLX_EMBED_W8A8.md` — this note.
