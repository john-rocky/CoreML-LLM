# Stage 1 — W4A8 retry v3 with AWQ smoothing — HOLD

**Date:** 2026-04-26
**Branch:** `stage1-w4a8`
**Verdict:** **HOLD v3** (AWQ assumption violated by cml9 W4 LUT precision)

Follow-up to `SESSION_2026_04_26_W4A8_HOLD_v2.md`. v2 confirmed
calibration data was the v1 root cause and gave a 5× cos lift
(0.108 → 0.501) but stayed below the cos ≥ 0.95 gate. v3 added AWQ
(Activation-aware Weight Quantization) per Lin et al. 2023 to migrate
activation outliers into weights. Result: AWQ at α=0.5 hurt W4 LUT
quality more than it helped INT8 activations. The math underlying
AWQ assumes weight quantization precision > activation precision —
a regime cml9's W4 kmeans LUT (16 levels per 32-channel group) does
not satisfy versus INT8 (256 levels).

---

## 0. TL;DR

| variant | calib data | quant | AWQ α | cos sim vs W4 baseline |
|---|---|---|---|---|
| W4 (Linear, no A8, no AWQ) | — | W4 LUT only | — | **1.000** (reference) |
| W4 + AWQ (no A8) | 64 real | W4 LUT only | 0.5 | **0.941** *(AWQ damages W4)* |
| W4A8 (real-calib, sym, no AWQ) | 64 real | W4 LUT + INT8 act | — | **0.501** |
| W4A8 + AWQ | 64 real | W4 LUT + INT8 act | 0.5 | **0.516** *(+0.015 vs no-AWQ)* |
| W4A8 + AWQ | 64 real | W4 LUT + INT8 act | 0.7 | **0.533** *(+0.032 vs no-AWQ)* |

The W4-only-with-AWQ row is the diagnostic: AWQ's weight scaling alone
(no activation quant) drops cos sim from 1.0 to 0.94. This is the W4
LUT struggling to represent the wider weight magnitudes after
per-channel scaling. The activation-quant gain (+0.015 at α=0.5,
+0.032 at α=0.7) is swamped by the W4 damage.

Note α=0.7 outperforms α=0.5 by +0.017 — pushing α higher migrates
more outliers into weights, helping A8 quant more than it hurts W4.
This suggests a sweet spot might exist closer to α=1.0, but
extrapolating linearly: α=1.0 ≈ 0.55-0.58. Still well below 0.95.

---

## 1. Why AWQ doesn't fit cml9 W4 LUT

AWQ's principle (Lin et al. 2023, MIT Han Lab):
- Per-input-channel scale `s_i = activation_max[i]^α / weight_max[i]^(1-α)`
- Apply `weight ← weight × s` and `RMSNorm.weight ← RMSNorm.weight / s`
- Output is mathematically unchanged (in fp32/fp16); but the redistributed
  magnitudes make activation quantization easier — outlier channels are
  shrunk in the activation domain.

**Critical assumption:** weight quantization preserves more precision
than activation quantization, so migrating outliers FROM activations
INTO weights nets a quality gain. AWQ paper validates this with W4
group-quant + A16 fp16 activations — weights have hundreds of
effective levels per group, activations have full fp16 dynamic range.

**cml9 architecture this stage uses:**
- `palettize_weights(nbits=4, granularity="per_grouped_channel", group_size=32)`:
  kmeans with **16 cluster centers per 32-channel group**
- `linear_quantize_activations`: INT8, **256 levels** per per-channel scale

So per-value precision: weights ≈ 4 bits effective × kmeans choice = ~16
representable values per group; activations = 256 levels uniformly.
**Activations are higher precision than weights.** AWQ's migration
direction is backwards for this regime.

The empirical confirmation (W4-AWQ vs W4-baseline cos = 0.94 without
any activation quant) shows the W4 LUT can't follow the AWQ-rescaled
weight distribution. Magnitudes that were ~1 are now ~32 in some
channels — the kmeans clusters are forced to spread out, and the
remaining mass loses precision.

---

## 2. What we tried in v3

### 2.1 `conversion/awq_smoothing.py` (new module)

Pure-PyTorch in-place smoothing. For each layer in chunk_1:
- Hooks q_proj and gate_proj inputs during a forward pass on 64 real-
  prompt calibration samples
- Per-input-channel max-abs reduction → activation_max
- Per-input-channel weight max from concat(q/k/v) and concat(gate/up)
- Compute s = activation_max^α / weight_max^(1-α), clamp ≥ 1e-5
- Mutate `input_layernorm.weight ÷= s`, `q/k/v_proj.weight ×= s`
- Mutate `pre_feedforward_layernorm.weight ÷= s`, `gate/up_proj.weight ×= s`
- o_proj and down_proj are not smoothed (no preceding norm to fold into).
  Coverage = 5/7 linears per layer × 8 layers = 71 %.

### 2.2 `--awq` and `--awq-alpha` flags (converter)

Minimal CLI surface, opt-in. Re-uses `--calib-data` for the activation
stat collection. Default behaviour unchanged.

### 2.3 Per-layer scaling stats (chunk_1, α=0.5, 64 real samples)

```
L0: qkv s∈[6.430, 31.990] med=11.255; gateup s∈[0.244, 10.952] med=3.039
L1: qkv s∈[0.127, 15.800] med=5.048;  gateup s∈[0.234, 13.004] med=3.407
L2: qkv s∈[0.362, 11.732] med=4.921;  gateup s∈[0.302, 8.862]  med=2.867
L3: qkv s∈[0.374, 12.176] med=3.425;  gateup s∈[0.117, 9.555]  med=2.184
L4: qkv s∈[0.186,  4.812] med=1.880;  gateup s∈[0.266, 11.531] med=3.525
L5: qkv s∈[0.207,  7.432] med=3.329;  gateup s∈[0.239, 8.645]  med=2.409
L6: qkv s∈[1.570, 14.128] med=4.582;  gateup s∈[0.377, 7.377]  med=1.914
L7: qkv s∈[1.059, 18.626] med=4.902;  gateup s∈[0.764, 7.303]  med=1.845
```

Scales s ∈ [0.13, 32.0]. The 32× outliers concentrate weight magnitudes
in those channels — exactly what's hurting W4 LUT.

---

## 3. Math summary across all three Stage 1 attempts

Each chunk_1 has 8 transformer layers × 7 linears = 56 quantize/dequantize
rounds. Per-op effective cosine similarity (1 - ε):

| variant | observed cos | per-op cos | per-op ε | gate (cos≥0.95 = ε≤0.001) |
|---|---|---|---|---|
| v1 synth + W4A8 sym | 0.108 | 0.961 | ~3.9% | ✗ |
| v2 real + W4A8 sym | 0.501 | 0.988 | ~1.2% | ✗ |
| v3 AWQ α=0.5 + W4A8 sym | 0.516 | 0.988 | ~1.2% | ✗ |
| target gate | ≥ 0.95 | ≥ 0.999 | ≤ 0.1% | ✓ |

To reach gate, per-op ε must drop ~12× from v3. AWQ's design space
(α ∈ [0, 1]) cannot deliver that — and the W4 LUT damage from AWQ
caps the achievable cos sim from this direction.

---

## 4. Real fix paths (none in cml9 today)

These are the techniques that *do* close the gap on AWQ-style
quantization for transformer attention paths:

| technique | per-op cos delta | cml9 support | effort |
|---|---|---|---|
| **GPTQ-W4** | ε drops 5-10× via Hessian-aware weight choice | None | implement W4 calibration algorithm, ~1 wk |
| **W8 + A8 + AWQ** | weight precision matches AWQ assumption | Yes, just `--nbits 8` | trivial **but doubles weight bandwidth — kills the entire W4A8 perf goal** |
| **SmoothQuant per-tensor** | helps when act quant is per-tensor | cml9 act quant is per-channel by default — no headroom for SmoothQuant to help | irrelevant |
| **QAT** | trains the quantizers; ε drops to 0.5-1% | None; needs HF-side training infra | ~3-5 days A100 + tooling |
| **Mixed precision (W4 most layers, W8 critical few)** | bypass per-layer | cml9's `op_name_configs` gives the hooks; needs analysis to pick layers | ~3 days, may not reach gate alone |

The W8 path is the only one that's mechanically trivial but it
**defeats the perf rationale** — Stage 1 exists to halve weight
bandwidth on the iPhone ANE; W8 doubles it.

GPTQ-W4 or QAT-W4 is the technically right answer but neither is in
cml9; both are multi-day implementation efforts plus iPhone validation.

---

## 5. Recommendation: close Stage 1, retain framework

**Close** Stage 1 W4A8 with three HOLD docs (v1 synth, v2 real-calib,
v3 AWQ) collectively documenting the path. The converter framework
(opt-in `--activation-quant`, `--calib-data`, `--awq`) stays in the
repo for future GPTQ/QAT integrations — those will reuse the same
calibration-data and probe scaffolding.

If a future stage wants to revisit:
1. **Implement GPTQ-W4 in PyTorch pre-conversion.** Reuse
   `gen_calib_data_real.py` and `awq_smoothing.py` skeletons. ~1 week.
2. **Or retrain with QAT.** Use the existing W4 + A8 graph as the
   target; train activation quantizers jointly. A100, ~3-5 days.

Neither is on the v1.7.0 critical path.

---

## 6. What lands in this commit

- `conversion/awq_smoothing.py` (new): AWQ smoothing module with hooks +
  in-place layer mutation. Tested on chunk_1 — correct math (W4-AWQ vs
  W4-baseline cos 0.94 matches expected weight-precision-loss prediction).
- `conversion/build_gemma4_e2b_stateful_chunks.py`: `--awq` and
  `--awq-alpha` CLI flags. Default behaviour unchanged.
- This doc.
- Roadmap §6 status row updated to reflect v3 close.

INFLIGHT claim removed.

---

## 7. Reproduction

```bash
# Existing real calibration data from v2 (regen if needed):
python3.12 conversion/gen_calib_data_real.py \
    --hf-dir output/gemma4-e2b/hf_model \
    --output conversion/calibration_data/gemma4_chunk1_real.npz

# v3 build with AWQ + W4 + A8:
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/g4_w4a8/w4a8_awq \
    --hf-dir output/gemma4-e2b/hf_model \
    --only-chunk 1 --linear-projections --activation-quant \
    --calib-data conversion/calibration_data/gemma4_chunk1_real.npz \
    --awq --awq-alpha 0.5

# Diagnostic: AWQ alone (no A8) — confirms AWQ scaling damages W4 LUT
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/g4_w4a8/w4_awq \
    --hf-dir output/gemma4-e2b/hf_model \
    --only-chunk 1 --linear-projections \
    --calib-data conversion/calibration_data/gemma4_chunk1_real.npz \
    --awq --awq-alpha 0.5

# Quality compare (real-prompt forward through both):
python3.12 conversion/probe_w4a8_quality.py \
    --w4   /tmp/g4_w4a8/w4_linear/chunk_1.mlpackage \
    --w4a8 /tmp/g4_w4a8/w4a8_awq/chunk_1.mlpackage \
    --real-data conversion/calibration_data/gemma4_chunk1_real.npz
```

Each AWQ + W4A8 build ≈ 10 min on idle Mac Studio.
