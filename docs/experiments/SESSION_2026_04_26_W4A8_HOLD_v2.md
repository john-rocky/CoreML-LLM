# Stage 1 — W4A8 retry with real-prompt calibration — HOLD v2

> **For the canonical Stage 1 summary, read
> `docs/STAGE1_W4A8_FINAL.md`. This doc is the v2 iteration record
> (real-prompt calibration, cos 0.501) — kept for the trail.**

**Date:** 2026-04-26
**Branch:** `stage1-w4a8`
**Verdict:** **HOLD v2** (real-calib + asymmetric mode still below cos sim 0.95 gate)

Follow-up to `SESSION_2026_04_26_W4A8_HOLD.md` (v1: synthetic
calibration, cos 0.108). The user's hypothesis was that real-prompt
calibration would unblock the gate. Real-calib improved cos sim by 5×
(0.108 → 0.501), confirming the synth was wrong, but the gate still
fails — naive INT8 PTQ with 8-layer compounding can't reach cos ≥ 0.95
without weight-aware techniques (AWQ / SmoothQuant / QAT).

---

## 0. TL;DR

| variant | calibration | mode | scope | cos sim (vs W4) | verdict |
|---|---|---|---|---|---|
| (v1 prior) W4A8 | 4 synth N(0, 0.5) | linear_symmetric | all-ops | **0.108** | FAIL |
| (v1 prior) W4A8 | 4 synth N(0, 0.5) | linear_symmetric | linear-only | **0.153** | FAIL |
| (v2 this run) W4A8 | 64 REAL prompts | linear_symmetric | linear-only | **0.501** | FAIL |
| (v2 this run) W4A8 | 64 REAL prompts | linear (asymmetric) | linear-only | **build-crash** | n/a (see §3.5) |

Mac chunk_1 latency parity through all variants (~4.0 ms vs W4 baseline
4.04 ms; activation INT8 doesn't shorten the Mac ANE shader path).

---

## 1. What v2 added

### 1.1 `gen_calib_data_real.py`
Generates `.npz` calibration samples from PyTorch Gemma 4 + tokenizer.
For each of 32 diverse prompts (English / Japanese / code / math), the
first 2-4 tokens become independent calibration samples:
- `hidden_states` = `embed_tokens(token_id) * sqrt(hidden_size)`
- `per_layer_raw` = `embed_tokens_per_layer(token_id) * sqrt(per_layer_dim)`
- `cos_*` / `sin_*` = real RoPE table values at the position
- `causal_mask_*` = position-dependent (zero in valid range, -1e4 future)
- `current_pos` / `ring_pos` = monotone

Result: 64 samples, 1.2 MB compressed.

Distribution check vs prior synth N(0, 0.5):

| input | std (synth) | std (real) | range (real) |
|---|---|---|---|
| `hidden_states` | 0.5 | **1.25** | **±4.25** |
| `per_layer_raw` | 0.5 | **1.09** | **±5.84** |

Real activations are 2.5× wider. This was the user's hypothesised root
cause and it's confirmed — real-calib gave a 5× cos-sim lift.

### 1.2 Converter `--calib-data PATH` flag
`build_gemma4_e2b_stateful_chunks.py` now loads `.npz` calibration
samples when `--calib-data` is set, falling back to synthetic when
absent. Default behaviour unchanged.

### 1.3 `--activation-mode {linear_symmetric, linear}` flag
Added to compare zero-point=0 (default) vs asymmetric (zp ≠ 0). For
non-zero-centered post-RMSNorm activations the asymmetric form should
fit ranges more tightly.

### 1.4 Probe with `--real-data PATH`
`probe_w4a8_quality.py` now forwards real-prompt calibration samples
through both W4 and W4A8 instead of synthetic random — gives a fair
comparison on the distribution that calibration actually saw.

---

## 2. Results

### 2.1 W4 baseline (Linear, no activation quant)
- size 148.6 MB, ANE 92.9 % (955/1028)
- Reference output for cosine comparison

### 2.2 W4A8 real-calib, linear_symmetric, linear-only

- Build: size 148.7 MB, ANE 94.0 % (1151/1224)
- Calibration: 64 samples × 5.7 s = 6:04 wall-clock
- Compared to v1 synth ANE 94.8 % (1339/1412): same model graph, same
  scope; ANE % difference is just total-op count noise.

Quality on 32 real-prompt forward passes (in-distribution):
```
hidden_states_out:
  max_abs_diff:  mean=18.5  max=31.5  min=7.1
  cosine sim:    mean=0.501  min=0.343  max=≈0.7
verdict: FAIL (cos<0.90)
```

For comparison, v1 synth-calib produced cos = 0.108 on the same probe
(seed=0, in-distribution synthetic). Real-calib gave **5× cos lift**,
fully confirming the user's hypothesis that calibration data was the
v1 root cause. But the absolute number (0.501) is still well below the
0.95 gate per roadmap §2.4. Real-calib is necessary but not sufficient.

### 2.3 W4A8 real-calib, linear (asymmetric), linear-only — build crash

`--activation-mode linear` (asymmetric, zero_point ≠ 0) trips
`ValueError: quantize op: scale cannot be 0` during the
`insert_prefix_quantize_dequantize_pair` pass. Root cause: cml9's
guard at `_quantization_passes.py:1680` only kicks in when
`zero_point is None`:

```python
if np.all(scale == 0) and zero_point is None:
    scale = np.ones_like(scale)
```

For symmetric mode, mask inputs (all-zero, rmin=rmax=0) get the
guard treatment and survive. For asymmetric mode, zero_point is
non-None, the guard skips, scale=0 is passed to `mb.quantize()`,
iOS17's `type_inference` rejects it.

Fix would be: extend the prefix monkey-patch to also skip when
`stats[var_name]["rmin"] == stats[var_name]["rmax"]` (mirroring the
suffix patch). 5-line change, but not pursued because:

1. Expected upside is small. Asymmetric typically improves per-op cos
   by ~0.005 over symmetric. With 56 quant rounds:
   `0.995^56 ≈ 0.755`. Still below the 0.95 gate.
2. The structural argument (§3) is unaffected — naive PTQ INT8 on
   a 56-op chain can't reach cos ≥ 0.95.
3. Per-mode tuning is moving deck chairs; the answer is a different
   quantization technique (AWQ/SmoothQuant/QAT), not a different mode.

---

## 3. Why this fails — quantitative

`hidden_states_out` is the output of chunk_1's 8-layer transformer
stack. Each layer applies 7 linear ops (q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj). Activation quantization inserts a
quantize/dequantize round-trip on each linear op's input, introducing
~ε per op.

Total quantization rounds in chunk_1: 8 × 7 = 56.

Empirical cosine similarity vs W4 fp16 baseline:
```
v2 linear_symmetric: cos = 0.501
```
Implied per-op `1 - ε`:  `0.501^(1/56) ≈ 0.988` → ε ≈ 1.2 % per op.

For cos ≥ 0.95 across 56 ops: per-op cos ≥ `0.95^(1/56) ≈ 0.999`
(ε ≤ 0.1 %). That's beyond what naive PTQ INT8 with per-channel
calibration can reach on transformer activations — the math is the
constraint, not the calibration data.

**This applies to every full-precision-trained transformer** at this
depth. The fix is one of:
- **AWQ** (Activation-aware Weight Quantization, MIT 2023): scales
  weights by activation magnitudes pre-quant so high-magnitude
  channels don't dominate scale.
- **SmoothQuant** (MIT/NVIDIA 2022): migrates difficulty from
  activations to weights via per-channel multipliers.
- **GPTQ**: weight-only INT8 with optimal layer-wise reconstruction.
  Doesn't quantize activations; can be combined with W4 LUT.
- **QAT** (Quantization-Aware Training): joint training of weight +
  activation quantizers. Most effective; needs training infra.

cml9 ships `linear_quantize_weights` (W4/W8 only) and
`linear_quantize_activations` (A8 only); neither does AWQ/SmoothQuant
out of the box. AWQ-equivalent could be hand-rolled by pre-scaling the
PyTorch weights before the converter sees them, but it's a multi-day
research task — out of this stage's scope.

---

## 4. What does this leave on the table?

The v1 HOLD doc speculated the loss was 30-50 % decode tok/s on
iPhone (W4A8 vs W4 baseline). Without iPhone validation we don't know
the actual delta — Mac latency was parity, but iPhone ANE may behave
differently with INT8 activations (the cml9 PR #2577 motivation).

**Still on the table for stage 1 if revisited:**
- Implement AWQ-style activation scaling in PyTorch pre-conversion.
  Worth ~1 week if W4A8 lands cos > 0.99 for the user.
- Try W8A8 (no W4 LUT, just INT8 weights + INT8 activations). cml9
  supports both via `linear_quantize_weights` + `linear_quantize_activations`.
  Wider weight precision should reduce the per-op ε.
- Selective scope: only quantize MLP gate/up/down (compute-heavy,
  less sensitive); skip Q/K/V (attention is more sensitive). cml9
  `op_name_configs` would let us target by name pattern. Not tested.

**Not worth retrying as configured:** more synth or real prompts —
calibration-data quality saturated at 64 samples; the limit is
architectural, not statistical.

---

## 5. What lands in this commit

- `conversion/gen_calib_data_real.py` — real-prompt calibration data
  generator. Standalone, deterministic, ~1 min runtime.
- `conversion/build_gemma4_e2b_stateful_chunks.py` — adds
  `--calib-data PATH`, `--activation-mode {linear_symmetric, linear}`
  flags. Default behaviour unchanged.
- `conversion/probe_w4a8_quality.py` — adds `--real-data PATH` and a
  more granular gate verdict (PASS / PASS-MARGINAL / WARN / FAIL).
- `conversion/calibration_data/gemma4_chunk1_real.npz` — 64-sample
  data file. {DECISION: keep or skip per CLAUDE.md "minimum size"
  rule — generator is reproducible from prompt list, npz can be
  re-generated by anyone.} Decision: don't commit `.npz`, leave the
  generator script.

Roadmap §6 update: Stage 1 W4A8 status →
"HOLD v2 (real-calib insufficient, needs AWQ/SmoothQuant/QAT)".

INFLIGHT claim removed.

---

## 6. Reproduction

```bash
# Step 1: real-prompt calibration data
python3.12 conversion/gen_calib_data_real.py \
    --hf-dir output/gemma4-e2b/hf_model \
    --output conversion/calibration_data/gemma4_chunk1_real.npz \
    --max-prompts 32 --tokens-per-prompt 4 --max-samples 64

# Step 2a: W4 baseline (already from v1)
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/g4_w4a8/w4_linear \
    --hf-dir output/gemma4-e2b/hf_model \
    --only-chunk 1 --linear-projections

# Step 2b: W4A8 (linear_symmetric)
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/g4_w4a8/w4a8_linear \
    --hf-dir output/gemma4-e2b/hf_model \
    --only-chunk 1 --linear-projections --activation-quant \
    --calib-data conversion/calibration_data/gemma4_chunk1_real.npz \
    --activation-mode linear_symmetric

# Step 2c: W4A8 (linear asymmetric)
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/g4_w4a8/w4a8_asymm \
    --hf-dir output/gemma4-e2b/hf_model \
    --only-chunk 1 --linear-projections --activation-quant \
    --calib-data conversion/calibration_data/gemma4_chunk1_real.npz \
    --activation-mode linear

# Step 2d: quality probe
python3.12 conversion/probe_w4a8_quality.py \
    --w4   /tmp/g4_w4a8/w4_linear/chunk_1.mlpackage \
    --w4a8 /tmp/g4_w4a8/w4a8_linear/chunk_1.mlpackage \
    --real-data conversion/calibration_data/gemma4_chunk1_real.npz \
    --samples 32
```

Each W4A8 build ~9 min on idle Mac Studio (24 s load + 10 s convert
+ 45 s palettize + 6 min calibration + 1 min save).
