# Stage 1 — W4A8 via cml9 `linear_quantize_activations` — HOLD

> **For the canonical Stage 1 summary, read
> `docs/STAGE1_W4A8_FINAL.md`. This doc is the v1 iteration record
> (synthetic calibration, cos 0.108) — kept for the trail but the
> conclusion is in the FINAL doc.**

**Date:** 2026-04-26
**Branch:** `stage1-w4a8`
**Verdict:** **HOLD** (Mac chunk_1 quality regression)

Per `docs/ROADMAP_2026_04_26.md` §2 step 4: *"≥ +20 % decode tok/s and PPL
parity → ship; otherwise HOLD."* Mac chunk_1 quality probe shows
`hidden_states_out` cosine similarity ≈ 0.1 vs W4 baseline (target
> 0.995). Not shipped to iPhone; no iPhone trip burnt.

---

## 0. TL;DR

| metric | W4 baseline (Linear) | W4A8 (Linear, all-ops) | W4A8 (Linear, linear-op only) |
|---|---|---|---|
| size | 148.6 MB | 148.7 MB | 148.7 MB |
| ANE placement | 92.9 % (955/1028) | 94.8 % (1339/1412) | 94.0 % (1151/1224) |
| Mac CPU+NE median latency (chunk_1) | 4.04 ms | 4.09 ms (+1.3 %) | (not re-measured) |
| `hidden_states_out` cos sim | — | **0.108** | **0.153** |
| `per_layer_combined_out` cos sim | — | 0.999 | 0.999 |

Mac latency is parity (no win from INT8 activations). Quality is
catastrophically broken on the chunk_1 main output that feeds chunks
2-4. The auxiliary `per_layer_combined_out` (computed pre-attention) is
fine — the regression is concentrated in the attention path.

---

## 1. What we built

### 1.1 Converter (`build_gemma4_e2b_stateful_chunks.py`)

- `--activation-quant` flag: opt-in. Default behaviour unchanged.
- `--calib-samples N` (default 4): synthetic calibration draw count.
- After `palettize_weights`, optionally calls
  `coremltools.optimize.coreml.linear_quantize_activations(mlmodel,
   OptimizationConfig(op_type_configs={"linear": cfg}), sample_data)`.
- Synthetic `sample_data` per input spec:
  - mask inputs: zero
  - `cos_*` / `sin_*`: U(-1, 1)
  - position counters (int32): monotone
  - everything else: N(0, 0.5) fp16

### 1.2 cml9 monkey-patches (in converter, applied at convert time)

cml9's `linear_quantize_activations` was authored against stateless
Conv2d-1×1 image models. Three issues blocked stateful Gemma 4:

1. **`_patch_calibrator_for_stateful`** — calibrator's
   `predict_intermediate_outputs` calls `model.predict(inputs)` with no
   state, raising "The input feature for kv_cache_full must be an
   MLState, but it was not." Patch detects stateful temp model via
   `_is_stateful()` and allocates fresh state via `make_state()` per
   call.

2. **`_patch_quant_dequant_skip_missing`** — `insert_prefix_…` walks
   every `linear`/`conv`/`add`/pool op; for `add` ops with int32 inputs
   (e.g. `current_pos + 1`) it tries to insert a fp32-scale quantize op,
   tripping "scale must have the same data type as input" validator.
   Patch skips ops where `op.inputs["x"].dtype` isn't fp16/fp32. Also
   skips when input has no calibration stats (some intermediate outputs
   of bool type are silently skipped during calibration but not during
   pattern matching).

3. **`_patch_suffix_skip_missing`** — `insert_suffix_…` looks up rmin/
   rmax for the OUTPUT var of every dequantize→linear/add pattern. Same
   missing-stats failure mode. Also skips when `rmin == rmax`
   (degenerate range produces scale = 0, tripping iOS17 quantize op
   validation).

All three patches are idempotent and apply only when
`--activation-quant` is set, so they don't affect the default code path.

### 1.3 Probe scripts

- `conversion/probe_chunk1_w4a8.py` — Mac CPU+NE median-of-20 latency,
  W4 vs W4A8.
- `conversion/probe_w4a8_quality.py` — N-sample synthetic forward,
  cosine similarity / max abs diff / rel L2 on `hidden_states_out`.

---

## 2. What worked

- **Build pipeline succeeds.** chunk_1 W4A8 mlpackage is produced after
  five iterations refining the patch set. Calibration runs in ~25 s,
  prefix/suffix passes in ~5 s.
- **Stateful model compatibility.** Once the calibrator is patched, the
  `linear_quantize_activations` call works on the dual-state Gemma 4
  chunk_1 (kv_cache_sliding + kv_cache_full).
- **ANE placement preserved.** Quantize/dequantize ops insert ~30 %
  more total ops, but ANE placement stays > 94 %. So compiler-side this
  is fine.
- **Mac latency parity.** No regression. No improvement either.

## 3. What didn't work

### 3.1 Output quality

Quality probe (`probe_w4a8_quality.py`) on 8 synthetic forward passes
with seed=0 (the same seed as calibration, so in-distribution):

```
hidden_states_out:
  max_abs_diff:  mean=12.7  max=27.3  min=7.9
  cosine sim:    mean=0.108  min=0.054
verdict: WARN
```

For a linear-only scope (only `linear` op inputs quantized, not residual
adds):

```
hidden_states_out:
  max_abs_diff:  mean=12.7  max=30.2  min=7.9
  cosine sim:    mean=0.153  min=0.110
verdict: WARN
```

Limiting scope to linear ops barely improved cosine. The error is not
coming from residual `add` quantization alone.

### 3.2 Mac latency win

Mac chunk_1 median latency:
- A (W4 + Linear): 4.04 ms
- B (W4A8 + Linear): 4.09 ms (+1.3 %)

Mac ANE shader chose the same dispatch path; INT8 activations don't
exercise a different kernel here. Memory-bandwidth halving theory
doesn't manifest on Mac. Whether iPhone ANE picks a different shader
is the open question — but quality regression makes that test moot.

---

## 4. Root cause hypothesis

The most likely cause is **synthetic calibration data inadequacy for
Gemma 4's intermediate activations**:

1. **Distribution mismatch.** Our calibration samples draw `hidden_states`
   from N(0, 0.5). Real Gemma 4 hidden states post-RMSNorm have
   different distributions, and intermediate activations after each of
   the 8 transformer layers in chunk_1 evolve in scales/skewness our
   synthetic walks won't match. Calibration captures rmin/rmax of
   synthetic activations; real activations exceed those ranges and get
   clipped.

2. **Compounding error across layers.** Even small per-linear-op
   quantization error compounds through 8 sequential transformer
   layers. With INT8 round-trip ≈ 0.5 % per op, 8 layers × 4
   quantize-dequantize pairs = ~30 ops, error budget exceeded.

3. **Auxiliary output sanity check confirms (1).** `per_layer_combined_out`
   is computed *before* any transformer layer (it's just a linear over
   `per_layer_raw`). Its cosine is 0.999, indicating the activation
   quantization on a single linear op is fine in isolation. The
   regression appears only after multi-layer compounding.

This is a known issue with PTQ activation quantization for transformers
without proper calibration data. Mitigation paths (none cheap):

- **Real-prompt calibration.** Capture chunk_1 inputs from a real
  Gemma 4 forward pass (PyTorch fp16) on diverse prompts, use those as
  `sample_data`. ~1-2 days of work; would need at least 16-64 diverse
  prompts to cover the activation distribution adequately.
- **Per-layer mixed precision.** Quantize early layers' linears (where
  ranges are stable) and leave later layers fp16. Requires per-op
  `op_name_configs`; substantial engineering.
- **QAT fine-tune.** Train activation quantization scales jointly with
  weights. Out of scope (we don't have training infra for E2B).

## 5. Recommendation

**HOLD this stage**. cml9 PR #2577 is a real capability and our
converter framework is sound, but PTQ activation quantization on
chunk_1 with synthetic calibration is below the production quality
bar. The roadmap's expected +30-50 % decode lift wasn't directly
testable on Mac (chunk_1 latency parity), and quality fails before
iPhone validation.

Future work (not committing to a stage right now):
- Real-prompt calibration data harness (`probe_real_prompt_calibration.py`).
- W8A8 (8-bit weights too) instead of W4A8 — palettize stays fp16
  weights; activation quant alone may behave differently.
- AWQ-style calibration (activation-aware weight quantization).

## 6. What lands in this branch (commit)

- `conversion/build_gemma4_e2b_stateful_chunks.py` — `--activation-quant`
  flag + three cml9 monkey-patches + synthetic calibration helper.
  Default behaviour identical to pre-change.
- `conversion/probe_chunk1_w4a8.py` — Mac latency probe (W4 vs W4A8).
- `conversion/probe_w4a8_quality.py` — quality probe (cosine sim,
  max abs diff).
- This doc.
- Roadmap §6 update: Stage 1 status → HOLD with reason.
- `INFLIGHT.md` claim removed.

## 7. Reproduction

```bash
# Baseline (W4 + Linear)
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/g4_w4a8/w4_linear \
    --hf-dir output/gemma4-e2b/hf_model \
    --only-chunk 1 --linear-projections

# W4A8 (Linear, linear-op only — narrowest scope)
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/g4_w4a8/w4a8_linear \
    --hf-dir output/gemma4-e2b/hf_model \
    --only-chunk 1 --linear-projections --activation-quant --calib-samples 4

# Probes
python3.12 conversion/probe_chunk1_w4a8.py
python3.12 conversion/probe_w4a8_quality.py \
    --w4   /tmp/g4_w4a8/w4_linear/chunk_1.mlpackage \
    --w4a8 /tmp/g4_w4a8/w4a8_linear/chunk_1.mlpackage \
    --samples 8
```

Build wall-time ~3 min per chunk_1 mlpackage on idle Mac Studio.
