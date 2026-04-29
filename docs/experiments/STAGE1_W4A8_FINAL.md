# Stage 1 W4A8 — final report (CLOSED)

**Branch:** `stage1-w4a8`
**Commits:** `ef4cb0e` (v1) → `eef2e39` (v2) → `f88360f` (v3 AWQ) →
`716706c` (FP16 diagnostic) → `fae395b` (W8A8 + SmoothQuant)
**Verdict:** **CLOSED — A8 quantization on Gemma 4 attention is
structurally beyond cml9 PTQ. W4A16 stateful is the operating point.**

This is the canonical single-page reference. Three chronological HOLD
docs (`SESSION_2026_04_26_W4A8_HOLD{_v2,_v3}.md`) capture the iteration
trail; this doc consolidates the conclusions.

---

## TL;DR

Stage 1's perf goal was halving iPhone ANE memory bandwidth via
INT8 activations on top of W4 weights (target +30-50 % decode
tok/s). After five build iterations and three calibration / migration
techniques, **A8 cos sim plateaus at 0.5-0.6 vs FP16 baseline** —
0.4 cos below the 0.95 gate. The technique trail (AWQ, SmoothQuant,
real-prompt calibration, alpha sweep, W4 vs W8) all confirms a single
structural fact:

> **cml9 PTQ activation quantization on the Gemma 4 attention path
> destroys ~12× more per-op precision than W4 weight quantization
> alone, and no PTQ technique we tested closes that gap.**

The W4A16 stateful path already shipped in commit `7c9cfea`
(`gemma4e2bStatefulLinear` ModelInfo entry) is the production
operating point. Stage 1's perf rationale is unattainable on the
current cml9 + Gemma 4 stack.

---

## The cos sim ladder (32 real-prompt samples, vs FP16 baseline)

| variant | weights | activations | size | **cos vs FP16** |
|---|---|---|---|---|
| FP16 baseline (`--nbits 0`) | FP16 | FP16 | 592 MB | 1.000 |
| **W4A16 stateful** (current production) | W4 LUT | FP16 | 148 MB | **0.949** |
| W4 + AWQ α=0.5 (no A8) — diagnostic | W4 LUT smoothed | FP16 | 148 MB | ≈ 0.89 |
| W4A8 synth calibration (v1) | W4 LUT | INT8 sym | 148 MB | ≈ 0.10 |
| W4A8 real calibration (v2) | W4 LUT | INT8 sym | 148 MB | ≈ 0.475 |
| W4A8 + AWQ α=0.5 (v3) | W4 LUT smoothed | INT8 sym | 148 MB | ≈ 0.490 |
| W4A8 + AWQ α=0.7 | W4 LUT smoothed | INT8 sym | 148 MB | ≈ 0.506 |
| **W8A8 + SmoothQuant α=0.5** | W8 LUT smoothed | INT8 sym | **299 MB** | **0.574** |

Per-op (1-ε) at gate cos ≥ 0.95 across 56 quant rounds: ε ≤ 0.1 %/op.

W4 LUT alone hits this (W4A16 cos 0.949 → ε ≈ 0.094 %/op). A8 spikes
ε ~12× to 1.2 %/op regardless of weight precision or migration.

---

## The five things we learned

### 1. AWQ is a weight-only technique by design
Lin et al. 2023 frame AWQ for **W4A16**. The migration formula
`s_i = act_max[i]^α / w_max[i]^(1-α)` shifts outliers from
activations *into* weights — useful only if weight quantization can
absorb the migrated magnitude. With cml9's W4 LUT (16 cluster centers
per group), AWQ-induced weight magnitudes spread the kmeans range and
hurt W4 accuracy by ~0.06 cos (W4-AWQ vs W4 baseline = 0.94 instead
of ~1.0 expected). We ran AWQ as if it were a quantization-direction-
agnostic tool. It isn't.

### 2. SmoothQuant is the proper W8A8 paper, and it still falls short
SmoothQuant (Xiao et al. 2022) shares AWQ's smoothing formula but is
designed for **W8A8** with per-tensor activation scales. cml9's
`palettize_weights(nbits=8)` (256 cluster W8 LUT) approximates the
W8 INT8 paper assumption; we built chunk_1 with W8 + INT8 sym + α=0.5
migration → cos 0.574 vs FP16. Better than W4A8 (+0.07) but still
0.4 cos below the gate. **The proper-regime paper-of-record technique
gives ~+13 % cos lift; the regime fundamentals leave Gemma 4
attention path beyond reach.**

### 3. The skipped diagnostic was the cheapest and most informative
We spent 4 builds (v1, v2, v3 AWQ x2) iterating on calibration data
and α before measuring W4A16 vs FP16 directly. That single 2-minute
diagnostic showed W4 LUT was clean (cos 0.949) and A8 was the wall
(adds 0.4 cos drop independent of every other lever). **Lesson: when
debugging a quantization stack, build the precision ladder
unprejudiced first.** FP16 → W-only → +A8 → +migration. Don't start
with the technique under suspicion.

### 4. cml9 PTQ activation quant has stateful + int32 + degenerate-
range gaps that need three monkey-patches to fire on Gemma 4 chunks
For anyone attempting A8 (or W8A8, or QAT-PTQ-hybrid) on a stateful
multi-input MIL graph, expect to:
- Patch `predict_intermediate_outputs` to allocate `make_state()`
  per calibration sample (cml9 calibrator hard-codes
  `model.predict(inputs)` with no state).
- Patch `insert_prefix_quantize_dequantize_pair.transform_op` to skip
  ops whose `x` input is int32 (e.g. `current_pos + 1`) or whose
  calibration stats are missing (bool intermediates filtered during
  output cloning).
- Patch `insert_suffix_quantize_dequantize_pair._try_apply_transform`
  similarly for missing stats and degenerate ranges (rmin == rmax
  produces scale=0, tripping iOS17 quantize op validator).

All three are in `conversion/build_gemma4_e2b_stateful_chunks.py`,
applied only when `--activation-quant` is set.

### 5. The Stage 1 perf goal was structurally unattainable
"+30-50 % decode tok/s via memory bandwidth halving" requires A8 to
fire reliably on iPhone ANE. cml9 PTQ + Gemma 4 attention can't
produce a ship-quality A8 model. The two remaining levers are:
- **QAT** (joint-training of weight + activation quantizers) —
  research-grade, A100-required, ~3-5 days plus iPhone validation.
  *Not on v1.7.0 critical path.*
- **Selective A8** (skip residual / RMSNorm-out / KV-state-feeding
  ops via `op_name_configs`) — partial perf gain at best, since the
  unsafe ops are the dominant intermediate-tensor sites. Estimated
  worst-case cos 0.85, best-case 0.93. Worth ~3 days if QAT is
  unavailable and someone wants to grind partial wins.

---

## Why we report this (operations note)

The cml9 PR #2577 release notes claim "INT8 activation quantization
for `linear` op" — true for the API, misleading for the use-case.
**The PR enables the technique mechanically; whether it produces
ship-quality output for a given model is application-dependent.**
For Gemma 4 (RMSNorm-heavy, residual-stream-deep, GQA-attention
architecture), cml9 PTQ A8 is structurally inadequate.

This is documented so the next person who reads the PR #2577 release
notes and considers W4A8 for Gemma 4 has a clean prior:
"PTQ-A8 has been tried and quantitatively measured below ship
quality. Skip to QAT or selective A8."

---

## What's left in the repo (re-usable for future work)

All of these are opt-in flags / additional modules; **default build
behaviour is unchanged from main** (W4 weight-only palettize +
Linear projections + stateful chunks).

### Converter (`conversion/build_gemma4_e2b_stateful_chunks.py`)

| flag | purpose | default |
|---|---|---|
| `--activation-quant` | adopts cml9 `linear_quantize_activations` after palettize | OFF |
| `--calib-data PATH` | uses real-prompt `.npz` instead of synthetic | OFF |
| `--calib-samples N` | synthetic calibration sample count | 4 |
| `--awq` | applies AWQ / SmoothQuant smoothing pre-conversion | OFF |
| `--awq-alpha α` | smoothing exponent | 0.5 |
| `--activation-mode {linear_symmetric,linear}` | act-quant mode | linear_symmetric |
| `--activation-scope {linear,all}` | which op types get quantized | linear |
| `--nbits {0,4,8}` | weight palettize nbits (0=fp16) | 4 |

Three cml9 monkey-patches (idempotent, applied only when
`--activation-quant` is set):
- `_patch_calibrator_for_stateful` — stateful predict via `make_state()`
- `_patch_quant_dequant_skip_missing` — int32 / no-stats skip in prefix
- `_patch_suffix_skip_missing` — no-stats / degenerate skip in suffix

### Other modules

- `conversion/awq_smoothing.py` (96 lines) — SmoothQuant /
  AWQ-formula module: hooks q_proj / gate_proj inputs, computes
  per-channel `s`, mutates `input_layernorm.weight ÷= s` /
  `pre_feedforward_layernorm.weight ÷= s` and the corresponding
  linear weights `×= s`. In-place layer mutation, output unchanged
  in fp32 ground truth.
- `conversion/gen_calib_data_real.py` — generates real-prompt `.npz`
  calibration data from PyTorch Gemma 4 + tokenizer (32 prompts ×
  first-N tokens, configurable). Standalone, deterministic.
- `conversion/probe_w4a8_quality.py` — cosine sim / max abs diff
  between two chunk_1 mlpackages on either synthetic or real-data
  inputs. PASS / WARN / FAIL gate verdict.
- `conversion/probe_chunk1_w4a8.py` — Mac CPU+NE 20-iter median
  latency probe.

### Documentation

- `docs/SESSION_2026_04_26_W4A8_HOLD.md` (v1, synthetic calibration)
- `docs/SESSION_2026_04_26_W4A8_HOLD_v2.md` (real-prompt calibration)
- `docs/SESSION_2026_04_26_W4A8_HOLD_v3.md` (AWQ + FP16 diagnostic +
  W8A8 SmoothQuant)
- `docs/STAGE1_W4A8_FINAL.md` ← this file (canonical summary)

All of the above can be picked up directly when QAT or selective-A8
work resumes — none of it is W4A8-specific. The calibration data,
probe scripts, and AWQ smoothing module are reusable for any
quantization variant.

---

## Reproduction (for the QAT / selective-A8 successor)

```bash
# 1. Generate real-prompt calibration data (~1 min)
python3.12 conversion/gen_calib_data_real.py \
    --hf-dir output/gemma4-e2b/hf_model \
    --output conversion/calibration_data/gemma4_chunk1_real.npz

# 2. FP16 baseline (~2 min, no quant)
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/g4_w4a8/fp16 \
    --hf-dir output/gemma4-e2b/hf_model \
    --only-chunk 1 --linear-projections --nbits 0

# 3. W4A16 reference (~3 min, current production)
python3.12 conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/g4_w4a8/w4_linear \
    --hf-dir output/gemma4-e2b/hf_model \
    --only-chunk 1 --linear-projections

# 4. Quality probe (vs FP16 ground truth)
python3.12 conversion/probe_w4a8_quality.py \
    --w4   /tmp/g4_w4a8/fp16/chunk_1.mlpackage \
    --w4a8 /tmp/g4_w4a8/w4_linear/chunk_1.mlpackage \
    --real-data conversion/calibration_data/gemma4_chunk1_real.npz \
    --samples 32

# Expected output: hidden_states_out cos sim mean ≈ 0.949 vs FP16.
# That's the high water mark on this stack until QAT lands.
```

---

## Stage status row (`docs/ROADMAP_2026_04_26.md` §6)

> | Stage 1 W4A8 | **CLOSED — A8 is the wall, even at W8A8 + SmoothQuant** | `stage1-w4a8` | 2026-04-26 |

Branch left open and pushed; can be merged to main as docs-only
(no Sources/ changes), or rebased + dropped depending on archive
preference. The opt-in converter flags don't affect any production
build path, so merge is safe.
