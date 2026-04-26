# Session 2026-04-26 — Round 8 INT8 LUT entries probe (chunk_1, MacBook Air M3)

**Status:** **DEAD on this stack.** Mac probe disproves drop-in viability.
**Branch:** `main` (research-only converter flag, no Sources/ change).
**Round 8 candidate:** #1 INT8 LUT entries (`docs/ROUND8_FINDINGS.md` §1).

## TL;DR

Apple's `linear_quantize_weights(joint_compression=True)` after
`palettize_weights` does store the W4 LUT entries as INT8 instead of
FP16 — the API works exactly as documented. But on the stateful Linear
chunk_1 graph for Gemma 4 E2B:

| metric | baseline (W4 LUT FP16) | INT8 LUT entries | Δ |
|---|---:|---:|---:|
| Bundle size | 148.6 MB | 148.6 MB | **0.0 %** |
| Mac CPU+NE latency (chunk_1, 20-iter median) | 4.56 ms | 4.72 ms | **+3.4 % (slower)** |
| ANE placement | 955/1028 (92.9 %) | 955/1101 (86.7 %) | **-6.2 pt** |
| Cos sim vs baseline (16 synth N(0,0.5) samples) | — | mean 0.829, min 0.607 | **FAIL** (gate ≥0.95) |

The exact number of ANE-placed ops is **identical** (955 in both). The
INT8 LUT variant adds **73 extra ops** (1101-1028) handling the
INT8→FP16 dequant of the LUT entries themselves, and **all 73 fall off
ANE** (CPU/GPU "unknown" device). Bundle size is unchanged because the
W4 indices dominate weight bytes — the LUT entries (16 × FP16 per group)
are a tiny fraction of total bytes.

**This matches the "top failure mode" predicted in
`docs/ROUND8_FINDINGS.md` §1** verbatim: *"ANE dequantizes LUT→FP16 on
read regardless, so the steady-state per-step bandwidth saving is zero
(only weight loading benefits). If that happens, this collapses to a
bundle-size win only."* Bundle-size win didn't materialize either — so
it collapses to **pure regression**.

## Recommendation

**DEAD on Gemma 4 E2B / cml9 / iOS 26 ANE path.** Do not pursue further
for this model. Move ROUND8 candidate #1 from "alive" to "dead" in the
roadmap.

## What this probe answered

1. ✅ cml9 accepts `palettize → joint_compression` on the stateful
   Linear chunk_1 graph (build succeeds in ~80 s on M3 16 GB).
2. ❌ MIL graph does NOT stay 100 % on ANE — drops to 86.7 % from
   baseline 92.9 %, with 73 new INT8-dequant ops landing on CPU/GPU.
3. ❌ Mac CPU+NE latency regresses 3.4 %, not improves.
4. ❌ Cos sim against W4-LUT-FP16 baseline FAILs (mean 0.83 vs gate 0.95).
5. ❌ Disk size is identical to baseline (W4 indices dominate, LUT is
   small — INT8 LUT entries don't move the needle).

## What this probe does NOT answer

- iPhone-specific behaviour. Skipped because Mac results are decisive
  enough (ANE drop + latency regression on Mac strongly predict iPhone
  parity or worse — A19 Pro ANE compiler is the same generation).

## Setup

- Hardware: MacBook Air M3, 16 GB RAM, macOS 26.0.1
- Python: `.venv-ss/bin/python` (3.11, coremltools 9.0, torch 2.11)
- Model: `output/gemma4-e2b/hf_model` (`google/gemma-4-E2B-it`,
  downloaded fresh)
- Build: `--only-chunk 1 --linear-projections` (Plan 3 stateful path)
- Build times: HF load ~190 s (slow on 16 GB), MIL convert 22 s,
  palettize 50 s, joint INT8-LUT 9 s. Total ~5 min per variant on M3.

## Build commands (reproducible)

```bash
# Baseline (W4 LUT, FP16 entries)
.venv-ss/bin/python conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/r8_int8lut/baseline \
    --hf-dir output/gemma4-e2b/hf_model \
    --only-chunk 1 --linear-projections

# INT8 LUT entries (Round 8 candidate #1)
.venv-ss/bin/python conversion/build_gemma4_e2b_stateful_chunks.py \
    --output /tmp/r8_int8lut/int8lut \
    --hf-dir output/gemma4-e2b/hf_model \
    --only-chunk 1 --linear-projections \
    --joint-int8-lut

# Latency + ANE audit
.venv-ss/bin/python conversion/probe_int8_lut_chunk1.py \
    --baseline /tmp/r8_int8lut/baseline/chunk_1.mlpackage \
    --int8lut  /tmp/r8_int8lut/int8lut/chunk_1.mlpackage

# Cos sim (reuse Stage 1 probe, --w4 = baseline, --w4a8 = int8lut)
.venv-ss/bin/python conversion/probe_w4a8_quality.py \
    --w4   /tmp/r8_int8lut/baseline/chunk_1.mlpackage \
    --w4a8 /tmp/r8_int8lut/int8lut/chunk_1.mlpackage \
    --samples 16
```

## Why this happened (mechanism)

The Apple docs claim is true but conditional: INT8 LUT entries shrink
on disk and reduce DRAM bandwidth **per LUT lookup**. On A-series ANE,
LUT lookups are decoded inline by the ANE compiler into FP16 weight
tensors at compile time, not at run time. The `constexpr_lut_to_dense`
op family with INT8-typed LUT entries forces an extra dequant op (INT8
→ FP16 LUT) that the ANE compiler doesn't yet inline; it falls back to
CPU/GPU. Net: more ops, same bytes (since indices dominate weight
bytes), worse latency, worse quality.

Per `docs/COREMLTOOLS_AND_IOS18.md` §3.1: "Output MIL op:
`constexpr_lut_to_dense`." Per Round 8 verification, joint compression
adds upstream INT8 dequant ops that don't fuse into the ANE
`constexpr_lut_to_dense`. **The optimization is real but currently
ANE-incompatible** for this MIL pattern in cml9.

## Code changes landed (kept for reproducibility — flag is opt-in)

- `conversion/build_gemma4_e2b_stateful_chunks.py` — added
  `--joint-int8-lut` CLI flag + plumbing through `_trace_and_convert_stateful`
  via `joint_int8_lut` keyword. Calls
  `linear_quantize_weights(joint_compression=True)` after palettize when
  set. **Default OFF.**
- `conversion/probe_int8_lut_chunk1.py` — new probe script (latency +
  ANE audit + size). Reusable for future joint-compression variants.
- `docs/SESSION_2026_04_26_ROUND8_INT8_LUT_PROBE.md` — this file.

The flag is harmless when off; kept in tree so a future cml/ANE compiler
update can be re-tested cheaply (~5 min build + probe).

## Implications for ROUND8 sibling candidates

- **#2 PALU** (low-rank K/V projection) — unaffected. PALU's
  reconstruction matmul is a standard MIL op, not a LUT-dequant
  pattern. Still the headline ROUND8 lead.
- **#3 Joint sparse + palettized** — partially related risk. Joint
  sparsity uses `constexpr_lut_to_sparse + constexpr_sparse_to_dense`,
  a different op family from `constexpr_lut_to_dense + INT8-dequant`.
  May not hit the same ANE-fallback mode but ANE sparse-tensor support
  is "Unknown on ANE" per `QUANTIZATION_SURVEY.md` line 39 — the binary
  outcome warning in ROUND8_FINDINGS still holds.
