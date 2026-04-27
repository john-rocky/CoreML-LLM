# Session 2026-04-27 — ROUND8 #3 lever (Joint sparse + W4 LUT) probe — DEAD

**Branch:** `feat/joint-sparse-palettized`
**Base:** `54b8705` — main with 3-chunk mlstate-linear E2B default
**Goal:** binary go/no-go on `prune_weights(2:4)` + `palettize_weights(W4 LUT)`
on Gemma 4 E2B chunk_1 per `docs/ROUND8_FINDINGS.md` §3.

## Verdict — DEAD

All four probe axes regressed vs the W4 LUT baseline. No path to ship.

## Probe setup

- Converter: `conversion/build_gemma4_e2b_stateful_3chunks.py --only-chunk 1
  --linear-projections --nbits 4 --prune-n-m 2:4` (variant) and same minus
  `--prune-n-m` (baseline). ctx=2048.
- coremltools 9.0; Apple-blessed compose path: `prune_weights(
  OpMagnitudePrunerConfig(n_m_ratio=(2, 4), dim=1, weight_threshold=2048))`
  followed by `palettize_weights(OpPalettizerConfig(nbits=4,
  granularity="per_grouped_channel", group_size=32))`.
- Mac probe: `conversion/probe_sparse_chunk1.py`, 30 decode steps with
  `CPU_AND_NE`, identical synthetic inputs across both bundles.

## Numbers

| Axis | W4 LUT baseline | Sparse 2:4 + W4 | Delta |
|---|---|---|---|
| Bundle size | 155.8 MB | 349.6 MB | **+124% worse** |
| Mean step latency | 5.12 ms | 5.69 ms | **+11.2% worse** |
| Load time | 5.2 s | 13.6 s | +162% worse |
| ANE placement | 955 / 1028 = 92.9% | 955 / 1028 = 92.9% | unchanged |
| Cos sim `hidden_states_out` | reference | **0.449** | destroyed |
| Cos sim `per_layer_combined_out` | reference | 0.927 | drift |

## Why each axis lost

1. **Size**: theoretical sparse + W4 ≈ 2.5 bit/weight vs W4 = 4 bit/weight,
   should be ~62 %. Measured 225 %. coremltools 9.0 emits
   `constexpr_lut_to_sparse + constexpr_sparse_to_dense` but the
   `.mlpackage` storage is not actually bit-packed for the sparse case —
   it appears to materialise both the LUT-sparse and a dense fallback in
   the package. Apple-side limit, not fixable in our pipeline.
2. **Latency**: ANE op count identical (955), so the only added work is
   the sparse-to-dense reconstruction step. ANE does NOT skip zero
   multiplies at the dispatch level — confirmed by zero-improvement op
   count and +11 % wall clock.
3. **Quality**: `OpMagnitudePrunerConfig` is calibration-free (zero the
   smallest 2 of every 4 weights along dim 1). On Gemma 4 E2B's L0-7
   weights this collapses cos sim on `hidden_states_out` to 0.449 vs
   gate ≥0.95. Calibrated pruning (Fisher / activation-aware) could
   recover quality, but cannot fix the size or latency regressions.
4. **ANE placement**: 92.9 % is the **baseline** value for chunk_1 in
   the new 3-chunk default — not a sparse-induced regression. The 73
   non-ANE ops (7.1 %) appear in the W4 LUT path too.

## Failure mode confirmed

Matches the predicted top failure mode from `ROUND8_FINDINGS.md` §3:

> "ANE compiler treats sparse weights as dense at runtime (just smaller
> storage on disk). Probe outcome is binary — either you see ANE_wait
> drop on the bench, or you don't."

Worse than predicted on the size axis: the `.mlpackage` is **larger**,
not smaller. The "size-only win" fallback the doc considered does not
materialise on cml9 / iOS 26.

## What we did NOT try (gate-and-bail rationale)

- Calibrated pruning (Fisher / activation-aware) — would fix quality but
  not size or latency.
- Lower sparsity (1:4, 1:8) — same architectural failure mode at smaller
  amplitude.
- `granularity="per_channel"` palettize — orthogonal change, would not
  alter the sparse-to-dense materialisation behavior.

The gate was binary per ROUND8 protocol: ANE win or fall back. ANE %
unchanged + size + latency regression = fall back.

## Files changed (probe scaffolding only — keep or revert)

- `conversion/build_gemma4_e2b_stateful_chunks.py` — `prune_n_m` parameter
  added to `_trace_and_convert_stateful` and `convert_chunk1`. Inert
  unless the new flag is passed.
- `conversion/build_gemma4_e2b_stateful_3chunks.py` — `--prune-n-m` CLI
  flag. Inert unless passed.
- `conversion/probe_sparse_chunk1.py` — Mac latency + cos sim probe
  script. Reusable for any A/B chunk_1 comparison.

The `prune_n_m` plumbing is benign; recommend keeping it as a future
escape hatch if cml10 fixes the joint serialisation. Probe script is
reusable for sibling A/B comparisons.

## Closes

Closes ROUND8 alive lead #3 (`docs/ROUND8_FINDINGS.md` line 24-32 and
table B in §"Recommended ordering"). Status update for that doc:

| # | Candidate | Status |
|---|---|---|
| 1 | Joint compression: INT8 LUT entries | DEAD 2026-04-26 |
| 2 | PALU low-rank K/V projection | HOLD 2026-04-27 — memory-only lever, not tok/s win on ANE per agent digest |
| 3 | Joint sparse + palettized | **DEAD 2026-04-27** — this session |

ROUND8 alive table is now empty. Next moves are infrastructural (Metal
port / 11c) per `ROADMAP_2026_04_26.md`, not PTQ-class.
