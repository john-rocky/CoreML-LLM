# Phase β-1 Stage 1 — Findings & Architecture Ceiling

Date: 2026-05-15 (overnight session continuation of 2026-05-14 handoff)
Branch: `feat/mtp-iphone-perf` working state
Predecessor: `docs/SESSION_2026_05_15_HANDOFF.md`

## TL;DR

Phase β-1 Stage 1 (single-layer Python prototype) revealed two problems that
collectively cap the approach far below the user-stated 1.5-2× target:

1. **Empirical** — Gemma 3n's trained 95 % activation sparsity is at the
   individual-neuron level and is spatially scattered across the 8192-wide
   intermediate. Contiguous-index slicing gets oracle cos sim only **0.766
   at K=2 of 16** (handoff promise was cos sim > 0.98). Co-firing
   permutation lifts oracle to **0.987 at K=6 of 16** but the linear
   router only achieves **0.929 at K=6** — gap not closeable without a
   non-trivial router (and even then the next problem applies).

2. **Architectural ceiling** — even with a perfect oracle router that saves
   95 % of sparse-FFN bandwidth, the whole-model speedup ceiling is **only
   1.18×**:

   | Component | Share of total decode bandwidth |
   |---|---|
   | Attention (30 layers) | 16.3 % |
   | FFN sparse (L0-9) | **16.3 %** |
   | FFN dense (L10-29) | 32.6 % |
   | Embed | 17.4 % |
   | LM head | 17.4 % |

   The handoff cited "21 % of total bandwidth saveable on sparse-FFN
   layers" — that number was computed against decoder-only bandwidth and
   ignored the 34.8 % of total bandwidth spent on embed + lm_head. The
   correct figure is **16.3 %**, giving a max sparse-FFN-only speedup of
   `1 / (1 - 0.163 × 0.95) = 1.18×`.

   To reach 1.5× would require saving roughly 33 % of TOTAL model
   bandwidth — i.e. either:
   * 100 % of FFN sparse bandwidth + 30 % of every other component
     (infeasible — attention can't be pruned, embed/lm_head are
     contiguous gather operations), or
   * combining FFN sparse routing with aggressive dense-layer static
     prune (handoff already proved this breaks code prompts at >30 %
     retention loss).

## Evidence files

* `conversion/stage1_dynamic_router_l0.py` — original prototype
  (cos sim oracle 0.75, router 0.68 at K=2 of 16)
* `conversion/stage1_analyze_slice_structure.py` — slice-entropy
  diagnostic (entropy 2.71 of max 2.77 → near-uniform firing)
* `conversion/stage1_verify_multi_layer.py` — multi-layer (L0/4/9/14)
  confirmation: same near-uniform entropy on every sparse layer
* `conversion/stage1_capture_large_corpus.py` — 2229-token mixed
  corpus capture (`/tmp/l0_activations_large.npz`)
* `conversion/stage1_cofiring_cluster.py` — first positive signal:
  co-firing permutation lifts K=2 oracle 0.766 → 0.903
* `conversion/stage1_cluster_eval.py` — full eval with held-out test
  set: oracle 0.987 at K=6, router 0.929 at K=6, router 0.995 at K=12

## Numerical summary

(2229-token corpus, train/val/test 70/15/15, N=16 slices, balanced
k-means clustering, linear router 2048→16.)

| K | Oracle cos | Router cos | BW save (kept frac) |
|---:|---:|---:|---:|
| 2 | 0.903 | 0.819 | 87.5 % |
| 3 | 0.947 | 0.861 | 81.2 % |
| 4 | 0.967 | 0.889 | 75.0 % |
| 6 | 0.987 | 0.929 | 62.5 % |
| 8 | 0.994 | 0.956 | 50.0 % |
| 12 | 1.000 | 0.995 | **25.0 %** |

Acceptance gate from handoff (router cos ≥ 0.98) only reached at K=12,
which saves only 25 % of sparse-FFN bandwidth. Whole-model speedup at
that point: `1 / (1 - 0.25 × 0.163) ≈ 1.043×` ≈ **30 → 31.3 tok/s**.

That's the realistic ceiling for this approach as designed.

## What would be needed to deliver 1.5×

To hit the user's 1.5× iPhone target, training-free, you'd need ALL of:

1. Sparse-FFN routing on L0-9 saving ~70 % of those layers' bandwidth
   (router cos sim ≥ 0.99 at K=4 of 16 — possible with MLP router but
   not validated)
2. Dense-FFN static prune on L10-29 retaining ~70 % of weights without
   killing code prompts (handoff says this failed even at 70 % retention)
3. Attention KV-cache compression (separate workstream)
4. Embed/lm-head bandwidth reduction (no known training-free win — INT4
   already deployed)

All four would compound to roughly 1.5×. Items 2-4 are all empirically
or structurally blocked in current code; item 1 alone delivers ~1.2×.

The remaining 0.3× has to come from somewhere. The only candidates per
the existing memory are training-required: drafter retraining (Path B
on Gemma 4) or DVI online drafter update.

## Recommended next steps (user decision)

Three options, ordered by my best guess of value:

### A. Stop β-1 and capture the negative finding

* Commit Stage 1 scripts + this doc + a `REJECTED_APPROACHES` update.
* No new branches, no CoreML work.
* Save ~5 days of engineering. iPhone state stays at current ~30-37 tok/s.
* Re-prioritise to the training paths (DVI online drafter, Mirror-SD
  drafter→GPU overlap from `RESEARCH_FINDINGS_2026_05_13`).

### B. Complete β-1 anyway to capture the +6-10 % win

* Build all 16 CoreML slices + Swift orchestrator (3-5 days more).
* Realistic outcome: 30 → 32-33 tok/s on Mac (+5-10 %), iPhone TBD.
* Value: incremental win + complete infrastructure for future routing
  experiments on different models (Apple FM, future block-sparse
  models).
* Cost: 1 week, no guarantee of net positive on iPhone after thermal
  and dispatch overhead.

### C. Pivot to a training-required path

* Acknowledge training is needed for >1.25× iPhone.
* Run Mirror-SD or DVI prototypes (both in `RESEARCH_FINDINGS_2026_05_13`).
* Mac-only LoRA-from-history (DVI) is the closest to "training-free"
  while still being a learned drafter.

My recommendation: **A then C**. The structural ceiling for Strategy A
is too low to justify the engineering. The negative finding is itself
valuable — it cleans up a class of "MoE-routing speedup" proposals
that will keep coming up otherwise.
