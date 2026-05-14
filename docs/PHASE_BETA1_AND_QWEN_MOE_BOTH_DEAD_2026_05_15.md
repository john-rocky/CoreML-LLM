# Phase β-1 + Qwen MoE pivot — Both dead. Training-free 1.5× ceiling reaffirmed.

Date: 2026-05-15 (single-session synthesis)
Branch: `feat/mtp-iphone-perf`
Predecessors: `docs/SESSION_2026_05_15_HANDOFF.md`,
`docs/PHASE_BETA1_STAGE1_FINDINGS_2026_05_15.md`,
`docs/SESSION_2026_05_15_DYNAMIC_ROUTING_TO_MOE_PIVOT.md`

## Headline

In one session we empirically refuted **two** distinct paths to
training-free 1.5× iPhone speedup over Gemma 4 E2B. The existing memory
`project_lever_hunt_ceiling.md` ("training-free iPhone ceiling = 1.22-1.25×")
is fully reaffirmed. The roadblock is structural — ANE dispatch overhead
× small kernel + dynamic routing pattern × per-token decision.

| Path | Outcome | Why |
|---|---|---|
| Phase β-1 — Gemma 3n FFN slice routing | DEAD | Structural ceiling 1.18× (handoff bandwidth math used decoder-only share; corrected reveals the actual share of FFN-sparse-only is 16.3% of total bandwidth). |
| Qwen1.5-MoE-A2.7B per-expert mlpackages | DEAD | Per-expert ANE dispatch 0.35 ms → 144 dispatches/tok = 50 ms = 20 tok/s (regression vs 35 baseline). |
| Qwen1.5-MoE-A2.7B batched-gather mlpackage | DEAD | ANE gather on runtime indices falls off-engine at 45.69 ms median (50× worse than dense). GPU 0.88 ms/layer × 96 dispatches = 11 tok/s. |

## Empirical detail

### Path 1: Gemma 3n β-1 (sparse FFN slice routing)

* **L0 oracle K=2 of 16 cos sim = 0.766** vs handoff target ≥0.98.
* Co-firing-clustered permutation lifts oracle K=6 cos sim to 0.987,
  but linear router stays at 0.929 at K=6. Router only reaches 0.98 at
  K=12 of 16 (25% bandwidth save → 1.04× iPhone speedup).
* Architecture ceiling: even oracle-perfect on sparse-FFN-only = 1.18×
  (the handoff's 21 % figure was computed against decoder-only bandwidth
  and ignored 35 % of total bandwidth spent on embed+lm_head for
  Gemma 3n's 262 k vocab).

See `docs/PHASE_BETA1_STAGE1_FINDINGS_2026_05_15.md` for full numbers.

### Path 2: Qwen1.5-MoE-A2.7B port

* Bandwidth math is favourable: INT4 1.34 GB/tok vs Gemma 4's 1.97 →
  1.46× lower bandwidth → 52 tok/s ceiling on iPhone (if compute-bound
  by bandwidth alone).
* **But** ANE dispatch overhead dominates for the small per-expert
  kernel size:

  Phase B-1 (per-expert):
  | Compute | Median per-dispatch | × 144 dispatches | Tok/s |
  |---|---|---|---|
  | ANE | 0.35 ms | 50 ms | **20** (regression) |
  | GPU | 0.17 ms | 24 ms | 41 (only 1.17× over baseline) |
  | ALL | 0.34 ms | 49 ms | 20 (defers to ANE) |

  Phase B-2 (layer-gather, 60 experts as constants in one mlpackage,
  runtime gather to select 4):
  | Compute | Median per-dispatch | × 96 dispatches (24 layers × 4) | Tok/s |
  |---|---|---|---|
  | ANE | 45.69 ms | 4386 ms | **0.2** (gather kills ANE) |
  | GPU | 0.88 ms | 85 ms | **11** (regression) |
  | ALL | 45.78 ms | 4395 ms | 0.2 |

  ANE has no efficient path for runtime-indexed gather on a constant
  weight tensor (~1 GB at fp16 across 60 experts). It falls back to
  CPU/GPU emulation at heavy cost.

  Phase B-3 (weights-as-inputs — Swift gathers, passes 4 expert
  weights per layer call as input tensors, no in-graph constants):
  | Compute | Median per-call | × 24 layers (routed only) | Tok/s |
  |---|---|---|---|
  | ANE | 10.87 ms | 261 ms | **3.8** |
  | GPU | 9.80 ms  | 235 ms | **4.3** |
  | ALL | 10.97 ms | 263 ms | 3.8 |

  The 70 MB per-call weight transfer kills bandwidth. INT4 would help
  (8.75 MB per call), but ANE/CoreML doesn't accept INT4 input tensors
  in current coremltools. The bandwidth wall is structural.

## What's actually true about iPhone speedup

The combination of constraints — Apple Neural Engine's per-dispatch
overhead, lack of efficient runtime gather, iPhone DRAM bandwidth wall
(60-77 GB/s) — bounds training-free improvements to **1.22-1.25×** over
the current Gemma 4 E2B production state of ~30-37 tok/s. This is the
same ceiling reached by independent investigations going back weeks
(`project_lever_hunt_ceiling.md`, `project_drafter_structurally_dead.md`).

Per the existing roadmap memory, the only remaining levers above this
ceiling require **training**:

1. **Path B drafter retraining** (Gemma 4 native vocab drafter trained
   on real distribution): predicted +30-50 % decode speed via SpecDec.
   Cost: ~1 GPU-week training, drafter weight memory ~150 MB.

2. **DVI online drafter update** (arxiv 2510.05421 from
   `RESEARCH_FINDINGS_2026_05_13.md`): Mac sidecar continuously
   updates a LoRA on the drafter using user history. Less data
   than full retrain, possibly trainable on Mac GPU overnight.

3. **Mirror-SD drafter→GPU overlap** (Apple arxiv 2510.13161): +20-30 %
   by overlapping the drafter compute with target on GPU side.
   Architecture-level change to the inference loop.

## What we learned that's reusable

* **`stage1_*.py` scripts** are a portable test for "is this model's
  trained sparsity slice-routable?" — answer for Gemma 3n: no
  (neuron-level scatter, near-uniform slice entropy). Useful for any
  future block-sparse-trained model that emerges.
* **`phase_b_*.py` scripts** characterise ANE dispatch overhead on
  small kernels — measured at 0.28-0.35 ms on Mac M-series for a
  ~8.65 M-param SwiGLU. Iphone will be similar or worse. Useful for
  sanity-checking any future routing scheme.
* **The handoff's bandwidth math error** is documented; future plans
  should compute speedup against TOTAL model bandwidth including
  embed/lm_head, not decoder-only.

## File index (this session)

### New conversion scripts
* `conversion/stage1_dynamic_router_l0.py`
* `conversion/stage1_analyze_slice_structure.py`
* `conversion/stage1_verify_multi_layer.py`
* `conversion/stage1_capture_large_corpus.py`
* `conversion/stage1_cofiring_cluster.py`
* `conversion/stage1_cluster_eval.py`
* `conversion/phase_a_qwen_moe_analysis.py` (incomplete — download aborted)
* `conversion/phase_b_single_expert_mlpackage.py`
* `conversion/phase_b_layer_gather.py`

### Docs
* `docs/PHASE_BETA1_STAGE1_FINDINGS_2026_05_15.md`
* `docs/SESSION_2026_05_15_DYNAMIC_ROUTING_TO_MOE_PIVOT.md`
* `docs/PHASE_BETA1_AND_QWEN_MOE_BOTH_DEAD_2026_05_15.md` (this file)

### Memory
* `memory/project_qwen15_moe_pivot.md` (marked as DEAD, but kept for
  the bandwidth math reference)

### Cleanup needed (~30 GB freed if user agrees)
* `/tmp/gemma3n-e2b-pruned/`, `pruned50/`, `pruned70/` (static-prune
  variants, ~30 GB) — finding obsolete
* `/tmp/qwen15-moe-chat/` (~12 GB partial download) — aborted
* `/tmp/l0_activations*.npz` (~200 MB)
* `/tmp/stage1_*.json` (small)

## Decision for user

Both training-free paths in tonight's exploration converge at the
same answer: 1.22-1.25× is the iPhone ceiling. Options:

1. **Accept the ceiling.** Stay on Gemma 4 E2B production at 30-37 tok/s.
   Capture remaining incremental wins (fix iPhone Round F MTP crash,
   sweep Round E MTP_PER_PROMPT_KUSE, fallback threshold tuning) for
   maybe +5 % more.

2. **Commit to a training path.** DVI online drafter (`RESEARCH_FINDINGS
   _2026_05_13.md` T8) is the cheapest entry — Mac-side LoRA training
   from user history, no GPU rental. Path B drafter retraining is the
   more aggressive option (1 GPU-week, third-party rental).

3. **Wait for hardware.** iPhone 18 Pro / future SoC may bring native
   MoE-routing support or higher DRAM bandwidth. The ceiling is
   silicon-bound, not software-bound.

## Honest assessment

Two negative results in one session, both empirically grounded with
working scripts and concrete dispatch-latency measurements. The
project-memory ceiling of 1.22-1.25× is real; it has now been
independently triangulated by three different lines of investigation
(static prune dead end + dynamic FFN routing dead end + MoE port dead
end).

This is not nothing — the empirical refutations save weeks of
exploratory work for future sessions and rule out a class of
training-free MoE-style proposals that would otherwise keep surfacing.
But for the next 1.5× iPhone speedup, training is required.
