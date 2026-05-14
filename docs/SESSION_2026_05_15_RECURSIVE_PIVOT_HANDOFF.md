# Session 2026-05-15 — From β-1 to the recursive Gemma 4 path

Branch: `feat/mtp-iphone-perf`
8 commits this session (c9a803c was the starting point).

## The arc, in one paragraph

Started on the handoff's Phase β-1 (Gemma 3n FFN slice routing) → dead,
1.18× structural ceiling. Pivoted to Qwen1.5-MoE-A2.7B → CoreML/ANE
designs all failed. Pursued the ANE route exhaustively per the user's
directive — **8 distinct designs, every one Swift-measured** — ANE
caps at ~32 tok/s for MoE because MoE's training objectives
(load-balancing, expert-diversity) structurally produce exactly the
properties ANE can't exploit. Found along the way that the same model
via **MLX** runs at 219 tok/s on Mac (GPU, not ANE). User chose to
keep ANE and accept training: **recursive Gemma 4**. Built + validated
the recursive training harness on Mac. Phase 1 (all Mac-side prep) is
complete; Phase 2 is the GPU uptraining run.

## What's settled (don't re-investigate)

| Path | Verdict | Evidence |
|---|---|---|
| Gemma 3n β-1 FFN slice routing | DEAD — 1.18× ceiling | `docs/PHASE_BETA1_STAGE1_FINDINGS_2026_05_15.md` |
| Qwen MoE on CoreML/ANE | DEAD — 8 measured designs, ~32 tok/s ceiling | `docs/ANE_MOE_ROUTE_EXHAUSTED_2026_05_15.md` |
| Qwen MoE via MLX | WORKS — 219 tok/s Mac 3-bit, but GPU not ANE | `docs/QWEN_MOE_MLX_BREAKTHROUGH_2026_05_15.md` |
| Naive recursive tie | DEAD — breaks Gemma 4 completely | commit 3331552 |
| Recursive low-rank, no training | DEAD — broken at every rank | commit 0ff9f6c |
| **Recursive Gemma 4 + uptraining** | **CHOSEN PATH — harness validated** | commits 0ff9f6c, d41e974 |

Root cause for the whole "training-free ANE 1.5×" search coming up
empty: it's structural. MoE's dynamic routing ⊥ ANE's static graph.
Dense models are already at their ANE ceiling. Activation sparsity is
scattered. The training-free iPhone ceiling of ~1.22-1.25× (from
`project_lever_hunt_ceiling.md`) is now triangulated by ~12
independent investigations.

## The chosen path: recursive Gemma 4

A recursive model ties consecutive decoder layers to a shared block
applied twice — half the per-token weight bandwidth, **and** it's a
static graph with no gather, so ANE runs it natively (unlike MoE).
The catch: the recursive structure must be **trained** to work — the
init alone is broken (commit 0ff9f6c proved this at every rank).

### Phase 1 — COMPLETE (Mac-side, this session)

* `conversion/recursive_tie_experiment.py` — naive tie, signature-aware
  (Gemma 4 E2B is non-uniform: FFN size shifts at L15, every 5th layer
  is global-attention). Result: naive tie → gibberish.
* `conversion/recursive_lowrank_experiment.py` — RRT-style shared base
  + per-position rank-r SVD delta. No-train sweep r∈{0,64,128,256,512}:
  broken at every rank (even r=512/1.08× bandwidth → top-1 0.36,
  repetitive degeneration). Sizes the gap: training is the core method.
* `conversion/recursive_distill_train.py` — distillation uptraining
  harness. Frozen Gemma 4 teacher, recursive student (21/35 unique
  blocks, 220 LoRA modules, 6% trainable). **Mac-validated**: bf16,
  per-token fp32 KL, rank 256, 40 steps on a tiny corpus → KL 31.4→3.98,
  top-1 teacher-agreement 0.8%→37%. Pipeline works, student learns fast.

### Phase 2 — NEXT (needs GPU; user initiates)

Run the full distillation uptrain on a rented A100/H100.
* Command shape:
  ```
  python conversion/recursive_distill_train.py --train \
    --model output/gemma4-e2b/hf_model --rank 256 \
    --corpus <real-corpus.txt> --steps <20k+> --batch 8 \
    --dtype bfloat16 --out output/recursive-gemma4
  ```
* Decisions to make before the run:
  - **Rank**: 256 gives 1.38× tied-pair bandwidth; lower rank = more
    speedup but more to recover. Sweep on a short run first.
  - **train-base**: `--train-base` also unfreezes the shared base
    (heavier, closer to RRT full uptrain — likely needed for real
    quality, LoRA-only may not be enough).
  - **Corpus**: RRT used 60B tokens. With distillation from Gemma 4
    itself as teacher, fewer may suffice — but plan for the
    ~A100 3-5 day / $700-1400 regime.
  - **Init**: current init is average-base. RRT's SVD/Stepwise init
    may give a better starting point — worth a quick A/B on a short run.
* Cost: ~$700-1400 GPU rental, 3-5 days wall-clock. The user must set
  up GPU access — I can't rent hardware.

### Phase 3 — AFTER Phase 2

Convert the trained recursive model to CoreML/ANE. The recursive
structure → ANE chunks where the shared block's weights are read once
and reused. Mac smoke (quality cos sim vs trained PyTorch reference) →
iPhone 17 Pro bench. Acceptance: iPhone tok/s ≥ 1.5× current Gemma 4
baseline AND quality parity. This is where the training effort lands
on an ANE-deployable result.

## Tasks state (TaskList)

* #17 Phase 1a recursive low-rank no-train sweep — COMPLETED
* #18 Phase 1b distillation harness — COMPLETED
* #19 Phase 2 GPU uptraining — PENDING (user initiates GPU access)
* #20 Phase 3 convert + iPhone bench — PENDING (after Phase 2)
* #11 MLX Qwen MoE iPhone path — COMPLETED Mac-side (the alternative,
  not chosen, but documented; 219 tok/s Mac, iPhone unmeasured)

## Off-tree resources

* `/tmp/qwen_moe_3bit/` — 3-bit MLX Qwen MoE (5.9 GB). The MLX path's
  artifact. Keep if MLX stays a fallback; delete to reclaim disk.
* `output/gemma4-e2b/hf_model/` — Gemma 4 E2B HF weights (in-repo,
  used by the recursive scripts as teacher).
* `/tmp/recursive_*.json`, `/tmp/tie_experiment.json` — experiment
  reports (small).

## Memory updated this session

* `project_qwen15_moe_pivot.md` — ANE route exhausted (8 designs),
  MLX path documented.
* `feedback_no_proxy_benchmarks.md` — don't trust Python predict() for
  ANE latency verdicts.
* `feedback_dont_give_up_single_perspective.md` — question premises,
  enumerate alternatives, don't give up from one angle.

## Resume protocol for next session

```
Branch: feat/mtp-iphone-perf @ d41e974
State: recursive Gemma 4 path — Phase 1 complete, Phase 2 (GPU) next.
Read: docs/SESSION_2026_05_15_RECURSIVE_PIVOT_HANDOFF.md (this file)

Phase 1 artifacts (Mac-validated):
  conversion/recursive_lowrank_experiment.py  — sized the gap
  conversion/recursive_distill_train.py       — the training harness

Phase 2 is the GPU uptrain. Before spending:
  1. Short-run rank sweep + init A/B (average vs SVD) + LoRA-only vs
     --train-base, to pick the operating point.
  2. Pick a real corpus.
  3. User sets up A100/H100 access.
  4. Run --train, target near-parity top-1 agreement with Gemma 4.

Then Phase 3: convert to CoreML, iPhone bench.

Do NOT re-investigate: Gemma 3n β-1, Qwen MoE on ANE (8 designs all
measured dead), naive recursive tie, recursive-no-train. All settled.
```

## Honest assessment

This session converted "is there a training-free ANE 1.5×?" from an
open question into a measured **no** — ~12 investigations, all
negative, root cause identified. That negative is itself valuable: it
ends a long line of speculative levers. The recursive Gemma 4 path is
the one survivor — it's ANE-compatible (the training effort actually
lands on ANE, unlike MoE or drafter training) and the harness is built
and validated. But it is a real training commitment ($700-1400, days),
not a free lunch — Phase 1a proved the init alone doesn't work. Phase 2
is a genuine spend decision the user makes with eyes open.
