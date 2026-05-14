# Session 2026-05-15 — From β-1 (Gemma 3n routing) to Qwen MoE pivot

Branch: `feat/mtp-iphone-perf` (continuing 2026-05-14 handoff state)
Predecessor: `docs/SESSION_2026_05_15_HANDOFF.md`,
`docs/PHASE_BETA1_STAGE1_FINDINGS_2026_05_15.md`

## Two-sentence summary

Phase β-1 (Gemma 3n sparse-FFN routing) failed Stage 1: oracle K=2 of 16
cos sim was 0.766 (vs handoff target ≥0.98), and the architecture ceiling
even with perfect routing maxes at 1.18× iPhone speedup — not the 1.5×
the handoff promised. After verifying with bandwidth math, pivoted to
**Qwen1.5-MoE-A2.7B-Chat** (1.46× iPhone BW reduction vs Gemma 4 E2B at
INT4) as the new deployment target.

## What was tried (β-1)

### Stage 1 sub-experiments

1. **`conversion/stage1_dynamic_router_l0.py`** — original prototype:
   contiguous-index slices, linear router. Result: oracle cos=0.75 at K=2/16,
   router cos=0.68. Below acceptance.

2. **`conversion/stage1_analyze_slice_structure.py`** — slice entropy
   diagnostic. Result: entropy 2.71 of max 2.77 — firing neurons
   essentially uniform across contiguous slices.

3. **`conversion/stage1_verify_multi_layer.py`** — L0/4/9/14 all show
   same near-max entropy. Plus L0-9 had 94-98% of |Y| values *exactly
   zero* (trained sparsity is real and hard), but those firing 2-5%
   are spatially random.

4. **`conversion/stage1_capture_large_corpus.py`** — re-captured on
   2229-token mixed corpus.

5. **`conversion/stage1_cofiring_cluster.py`** — first positive
   signal. Co-firing-based spectral clustering before slicing pushed
   K=2 oracle cos from 0.77 → 0.90.

6. **`conversion/stage1_cluster_eval.py`** — full eval with held-out
   test set + linear router. Spectral and balanced k-means both reach
   oracle cos 0.987 at K=6/16, but the linear router lags by ~0.06.
   Router only hits 0.98 at K=12/16 (25% bandwidth save — useless).

### Architecture ceiling computation

A separate sanity check (inline Python in this session) revealed the
handoff's "21 % bandwidth saveable" figure was computed against decoder
bandwidth only, ignoring embed+lm_head (35 % of total on Gemma 3n due
to its 262 k vocab). Correct sparse-FFN-only bandwidth share = 16.3 %
of total. Even oracle-perfect routing caps speedup at 1.18×.

Conclusion: β-1's premise was based on flawed bandwidth math.

### What was good about β-1 (for future reference)

* The co-firing permutation result is a real finding — if a future
  block-sparse-trained model comes along, "permute neurons by co-firing
  before slicing" is a real lever, validated.
* The slice-entropy diagnostic is a portable test for "is this model
  block-sparse-friendly". Reusable on any architecture.
* The training-free router framework (Python BCE on slice-membership)
  is a reference if we ever try this on a future block-clustered model.

## The pivot — Qwen1.5-MoE-A2.7B-Chat

Memory entry: `project_qwen15_moe_pivot.md`.

Key facts from the Qwen team's published config + HF model card:

| Property | Value |
|---|---|
| Architecture | Qwen2MoeForCausalLM |
| Total params | 14.3 B |
| Active params/tok | 2.7 B |
| Layers | 24 |
| Hidden | 2048 |
| Num experts | 60 |
| Top-K routing | 4 |
| Expert intermediate | 1408 |
| Shared expert intermediate | 5632 |
| Attention heads | 16 (no GQA) |
| Vocab | 151,936 |
| Tied embed | **false** |
| License | Tongyi Qianwen |

iPhone INT4 bandwidth math:

* Per-token bandwidth: 5.38 GB at fp16 → 1.34 GB at INT4
* iPhone 17 Pro 70 GB/s ceiling: 52 tok/s
* Gemma 4 E2B at INT4: 1.97 GB/tok → 35.6 tok/s ceiling
* **Pure-BW speedup vs Gemma 4: 1.46×**

This is "trained β-1": the per-token expert routing the original Phase
β-1 plan tried to engineer from scratch is already pre-trained in the
gate networks of Qwen MoE.

## Engineering plan

Now claimed in tasks (#6 / #7 / #8):

### Phase A — Mac validation (3-5 days)

* Download fp16 weights (~28 GB; download started in background)
* Run baseline tok/s on Mac MPS for 4 prompt classes
* Hook every layer's expert gate, capture which 4 of 60 experts fire
  per token across a 1k-2k token corpus
* Quality smoke vs Gemma 4 E2B output for same prompts
* Gate: routing matches assumed 4/60 + shared structure, no degenerate
  collapse, no expert pruning. PASS → Phase B.

Script: `conversion/phase_a_qwen_moe_analysis.py`.

### Phase B — CoreML single-expert feasibility (1-2 weeks)

* Extract L0 expert 0 weights (~8.65 M params)
* Convert to minimal SwiGLU mlpackage via coremltools
* Verify ANE residency on Mac, time dispatch
* Extrapolate per-token Mac wall-clock: 5 dispatches × 24 layers
* Gate: extrapolated Mac tok/s ≥ Gemma 4 baseline. PASS → Phase C.
  This is the critical "can ANE handle MoE dispatch" question; if NO,
  the whole pivot fails and we stop.

### Phase C — Full port (2-4 weeks)

* 60 × 24 = 1440 expert mlpackages OR a smarter packaging scheme
* 24 router mlpackages
* INT4 palettization compatible with CoreML
* Swift orchestrator mirroring current `ChunkedEngine` for dynamic
  expert dispatch
* Mac smoke (quality cos sim parity vs fp16 reference, tok/s)
* iPhone bench under thermal budget
* HF upload of finalised .mlpackages

Acceptance: iPhone ≥ 50 tok/s (1.43× over current Gemma 4)
AND quality cos sim parity vs fp16.

## Risks tracked

1. **ANE MoE dispatch overhead** — 60 experts × 24 layers = 1440 potential
   dispatch sites. If ANE serialises and each dispatch is ≥1 ms, we lose
   to Gemma 4. Phase B answers this.
2. **iPhone RAM pressure** — 7 GB INT4 model vs Gemma 4's 1.5 GB.
   iPhone 17 Pro nominal RAM is ~12 GB but the OS won't give us all of
   it. Memory-pressure kills are a real risk; mmap-only access patterns
   may be required.
3. **Quality vs Gemma 4** — Qwen team's claim is "comparable to Qwen1.5-7B
   at 1.74× faster" but we're comparing to Gemma 4 E2B. Phase A
   validation will tell us if quality is at least parity.
4. **License** — Tongyi Qianwen, not Apache. Commercial deployment
   needs legal review; not blocking research.
5. **No prior ART** — HF model card shows no iOS/CoreML conversion
   attempt for Qwen MoE. We're first. Higher risk of unknown unknowns.

## Files modified / added this session

### New scripts
* `conversion/stage1_dynamic_router_l0.py`
* `conversion/stage1_analyze_slice_structure.py`
* `conversion/stage1_verify_multi_layer.py`
* `conversion/stage1_capture_large_corpus.py`
* `conversion/stage1_cofiring_cluster.py`
* `conversion/stage1_cluster_eval.py`
* `conversion/phase_a_qwen_moe_analysis.py`

### New docs
* `docs/PHASE_BETA1_STAGE1_FINDINGS_2026_05_15.md` (β-1 negative result)
* `docs/SESSION_2026_05_15_DYNAMIC_ROUTING_TO_MOE_PIVOT.md` (this file)

### Memory
* `memory/project_qwen15_moe_pivot.md`

### Off-tree (not committed)
* `/tmp/l0_activations.npz`, `/tmp/l0_activations_large.npz`
* `/tmp/stage1_*.json` reports
* `/tmp/qwen15-moe-chat/` (download target, ~28 GB fp16)

## Resume protocol for next session

If today's work gets interrupted before Phase A completes:

```
Branch: feat/mtp-iphone-perf @ <commit hash after this session>
State: Qwen1.5-MoE pivot — Phase A in progress
Goal: Complete Phase A (Mac validation + routing analysis) on
      /tmp/qwen15-moe-chat (verify download completed first)

Read first:
* docs/SESSION_2026_05_15_DYNAMIC_ROUTING_TO_MOE_PIVOT.md (this)
* docs/PHASE_BETA1_STAGE1_FINDINGS_2026_05_15.md (why we abandoned β-1)
* memory/project_qwen15_moe_pivot.md

Do not re-investigate: Gemma 3n FFN routing. Closed by structural
math. The handoff's bandwidth math was wrong (used decoder-only share
instead of total-model share).

Next action: run
  PYENV_VERSION=lama-cml python conversion/phase_a_qwen_moe_analysis.py
on /tmp/qwen15-moe-chat. Check /tmp/phase_a_qwen_moe_report.json
output. Decide GO/NOGO for Phase B.
```
