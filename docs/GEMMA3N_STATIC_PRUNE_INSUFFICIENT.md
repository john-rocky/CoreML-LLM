# Gemma 3n Static Prune — Insufficient (2026-05-14)

Phase α-2 follow-up to `docs/GEMMA3N_SPARSITY_VALIDATED_2026_05_14.md`.

The empirical sparsity (95% dormant neurons on L0-9) is REAL but
**static structural pruning fails on specialised tasks** because the
trained sparsity is dynamic — a different 5% of neurons fires per
token. The calibration set's union of "frequently fired" neurons does
not cover what code generation needs.

## Methodology

Tool: `conversion/prune_gemma3n_sparse_ffn.py`. Reads the sparsity
calibration JSON (Phase α-2 outputs from
`conversion/calibrate_ffn_sparsity.py`) and slices each sparse layer's
`mlp.{gate,up,down}_proj` to the top-K most-frequently-fired neurons.
Updates the HF `config.json` with per-layer `intermediate_size` so the
stock Gemma 3n loader honours the reduction.

Pruned three variants (keep-pct = 0.30, 0.50, 0.70), each on top of
the original 5.4 B-param model. Sparse layers retained 2457 / 4096 /
5734 neurons respectively (of 8192).

Quality smoke (`/tmp/gemma3n_sweep_smoke.py`) — 32-token completions,
greedy, on three prompts:

| keep-pct | code BST | AI essay | "say yes 10 times" |
|---|---|---|---|
| ORIGINAL | ✅ proper `class BinarySearchTree:` def | ✅ coherent | ✅ "Okay, I've said yes 10 times" |
| 70% | ❌ stuck looping `class BinarySearchTree:` | ✅ slight rewording | ✅ "Yes Yes Yes" properly |
| 50% | ❌ same code loop | ⚠️ coherence borderline | ⚠️ awkward repeat pattern |
| 30% | ❌ ``` markdown noise | ✅ surprisingly coherent | ❌ off-topic ("What does it mean?") |

## What this tells us

1. **The sparsity IS real** (calibration: 95% dormant matches config).
2. **Static prune cannot exploit it** for general use because:
   - Trained 95% sparsity is **dynamic** — the specific 5% varies per
     token
   - Our calibration corpus (mixed English narrative + light tech)
     does NOT capture the neurons needed for code generation
   - Even keep-pct = 0.70 (5734 / 8192 retained, lossing 30% of
     trained sparsity-zone freedom) **already breaks code prompts**
3. **Essay-style prompts tolerate static prune** because the
   high-magnitude neurons are well-covered by a generic calibration.
4. **Per-prompt calibration is impractical** — we can't ship a
   different model per query type.

## The real path forward

To actually exploit Gemma 3n's trained sparsity at inference time, we
need **dynamic top-K dispatch** — the Apple Foundation Model approach:

1. At inference, every FFN block runs a tiny predictor over the
   intermediate activations (or a learned router on the hidden state)
   and selects the top-K neurons to compute.
2. Only the selected K neurons' weights are read from DRAM → 95%
   bandwidth savings on sparse layers.
3. The selection MUST be per-token because the trained sparsity
   pattern is per-token.

This maps directly to the user-proposed architecture: **MoE-style
routing with the router living outside ANE (CPU/GPU side) + many
small CoreML-callable expert kernels**. Concretely:

* Sparse layer's FFN gets exported as ~N small CoreML graphs, each
  containing a slice of the intermediate neurons (`8192 / N` per
  graph)
* At inference, a tiny CPU-side predictor scores each slice
* Only top-K slices fire on ANE
* Output sum across selected slices

This is the user's instinct restated. It requires:

* A CoreML graph partition pass that emits per-slice mlmodelc files
* A Swift orchestrator that scores + dispatches
* The router can be a learned linear layer trained on per-token gate
  values during calibration

Effort: **~1 week prototype**, much more for production.

## Alternative: live with what we have

Without dynamic routing, the practical options on Gemma 3n are:

1. **Don't prune** — use Gemma 3n at full size, get its baseline
   quality (likely ~13-14 tok/s on Mac MPS fp16 per smoke; would be
   25-30 tok/s on ANE INT4 same as Gemma 4 E2B). No win.
2. **Prune dense layers only (L10-29)** — wait, those aren't sparse,
   so we can't prune them.
3. **Use Gemma 3n as a multi-turn drafter / cross-vocab drafter** —
   different question.

So **without dynamic routing infrastructure, Gemma 3n offers no
clear win over Gemma 4 E2B for our use case**.

## Decision

Given the user-stated constraint of "no training" and the user's
explicit instinct toward "MoE routing outside CoreML," the right next
direction is to **prototype the dynamic top-K dispatch architecture**.
This is a multi-day project; tonight's bench session confirmed the
upper bound (95% sparse is real) and the lower bound (static prune
fails for code).

Recommended for next session:

1. Investigate Apple's `ml-compress` / `coremltools` slice-export
   primitives for per-slice CoreML graphs
2. Design the Swift orchestrator API (synchronous CPU score → ANE
   dispatch top-K)
3. Build a single-layer prototype on Mac before scaling to full model
4. Compare against full-dense Gemma 3n baseline for both quality and
   wall-clock

If the dynamic-routing infrastructure doesn't pan out within ~1 week,
the realistic conclusion is **training is required** (Path B drafter
distillation remains the only training-free [sic] ceiling-lift).

## Files

* `conversion/prune_gemma3n_sparse_ffn.py` — static prune helper
* `/tmp/gemma3n_pruned_smoke.py` — quality comparator (not committed,
  trivial)
* `/tmp/gemma3n_sweep_smoke.py` — 50%/70% sweep
* `/tmp/gemma3n-e2b-pruned{,50,70}/` — pruned checkpoints (not
  committed, ~5 GB each)
* `/tmp/sparsity_gemma3n.json` — input calibration data
