# Gemma 3n FFN Activation Sparsity — Empirically Validated (2026-05-14)

Phase α-2 of the sparse-activation roadmap. Following the Phase α-1
finding that Gemma 4 E2B is densely activated and therefore unsuitable
for post-training sparsification (see
`docs/SPARSITY_CALIBRATION_2026_05_14.md`), we ran the same calibration
on Gemma 3n E2B (`google/gemma-3n-E2B-it`).

## TL;DR

**Gemma 3n's `activation_sparsity_pattern` in the HF config is REAL.**
Layers 0-9 have 95% of FFN intermediate neurons dormant per token, as
the config advertises. Layers 10-29 are dense (config `0.0`, observed
hit rate 0.34-0.86).

| layer | hits/tok | top10%cov | classification |
|---:|---:|---:|---|
| 0  | 0.033 | 0.76 | 95% sparse (validated) |
| 1  | 0.054 | 0.60 | 95% sparse (validated) |
| 4  | 0.025 | 0.81 | 95% sparse (validated) |
| 9  | 0.057 | 0.60 | 95% sparse (validated) |
| 10 | 0.798 | 0.21 | dense |
| 14 | 0.770 | 0.23 | dense |
| 20 | 0.738 | 0.22 | dense |
| 29 | 0.817 | 0.34 | dense |

Across sparse layers (0-9), **hit rate per neuron averages 0.046** —
only 4.6% of the 8192-wide intermediate neurons fire per token. Top
10% of neurons cover 49-81% of total magnitude.

## What this enables

**For sparse layers:**
- Bandwidth: skip ~95% of FFN weight reads
- Compute: skip ~95% of FFN matmul work (and ANE is bandwidth-bound
  anyway, so this is the real win)
- Quality: minimal regression because the model was trained with this
  sparsity baked in — those 95% of neurons WERE supposed to be ~0

**For the model overall:**
- 10/30 layers (33% of network) get the sparsity treatment
- FFN ≈ 2/3 of decoder bandwidth (rest is attention QKV/O + RMSNorm)
- Sparse-layer FFN bandwidth saving: 95% × (2/3) × (10/30) ≈ **21%**
  of total model bandwidth
- At iPhone 17 Pro's 60-77 GB/s and E2B at ~1.5 GB/token, theoretical
  decode ceiling lifts from ~34 tok/s to ~43 tok/s (a 1.26× win)
- If we ALSO get post-training top-K sparsification on layers 10-29
  (at lossier 0.5× sparsity), additional ~10% bandwidth win
- Combined ceiling: **~50 tok/s** training-free on iPhone 17 Pro

## Calibration setup (same as Phase α-1)

Identical methodology to `docs/SPARSITY_CALIBRATION_2026_05_14.md`:
- 681-token English calibration corpus
- fp16 weights on MPS, fp32 accumulation on CPU
- forward-pre-hook on each layer's `down_proj` captures the SwiGLU
  intermediate magnitude per token
- threshold for "hit": |activation| > 0.01

Numpy 2.0 / wandb / timm import-chain compatibility shims required for
Gemma 3n loading (committed in the calibration script).

## What's needed to convert this to a CoreML speedup

Three distinct paths:

### Path 1 — Static structural pruning (training-free)

For each sparse layer (0-9), identify the top-K most-frequently-fired
neurons across the calibration corpus and **physically delete the
others** at conversion time. The resulting FFN has only K (e.g. 256)
intermediate neurons instead of 8192.

* Pros: easy CoreML conversion (static graph), no runtime overhead,
  no routing logic, no quality loss vs the sparse mask itself
* Cons: K must be chosen high enough to cover unseen contexts
  (probably K ≈ 512-1024 not the 410 = 5% of 8192 used in training)
* Estimated wall-clock win: 80% of theoretical (since K is somewhat
  bigger than the trained sparsity rate)

### Path 2 — Dynamic top-K masking (runtime routing)

Predict per-token which K neurons to evaluate. Apple FM 2025 uses
a tiny predictor head + top-K selection. CoreML supports `topk` and
`gather`, but dynamic neuron selection per-token would require either
(a) compiling all 2^N possible subsets ahead of time (impossible) or
(b) leaving the dense compute on ANE but masking outputs (no win).

* Pros: closer to Apple's actual production design
* Cons: doesn't naturally fit CoreML's static graph model. Plausibly
  feasible via MLState + multifunction but blocked by ANE MLState
  compiler rejection.

### Path 3 — Hybrid: ANE-dense + GPU-routed

Run dense FFN on ANE for first cycles to gather statistics, then
switch to GPU/CPU routing for the sparse subset. Discarded — same
complexity as Path 2, smaller absolute gain.

**Recommended next step**: Path 1 with K=1024 (12.5% retention, vs
5% trained sparsity). Calibrate on a larger corpus (4-8k tokens
mixed domains) to ensure neuron coverage is robust to unseen
prompts. Build a conversion script that produces shrunken FFNs
for the sparse layers, leaves dense layers untouched. Verify
output parity on Mac, then iPhone bench.

## Comparison to Gemma 4 E2B

| metric | Gemma 4 E2B | Gemma 3n E2B |
|---|---|---|
| Sparse layers | 0 / 35 | **10 / 30** |
| Avg hit rate (across all layers) | 0.32-0.84 (dense) | 0.025-0.86 (bimodal) |
| Top 10% coverage on most-sparse layer | 0.50 (dense L34) | **0.81 (sparse L4)** |
| Trained for sparsity | ❌ | ✅ |
| Training-free sparsification viable | ❌ | **✅** |
| Cross-vocab compatibility with our pipeline | native | shares tokenizer with Gemma 4 |

## Path forward implication

Gemma 3n becomes a **legitimate alternative target model**. The
existing Gemma 4 E2B pipeline can probably be adapted because:
- Same vocab (Gemma 4 uses gemma-tokenizer, Gemma 3n's text submodel
  uses same vocab_size=262144 — needs verification of exact merges)
- Same architecture family (decoder-only transformer with SwiGLU)
- Different layer count (30 vs 35) — chunking would shift
- Different sparsity pattern — needs the structural-prune conversion
  pass

The user-stated directive "別モデルでいいからね" (it's fine to use a
different model) authorises this pivot.

Open questions for next session:
1. Is Gemma 3n quality comparable to Gemma 4 E2B on our test prompts?
   (PR/HF reports it's competitive on benchmarks but downstream
   instruction-following quality is another question)
2. Does Apple ANE compile the Gemma 3n architecture with the same
   chunking we use for Gemma 4? (vision/audio tower complications
   need to be excluded — text-only path matters)
3. What K do we pick for structural pruning? (trade quality vs
   speedup)

## Files

* `conversion/calibrate_ffn_sparsity.py` — script (numpy/wandb shims
  added for Gemma 3n loading)
* `/tmp/sparsity_gemma3n.json` — raw per-neuron data (3.5 MB,
  regenerate via the script)
* `/tmp/gemma3n-e2b/` — HF weights (10.9 GB, not committed)
