# FFN Activation Sparsity — Calibration Findings (2026-05-14)

Phase α-1 of the sparse-activation roadmap (`docs/SESSION_2026_05_14_HANDOFF.md`
Round F + follow-up). Goal: figure out whether post-training FFN
sparsification on Gemma 4 E2B would deliver a meaningful bandwidth win
without retraining.

## TL;DR

**Gemma 4 E2B is densely activated.** Post-training FFN sparsification
will not deliver the 5-10× bandwidth win that Apple-FM-class models
(with built-in activation sparsity) achieve. Lossy without retrain.

The sparse-activation lever requires either:
1. switching to a model trained with `activation_sparsity_pattern`
   (Gemma 3n is the obvious candidate)
2. or accepting major quality loss

## Methodology

Script: `conversion/calibrate_ffn_sparsity.py`.

```bash
pyenv shell lama-cml
python conversion/calibrate_ffn_sparsity.py \
  --model output/gemma4-e2b/hf_model \
  --tokens 1024 \
  --out output/sparsity_gemma4_e2b.json
```

For each FFN layer, forward-pre-hook on `down_proj` captures the
SwiGLU intermediate `act(gate) * up` per token. Per-neuron magnitude
sums and hit counts (|act| > 0.01) accumulate across the calibration
corpus. Output JSON has top-K indices at 10/20/30/50/70% thresholds
plus cumulative magnitude coverage curve per layer.

Calibration corpus: 680-token mixed English passage covering
mechanical/electronic computing history, mobile NPU constraints,
activation-sparsity context, and reasoning tasks. Single forward.

Loaded as fp16 on MPS (Mac Silicon GPU). Activations moved to CPU
fp32 for accumulation (MPS doesn't support fp64).

## Per-layer coverage table

| layer | top10% cov | top20% cov | top30% cov | top50% cov | top70% cov | hits/tok |
|---:|---:|---:|---:|---:|---:|---:|
|  0 | 0.29 | 0.45 | 0.57 | 0.75 | 0.88 | 0.84 |
|  1 | 0.32 | 0.49 | 0.62 | 0.80 | 0.92 | 0.73 |
|  2 | 0.26 | 0.42 | 0.54 | 0.71 | 0.85 | 0.84 |
|  3 | 0.25 | 0.40 | 0.51 | 0.69 | 0.83 | 0.79 |
|  4 | 0.40 | 0.57 | 0.70 | 0.86 | 0.94 | 0.63 |
|  5 | 0.28 | 0.43 | 0.54 | 0.71 | 0.85 | 0.83 |
|  6 | 0.27 | 0.41 | 0.52 | 0.68 | 0.82 | 0.80 |
|  7 | 0.30 | 0.44 | 0.55 | 0.71 | 0.84 | 0.79 |
|  8 | 0.33 | 0.47 | 0.58 | 0.73 | 0.85 | 0.73 |
|  9 | 0.29 | 0.44 | 0.55 | 0.72 | 0.86 | 0.68 |
| 10 | 0.30 | 0.45 | 0.56 | 0.73 | 0.86 | 0.71 |
| 11 | 0.30 | 0.45 | 0.56 | 0.74 | 0.86 | 0.69 |
| 12 | 0.31 | 0.46 | 0.58 | 0.75 | 0.87 | 0.66 |
| 13 | 0.27 | 0.44 | 0.56 | 0.74 | 0.87 | 0.55 |
| 14 | 0.34 | 0.50 | 0.62 | 0.78 | 0.89 | 0.67 |
| 15 | 0.39 | 0.54 | 0.65 | 0.80 | 0.90 | 0.77 |
| 16 | 0.38 | 0.52 | 0.62 | 0.77 | 0.89 | 0.65 |
| 17 | 0.39 | 0.51 | 0.60 | 0.76 | 0.88 | 0.69 |
| 18 | 0.47 | 0.58 | 0.67 | 0.80 | 0.90 | 0.62 |
| 19 | 0.35 | 0.49 | 0.60 | 0.76 | 0.88 | 0.66 |
| 20 | 0.33 | 0.47 | 0.58 | 0.75 | 0.87 | 0.72 |
| 21 | 0.32 | 0.46 | 0.56 | 0.73 | 0.86 | 0.66 |
| 22 | 0.33 | 0.47 | 0.57 | 0.74 | 0.87 | 0.67 |
| 23 | 0.39 | 0.53 | 0.63 | 0.77 | 0.88 | 0.79 |
| 24 | 0.32 | 0.46 | 0.57 | 0.74 | 0.87 | 0.75 |
| 25 | 0.37 | 0.50 | 0.60 | 0.75 | 0.87 | 0.76 |
| 26 | 0.33 | 0.45 | 0.55 | 0.71 | 0.84 | 0.79 |
| 27 | 0.35 | 0.46 | 0.56 | 0.72 | 0.85 | 0.75 |
| 28 | 0.38 | 0.49 | 0.58 | 0.73 | 0.86 | 0.67 |
| 29 | 0.36 | 0.47 | 0.56 | 0.71 | 0.84 | 0.62 |
| 30 | 0.32 | 0.43 | 0.52 | 0.68 | 0.82 | 0.49 |
| 31 | 0.31 | 0.42 | 0.51 | 0.68 | 0.82 | 0.40 |
| 32 | 0.36 | 0.46 | 0.55 | 0.70 | 0.83 | 0.32 |
| 33 | 0.42 | 0.54 | 0.62 | 0.76 | 0.87 | 0.45 |
| 34 | 0.50 | 0.61 | 0.69 | 0.81 | 0.90 | 0.66 |

## Reading the table

* **Top-K cov** = fraction of total |activation| magnitude captured by
  taking the K%-largest neurons (sorted by sum |a| across all tokens).
* **hits/tok** = average per-neuron fire rate at threshold 0.01.

To match Apple Foundation Model 2025 sparsity (~95% magnitude in top
10%), top10%-cov needs to be ≈ 0.95 on every layer. Observed:
**0.25–0.50**, half of what's needed.

## Comparison to a useful target

| status | top10%cov | top30%cov | bandwidth win |
|---|---|---|---|
| ideal (Apple FM-class) | 0.95 | 0.99 | 5-10× |
| Gemma 4 E2B (this study) | **0.25–0.50** | **0.51–0.70** | 1.4-1.5× (lossy) |

The 1.4-1.5× win comes from taking top 70% of neurons and dropping the
rest. That's lossy without fine-tuning to compensate, and the speedup
margin barely justifies the quality regression.

## Conclusion

Post-training FFN sparsification on Gemma 4 E2B is **not worth
pursuing**. Two paths forward:

1. **Switch to Gemma 3n** — model is trained with
   `activation_sparsity_pattern` baked into the config. Designed for
   on-device sparse activation. Apple's own foundation models use a
   related pattern. Repeat this calibration on Gemma 3n; expect
   top10%cov ≈ 0.90+ if the sparsity pattern is honored. (Gemma 3n
   download in progress at end of session.)

2. **Switch focus to small-drafter approach** — SmolLM2 135M / 360M
   already downloaded, cross-vocab map to Gemma 4 built (94.3%
   coverage), conversion pipeline blocker is the absence of a Llama
   model module under `conversion/models/`. Building that wrapper is
   tractable but ~1 day of work.

Both paths preserve the "different model is OK" directive.

## Files

* `conversion/calibrate_ffn_sparsity.py` — script (this report's source)
* `output/sparsity_gemma4_e2b.json` — full per-neuron data (3.5 MB,
  not committed; regenerate via the script)
* `output/smollm135_gemma_vocab.bin` — SmolLM 135M ↔ Gemma 4 cross-vocab
  map, 94.3% qwen→gemma coverage (not committed; rebuild via
  `conversion/build_qwen_gemma_vocab_map.py`)
* `output/smollm360_gemma_vocab.bin` — same for SmolLM 360M
  (identical vocab to 135M)
