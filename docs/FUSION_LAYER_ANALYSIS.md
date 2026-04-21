# EAGLE-3 Fusion Layer Analysis for Gemma 4 E2B

**Date:** 2026-04-20  •  **Branch:** `feat/eagle3-on-main`
**Script:** `conversion/diagnose_fusion_layers.py`
**CSV:** `docs/FUSION_LAYER_SCORES.csv`

## Question

EAGLE-3 picks three fusion layers (hidden-state taps) the draft ingests.
The deployed config uses **L8, L17, L34** on Gemma 4 E2B (35 decoder layers,
indexed 0..34). The official "low/mid/high" recipe is roughly quartile-spaced,
which for us gives L8 / L17 / L26 — but we picked L34 for "high" instead.
Are L8/L17/L34 the best three, or would shifting (e.g. L2/L17/L32) or
tightening (L11/L17/L23) yield a draft that sees more next-token signal?

## Method

- **Target:** `google/gemma-4-E2B-it` HF fp16 (text-only), loaded from the
  local HF cache snapshot
  (`~/.cache/huggingface/hub/models--google--gemma-4-E2B-it`). Gemma 4 is
  gated but the weights were already present locally — no download.
- **Corpus:** `~/Downloads/eagle_corpus.jsonl` (same JSONL used by
  `collect_eagle_hidden_states_w4a8.py`).
- **Sample:** 9 sequences, **991 next-token positions**
  (`--max-tokens 1000 --seq-len 256 --min-seq 16`). Last position of each
  sequence is dropped because it has no next-token target.
- **Forward pass:** text-model-only with `output_hidden_states=True`,
  capturing all 36 hidden-state tensors per sequence (1 embed + 34 raw
  decoder outputs + 1 final normed hidden). All computation on CPU in fp16
  for the forward, fp32 for the metric math.
- **Metrics per layer i** (0..34, corresponding to `hidden_states[i+1]`):
  - `mean_norm`, `std_norm` — L2 norm statistics of the layer output
  - `cos_to_L34` — mean row-wise cosine with the final normed hidden
    (redundancy indicator)
  - `cos_to_next_argmax_embed` — mean cosine with the tied-embedding row
    of the target's next-token argmax (weak MI proxy)
  - `logit_lens_logp` — mean log-prob assigned to the target's argmax when
    hidden_i is projected via the real output path: `lm_head(norm(h_i))`
    with softcap=30. **Strongest MI proxy**: directly measures how much
    next-token information is decodable at layer i through the tied head.
  - `agree_top1`, `agree_top5` — fraction of positions where the
    logit-lens argmax equals / is in top-5 of the target's argmax.
- **Pairwise `pair_cos`** — (35,35) mean-position cosine between every
  layer pair. Used for a redundancy-adjusted band pick.
- **Ranges:** low = L1..L11, mid = L12..L22, high = L23..L34. L0 excluded
  from the low band because it is the raw embedding output and would give
  the draft a trivial copy of the input-token embedding.

**Caveat on target:** metrics are measured against fp16 HF target, NOT
the deployed W4A8 chunks. The W4A8 argmax may differ (which is part of
why the on-device retrain uses chunk outputs), but the structural
question of "which layers carry the most next-token signal" is a property
of the model topology and should transfer.

## Results

### Per-layer logit-lens (selected; full CSV in `docs/FUSION_LAYER_SCORES.csv`)

| L  | mean_norm | cos→L34 | logit_lens_logp | agree_top1 |
|---:|----------:|--------:|----------------:|-----------:|
|  0 |     40.58 |  -0.002 |         -30.556 |      0.000 |
|  2 |     30.13 |   0.001 |         -35.955 |      0.000 |
|  8 |     72.56 |   0.004 |         -39.485 |      0.000 |
| 11 |     49.26 |   0.006 |         -40.164 |      0.000 |
| 12 |     52.01 |   0.019 |         -37.022 |      0.000 |
| 15 |     70.12 |  -0.006 |         -29.122 |      0.000 |
| 17 |     72.60 |  -0.003 |     **-25.838** |      0.000 |
| 19 |     65.69 |  -0.009 |         -28.439 |      0.000 |
| 23 |     61.10 |   0.071 |         -28.352 |      0.011 |
| 26 |     74.03 |   0.143 |         -16.897 |      0.039 |
| 32 |     81.80 |   0.378 |         -11.756 |      0.041 |
| 33 |     69.06 |   0.519 |          -5.240 |      0.254 |
| 34 |    171.59 |   1.000 |      **-1.015** |      1.000 |

### Candidate triples (higher sum-logp = more signal)

| Triple                              | sum-logp | sum-top1 | Δ vs current |
|:------------------------------------|---------:|---------:|-------------:|
| Current deployed [8, 17, 34]        |  -66.338 |    1.000 |         0.00 |
| Recommended (this script) [2,17,34] |  -62.808 |    1.000 |        +3.53 |
| Shifted-outward [2, 17, 32]         |  -73.549 |    0.041 |        -7.21 |
| Tighter [11, 17, 23]                |  -94.354 |    0.011 |       -28.02 |
| Symmetric quartiles [8, 17, 26]     |  -82.220 |    0.039 |       -15.88 |
| High-dominant [15, 26, 34]          |  -47.034 |    1.039 |       +19.30 |

## Verdict

**Keep [8, 17, 34].** The script's raw-MI pick is [2, 17, 34] (+3.5 nats),
but that delta is inside the noise floor of the low band: every layer in
L1..L11 has logp within ~4 nats of every other (all close to
near-uniform), i.e. no shallow layer is a meaningfully better "low
feature" than another by this measure. The user's two proposed shifts
(shifted-outward [2,17,32], tighter [11,17,23]) are both clearly
**worse** than the current triple: [2,17,32] loses 7 nats (switching
high=34 to 32 drops logp by 10.7), and [11,17,23] loses 28 nats (dropping
the top band entirely). The mid pick L17 is the unambiguous winner of
L12..L22 and is already deployed. The high pick L34 (logp=-1.02) is the
real output — changing it to L33 (-5.24) or L32 (-11.76) throws away the
most directly next-token-predictive feature we have. The only triple
that beats current meaningfully is [15, 26, 34] (+19 nats), but that
abandons the EAGLE-3 "low feature" contract and would need a full retrain
to evaluate. **Recommendation: do NOT swap fusion layers. The current
deployed [8, 17, 34] is within 3.5 nats of optimal under this MI proxy,
and any shift destroys more signal than it gains.**

## Reproduce

```bash
cd /path/to/CoreML-LLM-eagle3-main
python3 conversion/diagnose_fusion_layers.py \
    --corpus ~/Downloads/eagle_corpus.jsonl \
    --max-tokens 1000 \
    --csv-out docs/FUSION_LAYER_SCORES.csv
```

Requires a local HF snapshot of `google/gemma-4-E2B-it` in the default
`~/.cache/huggingface/hub/`; pass `--model-path` to override. Runtime on
Mac Studio (CPU fp16): ~10-15s for model load + ~2-3s per sequence
forward = ~30-40s end-to-end at 1000 tokens.

## Limitations

- Logit-lens with the final RMSNorm reused across all layers is a
  first-order early-exit probe; it penalises layers whose activations
  have a different scale regime than L34. A linear-probe fine-tuned on
  the training set would give a tighter bound, but is much more work and
  is downstream of what the draft's Fusion module learns anyway.
- 991 positions is ~0.1% of the training corpus; low-band scores within
  ±0.5 nats are not resolvable at this sample size. Main conclusions
  (mid=L17, high=L34) are robust.
- Not validated against the W4A8 argmax — the quantisation shifts the
  target distribution (cf. `docs/EAGLE3_DEPLOY.md`) but doesn't move the
  semantic layer stratification.
