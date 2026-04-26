# PLE INT4 quantization probe

**Date:** 2026-04-26
**Branch:** `research/litert-lm-transfer` (follow-up to `LITERT_LM_TRANSFER_ANALYSIS.md`)
**Question:** Can the 2.20 GB PLE table (`embed_tokens_per_layer_q8.bin`,
the largest single file in the bundle) be re-quantized from INT8 to INT4
without quality regression vs the current W4A16 production point?

## TL;DR

- **Strict lossless: NO.** Production INT8 row-wise quantization is
  effectively lossless against BF16 ground truth (mean cos
  **0.999937**, min 0.999291). INT4 cannot match that.
- **Practical "ship-quality": likely YES at group=32**, mean cos
  **0.994948**, min **0.987177** vs BF16. -0.005 cos vs INT8 baseline.
  Saves **~980 MB on disk** (PLE 2.20 GB → 1.26 GB).
- **Row-wise INT4 is too aggressive.** Min cos drops to **0.823137** —
  ~1 in 100K vocab tokens get hit hard. Pick a grouped scheme.
- **End-to-end validation still required** before shipping. The PLE-step
  cos sim doesn't tell us how the error compounds through 35 transformer
  layers + lm_head onto the final logit distribution. The W4 decoder
  noise floor is already cos 0.949 vs FP16 — there's headroom, but
  empirical e2e measurement is the gating step.

## Method

`scripts/probe_ple_int4.py`. Loads the BF16 reference from
`output/gemma4-e2b/hf_model/model.safetensors` key
`model.language_model.embed_tokens_per_layer.weight` (shape
`[262144, 8960]` = vocab × num_layers × per_layer_dim = 262144 × 35 × 256),
plus the production INT8 + per-row FP16 scales from
`embed_tokens_per_layer_q8.bin` / `embed_tokens_per_layer_scales.bin`.

Runs:
1. **Production INT8 vs BF16** — baseline (the on-disk format we'd be
   replacing). Per-row symmetric INT8 with FP16 scale, dequant
   `out = int8 × (scale_fp16 / 127.0)`.
2. **Synthetic INT8 row-wise vs BF16** — methodology sanity check.
   Result matches production exactly → confirms the production recipe.
3. **INT4 row-wise vs BF16** — simplest INT4 replacement.
4. **INT4 grouped (g=128, 64, 32) vs BF16** — finer-grained scaling.
5. **INT4 vs production INT8** — practical "vs deployable baseline" view.

Per-row cosine similarity, computed in fp64 over chunks of 16384 rows for
memory efficiency. ~89 s on Mac Studio, peak RAM ~25 GB.

## Results

### Quality (cosine similarity per row)

| recipe | mean | min | p50 | p99 | p99.9 |
|---|---:|---:|---:|---:|---:|
| **production INT8 vs BF16** | **0.999937** | 0.999291 | 0.999942 | 0.999965 | 0.999968 |
| synth INT8 row-wise vs BF16 | 0.999937 | 0.999291 | 0.999942 | 0.999965 | 0.999968 |
| INT4 row-wise vs BF16 | 0.979847 | **0.823137** | 0.981552 | 0.988569 | 0.989737 |
| INT4 group=128 vs BF16 | 0.992241 | 0.968021 | 0.992243 | 0.993376 | 0.993639 |
| INT4 group=64 vs BF16 | 0.993637 | 0.978756 | 0.993632 | 0.994387 | 0.994572 |
| **INT4 group=32 vs BF16** | **0.994948** | **0.987177** | 0.994943 | 0.995417 | 0.995541 |

Vs the production INT8 baseline (the deployable replacement question):

| recipe | mean | min |
|---|---:|---:|
| INT4 row-wise vs prod INT8 | 0.979754 | 0.822148 |
| INT4 group=128 vs prod INT8 | 0.992178 | 0.967744 |
| INT4 group=64 vs prod INT8 | 0.993574 | 0.978431 |
| INT4 group=32 vs prod INT8 | 0.994885 | 0.986880 |

(The INT4 vs BF16 and INT4 vs INT8 columns are nearly identical because
INT8 itself is so close to BF16 that the comparison reduces to INT4 vs
BF16.)

### Storage (vocab × hidden, no group overhead unless noted)

| layout | size on disk |
|---|---:|
| BF16 reference | 4480 MB |
| **prod INT8 + row scales (today)** | **2240 MB** |
| INT4 row-wise + row scales | 1120 MB (-1120 vs INT8) |
| INT4 group=128 + scales | 1155 MB (-1085 vs INT8) |
| INT4 group=64 + scales | 1190 MB (-1050 vs INT8) |
| **INT4 group=32 + scales** | **1260 MB (-980 vs INT8)** |

The g=32 metadata overhead is ~140 MB (8960/32 = 280 groups per row × 2 B
fp16 scale × 262144 vocab) — small relative to the saving.

## Interpretation

1. **Production INT8 PLE is essentially lossless.** mean cos 0.999937
   means INT8 is contributing zero detectable error to the embedding
   path. Anything that gives up here is a regression.

2. **INT4 row-wise has unacceptable tail behaviour.** min cos 0.823 means
   at least some vocab tokens take a ≥ 0.18 cos hit on the PLE
   contribution alone. Even if mean is fine, those tokens probably
   include real, frequent ones (vocab is 262144; the first ~50K are the
   common tokens). Pick a grouped scheme.

3. **g=32 is the quality sweet spot.** mean 0.99495 / min 0.98718 vs
   BF16 — comparable to typical "lossless INT4 quant for general weights"
   in the published literature (AWQ, GPTQ, OmniQuant all report ~0.99-0.999
   per-row cos at g=128 on attention/MLP weights).

4. **g=32 still adds ~0.005 cos error vs the deployed INT8 baseline.**
   Whether that propagates through the model is an empirical question.
   Two reasons it should be safe in our specific stack:
   - The W4 LUT decoder already operates at cos 0.949 vs FP16 — a 0.005
     extra error at the embedding step would nudge that into ~0.945,
     within the noise of our 32-prompt cos-sim measurement.
   - PLE feeds into the residual stream via `per_layer_projection`
     (1536x256 matmul, also W4-quantized in chunk_1) + `per_layer_input_gate`
     + `post_per_layer_input_norm` — three lossy operations downstream of
     PLE will dilute small input perturbations.

5. **End-to-end logits cos sim is the gating measurement.** This probe
   answers "what does INT4 do at the embedding lookup step?". It does
   not answer "what does INT4 do to the output token distribution?".
   That's the next experiment.

## Recommendation

A follow-up branch could:

1. Generate `embed_tokens_per_layer_q4_g32.bin` + grouped scales
   (~1.26 GB on disk).
2. Add an `EmbeddingLookup`-style INT4-grouped path in Swift
   (`Data(...,options:.mappedIfSafe)` + group-aware dequant). The
   per-token cost is essentially unchanged from the current INT8 path
   — just smaller LUT to read.
3. Run the existing `probe_w4a8_quality.py`-style 32-prompt cos sim
   measurement against the FP16 baseline, comparing prod (W4A16 + INT8
   PLE) vs candidate (W4A16 + INT4-g32 PLE). Gate at cos ≥ 0.945.
4. If gate passes, ship behind a `LLM_PLE_INT4=1` sideload flag for an
   iPhone session, then default-on if confirmed.

Engineering: ~1-2 days.
Bundle saving: ~980 MB (3.71 GB → 2.73 GB).
Risk: low — falls within the W4A16 noise floor.

This is the largest single bundle-size win achievable on this stack
without changing the model architecture.

## Where this leaves the broader picture

After this branch:
- Bundle 3.71 GB. After hypothetical INT4-g32 PLE: **2.73 GB**.
- LiteRT-LM-equivalent: 2.21 GB. Remaining gap: ~0.52 GB, almost
  entirely the embedding (384 → 104 MB) and a slightly tighter decoder
  quant scheme (1.09 → 0.82 GB).
- Closing the embedding gap requires either tied-weight dedup with
  lm_head (-192 MB at most) or INT4 embedding with group quantization
  (-192 MB). Both need their own quality probe.

So: **INT4 PLE g=32 + tied embedding dedup, both validated, gets us
within ~330 MB of LiteRT.** Beyond that requires graph-level
restructuring.

## Reproduce

```bash
python3.12 scripts/probe_ple_int4.py
# ~90s, peak ~25 GB RAM
```

The script is self-contained — it reads the HF safetensors at the
hardcoded path and the production INT8 sidecar, no env / args.
