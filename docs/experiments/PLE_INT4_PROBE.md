# PLE INT4 quantization probe

**Date:** 2026-04-26
**Branch:** `research/litert-lm-transfer` (follow-up to `LITERT_LM_TRANSFER_ANALYSIS.md`)
**Question:** Can the 2.20 GB PLE table (`embed_tokens_per_layer_q8.bin`,
the largest single file in the bundle) be re-quantized from INT8 to INT4
without quality regression vs the current W4A16 production point?

## TL;DR

- **Strict lossless: NO.** Production INT8 row-wise quantization is
  effectively lossless against BF16 ground truth (mean cos
  **0.999937**, min 0.999291). INT4 cannot match that at any
  group size we tested.
- **End-to-end test contradicts the per-row optimism.** Multi-step
  prefill on 4 prompts × 16 positions: even at INT4 group=8 (the
  finest group we tested), **token argmax flips ~35% of the time** vs
  the production INT8 PLE. Smaller groups improve cos sim at the
  embedding step but token agreement plateaus around 65% — meaning
  the PLE perturbation compounds through 35 transformer layers more
  aggressively than the 0.005 input-cos-drop suggests.
- **INT4 PLE is NOT a viable PTQ drop-in.** Saving ~980 MB this way
  trades the largest single bundle-size win against ~35% token
  divergence in real prefill, and that ratio doesn't change with
  finer group sizes (g=32, 16, 8 all sit in the 64-67% agreement
  band).
- **Paths forward that would actually work** (none of them "quick"):
  (a) QAT — train the model to be robust to INT4 PLE, ~A100 + 3-5 d.
  (b) LUT palettization (16-centroid kmeans per group) — more
      expressive than absmax-symmetric, may close some of the gap
      but unlikely to recover the full 35% argmax delta on its own.
  (c) Vocab trim — drop the ~50K tail vocab entries that contribute
      most of the PLE byte mass and rarely appear, saves storage
      orthogonally to quant scheme.
  (d) Tied-weight dedup (embedding ↔ lm_head, see §1 of the parent
      transfer doc) — saves 192 MB without touching PLE.

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

## End-to-end measurement (the gating test)

Two scripts run the chunk_1→chunk_2→chunk_3→chunk_4 stateful chain
twice per test (INT8 PLE vs INT4-grouped PLE), with the token
embedding identical between runs so the only delta is the PLE input.

### Test 1: fresh-state pos=0, 32 vocab tokens
`scripts/probe_ple_int4_e2e.py` — 16 random + 16 worst-case (lowest
PLE cos sim under INT4-g32) vocab IDs, each at position 0 with fresh
MLState. INT4 g=32:

| metric | value |
|---|---:|
| mean cos(per_layer_combined_out) | 0.996804 |
| mean cos(hidden_states from chunk_4) | **0.943404** |
| min cos(hidden_states from chunk_4) | 0.775465 |
| **token_id argmax agreement** | **14/32 (43.8%)** |

### Test 2: real prefill, 4 prompts × 16 positions, multiple group sizes
`scripts/probe_ple_int4_e2e_prefill.py 32,16,8` — short token-ID
prefill sequences (e.g. tokens 2..17, 40..55, 100..115, 200..215),
T=1 prefill at each position, comparing both runs' chunk_4 outputs:

| group | mean cos(plc) | mean cos(h4) | min cos(h4) | tok agreement |
|---:|---:|---:|---:|---|
| **32** | 0.996998 | 0.929457 | 0.735551 | 42/64 (**65.6%**) |
| 16 | 0.997782 | 0.954497 | 0.794194 | 43/64 (67.2%) |
| 8 | 0.998470 | 0.961482 | 0.824102 | 41/64 (64.1%) |

Token agreement **plateaus at ~65%** across group sizes, even though
input cos sim and h4 cos sim both improve monotonically as groups
shrink. The argmax over 262K vocab tokens is sensitive enough that
even a residual stream cos sim of 0.96 produces frequent argmax
flips.

### What this tells us

The PLE input perturbation amplifies through the 35-layer transformer
faster than per-row cos numbers suggested:

- input PLE row cos sim: 0.995 (g=32)
- chunk_1 plc output cos sim: 0.997 — slight smoothing from the
  per-layer gate + projection
- chunk_4 h4 cos sim: **0.93** — significant amplification
- argmax token agreement: **65%** — argmax instability in a 262K-vocab
  space

Smaller groups (g=16, g=8) reduce the input perturbation but don't
break out of the 65% argmax-agreement ceiling. This is consistent
with the PLE's role as a *per-layer additive contribution to the
residual stream*: the same tokenwise perturbation enters at every
one of the 35 layers, so the compounding is structurally larger than
"one perturbed input × one model".

## Recommendation: NOT viable as a PTQ drop-in

The original recommendation (INT4 g=32 + Swift dequant path, 1-2 day
engineering, "low risk") is **withdrawn**. The empirical e2e numbers
say it isn't ship-quality.

If saving ~980 MB on the PLE table is a goal, the realistic options
require investment beyond a runtime port:

1. **Tied-weight dedup (the actual quick win — embedding ↔ lm_head)**.
   Saves 192 MB at zero quality cost. ~1-3 days. See parent doc §0
   item 1.
2. **Vocab pruning**. Drop the rare ~50K tail vocab entries (262144 →
   ~210K). Saves storage proportionally on PLE, embedding, and lm_head
   simultaneously. Quality cost: depends on vocab usage frequency in
   real workloads.
3. **QAT for INT4 PLE**. Train the model to absorb INT4 PLE
   perturbations. A100 + 3-5 days, mirrors the QAT path called out
   in `STAGE1_W4A8_FINAL.md` for activation quant.
4. **Don't try INT4 PLE.** Accept the 2.20 GB cost. The remaining
   bundle delta against LiteRT-LM (1.50 GB) is mostly PLE; if we're
   not going to attack PLE, accept being 1.68× LiteRT bundle size.

The cheapest-to-most-impactful order is: 1 → 2 → 4 → 3.

## Where this leaves the broader picture

After this probe (INT4 PLE confirmed not viable as drop-in):
- Bundle stays at **3.71 GB**. INT4 PLE shaves ~980 MB on disk but
  flips ~35% of token argmaxes vs INT8 — not shippable.
- The realistic quick-win path is tied-weight dedup (embedding ↔
  lm_head, -192 MB, zero-quality-cost). Bundle floor without QAT or
  vocab trim: **3.52 GB** (1.59× LiteRT).
- Closing the rest of the LiteRT gap requires architecture work
  (QAT for INT4 PLE) or vocab pruning.

What we learned about LiteRT-LM specifically: their 1.28 GB PLE table
must be using either (a) an INT4 LUT/palettization scheme that
preserves more directional information than absmax-symmetric scaling,
(b) QAT-trained INT4-friendly weights, or (c) a smaller PLE matrix
(less per-layer dim, smaller vocab subset). Without a source-read of
their `.litertlm` Embedder section internals, we can't tell which.
Source comparison would be a separate research item.

## Reproduce

```bash
python3.12 scripts/probe_ple_int4.py
# ~90s, peak ~25 GB RAM
```

The script is self-contained — it reads the HF safetensors at the
hardcoded path and the production INT8 sidecar, no env / args.
