# EAGLE-3 `e_next` Embed Parity — Track 7

**Date:** 2026-04-20
**Question:** Does the trainer's `e_next` (fp16 lm_head lookup) match the
deployed draft's `e_next` (int8-dequantized table), or does int8 quantization
produce OOD inputs to the draft's attention at inference?

## Trainer path (what `e_next` is during Colab training)

- `conversion/train_eagle3_ttt.py:282` — sets `embed_table = lm_head_t` where
  `lm_head_t` is the fp16 lm_head buffer loaded from
  `--lm-head lm_head_weight.bin`. Tied-embedding Gemma 4 shares weights, so
  this is semantically the token embedding table.
- `conversion/train_eagle3_ttt.py:374-375` — per-step `e_next` construction:
  `e_next = (embed_table[tok_for_e].unsqueeze(1).float() * float(cfg["embed_scale"]))`
  i.e. `fp16_tbl[tok] * 39.1918...` (no quantization anywhere).
- The PyTorch sanity check in `conversion/test_eagle3_infer.py:193,199` uses
  `target.get_input_embeddings()` (HF fp16 table) × `embed_scale` — same
  numeric path, different source for `embed_table` (HF vs dumped lm_head).

## Inference path (what `e_next` is on iPhone ANE)

- `Sources/CoreMLLLM/CoreMLLLM.swift:806` — `drawBurst` is invoked with
  `tokenEmbed: { try engine.embedToken($0) }`.
- `Sources/CoreMLLLM/ChunkedEngine.swift:1411-1413` — `embedToken` forwards
  to `embedTokens.lookup(tokenID, shape: [1,1,hidden])`.
- `Sources/CoreMLLLM/EmbeddingLookup.swift:35-58` — dequantization formula:
  `f16_out[i] = int8[tok,i] * (fp16_scale[tok] / 127.0) * embedScale`
  where `embedScale = config.embedScale = 39.1918...` (ModelConfig.swift:39,
  loaded from `model_config.json`).
- `Sources/CoreMLLLM/SpeculativeLoop.swift:182,215` — this dequantized array
  is fed directly as the draft's `e_next` input and reused after each draft
  step with the draft's own predicted token.

## Numerical comparison (n=10000 tokens, seed=0)

| metric                    | mean       | min        | max        |
|---------------------------|-----------:|-----------:|-----------:|
| cos_sim(train, device)    | 0.999954   | 0.998866   | 0.999979   |
| abs_err per element       | 9.49e-03   | 6.04e-03   | 4.29e-02   |
| abs_err per row (max)     | 1.99e-02   | 1.29e-02   | 8.61e-02   |
| rel_err (row_max/\|train\|) | 4.29e-04   | 3.03e-04   | 2.10e-03   |
| \|e_train\| (L2 norm)     | 46.78      | 36.98      | 60.56      |
| \|e_device\| (L2 norm)    | 46.79      | 37.00      | 60.59      |

Rows with `cos_sim < 0.998`: **0 / 10000**. Max relative error is **0.21%**
(row max absolute / row L2 norm), i.e. the int8-dequantized embedding is
within 0.21% of the fp16 ground truth on the worst-case token in the sample.

## Verdict: **NO DRIFT**

The int8 per-token-scaled embed table is numerically equivalent to the fp16
table within fp16 quantization noise (cos ≥ 0.9989 on every sampled row,
mean cos = 0.99995). The draft's attention at inference sees essentially
the same `e_next` distribution it saw during training.

**This is NOT the source of the accept-rate gap.** The trainer does not
need to be fixed to use the int8-dequant table; switching `embed_table` to
the dequantized int8 tensor in the trainer would change `e_next` by ≤0.21%
per token, far below any signal the draft's attention can meaningfully
learn from.

### Why the parity is this tight

Per-row fp16 scales (not per-tensor) + `/127.0` int8 mapping capture ≈8
bits of per-row precision on a tensor that is itself fp16 (≈10-bit
mantissa). Residual error is int8 round-off on small entries, which
cancels on the L2 norm and contributes negligibly to cosine similarity.

### Where to look next (Track 7 follow-ups)

Given this result, the train↔inference drift investigation should move on:

1. **Hidden tap parity** — compare `h_low/h_mid/h_high` from
   `collect_eagle_hidden_states_w4a8.py` vs iPhone-runtime's
   `lastHiddenAtL{8,17,34}` taps on the same stream.
2. **Fusion parity** — `eagle3_fusion.mlpackage` INT4 palettized fusion
   Linear vs trainer's fp32 `FeatureFusion.proj` (build_eagle3.py:229-235).
3. **Draft palettization** — `build_eagle3.py:340-346` palettizes draft
   weights INT4/group=32. Compare PyTorch vs mlpackage logits on a fixed
   `(h_prev, e_next)` stream.

## Reproducing

```bash
python conversion/diagnose_embed_parity.py \
  --int8   ~/Downloads/gemma4-e2b-eagle3-sideload/embed_tokens_q8.bin \
  --scales ~/Downloads/gemma4-e2b-eagle3-sideload/embed_tokens_scales.bin \
  --fp16   ~/Downloads/lm_head_weight.bin \
  --n-sample 10000 --seed 0
```

Uses only the on-disk sideload assets and the existing fp16
`lm_head_weight.bin` dump (~/Downloads/lm_head_weight.bin, 805 MB,
(262144, 1536) fp16). No HF download required.
