# Experiments — What We Tried, What Shipped, What Didn't

This is the decision log for speed / quality experiments layered on top of the baseline SWA 4-chunk pipeline. Anything *not* in `conversion/convert.py`'s shipping path lives here.

Legend:

- **Shipping** — in `convert.py` or a default path, used by pre-converted HuggingFace models.
- **Prototype** — runs end-to-end, numerically validated, not yet plumbed into the default pipeline or the Swift runtime.
- **Shelved** — implemented far enough to judge, then rejected. Reason recorded.
- **Planned** — designed but not implemented.

See `docs/SPEED_8K.md` for the overall speed roadmap and tier assignments.

---

## Attention-variant experiments (8K context)

### WFA — Windowed Full Attention  —  Shelved

- Files: `conversion/build_wfa.py`, `conversion/models/gemma4_swa_wfa.py`, `conversion/benchmark_wfa.py`
- Idea: Cap the 7 full-attention layers' KV at a fixed window `FW` (e.g. 2048) instead of the full context. Same shift-based update as the 28 sliding layers, so every layer is O(W) per step.
- Build speed: at `FW=2048`, decode throughput matches the 2K baseline (~31 tok/s) even at 8K context. Bandwidth cost is the bottleneck the knob hits.
- **Reason shelved**: recall drops on prompts that need attention to tokens beyond `FW`. The quality regression on long-context tasks is the exact class of failure this project refuses to absorb silently. See `docs/SPEED_8K.md §0` ("naive WFA quality NG past FW"). Kept around because DuoAttention-style per-head windowing (Tier A) reuses the same shift-cache machinery.

### Flash Decoding  —  Prototype

- Files: `conversion/build_flash.py`, `conversion/models/gemma4_swa_flash.py`
- Idea: Split `Q @ K^T` along the K dimension into fixed-size chunks (default 1024) and recombine with an online softmax. Mathematically identical to standard attention within fp16 rounding (cosine > 0.999999 vs standard attention on test tensors).
- Status: builder works, outputs validated. Not integrated into the Swift runtime because the expected win on ANE is small — ANE is not SRAM-bound for 8K × 512 K/V (1 MB per chunk fits easily in 32 MB SRAM), so tiling buys less than it does on GPU. Keep for future much-longer-context variants.

### Merged chunk2+chunk3  —  Prototype

- File: `conversion/models/gemma4_swa_merged.py`
- Idea: Merge chunks 2 and 3 (L8–24) so `kv13` / `kv14` are produced and consumed on the ANE side without a round-trip to the Swift runtime. Saves 32 MB of fp16 KV I/O per decode step.
- Status: builder works, not yet benched on-device. Trade-off vs current 4-chunk split: the merged chunk is larger, which may push the ANE compiler past the ~15-layers-per-chunk stability line we hit elsewhere. Measure before shipping.

### Lite 2-chunk variant  —  Prototype

- Files: `conversion/models/gemma4_lite_chunks.py`, `conversion/models/gemma4_lite_wrapper.py`
- Idea: Two chunks (embedding + L0–14) and (L15–34 + LM head). Fewer chunks = less per-step overhead, but risks the ANE compiler stability ceiling.
- Status: kept as a fallback in case the 4-chunk split has problems on a new OS release. Not the default.

### Stateless 4-chunk (no MLState)  —  Shipping

- File: `conversion/models/gemma4_stateless_chunks.py`
- Idea: Explicit KV input/output tensors instead of Apple's `MLState` API. Chosen over the Monolithic/`MLState` path because `MLState` introduces int64 state indices that break ANE placement on the model sizes we ship (see `docs/CONVERSION.md` on "Explicit KV I/O").
- This is what the default Gemma 4 E2B conversion actually produces.

---

## Quantization experiments

### INT4 palettization, group_size=32  —  Shipping

- File: `conversion/exporter.py :: _quantize_model`
- Chosen for weight-only compression. See `docs/CONVERSION.md` "Quantization" section for the size/quality/latency trade-off and why not INT8 or FP16.

### INT8 weight-only  —  Prototype

- File: `conversion/build_w8a8.py` (mode `w8`)
- Idea: symmetric per-channel INT8. Smaller than FP16, larger than INT4. Zero latency gain on ANE (ANE is FP16 internally, so weight-only INT8 gives size only — same as INT4 but at double the storage).
- Status: shelved as a *shipping* option for size reasons, kept as a baseline for measuring INT4's quality cost. Not useful standalone.

### W8A8 (naive calibration)  —  Shelved

- File: `conversion/build_w8a8.py` (mode `w8a8`)
- Idea: activation + weight INT8. This is the only quantization that unlocks ANE's INT8×INT8 compute path (~1.3–1.6× per Apple's ResNet-50 docs).
- Calibration used 5 random samples. Quality regressed visibly on chat outputs.
- **Reason shelved**: insufficient calibration. Superseded by `build_w8a8_proper.py`.

### W8A8 (realistic calibration)  —  Shelved 2026-04-13 (ANE incompatibility)

- File: `conversion/build_w8a8_proper.py`
- Idea: collect real activation traces by running the INT4 model on 32+ prompts at positions 0..31, then quantize. Also provides a W4A8 fallback (INT4 palette weights + INT8 activations).
- **Dead-end**: `coremltools.experimental.linear_quantize_activations` inserts quantize / dequantize MIL ops that the **iPhone ANE compiler refuses**. This is not a calibration or file-mixing issue — the op set itself is unsupported on ANE on the A17/A18/A19 Pro generation as of 2026-04. Mac Studio M4 Max independently measures 0% speedup (no INT8-INT8 fast path on M4, different hardware story but same conclusion for us).
- The 1.3-1.6× number in Apple's ResNet50 doc refers to a vision-model ANE path that doesn't generalize to transformer decoder graphs produced by coremltools.
- Scripts are kept for historical reference and in case a future coremltools release adds ANE support for these ops. Do **not** reintroduce W8A8 into the default stack without re-verifying the op support situation.

### INT8 KV cache  —  Shelved (same reason as W8A8)

- The pathway of quantize-on-write / dequantize-on-read adds the same coremltools quant/dequant MIL ops that the iPhone ANE compiler rejects.
- Even the naive "store INT8, convert to FP16 in Swift before the prediction call" variant shows 0% wall-clock gain — the attention op still runs in FP16 and CoreML dequantizes the I/O on each call.
- Neither path is worth reviving on the current ANE stack. 8K decode speed has to come from Q-batching, DuoAttention, EAGLE-3, and the TriForce/Cascading sparse-attention variants instead.

---

## Speculative decoding experiments

### EAGLE-3  —  In training

- Files: `conversion/collect_eagle_hidden_states.py`, `conversion/download_eagle_corpus.py`, `conversion/train_eagle_draft.ipynb`, `conversion/train_eagle3_draft.ipynb` (notebooks are untracked while actively iterated)
- Idea: train a small decoder-layer draft model on Gemma 4 E2B hidden states; verify in-graph against the target.
- Training corpus: WikiText + C4 + Alpaca + Dolly + CodeAlpaca + UltraChat, formatted with Gemma 4's chat template. ~50 k samples.
- Current acceptance metric (acc0) tracked in notebook; see `docs/SPEED_8K.md §3 P1` for the latest snapshot (dated, not live).
- Integration path: once trained, draft model → `build_speculative.py` → paired with verify chunks.

### Medusa (3 heads)  —  Shelved

- Files: `conversion/train_medusa_heads.py`, parts of `conversion/build_speculative.py`
- Idea: 3 lightweight ResBlocks predict the next 3 tokens from the final hidden state; verify with target model.
- **Reason shelved**: published Medusa acceptance on Gemma-class models is ~1.3 %, far below EAGLE-3's 50–70 %. Confirmed on a small internal run. Kept because `build_speculative.py` reuses the same verify-chunk plumbing for EAGLE-3.

### TriForce / Quest (sparse KV retrieval)  —  Planned

- See `docs/SPEED_8K.md §1 A3 / A4`. Requires block-static top-k redesign to stay on ANE. Not started.

### DuoAttention (retrieval vs streaming heads)  —  Planned

- See `docs/SPEED_8K.md §1 A1 / §3 P3`. Offline head classification + two KV banks per layer. High ROI, training-free at inference time. Next candidate after EAGLE-3 lands.

---

## Vocabulary pruning  —  Abandoned

- File: `conversion/prune_vocab.py`
- Idea: the Gemma 4 vocab is 262 K tokens. Embedding + LM-head weights dominate on-device size. Analysis shows large blocks (rare scripts, emoji variants) are near-never used for English chat.
- **Reason abandoned**: (1) Gemma's tokenizer is sentencepiece — dropping tokens changes BPE merges and breaks round-trip tokenization; (2) the `gemma4_lite_wrapper.py` route (external per-layer embedding in Swift) already reclaimed the main memory win (~40 %). Keep the analysis as reference.

---

## 8K chunk4 rebuild  —  Shipping (maintenance fix)

- File: `conversion/rebuild_chunk4_8k.py`
- Purpose: regenerate `chunk4` with `causal_mask_full` at `(1,1,1,8192)` when the earlier build shipped with a 2048-sized mask. This is the kind of silent mismatch the `ChunkedEngine` auto-detection (see commit `4311991`) now guards against at load time.

---

## How to add a new experiment

1. Pick a short name (e.g. `wfa`, `flash`, `merged`).
2. Put the model variant in `conversion/models/gemma4_<name>.py`.
3. Put the builder in `conversion/build_<name>.py`. Keep it runnable standalone (`python build_<name>.py --output ./output/<name>`).
4. Put the A/B benchmark in `conversion/benchmark_<name>.py` comparing against the shipping baseline on the same prompts.
5. Write one row in this file: what you tried, status after the first real measurement, reason if shelved.

Rule: **every experiment that gets shelved gets a one-paragraph obituary here with the numeric reason**. Future-us will otherwise rebuild the same thing.
