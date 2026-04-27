# SESSION 2026-04-28 — Qwen3.5 0.8B + 2B → 2K context + Gemma 4 ANE recipe end-to-end

**Goal**: ship Qwen3.5-0.8B and -2B on iPhone 17 Pro with the Gemma 4 ANE
recipe (Conv2dLinear + ANERMSNorm + ane_softmax + repeat_kv_ane) at
MAX_SEQ=2048 in a single PR. World-fastest target.

**Status**: code changes done in this session. Re-conversion + iPhone bench
are user-side actions (Mac M-series + USB deploy). All edits land on
`fix/qwen35-mseq-reconcile` (PR title at open time should reflect the
broader scope, e.g., `feat(qwen3.5): v1.x — 2K + Gemma 4 ANE recipe`).

## What changed

### 1. MAX_SEQ → 2048 default
`conversion/test_qwen3_5_full_decode_trace.py:47` and
`conversion/test_qwen3_5_full_prefill_stateful.py:41` bumped 128 → 2048.
`conversion/build_qwen35_decode_int4.py:34` removed its hard-coded copy
and now imports MAX_SEQ from the trace module. All builders inherit.

KV state cost at 2048: ~22 MB extra on the 6 full_attention layers
(0.8B) / ~46 MB (2B). SSM state is O(1) regardless of seq.

### 2. Chunk-level Gemma 4 ANE recipe
`conversion/qwen3_5_decode_layer_ane.py` (NEW) exports
`ANEDecoderDecodeLayer` — a layer-type-aware (linear_attention or
full_attention) decode step using:

- `Conv2dLinear` (kernel=1) for every projection (in_proj_qkv/z/b/a +
  out_proj on SSM; q/k/v/o + gate/up/down on full_attn).
- `ANERMSNorm` (cat([x,-x]) → LayerNorm identity) for the layer-level
  input/post_attn norms and the per-head q_norm/k_norm. SSM's
  RMSNormGated stays in fp32 (SiLU z-gate doesn't fit the [x,-x] form).
- `ane_softmax` (max/sub/exp/sum/div fp16) for full_attention.
- `repeat_kv_ane` (reshape+repeat+view) instead of repeat_interleave.
- KV cache update via `where(range == position)` — scatter-free, ANE.

The monolithic version of this recipe was tried at 8909ecd / 907e029
and reverted twice (5be231b, f3ec1ef) because the 24-layer graph blew
the iOS 26.1 BNNS/ANEF compiler ceiling ("No space left on device").
Per-chunk graphs (6 layers each) fit comfortably; this is the same
escape hatch the VL 4B `_ane` variant used to get 11→31 tok/s on
Gemma 4 (`build_qwen3_vl_4b_text_decode_chunks_ane.py` header).

### 3. Unified chunked builder (drives 0.8B + 2B)
`conversion/build_qwen35_decode_chunks_ane.py` (NEW) replaces the prior
0.8B INT4 monolithic + 2B-specific chunk paths. 4 chunks (6 layers each)
+ embed sidecar + chunk_d carries final_norm + lm_head. Layout-compatible
with the existing Swift loader (`Qwen35Generator.swift` chunked path).

Usage:
```bash
# 0.8B (default model id, bundle name)
python conversion/build_qwen35_decode_chunks_ane.py \
    --out-dir /tmp/qwen35_0_8b_ane

# 2B
python conversion/build_qwen35_decode_chunks_ane.py \
    --out-dir /tmp/qwen35_2b_ane \
    --model-id Qwen/Qwen3.5-2B \
    --bundle-name qwen3_5_2b_decode_chunks
```

Outputs `<bundle-name>/{chunk_a..d}.mlpackage` + `embed_weight.bin` per
the v1.1.0 layout convention.

### 4. Swift loader: 0.8B + 2B chunked, mseq128 retired
`Examples/CoreMLLLMChat/CoreMLLLMChat/Qwen35Generator.swift`:
- `resolveChunkedDecodeURLs()` now tries
  `qwen3_5_0_8b_decode_chunks/` then `qwen3_5_2b_decode_chunks/`.
- Hidden size detected from chunk_a's `hidden_in` input spec (1024 vs
  2048); `reusableHidden` re-allocated to match. Replaces the
  hardcoded `2048` in `embedLookup`.
- Variant label derived from the resolved subdir name.
- mseq128 monolithic fallback paths retired — `loadDecodeOnly()` now
  throws if no chunked bundle is found.

`Sources/CoreMLLLM/ModelDownloader.swift`:
- `buildQwen35FileList()` now downloads the 4-chunk + embed bundle
  under `qwen3_5_0_8b_decode_chunks/` (was the mseq128 monolithic
  mlpackage).
- `localModelURL` (Qwen3.5 detection branch) tries both 0.8B and 2B
  chunked subdirs; mseq128 monolithic detection removed.

`Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift`:
- Both load detection and `modelName` mapping updated to recognize
  `qwen3_5_0_8b_decode_chunks/` and `qwen3_5_2b_decode_chunks/`.
- mseq128 monolithic detection removed.

`Examples/CoreMLLLMChat/CoreMLLLMChat/Qwen35DecodeBenchmark.swift`:
**Not yet updated** — still references `qwen3_5_0_8b_decode_fp16_mseq128`.
Bench will fail to load until either (a) the file is updated to use
chunked path, or (b) a parity-only fp16 monolithic build is rebuilt at
MAX_SEQ=2048. Defer to follow-up commit.

## Stretch items deferred (not in this PR)

- **In-graph TopK head**: split chunk_d into `chunk_d` (body only) +
  `chunk_head` (final_norm + Conv2d lm_head + topk[k=1]). Saves the
  ~1 MB ANE→Swift logits transfer per step. Implementation drafted in
  the original `ANEHeadChunk` code (now removed) — revive when Swift
  loader is taught to handle int32 `next_token` output.
- **IOSurface-backed inputs**: VL Phase 1 stretch memo says ≥40 tok/s
  needs IOSurface. Not implemented.
- **PREFILL_BYPASS env opt-in**: B1 PR #92 -43% TTFT pattern. Not
  ported to Qwen3.5.
- **Bench harness update** for the new chunked layout: keep existing
  bench; add an iPhone 17 Pro 2K bench protocol when artifacts ship.

## Re-conversion commands (run on Mac M-series with `lama-cml` env)

```bash
# Setup
source ~/lama-cml/bin/activate   # or whichever venv has coremltools 8.3
cd /Users/<you>/Downloads/CoreML-LLM

# 0.8B re-conversion (INT8)
python conversion/build_qwen35_decode_chunks_ane.py \
    --out-dir /tmp/qwen35_0_8b_ane \
    --nbits 8

# 2B re-conversion (INT8)
python conversion/build_qwen35_decode_chunks_ane.py \
    --out-dir /tmp/qwen35_2b_ane \
    --model-id Qwen/Qwen3.5-2B \
    --bundle-name qwen3_5_2b_decode_chunks \
    --nbits 8
```

Expected per chunk on Mac M4: ~5-15 min trace + convert, +1-3 min
palettize. Each builder prints ANE placement % (audit) at the end —
target ≥ 95% per chunk.

## HF re-upload (after conversion)

```bash
# 0.8B (replaces existing mseq128 artifacts on
# `mlboydaisuke/qwen3.5-0.8B-CoreML`)
huggingface-cli upload mlboydaisuke/qwen3.5-0.8B-CoreML \
    /tmp/qwen35_0_8b_ane/qwen3_5_0_8b_decode_chunks \
    qwen3_5_0_8b_decode_chunks \
    --repo-type model

# Then delete the retired mseq128 entries from the HF repo:
huggingface-cli delete-files mlboydaisuke/qwen3.5-0.8B-CoreML \
    qwen3_5_0_8b_decode_int8_mseq128.mlpackage \
    qwen3_5_0_8b_decode_fp16_mseq128.mlpackage \
    qwen3_5_0_8b_decode_argmax_fp16_mseq128.mlpackage

# 2B (replace v1.1.0 chunks with ANE-recipe re-conversion)
huggingface-cli upload mlboydaisuke/qwen3.5-2B-CoreML \
    /tmp/qwen35_2b_ane/qwen3_5_2b_decode_chunks \
    qwen3_5_2b_decode_chunks \
    --repo-type model
```

## iPhone bench protocol (after deploy)

```bash
# Sideload via devicectl (if not waiting for HF download in app)
xcrun devicectl device process launch ...  # use existing dev workflow

# In the app: Models → (Qwen3.5 0.8B or 2B) → tap to download / load
# Then in the chat or benchmark view, run:
#   - Prompt at 2048 tokens (long context)
#   - Measure decode tok/s, prefill TTFT, phys_footprint
```

## Mac M4 measurement (2026-04-28, this session)

Greedy decode, 3 prompts × 40 new tokens. Two builds tested:
- **stateless** — `qwen35_chunks_ane_parity.py` runs the explicit-I/O
  4-chunk path with `state_X_a` / `new_state_X_a` per layer.
- **MLState** — `qwen35_chunks_mlstate_parity.py` runs the
  `make_state()` + `slice_update` path (per VL Phase 1 recipe).

| Model | Build | Compute | Prefill (tok/s) | Decode (tok/s) | Output | Notes |
|-------|-------|---------|----------------|----------------|--------|-------|
| 0.8B | stateless | ANE | 29.4 / 32.9 / 33.0 | 33.1 | ✓ | fp16 100% / INT8 91% ANE |
| 0.8B | stateless | GPU | 26.2 / 27.7 | 27-28 | ✓ | |
| 0.8B | stateless | CPU | 23.3 / 34.0 / 34.8 | 34.7 | ✓ | M4 CPU is unusually fast |
| 0.8B | MLState   | GPU | 21.5 / 39.0 / 39.6 | **40.1** | ✓ | +21% over stateless ANE |
| 0.8B | MLState   | ANE | — | — | — | error 11 ANEProgramProcessRequestDirect (multi-StateType incompat suspected; 4D conv_state didn't help) |
| 0.8B | MLState   | CPU | — | 57-59 | ✗ broken | CoreML CPU runtime miscompiles slice_update on multi-state model |
| 0.8B | stateless | 2-chunk ANE | — | 33.1 | ✓ | dispatch overhead is negligible — chunk consolidation no-op |
| 2B    | stateless | ANE | 22.1 / 24.6 / 24.5 | 24.6 | ✓ | fp16 100% / INT8 91% ANE |

ANE placement per chunk (INT8 palettized, post-compile audit):
- chunk_a: 524/576 ANE (91.0%)  — body layers 0-5
- chunk_b: 529/579 ANE (91.4%)  — body layers 6-11
- chunk_c: 524/576 ANE (91.0%)  — body layers 12-17
- chunk_d: 537/589 ANE (91.2%)  — layers 18-23 + final_norm + lm_head

The non-ANE 8-9% are the const palettize ops + the fp32 logits cast at
chunk_d's tail. All compute ops land on ANE.

Quality (greedy, no sampling):
- English fact prompt: "Paris" / "Berlin" / "Rome" — correct.
- Japanese recipe (餃子): coherent recipe text, both 0.8B (Japanese)
  and 2B (English `<think>` block then Japanese).
- Japanese short greeting (こんにちは): repetition loop (known v1.0.3
  greedy issue, sampling required for coherent output).

Bundle sizes:
- 0.8B: 4× INT8 chunks (126 + 123 + 126 + 377 MB) + 509 MB embed
  = 1.26 GB total.
- 2B: 4× INT8 chunks (347 + 340 + 347 + 849 MB) + 1017 MB embed
  = 2.90 GB total.

## iPhone projection

The MLState path moves the headline number from 33.1 (stateless ANE) to
**40.1 tok/s (MLState GPU)** on Mac M4. ANE remains blocked behind a
multi-StateType iOS 18 issue (3 separate state types — kv_cache,
conv_state, rec_state — interact with the slice_update lowering in a
way that CPU/ANE runtime can't execute; GPU runtime is fine).

Mac M4 / iPhone GPU ratio is more stable than ANE (~1.3-1.4× per the
Gemma 4 vision A19 retest memo). Naively scaled iPhone projections:
- **0.8B MLState GPU @ 2K: ~30-35 tok/s on iPhone 17 Pro**, vs
  v1.0.3 monolithic GPU 27.7 tok/s @ 128 — beats current ship.
- 0.8B stateless ANE @ 2K: ~22 tok/s flat vs v1.0.3 ANE.
- 2B  MLState GPU @ 2K: pending bench.
- 2B  stateless ANE @ 2K: ~17 tok/s flat vs v1.1.0.

Win conditions to validate on iPhone 17 Pro:
- 0.8B MLState GPU: ≥ 30 tok/s (1.4× over v1.0.3 ANE 22) — should be
  the easiest win; just deploy and measure.
- 2B  MLState GPU: ≥ 22 tok/s (1.3× over v1.1.0 ANE 17).

If MLState ANE on iPhone also fails with error 11, GPU is the ship
default. If it works (iPhone A18 ANE may handle multi-state where M4
ANE doesn't), even better — projected iPhone MLState ANE ≈ 27 tok/s
via the same Mac/iPhone ratio applied to the 40 tok/s GPU number.

The Mac numbers do not yet prove the recipe wins on iPhone for ANE,
but **MLState GPU is a real speedup** that ports cleanly via the Swift
loader update.

## Files touched

```
conversion/test_qwen3_5_full_decode_trace.py        # MAX_SEQ 128→2048
conversion/test_qwen3_5_full_prefill_stateful.py    # MAX_SEQ 128→2048
conversion/build_qwen35_decode_int4.py              # import MAX_SEQ from trace
conversion/qwen3_5_decode_layer_ane.py              # NEW — ANE decode layer
conversion/build_qwen35_decode_chunks_ane.py        # NEW — chunked builder
Examples/CoreMLLLMChat/CoreMLLLMChat/Qwen35Generator.swift  # 0.8B+2B detection
Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift        # detection
Sources/CoreMLLLM/ModelDownloader.swift             # 0.8B chunked download
docs/SESSION_2026_04_28_QWEN35_2K_ANE.md            # this file
```

## Follow-up work (next PR)

1. Update `Qwen35DecodeBenchmark.swift` to use chunked path.
2. Update `docs/QWEN35_IPHONE_BENCH.md` 2K bench protocol + LiteRT-LM /
   llama.cpp / MLC comparison table.
3. In-graph TopK head (`chunk_head` split + Swift int32 `next_token`).
4. IOSurface inputs + PREFILL_BYPASS opt-in (stretch ≥40 tok/s).
5. Append iPhone 17 Pro measured numbers to `QWEN35_LESSONS.md`.
