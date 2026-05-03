# Gemma 4 E4B multimodal CoreML — build & sideload guide

**Status:** Validated 2026-05-03 on iPhone 17 Pro. Text 15.7 tok/s + image / video / audio all functional.

**Working bundle:** `gemma4-e4b-multimodal` (~7.6 GB).

---

## Bundle topology (what works on iPhone)

| Component | File(s) | Source |
|---|---|---|
| Decode (Topology II 3-chunk) | `chunk1` (legacy) + `chunk2_3way` + `chunk3_3way` | legacy E4B HF + `build_gemma4_3way.py --model gemma4-e4b` |
| Prefill (multifunction `prefill_b8`) | `chunk1` / `chunk2` / `chunk3` / `chunk4` (legacy) | legacy E4B HF (`mlboydaisuke/gemma-4-E4B-coreml`) |
| Vision encoder | `vision.ane.mlmodelc` (output `[1, 256, 2560]`) | `convert_gemma4_multimodal.py --vision-ane --model-path <E4B HF>` |
| Audio encoder | `audio.mlmodelc` (output `[1, 50, 1024]`) | `convert_audio.py --model-path <E4B HF>` |
| Audio projection | `output_proj_*.npy` (1024→1536) + `embed_proj_weight.npy` (1536→2560) | from `convert_audio.py` |
| Text sidecars | `embed_tokens_*`, RoPE tables, `model_config.json`, `hf_model/` | legacy E4B HF |

`AudioProcessor.swift` `projectHiddenStates` runs the two-stage projection in Swift/Accelerate. `embed_proj` is now non-square aware (E4B `(2560, 1536)` vs E2B `(1536, 1536)`).

---

## What was tried and rejected

### `prefill_chunk{1..4}.mlmodelc` separate-file multifunction (T=64/128/256/512)

Built via `build_prefill_multifunction.py` (the production E2B `gemma4e2b3way` path).

- **Mac**: works fine, 16.5 tok/s text + correct multimodal.
- **iPhone**: text and image/audio prompts both produce degenerate output (e.g. `こんにちは` → `こんにちは。\n(同じトーンで)\nこんにちは。`).
- Likely cause: int4 quantization noise on iPhone ANE 18 + E4B-specific graph (HKV=2, 21 merged layers in `chunk2_3way`) tips greedy argmax into a degenerate loop. E2B ships the same multifunction layout and works on iPhone.
- Engine code is unchanged; the bundle ships **without `prefill_chunk*`** so the umbrella engine falls back to legacy `prefill_b8` multifunction. Vision-aware bidirectional mask within the image span still functions through `fillBatchMasksVisionAware` in `ChunkedEngine.swift`.

### Stateful (MLState) E4B multimodal

Engine class `Sources/CoreMLLLM/Gemma4StatefulMultimodalEngine.swift` builds and runs on Mac. iPhone ANE 18 fails to compile `chunk_2` with `std::bad_cast` in MIL→EIR translation when the producer layer's `kv13_k`/`kv14_k` alias slice is exposed as a chunk output. `.clone()` in PyTorch and 4-chunk decode split (each chunk smaller) both produce the same compile failure. Stateful path remains Mac-only / dev-only.

### iPhone bundle pushes

`xcrun devicectl device copy to` does **not** delete files that aren't in the source. Switching bundle layouts (e.g. multimodal → baseline) requires deleting and re-installing the app to clear the data container — otherwise orphan files (a leftover `prefill_chunk1.mlmodelc` is enough) silently override the new bundle's behaviour.

---

## Build steps

Run on Mac with a working `coremltools 9.0` venv. macOS 26 needs the source-built wheel (see `docs/MACOS_26_BUILD_ENV.md`).

```bash
PY=/tmp/ct_build_venv/bin/python
HF_DIR=/path/to/gemma4-e4b/hf_model     # local clone of google/gemma-4-E4B-it
ROOT=$(pwd)

# 1. 3-chunk decode (Topology II merged middle chunk).
mkdir -p /tmp/gemma4-e4b-3way
$PY conversion/build_gemma4_3way.py \
    --model gemma4-e4b --hf-dir "$HF_DIR" \
    --output /tmp/gemma4-e4b-3way --ctx 2048

# 2. Vision encoder (ANE-targeted, square 48×48 grid → 256 soft tokens at LM hidden 2560).
mkdir -p /tmp/gemma4-e4b-vision-ane
$PY ../CoreML-LLM/conversion/convert_gemma4_multimodal.py \
    --model-path "$HF_DIR" \
    --output /tmp/gemma4-e4b-vision-ane \
    --quantize int4 \
    --vision-ane

# 3. Audio encoder (Conformer + Swift projection sidecars).
mkdir -p /tmp/gemma4-e4b-audio
$PY ../CoreML-LLM/conversion/convert_audio.py \
    --model-path "$HF_DIR" \
    --output /tmp/gemma4-e4b-audio \
    --quantize int4

# 4. Assemble bundle (compiles mlpackage→mlmodelc, copies sidecars + legacy chunks).
LEGACY=/path/to/gemma4-e4b-coreml-bundle bash scripts/assemble_gemma4_e4b_multimodal.sh
# → build/gemma4-e4b-multimodal/   (~7.6 GB)
```

The assembler script accepts env vars `THREEWAY` / `VISION_ANE` / `AUDIO` / `LEGACY` / `MEL_FALLBACK` / `OUT` to override defaults. See the script header for the full layout description.

---

## Sideload to iPhone

```bash
DEVICE=$(xcrun devicectl list devices | awk '/iPhone 17 Pro/{print $3}')

# 1. Delete CoreMLLLMChat app on iPhone (long-press home icon → "Remove App"
#    → "Delete App"). devicectl doesn't remove orphan files; switching from a
#    previous bundle without a clean sandbox WILL produce broken output.
# 2. In Xcode, Cmd+R to reinstall a fresh app. Launch once to create the
#    Documents container.
# 3. Force-quit the app (swipe up in app switcher) so devicectl can write.

xcrun devicectl device copy to --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier com.example.CoreMLLLMChat \
    --source build/gemma4-e4b-multimodal \
    --destination Documents/Models/gemma4-e4b

# 4. Xcode scheme env vars:
#      LLM_SHOW_EXPERIMENTAL=1  (already required for some pickers)
#      LLM_VISION_FORCE_ANE=1   (route vision.ane.mlmodelc through ANE)
# 5. Cmd+R, pick "Gemma 4 E4B" in the picker, test.
```

---

## Verified iPhone 17 Pro results (2026-05-03)

| Modality | Result |
|---|---|
| Text-only | 15.7 tok/s, baseline-quality response (matches Mac) |
| Image + text | Coherent description, no gibberish |
| Video + text | Coherent description |
| Audio + text | Correct response (after `AudioProcessor` `embed_proj` non-square fix) |

---

## Files of interest

| File | Role |
|---|---|
| `Sources/CoreMLLLM/AudioProcessor.swift` | Two-stage Swift projection. `ProjectionWeights` now derives `inDim` / `outDim` / `finalDim` from weight tensor sizes; embed_proj sgemm uses `finalDim` (E4B 2560) instead of hard-coded `outDim`. |
| `conversion/models/gemma4_swa_merged.py` | `MergedChunk23` accepts `own_range` / `shared_range`; defaults E2B (L8-14 / L15-24); E4B passes (12,24)/(24,33). |
| `conversion/build_gemma4_3way.py` | Threads `compute_chunk_boundaries(cfg)` into the merged chunk so `--model gemma4-e4b` produces correct 3-way decode. |
| `scripts/assemble_gemma4_e4b_multimodal.sh` | Reproducible bundle assembly. |
| `Sources/CoreMLLLM/ChunkedEngine.swift` | Auto-detects Topology II via `chunk2_3way` + `chunk3_3way` presence. Routes prefill via `prefill_b8` multifunction in legacy chunks 1-4 when `prefill_chunk1` is absent (our case). |

---

## What's NOT in this bundle (intentional)

- **`prefill_chunk{1..4}.mlmodelc` (multifunction T=64/128/256/512)**: see "What was tried and rejected" above.
- **`vision.mlmodelc` (GPU variant, output `[1, 280, hidden]`)**: not built for E4B. We ship `vision.ane.mlmodelc` only and rely on `LLM_VISION_FORCE_ANE=1`.
- **`vision_video.mlmodelc`**: video runs through still-image vision with 2×2 pooling fallback in the engine. Adequate quality on validation.
- **Stateful chunks**: `Gemma4StatefulMultimodalEngine` is Mac-only / dev-only.
