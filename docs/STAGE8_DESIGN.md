# Stage 8: E4B + E2B Multimodal Stateful Engine — Implementation Plan

**Status:** Phase A code complete (text-only E4B parity with E2B). Phase B (multimodal wiring) deferred to a follow-up session.

**Branch:** `feat/e4b-optimize-multimodal`
**Date:** 2026-04-28
**Predecessor docs:** `docs/MLSTATE_MULTIMODAL_PROBE.md`, `docs/HANDOFF_STAGE8_MLSTATE_MULTIMODAL.md`, `docs/SESSION_2026_04_27_STAGE6_MULTIMODAL.md`

---

## Goal

Ship Gemma 4 E4B and E2B with the stateful Linear decode path **and** vision + video + audio multimodal input on iPhone 17 Pro. Reach the fastest decode tok/s the architecture allows while keeping multimodal correctness intact.

**Scope split chosen on 2026-04-28:**
- **Option A** — separate Swift class `Gemma4StatefulMultimodalEngine`, leaving the legacy `Gemma4StatefulEngine` (text-only multifunction prefill_b8 path) untouched.
- **E2B + E4B both get multimodal stateful** — same engine class drives both via `model_config.json`-derived dimensions.
- **Existing HF repos preserved** — new `mlboydaisuke/gemma-4-{E2B,E4B}-stateful-multimodal-coreml` repos rather than mutating the existing stateful repos. Mirrors the dual-repo pattern.

---

## What's already shipped (Phase A — code only, builds + iPhone gates pending)

Two commits on `feat/e4b-optimize-multimodal`:

1. **`4665ab2`** — generalize 3-chunk + 4-chunk converters to E2B + E4B
   - `SWAStatefulMergedChunk23{,Prefill,Single,PrefillSingle}` accept `own_range` / `shared_range`. Defaults E2B (own=L8-14, shared=L15-24); E4B passes (12,24)/(24,33).
   - `build_gemma4_e2b_stateful_3chunks.py` --model gemma4-e4b now produces a 3-chunk merged bundle (chunk_1 L0-11 / chunk_2 L12-32 / chunk_3 L33-41).
   - `sanity_stateful_chunks.py` model presets (--model gemma4-e2b / gemma4-e4b).
   - `scripts/assemble_gemma4_stateful_e4b.sh` bundle assembler for iPhone sideload.
   - `Sources/CoreMLLLM/ModelDownloader.swift` — `gemma4e4bStateful` + `gemma4e4bStatefulLinear` ModelInfo entries (slots 6/7 under `LLM_SHOW_EXPERIMENTAL=1`, sideload-only — `downloadURL: ""`).
   - `Examples/.../LLMRunner.swift` — stateful detection comment now lists all four E2B+E4B folders.

2. **`2655c17`** — single-function T=288 prefill builder accepts E4B
   - `build_gemma4_stateful_singlefunc_prefill.py` plumbs `own_range` / `shared_range` through `convert_chunk2_merged_prefill`. Without this, on E4B the merged middle prefill chunk silently used E2B layer ranges.

**Pending Phase A work (hardware-blocked):**
- A4: Mac build (3-chunk decode + multifunction prefill_b8) — kicked off in background after this doc lands.
- A4': Mac build (T=288 single-function prefill) for E2B + E4B — same session.
- A5: iPhone 17 Pro A/B for E4B 3-chunk merged stateful Linear — needs device.
- A6: HF upload `mlboydaisuke/gemma-4-E4B-stateful-coreml` once iPhone clears.

---

## Phase B — Stage 8 multimodal stateful engine

### B1. T=288 single-function prefill mlpackages (DONE script-side, build pending)

E2B and E4B variants of:
- `chunk_1_prefill_T288.mlpackage` (own KV)
- `chunk_2_3way_prefill_T288.mlpackage` (merged: own + shared internal)
- `chunk_3_prefill_T288.mlpackage` (KV-shared + lm_head + argmax)

T=288 = 256-token image span + ~32 text margin (BOS / turn markers). Drop to T=224 if 8 GB iPhone non-Pro rejects T=288 compile peak (probe required, see C1).

**Single-function** (separate mlpackage per T) instead of multifunction merge — iPhone ANE 18 rejects multifunction T>1 + dual MLState with `ANECCompile FAILED 11`. Probe 2 verified single-function T=288 compiles in 7.3 s on iPhone 17 Pro A19 Pro.

### B2. New Swift class `Gemma4StatefulMultimodalEngine`

**Location:** `Sources/CoreMLLLM/Gemma4StatefulMultimodalEngine.swift` (new file).

**Why a separate class (not inheritance / extension):**
- `Gemma4StatefulEngine` is `public final class` — not extensible.
- The two engines have different prefill path topology (multifunction merged vs separate single-func mlpackages), different state lifecycle (decode-only state vs decode+prefill state with bridge), and different public API shape (`generate(prompt:)` vs `generate(prompt:images:audio:)`).
- Keeps the Stage 3 stateful Linear 33.4 tok/s text-only path bit-identical for users who don't want multimodal.

**Storage:**

```swift
@available(iOS 18.0, macOS 15.0, *)
public final class Gemma4StatefulMultimodalEngine {
    // Decode chunks (3-chunk merged stateful Linear)
    private var decodeChunk1: MLModel?  // L0-7 / L0-11 (E4B) — own KV
    private var decodeChunk2: MLModel?  // L8-24 / L12-32 — merged own+shared
    private var decodeChunk3: MLModel?  // L25-34 / L33-41 + lm_head + argmax

    // Prefill T=288 chunks (separate mlpackages, single-function)
    private let prefillT: Int = 288
    private var prefillChunk1: MLModel?
    private var prefillChunk2: MLModel?
    private var prefillChunk3: MLModel?  // identical structure to decodeChunk3

    // Per-chunk MLState (chunk_3 is stateless — KV-shared from chunk_2)
    private var decodeState1: MLState?
    private var decodeState2: MLState?
    private var prefillState1: MLState?
    private var prefillState2: MLState?

    // Multimodal encoders (lazy)
    private var visionModel: MLModel?         // SigLIP still-image, 256 tokens
    private var videoVisionModel: MLModel?    // pooled SigLIP, 64 tokens/frame
    private var audioModel: MLModel?          // Conformer, ~50 tokens/2sec
    private var audioProjection: ProjectionWeights?
    private var melFilterbank: Data?

    // Sidecars (same as legacy engine)
    private var embedTokens: EmbeddingLookup?
    private var embedTokensPerLayer: EmbeddingLookup?
    private var cosSlidingTable: Data?  // mmap
    private var sinSlidingTable: Data?
    private var cosFullTable: Data?
    private var sinFullTable: Data?

    // Cross-turn state (Phase 2a — LCP match)
    private var persistedInputIds: [Int32] = []
    private var persistedPosition: Int = 0

    // Reusable scratch (T=1 decode + T=288 prefill)
    private var maskFullDecode, maskSlidingDecode: MLMultiArray!
    private var maskFullPrefill288, maskSlidingPrefill288: MLMultiArray!
    // ... batch hidden, per-layer raw, RoPE batched, etc.
}
```

**Public API:**

```swift
public init(config: Config = Config())
public func resetPersistedState()
public func load(modelDirectory: URL) async throws

public func generate(
    prompt: String,
    images: [CGImage] = [],
    audioPCM16k: [Float]? = nil,
    maxNewTokens: Int = 512,
    eosTokenIds: Set<Int32> = [],
    onToken: ((Int32) -> Void)? = nil
) async throws -> [Int32]
```

Video is a series of CGImages produced by `VideoProcessor.extractFrames` — exposed at the LLMRunner layer, not the engine.

### B3. Multimodal helpers — port from Stage 6 (`origin/stage6-multimodal-stateful` commit `02ac583`)

The Stage 6 patch added 528 lines to the legacy engine. We port these into the new class with one structural change: feature splice happens **during prefill at T=288** instead of during multifunction prefill_b8.

**Helpers to port (file:line references for `02ac583`):**

| Helper | Purpose | Adaption needed |
|---|---|---|
| `loadMultimodalEncoders` | Probe + load vision/video/audio mlmodelc + sidecars | Same layout, new file paths |
| `processImage(_: CGImage)` | UIImage → 256-token feature MLMultiArray | None — same encoder |
| `processVideoFrame` | Per-frame still vision encoding (64 tokens) | None |
| `processAudio(_: [Float])` | PCM 16k → mel → Conformer → projection | None |
| `computeVisionGroupIds` | Per-token group label (which image each token belongs to) | T=8 → T=288 generalization |
| `fillBatchMasksVisionAware` | Bidirectional within-image, causal across | T=8 → T=288 generalization |
| `multimodalSpliceT1` | Per-token feature splice at IMAGE/AUDIO_TOKEN_ID position | Reused for tail of prompt that doesn't fit T=288 |

**Special token IDs (preserved from Stage 6):**
- `IMAGE_TOKEN_ID = 258880`
- `AUDIO_TOKEN_ID = 258881`
- `VIDEO_TOKEN_ID = 258884`

### B4. State bridge (probe 2 verified)

After prefill completes, copy `kv_cache_sliding` and `kv_cache_full` from prefill MLState to decode MLState. Critical requirement: **nested closures** — the buffer pointer is only valid within `withMultiArray(for:)` scope.

```swift
private func bridgeKVState(from src: MLState, to dst: MLState) {
    let names = ["kv_cache_sliding", "kv_cache_full"]
    for name in names {
        src.withMultiArray(for: name) { srcArr in
            dst.withMultiArray(for: name) { dstArr in
                let bytes = srcArr.count * MemoryLayout<UInt16>.stride  // fp16
                memcpy(dstArr.dataPointer, srcArr.dataPointer, bytes)
            }
        }
    }
}
```

Called twice per generate(): once for chunk_1 state (sliding-only on E2B / sliding+full on E4B), once for chunk_2 state (sliding+full both).

Pitfall: chunk_3 is **stateless** in the 3-chunk variant (KV-shared from chunk_2 outputs kv13/kv14). No state to bridge for chunk_3.

### B5. Generate flow (single-image text+image example)

```
Input: prompt = "What's in this picture?", images = [oneImage]

1. Build inputIds:
     [BOS] <image_pad×256> "What's in this picture?" [EOT]
     ≈ 1 + 256 + 8 + 1 = 266 tokens — fits in T=288.

2. Preprocess image:
     features = visionModel(processImage(oneImage))  // (1, 256, hiddenSize)

3. Build prefill input:
   - embed_lookup(inputIds) → hidden (1, 266, hidden)
   - splice features[0..<256] into hidden[1..<257]
   - zero per_layer_raw at image positions
   - vision-aware mask: bidirectional within hidden[1..<257],
     causal elsewhere

4. Run prefill T=288 (pad inputIds to 288 with mask = -inf):
   - prefillChunk1(hidden, masks, rope, pos=0..287, ringPos=0)
       → updates prefillState1 (kv_cache_sliding[0..287])
   - prefillChunk2(prefill1.hidden, ..., pos=0..287, ringPos=0)
       → updates prefillState2; outputs kv13_k/v + kv14_k/v at last layer
   - prefillChunk3(prefill2.hidden, kv13_*, kv14_*, ...)
       → outputs token_id (last decode token)

5. Bridge state:
     bridgeKVState(prefillState1 → decodeState1)
     bridgeKVState(prefillState2 → decodeState2)

6. Decode loop (T=1, position=266, 267, ...):
   - decodeChunk1(emb(token), masks, rope, pos, ringPos)
     state: decodeState1
   - decodeChunk2(...) state: decodeState2
   - decodeChunk3(..., kv13, kv14) → next token
   - emit, append to output, repeat until EOS or maxTokens
```

For prompts longer than T=288: **split into multiple T=288 prefill passes** (no overlap; each pass writes consecutive ring positions). Image span must NOT split across passes — push image to first pass and chunk text after.

### B6. ModelDownloader bundle layout

Mirror E2B's existing `gemma-4-E2B-stateful-coreml` layout but add a `prefill_T288/` subdir:

```
mlboydaisuke/gemma-4-{E2B,E4B}-stateful-multimodal-coreml/
  gemma4_e2b_stateful_chunks/         # subdir kept for engine compat
    chunk_1.mlmodelc                  # decode multifunction merged
    chunk_2.mlmodelc
    chunk_3.mlmodelc
    prefill_T288/
      chunk_1_prefill_T288.mlmodelc
      chunk_2_3way_prefill_T288.mlmodelc
      chunk_3_prefill_T288.mlmodelc
    embed_tokens_q8.bin               # sidecars
    embed_tokens_scales.bin
    embed_tokens_per_layer_q8.bin
    embed_tokens_per_layer_scales.bin
    per_layer_projection.bin
    per_layer_norm_weight.bin
    cos_sliding.npy / sin_sliding.npy
    cos_full.npy    / sin_full.npy
    hf_model/                         # tokenizer
    model_config.json
    vision.mlmodelc                   # multimodal encoders
    vision_video.mlmodelc
    audio.mlmodelc
    output_proj_weight.npy            # audio projection sidecars
    output_proj_bias.npy
    embed_proj_weight.npy
```

Total bundle size:
- Decode chunks: ~1.15 GB (E2B) / ~1.6 GB (E4B)
- T=288 prefill chunks: ~1.50 GB (E2B) / ~2.0 GB (E4B)
- Encoders: ~0.99 GB (shared between models)
- Sidecars + tokenizer: ~0.4 GB
- **Total: ~4.0 GB (E2B) / ~5.0 GB (E4B)** download.

`ModelDownloader.buildGemma4StatefulMultimodalE{2,4}BFileList()` enumerates all files. Mirror the existing E2B helpers' pattern.

### B7-B8. iPhone tests + parity

- B7: Real-device test — image+text, video+text, audio+text → correct output.
- B8: Parity test — fixed image prompt through legacy 4-chunk prefill+decode vs new T=288 stateful prefill+bridge+decode. First 32 decode tokens must agree (top-1).

### C1. 8 GB iPhone non-Pro probe

Probe 1 only validated 12 GB iPhone 17 Pro at T=288. iPhone 15 / 16 / 17 non-Pro have 8 GB RAM — chunk_2 prefill at T=288 may fail compile peak (chunk_2 is the largest at 21 layers for E4B). If 8 GB fails, fall back to T=224 (image still fits 256 tokens; text margin shrinks to ~−32 — acceptable since prompt-tail fallback to T=1 already exists).

---

## Open questions for next session

1. **Picker entry naming.** "Gemma 4 E4B (multimodal stateful)" or "Gemma 4 E4B (stateful, vision+audio)"? UI clarity vs concision.
2. **Default model swap.** When B is shipped, should `gemma4e2b3way` (current production multimodal) be deprecated in favor of `gemma4e2bStatefulMultimodal` (faster decode + multimodal)? Memory note says current E2B 3-chunk is the multimodal default; swapping requires a soft-deprecation cycle for users mid-download.
3. **Cross-turn KV with vision.** Phase 2a LCP match assumes prefix invariance. If turn 1 has image and turn 2 reuses the same image, the image features may have been re-encoded — does the LCP match still hold? Stage 6 had this concern unresolved.

---

## Build commands (Phase A — kick off after this doc lands)

```bash
# Build 1: E4B 3-chunk merged decode + multifunction prefill_b8
HF_DIR=/Users/majimadaisuke/Downloads/CoreML-LLM/output/gemma4-e4b/hf_model
python conversion/build_gemma4_e2b_stateful_3chunks.py \
    --model gemma4-e4b \
    --hf-dir "$HF_DIR" \
    --output /tmp/gemma4-e4b-stateful-3chunk \
    --linear-projections \
    --prefill-batches "8" \
    --ctx 2048 \
    --nbits 4

# Build 2: E4B T=288 single-function prefill (Stage 8)
python conversion/build_gemma4_stateful_singlefunc_prefill.py \
    --model gemma4-e4b \
    --hf-dir "$HF_DIR" \
    --output /tmp/gemma4-e4b-singlefunc-prefill-T288 \
    --t 288 \
    --linear-projections \
    --ctx 2048 \
    --nbits 4

# Sanity (chunk shape + chained 1→2→3 forward, CPU_AND_NE):
python conversion/sanity_stateful_chunks.py \
    --model gemma4-e4b \
    --artifacts /tmp/gemma4-e4b-stateful-3chunk

# Bundle assemble for sideload (assumes legacy E4B sidecars in
# CoreML-LLM/output/gemma4-e4b/bundle):
SIDECARS=/Users/majimadaisuke/Downloads/CoreML-LLM/output/gemma4-e4b/bundle \
SRC_CHUNKS=/tmp/gemma4-e4b-stateful-3chunk \
bash scripts/assemble_gemma4_stateful_e4b.sh
```

Both builds load the 15 GB E4B safetensors; estimated 60-120 min total. Run sequentially to avoid memory contention.

---

## Reference

- `docs/MLSTATE_MULTIMODAL_PROBE.md` — probe 1 (T=288 chunk_1 compiles) + probe 2 (state bridge memcpy works).
- `docs/HANDOFF_STAGE8_MLSTATE_MULTIMODAL.md` — original Stage 8 handoff with 5-step plan.
- `docs/SESSION_2026_04_27_STAGE6_MULTIMODAL.md` — Stage 6 multimodal in legacy engine (in-place patch; ours is fresh class).
- Stage 6 commits: `origin/stage6-multimodal-stateful` (`02ac583`, `2432995`, `987ad86`) — port these helpers verbatim into new class.
- `Sources/CoreMLLLM/Gemma4StatefulEngine.swift` — legacy engine, reference for the patterns we duplicate (mask filling, RoPE lookup, EmbeddingLookup wiring, position scratch, etc.).
