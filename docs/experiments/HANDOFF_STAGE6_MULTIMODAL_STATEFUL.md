# Handoff — Stage 6: Multimodal stateful (vision + audio + video)

**Created:** 2026-04-27
**Branch suggestion:** `stage6-multimodal-stateful`
**Goal:** add vision (image), audio, and video support to the
Gemma 4 E2B stateful Linear path so the new HF default
(`mlboydaisuke/gemma-4-E2B-stateful-coreml`) can replace the legacy
4-chunk multimodal bundle entirely.

---

## Why this is needed

Stage 3 (PR #141 + #142, merged 2026-04-26) shipped the stateful
Linear 3-chunk path as a default-visible picker option, but it's
**text-only** — `Gemma4StatefulEngine.swift` has no hooks for vision /
audio encoders. Multimodal users stay on the legacy 4-chunk bundle
(`gemma4e2b`, multimodal 4-chunk recurrent).

Stage 6 brings the stateful path to feature parity with legacy so the
two can collapse into one default that's faster AND multimodal.

---

## What needs to land

### 1. Engine API extension

`Sources/CoreMLLLM/Gemma4StatefulEngine.swift` currently:
```swift
public func generate(inputIds: [Int32], maxNewTokens: Int = 64,
                     eosTokenIds: Set<Int32> = [],
                     onToken: ((Int32) -> Void)? = nil) async throws -> [Int32]
```

Add a multimodal variant that accepts encoded vision/audio features
and image-pad token spans:

```swift
public func generate(
    inputIds: [Int32],
    visionFeatures: VisionFeatures? = nil,    // image hidden states + DeepStack taps
    audioFeatures: AudioFeatures? = nil,      // audio hidden states
    imagePadTokenId: Int32 = 256000,          // Gemma 4 image-pad
    audioPadTokenId: Int32 = 256001,          // Gemma 4 audio-pad
    maxNewTokens: Int = 64,
    eosTokenIds: Set<Int32> = [],
    onToken: ((Int32) -> Void)? = nil
) async throws -> [Int32]
```

The engine's prefill loop replaces image-pad / audio-pad token embeds
with the encoded features at the matching positions before passing
into chunk_1.

### 2. Vision/audio encoder loading

Load `vision.mlmodelc` / `vision_video.mlmodelc` / `audio.mlmodelc`
alongside the chunks in `Gemma4StatefulEngine.load()`. Existing legacy
path already does this in `CoreMLLLM/ChunkedEngine.swift` — pattern
to mirror is ~lines 800-900 there (MLModel init, deepstack tap
extraction, etc.).

### 3. DeepStack injection

Gemma 4 uses **DeepStack** for vision (3 layer-tap injections at L0,
L8, L17 from `vision.mlmodelc`'s outputs `ds_0`, `ds_1`, `ds_2`). The
stateful chunks need to accept these taps and add them to hidden
states at the right layers — this is a converter change too:
`conversion/models/gemma4_swa_stateful_chunks.py` needs DeepStack
injection mirroring `gemma4_swa_chunks.py`'s pattern (search for
`ds_0` / `ds_1` / `ds_2` in the legacy converter).

For the stateful path:
- chunk_1 (L0-7) receives `ds_0` and `ds_1` (L0 + L7 injection)
- chunk_2 merged (L8-24) receives `ds_2` (L17 injection — L17 is
  inside chunk_2's range)
- chunk_3 (L25-34) no DS

Per-chunk DeepStack input + gate scalar pattern is the same as
Qwen3-VL stateful (see
`Examples/CoreMLLLMChat/CoreMLLLMChat/Qwen3VL2BStatefulGenerator.swift`
for the "ds_0/ds_1/ds_2 + visual_active gate" Provider class).

### 4. HF repo update

Add encoder mlmodelc + sidecar files to
`mlboydaisuke/gemma-4-E2B-stateful-coreml`:
- `vision.mlmodelc/` (~320 MB)
- `vision_video.mlmodelc/` (~338 MB)
- `audio.mlmodelc/` (~295 MB)
- `mel_filterbank.bin`, `audio_config.json`
- `output_proj_*.npy`, `embed_proj_weight.npy`

Bundle grows: 3.7 GB → ~4.7 GB.

### 5. Downloader file list

`Sources/CoreMLLLM/ModelDownloader.swift:buildGemma4StatefulLinearFileList()`
needs the new files added (mirror `gemma4e2b`'s extraFiles entries
for vision/audio).

### 6. LLMRunner glue

`Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift` already routes
to `Gemma4StatefulEngine` when `gemma4_e2b_stateful_chunks/` is
present. Extend the route to:
- Pass image / audio attachments to the engine
- Reuse the existing `cachedVisionImage` / `cachedVisionFeatures`
  pattern (see Qwen3-VL stateful)

---

## Reference implementations

| reference | what to copy |
|---|---|
| `Sources/CoreMLLLM/ChunkedEngine.swift` (legacy gemma4-e2b) | vision/audio mlmodelc loading, DeepStack tap extraction, image-pad / audio-pad token injection |
| `Examples/CoreMLLLMChat/CoreMLLLMChat/Qwen3VL2BStatefulGenerator.swift` | stateful + multimodal pattern: `chunk0VisionPrefill`, ds_0/1/2 inputs, visual_active gate, vision feature caching across turns |
| `conversion/models/gemma4_swa_chunks.py` (legacy converter) | DeepStack injection sites in chunk_1 + chunk_2 forward |

---

## Estimated effort

- Day 1 (Mac): converter DeepStack injection + chunk_1 PoC build + Mac
  sanity (vision encoder output shape match) — 4-6 h
- Day 1-2: full bundle build + LLMRunner wiring + Mac end-to-end
  (image prompt → text response) — 4-6 h
- Day 2 (iPhone): push + test image / audio / video, regression check
  on text-only — 2-4 h

Total: 1-2 working days.

---

## Risks / unknowns

1. **iPhone ANE compile** of the new chunk_1 with DeepStack adds.
   The DeepStack adds are simple residuals; should compile fine.
   But the multifunction `prefill_b8` of the multimodal chunk_1 is
   another graph — re-test for `ANECCompile FAILED` (Stage 3 hit
   this for ISWA + dual-state + multifunction; multimodal version
   may or may not).
2. **Vision token splicing** in stateful prefill: the engine needs to
   handle a position where the embed comes from the vision encoder,
   not the embedding lookup. Hidden state position-by-position
   override is a new code path.
3. **Multifunction prefill_b8 + DeepStack injection at T=8**: more
   complex graph. May fall back to T=1 prefill for vision turns
   (acceptable — vision turns are rare relative to chat turns).

---

## Pre-merge gating

Same as Stage 3: iPhone validation required. Specifically:
- Image prompt → response works
- Audio prompt → response works
- Video prompt → response works
- Text-only chat unchanged (33+ tok/s decode preserved)
- Cross-turn KV reuse intact

---

## Picker default

Once Stage 6 ships, swap `gemma4e2b` and `gemma4e2bStatefulLinear` in
`ModelDownloader.swift:defaults` so stateful Linear becomes default
again. The legacy `gemma4e2b` stays as a fallback for one release
cycle then drops.
