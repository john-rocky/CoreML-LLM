# Session 2026-04-27 — Stage 6: Multimodal stateful (image + audio)

**Branch:** `stage6-multimodal-stateful`
**Roadmap entry:** `docs/HANDOFF_STAGE6_MULTIMODAL_STATEFUL.md`
**Goal:** add image + audio support to the Gemma 4 E2B stateful Linear
path so the new HF default
(`mlboydaisuke/gemma-4-E2B-stateful-coreml`) can handle multimodal
prompts without falling back to the legacy 4-chunk recurrent bundle.

---

## Headline finding — handoff DeepStack premise was wrong

The handoff doc said Gemma 4 uses **DeepStack** vision injection
(layer-tap inputs `ds_0`/`ds_1`/`ds_2` plus a `visual_active` gate at
layers L0/L7/L17). It does not. Concrete evidence:

| source | what it shows |
|---|---|
| `conversion/models/gemma4_swa_chunks.py` | no `ds_*`/`visual_active` anywhere; chunks accept only `hidden_states` + `per_layer_raw` |
| `conversion/models/gemma4_swa_stateful_chunks.py` | same — stateful is structurally identical to the legacy chunks for the prefill-input contract |
| `conversion/models/gemma4_vision.py` | `vision.mlpackage` / `vision_video.mlpackage` produce a single `image_features` output; no DeepStack slices |
| `Sources/CoreMLLLM/ChunkedEngine.swift` (legacy production) | `buildPrefillHidden` (line 2171) splices `imageFeatures` rows directly into the prefill hidden buffer at IMAGE_TOKEN_ID positions; PLR gets zero. No DS, no gate. |

DeepStack is a **Qwen3-VL** pattern (see
`conversion/build_qwen3_vl_2b_stateful_chunk0_vision.py`). The handoff
author appears to have confused the two architectures. **Gemma 4
multimodal is much simpler than the handoff implied.**

→ No converter changes. No build script changes. The existing 3-chunk
merged stateful Linear bundle (Stage 3 ship) handles vision/audio
unmodified — the chunks just need a hidden state with encoder rows
spliced in at the right positions, which is purely a Swift-side
responsibility.

This collapses Stage 6 from "1-2 working days" to "Swift engine
extension + HF re-upload of encoder mlmodelc files".

---

## What landed

### 1. `Gemma4StatefulEngine.swift` — multimodal storage + load + helpers

Added a self-contained multimodal layer that mirrors
`Sources/CoreMLLLM/CoreMLLLM.swift`'s vision/audio loading paths so the
stateful engine doesn't depend on the legacy CoreMLLLM class.

- **Encoder probing** at `load()`: `vision.mlmodelc` /
  `vision_video.mlmodelc` / `audio.mlmodelc` (with `.mlpackage`
  fallbacks; `LLM_VISION_FORCE_ANE=1` honors the ANE build).
- **Sidecar loading**: `mel_filterbank.bin`, `audio_config.json`
  (driving `audioMelFrames`, `audioNumTokensConfig`, `audioMelFloor`),
  `output_proj_weight.npy` / `output_proj_bias.npy` /
  `embed_proj_weight.npy` (Swift-side audio projection fallback).
- **Background prewarm** of the GPU vision graph (`prewarmVisionInBackground`)
  — first-call compile is ~30 s on iPhone GPU; running it on a utility
  queue at load time hides it behind text-only chat startup.
- **Public helpers**: `processImage`, `processVideoFrame`,
  `processAudio` (returning `(features, actualTokenCount)`).
- **`hasVision`/`hasAudio`/`hasVideoVision`/`defaultAudioNumTokens`**
  surfaces probe results to LLMRunner.

### 2. `Gemma4StatefulEngine.swift` — multimodal generate path

Added a new `generate(...)` overload:

```swift
public func generate(
    inputIds: [Int32],
    imageFeatures: MLMultiArray? = nil,
    imageNumTokens: Int = 0,
    audioFeatures: MLMultiArray? = nil,
    audioNumTokens: Int = 0,
    maxNewTokens: Int = 64,
    eosTokenIds: Set<Int32> = [],
    onToken: ((Int32) -> Void)? = nil
) async throws -> [Int32]
```

Existing text-only callers continue working unchanged because the
multimodal params default to nil.

Internal flow (per-token):

- `step()` (T=1): if token is `IMAGE_TOKEN_ID`/`VIDEO_TOKEN_ID` and
  `imageFeatures` was provided, splice the appropriate row instead of
  the embed lookup. Same for `AUDIO_TOKEN_ID` + `audioFeatures`.
  `per_layer_raw` is forced to zero for those positions (matches
  legacy ChunkedEngine).
- `prefillStep()` (T=N multifunction): same splice logic, batched.
- **Vision-aware mask** — `mmVisionGroupIds` is computed once at
  generate() entry. Each contiguous run of image-pad / video-pad
  tokens forms one bidirectional group; the mask buffers
  (`fillBatchMasksVisionAware` / `fillFullCausalMaskVisionAware` /
  `fillSlidingCausalMaskVisionAware`) unmask within-group attention
  so vision spans match HF's `token_type_ids_mask_function`. Audio
  spans stay strictly causal (matches Gemma 4 audio behavior).

**Cross-turn KV reuse**: when `resumeAt > 0` after the LCP match, the
engine advances `mmImageIdx`/`mmAudioIdx` past the resumed prefix so
the first new token splices the correct encoder row. LLMRunner is
responsible for `resetPersistedState()` when image/audio fingerprint
changes.

### 3. Optimizations shipped

| ID | name | win | notes |
|---|---|---|---|
| **B** | cross-turn KV reuse + cached vision features | follow-up text turns about cached image: TTFT ≈ resume cost only | LLMRunner caches `(CGImage, MLMultiArray)`; engine advances mm counters past resumed prefix |
| **D** | vision encoder background prewarm | -~30 s first-image-prompt TTFT | `.utility` queue at `load()`; safe — text-only chats unaffected |
| **E** | vision-aware mask in multifunction prefill_b8 | preserves the 7.77× Mac prefill win on vision turns (else T=1 fallback for the 256-token image span = ≈8× slower) | mask is a chunk INPUT; just write a different mask buffer in Swift, no graph changes |

### 4. `ModelDownloader.swift`

`buildGemma4StatefulLinearFileList` now pulls
`vision.mlmodelc` / `vision_video.mlmodelc` / `audio.mlmodelc` plus
sidecars (`mel_filterbank.bin`, `audio_config.json`,
`output_proj_*.npy`, `embed_proj_weight.npy`). Bundle grows
3.7 GB → ~4.7 GB; encoders are still optional at runtime.

### 5. `LLMRunner.swift`

- `loadGemma4Stateful` now flags `hasVision`/`hasAudio` from
  `engine.hasVision`/`engine.hasAudio`.
- `generate(messages:image:audio:)` routes attachments through to
  `generateGemma4Stateful(messages:image:audio:)`.
- `generateGemma4Stateful` builds the multimodal prompt
  (`<|image>...<image|>` / `<|audio>...<audio|>` blocks pinned to the
  last user turn so cross-turn KV resume keeps working), runs the
  vision/audio encoder (cache hit when image is the same `CGImage`
  instance, audio matches a cheap signature), and calls
  `engine.generate(imageFeatures:audioFeatures:)`.
- `resetConversation()` clears the new Gemma 4 multimodal caches
  alongside the existing Qwen3-VL caches.

---

## Out of scope for this PR

- **Video routing in LLMRunner.** The engine accepts video frame
  features (VIDEO_TOKEN_ID splices from `imageFeatures`), but
  `LLMRunner.generate(messages:videoURL:videoOptions:)` still routes
  through the legacy `CoreMLLLM` engine. Adding the
  `generateGemma4StatefulVideo` plumbing (frame extraction → encoder
  → multi-frame splice) is a follow-up PR.
- **Picker default swap.** Holding off on flipping
  `gemma4e2b` ↔ `gemma4e2bStatefulLinear` until video is supported on
  the stateful path; otherwise users on the new default would lose
  video-chat. Same iPhone-baseline gate applies.

---

## Open follow-ups

1. **Mac sanity** — image prompt → response on Mac CLI (text-only
   regression check + image parity vs legacy gemma4e2b).
2. **HF re-upload** — incrementally add encoder mlmodelc + sidecar to
   `mlboydaisuke/gemma-4-E2B-stateful-coreml`. Don't overwrite
   existing chunk files. Confirm with user before pushing (visible).
3. **iPhone push + image prompt test** — TTFT on first image (vision
   encoder cost), decode tok/s parity with text-only (33+), cross-turn
   KV reuse intact when asking follow-ups about the same image.
4. **Video routing** (new branch).
5. **Picker default swap** (after video).

---

## Risks

1. **Vision-aware mask quality vs legacy.** The legacy 4-chunk uses a
   `makePrefillVisionMask` that's identical in spirit to what we ship
   here (within-group bidirectional). Numerics should match closely
   modulo fp16 rounding. Mac sanity should compare top-1 generation
   against legacy on a fixed image.
2. **Multifunction prefill_b8 + vision-aware mask on iPhone ANE 18.**
   Stage 3 hit `ANECCompile FAILED 11` on the dual-state multifunction
   prefill graph; we landed with T=1 fallback on iPhone. Vision
   support inherits that fallback — when iPhone ANE rejects the
   multifunction graph, vision turns degrade gracefully to T=1
   (correct, but slower; we lose Optimization E's win on iPhone). Mac
   keeps the 7.77× win for image prefill.
3. **Audio cache signature collisions.** The 4-probe XOR signature is
   fast but not collision-resistant against an adversary. For chat
   use, a collision would mean the engine reuses stale audio features
   for a different clip — easily caught (response is wrong) and
   recovered (`resetConversation()`). Acceptable for MVP.
