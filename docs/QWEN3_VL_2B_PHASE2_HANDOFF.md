# Qwen3-VL 2B Vision Integration — Session Handoff

Handoff for **Phase 2c–2d** of the Qwen3-VL 2B iPhone ship. Phase 1
(text-only decode) and Phase 2a (vision encoder converter) + Phase 2b
(chunk_0 DeepStack-aware converter) are done on branch
`research/qwen3-vl-4b` (yes, the branch kept the 4B name; 4B was
abandoned, 2B is the ship path).

---

## Where things are now

**iPhone 17 Pro, Qwen3-VL 2B text-only (shipping):**
- 7.5 tok/s decode, max_seq=2048, INT8, 4 body chunks × 7 layers
- ~200 MB phys_footprint (mmap embed sidecar)
- Model on iPhone at `Documents/Models/qwen3-vl-2b/qwen3_vl_2b_decode_chunks/`

**Artifacts on disk:**
- `/tmp/qwen3_vl_2b/qwen3_vl_2b_decode_chunks/` — text chunks (chunk_0..3
  + chunk_head + embed_weight.bin, INT8, ~2.1 GB total)
- `/tmp/qwen3_vl_2b_vision/qwen3_vl_2b_vision/vision.mlpackage` — vision
  encoder (388 MB INT8, 90.4% ANE, 99.8% fp16)
- `/tmp/qwen3_vl_2b_chunk0_v/` — **set by Phase 2b**, DeepStack-aware
  chunk_0 replacement

**HF repos:**
- `mlboydaisuke/qwen3-vl-4b-coreml` — stale, from the abandoned 4B run.
  **Delete or overwrite** with 2B artifacts at ship time.
- (not yet created) `mlboydaisuke/qwen3-vl-2b-coreml` — target for ship.

---

## Phase 2c: Swift vision integration

### Classes to add / modify

**New**: `Qwen3VL2BVisionEncoder.swift` (under `Examples/CoreMLLLMChat/
CoreMLLLMChat/`)
- Wraps `vision.mlmodelc` / `.mlpackage`.
- Input: `CGImage` → preprocess → `MLMultiArray` (3, 2, 448, 448) fp16.
- Output: `VisionFeatures` struct holding `{hidden: MLMultiArray (196,
  2048), deepstack: [MLMultiArray; 3]}` — each deepstack tensor also
  (196, 2048). All fp16, ANE-resident.
- Preprocess: resize to 448×448 using `CIImage`/`vImage`, mean-normalize
  with Qwen-VL ImageNet stats (mean=[0.48145466, 0.4578275, 0.40821073],
  std=[0.26862954, 0.26130258, 0.27577711]), duplicate frame across
  `temporal_patch=2` dimension.

**Modified**: `Qwen3VL2BGenerator.swift`
- Extend the generator to accept a `VisionFeatures?` alongside
  `inputIds` in `generate(…)`.
- The prefill loop currently feeds tokens one by one through the decode
  chunks. When a prompt token equals `151655` (`<|image_pad|>`), the
  generator must:
    1. Replace the embed lookup for that step with the pooled vision
       token from `features.hidden[imageTokenIdx]`.
    2. Supply `ds_0`, `ds_1`, `ds_2` to `chunk_0`'s predict call (as
       new input slots Phase 2b added).
  For non-image tokens, supply a zeroed deepstack buffer + a gate input
  (`visual_active = 0`) so the injection is a no-op. Phase 2b's chunk_0
  **gates** the deepstack adds on that scalar so the graph stays static.
- Image tokens appear as a contiguous run in the prompt between
  `<|vision_start|>` (151652) and `<|vision_end|>` (151653). Count the
  run, indexes 0..195 are the 196 vision tokens from the encoder output.
  See `run_chunked_decode` in the VL4B parity script for shape handling.

**Modified**: `LLMRunner.swift`
- Qwen3VL2B dispatch path needs a new `generate(messages:image:)`
  overload. When `image != nil`:
    1. Preprocess the image.
    2. `await encoder.encode(pixelValues)` → `VisionFeatures`.
    3. Build input IDs using the vision chat template (see below).
    4. `gen.generate(inputIds:, visionFeatures:)`.

### Chat template

Qwen3-VL uses:
```
<|im_start|>user
<|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>{text prompt}<|im_end|>
<|im_start|>assistant
```

Number of `<|image_pad|>` tokens = 196 per image (spatial_merge applied).

`AutoTokenizer.applyChatTemplate` with `messages = [{role: "user",
content: [{type: "image"}, {type: "text", text: "…"}]}]` should emit
the right sequence. **Verify** with a roundtrip against HF's
`processor.apply_chat_template` before trusting it.

### Image token embedding

For image tokens, the "embed lookup" isn't the vocab embed — it's the
corresponding row of the vision merger output. So the generator's
embed-lookup branch becomes:

```swift
if tokenId == 151655 {
    // image_pad token; read merger output row for this image position
    memcpy(reusableHidden.dataPointer,
           featuresHiddenPtr.advanced(by: imageTokenIdx * hiddenSize),
           hiddenSize * 2)
    imageTokenIdx += 1
} else {
    // regular token; mmap embed table
    embedLookup(token: tokenId)
}
```

`imageTokenIdx` resets at each generate call, increments every time an
image_pad token is consumed.

### DeepStack feed

On every prefill step, chunk_0's `ds_0`, `ds_1`, `ds_2` inputs come
from `features.deepstack[0..2]` indexed by the same `imageTokenIdx`.
For non-image prefill tokens and all decode tokens, pass a zero
(1, 1, 2048) MLMultiArray and set `visual_active` to 0.0.

Reuse a single zero MLMultiArray per slot; don't alloc per step.

## Phase 2d: UI + chat template integration

**ChatView.swift**: extend to pick an image alongside the text prompt.
Look at `PhotosPicker` wiring for reference (Gemma path has one, same
pattern works here). Show a thumbnail inline in the user bubble.

**Message → generate pipeline**: route `image != nil` to the new
`generate(messages:image:)` overload.

**Error handling**: if the image fails to decode / preprocess, fall
back to text-only with a `[Error: …]` bubble so users know what
happened.

## What to push + upload at ship time

1. Compile the final chunks + vision.mlmodelc:
   ```bash
   for n in chunk_0 chunk_1 chunk_2 chunk_3 chunk_head; do
     xcrun coremlcompiler compile \
       /tmp/qwen3_vl_2b/qwen3_vl_2b_decode_chunks/${n}.mlpackage \
       /tmp/qwen3_vl_2b_final/
   done
   xcrun coremlcompiler compile \
     /tmp/qwen3_vl_2b_vision/qwen3_vl_2b_vision/vision.mlpackage \
     /tmp/qwen3_vl_2b_final/
   ```
2. `hf upload mlboydaisuke/qwen3-vl-2b-coreml /tmp/qwen3_vl_2b/qwen3_vl_2b_decode_chunks qwen3_vl_2b_decode_chunks`
3. `hf upload mlboydaisuke/qwen3-vl-2b-coreml /tmp/qwen3_vl_2b_vision/qwen3_vl_2b_vision qwen3_vl_2b_vision`
4. `ModelDownloader.buildQwen3VL2BFileList` currently lists only the
   text chunks; **add the `qwen3_vl_2b_vision/vision.mlpackage` entry**
   so the download populates both.
5. README + release notes for v1.2.0 (text ship + vision). Keep the
   Qwen3.5 2B (v1.1.0) block, add a new v1.2.0 section for Qwen3-VL 2B.

## Known gotchas

- **Qwen3-VL mRoPE**: the text-only decode path collapses the mRoPE
  [24,20,20] interleave into plain 1D RoPE because T=H=W=position.
  **With images this no longer holds** — image tokens have real T/H/W
  indices (in chat template the image occupies positions p..p+195 with
  a 2D grid structure). The simplest correct behavior is to give every
  image_pad token a unique `position` scalar in the order emitted by
  the tokenizer; the HF forward does this internally. If outputs look
  off, check that image_pad tokens are not all getting `position=0`.
- **Image preprocessing**: HF's `Qwen3VLProcessor` does min/max size
  clamping + patch alignment. For 448×448 fixed input, skip all that
  and just resize + normalize. Mean/std values above.
- **Trailing FFFD fix**: Phase 1 fix in `LLMRunner.generateQwen3VL2B`
  already strips trailing U+FFFD before emitting deltas. Leave as-is.
- **Memory on device**: vision.mlmodelc loads ~400 MB, keep it
  allocated between calls (not recreated per image).

## Commit history (this session)

Main commits on `research/qwen3-vl-4b`:
- `9e4980d` initial 4B converter (ABANDONED, 1.7 tok/s)
- `0687d53` 4B Mac parity harness
- (ANE v2, max_seq=512, SDPA fused, fused MLP, 3-chunk) — iteration
  commits, superseded
- **2B pivot** (this session's ship path): 2B converter, 2B Swift
  generator, 2B ModelDownloader entry, vision encoder converter,
  chunk_0 DeepStack converter — **commit at session end**

## Next session prompt

> Continue Qwen3-VL 2B vision integration (Phase 2c/2d). Text ship
> ready at 7.5 tok/s, vision encoder + DeepStack-aware chunk_0 already
> built on disk. Read `docs/QWEN3_VL_2B_PHASE2_HANDOFF.md`, implement
> `Qwen3VL2BVisionEncoder.swift`, extend the generator + LLMRunner to
> feed vision features + DeepStack, wire PhotosPicker into ChatView,
> test on iPhone, then commit + HF upload + README + v1.2.0 release.
