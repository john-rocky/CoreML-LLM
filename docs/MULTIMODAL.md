# Multimodal (Image / Video / Audio) — Architecture & Debugging Notes

How Gemma 4 E2B processes images, video, and audio on-device via CoreML, and what went wrong along the way.

## Architecture Overview

```
Image (CGImage)
    │
    ▼
ImageProcessor.process()
    │  Aspect-ratio preserving resize to fit 2520 × 16² = 645,120 pixels
    │  Each side rounded to multiple of 48 (pooling_kernel × patch_size)
    │  512×512 → 768×768 → 48×48 = 2304 patches → 256 soft tokens
    │
    ▼
vision.mlmodelc  (GPU, .cpuAndGPU)
    │  Input:  pixel_values (1, 2520, 768) fp32
    │          pixel_position_ids (1, 2520, 2) int32
    │  Output: image_features (1, 280, 1536) fp16
    │  Includes: vision_tower + pooler + embed_vision.embedding_projection (768→1536)
    │
    ▼
Feature injection into LLM decode/prefill
    │  At each IMAGE_TOKEN_ID (258880) position in the prompt:
    │    hidden_states[i] = image_features[imageIdx]
    │    per_layer_raw[i] = ZERO (not from token lookup!)
    │
    ▼
chunk1-4 / prefill_chunk1-4  (ANE, .cpuAndNeuralEngine)
    │  Normal transformer decode with injected vision features
    ▼
Generated text
```

## Token IDs

| Token | ID | Role |
|-------|------|------|
| `<\|image>` (BOI) | 255999 | Begin-of-image marker (text token, normal embedding). Shared by image and video frames. |
| `<\|image\|>` | 258880 | **Image** placeholder (replaced with vision features) |
| `<\|video\|>` | 258884 | **Video frame** placeholder (replaced with vision features). Distinct id from `<\|image\|>` so the model treats the sequence as a video, not a series of stills. |
| `<image\|>` (EOI) | 258882 | End-of-image marker (text token, normal embedding). Shared by image and video frames. |

## Prompt Format

Single square image:
```
<bos><|turn>user\n<|image><|image|>×256<image|>\nDescribe this image<turn|>\n<|turn>model\n
```

Video clip (N frames at fps=1.0, tokensPerFrame=64):
```
<bos><|turn>user\n00:00 <|image><|video|>×64<image|> 00:02 <|image><|video|>×64<image|> … MM:SS <|image><|video|>×64<image|>\nDescribe this video<turn|>\n<|turn>model\n
```

BOI and EOI are regular text tokens — they get their normal text embeddings. Only the placeholder positions (`<|image|>` for images, `<|video|>` for video frames) get vision encoder features injected. Both go through the same `imgFeats` buffer in Swift; the prefill / decode paths in `ChunkedEngine` and `CoreMLLLM` recognize either id as a substitution slot.

## HF vs Our Approach

HuggingFace's `Gemma4ForConditionalGeneration` does NOT insert 256 `<|image|>` tokens into `input_ids`. Instead:
- `input_ids` has only ~13 tokens (BOI marker + text)
- A separate `mm_token_type_ids` mask tells the model where to insert image features
- The model internally expands the BOI marker into 256 feature positions

We take a different approach:
- Insert 256 `<|image|>` tokens explicitly in the prompt
- Tokenize normally (271 tokens for a simple image prompt)
- At inference time, substitute the hidden states at IMAGE positions with vision features

Both approaches are mathematically equivalent **IF the per-layer embedding (PLE) is handled correctly** — see below.

## Bug: PLE Corruption at Image Positions (Fixed in v0.3.1)

### Symptom
Model says "I have provided a string of characters that resembles a sequence of letters" — treating vision features as garbled text.

### Root Cause
For image token positions, the code was computing `per_layer_raw` from `tokenID = 0` (PAD) or `tokenID = 258880` (IMAGE). Both tokens have non-zero PLE embeddings (norm ≈ 94), which corrupted the per-layer input for all 256 image positions.

In Gemma 4, the per-layer combined input is:
```
per_layer_combined = per_layer_model_projection(hidden_states) * scale + per_layer_raw * inputScale
```

For image positions:
- `per_layer_model_projection(vision_features)` — correct (done inside chunk1 on ANE)
- `per_layer_raw` — should be ZERO, was non-zero garbage

### Fix
Set `per_layer_raw = zeros` for any position where `tokenID == IMAGE_TOKEN_ID`:
- **Decode path**: `predictStep()` creates a zero MLMultiArray when `imageEmbedding` is non-nil
- **Prefill path**: `buildPrefillPLR()` skips `embedPerLayer.lookupRaw()` for IMAGE positions (memset zero from init)

## Bug: Image Token Count 280 vs 256 (Fixed in v0.3.0)

The vision encoder output tensor is always `(1, 280, 1536)` (`max_soft_tokens = 280`), but for a square 768×768 input, only the first 256 tokens are real. Tokens 256–279 are zero padding inside the encoder.

The prompt previously inserted 280 `<|image|>` placeholders, feeding 24 zero hidden states to the LLM. These zeros confused the model into saying "I can't describe this image."

Fix: use 256 placeholders for square images.

## Vision Encoder Verification

Tested on Mac (2026-04-10): CoreML vision output matches HuggingFace Python output with **cosine similarity 0.993** across all 256 tokens. The small difference (0.007) is from:
- Resize algorithm: PIL LANCZOS vs CoreGraphics bicubic (max pixel diff: 0.047 ≈ 12/255)
- fp16 quantization drift in CoreML

The vision encoder conversion is correct.

## Image Preprocessing Details

Matches HuggingFace `Gemma3nImageProcessor` (`processor_config.json`):
- `do_rescale`: true (÷255, range [0, 1])
- `do_normalize`: **false** (no mean/std normalization)
- `do_resize`: true (aspect-ratio preserving, sides rounded to multiples of 48)
- `do_convert_rgb`: true
- Patch extraction: 16×16, channels-last, row-major
- Position IDs: meshgrid (x, y) = (px, py), padding positions = -1

## Video (v0.4.0 — Phase 1, mobile-scoped)

Gemma 4's video path is conceptually "a sequence of image frames, each with a
timestamp, optionally plus an audio track." No new CoreML model is required
on our side — the existing `vision.mlpackage` runs per frame and the existing
`audio.mlpackage` handles the soundtrack.

### Pipeline

```
video.mp4 / .mov
    │
    ▼
VideoProcessor.extractFrames(options.fps, options.maxFrames)
    │  AVAssetImageGenerator, aspect-ratio preserving.
    │  `maxFrames` distributed evenly across the full clip duration;
    │  `fps` caps the sampling rate so short clips don't duplicate frames.
    │
    ▼
ImageProcessor.process() × N frames       (optional audio branch)
    │                                     VideoProcessor.extractAudioPCM16k
    │                                     AudioProcessor.process
    ▼
concatFrameFeatures() → (1, N×256, 1536)  audio_features
    │
    ▼
Gemma video chat template:
  MM:SS <|image><|image|>×256<image|>  MM:SS <|image>…<image|>  ...
  (frames space-joined; audio block appended if present)
    │
    ▼
ChunkedEngine prefill/decode with imageNumTokens = N×256
```

### Per-frame chat-template block

Matching HF `Gemma4Processor.apply_chat_template` for video messages:

```
{MM:SS} <|image><|image|>×n_tokens<image|>
```

Frames are joined by single spaces, audio block (if any) goes after.

### Why `n_tokens = 64` (not 256)

Gemma 4's processor config exposes two separate token budgets:

| processor        | `max_soft_tokens` | ≈ real tokens/frame (square) |
|------------------|-------------------|------------------------------|
| `image_processor` | 280              | 256 (16×16 grid)             |
| `video_processor` | 70               | 64  (8×8 grid)               |

The shipped `vision.mlpackage` was built for the still-image path, so it
always emits 280 tokens per frame (256 real). Feeding those 256 tokens per
frame into the LLM as if they were video produced garbage output — the
model was trained expecting ~64 tokens per frame for video and got
confused by ×4 the expected density, emitting EOS after ~100 characters.

When `vision_video.mlpackage` is present (produced by
`convert_gemma4_multimodal.py --video-vision`), CoreMLLLM uses it
directly and the encoder itself emits 64 tokens per frame — this is the
correct video-grade path and matches the HF forward at cosine=1.0000.

For model bundles that predate Phase 2, `tokensPerFrame = 64` falls back
to 2×2-average-pooling each frame's 16×16 still-image token grid down
to 8×8 in fp32 on-the-fly in Swift. Output length jumps from ~100 chars
to ~900 and the model starts explicitly referencing frames by their
`MM:SS` timestamps; the pool is a drop-in approximation of the
video-grade encoder but is expected to drift more under motion.

### Mobile context-length budget

Each frame costs ~261 tokens (256 placeholders + BOI + EOI + `MM:SS` + space).

| Chunk size | Max frames fittable | Recommended `maxFrames` |
|------------|---------------------|--------------------------|
| 512        | ~1                  | 1 (use the still-image path instead) |
| 2048       | ~7                  | 6                        |
| 8192       | ~30                 | 24                       |

`VideoProcessor.Options(fps:maxFrames:includeAudio:)` defaults to
`fps=1.0, maxFrames=8, includeAudio=false` — safe on a 2K chunk.
`maxFrames` is the actual count target (frames are spaced evenly across
the clip); `fps` caps the rate so clips shorter than `maxFrames / fps`
seconds don't produce near-duplicate samples.

For longer clips on mobile, `vision_video.mlpackage` provides the
low-token-budget encoder variant (`max_soft_tokens=70`, ≈64 real
tokens/frame) so 30–60 s at 1 fps fits in 8K. Build it by re-running
`convert_gemma4_multimodal.py --video-vision`.

### Usage

```swift
let llm = try await CoreMLLLM.load(model: .gemma4e2b)
let answer = try await llm.generate(
    "What is happening in this clip?",
    videoURL: URL(fileURLWithPath: "/tmp/clip.mp4"),
    videoOptions: .init(fps: 1.0, maxFrames: 6, includeAudio: true)
)
```

## Files

| File | What |
|------|------|
| `Sources/CoreMLLLM/ImageProcessor.swift` | Image preprocessing + vision encoder call |
| `Sources/CoreMLLLM/AudioProcessor.swift` | Mel spectrogram + Conformer + projection |
| `Sources/CoreMLLLM/VideoProcessor.swift` | AVFoundation frame + audio extraction |
| `Sources/CoreMLLLM/ChunkedEngine.swift` | Feature injection in decode/prefill paths |
| `conversion/models/gemma4_vision.py` | Vision weight extraction (not CoreML conversion) |
| `conversion/output/gemma4-mobile/vision.mlpackage` | CoreML vision encoder |
| `docs/MULTIMODAL.md` | This file |
