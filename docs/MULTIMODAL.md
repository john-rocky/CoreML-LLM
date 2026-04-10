# Multimodal (Image Understanding) — Architecture & Debugging Notes

How Gemma 4 E2B processes images on-device via CoreML, and what went wrong along the way.

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
| `<\|image>` (BOI) | 255999 | Begin-of-image marker (text token, normal embedding) |
| `<\|image\|>` | 258880 | Image placeholder (replaced with vision features) |
| `<image\|>` (EOI) | 258882 | End-of-image marker (text token, normal embedding) |

## Prompt Format

For a single square image:
```
<bos><|turn>user\n<|image><|image|>×256<image|>\nDescribe this image<turn|>\n<|turn>model\n
```

BOI and EOI are regular text tokens — they get their normal text embeddings. Only the 256 `<|image|>` positions get vision encoder features injected.

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

## Files

| File | What |
|------|------|
| `Sources/CoreMLLLM/ImageProcessor.swift` | Image preprocessing + vision encoder call |
| `Sources/CoreMLLLM/ChunkedEngine.swift` | Feature injection in decode/prefill paths |
| `conversion/models/gemma4_vision.py` | Vision weight extraction (not CoreML conversion) |
| `conversion/output/gemma4-mobile/vision.mlpackage` | CoreML vision encoder |
| `docs/MULTIMODAL.md` | This file |
