# Audio (Speech Understanding) — Architecture & Notes

How Gemma 4 E2B processes audio on-device, and why the pipeline is split across CoreML and Swift/Accelerate.

## Pipeline Overview

```
Raw PCM (Float, mono, 16 kHz)
    │
    ▼
AudioProcessor.computeMelSpectrogram()           Swift / Accelerate
    │  frame_length=320, hop=160, fft=512, n_mel=128
    │  log-mel with floor 1e-3, 0.5× magnitude scaling,
    │  Hann window, reflection pad on the left (frame_length/2).
    │  Shape: (1, T, 128) fp16. T is fixed per mlpackage.
    │
    ▼
audio.mlmodelc   (12-layer Conformer)            CoreML (ANE preferred)
    │  Input:  input_features (1, T, 128) fp16
    │  Output: hidden_states  (1, S, 1024) fp16      S ≈ T / 4
    │  Includes: SubSampleConv → 12× Conformer blocks
    │  Excludes: output_proj, RMSNorm(no scale), embed_proj   ← see below
    │
    ▼
AudioProcessor.projectHiddenStates()             Swift / Accelerate (fp32)
    │  output_proj   (1024 → 1536)   cblas_sgemm + bias
    │  RMSNorm       (no weight, eps=1e-6)   vDSP_svesq + vDSP_vsmul
    │  embed_proj    (1536 → 1536)   cblas_sgemm
    │  fp32 internally, fp16 on the way in and out.
    │
    ▼
Feature injection into LLM decode/prefill        ChunkedEngine.swift
    │  At each AUDIO_TOKEN_ID (258881) position in the prompt:
    │    hidden_states[i] = audio_features[audioIdx]
    │    per_layer_raw[i] = 0            (same rule as vision)
    │
    ▼
chunk1-4 / prefill_chunk1-4 (ANE)
    ▼
Generated text
```

`S ≈ T / 4` because SubSampleConv downsamples 4×. Each output token is ≈40 ms of audio.

| T (mel frames) | S (audio tokens) | Approx. audio length |
|---:|---:|---|
| 200  | 50  | ~2 sec  |
| 500  | 125 | ~5 sec  |
| 3000 | 750 | ~30 sec |

## Token IDs (Gemma 4 tokenizer)

| Token | ID | Role |
|---|---:|---|
| `<\|audio\|>` (BOA) | 256000 | Begin-of-audio marker (text token, normal embedding) |
| `<\|audio\|>` (placeholder) | 258881 | Audio placeholder — replaced with encoder features |
| `<audio\|>` (EOA) | 258883 | End-of-audio marker |

## Why the Projection Runs in Swift (not CoreML)

The Gemma 4 audio tower ends with:

```
output_proj  : Linear(1024 → 1536) with bias
norm         : RMSNorm without learnable scale (scale=1 implicit)
embed_proj   : Linear(1536 → 1536)
```

Shipping this inside the CoreML graph causes the **RMSNorm-without-scale** op to emit all-zero outputs on the CoreML GPU runtime. ANE does not support all the ops in this tail block cheaply either. Running it in Swift/Accelerate with fp32 precision gives:

- Correct numerics (validated with cosine similarity > 0.99 vs HF)
- Predictable performance (`cblas_sgemm` is faster than small per-token CoreML predictions)
- No GPU fallback for the decoder itself — ANE placement stays at 99.78%

The conformer body (12 layers, almost all the compute) still runs on ANE.

## Fixed-Shape Conformer Attention (conversion-time gotcha)

HF `Gemma4AudioAttention` uses `.shape`, `//`, and dynamic reshape, which produce `aten::Int` ops that coremltools cannot lower. `conversion/convert_audio.py :: FixedShapeConformerAttention` rewrites it with:

- Hardcoded `num_blocks`, `pad_amount`, `padded_seq`, `context_size`, `chunk_size`
- Precomputed relative-position embeddings as a registered buffer
- Precomputed 5D blocked attention mask (buffer, not a runtime call to `create_bidirectional_mask`)
- Trans-XL-style relative position shift expressed as `pad → reshape → slice → reshape`

These rewrites are why audio conversion has its own builder (`convert_audio.py`) rather than reusing `exporter.py`.

## Mel Spectrogram Matching

The Swift implementation (`Sources/CoreMLLLM/AudioProcessor.swift :: computeMelSpectrogram`) reproduces HF `Gemma4AudioFeatureExtractor` bit-for-bit:

- Hann window: `0.5 - 0.5·cos(2π·n / frame_length)`
- Reflection-style left pad: `frame_length / 2` zeros prepended
- 0.5× magnitude scaling on both real and imaginary FFT components (HF quirk)
- DC bin index 0 = `|real[0]|`; Nyquist index `fft_length/2` = `|imag[0]|` (packed in place as vDSP does)
- `log(mel + mel_floor)` with `mel_floor = 1e-3`

If you change any of these constants, compare a `.wav` through both the HF extractor and Swift to avoid silent drift.

## Inputs the App Needs on Disk

`convert_audio.py` writes four files under `--output`:

| File | Purpose |
|---|---|
| `audio.mlpackage` (or compiled `.mlmodelc`) | Conformer encoder |
| `output_proj_weight.npy` fp16 (1536, 1024) | Swift-side gemm |
| `output_proj_bias.npy` fp16 (1536,) | Swift-side bias add |
| `embed_proj_weight.npy` fp16 (1536, 1536) | Swift-side gemm |
| `audio_config.json` | Feature-extractor constants + token IDs + `mel_frames` / `num_tokens` |

The mel filterbank itself is expected as a 257×128 float32 binary (`loadMelFilterbank`), produced once and shipped with the model.

## Fixed vs Variable Audio Length

CoreML requires static shapes on ANE. Current strategy: **one mlpackage per fixed `T`**. Options:

1. Ship the longest needed (e.g. `T=3000`, ~30 s) and zero-pad shorter clips. Simple, correct, but wastes Conformer compute on padding.
2. Ship a small `EnumeratedShapes` set (e.g. 200 / 1000 / 3000) and pick the smallest bucket that fits. This is ANE-compatible per Apple's docs.
3. Chunk long audio in Swift and concatenate outputs. Chunk boundary artifacts possible because the Conformer uses bidirectional local attention; validate on a speech benchmark before shipping.

`convert_audio.py --mel-frames` controls which bucket is produced. There is no runtime dynamic-length support yet.

## Chunked Variant

`conversion/convert_audio_chunked.py` splits the 12 Conformer layers into three 4-layer CoreML models. Purpose: let the ANE compiler plan each chunk independently (>15 layers per chunk can hang the compiler) and reduce peak memory during conversion. Not required for inference correctness — keep the monolithic variant unless you hit compile failures.

## Verification Checklist

1. Load any Gemma 4 E2B audio sample through HF and through this pipeline.
2. Compare `hidden_states` (pre-projection): cosine similarity ≥ 0.99, max abs error ≤ 5e-3 on fp16.
3. Compare final `audio_features` after Swift projection: cosine similarity ≥ 0.99 vs HF full tower.
4. Run a real transcription prompt end-to-end; check the model emits plausible text rather than repeats or garbage.

Items 2 and 3 are automated inside `convert_audio.py :: verify_output`. Item 4 is manual.
