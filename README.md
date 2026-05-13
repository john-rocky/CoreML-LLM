# CoreML-LLM

**Run LLMs on iPhone / iPad with Core ML. Optimised for the Neural Engine + GPU.**

Quantized 1-3B parameter models running at 15-30 tokens/s on iPhone 15 Pro. Useful for: on-device summarisation, structured extraction, classification, function-calling. Zero per-token cost.

By [Daisuke Majima](https://john-rocky.github.io). Maintainer of [CoreML-Models](https://github.com/john-rocky/CoreML-Models) (1,749★).

💼 **Currently solo-lead on a 94k-line production iOS retail-vision app using on-device LLM in real-time. Open to Staff / Senior iOS / Mobile ML roles.** — [john-rocky.github.io](https://john-rocky.github.io)

---

# CoreML-LLM

**On-device LLMs on the Apple Neural Engine.** Run Gemma 4, Qwen3.5, Qwen3-VL, FunctionGemma, EmbeddingGemma, and Liquid AI's LFM2.5 on iPhone with CoreML — ANE-first, battery-friendly, no server.

Where [MLX Swift](https://github.com/ml-explore/mlx-swift) is the right call when you want maximum GPU throughput, CoreML-LLM is what you use when the LLM should live on the **ANE** so the GPU stays free for the rest of the app.

[![App Store](https://toolbox.marketingtools.apple.com/api/v2/badges/download-on-the-app-store/black/en-us?releaseDate=1735689600)](https://apps.apple.com/jp/app/models-zoo/id6762083207)

## Use in your app

Add the package, name a model, generate.

```swift
// Package.swift
.package(url: "https://github.com/john-rocky/CoreML-LLM", from: "1.9.0")
```

```swift
import CoreMLLLM

let llm = try await CoreMLLLM.load(repo: "lfm2.5-350m")
let answer = try await llm.generate("What is the capital of France?")
```

`repo:` accepts a registered model id (`"gemma4-e2b"`, `"qwen3.5-0.8b"`, `"lfm2.5-350m"`, …) or a full HuggingFace path — first call downloads, later calls reuse the on-device bundle. Streaming, multi-turn chat, image / video / audio, FunctionGemma, EmbeddingGemma → [package docs (Quick Start → Swift Package)](#swift-package).

## Models

| Model | Size | Task | iPhone 17 Pro decode | HuggingFace |
|---|---:|---|---:|---|
| **Gemma 4 E2B** | 5.4 GB (4.4 GB text-only) | Text + image + video + audio | **34.2 tok/s** | [mlboydaisuke/gemma-4-E2B-coreml](https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml) |
| **Gemma 4 E4B** | 8.16 GB multimodal / 5.5 GB text-only | Text + image + video + audio | **15.7 tok/s** | [multimodal](https://huggingface.co/mlboydaisuke/gemma-4-E4B-multimodal-coreml) · [text-only](https://huggingface.co/mlboydaisuke/gemma-4-E4B-coreml) |
| **Qwen3.5 2B** | 2.8 GB | Text | **~27 tok/s** | [mlboydaisuke/qwen3.5-2B-CoreML](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreML) |
| **Qwen3.5 0.8B** | 1.2 GB | Text | **~48 tok/s** | [mlboydaisuke/qwen3.5-0.8B-CoreML](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreML) |
| **Qwen3-VL 2B (stateful)** | 2.3 GB | Text + image | **~24 tok/s** | [mlboydaisuke/qwen3-vl-2b-stateful-coreml](https://huggingface.co/mlboydaisuke/qwen3-vl-2b-stateful-coreml) |
| **LFM2.5 350M** [†](#lfm2-license) | 810 MB | Text | **52 tok/s** | [mlboydaisuke/lfm2.5-350m-coreml](https://huggingface.co/mlboydaisuke/lfm2.5-350m-coreml) |
| **FunctionGemma-270M** | 850 MB | Function calling | (specialist) | [mlboydaisuke/functiongemma-270m-coreml](https://huggingface.co/mlboydaisuke/functiongemma-270m-coreml) |
| **EmbeddingGemma-300M** | 295 MB | Sentence embeddings | (specialist) | [mlboydaisuke/embeddinggemma-300m-coreml](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-coreml) |
| Qwen3-VL 2B (legacy) | 2.9 GB | Text + image | ~7.5 tok/s | [mlboydaisuke/qwen3-vl-2b-coreml](https://huggingface.co/mlboydaisuke/qwen3-vl-2b-coreml) |
| Qwen2.5 0.5B | 302 MB | Text | — | [mlboydaisuke/qwen2.5-0.5b-coreml](https://huggingface.co/mlboydaisuke/qwen2.5-0.5b-coreml) |

All numbers are iPhone 17 Pro A19 Pro, 2048-token context, ANE-only (no GPU fallback at runtime unless noted). Methodology: [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

**Which one should I pick?**
- Multimodal (image / video / audio), fastest → **Gemma 4 E2B** (34 tok/s)
- Multimodal, highest quality → **Gemma 4 E4B (multimodal)** (15.7 tok/s)
- Image + text chat, lowest memory + fastest follow-up → **Qwen3-VL 2B (stateful)**
- Text-only, maximum quality under ≤3 GB → **Qwen3.5 2B**
- Text-only, maximum quality → **Gemma 4 E4B (text-only)**
- Text-only, fast + chat-strong → **Qwen3.5 0.8B** (48 tok/s)
- Text-only, smallest at high tok/s on iPhone → **LFM2.5 350M** (52 tok/s, 810 MB) [†](#lfm2-license)
- Tool / function calling → **FunctionGemma-270M**
- Sentence embeddings / RAG → **EmbeddingGemma-300M**

## Demos

<table>
  <tr>
    <td align="center" width="50%"><b>Text</b><br><img src="https://github.com/user-attachments/assets/67584300-ce34-4aa5-b3bd-5521cfe8855a" width="100%"></td>
    <td align="center" width="50%"><b>Image</b><br><img src="https://github.com/user-attachments/assets/2a869bf5-8315-422d-8b06-a4a7edecd173" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>Video</b><br><img src="https://github.com/user-attachments/assets/1d2a9ff3-2912-40e9-895d-fbaa3c73ee3a" width="100%"></td>
    <td align="center"><b>Audio</b><br><video src="https://github.com/user-attachments/assets/e8deb6d0-d8b0-4210-885c-5d7a7ddc7ad3" controls></video></td>
  </tr>
</table>

## Quick Start

### Try it — App Store

**[Models Zoo](https://apps.apple.com/jp/app/models-zoo/id6762083207)** is a pre-built app shipping CoreML-LLM. Open it, pick a model, download, chat.

### Build from source

```bash
open Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj
```

Set your development team → build to an iOS 18+ device → **Get Model** → download → chat. Compute units default to `.cpuAndNeuralEngine` (ANE).

### Swift Package

```swift
dependencies: [
    .package(url: "https://github.com/john-rocky/CoreML-LLM", from: "1.9.0"),
]
```

```swift
import CoreMLLLM

// Download + load in one call
let llm = try await CoreMLLLM.load(model: .gemma4e2b) { print($0) }

// Simple / streaming / multi-turn
let answer = try await llm.generate("What is the capital of France?")
for await tok in try await llm.stream("Tell me a story") { print(tok, terminator: "") }

let messages: [CoreMLLLM.Message] = [
    .init(role: .user, content: "Hi!"),
    .init(role: .assistant, content: "Hello!"),
    .init(role: .user, content: "What is 2+2?"),
]
for await tok in try await llm.stream(messages) { print(tok, terminator: "") }

// Multimodal (Gemma 4)
let caption   = try await llm.generate("Describe this image", image: cgImage)
let transcript = try await llm.generate("What did they say?", audio: pcmSamples)
let analysis   = try await llm.generate(
    "Describe this video frame by frame.",
    videoURL: URL(fileURLWithPath: "/path/to/clip.mp4"),
    videoOptions: .init(fps: 1.0, maxFrames: 6))

// Fastest decode on iPhone 17 Pro A19 Pro: opt into the 3-chunk path.
// Set in the Xcode scheme: Environment Variables → LLM_3CHUNK = 1.
// +8.2 % tok/s, bit-equivalent to the default 4-chunk decode.
```

Downloads run in the background via `URLSessionConfiguration.background` with pause/resume support:

```swift
let url = try await ModelDownloader.shared.download(.gemma4e2b)
ModelDownloader.shared.pause()
ModelDownloader.shared.resumeDownload()
```

### FunctionGemma + EmbeddingGemma

Two specialists with their own narrow Swift APIs. Ship them alongside a chat model (Gemma 4, Qwen3.5) for tool calling + RAG.

```swift
import CoreMLLLM

let dir = FileManager.default
    .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]

// Function calling (850 MB, ≥ 92% ANE, batched prefill T=32)
let fg = try await FunctionGemma.downloadAndLoad(modelsDir: dir)
let (text, call) = try fg.generateFunctionCall(
    userPrompt: "Turn on the flashlight",
    tools: [[
        "type": "function",
        "function": [
            "name": "toggle_flashlight",
            "description": "Turn the phone flashlight on or off.",
            "parameters": ["type": "object", "properties": [:], "required": []],
        ],
    ]])
// call = "call:toggle_flashlight{}"

// Embeddings (295 MB, 99.80% ANE, Matryoshka 768/512/256/128)
let eg = try await EmbeddingGemma.downloadAndLoad(modelsDir: dir)
let vec = try eg.encode(text: "How do cats behave?",
                        task: .retrievalQuery, dim: 768)
```

Standalone sample at `Examples/Gemma3Demo/` imports `CoreMLLLM` and exercises both without pulling the Gemma 4 chat stack. Full I/O contracts in [docs/FUNCTIONGEMMA.md](docs/FUNCTIONGEMMA.md) + [docs/EMBEDDINGGEMMA.md](docs/EMBEDDINGGEMMA.md).

## Convert your own

```bash
cd conversion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Qwen2.5 0.5B (~2 min)
python convert.py --model qwen2.5-0.5b --output ./output/qwen2.5-0.5b

# Gemma 4 — one-shot bundle builder (chunks + embeds + PLE + RoPE +
# tokenizer + model_config.json, ready for USB sideload or HF upload)
python build_gemma4_bundle.py --model gemma4-e2b --ctx 2048
python build_gemma4_bundle.py --model gemma4-e4b --ctx 2048

# Gemma 4 E2B 3-chunk decode (default since v1.7, +8.2 % tok/s on iPhone A19 Pro)
python build_gemma4_3way.py --model gemma4-e2b --ctx 2048
python install_3way_bundle.py

# Specialists
python build_functiongemma_bundle.py --ctx 2048 --quantize int8 --prefill-t 32
python build_embeddinggemma_bundle.py --max-seq-len 128 --quantize int8

# LFM2.5 350M (Liquid AI hybrid attn + short-conv) — sideload-ready bundle
python build_lfm2_bundle.py --model lfm2.5-350m --l-pad 3
```

Step-by-step: [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md). Full reference (quant, `.mlpackage` → `.mlmodelc`, iPhone deployment): [docs/CONVERSION.md](docs/CONVERSION.md). LFM2-specific deep-dive (ChatML template, dual-state ANE blocker, fp16 short-conv drift): [docs/LFM2_CONVERSION_FINDINGS.md](docs/LFM2_CONVERSION_FINDINGS.md).

## Documentation

Design docs, benchmarks, and per-model conversion notes live in [docs/](docs/README.md). Start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the chunked decode design, ANE optimizations, MLX comparison, and project layout.

## What's new

Current release: **v1.9.0** ([release notes](https://github.com/john-rocky/CoreML-LLM/releases/tag/v1.9.0)).

- **v1.9.0** — Gemma 4 E4B multimodal (text + image + video + audio) on iPhone 17 Pro at **15.7 tok/s** decode. Topology II 3-chunk decode (`chunk1` + `chunk2_3way` + `chunk3_3way`) + legacy 4-chunk `prefill_b8` multifunction with vision-aware bidirectional mask. E4B-built `vision.ane.mlmodelc` (output `[1, 256, 2560]`) + Conformer audio + Swift two-stage projection (1024 → 1536 → 2560, non-square `embed_proj`). New picker entry "Gemma 4 E4B (multimodal)" auto-downloads from [`mlboydaisuke/gemma-4-E4B-multimodal-coreml`](https://huggingface.co/mlboydaisuke/gemma-4-E4B-multimodal-coreml) (~8.16 GB); text-only entry kept at the existing HF repo. Build + sideload guide: [docs/E4B_MULTIMODAL_BUILD.md](docs/E4B_MULTIMODAL_BUILD.md).
- **v1.8.0** — Qwen3.5 0.8B / 2B full-vocab rep_penalty masks iPhone A18 fp16 ANE reduction bias. 0.8B: 48 tok/s, 2B: 27 tok/s on iPhone 17 Pro, all clean output across English + Japanese. +45 % over the prior v1.x ceiling. See [docs/QWEN35_FULL_VOCAB_REP_PENALTY.md](docs/QWEN35_FULL_VOCAB_REP_PENALTY.md).
- **v1.7.0** — Gemma 4 E2B 3-chunk decode is the picker default + multimodal opt-out toggle. The new `gemma4e2b3way` ModelInfo ships `chunk2_3way` (L8-24 merged) + `chunk3_3way` (L25-34 + lm_head) and re-uses legacy `chunk1` + 4-chunk prefill graphs (vision-aware bidirectional mask preserved). Decode `c1+c2+c4` (chunk3 nil) — 3 ANE dispatches/step, **34.2 tok/s** on iPhone 17 Pro A19 Pro. The 4-chunk legacy entry stays as `Gemma 4 E2B (4-chunk legacy)`. ModelPickerView's "Download Options → Include multimodal" toggle drops vision/video/audio encoders + sidecars when off (~1 GB savings, text-only install). finishDownload now hardlinks shared decode↔prefill weights instead of copying (`chunk1↔prefill_chunk1` and `chunk3_3way↔prefill_chunk4`, **−682 MB on disk**).
- **v1.6.0** — Qwen3-VL 2B stateful Phase 2: cross-turn KV reuse + ANE prewarm. Same-prompt 2nd TTFT **4 s → 125 ms** (~32×), vision-chat 2nd-turn TTFT 125 ms (target was <500 ms). LCP-matched MLState resume + image-pinned-to-first-user-turn prompt builder + per-chunk dummy predict at load (231 ms total).
- **v1.5.0** — Qwen3-VL 2B stateful Phase 1: MLState + slice_update KV cache + multifunction prefill_b8. **24 tok/s decode at 256 MB phys_footprint** on iPhone 17 Pro (vs 7.5 tok/s / 1.7 GB on the v1.3 recurrent build — 3.2× decode, 6.4× memory drop). 4-chunk INT8 + fp16 embed sidecar.
- **v1.4.0** — Gemma 4 E2B 3-chunk decode (opt-in, `LLM_3CHUNK=1`): 31.6 → **34.2 tok/s** on iPhone 17 Pro A19 Pro (+8.2 %). Bit-equivalent to 4-chunk by construction. Closes the ANE-ceiling sweep for E2B; five additional lossless probes (SDPA fusion, K=V alias, Topology I boundary search, blockwise palettization, native softmax) all landed as negative results — see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
- **v1.3.0** — Qwen3-VL 2B (text + vision on ANE, 196 image tokens, DeepStack injection at L0/1/2, interleaved mRoPE for image tokens). 28-layer GQA, 2.9 GB bundle, ~7.5 tok/s text decode. (Recurrent KV — superseded by v1.5.0 stateful build; kept for backward compatibility.)
- **v1.2.0** — FunctionGemma-270M (function calling, batched prefill T=32) and EmbeddingGemma-300M (99.80 % ANE, Matryoshka 768/512/256/128). Standalone `Gemma3Demo` sample.
- **v1.1.0** — Qwen3.5 2B (4 INT8 chunks + mmap fp16 embed sidecar, ~200 MB phys_footprint for a 2B-param model).
- **v1.0.0** — Qwen3.5 0.8B (first hybrid SSM+attention LLM on CoreML, 99.9 % ANE).
- **v0.8.0** — Gemma 4 E4B (42-layer text decoder, 100 % ANE).
- **v0.7.0** — Video multimodal (native 384×384 vision encoder, 64 tokens/frame).
- **v0.6.2** — Audio multimodal (12-layer Conformer encoder).

Full history: [GitHub Releases](https://github.com/john-rocky/CoreML-LLM/releases).

## Requirements

- **Inference**: iOS 18+ / macOS 15+
- **Conversion**: Python 3.10–3.12, coremltools 8+, PyTorch 2.2+
- **Sample apps**: Xcode 16+

## License

MIT for the CoreML-LLM code. Model weights inherit the original licenses (Gemma weights: [Gemma Terms of Use](https://ai.google.dev/gemma/terms); Qwen weights: Apache 2.0; Qwen3-VL vision weights: Apache 2.0).

<a id="lfm2-license"></a>
**† LFM2.5 350M** weights are under [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE) (Liquid AI). Free for non-commercial use, research, and commercial use **up to a US $10M annual revenue threshold**. Above that threshold, see [Liquid AI](https://www.liquid.ai/) for a separate commercial license.
