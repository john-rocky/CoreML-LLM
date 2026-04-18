# CoreML-LLM

**On-device LLMs on Apple Neural Engine** — Run **Gemma 4** on iPhone with CoreML, ANE-optimized, no server.

CoreML-LLM targets the **Apple Neural Engine** rather than the GPU, making it a good fit for always-on, battery-friendly inference. [MLX Swift](https://github.com/ml-explore/mlx-swift) is the best choice when you want maximum throughput from the GPU; CoreML-LLM is the answer when you want the LLM to live on the ANE so the GPU stays free.

> **v0.7.0** — Video multimodal: native video vision encoder (64 tokens/frame), uniform frame sampling with per-frame thumbnails, `<|video|>` placeholder + bidirectional vision group attention. See [What's new](#whats-new).

| Text | Image | Audio (v0.6) | Video (v0.7) |
|------|-------|--------------|--------------|
| ![text](https://github.com/user-attachments/assets/67584300-ce34-4aa5-b3bd-5521cfe8855a) | ![multimodal](https://github.com/user-attachments/assets/2a869bf5-8315-422d-8b06-a4a7edecd173) | <video src="https://github.com/user-attachments/assets/e8deb6d0-d8b0-4210-885c-5d7a7ddc7ad3" controls></video> | ![video](https://github.com/user-attachments/assets/1d2a9ff3-2912-40e9-895d-fbaa3c73ee3a) |

## Performance (Gemma 4 E2B, iPhone 17 Pro)

| | v0.1.0 | v0.2.0 | v0.3.0 | v0.4.0 | v0.5.0 | v0.6.2 | **v0.7.0** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Context length | 512 | 2048 | 2048 | 2048 | 2048 | 2048 | **2048** |
| Decode speed | ~11 tok/s | ~11 tok/s | ~28 tok/s | ~28 tok/s | ~31 tok/s | ~31 tok/s | **~31 tok/s** |
| Prefill | ~11 tok/s | ~175 tok/s | ~96 tok/s | ~96 tok/s | ~154 tok/s | ~154 tok/s | **~154 tok/s** |
| Multimodal (image) | — | — | broken | working | working | working | **working** |
| Multimodal (audio) | — | — | — | — | — | working | **working** |
| Multimodal (video) | — | — | — | — | — | — | **working** |
| ANE placement | — | — | 99.78% | 99.78% | 99.78% | 99.78% | **99.78%** |
| Memory (`phys_footprint`) | — | — | — | ~1 GB | ~1 GB | ~1 GB | **~1 GB** |

Context length is 2048. Extended context (8K) is under active development (see `docs/SPEED_8K.md` for the roadmap) but not yet stable for shipping: 8K chunk conversion and the inference path are still in flight. The shipped model on HuggingFace is ctx=2048.

Ground-truth ANE placement measured via `MLComputePlan` (7,294 / 7,310 dispatched LLM ops on ANE). Vision encoder runs on GPU by design.

> **Memory:** ~1 GB `phys_footprint` (the iOS jetsam basis), measured via `task_vm_info`. Previous versions of this README quoted ~250 MB from Xcode's memory gauge, which underreports when CoreML loads INT4 palettized weights. The actual number is ~873 MB after load, ~981 MB during inference. `os_proc_available` remains ~5 GB on iPhone 17 Pro (8 GB RAM).

## Pre-converted Models

| Model | Size | Multimodal | Download |
|-------|------|------------|----------|
| **Gemma 4 E2B** | 3.1 GB | Image + Video + Audio + Text | [HuggingFace](https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml) |
| Qwen2.5-0.5B | 302 MB | Text only | [HuggingFace](https://huggingface.co/mlboydaisuke/qwen2.5-0.5b-coreml) |

The iOS sample app downloads models automatically. You can also convert your own.

## Quick Start

### Try it without building — App Store

If you just want to try Gemma 4 on your iPhone, grab **Models Zoo** — a pre-built app that ships CoreML-LLM:

[<img src="https://toolbox.marketingtools.apple.com/api/v2/badges/download-on-the-app-store/black/en-us?releaseDate=1735689600" alt="Download on the App Store" height="48">](https://apps.apple.com/jp/app/models-zoo/id6762083207)

### iOS App (build from source)

```bash
open Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj
```

Set your development team → build to an iOS 18+ device → **Get Model** → Download → Chat.

The app uses `.cpuAndNeuralEngine` to force ANE execution.

### Swift Package

```swift
dependencies: [
    .package(url: "https://github.com/john-rocky/CoreML-LLM", from: "0.6.0"),
]
```

```swift
import CoreMLLLM

// Download (if needed) + load in one line
let llm = try await CoreMLLLM.load(model: .gemma4e2b) { status in
    print(status)  // "Downloading...", "Loading chunks...", "Ready"
}

// Or load from a local directory
// let llm = try await CoreMLLLM.load(from: modelDirectory)

// Simple generation
let answer = try await llm.generate("What is the capital of France?")

// Streaming
for await token in try await llm.stream("Tell me a story") {
    print(token, terminator: "")
}
print("\(llm.tokensPerSecond) tok/s")

// Multi-turn conversation
let messages: [CoreMLLLM.Message] = [
    .init(role: .user, content: "Hi!"),
    .init(role: .assistant, content: "Hello!"),
    .init(role: .user, content: "What is 2+2?"),
]
for await token in try await llm.stream(messages) {
    print(token, terminator: "")
}

// Image understanding (Gemma 4)
let caption = try await llm.generate("Describe this image", image: cgImage)

// Audio understanding (Gemma 4)
let transcript = try await llm.generate("What did they say?", audio: pcmSamples)

// Video understanding (Gemma 4)
let analysis = try await llm.generate(
    "Describe this video frame by frame.",
    videoURL: URL(fileURLWithPath: "/path/to/clip.mp4"),
    videoOptions: .init(fps: 1.0, maxFrames: 6))
```

#### Model Download

```swift
import CoreMLLLM

// Background download with pause/resume
let model = ModelDownloader.ModelInfo.defaults[0]  // Gemma 4 E2B
let url = try await ModelDownloader.shared.download(model)

// Pause / Resume
ModelDownloader.shared.pause()
ModelDownloader.shared.resumeDownload()
```

Downloads continue in the background via `URLSessionConfiguration.background`. Resume data is persisted to disk.

Auto-detects model layout: chunked SWA (Gemma 4 E2B) or monolithic (Qwen2.5).

### Convert a Model

```bash
cd conversion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install scikit-learn

# Qwen2.5-0.5B (~2 min)
python convert.py --model qwen2.5-0.5b --output ./output/qwen2.5-0.5b

# Gemma 4 E2B (~15 min, 10 GB download)
python convert.py --model gemma4-e2b --output ./output/gemma4-e2b

# List available models
python convert.py --list
```

## What's new

Current release: **v0.7.0** ([release notes](https://github.com/john-rocky/CoreML-LLM/releases/tag/v0.7.0)).

### v0.7.0 — Video multimodal

- **Native video vision encoder** — `vision_video.mlmodelc` traces the HF vision tower at video-grade resolution (384×384, 64 tokens/frame) and ships as part of the Gemma 4 E2B bundle (3.1 GB). Parity vs HF forward: cosine = 1.0000. Falls back to Swift-side 2×2 pooling of the still-image encoder when absent.
- **Uniform frame sampling** — `maxFrames` are distributed evenly across the full clip duration instead of just the first N seconds. `fps` caps the sampling rate so short clips don't duplicate frames. Matches Gemma 4's `num_frames` semantic.
- **Per-frame thumbnails** — the chat bubble shows the exact frames the encoder received, with `MM:SS` captions. Lets you see what the model sees.
- **`<|video|>` placeholder (258884)** — uses the correct video-specific token id per HF's `Gemma4Processor`, with bidirectional attention within each frame's vision group during prefill.

### v0.6.2 — Audio multimodal

- **Audio multimodal** — Gemma 4 E2B can hear. Record on the phone, get an answer. 12-layer Conformer encoder, INT4-palettized, ANE-resident; features injected at `<|audio|>` placeholders via the same path as the vision encoder. See [docs/AUDIO.md](docs/AUDIO.md).
- **`ModelDownloader` in the library** — `import CoreMLLLM` → `ModelDownloader.shared`. 404-tolerant on optional `mlmodelc` metadata.
- **v0.6.2 download fix** — v0.6.0 / v0.6.1 pulled 8K chunks for a 2K model and hard-failed on load. v0.6.2 downloads the real 2K chunks (`swa/`, `prefill/`) and auto-invalidates stale caches on upgrade — no manual delete.

Full release history: [GitHub Releases](https://github.com/john-rocky/CoreML-LLM/releases).

## Architecture

```
      Prompt  ─┐
               ▼
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │ Prefill ch1  │─► │ Prefill ch2  │─► │ Prefill ch3  │─► │ Prefill ch4  │─► first token
        │ L0-7  + PLE  │   │ L8-14, kv13/ │   │ L15-24 shared│   │ L25-34 + LM  │
        └──────────────┘   │  kv14 out    │   └──────────────┘   └──────────────┘
               │           └──────────────┘          ▲                  ▲
               │                  │                  │                  │
               │                  └────────────┬─────┴──────────────────┘
               │                                          kv13_k/v, kv14_k/v
               ▼                                              (shared)
          writes K/V to persistent SWA caches
                      │
                      ▼  (decode loop, 1 token per step)
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │  Decode ch1  │─► │  Decode ch2  │─► │  Decode ch3  │─► │  Decode ch4  │─► next token
        └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### ANE Optimizations

| Technique | What | Why |
|-----------|------|-----|
| ANERMSNorm | `cat([x,-x])` → LayerNorm → slice | ANE has optimized LayerNorm; bare RMSNorm is slow |
| Conv2d Linear | `nn.Linear` → `nn.Conv2d(1)` | ANE executes Conv2d ~3× faster than MatMul |
| In-Model Argmax | Argmax inside the CoreML graph | Avoids shipping 256K logits from ANE to CPU |
| Manual softmax | `max/sub/exp/sum/div` with explicit fp16 casts | Prevents PyTorch's silent fp16→fp32 upcast in `torch.exp` |
| Pre-computed RoPE | cos/sin as model inputs, looked up in Swift | Eliminates `gather`/`greater_equal` (int ops → CPU) |
| Explicit KV I/O | Plain tensor inputs/outputs, no `MLState` | Avoids int64 state indices that break ANE placement |
| Sliding Window | Shift-based cache for 28/35 layers | O(W) per step instead of O(ctx) |
| Batched Prefill | One CoreML call for up to 512 tokens | Order-of-magnitude faster TTFT vs per-token |
| PLE in graph | Conv2d projection + per-layer norm | 8 ms → 1.8 ms/token |

### Why not MLX?

MLX Swift targets the Apple GPU (Metal) and is excellent when you want maximum throughput on a plugged-in Mac. CoreML-LLM targets the Apple Neural Engine instead — which matters when:

- You want the GPU to stay free for rendering, games, or other ML work
- You want the LLM to coexist with foreground apps without competing for the same silicon
- You want the model placed on the most power-efficient compute unit available on Apple devices

The two are complementary, not competing. If you're on a Mac and want to burn through a 70B model as fast as possible, use MLX. If you want a 2B model quietly running on ANE inside an iPhone app, this library is aimed at that case.

## Adding New Models

See [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) for a step-by-step guide and [docs/CONVERSION.md](docs/CONVERSION.md) for the full conversion reference (including quantization rationale and the `.mlpackage` → `.mlmodelc` deployment recipe).

## Documentation map

| Topic | File |
|---|---|
| How to convert HF weights, ANE tricks, precision checklist, INT4/INT8/W8A8 rationale, iPhone deployment recipe | [docs/CONVERSION.md](docs/CONVERSION.md) |
| Step-by-step guide to adding a new architecture | [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) |
| `.mlpackage` vs `.mlmodelc`, byte-level format gotchas | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| Image pipeline (vision encoder, token injection, debug notes) | [docs/MULTIMODAL.md](docs/MULTIMODAL.md) |
| Video pipeline (frame sampling, video encoder, prompt format) | [docs/VIDEO_PHASE2_CONTINUATION.md](docs/VIDEO_PHASE2_CONTINUATION.md) |
| Audio pipeline (mel spec, Conformer, Swift-side projection) | [docs/AUDIO.md](docs/AUDIO.md) |
| 8K context speed roadmap, ANE-compat matrix, reading list | [docs/SPEED_8K.md](docs/SPEED_8K.md) |
| EAGLE-3 speculative decoding: trained ckpt → iPhone deployment contract | [docs/EAGLE3_DEPLOY.md](docs/EAGLE3_DEPLOY.md) |
| Competitive landscape, energy/efficiency notes, spec-decoding background | [docs/RESEARCH.md](docs/RESEARCH.md) |
| Decision log for experimental variants (WFA, Flash, W8A8, Medusa, EAGLE-3, …) | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| Methodology behind the tok/s, ANE%, memory numbers | [docs/BENCHMARKING.md](docs/BENCHMARKING.md) |

## Project Structure

```
CoreML-LLM/
├── Package.swift
├── Sources/CoreMLLLM/                   # Swift Package (import CoreMLLLM)
│   ├── CoreMLLLM.swift                  # Public API — load, generate, stream
│   ├── ChunkedEngine.swift              # SWA 4-chunk decode + prefill engine
│   ├── EmbeddingLookup.swift            # INT8 quantized embedding lookup
│   ├── ImageProcessor.swift             # Vision encoder preprocessing (image + video)
│   ├── VideoProcessor.swift             # Frame extraction + audio track decode
│   ├── AudioProcessor.swift             # Mel spectrogram + audio encoder
│   ├── ModelDownloader.swift            # Background download with pause/resume
│   └── ModelConfig.swift
├── Examples/CoreMLLLMChat/              # iOS sample app (uses CoreMLLLM package)
│   └── CoreMLLLMChat/
│       ├── LLMRunner.swift              # @Observable wrapper around CoreMLLLM
│       ├── ChatView.swift               # SwiftUI chat UI
│       ├── ModelPickerView.swift         # Model download/selection UI
│       ├── AudioRecorder.swift          # Microphone recording
│       └── CoreMLLLMChatApp.swift
├── conversion/                          # Python conversion pipeline
│   ├── convert.py                       # CLI entry point
│   ├── ane_ops.py                       # ANE-optimized ops
│   ├── exporter.py                      # CoreML export + INT4 palettization
│   └── models/
│       ├── gemma4.py / gemma4_swa_chunks.py / gemma4_prefill_chunks.py
│       ├── gemma4_vision.py / gemma4_audio.py
│       └── qwen2.py
└── docs/
```

## Requirements

- **Conversion**: Python 3.10-3.12, coremltools 8+, PyTorch 2.2+
- **Inference**: iOS 18+ / macOS 15+ (MLState-free stateless path)
- **Sample app**: Xcode 16+

## License

MIT for the CoreML-LLM code. Gemma 4 weights are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
