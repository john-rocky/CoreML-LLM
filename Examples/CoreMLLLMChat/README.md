# CoreMLLLMChat

Sample iOS app built with the **CoreMLLLM** Swift package. Demonstrates text chat, image understanding, audio input, and model management — all running on-device via Apple Neural Engine.

The app is a thin UI layer over the package. All inference, audio/image processing, and model downloading is handled by `import CoreMLLLM`.

## Setup

### 1. Open in Xcode

```bash
open Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj
```

The project references the local `CoreMLLLM` package (relative path `../..`). Xcode resolves it automatically.

### 2. Build

- Set development team in Signing & Capabilities
- Build to device (iOS 18+, iPhone or iPad)

### 3. Use

- Tap **"Get Model"** → select and download a model
- Chat with text
- Tap the photo icon to attach an image (Gemma 4)
- Tap the video icon to attach a clip (Gemma 4)
- Tap the mic icon to record audio (Gemma 4)

### 4. Trying the Phase 2 video encoder (optional)

The `feat/gemma4-video-input` branch adds a purpose-built video vision
encoder (`vision_video.mlpackage`, ~323 MB, 64 tokens/frame natively)
that replaces the Phase 1 Swift-side 2×2 pool. The HF release does
not yet carry the artifact, so the Phase 2 path only activates when
the file is present on-device. Without it, the existing pool fallback
keeps working unchanged.

**Build it** (Mac, ~5 min — env details in
`docs/VIDEO_PHASE2_CONTINUATION.md`):

```bash
# From the worktree root:
uv venv --python 3.11 .venv-phase2
source .venv-phase2/bin/activate
uv pip install 'git+https://github.com/huggingface/transformers@main' \
               torch==2.7.0 coremltools==9.0 accelerate pillow numpy
python conversion/phase2/trace_video_vision.py ~/Downloads/vision_video_out
xcrun coremlcompiler compile \
    ~/Downloads/vision_video_out/vision_video.mlpackage \
    ~/Downloads/vision_video_out/
```

**Sideload** next to the existing Gemma 4 bundle (full workflow in
`docs/USB_MODEL_SIDELOAD.md`):

```bash
DEVICE=$(xcrun devicectl list devices | awk '/connected/{print $3}' | head -1)
xcrun devicectl device copy to \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier com.example.CoreMLLLMChat \
    --source ~/Downloads/vision_video_out/vision_video.mlmodelc \
    --destination Documents/Models/gemma4-e2b/vision_video.mlmodelc
```

Relaunch the app. `CoreMLLLM.load(from:)` auto-detects
`vision_video.mlmodelc` and routes the video picker's frames through
it; when absent, the Phase 1 pool fallback is used transparently.

## Architecture

```
┌───────────────────────────────────────────────────┐
│  App (UI layer only)                              │
│  ChatView ─── LLMRunner ─── ModelPickerView       │
│  AudioRecorder            ChatMessage             │
└───────────┬───────────────────────┬───────────────┘
            │                       │
            ▼                       ▼
┌───────────────────┐   ┌───────────────────────┐
│  CoreMLLLM        │   │  ModelDownloader      │
│  .load()          │   │  .download()          │
│  .stream()        │   │  .pause() / .resume() │
│  .generate()      │   │  Background URLSession│
│  AudioProcessor   │   └───────────────────────┘
│  ImageProcessor   │
│  ChunkedEngine    │
└───────────────────┘
      CoreMLLLM Swift Package
```

### Key files

| File | Lines | Role |
|------|------:|------|
| `LLMRunner.swift` | ~280 | `@Observable` wrapper around `CoreMLLLM`. Adds benchmark, ANE verification, memory report |
| `ChatView.swift` | ~400 | SwiftUI chat interface with image picker, audio recorder, benchmark UI |
| `ModelPickerView.swift` | ~150 | Model download/selection UI with pause/resume/cancel |
| `AudioRecorder.swift` | ~90 | AVAudioEngine microphone capture → `[Float]` PCM samples |
| `ChatMessage.swift` | ~16 | UI data model (role, content, image thumbnail) |
| `CoreMLLLMChatApp.swift` | ~25 | App entry + background download handler |

## Features

- **Streaming generation** with real-time tok/s display
- **Multi-turn conversation** with chat history
- **Image understanding** via PhotosPicker (Gemma 4 multimodal)
- **Audio understanding** via microphone recording (Gemma 4 multimodal)
- **Background model download** with pause/resume (survives app backgrounding)
- **Battery benchmark** — sustained generation with SoC drain tracking
- **ANE verification** — `MLComputePlan` op placement report
- **Memory diagnostics** — `task_vm_info` phys_footprint

## Requirements

- iOS 18+ (CoreML stateless KV path)
- iPhone 15 Pro or newer recommended (8 GB RAM)
- Xcode 16+
