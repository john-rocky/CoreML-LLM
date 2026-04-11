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
- Tap the mic icon to record audio (Gemma 4)

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
