# CoreMLLLMChat

iOS chat app with on-device LLM inference using CoreML (ANE+GPU).

Supports text chat (Qwen2.5, Gemma 4) and image understanding (Gemma 4 multimodal).

## Setup

### 1. Open in Xcode

```bash
open Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj
```

### 2. Build

- Set development team in Signing & Capabilities
- Build to device (iOS 18+, iPhone or iPad)
- Xcode will resolve swift-transformers dependency automatically

### 3. Use

- Tap **"Get Model"** → select and download a model
- Chat with text
- For Gemma 4 multimodal: tap the photo icon to attach an image

## Features

- **swift-transformers tokenizer** — proper BPE tokenization via HuggingFace
- **Streaming generation** — tokens appear as generated, with tok/s display
- **Image input** — PhotosPicker for Gemma 4 multimodal image captioning
- **Model download** — downloads from GitHub Releases with progress
- **Qwen/Gemma chat templates** — auto-detected from model config

## Requirements

- iOS 18+ (MLState API)
- iPhone 15 Pro or newer recommended (8 GB RAM)
- Xcode 16+
