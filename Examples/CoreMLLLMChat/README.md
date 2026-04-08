# CoreMLLLMChat

A minimal iOS chat app demonstrating on-device LLM inference using CoreML with ANE+GPU optimization.

## Setup

### 1. Convert a Model

```bash
cd ../../conversion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install scikit-learn
python convert.py --model qwen2.5-0.5b --context-length 512 --output ./output/qwen2.5-0.5b
```

### 2. Transfer Model to Device

Copy the output folder (containing `model.mlpackage`, `model_config.json`, and `hf_model/`) to your device via AirDrop, Files app, or Xcode.

### 3. Run the App

1. Open `CoreMLLLMChat.xcodeproj` in Xcode
2. Set your development team in Signing & Capabilities
3. Build and run on device (iOS 18+)
4. Tap "Load Model" and select the folder containing the model files

## Architecture

```
CoreMLLLMChatApp.swift  — App entry point
ChatView.swift          — SwiftUI chat interface with streaming
LLMRunner.swift         — CoreML model loading and inference loop
SimpleTokenizer.swift   — Minimal BPE tokenizer (loads tokenizer.json)
ChatMessage.swift       — Message data model
```

## Notes

- Requires iOS 18+ (MLState API for stateful KV cache)
- First model load compiles the CoreML model (may take 30-60 seconds)
- The SimpleTokenizer is a minimal implementation; for production, use [swift-transformers](https://github.com/huggingface/swift-transformers)
- Model files are NOT bundled — load from device storage at runtime

## Supported Models

- **Qwen2.5-0.5B** (302 MB, int4) — Fast, accurate
- **Gemma 4 E2B** (2.4 GB, int4) — Larger, multimodal text decoder
