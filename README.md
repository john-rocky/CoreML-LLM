# CoreML-LLM

**On-device LLMs on Apple Neural Engine** — Run **Gemma 4 E2B / E4B**, **Qwen3.5 0.8B / 2B**, and **Qwen3-VL 2B** on iPhone with CoreML, ANE-optimized, no server.

CoreML-LLM targets the **Apple Neural Engine** rather than the GPU, making it a good fit for always-on, battery-friendly inference. [MLX Swift](https://github.com/ml-explore/mlx-swift) is the best choice when you want maximum throughput from the GPU; CoreML-LLM is the answer when you want the LLM to live on the ANE so the GPU stays free.

> **v1.2.0** — **Qwen3-VL 2B** (text + vision on ANE: 28-layer GQA backbone + DeepStack-injected vision tower, 196 image tokens/picture, ~7.5 tok/s text decode on iPhone 17 Pro, image-conditioned prefill via DeepStack-aware chunk_0_vision).
> **v1.1.0** — **Qwen3.5 2B** (hybrid SSM + attention, ~17 tok/s decode on iPhone 17 Pro, 4 transformer chunks all on ANE, mmap'd fp16 embed sidecar keeps the app at ~200 MB phys_footprint while the 2B-param model runs).
> **v1.0.0** — **Qwen3.5 0.8B** (hybrid Gated-DeltaNet SSM + attention, ~20 tok/s decode on iPhone 17 Pro, 99.9% ANE, 0 GB sustained Metal heap). First hybrid SSM/attention LLM shipped on CoreML. See [Qwen3.5 performance](#qwen35-08b-new-v100).
> **v0.8.0** — **Gemma 4 E4B** (42 layers, hidden=2560, ~14 tok/s on iPhone 17 Pro, 100% ANE). Second model option alongside E2B; swap in the Models picker.
> **v0.7.0** — Video multimodal: native video vision encoder (64 tokens/frame), uniform frame sampling with per-frame thumbnails, `<|video|>` placeholder + bidirectional vision group attention.

<table>
  <tr>
    <td align="center" width="50%"><b>Text (E2B)</b><br><img src="https://github.com/user-attachments/assets/67584300-ce34-4aa5-b3bd-5521cfe8855a" width="100%"></td>
    <td align="center" width="50%"><b>Text (E4B, v0.8)</b><br><img src="https://github.com/user-attachments/assets/5d514739-8538-4048-bfce-78605de64e83" width="100%"></td>
  </tr>
  <tr>
    <td align="center"><b>Image</b><br><img src="https://github.com/user-attachments/assets/2a869bf5-8315-422d-8b06-a4a7edecd173" width="100%"></td>
    <td align="center"><b>Video (v0.7)</b><br><img src="https://github.com/user-attachments/assets/1d2a9ff3-2912-40e9-895d-fbaa3c73ee3a" width="100%"></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Audio (v0.6)</b><br><video src="https://github.com/user-attachments/assets/e8deb6d0-d8b0-4210-885c-5d7a7ddc7ad3" controls></video></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><b>Image (Qwen3-VL 2B, v1.2.0)</b><br><video src="https://github.com/user-attachments/assets/f9a257cf-cf10-4968-8964-b979b0a352d1" controls></video></td>
  </tr>
</table>

## Performance (iPhone 17 Pro)

### Gemma 4 E2B (shipping)

| | v0.1.0 | v0.2.0 | v0.3.0 | v0.4.0 | v0.5.0 | v0.6.2 | v0.7.0 | **v0.8.0** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Context length | 512 | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 | **2048** |
| Decode speed | ~11 tok/s | ~11 tok/s | ~28 tok/s | ~28 tok/s | ~31 tok/s | ~31 tok/s | ~31 tok/s | **~31 tok/s** |
| Prefill | ~11 tok/s | ~175 tok/s | ~96 tok/s | ~96 tok/s | ~154 tok/s | ~154 tok/s | ~154 tok/s | **~154 tok/s** |
| Multimodal (image) | — | — | broken | working | working | working | working | **working** |
| Multimodal (audio) | — | — | — | — | — | working | working | **working** |
| Multimodal (video) | — | — | — | — | — | — | working | **working** |
| ANE placement | — | — | 99.78% | 99.78% | 99.78% | 99.78% | 99.78% | **99.78%** |
| Memory (`phys_footprint`) | — | — | — | ~1 GB | ~1 GB | ~1 GB | ~1 GB | **~1 GB** |

### Gemma 4 E4B (v0.8.0, new)

| | E2B | **E4B** |
|---|---:|---:|
| Parameters | ~2B effective | **~4B effective** |
| num_hidden_layers | 35 | **42** |
| hidden_size | 1536 | **2560** |
| num_key_value_heads | 1 | **2** |
| Context length | 2048 | **2048** |
| Decode speed | ~31 tok/s | **~14 tok/s** |
| Per-step latency | ~32 ms | **~71 ms** |
| ANE placement | 99.78% | **100%** |
| Bundle size (INT4) | 3.1 GB | **5.5 GB** |

Context length is 2048 on both variants. Extended context (8K) is under active development (see `docs/SPEED_8K.md` for the roadmap) but not yet stable for shipping: 8K chunk conversion and the inference path are still in flight. The shipped model on HuggingFace is ctx=2048.

Ground-truth ANE placement measured via `MLComputePlan` (E2B: 7,294 / 7,310 dispatched LLM ops on ANE; E4B: 100%). Vision encoder runs on GPU by design.

### Qwen3.5 0.8B (new, v1.0.0)

First CoreML port of a hybrid SSM/attention LLM on iPhone. The 24 layers are 18× Gated-DeltaNet linear-attention (SSM) interleaved with 6× full GQA attention. Text-only.

| | Qwen3.5 0.8B |
|---|---:|
| Parameters | ~0.8B |
| Architecture | **Gated-DeltaNet SSM + attention (hybrid)** |
| num_hidden_layers | 24 (18 SSM + 6 full-attn) |
| hidden_size | 1024 |
| Context length | 128 (decode mlpackage mseq) |
| Decode speed (ANE) | **~20 tok/s** |
| Decode speed (GPU) | ~22 tok/s (bit-exact) |
| Prefill (ANE, non-stateful chunks) | ~170 tok/s (3× LiteRT-LM baseline) |
| ANE placement | 99.9% |
| Bundle size (fp16) | 1.4 GB |
| Metal heap (ANE mode, sustained) | **0 GB** |

Qwen3.5 is instruct-tuned — the app automatically wraps input in the chat template (`<|im_start|>`/`<|im_end|>`). On ANE, strict argmax-vs-fp32 top-1 match is 60% (argmax-fragility on 248K vocab), but oracle top-1 is in ANE top-3 for 100% of positions: sampling-mode generation is indistinguishable from fp32.

First load triggers a ~4 min on-device ANE E5 compile; subsequent loads are cached.

> **Memory:** ~1 GB `phys_footprint` (the iOS jetsam basis), measured via `task_vm_info`. Previous versions of this README quoted ~250 MB from Xcode's memory gauge, which underreports when CoreML loads INT4 palettized weights. The actual number is ~873 MB after load, ~981 MB during inference. `os_proc_available` remains ~5 GB on iPhone 17 Pro (8 GB RAM).

## Pre-converted Models

| Model | Size | Multimodal | Download |
|-------|------|------------|----------|
| **Qwen3-VL 2B** (new, v1.2.0) | 2.9 GB | Image + Text | [HuggingFace](https://huggingface.co/mlboydaisuke/qwen3-vl-2b-coreml) |
| **Gemma 4 E2B** | 3.1 GB | Image + Video + Audio + Text | [HuggingFace](https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml) |
| **Gemma 4 E4B** (v0.8.0) | 5.5 GB | Text only | [HuggingFace](https://huggingface.co/mlboydaisuke/gemma-4-E4B-coreml) |
| **Qwen3.5 2B** (v1.1.0) | 2.4 GB | Text only | [HuggingFace](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreML) |
| **Qwen3.5 0.8B** (v1.0.0) | 1.4 GB | Text only | [HuggingFace](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreML) |
| Qwen2.5-0.5B | 302 MB | Text only | [HuggingFace](https://huggingface.co/mlboydaisuke/qwen2.5-0.5b-coreml) |

The iOS sample app downloads models automatically. You can also convert your own.

**Which model to pick?**
- **Multimodal (image / video / audio)** → Gemma 4 E2B
- **Text-only, maximum quality on ≤3 GB** → Qwen3.5 2B (17 tok/s, ~200 MB phys_footprint)
- **Text-only, maximum quality** → Gemma 4 E4B (~4B effective params)
- **Text-only, fastest + smallest** → Qwen3.5 0.8B (20 tok/s, 754 MB bundle)

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

# Gemma 4 E4B — one-shot bundle builder (chunks + embeds + PLE + RoPE +
# tokenizer + model_config.json, ready for USB sideload or HF upload)
python build_gemma4_bundle.py --model gemma4-e4b --ctx 2048

# List available models
python convert.py --list
```

## What's new

Current release: **v1.2.0** ([release notes](https://github.com/john-rocky/CoreML-LLM/releases/tag/v1.2.0)).

### v1.2.0 — Qwen3-VL 2B (text + vision on ANE)

- **Qwen3-VL 2B on iPhone ANE** — 28-layer GQA backbone (hidden=2048, 16 heads / 8 KV heads, head_dim=128, vocab=151936) split into 4 INT8 body chunks + chunk_head. ~7.5 tok/s text decode at max_seq=2048 on iPhone 17 Pro. mmap'd fp16 `embed_weight.bin` keeps app ~200 MB phys_footprint. Bundle at [`mlboydaisuke/qwen3-vl-2b-coreml`](https://huggingface.co/mlboydaisuke/qwen3-vl-2b-coreml).
- **DeepStack-aware vision on ANE** — fixed-grid 448×448 vision encoder (196 image tokens / image) ships as a separate `vision.mlpackage` alongside a `chunk_0_vision.mlpackage` that injects the 3 DeepStack residual slices at text layers 0 / 1 / 2. On image-pad token steps the generator memcpy's the merger row into `hidden_in` and flips a `visual_active=1.0` gate; all other steps feed zeroed DeepStack buffers with the gate off so the graph stays static.
- **Chat template built as IDs** — vision prompts splice `<|vision_start|>` + 196 × `<|image_pad|>` + `<|vision_end|>` into the user turn as raw token IDs (151652 / 151655 / 151653) instead of relying on the tokenizer to round-trip the special strings.
- **2B pivot from VL4B** — the 4B path shipped at 1.7–6 tok/s on iPhone and OOM'd when merged to 3 chunks; 2B is 48% of the parameters on the same vision tower and fits the same 4-chunk × 7-layer ANE envelope that Qwen3.5 2B proved shippable in v1.1.0.

### v1.1.0 — Qwen3.5 2B (4-chunk ANE + mmap embed sidecar)

- **Qwen3.5 2B on iPhone ANE** — 24-layer hybrid SSM + attention (2.04 B params) split into 4 INT8 transformer chunks (6 layers each). Every chunk compiles on iPhone 17 Pro ANE at ≥ 90% op placement. ~17 tok/s decode, 2048-token context. First 2B-class hybrid LLM shipped on ANE. Bundle at [`mlboydaisuke/qwen3.5-2B-CoreML`](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreML).
- **mmap'd fp16 embed sidecar** — `embed_tokens.weight` ships as a raw 1 GB fp16 file (`embed_weight.bin`), Swift `mmap(..., MAP_PRIVATE)` with `MADV_RANDOM`. Only the handful of rows actually touched per prompt page in, and those pages stay "clean" so they don't count against `phys_footprint`. Reported memory during generation: **~200 MB** (vs ~2 GB with a CoreML chunk_embed mlpackage that would dequantize the full 1 GB into process memory).
- **Why 4 chunks (not 2)** — palettization shrinks disk, not ANE memory: INT8 weights re-expand to fp16 inside the ANE region, and iPhone ANE's per-mlprogram compile envelope rejected the 2-chunk split (~2 GB fp16/chunk) with `MILCompilerForANE: Couldn't communicate with a helper application` → silent GPU fallback at 3.4 GB Metal heap and 7 tok/s. 4 chunks at ≤ 1 GB INT8 (≤ ~1.7 GB fp16) each match the Gemma 4 E4B envelope that's proven ANE-resident.
- **2048-token context window** — bumped from 128 so chat turns don't truncate after ~10 lines. Only full-attention state scales with `max_seq` (6 layers × 2 × `max_seq` × 256 × fp16 × 2); +22 MB total vs the 128-token ceiling. Per-step SDPA compute scales linearly with max_seq but SDPA is ~5% of total per-step work — throughput cost is minor (17 → ~15 tok/s at 2048).
- **App binary shrunk ~5 GB** — removed stale Qwen3.5-0.8B mlpackages from the Xcode target (fp16 decode, fp16 prefill, stateful prefill, legacy 2-chunk a/b). Models arrive via HF download or `devicectl` sideload; nothing is compiled into the app binary.

### v1.0.0 — Qwen3.5 0.8B (hybrid SSM + attention)

- **Qwen3.5 0.8B on ANE** — hybrid Gated-DeltaNet SSM + attention model, 24 layers (18 linear-attention + 6 full-attention), 99.9% ANE operator placement. ~20 tok/s decode and ~170 tok/s prefill on iPhone 17 Pro. First CoreML port of a hybrid SSM/attention LLM we're aware of. Bundle at [`mlboydaisuke/qwen3.5-0.8B-CoreML`](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreML).
- **Zero sustained Metal heap** — decode + recurrent prefill run entirely on ANE; GPU stays free for the rest of the app. Total memory ~1.6 GB (mostly the fp16 weight mmap).
- **Integrated into the regular ChatView** — Qwen3.5 shows up in the main Models picker alongside Gemma; chat template is applied automatically (instruct-tuned model).
- **Swift marshaling optimizations** — 1.5× decode throughput vs the naive Core ML call pattern. Custom `MLFeatureProvider` delegates state handoff zero-copy, reusable `MLMultiArray` pool for per-step inputs, single-pass native Float16 argmax (NEON-accelerated). Applies to Gemma as well via the same runner.

### v0.8.0 — Gemma 4 E4B

- **Gemma 4 E4B** — 42 layers, hidden=2560, 2 KV heads, text-only decoder. ~14 tok/s at INT4 on iPhone 17 Pro with 100% ANE placement. Second shipping model alongside E2B; switch between them in the Models picker. Bundle published at [`mlboydaisuke/gemma-4-E4B-coreml`](https://huggingface.co/mlboydaisuke/gemma-4-E4B-coreml).
- **Generalized Gemma 4 conversion pipeline** — chunk boundaries and KV-producer layer indices are derived from the HF model config (`kv_sliding_producer` / `kv_full_producer` computed from `layer_types` + `num_kv_shared_layers`). Adding future Gemma 4 variants is now a registry entry away.
- **One-shot bundle builder** — `python conversion/build_gemma4_bundle.py --model gemma4-e4b --ctx 2048` produces a complete ready-to-ship directory (chunks + compiled `.mlmodelc` + INT8 embeds + INT8 PLE + RoPE + tokenizer + `model_config.json`).
- **Dynamic KV cache shapes in Swift** — `ChunkedEngine` reads `(slots, num_kv_heads)` from each chunk's `K_sliding_in` / `K_full_in` input description. Preserves E2B's (7/1, 5/2, nkv=1) exactly; enables E4B's (10/2, 10/2, nkv=2) with zero chunk-specific code.
- **Safer model switching** — `LLMRunner.loadModel` now releases the previous model before allocating the new one, avoiding a ~8 GB double-buffer peak during E2B ↔ E4B swaps.
- **Self-healing downloader** — skips prefill weight-sharing when prefill metadata wasn't part of the fetched file list; the engine cleans up zero-metadata `prefill_chunk*.mlmodelc` directories on launch.

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
