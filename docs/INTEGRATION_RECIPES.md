# Integration Recipes — using CoreMLLLM from your app

The shortest path from "I have a Swift project" to "I'm running an LLM
on iPhone ANE." Code-first. Each recipe is copy-paste-able.

> **What this is.** A task-oriented companion to the top-level
> [`README.md`](../README.md) (which is model-oriented). If you're
> deciding *which* model to use, read the README. If you've decided
> *what you want to do* and need code, read this.

---

## 0. One-time setup

Add the package to your `Package.swift` (or via Xcode → Add Package):

```swift
.package(url: "https://github.com/john-rocky/CoreML-LLM", from: "1.9.0"),
```

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "CoreMLLLM", package: "CoreML-LLM"),
    ]
),
```

iOS target only: add the **Increased Memory Limit** entitlement
(`com.apple.developer.kernel.increased-memory-limit`) — required for
the 3 GB-class models (Gemma 4 E2B, Qwen 3.5 2B, Qwen 3-VL 2B).
Smaller models (EmbeddingGemma, FunctionGemma, LFM2.5 350M) don't need it.

HuggingFace gated models (FunctionGemma, EmbeddingGemma) need a token:
generate one at <https://huggingface.co/settings/tokens> and pass it to
`download(..., hfToken: "hf_...")`. Public models (Gemma 4, Qwen) work
without auth.

---

## 1. Recipe: chat (Gemma 4 / Qwen 3.5 / LFM2.5 / ...)

The unified facade — works for every general-purpose chat model
registered in the package.

```swift
import CoreMLLLM

let llm = try await CoreMLLLM.load(repo: "gemma4-e2b")
//                                       └─ or "qwen3.5-0.8b", "lfm2.5-350m", "qwen3.5-2b", ...

// One-shot
let answer = try await llm.generate("What is the capital of France?")

// Streaming
for try await piece in llm.stream("Write a haiku about Tokyo.") {
    print(piece, terminator: "")
}

// Multi-turn
let messages: [CoreMLLLM.Message] = [
    .init(role: .system, content: "You are a concise assistant."),
    .init(role: .user, content: "Explain ANE in one sentence."),
]
let reply = try await llm.generate(messages)
```

First call downloads the bundle (~1-5 GB depending on model) into the
app's caches dir; subsequent launches are instant. To inspect or pre-
warm, see `Gemma3BundleDownloader.localBundle(_:under:)`.

---

## 2. Recipe: function calling (FunctionGemma 270M)

For structured-output / tool-calling intents — *not* a Gemma 4
replacement. Emits `<start_function_call>...<end_function_call>` only.

```swift
import CoreMLLLM

let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
let fg = try await FunctionGemma.downloadAndLoad(
    into: cacheDir,
    hfToken: "hf_..."  // FunctionGemma is gated on HF
)

let tools: [[String: Any]] = [
    [
        "type": "function",
        "function": [
            "name": "set_timer",
            "description": "Set a countdown timer.",
            "parameters": [
                "type": "object",
                "properties": ["minutes": ["type": "integer"]],
                "required": ["minutes"],
            ],
        ],
    ],
    // ... more tools
]

let text = try fg.generate(
    messages: [["role": "user", "content": "Set a timer for 5 minutes"]],
    tools: tools,
    maxNewTokens: 64
)
// → "<start_function_call>call:set_timer{minutes:5}<end_function_call>"

if let call = fg.extractFunctionCall(from: text) {
    // call = "call:set_timer{minutes:5}"  — parse and dispatch to your handler
}
```

A typical app pattern: route structured/imperative user input through
FunctionGemma, fall back to Gemma 4 (recipe §1) for free-form chat.

See `docs/FUNCTIONGEMMA.md` for the architecture detail.

---

## 3. Recipe: semantic search / RAG (EmbeddingGemma 300M)

ANE 1-shot encoder, ~10 ms per embedding on Mac, ~50-70 ms on iPhone
17 Pro. 768-d unit-norm output, Matryoshka-truncatable to 512 / 256 / 128.

### Minimal — encode + cosine similarity

```swift
import CoreMLLLM

let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
let eg = try await EmbeddingGemma.downloadAndLoad(
    into: cacheDir,
    hfToken: "hf_..."  // EmbeddingGemma is gated on HF
)

let v1 = try eg.encode(text: "The cat sat on the mat.")
let v2 = try eg.encode(text: "A feline rested on the rug.")
// Both vectors are unit-norm, so dot product = cosine similarity.
let sim = zip(v1, v2).reduce(Float(0)) { $0 + $1.0 * $1.1 }
```

### RAG — build an index, retrieve top-K

```swift
struct Indexed { let id: UUID; let text: String; let vector: [Float] }

// Index a corpus (call once at app start, or as new items arrive).
let corpus: [String] = [
    "Carbonara: eggs, pancetta, pecorino.",
    "Pad thai: rice noodles, tamarind, fish sauce.",
    "Risotto: arborio rice, broth, parmesan.",
    // ... 100s-1000s OK; ~10 ms each on Mac
]
var index: [Indexed] = []
for text in corpus {
    let v = try eg.encode(text: text, task: .retrievalDocument)
    index.append(.init(id: UUID(), text: text, vector: v))
}

// At query time.
let query = "what was that egg pasta recipe"
let qv = try eg.encode(text: query, task: .retrievalQuery)
let topK = index
    .map { ($0, zip(qv, $0.vector).reduce(Float(0)) { $0 + $1.0 * $1.1 }) }
    .sorted { $0.1 > $1.1 }
    .prefix(3)

for (item, sim) in topK {
    print("\(String(format: "%.3f", sim)): \(item.text)")
}
```

Note the `task:` parameter — `.retrievalQuery` and `.retrievalDocument`
add asymmetric prompt prefixes that the model was trained with, and
give noticeably better retrieval than encoding both sides identically.

### Combine with a chat model (full RAG)

```swift
// Inject retrieved context as system message into Gemma 4.
let context = topK.map { $0.0.text }.joined(separator: "\n\n")
let llm = try await CoreMLLLM.load(repo: "gemma4-e2b")
let answer = try await llm.generate([
    .init(role: .system, content: "Answer using ONLY the following notes:\n\n\(context)"),
    .init(role: .user, content: query),
])
```

For corpus sizes > ~10 k items, swap the linear scan for HNSW or a
SQLite-backed index — the encoder itself stays the same.

See `docs/EMBEDDINGGEMMA.md` for the architecture detail.

---

## 4. Recipe: multimodal (image / video / audio in chat)

Gemma 4 E2B (and E4B multimodal) accept image / video / audio through
the same `.generate` / `.stream` entry points.

```swift
import CoreMLLLM
import CoreGraphics

let llm = try await CoreMLLLM.load(repo: "gemma4-e2b")

// Image — pass a CGImage.
let cg: CGImage = /* from UIImage.cgImage or PhotosUI */
let caption = try await llm.generate("What's in this image?", image: cg)

// Video — pass a file URL (mp4/mov).
let videoURL = URL(fileURLWithPath: "/path/to/clip.mp4")
let summary = try await llm.generate("Summarise this clip.", videoURL: videoURL)
```

Audio uses the same pattern via `audioURL:` — see the model card on HF
for accepted formats. The 5.4 GB E2B bundle contains the vision /
audio adapters; first load triggers the multimodal download.

---

## 5. Integration concerns

### Where to store bundle files

Default: `Gemma3BundleDownloader.download(_:into:)` writes to the
directory you pass. For iOS, use the **caches directory** so the system
can evict on storage pressure (re-download is one-shot, ~minutes):

```swift
let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
```

For "must keep forever" semantics use the app's Documents dir instead.

### Compute units

All loaders default to `.cpuAndNeuralEngine`. Override only if you're
benching or have a specific reason:

```swift
let eg = try await EmbeddingGemma.load(bundleURL: url, computeUnits: .cpuAndGPU)
```

ANE is the right default for power-efficient sustained workloads
(Gemma 4 was tuned for it); GPU is faster on a Mac for short bursts
but eats more battery on iPhone.

### Threading

All `.generate` / `.encode` calls are synchronous-on-the-current-thread
(the underlying CoreML `prediction(from:)` is). Wrap them in a task
to keep the UI responsive:

```swift
Task.detached {
    let answer = try await llm.generate(prompt)
    await MainActor.run { /* update UI */ }
}
```

### Concurrent models

Loading both Gemma 4 + EmbeddingGemma (a typical RAG setup) at the same
time on iPhone is fine with the Increased Memory Limit entitlement —
Gemma 4 E2B is ~3.7 GB resident, EmbeddingGemma is ~300 MB. They share
no runtime state, so calls can be interleaved.

### Pre-warming

First call after `load(...)` includes ANE compile (~20-30 s for Gemma 4,
~3-5 s for the small models). To warm up off the critical path:

```swift
// Right after load:
_ = try? await llm.generate("hi", maxTokens: 1)
```

---

## 6. Where to look next

- [`README.md`](../README.md) — model table, iPhone tok/s benchmarks
- [`docs/EMBEDDINGGEMMA.md`](EMBEDDINGGEMMA.md) — embedding model internals
- [`docs/FUNCTIONGEMMA.md`](FUNCTIONGEMMA.md) — function-call model internals
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — package layout
- [`docs/ADDING_MODELS.md`](ADDING_MODELS.md) — convert a new HF model
- `Examples/CoreMLLLMChat/` — full SwiftUI chat app reference
- `Examples/Gemma3Demo/` — minimal Gemma 3 / Embedding / Function tabs
- `Sources/{functiongemma,embeddinggemma,coreml-llm-smoke}-demo/` —
  CLI references you can copy-paste from
