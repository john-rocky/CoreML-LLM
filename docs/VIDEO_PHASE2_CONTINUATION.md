# Gemma 4 Video Input — Phase 2 Continuation Guide

Phase 1 (this branch) ships a working video path by average-pooling the
still-image encoder's 256 tokens/frame down to 64 in Swift. That fix was
enough to unlock temporal reasoning (see MULTIMODAL.md), but it is a
post-hoc workaround — the Gemma 4 team trained the model with a
separately-configured `video_processor` (`max_soft_tokens=70`) that pools
inside the encoder. Phase 2 replaces our Swift-side pool with a
CoreML-converted video-grade vision encoder.

This doc is written for the next engineer (or a beefier build machine)
to pick up cleanly. All Phase 1 code paths stay as fallback — Phase 2
only adds a `vision_video.mlpackage` alongside the existing
`vision.mlpackage`.

---

## Prerequisites

| Requirement          | Why                                                    |
|----------------------|--------------------------------------------------------|
| Python 3.10+         | `transformers` ≥ latest main (Gemma 4 processor)      |
| `coremltools ≥ 8.x`  | fp16 palettization, iOS 18 target                     |
| 64 GB RAM or swap    | Gemma 4 E2B vision tower trace + CoreML convert       |
| ~10 GB disk          | HF weights (`google/gemma-4-E2B-it`) + mlpackage      |
| Mac w/ Apple Silicon | `ct.convert(..., compute_units=ct.ComputeUnit.ALL)` sanity run |

Get the HF weights once (the repo's conversion output dir already has
`hf_model/` with tokenizer + config only):

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/gemma-4-E2B-it',
  local_dir='$HOME/Documents/Models/gemma4-e2b/hf_model',
  allow_patterns=['*.safetensors', '*.json', 'tokenizer*'])
"
```

---

## Task list

1. Add a video-grade vision encoder to the conversion script.
2. Swap the Phase 1 2×2 pool for the real encoder output in Swift.
3. Validate parity (HF reference vs. our output) on a known clip.
4. Ship.

---

## Step 1 — Add `vision_video.mlpackage` conversion

The existing `conversion/convert_gemma4_multimodal.py` calls
`save_vision_weights(hf_model, args.output)` which currently only dumps
an `.npz` — the still-image `vision.mlpackage` shipped on HF was built
with an out-of-tree script. **The quickest Phase 2 path is: trace
`hf_model.model.vision_tower` twice** — once at the image-path resolution
and once at the video-path resolution — and emit two `.mlpackage`s.

The processor config tells us the exact knobs:

```json
"image_processor":  { "max_soft_tokens": 280, ... }
"video_processor":  { "max_soft_tokens": 70, ... }
```

Both use `patch_size=16, pooling_kernel_size=3`. So for video:

| field             | image path | video path           |
|-------------------|------------|----------------------|
| target pixel budget | 645,120  | 161,280 (= 70·48²·0.5 rounded) — empirically 384² for square |
| num_patches (input) | 2520     | 630 (= 384²/16²)     |
| max_soft_tokens     | 280      | 70                   |
| real tokens (square)| 256      | 64                   |
| output shape        | (1,280,H)| (1,70,H)             |

### 1.1 Extend the Python-side image processor to emit video-grade inputs

In `save_vision_weights` (or in a new `build_video_vision(hf_model, ...)`
helper in `conversion/models/gemma4_vision.py`), run the HF
`Gemma4VideoProcessor` on a dummy 384×384 frame to confirm the shape
contract. Use `processor.video_processor(...)` directly — don't go
through `processor(...)` which expects full video tensors.

### 1.2 Trace and convert the tower

```python
class VisionVideoWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.vision_tower = hf_model.model.vision_tower
        self.embed_vision = hf_model.model.embed_vision

    def forward(self, pixel_values, image_position_ids):
        out = self.vision_tower(
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            return_dict=True)
        # pool already happens inside the tower when max_soft_tokens is
        # honored via position_ids; embed_vision does the 768→1536 proj.
        return self.embed_vision.embedding_projection(out.pooler_output)

traced = torch.jit.trace(
    VisionVideoWrapper(hf_model).eval(),
    (torch.zeros(1, 630, 768, dtype=torch.float32),
     torch.zeros(1, 630, 2,   dtype=torch.int32)),
)

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="pixel_values",
                      shape=(1, 630, 768), dtype=np.float32),
        ct.TensorType(name="pixel_position_ids",
                      shape=(1, 630, 2),   dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="image_features", dtype=np.float16)],
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.ALL,
    compute_precision=ct.precision.FLOAT16,
)
mlmodel.save(os.path.join(output_dir, "vision_video.mlpackage"))
```

**Gotchas:**
- The vision tower has aspect-ratio-aware pooling; if the trace bakes
  the still-image pooling kernel in, override it before tracing:
  `hf_model.model.vision_tower.config.max_soft_tokens = 70`.
- If `torch.jit.trace` complains about dynamic masking (it did when the
  image encoder was first traced — see the `gemma4_vision.py` docstring),
  fall back to `ct.convert` via the new `ct.convert(..., source="pytorch")`
  + `torch.export` path instead of `jit.trace`.
- Palettize with int4 per-grouped-channel **only if** output cosine
  similarity vs. the un-palettized HF forward pass stays ≥ 0.98 on a
  384×384 synthetic frame. Video fidelity is more sensitive to encoder
  drift than still-image captioning.

### 1.3 Wire the CLI flag

Add to `convert_gemma4_multimodal.py`:

```python
parser.add_argument("--video-vision", action="store_true",
                    help="Also produce vision_video.mlpackage (max_soft_tokens=70)")
...
if args.video_vision:
    save_video_vision_weights_and_convert(hf_model, args.output)
```

Update the `parts` block in the emitted `model_config.json`:

```python
"parts": {
    ...,
    "vision_video": "vision_video.mlpackage",  # optional
},
```

---

## Step 2 — Swift-side switch

Two files to touch: `CoreMLLLM.swift` and `ModelConfig.swift`.

### 2.1 Load the video encoder lazily

In `CoreMLLLM.load(from:)`, next to where `vision.mlpackage` is resolved:

```swift
let videoVisionCompiled = directory.appendingPathComponent("vision_video.mlmodelc")
let videoVisionPkg      = directory.appendingPathComponent("vision_video.mlpackage")
if FileManager.default.fileExists(atPath: videoVisionCompiled.path) {
    llm.videoVisionModelURL = videoVisionCompiled
} else if FileManager.default.fileExists(atPath: videoVisionPkg.path) {
    llm.videoVisionModelURL = videoVisionPkg
}
```

Add `videoVisionModel` / `videoVisionModelURL` / `videoVisionConfig`
properties mirroring the still-image ones.

### 2.2 Branch in `concatFrameFeatures`

Today `concatFrameFeatures` always calls `processImage(frame)` (256-token
path) and then either pools to 64 or memcpys. Replace with:

```swift
private func concatFrameFeatures(_ frames: [CGImage],
                                  tokensPerFrame: Int) throws -> MLMultiArray {
    // If a purpose-built video encoder is available, use it — it already
    // emits 64 tokens/frame (max_soft_tokens=70 inside the encoder) so
    // we can skip the 2×2 Swift pool entirely.
    if tokensPerFrame == 64, let url = videoVisionModelURL {
        return try concatVideoFrameFeatures(frames, encoderURL: url)
    }
    // … existing image-encoder + 2×2 pool fallback …
}
```

Implement `concatVideoFrameFeatures` with:
- Resize each frame to 384×384 (matches `video_processor` pixel budget).
- Build `pixel_values (1, 630, 768)` and `pixel_position_ids (1, 630, 2)`
  the same way `ImageProcessor.process` does, but with Hp=Wp=24 and
  `total = 630`.
- Run the video encoder; memcpy the first 64 of 70 tokens per frame into
  the concatenated output.

Keep the Phase 1 pool path behind `videoVisionModelURL == nil` so old
model bundles still work.

### 2.3 Update `VideoProcessor.Options.tokensPerFrame` semantics

Current: `tokensPerFrame=64` means "image encoder + 2×2 pool."
After Phase 2: `tokensPerFrame=64` means "use video encoder if present,
else 2×2 pool as fallback."

No API break — just the docstring changes.

---

## Step 3 — Validation

Run the existing Mac CLI with `--compare` against a motion clip and
confirm:

1. Multi-frame output length stays high (>500 chars) and mentions
   timestamps / motion.
2. First-token latency doesn't regress (video encoder should be ~same
   cost as image encoder; we're running it 3–8 times instead of 1, so
   budget accordingly).
3. Memory peak on iPhone stays under the existing multimodal headroom.

HF-reference parity check (optional but recommended) — install a main
build of transformers that includes `Gemma4Processor`:

```bash
pip install 'git+https://github.com/huggingface/transformers@main'
```

then:

```python
proc = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
model = AutoModelForMultimodalLM.from_pretrained("google/gemma-4-E2B-it")
msgs = [{"role":"user","content":[
    {"type":"video","url":"/tmp/clip.mp4"},
    {"type":"text","text":"Describe what happens."}]}]
inputs = proc.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                   return_tensors="pt",
                                   num_frames=5, fps=1.0,
                                   add_generation_prompt=True)
print(proc.decode(model.generate(**inputs, max_new_tokens=200)[0]))
```

Compare that to `swift run video-test …` output — they should agree on
macro content (object names, motion direction) even if phrasing differs.

---

## Step 4 — Ship

- Update `docs/MULTIMODAL.md` — remove the "Phase 2 pending" paragraph
  and the pool-hack note; add `vision_video.mlpackage` to the file list.
- Bump the HF release (`mlboydaisuke/gemma-4-E2B-coreml`) to include
  `vision_video.mlpackage`.
- Update `ModelDownloader.ModelInfo.gemma4e2b.size` to reflect the new
  artifact (adds ~150 MB).
- Post-merge: audit `tokensPerFrame` default — once every shipped model
  has the video encoder, the pool-fallback branch can be deleted.

---

## Known unknowns

- **`torch.jit.trace` on the vision tower**: the `gemma4_vision.py`
  docstring says tracing was hard last time because of dynamic masking.
  **Partially resolved** on this branch — see
  `conversion/phase2/trace_video_vision.py`. Summary of what we learned
  in a ~1.5 hr Phase 2 spike on 2026-04-15:
  - Env that works: **Python 3.11 + `transformers@main` (5.6.0.dev0) +
    `torch 2.7.0` + `coremltools 9.0` + `accelerate 1.13`** via `uv`.
    Python 3.9 is too old for transformers main.
  - `hf.model.get_image_features(pixel_values=(1,630,768),
    image_position_ids=(1,630,2) int64)` returns `(64, 1536)` directly
    for a 24×24 patch grid — the vision tower **adaptively pools** to
    the video-grade token count without any config change. That's the
    output we want.
  - `torch.export` path: conversion fails at coremltools-9.0 on
    unsupported fx nodes (`new_ones`, then `__and__`, then …). A
    whack-a-mole. Unusable until coremltools 10+.
  - `torch.jit.trace` path: the vision encoder's `forward` calls
    `create_bidirectional_mask(...)` which internally does
    `q_length.shape[0]` on what's effectively a scalar during trace and
    raises `IndexError`. Fix is to monkey-patch
    `hf.model.vision_tower.encoder.forward` with a replacement that
    builds a static additive mask from the `attention_mask` input —
    implementation is in `phase2/trace_video_vision.py`
    (`patched_vision_forward`).
  - With that patch, `torch.jit.trace` succeeds. **`ct.convert` then
    fails** in MIL on a cast op (`only 0-dimensional arrays can be
    converted to Python scalars`) — probably inside `rotary_emb` or
    similar. Next engineer picks up here.
  - Suggested next step: run with
    `ct.convert(..., debug=True)` to localize the offending node, then
    either (a) patch that node's handler in
    `coremltools/converters/mil/frontend/torch/ops.py`, or (b)
    rewrite the vision tower's `rotary_emb` path to avoid the cast.
- **ANE placement**: the still-image encoder runs `.cpuAndGPU` by design
  (see BENCHMARKING.md). The video encoder inherits the same constraint
  unless ANE-friendly ops can be substituted. Do not chase ANE placement
  for the video encoder in Phase 2 — correctness first.
- **`num_frames=32` on mobile**: HF recommends sampling 32 frames for
  E2B at 1 fps. On a 2K chunk you can only fit ~7. Don't bump
  `maxFrames` in `VideoProcessor.Options` past what the chunk size
  actually allows. Wait for 8K chunks (see `MOBILE_2K_COMPETITIVE_PLAN.md`).

---

## File inventory for Phase 2

| Path                                                 | Change               |
|------------------------------------------------------|----------------------|
| `conversion/phase2/trace_video_vision.py`            | **starting point** — partial trace+convert, blocked in MIL cast |
| `conversion/phase2/probe_vision.py`                  | reference: confirms `(64,1536)` output for 24×24 patches |
| `conversion/models/gemma4_vision.py`                  | add video converter (productionize phase2/)  |
| `conversion/convert_gemma4_multimodal.py`            | `--video-vision` flag|
| `Sources/CoreMLLLM/CoreMLLLM.swift`                  | load/use new encoder |
| `Sources/CoreMLLLM/VideoProcessor.swift`             | doc tweaks only      |
| `Sources/CoreMLLLM/ModelDownloader.swift`            | new artifact in manifest |
| `docs/MULTIMODAL.md`                                 | remove Phase 1 hack note |
