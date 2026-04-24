# Probe: Gemma 4 vision encoder → ANE

**Date:** 2026-04-24
**Branch:** `worktree-gemma4-vision-ane`
**Goal (TTFT roadmap #1):** Move the Gemma 4 E2B still-image vision
encoder off GPU (`.cpuAndGPU`, ~200–300 ms per image on iPhone 17 Pro)
onto ANE, targeting a **60–100 ms TTFT reduction** for image and video
prompts.

## TL;DR — result

- **98.3% ANE placement** (1937/1970 compute ops; 33 CPU, 0 GPU).
- Real-input parity vs HF: **overall cos = 1.0000**, per-token min cos
  0.9996, max_abs_diff 0.086 (fp16 noise).
- Mac Studio (M-series ANE) steady-state predict:
  - **`.cpuAndNeuralEngine`: ~473 ms/image**
  - `.cpuAndGPU` (same file, forced GPU): ~3800 ms/image (8× slower)
- Ships as `vision.ane.mlpackage` (326 MB compiled to `vision.ane.mlmodelc`);
  coexists with the legacy `vision.mlpackage` — runtime prefers the ANE
  build when present.

## Two ANE-only bugs (not visible on GPU)

On-device iteration caught two bugs that the convert-time zero-input
parity check missed. Both are fixed in this PR.

1. **Dynamic-shape `gather_nd` silent zero output**. The stock HF vision
   tower strips padding via `hidden_states[pooler_mask]`. With a full
   grid there is no padding, but the trace still emits a data-dependent
   `gather_nd` that iOS ANE miscompiles — Mac CPU/GPU produced correct
   features while ANE returned an all-zero tensor (norm = 0 vs 521).
   Fix: monkey-patch `Gemma4VisionModel.forward` to skip the strip when
   there are no padding positions.

2. **fp16 subnormal `rsqrt(0) = Inf` in `embedding_pre_projection_norm`**.
   Torch's `Gemma4RMSNorm.forward` casts to fp32 before `rsqrt`, but
   coremltools' `FLOAT16` compute_precision folds that cast away. iOS
   ANE flushes subnormals to zero, so `mean(x²)` for small-activation
   tokens lands at 0, `rsqrt(0) → Inf`, and a handful of feature
   vectors end up ~1000× their true magnitude (direction still correct,
   per-token cos ≈ 1.0). Fix: replace the final RMSNorm with the
   `layer_norm([x, -x])` identity — ANE's native `layer_norm` handles
   the reduction correctly in fp16.

The second bug is the one that produced garbled output on the first
device test (sunflower image → "human eye" response); the first bug
would have been caught earlier but GPU happened to tolerate the
`gather_nd`.

## Background

- Shipped `vision.mlpackage` was built by an earlier (no-longer-in-tree)
  script. Input `pixel_values (1, 2520, 768) fp32` + `pixel_position_ids
  (1, 2520, 2) int32` → output `image_features (1, 280, 1536) fp16`.
  Runs on GPU because it contains dynamic ops (pooler one-hot grouping
  gated on padding positions, masked_fill with variable padding).
- Qwen3-VL 2B vision already ships on ANE at 448×448 fixed grid
  (`build_qwen3_vl_2b_vision.py`, cos ≥ 0.95 vs HF, 196 merged tokens).
  The recipe:
  - Fixed grid → position / rotary embeddings are compile-time
    constants.
  - Static additive attention mask (no cu_seqlens).
  - Pre-patchify in Swift → rank-5 reshape max inside the graph
    (A18 Pro ANE faults on rank-10 patchify views that Mac ANE
    accepts).
- `convert_video_vision_to_coreml` (for `vision_video.mlpackage`,
  24×24 = 630-patch video-grade variant) already applies the static
  attention mask patch for Gemma 4; we just have never pointed it at
  ANE.

## This PR

Adds `convert_still_image_vision_ane_to_coreml` in
`conversion/models/gemma4_vision.py` and a `--vision-ane` flag on
`convert_gemma4_multimodal.py`.

### Design choices

| Choice | Value | Why |
|---|---|---|
| Grid | Square 48×48 = 2304 patches | 768×768 canvas matches HF's aspect-ratio-preserving resize for most photos. Even split 48 / 3 = 16 → integer pooling, no padding. |
| Pool kernel | k = 3 | `48 ÷ 3 = 16 → 16×16 = 256 soft tokens`. Matches HF `pooling_kernel_size=3`. |
| Output tokens | 256 (not 280) | No padding slots. The runtime already tolerates 256 for square images (MULTIMODAL.md bug #2 fix in v0.3.0). |
| Input layout | `(1, 2304, 768)` pre-patchified | Swift `ImageProcessor.process()` already emits this layout (without trailing -1 padding), so the preprocessing change on-device is trivial. |
| Attention mask | Static all-ones (no padding) | With no padding positions, the existing `_static_mask_forward` patch degenerates to a constant-zero additive mask; coremltools folds it into const. |
| Pooler | Static one-hot weights | With deterministic `pixel_position_ids` (0..47 × 0..47), the `one_hot(kernel_idxs, 256)` matrix folds to a constant at trace time. |
| Compute units | `CPU_AND_NE` at convert | Surface ANE-hostile ops early via the audit helper. |

### What's intentionally NOT in this PR

- **No Swift changes** yet. Path-flipping to `.cpuAndNeuralEngine` gets a
  separate PR once the Mac conversion shows ≥ 80 % ANE placement and
  cos ≥ 0.99 parity.
- **No multifunction.** Variable aspect ratios (portrait, landscape)
  fall back to the existing GPU `vision.mlpackage`. Adding a handful of
  fixed grids (e.g. 42×56, 56×42) is straightforward follow-up once the
  square grid is validated end-to-end.
- **No iPhone deploy.** Mac parity first, then ComputePlanAudit, then
  device.

## Reproducing the Mac result

```bash
# from repo root, with lama-cml env active
python conversion/convert_gemma4_multimodal.py \
    --output /tmp/gemma4-vision-ane \
    --vision-ane
# compile for iOS
xcrun coremlcompiler compile \
    /tmp/gemma4-vision-ane/vision.ane.mlpackage /tmp/gemma4-vision-ane/
# latency bench (Mac)
python conversion/bench_vision_ane_mac.py \
    --ane /tmp/gemma4-vision-ane/vision.ane.mlpackage --iters 40
```

## Swift wiring (this PR)

`CoreMLLLM.load()` checks for, in order: `vision.ane.mlmodelc`,
`vision.ane.mlpackage`, `vision.mlmodelc`, `vision.mlpackage`. When an
ANE build is found it sets `.cpuAndNeuralEngine` and routes
`processImage` through `ImageProcessor.processANE(...)` (48×48 fixed
grid, fp16 pixel values, no padding positions). The legacy variable-
grid GPU path is untouched and remains the fallback.

## iPhone deploy

```bash
xcrun devicectl device copy to \
    --device <iPhone UUID> \
    --domain-type appDataContainer \
    --domain-identifier com.example.CoreMLLLMChat \
    --source /tmp/gemma4-vision-ane/vision.ane.mlmodelc \
    --destination Documents/Models/gemma4-e2b-fashion/vision.ane.mlmodelc \
    --user mobile
```

Then rebuild CoreMLLLMChat in Xcode and launch. The runtime log prints
the vision path it picked up; verify it reads `vision.ane.mlmodelc`
and that the image-prompt TTFT drops vs the legacy build.

## Aspect-ratio tradeoff

The ANE build is locked to a square 48×48 grid. Images are force-
resized to 768×768, so non-square photos have their aspect ratio
squashed. For natural photos this is acceptable (the vision tower
tolerates it, cos=1.0 at convert time on a zero-frame sanity check).
For extreme aspect ratios the legacy GPU encoder remains the correct
choice — delete `vision.ane.mlmodelc` to force-revert.

## Follow-ups (separate PRs)

- **Aspect-ratio multifunction**: add portrait/landscape fixed grids as
  additional functions so CoreML picks the closest match at predict
  time.
- **Session #4** (video batch encoding): apply the same static-grid
  recipe to `vision_video` with batch=4 so a 6-frame clip encodes in
  two predict calls instead of six.
- **Session #2** (image embedding cache): hash pixels, cache image
  features on disk, skip re-encode for follow-up turns.
