# Probe: Gemma 4 vision encoder → ANE

**Date:** 2026-04-24
**Branch:** `worktree-gemma4-vision-ane`
**Goal (TTFT roadmap #1):** Move the Gemma 4 E2B still-image vision
encoder off GPU (`.cpuAndGPU`, ~200–300 ms per image on iPhone 17 Pro)
onto ANE, targeting a **60–100 ms TTFT reduction** for image and video
prompts.

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

## Handoff — run on Mac Studio

```bash
# from repo root, with lama-cml env active
python conversion/convert_gemma4_multimodal.py \
    --output /tmp/gemma4-vision-ane \
    --vision-ane
```

Expected console output (the two things to copy back):

```
  ANE placement: <ane>/<compute> (<pct>%) — CPU=<n> GPU=<n>
    cosine=0.99xx  max_abs_diff=0.0xxx
```

### Decision matrix

| ANE % | cos | Next step |
|-------|-----|-----------|
| ≥ 80 | ≥ 0.99 | Swift wiring PR (prefer `vision.ane.mlpackage` when present, `.cpuAndNeuralEngine`) → iPhone bench. |
| ≥ 80 | < 0.99 | Disable ANE fp16 drift sources: upcast pooler matmul to fp32, re-check. |
| < 80 | any | Inspect which ops fell back (CPU vs GPU) via audit output. Most likely culprits: `one_hot`, `masked_fill` on const mask, `gather` in rotary embedding. Replace with static constants. |

## Follow-ups (separate PRs once #1 lands)

- **#1b:** Swift wiring + iPhone bench.
- **Aspect-ratio multifunction:** add portrait/landscape grids once
  square path is on ANE.
- **Session #4:** extend this same recipe to batch=4 for video frames.
