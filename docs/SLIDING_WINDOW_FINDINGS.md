# Sliding-Window Local Attention — Already Exploited

**Status (2026-04-15):** DONE. No further work warranted on this axis at the
KV-layout level. The codebase has implemented the two-tier KV layout, per-layer
type distinction, shift-based circular sliding KV, separate sliding vs full
causal masks, and IOSurface-backed KV buffers.

This document records the investigation so future sessions can skip it.

## 1. Gemma 4 E2B config, measured

From `conversion/models/gemma4.py` (authoritative for the builder):

- `num_hidden_layers = 35`
- `sliding_window = 512` (NOT 1024 or 4096)
- `layer_types`: pattern `(i+1) % 5 == 0 → full_attention`, else `sliding_attention`
- Result: **7 full-attention layers** (L4, L9, L14, L19, L24, L29, L34) and
  **28 sliding-attention layers**. Ratio is 4:1 (sliding:full) per block, not
  5:1 as the task prompt guessed.
- Additional wrinkle: `num_kv_shared_layers = 20`. Layers 15-34 do **not own**
  KV at all; they read shared KV from L13 (sliding) and L14 (full). So the
  only layers that allocate KV are L0-L14 (15 layers = 12 sliding + 3 full).
- Attention head dims differ by type: `head_dim=256` for sliding,
  `global_head_dim=512` for full. RoPE tables are also separate
  (`theta=10000` sliding, `theta=1M` full).

## 2. mlpackage graph: is the layer-type distinction baked in?

Yes. See `conversion/models/gemma4_swa_chunks.py`:

- Function `_run_layer_swa` branches on `config.is_full_attention(layer_idx)`
  and `config.is_kv_shared(layer_idx)`. Sliding layers take
  `K_sliding_slot (1,1,W,maxHd)`, full layers take `K_full_slot (1,1,ctx,maxHd)`.
- Sliding KV update is shift-based: `torch.cat([K[:, :, 1:, :], new_k], dim=2)`
  — a ring buffer expressed as a static-shape op so the CoreML graph stays
  fully static. Position/RoPE indexing is the absolute step `position`; the
  sliding causal mask in Swift (`makeSlidingCausalMask`) handles the
  "fewer than W tokens seen so far" case by masking the unused prefix.
- Full KV update is mask-based via `update_mask: (1,1,ctx,1)` one-hot at the
  current position.
- Chunk partitioning:
  - chunk1 (L0-7): 7 sliding + 1 full → `(7,1,W,512)` + `(1,1,ctx,512)`
  - chunk2 (L8-14): 5 sliding + 2 full → `(5,1,W,512)` + `(2,1,ctx,512)`
  - chunk3, chunk4 (L15-34): no KV in/out, read kv13 and kv14 from chunk2
- Two causal masks flow to every chunk as separate inputs:
  `causal_mask_full (1,1,1,ctx)` and `causal_mask_sliding (1,1,1,W)`.

So the mlpackage graph **does** have layer-type distinction, with genuinely
different KV tensor shapes for sliding vs full.

## 3. Swift runtime (ChunkedEngine.swift): KV allocation

Lines 51–58 declare the tiered KV buffers:

```
kSliding1: MLMultiArray  // (7, 1, W=512, maxHd=512)
vSliding1: MLMultiArray
kFull1:    MLMultiArray  // (1, 1, ctx,  maxHd=512)
vFull1:    MLMultiArray
kSliding2: MLMultiArray  // (5, 1, W,    maxHd)
vSliding2: MLMultiArray
kFull2:    MLMultiArray  // (2, 1, ctx,  maxHd)
vFull2:    MLMultiArray
```

Allocation (lines 314-317) uses IOSurface-backed `CVPixelBuffer` for
zero-copy CPU↔ANE transfer, with fallback to plain `MLMultiArray`.

So on the Swift side too, sliding layers carry only `(slots, 1, 512, 512)` =
~1.8 MB total (chunk1) and ~1.3 MB (chunk2), versus `(slots, 1, ctx, 512)`
for the 3 full layers. At `ctx=8192`, the full-attn KV is `3 × 8192 × 512 × 2 B`
= 24 MB; if we had naively kept sliding layers at full ctx, they would have
added `12 × 8192 × 512 × 2 B` = 96 MB. **The 96 MB saving is already realized.**

## 4. Quantitative confirmation that the saving is non-trivial

`docs/SPEED_8K.md` table at 8K (excerpt from line 46-50):

| Chunk | ms | layers | ms/layer |
|---|---|---|---|
| chunk1 (L0-7, SWA) | 12.4 | 8 | 1.55 |
| chunk2 (L8-14, full-attn mix) | 20.7 | 7 | 2.96 |
| chunk3 (L15-24, SWA) | 15.2 | 10 | 1.52 |
| chunk4 (L25-34, SWA+LMhead) | 17.3 | 10 | 1.73 |

Sliding layers run at ~1.5 ms/layer vs ~3.0 ms/layer for full-attn layers.
Full-attn layers are the known 8K bottleneck; sliding layers are already
cheap because of the tiered KV. The remaining optimization axis inside chunk2
is retrieval-based attention on the 7 full-attn layers (discussed in
`docs/SPEED_8K.md` §1 A3/A4 and `docs/EXPERIMENTS.md` WFA), NOT SWA.

## 5. Correctness anchors

- `makeSlidingCausalMask(position, W)` at line 1090: builds a length-W causal
  mask where the valid-token prefix has width `min(position+1, W)`, packed
  against the right edge to match the shift-based KV layout. Tokens older
  than `position - W + 1` are naturally absent from the ring, and masked.
- Position/RoPE fed to the model is the absolute step (`currentPosition`),
  not the ring offset. This is the correct Gemma 4 behavior because RoPE
  positions are applied to Q/K **before** they enter the cache, so the cache
  stores pre-rotated K. Rolling out via `cat([K[:,:,1:], new_k])` preserves
  the relative-position phase relationships inside the window.
- Parity of the tiered KV path is covered by the existing mlpackage parity
  suite (referenced in `docs/CONVERSION.md`); the models ship with
  `sliding_window: 512` in `model_config.json` and the mlpackage causal-mask
  shape is validated at load (ChunkedEngine line 261-268).

## 6. What is NOT already exploited (adjacent, but not in scope for this task)

These are tracked elsewhere; do not confuse with the SWA KV layout that IS done:

1. **Full-attn KV sparsity / retrieval** on the 3 full-owning layers at 8K
   (TriForce / Quest / DuoAttention). Documented in `docs/SPEED_8K.md` and
   `docs/EXPERIMENTS.md`. Not done.
2. **Sliding layer FlashAttention fusion** — `gemma4_swa_flash.py` explores
   this; status per file timestamp unclear, but not the critical path per
   the 8K audit.
3. **Per-head classification** to shrink the full-attn layer count further
   (DuoAttention-style). Not done.

## 7. Verdict

Task objective — exploit sliding-window local attention for decode-time
savings on KV compute and IOSurface payload — is already implemented end-to-end:
Python builder has layer-typed KV shapes, Swift runtime allocates tiered KV,
masks are built per-type, the graph is static, and the 8K profiler confirms
sliding layers run at ~1.5 ms/layer (approx. half the full-attn cost). The
96 MB per-step KV transfer saving at 8K is already captured.

**Recommendation:** do not rebuild. Spend effort on the full-attn retrieval
redesign (the real 8K bottleneck), which is the open item in
`docs/SPEED_8K.md`. That is independent of, and composes with, the
already-shipped SWA layout.
