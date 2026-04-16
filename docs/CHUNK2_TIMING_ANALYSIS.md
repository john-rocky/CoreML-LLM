# Chunk 2 Timing Analysis: Why 29.2ms for 7 Layers?

**Measured**: iPhone 17 Pro, 8K context, 4-chunk decode_q1 layout.  
**Anomaly**: c2 (L8-L14, 7 layers) = 29.2ms vs c1 (L0-L7, 8 layers) = 15.9ms. Nearly 2x slower with fewer layers.

## Root Cause: Full-Attention KV Memory Bandwidth

Chunk 2 has **two full-attention layers** (L9, L14) vs chunk 1's **one** (L4). At 8K context, each full-attention layer touches 16x more KV data than a sliding layer. This single asymmetry accounts for the entire timing gap.

## Per-Layer Breakdown Estimate

### KV cache sizes per attention type (fp16)

| Type | K shape | V shape | Bytes per K or V | Combined K+V |
|------|---------|---------|-------------------|--------------|
| Sliding | (1, 1, 512, 512) | same | 512 KB | 1.0 MB |
| Full | (1, 1, 8192, 512) | same | 8.2 MB | 16.4 MB |

Note: max_hd=512 is used for all cache slots (padded for sliding's hd=256).

### Memory traffic per layer

- **Sliding layer**: Read 1.0 MB (K+V), write 1.0 MB (shift update), attention matmul over W=512 tokens. Plus compute (projections, MLP). Estimated ~1.8 ms/layer.
- **Full-attention layer**: Read 16.4 MB (K+V), write 16.4 MB (mask update), attention matmul over ctx=8192 tokens. Plus compute. Estimated ~5.5 ms/layer.

### Chunk 1 (L0-L7): 8 layers

| Layers | Type | Count | Est. ms |
|--------|------|-------|---------|
| L0-3, L5-7 | sliding | 7 | 7 x 1.8 = 12.6 |
| L4 | full | 1 | 1 x 5.5 = 5.5 |
| PLE compute | - | 1 | ~1.0 |
| **Subtotal** | | | **~16.0** (measured 15.9) |

PLE (per-layer embedding) computation is done once in chunk 1 via `_compute_ple()` -- a single projection + RMSNorm over (1, 35, 256). This adds ~1ms but chunk 1 amortizes it over 8 layers.

### Chunk 2 (L8-L14): 7 layers

| Layers | Type | Count | Est. ms |
|--------|------|-------|---------|
| L8, L10-13 | sliding | 5 | 5 x 1.8 = 9.0 |
| L9 | full | 1 | 1 x 5.5 = 5.5 |
| L14 | full | 1 | 1 x 5.5 = 5.5 |
| KV13/KV14 output copy | - | 1 | ~2.5 |
| Extra full-attn I/O overhead | - | 1 | ~4.5 |
| **Subtotal** | | | **~27.0** (measured 29.2) |

## Three Contributing Factors (ranked by impact)

### 1. Double Full-Attention Layers (dominant, ~11ms extra vs chunk 1)

Chunk 2 has L9 and L14 as full-attention. Each full-attention layer at 8K context:
- Reads K_full (8192 x 512 x fp16 = 8.2 MB) and V_full (8.2 MB) from IOSurface
- Writes back the updated cache (16.4 MB)
- Computes attention over 8192 positions instead of 512
- Total memory traffic per full layer: ~33 MB read+write

Chunk 1 has only L4 as full. So chunk 2 pays the full-attention penalty **twice** while chunk 1 pays it once. At ANE's ~50 GB/s effective bandwidth for IOSurface-backed tensors, the extra 33 MB costs ~0.7ms in pure bandwidth -- but the real cost is higher due to (a) the attention matmul being 16x larger (Q @ K^T over 8192 vs 512), and (b) the mask-based KV update (multiply + add over 8.2M elements twice) which does not pipeline as efficiently as the sliding shift.

**Extra cost from second full-attention layer: ~5.5ms**

### 2. KV13/KV14 Output Emission (secondary, ~2-5ms)

Chunk 2 uniquely emits 4 extra output tensors for KV sharing:
- `kv13_k`: (1, 1, 512, 256) = 256 KB -- sliced from L13's sliding cache
- `kv13_v`: (1, 1, 512, 256) = 256 KB
- `kv14_k`: (1, 1, 8192, 512) = 8.2 MB -- sliced from L14's full cache
- `kv14_v`: (1, 1, 8192, 512) = 8.2 MB

Total extra output: ~17 MB. This is not a simple pointer pass -- CoreML materializes these as separate IOSurface-backed output tensors. The kv14 pair alone is 16.4 MB of data that must be written to new IOSurface buffers so chunks 3 and 4 can read them as inputs.

The slice operations (`K_full_out[..., :512]`, `K_sliding_out[..., :256]`) also prevent zero-copy: the slice creates a non-contiguous view that CoreML must copy to a contiguous output buffer.

**Extra cost: ~2.5ms** (bandwidth + IOSurface allocation for 4 tensors)

### 3. Higher Aggregate KV I/O (contributing, ~2ms)

Chunk 2's total KV input tensors:
- `K_sliding_in`: (5, 1, 512, 512) = 2.5 MB
- `V_sliding_in`: (5, 1, 512, 512) = 2.5 MB
- `K_full_in`: **(2, 1, 8192, 512) = 16.4 MB**
- `V_full_in`: **(2, 1, 8192, 512) = 16.4 MB**
- **Total KV input: 37.8 MB**

Chunk 1's total KV input tensors:
- `K_sliding_in`: (7, 1, 512, 512) = 3.6 MB
- `V_sliding_in`: (7, 1, 512, 512) = 3.6 MB
- `K_full_in`: (1, 1, 8192, 512) = 8.2 MB
- `V_full_in`: (1, 1, 8192, 512) = 8.2 MB
- **Total KV input: 23.6 MB**

Chunk 2 moves 14 MB more KV data in, plus 17 MB more KV data out (the kv13/kv14 exports). That is +31 MB of IOSurface traffic.

## What Could Fix It

### Option A: Rebalance chunks to equalize full-attention layers
Move L9 into chunk 1 (making chunk 1 = L0-L9, 2 full). Chunk 2 becomes L10-L14 (5 layers, 1 full = L14 only). This equalizes the full-attention cost:
- c1: L0-L9, 8 sliding + 2 full, ~21ms
- c2: L10-L14, 4 sliding + 1 full, ~13ms + kv13/14 overhead ~15ms
- Total c1+c2: ~36ms (vs current 45.1ms) -- saves ~9ms

### Option B: Split L14 into its own mini-chunk
L14 is special: it is the only full-attention layer whose KV is shared downstream. Making it a standalone chunk 2b would isolate its cost and allow kv14 to be passed by reference (the chunk's output IS the cache). This eliminates the 8.2 MB x 2 copy for kv14_k/kv14_v.

### Option C: Fuse kv13/kv14 output with the cache output
Instead of emitting kv13/kv14 as separate tensors, have chunks 3/4 read directly from chunk 2's K_sliding_out[slot_for_L13] and K_full_out[slot_for_L14]. This requires the Swift runtime to extract the correct slot and pass it, but eliminates the 17 MB copy inside the ANE graph.

### Option D: Reduce full-attention KV cache waste
The full-attention cache uses max_hd=512 for all layers, but only full-attention layers need 512. If the cache were split into separate sliding (hd=256) and full (hd=512) tensors without padding, the full-attention KV I/O would be the same, but sliding KV I/O would halve. This saves ~5 MB for chunk 2's sliding layers. Minor impact.

## Summary

The 29.2ms is **not** caused by extra computation or per-layer processing differences -- layers L8-L14 have identical compute structure to L0-L7. The bottleneck is **memory bandwidth**: chunk 2's two full-attention layers (L9, L14) move 2x the KV data of chunk 1's single full-attention layer (L4), and the kv13/kv14 export adds another 17 MB of materialized output. On ANE, where arithmetic is cheap but IOSurface memory traffic is the bottleneck, this dominates.

**Dominant factor**: 2 full-attention layers (L9 + L14) instead of 1 -- accounts for ~60% of the gap.  
**Secondary factor**: kv13/kv14 output materialization -- accounts for ~25% of the gap.  
**Tertiary factor**: Higher aggregate KV I/O tensor sizes -- accounts for ~15% of the gap.
