# 11C Proposed Contract: Write-After-Accept Verify Protocol

**Status:** Spec — 2026-04-17. Implementation in progress on `feat/verify-protocol-redesign`.

## Resolutions (2026-04-17 implementation session)

Three ambiguities in the original spec were resolved before code work began:

- **A1 — `update_indicator` STAYS as a chunk1/chunk2 input.** It is still required inside the graph to scatter the K new positions into the ctx-length `K_for_attn` for full-attention layers. Only the graph *output* changes: chunks return raw `k_padded` slices instead of the blended ctx-length cache. Mask shapes are unchanged.
- **B — chunks 3 and 4 are UNCHANGED.** The original concern (argmax[1]/[2] would lose dependence on d_0/d_1) is already addressed by chunk2's existing `kv13_k`/`kv13_v`/`kv14_k`/`kv14_v` outputs, which it emits in their *extended* within-verify form (W- or ctx-sized, with the K new positions blended in). Chunks 3/4 consume these as inputs and attend correctly, exactly as today. The fix is purely that **Swift does not write these extended views back to persistent kv13/kv14 storage**. Persistent kv13/kv14 are slots of `kSliding2`/`kFull2`, and Swift commits only N accepted per-T slices to those slots through the new per-T outputs (same path as the other sliding/full layers).
- **C — Bonus token (K+1):** when N==K+1, Swift runs one T=1 `predictStep` with input `embed(token_ids[K-1])` (NOT the verify hidden state) at position `currentPosition + K` after writing the K verified slices. This populates the bonus token's KV in the persistent cache.

---

## Design Principle

Verify chunks compute attention and produce logits *without writing any KV to the persistent cache*. They instead return per-T-position K/V slices as additional outputs. Swift receives all K slices, decides the accepted prefix (length N, 1 ≤ N ≤ K+1), and writes exactly those N slices into the IOSurface-backed KV caches at the correct positions.

This collapses Blocker 2 (`commitAccepted` per-token replay) and item 11c (KV contamination) into a single fix.

---

## New Chunk Output Additions

### Overview

The verify variant of each chunk adds per-T-position K/V slice outputs alongside the existing `hidden_states_out`. The full K×(per-layer K/V) are returned as flat tensors; Swift slices them by position index.

**No inputs change.** The chunks still read the *pre-verify* persistent KV caches as inputs for attention context. The `update_indicator` (one-hot mask) is **removed** from the verify variant — KV computation is still performed inside the graph, but the result is returned as an output instead of being blended back into the input cache.

### Why keep KV compute inside the graph

The graph still needs to compute K and V for each new token to run attention at that token's position. The semantic change is only in *what happens to those computed values*: instead of blending them in-place into the W-sized sliding window (or writing into the full ctx buffer), the graph returns them raw and lets Swift decide what to commit.

---

## Tensor Spec

Throughout: K = number of draft positions (default 3). ctx = 2048, W = 512.

### Notation

- `hd_sliding` = 256 (head dim for sliding layers, L0–L13, L15–L24 shared sliding)
- `hd_full` = 512 (head dim for full layers L4, L9, L14)
- `nkv` = 1 (GQA: 1 KV head)
- Per-position K/V shape = `(1, nkv, K, hd)` — all K positions stacked in seq dim

### verify_chunk1 (L0–L7)

Sliding layers: L0,L1,L2,L3,L5,L6,L7 (7 layers). Full layers: L4 (1 layer).

**Removed input:** `update_indicator`

**New outputs** (replacing `K_sliding_out`, `V_sliding_out`, `K_full_out`, `V_full_out`):

| Output name | Shape | Dtype | Semantics |
|-------------|-------|-------|-----------|
| `new_K_sliding_c1` | (7, nkv, K, 256) | fp16 | New K slices for all 7 sliding layers, K positions |
| `new_V_sliding_c1` | (7, nkv, K, 256) | fp16 | New V slices for all 7 sliding layers, K positions |
| `new_K_full_c1`    | (1, nkv, K, 512) | fp16 | New K slices for L4 (full), K positions |
| `new_V_full_c1`    | (1, nkv, K, 512) | fp16 | New V slices for L4 (full), K positions |
| `hidden_states_out` | (1, K, 1536) | fp16 | Unchanged |
| `per_layer_combined_out` | (1, K, nlayers*pld) | fp16 | Unchanged |

**Attention inside the graph:** each layer still runs attention against the full W-sized (sliding) or ctx-sized (full) KV context passed as input. The new K/V values are computed but not blended back — they are passed out as raw slices.

**Python model change (chunk1):** In `_run_layer_verify`, replace:
```python
# Old: blend new k into sliding cache
K_sliding_out = torch.cat([K_sliding_slot[:, :, 1:], k_padded], dim=2)
```
with:
```python
# New: keep original cache for attention; stack new k separately
K_for_attn = torch.cat([K_sliding_slot[:, :, 1:], k_padded], dim=2)  # only for attn
# ...do not assign K_sliding_out...
new_K_sliding[layer_slot] = k_padded  # (1, nkv, K, 256) stacked across T
```
Return new K/V stacks as outputs, not the full updated cache.

### verify_chunk2 (L8–L14)

Same pattern as chunk1. Sliding layers: L8, L10, L11, L12, L13 (5 layers). Full layers: L9, L14 (2 layers).

**New outputs:**

| Output name | Shape | Notes |
|-------------|-------|-------|
| `new_K_sliding_c2` | (5, nkv, K, 256) | Sliding slices for L8,L10-L13 |
| `new_V_sliding_c2` | (5, nkv, K, 256) | |
| `new_K_full_c2`    | (2, nkv, K, 512) | Full slices for L9, L14 |
| `new_V_full_c2`    | (2, nkv, K, 512) | |
| `kv13_k_slices`    | (1, nkv, K, 256) | New K at L13, all K positions (for sharing) |
| `kv13_v_slices`    | (1, nkv, K, 256) | |
| `kv14_k_slices`    | (1, nkv, K, 512) | New K at L14, all K positions (for sharing) |
| `kv14_v_slices`    | (1, nkv, K, 512) | |

**For chunks 3/4:** instead of passing the full updated kv13/kv14, pass the *pre-verify* kv13/kv14 (already in persistent cache) PLUS the new slices appended logically. Chunk3/4 still receive:
- `kv13_k` shape (1,1,W,256) — the pre-verify window
- `kv14_k` shape (1,1,ctx,512) — the pre-verify full cache

The verify graph builds a *temporary* extended key for attention at each new position:
```
K_for_attn_at_Pt = cat([kv13_k[:, :, 1:, :], new_k13_at_Pt], dim=2)  # temporary, W slots
```
This is never written back. Chunks 3/4 receive the pre-verify kv13/kv14 unchanged.

### verify_chunk3 and verify_chunk4

No KV outputs. Already read-only. No changes required except removing `update_indicator` from inputs if present.

**verify_chunk4 output addition:**

| New output | Shape | Notes |
|------------|-------|-------|
| `token_ids` | (K,) int32 | Per-position argmax (existing) |
| `hidden_states_out` | (1, K, 1536) | Existing; Swift takes slice [N-1] for bonus decode |

---

## Swift Write-Back Algorithm

### Data structures

After the verify call, Swift holds:
- `newKSlidingC1`: (7, 1, K, 256) — new K at each of 7 sliding layers, all K positions
- `newVSlidingC1`: (7, 1, K, 256)
- `newKFullC1`:    (1, 1, K, 512)
- `newVFullC1`:    (1, 1, K, 512)
- `newKSlidingC2`: (5, 1, K, 256)
- `newVSlidingC2`: (5, 1, K, 256)
- `newKFullC2`:    (2, 1, K, 512)
- `newVFullC2`:    (2, 1, K, 512)
- `kv13Slices`, `kv14Slices`: (1, 1, K, 256/512)
- `N`: accepted count (from acceptance decision)

### Acceptance decision (unchanged from current)

```swift
var accepted: [Int32] = [tTokNext]
for k in 0..<K {
    if proposals[k] == targetArgmax[k] { accepted.append(proposals[k]) }
    else { accepted.append(targetArgmax[k]); break }
}
let N = accepted.count  // 1..K+1
```

### Write-back

For each accepted position `n` in `0..<N`:
- The corresponding verify position index is `n` (0-based, mapping to position `currentPosition + n`)

**Sliding layers (chunk1, 7 slots):**
```swift
for slot in 0..<7 {
    // Shift existing window by N positions
    // Then write new slices into positions [W-N .. W-1]
    let k_new_slice = newKSlidingC1[slot, 0, 0..<N, :]  // (N, hd)
    let v_new_slice = newVSlidingC1[slot, 0, 0..<N, :]
    shiftAndAppend(kSliding1, slot: slot, newSlices: k_new_slice)
    shiftAndAppend(vSliding1, slot: slot, newSlices: v_new_slice)
}
```

`shiftAndAppend(buf, slot, newSlices)`:
1. `memmove` the last `W - N` entries of slot `slot` to the front (i.e., shift left by N)
2. Copy `newSlices` (shape N × hd) into the last N entries of slot `slot`

**Full attention layers (chunk1, 1 slot = L4):**
```swift
for n in 0..<N {
    let pos = currentPosition + n
    kFull1[0, 0, pos, 0..<hd] = newKFullC1[0, 0, n, 0..<hd]
    vFull1[0, 0, pos, 0..<hd] = newVFullC1[0, 0, n, 0..<hd]
}
```

Same pattern for chunk2 sliding slots (5 layers) and chunk2 full slots (2 layers).

**kv13 (sliding, W-sized):**
```swift
shiftAndAppend(kv13KCache, slot: 0, newSlices: kv13Slices[0..<N])
shiftAndAppend(kv13VCache, slot: 0, ...)
```

**kv14 (full, ctx-sized):**
```swift
for n in 0..<N {
    kv14KCache[0, 0, currentPosition + n, :] = kv14Slices[n, :]
    kv14VCache[0, 0, currentPosition + n, :] = kv14Slices[n, :]
}
```

**Advance position:**
```swift
currentPosition += N
```

### Bonus token (K+1) decode

When all K drafts accept (N = K+1), the bonus token is the target's argmax at position K, returned in `token_ids[K-1]` — this is already in the verify output.

The bonus token's K/V are NOT in the verify slices (verify only ran K positions). For the bonus token:
- Its hidden state is available: `hidden_states_out[K-1]` from verify_chunk4
- **Option A (simple):** run one full T=1 decode step for the bonus position. This is the same cost as the current `commitAccepted` single decode, but now only triggered on the all-accept path (K+1 case), not every burst.
- **Option B (deferred):** carry the bonus hidden and run the full T=1 decode at the *start* of the next burst (before drafting). Same correctness, slightly different scheduling.

**Option A is recommended for the initial implementation.** Option B can be a follow-on.

---

## Composability with EAGLE-3 Blocker 2

EAGLE-3 Blocker 2 (`commitAccepted` replaying T=1 per token) is fully replaced by the write-back algorithm above. The two fixes are the same fix: once verify returns per-position K/V slices and Swift commits only the accepted prefix, there is no need for per-token replay.

The EAGLE-3 document's "K/V direct-write + 1 decode" entry in the burst-formula table corresponds exactly to this contract, projecting 40.7 tok/s at N=3.05.

---

## KV Cache Buffer Compatibility

The IOSurface-backed `MLMultiArray` buffers already in use (`kSliding1`, etc.) are the target of Swift's write-back. The format is:
- Sliding: `(num_slots, 1, W, max_hd)` fp16, zero-padded to `max_hd=512`
- Full: `(num_slots, 1, ctx, max_hd)` fp16, zero-padded to `max_hd=512`

The new K/V slice outputs from the graph will be:
- Sliding: `(num_slots, 1, K, hd)` fp16 where `hd < max_hd` for sliding layers
- Full: `(num_slots, 1, K, hd)` fp16

Swift zero-pads `hd → max_hd` before writing into the cache buffer. This matches the existing padding logic in `_run_layer_swa`:
```python
if hd < max_hd:
    k_padded = F.pad(k, (0, max_hd - hd))
```
The graph already pads; outputs should match what is currently written into the in-graph buffers.

---

## What Does NOT Change

- Attention computation inside each layer is unchanged; logits and argmax are identical.
- Chunk3/4 KV-sharing structure is unchanged; they still read kv13/kv14 passed as inputs.
- `decode_q1` function inside the multi-function mlpackage is completely unchanged.
- Swift embedding, RoPE, mask-building, PLE code is unchanged.
- The `SpeculativeLoop.drawBurst` acceptance decision logic is unchanged.

---

## Open Questions for the Implementation Session

1. **Graph export size:** returning `(num_slots, 1, K, hd)` outputs per chunk adds ~4 new output tensors per chunk. For K=3 these are small (e.g., 7×1×3×256×2 bytes = 10.7 KB). Weight file is unchanged. ANE dispatch overhead should be unaffected.
2. **`update_indicator` removal:** the full-attention layers use `update_indicator` as a mask to blend new K/V into the pre-existing ctx-length buffer (`K_full_out = K_full_slot * (1-mask) + k_padded * mask`). In the new design this blend does not happen inside the graph. Verify the graph still computes the correct K/V for attention (it should — attention over full cache + new position is computed before the blend, using the same `K_for_attn` path).
3. **Stale kv13/kv14 in chunk2:** in the new design, chunks 3/4 receive the *pre-verify* kv13/kv14 (from the persistent cache), not the freshly updated ones. This is semantically correct because verification is computing logits for positions P..P+K-1, and the attention context for those positions should be the prefix up to P, not including the new positions themselves. Confirm via Mac parity test.
4. **kv13 sliding: correct shift size for attention inside verify.** When verify_chunk2 runs L13, it must build a K-position-wide K_for_attn from the existing W-slot window. For position P+t (t in 0..K-1), the correct attention window is `kv13[W-(P+t)'s position in window..] + newly-computed k13 for P..P+t-1`. This is equivalent to a causal build-up within the verify call. The `causal_mask_sliding` input already encodes this triangular structure — confirm its mask shape `(1,1,K,W)` is computed correctly in `makeVerifySlidingMask`.
