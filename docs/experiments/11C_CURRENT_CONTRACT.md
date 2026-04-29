# 11C Audit: Current Verify I/O Contract

**Status:** Phase 0 audit — 2026-04-17. Read-only; no code changed.

---

## Layer / Chunk Topology

Gemma 4 E2B has 35 layers split across 4 chunks:

| Chunk | Layers | KV type | Cache buffers |
|-------|--------|---------|---------------|
| chunk1 | L0–L7  | 7 sliding (W=512) + 1 full (L4) | `kSliding1` (7,1,W,512), `kFull1` (1,1,ctx,512) |
| chunk2 | L8–L14 | 5 sliding + 2 full (L9,L14) | `kSliding2` (5,1,W,512), `kFull2` (2,1,ctx,512) |
| chunk3 | L15–L24 | all KV-shared (reads kv13/kv14) | — (no own cache) |
| chunk4 | L25–L34 + norm + lm_head | KV-shared | — (no own cache) |

L13 is the last sliding layer owned by chunk2; L14 is the last full layer owned by chunk2.
Their K/V outputs are tagged `kv13_k/v` and `kv14_k/v` and are passed into chunks 3 and 4.

---

## Current `verify_qK` I/O Contract (K=3, ctx=2048, W=512)

### verify_chunk1 (L0–L7)

| Direction | Tensor name | Shape | Dtype | Notes |
|-----------|-------------|-------|-------|-------|
| IN | `hidden_states` | (1,K,1536) | fp16 | Batch-embedded draft tokens |
| IN | `causal_mask_full` | (1,1,K,ctx) | fp16 | Full-attn mask for K positions |
| IN | `causal_mask_sliding` | (1,1,K,W) | fp16 | SWA mask for K positions |
| IN | `update_indicator` | (1,1,ctx,K) | fp16 | One-hot write mask per position |
| IN | `per_layer_raw` | (1,K,nlayers*pld) | fp16 | Per-layer embedding concat |
| IN | `cos_s`,`sin_s` | (1,1,K,256) | fp16 | Sliding RoPE for K positions |
| IN | `cos_f`,`sin_f` | (1,1,K,512) | fp16 | Full RoPE for K positions |
| IN | `K_sliding_in` | (7,1,W,512) | fp16 | **Persistent cache read-in** |
| IN | `V_sliding_in` | (7,1,W,512) | fp16 | Persistent cache read-in |
| IN | `K_full_in` | (1,1,ctx,512) | fp16 | Persistent cache read-in |
| IN | `V_full_in` | (1,1,ctx,512) | fp16 | Persistent cache read-in |
| OUT | `hidden_states_out` | (1,K,1536) | fp16 | |
| OUT | `per_layer_combined_out` | (1,K,nlayers*pld) | fp16 | |
| **OUT** | **`K_sliding_out`** | **(7,1,W,512)** | **fp16** | **Full updated KV cache** |
| **OUT** | **`V_sliding_out`** | **(7,1,W,512)** | **fp16** | **Full updated KV cache** |
| **OUT** | **`K_full_out`** | **(1,1,ctx,512)** | **fp16** | **Full updated KV cache** |
| **OUT** | **`V_full_out`** | **(1,1,ctx,512)** | **fp16** | **Full updated KV cache** |

Outputs in **bold** are *immediately* copied back into the engine's persistent cache buffers (`kSliding1`, `vSliding1`, `kFull1`, `vFull1`) via `copyBack()` before chunk2 runs.

### verify_chunk2 (L8–L14)

Same input/output pattern as chunk1 but with 5 sliding + 2 full slots, plus extra outputs:

| Extra OUT | Shape | Notes |
|-----------|-------|-------|
| `kv13_k`, `kv13_v` | (1,1,W,256) | Sliding KV from L13, passed to chunks 3/4 |
| `kv14_k`, `kv14_v` | (1,1,ctx,512) | Full KV from L14, passed to chunks 3/4 |

Same write-through pattern: `K_sliding_out` / `V_sliding_out` / `K_full_out` / `V_full_out` are copied back into `kSliding2`, `vSliding2`, `kFull2`, `vFull2` immediately.

### verify_chunk3 (L15–L24) — KV-shared, read-only

| Direction | Tensor | Shape | Notes |
|-----------|--------|-------|-------|
| IN | `hidden_states` | (1,K,1536) | from chunk2 |
| IN | `causal_mask_full`, `causal_mask_sliding` | (1,1,K,ctx/W) | |
| IN | `per_layer_combined` | (1,K,nlayers*pld) | |
| IN | `cos_s`,`sin_s`,`cos_f`,`sin_f` | as above | |
| IN | `kv13_k`,`kv13_v` | (1,1,W,256) | shared from chunk2 out |
| IN | `kv14_k`,`kv14_v` | (1,1,ctx,512) | shared from chunk2 out |
| OUT | `hidden_states_out` | (1,K,1536) | No KV writes |

No KV write-back in chunk3. All layers read from the fixed kv13/kv14.

### verify_chunk4 (L25–L34 + LM head)

Same KV-shared read pattern as chunk3. Outputs:

| OUT | Shape | Notes |
|-----|-------|-------|
| `token_ids` | (K,) int32 | Target argmax at each of K positions |
| `hidden_states_out` | (1,K,1536) | Used for MTP drafter carry state |

---

## The Bug: KV Write-Through During Verify

```
Timeline of a K=3 verify call at position P:
  
  Draft proposals: [d0, d1, d2]
  verifyTokens fed to graph: [tTokPrev, d0, d1]  (slot 0 = tTokPrev, slots 1-2 = drafts)
  
  verify_chunk1 runs:
    - Reads kSliding1 / kFull1 (current prefix up to position P)
    - Computes new K/V for positions P+0, P+1, P+2 (all three draft slots)
    - **Writes updated kSliding1 / kFull1 immediately** (copyBack)
    ← kSliding1 now contains KV at P+0, P+1, P+2, including d0 and d1

  verify_chunk2 runs:
    - Reads kSliding2 / kFull2 (still clean)
    - **Writes updated kSliding2 / kFull2 immediately**
    - Produces kv13_k/v, kv14_k/v containing contributions from d0, d1
    ← kv13_k/v, kv14_k/v now conditioned on d0 and d1

  verify_chunk3 runs:
    - Reads kv13/kv14 produced above (contaminated with d0, d1)
    - Produces hidden states conditioned on the contaminated cache

  verify_chunk4 runs:
    - argmax[0] at position P+0 = target's token T0
    - argmax[1] at position P+1 sees attention over keys including K(d0)
    - argmax[2] at position P+2 sees attention over keys including K(d0), K(d1)
    ← argmax[1] and argmax[2] are CONTAMINATED by unaccepted draft KV
```

### What "contamination" means at the next burst

After the verify call returns, the engine calls `commitAccepted(accepted)` which **only advances `currentPosition`** (Swift line 1429–1434). It does not undo any KV writes.

The next decode step starts with `kSliding1`, `kFull1`, `kSliding2`, `kFull2` already containing KV entries from the *rejected* draft tokens `d1` (if only `d0` was accepted). Layer-13's sliding window has been shifted by 3 positions regardless of how many tokens were accepted.

This is the mechanism identified by PR #72 and documented in `docs/PHASE_B_DECISION.md`:
> "verify writes drafter proposals into the KV cache at positions P+1..P+K-1 before the acceptance decision. Subsequent target argmaxes condition on contaminated cache."

---

## Swift Call Path Summary

```
SpeculativeLoop.drawBurst()
  └─ target.verifyCandidates(tokens, K)          ← calls ChunkedEngine
       └─ verifyCandidates(tokens:startPosition:)
            ├─ verifyChunk1.prediction(...)
            │    └─ copyBack(out1, "K_sliding_out", into: kSliding1)  ← WRITE
            │    └─ copyBack(out1, "V_sliding_out", into: vSliding1)  ← WRITE
            │    └─ copyBack(out1, "K_full_out",    into: kFull1)     ← WRITE
            │    └─ copyBack(out1, "V_full_out",    into: vFull1)     ← WRITE
            ├─ verifyChunk2.prediction(...)
            │    └─ copyBack → kSliding2, vSliding2, kFull2, vFull2   ← WRITE
            │    └─ kv13k/v, kv14k/v captured for chunks 3/4
            ├─ verifyChunk3.prediction(...)        ← read-only
            └─ verifyChunk4.prediction(...)        ← read-only, returns token_ids
  └─ acceptance decision in Swift (SpeculativeLoop.drawBurst step 5)
  └─ target.commitAccepted(accepted)
       └─ currentPosition += tokens.count         ← only counter advance, no KV fixup
```

**KV writes happen at verify_chunk1 + verify_chunk2 graph boundaries, which is before the acceptance decision. There is no rollback path.**

---

## Summary of Issues

| # | Issue | Location |
|---|-------|----------|
| 1 | KV written for all K positions (incl. rejected) before acceptance decided | `verifyCandidates` lines 776–800 |
| 2 | Sliding window shifted by K regardless of actual acceptance count | `_run_layer_swa` cat+shift in chunk1/2 |
| 3 | `commitAccepted` never corrects or partially-undoes the KV writes | line 1429 |
| 4 | `kv13_k/v` and `kv14_k/v` passed to chunks 3/4 already include rejected-draft context | chunk2 out lines 801–806 |
| 5 | No per-T-position K/V slice outputs exist; only full cache buffers are emitted | — |

Issue 5 is the structural prerequisite for the fix: the redesign requires chunks to emit per-position K/V slices so Swift can selectively commit only accepted positions.
