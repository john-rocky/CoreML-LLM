# Embedding Bypass Findings

## Honest answer to the brief

**The task's premise — "remove the embedding chunk from the ANE dispatch chain"
— is moot. There is no embedding chunk on ANE.** The CoreML package ships four
decode chunks (`chunk1..chunk4`) and `chunk1`'s first input is already
`hidden_states: (1, 1, hidden)`. Swift performs the int8 embedding gather on
CPU and passes the fp16 tensor into `chunk1`.

Evidence:
- `conversion/models/gemma4_swa_chunks.py:256` — `SWAChunk1.forward` signature:
  `def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, ...)`
  — no `input_ids` anywhere in any chunk.
- `Sources/CoreMLLLM/ChunkedEngine.swift:411` (decode path) — feeds `chunk1`
  with `hidden_states = embedTokens.lookup(tokenID, ...)`.
- `per_layer_raw` (~17 KB/step) is also gathered on CPU
  (`Sources/CoreMLLLM/ChunkedEngine.swift:412` -> `lookupPerLayerRawInto`).

The only four ANE dispatches per decode step are `chunk1..chunk4`. The
embedding has never been in that chain in this repo.

## Prior state of the CPU lookup (already optimized before this session)

- `e4524bc`: Vectorized int8 -> fp16 via vDSP + vImage; 2.0 ms -> <0.5 ms.
- `222e7a0`: PLE raw path switched from scalar loop to `memcpy`; 1.1 ms -> 0.4 ms.

The `[Profile]` line already reports `emb=Xms` averaged across decode steps —
on a warm iPhone 17 Pro run this is typically ~1 ms total (tok + PLE).

## Changes made in this patch (small, honest wins only)

1. **`EmbeddingLookup.lookupInto(tokenID:, dstPtr:)`** — new direct-write API
   that dequantizes int8 -> fp16 into a caller-owned pointer, bypassing the
   per-step `MLMultiArray` allocation. Internally uses the same vDSP +
   `vImageConvert_PlanarFtoPlanar16F` path as before.
2. **Decode scratch pool** — `scratchHiddenIn` (~4 KB fp16) and `scratchPlRaw`
   (~17 KB fp16) added alongside the existing mask pool. Decode path now
   writes directly into these buffers each step instead of allocating.
3. **`lookupPerLayerRawInto`** — new variant that targets the pooled buffer;
   old `lookupPerLayerRaw` retained for any future non-decode callers.
4. **`EMBEDDING_BENCH=1` gated print** — splits CPU embed cost into
   `tok_lookup` (int8->fp16 gather) and `ple_lookup` (int8 memcpy), printed
   alongside the existing `[Profile]` line every 10 steps.

## Predicted savings

Per-step:
- Avoided `MLMultiArray(shape:dataType:.float16)` calls: 2 (hidden + plRaw).
  Foundation allocation + zero-init for these sizes is on the order of
  **~50-150 us** combined on A19. This is a single-digit-percent slice of the
  current ~1 ms `emb` budget.
- No ANE round-trip removed — **none existed** for the embedding.

Expected `emb=` reading: roughly **1.0 ms -> 0.8-0.9 ms**. Total tok/s delta:
well below +1 tok/s. This is an honest, small optimization — the brief's
target of "+5 tok/s from removing an ANE dispatch" is not achievable here
because there is no ANE embedding dispatch to remove.

## The real remaining bottlenecks (for the next session)

Per `[Profile]` output: `c1+c2+c3+c4` sums to ~30 ms/step @ 2K context at
baseline, dominated by the four ANE dispatches (~9 ms overhead + compute).
Moving compute off-ANE is not profitable. Candidate wins:
- Merge chunks so fewer `prediction(from:)` round-trips occur (Track A).
- `outputBackings` wiring for chunk outputs so `copyBack` memcpy disappears.
- Speculative decoding (MTP / EAGLE3) amortizes the 4-chunk cost over
  multiple accepted tokens.

## Device-run instructions

Build and install the Xcode target as usual, then run the chat example with:

```bash
EMBEDDING_BENCH=1 <your usual launch>
```

(Xcode: edit the scheme -> Run -> Arguments -> Environment Variables, add
`EMBEDDING_BENCH = 1`.)

Console will print, every 10 decode steps:

```
[Profile] emb=0.8ms mask=... | c1=... c2=... c3=... c4=... ...
[EmbeddingBench] tok_lookup=0.420ms ple_lookup=0.380ms total_cpu_embed=0.800ms (n=20)
```

Expected ranges on iPhone 17 Pro, iOS 26, warm run:
- `tok_lookup`: 0.35-0.55 ms (int8 gather + vDSP + vImageConvert for 2048 dims)
- `ple_lookup`: 0.30-0.45 ms (int8 memcpy for ~9K dims; no dequant)
- `total_cpu_embed`: < 1 ms, same order as existing `emb=` reading (minus the
  saved alloc overhead).

If `tok_lookup` exceeds ~1 ms, Accelerate isn't being used — check that
`EmbeddingLookup.dequantRow` is the call site (breakpoint in Xcode).

## Files touched

- `Sources/CoreMLLLM/EmbeddingLookup.swift` — factored shared `dequantRow`
  helper, added `lookupInto`, `lookupUnscaled` reuses the helper.
- `Sources/CoreMLLLM/ChunkedEngine.swift` — added `scratchHiddenIn` /
  `scratchPlRaw`, wired decode path to use them, split embed timer, added
  `EMBEDDING_BENCH` gate.
