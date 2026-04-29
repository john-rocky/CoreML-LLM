# Qwen3.5 future improvement ideas

After v1.8.0 ship: 0.8B 48→43 tok/s + 2B 27→25 tok/s clean output on
iPhone 17 Pro using full-vocab rep_penalty workaround for fp16 ANE
bias. These ideas are NOT yet implemented; record here for the next
iteration.

## C. Multi-token batched prefill

**Goal:** cut TTFT for long prompts (>20 tokens) by ~3-5×.

**Approach:** build a separate `chunk_*_prefill.mlpackage` that takes
seq=N input tokens at once, processes them in parallel where possible.

**Challenge:** Qwen3.5 hybrid architecture limits batching benefit:
- 18 of 24 layers are **SSM (Gated DeltaNet)** — sequential by design
  (`rec_state` updates token-by-token), zero batching gain.
- 6 of 24 layers are **full attention** — batched matmul + slice_update
  KV cache scales linearly with seq, ~5-10× faster than recurrent.
- Net: 25% layers benefit fully, 75% remain sequential. Expected
  speedup ~2-3× for prefill (vs 5-10× for pure transformers).

**Cost:** doubles bundle size (separate prefill mlpackages), adds
KV-state-transfer logic at the prefill→decode boundary. iPhone A18
ANE rejects multi-StateType, so prefill and decode chunks must share
the same single MLState (KV-only) — likely requires custom MLState
dispatch helpers in Swift.

**Verdict:** Worth it if average prompt length > 30 tokens. For chat
use cases (5-15 tokens typical), TTFT is already < 200 ms with the
warm-up + Swift-sampling-skip combo.

## D. SSM state in MLState (multi-StateType packing)

**Goal:** eliminate per-step host I/O of ~10 MB SSM state, reducing
thermal throttle from 10% → 2-3%.

**Current cost:** 18 SSM layers × (conv_state 49 KB + rec_state 524 KB)
= ~10 MB Swift↔ANE I/O per step. Continuous bus pressure heats up
the chip faster than pure-transformer models.

**Approach:** pack all 18 SSM layers' conv_state + rec_state into one
giant MLState `state` tensor (e.g., 18 × 573 KB ≈ 10.3 MB resident
on ANE, never crosses the bus during decode).

**Challenge:**
- Indexing into a flat MLState buffer requires `slice_update` per
  layer per call — 18 × 2 = 36 slice_updates per chunk per step.
  Each slice_update has fixed cost; may exceed the I/O it replaces.
- ANE compiler may not fuse 36 slice_updates efficiently — risk of
  ANE rejection or silent CPU fallback for the slice ops.
- Multi-StateType (kv_cache + ssm_state as two MLStates) was the
  cleanest design but iPhone A18 rejects with Error 11.

**Verdict:** Architecture-level change. Try after multi-StateType is
fixed in a future iOS release, OR accept the 10% throttle as Qwen3.5's
"hybrid SSM tax" on iPhone.

## E. INT8 SSM state

**Goal:** halve the 10 MB/step SSM state I/O via INT8 quantization
of `conv_state` and `rec_state` at the chunk boundary.

**Approach:**
- Output `conv_state` / `rec_state` as INT8 from each chunk (with
  per-tensor scale/zero-point).
- Input as INT8 to next step, dequantize-on-load inside the chunk.

**Cost:** quantization noise accumulates per step. SSM recurrence
multiplies state by gates each step; rounding error compounds. Need
empirical drift testing — likely OK for short generations (~50 tok),
risky for long (>200 tok).

**Effort:** medium. Need to build a quant/dequant module in PyTorch,
re-trace, ensure ANE accepts the INT8 tensor I/O.

**Verdict:** Worth a one-day spike. If drift over 256 tokens stays
below 5% top-1 mismatch vs fp16-state path, ship it.

## F. MPS GPU lm_head matmul

**Goal:** alternate path for environments where ANE fp16 isn't even
available (older iOS / iPadOS / macOS without Apple Silicon).

**Approach:** use `MPSMatrixVectorMultiplication` for the lm_head
matmul on Metal GPU with fp16 weights. fp32 accumulator built in
on iPhone GPU; gives clean output without CPU cblas overhead.

**Speed estimate:** ~3-5 ms head matmul on iPhone A18 GPU
(vs current ANE+Swift sampling ~3-5 ms total). Marginal benefit on
A18 + ANE, but valuable as fallback for unsupported devices.

**Effort:** medium. Metal buffer setup + kernel launch + sync.

**Verdict:** Defer until a non-ANE platform actually requires it.

## G. Per-channel palettize lm_head

**Goal:** reduce fp16 reduction tie frequency on iPhone A18 ANE,
potentially eliminating the need for full-vocab rep_penalty.

**Approach:** `palettize_weights(granularity="per_grouped_channel",
group_size=64)` on the lm_head Conv2d (vocab=248K outputs, hidden
input). Each output token's weight row gets its own scale, preserving
relative magnitudes.

**Cost:** per-channel scales add ~1 MB to bundle (248K fp16 scales).
ANE compatibility needs verification — palettize per-grouped-channel
may force CPU/GPU fallback for some ops.

**Effort:** small (config change in build script).

**Verdict:** Easy experiment. If iPhone A18 produces clean greedy
output without rep_penalty, we could drop the rep_penalty=1.1
default and improve quality slightly.

## H. Speculative decoding via small Qwen3.5 drafter

**Goal:** 1.5-2× tok/s by speculatively drafting K tokens with a
smaller model, verifying with the main model.

**Challenge:** Qwen3.5 family's smallest is 0.8B. No "drafter-sized"
variant. Could try Qwen2.5-0.5B as drafter — different vocab, would
require token-id translation.

**Verdict:** Spec decode infrastructure already exists in this repo
(MTP / EAGLE-3 work). Adding Qwen3.5 spec is an extension day, but
needs a compatible drafter model first.

## I. Reduce body chunk count (4 → 2)

**Goal:** cut 4 ANE dispatches per step to 2.

**Memory result (already tested):** "2-chunk works but only +0.7
tok/s (31→31.7); dispatch overhead is not the bottleneck"
([chunk_consolidation_blocked.md](../../.claude/projects/-Users-majimadaisuke-Downloads-CoreML-LLM/memory/chunk_consolidation_blocked.md)).

**Verdict:** Marginal. Skip.
