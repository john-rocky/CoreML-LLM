# ANE-only Levers for TTFT/Decode (post-Gallery analysis, 2026-04-23)

## Scope

Research pass on LiteRT-LM / AI Edge Gallery (see `MOBILE_RUNTIME_COMPARISON.md`,
`LITERT_CONTAINER_ANALYSIS.md`) surfaced a candidate list of speedups. This
doc filters that list under two project constraints:

- **ANE-only.** No Metal/GPU decode path for now.
- **MLState is off the table.** Prior attempt failed; current KV approach is
  persistent `MLMultiArray` buffers with IOSurface backing (see
  `ChunkedEngine.swift:50-124`, `OUR_STACK_ANATOMY.md:48-57`).

Gallery's published iPhone 17 Pro "TTFT 0.3s / 56.5 tok/s" is its **GPU**
number (ML Drift / Metal), not ANE. Matching it on ANE requires a different
route: smaller prefill, not faster prefill.

## Reference numbers

| Metric | Our E2B v0.8.0 (iPhone 17 Pro, ANE) | Gallery E2B (iPhone 17 Pro, GPU) |
|---|---|---|
| Prefill | 154 tok/s | 2878 tok/s |
| Decode | 31 tok/s | 56.5 tok/s |
| TTFT (512 prompt) | ~3.3 s (prompt / prefill tok/s) | 0.3 s |
| Bundle size | 3.1 GB | 2.58 GB |

Sources: `README.md:27-38`, LiteRT-LM HF card `litert-community/gemma-4-E2B-it-litert-lm`.

## Re-evaluation of original candidate list

| # | Candidate | Status in our stack | Verdict |
|---|-----------|---------------------|---------|
| #3 | Startup preload + warmup | Eager load + two-phase prewarm (4+8 decode steps, `ChunkedEngine.swift:607-647`), `onProgress "Ready"` | **Shipped**. Missing: prefill-path warmup (see #1 below). |
| #4 | Engine / conversation split | Singleton engine, `reset()` zeros KV only; no model destruction per chat | **Shipped**. |
| #5 | Multiple prefill lengths | Fixed `PREFILL_N=512` (`gemma4_prefill_chunks.py:29`); short prompts pay full 3.3 s | **Not done. Top TTFT lever.** |
| #6 | Sliding window for local layers | W=512, 4:1 local:global, 28 sliding / 7 full (`config.py`, `gemma4_swa_chunks.py:4`, mask construction at `ChunkedEngine.swift:843-880`) | **Shipped**. |
| #7 | Shared KV for global layers | Cross-layer alias done (L13→sliding, L14→full; L15-34 are Q-only, consume `kv13_*` / `kv14_*`). **Within-layer K=V alias per Gemma 4 design NOT done** — we store `kv14_k` and `kv14_v` as separate tensors even though the architecture guarantees equality in global layers. | **Half-done**. Remaining work cuts global-layer KV ~50%. |

## Prefill size math (for #5 multifunction)

Multifunction mlpackage (`coremltools.utils.save_multifunction`) deduplicates
weights by hash, so variants share the underlying palletized weights and only
add graph/MIL spec.

- Current E2B bundle: **3.1 GB**, weights ~500-700 MB post-quant (INT4
  per_grouped_channel group_size=32; INT8 embeddings external).
- Per-variant MIL spec, 4 chunks combined: rough **20-80 MB**.
- Adding three variants (64 / 128 / 256 alongside 512): **+60-240 MB (+2-8%)**.
- If weight dedup does not apply (verify on Mac first): 3× bundle size → NG.

⚠ Apple docs do not confirm ANE execution of multifunction mlpackages. First
validation step must be an on-device load + predict test with a
two-variant spike.

## Newly-found ANE-scope levers (from the audit)

### A. Prefill path warmup (hours → done in this commit)
`finalPrewarm()` previously warmed decode (8 steps) + verify, not prefill.
Cold first prefill pays ANE specialization on top of the normal 3.3 s. Added
one dummy `runPrefill(tokenIDs: [0])` to `finalPrewarm()`; 1-token input is
enough because the buffer width is still `prefillN` and ANE compiles by shape.

### B. Chunk pipelining during decode (week)
`BASELINE_SPEED_AUDIT.md:59-64` observed chunks run serially with no
pipelining. ANE is idle during CPU prep/copy between chunks. Async dispatch
of chunk N+1 prep during chunk N compute could gain **+10-20% decode tok/s**.

### C. System-prompt KV persistence to disk (days)
`reset()` zeros KV; the system prompt is re-prefilled every new conversation.
Serialize the post-system-prompt KV snapshot once, restore on `reset()` to
skip that prefill entirely. First-message TTFT of every new session drops
to decode-only. The prefix cache machinery at
`ChunkedEngine.swift:971-1001` can be the foundation (already handles
`restoreKVSnapshot`).

### D. Within-layer K=V alias for global layers (#7 remainder, week)
Export surgery: emit single `kv14_kv` output from Chunk 2 instead of
separate `kv14_k` / `kv14_v`, consume same tensor for both K and V in
downstream global layers. Halves global-layer KV memory and bandwidth —
meaningful at long context (global KV grows linearly with context while
local caps at 512).

### E. Bundle size trim (days)
Per-layer embeddings (Gemma 4: 262144 × 256 × 35 layers) likely dominate
the 3.1 GB vs Gallery's 2.58 GB. Audit what precision they ship at and
whether an INT4 path is viable. Affects load time / RAM, not TTFT.

## Priority order

| Rank | Item | Effort | Expected gain | Validation |
|------|------|--------|----------------|------------|
| 1 | **A. Prefill warmup** | hours (done) | first-TTFT -50-300 ms | iPhone: measure first vs second user message TTFT |
| 2 | **#5 Multi-prefill length (64/128/256/512) via multifunction** | 1-2 weeks | short-prompt TTFT **3.3 s → 0.3-0.5 s** | Mac spike (2 variants) first, verify weight dedup + ANE load |
| 3 | **B. Chunk pipelining** | week | decode **31 → 35-37 tok/s** | iPhone bench before/after |
| 4 | **#7 (remainder) Within-layer K=V alias** | week | -25% total KV at long ctx, slight decode gain | Long-ctx decode bench |
| 5 | **C. System-prompt KV persistence** | days | new-session first-TTFT ≈ 0 when prompt fixed | iPhone: new chat with fixed system prompt |
| 6 | **E. Bundle size trim** | days | load/RAM; no TTFT | Size + eval accuracy |

## Action this pass

1. ✅ **A implemented** (prefill added to `finalPrewarm()`, `ChunkedEngine.swift`).
2. ✅ **Transition warmup added** — one extra prefill+decode cycle at end of
   `finalPrewarm()` to pre-pay the prefill→decode kernel handoff (observed
   27ms cold cost on first real decode after prefill).
3. ✅ **#5 Mac spike passed** (`conversion/spikes/multifunction_prefill_spike.py`).
   On a 7-layer 1536-dim stand-in: n64 variant 7.9MB, n512 variant 7.9MB,
   sum 15.8MB, **merged 7.9MB** (1.00x of larger, perfect dedup). Both
   functions load via `function_name=`. Clears the gate on multi-prefill
   lengths — graph-only delta is below measurement granularity at this scale.
4. ⏳ **ANE on-device multifunction load test** — wire a minimal 2-variant
   into a real Gemma 4 chunk, ship to iPhone, confirm ANE executes
   function-selected paths. Only hard unknown left.
5. ✅ **Deferred prefill chunk load** (`LLM_DEFER_PREFILL=1`). Existing
   device bench (see `ChunkedEngine.swift:324`) shows parallel load is
   *slower* than sequential because ANECompilerService serializes
   internally, so straight parallelism is a dead end. Structural lever
   instead: decode chunks (~35s) are enough to start chatting; prefill
   chunks (~30s more) load in a background `Task.detached`. Call sites
   already fall back to per-token decode via the `hasPrefill` gate, so
   the engine is usable during the load window. Prefill storage moved
   behind `NSLock` to make `attachPrefill` atomic. Opt-in env var for
   A/B on device; default off until verified.

## Known unknowns

- ANE execution of multifunction mlpackages — Apple docs silent. Must verify
  on-device.
- Whether `runPrefill(tokenIDs: [0])` warms the exact same ANE graph that a
  real prompt will hit. Buffer width matches, token pattern doesn't — should
  be enough but confirm via first-user-message TTFT delta.
- Our 154 tok/s prefill vs Gallery CPU 532 tok/s — is this pure ANE
  architecture (large matmul on A18 AMX beats ANE's TOPS profile for this
  shape), or is there headroom on the ANE side too? Needs deeper instrumenta
  tion of per-chunk ANE time vs CPU active time.
