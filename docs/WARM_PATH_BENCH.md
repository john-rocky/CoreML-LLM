# Warm-Path Runtime Bench (V6-1, V6-2, outputBackings, scratch reuse)

**Date:** 2026-04-15
**Scope:** Track the per-chunk dispatch-overhead win from the runtime
warm-path optimizations landed in this branch. None of these change the
model graph or numerical outputs — they target Swift-side allocation,
plan-build, and shape-trace overhead.

Baseline (per `docs/BASELINE_SPEED_AUDIT.md`, Mac Studio):
`c1=11.0 c2=12.8 c3=11.8 c4=16.1` ms (sum 51.7 ms), 19.4 tok/s.

iPhone 17 Pro reference baseline: 31.4 tok/s @ 2K (per project memory).

---

## What landed

| ID | Item | Where | Predicted gain (per step) |
|----|---|---|---|
| **V6-1** | `MLModelConfiguration.optimizationHints.reshapeFrequency = .infrequent` (iOS 18.2+) | `ChunkedEngine.load`, `CoreMLLLM.load` | -0.3 to -0.6 ms |
| **V6-2** | `MLComputePlan.load(...)` warm-pool, retained on engine | `ChunkedEngine.load` | First-token: -0.8 ms; steady-state: 0 (already prewarmed) |
| **outputBackings** | Pre-allocated MLMultiArray output buffers reused every step via `MLPredictionOptions.outputBackings`, applied to chunk1–chunk4 decode | `ChunkedEngine.predictStep` | -0.1 to -0.4 ms across 4 chunks (eliminates IOSurface alloc + refcount churn) |
| **scratch input reuse** | Reuse persistent `MLMultiArray` scratch buffers for hidden_states, per_layer_raw, and 4× RoPE rows. New `EmbeddingLookup.lookupInto(_:dst:)` writes directly into the scratch pointer. New `lookupRoPEInto` ditto. | `ChunkedEngine.predictStep` | -0.05 to -0.2 ms |
| **bench gating** | `WARM_PATH_BENCH=1` env var or UserDefaults bool turns on extra logging without running on shipping releases | `ChunkedEngine` (cached static) | n/a |

Aggregate predicted gain on iPhone 17 Pro warm path: **+1.0 to +2.5 ms / step → +1 to +5 tok/s**.

The existing `[Profile]` log line in `ChunkedEngine.predictStep` (printed on
step 1 and every 10th step) is the per-chunk timing source; no extra
hooks were needed.

## What was deferred and why

| Item from task list | Status | Reason |
|---|---|---|
| Async predict pipelining | Not implemented | The four chunks have a strict data dependency (`c1.h → c2 in → c2.h → c3 in → ...`); no parallelism is available within a single decode step. Cross-step overlap (staged pipelining) is tracked separately in `BASELINE_SPEED_AUDIT.md` as a multi-day Phase D1 effort, not a runtime micro-op. |
| MLDictionaryFeatureProvider → MLMultiArrayFeatureProvider | Not implemented | The chunk inputs are heterogeneous (hidden_states, masks, RoPE, KV) which `MLDictionaryFeatureProvider` already represents directly. Replacing it would require a custom MLFeatureProvider subclass with no measurable win because the dictionary literal is built on the stack and the bottleneck is ANE dispatch, not feature-provider lookup. |
| iOS 26 `preferredSchedulingPriority` | Not implemented | No such public API surfaced as of coremltools 9.0 / iOS 26 SDK. Left a `// TODO` placeholder is unnecessary since adding one without a known API contract is dead code. Re-evaluate if Apple ships an iOS 26.x point release with new `MLPredictionOptions` / `MLModelConfiguration` flags. |
| KV-output aliasing (zero-copy) | Backed out | Initial implementation aliased `K_*_out` / `V_*_out` directly to the persistent input KV buffers. This introduces a read-after-write hazard inside the chunk's attention rewrite (the chunk reads the pre-rolled cache to compute attention before writing the new entry). Kept the explicit `copyBack(...)` memcpy for parity safety. We still win the per-step output-buffer allocation; the memcpy itself is small (~0.05 ms total). |

## How to verify on device

1. Pull this branch, install the CoreMLLLMChat example app on iPhone 17 Pro.
2. Bench mode: set the env var via Xcode scheme → Run → Arguments → Environment Variables: `WARM_PATH_BENCH=1`. The extra `[WarmPath]` log line confirms backings were wired.
3. Run a single 100-token generation on each prompt category from `Sources/accept-rate-bench/Prompts.swift` (`chat-define-transformer`, `code-complete-sum`, `qa-where-is-swift`, `sum-para-ane`).
4. Capture the final `[Profile] ... c1=... c2=... c3=... c4=...` line for each run. Compare against the same prompts on `main`.

CSV schema for collation (`/tmp/warm-path-bench.csv`):

```
prompt,branch,c1_ms,c2_ms,c3_ms,c4_ms,sum_ms,total_ms,tok_per_s
chat,main,11.0,12.8,11.8,16.1,51.7,51.9,19.3
chat,warm,10.7,12.4,11.5,15.7,50.3,50.5,19.8
...
```

## Acceptance criteria

- `[ComputePlan] warm: 4 plan(s) in <0.5s` line appears on iPhone load.
- First decode step from a clean reset is ≤ baseline first-step latency
  (V6-2 effect is most visible here — steady-state may be unchanged
  because the existing 4-step prewarm already builds the plan).
- Steady-state per-chunk times drop by ≥ 0.1 ms on at least one chunk.
- No quality regression vs `main` (compare first-100 tokens byte-for-byte
  on a fixed prompt; warm-path changes are numerically lossless by
  construction).

## Rollback

All four items are independent commits. To revert one, `git revert
<sha>` is safe — no shared state was added beyond the engine's
`backings*` / `options*` / `warmComputePlans` properties.
