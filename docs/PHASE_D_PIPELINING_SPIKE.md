# Phase D1 staged chunk pipelining — feasibility spike

Date: 2026-04-15. Branch: `spike/d1-chunk-pipelining`.
Scope: answer the binary question "can two `MLModel.prediction` calls
on distinct chunk models overlap on Mac CoreML / ANE?" — the
prerequisite for the staged pipelining candidate ranked #1 in
`docs/BASELINE_SPEED_AUDIT.md`.

## TL;DR

**Pipelining is infeasible as currently conceived.** When `chunk1`
and `chunk3` `MLModel.prediction` calls are dispatched on separate
`DispatchQueue`s with `DispatchGroup`-based join, wall-clock is
indistinguishable from running them back-to-back serially. Overlap
factor = 0.02–0.06 across all four prompt categories (1.0 = perfect
overlap, 0.0 = full serialisation).

The ANE serialises distinct-model `MLModel.prediction` submissions at
the driver level. A Swift-side `DispatchQueue` concurrency strategy
cannot produce overlap. The expected 1.5–2.0× audit headroom does
**not** materialise at this layer.

## Methodology

- Host: Mac Studio (M-series, macOS 25.0.0). Release build.
- Model: `~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b`.
- Probe added to `ChunkedEngine.runConcurrencyProbe()`, gated by
  `CHUNK_PIPELINE_SPIKE=1`. Runs once after ANE prewarm, before any
  user-visible decode. Does not modify `predictStep`. Default-off.
- Inputs mirror a real decode step at position=1: `chunk1` receives
  embed(0) + RoPE rows + mask + persistent KV buffers; `chunk3`
  receives `h2` + `kv13/kv14` produced by a prior `chunk2` call.
- For each chunk and for the pair:
  - 10 trials serial `chunk1.prediction`.
  - 10 trials serial `chunk3.prediction`.
  - 10 trials `chunk1` then `chunk3` back-to-back (bound on wall-clock).
  - 10 trials `chunk1` on queue A, `chunk3` on queue B, joined on
    `DispatchGroup`. Both queues `.userInitiated`.
- Overlap factor = `(T_c1 + T_c3 − T_parallel) / (T_c1 + T_c3 − max(T_c1, T_c3))`.
  1.0 = achieved the theoretical overlap ceiling; 0.0 = no overlap.

## Results (4 prompts, one per audit category)

| Prompt category        | c1 (ms) | c3 (ms) | serial both (ms) | parallel (ms) | overlap factor |
|------------------------|--------:|--------:|-----------------:|--------------:|---------------:|
| chat-define-transformer|    5.12 |    7.39 |            12.66 |         12.26 |           0.05 |
| code-complete-sum      |    5.10 |    7.41 |            12.53 |         12.38 |           0.02 |
| qa-where-is-swift      |    5.13 |    7.53 |            12.69 |         12.40 |           0.05 |
| sum-para-ane           |    5.09 |    7.41 |            12.38 |         12.21 |           0.06 |

Parallel dispatch completes in essentially the same wall-clock as
serial dispatch. The ~0.2–0.4 ms improvement over `serial both`
corresponds to removing the Swift-side loop + dict construction
overhead, not actual kernel overlap.

**Decode tok/s with `CHUNK_PIPELINE_SPIKE=1` vs default-off** (same
prompts, 64 max tokens):

| Prompt category        | baseline tok/s | spike tok/s |
|------------------------|---------------:|------------:|
| chat-define-transformer|          33.25 |       33.12 |
| code-complete-sum      |          32.84 |       33.11 |
| qa-where-is-swift      |          32.70 |       32.91 |
| sum-para-ane           |          33.00 |       33.07 |

Within run-to-run noise. The probe runs once pre-stream and has no
effect on user-visible decode (engine `reset()` is called afterwards).

## Interpretation

Even though `chunk1` and `chunk3` are distinct `MLModel` instances
backed by distinct `.mlmodelc` packages with distinct resident weights
on the ANE, the Apple Neural Engine driver appears to serialise
submissions from a single process. Dispatch-level parallelism on the
Swift side cannot beat this. This matches prior anecdotal reports in
`docs/ANE_OPTIMIZATION_SURVEY.md` and general guidance that the ANE
presents a single-queue abstraction to user space.

What was not tested (out of scope for this one-day spike):

- CPU-placed or GPU-placed `MLComputeUnits` for one of the two chunks.
  Could allow genuine overlap, but trades ANE perf for fallback
  compute — likely net-negative on chunk-by-chunk basis.
- Async prediction via `prediction(from:options:completionHandler:)`.
  Same underlying ANE queue; would be surprising to see a different
  result but not tested.
- `MLModelCollection` / batched multi-function predict. Different API
  surface; might be worth a second spike.

## Verdict

**Do NOT pursue full Phase D1 staged pipelining.** The
`1.5–2.0× realistic` target from the baseline audit assumed Swift
`DispatchQueue`-level concurrency would yield kernel overlap, which
this probe disproves. Any further pipelining work on Mac requires
first answering "how do we get the ANE to accept concurrent
submissions" at the CoreML layer (non-trivial, likely requires Apple
engineering contact or a different compute unit mix).

## Concrete next steps

1. **Drop Phase D1 from the priority roadmap.** Update
   `docs/PRIORITY_ROADMAP.md` with this finding.
2. **Escalate the other non-speculative lever** from the audit:
   GPU prefill via MLX-Swift (item 27). TTFT win, independent of the
   decode serialisation constraint discovered here.
3. If staged pipelining must be revisited: run a follow-up spike that
   places `chunk3` on `.cpuAndGPU` while `chunk1` stays on `.all`
   (ANE). Measure whether kernel-level overlap appears under that
   compute-unit split. Budget: half-day. Risk: GPU-placed chunks are
   ~3–5× slower than ANE-placed per prior ad-hoc tests, so the
   overlap has to beat a large deficit.

## Files touched

- `Sources/CoreMLLLM/ChunkedEngine.swift`:
  - Added env-gated `runConcurrencyProbe()` (~130 lines).
  - Called once after prewarm when `CHUNK_PIPELINE_SPIKE=1`.
  - Default decode path (`predictStep`) is unchanged.
- `docs/PHASE_D_PIPELINING_SPIKE.md` (this file).

## Raw data

Logs in `/tmp/d1-spike/` (not committed; regenerate with the commands
in §Methodology).
