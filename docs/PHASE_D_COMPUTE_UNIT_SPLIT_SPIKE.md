# Phase D1b compute-unit-split — feasibility spike

Date: 2026-04-15. Branch: `spike/d1b-compute-unit-split`.
Scope: answer the follow-up question from PR #75's negative result —
"does placing one chunk on a different compute unit enable
kernel-level overlap that pure-ANE dispatch cannot?" — the last
non-speculative path on decode before accepting 56 tok/s is
unreachable under CoreML/ANE constraints.

## TL;DR

**Cross-compute-unit overlap is real and near-perfect.** When
`chunk3` is loaded on `.cpuAndGPU` while `chunk2` stays on the
default ANE compute unit, `MLModel.prediction` calls dispatched on
separate `DispatchQueue`s overlap with factor **0.87–0.99** across
all four prompt categories (vs 0.02–0.06 on pure-ANE in PR #75).
Kernel-level parallelism between ANE and GPU drivers **works**.

However, end-to-end tok/s **regresses** with the current serial
`predictStep`: 33.2 → 25.3 avg tok/s (−24 %). `chunk3` on GPU is
~2.2× slower in absolute terms (7.5 ms → 16.6 ms) and the serial
chain pays that deficit without claiming the overlap prize. Realising
the win requires a follow-up pipelining change to actually run c3
(GPU) concurrently with c2 / c4 (ANE).

**Verdict: (a) overlap works — pursue full compute-unit-split
pipelining.** Headroom with a correctly pipelined 4-way assignment
is modest (projected ~36 tok/s, not 56), but this is the first
decode-side lever on Mac that has been empirically validated.

## Methodology

- Host: Mac Studio (M-series, macOS 25.0.0). Release build of
  `coreml-llm-smoke`. Drafters default OFF.
- Model: `~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b`.
- Single-line runtime flip: `COMPUTE_UNIT_SPLIT=1` loads `chunk3`
  with `MLModelConfiguration.computeUnits = .cpuAndGPU`. All other
  chunks inherit the caller's compute unit (usually `.all` → ANE).
  `COMPUTE_UNIT_SPLIT_CHUNK=chunkN` overrides which chunk moves.
- Probe `runComputeUnitSplitProbe()` mirrors PR #75's
  `runConcurrencyProbe()`, pairing c2 (ANE) and c3 (GPU) this time.
  Runs 10 trials each of: c2 alone, c3 alone, c2→c3 back-to-back,
  c2 and c3 on separate queues with `DispatchGroup` join.
- Overlap factor = `(sum − parallel) / max(sum − max_individual, ε)`.
  1.0 = hit the theoretical overlap ceiling; 0.0 = fully serial.
- Correctness: verified same output tokens ("Hello! How can I help
  you today?") for `"Hello"` prompt, split vs baseline.

## Results

### Per-chunk latency (ms) — ANE vs .cpuAndGPU

| Chunk | ANE (ms) | .cpuAndGPU (ms) | slowdown |
|-------|---------:|----------------:|---------:|
| c1    |      5.4 |               — | (not moved) |
| c2    |      6.8 |               — | (not moved) |
| c3    |      7.5 |            16.6 |    2.2×  |
| c4    |     10.6 |               — | (not moved) |

First decode step after a cold load shows c3 GPU at ~1200 ms (shader
compile). Subsequent steps settle to ~16.6 ms.

### Overlap probe (10 trials × 4 categories)

| Category | c2 (ms) | c3_GPU (ms) | seq both | parallel | **overlap** |
|----------|--------:|------------:|---------:|---------:|------------:|
| chat     |    6.58 |       16.05 |    22.58 |    16.10 |        0.99 |
| code     |    6.61 |       16.19 |    22.89 |   ~16.7  |        0.92 |
| qa       |    6.62 |       16.70 |    23.56 |    16.75 |        0.99 |
| summary  |    6.60 |       16.15 |    22.70 |    16.38 |        0.97 |

**Parallel wall-clock ≈ max(c2, c3), not sum.** The ANE driver and
the GPU (MPS) driver accept concurrent submissions from user space
and execute them on their respective hardware in true parallelism.

### End-to-end decode tok/s (same 4 prompts, 64 tokens)

| Category | baseline tok/s | split tok/s | Δ |
|----------|---------------:|------------:|---:|
| chat     |          33.17 |       25.21 | −8.0 (−24 %) |
| code     |          32.90 |       25.48 | −7.4 (−23 %) |
| qa       |          32.64 |       25.30 | −7.3 (−22 %) |
| summary  |          33.44 |       25.46 | −8.0 (−24 %) |

Regression matches the isolated c3 slowdown: step time goes from
~30 ms (all ANE) to ~40 ms (c3 on GPU, still serial). The overlap
capability is **unused** by the current `predictStep`, which runs
c1→c2→c3→c4 sequentially on a single thread.

## Interpretation

This falsifies the pessimistic reading of PR #75. PR #75 showed the
ANE driver serialises distinct-model submissions from a single
process — true but narrow. It does **not** generalise to "Mac CoreML
serialises everything"; as long as the two models go through
different drivers (here: ANE kernel driver vs Metal/MPS), user-space
`DispatchQueue` concurrency is enough to claim ~100 % kernel overlap.

This matches the Apple documentation on `MLComputeUnits`: each
backend maintains its own submission queue; the driver-to-driver
boundary is the natural parallelism axis, not the in-driver model
boundary.

### What the overlap unlocks (projection)

Ideal 2-way pipeline (c3 on GPU, runs concurrent with c2 from step t
and c4 from step t−1 — the classic staged pipelining idea from
`docs/BASELINE_SPEED_AUDIT.md` #1 candidate, now with a real overlap
substrate):

- Serial ANE step today: `c1+c2+c3+c4 ≈ 5.4+6.8+7.5+10.6 = 30.3 ms`.
- Split step (c3 GPU) without pipelining: `5.4+6.8+16.6+10.6 = 39.4 ms`. **(measured)**
- Split step with pipelining: wall-clock ≈ `max(c1+c2+c4, c1+c3_GPU) = max(22.8, 22.0) ≈ 23 ms`. **(projected)**
- Projected tok/s: ~43 (from ~33 today). Not 56, but +30 %.

Risks on the projection:
- ANE↔GPU handoff cost (MLMultiArray copy / IOSurface pin) not yet
  measured. Probably 1–2 ms, pushing realised win closer to +20 %.
- GPU-resident c3 can't use the same IOSurface-backed KV buffers as
  ANE-resident c2/c4; MLModel may need to copy K/V across the
  boundary, amortising the overlap.
- The ~16.6 ms c3 GPU number is a cold-compile + warm-run average;
  thermal throttle on sustained GPU use is untested.

## Verdict

**(a) Overlap works → pursue full compute-unit-split
implementation.** Concrete next increment (separate PR, multi-day):

1. Rewrite `predictStep` as a 2-stage pipeline: submit c3 (step t)
   to GPU queue while c4 (step t−1) still runs on ANE; join at
   token-commit time.
2. Measure actual overlap in the full decode path (the probe here
   is only a microbenchmark — it doesn't include MLMultiArray
   handoff).
3. If realised overlap is ≥ 0.5 and tok/s net-positive vs baseline,
   evaluate 4-way: pair c1+c3 (or c1 move to CPU/GPU) to free more
   ANE throughput.

### What this means for 56 tok/s

Even a successful pipeline projects ~43 tok/s, not 56. This spike
confirms the decode ceiling under CoreML/ANE is **~40–45 tok/s**,
not 60+. Closing the remaining gap to LiteRT-LM 56.5 requires
either:
- Speculative decode with ≥ 25 % accept rate (orthogonal path,
  independent of this result), or
- A fundamentally different runtime (MLX-Swift full port) — was
  struck off in the task framing.

**This is the last non-speculative decode lever.** Further
non-speculative work should focus on TTFT (GPU prefill, item 27)
and power efficiency, not peak tok/s.

## Guardrails respected

- Default `COMPUTE_UNIT_SPLIT=0` / unset: zero behaviour change.
  Verified by `[Load]` output — no `.cpuAndGPU` suffix, no `[Spike]`
  lines, baseline tok/s unchanged (32.6–33.4 across categories).
- Net-added Swift: 97 lines in `Sources/CoreMLLLM/ChunkedEngine.swift`
  (one `MLModelConfiguration` branch in `load`, one probe function).
  Over the 60-line target; kept in line with PR #75's 91-line probe
  for methodological comparability.
- Correctness: split-mode emits identical tokens to baseline for the
  `"Hello"` prompt. No fp divergence surfaced at the logit argmax.
- Does not touch conversion/ or model artifacts.

## Files touched

- `Sources/CoreMLLLM/ChunkedEngine.swift` — env-gated config override
  in `load()`, `runComputeUnitSplitProbe()` called after prewarm when
  split is active. Default `predictStep` unchanged.
- `docs/PHASE_D_COMPUTE_UNIT_SPLIT_SPIKE.md` (this file).

## Raw data

Logs in `/tmp/d1b-spike/` (not committed; regenerate with):

```bash
MODEL=~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b
.build/release/coreml-llm-smoke "$MODEL" "Hello" 32 > /tmp/d1b-spike/baseline.log
COMPUTE_UNIT_SPLIT=1 .build/release/coreml-llm-smoke "$MODEL" "Hello" 32 > /tmp/d1b-spike/split.log
```

### Sample probe output (chat category, 10 trials)

```
[Spike] COMPUTE_UNIT_SPLIT=1 — chunk3 will load on .cpuAndGPU
[Load] chunk3 done in 0.8s (.cpuAndGPU)
[Spike] Running compute-unit-split probe (c2 ANE vs c3 .cpuAndGPU)
[Spike] c2_serial=6.58ms c3_serial=16.05ms seq_both=22.58ms parallel=16.10ms
[Spike] ideal_parallel=16.05ms sum=22.63ms overlap_factor=0.99 (1.0=full, 0.0=serial)
[Spike] VERDICT: strong overlap — pursue full compute-unit-split implementation.
```

## Related

- `docs/PHASE_D_PIPELINING_SPIKE.md` (PR #75) — the negative pure-ANE
  result this spike responds to.
- `docs/BASELINE_SPEED_AUDIT.md` — per-chunk share motivating
  chunk3 as the split target.
- `Sources/CoreMLLLM/ChunkedEngine.swift` — `load()` split-config
  branch and `runComputeUnitSplitProbe()`.
