# Benchmarking — How the Numbers in README Were Measured

Performance numbers in this repo should be **reproducible**. This doc specifies the exact methodology so anyone can verify and compare against their own runs.

## Device & OS

| Item | Setting |
|---|---|
| Device | iPhone 17 Pro (A19 Pro, 8 GB RAM) |
| iOS | 18.0+ |
| Xcode | 16+ (to build the sample app) |
| App build config | Release (not Debug) |
| Thermal state at start | `nominal` — if screen has been kept warm, let the device sit 5 min before running |
| Low Power Mode | OFF |
| Battery | ≥ 50 %, not charging (charging changes thermal envelope) |

ANE placement and decode throughput both depend on the compute units actually chosen at load time. The sample app forces `.cpuAndNeuralEngine` (see `LLMRunner.swift`). On older A-chips, the numbers will differ.

## Decode speed (`tok/s`)

**Measured inside the sample app**, not from an external stopwatch.

- `CoreMLLLM` streams tokens one at a time.
- `ChunkedEngine` records `tokensPerSecond` as an EMA-style running average over a window of decoded tokens (see `ChunkedEngine.swift` — the value surfaced on `CoreMLLLM.tokensPerSecond` after each stream completes).
- The reported number is over the **last generation**, not global-average since app launch.

### Prompt & sampling

- Sampling is **argmax in-graph** (`ane_ops.InModelArgmax`). No temperature, no top-k, no top-p. This is deterministic and isolates engine speed from sampler overhead.
- README numbers use the benchmark prompt in `LLMRunner.swift :: benchmarkPrompt` (a long-form article request about AI history). This reliably produces >256 continuation tokens, so the tok/s average is taken over a meaningful window, not warm-up.
- First-token latency (prefill) is excluded from decode tok/s — prefill is reported separately (see below).

### Context length

- Default `contextLength: 2048` → README "decode speed" row.
- `contextLength: 8192` loaded explicitly in the app → README "Decode @ 8K ctx" row.
- KV buffers at 8K make the 7 full-attention layers scale with the whole context, which is why 8K decode is ~½ the 2K speed even though the 28 sliding layers are unchanged.

### Sustained vs peak

- **Peak**: first 30 s of decode on a thermally-nominal device.
- **Sustained**: "Bench" button in the sample app runs for a user-set duration (default 120 s). It stops early on thermal `serious`/`critical`. The battery / `phys_footprint` curve is logged per minute. Reported as `avgTokPerSec` in `BenchmarkResult`.

We report **peak** in the README performance table because that matches how competing engines publish their numbers (MLX, LiteRT, MLC). For a sustained number, use the Bench screen.

## Prefill speed (`tok/s`)

- Measured as `N / t` where `N` is the prefill batch size the model was built with (default 512, see `ChunkedEngine.swift :: prefillN`) and `t` is the wall time of the 4-chunk prefill call (from `ChunkedEngine.swift :: prefill()`).
- README "Prefill" row: single batched call on a fresh KV cache with the sample prompt padded to `N`.
- Prefill tok/s is order-of-magnitude higher than decode tok/s because the 4 prefill chunks process all `N` tokens in parallel per chunk.

## ANE placement (`99.78 %`)

- Measured via `MLComputePlan` on-device — tap "ANE?" in the sample app (`LLMRunner.swift :: verifyANEPlacement`).
- The reported percentage is `(ops dispatched to ANE) / (total ops)` across **all four decode chunks**, summed.
- Current snapshot: 7,294 / 7,310 ops on ANE for Gemma 4 E2B at ctx=2048 with INT4 palettized weights.
- The 16 non-ANE ops are embedding-lookup scatter/gather patterns at chunk boundaries. They are deliberately kept on CPU because forcing them to ANE requires converting to Conv2d-style scatter that adds more latency than the CPU hop.
- Vision encoder (`vision.mlpackage`) is **excluded** from this percentage — it runs on `.cpuAndGPU` by design. If you include it, end-to-end placement is lower.
- Prefill chunks are counted separately; their placement is similar but not identical. The 99.78 % figure is for **decode chunks**.

## Memory (`phys_footprint` ~1 GB)

- Measured via `task_vm_info()` — tap "Mem" in the sample app.
- Reported fields:
  - `phys_footprint` — iOS jetsam basis. **This is the number that matters** for "will my app get killed".
  - `resident_size` — classic RSS.
  - `compressed` — pages the OS has compressed (ANE weights often land here).
  - `os_proc_available_memory()` — remaining headroom before jetsam.
- Why not Xcode's memory gauge? Xcode consistently under-reports INT4-palettized CoreML weights by ~700 MB vs `task_vm_info` on iOS 18. The gauge is a debugger estimate, not the jetsam metric. README v0.4 and earlier quoted the gauge number; v0.5+ uses `phys_footprint`.
- Snapshot on iPhone 17 Pro, Gemma 4 E2B, ctx=2048, all 8 chunks loaded:
  - After load, idle: ~873 MB
  - Mid-decode: ~981 MB
  - Headroom (`os_proc_available`): ~5 GB

## Energy (`mJ/tok`, `%/hour`, thermal trajectory)

The sample app's **Bench** menu now exposes three presets aimed at
power reporting:

- **2 min (speed)** — quick peak tok/s check
- **15 min (power)** — minimum duration for a defensible `mJ/tok`
  number given the iOS battery gauge's 1 % resolution
- **60 min** — long-haul thermal profile, useful for "will this
  throttle in a real session" questions

After each run the app writes a CSV to `Documents/bench-<unix_ts>.csv`
with the per-30s thermal trajectory, battery log, and a `# summary`
block. The CSV filename is printed in the in-app result and to the
console. Retrieve via Files app (the target already has
document-sharing entitlements).

`BenchmarkResult` exposes:

- `mJPerToken` — `drainedPercent × batteryCapacityWh × 36000 / totalTokens`.
  iPhone 17 Pro nominal capacity is 14.03 Wh; override
  `batteryCapacityWh` for other devices.
- `drainedPerHour` — extrapolated from the run duration.
- `timeToFair`, `timeToSerious` — first elapsed second at which
  `ProcessInfo.thermalState` transitioned.
- `thermalTrajectory` — array of `ThermalSample(t, state, batteryLevel)`
  at 30-second intervals.

For the methodology, metric tiers, and head-to-head protocol against
other engines, see [POWER_BENCHMARK_PLAN.md](POWER_BENCHMARK_PLAN.md).

## Energy (`J/tok`) — legacy derivation

The ~0.07 J/tok figure in `docs/RESEARCH.md` is **derived**, not directly measured:

- Observed battery drain in the Bench screen (`drainedPerMinute`) × nominal battery capacity in watt-hours → watts during the run.
- Tokens per second from the same run → tokens per minute.
- J/tok = (W × 60) / (tokens / minute).

This is order-of-magnitude correct but not precise — iOS does not expose per-core power rails. Treat it as "about 10× better than GPU-based LLM engines on the same device", not as a calibrated lab measurement.

## Reproducing a number

To reproduce any README performance row on your own iPhone 17 Pro:

1. Build `Examples/CoreMLLLMChat` in Release, set your team, install on the device.
2. Download Gemma 4 E2B via the app's **Get Model** button (pre-converted from HuggingFace).
3. Wait 5 min with the app closed so thermal state is `nominal`.
4. Open the app. Tap **Bench** (or type the benchmark prompt in Chat).
5. Record the `avgTokPerSec` reported by the Bench screen. Tap **ANE?** for placement and **Mem** for memory.

Numbers should match the README within ±5 % on the same hardware. Larger deltas usually mean thermal throttling (check the Bench `thermalEnd`) or a different context length.

## What we *don't* measure

- **Perplexity / quality**. We've spot-checked INT4 vs FP16 outputs and they match qualitatively, but there is no automated perplexity regression test. See `docs/EXPERIMENTS.md` — this is a known gap before promoting W8A8 out of prototype.
- **First-token latency (TTFT)** in the README table. Prefill tok/s is proxy; actual TTFT for a 512-token prompt is `512 / prefill_tok_s + per-chunk-launch-overhead`.
- **Cold-start load time**. Dominated by CoreML's first-run ANE compile (parallelized across chunks since v0.5), typically 15–30 s on first launch, <1 s on subsequent launches.

## When the numbers change

If you ship a model change (different chunking, different quantization, a new Gemma version) and the decode number drops by more than 5 %, that is a regression to investigate, not noise. Run the Bench screen twice on a cool device to confirm, then bisect against the last known-good build.
