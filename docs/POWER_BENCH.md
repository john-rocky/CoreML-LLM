# Power & Throughput Bench (`ComputeProfile` sweep)

This bench characterises the three user-facing compute profiles for the
on-device Gemma 4 E2B (INT4 chunked) model:

| Profile          | `MLComputeUnits`          | Intended use                |
| ---------------- | ------------------------- | --------------------------- |
| `.efficient`     | `.cpuAndNeuralEngine`     | Default. Lowest power.      |
| `.balanced`      | `.all`                    | Core ML auto-partitions.    |
| `.performance`   | `.cpuAndGPU`              | "Gachi" / full-send mode.   |

The harness lives in `Examples/CoreMLLLMChat/CoreMLLLMChat/PowerBench.swift`.
It loads a fresh `CoreMLLLM` instance per profile (so Core ML honours the
new compute units), runs N trials of fixed-length generation, and writes
`Documents/power_bench.csv`.

## How to run on device

1. Build & install `CoreMLLLMChat` to the iPhone 17 Pro from Xcode 26.1+.
2. Load any chunked model (Gemma 4 E2B recommended).
3. Tap the `cpu` icon in the toolbar to pick a default profile.
4. Tap `Power` in the toolbar to start the sweep.
5. Wait ~5-10 minutes — keep the screen on, unplug from charger.
6. Result lines appear in chat; CSV is in the app's Documents folder
   (open via Files → On My iPhone → CoreMLLLMChat → `power_bench.csv`).

## CSV schema

```
profile,trial,tokens,seconds,tok_per_sec,thermal_start,thermal_end,
battery_start,battery_end,cpu_pct,joules_estimate
```

Followed by a per-profile summary block:

```
profile,mean_tok_per_sec,peak_tok_per_sec,mean_cpu_pct,thermal_exit,
joules_total_estimate
```

`joules_estimate` uses the nominal iPhone 17 Pro battery (13.97 Wh) and is
only meaningful for trials long enough to see a battery tick (~30 s+). For
authoritative power numbers use **tethered powermetrics** — see below.

## Tethered powermetrics (authoritative joules)

The on-device joule estimate is coarse. Apple's `powermetrics` exposes
per-domain power on macOS; when the iPhone is tethered via USB (or
internally, on a Mac mini doing the test), you can sample CPU / GPU / ANE
power at 1-second granularity:

```sh
sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -n 60 \
  > ~/Desktop/powermetrics_$(date +%s).log
```

Run this **on the Mac** in parallel with the on-device bench. The 60-sample
window covers ~1 minute — long enough to span 1-2 profile trials. Match
timestamps in the log against the trial start lines printed by the bench.

To keep the iPhone awake under USB:

```sh
caffeinate -dimsu &
```

If you want continuous power for the entire 5-10 min sweep, raise `-n` to
`600` (10 min @ 1 Hz).

### What to look for in `powermetrics`

- `ANE Power`: should peak ~0.5-1 W under `.efficient`, near zero under
  `.performance`.
- `GPU Power`: should peak ~3-5 W under `.performance`, near zero under
  `.efficient`.
- `Combined Power`: total platform draw. `.balanced` typically lands in
  the middle (~1.5-3 W).

## Predicted numbers (iPhone 17 Pro, A19 Pro, iOS 26)

These are predictions filed before on-device measurement. **User: please
fill in the "Actual" column after the first run.**

| Profile        | tok/s @ 2K (predicted) | Thermal after 5 min | Drain rate     | Actual |
| -------------- | ---------------------- | ------------------- | -------------- | ------ |
| `.efficient`   | 31 tok/s               | nominal / fair      | ~3-4 %/min     |        |
| `.balanced`    | 35-40 tok/s            | fair                | ~5-7 %/min     |        |
| `.performance` | 45-60 tok/s            | serious (likely)    | ~10-12 %/min   |        |

### Reasoning

- `.efficient` is today's measured baseline (31.4 tok/s @ 2K) — no change.
- `.balanced` lets Core ML partition; on A19 Pro this typically gives a
  modest 10-20% speedup as the GPU absorbs the residual matmul tail.
- `.performance` predicted upper bound assumes the GPU dispatch latency
  is ~half of ANE's ~2.3 ms round-trip. With 4 chunks × 4 dispatches per
  decode step → ~9 ms total today; ~4-5 ms on GPU could net 60-70 tok/s.
  The lower bound (45) covers GPU memory-bandwidth bottleneck on the
  INT4 palettised weights — INT4 lookup may not be fast on Metal.

### Will `.performance` mode alone beat LiteRT-LM 56.5 tok/s @ 2K?

**Honest answer: borderline. 50-50.**

Reasons for optimism:
- LiteRT-LM uses GPU delegate too, so the platform supports the speed.
- Per-dispatch overhead on the GPU is empirically lower than ANE on A-series.

Reasons for skepticism:
- The current 4 mlpackages were ANE-tuned (INT4 palettisation, specific
  KV layout). GPU may prefer fp16 weights and a different attention
  layout (sliced-Q is in `conversion/build_prefill_gpu.py` for prefill;
  decode would need an analogous variant).
- Without a GPU-specific decode mlpackage, `.performance` on the same
  ANE-tuned chunks may only reach 40-45 tok/s — below the 56.5 target.
- LiteRT-LM also uses GPU-side speculative decoding; we'd need to compose
  with MTP + drafter-union to match.

**To beat 56.5 tok/s on `.performance` alone you almost certainly need a
GPU-specialised decode mlpackage** (see "GPU mlpackage variant" below).
Without one, expect `.performance` to land 40-50 tok/s and rely on
`.balanced + speculative` to exceed 56.5.

## GPU mlpackage variant — decision

`conversion/build_prefill_gpu.py` already produces a GPU-targeted
**prefill** variant (with sliced-Q attention) that drops a
`compute_preference.json` sidecar so `ComputePreferenceLoader` can pick it
up automatically. There is no equivalent **decode** GPU variant yet.

Decision for this patch: **ship the user-facing toggle now, defer the
decode-GPU mlpackage**. Rationale:

1. The same ANE-tuned mlpackages compile and run under `.cpuAndGPU` —
   Core ML transparently re-targets ops. We get a real measurement out
   of the existing assets without spending a conversion cycle.
2. The bench harness will tell us whether the GPU is bandwidth-limited
   (in which case fp16 weights help) or compute-limited (in which case
   layout changes help). Without numbers, building a GPU-specific
   variant would be premature.
3. If `.performance` lands well below 56.5 tok/s, the next conversion
   cycle should produce `chunk{1..4}_gpu.mlpackage` as fp16 (un-palettised)
   with sliced-Q attention, mirroring `build_prefill_gpu.py`'s precedent.

The ANE-only quirks to watch for in the GPU run:
- INT4 lookup-tables (LUT) for palettised weights — may not vectorise on
  Metal; fp16 weights would be faster but ~3× larger.
- Per-chunk output reshape that inserts a 1×1×1×N axis for ANE alignment
  — harmless on GPU but wastes a bit of bandwidth.

## Apples-to-apples discipline

- Disable speculation (`mtpEnabled`, `crossVocabEnabled`, `drafterUnionEnabled`)
  during the bench — the harness already does this. Leaves only the
  compute-unit effect.
- Run on battery, screen at minimum brightness, airplane mode. Otherwise
  the joule estimate is contaminated by radio/display draw.
- Wait for `thermalState == .nominal` between runs — the harness does a
  30 s cool-down by default; lengthen if you see `.serious` carry over.

## Reference: programmatic invocation

```swift
let bench = PowerBench(llmFolder: modelFolder)
let summaries = try await bench.run { print($0) }
for s in summaries {
  print("\(s.profile.displayName): \(s.meanTokPerSec) tok/s")
}
```
