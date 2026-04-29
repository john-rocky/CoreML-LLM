# CPU Bottleneck Investigation — Hypothesis Refuted (2026-04-18)

## TL;DR

User reported the device gets noticeably hotter on this branch (ANE
path) than on a public LiteRT-LM iOS app (Metal GPU). Hypothesis: heat
came from CPU orchestration saturating a P-core.

**Hypothesis refuted by E4B device measurement.** CPU is essentially
idle (4% of step time). ANE itself is the heat source — it's busy 97%
of step time and physically pinned at high DVFS the whole time.

What this branch lands:

- **Measurement** (always on) — `[ANE/CPU]` log line that produced the
  finding.
- **`LLM_DECODE_QOS` env var** (opt-in) — kept; tiny CPU draw still
  benefits from E-core scheduling.
- **`LLM_FAST_PREDICTION` (default ON)** — `MLOptimizationHints.specializationStrategy = .fastPrediction`
  reduces per-prediction wall time on iOS 18+ without changing tok/s
  semantics. Direct heat reduction at fixed throughput.
- **Powermetrics + Mac smoke scripts** — kept for future investigations.

What was tried and **removed**:

- `LLM_DOUBLE_BUFFER_KV` — failed: IOSurface-backed `MLMultiArray`
  becomes locked once used as model input and **cannot be reused as
  output backing** in subsequent inferences (Apple warning:
  *"The underlying pixel buffer has been locked. Use a newly created
  pixel buffer..."*). The fallback path was supposed to be transparent
  but in practice CoreML allocated fresh IOSurfaces per step → 16×
  slowdown observed (1129 ms/step vs 71 ms baseline). Removed as a
  structural dead-end. See "Why double-buffer can't work" below.

---

## The E4B measurement that refuted the hypothesis

Steady state (post-warmup), iPhone 17 Pro, Gemma 4 E4B, no env vars
except `LLM_PROFILE_EVERY_STEP=1`:

```
[Profile] emb=0.8ms mask=0.2ms | c1=21.7 c2=21.6 c3=11.3 c4=16.0 (sum=70.6ms)
                                                          → 14.0 tok/s
[ANE/CPU] ANE_wait=68.7ms copyBack=1.9ms cpu_active=3.0ms (4% CPU)
```

| Component   | ms    | % of step |
|-------------|-------|-----------|
| ANE wait    | 68.7  | **97%**   |
| copyBack    | 1.9   | 3%        |
| cpu_active  | 3.0   | **4%**    |

Implications:
- CPU is essentially doing nothing — async/double-buffer/QoS can't
  meaningfully reduce heat.
- copyBack is 1.9 ms — much smaller than the originally estimated
  10 ms. Even a perfect outputBackings implementation caps gain at 2.7%.
- The 1 W ANE the docs cite is correct; what makes it feel hot is
  *sustained dispatch keeping ANE pinned at high DVFS for 14 seconds
  per response*, not a CPU-side heater.

---

## Why ANE *feels* hotter than GPU at lower watts (physics)

1. **Heat flux density.** ANE die ≈ 10 mm². 1 W / 10 mm² = 10 W/cm².
   GPU 4 W / 30 mm² ≈ 13 W/cm² but spread across more silicon and
   nearer the iPhone 17 Pro vapor chamber centroid. Small dies with
   no spreader produce localized hotspots that show up as skin temp
   even at low total wattage.
   ([SemiEngineering — Getting rid of heat in chips](https://semiengineering.com/getting-rid-of-heat-in-chips/))

2. **Vapor chamber geometry.** iPhone 17 Pro VC sits over the A19
   Pro CPU/GPU cluster. ANE is off to one side — its waste heat takes
   a longer thermal path to the chassis spreader.
   ([iFixit / 9to5Mac teardown](https://9to5mac.com/2025/09/23/iphone-17-pro-teardown-reveals-vapor-chamber-internals-scratchgate-details-more/))

3. **Time-integral.** Generating 200 tokens at 14 tok/s = 14.3 s
   sustained ANE busy. Same response on LiteRT GPU at 56 tok/s =
   3.6 s active + idle. Skin temperature is a low-pass filter of die
   power; sustained midpoint feels warmer than peak-then-idle even
   when the integral is similar.

4. **DRAM is the real heater.** LLM decode is bandwidth-bound. At
   14 tok/s × ~3 GB resident model ≈ 42 GB/s sustained LPDDR5X reads.
   The DRAM controllers and memory package heat irrespective of which
   compute engine drives them.
   ([SemiEngineering — LPDDR5X](https://semiengineering.com/lpddr5x-high-bandwidth-power-efficient-performance-for-mobile-beyond/))

5. **ANE DVFS.** Per Maderix M4 ANE reverse-engineering, ANE has
   *independent DVFS and 0 mW hard power-gating* when idle, but stays
   pinned at a high DVFS state under continuous dispatch. GPU DVFS is
   more aggressive at dropping clocks between Metal command buffers.
   ([maderix.substack.com — Inside M4 ANE Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine))

---

## Why the double-buffer KV optimization can't work

The intent was: allocate sibling buffers, pass via
`MLPredictionOptions.outputBackings`, swap (in, out) roles after each
prediction so the model writes KV into the future-input buffer,
eliminating the 1.9 ms/step `copyBack` memcpy.

What actually happens on iOS:

1. Step 1 — chunk1 input = buffer A (IOSurface), backing = buffer B.
   Backings honored, model writes into B.
2. Swap — kSliding1 = B, kSliding1Out = A.
3. Step 2 — chunk1 input = B, backing = A.
   **CoreML rejects A** as a backing because A's `CVPixelBuffer` was
   locked when CoreML used it as input in step 1. Apple's warning is
   explicit:

   > "The underlying pixel buffer (0x...) used in the output backing
   > MLMultiArray object for feature K_sliding_out has been locked.
   > The output backing cannot use such an object. ... Use a newly
   > created pixel buffer and MLMultiArray to avoid this error."

4. CoreML falls back to allocating a fresh IOSurface per step. The
   alloc path is dramatically slower than the warm-cached one — we
   measured 1129 ms/step vs 71 ms baseline (16× slowdown).

This is structural: any buffer used as model input enters a locked
state that excludes it from being a future output backing. Triple-buffer
would face the same issue eventually. The only way to do double-buffer
on iOS is to **not** use IOSurface for the output backings (plain
MLMultiArrays), but those lose the zero-copy benefit on the input
side. Net win with current ANE driver is ≤ copyBack itself (1.9 ms),
not worth the complexity.

**This is a real Apple-side constraint, not a bug in our implementation.**
Filed in the doc so future investigators don't re-attempt the same path.

---

## Mitigation menu (post-investigation)

After confirming ANE itself is the heat source, the available levers
are limited. Ranked by ROI for "reduce sustained ANE heat without
changing model architecture":

| Lever | Status | Rationale |
|-------|--------|-----------|
| `MLOptimizationHints.specializationStrategy = .fastPrediction` | **shipped (default ON)** | Shorter ANE busy per dispatch; same tok/s, less heat per token |
| Inter-dispatch `Task.sleep(Xms)` keyed off `thermalState` | rejected by user 2026-04-18 | Effective (ANE drops to 0 mW between chunks per Maderix) but trades throughput for heat |
| Chunk3+4 merge (4→3 dispatches) | conversion-side; out of this PR | Tracked in `LITERT_RUNTIME_ANALYSIS.md` Tier S2 |
| INT8 KV cache (50% bandwidth) | conversion-side; out of this PR | Tracked in `LITERT_RUNTIME_ANALYSIS.md` Tier S1; reduces DRAM heat |
| Apple FM-style architectural KV-share split | already in Gemma 4 | L0–14 own KV, L15–34 shared — more aggressive than Apple's 5:3 |
| W2 QAT weights | rejected | Post-training W2/W3 produces gibberish; QAT is days of GPU |
| Async ANE dispatch | rejected | Increases concurrency = more sustained heat (per WWDC23 / our analysis) |
| `_ANEClient` private `qos:21` | rejected | App Store reject risk |

---

## How to validate on iPhone

### Step 1 — Baseline
No env vars. Run the iOS app, type a prompt, generate 100 tokens.
Read the **last** `[ANE/CPU]` log line (most stable post-warmup).

Expected on E4B: `cpu_active ≈ 3 ms (4% CPU)`, `ANE_wait ≈ 70 ms`,
14 tok/s. If `cpu_active > 10 ms`, the original CPU hypothesis may
still hold for your workload — file the numbers.

### Step 2 — Tethered powermetrics

```sh
caffeinate -dimsu &
sudo ./scripts/powermetrics_bench.sh 60 1000
```

Start the iPhone generation when the script says "Starting…". CSV at
`/tmp/powermetrics_<unix>.csv`. The mean `combined_w` is the actual
heat-relevant number.

### Step 3 — Toggle `LLM_FAST_PREDICTION`

Default is ON in this PR. To compare, set `LLM_FAST_PREDICTION=0` in
the scheme. Expect:
- Same tok/s (within noise)
- First-load specialization a bit longer (~1-3 s extra)
- Slightly lower `ANE_wait` per chunk → marginally cooler per token

### Step 4 — `LLM_DECODE_QOS=utility` (optional)

Forces the decode loop onto efficiency cores. With cpu_active = 3 ms,
expected impact is small (saves ~1 ms of CPU per token at lower
watt/cycle). Try if combined_w needs further trimming.

---

## How to validate on Mac

### Single command — A/B in one shot

```sh
./scripts/test_double_buffer_locally.sh /path/to/model_dir 64
```

Note: this script was originally written to test the now-removed
double-buffer flag. The script still works for comparing
baseline vs `LLM_FAST_PREDICTION=0` — useful for sanity-check before
shipping the on-device build.

### Manual

```sh
swift build -c release

# baseline (fast prediction default on)
./.build/release/coreml-llm-smoke /path/to/model_dir "Hi" 64

# disable fast prediction
LLM_FAST_PREDICTION=0 LLM_PROFILE_EVERY_STEP=1 \
  ./.build/release/coreml-llm-smoke /path/to/model_dir "Hi" 64
```

Mac absolute numbers don't transfer to iPhone — the M-series ANE is
much larger. Use deltas only.

---

## References

- maderix — [Inside the M4 Apple Neural Engine, Part 1](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine) and [Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) (DVFS, power gating, qos:21)
- Orion — [Characterizing and Programming Apple's Neural Engine, arXiv 2603.06728](https://arxiv.org/abs/2603.06728) (dispatch overhead, INT8/FP16 throughput equivalence)
- Apple — [Foundation Language Models 2025 Tech Report, arXiv 2507.13575](https://arxiv.org/abs/2507.13575) (KV-share, W2 QAT, ReDrafter)
- Google — [ML Drift: scaling on-device GPU inference, arXiv 2505.00232](https://arxiv.org/abs/2505.00232) (LiteRT-LM Metal kernels)
- Apple — [MLOptimizationHints / specializationStrategy](https://developer.apple.com/documentation/coreml/mloptimizationhints-swift.struct/specializationstrategy-swift.property)
- Apple — [ProcessInfo.thermalState](https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.property)
- LiteRT-LM source — [google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) (no thermal management code; "fast enough that workload is short" pattern)
