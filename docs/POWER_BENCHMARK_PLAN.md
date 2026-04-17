# Power & Energy Benchmark Plan

**Goal:** publish a reproducible, defensible set of power/energy numbers
for Gemma 4 E2B on iPhone 17 Pro that competitors (LiteRT-LM iOS,
llama.cpp Metal, MLX) cannot easily match, and that makes the ANE
placement advantage visible as a *user-facing* metric — not just a
compute-unit placement percentage.

Speed parity is the second-order goal (see
`docs/MOBILE_2K_COMPETITIVE_PLAN.md`). This doc covers the first-order
goal: prove that on a phone in a pocket, we draw less current and
stay cooler for the same work.

---

## Why this matters competitively

| Engine | Primary compute | Expected sustained behaviour |
|---|---|---|
| llama.cpp (Metal) | GPU | Thermal throttles within 3–5 min of continuous decode; iPhone chassis gets hot; drains battery fast |
| LiteRT-LM iOS | GPU + CPU hybrid | Better than pure-Metal but still GPU-heavy; 56 tok/s peak implies high wattage |
| MLX | GPU | Same class as llama.cpp Metal |
| **CoreMLLLM (ours)** | ANE (99.78 %) | ANE power envelope is ~1/3 to 1/5 of GPU for the same ops; no thermal impact on the rest of iOS |

Nobody publishes *mWh/token* or *sustained tok/s at thermal=nominal for
15 minutes* for on-device LLMs. Whoever publishes first owns that
narrative. We are the only engine that can publish these numbers
honestly.

---

## Metrics to publish

### Tier 1 — must ship (v0.6 README)

1. **Energy per token** — `mJ/tok` at ctx=2K and ctx=8K, decode only.
2. **Sustained tok/s (15 min)** — average over a 15-minute continuous
   decode run, device starting at `thermal=nominal`, not charging.
3. **Thermal trajectory** — `ProcessInfo.thermalState` at t=0, 1, 3, 5,
   10, 15 min. Reported as "time to first `fair`" and "time to first
   `serious`".
4. **Battery drain** — `%/hour` extrapolated from a 15-minute run,
   corrected for battery capacity (`UIDevice.batteryLevel` delta ×
   declared Wh).

### Tier 2 — if infra allows (v0.7)

5. **Per-subsystem power** — ANE vs GPU vs CPU wattage breakdown,
   sampled from `powermetrics` (macOS host, device tethered).
6. **mW during idle-after-decode** — cost of holding KV cache resident
   vs releasing.
7. **Chassis surface temperature** — IR thermometer, 3 points (back
   center, top, camera bump). Manual, one-shot.
8. **Energy per user turn** — prefill + decode, typical 512-token
   prompt → 128-token response.

### Tier 3 — research-grade (not blocking)

9. **Joules per correct answer** — hook MMLU subset into Bench, measure
   energy to produce each answer.
10. **Energy parity vs llama.cpp / LiteRT-LM** — head-to-head on the
    same physical device, same prompt, same output length.

---

## Measurement methodology

### On-device (no host required)

The sample app already has the scaffolding:

- `LLMRunner.BenchmarkResult.drainedPercent` — from `UIDevice.batteryLevel`
  delta across the run. Resolution is 1 % on iOS, so runs must be
  ≥ 10 minutes to get < 10 % error. **Raise default Bench duration from
  120 s to 900 s (15 min)** and expose it in `ChatView.swift`.
- `thermalStart` / `thermalEnd` — already captured. **Add
  thermal sampling every 30 s** into a `[ThermalSample]` array on
  `BenchmarkResult`. This gives the "time to fair/serious" number.
- `drainedPerMinute` × iPhone 17 Pro nominal capacity (14.03 Wh) →
  watts → mJ/tok.

The existing `Energy (J/tok)` section in `docs/BENCHMARKING.md`
acknowledges this is derived, not measured. That's fine — lab-grade
per-rail power is not available without tethering. Be explicit in the
README.

### Tethered (macOS host, `powermetrics`)

For Tier 2, use `sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 1000`
on a **Mac connected to the iPhone via USB-C** running the
CoreMLLLM Bench. Note: `powermetrics` on macOS reports the **Mac's**
subsystems, not the iPhone's. For the iPhone, use **Instruments →
Energy Log** instead — it gives CPU/GPU/networking energy estimates
per process but **does not break out ANE power**.

**Honest conclusion:** iOS does not expose per-rail ANE power to
third parties. Tier 2 #5 is a "best effort with disclosed limits"
metric, not a lab number. Publish the raw Instruments screenshots.

### Lab-grade (optional, nice-to-have)

- External USB-C power meter (e.g. ChargerLAB POWER-Z KM003C) between
  charger and phone, measure Wh during a 15-min decode with phone at
  exactly 50 % battery. Subtract idle baseline measured for 15 min
  immediately before with the app backgrounded.
- This is the only way to get an end-to-end "wall-plug" number. It
  includes screen, radios, and everything else, but with a clean
  baseline-subtraction it is defensible.

---

## Test matrix

Run each configuration **three times** on a cold device (5 min rest
between runs). Report median.

| Ctx | Duration | Sampling | KV reset | Purpose |
|---|---|---|---|---|
| 2048 | 15 min | argmax | no | Tier-1 headline: sustained 2K mJ/tok |
| 8192 | 15 min | argmax | no | Long-ctx sustained — differentiator vs llama.cpp (they OOM or crawl) |
| 2048 | 15 min | argmax | every 256 tok | Shows steady-state with realistic turns, not a single long generation |
| 2048 | 5 min | argmax | no | Peak number for README (matches competitors' reporting) |
| 2048, bench-prefill | N/A | prefill only | N/A | mJ/tok for prefill (usually 3–10× cheaper per token than decode) |

All runs: airplane mode ON, screen brightness at 50 % fixed (manual —
auto-brightness adds noise), Low Power Mode OFF, not charging, same
benchmark prompt as `LLMRunner.swift :: benchmarkPrompt`.

---

## Head-to-head protocol (Tier 3 #10)

To make an apples-to-apples claim against LiteRT-LM or llama.cpp:

1. Same iPhone 17 Pro, same iOS version, same battery level (50 % start).
2. Same prompt, same max-tokens cap (e.g. 256 decoded tokens).
3. Same starting thermal state (`nominal`, 5 min rest between runs).
4. Airplane mode ON. Screen brightness fixed.
5. Measure: wall-clock duration, battery drain %, ending thermal state.
6. Derive: J/tok = (drain% × 14.03 Wh × 3600) / (tokens × 100).
7. Repeat each engine 3×, report median + min/max.

**Risk:** LiteRT-LM iOS distribution may not be publicly installable.
If so, publish our number standalone and invite Google to respond —
that itself is a win narratively.

---

## Implementation plan (code changes)

Ordered by cost. Each step is standalone-shippable.

### Step 1 — extend `BenchmarkResult` (0.5 day)

`Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift:116`

Add:
```swift
struct ThermalSample { let t: TimeInterval; let state: ProcessInfo.ThermalState; let batteryLevel: Float }
var thermalTrajectory: [ThermalSample]
var mJPerToken: Double { /* drained% × 14030 J / tokens / 100 */ }
var timeToFair: TimeInterval?
var timeToSerious: TimeInterval?
```

Sample every 30 s from a `Task` running alongside decode.

### Step 2 — raise default duration, add presets (0.5 day)

`ChatView.swift:394` area — add a picker: `2 min` / `15 min` / `60 min`.
Default to 15 min for the "Power" tab, 2 min for the "Speed" tab.

### Step 3 — CSV export (0.5 day)

Add a "Share CSV" button on the Bench result sheet. Columns:
`t_seconds, tok_per_sec_window, battery_pct, thermal_state, phys_footprint_mb`.
Let users (and us) paste into spreadsheets.

### Step 4 — README rewrite (0.5 day)

Add a **"Power & Thermal"** table to README above the speed table.
Include mJ/tok at 2K and 8K, sustained tok/s, time-to-`fair`. Link to
this doc for methodology.

### Step 5 — head-to-head blog post / gist (1 day)

Run the protocol above against whichever competitor we can actually
install. Publish the CSVs. Do not editorialise — let numbers speak.

---

## What we will *not* claim

- **"X watts on the ANE"** — iOS does not give us this. We will not
  fabricate a per-rail number.
- **"Zero GPU usage"** — the vision encoder runs on GPU by design. Any
  multimodal turn has GPU energy in it. Text-only is clean.
- **"Better battery life than the OS baseline"** — untrue and not the
  claim. The claim is "less energy per token than competing LLM
  engines", which is narrower and defensible.
- **Lab-calibrated J/tok** — clearly label the headline number as
  *derived from battery-gauge delta*, with error bars.

---

## Success criteria

v0.6 README ships with:

- mJ/tok at 2K decode, ±15 % error bar, methodology linked.
- Sustained 15-min tok/s at 2K, with thermal trajectory.
- At least one head-to-head comparison (even if only against a
  hypothetical "GPU-based engine on same device" using published
  llama.cpp Metal numbers — clearly marked as indirect).

Stretch:

- Instruments Energy Log screenshot showing our process's energy
  score vs a Metal-based LLM on the same device.
- External USB-C power meter measurement with baseline subtraction.

---

## Timeline

| Week | Deliverable |
|---|---|
| 1 | Steps 1–3 (code changes), first internal 15-min runs logged |
| 2 | Step 4 (README), publish Tier 1 metrics |
| 3 | Tier 2 attempt (Instruments tethered), publish whatever we get |
| 4 | Head-to-head against one competitor, blog post |

Total: ~4 weeks for 1 person, can run in parallel with speed work.
