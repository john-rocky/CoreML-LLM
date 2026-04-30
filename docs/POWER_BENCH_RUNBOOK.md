# Power Benchmark Runbook (shareable)

This is a self-contained guide for running the 15-minute sustained
power benchmark on **any supported iPhone**. It's the practical
counterpart to `POWER_BENCHMARK_PLAN.md` (methodology). Hand this to
whoever is running the test.

**Supported**: iPhone with A17 Pro or newer (iPhone 15 Pro, 15 Pro Max,
16, 16 Plus, 16 Pro, 16 Pro Max, 17, 17 Pro, 17 Pro Max). Older chips
lack the ANE headroom to hit reasonable tok/s and will skew numbers.

**Time budget**: ~35 min end-to-end (5 min cool-down + 15 min bench +
~15 min setup + reporting).

---

## 1. Device-specific setup (one-time)

### 1a. Set the battery capacity for your device

`mJ/token` is derived from battery drain × battery capacity. The code
defaults to iPhone 17 Pro (14.03 Wh). For other devices, edit
`Examples/CoreMLLLMChat/CoreMLLLMChat/LLMRunner.swift`, find:

```swift
var batteryCapacityWh: Double = 14.03
```

and replace with your device's nominal capacity:

| Device | Battery (Wh) |
|---|---:|
| iPhone 15 Pro | 13.35 |
| iPhone 15 Pro Max | 17.11 |
| iPhone 16 | 13.63 |
| iPhone 16 Plus | 17.16 |
| iPhone 16 Pro | 13.88 |
| iPhone 16 Pro Max | 17.15 |
| iPhone 17 | 14.34 |
| iPhone 17 Pro | **14.03** (default) |
| iPhone 17 Pro Max | 17.20 |

Source: Apple "environmental report" PDFs per device. If unsure, look
it up — a wrong value scales `mJ/token` linearly, so use the right one.

### 1b. Build in Release

1. Open `Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj` in Xcode.
2. Product → Scheme → **Edit Scheme** → **Run** tab → **Build
   Configuration: Release**. Close the dialog.
3. Select your physical iPhone as the run destination.
4. Set a Development Team (Signing & Capabilities → pick your team).
5. ⌘R to build & install. Once launched, Xcode can be disconnected.

Debug builds are ~10–20 % slower and will under-report both tok/s and
energy efficiency. **Release is not optional.**

---

## 2. Pre-run checklist (every run)

Do this immediately before each benchmark run. Skipping any of these
biases the result.

| Check | Setting | Why |
|---|---|---|
| Battery ≥ 60 % | — | 15-min run drains ~3–8 %; under 60 % risks Low Power Mode auto-enabling |
| **Unplug charger** | Cable out | Charging masks drain measurement |
| Low Power Mode | **OFF** (Settings → Battery) | LPM throttles CPU/ANE and invalidates the number |
| Airplane Mode | **ON** | Radio traffic adds uncontrolled energy |
| Wi-Fi / Bluetooth | OFF (Control Center) | Same as above; verify even with airplane mode on |
| Auto-Brightness | **OFF** (Settings → Accessibility → Display & Text Size) | Brightness changes shift power draw mid-run |
| Brightness slider | **50 % exactly** | Reproducibility |
| Background apps | All killed (swipe up from each) | Other processes burn battery |
| Do Not Disturb | ON | Notifications wake the display |
| Device temperature | Cool to the touch | Start from `thermal = nominal` |
| **5-min rest** | Close app, set phone down, wait 5 min | Lets residual thermal clear |

If any check fails and you run anyway, **note it in the report** —
the number is still useful, it's just not clean.

---

## 3. Run steps

1. Unlock the phone, open **CoreMLLLMChat**.
2. If no model is loaded: tap **Get Model** → pick Gemma 4 E2B → wait
   for download + load (~2.7 GB on first run, ~15–30 s compile on
   first launch after download, <1 s on subsequent).
3. Tap **ANE?** once. Confirm the result shows `TOTAL … (99 %)` or
   higher. If it's lower, something is wrong — stop and investigate.
4. Put the phone **face-up on a flat surface** (table, not fabric —
   fabric insulates and skews thermal).
5. **Don't touch it. Don't hold it. Don't charge it.** The body heat
   from your hand changes the thermal profile.
6. Tap **Bench → 15 min (power)**.
7. A banner says `[Benchmark] Starting 15-minute sustained
   generation…` — screen will stay on automatically
   (`isIdleTimerDisabled`).
8. **Wait 15 minutes**, hands off.
9. When done, a `[Benchmark RESULT]` block appears in the chat.

If it aborts early with `Aborted: YES (thermal .serious)`, **keep
that result** — "device can only sustain N minutes before throttling"
is itself the thing we want to publish.

---

## 4. What to send back

Paste the full `[Benchmark RESULT]` block into a reply. It looks like:

```
[Benchmark RESULT]
Duration      : 900s (15.0 min)
Rounds        : 4
Total tokens  : 27431
Avg tok/s     : 30.48
Battery       : 82% → 76%  (Δ 6.00%)
Drain rate    : 0.400%/min (~24.0%/hr)
Tokens/%SoC   : 4572
Energy/token  : 92.0 mJ/tok
Thermal       : nominal → fair
Time→fair     : 420s
Time→serious  : never
CSV           : bench-1744800000.csv
Thermal trajectory:
    0s → nominal  bat=82%
   30s → nominal  bat=82%
   ...
Battery log:
    0s → 82%
  120s → 81%
  ...
```

Also attach the CSV if possible (see §5).

Plus, these side-channel facts:

- **Device model** (Settings → General → About → Model Name)
- **iOS version** (Settings → General → About → iOS Version)
- Whether any pre-run check was skipped, and which
- Ambient room temperature rough estimate (cold room vs warm room
  matters — ANE has ~5–10 °C thermal headroom before throttle)

---

## 5. Getting the CSV off the device

The CSV is saved to the app's `Documents/` folder as
`bench-<unix_timestamp>.csv`. Three ways to extract:

**Easiest — Files app + AirDrop (no cable):**

1. Open **Files** app on the iPhone → Browse → **On My iPhone** →
   **CoreMLLLMChat**.
2. Long-press `bench-<ts>.csv` → **Share** → **AirDrop** → send to
   Mac.

**Files + iCloud Drive (if you have iCloud):**
Copy the file from On My iPhone → CoreMLLLMChat into iCloud Drive,
retrieve on Mac.

**Xcode (cable required, dev machine):**
Xcode → Window → **Devices and Simulators** → select iPhone → select
`CoreMLLLMChat` under Installed Apps → gear icon → **Download
Container** → right-click the `.xcappdata` → **Show Package Contents**
→ `AppData/Documents/`.

---

## 6. Running more than once

Recommended: run the full 15-min bench **3 times** and report median.

Between runs:
- Let the phone **cool for 10+ min** (not 5 — second run starts warmer
  than first).
- Recheck §2 (Low Power Mode sometimes auto-enables when battery
  drops).

If you can only do one run, that's fine — just note "n=1" in the
report.

---

## 7. Sanity ranges (so you know if something's wrong)

On an A17 Pro or newer iPhone, expect roughly:

| Number | Expected range | Red flag if… |
|---|---|---|
| Avg tok/s (15 min) | 25–35 tok/s | < 15: Debug build, or thermal throttle, or wrong compute unit |
| Drain rate | 0.3–0.6 %/min | > 1 %/min: another app is active, or radios are on |
| mJ/token | 50–150 | > 300: drain didn't register (too-short run), or wrong capacity Wh |
| Time → fair | 180–900 s (or `never`) | < 60 s: device started warm, or ambient too hot |
| Time → serious | `never` preferred | Happens: reportable data, not a failure |

If numbers are **way** outside these, rerun after confirming §2, and
double-check step 1b (Release build).

---

## 8. Troubleshooting

**`Energy/token : n/a (gauge noise, need ≥10 min run)`**
→ The battery gauge didn't change. Means the run was too short, or
the phone was charging, or drain was < 1 %. Rerun at 15 min with
charger unplugged.

**`Aborted: YES (thermal .serious)`**
→ Device threw thermal state before the 15 min mark. That IS the
result. Note `Time→serious` — that's the sustained-duration number.

**`Avg tok/s` much lower than expected**
→ Check that `ANE?` showed ≥ 99 %. If it's lower, the model loaded on
GPU/CPU and numbers are meaningless. Force-quit and relaunch.

**No CSV file in Files app**
→ Verify `Info.plist` has `UIFileSharingEnabled = YES` and
`LSSupportsOpeningDocumentsInPlace = YES` (already set in this repo,
but check if modified). The folder won't appear until the first
file is written, so run at least one bench first.

**Battery delta is 0 or negative**
→ Charger was connected, or Low Power Mode kicked in mid-run. Both
ruin the measurement. Start over.
