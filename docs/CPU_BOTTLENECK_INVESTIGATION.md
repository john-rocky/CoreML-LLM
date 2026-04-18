# CPU Bottleneck Investigation (2026-04-18)

User-reported symptom: device gets noticeably hotter on this branch (ANE
path, ~31 tok/s) than on a public LiteRT-LM iOS app (Metal GPU). Per
docs the ANE path should draw ~0.5–1 W vs LiteRT's 3–5 W on GPU, so the
heat asymmetry contradicts the documented power profile.

Hypothesis: ANE itself is cool, but the CPU orchestration around it
(KV `copyBack` memcpy, dispatch prep, embed dequant, mask building)
saturates a P-core. At 31 tok/s × 4 chunks/tok = 124 dispatches/sec,
even small per-dispatch CPU work compounds into a sustained hot loop.

This branch lands the measurement to confirm or refute the hypothesis,
plus three opt-in mitigations.

---

## What landed

### 1. CPU vs ANE wait split (always on)

`ChunkedEngine.predictStep` now logs an extra line every 10 decode
steps:

```
[Profile] emb=… mask=… | c1=… c2=… c3=… c4=… (sum=…) | predict=… total=… (… tok/s)
[ANE/CPU] ANE_wait=Xms copyBack=Yms cpu_active=Zms (W% CPU)
```

- `ANE_wait`  — sum of time inside the four `chunk.prediction(from:)` calls.
- `copyBack`  — sum of `copyBack` memcpy time on the persistent KV buffers.
- `cpu_active` — `total - ANE_wait` (everything the CPU did between dispatches:
  dictionary build, embed dequant, mask build, KV copy, output extraction).
- `W% CPU`    — `cpu_active / total * 100`.

**Interpretation guide:**

| W% CPU | Likely heat source                    | What to try next                    |
|--------|---------------------------------------|-------------------------------------|
| < 25%  | ANE itself is sustained-on            | Reduce dispatch count (chunk merge) |
| 25–50% | Mixed                                  | Try LLM_DOUBLE_BUFFER_KV first      |
| > 50%  | CPU orchestration dominates           | LLM_DOUBLE_BUFFER_KV + LLM_DECODE_QOS |

If `copyBack` alone is ≥ 30% of the step (i.e., ≥ 15 ms at 51 ms/step),
LLM_DOUBLE_BUFFER_KV is the right first lever.

### 2. Tethered powermetrics helper (`scripts/powermetrics_bench.sh`)

Run on the Mac while the iPhone is tethered:

```sh
sudo ./scripts/powermetrics_bench.sh 60 1000   # 60s @ 1Hz
```

Writes `/tmp/powermetrics_<unix>.csv` with columns:
`ts_sec, cpu_w, gpu_w, ane_w, combined_w` plus a mean-power summary.

Reading the result:
- `ane_w` near 0.5–1 W is expected for the ANE path.
- If `cpu_w` is 2–3 W during sustained generation, **CPU is the heat source**.
- If `combined_w` exceeds 3 W, the user's "hotter than LiteRT" report is
  reproduced — and the split shows whether CPU or ANE owns it.

### 3. Decode-loop QoS (env-gated, opt-in)

`LLM_DECODE_QOS=utility` (or `background`) sets the priority of the
`Task` that owns the generation loop in `streamFromPrompt`. The Swift
runtime maps lower priorities to efficiency cores when possible.

```sh
LLM_DECODE_QOS=utility ./CoreMLLLMChat
```

Trade-off: E-cores are ~25% as fast as P-cores per cycle but draw
~25% the power. The ANE wait is unchanged; only the CPU work between
dispatches is affected. Expect a small tok/s loss (≤ 5%) for a
meaningful sustained-temperature win.

Defaults to inherited (`.userInitiated`) — no behavior change unless
the env var is set.

### 4. outputBackings + double-buffer KV (env-gated, opt-in)

`LLM_DOUBLE_BUFFER_KV=1` allocates sibling output backings for chunk1
and chunk2 KV outputs (`kSliding{1,2}Out`, `vSliding{1,2}Out`,
`kFull{1,2}Out`, `vFull{1,2}Out`) and passes them via
`MLPredictionOptions.outputBackings`. After each prediction, the engine
verifies the model wrote into the supplied backing (compares
`dataPointer`); if so, it swaps the (in, out) roles instead of running
the four `copyBack` memcpys (~99 MB/step at 8K context). If the model
didn't honor the backing, the engine falls back to copyBack and logs
once per chunk.

```sh
LLM_DOUBLE_BUFFER_KV=1 ./CoreMLLLMChat
```

Memory cost: ~100 MB extra resident KV (sibling buffers). Acceptable
on 8 GB+ iPhones.

Risk: if the ANE compiler ignores `outputBackings` for IOSurface-backed
arrays, the swap path never triggers and behavior matches the default
copyBack path. The `[KV] chunk1 outputBackings honored — copyBack skipped`
log line confirms it took effect.

Expected impact (if backings honored): -8 to -15 ms/step → ~38–45 tok/s
and a sharp drop in `cpu_active`. **Must be validated on iPhone — Mac
behavior may differ.**

---

## What was deferred and why

### Async `predictStep` (originally Task #3)

Rejected after analysis. CoreML's sync `prediction(from:)` already
blocks on a Mach IPC kernel wait — the calling thread sleeps, it
doesn't spin. Switching to `await chunk.prediction(from:)` would block
on the same primitive without any thermal benefit, and would force
all sync callers (MTP, CrossVocab, DrafterUnion) into async contexts
they don't currently need.

If we later want **concurrent** mask/RoPE precomputation during ANE
wait, we'll add it then with a measurement-justified design.

### Cross-step dispatch pipelining (originally Task #5)

Within a single decode step, chunk1→chunk2→chunk3→chunk4 is strictly
data-dependent — no parallelism is possible. Across steps, only
mask/RoPE/umask for position+1 can be precomputed (embed and PLE
depend on the next token, which is chunk4's argmax). Mask + RoPE
together are ~1 ms/step today, so the upper bound on this win is
~2% — not worth the concurrency complexity until the bigger levers
land.

### INT8 KV cache, chunk3+4 merge, in-place KV via dynamic_update_slice

All require Python conversion changes + Mac parity testing + iPhone
re-export. Out of scope for this Swift-side investigation pass. See
`docs/LITERT_RUNTIME_ANALYSIS.md` Tier S/A for the conversion-side
plan once measurements justify the effort.

---

## iPhone arrival checklist (start here when device is in hand)

Quick path for the next session — every step takes < 5 minutes. Print
this section, run top-to-bottom, write the numbers in the table at the
bottom.

### Step 1 — Open Xcode, set env vars

`open Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj`

Xcode → Edit Scheme → Run → Arguments → **Environment Variables**.
Add (initially DISABLED — toggle them in Step 4):

| Name                     | Value     |
|--------------------------|-----------|
| `LLM_PROFILE_EVERY_STEP` | `1`       |
| `LLM_DOUBLE_BUFFER_KV`   | `1`       |
| `LLM_DECODE_QOS`         | `utility` |

`LLM_PROFILE_EVERY_STEP` is safe to keep on for all four configs — it
just controls log frequency.

### Step 2 — Baseline run (no env vars enabled)

Untick `LLM_DOUBLE_BUFFER_KV` and `LLM_DECODE_QOS` in the scheme. Run
on iPhone, type a prompt, generate ~100 tokens. From the Xcode console:

- Look for the **last** `[ANE/CPU]` line — that's the most stable
  reading after warmup.
- Note `tok/s` from the summary or last `[Profile]` line.

Record in the table below as **(A) baseline**.

### Step 3 — powermetrics on the Mac (in parallel with Step 2 repeat)

Tether iPhone → Mac via USB. In Mac terminal:

```sh
caffeinate -dimsu &
sudo ./scripts/powermetrics_bench.sh 60 1000
```

Start the iPhone generation IMMEDIATELY when the script says "Starting…".
The CSV at `/tmp/powermetrics_<unix>.csv` has the per-second power
breakdown; the script also prints means at the end.

Record `cpu_w`, `gpu_w`, `ane_w`, `combined_w` (means) for **(A) baseline**.

### Step 4 — Enable double-buffer KV

Re-tick **only** `LLM_DOUBLE_BUFFER_KV` in the scheme. Run again with
the same prompt and length. Watch the Xcode console for one of:

- `[KV] chunk1 outputBackings honored — copyBack skipped` → ✅ working
- `[KV] chunk1 outputBackings NOT honored — falling back to copyBack`
  → ❌ ANE compiler doesn't accept our backings; this is itself a
  finding — file under "ANE constraints to investigate"

Record as **(B) double-buffer**.

### Step 5 — Add utility QoS

Re-tick `LLM_DECODE_QOS` (value `utility`). Run again. Record as
**(C) double-buffer + utility QoS**.

### Step 6 — Fill in the table and decide

| Config                        | tok/s | ANE/copyBack/cpu (ms) | %CPU | cpu_w | combined_w |
|-------------------------------|-------|------------------------|------|-------|------------|
| (A) baseline                  |       |                        |      |       |            |
| (B) double-buffer             |       |                        |      |       |            |
| (C) double-buffer + utility   |       |                        |      |       |            |

**Decision rules:**

- If (A) shows **`%CPU` > 50** → CPU is the bottleneck (hypothesis confirmed).
- If (B) shows **`copyBack` ≈ 0 and tok/s ≥ (A)** → outputBackings works on
  ANE; ship `LLM_DOUBLE_BUFFER_KV=1` as default and re-test thermals.
- If (B) shows **NOT honored fallback** → file an issue; the fallback path
  is identical to (A) so no regression. Next investigation: check if
  `outputBackings` works for any chunk (try `MLPredictionOptions` with
  hidden_states_out only — those aren't IOSurface-backed).
- If (C) shows **`cpu_w` lower than (B), tok/s within 5%** → ship utility
  QoS. If tok/s drops > 5%, leave QoS as inherited.

If `combined_w` in (B)+(C) is now < 2 W and the device is no longer
visibly hotter than LiteRT-LM, the investigation is closed. Otherwise,
the residual heat is in ANE itself or DRAM bandwidth, and the next
levers are conversion-side (INT8 KV, chunk merge — see
`docs/LITERT_RUNTIME_ANALYSIS.md` Tier S).

---

## How to validate on Mac (before iPhone)

The Mac path is for sanity-checking the new code paths against any
locally-converted Gemma 4 chunks (e.g., `output/gemma4-e2b-2k/`). Mac
absolute numbers don't transfer to iPhone — the M-series ANE is much
larger — but the **deltas** between configs do.

### Single command — A/B/C in one shot

```sh
./scripts/test_double_buffer_locally.sh /path/to/model_dir 64
```

Builds release, runs three configurations against the same prompt, and
prints a comparison table. Detects whether `outputBackings` is honored
on Mac ANE (proxy for iPhone behavior). Raw logs are saved to
`/tmp/coreml-llm-bench.<random>/`.

### Manual smoke runs

```sh
swift build -c release

# baseline
./.build/release/coreml-llm-smoke /path/to/model_dir "Hi" 64

# double buffer
LLM_DOUBLE_BUFFER_KV=1 LLM_PROFILE_EVERY_STEP=1 \
  ./.build/release/coreml-llm-smoke /path/to/model_dir "Hi" 64

# double buffer + utility QoS
LLM_DOUBLE_BUFFER_KV=1 LLM_DECODE_QOS=utility LLM_PROFILE_EVERY_STEP=1 \
  ./.build/release/coreml-llm-smoke /path/to/model_dir "Hi" 64
```

What to look for in the logs:

- `[KV] chunk1 outputBackings honored — copyBack skipped` (only logs once)
- `[ANE/CPU] ANE_wait=Xms copyBack=Yms cpu_active=Zms (W% CPU)` per step
- `[QoS] LLM_DECODE_QOS=utility — decode loop priority overridden`

If the Mac smoke run shows **outputBackings NOT honored**, there's still
a chance iPhone behaves differently (the iPhone ANE compiler is a
separate binary). It's worth trying on iPhone regardless, since the
fallback path is identical to baseline.
