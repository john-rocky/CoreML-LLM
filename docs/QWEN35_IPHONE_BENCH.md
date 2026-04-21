# Qwen3.5-0.8B iPhone Benchmark — USB Deploy Guide

Measures ANE drift and prefill throughput for the fp16 Qwen3.5-0.8B prefill
mlpackage on iPhone 17 Pro (A18 Pro ANE).

## What you're measuring

Mac-side reference numbers (from `test_qwen3_5_full_model_trace.py` +
`palettize_qwen3_5_int4.py` runtime predict on Mac Studio M4 / coremltools 8.3):

| runtime | mean cos | worst-pos cos | top-1 match |
|---|---|---|---|
| torch fp32 (oracle) | 1.0 | 0.999987 | 100% |
| CoreML fp16, CPU only | 0.99988 | 0.998145 | **100%** |
| CoreML fp16, CPU+ANE (M4) | 0.9830 | 0.8427 | **80%** |

The A18 Pro ANE is a different chip. This benchmark answers:
- Is the A18 drift similar, better, or worse than M4?
- What is the actual prefill throughput (tok/s) on device?

## Deploy steps

**Prereqs**: Xcode 15+, iPhone 17 Pro (or any iOS 18+ device), Mac.

### 1. Generate the mlpackage on Mac

The fp16 mlpackage is already at `/tmp/qwen35_build_fwdsub/qwen3_5_0_8b_fp16_seq64.mlpackage`
if you ran the conversion earlier this session. If not:

```bash
cd conversion
/Users/$(whoami)/.pyenv/versions/3.10.13/envs/lama-cml/bin/python \
    test_qwen3_5_full_model_trace.py \
    --seq-len 64 --precision fp16 \
    --out-dir /tmp/qwen35_build_fwdsub
```

Expect ~2-4 minutes. Output: `/tmp/qwen35_build_fwdsub/qwen3_5_0_8b_fp16_seq64.mlpackage` (≈ 1.5 GB).

### 2. Generate the oracle bundle

```bash
/Users/$(whoami)/.pyenv/versions/3.10.13/envs/lama-cml/bin/python \
    conversion/export_oracle_for_ios.py
```

Output: `conversion/qwen3_5_oracle_ios.json` (≈ 6.3 MB).

### 3. Add both files to the Xcode project

Open `Examples/CoreMLLLMChat/CoreMLLLMChat.xcodeproj` and drag both:
- `/tmp/qwen35_build_fwdsub/qwen3_5_0_8b_fp16_seq64.mlpackage`
- `conversion/qwen3_5_oracle_ios.json`

into the `CoreMLLLMChat` target's file list. Make sure:
- Target membership: `CoreMLLLMChat` ✅
- For the mlpackage, Xcode auto-compiles it into `qwen3_5_0_8b_fp16_seq64.mlmodelc`
  inside the app bundle at build time.
- For the JSON, it's copied verbatim as a bundle resource.

### 4. Build + run on device

1. Connect iPhone 17 Pro via USB (or wireless debugging if paired).
2. Select the iPhone as the build target.
3. Build & run (⌘R).
4. In the app's model picker screen, scroll to **Research → Qwen3.5-0.8B
   prefill benchmark**.
5. Pick a compute unit (`CPU+ANE` is the default; compare against `CPU only`
   as a baseline) and tap **Run benchmark**.

### 5. Read results

The summary row shows mean cos / worst cos / top-1 match / prefill latency
/ throughput. Per-prompt rows show cos and latency for each of the 10 oracle
prompts.

Device-side data points to collect:
- `CPU+ANE` top-1 and worst-pos cos on A18 — compare to 80% / 0.843 on M4
- `CPU only` top-1 on A18 — should match 100% / 0.998 on Mac (same math)
- Prefill tok/s in both modes — LiteRT-LM baseline is 56.5 tok/s

## What to do with the results

- **If A18 ANE top-1 ≥ 95%**: huge win, the Mac M4 ANE drift doesn't
  generalize; proceed to Phase 4e-1 (full stateful decode converter) and
  Phase 4e-4 (Swift generation loop).
- **If A18 ANE top-1 ≈ 80% similar to M4**: ANE fp16 ceiling is real for
  this architecture; either (a) invest in mixed-precision via
  MIL-graph-level intervention, or (b) ship the `CPU only` path with
  acceptable throughput.
- **If A18 ANE top-1 < 70%**: A18 ANE has stricter fp16 behavior than M4;
  same options as above, tipped toward option (b).

## Decode benchmark (Phase 4e-1)

A second benchmark screen measures the **stateful decode** path — the
per-token auto-regressive step used during text generation. This is
the component whose throughput determines tok/s during real generation.

**Mac M4 Studio reference numbers** (from `test_qwen3_5_full_decode_trace.py`
runtime predict, zero-init states, HF recurrent oracle):

| runtime | mean cos | worst cos | top-1 | tok/s |
|---|---|---|---|---|
| CPU fp16 | 0.99992 | 0.99985 | 100% | ~50 |
| CPU+ANE fp16 | 0.99 | 0.977 | 40% | ~40 |

**Setup:**

1. Build the decode mlpackage:
   ```bash
   /Users/$(whoami)/.pyenv/versions/3.10.13/envs/lama-cml/bin/python \
       conversion/test_qwen3_5_full_decode_trace.py \
       --out-dir /tmp/qwen35_build_decode
   ```
   Output: `/tmp/qwen35_build_decode/qwen3_5_0_8b_decode_fp16_mseq128.mlpackage` (≈ 1.5 GB).

2. Drag into the Xcode project alongside the prefill mlpackage.
   The decode benchmark loads both `qwen3_5_0_8b_decode_fp16_mseq128.mlmodelc`
   and the shared `qwen3_5_oracle_ios.json`.

3. In the app: **Models → Research → Qwen3.5-0.8B decode benchmark**.

**Decision rule** for the decode benchmark:
- If CPU-only tok/s on A18 Pro >= 56.5 → **shipping path found** (CPU fp16
  clears LiteRT without needing ANE). Proceed to Phase 4e-2 (prefill
  state export) + 4e-4 (Swift generation loop).
- If CPU-only < 56.5 but CPU+ANE top-1 recovers on A18 → invest in ANE
  drift investigation (a win doubles as speed + accuracy).
- If both fall short → consider INT4 weight-only (with relaxed kmeans
  settings) to shrink memory and cache-miss overhead.

## Known limitations of this harness

- Measures **prefill only** (seq=64 fixed). Generation requires the decode
  converter (Phase 4e-1) which is not in this build.
- Cos measured at **last position only** (keeps the JSON bundle small at
  6 MB). Per-position worst-pos drift on Mac uses all positions, so
  A18 worst-pos numbers here aren't directly comparable to the
  `palettize_qwen3_5_int4.py` Mac numbers — but the mean + top-1 are.
- Throughput is the full-sequence prefill time, not per-generated-token.
