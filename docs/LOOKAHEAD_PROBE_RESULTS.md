# LookAhead K=8 probe — results

**Status:** Mac sanity = GO (30.15 ms median), iPhone measurement pending
**Branch:** `probe/lookahead-k8`
**Gate source:** `docs/LOOKAHEAD_PROBE_HANDOFF.md` §2.4

## Decision gate

| Measured verify_qK=8 median | Verdict | Next step |
|---|---|---|
| < 50 ms | GO | Full LookAhead implementation (§3 of handoff) |
| 50 – 70 ms | MARGINAL | Defer; revisit if Metal Phase 3 slips |
| > 70 ms | KILL | K-linear scaling confirmed; close LookAhead route |

## Setup

**Mac (conversion):**
```bash
cd conversion
PYENV_VERSION=lama-cml python build_verify_chunks.py --K 8 \
  --model gemma4-e2b --ctx 2048 \
  --output ../output/gemma4-e2b/chunks-k8 --keep-tmp
```

Build target: `output/gemma4-e2b/chunks-k8/chunk{1-4}.mlpackage`
(multifunction: `decode_q1` + `verify_qK` @ K=8, W4A8 palettized).

**Mac (compile):**
```bash
cd /Users/majimadaisuke/Downloads/workspace/CoreML-LLM
for c in chunk1 chunk2 chunk3 chunk4; do
  xcrun coremlcompiler compile \
    output/gemma4-e2b/chunks-k8/${c}.mlpackage \
    output/gemma4-e2b/chunks-k8/
done
```

**Mac (bundle assembly):**
```bash
./scripts/assemble_lookahead_probe_bundle.sh
# → /Users/majimadaisuke/Downloads/device_deploy_lookahead_probe/
```

Copies `coreml-llm-artifacts/backup-iphone-2k/` as base (no drafter artefacts,
2K ctx, W=512) and swaps `chunk1-4.mlmodelc` with the K=8 variants. Existing
bundles at `backup-iphone-2k/` and on-device `Documents/Models/gemma4-e2b/`
are NOT modified.

**iPhone (sideload — back up first per `docs/USB_MODEL_SIDELOAD.md`):**
```bash
DEVICE=$(xcrun devicectl list devices | awk '/connected/{print $3}' | head -1)

# 1. Backup current device bundle
xcrun devicectl device copy from \
  --device "$DEVICE" \
  --domain-type appDataContainer \
  --domain-identifier com.example.CoreMLLLMChat \
  --source Documents/Models/gemma4-e2b \
  --destination ~/Downloads/coreml-llm-artifacts/backup-pre-k8-probe-$(date +%Y%m%d-%H%M)

# 2. Push probe bundle
xcrun devicectl device copy to \
  --device "$DEVICE" \
  --domain-type appDataContainer \
  --domain-identifier com.example.CoreMLLLMChat \
  --source /Users/majimadaisuke/Downloads/device_deploy_lookahead_probe \
  --destination Documents/Models/gemma4-e2b \
  --remove-existing-content true
```

**iPhone (measurement):**
Build `verify-k8-probe` for iOS (add to CoreMLLLMChat app or run via Xcode
on-device test target). Or run on Mac first:

```bash
swift run -c release verify-k8-probe \
  /Users/majimadaisuke/Downloads/device_deploy_lookahead_probe 25
```

Mac-side numbers are a sanity check only — the go / no-go gate uses
**iPhone 17 Pro** numbers because ANE dispatch behaviour differs.

## Measurements

### Mac (Mac Studio, 2026-04-22, 15 iter)

Hardware: M-series Mac Studio. Bundle:
`/Users/majimadaisuke/Downloads/device_deploy_lookahead_probe` (2K ctx,
K=8 verify chunks). Compute units: `cpuAndNeuralEngine` (default).

```
[Profile] c1=6.3 c2=7.7 c3=7.6 c4=10.3 (sum=31.9ms)     # decode_q1 (prewarm 8/8)
[probe] benchVerifyK=8 ✓
[Prefill] prep=7.8ms c1=7.4ms c2=6.4ms c3=10.8ms c4=15.0ms total=47.5ms (19 tokens, 400 tok/s)

[probe] verify_qK=8 timing over 15 iters (ms)
[probe]   min    =  30.05
[probe]   median =  30.15
[probe]   mean   =  30.29
[probe]   p90    =  30.50
[probe]   p99    =  31.90
[probe]   max    =  31.90
[probe]   std    =   0.45

[probe] verdict: GO — LookAhead implementation is worth committing to
```

| Metric | Value (ms) |
|---|---|
| min | 30.05 |
| median | **30.15** |
| mean | 30.29 |
| p90 | 30.50 |
| p99 | 31.90 |
| max | 31.90 |
| std | 0.45 |

**Interpretation.** `verify_qK=8` (30.15 ms) is essentially identical to
`decode_q1` (≈31.9 ms sum-of-chunks after final prewarm). K=8 batch costs
the same as K=1 single-step on Mac ANE — batch-invariance regime holds at
8× the batch. This is the strongest possible Mac signal for the LookAhead
route.

**Caveat:** Mac ANE ≠ iPhone 17 Pro ANE (different silicon, different
scheduler). iPhone number is still the decision point. But a strong Mac
signal rules out the "build failed, measurement nonsense" class of
failures.

## Full-stack Mac measurement (LookAhead vs serial decode)

With the Mac signal for verify_qK=8 ≈ decode_q1, built
`Sources/CoreMLLLM/LookaheadEngine.swift` (linear Jacobi + PLD + suffix
trie integration), wired behind `lookaheadEnabled` / `LLM_LOOKAHEAD_ENABLE=1`,
Mac-measured against serial decode on the same probe bundle:

| Prompt | Baseline tok/s | LookAhead tok/s | Speedup | Rolling accept |
|---|---|---|---|---|
| "Write a short poem about the ocean." | 33.18 | 34.90 | **+5.2%** | 1.4% |
| "Write a Python function named factorial…" | 32.9 | 33.17 | +0.8% | 0.4% |
| "Count from 1 to 20, each on a new line:" | 33.44 | 36.05 | **+7.8%** | 3% |
| "List months of the year, days in each…" | 33.42 | **57.39** | **+71.7%** | 11–13% |

**Pattern.** Free-form outputs (poetry, novel code) get marginal win —
PLD / suffix n-grams rarely hit, Jacobi fixed-point iteration alone
can't guess well. Structured outputs with repeated templates (lists,
formatted output, boilerplate) get a large win because PLD proposes
the next 4-5 tokens correctly and verify_qK=8 accepts them in one
ANE dispatch.

**Per-cycle detail (months prompt):**
```
[SpecProfile lookahead #0051] accepted=4/7 emitted=5 rolling=0.123
[SpecProfile lookahead #0053] accepted=4/7 emitted=5 rolling=0.132
[SpecProfile lookahead #0074] accepted=5/7 emitted=6 rolling=0.114
```

Best case: 6 tokens committed in one 30 ms verify = 5 tok / verify =
~170 tok/s local peak. Amortised over cycles with 0-accepts, settles
at the observed 57 tok/s.

**Numerics caveat.** verify_qK=8 graph produces slightly different
argmax than decode_q1 graph (K-dependent fp16 drift, same issue that
killed MTP Path A). LookAhead output is coherent but NOT bit-exact
with serial decode — same prompt lands on a different coherent
trajectory. This is stack-inherent and applies to any K-based
speculation path on our pipeline.

### iPhone 17 Pro (decision-point)

```
TODO: paste [probe] output here
```

| Metric | Value (ms) |
|---|---|
| min | — |
| median | — |
| mean | — |
| p90 | — |
| p99 | — |
| max | — |
| std | — |

**Reference anchors (docs/ROUND7_FINDINGS.md:39-45):**

- decode_q1: 32.3 ms
- verify_qK=3: 31.5 ms (~flat vs decode_q1)

Near-flat scaling at K=3 is the reason LookAhead is worth probing at all.
Question: does the flat regime extend to K=8, or does K=8 cross into a
different ANE scheduling regime (e.g. multi-tile dispatch, additional
partial-buffer spills)?

## Verdict

**TODO — fill in after measurement.**

```
GO | MARGINAL | KILL — <one-sentence reason citing the median>
```

### If GO

Proceed to §3 of `docs/LOOKAHEAD_PROBE_HANDOFF.md`:

1. Write `Sources/CoreMLLLM/LookaheadEngine.swift` (~300 LOC)
2. Extend `SpeculativeTarget` protocol for K≥8 verify
3. N-gram cache ring buffer (~150 LOC)
4. Wire into `CoreMLLLM.swift` as opt-in via env gate
5. iPhone validation: ≥ +15% over 30 tok/s baseline

### If MARGINAL

- Park the K=8 chunks + probe CLI on this branch for later reuse
- Let Metal Phase 3 progress dictate revival
- Document decision in `project_drafter_structurally_dead.md` memory

### If KILL

- Append "LookAhead K=8 — ANE K-linear confirmed, dead" to the drafter
  memory's "Still dead" list
- Close this branch without merging (or merge docs-only update)
- Metal Phase 3 becomes the sole remaining path to LiteRT 56 parity

## Known risks and caveats

- **ANE compile budget.** Replacing K=3 chunks with K=8 chunks keeps the
  per-process graph count stable (still 2 functions per chunk: `decode_q1` +
  `verify_qK`). Should not push toward the ~100/process limit.
- **Warmup.** First verify call pays ANE compile-schedule cost. Probe
  discards the first 3 iterations.
- **Sideloaded file ownership.** Pushed files are owned by root; the app
  runs as mobile. Mass deletion needs reinstall. See USB_MODEL_SIDELOAD.md.
- **Multifunction PTQ.** `build_verify_chunks.py` uses per-function
  palettization then bundles via `save_multifunction`. Tested at K=3;
  K=8 should behave identically but worth watching weight-bin size.
