# iPhone Lever Vault (2026-05-14)

Untested levers researched 2026-05-13 evening for tomorrow's iPhone
session. Every entry has been cross-checked against existing memory
to avoid duplication.

Pre-conditions for any bench:
* iPhone 17 Pro at room temp, `state=fair` (first chunk load ≤ 0.2 s).
* App installed from today's `feat/mtp-iphone-perf` branch (commit
  `6e422a0` or newer).
* Bundle: `Documents/Models/gemma4-e2b/` (push via
  `bash scripts/push_gemma4_e2b_bundle.sh /tmp/push-bundle`).

## Lever queue — env-only (no rebuild)

Order: lowest cost first, highest expected info-per-bench.

### L1. K_USE=1 + warmup cool retest (REPRODUCIBILITY)

```bash
DEVICE=$(xcrun devicectl list devices | grep "iPhone 17 Pro" | grep connected \
  | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}' | head -1)
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "code", "MTP_K_USE": "1"}' \
  com.example.CoreMLLLMChat > /tmp/L1_kuse1_warm.log 2>&1
```

Hypothesis: AutoBench warmup (`bc5b04a`) activates drafter so K_USE=1
cold-bail doesn't fire. Run 2-3 times with 10-min cool gaps; if code
hits **50+ tok/s consistently**, K_USE=1 is the right code default
and we revert `e9af22d`'s K_USE=2 fallback.

Cool: ~3 mins per bench. Cost: 3 cool gaps × 10 min = 35-40 min total.

### L2. FLy K sweep on iPhone

```bash
bash scripts/iphone_autobench_sweep.sh fly_topk code
```

Tests `MTP_FLY_TOPK` ∈ {8, 16, 24, 32}. **Manually insert 10-min cool
gaps between values** (script doesn't pace).

Mac plateau: K=16 (K=24/32 no further lift, K=32 collapse). iPhone
TBD — verify cycle ratio is different so the curve may shift.
Hypothesis: K=24 or K=32 might unlock another +5-10 % on iPhone code
without quality collapse. K=8 likely strict-narrative-aware (less
lossy, may help narrative).

### L3. Fallback threshold sweep

```bash
bash scripts/iphone_autobench_sweep.sh bail_threshold narrative
```

`MTP_FALLBACK_THRESHOLD` ∈ {0.20, 0.25, 0.30 (current), 0.35, 0.40}
on narrative. 0.30 was a snap choice; sweet spot may be elsewhere.

* 0.20 narrative: lets MTP try more drafter cycles before bail —
  might recover small narrative gain at cost of some net-negative cycles.
* 0.35-0.40: bails faster — more conservative, may slightly improve
  steady-state narrative.

### L4. MTP_PLD_PREFETCH_ENABLE on code

```bash
# code + structured prompts have repetitive token n-grams
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "code,list", "MTP_PLD_PREFETCH_ENABLE": "1"}' \
  com.example.CoreMLLLMChat > /tmp/L4_pld.log 2>&1
```

Memory note: PLD net-negative on Mac free-form (drops emit 1.78 →
1.63). But for code/list with high n-gram repetition, may save 11 ms
drafter call per hit. iPhone code with PLD untested.

Expected: +5-15 % on code-heavy prompts with repetition, neutral or
negative on narrative.

### L5. MTP_SELF_BAIL_DISABLE=1

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "code", "MTP_SELF_BAIL_DISABLE": "1"}' \
  com.example.CoreMLLLMChat > /tmp/L5_no_self_bail.log 2>&1
```

Self-bail kicks in when drafter softmax probability on its top-1
drops below 0.40. Default ON because "subsequent tokens likely wrong
anyway". On iPhone where drafter cost is small (1.8 ms warm), running
all K drafter steps and letting verify reject is cheap. Disable may
let through more partial accepts.

### L6. LLM_FAST_PREDICTION=1 on iPhone

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "narrative,code", "LLM_FAST_PREDICTION": "1"}' \
  com.example.CoreMLLLMChat > /tmp/L6_fast_pred.log 2>&1
```

`MLOptimizationHints.specializationStrategy = .fastPrediction`. Mac:
within-noise gain. iPhone: untested, but A19 ANE may benefit
differently than M-series. Cost is longer first-load (longer ANE
compile); steady state TBD.

### L7. MTP_DRAFT_POS_MODE=perstep variant

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "code", "MTP_DRAFT_POS_MODE": "perstep"}' \
  com.example.CoreMLLLMChat > /tmp/L7_perstep.log 2>&1
```

Default is `constpm1` (HF behavior, +3 % vs perstep on Mac centroid).
iPhone untested — perstep may give different acc on iPhone ANE
quantization noise pattern.

### L8. MTP_L5_ASYNC_DISABLE=1 — disable async drafter

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "code", "MTP_L5_ASYNC_DISABLE": "1"}' \
  com.example.CoreMLLLMChat > /tmp/L8_no_l5.log 2>&1
```

L5 async drafter speculation runs background drafter for cycle N+1
during cycle N's verify, hoping for full-accept hit. If iPhone has
limited ANE concurrency, L5 may compete with main MTP and slow
things. Disable to measure if L5 actually wins on iPhone (Mac:
shipped default ON, iPhone shipped ON, untested OFF).

### L9. Stateful Linear path A/B

This requires pushing a different bundle (`gemma4-e2b-stateful-linear`).

```bash
# Pre-step: assemble + push the stateful Linear bundle
# Source: /Users/majimadaisuke/Downloads/workspace/CoreML-LLM-stage3/build/gemma4_stateful_3chunk_linear/gemma4_e2b_stateful_chunks/
# Target: Documents/Models/gemma4-e2b-stateful-linear/gemma4_e2b_stateful_chunks/
# Use scripts/push_gemma4_e2b_bundle.sh logic as template — needs new push script.
```

Memory: `project_stateful_plan3_phase2a.md` says Mac 34.6 tok/s,
"iPhone parity". Implemented as separate `Gemma4StatefulGenerator`
codepath in LLMRunner. AutoBench currently loads `gemma4-e2b`
(ChunkedEngine path); stateful would need
`LLM_AUTOBENCH_MODEL=gemma4-e2b-stateful-linear`.

**Caveat**: stateful path doesn't use MTP — it's a pure T=1 decode
with Linear LM head. Bench comparison is T=1 stateful vs T=1
ChunkedEngine. Memory says +5 % on Mac (32 → 34.6). iPhone similar
gain projection.

## Lever queue — Swift change required (rebuild)

### L10. Per-prompt K_USE adapter

File: `Sources/CoreMLLLM/MtpSpeculativeEngine.swift` ~line 384.

Sketch:
```swift
// After kEffective is determined, also track first-cycle acc.
// If first MTP cycle hits accept >= drafter K (warm), drop K_USE
// to 1 for the rest of this prompt. Else stay at K-1.
private var firstCycleAccept: Int? = nil
private var adaptedKEffective: Int? = nil

// In speculateStep, after the first cycle:
if firstCycleAccept == nil {
    firstCycleAccept = matched
    if matched >= kEffective {
        adaptedKEffective = 1   // drop to 1 — warm code path
    } else {
        adaptedKEffective = nil // keep K-1
    }
}
// On reset (new prompt), clear both.
```

Expected: warm code prompts auto-drop to K_USE=1 (+20 % per code), cold
or narrative stay at K_USE=2 (no regression). Code change ~30 lines.

### L11. Custom MLFeatureProvider — skip dict alloc

File: `Sources/CoreMLLLM/ChunkedEngine.swift` — 27 sites.

Sketch:
```swift
final class MutableFeatureProvider: NSObject, MLFeatureProvider {
    var values: [String: MLFeatureValue] = [:]
    var featureNames: Set<String> { Set(values.keys) }
    func featureValue(for featureName: String) -> MLFeatureValue? {
        values[featureName]
    }
}

// Per-chunk persistent providers (one each for c1/c2/c3/c4, decode+verify).
private let decodeIn1Provider = MutableFeatureProvider()
// ... etc

// In predictStep/verifyCandidates:
decodeIn1Provider.values["hidden_states"] = MLFeatureValue(multiArray: hiddenIn)
decodeIn1Provider.values["causal_mask_full"] = ...
let out1 = try chunk1.prediction(from: decodeIn1Provider)
```

Saves: 27 × ~30-50 µs = 0.8-1.4 ms per decode/verify cycle. Mac
within noise; iPhone autorelease churn benefit. Risk: low. Cost:
~80 lines.

### L12. Drafter call A/B — 38MB vs centroid 149MB iPhone

Already known Mac result (38MB slower drafter, centroid faster). On
iPhone untested directly. If centroid drafter still wins big on
iPhone, it confirms production pick. If 38MB drafter has unexpected
iPhone-ANE advantage (smaller weight residency), revisit.

```bash
# Restore 38MB drafter then push:
mv output/gemma4-e2b/bundle_diff_logits/mtp_drafter.mlmodelc \
   /tmp/mtp_drafter_centroid_149mb.bak.mlmodelc
cp -R /tmp/mtp_drafter_38mb_bak.mlmodelc \
   output/gemma4-e2b/bundle_diff_logits/mtp_drafter.mlmodelc
# rebuild push bundle, push, bench
# Then restore centroid:
cp -R /tmp/mtp_drafter_centroid_149mb.bak.mlmodelc \
   output/gemma4-e2b/bundle_diff_logits/mtp_drafter.mlmodelc
```

## Lever queue — model rebuild (Python conversion)

Skip unless above levers exhausted.

### L13. Drafter K=2 variant — model rebuild

Currently K=3 drafter (3 proposed tokens per call). K=2 drafter would
shrink verify cycle to K=2 instead of K=3. iPhone verify cycle 35 ms →
maybe 28 ms. But also means K_USE max is 2, and K_USE=1 is the
practical default — so the speed gain only manifests if K_USE=2 was
already winning.

Cost: Python `build_mtp_drafter.py` rerun + verify_qK K=2 rebuild for
the chunks (`build_verify_chunks.py --K 2`). ~30-60 min Python.

### L14. K=4 drafter — opposite direction

Test "more draft per cycle" with K=4 verify chunks. Drafter call only
slightly longer; verify cycle ~+5 ms; but K_USE max 3 = more accepted
per cycle when drafter agrees. ROI depends on per-slot accept rate.

Cost: same as L13.

### L15. Sliding window W=256 (was 512)

Memory `project_iphone_ane_sparsity.md` says padding free on iPhone.
But W=256 reduces verify chunk K cache + drafter kv13 cache size.
iPhone-side may benefit.

Cost: full chunk rebuild + drafter rebuild compatible. 2-3 hours.

## Cross-cutting research worth doing

### R1. iPhone verify cycle timing decomposition

Mac verify: ~35 ms. iPhone verify: ~52 ms. The 17 ms iPhone overhead
could be:
* ANE 18 dispatch overhead per chunk × 4 chunks
* CPU-side dict construction / autorelease churn
* iPhone-ANE memory bandwidth bottleneck on kv13/14 read

**Pre-bench profile run on cool iPhone** with extra tracing in
verifyCandidates would tell us where. Currently the
`[SpecProfile mtp #...]` lines aggregate to `verify=33.5ms` — but
don't break down by chunk.

### R2. ANE residency thrash audit

If drafter (149MB) competes with chunks (1.3 GB) for ANE memory,
moving drafter to GPU could free ANE space → faster chunks. Mac:
GPU drafter slower per call (Mac M3 CPU is fast). iPhone: untested.

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "code", "MTP_DRAFTER_DEVICE": "gpu"}' \
  com.example.CoreMLLLMChat > /tmp/R2_drafter_gpu.log 2>&1
```

Expected: drafter call time up (gpu slower than ane for small model),
but ANE chunks may run faster. Net TBD.

## Already-tried, do NOT retest

| lever | result | source |
|---|---|---|
| 16-lever inventory | all training-free Mac levers exhausted | `IPHONE_SPEEDUP_LEVER_INVENTORY_2026_05_13.md` |
| L12 subset LM head | iPhone -45 % | `project_l12_iphone_regression` |
| 3-chunk + MTP | drafter mismatch | `project_3way_mtp_dead_drafter_mismatch` |
| AWQ chunk2 | no go | `project_awq_nogo_e2b` |
| Path A drafter | dead | `project_drafter_structurally_dead` |
| LayerSkip | 0 % twice | (memory) |
| EAGLE-3 HASS | 14 % live | (memory) |
| Cross-vocab Qwen | net negative | (memory) |
| ANEMLL 16-way LM head split | -4.6 % iPhone | `project_lmsplit_rejected` |
| 4→2 chunk consolidation | +1 tok/s only | `project_chunk_consolidation_dead` |
| Decode output backings | iPhone crash | `5b68fb3` |
| Tree verify | -20 % yes-yes | (code note) |
| Adaptive K_USE | Mac +3.5 % vs static +13.8 % | (code note) |
| L5 async drafter | Mac shipped default ON | (memory) |
| K_USE=1 iOS default | cold-bails (`e9af22d` reverted) | mistake #11 |

## Tomorrow's recommended order

If iPhone fully cool and 1 hour budget:

1. **L1** (15 min) — K_USE=1 warmup retest. Gives the headline answer
   about whether warmup unlocks the +24 % code default.
2. **L4** (15 min) — PLD prefetch on code. Cheap test, possible code +10 %.
3. **L2** (45 min) — FLy K sweep (4 values × ~5 min run + cool gaps). Highest
   info per bench across the lever space.
4. **L3** (30 min) — Fallback threshold on narrative. Confirms 0.30
   choice or finds better.

If 30-min budget:
* L1 + L4 only. Skip the sweeps until next session.

If 2+ hour budget AND extra cool time:
* L1, L2, L4 → L9 (stateful Linear push + bench) → L11 (Custom
  MLFeatureProvider) Swift change.

Never run more than 4 timed prompts in a single launch — iPhone
thermal=serious kicks in around the 5th and contaminates results.
