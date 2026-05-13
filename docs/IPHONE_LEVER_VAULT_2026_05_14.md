# iPhone Lever Vault (2026-05-14)

**⚠️ SUPERSEDED BY `docs/RESEARCH_FINDINGS_2026_05_13.md` FOR TIER-S LEVERS**

The Tier 1-6 below was an early brainstorm that included speculative
padding (L16-L46). After 6 parallel research agents investigated 2026
H1 papers + Apple disclosures + competing iPhone LLM projects + ANE
empirical tricks, the high-conviction shortlist is in
`RESEARCH_FINDINGS_2026_05_13.md`. Read THAT first.

Below sections L1-L8 remain valid (env-only sweeps with concrete
commands); L16-L46 are kept only as historical brainstorm and should
NOT be acted on without re-checking against the research findings.

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

## More env-only levers (L16-L25, no rebuild)

### L16. LLM_DECODE_QOS=high — promote decode loop to perf cores

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "narrative,code", "LLM_DECODE_QOS": "high"}' \
  com.example.CoreMLLLMChat > /tmp/L16_qos_high.log 2>&1
```

Decode loop QoS defaults to `.userInitiated` (inherited from UI). Set
`high` to bias toward P-cores. Comment says "trades tok/s loss for
cooler operation" for utility; `high` is the inverse direction. May
help short-burst peak tok/s; risk: hotter thermal escalation.

### L17. LLM_DECODE_QOS=utility — anti-thermal sustained mode

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "narrative,code", "LLM_DECODE_QOS": "utility"}' \
  com.example.CoreMLLLMChat > /tmp/L17_qos_utility.log 2>&1
```

Bias toward E-cores. Lower peak, but **may sustain longer without
thermal=serious** — interesting for chat-bot UX (long sessions).

### L18. MTP_MARS_ENABLE=1 — Margin-Aware Verification (lossy)

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "code", "MTP_MARS_ENABLE": "1"}' \
  com.example.CoreMLLLMChat > /tmp/L18_mars.log 2>&1
```

Accept drafter token iff equals target's top-2 AND z2/z1 ≥ theta
(default 0.9). arxiv 2601.15498. Memory note: "1.5-1.8× initial
claim was incoherent output." Quality-bounded; verify output coherent
before claiming gain.

### L19. MTP_CSD_ENABLE=1 — Calibrated Speculative Decoding

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "code", "MTP_CSD_ENABLE": "1"}' \
  com.example.CoreMLLLMChat > /tmp/L19_csd.log 2>&1
```

arxiv 2604.13634. Historical bias correction: after λ historical
(drafter, target_top1) divergences, rescue subsequent occurrences
when z_drafter ≥ z_top1 + log τ. Default λ=2, τ=0.01. Same lossy
caveat as MARS.

### L20. LLM_LOOKAHEAD_ENABLE=1 — Jacobi/Lookahead (drafter-free)

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "narrative,code", "LLM_LOOKAHEAD_ENABLE": "1"}' \
  com.example.CoreMLLLMChat > /tmp/L20_lookahead.log 2>&1
```

Drafter-free speculative path. Memory `project_drafter_structurally_dead`
mentions "shipped opt-in 2026-04-22" with mac +5-72 % by workload
(structured > free-form), iPhone "verify_qK=8 GO gate" but iPhone
production left ON only in probe bundle. Untested with today's
fc31660 stack. Could complement MTP.

### L21. LLM_PREFIX_CACHE=1 — multi-turn TTFT (not tok/s)

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "narrative", "LLM_PREFIX_CACHE": "1"}' \
  com.example.CoreMLLLMChat > /tmp/L21_prefix_cache.log 2>&1
```

Disk-backed prefix cache. Saves on multi-turn TTFT (prefill re-use).
Not a per-token tok/s lever — but UX win for chat-bot conversations.
Memory: `project_stateful_plan3_phase2a` says multi-turn TTFT
-95 % (Phase 2a). Verify on iPhone.

### L22. MTP_TEMPERATURE=0.7 — rejection sampling

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "narrative", "MTP_TEMPERATURE": "0.7"}' \
  com.example.CoreMLLLMChat > /tmp/L22_temp.log 2>&1
```

Sampling-based MTP. Mac result: T>0 SLOWED narrative (~0.85× of
greedy on free-form per memory). iPhone CPU different — may differ.
Quality changes from greedy; verify coherence.

### L23. MTP_DRAFTER_T=0.5 — asymmetric drafter temperature

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "narrative", "MTP_TEMPERATURE": "1.0", "MTP_DRAFTER_T": "0.5"}' \
  com.example.CoreMLLLMChat > /tmp/L23_async_temp.log 2>&1
```

Drafter samples at T=0.5, target verifies at T=1.0. Softens drafter
top-1 over-confidence. iPhone untested.

### L24. LLM_LOAD_MAX_PARALLEL — chunk load parallelism

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_LOAD_MAX_PARALLEL": "0"}' \
  com.example.CoreMLLLMChat > /tmp/L24_load_parallel.log 2>&1
```

Default = max parallel chunk loads. `0` = sequential. Mostly TTFT
lever (one-time load cost), not steady-state tok/s. Useful when iOS
thermal kicks in during parallel load (sequential may avoid
state=serious during launch).

### L25. LLM_COMPUTE_UNITS=cpuOnly — pin to CPU (diagnostic only)

```bash
# Diagnostic — confirms ANE is the bottleneck. Will be 5-10x slower.
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_COMPUTE_UNITS": "cpuOnly", "LLM_AUTOBENCH_PROMPTS": "code"}' \
  com.example.CoreMLLLMChat > /tmp/L25_cpu_only.log 2>&1
```

Confirms ANE path is the gain source. Not a production lever.

## Algorithmic ideas (L26-L31, Swift change or research)

### L26. Two-stage drafter cascade

Small drafter A (38 MB, fast 1.0 ms) generates proposal; if low
confidence per-step, fallback to centroid drafter B (149 MB, 1.8 ms)
for that step only. Bottom-of-cycle drafter time amortizes.

Memory says 38 MB drafter has worse accept on Mac. But cascade
philosophy: 38 MB accept rate × 1.0 ms cost may beat centroid for
"easy" tokens. Test by adding a confidence threshold + fallback.

Implementation: ~100 lines Swift. Risk: medium (drafter state
management).

### L27. PLD-only mode (no drafter)

Drop drafter entirely; use PromptLookupDraft for repetitive prompts.
Saves 1.8 ms drafter call per cycle. Wins on yes-yes-style content;
loses on novel narrative.

`MTP_DRAFTER_DISABLE=1` env knob would need to be added. ~10 line
Swift change.

### L28. Token recycling

Rejected drafter tokens stored as candidate PLD n-grams for next
cycle. When future drafter proposes the same token and target also
emits it, that's a no-cost re-acceptance.

Implementation: ~50 lines Swift. Speculative gain: niche.

### L29. Speculative streaming UI

Yield drafter tokens to the UI BEFORE verify completes. On miss,
"rewind" the displayed text. User perceives faster output even when
acceptance is medium. UX lever, not tok/s.

Implementation: UI + cancellation. Medium-large change.

### L30. Adaptive drafter temperature

Increase drafter temperature on rejection streak. Counter-intuitive
but: drafter's argmax may be wrong, top-2/3 may be right. Lifting T
explores those.

Implementation: ~30 lines Swift.

### L31. Multiple parallel drafters voting

Run drafter A and drafter B concurrently (different training). Accept
if either matches target. Requires two drafters loaded → memory
budget; may not fit on iPhone.

## Build-time levers (L32-L37)

### L32. group_size sweep — INT4 quantization

`PALETTIZE_GROUP_SIZE` ∈ {16, 32 (current), 64, 128}. Smaller = more
buckets = less quant noise but bigger weights. Memory `project_stage1_w4a8_closed`
says W4A8 closed, but **W4 group_size 16** might help K cache
fidelity → drafter accept.

Cost: full chunks rebuild × 4. ~30 min Python.

### L33. AWQ on chunk1 (NOT chunk2)

`project_awq_nogo_e2b` says chunk2 alone no go. chunk1 (L0-7) is
where the drafter's first KV reads come from. AWQ on chunk1 might
shift drafter's prior. Untested.

Cost: AWQ smooth pass on chunk1 only + rebuild. ~1 hour.

### L34. INT8 LM head only (rest INT4)

LM head (chunk4 last Conv2d) has the most quant noise (cos 0.834 in
INT4 per memory). Keeping it INT8 raises chunk4 size by ~3 % but
improves numerical fidelity. Doesn't directly speed up, but may
improve drafter accept by sharpening target's argmax.

Cost: split palettize config + rebuild. ~30 min.

### L35. fp16 attention on full-attn layers only

Memory `project_iphone_ane_sparsity.md` notes iPhone padding-free for
sliding. Keep sliding INT4 but full-attn layers (L13/L14) fp16. May
improve K cache fidelity selectively.

Cost: palettize config tuning + rebuild. ~30 min.

### L36. Embed table INT8 → fp16

Embed table is q8. Going fp16 means doubled embed memory (~768 MB)
but cleaner first-layer signal. Could be win on iPhone with abundant
memory.

Cost: rebuild embed.bin. ~10 min.

### L37. Drafter group_size sweep

Drafter is centroid-quantized. Re-quantize with finer groups (g=16).
Cost: ~30 min.

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

### R3. Drafter cold→warm latency curve

Today saw drafter cold cycle = 81 ms (vs warm 1.8 ms). Plot per-cycle
drafter time for cycle 0 through 20 to characterise the ANE warmup
curve. May suggest whether multi-prompt warmup or longer single
warmup is optimal.

```bash
# Add `MTP_VERBOSE_SETUP=1` and `SPECULATIVE_PROFILE=1` to get per-
# cycle draft= timings; parse with awk for cycle index → draft ms.
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_PROMPTS": "code", "MTP_VERBOSE_SETUP": "1", "SPECULATIVE_PROFILE": "1"}' \
  com.example.CoreMLLLMChat > /tmp/R3_warmup_curve.log 2>&1
grep -oE 'mtp #[0-9]+\] draft=[0-9.]+ms' /tmp/R3_warmup_curve.log
```

### R4. ANE residency monitor

iOS has no direct ANE residency API. Indirect signal: chunk1 .prediction
wall time should be flat at ~6 ms; if it climbs to 20+ ms mid-bench,
ANE pages evicted (or thermal throttle, hard to distinguish).
Cross-reference with `[Thermal] state=` log lines.

### R5. Long-form steady-state bench

Currently bench at 256 tokens. Push to 1000 tokens to measure
sustained-session tok/s post-warmup, post-thermal escalation. Drives
realistic chat-bot UX estimate.

```bash
xcrun devicectl device process launch --device "$DEVICE" --console \
  --environment-variables '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_MAX_TOKENS": "1000", "LLM_AUTOBENCH_PROMPTS": "narrative"}' \
  com.example.CoreMLLLMChat > /tmp/R5_long.log 2>&1
```

### R6. Per-chunk timing breakdown profiling

Mac smoke shows `c1=5.5 c2=6.8 c3=7.5 c4=10.5` decode breakdown.
iPhone should print same — verify which chunk dominates iPhone-side.
If chunk4 (LM head) dominates, focus on chunk4 tuning (subset L12,
INT8 LM head L34). If chunk1 dominates, sliding cache reduction
(L15) might pay off.

### R7. Drafter 38 MB vs 149 MB iPhone A/B (deferred from L12)

Mac proved centroid 149 MB wins. iPhone empirically untested
directly. ANE memory pressure may shift the trade-off.

### R8. Battery / energy per token

`hostInfo()` energy counters or `os_log` energy traces. Per-token
energy cost reveals whether MTP-on is net energy-positive (more
tokens emitted per ANE cycle) or net-negative (drafter cost without
acceptance).

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

## iOS system-level levers (L38-L42, advanced)

### L38. Game Mode trigger

iOS 17+ Game Mode boosts ANE / GPU clocks at the cost of background
process priority. Could expose via `CHGameMode` API or
`requestThermalState`. Untested for LLM workload.

### L39. Background ANE keep-alive

Run a 1-token dummy decode every 30 s in background to keep ANE
residency warm between user turns. Trades battery for snappier
first-token TTFT in subsequent turns.

### L40. UIApplication.isIdleTimerDisabled audit

Already set in ChatView.swift:539. Confirm AutoBench inherits this
(prevents screen sleep during long bench).

### L41. Pre-load mlmodelc pages via mmap_advise

CoreML lazy-loads weight pages on first inference. Pre-fault all
weights via `mmap` advise(WILLNEED) at load time. Could eliminate
cold-start residency miss spike (R3 above).

### L42. RTKit thread priority bump

Promote decode loop to QoS+thread priority both. Currently QoS is
configurable (L16/L17); pthread_setschedparam not used.

## Production-quality levers (L43-L46, ship-grade)

### L43. Auto-detect prompt class

Heuristic: tokenize prompt, count code-like tokens (brackets,
keywords) vs natural language. Switch K_USE / FLy K / fallback
threshold based on class.

* Narrative → K_USE=2, FLy K=8 (less lossy), threshold 0.30
* Code → K_USE=1, FLy K=24 (more permissive), threshold 0.20
* List → K_USE=2, FLy K=16, threshold 0.25

~100 lines Swift. Significant UX win if validated.

### L44. Per-user adaptive K_USE memory

Track per-conversation accept-rate history. Adapt K_USE / FLy K to
user's prompt distribution. Stored in UserDefaults.

### L45. Streaming output post-processing

After tok/s win confirmed, address output quality issues seen with
FLy K=16 ("def def def" loops). Either reduce K when streak detected,
or filter duplicate tokens.

### L46. Two-rate decode

Slow start (first 16 tokens at K_USE=2 for sample-quality assurance)
then accelerate (K_USE=1 + FLy K=24 for sustained throughput). Per
prompt response.

## Tomorrow's recommended order (updated with L16+)

### Tier 1 — 15-minute info-per-bench (env-only, cool iPhone)

Each launch = 1 cool bench. **Manually 10 min cool between each.**

1. **L1** — K_USE=1 + warmup retest (reproducibility)
2. **L4** — PLD prefetch on code
3. **L18** — MARS lossy verify (quality-gated)
4. **L20** — Lookahead engine A/B (drafter-free path)
5. **L16** — LLM_DECODE_QOS=high (burst speed)

Picks the lever that's most "binary" — either fixes the
reproducibility issue or directly stacks tok/s.

### Tier 2 — 30-minute sweeps (multi-value, cool)

6. **L2** — FLy K sweep (8/16/24/32)
7. **L3** — Fallback threshold sweep (0.20-0.40)

### Tier 3 — bundle / build levers (1-2 hour)

8. **L9** — Stateful Linear path A/B (separate codepath)
9. **L11** — Custom MLFeatureProvider Swift change
10. **L10** — Per-prompt K_USE adapter Swift change

### Tier 4 — Python rebuild levers (half-day each)

11. **L13** — Drafter K=2 rebuild
12. **L32** — group_size sweep on chunks
13. **L33** — AWQ on chunk1 only
14. **L34** — INT8 LM head only

### Tier 5 — research / probing (no immediate tok/s)

15. **R3** — Cold→warm drafter latency curve
16. **R5** — Long-form 1000-token steady-state
17. **R6** — Per-chunk timing breakdown
18. **R8** — Energy per token

### Tier 6 — production polish

19. **L43** — Auto-detect prompt class
20. **L46** — Two-rate decode

### Time budget chooser

* **15 min**: only L1 (reproducibility check).
* **30 min**: L1 + L4 (PLD on code).
* **60 min**: L1 + L4 + L2 (FLy K sweep) + L18 (MARS quality probe).
* **2 hours**: All of Tier 1 + L9 stateful Linear push + bench.
* **4 hours / Sunday morning session**: above + Tier 3 Swift changes.

### Never do in one sitting

* > 4 timed prompts per launch — thermal=serious by 5th.
* > 6 launches per hour — cool gap floor is 8-10 min.
* Sweep + manual single-prompt bench back-to-back without cool gap.
* Tier 4 Python rebuild within 30 min of a hot iPhone (rebuild
  uses Mac, then needs iPhone push + cool bench — pad cool gap).

## Combination ideas (stacked levers, advanced)

Beyond individual sweep values, some combos are worth one shot each:

| combo | hypothesis | command |
|---|---|---|
| L1 + L4 | warm K=1 + PLD on code | `{"MTP_K_USE": "1", "MTP_PLD_PREFETCH_ENABLE": "1"}` |
| L2:24 + L1 | K=1 + FLy K=24 | `{"MTP_K_USE": "1", "MTP_FLY_TOPK": "24"}` |
| L3:0.20 + L1 narrative | K=1 narrative w/ low bail | `{"MTP_K_USE": "1", "MTP_FALLBACK_THRESHOLD": "0.20"}` |
| L16 + L18 | high QoS + lossy MARS | `{"LLM_DECODE_QOS": "high", "MTP_MARS_ENABLE": "1"}` |

## Summary lever count

* **Env-only**: L1-L8 (today's playbook) + L16-L25 (this update) +
  L38-L42 (system-level) = **23 env knobs ready to sweep**.
* **Swift change**: L10-L12, L26-L31, L43-L46 = **13 implementation
  candidates** with risk + LOC notes.
* **Python rebuild**: L13-L15, L32-L37 = **9 model rebuild
  candidates** with cost notes.
* **Research**: R1-R8 = **8 probing tasks**.

Total candidates loaded: **53**. With memory cross-referencing,
no overlap with the 15+ already-tried levers.
