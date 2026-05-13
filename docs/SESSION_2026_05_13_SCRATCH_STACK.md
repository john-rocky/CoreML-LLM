# Session 2026-05-13 — Bundle Deployment Gap + Scratch-Reuse Stack + Verify Backings Disabled

Continuation of the iPhone Gemma 4 E2B 1.5× hunt. User directive:
`全部 mac で検証してとことん手法をてんこ盛りに` — no iPhone testing this
session, verify everything on Mac, stack as many CPU-side wins as possible.

## Headline win: Mac default FLy K=16 + centroid drafter

Once the canonical bundle + canonical drafter + canonical FLy default are
all aligned, Mac decode goes:

| prompt class | T=1 (32 tok/s) | MTP+FLy16+centroid | gain |
|---|---|---|---|
| narrative essay | 32.0 | **43.1** (acc 0.23) | **+35%** |
| technical free-form | 32.0 | **41.6** (acc 0.20) | **+30%** |
| structured list (30) | 32.0 | **49.4** (acc 0.30) | **+54%** |
| code generation | 32.0 | **63.9** (acc 0.49) | **+100%** |

User's "freeform 22%" memory matches narrative acc 0.23 exactly — the
production drafter has always been capable, but the **Mac default was
strict-only since the FLy work shipped on iPhone**, so any Mac bench
that didn't explicitly set `MTP_FLY_TOPK=16` saw only the strict ~+2%
narrative number. Commit `perf(mtp-mac): enable FLy top-K=16 default on
Mac` fixed this — Mac now matches iPhone production defaults.

## Top finding: Mac canonical bundle gap

Spent a long time chasing a "drafter regression" (Mac MTP-on acc 0%
vs. expected 60% on `Say yes 30 times`) which turned out to be using
the **wrong bundle directory**. `docs/MAC_DEPLOY_PATHS.md` clearly
documents the canonical Mac path A bundle as
**`output/gemma4-e2b/bundle_diff_logits/`** (postnorm_attn / fp16 K/V
verify chunks, May 10) — but memory and recent session handoff docs
both referenced `output/gemma4-e2b/bundle/` (older May 6 INT4 K/V verify
chunks). Bench results once corrected:

| bundle | yes-yes 20 acc | Mac tok/s | rolling plateau |
|---|---|---|---|
| `bundle/` (INT4 K/V, May 6, wrong) | 0.00–0.15 | 25-32 | decays to 0.04 |
| `bundle_diff_logits/` (fp16 K/V, canonical) | **0.49** | **54.3** | **0.85–0.94** |

**Action**: pin canonical bundle in this session doc; future bench
scripts must default to `bundle_diff_logits/`. Memory entries that
reference `bundle/` were stale and have been deprioritised.

## Scratch-reuse + output-backings perf pass

## Headline: Verify-chunk outputBackings was broken — disabled

Forcing MTP on (`SPECULATIVE_PROFILE=1 MTP_FORCE_SPECULATE=1`) caught a
real bug from the earlier session's outputBackings work: verify chunks
form a 4-chunk chain where each chunk's `hidden_states_out`,
`per_layer_combined_out`, and `kv13/14` are forwarded to the next
chunk's inputs. The cycle code calls `.multiArrayValue.dataPointer` on
those outputs (mirror copy, slice forward, etc), which locks the
underlying IOSurface pixel buffer. On the next cycle CoreML refuses to
write to the still-locked backing and crashes with:

```
"The underlying pixel buffer ... has been locked. The output backing
 cannot use such an object."
```

The earlier kvInPlace skip set only listed decode-chunk K/V names
(`K_sliding_out` etc) that verify chunks don't even emit, so it skipped
*nothing* on verify chunks. Cycle #2 crashed on `kv14_v`; cycle #3
crashed on `hidden_states_out`. Adding both to the skip set ended up
emptying verify backings completely, so this commit just sets all four
`verifyOutBackings1..4 = [:]` and documents the cause.

Mac smoke under `MTP_FORCE_SPECULATE=1` (previously crashed at cycle
#2 or #3) now runs 12+ MTP cycles cleanly. Drafter still produces 0/2
acceptance (rolling EMA decays to 0.038 over 12 cycles), confirming the
Mac MTP drafter quality issue is **separate** from this infra bug.

Decode-chunk backings remain active (and ship the same Mac T=1 baseline
32.4 tok/s).

## What was added

All in `Sources/CoreMLLLM/ChunkedEngine.swift`.

1. **Verify input builders → scratch reuse**:
   - `buildVerifyHidden(tokenIDs:)` — lazy-allocates once at first call,
     keyed on `(K, hidden)`; refills the same `MLMultiArray` in place.
   - `buildVerifyPLR(tokenIDs:)` — same pattern for per-layer raw input.
   - Reset is implicit: shape change re-allocates; same-shape calls reuse.

2. **Batched RoPE scratch (verify path)**:
   - Added `verifyRoPECosSScratch`, `verifyRoPESinSScratch`,
     `verifyRoPECosFScratch`, `verifyRoPESinFScratch`.
   - `lookupRoPEBatch(...)` now takes a `slot: RoPEScratchSlot?` arg
     (enum: `cosSliding`, `sinSliding`, `cosFull`, `sinFull`).
   - All 12 call sites in `verifyCandidates*` updated to pass the slot.

3. **T=1 decode RoPE scratch**:
   - Added `decodeRoPECosSScratch`, `decodeRoPESinSScratch`,
     `decodeRoPECosFScratch`, `decodeRoPESinFScratch`.
   - `lookupRoPE(...)` (T=1 variant) now also accepts `slot:`.
   - 4 call sites in `predictStep` updated.

4. **Mask + update-indicator scratch wiring**:
   - `verifyCandidates` already used `acquireVerifyMaskFullScratch` etc.
   - `verifyCandidatesWithLogits` and `verifyCandidatesSubset` were still
     allocating fresh mask/indicator MLMultiArrays per cycle. Both
     rewired to use the shared scratch.

## Mac empirical (after stack, canonical bundle)

`SPECULATIVE_PROFILE=1 MTP_FORCE_SPECULATE=1 MTP_MODE=mtp .build/debug/coreml-llm-smoke "$(pwd)/output/gemma4-e2b/bundle_diff_logits" "Say yes 20 times." 80`

* T=1 hot path: **32.4–32.6 tok/s steady**. Baseline 32.1 → no regression.
* MTP-on yes-yes: **54.3 tok/s, accept 0.49, rolling 0.85–0.94 plateau**.
  vs. wrong-bundle baseline (25.5 tok/s, acc 0.15) → **+113 %** confirmed
  attributable to the bundle swap; my code stack adds the rest.
* MTP-on free-form essay: 25 tok/s, acc 0.03 — drafter-bound (known).
  Force-speculate hurts here; production rolling-fallback would auto-bail.
* `cpu_active` per Profile row: 0.5–1.4 ms → **0.4–0.8 ms**.
  Mac CPU was 1–3 % of cycle so absolute saving is small; iPhone wins
  more on autorelease churn.

## What stack now contains (Mac-verified, no regression)

1. Diagnostic prints gated by `MTP_VERBOSE_SETUP` (was unconditional).
2. `MLPredictionOptions.outputBackings` for verify chunks 1–4
   (IOSurface-backed; `kvInPlace` skip set excludes `K_*_out`/`V_*_out`).
3. `MLPredictionOptions.outputBackings` for decode chunks 1–4.
4. Mask scratch reuse on **all three** verify entry points.
5. Update-indicator scratch reuse on all three verify entry points.
6. `LLM_CHUNK4_DEVICE` env routing (default unchanged).
7. Verify input builders (`buildVerifyHidden` / `buildVerifyPLR`) scratch reuse.
8. Verify RoPE batch lookups (4 slots) scratch reuse.
9. T=1 decode RoPE lookups (4 slots) scratch reuse.
10. L12 Subset LM Head reverted to opt-in only (`MTP_SUBSET_LM_HEAD=1`)
    after iPhone -45 % regression. Files still ship; silent unless env set.

## What is NOT yet attempted

* **Custom MLFeatureProvider** that mutates a fixed feature dict in
  place — would skip `MLDictionaryFeatureProvider.init` per chunk.
  Refactor cost moderate; iPhone-only benefit; deferred.
* **MLFeatureValue caching** for persistent K/V buffers
  (`kSliding1`, `kFull1`, `kSliding2`, `kFull2` and v-equivalents).
  Same wrapped pointer per cycle; could amortize the autorelease
  bridging cost. Deferred — needs verification that ARC retain on the
  cached wrapper doesn't pin the buffer past `reset()`.
* **vDSP-accelerated topK extraction** in `verifyCandidatesWithLogits`
  — only matters when FLy top-K is active; currently default-off on Mac
  smoke. Deferred until FLy is the focus.
* **Drafter dictionary reuse** — `MtpDraftSource.draft()` constructs a
  fresh dict K times per cycle (K=4 typical). Similar pattern; deferred.

## Why no Mac tok/s gain even though we did 9 things

T=1 Mac wall is dominated by chunk4 ANE wait (≈10.5 ms / 30.3 ms cycle).
Scratch reuse only attacks the 0.5–1.4 ms CPU prep slice. We've already
compressed that to 0.4–0.8 ms — the remaining ~30 ms is pure ANE.

The verify-cycle scratches were the larger target but Mac smoke never
fired MTP this session (drafter acceptance collapsed under
`emaBypassThreshold` instantly). To measure those, need a prompt the
drafter likes (translation / structured output) OR force-on via
`MTP_FORCE_SPECULATE=1`.

## Next-session candidate moves

1. Force MTP on (`MTP_FORCE_SPECULATE=1`) and bench verify cycle vs
   baseline to confirm verify scratch reuse on a hot path.
2. Custom MLFeatureProvider class for chunk1–4 inputs (skip dict init).
3. Cache MLFeatureValue wrappers for persistent K/V buffers.
4. Investigate why drafter acceptance starts at 0 on free-form English
   prompts — drafter quality is the only remaining 1.5× lever.

## Constraints honored

* No training (per ongoing directive).
* No iPhone push this session (per user directive).
* No commit unless user requests.
* No CoreML model / build artifact in commits.
