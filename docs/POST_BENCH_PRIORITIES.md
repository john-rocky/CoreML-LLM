# Post-Bench Priority Reassessment

**Updated:** 2026-04-12, after EAGLE-3 on-device (iPhone 17 Pro) bench.
**Priority axis:** *model performance (decode tok/s, then quality) first; ship
size last.*

Written to overrule the pre-bench ranking in `UNEXPLORED_APPROACHES.md`. The
pre-bench ranking placed ship-size (vocab pruning) and TTFT wins on top. That
recommendation is deprioritized here — the user's product direction is that
**raw decode throughput and model quality are more important than download
size or first-token latency.**

---

## What was measured on iPhone 17 Pro (2K ctx, Gemma 4 E2B)

| Metric | Observed | Notes |
|---|---:|---|
| Baseline T=1 decode | **28.6 tok/s** | steady-state post-ANE-warmup |
| verify T=3 per call | 31.5 ms | ≈ cost of one decode step |
| Single decode step | 33–36 ms | |
| Avg accepted/burst | **2.00-2.07** | Colab projected E[N]=3.05 — gap is diagnostic-confirmed distribution mismatch |
| Speculative effective tok/s | 11-17 tok/s | burst overhead not yet amortized |

See `EAGLE3_INTEGRATION_STATE.md` §Phase 3 for root-cause analysis of the
acceptance gap (draft trained on HF Gemma-4 hiddens, on-device uses our
custom `Gemma4Model` hiddens — distribution OOD).

---

## Priority, ordered by decode/quality payoff

### Tier 1 — decode-speed wins, training-free or cheap calibration

Ordered roughly by expected gain per engineering hour.

| # | Approach | Expected gain @ 2K | Expected gain @ 8K | Effort | Source of claim |
|---|---|---|---|---:|---|
| 1 | **F. MIL graph optimization pass** | +1–5% | +1–5% | ~1 day | `UNEXPLORED_APPROACHES.md` §F |
| 2 | **P2 from SPEED_8K: pre-alloc masks + KV-share Q-batching + INT8 KV cache** | small–moderate | +30–50% (stacked) | 2–4 days | `SPEED_8K.md` §3 P2 |
| 3 | **W8A8 (INT8 activations + INT8 weights)** | **~1.3–1.6×** | **~1.5×** | already on `feature/w8a8-8k`, await merge | `SPEED_8K.md` §3 P4 |
| 4 | **TriForce** (block-level top-k, training-free) | — | **~2.3×** (15→33 solo) | 3–5 days | `SPEED_8K.md` §S1 |
| 5 | **DuoAttention** (retrieval vs streaming heads) | — | +40–80% (15→22–27) | ~1 week (includes training) | `SPEED_8K.md` §3 P3 |

Rationale for ordering within Tier 1:
1. **F first** — lowest cost, no quality risk, verified-equivalent flag in
   the scaffold. Low upside but high confidence.
2. **P2 second** — the training-free bandwidth-reduction triad is the
   fastest ROI for 8K compute; each step is testable in isolation.
3. **W8A8 third** — work in progress on a sibling branch; merge when the
   sibling lands. Potentially the single largest decode win on any ctx.
4. **TriForce fourth** — is training-free and composes with EAGLE-3. The
   ANE compile story is "block-level top-k with fixed budget," so it's
   feasible here even though the original paper is more dynamic.
5. **DuoAttention fifth** — bigger quality story than TriForce but needs a
   calibration run (4–8 h A100). Comparable decode win.

### Tier 2 — EAGLE-3 follow-through

**Requires fixing both blockers identified in Phase 3 bench to deliver any
speedup at all.** Neither alone is sufficient (proved by bench math, see
§Phase 3 of `EAGLE3_INTEGRATION_STATE.md`).

| # | Sub-step | Expected gain | Effort |
|---|---|---:|---:|
| 6a | Retrain draft against our custom `Gemma4Model` hiddens (not HF's) | required to raise acceptance from ~0 to Colab-projected E[N]=3.05 | 3–4 h Colab + new collection script |
| 6b | Deploy Phase 2A v2 verify chunks + Swift K/V direct-write commit | lets 6a translate to tok/s; without it, commit-decode overhead wipes out any draft gain | 2–3 h |
| 6c | **B. Mirror Speculative Decoding v1** (draft → A19 Pro GPU) | +8–15% on top of 6a+6b | 2 days (scaffold exists) |

Combined 6a+6b: **1.45× baseline** (28→40 tok/s projected). Add 6c: **1.57×**
(28→44 tok/s). Mirror v2 (cross-burst pipelining) adds another ~15% for
research-tier effort.

Footgun for 6a: `conversion/collect_eagle_hidden_states.py` uses HF's
`Gemma4ForConditionalGeneration`. To train a draft that matches our deployed
target, collection must run through our custom `Gemma4Model` + `SWAChunk1..4`
(or equivalent monolithic forward) and emit `hidden_at_L{8,17,34}` from
those. A new collection script is required.

### Tier 3 — long-context quality (only if 8K becomes a product goal)

| # | Approach | Effort | Expected benefit |
|---|---|---:|---|
| 7 | **C. Cascading KV Cache** | 2–3 days | preserves 8K quality without a fine-tune, paper reports +5.6% on LongBench |

Only relevant for the 8K SKU. For 2K default, postpone. Prefer over
StreamingLLM+QLoRA because it's training-free.

### Tier 4 — UX (TTFT, not decode)

Important for perceived latency but do not raise decode throughput once
generation starts. Schedule after decode-speed work unless the product story
explicitly prioritizes first-use latency.

| # | Approach | Expected gain |
|---|---|---|
| 8 | **A. GPU prefill on A19 Pro tensor cores** | TTFT 13 s → ~5 s (-60%) |
| 9 | **E. Persistent prefix KV cache** | 4–35× TTFT on cache hit |

### Tier 5 — download size (deferred until performance work settles)

| # | Approach | Caveat |
|---|---|---|
| 10 | **D. Vocabulary pruning (262k → 50-150k)** | quality claim is for English-heavy benchmarks; multilingual coverage trade-off is *very* real — see `UNEXPLORED_APPROACHES.md` §D and the table below |

**Vocab pruning multilingual footgun** (do NOT ship without picking a coverage policy):

| keep | corpus | EN quality | JP quality | Other languages |
|---|---|---|---|---|
| 50k | English-heavy (C4-en) | <1% loss | ~5% | **30–80% loss** on Korean, Arabic, Hindi, Thai, Vietnamese, Swahili etc. |
| 50k | mC4 (natural proportions) | <1% | ~3% | top-10 ~8%, rare 20–50% loss |
| 100k | mC4 | <1% | <2% | top-10 <2%, rare 5–20% |
| 150k | mC4 | <1% | <1% | top-30 <2%, rare ~3% |
| 200k+ | any | <1% | <1% | <1% | ← almost no savings left |

Honest bottom line: aggressive prune (keep=50k) is only safe for an
English+Japanese-only SKU. Multilingual SKU needs keep ≥ 100k, which drops
the download savings from "-1.7 GB" to "-1 GB." `QLoRA` recovery can claw
back some of the loss for keep=50-80k but adds training cost.

---

## Recommended sequencing under performance-first priority

1. **Immediate (this week)**:
   - #1 F MIL optim (try it, revert if any graph pass regresses)
   - #3 W8A8 (merge when sibling branch lands) in parallel

2. **Next (2–4 weeks)**:
   - #2 P2 bandwidth-reduction triad (pre-alloc, KV-share Q-batching, INT8 KV)
   - **THEN decide Tier 2 entry**: if W8A8 + P2 leave a decode gap worth
     closing, pick up #6a + #6b together (indivisible — see bench math).
     If decode is already satisfactory, park Tier 2.

3. **After baseline decode speed is landed**:
   - #4 TriForce or #5 DuoAttention for 8K ctx if long-context is in scope.
   - Otherwise #7 Cascading KV if quality at 8K is the binding constraint.

4. **UX polish** (do last within scope, before shipping):
   - #8 A GPU prefill — big TTFT win, no decode impact
   - #9 E Prefix KV cache — incremental on top of A

5. **Ship size** (after all the above):
   - #10 D Vocab pruning — only after a product decision on multilingual
     coverage. Publish per-language eval numbers alongside any pruned SKU.

---

## What this means for the decode-tok/s ceiling

Cumulative best case at 2K ctx (single SKU, current target architecture):

```
Baseline (Gemma-4 E2B, MQA, current chunks)           : 28 tok/s
+ F MIL optim                          × 1.03        →  29
+ W8A8                                 × 1.4          →  41
+ P2 (INT8 KV + Q-batch + pre-alloc)   × 1.15        →  47
+ EAGLE-3 full (6a+6b+6c)              × 1.55        →  73
+ Mirror v2 cross-burst                × 1.15        →  84
                                                        ^^
                                                        realistic max @ 2K
```

At 8K with TriForce or DuoAttention stacked on the above: **~50–80 tok/s**
depending on the 8K path chosen. Matches `SPEED_8K.md` §2's honest
projection.

Above that ceiling, only a **smaller target model** (Tier 4 of
`ALTERNATIVE_APPROACHES.md`: distill to 1B) unlocks more decode speed.

---

## Links
- [EAGLE3_INTEGRATION_STATE.md](EAGLE3_INTEGRATION_STATE.md) — full Phase 3 bench report
- [UNEXPLORED_APPROACHES.md](UNEXPLORED_APPROACHES.md) — pre-bench deep dive (A–F)
- [SPEED_8K.md](SPEED_8K.md) — 8K roadmap, W8A8 / DuoAttention / TriForce rationale
- [ALTERNATIVE_APPROACHES.md](ALTERNATIVE_APPROACHES.md) — outside-Gemma-4 options
