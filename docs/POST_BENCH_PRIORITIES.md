# Post-Bench Priority Reassessment

**Updated:** 2026-04-12, after EAGLE-3 on-device (iPhone 17 Pro) bench.

Written to overrule the pre-bench ranking in `UNEXPLORED_APPROACHES.md`. The
pre-bench ranking assumed EAGLE-3 would reach Colab's projected 64 tok/s at
2K. On-device measurement contradicts that: with our current target
implementation, EAGLE-3 delivers **no speedup** because of two independent
blockers (see `EAGLE3_INTEGRATION_STATE.md` §Phase 3 for full details):

1. **Distribution mismatch**: draft trained against HF `Gemma4ForConditionalGeneration`;
   we deploy our custom `Gemma4Model`. On real prompts the two produce different
   hidden taps → draft acceptance rate is ~0 (observed avg N=2.07, vs Colab
   E[N]=3.05). Fixing requires retraining the draft on data collected from
   *our* custom forward (~3 h Colab + new data collection script).
2. **commit-decode overhead**: `ChunkedEngine.commitAccepted` re-runs
   `predictStep` per accepted token, so even a perfect draft cannot beat
   baseline. Fixing requires Phase 2A v2 verify chunks (K/V + hidden outputs
   already built in Python, not yet deployed) plus a Swift cache-writer
   implementation (~2–3 h).

Both blockers must be fixed together. Fixing only one leaves the speculative
path at or below baseline 28 tok/s.

---

## What was measured

| Metric (iPhone 17 Pro, 2K ctx, Gemma 4 E2B) | Observed |
|---|---:|
| Baseline T=1 decode (post-ANE-warmup) | **28.6 tok/s** |
| Baseline prefill (estimated from code path, not yet measured on-device) | ~150 tok/s |
| TTFT for 2K prompt (estimated) | ~13 s |
| verify call at T=3 | 31.5 ms |
| Single decode step | 33–36 ms |
| EAGLE-3 speculative effective throughput (current impl + broken draft) | 11–17 tok/s |
| EAGLE-3 projected if fully fixed | 40–44 tok/s (1.45–1.57× baseline) |

The decode-path ceiling on Gemma 4 E2B given the current conversion pipeline
is therefore **~45 tok/s at 2K** even with full EAGLE-3 investment. Further
decode wins require either a smaller target (Turbo SKU) or fundamentally
different architecture work (W8A8, sliding-only, etc. — see
`ALTERNATIVE_APPROACHES.md`).

TTFT on the other hand is **~13 s untouched** — the biggest user-visible pain
point, and one that no decode optimization addresses.

---

## Revised priority, in order

### Tier 1 — high-confidence, TTFT-focused, composable with everything

| # | Approach | Effort | Expected win | Quality risk |
|---|---|---:|---|---|
| 1 | **D. Vocab pruning (262k → ~50k)** | ~1 day | **-1.7 GB download**, lookup cost ↓ | <1% at keep ≥ 50k |
| 2 | **A. GPU prefill on A19 Pro tensor cores** | 1–2 days | **TTFT 13 s → ~5 s** (-60%) | none |
| 3 | **E. Persistent prefix KV caching** | ~1 day Swift | **4–35× TTFT on cache hit** | none |
| 4 | **F. MIL graph optimization pass** | ~1 day | op-count -20–40%, decode +1–5%, compile faster | none (verified equivalent) |

Rationale: this entire tier lands **without touching the trained artifacts**.
It addresses the dominant UX cost (cold-start latency) and the dominant ship
cost (download size). No retraining, no draft / verify rebuilds, no on-device
debug cycles that depend on Colab iteration. These four alone deliver the
biggest perceived speedup and the biggest perceived download-size win.

Recommended order within the tier:
1. **D first** — simplest, ship-size win is the most visible.
2. **F in parallel** — can be done while D's eval runs; reverts cleanly if a
   graph pass breaks something.
3. **A after D** — needs freshly-compiled prefill mlpackages; easier once
   the pruned-vocab pipeline is the source of truth.
4. **E last in the tier** — biggest code-path change (Swift side); benefits
   multiply with A.

After Tier 1: **TTFT ~5 s (first use) / ~0.3 s (cached prefix), download ~1 GB,
decode 28 tok/s.** That's the "ship it" state.

### Tier 2 — 8K-context quality, only if long-context is a product goal

| # | Approach | Effort | Expected win |
|---|---|---:|---|
| 5 | **C. Cascading KV Cache** | 2–3 days | 8K quality preserved without fine-tune |

Only relevant if/when we enable 8K ctx as a default. For 2K default it is
irrelevant. Postpone until a 8K-mode product decision is made. Preferable to
StreamingLLM+QLoRA because it's training-free, provided Gemma 4 plays as well
with cascading as Llama-2 does in the original paper.

### Tier 3 — EAGLE-3 follow-through (only if Tier 1 UX isn't enough)

**Entry cost is high**, payoff is modest on this decoder's architecture.
Proceed only if decode tok/s is the binding constraint after Tier 1 ships.

| # | Sub-step | Effort | Expected win |
|---|---|---:|---|
| 6a | Retrain draft against our custom target | 3–4 h | raises acc to ~Colab baseline (E[N]=3.05 projected) |
| 6b | Phase 2A v2 verify deploy + Swift K/V direct-write commit | 2–3 h | lets 6a translate to tok/s |
| 6c | **B. Mirror Speculative Decoding v1** (draft → A19 Pro GPU) | 2 days | +8–15% on top of 6a+6b |
| 6d | Mirror v2 (cross-burst pipelining) | 3–5 days | +30% over EAGLE-3 |

With 6a + 6b shipped: decode **40 tok/s** (1.45× baseline). Add 6c: **44
tok/s** (1.57×). 6d is research-tier effort for incremental 10–20% — not
worth until the rest is shipped.

**Critical**: 6a and 6b are a single indivisible unit. Shipping 6a without
6b leaves speculative slower than baseline (Phase 3 bench proved this).
Shipping 6b without 6a has the draft accepting nothing, so no benefit.

6a data-collection footgun: `conversion/collect_eagle_hidden_states.py`
currently uses `target.model(input_ids=..., output_hidden_states=True)` on
HF. To train a draft that matches our on-device target, it must instead run
the full pipeline through our `Gemma4Model` + `SWAChunk1..4` (or equivalent
monolithic forward) and emit `hidden_at_L{8,17,34}` from those — requires a
new collection script. See `EAGLE3_INTEGRATION_STATE.md` §Phase 3.

### Tier 4 — architectural swaps (separate product, not this SKU)

From `ALTERNATIVE_APPROACHES.md`:
- #1 Distill Gemma 4 → 1B (Turbo SKU; needs $500-1k A100)
- #5a Distill Gemma 4 → pretrained small student
- W8A8 end-to-end if not already shipped (see `SPEED_8K.md` §3 P1)

These are genuinely different products. Keep on the long-term roadmap, don't
expect payoff inside the current iteration.

---

## Final recommendation

**Build and ship Tier 1 next. Tier 3 can wait until after Tier 1 tells us
how much more speed the product needs.**

Concretely: do **D + F + A + E** in that order. That's roughly one person-
week for a large and tangible end-user improvement. At the end of it,
decode is 28 tok/s (unchanged), download is ~1 GB (was 2.7 GB), and first-
token wait is 5 s / 0.3 s (was 13 s). Most users care about that tradeoff
more than about raw decode tok/s in steady-state.

After Tier 1 ships, re-measure whether decode tok/s is actually the
binding constraint for user workflows. If it is, pick up Tier 3 (6a + 6b
together, then optionally 6c). If long-context becomes a product goal,
pick up Tier 2 (C).

---

## Links
- [EAGLE3_INTEGRATION_STATE.md](EAGLE3_INTEGRATION_STATE.md) — full Phase 3 bench report, diagnostic details
- [UNEXPLORED_APPROACHES.md](UNEXPLORED_APPROACHES.md) — pre-bench deep dive of A–F
- [ALTERNATIVE_APPROACHES.md](ALTERNATIVE_APPROACHES.md) — outside-Gemma-4 options
- [SPEED_8K.md](SPEED_8K.md) — 8K-context roadmap, W8A8 rationale
