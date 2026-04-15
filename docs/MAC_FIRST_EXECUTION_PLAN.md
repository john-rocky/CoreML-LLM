# Mac-first execution plan (2026-04-14 revision)

> **Partial correction 2026-04-15 late (PR #62).** Phase A and the
> Mac-vs-iPhone authority split below are still authoritative and
> held up well. Phase B's exit criteria ("≥ 50 tok/s chat", "accept
> rate within ±5 % of A5 projection") and Phase C's C2 Union-of-
> drafters arithmetic were built on oracle-replay bench numbers
> that PR #62 invalidated. Task #2 (target-argmax bench rebuild) is
> the new immediate work before Phase B/C exit criteria can be set.
> See `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` for details.

**Status:** authoritative priority order as of 2026-04-14 late. Supersedes
the phase-count ordering in `PRIORITY_ROADMAP.md` for the next 2–3 weeks
of work; that doc remains the menu, this one is the itinerary.

**Why this revision exists:** today's work hit two on-device regressions
(PR #17 ANE micro-opts, PR #33 prefill bypass) despite Mac smoke tests
passing. Net cost was ~10 device trips for two rejected changes and no
tok/s gain. The core problem: iPhone device iteration is the scarce
resource and we weren't maximizing the information per trip.

This plan enforces:

1. **Mac-side pre-validation.** Every candidate is run through an
   offline measurement to estimate gain / confirm correctness before
   burning a device trip. Mac is a development tool — not the product
   target.
2. **iPhone is the authoritative speed benchmark.** The product claim
   is "ANE-optimized LLM on iPhone"; headline tok/s numbers in
   README / release notes / this doc must be measured on iPhone. Mac
   numbers are proxies used to rank candidates and catch correctness
   regressions, never as the shipping number.
3. **Bundled device trips.** When the iPhone does come out, multiple
   unrelated data points are collected in one session.
4. **ROI-ranked execution.** Start with items that deliver the most per
   day, stop when the competitive target (> 56 tok/s @ 2K, Google's
   LiteRT-LM iOS) is comfortably exceeded on device.

### What each environment is authoritative for

| Quantity | Mac can answer? | iPhone required? |
|---|---|---|
| Correctness / numerical bit-exactness | ✅ (fp16 deterministic) | no |
| Accept rate for speculative drafters | ✅ (drafter proposals and target argmax are device-independent) | no |
| Model load sequence, file layout, IO | ✅ | no |
| Wall-clock chunk latencies (ms/step) | proxy only | **yes** |
| Final tok/s for README / release | no | **yes** |
| MLComputePlan ANE placement ratio | proxy only (Mac ANE ≠ iPhone ANE ops) | **yes** |
| TTFT with real prefill chunks | proxy only | **yes** |

---

## Phases

### Phase A — Mac Studio only, 3–5 days, **zero device trips**

Build offline evaluation that tells us which Route B drafter actually
delivers on our weights. Accept rate is the main independent variable;
everything else (drafter latency, verify cost, union gain) can be
derived.

| # | Item | Output |
|---|---|---|
| **A1** | Build offline accept-rate harness. Load the existing multi-function `iphone_8k/chunk{1-4}.mlpackage` (`verify_qK` + `decode_q1`, K=3 already present) via Python coremltools with `computeUnits=cpuAndGPU` on Mac Studio. Implement the verify loop identically to `ChunkedEngine.verifyCandidates`, plus a pluggable drafter source interface. | Script: `eval/accept_rate_harness.py` or similar, plus a prompt corpus. |
| **A2** | Measure Prompt Lookup Decoding accept rate across 4 prompt categories × 10 prompts (code, chat, summary, QA). | Accept-rate histogram per category. |
| **A3** | Measure SuffixDecoding accept rate with a multi-turn session simulator. | Same format. |
| **A4** | Measure Cross-vocab Qwen 2.5 0.5B drafter accept rate. Includes writing the vocab-translation layer (lookup table) since that's a prerequisite. | Same format. |
| **A5** | **Decision**: pick the winning drafter from A2–A4 by `(accept_rate × (1 − drafter_cost_ratio))`. Compute theoretical upper bound tok/s on device given measured chunk latencies from today's baseline (c1=5.9, c2=6.8, c3=8.1, c4=10.4 ms). If no drafter beats 1.5× on any workload, pivot to Harmony-Decoding (V6-8) or revisit assumptions. | A picked drafter, a theoretical tok/s estimate, a go/no-go on device wiring. |

Exit criterion for Phase A: one drafter choice with measured accept
rate, theoretical tok/s upper bound on iPhone, and a written prediction
we can compare against device reality in Phase B.

### Phase B — one bundled device trip, ≤1 day

Validate Phase A's theoretical prediction and collect four data points
in a single session.

| # | Item | Mac-side prep | Device observation |
|---|---|---|---|
| **B1** | Runtime hints V6-1 (`reshapeFrequency = .infrequent`) + V6-2 (`MLComputePlan` warm-pool). Numerically identical to baseline. | Edit `ChunkedEngine.load`, build on Mac. | Decode tok/s delta (expect +1–2%), no bit-exact regression. |
| **B2** | Blockwise-32 W4 palettization. | Reconvert chunks with `granularity="per_block", block_size=32"` via `conversion/build_*.py`, push to iPhone over USB using `push-model.sh`. | Quality spot-check only (same tok/s expected). |
| **B3** | ~~Wire winning drafter from A5 — DONE via PR #54's DrafterUnion orchestrator on `Sources/CoreMLLLM/DrafterUnion.swift`. Enable with `drafterUnionEnabled = true` after the perf investigation resolves the iPhone regression.~~ PR #54 landed with `drafterUnionEnabled = false` default; PR #62 showed Union regresses (15–21 tok/s Mac vs 32 baseline). Union shape pending re-decide after task #2. | Opt-in flag; no new Swift build needed. | ~~iPhone accept rate vs Mac prediction~~ Superseded — see PR #62. |
| **B4** | Fresh MLComputePlan audit on the hints-applied chunks. | — | Confirm ANE placement still ≥ 99%. Watch for hint-induced fallback. |

Exit criterion for Phase B (revised 2026-04-15 late, post-PR #62):

Prerequisites (before any tok/s target is set):

1. **Task #2 — target-argmax replay mode** for `accept-rate-bench`.
   Produces `eval/accept-rate-bench-v3-target-argmax.json` with
   live-equivalent accept rates per (drafter, category). Blocker
   for setting a realistic Union tok/s target.

After task #2 lands, the phase exit is re-defined based on v3
numbers. Previous exit criteria below are retained for history but
should not drive work:

~~1. Matched-prefix bookkeeping bit-exact vs serial decode on Mac.
   Still valid as a correctness gate (PR #54's
   `union-bitexact --mode fallback-only`).~~ (Still valid —
   unchanged by PR #62.)

~~2. On-device accept rate within ±5 % of per-category Mac
   projection in `docs/PHASE_A5_DECISION.md`.~~ Superseded —
   A5 projections were oracle-replay and over-claimed 3–9×.

~~3. Manual quality spot-check on 5 prompts per category.~~
   (Still valid.)

~~4. Reproducible chat tok/s ≥ 50 at 2K.~~ Superseded — Union on
   Mac measured 17.8 tok/s on chat (baseline 32.7), so 50 tok/s
   was never going to be reached by Union alone. Re-target after
   task #2.

Item 11c (verify-chunk numerical alignment) is now deprioritised:
PR #62 showed the live-vs-bench gap is present on Mac CPU+GPU too,
so closing 11c would not change the live-equivalent accept rate
delta. 11c remains interesting for its own reasons but is not the
Phase B bottleneck.

### Phase C — infrastructure & stacking, 3–4 days + 1–2 device trips

| # | Item | Mac-validatable | Device check |
|---|---|---|---|
| **C1** | I3 KV direct-write in `commitAccepted`. Without this, every accepted speculative token costs one extra T=1 decode, erasing most of the Phase B gain. Today's code re-runs `predictStep` per accepted token (see `MtpSpeculativeEngine.swift` commit path and `EAGLE3_INTEGRATION_STATE.md` Blocker 2). | Bit-exact re-verification on Mac CoreML using the same prompts as A1's harness. | Final tok/s check. |
| **C2** | ~~Union-of-drafters (ROUTE_B_EXECUTION_PLAN Task 4). Add second- and third-place drafters from Phase A as fallback sources. Since A already measured each, union gain is arithmetic on hit rates (`1 − Π(1−accept_i)`).~~ Landed in PR #54 (default off). Shape pending re-decide after task #2 per PR #62. The "arithmetic on hit rates" shortcut assumed oracle-replay numbers and over-claimed. | Mac simulation with **live-equivalent** accept rates (task #2 v3 json). | Final tok/s. |
| **C3** | I1 async ANE dispatch queue (ROUTE_B_EXECUTION_PLAN Task 5). Decoupling chunk dispatches from strict serial order. | Mac CoreML doesn't reproduce ANE's async behavior exactly, but determinism + ping-pong design can be validated with synthetic timing. | Race detection + tok/s on real ANE. |

Exit criterion for Phase C: ~~≥ 70 tok/s on mixed chat (> Google by 25%).~~ TBD post-PR #62; re-set after task #2 produces live-equivalent ceiling estimates.

### Phase D — stretch, only if Phase C saturates

| # | Item | Pre-req |
|---|---|---|
| **D1** | Mirror SD (GPU drafter async, T2). | C3 async infra. |
| **D2** | Staged chunk pipelining (T4). | C3 async infra. |
| **D3** | MoE / RWKV / W2-rotated — out of 2K scope. | — |

Exit criterion: ≥ 100 tok/s @ 2K, decisive win over Google.

---

## Explicit deprioritizations

| Item | Why paused |
|---|---|
| PR #33 prefill bypass (draft) | 6× regression, root cause not isolated. Mac reproduction pending; no Phase A–C item depends on it. |
| 8K ctx optimizations (DuoAttention, SparQ, TransMLA) | Compete with 2K target budget. Re-open after 100 tok/s @ 2K lands. |
| Route A drafter training (MTP / HASS / EAGLE-3 retrain) | Owned by the other session; this plan doesn't block on it. When it lands, drops into Phase C2's union as an additional source. |
| ANE micro-opts (PR #17 items: MLP tile, GQA broadcast, exp2) | Rejected. Don't revive without an isolated ANE? audit identifying which specific op fell off. |

---

## How Mac-side accept rate measurement works

The verify chunks (`verify_qK`, K=3) are already multi-function in
`conversion/output/iphone_8k/chunk{1-4}.mlpackage` and
`~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b/chunk{1-4}.mlpackage`.
Both use the same shapes the device expects:

- `hidden_states`: (1, 3, 1536) fp16 — K=3 queries
- `per_layer_raw`: (1, 3, 8960) fp16
- returns target's argmax at each of the 3 positions

Drafter proposal → feed as K=3 tokens → target returns K=3 argmax → walk
and count matches = accept count. This runs identically on ANE, GPU, or
CPU: fp16 numerics agree within rounding, so **accept rate on Mac
coremltools is a faithful predictor of accept rate on device**. The
device-only variables are per-chunk wall-clock time (already measured
in baseline profile) and async-dispatch behaviour (Phase C3 only).

What this means for prioritisation: we can reject any drafter whose
Mac-measured accept rate fails to clear `1 − ε / (average chunk time
× K)` = roughly 0.3 for our shape, **without burning a single device
trip**.

---

## Task list mapping

For tracking, the phases map directly to the session's task list. Each
Phase X item becomes one task. Phase A tasks are the starting queue.

---

## Relationship to existing docs

- `PRIORITY_ROADMAP.md` — the exhaustive menu and rejection register.
  Still authoritative; this doc just reorders and gates the Phase 0–2
  items for the next work cycle.
- `ROUTE_B_EXECUTION_PLAN.md` — per-task implementation details for
  Route B. Phase B/C here references specific task numbers there.
- `UNEXPLORED_APPROACHES_V6.md` — lossless candidate list. This plan
  picks specific candidates; V6 stays as the larger pool.
- `MOBILE_2K_COMPETITIVE_PLAN.md` — strategic framing and targets.
  Unchanged; this doc is the ordered execution view of the same
  strategy with device-iteration discipline added.
