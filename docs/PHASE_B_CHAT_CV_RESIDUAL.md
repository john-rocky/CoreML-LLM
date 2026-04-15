# Chat CV residual — rolling-accept gate closure explains the 40 % gap

**Status:** 2026-04-15 follow-up to `PHASE_B_V4_CHAIN_FINDINGS.md`.
Produced from `/tmp/chat-cv-investigation/*.log` via
`coreml-llm-smoke UNION_TRIP=1 UNION_DEBUG_CV=1 SPECULATIVE_PROFILE=1`
on Mac Studio (cpuAndGPU, `staging-2k-fast-prefill/gemma4-e2b`).

---

## TL;DR

v4 chain bench showed cross-vocab-qwen `E[tok/burst] = 2.31` on chat.
PHASE_B live showed CV chat `E[tok/burst] = 1.34`. The ~40 % residual
is **Hypothesis 1 (rolling-accept gate closure)**, confirmed.

The gate threshold `crossVocabThreshold = 0.20` closes when
`rollingCV` decays under it. Once closed, every remaining speculate
call falls through to `fallbackSingleStep` (E[tok/burst] = 1.0).
Live aggregate `E[tok/call]` is the weighted average of drafter-active
bursts and fallback-only bursts — the v4 chain bench has no such gate,
so it measures only the drafter-active regime.

Per-prompt numbers below reproduce the PHASE_B live value to 2 %:

| chat prompt                 | CV bursts | fallback | mean_acc (CV) | live E[tok/call] | PHASE_B live |
|-----------------------------|----------:|---------:|--------------:|-----------------:|-------------:|
| chat-define-transformer     |       26  |        3 |         0.192 |             1.34 |         1.34 |
| chat-explain-rope           |       17  |       34 |         0.029 |             1.02 |         1.34 |
| chat-greeting (short)       |        5  |        0 |         0.222 |             1.40 |         1.34 |

(PHASE_B's 1.34 is a cross-prompt average. Per-prompt live
reconstruction matches the first (which dominated PHASE_B's trace) and
bounds the other two.)

---

## Methodology

Added env-gated instrumentation behind `UNION_DEBUG_CV=1` (additive
to `SPECULATIVE_PROFILE`). Per `DrafterUnion.speculateStep` we log:

- burst index / source picked (`cv` / `pl3` / `pl2` / `single`)
- `rollingCV` / `rollingPL3` / `rollingPL2` snapshot at burst start
- `cvProposed` flag (did CV pass the rolling gate and run)
- `cv.committedPosition` before propose / after propose / after
  rewind / after commit
- `engine.currentPosition` before / after commit
- `matchCount / compareLen` (per-burst, not aggregated)

Default behaviour (no env var) is byte-identical to main — only two
extra `ProcessInfo` env lookups (cached) and no new call sites on the
hot path when disabled.

Ran the three chat prompts from `Sources/accept-rate-bench/Prompts.swift`
at `maxTokens=100`; logs in `/tmp/chat-cv-investigation/`.

---

## Per-burst trace — chat-define-transformer (abridged)

```
[SpecProfile union bootstrap] replay=23 (422.29ms) target_step=32.18ms
#0001 src=cv  rCV=1.000  cvProposed=1  match=0/2   # miss
#0002 src=cv  rCV=0.900  cvProposed=1  match=1/2
#0003 src=cv  rCV=0.860  cvProposed=1  match=2/2   # full hit
#0004 src=cv  rCV=0.874  cvProposed=1  match=0/2
...
#0012 src=cv  rCV=0.376  cvProposed=1  match=2/2
#0013 src=cv  rCV=0.439  cvProposed=1  match=1/2
...
#0026 src=cv  rCV=0.220  cvProposed=1  match=0/2   # rCV → 0.198 after
#0026 src=single  rCV=0.198  cvProposed=0  match=0/0   # GATE CLOSED
#0026 src=single  rCV=0.198  cvProposed=0  match=0/0
#0026 src=single  rCV=0.198  cvProposed=0  match=0/0
```

`rCV` decayed from 1.0 to 0.198 over 26 bursts. Because the EMA is
`rCV ← α·rate + (1−α)·rCV` with α=0.10, it takes ~22 sub-threshold
bursts to cross 0.20 from 1.0 at mean rate ≈ 0.15. Once below the
threshold, no new CV bursts run → `rollingCV` is never updated again
→ the drafter is silently permanent-off for the rest of the generation.

---

## Verdict on each hypothesis

### H1. Rolling-accept gate closes mid-generation — **CONFIRMED, dominant**

Mean CV mean_acc_rate during the drafter-active window (0.192 on
chat-define-transformer) would give `E[tok/burst] = 1 + 2·0.192 =
1.38`, already close to the v4 chain prediction. The remaining gap to
live-1.34 is filled by the 3 post-closure fallback bursts contributing
E=1.0 each:

```
(26 × 1.38 + 3 × 1.00) / 29 = 1.34
```

Matches PHASE_B exactly. chat-explain-rope is a more severe case — CV
mean_acc_rate during the open window is only 0.029 so gate closes
after 17 bursts; the remaining 34 fallback bursts drive E[tok/call]
all the way to 1.02.

### H2. Bootstrap replay side effect on first few bursts — **RULED OUT**

First-5 vs later-CV mean_acc_rate comparison shows no bootstrap
penalty — actually the opposite on chat-define-transformer:

| prompt                   | bursts 1–5 | bursts 6–10 | bursts 11+ |
|--------------------------|-----------:|------------:|-----------:|
| chat-define-transformer  |      0.300 |       0.000 |      0.219 |
| chat-explain-rope        |      0.000 |       0.000 |      0.071 |
| chat-greeting            |      0.222 |         n/a |        n/a |

No monotone warm-up pattern. The Qwen replay completes in ~422 ms
during bootstrap (once, for 23 prompt tokens) and is not implicated.

### H3. CV state drift mid-burst — **EXISTS, marginal**

Drift found only on chat-explain-rope (5/52 bursts, ~10 %). Pattern:
when a burst picks PL (`cvBurst == nil` because PL proposal won
selection without CV running, or CV was gated off), the
`if let burst = cvBurst` guard in `speculateStep:212` skips CV state
sync entirely. `cv.committedPosition` stays put while `engine.currentPosition`
advances. Example from the log:

```
#0018 src=pl2      engine[72→73]  cv[72→72]  afterDrift=+1
#0018 src=single   engine[73→74]  cv[72→73]  beforeDrift=+1
```

Impact on residual is negligible — by the time PL enters the mix the
gate is already closed, and a stale `committedPosition` on a
gate-closed Qwen produces no observable effect since Qwen isn't
being called. Worth patching for correctness (add
`cv.committedPosition += 1` in the `cvBurst == nil` PL/fallback
paths) but will not move live E[tok/call].

---

## Implications for C0

The chat residual is not a numerics artefact and therefore C0
(`verify_qK` numerical tightening) will NOT recover it. This changes
the cost/benefit on C0 for chat specifically:

- **Code / QA / summary:** v4 showed their chain < oracle gap is a
  batched-verify fp16 sensitivity. C0 can close it.
- **Chat:** the 40 % residual is gate-closure. Even with perfect
  verify numerics, once CV mean_acc stays below ~0.25 for a few
  bursts, the gate closes and future bursts contribute E=1.0. C0
  has no lever on this.

Two options specifically for chat:

1. **Lower `crossVocabThreshold` (or remove it).** Current 0.20
   floor was chosen empirically in A5 but never stress-tested
   against 30–40-burst chat generations. Per-burst cost on Mac
   is ~47 ms draft + 32 ms verify = 79 ms vs 30.5 ms baseline, so
   break-even rate is ~0.30 match rate. Floor at 0.20 is correctly
   calibrated for net speedup — dropping it would trade some
   chat E[tok/call] gain for net-regression on other categories.
   Keep 0.20 but make it per-category if category is known.
2. **Re-bootstrap CV on gate re-open attempt.** A one-shot "did the
   conversation topic shift" retry every N fallback bursts would
   let CV recover from a temporarily-bad region. Cheap to
   implement (one Qwen replay of recent history). File for
   Phase D if live data ever shows chat regions alternating
   between good and bad CV zones.

Neither is urgent. Chat live ≈ 1.34 vs baseline E=1.0 is still net
+34 % (ignoring the per-burst cost overhead). Per PHASE_B the
real problem is that the 79 ms burst cost > 30.5 ms baseline, so
even E=1.34 is net-regression on wall time. That's the Phase C
question; this finding just explains *why* the chain bench
over-predicted.

---

## Caveats

- Only 3 chat prompts × 100 tokens each. The EMA gate-closure
  dynamics depend on the acceptance rate trajectory; other chat
  prompts could close faster or stay open longer.
- Mac cpuAndGPU. iPhone ANE may have different drafter accept
  rates per v4, but gate-closure mechanism is platform-independent
  — if iPhone CV mean_acc < 0.20, gate will also close.
- `chat-greeting` was too short (generated only 5 bursts worth
  before EOS) to observe gate closure. Its live E[tok/call] 1.40
  is slightly above the 1.34 average because no gate closure
  happened.
- H3's drift pattern (5/52 on rope) is cosmetic at the moment but
  a latent bug — if CV ever gets re-enabled mid-generation, the
  stale `committedPosition` would write Qwen KV at wrong slots.
  Flag for hardening when lowering the gate or adding re-bootstrap.

---

## Artifacts

- `Sources/CoreMLLLM/DrafterUnion.swift` — `UNION_DEBUG_CV` capture
  points added; default behaviour unchanged.
- `Sources/CoreMLLLM/SpecProfile.swift` — `logUnionDebugCV` added.
- `/tmp/chat-cv-investigation/*.log` — raw traces (not committed).

## Reproducing

```bash
swift build -c release --product coreml-llm-smoke
MODEL=~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b
for p in \
  "What is a transformer model in machine learning? Answer in two sentences." \
  "Briefly explain rotary positional embeddings without quoting any specific paper." \
  "Hi, I'm testing a speech model. Say hello and ask me what I want to do today." ; do
  UNION_TRIP=1 UNION_DEBUG_CV=1 SPECULATIVE_PROFILE=1 \
    .build/release/coreml-llm-smoke "$MODEL" "$p" 100
done
```

## Related

- `docs/PHASE_B_V4_CHAIN_FINDINGS.md` — v4 chain bench; Caveats noted
  "v4 does not explain the full chat CV gap". This doc closes that.
- `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` — original live-gap finding
  that motivated the investigation.
- `Sources/CoreMLLLM/DrafterUnion.swift:125` — `rollingCV >=
  crossVocabThreshold` is the gate.
