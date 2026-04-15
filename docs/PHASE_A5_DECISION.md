# Phase A5 — drafter decision + theoretical on-device ceiling

> ⚠️ **SUPERSEDED by PR #62 (2026-04-15 late).** The decision in this
> doc ("ship a union-of-drafters: cross-vocab Qwen + PL-n3 + PL-n2")
> was based on oracle-replay bench numbers that turned out to
> over-claim live accept rates by 3–9×. Mac reproduction via
> `coreml-llm-smoke UNION_TRIP=1` shows Union is a net regression on
> all four categories (15–21 tok/s vs baseline 32). The theoretical
> tok/s projections below (30/42/57/63 serial; 44/57/61/63 Mirror
> SD) assume drafter accept rates that do not reflect live decoding.
>
> Read `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` for the updated
> picture. Union shape is being re-decided via task #2 (target-argmax
> replay mode for accept-rate-bench). Content below retained for
> historical context — **do not plan work on top of these numbers**.

**Status:** 2026-04-15 late. ~~Closes Phase A of
`docs/MAC_FIRST_EXECUTION_PLAN.md`. Three device trips saved (0 done).
Ready for Phase B.~~ Superseded by PR #62; see banner above.

---

## Measured accept rates (Mac Studio, `staging-2k-fast-prefill`)

Oracle replay at temperature = 0, K = 3, `max_tokens = 128`. 10-prompt
corpus across code / summary / qa / chat categories. Numbers are chain
acceptance probabilities and the derived `E[tokens/burst]`.

| workload | prompt-lookup n=2 | prompt-lookup n=3 | suffix-scan | **cross-vocab qwen** |
|---|---:|---:|---:|---:|
| chat | 1.35 | 1.01 | 1.35 | **2.31** ✅ |
| code | 2.72 | **2.94** ✅ | 2.93 | 2.63 |
| qa | 2.89 | 2.96 | 2.89 | **3.17** ✅ |
| summary | **3.26** ✅ | 3.22 | **3.26** ✅ | 3.12 |

The single-turn suffix scan tracks prompt-lookup-n2 exactly — Suffix
only earns its keep with a multi-turn session simulator, which isn't
needed for A5 because the winner on every category is already clear.

## Decision

**Ship a union-of-drafters.** No single drafter wins all four
categories, and the margins are significant (chat goes from unusable
1.35 to a respectable 2.31 once cross-vocab enters).

Minimum viable union:

- **cross-vocab Qwen 2.5 0.5B** — chat + qa anchor
- **prompt-lookup (n=3)** — code anchor
- **prompt-lookup (n=2)** — summary anchor (also matches suffix)

Per-burst, run all three drafters, take the longest matching prefix
across their proposals against the target's verify argmax. One verify
pass per burst regardless of how many drafters ran. Rolling-accept
gate disables drafting below ~0.2 acceptance so we never regress.

All three sources live under the `DrafterUnion` orchestrator
(`Sources/CoreMLLLM/DrafterUnion.swift`, landed in PR #54) that
asks each for up to K proposals per cycle and picks the best
according to the priority policy documented inline there. Opt-in
via `drafterUnionEnabled = true` — default off on main since the
Qwen drafter perf regresses on iPhone (see `docs/SESSION_STATE.md`
active tasks).

## Theoretical on-device ceiling (PROJECTION — requires iPhone confirmation)

> ⚠️ All tok/s numbers in this section are **Mac-side projections**
> derived from measured iPhone baseline chunk times + assumed verify
> cost. They are NOT iPhone measurements and must be confirmed in
> Phase B. The product's shipped speed claim is always the
> iPhone-measured number. See `MAC_FIRST_EXECUTION_PLAN.md` §"What
> each environment is authoritative for" for the split.

Inputs:

- iPhone 17 Pro baseline 2K decode = 31 tok/s (measured 2026-04-14).
  Per-step chunk cost c1=5.9 c2=6.8 c3=8.1 c4=10.4 ms; sum ≈ 31 ms.
- Verify chunk cost ≈ 1.7× decode (K=3 seq dim, same KV-read topology).
  Budget ~52 ms per verify dispatch on iPhone.
- Drafter cost on Qwen 2.5 0.5B @ GPU ≈ 8 ms / draft step × 3 = 24 ms.
  Prompt-lookup is a CPU n-gram scan ≈ 0.1 ms, negligible.
- Mirror SD (T2) would hide drafter cost entirely; without it the
  drafter cost sits on the critical path.

Serial throughput per category (without Mirror SD):

```
total burst time = drafter cost + verify cost
tok/s = E[tok/burst] / total burst time
```

| category | best drafter | E[tok/burst] | burst time (ms) | tok/s | vs Google 56 |
|---|---|---:|---:|---:|---:|
| chat | cv-qwen | 2.31 | 52 + 24 = 76 | **30** | −46 % |
| code | pl-n3 | 2.94 | 52 + 0.1 ≈ 52 | **57** | **+2 %** |
| qa | cv-qwen | 3.17 | 76 | **42** | −25 % |
| summary | pl-n2 | 3.26 | 52 | **63** | **+12 %** |

**With Mirror SD (Phase C/D) the drafter cost is hidden**, so the
numbers collapse to the 52 ms verify floor:

| category | E[tok/burst] | verify-only burst (52 ms) | tok/s |
|---|---:|---:|---:|
| chat | 2.31 | 52 | **44** |
| code | 2.94 | 52 | **57** |
| qa | 3.17 | 52 | **61** |
| summary | 3.26 | 52 | **63** |

Mixed-workload average (equal weight across categories):

- **Serial (no Mirror SD): ~48 tok/s** — (30 + 57 + 42 + 63) / 4.
  Below Google's 56 on average; chat is the bottleneck.
- **With Mirror SD: ~56 tok/s** — (44 + 57 + 61 + 63) / 4. Ties
  Google's iOS number.

The qualitative conclusion: **Route B's union-of-drafters alone
matches Google only once Mirror SD hides drafter cost**. Without
Mirror SD, the chat regression drags the average below Google.

What the *current main* branch delivers today (post-PR #54):
DrafterUnion is wired, but both `drafterUnionEnabled` and
`crossVocabEnabled` default to `false`, so the default path runs
pure target decode at ~31 tok/s. Opting in via
`drafterUnionEnabled = true` on iPhone 17 Pro was measured at
**1.8 tok/s decode and 25+ s TTFT on short prompts** — the Qwen
drafter was roughly 10× slower than the 24 ms estimate, likely
running on CPU rather than GPU. The 48 and 56 averages above assume
Mac-measured drafter cost plus Mirror-SD overlap, neither of which
reflects iPhone reality today. Perf investigation (PR #57) lands
the timing logs needed to isolate the regression.

## Known limits of this estimate

- Verify cost ratio (1.7×) is a guess; the MLComputePlan on chunks
  at K=3 will tell us. Phase B bundles that audit.
- Drafter cost on Qwen 0.5B at GPU is an order-of-magnitude estimate
  (Apple's MLX-LM reports ~100–150 tok/s at 0.5B on M-class GPUs;
  K=3 steps at that rate is 20–30 ms).
- Accept rates above were measured on 10 prompts per category. Noise
  band is ±0.1 E[tokens/burst]. Enough to rank drafters but the
  absolute tok/s number should be confirmed on device.
- The chat regression risk in the serial-drafter case (30 tok/s vs 31
  baseline) means the rolling gate must be aggressive — fall back to
  single-token decode if rolling accept ever drops below ~0.25.

## Next steps

### Phase B — one bundled device trip

All four items land together:

1. Union-of-drafters wiring (cross-vocab + prompt-lookup-n2/n3).
2. Rolling-accept gate with per-drafter thresholds (cross-vocab:
   ~0.3, prompt-lookup: ~0.1 since missing is cheap).
3. Runtime hints V6-1 + V6-2 (shape-trace + warm-pool).
4. MLComputePlan audit on the union path.

Exit criterion for Phase B: ≥ 50 tok/s on chat prompts (up from 31)
and ≥ 60 tok/s on other categories. Or a clear diagnosis of where
the theoretical estimate diverges from reality.

### Phase C — if Phase B under-delivers on chat

The serial-drafter ceiling on chat is 30 tok/s. If Phase B's measured
chat number tracks the estimate, the next unlock is **Mirror SD
(T2)** — parallel GPU drafter + ANE verify — which pushes chat to
~44 tok/s and overall to ~56.

Async ANE dispatch (I1) is the prerequisite. 4–6 days, Swift.

---

## Appendix — raw data

`eval/accept-rate-bench-v2.json` in repo, produced by
`swift run -c release accept-rate-bench --max-tokens 128 --out …` on
commit e0c8aee.

Reproduce:

```bash
# (one-time) compile mlpackage chunks to mlmodelc for Mac runtime
for i in 1 2 3 4; do
    xcrun coremlc compile \
        ~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b/chunk$i.mlpackage \
        /tmp/compile-2k/
done
# (one-time) install cross-vocab drafter
python conversion/setup_cross_vocab_drafter.py \
    --gemma-dir ~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b \
    --qwen-dir  ~/Downloads/CoreML-LLM/conversion/output/qwen2.5-0.5b
xcrun coremlc compile \
    ~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b/cross_vocab/qwen_drafter.mlpackage \
    /tmp/qwen-compile/

# run bench
swift build -c release --product accept-rate-bench
.build/arm64-apple-macosx/release/accept-rate-bench \
    --max-tokens 128 --out eval/accept-rate-bench-v2.json
```
