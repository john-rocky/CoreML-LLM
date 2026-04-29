# Phase B findings — oracle-bench vs live-decode accept-rate gap

**Status:** 2026-04-15 Mac-side finding. Invalidates Phase A5 drafter
projections. iPhone-trip results are consistent with Mac reproduction;
no iPhone-specific bug. Triggered by the `[SpecProfile union]` logs
from PR #57 and confirmed via `coreml-llm-smoke UNION_TRIP=1`.

---

## TL;DR

The `eval/accept-rate-bench-v2.json` numbers that drove Phase A5's
decision (cv=2.31–3.17 E[tok/burst] across categories) were measured
against a **ground-truth corpus**, not against the target Gemma's
actual argmax. In live decoding the two diverge significantly, and
CV's real accept rate is 3–9× lower than the bench projected.

Measured on Mac Studio, one prompt per category, `staging-2k-fast-prefill`,
`coreml-llm-smoke UNION_TRIP=1 SPECULATIVE_PROFILE=1`:

| category | bench cv draft-rate | live cv draft-rate | live zeros | ratio |
|---|---:|---:|---:|---:|
| code    | 0.54 | 0.059 | 15/17 | 9.1× low |
| chat    | 0.44 | 0.114 | 18/22 | 3.8× low |
| qa      | 0.72 | 0.212 | 23/33 | 3.4× low |
| summary | 0.71 | 0.111 | 15/18 | 6.4× low |

(Bench draft-rate = `(E[tok/burst] − 1) / 3`, normalised from the
chain-acceptance E-values in `PHASE_A5_DECISION.md`.)

Union's end-to-end tok/s on Mac is therefore net-regression across all
four categories:

| category | baseline (fallback step) | Union live | delta |
|---|---:|---:|---:|
| code    | 32.9 | 19.3 | −41 % |
| chat    | 32.7 | 17.8 | −46 % |
| qa      | 31.5 | 15.2 | −52 % |
| summary | 32.4 | 21.2 | −35 % |

**This is not an iPhone-specific bug.** The iPhone logs (bursts #60+
with `cv=0.00ms`) match the Mac behaviour once the rolling-accept gate
closes: after ~16 consecutive low-accept CV bursts, `rollingCV` drops
below the 0.20 threshold and CV is silently gated off for the rest of
the generation. iPhone logs just happened to start past that point.

---

## Why the bench over-claims

`Sources/accept-rate-bench/Bench.swift:9` explicitly documents the
methodology:

> Uses oracle replay rather than calling `verify_qK`, which is …

Oracle replay means:

1. Tokenise a corpus prompt → `[t0, t1, t2, …, tN]`.
2. At step `i`, feed `t_i` as seed to the drafter, ask for K draft
   tokens.
3. Compare draft tokens `[d0, d1, d2]` against corpus tokens
   `[t_{i+1}, t_{i+2}, t_{i+3}]`. Count a draft as accepted iff it
   matches the corpus.

This measures **drafter-vs-corpus agreement**. The actual live path is
different:

1. Target Gemma is decoding an OPEN-ended generation (no corpus
   constraint past the prompt).
2. At step `i`, drafter proposes K tokens conditioned on the tokens
   Gemma has emitted so far.
3. Verify chunk (`verify_qK`) returns Gemma's argmax at each of K
   positions.
4. A draft is accepted iff it matches Gemma's argmax.

The two measurements agree only when Gemma's argmax equals the
corpus's next token — which is surprisingly rare, especially past the
first few tokens of output. The model can validly continue in many
directions; the corpus gives one. The drafter, trained on similar
distributional data, produces something plausible but often not
byte-exact to Gemma's argmax.

**This gap is inherent to the oracle-replay methodology, not a bug in
the bench.** It means oracle-replay numbers are an upper bound on live
accept rate, and Phase A5 treated that upper bound as the expected
value.

---

## Per-source live numbers (raw from /tmp/mac-union-logs/)

### code.log (prompt: "Write a Python function to merge two sorted lists…")
```
total bursts: 25, fallback steps: 65, tok/s: 19.31
  cv:  n=17 mean_acc_rate=0.059 zeros=15/17 draft=46.45ms verify=32.27ms
  pl2: n=5  mean_acc_rate=0.200 zeros=4/5   draft=0.00ms  verify=31.44ms
  pl3: n=3  mean_acc_rate=0.667 zeros=1/3   draft=0.00ms  verify=30.93ms
fallback step: 30.44ms ≈ 32.9 tok/s
```

### chat.log (prompt: "Hey, how are you doing today? Tell me something interesting.")
```
total bursts: 23, fallback steps: 64, tok/s: 17.80
  cv:  n=22 mean_acc_rate=0.114 zeros=18/22 draft=46.25ms verify=32.74ms
  pl2: n=1  mean_acc_rate=0.000 zeros=1/1
fallback step: 30.55ms ≈ 32.7 tok/s
```

### qa.log (prompt: "What is the capital of France and why does it have that name?")
```
total bursts: 33, fallback steps: 5, tok/s: 15.20
  cv:  n=33 mean_acc_rate=0.212 zeros=23/33 draft=47.32ms verify=32.14ms
fallback step: 31.74ms ≈ 31.5 tok/s
```

### summary.log (prompt: 200-char Apple news blurb)
```
total bursts: 35, fallback steps: 38, tok/s: 21.16
  cv:  n=18 mean_acc_rate=0.111 zeros=15/18 draft=46.88ms verify=32.96ms
  pl2: n=10 mean_acc_rate=0.550 zeros=3/10
  pl3: n=7  mean_acc_rate=0.643 zeros=1/7
fallback step: 30.82ms ≈ 32.4 tok/s
```

Observations:
- **CV draft cost is ~47 ms per burst** (3 Qwen steps, each ~15–16 ms
  on Mac CPU+GPU). Matches the Phase A5 "order-of-magnitude" estimate
  but the 24 ms M-class projection was optimistic — Mac Studio hits
  47 ms.
- **Verify cost is ~32 ms**, consistent with Phase A5's 1.7× decode
  assumption (baseline decode = 30.5 ms).
- **Per-burst budget for break-even**: to match baseline 30.5 ms/token
  the union needs to emit ≥ `(draft + verify) / 30.5` tokens per
  burst. CV-only: (47+32)/30.5 ≈ 2.6 tokens. At live draft-rate 0.11,
  E[tok/burst] = 1 + 0.33 ≈ 1.33. Net: 59 % slower per burst.
- **Summary edges best** because PL matches repetitive phrases in the
  long prompt (summary has 200 chars of context to match against).
- **QA is the slowest** — CV can't shortcut a factual Q→A format, and
  PL never triggers (no repeating n-grams).

---

## Implications for Phase A5's decision

A5 concluded "ship union-of-drafters" based on:
- cv anchoring chat (2.31 E[tok/burst])
- pl-n3 anchoring code (2.94)
- pl-n2 anchoring summary (3.26)

All three anchors fall apart in live measurement:

| A5 anchor | A5 value | live value | ratio |
|---|---:|---:|---:|
| cv on chat     | 2.31 E[tok/burst] | ≈1.34 | −42 % |
| pl-n3 on code  | 2.94              | ≈2.00 | −32 % |
| pl-n2 on summary | 3.26            | ≈2.65 | −19 % |

(Live E[tok/burst] = 1 + 3 × mean_acc_rate for the source.)

pl-n2 on summary is the only anchor that retains plausibility. cv-chat
and pl-n3 code don't reach the break-even burst budget.

**The A5 decision should be revisited.** Options:

1. ~~**Ship PL-only union**~~ — **Rejected 2026-04-15 by v4 chain
   bench.** pl-n2 chain E[tok/burst]: chat 1.48, code 2.01, qa 1.00,
   summary 1.00. Break-even needs ≥ 2.6. Net-regression on 3 of 4
   categories. See `docs/PHASE_B_DECISION.md`.
2. ~~**Raise rolling-gate thresholds**~~ — **Noted but not
   implemented.** Under Union default OFF, gate tuning affects no
   out-of-box user. Existing gates (CV 0.20, PL 0.05) already
   collapse to baseline. File if a user reports regression after
   opting in.
3. ~~**Defer Union until Mirror SD**~~ — **Amended 2026-04-15 late
   by v4.** Mirror SD alone does not unlock Phase B: v4 showed
   acceptance ceiling is set by `verify_qK`'s fp16 batch-content
   sensitivity, not drafter speed. The correct gating item is
   **verify-chunk numerical tightening** (roadmap item 11c, now
   "C0"), which must close the batched-verify drift before Mirror
   SD can yield a speculative speedup.

See `docs/PHASE_B_DECISION.md` for the consolidated go-forward.

Option 3 aligns with Phase A5's own caveat:

> Route B's union-of-drafters alone matches Google only once Mirror SD
> hides drafter cost.

— but A5 expected baseline-matching on serial, not the observed
−35 to −52 % regression. Option 1 or 2 should be evaluated before
committing to C.

---

## Also: item 11c is NOT implicated here

The live-decode gap is present on **Mac** (no Neural Engine involvement
on the verify path — cpuAndGPU for `coreml-llm-smoke` at default).
Item 11c concerns K=3 vs K=1 fp16 drift that would only show on
Neural Engine. The gap measured here is the bench-methodology gap,
which manifests identically on Mac CPU+GPU and iPhone ANE.

iPhone-trip interpretation (post-PR #57) should be updated:

- Previous branching hypothesis in `docs/HANDOFF.md:105` assumed
  "iPhone accept << Mac oracle replay" implies 11c. That framing is
  wrong — iPhone accept is already << Mac oracle because live << oracle
  regardless of platform.
- The correct comparison is **iPhone live vs Mac live**. If those
  agree, 11c is not load-bearing for live decoding. If iPhone live is
  meaningfully lower than Mac live, then 11c's fp16 drift adds on top
  of the bench gap.

A quick data point for this: Mac live cv accept-rate mean across
categories is 0.124. iPhone mean across the ~10 bursts we have before
the rolling gate closed was qualitatively similar (mixed 0/2 and 1/2
and 2/2 visible). No large gap visible. **11c probably is NOT the
bottleneck; the bench methodology is.**

---

## Next steps (Mac-only, no iPhone trip needed)

1. **Extend accept-rate-bench with a target-argmax mode.** Instead of
   comparing draft tokens against corpus, open-generate from Gemma and
   at each step record `verify_qK`'s argmax for the drafter's current
   proposal. This gives live-equivalent numbers without running the
   chat app. ~30 min effort, reuses existing bench harness.
2. **Run the new bench across all 3 drafters × 4 categories × 10
   prompts.** Compare with oracle-replay numbers side-by-side. Produce
   `eval/accept-rate-bench-v3-target-argmax.json`.
3. **Decide Union shape based on v3 numbers** — probably PL-only, or
   defer until Phase C.
4. Once v3 bench exists, encode it as the regression test for future
   drafter work so we never ship on oracle-replay numbers again.

Estimated turnaround: 1 Mac-day.

---

## Reproducing

```bash
# build once
swift build -c release --product coreml-llm-smoke

# run per-category (UNION_TRIP enables mtp=off, union=on, cv=on;
# SPECULATIVE_PROFILE prints [SpecProfile union] lines; the temp
# CVDebug prints in Sources/CoreMLLLM/CrossVocabDraft.swift show
# whether CV actually runs or bails on mapSeed)
for cat in code chat qa summary; do
  UNION_TRIP=1 SPECULATIVE_PROFILE=1 \
    .build/arm64-apple-macosx/release/coreml-llm-smoke \
    ~/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b \
    "<prompt for $cat>" 100 > /tmp/mac-union-logs/$cat.log 2>&1
done

# analyse
python3 /tmp/mac-union-logs/analyze.py /tmp/mac-union-logs/*.log
```

`analyze.py` is the 50-line parser that produced the tables above;
not committed yet.

---

## Related

- `docs/PHASE_A5_DECISION.md` — the decision this finding invalidates.
- `docs/HANDOFF.md` §"Phase B priority ordering" — the branching chart
  that assumed 11c was the Phase B culprit; should be revised.
- `eval/accept-rate-bench-v2.json` — the oracle-replay numbers that
  over-claimed.
- `Sources/accept-rate-bench/Bench.swift:9` — the bench's methodology
  disclaimer ("Uses oracle replay rather than calling `verify_qK`"),
  which is now the pivot of this finding.
