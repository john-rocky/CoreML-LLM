# v4 chain-following bench — live gap is reproducible, methodology is the dominant factor

**Status:** 2026-04-15 follow-up to v3. Produced from
`eval/accept-rate-bench-v4-chain.json` (mode=all, 10 prompts, K=3,
maxTokens=128, staging-2k-fast-prefill, Mac Studio cpuAndGPU).

---

## TL;DR

A third mode (`--mode chain`) was added: drafter proposals feed verify
slots 1..K-1 (instead of zeros) and chain-accept is measured against
`verify_qK`'s returned argmax[0..K-2]. This is what
`DrafterUnion.speculateStep` actually measures on-device.

**Chain mode reproduces most of the PHASE_B live gap.** The
`decode_q1 → chain` drop on cross-vocab-qwen matches live numbers
directionally across 3 of 4 categories:

```
category   oracle   argmax   chain   live (PR #62)
code        2.63     2.45    1.01        1.18
chat        2.31     2.77    2.31        1.34
qa          3.17     3.17    2.04        1.64
summary     3.12     3.20    1.00        1.33
```

(E[tok/burst] for cross-vocab-qwen. `live` from
`docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` §Per-source live numbers:
1 + 3 × `mean_acc_rate`.)

**Conclusion:** the bench-vs-live gap is **methodological**, not
implementation. Oracle-replay and v3 argmax-mode both condition verify
slots 1..K-1 on "clean" input (corpus tokens or zeros). Live Union
conditions them on the drafter's proposals. That single difference
accounts for most of the 3–9× gap PHASE_B reported.

---

## Why does verify-slot content matter?

At temp=0, target's argmax at position P+1 depends on `verifyTokens[0]`
and past KV cache only — slots 1..K-1 should be irrelevant for
argmax[0]. `SESSION_STATE.md` gotcha #3 formalised this and argued that
chain-accept (at temp=0) is identical across oracle replay and live.

v4 empirically refutes that at the fp16 level. With drafter proposals
in slots 1..K-1, target's verify_qK argmax chain drifts noticeably from
the zero-padded chain — enough that prompt-lookup drafters (which need
byte-exact n-gram matches against prior history) lose almost all their
matches on summary / qa.

Mechanism is unconfirmed but most likely the batched multi-function
`verify_qK` compute path has fp16 ordering / reduction differences
that depend on all K input tokens jointly. At temp=0 the tiny logit
drift can flip argmax on marginal tokens, compounding over a chain.

This is not "item 11c" (`decode_q1` ↔ `verify_qK` drift); v3 already
ruled that out as the dominant factor. This is a *within-verify_qK*
sensitivity to batch content, and it explains the remaining gap.

> **UPDATE 2026-04-15 by PR #72 (Track B approach 3).** The
> speculation above — that "the batched multi-function `verify_qK`
> compute path has fp16 ordering / reduction differences that depend
> on all K input tokens jointly" — is **refuted**. PR #72 replaced
> the batched `verify_qK` call with K serial `decode_q1` calls (no
> joint-K-token compute at all) and re-ran the v4 chain bench. The
> cross-vocab code category stayed at 1.01 (oracle 2.63); qa stayed
> at 2.04; summary at 1.00; chat moved 2.31 → 2.09 within noise. If
> batched fp16 ordering were the mechanism, removing the batched
> path would have closed most of the gap. It didn't.
>
> The actual mechanism is **semantic, not numerical**: verify writes
> drafter proposals into the KV cache at positions P+1..P+K-1 *before*
> the acceptance decision. Subsequent target argmaxes (at the next
> committed position) read attention over a cache containing possibly-
> rejected drafter tokens, which is a real semantic difference from
> oracle replay (which never calls verify) and from the "clean" chain
> a pure `decode_q1` loop would produce. Serial decode reproduces the
> gap exactly because it writes KV at the same sites the batched path
> does — just sequenced. See
> `docs/PHASE_C_TIGHTENING_FINDINGS.md` for the per-category table
> and `docs/PHASE_B_DECISION.md` §"Phase C gating item" for the
> resulting C0 candidate-list update (fp32 upcast / accumulation-order
> are dead; output-space tolerance and verify-protocol redesign
> remain).

---

## Per-category results (E[tok/burst])

```
category   drafter              oracle   argmax   chain   Δ(chain-oracle)
chat       cross-vocab-qwen      2.31     2.77    2.31      0.0 %
chat       prompt-lookup-n2      1.35     1.00    1.48     +9.6 %
chat       prompt-lookup-n3      1.01     1.00    1.49    +47.5 %
chat       suffix-scan           1.35     1.00    1.49    +10.4 %
code       cross-vocab-qwen      2.63     2.45    1.01    −61.6 %
code       prompt-lookup-n2      2.72     2.65    2.01    −26.1 %
code       prompt-lookup-n3      2.94     2.60    1.01    −65.6 %
code       suffix-scan           2.93     2.63    1.01    −65.5 %
qa         cross-vocab-qwen      3.17     3.17    2.04    −35.6 %
qa         prompt-lookup-n2      2.89     2.89    1.00    −65.4 %
qa         prompt-lookup-n3      2.96     2.96    1.00    −66.2 %
qa         suffix-scan           2.89     2.89    1.00    −65.4 %
summary    cross-vocab-qwen      3.12     3.20    1.00    −67.9 %
summary    prompt-lookup-n2      3.26     3.41    1.00    −69.3 %
summary    prompt-lookup-n3      3.22     3.31    1.00    −69.0 %
summary    suffix-scan           3.26     3.39    1.00    −69.3 %
```

### Observations

- **3 of 4 categories show the bench-vs-live gap in chain mode.**
  code / qa / summary all drop 25–70 % vs oracle, bringing CV and PL
  numbers close to live.
- **Chat is the outlier.** CV chain = CV oracle = 2.31, whereas live
  shows 1.34. Chat's decode_q1 argmax chain seems robust to the
  batched-verify drift that kills the other categories. Possibly
  because chat generations are short enough that the target's
  confidence stays high (logit margins don't flip under fp16 drift).
- **Prompt-lookup collapses to near-zero accept on summary/qa in
  chain mode.** Summary pl-n2 oracle 3.26 → chain 1.00, qa pl-n2
  2.89 → chain 1.00. PL depends on byte-exact n-gram matches against
  prior history; if the verify chain drifts at all, history contains
  drifted tokens and PL's n-gram lookup misses.
- **On chat, chain > oracle for prompt-lookup.** Novel chat content
  has no n-gram matches in oracle mode (PL returns empty). In chain
  mode, the drifted chain happens to repeat tokens PL can match.
  Noise, not signal.

---

## Implications for Union-shape decision

With v4's methodology in place, the three candidates from
`PHASE_B_LIVE_ACCEPT_RATE_GAP.md` can be re-evaluated against
live-equivalent numbers:

1. **PL-only Union.** Chain E[tok/burst] for pl-n2: chat 1.48,
   code 2.01, qa 1.00, summary 1.00. Only code beats baseline
   (break-even = 2.6 per the per-burst arithmetic in PHASE_B).
   PL-only Union is net-regression on 3 of 4 categories. **Rejected.**
2. **Raise rolling gates.** Make rollingPL / rollingCV shut off
   faster so the system collapses to straight decode. v4 suggests
   this is the safest immediate move: baseline is preserved on
   all categories, no speculative gain, no regression. The
   current thresholds (0.05 / 0.20) already do this on iPhone
   but v4 suggests tighter floors may be needed. **Viable short-term.**
3. **Defer Union until Mirror SD.** v4 confirms drafters are
   fundamentally mis-matched against verify_qK's batched argmax
   on 3/4 categories. Even with perfect drafters, they'd lose
   30–70 % of matches to the batched-verify drift. Mirror SD
   doesn't fix this — same verify chunk is used. Unless verify
   chunks are re-quantised / re-exported with tighter numerics
   to reduce the batched-argmax sensitivity, speculative decoding
   under the current target is inherently capped. **Most likely
   right answer; filed as follow-up item.**

---

## Caveats

- `runChainMode` runs all drafters sequentially at the same
  `startPosition = P`. Each verify call overwrites KV slots
  [P..P+K-1]. Slot P's KV after the loop is from the LAST drafter's
  call, not the first (whose argmax[0] seeds the chain). At fp16
  the two should be identical (slot 0 input is constant) but
  measurable drift is possible. Running each drafter in isolation
  with independent prefills would eliminate this; not done here
  for cost reasons (would multiply prefill time by 4).
- `prompt-lookup` drafters in chain mode can show `chain > oracle`
  on categories where the drifted chain happens to repeat n-grams
  that the pristine chain didn't. See chat pl-n3 in the table.
- v4 does not explain the full chat CV gap (chain 2.31 vs live 1.34).
  Remaining ~40 % likely from live-specific mechanics (rolling-accept
  gate closing mid-generation, bootstrap replay through Qwen, or
  cross-vocab state drift mid-burst). Investigating this further
  requires instrumentation on `DrafterUnion.speculateStep`, not bench
  harness changes.

---

## Artifacts

- `eval/accept-rate-bench-v4-chain.json` — full per-prompt results,
  all three modes.
- `Sources/accept-rate-bench/Bench.swift` — `--mode {chain, all}`
  added alongside existing oracle / argmax / both.
- `Sources/CoreMLLLM/CoreMLLLM.swift` — no new public surface beyond
  v3's bench helpers.

## Reproducing

```bash
swift build -c release --product accept-rate-bench
.build/release/accept-rate-bench --mode all --max-tokens 128 \
  --out eval/accept-rate-bench-v4-chain.json
```

## Related

- `docs/PHASE_B_V3_ARGMAX_FINDINGS.md` — v3 ruled out `decode_q1`
  vs `verify_qK` drift as the dominant factor.
- `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` — original live-gap finding
  and the three candidate next steps (1 and 3 now have v4-grounded
  verdicts).
- `docs/SESSION_STATE.md` §"Gotchas" item 3 — the chain-accept
  equivalence argument. **Empirically contradicted by v4** at the
  fp16 level; the argument holds in exact arithmetic only.
