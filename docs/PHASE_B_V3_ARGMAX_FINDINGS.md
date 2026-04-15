# v3 argmax-mode bench findings — the `decode_q1` / `verify_qK` drift is real but does not explain the live 3–9× gap

**Status:** 2026-04-15 Mac-side follow-up to
`docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md`. Produced from
`eval/accept-rate-bench-v3-target-argmax.json` (mode=both, 10 prompts,
K=3, maxTokens=128, staging-2k-fast-prefill on Mac Studio cpuAndGPU).

---

## TL;DR

The v3 argmax mode was added to check the hypothesis that oracle-replay
over-claims because `decode_q1` and `verify_qK` disagree at the fp16
level (item 11c). Measurement on 10 prompts:

- `decode_q1` and `verify_qK` argmax **chains do diverge** on Mac
  cpuAndGPU — 4 of 10 prompts drop below 50 % agreement within the first
  20 tokens, one as low as 12 %.
- **But drafter chain-accept stats barely move** between the two modes
  (delta ±5 %–25 % on categories that diverge, 0 % on qa where chains
  fully agree). Nothing close to the 3–9× gap `PHASE_B_LIVE_ACCEPT_RATE_GAP.md`
  reported from live Union runs.

**Implication:** the `decode_q1 ↔ verify_qK` numerical drift is real and
on-Mac (not ANE-only), but it is **not** the dominant driver of the
bench-vs-live gap. Something else in the live Union path — bootstrap
replay, cross-vocab state, rolling-gate dynamics, or a subtle
drafter-vs-bench implementation delta — explains most of the 3–9×.

---

## Methodology

`accept-rate-bench` now accepts `--mode {oracle|argmax|both}`:

- **oracle** (pre-existing): compare drafter proposals against the
  token chain emitted by `llm.generate(...)`, which uses `decode_q1` at
  each step.
- **argmax** (new): for each prompt, open-generate a target-argmax
  chain by calling `verify_qK` at each position with
  `[nextID, 0, 0, ..., 0]` and consuming only `argmax[0]`. Slots
  1..K-1 are zero-padded dummies; target's prediction at P+1 depends
  only on slot 0's token at position P, so slots 1..K-1 inputs do not
  affect the argmax the bench uses. Then replay drafters against the
  resulting `emitted_verify` chain (same chain-accept semantics as
  oracle mode; only the corpus differs).
- **both**: run both modes per prompt; report both stats side-by-side
  and record the leading-token agreement between the two chains.

`Sources/accept-rate-bench/Bench.swift` + new narrow public helpers on
`CoreMLLLM` (`benchPrefill`, `benchVerify`, `benchAdvance`,
`benchCurrentPosition`, `benchVerifyK`). No changes to production APIs.

---

## Results

### Per-prompt chain divergence

```
category   prompt                         dec   arg   agree / min
code       code-complete-sum              11    19    3 / 11   ( 27 %)
code       code-explain-fib              128    75    9 / 75   ( 12 %)
code       code-refactor-loop             22    22   22 / 22   (100 %)
summary    sum-para-ane                   21    21   21 / 21   (100 %)
summary    sum-para-compiler              41    31   15 / 31   ( 48 %)
qa         qa-where-is-swift              11    11   11 / 11   (100 %)
qa         qa-what-is-ane                 12    12   12 / 12   (100 %)
chat       chat-define-transformer        66    21   10 / 21   ( 48 %)
chat       chat-explain-rope             128    20    6 / 20   ( 30 %)
chat       chat-greeting                  10    10    2 / 10   ( 20 %)
```

`dec` is the `decode_q1` chain length (hit EOS or maxTokens); `arg` the
`verify_qK` chain length. `agree / min` is how many leading tokens were
byte-identical across the two chains, over the common prefix length.

Two observations:

1. **`verify_qK` chains terminate earlier** on 6 of 10 prompts (argmax
   reaches EOS earlier than `decode_q1`). Unclear whether this is fp16
   rounding tipping EOS logits or a K-specific graph artefact. Not
   investigated further here.
2. **Chain agreement is bimodal** — either ~100 % (qa, one code, one
   summary) or quickly collapses (20–48 %). Prompts where the model is
   confident (short factual answers) stay in sync; longer open-ended
   generations diverge.

### Per-category aggregate (E[tok/burst], `oracle` vs `argmax`)

```
category   drafter                oracle   argmax   Δ
chat       cross-vocab-qwen         2.31     2.77   +20.0 %
chat       prompt-lookup-n2         1.35     1.00   −25.8 %
chat       prompt-lookup-n3         1.01     1.00    −0.5 %
chat       suffix-scan              1.35     1.00   −25.8 %
code       cross-vocab-qwen         2.63     2.45    −7.1 %
code       prompt-lookup-n2         2.72     2.65    −2.6 %
code       prompt-lookup-n3         2.94     2.60   −11.8 %
code       suffix-scan              2.93     2.63   −10.2 %
qa         cross-vocab-qwen         3.17     3.17     0.0 %
qa         prompt-lookup-n2         2.89     2.89     0.0 %
qa         prompt-lookup-n3         2.96     2.96     0.0 %
qa         suffix-scan              2.89     2.89     0.0 %
summary    cross-vocab-qwen         3.12     3.20    +2.4 %
summary    prompt-lookup-n2         3.26     3.41    +4.6 %
summary    prompt-lookup-n3         3.22     3.31    +2.9 %
summary    suffix-scan              3.26     3.39    +3.9 %
```

Compare against `PHASE_B_LIVE_ACCEPT_RATE_GAP.md`'s live-Union numbers
(Mac `coreml-llm-smoke UNION_TRIP=1`):

```
category   cv live E[tok/burst]   oracle   argmax(v3)
code       1.18                   2.63     2.45
chat       1.34                   2.31     2.77
qa         1.64                   3.17     3.17
summary    1.33                   3.12     3.20
```

(live cv E[tok/burst] = 1 + 3 × mean_acc_rate; values 1.18 / 1.34 /
1.64 / 1.33 from PHASE_B §Per-source live numbers.)

Argmax mode is **not** close to live. It is essentially oracle with
±25 % noise.

---

## What this rules out

- **"`decode_q1` vs `verify_qK` fp16 drift alone explains the gap"** —
  ruled out. Chain-accept numbers are stable across the two corpora on
  Mac Studio cpuAndGPU.
- **"The gap is an iPhone ANE-only phenomenon"** — already ruled out by
  PHASE_B. v3 reinforces: Mac live-Union bursts match the PHASE_B
  finding regardless of which chunk function is used to generate the
  oracle chain.

## What's still open

The live-Union path in `DrafterUnion.speculateStep` does things the
bench's oracle replay and argmax replay both skip:

1. **Bootstrap replay** — on first burst, CrossVocab replays the
   entire committed history through Qwen to seed its KV. Any
   bootstrap bug would systematically hurt live CV without affecting
   bench CV (bench rewinds/restores `committedPosition` but doesn't
   replay the full prompt through Qwen from scratch).
2. **Cross-vocab state advancement mid-decode** — live applyCommit /
   fallback handling vs bench's `defer { drafter.committedPosition = saved }`
   rewind (`Sources/accept-rate-bench/Drafters.swift:117`). Any Qwen
   state drift between bursts in live mode would reduce CV accept
   rate.
3. **Rolling-accept gate feedback** — live mode shuts CV off once
   rollingCV drops below 0.20, and the PL sources once rollingPL
   drops below 0.05. A single unlucky prefix can flip the gate and
   suppress the drafter for the rest of the generation. The bench
   has no such gate.
4. **Verify slot 1..K-1 contamination** — live Union feeds drafter
   proposals into verify slots 1..K-1; the bench's argmax mode fills
   slots 1..K-1 with zeros. If target's argmax at slot 1 depends on
   slot 0's token in a way that chain-accept is sensitive to (beyond
   what gotcha #3 in SESSION_STATE.md predicts), this could be part
   of the gap. Unlikely given the chain-accept equivalence proof but
   worth a targeted test.

Next concrete step before picking a Union shape: reproduce the live
gap under the bench harness by adding a **chain-following argmax
mode** — feed the drafter's own proposals into verify slots 1..K-1
(not zeros) and count chain-accept against the K argmaxes that come
back. That measures exactly what live Union measures. If it still
shows ±25 %, the gap is not methodological; it's implementation.

Estimated effort: 1–2 h on top of the v3 plumbing.

---

## Artifacts

- `eval/accept-rate-bench-v3-target-argmax.json` — full per-prompt
  results, both modes.
- `Sources/accept-rate-bench/Bench.swift` — `--mode` flag wiring.
- `Sources/CoreMLLLM/CoreMLLLM.swift` — `benchPrefill`, `benchVerify`,
  `benchAdvance`, `benchCurrentPosition`, `benchVerifyK` helpers (all
  public; intended for offline harness use only, not production).

## Reproducing

```bash
swift build -c release --product accept-rate-bench
.build/release/accept-rate-bench --mode both --max-tokens 128 \
  --out eval/accept-rate-bench-v3-target-argmax.json
```

## Related

- `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` — original finding and the
  three Union-shape candidates (PL-only / raise gates / defer to
  Mirror SD).
- `docs/PHASE_A5_DECISION.md` — the historical projections v3 is
  sanity-checking.
- `docs/SESSION_STATE.md` §"Gotchas" item 3 — the chain-accept
  equivalence argument at temp=0. Still holds; argmax and oracle
  numbers matching is consistent with it.
