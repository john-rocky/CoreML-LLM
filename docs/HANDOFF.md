# Next-session handoff

**Last updated:** 2026-04-15 post B.3 refutation (PR #72) — batched
fp16 hypothesis is dead; mechanism is semantic KV write-through.

## Read this first

To resume cleanly, the next session should:

1. **If starting C0 work:** open `docs/PHASE_C_TIGHTENING_FINDINGS.md`
   first — PR #72's B.3 result that rules out the batched-fp16
   hypothesis and pins the mechanism to semantic KV write-through.
   Then `docs/NEXT_SESSION_C0.md` for the two-track (loosen / rebuild)
   branching, **but read the updated candidate list in
   `docs/PHASE_B_DECISION.md` §"Phase C gating item" §"2026-04-15
   update — B.3 refutation" before acting on NEXT_SESSION_C0 —
   approaches 1 and 2 there are now dead.**
2. Open this file (`docs/HANDOFF.md`) — takes 5 minutes.
3. Read `docs/PHASE_B_DECISION.md` — **Phase B closes here; Union
   defaults stay OFF; Phase C is gated on item "C0".** Candidate
   list is now (a) output-space tolerance / (b) verify-protocol
   redesign; fp32 upcast and accumulation-order variants are dead
   per B.3.
4. Read `docs/PHASE_B_V4_CHAIN_FINDINGS.md` — empirical basis for
   the decision **layered with the 2026-04-15 UPDATE callout that
   retracts the batched-fp16 speculation**.
5. Read `docs/PHASE_B_V3_ARGMAX_FINDINGS.md` — v3 ruled out the
   narrower "`decode_q1` vs `verify_qK` drift" hypothesis.
6. Read `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` — original live-gap
   finding.
7. Skim `docs/SESSION_STATE.md` for exact PR / branch / task state
   (gotcha #3 has the short-form B.3 summary).
8. `docs/MAC_FIRST_EXECUTION_PLAN.md` for the phased itinerary.
9. `docs/PHASE_A5_DECISION.md` — historical, superseded.

No need to touch MTP Path A, PR #17, or PR #33 — all deprioritised
with reasons in the roadmap's Rejected table.

## Suggested opening prompt for the next session

Copy-paste this at the start of the next `/claude` session so the
model walks in with correct framing:

> Phase B closed 2026-04-15. Union defaults stay OFF on main.
> Phase C is gated on roadmap item 11c ("C0"). Refutation chain:
> v3 ruled out decode_q1 ↔ verify_qK drift; v4 hypothesised batched-
> fp16 ordering inside verify_qK; **B.3 / PR #72 refuted that** by
> swapping batched verify_qK for K serial decode_q1 calls and seeing
> no chain-gap closure. The mechanism is **semantic** — verify writes
> drafter proposals into KV before acceptance is decided, so target
> argmaxes at subsequent positions condition on the contaminated
> cache. This is not fixable by tightening verify numerics.
> Remaining C0 options: (a) output-space tolerance (Track A / PR #73
> patch-ready, cheap; blocked on verify chunks emitting top-K
> logits), (b) verify-protocol redesign with delayed KV write-through
> (multi-week).
>
> Read (in order):
> 1. docs/PHASE_C_TIGHTENING_FINDINGS.md — B.3 result and mechanism.
> 2. docs/PHASE_B_DECISION.md — go-forward plan with B.3-updated
>    candidate list.
> 3. docs/PHASE_B_V4_CHAIN_FINDINGS.md — empirical basis with the
>    2026-04-15 UPDATE callout.
> 4. docs/PHASE_B_V3_ARGMAX_FINDINGS.md — ruled out the narrower
>    drift hypothesis.
> 5. docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md — original live-gap finding.
> 6. docs/HANDOFF.md (this file) + docs/SESSION_STATE.md.
>
> Start options:
> - **Track A measurement (cheap):** unblock by landing the verify
>   chunk re-export that emits `logits_fp16` alongside `token_ids`
>   (conversion/ change, ~1 output-node addition); then Track A
>   bench (PR #73 wiring) measures how many chain flips are
>   within-margin and should be tolerated.
> - **Non-speculative D1 spike (also cheap, unblocked now):** staged
>   chunk pipelining per HANDOFF §"Phase D". Doesn't depend on
>   verify numerics and is independent of the C0 question.
> - Verify-protocol redesign (option (b)) is multi-week and should
>   wait until Track A has quantified whether tolerance alone
>   recovers a useful fraction of the chain gap.
>
> Bench + `bench*` helpers on CoreMLLLM are public (merged).
> Docs auto-merge; bench/harness code auto-merges too.

## Full roadmap — Phase A → final

Reference numbers: iPhone 17 Pro 2K baseline 31 tok/s (measured);
Google LiteRT-LM iOS 56 tok/s (user-supplied). All Phase B-onward
numbers below are **projections needing iPhone confirmation**, not
measured.

### Phase A — accept-rate measurement and drafter selection ✅ DONE

Mac-only. Oracle-replay bench on 40 prompts × 4 drafters. Decision:
union of cross-vocab Qwen 2.5 0.5B + prompt-lookup-n2 + prompt-lookup-n3.

Evidence: `eval/accept-rate-bench-v2.json`, `docs/PHASE_A5_DECISION.md`.

### Phase B — drafter union + basic hygiene (NEXT)

Goal: first tok/s gain on device.

| # | Task | Effort | Exit criterion |
|---|---|---|---|
| B1 | Union-of-drafters orchestrator class. Runs cross-vocab + prompt-lookup{n=2, n=3} per burst, picks longest matching prefix against single verify. | 2–3 days | (a) matched-prefix bookkeeping bit-exact vs serial decode (Mac); (b) on-device accept rate within ±5 % of Mac projection; (c) manual quality spot-check on 5 prompts per category |
| B2 | Rolling-accept gate per drafter. Fall back to single-token decode below threshold. | 0.5 day | no regression vs baseline on any workload |
| B3 | Runtime hints V6-1 (`reshapeFrequency = .infrequent`) + V6-2 (`MLComputePlan` warm-pool). | 1 day | no correctness change |
| B4 | MLComputePlan audit on the union path. | 0.5 day | ANE placement ≥ 99 % |
| B5 | Measure actual verify chunk cost on iPhone (current 1.7× estimate is unvalidated). | 0 (pure measurement on the B device trip) | real ms/dispatch logged |

**Projected outcome:** chat ~30 tok/s (cross-vocab anchor), other
categories 57–63 tok/s (prompt-lookup anchors), mixed average ~48
tok/s. Exit: chat ≥ 50 tok/s OR clear diagnosis of why the projection
breaks on-device.

One bundled iPhone trip for the whole phase.

> **B1 exit criterion update (2026-04-15).** Previous wording was "bit-exact
> @ temp=0 vs serial decode". Mac verifier on the union PR (#54) showed
> that the K=3 batched verify chunks and K=1 decode chunks diverge at the
> fp16 level by design — the same drift would affect the existing MTP /
> CrossVocab paths, not introduced by Phase B. Strict byte-equivalence is
> not achievable under the current verify chunks. The relaxed three-part
> criterion (matched-prefix bookkeeping bit-exact, accept rate within
> ±5 %, manual quality spot-check) preserves the correctness signal
> without demanding something unattainable. A separate investigation to
> close the K=3↔K=1 numerical gap is filed in
> `docs/PRIORITY_ROADMAP.md` (item 11c) — could lift accept rates by
> aligning verify and decode argmaxes.

#### Two-drafter confluence on item 11c (2026-04-15 evening)

Two independently-validated drafter approaches hit the same device-side
regression wall. The drafters themselves are correct — Mac equivalence
tests pass for both — yet each produces net-negative speed on iPhone
17 Pro. The common dependency is the target's `verify_qK` / `decode_q1`
multi-function pair (item 11c).

| Drafter | Mac validation | iPhone tok/s | iPhone symptom | Source doc |
|---|---|---|---|---|
| Cross-vocab Qwen 2.5 0.5B (PR #54 Union) | bookkeeping bit-exact in `union-bitexact --mode fallback-only` | ~1.8 (baseline 31) | 25 s TTFT, coherent fragments echoing earlier context | `docs/SESSION_STATE.md` Phase B Task 2 |
| MTP Path C self-trained K=2 modules | PT↔CoreML argmax parity via `verify_coreml_equiv.py` | 16.3 (baseline 31) | Japanese prompt degenerates into single-token loop after ~5 tokens | `docs/MTP_PATH_C_FINDINGS.md` |

Path A (TFLite MTP extraction, `docs/MTP_INTEGRATION_RESULTS.md`) is a
related but *distinct* failure: parked earlier for target-distribution
mismatch (W4A8 training vs HF fp target), not verify-chunk drift.
Mentioned for context, not counted as 11c evidence.

The break-even arithmetic from `MTP_PATH_C_FINDINGS.md` §4.1 shows why
this is load-bearing: per-cycle verify cost ≈ 1.8–2.3× decode on ANE,
so speculation needs ~77 % acceptance to match baseline. Training hits
~38 %; closing 11c would push the multiplier toward ~1.3× and the
break-even to ~40 %, which existing drafters already clear.

#### ~~Revised Phase B priority ordering (post-#57, pending iPhone trip)~~ RETRACTED by PR #62

> **Retracted 2026-04-15 late (PR #62).** The branching chart below
> assumed "iPhone live accept << Mac oracle replay" implied item 11c's
> fp16 drift was load-bearing. Mac reproduction via `coreml-llm-smoke
> UNION_TRIP=1` shows the same 3–9× gap on Mac (cpuAndGPU, no ANE).
> The gap is bench methodology (oracle-replay vs target-argmax), not
> device divergence. See `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` for
> the full analysis. Kept inline for history; do not act on it.

<details>
<summary>Retracted chart (expand for historical context)</summary>

```
iPhone trip with PR #57 env vars on (SPECULATIVE_PROFILE=1, COMPUTE_PLAN_AUDIT=1)
  → Measure drafter-vs-verify ms distribution (grounds the break-even math).
  → Simultaneously measure accept-rate ceiling: run Union with PLD only,
    compare iPhone per-burst accept rate against Mac oracle-replay in
    eval/accept-rate-bench-v2.json.
  ┌──────────────────────────────────────────┐
  │ iPhone accept rate << Mac oracle replay  │  → item 11c confirmed load-bearing.
  ├──────────────────────────────────────────┤
  │ iPhone accept rate ≈ Mac oracle replay   │  → 11c is NOT the bottleneck.
  └──────────────────────────────────────────┘
```

</details>

#### Actual Phase B priority ordering (post-PR #62, updated post-B.3)

1. ~~**Rebuild accept-rate-bench with target-argmax mode**~~ —
   **DONE 2026-04-15** (v3). `decode_q1`/`verify_qK` fp16 drift is
   real on Mac but does not explain the 3–9× live gap. See
   `docs/PHASE_B_V3_ARGMAX_FINDINGS.md`.
2. ~~**Chain-following argmax mode**~~ — **DONE 2026-04-15** (v4).
   Drafter proposals in verify slots 1..K-1 reproduce the live gap
   directionally on code/qa/summary. v4 attributed this to fp16
   sensitivity of batched verify_qK to slot 1..K-1 content.
   **Mechanism attribution updated 2026-04-15 by PR #72 (B.3):**
   batched fp16 is NOT the mechanism — K serial `decode_q1` calls
   reproduce the chain gap within noise, so joint-K-token compute
   isn't load-bearing. The mechanism is semantic: verify's KV
   write-through at P+1..P+K-1 contaminates subsequent target
   argmaxes. See `docs/PHASE_C_TIGHTENING_FINDINGS.md` and the
   UPDATE callout in `docs/PHASE_B_V4_CHAIN_FINDINGS.md`.
3. **Union-shape decision.** Per v4 §"Implications":
    - **PL-only Union:** rejected (net-regression on 3/4 categories
      in chain-mode numbers).
    - **Raise rolling gates:** viable short-term — preserves baseline,
      no speculative gain but no regression.
    - **Defer until Mirror SD + verify re-quantisation:** most likely
      right answer. Mirror SD alone doesn't fix the fp16 batched-verify
      sensitivity; the verify chunks need tighter numerics.
4. **Remaining chat CV gap** (v4 chain 2.31 vs live 1.34): investigate
   by instrumenting `DrafterUnion.speculateStep` directly
   (rolling-gate, bootstrap, CV state drift). Not bench-harness work.
4. **Then** iPhone trip to confirm Mac finding replicates on ANE
   hardware. Expected: same ~30 % regression pattern with current
   Union defaults off. Bundle with B4 MLComputePlan audit.
5. Item 11c investigation is deprioritised (not the Phase B driver).
   Revisit only if Mac live vs iPhone live shows a gap — which would
   genuinely implicate ANE fp16 drift. v3 confirmed the drift exists
   on Mac but does not dominate live accept rates.

The MTP session is working 11c directly on `feature/mtp-speculative-v1`;
coordinate with them before duplicating any verify-vs-decode argmax
diff work. Their Mac-first verification pattern (`verify_coreml_equiv.py`,
`verify_accept_logic.py`, `precompute.py` forward hooks; all under
`conversion/train_mtp_modules/`) is the methodological baseline.

### Phase C — Mirror Speculation + async dispatch infrastructure

Goal: break past 48 tok/s mixed average.

| # | Task | Effort | Exit criterion |
|---|---|---|---|
| C1 | **I1** Async ANE dispatch queues. Ping-pong IOSurface buffers between consecutive chunks. | 4–6 days | soak test 512 steps bit-exact, chunk overlap visible in profile |
| C2 | **I2** GPU drafter execution path. Compile drafter with `computeUnits = .cpuAndGPU`, IOSurface-backed tensor handoff. | 3–4 days | drafter wall-clock when run on GPU matches expectation |
| C3 | **I3** KV direct-write in `commitAccepted`. Eliminates the re-run T=1 decode per accepted token. | 1–2 days | verify tok/s doesn't lose the speculative-accept benefit |
| C4 | **T2** Mirror SD — drafter concurrent with ANE verify via I1+I2. | 2–3 days | drafter cost hidden in verify shadow |

**Projected outcome:** chat → 44 tok/s, other 57–63, mixed average
~56 tok/s. Ties Google LiteRT-LM iOS.

One or two iPhone trips (soak + final bench).

### Phase D — staged chunk pipelining + composition

Goal: decisive beat over Google.

| # | Task | Effort | Exit criterion |
|---|---|---|---|
| D1 | Staged chunk pipelining. Step N+1's chunk1 starts as soon as step N's chunk2 returns kv13/kv14. | 5–7 days | chunk3+chunk4 latency hides one dispatch per step |
| D2 | DISCO dynamic K per burst (adaptive speculative K based on recent accept rate). | 1 day | measurable on code where long matches are common |
| D3 | Y-tree (Sequoia) verify topology if we revisit K > 3. | 1–2 days offline DP | theoretical model matches bench |

**Projected outcome:** mixed average 70–80 tok/s. ~1.3–1.4× Google.

### Final phase — stretch

Unlocks only if everything above lands and 8K ctx or further speed
becomes the priority. Listed in `docs/PRIORITY_ROADMAP.md` under
Phase 3 / Phase 4 / Phase 5 plus `docs/UNEXPLORED_APPROACHES_V6.md`:

- **W2-rotated palettization** (V6-6 SpinQuant/QuaRot + 2-bit weights).
  Decode is dispatch-bound today, so only matters once compute or
  bandwidth becomes the bottleneck — realistic after Phase D lands.
- **MoE retrofit** (dense → 4-expert sparse via distill). 5–10 days
  training on A100. Only worth pursuing if the 8K ctx goal resurfaces
  and dense compute dominates.
- **BitNet b1.58** (ternary weights from QAT). Research-grade. Risk
  of quality collapse. Only if model size becomes the bottleneck.
- **GPU prefill via MLX-Swift** (Phase 5 item 27). TTFT 13s → ~1s.
  Separate axis from decode tok/s but matters for UX. 7–10 days.
- **System prompt KV cache persistence** (Phase 5 item 29). Multi-turn
  UX win, not decode tok/s.

All final-phase items are optional decisions; none is in the critical
path to beating Google at 2K. Re-evaluate when Phase D exits.

## Current open PRs and branches

See `docs/SESSION_STATE.md` for the live list. Summary at handoff:

- **Merged (2026-04-15)**: PR #54 — Phase B Task 1 DrafterUnion,
  back-ported KV-hole + carry-double-emit fixes for CV/MTP,
  CrossVocabDraft ctx auto-detect. Defaults
  `drafterUnionEnabled = false` / `crossVocabEnabled = false`.
- **Merged (2026-04-15)**: PR #57 — drafter perf instrumentation
  (env-var gated per-phase timing logs + ComputePlanAudit extended
  to Qwen drafter). Ready for the next iPhone trip.
- **Draft / held**: PR #33 (0d prefill bypass — 6× regression,
  fresh-eyes investigation).

## Key discipline reminders

1. **Mac is for correctness + accept-rate measurement. iPhone is for
   tok/s.** Shipping numbers are iPhone-measured.
2. **Code PRs hold until an iPhone regression check clears.** User
   approval alone is insufficient; measurement on device is.
3. **Docs-only PRs auto-merge is fine.**
4. **Bundle device trips.** One trip ≥ one data point, ideally
   several.
5. **Avoid the "projection" English word in running prose** — it
   hides "推定" under technical framing. Say "推定" or "projected"
   with caveats explicit.
