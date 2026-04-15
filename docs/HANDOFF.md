# Next-session handoff

**Last updated:** end of 2026-04-15 session.

## Read this first

To resume cleanly, the next session should:

1. Open this file (`docs/HANDOFF.md`) — takes 5 minutes.
2. Skim `docs/SESSION_STATE.md` for the exact PR / branch / task state.
3. Read `docs/MAC_FIRST_EXECUTION_PLAN.md` for the phased itinerary
   and the Mac-vs-iPhone authority split.
4. Read `docs/PHASE_A5_DECISION.md` for the drafter-union decision
   and per-category projections.
5. Start on Phase B Task 1 (Union orchestrator).

No need to touch MTP Path A, PR #17, or PR #33 — all deprioritised
with reasons in the roadmap's Rejected table.

## Suggested opening prompt for the next session

Copy-paste this at the start of the next `/claude` session so the
model walks in with correct framing:

> Continue the Phase B work on this repo. Read `docs/HANDOFF.md`,
> `docs/SESSION_STATE.md`, `docs/MAC_FIRST_EXECUTION_PLAN.md`, and
> `docs/PHASE_A5_DECISION.md` in that order — those cover where we
> stand and what Phase B means. Phase A (accept-rate measurement
> and drafter selection) is done; the PR #45 regression check on
> iPhone already passed at 31–32 tok/s baseline. Start with Phase B
> Task 1 — the Union-of-drafters orchestrator — and hold any code
> PR until it clears an iPhone baseline check per our merge
> discipline. Docs-only PRs can still auto-merge.

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
| B1 | Union-of-drafters orchestrator class. Runs cross-vocab + prompt-lookup{n=2, n=3} per burst, picks longest matching prefix against single verify. | 2–3 days | bit-exact @ temp=0 vs serial decode |
| B2 | Rolling-accept gate per drafter. Fall back to single-token decode below threshold. | 0.5 day | no regression vs baseline on any workload |
| B3 | Runtime hints V6-1 (`reshapeFrequency = .infrequent`) + V6-2 (`MLComputePlan` warm-pool). | 1 day | no correctness change |
| B4 | MLComputePlan audit on the union path. | 0.5 day | ANE placement ≥ 99 % |
| B5 | Measure actual verify chunk cost on iPhone (current 1.7× estimate is unvalidated). | 0 (pure measurement on the B device trip) | real ms/dispatch logged |

**Projected outcome:** chat ~30 tok/s (cross-vocab anchor), other
categories 57–63 tok/s (prompt-lookup anchors), mixed average ~48
tok/s. Exit: chat ≥ 50 tok/s OR clear diagnosis of why the projection
breaks on-device.

One bundled iPhone trip for the whole phase.

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

- **Open, user-merge-ready**: none (PR #45 already merged after
  iPhone baseline cleared).
- **Draft / held**: PR #33 (0d prefill bypass — 6× regression,
  fresh-eyes investigation).
- **Code branch not pushed**: `feat/route-b-task1-prompt-lookup-wiring`
  at SHA `18c5454`. Agent-authored Prompt Lookup decode-loop wiring.
  Drop into the Phase B union orchestrator.

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
