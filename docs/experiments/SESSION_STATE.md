# Session state — rolling handoff

**Last updated:** 2026-04-15 late (post v4 chain-mode bench).

Keep this doc short and always-current. Any engineer / agent picking up
should be able to cold-start from it in under 10 minutes.

---

## Where we are

**Plan:** beat Google LiteRT-LM iOS 56 tok/s @ 2K. See
`docs/MOBILE_2K_COMPETITIVE_PLAN.md` for the strategic framing and
`docs/MAC_FIRST_EXECUTION_PLAN.md` for the phased itinerary.

**Phase A — DONE.** Accept-rate measurement picked the winning drafter
combination. Detailed decision in `docs/PHASE_A5_DECISION.md` §Decision.

**Phase B Task 1 — MERGED (PR #54, commit `ec7616b`) but defaults
flipped OFF.** The DrafterUnion orchestrator + KV-hole and
carry-double-emit fixes for the legacy MTP / CrossVocab engines + the
CrossVocabDraft ctx auto-detect all shipped together. iPhone 17 Pro
measurement with Union ON delivered **1.8 tok/s decode and 25+ second
TTFT on short prompts** — roughly 10× below Mac projection. Output
quality also degraded (coherent fragments echoing earlier context).
To prevent any model bundle with `cross_vocab/qwen_drafter.mlmodelc`
auto-activating the regression, **`crossVocabEnabled` and
`drafterUnionEnabled` both default `false` on main**. Re-enable is
opt-in until the perf investigation closes.

**Phase B Task 2 (next) — drafter perf investigation.** See task list
below. Gates whether defaults can be flipped back on and whether the
A5 30–63 tok/s projection is recoverable, or whether Phase B has to
pivot to MTP / Mirror-SD without ever shipping the cross-vocab path.

> **Coordination note for sibling sessions.** Drafters being default-off
> means any model bundle including `cross_vocab/` assets currently
> behaves as a pure baseline target — no speculation. Don't assume a
> regression on your branch comes from your changes if you're seeing
> baseline-tier numbers; check `crossVocabEnabled` / `drafterUnionEnabled`
> first. The MTP session's upcoming Path C drafter trip is unaffected
> (`mtpEnabled` still defaults true and lands ahead of the cross-vocab
> path in the engine selector).

---

## Open / draft PRs (end-of-session)

| PR | Status | What | Why blocked / what next |
|---|---|---|---|
| [#45](https://github.com/john-rocky/CoreML-LLM/pull/45) | **merged 2026-04-15** | Mac accept-rate bench + A1–A4 wiring (incorporates `origin/feature/route-b-cross-vocab-drafter` via merge) | iPhone baseline cleared at 31.4 tok/s; merged. |
| [#54](https://github.com/john-rocky/CoreML-LLM/pull/54) | **merged 2026-04-15 evening** (`ec7616b`) | Phase B Task 1 — DrafterUnion + back-port of legacy MTP/CV bookkeeping fixes (KV-hole, carry double-emit) + CrossVocabDraft ctx auto-detect + Mac bit-exact verifier | Defaults flipped OFF in same merge after iPhone returned 1.8 tok/s + degraded quality; perf investigation owns re-enabling. |
| [#55](https://github.com/john-rocky/CoreML-LLM/pull/55) | **merged 2026-04-15** | Docs: relax Phase B B1 exit criterion (matched-prefix bookkeeping bit-exact + accept rate ±5 % + manual quality + tok/s) and file roadmap item 11c | n/a |
| [#33](https://github.com/john-rocky/CoreML-LLM/pull/33) | **draft** | 0d prefill bypass | 6× decode regression on device (2026-04-14); root cause not isolated. Don't merge as-is. Fresh-eyes investigation later |
| #27 | open | MTP TFLite path fix | Owned by another session |
| #16 | open | SuffixDecoding implementation | Superseded in effect by `feat/accept-rate-bench-phase-a1` which measured suffix = prompt-lookup in single-turn. Re-evaluate if someone wants a multi-turn session bench |

---

## Branches with useful work (not merged yet)

| Branch | Where | What it has |
|---|---|---|
| `feat/accept-rate-bench-phase-a1` | `origin` | PR #45. Merged. Safe to delete. |
| `feat/drafter-union` | `origin` | PR #54. Merged as `ec7616b`. Safe to delete. |
| `feat/route-b-task1-prompt-lookup-wiring` | local worktree only (`.claude/worktrees/agent-ab599231`) | Folded into PR #54 (commit 1). Safe to drop the worktree. |
| `feature/route-b-cross-vocab-drafter` | `origin` | Already absorbed via PR #45. Safe to delete. |
| `feature/qk-multi-function-verifier` | `origin` + local worktree | Verify write-through KV work (other session). Referenced in roadmap Phase 2 Shared item 9 |
| `feat/0d-prefill-bypass` | `origin` | PR #33 draft content |

---

## Phase B task list (post-#54)

Phase B Task 1 (Union orchestrator) shipped as PR #54. The remaining
work is gated by the perf investigation (Task 2 below). All other
items continue to share the next bundled iPhone trip.

1. ~~**Union-of-drafters orchestrator**~~ — **MERGED 2026-04-15** as
   PR #54 (`ec7616b`). Bookkeeping bit-exact (criterion (a)).
   Defaults flipped OFF after iPhone returned 1.8 tok/s.
2. **Drafter perf investigation — Qwen 10× slower on iPhone.**
   *Gates re-enabling default-on for `crossVocabEnabled` /
   `drafterUnionEnabled`.* iPhone 17 Pro with Union ON delivers
   ~1.8 tok/s decode and 25+ s TTFT vs 30–63 tok/s Mac projection.
   Two non-overlapping signals to collect on a single device trip:
   - **Per-phase timing logs** in `CrossVocabSpeculativeEngine`
     (and ideally the parallel `MtpSpeculativeEngine` so the MTP
     session shares the format) — ms in drafter forward vs verify
     vs fallback per burst, env-var gated so baseline is unaffected.
     Tells us *how much* time is being lost to the drafter.
   - **MLComputePlan audit on `cross_vocab/qwen_drafter.mlmodelc`** —
     extend `ComputePlanAudit.run` to cover the drafter, surface the
     placement table on iPhone. Tells us *where* the drafter ran
     (CPU vs GPU vs ANE). A 500 ms drafter forward is interpretable
     only once you know whether the fix is "force compute units" or
     "model surgery".
   - Exit: log artifact + placement table from one device run; we
     can choose between the two failure modes (CPU fallback vs slow
     GPU) without burning a second trip.
   - ~~**Accept-rate ceiling check (bundle in same trip).**~~
     **Retracted 2026-04-15 late by PR #62.** The branching logic
     below assumed comparing iPhone live vs Mac oracle-replay would
     reveal 11c as the bottleneck. PR #62's Mac reproduction
     (`coreml-llm-smoke UNION_TRIP=1`) shows the same live-vs-oracle
     gap on Mac (cpuAndGPU, no ANE). The gap is bench methodology,
     not ANE fp16 drift. Item 11c is deprioritised as the Phase B
     driver. See `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md`.

     Kept for history:
     > Path C shelving (`docs/MTP_PATH_C_FINDINGS.md`) gives an
     > independent corroboration that iPhone acc0 may be capped by
     > item 11c rather than drafter quality. While the trip is
     > running with `SPECULATIVE_PROFILE=1`, also run Union in
     > PLD-only mode (CV disabled) and compare on-device per-burst
     > accept rate against Mac oracle-replay. Decision branch:
     > – iPhone acc ≪ Mac acc → item 11c. – iPhone acc ≈ Mac acc →
     > Task 3 + Phase C remain critical path.
3. **Bootstrap optimisation** — gated on Task 2's data. Two
   candidates depending on what the timing log says:
   (i) batched Qwen prefill shape so the N sequential `consume()`
   calls during prompt replay collapse into one or a few prefill
   chunks; (ii) skip bootstrap entirely and let rolling accept
   recover over a few bursts (cheap if accept rate stays high).
   Likely ~3 days Swift after Task 2 lands.
4. **Output quality investigation** — concurrent with Task 2.
   Visible degradation observed on iPhone is *not* the
   carry-double-emit / KV-hole bugs (those were back-ported and
   bookkeeping is bit-exact), so it's plausibly Finding 2 / roadmap
   item 11c (K=3↔K=1 verify-chunk drift) surfacing as actual
   coherence loss, or a subtler carry-semantics mismatch escaping
   the Mac verifier. Priority bumps if Task 3 fixes perf — quality
   is independent.
5. **Runtime hints** V6-1 (`optimizationHints.reshapeFrequency = .infrequent`)
   + V6-2 (`MLComputePlan` warm-pool). See V6.md for details.
   - Exit: no correctness change; hint loadable at init.
6. **iPhone-measured verify chunk cost**. The A5 decision doc
   projected ~52 ms per verify dispatch; actual Mac number from PR
   #62 logs is ~31 ms (close to decode cost, not 1.7×). iPhone
   number still unmeasured but less load-bearing now that A5 tok/s
   ceilings are superseded. Log opportunistically when next trip
   runs.

Critical path: Task 2 → Task 3 → re-enable defaults → re-measure
against B1's (b)(c)(d) criteria. Tasks 4, 5, 6 piggyback on the same
device trips opportunistically.

---

## Gotchas / non-obvious bits discovered this session

1. **Mac CoreML can't load `.mlpackage` via `MLModel(contentsOf:)`
   directly** — you get `"Failed to open file: .../coremldata.bin.
   It is not a valid .mlmodelc file"`. Pre-compile with `xcrun coremlc
   compile foo.mlpackage /tmp/out/` and use the `.mlmodelc` output.
   iOS handles this automatically; Mac does not.
2. ~~**Qwen 2.5 0.5B bundled in sibling repo was compiled at
   `contextLength = 512`**~~ — **resolved in PR #54.**
   `CrossVocabDraft.init` now auto-detects ctx from the model's
   `causal_mask` input shape and ignores the caller-supplied
   `contextLength` if it conflicts. The accept-rate bench's
   hard-coded 512 still works but is no longer load-bearing.
3. **Oracle replay ≈ real verify semantics — but only in exact
   arithmetic.** At temp=0, verify's argmax at P+k+1 depends only on
   history ending with the drafter's d_k; if d_k matches the true
   emitted[P+k] (by chain-accept definition), the two agree. **v4
   empirically contradicts this at fp16.** Drafter proposals in
   verify slots 1..K-1 cause batched-computation fp16 drift that
   changes `argmax[0]` even when slot 0 is the same. This is how the
   bench-vs-live 3–9× gap arises.
   See `docs/PHASE_B_V4_CHAIN_FINDINGS.md`. Don't count "all K
   positions independently" either — that over-counts. Use chain-mode
   or `--mode chain` in the bench to get live-equivalent numbers.
   - **2026-04-15 (PR #72 / B.3) — mechanism is semantic, not
     numerical.** Replacing batched `verify_qK` with K serial
     `decode_q1` calls still reproduces the chain gap (cross-vocab
     code 1.01, chat 2.31→2.09). So the v4 fp16-ordering speculation
     is refuted; the chain gap comes from verify writing drafter
     proposals into KV at positions P+1..P+K-1 *before* acceptance
     is decided. Subsequent target argmaxes then condition on a
     contaminated cache. Oracle replay avoids this (no verify call);
     serial decode reproduces it (same writes, sequenced). Not
     fixable by tightening verify numerics. See
     `docs/PHASE_C_TIGHTENING_FINDINGS.md`.
4. **Per-model caches.** Staging directories under
   `~/Downloads/coreml-llm-artifacts/` contain live model files. Don't
   `rm -rf` them casually. `staging-2k-fast-prefill` currently has the
   installed cross-vocab drafter under `cross_vocab/`.
5. **`.mlpackage` Qwen symlink.** `setup_cross_vocab_drafter.py` links
   `cross_vocab/qwen_drafter.mlpackage` to the sibling's
   `qwen2.5-0.5b/model.mlpackage`. After `xcrun coremlc compile` the
   staging has `cross_vocab/qwen_drafter.mlmodelc` (compiled from that
   symlink). Both can coexist; code prefers the `.mlmodelc`.
6. **Dependency on sibling repo.** The bench reads `qwen2.5-0.5b`
   from `/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/output/`
   — that's the other working tree (MTP / training session). Don't
   delete it.

---

## Key numbers to remember

- iPhone 17 Pro 2K baseline (main, drafters OFF — current default):
  **31.4 tok/s**. Per-chunk c1=5.9 c2=6.8 c3=8.1 c4=10.4 ms.
- Google LiteRT-LM iOS @ 2K Gemma 4 E2B: **56 tok/s** (user-supplied).
- **Mac Union live (PR #62 repro, `coreml-llm-smoke UNION_TRIP=1`):**
  code 19.3, chat 17.8, qa 15.2, summary 21.2 tok/s vs Mac baseline
  ≈ 32. Net regression on all four categories.
- ~~Phase A accept-rate winners: chat=cross-vocab 2.31, code=pl-n3 2.94,
  qa=cross-vocab 3.17, summary=pl-n2/suffix 3.26.~~ **Superseded by
  PR #62.** These were oracle-replay numbers; live draft-rate on Mac
  is 0.059–0.212 — 3–9× lower. Task #2 rebuilds the bench with a
  target-argmax mode.
- ~~Phase B theoretical target: 44–63 tok/s with Mirror SD on, 30–63
  serial.~~ Deferred; ceilings re-compute once v3 bench lands.

---

## Merge discipline (corrected 2026-04-15)

Previous sessions auto-merged several code PRs based on "user approved"
as sufficient. That was wrong. **Code PRs require an iPhone baseline
check before merge**, regardless of user approval, because speed
regressions only show up on device and the product's headline number
is iPhone tok/s. Docs-only PRs keep the auto-merge permission.

Rules going forward:

- **Docs-only PR** (content under `docs/` only): auto-merge OK after
  user approval.
- **Code PR touching Sources/ or conversion/**:
  1. Mac build + tests must pass.
  2. Land it on a feature branch and push.
  3. At least the next bundled iPhone trip must measure baseline
     regression on a clean prompt; log the tok/s before / after.
  4. Merge *only* after iPhone baseline check clears.
  5. If the change is genuinely runtime-free (e.g. only populates
     an observable array) Mac CoreML on Apple Silicon is acceptable
     as a proxy, but document the Mac numbers in the PR body.

Retroactive audit of 2026-04-14 / 2026-04-15 merges:

- **PR #45** (accept-rate-bench + CrossVocab merge + runtime hooks):
  merged without an iPhone check. iPhone baseline measurement
  2026-04-15 confirmed regression-free (31.4–32.8 tok/s steady-state
  with drafters held off). No revert needed.
- **PR #54** (DrafterUnion + back-ports + ctx auto-detect): iPhone
  baseline check done before merge; revealed 1.8 tok/s + degraded
  quality with Union ON. Merged anyway with **defaults flipped OFF
  in the same merge** so the regression is opt-in only and main
  remains baseline-clean. Re-enabling is gated by the Phase B Task 2
  perf investigation.
- All other 2026-04-14/15 merges are docs-only.

## Housekeeping

`model_config_8k_backup.json` at repo root is an untracked leftover;
safe to remove next session if no one has claimed it. `.claude/`
directories (agent worktrees, session cache) should stay in
`.gitignore` — they already are.
