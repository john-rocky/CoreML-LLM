# Session state — rolling handoff

**Last updated:** 2026-04-15 (end of Phase A on MAC_FIRST_EXECUTION_PLAN).

Keep this doc short and always-current. Any engineer / agent picking up
should be able to cold-start from it in under 10 minutes.

---

## Where we are

**Plan:** beat Google LiteRT-LM iOS 56 tok/s @ 2K. See
`docs/MOBILE_2K_COMPETITIVE_PLAN.md` for the strategic framing and
`docs/MAC_FIRST_EXECUTION_PLAN.md` for the phased itinerary.

**Phase A — DONE (all Mac-side, zero device trips).** Accept-rate
measurement picked the winning drafter combination. Detailed decision
in `docs/PHASE_A5_DECISION.md` §Decision.

Headline number (PROJECTED, not yet measured on iPhone): Route B
alone gets us to ~56 tok/s at 2K (ties Google LiteRT-LM iOS); Mirror
SD (Phase C/D async dispatch) is load-bearing for the decisive beat.
Phase B must confirm this on iPhone — all shipped speed claims are
iPhone-measured, Mac is only for accept-rate / correctness.

**Phase B — NEXT.** One bundled iPhone trip. Exit criterion: chat
≥ 50 tok/s, other categories ≥ 60.

---

## Open / draft PRs (end-of-session)

| PR | Status | What | Why blocked / what next |
|---|---|---|---|
| [#45](https://github.com/john-rocky/CoreML-LLM/pull/45) | **merged 2026-04-15** | Mac accept-rate bench + A1–A4 wiring (incorporates `origin/feature/route-b-cross-vocab-drafter` via merge) | iPhone baseline cleared at 31.4 tok/s; merged. |
| [#54](https://github.com/john-rocky/CoreML-LLM/pull/54) | **open, held for iPhone baseline** | Phase B Task 1 — DrafterUnion (cv + pld-n2 + pld-n3) + Mac bit-exact verifier | Bookkeeping bit-exact (fallback-only mode); iPhone trip should record baseline + per-source picks before merge. |
| [#33](https://github.com/john-rocky/CoreML-LLM/pull/33) | **draft** | 0d prefill bypass | 6× decode regression on device (2026-04-14); root cause not isolated. Don't merge as-is. Fresh-eyes investigation later |
| #27 | open | MTP TFLite path fix | Owned by another session |
| #16 | open | SuffixDecoding implementation | Superseded in effect by `feat/accept-rate-bench-phase-a1` which measured suffix = prompt-lookup in single-turn. Re-evaluate if someone wants a multi-turn session bench |

---

## Branches with useful work (not merged yet)

| Branch | Where | What it has |
|---|---|---|
| `feat/accept-rate-bench-phase-a1` | `origin` | PR #45. Accept-rate harness + 4 drafters measured. Includes the merged cross-vocab drafter from another session |
| `feat/route-b-task1-prompt-lookup-wiring` | local worktree only (`.claude/worktrees/agent-ab599231`) | Prompt Lookup runtime wiring in `CoreMLLLM.swift`. Not pushed. Agent-authored 2026-04-14. Integrate during Phase B Union orchestrator |
| `feature/route-b-cross-vocab-drafter` | `origin` | Cross-vocab Qwen drafter + vocab map + build scripts. Already merged into `feat/accept-rate-bench-phase-a1` |
| `feature/qk-multi-function-verifier` | `origin` + local worktree | Verify write-through KV work (other session). Referenced in roadmap Phase 2 Shared item 9 |
| `feat/0d-prefill-bypass` | `origin` | PR #33 draft content |

---

## Phase B task list (for next session)

All four items bundle into one device trip. Exit criterion listed
with each.

1. **Union-of-drafters orchestrator**. New class wrapping
   `PromptLookupLoop` (exists in draft branch) + `CrossVocabSpeculativeEngine`
   (exists in merged work). Per burst: run all three drafters
   (cross-vocab, prompt-lookup n=2, prompt-lookup n=3), pick the
   proposal with the longest matching prefix against the single target
   verify call.
   - Exit (revised 2026-04-15 — see HANDOFF.md §B1):
     (a) matched-prefix bookkeeping bit-exact vs serial on Mac,
     (b) on-device accept rate within ±5 % of A5 projection,
     (c) manual quality spot-check on 5 prompts/category,
     (d) chat ≥ 50 tok/s, other categories ≥ 60.
   - **In-flight**: PR #54 (`feat/drafter-union`) implements (a) +
     orchestration; (b)–(d) need iPhone trip.
2. **Rolling-accept gate** per drafter. Cross-vocab threshold ~0.3,
   prompt-lookup ~0.1 (missing is cheap). Fall back to single-token
   decode below threshold.
   - Exit: no regression vs baseline on any workload.
3. **Runtime hints** V6-1 (`optimizationHints.reshapeFrequency = .infrequent`)
   + V6-2 (`MLComputePlan` warm-pool). See V6.md for details.
   - Exit: no correctness change; hint loadable at init.
4. **MLComputePlan audit** on the union path. Confirm ANE placement
   stays ≥ 99% under the new orchestrator.
   - Exit: placement table recorded for posterity.
5. **iPhone-measured verify chunk cost**. The A5 decision doc
   projected ~52 ms per verify dispatch; the actual number is an
   unknown that gates whether the 44–63 tok/s ceiling holds. Log a
   `[Profile]` on a verify call directly.
   - Exit: real verify ms/dispatch recorded; Phase A5 ceiling numbers
     updated with measured verify cost instead of the 1.7× estimate.
6. ~~**Regression sanity check** for PR #45's runtime changes~~ —
   **DONE 2026-04-15**. On-device measurement (backup 2K model,
   `Explain what a transformer is...` prompt) held steady-state
   at 31.4–32.8 tok/s, c1=5.5 c2=6.7 c3=8.0 c4=10.1 ms. Matches
   pre-#45 baseline within noise. PR #45 confirmed regression-free.

Order 1 → 3 → 2 → 4 → 5 → 6. Item 1 is the critical path. Items 3, 5,
6 are cheap to bundle into the same trip; 6 should run first (before
the new orchestrator is enabled) so any regression is cleanly
attributable.

---

## Gotchas / non-obvious bits discovered this session

1. **Mac CoreML can't load `.mlpackage` via `MLModel(contentsOf:)`
   directly** — you get `"Failed to open file: .../coremldata.bin.
   It is not a valid .mlmodelc file"`. Pre-compile with `xcrun coremlc
   compile foo.mlpackage /tmp/out/` and use the `.mlmodelc` output.
   iOS handles this automatically; Mac does not.
2. **Qwen 2.5 0.5B bundled in sibling repo was compiled at
   `contextLength = 512`**, not 2048. Passing 2048 to `CrossVocabDraft.init`
   triggers a MultiArray shape mismatch at the first prediction. The
   bench hard-codes 512; if someone reconverts Qwen at 2K or 8K, update
   the bench `contextLength` argument.
3. **Oracle replay ≠ real verify semantics only past the first miss.**
   At temp=0, verify's target argmax at position P+k+1 conditions on
   history ending with the drafter's d_k (NOT the true emitted[P+k]).
   Oracle replay conditions on the true history. For "chain-accept"
   semantics where we only count matches while every prior position
   also matched, the two are identical. Don't change the replay to
   "count all K positions independently" — it breaks this equivalence.
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

- iPhone 17 Pro 2K baseline (main): **31.4 tok/s**. Per-chunk
  c1=5.9 c2=6.8 c3=8.1 c4=10.4 ms.
- Google LiteRT-LM iOS @ 2K Gemma 4 E2B: **56 tok/s** (user-supplied).
- Phase A accept-rate winners: chat=cross-vocab 2.31, code=pl-n3 2.94,
  qa=cross-vocab 3.17, summary=pl-n2/suffix 3.26.
- Phase B theoretical target: 44–63 tok/s with Mirror SD on, 30–63
  serial.

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
  merged without an iPhone check. Mac Studio same-model decode
  profile pre-merge ≈ 33 tok/s (c1=5.9 c2=6.7 c3=7.5 c4=10.3), post-
  merge bench on same model ≈ 29-31 tok/s per prompt — essentially
  baseline. Evidence (not proof) of no regression. iPhone
  confirmation is Phase B task #6 (see above). If the Phase B trip
  finds a regression, revert.
- All other 2026-04-14/15 merges are docs-only.

## Housekeeping

`model_config_8k_backup.json` at repo root is an untracked leftover;
safe to remove next session if no one has claimed it. `.claude/`
directories (agent worktrees, session cache) should stay in
`.gitignore` — they already are.
