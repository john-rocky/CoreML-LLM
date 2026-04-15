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

Headline number: Route B alone gets us to ~56 tok/s (ties Google);
Mirror SD (Phase C/D async dispatch) is load-bearing for the decisive
beat.

**Phase B — NEXT.** One bundled iPhone trip. Exit criterion: chat
≥ 50 tok/s, other categories ≥ 60.

---

## Open / draft PRs (end-of-session)

| PR | Status | What | Why blocked / what next |
|---|---|---|---|
| [#45](https://github.com/john-rocky/CoreML-LLM/pull/45) | **open, awaiting code approval** | Mac accept-rate bench + A1–A4 wiring (incorporates `origin/feature/route-b-cross-vocab-drafter` via merge) | User approval. It's the foundation for Phase B; merging is cheap and lets the bench keep running |
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
   - Exit: bit-exact output at temp=0 vs serial decode, ≥ 50 tok/s
     chat, ≥ 60 tok/s other categories.
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

Order 1 → 3 → 2 → 4. Item 1 is the critical path; 3 is free-standing
and risk-free, good to land first as a warm-up.

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

## Housekeeping

`model_config_8k_backup.json` at repo root is an untracked leftover;
safe to remove next session if no one has claimed it. `.claude/`
directories (agent worktrees, session cache) should stay in
`.gitignore` — they already are.
