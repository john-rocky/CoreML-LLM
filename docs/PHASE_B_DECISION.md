# Phase B decision — post-v4

**Status:** 2026-04-15 late. Decides the three Union-shape candidates
from `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` §"Implications" based on
v4's chain-mode numbers (`docs/PHASE_B_V4_CHAIN_FINDINGS.md`).

---

## Decision

**Phase B ships with Union defaults OFF (status quo on main) and no
further speculative-decode work on the current verify chunks.** Phase C
gating item promoted: **verify-chunk numerical tightening** (not just
Mirror SD) is now the unlock for any speculative strategy.

Concrete:

1. `drafterUnionEnabled = false` and `crossVocabEnabled = false` stay as
   the defaults. Main is baseline-clean at 31.4 tok/s on iPhone 17 Pro
   and 32–33 tok/s on Mac Studio.
2. **No further drafter / Union tuning** until the verify path is
   numerically aligned. v4 showed that the bench-vs-live gap is
   dominated by batched-`verify_qK` fp16 sensitivity to slot 1..K-1
   content — a target-side issue no drafter can work around.
3. **PL-only Union is rejected.** v4 chain numbers: pl-n2 E[tok/burst]
   on chat 1.48, code 2.01, qa 1.00, summary 1.00. Break-even needs
   ≥ 2.6. Net-regression on 3 of 4 categories.
4. **Raise-gates is noted but not implemented.** Under Union default
   OFF, rolling-gate tuning affects no out-of-box user. If opt-in users
   ever enable Union, the existing gates (CV 0.20, PL 0.05) already
   collapse the system to baseline fast enough on the v4 data. A
   stricter default would mask a potential future speedup rather than
   earn one. File if a user reports regression.
5. **Verify-chunk re-quantisation** is the next speculative-decode
   unlock. Roadmap item 11c is elevated: the specific failure mode is
   batched verify_qK's slot-interaction fp16 drift, not the narrower
   K=3↔K=1 argmax mismatch. Any re-quantisation attempt must target
   this jointly. Filed to Phase C's critical path alongside Mirror SD.

---

## Why Mirror SD alone doesn't unlock Phase B

Mirror SD hides drafter cost by running the drafter concurrent with
target verify (Phase A5 / Phase C plan). It does not change what
`verify_qK` returns. v4 shows that even if the drafter is free, the
drafter's proposals induce fp16 drift in verify's argmax chain, and the
drafter loses byte-exact n-gram matches against that drifted chain. The
acceptance ceiling is set by the verify chunk, not the drafter speed.

Therefore Mirror SD without verify-chunk tightening would yield ties at
baseline (drafter cost hidden, accept rate still ≈ chain numbers). Any
speculative *gain* requires closing the fp16 drift first.

---

## Phase B exit criterion

The B1 exit criterion (docs/HANDOFF.md §"Phase B") is declared met in
its relaxed form (matched-prefix bookkeeping bit-exact, tolerance
on accept rate, quality spot-check). No further Phase B tok/s exit
criterion will be pursued. Phase B closes here.

Phase B did deliver:

- `DrafterUnion` orchestrator (PR #54, on main, default OFF).
- Accept-rate bench v2/v3/v4 harness with oracle / argmax / chain modes
  (PRs #45, #65, #66).
- Mac-first verification tools: `union-bitexact`, `ComputePlanAudit`,
  `SpecProfile` (PRs #54, #57).
- Diagnosis of the bench-vs-live gap and its mechanism
  (`docs/PHASE_B_V3_ARGMAX_FINDINGS.md`, `docs/PHASE_B_V4_CHAIN_FINDINGS.md`).

These are useful going into Phase C. They are NOT an iPhone tok/s
improvement.

---

## Phase C gating item (new)

**C0: Verify-chunk numerical tightening.** Before any Phase C
speculative work, the verify chunks must be re-exported / re-quantised
such that `verify_qK([nextID, d0, d1])[0]` is bit-exact with
`verify_qK([nextID, 0, 0])[0]` at temp=0, up to argmax stability (not
bit-exact in logits, just argmax-stable across slot content). Candidate
approaches:

- ~~Explicit fp32 upcast on the logit projection in the verify
  function.~~ **REFUTED 2026-04-15 by PR #72 (Track B approach 3).**
  See `docs/PHASE_C_TIGHTENING_FINDINGS.md`.
- ~~Re-quantise verify weights with accumulation-order control.~~
  **REFUTED 2026-04-15 by PR #72 (Track B approach 3).** Both of
  these are strict subsets of what replacing batched `verify_qK`
  with K serial `decode_q1` calls already removed — and that swap
  did not close the chain gap (cross-vocab code stayed at 1.01,
  chat moved 2.31 → 2.09 within noise).
- ~~Switch to `MLMultiArray` inputs with explicit fp32 zeroing so
  batched compute path takes a canonical ordering.~~ Same class —
  also predicted no-op by PR #72.
- **Output-space tolerance (Track A).** Instead of byte-exact
  argmax, compare `logit[argmax(chain)]` vs `logit[argmax(verify[0])]`
  and accept if the margin is below threshold. A measurement-only
  relaxation; doesn't require verify to be deterministic. Bench-side
  patch is ready on PR #73 and lands independently of any chunk
  re-export.
- **Delayed KV write-through (verify-protocol redesign).** Verify
  computes logits but commits KV only after the acceptance decision
  is made, so subsequent target argmaxes don't condition on
  drafter-proposal residue at positions P+1..P+K-1. Multi-week work
  (verify chunk I/O contract re-designed, write-through semantics
  moved into Swift), but directly addresses the semantic mechanism
  PR #72 isolated.

### 2026-04-15 update — B.3 refutation

PR #72 (`experiment/c0-verify-serial`) replaced the batched
`verify_qK` call with K serial `decode_q1` calls and ran the v4 chain
bench. The cross-vocab code category stayed at E[tok/burst] = 1.01
(oracle = 2.63), qa stayed at 2.04, summary at 1.00, chat moved
2.31 → 2.09. That rules out batched-compute fp16 as the chain-gap
mechanism: if joint-K-token fp16 ordering were the cause, removing
batched compute entirely would have closed the gap.

The mechanism is instead **semantic**: verify writes drafter proposals
into the KV cache at positions P+1..P+K-1 *before* acceptance is
decided. Subsequent target argmaxes condition on the contaminated
cache. Oracle replay avoids this because it never calls verify.
Serial decode reproduces it exactly (same write sites, sequenced).
Full detail in `docs/PHASE_C_TIGHTENING_FINDINGS.md`.

Implication: the C0 candidate list collapses to **(a) output-space
tolerance** (cheap; Track A / PR #73 wiring) and **(b) verify-protocol
redesign** (multi-week; delays KV write-through until after
acceptance). The fp32-upcast / accumulation-order variants are dead
ends.

Effort: (a) is a bench-measurement exercise once verify chunks emit
top-K logits; (b) is a chunk re-export plus a Swift engine re-design.
Blocking item for Phase C.

---

## What this means for the 56 tok/s target

`MOBILE_2K_COMPETITIVE_PLAN.md` targets 56 tok/s (Google LiteRT-LM
parity). Phase B's projected path there was via speculative decoding.
Post-v4, that path is blocked at verify-chunk tightening. Options:

- **Accept delay.** Finish C0 → Mirror SD → measurable speculative
  gain. Multi-week path, well-defined.
- **Switch priorities.** Non-speculative wins that Phase A5 deferred:
  staged chunk pipelining (Phase D1 in HANDOFF), GPU prefill via
  MLX-Swift (Phase 5 item 27). These don't depend on verify-chunk
  numerics and may close part of the gap independently.
- **Revisit the target.** 56 tok/s is a specific model/prompt
  configuration; at 2K ctx on 17 Pro with current chunks, the
  fallback-only path delivers ~31–32. Whether the 25-tok/s gap is
  entirely speculative-attributable vs also chunk-dispatch overhead
  hasn't been decomposed cleanly. A per-chunk audit (item B4, not
  yet run) would clarify.

Recommend: schedule C0 investigation as highest-priority next session
and run the chunk-dispatch audit opportunistically on whatever iPhone
trip follows.

---

## Related

- `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` — originator of the three
  candidates this doc decides.
- `docs/PHASE_B_V3_ARGMAX_FINDINGS.md` — ruled out `decode_q1` vs
  `verify_qK` chain-level drift.
- `docs/PHASE_B_V4_CHAIN_FINDINGS.md` — identified the actual
  mechanism and provides the empirical grounding for this decision.
- `docs/PHASE_A5_DECISION.md` — historical; its projections are
  superseded.
- `docs/PRIORITY_ROADMAP.md` item 11c — upgraded from "correctness
  only" to "Phase C gating item".
