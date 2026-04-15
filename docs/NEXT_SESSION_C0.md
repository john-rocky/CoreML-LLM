# Next-session instructions — C0 (verify-chunk tightening)

**Status:** 2026-04-15 late. Written as the concrete handoff after
PR #67 (`docs/PHASE_B_DECISION.md`). Two tracks can run **in
parallel** across two sessions because they touch almost disjoint
files. Either one alone may unblock Phase C.

Both tracks share the same measurement instrument: `accept-rate-bench
--mode chain` (v4). Success = chain-mode E[tok/burst] approaches the
oracle/argmax numbers (i.e., gap closes).

---

## Shared context — read first (both tracks, 10 min)

1. `docs/PHASE_B_DECISION.md` — why we're here.
2. `docs/PHASE_B_V4_CHAIN_FINDINGS.md` §"Per-category results" and
   §"Caveats" — the baseline numbers to beat, and the known
   methodology caveats.
3. `Sources/CoreMLLLM/ChunkedEngine.swift:641-800` — current
   `verifyCandidates` implementation. Note it returns `[Int32]`
   (argmax only); no logits exposed.
4. `Sources/accept-rate-bench/Bench.swift` — `runChainMode` is the
   live-equivalent measurement. Read it end-to-end (~100 lines).

Reference numbers (from v4, cross-vocab-qwen E[tok/burst], K=3, mode=chain):
```
chat    2.31    code    1.01    qa    2.04    summary    1.00
```
Oracle targets: 2.31 / 2.63 / 3.17 / 3.12 respectively. Closing the
gap in at least 2 categories counts as partial success; all 4 counts
as unblocking Phase C.

---

## Track A — loosen (output-space tolerance)

**Session label:** `feat/c0-tolerance-bench`
**Estimated effort:** 3–4 h
**Touches:** `Sources/CoreMLLLM/ChunkedEngine.swift`,
`Sources/CoreMLLLM/CoreMLLLM.swift`, `Sources/accept-rate-bench/Bench.swift`.
**Risk:** changes `verifyCandidates` return type; must not break
existing callers (`DrafterUnion`, `MtpSpeculativeEngine`, etc.).

### Goal

Measure whether relaxing the acceptance test from byte-exact argmax to
a tolerance-based check closes the v4 chain-mode gap. Hypothesis: most
"misses" under chain mode have the drafter's proposed token as a
high-ranked alternative (top-2 or within ε of argmax), not a complete
miss. If true, tolerance acceptance recovers the accept rate.

### Implementation plan

1. **Expose top-K logits from verifyCandidates.** Options in order of
   reversibility:
   a. Add a SECOND public method `verifyCandidatesWithLogits(tokens:
      startPosition:) throws -> (argmax: [Int32], topK: [[(Int32,
      Float)]])` that returns both argmax and top-K alternatives at
      each position. Leaves existing callers untouched.
   b. Change `verifyCandidates` signature (breaking). Avoid unless 2
      reasons force it.
   Go with (a). Pass a `topK: Int = 3` parameter so you get the top 3
   tokens + their logits at each of the K positions.

2. **Thread through CoreMLLLM.** Add `benchVerifyTopK(_ tokens: [Int32],
   topK: Int) throws -> [[(Int32, Float)]]` alongside existing
   `benchVerify`. Return shape: K positions, each with topK (token_id,
   logit_fp32) pairs.

3. **New bench mode.** `--mode tol` (or reuse `--mode chain` with a
   new `--tolerance` flag). Easiest: add a `--tolerance <N>` CLI arg
   that, when >0, switches matching to:
   - top-N: accept if drafter's proposal is in verify's top-N at that
     position
   - margin: accept if drafter's logit is within ε of argmax's logit
     (ε = 0.5 fp16 default, override via `--logit-margin`)
   Implement both; one is enough for a first measurement.

4. **Measure.** Run `--mode chain --tolerance 2` (top-2 acceptance)
   across all 10 prompts. Write to
   `eval/accept-rate-bench-v5-tolerance.json`.

5. **Document.** Compare v4 chain numbers vs v5 tolerance numbers in
   a short `docs/PHASE_C_TOLERANCE_FINDINGS.md` (≤ 2 pages). Table
   form, like v4's findings.

### Exit criteria

- **Hard:** build clean, existing `DrafterUnion` / `MtpSpeculativeEngine`
  still compile unchanged.
- **Hard:** v5 JSON produced.
- **Soft:** if chain-mode E[tok/burst] under top-2 tolerance rises
  above break-even (≥ 2.6) on ≥ 2 categories that previously failed
  (code / qa / summary), C0 path is viable via tolerance alone.

### Gotchas

- Verify_qK's output at position k is `logit_k[vocab]`. For top-K
  extraction, access the full logit tensor before the current argmax
  reduction. Look for `argmax` / `token_ids` in the verify function's
  CoreML graph — the logit tensor may not be an output today.
  If it isn't, this is blocked; need to re-export verify chunks with
  an additional `logits_fp16` output. That pushes Track A into
  Track B territory. Check FIRST before doing the Swift work.
- Accepting tokens outside the argmax changes semantics: the emitted
  sequence is no longer temp=0 target argmax. For the BENCH this is
  measurement-only (no emission). But if this mode goes into
  production (`DrafterUnion`), output quality must be re-validated.
  Out of scope for this session.

---

## Track B — rebuild (verify chunk numerical tightening)

**Session label:** `feat/c0-verify-requant`
**Estimated effort:** 2–3 days (bounded by CoreML re-conversion time).
**Touches:** `conversion/` scripts, CoreML verify function export,
staging model dir. **No Swift code changes** (other than maybe
adjusting if output shapes change).
**Risk:** partial model re-export can take 30–90 min wall-clock per
try; plan for 2–3 iterations.

### Goal

Produce a verify chunk where `argmax` is stable across batch content
— i.e., calling `verify_qK` with the same slot 0 token but different
slot 1..K-1 tokens gives the same `argmax[0]` (modulo genuine
sample-level variation, not fp16 jitter). This restores gotcha #3's
equivalence argument empirically.

### Approach options (pick one to try first)

1. **fp32 upcast on logit projection.** In the verify function's
   export, find the final linear (logit projection) op and cast its
   accumulation to fp32 before argmax. Most `coremltools` converters
   allow this via dtype hints or a manual graph edit. Cheapest
   re-export (no retraining).
2. **Accumulation-order control.** If the batched verify uses a
   different matmul tile order than decode_q1, align them. Requires
   digging into the CoreML-converted MIL program. Harder to pin down.
3. **Split verify_qK into K serial decode_q1 calls.** Not really
   "tightening" but eliminates the batched-computation effect
   entirely. Expected to restore oracle numbers but at K× verify
   latency. Useful as a sanity-check baseline — if this doesn't close
   the gap, something else is wrong. Fast to try.

Recommend order: **3 first (sanity check), then 1, then 2**.

### Implementation plan

1. **Locate the verify export.** Find where `chunk1.mlmodelc`'s
   `verify_qK` function is built. Likely in
   `conversion/` — grep for `verify_qK` or `EnumeratedShapes`.
2. **Try approach 3 (sanity check).** Change the Swift caller (NOT
   re-export) to run `decode_q1` K times serially instead of one
   `verify_qK` call. One-off experiment branch:
   `experiment/c0-verify-serial`. Run `--mode chain` against this and
   see if numbers lift. If yes: confirms the batched-compute is the
   culprit. If no: the drafter proposals affect state via KV cache,
   and the gap isn't batching — needs different investigation.
3. **Try approach 1 (fp32 upcast).** Re-export verify function with
   fp32 logit accumulation. Install compiled chunks into
   `staging-2k-fast-prefill/gemma4-e2b/`. Run `--mode chain`.
4. **If 1 improves but doesn't close the gap:** try approach 2
   (accumulation order) OR combine 1 + 3.

### Exit criteria

- **Hard:** reproducible recipe in `conversion/` that produces
  tightened verify chunks. Committed to a branch.
- **Hard:** `--mode chain` re-measurement recorded
  (`eval/accept-rate-bench-v5-requant.json`).
- **Soft:** chain-mode E[tok/burst] rises above break-even on ≥ 2
  categories. Same threshold as Track A.

### Gotchas

- Don't clobber `staging-2k-fast-prefill/` in-place. Produce a new
  staging dir (e.g., `staging-2k-fp32-verify/`) and point the bench at
  it via `--model`. Keeps a rollback path and allows A/B measurement
  in the same session.
- Mac CoreML compile step: `xcrun coremlc compile foo.mlpackage
  /tmp/out/` per chunk. ~1–2 min each. Script it.
- If re-export requires retraining (shouldn't for approaches 1/2/3
  but verify), it's out of scope — flag to user and switch to
  Track A. CLAUDE.md forbids long training runs without explicit ask.

---

## Parallel-session coordination

Both tracks target `main`. They touch disjoint files so rebase
conflicts are unlikely. Merge order (if both succeed):

1. Track A merges first (pure Swift, smaller diff, faster review).
2. Track B merges second. If Track B changes verify output tensor
   shapes in a way that affects Track A's topK extraction, rebase
   Track A on top of Track B before merging.

If Track A's gotcha triggers (logits not exposed without re-export):
- Track A blocks on Track B's logit-output addition.
- Track A session should stop Swift work and wait. Use session time
  to prepare the tolerance-check logic as a patch, ready to apply
  once Track B lands the logit output.

If Track B's approach 3 (serial decode_q1) closes the gap alone:
- Track A becomes redundant; skip the tolerance work.
- Document in `docs/PHASE_C_TIGHTENING_FINDINGS.md` and move on to
  Mirror SD planning.

---

## Measurement protocol (both tracks)

Same harness, same prompts, reproducible numbers:

```bash
swift build -c release --product accept-rate-bench
.build/release/accept-rate-bench --mode chain --max-tokens 128 \
  [--tolerance 2]                                  # Track A only
  [--model <alt-staging-dir>]                      # Track B only
  --out eval/accept-rate-bench-v5-<track>.json
```

Baseline to compare against: `eval/accept-rate-bench-v4-chain.json`
(chain mode numbers already per-category aggregated in
`docs/PHASE_B_V4_CHAIN_FINDINGS.md`).

Key metric: cross-vocab-qwen E[tok/burst] per category under
`--mode chain`. Target ≥ 2.6 on ≥ 2 of {code, qa, summary}.

---

## After C0 (either track succeeds)

1. Re-enable `drafterUnionEnabled = true` as a non-default opt-in
   flag with the C0 fix active. Do NOT flip the global default
   without an iPhone trip.
2. iPhone trip: measure `--mode chain` numbers on device (Mac CV
   accept != iPhone CV accept might still diverge).
3. Mirror SD work resumes (Phase C original plan).

## After C0 (both tracks fail)

Phase C via speculative decoding is shelved. Pivot to non-speculative
wins:
- Phase D1 staged chunk pipelining.
- Phase 5 item 27 (GPU prefill via MLX-Swift).
- Item 10 (KV direct-write in `commitAccepted`) may still help serial
  decode by 5–10%.

Update `MOBILE_2K_COMPETITIVE_PLAN.md` to reflect the speculative-free
path to 56 tok/s, and revisit whether that target is reachable under
the current chunks at all.
