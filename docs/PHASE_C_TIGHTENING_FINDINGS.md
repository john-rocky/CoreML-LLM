# Phase C C0 — verify-chunk tightening findings

**Status:** 2026-04-15. Track B approach-3 sanity check complete.
Result: **the batched-compute hypothesis is refuted**. Switching
`verify_qK` (K queries, one dispatch) for K serial `decode_q1`
calls does not close the v4 chain-mode gap.

Baseline v4 chain numbers came from `eval/accept-rate-bench-v4-chain.json`.
Serial-verify numbers come from `eval/accept-rate-bench-v5-requant.json`
(new this session, produced by `--verify-serial` flag on
`accept-rate-bench --mode chain`). A fresh batched re-run is in
`eval/accept-rate-bench-v5-chain-batched-recheck.json` and matches v4
exactly — confirming the two runs are measuring the same pipeline.

---

## TL;DR

Replacing the batched `verify_qK` call with K sequential `decode_q1`
calls (approach 3 per `docs/NEXT_SESSION_C0.md`) leaves the cross-vocab
chain-mode accept rate unchanged within noise. On 3 of 4 categories
the numbers are identical; on chat serial is slightly worse
(2.31 → 2.09).

```
cross-vocab-qwen E[tok/burst]
                 oracle     batched verify_qK     serial decode_q1×K
chat              2.31           2.31                  2.09
code              2.63           1.01                  1.01
qa                3.17           2.04                  2.04
summary           3.12           1.00                  1.00
```

**Conclusion:** the v4 hypothesis that batched-compute fp16 ordering
drives the oracle→chain gap is wrong. Serial decode gives the same
chain argmax as batched verify — no "tightening" available from that
angle. Approach 1 (fp32 logit upcast in the verify export) and
approach 2 (accumulation-order alignment) would likely show the same
result, because the smoking gun isn't batched compute.

The exit criteria in `NEXT_SESSION_C0.md` Track B are met with a
negative outcome: chain-mode E[tok/burst] did **not** rise above
break-even (≥2.6) on any category. Per the playbook, this triggers the
"approach 3 doesn't close the gap → different investigation needed"
branch, not further verify-chunk re-export work.

---

## Per-category detail

```
category   drafter              oracle   batched   serial   Δ(serial−batched)
chat       cross-vocab-qwen      2.31     2.31     2.09      −0.22
chat       prompt-lookup-n2      1.35     1.48     1.34      −0.14
chat       prompt-lookup-n3      1.01     1.49     1.24      −0.25
chat       suffix-scan           1.35     1.49     1.42      −0.07
code       cross-vocab-qwen      2.63     1.01     1.01       0.00
code       prompt-lookup-n2      2.72     2.01     2.01       0.00
code       prompt-lookup-n3      2.94     1.01     2.01      +1.00
code       suffix-scan           2.93     1.01     1.00      −0.01
qa         cross-vocab-qwen      3.17     2.04     2.04       0.00
qa         prompt-lookup-n2      2.89     1.00     1.00       0.00
qa         prompt-lookup-n3      2.96     1.00     1.00       0.00
qa         suffix-scan           2.89     1.00     1.00       0.00
summary    cross-vocab-qwen      3.12     1.00     1.00       0.00
summary    prompt-lookup-n2      3.26     1.00     1.00       0.00
summary    prompt-lookup-n3      3.22     1.00     1.00       0.00
summary    suffix-scan           3.26     1.00     1.00       0.00
```

Code pl-n3 (+1.00) is the only category/drafter where serial exceeds
batched by more than noise. v4 explicitly called out PL's sensitivity
to drifted chains: the drifted serial chain happens to repeat n-grams
the batched chain didn't — same class of noise finding as chat pl-n3
in v4. Not a signal.

Chat numbers go slightly down across the board under serial
(CV −0.22, PL −0.14 / −0.25, suffix −0.07). Same mechanism in reverse
— serial emits a *different* chain than batched (both are target's
argmax at temp=0, but the drafter-proposal-conditioned KV state
writes differ between the two paths), and the new chain happens to
match drafter outputs slightly less on chat.

---

## What this rules out

- **Batched-verify fp16 ordering.** Hypothesised in
  `docs/PHASE_B_V4_CHAIN_FINDINGS.md` §"Why does verify-slot content
  matter?" as "the batched multi-function `verify_qK` compute path has
  fp16 ordering / reduction differences that depend on all K input
  tokens jointly." If that were the cause, switching to serial
  decode_q1 — which has *no* joint K-token compute at all — should
  close most of the gap. It doesn't. Rejected.
- **By extension, fp32 upcast on the verify logit projection
  (approach 1) is not worth trying.** Approach 1 addresses a subset of
  what approach 3 already covers (batched logit precision). If fully
  removing the batched path doesn't help, sharpening its precision
  won't either.

---

## What this shifts attention to

The remaining candidate mechanisms for the oracle → chain gap are now:

1. **KV-write side effects in slots 1..K-1.** Both batched and serial
   paths write K KV entries (drafter proposals at positions P+1..P+K-1
   in the write-through cache). A rejected proposal leaves a
   *different* KV value at those positions than the accepted token
   would have. Under chain mode, the `argmax[0]` at iteration *t+1*
   reads from KV[0..P+K-1] — which includes whatever drafter-proposal
   residue the previous iteration wrote. Serial decode replicates this
   exactly (same writes, just sequenced), so it reproduces the gap.
   Oracle replay avoids it entirely (no verify calls).
2. **KV-cache write-through semantics during the multi-drafter loop.**
   `runChainMode` runs all drafters at the same `startPosition`, so
   each drafter's verify call overwrites slots P..P+K-1 with that
   drafter's proposals. The chain's `nextID` uses drafter #0's
   argmax[0], but slot 0's KV at position P after the loop is from the
   LAST drafter's call. At temp=0 and exact arithmetic the KV values
   at slot 0 are identical across drafters (same input token), but
   under fp16 they can drift. v4's caveats section already flagged
   this as a second-order effect; v5 serial doesn't isolate it either.
3. **Genuine logit-margin flips from drafter-conditioned KV.** The
   drafter's proposals at P+1..P+K-1 enter KV write-through. At the
   *next* committed position P' = P+1 the target's argmax reads
   attention over a KV cache that contains the drafter's (possibly
   rejected) token at P+1 — which is a semantic difference from
   oracle replay, not a numerical drift. This would be intrinsic to
   the write-through design and not fixable by re-exporting chunks.

Candidate 3 is the most likely dominant mechanism and is not a
numerics problem: it is a *semantic* consequence of chain-mode's
multi-drafter measurement writing drafter proposals into KV before
acceptance is decided. Oracle replay doesn't have this because it
never calls verify. Live `DrafterUnion` has the same effect. This
would explain why v4's 3-of-4 category gap matches live directionally.

---

## Recommendation

Stop further verify-chunk re-export work (approach 1 / approach 2 now
both predicted to be no-ops by the approach-3 result).

Two paths forward, both higher-value than more Track B iteration:

1. **Pivot to Track A (tolerance-based acceptance).** Per
   `NEXT_SESSION_C0.md` Track A, measure top-N / logit-margin
   acceptance. If drafter proposals live near the argmax in the verify
   output (top-2 / within-ε-of-max), tolerance acceptance recovers
   bursts lost to the single-token flip. This does not require any
   chunk re-export — but Track A's gotcha note says it *does* require
   verify chunks to emit top-K logits. That is now the blocker for
   Track A. See "Track A unblocker" below.

2. **Pivot to non-speculative wins.** Per `NEXT_SESSION_C0.md`
   §"After C0 (both tracks fail)", the alternatives are Phase D1
   pipelining, Phase 5 item 27 (GPU prefill via MLX-Swift), and
   item 10 (KV direct-write in `commitAccepted`). Any of these is
   tractable without chunk re-export.

### Track A unblocker

If tolerance-based acceptance is taken up, the verify function needs a
new `logits_fp16` output alongside `token_ids`. That is a
`conversion/` change (one output node added to verify chunk4, not a
retraining step) plus a small Swift shim to read it. This session
scoped to Swift-only, so the conversion work is still pending.

---

## Artifacts

- `Sources/CoreMLLLM/ChunkedEngine.swift` —
  `verifyCandidatesSerial(tokens:startPosition:)` added as a sibling
  of `verifyCandidates`. Bench-only; not wired into production
  callers.
- `Sources/CoreMLLLM/CoreMLLLM.swift` — `benchVerifySerial` exposed
  alongside `benchVerify`.
- `Sources/accept-rate-bench/Bench.swift` — `--verify-serial` flag
  toggles between the two paths inside `runChainMode`.
- `eval/accept-rate-bench-v5-requant.json` — full serial results
  (Track B approach-3 sanity check).
- `eval/accept-rate-bench-v5-chain-batched-recheck.json` — same-day
  batched re-run, numerically identical to v4 (sanity check).

## Reproducing

```bash
swift build -c release --product accept-rate-bench

# batched baseline (same as v4)
.build/release/accept-rate-bench --mode chain --max-tokens 128 \
  --out eval/accept-rate-bench-v5-chain-batched-recheck.json

# serial experiment
.build/release/accept-rate-bench --mode chain --max-tokens 128 \
  --verify-serial \
  --out eval/accept-rate-bench-v5-requant.json
```

## Related

- `docs/PHASE_B_V4_CHAIN_FINDINGS.md` — v4 findings whose batched-
  compute hypothesis this session refutes.
- `docs/NEXT_SESSION_C0.md` — Track B approach ordering and exit
  criteria.
- `docs/PHASE_B_DECISION.md` — Phase B closure + Phase C gating.
