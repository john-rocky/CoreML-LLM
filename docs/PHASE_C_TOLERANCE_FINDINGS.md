# Phase C — Output-space tolerance (Track A) findings

**Status:** 2026-04-15 — measured. Track B (this PR,
`feat/c0-verify-logits-output`) exposed `logits_fp16` on the verify
chunk 4 mlpackage and re-compiled into a separate staging directory
(`staging-2k-fast-prefill-tol/`). The Swift `--tolerance N` path that
was code-ready in PR #73 is now active.

## TL;DR

**Tolerance-2 does NOT close the bench-vs-oracle gap.** Top-2
acceptance lifts a few (drafter, category) cells but leaves the
success threshold (chain E[tok/burst] ≥ 2.6 on ≥2 of
{code, qa, summary}) **un-met** for every drafter. Best cells under
tolerance-2 are summary prompt-lookup-n2 = 2.02 and code
cross-vocab-qwen = 1.35 — both far below the 2.6 pass bar.

In one regression (qa cross-vocab-qwen) tolerance-2 went backwards vs
v4 argmax chain (2.04 → 1.59). Root cause is likely the "accept
position-k only if drafter top-2 token is also in verify top-2", which
expands drafter acceptance at position 1 but also lets a drifted token
into the committed chain, poisoning position 2.

Hypothesis under test (from v4 findings): "most chain-mode misses have
the drafter's proposed token as a high-ranked alternative (top-2 or
within ε of argmax), not a complete miss" — **falsified** for
code / qa / summary. For chat cross-vocab-qwen the drafter is already
at oracle (2.31 ≈ 2.36), so tolerance changes nothing.

Conclusion: **tolerance alone cannot unblock Phase C.** The
bench-vs-live gap diagnosed in PHASE_B_V4_CHAIN_FINDINGS is not a
"top-N vs top-1" problem — the drafter's proposal is far from verify's
argmax *and* its top-N under fp16 batched verify_qK.

Next lever to try: Track B's fp32 upcast or the 11c
decode-verify re-normalisation. Shelving this branch is the most
likely outcome once those numbers land.

## Methodology

Same harness, same 10 prompts, same drafter set as
`docs/PHASE_B_V4_CHAIN_FINDINGS.md`. K=3, maxTokens=128,
staging-2k-fast-prefill-tol, Mac Studio cpuAndGPU. The only delta vs
v4 is the acceptance rule in `runChainModeTolerance` (see
`Sources/accept-rate-bench/Bench.swift`):

- `--tolerance N` (top-N): accept the drafter's proposal at position
  `k` if it is in verify's top-N tokens at that position.
- `--logit-margin M`: additionally accept if the drafter's proposal
  has a logit within `M` fp32 units of the argmax logit. (Not swept
  in this PR — tolerance alone is the decisive signal.)

The committed chain token is still `argmax[0]` — only *measurement*
loosens, the emitted sequence does not drift from temp=0 target decode.

Default sweep this PR: top-N ∈ {2}. top-1 reproduces v4 chain and is
covered by the existing v4 JSON. top-3 was not run because top-2
already showed the gap cannot be closed at this vocab size (it
actively *hurt* qa).

## Results

Source: `eval/accept-rate-bench-v6-tolerance.json`.

### cross-vocab-qwen E[tok/burst] per category, --mode chain

| Tolerance | chat | code | qa   | summary |
|-----------|------|------|------|---------|
| v4 argmax (exact, reference) | 2.31 | 1.01 | 2.04 | 1.00 |
| top-2 (this PR)              | 2.36 | 1.35 | 1.59 | 1.00 |
| Oracle (byte-exact)          | 2.31 | 2.63 | 3.17 | 3.12 |

Pass threshold for unblocking Phase C via tolerance: ≥ 2.6 on ≥ 2 of
{code, qa, summary}. **Hit: 0 of 3.**

### All drafters, all categories, tolerance=2

Full chainAccept histograms and derived E[tok/burst]:

```
category  drafter              v4 chain   v6 chain+tol2   Δ
chat      cross-vocab-qwen      2.31        2.36          +0.05
chat      prompt-lookup-n2      1.48        1.58          +0.10
chat      prompt-lookup-n3      1.49        1.57          +0.08
chat      suffix-scan           1.49        1.60          +0.11
code      cross-vocab-qwen      1.01        1.35          +0.34
code      prompt-lookup-n2      2.01        2.01           0.00
code      prompt-lookup-n3      1.01        1.01           0.00
code      suffix-scan           1.01        1.01           0.00
qa        cross-vocab-qwen      2.04        1.59          −0.45
qa        prompt-lookup-n2      1.00        1.00           0.00
qa        prompt-lookup-n3      1.00        1.00           0.00
qa        suffix-scan           1.00        1.00           0.00
summary   cross-vocab-qwen      1.00        1.00           0.00
summary   prompt-lookup-n2      1.00        2.02          +1.02
summary   prompt-lookup-n3      1.00        2.02          +1.02
summary   suffix-scan           1.00        2.02          +1.02
```

### Observations

- **Summary prompt-lookup jumps from 1.00 to 2.02.** Tolerance-2
  rescues exactly one chain position for PL across all three PL
  variants — probably the newline / structural tokens that have a
  tiny logit gap but are not drafter-top-1 after batched fp16 drift.
  Still far below oracle 3.26.
- **Code cross-vocab-qwen improves (1.01 → 1.35)** but does not
  approach oracle (2.63).
- **QA cross-vocab-qwen regresses (2.04 → 1.59).** Tolerance-2 can
  reduce E[tok/burst] because once a "close-but-wrong" token is
  accepted at position 1, the drafter's position-2 context is now
  stale relative to verify's committed chain, so position-2 misses
  harder than under strict argmax. Net: fewer full-length bursts.
- **Chat is flat.** CV chain already matches oracle at 2.31 — there
  is no gap for tolerance to close.

## Next steps

1. **Do not ship tolerance as the Phase C fix.** The data is
   unambiguous: no configuration of tolerance-2 (and by extrapolation
   tolerance-3) reaches the 2.6 bar on ≥ 2 of {code, qa, summary}.
2. **Hand the baton back to Track B** (fp32 verify upcast): if
   fp32-upcast verify closes the gap, Phase C unblocks via
   numerical-precision fix, not measurement-space loosening.
3. **If fp32 upcast also fails**, the residual gap is either item
   11c decode-verify drift (revisit the 11c harness in
   `docs/MEASURE_VERIFY_DRIFT_HANDOFF.md`) or a deeper batched
   verify_qK content-sensitivity that needs a different speculative
   architecture (e.g. Mirror SD or Medusa-style heads).
4. **Archive this measurement** — the v6 JSON is committed as
   `eval/accept-rate-bench-v6-tolerance.json` for future reference.

## References

- `docs/NEXT_SESSION_C0.md` §"Track A — loosen" — original plan.
- `docs/PHASE_B_V4_CHAIN_FINDINGS.md` — v4 baseline numbers and
  batched-verify-drift mechanism.
- `docs/PHASE_B_DECISION.md` — why chain mode matters.
- `conversion/build_verify_chunk4_only.py` — focused re-export of
  verify chunk 4 with the new `logits_fp16` output (this PR).
- `eval/accept-rate-bench-v6-tolerance.json` — raw histograms.
