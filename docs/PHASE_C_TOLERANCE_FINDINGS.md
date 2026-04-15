# Phase C — Output-space tolerance (Track A) findings

**Status:** 2026-04-15 — skeleton. Swift side (this PR) is code-ready;
numerical results are **awaiting Track B's `logits_fp16` verify-chunk
re-export** before they can be measured.

## TL;DR

*(Placeholder — will be filled in after Track B lands.)*

Hypothesis under test: most v4 chain-mode misses have the drafter's
proposed token as a high-ranked alternative (top-2 or within ε of
argmax), not a complete miss. Relaxing acceptance to a top-N or
logit-margin test should close the code / summary gap without changing
the oracle/argmax numbers (which already hit exact argmax).

## Methodology

Same harness, same 10 prompts, same drafter set as
`docs/PHASE_B_V4_CHAIN_FINDINGS.md`. The only delta is the acceptance
rule in `runChainModeTolerance` (see `Sources/accept-rate-bench/Bench.swift`):

- `--tolerance N` (top-N): accept the drafter's proposal at position
  `k` if it is in verify's top-N tokens at that position.
- `--logit-margin M`: additionally accept if the drafter's proposal
  has a logit within `M` fp32 units of the argmax logit.

The two rules are OR'd. The committed chain token is still
`argmax[0]` — only *measurement* loosens, the emitted sequence does
not drift from temp=0 target decode.

Default sweep (when Track B lands):
- Top-N: `{1, 2, 3}` — 1 reproduces v4.
- Logit-margin: `{0.0, 0.25, 0.5, 1.0}` — 0 disables.

## Results

*(Placeholder — these tables will populate once Track B exposes
`logits_fp16` on the verify chunk 4 export.)*

### cross-vocab-qwen E[tok/burst] per category, --mode chain

| Tolerance | chat  | code  | qa    | summary |
|-----------|-------|-------|-------|---------|
| v4 (exact, reference) | 2.31 | 1.01 | 2.04 | 1.00 |
| top-2     | TBD   | TBD   | TBD   | TBD     |
| top-3     | TBD   | TBD   | TBD   | TBD     |
| margin=0.5| TBD   | TBD   | TBD   | TBD     |

Oracle targets (byte-exact): 2.31 / 2.63 / 3.17 / 3.12.
Pass threshold for unblocking Phase C via tolerance: ≥2.6 on ≥2 of
{code, qa, summary}.

### Other drafters

*(Placeholder — ngram-2/3 and SuffixTree tables; structure identical
to cross-vocab table above.)*

## Next steps

1. **Activate** once Track B's `feat/c0-verify-requant` merges to
   `main`. Rebase this branch if it landed first (the bench code path
   is already gated on runtime presence of `logits_fp16`, so no code
   change is expected).
2. **Measure.** Run:
   ```bash
   swift build -c release --product accept-rate-bench
   .build/release/accept-rate-bench --mode chain --max-tokens 128 \
       --tolerance 2 --out eval/accept-rate-bench-v5-tolerance-top2.json
   .build/release/accept-rate-bench --mode chain --max-tokens 128 \
       --tolerance 3 --out eval/accept-rate-bench-v5-tolerance-top3.json
   .build/release/accept-rate-bench --mode chain --max-tokens 128 \
       --logit-margin 0.5 --out eval/accept-rate-bench-v5-tolerance-m05.json
   ```
3. **Fill in tables** above and compare against v4 baseline.
4. **Decide.** If ≥ 2 of {code, qa, summary} clear 2.6 under top-2,
   Phase C unblocks on tolerance alone — proceed to
   `DrafterUnion` production integration (opt-in flag, separate PR)
   and iPhone re-measurement. If tolerance does not close the gap
   either, this branch stays archived and Track B's fp32 upcast
   numbers are the source of truth; document shelving in
   `docs/PHASE_C_TIGHTENING_FINDINGS.md` and move to Mirror SD.

## References

- `docs/NEXT_SESSION_C0.md` §"Track A — loosen" — original plan.
- `docs/PHASE_B_V4_CHAIN_FINDINGS.md` — v4 baseline numbers and caveats.
- `docs/PHASE_B_DECISION.md` — why chain mode matters.
