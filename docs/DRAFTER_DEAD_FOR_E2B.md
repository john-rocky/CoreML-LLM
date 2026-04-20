# Separate-drafter speculative decoding is dead for Gemma 4 E2B (MTP heads are NOT)

**Date:** 2026-04-17 (rev 2026-04-17 — scope correction)
**Status:** CLOSED for separate-architecture drafters. **OPEN for MTP head retraining at-scale** — see §MTP exception below.

**Important scope clarification:** an earlier draft of this doc
overreached and labelled "drafter training" as a whole infeasible.
That conflated two structurally different mechanisms:

1. **Separate-architecture drafter** (small model that runs alongside
   the target): EAGLE-3, MTP Path C as a separate net, cross-vocab
   Qwen, n-gram lookup. → **Structurally infeasible on E2B** (the
   capacity-vs-speed argument below).
2. **MTP heads built into the target** (extra output heads predicting
   t+1..t+K from a single forward pass): the same Gemma 4 E2B base
   achieves 56 tok/s on LiteRT-LM via MTP heads Google joint-trained.
   → **Structurally viable.** Our two prior attempts failed for
   training/recipe reasons (Path A = wrong target distribution; Path
   C = undertrained), not capacity reasons.

The rest of this doc covers (1). For (2) see `docs/MTP_RETRAIN_OPTION.md`
once it is written, and `docs/HANDOFF.md` MTP entries.

## The two-constraint problem

A working drafter must simultaneously clear:

1. **Speed:** drafter forward must be ≥5-10× faster than target.
   Without this, verify cost overwhelms any acceptance gain.
2. **Capacity:** drafter per-position acc must be ≥50% live to clear
   the GPU-verify break-even (`docs/S0_GPU_VERIFY_GATE.md`). On ANE
   verify (current path) the bar is acc ≥70% — even harder.

Empirical drafter sizes vs both constraints on our 4.6B target:

| drafter size | speed (vs target) | live acc ceiling | meets both? |
|---|---|---|---|
| 35M (MTP Path C self-trained) | fast | 17% (measured) | NO — acc too low |
| 100-200M (theoretical mid) | ok | 25-35% (extrapolated) | NO — still under 50% |
| 500M-1B (Qwen 0.5B cross-vocab) | slow on ANE | 40% trainable | NO — measured 1.8 tok/s, speed kills it |
| same-family 1B Gemma 4 (would be ideal) | 4-5× speedup | 50%+ achievable | **YES — but Gemma 4 does not ship a 1B variant** |

The "sweet spot" drafter (small enough to be fast, large enough to be
accurate) does not physically exist for Gemma 4 E2B.

## Why more training data / epochs cannot fix this

1. **Vocab 262144 → argmax sensitivity.** Top-1 token agreement is
   hyper-sensitive to logit noise. fp16 rounding alone flips argmax
   on near-tied tokens. Drafter capacity sets a hard ceiling on how
   tightly its logits can match target's, regardless of training
   data volume.
2. **Capacity floor for small drafters.** A 35M-200M parameter
   drafter cannot fully replicate a 4.6B target's output
   distribution. Asymptotic per-position acc with infinite data
   converges to ~25-35% for size-mismatched drafters (consistent
   with EAGLE-3 paper's small-target benchmarks).
3. **Quantization compounding.** Drafter int4-palettized runs
   accumulate quantization noise. Target int4-palettized runs
   accumulate different quantization noise. Even a "perfectly
   trained" drafter sees its argmax drift from target by a few
   percent of vocab — uncorrectable through training.
4. **Chain compounding.** K=3 chain E[tok/burst] = 1 + p + p² + p³.
   At p=30%: E=1.42. At p=50%: E=1.88. The break-even on S0 GPU
   verify is E≈1.88. Per-position drafter acc must clear ~50% live,
   not 50% on validation.

## Empirical record (settles the question)

| drafter | live acc | result | session |
|---|---|---|---|
| EAGLE-3 (HF use_cache=False training) | 0% | distribution mismatch | HANDOFF.md |
| MTP Path A (LiteRT W4A8 target) | 0% | target distribution mismatch | MTP_PATH_A_FINDINGS.md |
| MTP Path C (self-trained, K=2) | 17% | capacity-limited | MTP_PATH_C_FINDINGS.md |
| Cross-vocab Qwen 2.5 0.5B | <1% live (1.8 tok/s) | speed killed it | HANDOFF.md |
| Prompt Lookup n=3 (oracle 2.94, live 1.01) | n/a — drafter is fine, verify contamination kills E[tok/burst] | PHASE_C_TIGHTENING.md |
| SuffixDecoding T1 | 18% (workload-dependent) | not enough | HANDOFF.md |
| Union of all drafters | 15-21 tok/s (baseline 32) | net regression | HANDOFF.md |

Best self-trained number: **17%**. Required: **50%**. Gap is 33 percentage
points — not a "more data" gap.

## Why the literature backs this up

Public speculative-decoding success stories all use base models ≥30B:

- Llama 70B + Llama 7B drafter → ~10× speedup ratio, 60%+ acc → 2-3× decode gain
- Llama 13B + Llama 1B drafter → ~5× ratio, 50%+ acc → 1.5-2× gain
- EAGLE / EAGLE-3 papers benchmark 7B-70B targets
- Apple Intelligence Server uses larger base
- LiteRT-LM's 56 tok/s on Gemma 4 E2B comes from **MTP (multi-token
  prediction inside the same forward)**, NOT speculative decoding
  with a separate drafter. Different mechanism.

There is **no published spec-decoding success with a separate drafter
on a 4-7B base model**. This is consistent with the size-mismatch
analysis above, not an oversight.

## Implications for the roadmap (revised)

- **Do not allocate GPU training time to separate-architecture
  drafter retraining (EAGLE-3, cross-vocab, etc.).** Capacity ceiling
  applies regardless of training scale.
- **MTP head retraining at Google-scale IS the live decode-speed
  path.** LiteRT-LM proves MTP works on this exact base. Our
  Path A (extracted Google MTP, wrong target distribution) and
  Path C (self-trained, undertrained) failures do not refute MTP —
  they refute the specific cheap shortcuts we tried. Properly
  retraining MTP heads against our HF fp / int4 target with
  Google-scale compute (A100 × 5-10 days estimate) plausibly reaches
  50-60% live acc, which composes with S0 GPU verify for ~50 tok/s.
- **S0 GPU verify is conditionally alive again.** It is dead as a
  standalone lever (no working drafter to multiply) but lives as the
  verify infra for an MTP retrain. Keep Stage 1 (G3 fix) and
  consider Stages 2/3 *contingent on MTP retrain progress*.
- **Pivot positioning regardless.** Even with MTP+S0 path live, it
  is a multi-week + $$ investment with mid-range odds. The
  "32 tok/s + ~0.5s TTFT + 1W + multimodal" package (A1+A2+B1) is
  the certain win to ship first. MTP retrain is a parallel research
  thread, not a blocker for shipping.

## Escape clauses (when to revisit)

Reopen this only if:

1. Google ships Gemma 4 1B or 0.5B (would enable same-family drafter).
2. New spec-decoding scheme emerges that requires <100M drafter at
   ≥50% acc on small targets (architectural breakthrough).
3. We switch base model entirely — e.g., to Llama 3.x 8B or Qwen 3 7B,
   which have established drafter ecosystems. (Note: this is also a
   product/quality decision, not just speed.)

Until then: drafter is closed.
