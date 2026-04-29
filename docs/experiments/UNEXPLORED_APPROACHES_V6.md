# Unexplored Approaches — Round 6 (lossless only)

**Date:** 2026-04-14
**Scope:** techniques missing from rounds V1–V5 and related surveys
(UNEXPLORED_APPROACHES.md, FUNDAMENTAL_UNTRIED.md, ANE_OPTIMIZATION_SURVEY.md,
UNEXPLORED_SOURCES.md, LITERT_*_ANALYSIS.md, MTP_PATH_A_FINDINGS.md).
Filtered to **strict no-quality-loss**: either runtime-only (zero
numerical change), speculative-decoding with a target verifier (output
equals greedy target decode by construction), or palettization granularity
changes that are quality-neutral-or-positive vs current INT4.

Items that trade quality for speed (TEAL activation sparsity, RWKV-7
distillation, PowerInfer-2 hot/cold split, Deja Vu, CLLM) were surfaced
by the research pass but are **excluded here** — see §"Lossy, tracked
elsewhere" at the bottom for pointers.

Baseline: 15 tok/s @ 8K on iPhone 17 Pro (A19). Dispatch-overhead bound.
All items below compose with MTP / EAGLE-3 since they either run before
the draft/verify loop (runtime hints, palettization) or ARE the loop
(spec-decoding drafters).

---

## Priority order — lossless, ROI-ranked

| # | Item | Effort | Lossless because | Compose with |
|---|---|---|---|---|
| **V6-1** | iOS 18.2+ `optimizationHints.reshapeFrequency = .infrequent` | 0.5 d | Runtime-only hint — same compute, same outputs; skips per-call shape-trace | everything |
| **V6-2** | `MLComputePlan` warm-pool reuse across predictions | 1 d | Runtime-only — plan is the *same* compute plan, reused not rebuilt | everything |
| **V6-3** | coremltools 8 `OpLinearQuantizerConfig(granularity="per_block", block_size=32)` re-palettization | 1 d reconvert | Per-block 32 is strictly finer-grained than current per-channel INT4 — quantization error ≤ current | everything |
| **V6-4** | Prompt Lookup Decoding wiring (algorithm already merged in PR #36) | 0.5 d wire | Verify-based — target verifier rejects wrong drafts; output identical to greedy | stacks with drafter |
| **V6-5** | GliDe + CaPE drafter (ICML'24) — drafter shares target's KV cache | 4 d | Verify-based | replaces EAGLE/MTP |
| **V6-6** | SpinQuant / QuaRot rotation before INT4 palettization | 2–3 d | Hadamard rotation is orthogonal (invertible); rotated weights palettize with lower error than raw | V6-3 |
| **V6-7** | HASS drafter (EAGLE train/infer mismatch fix, +10–15% accept) | 1 A6000-day + 2 d | Verify-based | replaces EAGLE |
| **V6-8** | Harmony-Decoding (HD) — target's own shallow layers as drafter, no training | 6 d | Verify-based | drafter alternative |
| **V6-9** | Clover-2 — RNN drafter matching EAGLE at 1/3 params | 7 d (incl. train) | Verify-based | replaces EAGLE |
| **V6-10** | llama.cpp per-head-scaled INT8 KV (the quality-safe version of the naive INT8 we rejected in V3) | 3 d custom CoreML op | Per-head scale preserves precision; still fp16 execution on ANE post-dequant | Phase 4 KV optim |
| **V6-11** | Union-of-drafters composition: Prompt Lookup ∪ SuffixDecoding ∪ {HASS or EAGLE-3} gated by max-accept-length, single verify pass when all miss | 6 d after V6-4 + drafter | Each source is verify-gated | requires V6-4, V6-5/7/9 |

---

## Per-item details

### V6-1. `reshapeFrequency = .infrequent` hint

iOS 18.2 added a MLModelConfiguration hint that tells the runtime the
model's input shapes will not vary between predictions — skips a
per-call shape-trace + bounds-check pass. Net: same ANE dispatch, same
output, ~0.5 ms/step shaved on a hot pipeline.

- Source: WWDC24 session 406 sample code; limited public documentation
  on developer.apple.com.
- Applies at `MLModelConfiguration` init time, before `MLModel(contentsOf:)`.
- Risk: none — if set on a dynamic-shape model the runtime re-traces
  silently (no crash, just no speedup). We use fixed shapes everywhere,
  so it's a pure win.

### V6-2. MLComputePlan warm-pool reuse

`MLComputePlan` can be materialized once at load time (similar to how
0a does the audit) and its underlying compute-plan blob cached for
every subsequent `prediction(from:)`. The runtime currently rebuilds
the plan on first prediction after load; pooling skips ~0.8 ms on
first-call and correspondingly on infrequent "cold" sub-paths.

- Source: iOS 18 `MLComputePlan` class additions; underdocumented on
  developer.apple.com.
- Implementation: likely slot the plan into the `ChunkedEngine`
  per-chunk struct, reuse via the model's `MLPredictionOptions`.
- Risk: low — same plan, just not rebuilt.

### V6-3. coremltools 8 blockwise-32 palettization

coremltools 8.1 exposes `granularity="per_block"` on
`OpLinearQuantizerConfig` / `OpPalettizerConfig`, matching AWQ / GGUF
blockwise-32 grouping. Current conversion uses per-channel INT4
(large groups). Per-block 32 means each 32-element slice of the weight
gets its own scale+zero-point, so quantization error is uniformly
bounded. Quality is ≥ current by construction for the same bit-width.

- Source: coremltools release notes 8.1 (`OpLinearQuantizerConfig`
  granularity parameter).
- Effort: one reconversion pass (touch `build_*.py` scripts;
  `group_size=32` already present, add `granularity="per_block"`).
- Paired with V6-6 for further quality headroom.

### V6-4. Prompt Lookup wiring

PR #36 landed the n-gram matching algorithm (`PromptLookupDraft.swift`)
as dead code. Wiring into the decode loop (`CoreMLLLM.swift` main
generation path + `SpeculativeTarget.verifyCandidates`) converts it
into a zero-training drafter usable alongside MTP / EAGLE as a fallback
or primary on prompt-quoting workloads (summaries, QA, code).

- Verifier: existing `verifyCandidates` in `ChunkedEngine`.
- Expected gain: ×2.4 on summaries, ~1.0× on free-form chat.
- Trade: adds one verify pass per burst; acceptance gate threshold needs
  tuning (drop to single-token when rolling accept < ~0.3).

### V6-5. GliDe + CaPE

- Paper: ICML'24, arxiv 2402.11131.
- Repo: github.com/NonvolatileMemory/GliDe_with_a_CaPE_ICML_24.
- Drafter *shares the target's KV cache* instead of maintaining its
  own. Eliminates drafter-side KV memory (critical on iPhone at 8K
  ctx: drafter KV can cost 20–50 MB).
- ANE fit: drafter becomes a thin MLP head reading target's
  `kv13`/`kv14` (same interface MTP's Path A tried to use — but GliDe
  designs the drafter *for* this sharing rather than extracting a
  pre-trained mismatched one).
- Training: ~8 A100-hours for a Gemma 4 E2B drafter.

### V6-6. SpinQuant / QuaRot rotation

- Papers: SpinQuant arxiv 2405.16406 (learned), QuaRot arxiv 2404.00456
  (Hadamard, training-free).
- Idea: apply an orthogonal rotation `R` to weights and inputs such
  that `W x = (W R⁻¹)(R x)`. Rotations smear outliers across channels,
  shrinking the dynamic range each channel covers → finer palettize
  quantization buckets.
- Fits INT4 palettization: rotated weights palettize with ~0.3–0.5
  lower PPL at same bits. QuaRot is training-free and cheap.
- Risk: rotation must be materialized into weights at conversion time
  and into activations at prefill boundary (or fused inline). Cheap on
  ANE since it's just an extra matmul pre-attn.

### V6-7. HASS (Harmonized Speculative Sampling)

- Paper: arxiv 2408.15766.
- Repo: github.com/HArmonizedSS/HASS.
- Fixes EAGLE's train/inference distribution mismatch — trains the
  drafter on target outputs seen during inference, not on pre-collected
  hidden states. +10–15% accept rate for same drafter size.
- Directly addresses EAGLE-3 Blocker 1's reframed root cause
  (use_cache=True vs False at training time).
- Compatible with our existing `verify_chunk{1..4}` infrastructure.

### V6-8. Harmony-Decoding (HD)

- Paper: arxiv 2502.18008 (early 2025).
- Training-free. Uses the target's own shallow layers (e.g. L0–L14)
  as a drafter via a phase-dependent gate — smarter than LayerSkip's
  naive early-exit.
- No separate drafter artifact to ship; reuses chunk1+chunk2 shapes
  we already have.
- Risk: acceptance rate unmeasured on our weight distribution.

### V6-9. Clover-2

- Paper: arxiv 2408.00264.
- Repo: github.com/XiaoMi/Clover / Clover-2.
- RNN-based drafter, ~1/3 the parameters of a comparable EAGLE drafter
  at same accept rate. Smaller drafter = less ANE dispatch cost per
  speculative step.

### V6-10. Per-head-scaled INT8 KV

V3 rejected naive INT8 KV because ANE dequantizes to fp16 before
compute — no wall-clock gain. The quality-safe variant, which
llama.cpp ships as `--cache-type-k q8_0`, stores an INT8 KV with a
**per-head scale** (rather than one global scale). Memory footprint
drops 50% at negligible PPL cost. On ANE the compute is still fp16
post-dequant, so the win is purely KV memory — enabling longer
context within the same memory budget, not faster decode.

- Slot in Phase 4 (long-context / KV optim), not Phase 0–3.
- Implementation: Swift-side quant/dequant around the `kFull*` buffers,
  or a custom CoreML op in chunk2.

### V6-11. Union-of-drafters composition

Each lossless drafter (Prompt Lookup, SuffixDecoding, HASS/EAGLE-3,
GliDe) has strong workloads and weak workloads. Union = take the
longest accepted draft across all sources per burst, with a single
target verify pass when all miss. Marginal cost on ANE: one verify
pass per burst (the expensive part), paid anyway. Marginal benefit:
drafter-source diversity matches workload-type diversity.

Expected composition gain is +30–40% over the best single source — not
multiplicative because sources correlate on easy prompts and anti-
correlate on hard ones. Empirically this caps around +40% on mixed
workloads (LookaheadDecoding paper table 4 reports similar numbers
for their tree-union variant).

---

## How to integrate with the existing roadmap

Slot the items into `PRIORITY_ROADMAP.md` as follows — lossless-only,
and only where the existing phase's thesis still applies:

- **Phase 0** (runtime micro-ops): add V6-1, V6-2
- **Phase 1** (reconversion-based, quality-neutral): add V6-3, V6-6
- **Phase 2C** (zero-training auxiliaries): add V6-4 wiring, V6-8
- **Phase 2B** (EAGLE-3 retrain track): add V6-7 (replaces plain EAGLE-3
  retrain; same effort, better result)
- **Phase 2 shared**: add V6-5 (GliDe) as a drafter alternative to
  EAGLE-3 / MTP
- **Phase 3** (post-drafter stacking): add V6-11 (union composition)
- **Phase 4** (long-context KV): add V6-10 (per-head INT8 KV)

The roadmap also gains a diagnostic prerequisite: before V6-5 / V6-7 /
V6-9 is worth committing, confirm the MTP self-trained path (other
session) or EAGLE-3 retrain lands; otherwise the verify-chunk
infrastructure may itself need fixing first.

---

## Lossy, tracked elsewhere

The research pass surfaced these quality-trading items. Kept out of
the priority list above but logged here so they're not re-discovered:

- **TEAL** (arxiv 2408.14690) — training-free activation sparsity,
  40–50% FFN rows zeroed, <1 PPL reported on Llama/Mistral but untested
  on Gemma's GeGLU. Quality-dependent.
- **RWKV-7 "Goose"** (arxiv 2503.14456) — pure Conv1×1 architecture,
  softmax-free, natively ANE-friendly. Requires re-distill from Gemma
  (10–14 d). Different model.
- **PowerInfer-2** (arxiv 2406.06282) — hot/cold neuron split across
  NPU+CPU. Needs predictor training.
- **Deja Vu** (ICML'23) — activation sparsity with trained predictor MLP.
- **Q-Sparse / CATS** — similar class, mostly training-dependent.
- **CLLM / Consistency LLM** — 64 GPU-days per 7B.
- **BitDelta** — 1-bit delta for fine-tuned variants; only useful if
  shipping many Gemma variants.

---

## Not novel (already in existing docs, flagged by research pass)

- ReDrafter — in ANE_OPTIMIZATION_SURVEY.md.
- Ouroboros — single-session variant reduces to SuffixDecoding (already
  in FUNDAMENTAL_UNTRIED.md).
- EAGLE-3 — already in roadmap Phase 2B.
- Mamba-2, RetNet, Zamba — checked, ANE-incompatible or weaker than
  Gemma.
- vLLM PagedAttention — ANEMLL covers equivalent territory.
- TensorRT-LLM tree-attn — Sequoia covers this.
