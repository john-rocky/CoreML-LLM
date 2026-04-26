# Round 8 Findings — Decode-side leads (drafter excluded)

**Date:** 2026-04-26
**Scope:** Web research pass with 4 parallel Opus 4.7 max agents covering
(1) 2025-2026 decode speedup techniques, (2) iOS 26 / coremltools 9 /
WWDC25, (3) Gemma 4 / 3n specific optimizations, (4) ANE
reverse-eng / driver tricks. Drafter / speculative decoding paths
excluded by user choice.

All findings cross-checked against `REJECTED_APPROACHES.md`,
`SURVIVING_HYPOTHESES.md`, `PRIORITY_ROADMAP.md`, `ROUND7_FINDINGS.md`,
`STAGE1_W4A8_FINAL.md`, `LITERT_LM_TECHNIQUE_TRIAGE.md`,
`COREMLTOOLS_AND_IOS18.md`, `SPEED_8K.md`, `UNEXPLORED_SOURCES.md`.

---

## TL;DR

Two genuinely new alive candidates not in any prior doc, plus one
already-documented POC re-confirmed. Three published-but-dead candidates
recorded so they aren't re-discovered. One strategic-pivot lead
re-confirms the existing roadmap.

| # | Candidate | Effort | Effect (decode) | Risk | Status |
|---|---|---|---:|---|---|
| 1 | Joint compression: INT8 LUT entries | 1-3 days | +1-2 tok/s | low | NEW |
| 2 | PALU low-rank K/V projection | 4-6 days | +5-8 tok/s | low ANE compat | NEW |
| 3 | Joint sparse + palettized POC | 2-3 days | binary, 0 or significant | medium | ALIVE in `COREMLTOOLS_AND_IOS18.md` §7.4 |

All three are independent, single-function (no multifunction PTQ
constraint per `COREMLTOOLS_AND_IOS18.md` §6.1), and can be claimed by
different sessions. **PALU is the highest-effect new lead from this round.**

---

## 1. Joint compression: INT8 LUT entries — NEW

- **Source:** Apple official docs.
  https://apple.github.io/coremltools/docs-guides/source/opt-joint-compression.html
- **Mechanism:** Apply `OpPalettizerConfig` to produce a W4 LUT
  (16 entries × FP16 each), then apply `OpLinearQuantizerConfig` with
  `joint_compression=True`. Result: **the 16 LUT entries themselves
  become INT8/UINT8 instead of FP16**. Per Apple docs verbatim: "This
  means using a lookup table (LUT) whose values are of the dtype
  INT8/UINT8 instead of Float16 which is the default."
- **Why it could help on this stack:** Gemma 4 E2B decode is
  memory-bandwidth-bound at the per-step weight read level (ANE_wait
  97 % per `CPU_BOTTLENECK_INVESTIGATION.md`). Halving LUT entry
  precision shrinks the per-blob bytes the ANE must read on each
  dispatch. Bundle size also drops ~1.8× on the W4-LUT-dominated
  weights (1.09 GB decoder graph per `LITERT_LM_TRANSFER_ANALYSIS.md`).
- **ANE compatibility risk:** Low. Apple-blessed format; the same MIL
  output op (`constexpr_lut_to_dense`) just carries int8 LUT instead of
  fp16. Verify on Mac first that the runtime accepts the int8-LUT
  variant before pushing to iPhone (cml9 surface area incomplete per
  Issue #2548).
- **Effort:** 1-3 days. Single converter flag change in
  `build_gemma4_e2b_stateful_chunks.py`, no architecture change.
- **Top failure mode:** ANE dequantizes LUT→FP16 on read regardless,
  so the steady-state per-step bandwidth saving is zero (only weight
  loading benefits). If that happens, this collapses to a bundle-size
  win only.

---

## 2. PALU — low-rank K/V projection — NEW

- **Source:** arXiv 2407.21118 (ICLR 2025).
  https://arxiv.org/abs/2407.21118
  Repo: https://github.com/shadowpa0327/Palu
  Authors: Chi-Chih Chang, Wei-Cheng Lin, Chien-Yu Lin, ...,
  Mohamed S. Abdelfattah (Cornell), Kai-Chiang Wu — paper confirmed
  via WebFetch 2026-04-26.
- **Mechanism:** Decompose K/V projection linear layers into low-rank
  matrices. Cache the compressed intermediate state instead of full
  K/V. Reconstruct full K/V on the fly inside the attention block.
  Includes a rank-search algorithm and quantization-friendly design.
  Reported 50 % KV compression; 1.89× attention speedup (2.91× with
  quantization) on RoPE-based attention.
- **Why it could help on this stack:** Decode is memory-bandwidth bound
  at 2K context where KV reads dominate per-step DRAM traffic. PALU
  shrinks the K/V tensors directly. Even if half the paper's speedup
  realises on ANE → +5-8 tok/s on the 31 tok/s baseline. Gemma 4 E2B's
  KV-sharing L15-34 from L13/L14 means PALU only needs to apply to L0-14
  (the layers that own K/V projections). Smaller surface area than a
  full-model port.
- **ANE compatibility risk:** Low. Reconstruction is `K = K_low @ B_proj`
  (and similarly V) — both standard matmuls, no exotic ops, no MLState,
  no dynamic shapes. The "optimized GPU kernel" in the paper is for batch
  throughput, not a correctness requirement; the math runs on any
  matmul-capable backend.
- **Effort:** 4-6 days. Calibration on a small sample, refit ranks,
  re-export 4 chunks. No retraining. PALU has a published rank-search
  procedure that hands you the rank assignments; the engineering work
  is the MIL-graph rewrite to insert the reconstruction matmul in front
  of QK and SV.
- **Top failure mode:** The reconstruction matmul re-introduces an op in
  the inner attention loop. If it doesn't fuse with the existing QKV
  ops at the MIL level, dispatch count rises rather than falls.
  Mitigation: fold reconstruction into the existing K/V projection of
  the same chunk so the fused op is one matmul-with-rank-r-bottleneck
  rather than two sequential matmuls. Validate via `MLComputePlan`
  audit (already shipped in `Sources/CoreMLLLM/ComputePlanAudit.swift`).
- **Top risk #2:** Rank-search may produce ranks not divisible by
  ANE-preferred tile widths (16, 32, 64). Round up at conversion time
  or accept the small accuracy degradation.

---

## 3. Joint sparse + palettized — ALIVE in `COREMLTOOLS_AND_IOS18.md` §7.4

- **Source:** Apple coremltools 8+ joint compression. Documented in
  this repo at `docs/COREMLTOOLS_AND_IOS18.md` §3.3 (mechanism) and
  §7.4 (POC item, "1 week"). Apple docs:
  https://apple.github.io/coremltools/docs-guides/source/opt-joint-compression.html
- **Mechanism:** Prune to N:M sparsity (e.g. 2:4) first, then palettize
  on the non-zero values. Output MIL ops:
  `constexpr_lut_to_sparse + constexpr_sparse_to_dense`. Apple docs
  claim "additional latency and memory savings, over and above those
  achieved by applying those techniques individually."
- **Why this round re-confirms it:** The Round 1 Apple-blessed feature
  has not been benchmarked on this stack. Round 8 web search surfaced no
  contradicting evidence; Apple still documents the path on cml9.
- **ANE compatibility risk:** Medium. Sparse-tensor support on ANE is
  limited — historical project measurement (`docs/QUANTIZATION_SURVEY.md`
  line 39 + Anemll comparison) marks N:M as "Unknown on ANE". Outcome is
  binary: either ANE skips zeros at the dispatch level (large win), or
  it materializes sparse-to-dense at load (size-only win, 0 tok/s).
- **Effort:** 2-3 days for the first probe. Reuses
  `build_gemma4_e2b_stateful_chunks.py` flag wiring path that Stage 1
  already added.
- **Top failure mode:** ANE compiler treats sparse weights as dense at
  runtime (just smaller storage on disk). Probe outcome is binary —
  either you see ANE_wait drop on the bench, or you don't.
- **Caveat (`COREMLTOOLS_AND_IOS18.md` §6.1):** PTQ functions
  (`palettize_weights`, `linear_quantize_weights`, joint compression,
  pruning) do **NOT** support multifunction models. The stateful Linear
  chunks are single-function so this is fine; the multifunction prefill
  variants (already-rejected per `IPHONE_ANE_SPARSITY_FINDING.md`) would
  not be a target either way.

---

## Re-confirmed dead by this round

| Candidate | Why dead | Source |
|---|---|---|
| TurboQuant 3-bit KV (Walsh-Hadamard) | ANE forces FP16 decomp regardless of stored precision; Hadamard has no native ANE op (matmul replacement = 0 efficiency) | `SPEED_8K.md:34`, `UNEXPLORED_SOURCES.md:183`, `PRIORITY_ROADMAP.md:269` |
| Spark Transformer GeGLU statistical top-k (NeurIPS 2025, arXiv 2506.06644) | Production version of Gemma 3n's training-time activation sparsity, but Gemma 4 E2B confirmed to NOT inherit `activation_sparsity_pattern` per HF config inspection. Static-mask offline variant = ROUND7 R7-2 R-Sparse / R7-4 SCAP territory, already evaluated | `ROUND7_FINDINGS.md:102-130`, `ROUND7_FINDINGS.md:200-218` |
| AFM-style cross-block KV sharing (5:3 / 62.5-37.5 split) | Gemma 4 E2B already shares L15-L34 (20/35 layers, 57 % shared) — more aggressive than AFM's 37.5 %. No headroom from this lever without retraining a different KV-share topology | `CPU_BOTTLENECK_INVESTIGATION.md:150` ("Apple FM-style architectural KV-share split — already in Gemma 4") |
| Cross-Layer Attention (CLA, NAACL 2025), Tensor Product Attention (TPA, NeurIPS 2025) | Both require fine-tuning (~1B+ tokens). Same effort tier as W2-QAT, listed as alternatives but bundled with W2-QAT campaign rather than separate workstreams | (Round 8 agent reports — not landed as separate items) |

---

## Strategic-pivot — re-confirms existing roadmap

**A19 Pro GPU Neural Accelerators (3.1× transformer)**

- Apple M5 LLM paper:
  https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Argmax iPhone 17 benchmarks:
  https://www.argmaxinc.com/blog/iphone-17-on-device-inference-benchmarks
- New A19 Pro GPU has matmul "Neural Accelerator" tensor units; reported
  +19-27 % gen-token speedup with MLX/Metal, 4× TTFT vs M4.
- **GPU prefill (item 27 in `PRIORITY_ROADMAP.md`)** — already current
  focus per `HANDOFF.md`. Round 8 evidence strengthens the case.
- **GPU decode** — explicitly out of scope per
  `MOBILE_2K_COMPETITIVE_PLAN.md` ANE-native ~1 W value prop. Round 8
  does not change this; pivot is a product decision, not a research
  decision.
- **No new action.** This round confirms the existing item 27 prioritisation.

---

## Recommended ordering for downstream sessions

Run the three candidates above in dependency-free parallel:

| Session | Branch | Candidate | First go/no-go gate |
|---|---|---|---|
| A | `feat/joint-int8-lut` | #1 INT8 LUT entries | Mac chunk_1 build with `joint_compression=True`; ANE placement audit; cos sim vs W4 LUT FP16 reference |
| B | `feat/joint-sparse-palettized` | #3 Joint sparse + palettized | Mac chunk_1 build with N:M (2:4) sparse + W4 LUT; ANE placement audit; if `non-ANE op(s) > 5 %` → fall back; cos sim vs W4 LUT |
| C | `feat/palu-low-rank-kv` | #2 PALU | PyTorch reference run (Mac, no ANE) of PALU on Gemma 4 E2B HF model; verify accuracy holds at rank-r per paper; only then proceed to MIL graph rewrite |

Cross-session protocol per `docs/ROADMAP_2026_04_26.md` §1: each
session adds a line to `docs/INFLIGHT.md`; rebase onto current main;
PR-or-fast-forward merge.

---

## Sources (Round 8 web research)

- PALU paper (verified via arXiv): https://arxiv.org/abs/2407.21118
- PALU repo: https://github.com/shadowpa0327/Palu
- Joint compression docs (verified via Apple developer site):
  https://apple.github.io/coremltools/docs-guides/source/opt-joint-compression.html
- Spark Transformer (NeurIPS 2025): https://arxiv.org/abs/2506.06644
- TurboQuant (ICLR 2026, ref impl): https://github.com/AmesianX/TurboQuant
- Apple AFM 2025 tech report: https://arxiv.org/abs/2507.13575
- Apple M5 LLM (MLX + GPU): https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Argmax iPhone 17 benchmarks: https://www.argmaxinc.com/blog/iphone-17-on-device-inference-benchmarks
- Cross-Layer Attention (NAACL 2025): https://arxiv.org/abs/2410.14442
- Tensor Product Attention (NeurIPS 2025): https://arxiv.org/abs/2501.06425

## How this round was conducted

Four parallel Opus 4.7 max agents, each given a different angle and
the full dead-list from existing docs. Outputs cross-checked against
this repo's docs to filter known-dead, known-done, known-in-flight.
Each surviving candidate independently URL-verified to avoid the
hallucinated-arxiv-ID failure mode flagged in
`REJECTED_APPROACHES.md` Round 7 (FlashHead, Garbage Attention).
