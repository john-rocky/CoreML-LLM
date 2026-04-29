# Round 7 Findings — ANE-only acceleration, drafter-dead era

**Date:** 2026-04-21
**Context:** EAGLE-3 on Gemma 4 E2B concluded unable to produce speedup
on small models (today's finding). Research pass to find genuinely
novel ANE-only methods that are NOT speculative decoding and NOT KV
compression. Every arxiv ID below was fact-checked via WebFetch.

---

## TL;DR

- Spec decoding on ANE is **not structurally impossible on paper** —
  with EAGLE-3 training acc τ=2.13 and measured verify K=3 = 31.5 ms
  @ 2K, break-even = 1.13 and theoretical speedup = 1.89×.
- **But oracle-replay vs live-decoding gap collapses this in practice.**
  Project measurements in `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md` show
  live acc is 3-9× lower than oracle bench across all categories.
  Corrected live speedup: **0.96-1.11× = net even** at best.
- **No 2025-2026 arxiv paper benchmarks spec decoding on Apple ANE for
  ≤3B models.** sd.npu (2510.15312) is Qualcomm Hexagon only;
  Mirror SD (2510.13161) is GPU+Hexagon for 14-66B. Literature has
  no counter-evidence to the "EAGLE-3 dead on E2B" finding.
- **Gemma 4 E2B does not inherit Gemma 3n's `activation_sparsity_pattern`**
  (confirmed by HF config inspection) — no free native sparsity win.
- **4 post-hoc candidates survive verification**: COMPACT, LaRoSA,
  R-Sparse, SCAP. Top pick is COMPACT (3-5 days, +2-3 tok/s @ 2K).

---

## Math verification — spec decoding on ANE

### Measured inputs (from `docs/EAGLE3_INTEGRATION_STATE.md:101`, `docs/PHASE_B_LIVE_ACCEPT_RATE_GAP.md`)

| Quantity | Value | Source |
|---|---|---|
| Base decode @ 2K | 32.3 ms | 31 tok/s measured |
| Base decode @ 8K | 66 ms | 15 tok/s measured |
| EAGLE-3 drafter forward (ANE) | ~5 ms | `ANE_OPTIMIZATION_SURVEY.md:90` |
| Verify K=3 per call @ 2K | **31.5 ms** (iPhone measured) | `EAGLE3_INTEGRATION_STATE.md:101` |
| CV drafter forward (Mac CPU+GPU, 3 Qwen steps) | **46-47 ms** | `PHASE_B_LIVE_ACCEPT_RATE_GAP.md:95-119` |
| EAGLE-3 training acc | 0.7494 / 0.406 / 0.239 | training eval |
| EAGLE-3 training τ | 2.13 | training eval |
| **Live CV acc (code/chat/qa/summary)** | **0.059 / 0.114 / 0.212 / 0.111** | `PHASE_B_LIVE_ACCEPT_RATE_GAP.md:95-119` |

### Break-even analysis (EAGLE-3 on ANE)

```
cycle_2K  = 5 (drafter) + 31.5 (verify) = 36.5 ms
base_2K   = 32.3 ms
break_even = 36.5 / 32.3 = 1.13

With training τ=2.13:
  paper_speedup = 2.13 / 1.13 = 1.89× → 31 → 58 tok/s (on paper)
```

### The oracle-vs-live gap

`PHASE_B_LIVE_ACCEPT_RATE_GAP.md` documents a 3-9× gap between
oracle-replay bench numbers (used to justify Phase A5) and actual live
decoding. Oracle replay measures drafter-vs-corpus agreement; live
decoding needs drafter-vs-Gemma-argmax agreement. Gemma's argmax often
diverges from the corpus after a few tokens even when both are valid
continuations.

Apply the same gap to EAGLE-3 (which has never been live-measured due
to the `use_cache=False` training-corpus bug producing 0% acc[0]):

```
Training acc[0] = 0.75
Live acc[0]     = 0.75 × (1/3 to 1/9) = 0.08–0.25

Live E[accepted prefix] ≈ 0.08–0.25 (mostly k=0 or k=1)
Live E[total tokens]    ≈ 1.08–1.25
Live speedup @ 2K       = 1.08–1.25 / 1.13 = 0.96–1.11× = net even
```

### Conclusion

Spec decoding on ANE **is mathematically viable in theory** but the
project's empirical oracle-live gap (documented) **eats the margin**.
The "EAGLE-3 can't speed up small models on ANE" finding is consistent
with the math once live-acc correction is applied.

**No published ANE result for ≤3B spec decoding exists to refute this.**
sd.npu's 1.06-3.81× is Qualcomm Hexagon with retrieval drafters, not
Apple ANE with EAGLE-class drafters.

### Caveats / open directions (not pursued this round)

- Retrieval drafters (PLD/SAM/SuffixDecoding) have drafter_cost = 0 ms
  (CPU, overlapped). Break-even drops further. Per live measurement
  PL3 hit 0.667 acc on one prompt, PL2 hit 0.55 on summaries. These
  paths remain viable in principle, just narrow in applicability
  (only help on repetitive / prompt-quoting workloads).
- If oracle-live gap can be closed (training corpus matching live
  distribution), EAGLE-3 math still works. Gap closure is an
  engineering question, not a math question.

---

## Gemma 3n native sparsity check (R7-5)

**Result:** REFUTED. Gemma 4 E2B does NOT inherit Gemma 3n's
`activation_sparsity_pattern` config field.

### Verified

```
$ cat ~/.cache/huggingface/.../gemma-4-E2B-it/config.json | grep -iE "sparsity|sparse"
(no output)

text_config: {
  "hidden_activation": "gelu_pytorch_tanh",
  "intermediate_size": 6144,
  "enable_moe_block": false,
  "num_experts": null,
  "expert_intermediate_size": null,
  ...
}
```

No `activation_sparsity_pattern`, no sparsity fields. Gemma 4 E2B is a
dense GeGLU FFN. The "free native sparsity" hypothesis fails.

### Implication

Retrofit activation sparsity methods (R7-1/3/4) have real headroom —
Gemma 4 E2B genuinely runs dense FFN with no sparsification.

---

## Verified candidates (retrofit post-training)

### R7-1. COMPACT — Joint vocab + FFN pruning  ★ top pick

- **arXiv 2509.06836** (Eugene Kwek, Wenpeng Yin; Sep 2025, v3 Oct 2025)
- **Verified:** Abstract mentions "Qwen, LLaMA, and Gemma families
  (0.5B-70B)". Doesn't specify Gemma 3 explicitly in the fetched
  abstract — "Gemma 3 tested" is inferred, unconfirmed.
- **Mechanism:** Removes rare vocabulary tokens to shrink
  embedding/lm_head; prunes FFN intermediate channels using
  common-token-weighted activations. Output is a **standard
  Transformer** with smaller `vocab_size` and `intermediate_size`.
- **ANE fit:** ✅ Static reduced-dim weights. Compiles clean via
  existing `build_*.py` pipeline.
- **Training-free:** Calibration required (not explicit in abstract
  but consistent with "post-training pruning"). Paper details needed
  for exact compute; likely ~1 A100-hour.
- **Expected gain on our pipeline:**
  - chunk4 (lm_head dominates ANE time): vocab 20-40% cut → ~0.5-1 tok/s @ 2K
  - chunk2/3 (FFN-heavy): 20% intermediate cut → ~1.5-2 tok/s @ 2K
  - **Combined: +2-3 tok/s @ 2K, +1.5-2 @ 8K**
- **Effort:** 3-5 days. Offline calibration + 4-chunk re-conversion.
- **Top risk:** Repo not public yet as of this writing. May need
  re-implementing calibration scoring from paper.

### R7-3. LaRoSA — Layerwise rotated top-k sparsity

- **arXiv 2507.01299** (Kai Liu et al., ICML 2025)
- **Verified:** Training-free, layerwise orthogonal rotations
  (per-layer static matrices), 1.30× wall-clock on LLaMA-2-7B @ 40%
  sparsity, beats TEAL by +1.77% zero-shot accuracy.
- **Mechanism:** Offline rotation folds into weights (like
  SpinQuant/QuaRot). Runtime: top-k selection on rotated activations.
- **ANE fit:** ⚠ Rotation offline clean; runtime top-k is dynamic →
  CoreML compiles as dense-then-gather (ANE-hostile). Need
  **static-mask variant** (calibration-time fixed sparsity pattern,
  losing adaptivity).
- **Training-free:** ✅ YES
- **Expected gain:** Paper 1.30× is GPU with sparse kernels. ANE
  static-mask variant trades adaptivity for convertibility; realistic
  **+1.5-2.5 tok/s @ 2K, +1-1.5 @ 8K** at 35% static sparsity.
- **Effort:** 4-5 days. Compose with V6-6 (SpinQuant/QuaRot) once that
  lands — shared rotation infra.
- **Top risk:** Static-mask quality drop on Gemma 4 (untested). 1-day
  calibration PPL sweep before committing.

### R7-4. SCAP — Per-layer statistical sparsity threshold

- **arXiv 2412.07174** (Chua/Pan/Jain; NeurIPS 2024 ENLSP-IV
  Workshop — NOT main track)
- **Verified (PARTIAL):** Paper real. "Intel Labs" affiliation is
  inferred from `IntelLabs/SCAP` repo name, not stated in fetched
  abstract. Specific "48.5% sparsity -1.5% accuracy on Mistral-7B"
  numbers NOT in abstract — need PDF.
- **Mechanism:** Mode-Centering pre-calibration shifts FFN activation
  distributions so sparsity threshold lands at high-density region.
  Per-layer adaptive threshold.
- **ANE fit:** ⚠ Mode-center offset foldable (static bias). Sparsity
  gating at runtime dynamic — same static-mask caveat as LaRoSA.
- **Training-free:** ✅ YES (calibration only)
- **Expected gain:** Modest. **+0.5-1.5 tok/s @ 2K, +0.3-1 @ 8K**.
- **Effort:** 2 days. Cheapest of the set; worth doing if R7-1 already
  cut the obvious neurons.
- **Top risk:** Weakest evidence base (workshop paper, numbers not
  abstracted). POC first before commit.

### R7-2. R-Sparse — Rank-aware GeGLU activation sparsity

- **arXiv 2504.19449** (Zhang/Liu/Tian/Khaitan/Wang/Li; ICLR 2025,
  VITA-Group attribution unconfirmed from abstract)
- **Verified (PARTIAL):** Training-free YES. GeGLU handling not
  explicit in abstract — abstract says "addresses limitations with
  non-ReLU activation functions" but doesn't name GeGLU/SiLU
  explicitly. Specific "50% sparsity ~0.5 PPL gap" not in abstract.
- **Mechanism:** SVD-decomposed form `W·x ≈ non-sparse bias +
  rank-sparse component`. Runtime: top-k + low-rank matmul branches.
- **ANE fit:** ❌ Runtime dynamic sparsity is ANE-hostile.
  Decomposition itself (U·S·Vᵀ) would compile as extra dense
  matmuls — dispatch overhead may eat the FLOP win on our
  dispatch-bound setup.
- **Training-free:** ✅ YES
- **Expected gain:** Lower than paper. **+1-2 tok/s @ 2K, +0.5-1.5
  @ 8K** at 30% rank reduction.
- **Effort:** 4-6 days, high infra cost.
- **Top risk:** The extra 2 dispatches per FFN may cost more than the
  smaller matmul saves. **Prototype one layer first (1 day) before
  full commit.**

---

## Rejected from Round 7

| Item | arxiv | Why rejected |
|---|---|---|
| FlashHead (lm_head 1.75×) | 2603.14591 | Hallucination — 2026-06 future date |
| Garbage Attention (BOS-sink prune) | 2601.06787 | WebFetch anomaly; abstract doesn't quantify Gemma-3-4B results |
| Amber Pruner (2:4 N:M sparsity) | 2508.02128 | Prefill-only per paper. Decode-bound is our issue |
| SparseInfer, TurboSparse, dReLU | various | All require ReLU; Gemma 4 uses GeGLU |
| FlexiDepth, ConfLayers, GateSkip | various | Dynamic per-token layer dispatch; static CoreML incompatible |
| ELUTQ, Vec-LUT | 2510.19482, 2512.06443 | LUT-GEMM CPU/FPGA-oriented; ANE has no LUT primitive |
| Mirror SD | 2510.13161 | Apple paper but GPU+Qualcomm NPU, 14-66B server |
| sd.npu | 2510.15312 | Qualcomm Hexagon only; drafter mechanism is retrieval-based (PLD/SAM are baselines it beats, not the proposed method); "EuroSys 2026" venue claim unverified |

---

## Proposed execution order

1. **R7-1 COMPACT POC** (3-5 days) — top pick. Gemma family tested per
   paper; static-graph clean.
2. **R7-4 SCAP audit** (2 days) — cheap supplement, only if R7-1 hasn't
   already captured obvious prunable neurons.
3. **R7-3 LaRoSA** (4-5 days) — after SpinQuant/QuaRot (V6-6) lands.
4. **R7-2 R-Sparse** (4-6 days) — last, highest infra risk.

### Composability

R7-1 shrinks dims. R7-3/4 sparsify within shrunken FFN. All stack
with existing Phase 0/1 roadmap items.

### Realistic aggregate

If R7-1 + R7-3 (static-mask) + R7-4 all land:
- @ 2K: 31 → **35-37 tok/s** (+4-6)
- @ 8K: 15 → **17-19 tok/s** (+2-4)

Still does not reach LiteRT 56 tok/s. ANE-only ceiling after all
Round 7 candidates plus existing Phase 0/1 roadmap items:
~40 tok/s @ 2K, ~22 tok/s @ 8K. This is a hard ceiling under the
"no Metal port, no fine-tune, no new drafter" constraint.

---

## What the Round 7 pass confirmed and refuted

### Refuted
- "Gemma 3n-style activation sparsity is inherited for free" — No.
  HF config shows Gemma 4 E2B has no sparsity fields.
- "sd.npu empirically proves ANE spec decoding works on 2-3B" —
  No. sd.npu is Qualcomm Hexagon only; no ANE result exists.
- "xKV / DuoAttention / HeadKV / Ada-KV / Expected Attention fit
  our stack" — All fail (online SVD / per-head variable KV / dynamic
  eviction / redundancy with existing sliding window).
- "First-pass research agent found 5 solid candidates" — Two of five
  had hallucinated numbers or fabricated claims. Independent WebFetch
  verification caught the issues.

### Confirmed
- Spec decoding math is not structurally broken for ANE small models
  in principle, but oracle-live gap closes the margin in practice.
- No published ANE 2-3B spec decoding success exists as of 2026-04.
- Gemma 4 E2B architectural features (QK-norm, KV-sharing, sliding-512)
  absorb most "easy" compression slack that published methods target.
- Retrofit activation sparsity (COMPACT, LaRoSA family) is the
  cleanest remaining ANE-only angle under our constraints.
- ANE Private API path (maderix/ANE + salescore Zenn, reviewed 2026-04-22,
  source-verified) does **not** reverse drafter death and offers no
  ROI-positive decoder recipe for Gemma 4 E2B — see
  `docs/ANE_PRIVATE_API_CONSTRAINTS.md`.
- Source-verified reference pass 2026-04-22 over maderix/ANE, ANEMLL,
  LiteRT-LM, llama.cpp, MLX (file:line citations in each doc):
  - **Metal Phase 3 blueprint is concrete.** `llama.cpp/src/models/gemma4-iswa.cpp`
    ships a full Gemma 4 E2B path with ISWA dual-KV, Q-only KV-sharing
    via `has_kv()`, QK-norm, fused GeGLU, and FlashAttention kernel
    `f32_dk256_dv256` matching our head-dim. See `docs/METAL_PORT_REFERENCE.md`.
  - **LiteRT-LM 56 tok/s is source-confirmed Metal-GPU path with MTP.**
    No ANE backend exists. Advantage ranked: Metal GEMM+fusion (~30%),
    MTP (~20%), KV-sharing (~10%), sliding+bound (~5%), GPU sampling
    (~5%), graph quant (~5%). See `docs/LITERT_LM_ARCH_VERIFIED.md`.
  - **Correction to earlier claim:** attention-on-ANE is NOT physically
    blocked. ANEMLL passes explicit causal mask MLMultiArray and
    attention runs on ANE. Maderix's "attn_mask ignored" framing was
    a Private-API-path observation, not a generic ANE constraint.
  - **MLX's ANE support:** does not exist in source. Zenn article's
    "lazy eval keeps intermediates in L2" mechanism is also
    unsupported by MLX source (fusion is compile-time op-inlining,
    not runtime scheduling). See `docs/ANEMLL_SOURCE_NOTES.md`.

---

## Sources (all fetched 2026-04-21)

- [COMPACT (arXiv 2509.06836)](https://arxiv.org/abs/2509.06836)
- [R-Sparse (arXiv 2504.19449)](https://arxiv.org/abs/2504.19449)
- [LaRoSA (arXiv 2507.01299)](https://arxiv.org/abs/2507.01299)
- [SCAP (arXiv 2412.07174)](https://arxiv.org/abs/2412.07174)
- [sd.npu (arXiv 2510.15312)](https://arxiv.org/abs/2510.15312) — Qualcomm
- [Mirror SD (arXiv 2510.13161, Apple)](https://arxiv.org/abs/2510.13161) — GPU+Hexagon
- [EAGLE-3 (arXiv 2503.01840)](https://arxiv.org/abs/2503.01840) — no target <8B benchmarked
- [Spec Decoding × Quantization (arXiv 2505.22179)](https://arxiv.org/abs/2505.22179)
