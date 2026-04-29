# Quantization + sparsity — comprehensive survey for Gemma 4 E2B W4A8 stack

**Date:** 2026-04-22
**Scope:** All quantization methods across recent literature + shipped reference impls, ranked by ANE + W4A8 fit.

## 0. TL;DR

- **Currently shipping:** INT4 palettize (per_grouped_channel, group_size=32) + experimental W8A8 via coremltools.
- **Highest ROI next step:** SpinQuant (V6-6). Learned orthogonal rotations fold into weights → no runtime cost, ~+2% accuracy on W4.
- **Second step:** COMPACT (ROUND7 top pick). Vocab + intermediate pruning. +2-3 tok/s @ 2K.
- **Dead ends:** LUT-GEMM-specific methods (ELUTQ, Vec-LUT) — no ANE primitive.
- **ROUND7 winners unchanged:** COMPACT ★ > LaRoSA > SCAP > R-Sparse. Order: R7-1 → R7-4 → R7-3 → R7-2.

## 1. Classification matrix

| Method | Type | ANE fit | Training-free | Runtime ops | Reference impl |
|---|---|---|---|---|---|
| **INT4 palettize** (our default) | Weight-only LUT | ✅ Native | ✅ | `constexpr_lut_to_dense` | coremltools |
| **INT8 linear** | Weight-only affine | ✅ Native | ✅ | `constexpr_affine_dequantize` | coremltools |
| **W8A8** (our exp.) | Weight + Activation INT8 | ✅ Native (iOS 18) | ✅ (calib) | `linear_quantize_activations` | coremltools |
| **SpinQuant** | Learned rotation + quant | ✅ (rotations fold into weights) | ✅ | Identity runtime | [Facebook Research](https://github.com/facebookresearch/SpinQuant) |
| **QuaRot** | Hadamard rotation + quant | ✅ (random rotations fold) | ✅ | Identity runtime | [spcl/QuaRot](https://github.com/spcl/QuaRot) |
| **AWQ** | Activation-aware Weight Quantization | ✅ (weight-only, per-channel) | ✅ | Scale vector | [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) |
| **SmoothQuant** | Activation scale migration | ✅ (folds into weights) | ✅ | Identity runtime | Xiao et al. 2022 |
| **OmniQuant** | Learned bound | ✅ | Partial (calib) | Identity runtime | [OpenGVLab](https://github.com/OpenGVLab/OmniQuant) |
| **GPTQ** | Error-guided sequential quant | ✅ | ✅ | Identity runtime | IST-DASLab |
| **QuIP / QuIP#** | 2-bit with vector LUT | ❌ (LUT gather on ANE hostile) | ✅ | Gather ops | Cornell |
| **ELUTQ / Vec-LUT** | LUT-GEMM for CPU/FPGA | ❌ (no ANE LUT primitive) | ✅ | LUT table | Papers only |
| **K-means palettize** (coremltools) | Clustering LUT | ✅ | ✅ | LUT lookup | coremltools |

## 2. Sparsity / structural compression

| Method | Type | ANE fit | Training-free | Expected gain @ 2K | Reference |
|---|---|---|---|---|---|
| **COMPACT** ★ | Vocab + FFN pruning (structural) | ✅ Native | ✅ (calib) | +2-3 tok/s | [arxiv 2509.06836](https://arxiv.org/abs/2509.06836) |
| **LaRoSA** | Layerwise rotated top-k sparsity | ⚠ Static mask variant needed | ✅ | +1.5-2.5 tok/s | [arxiv 2507.01299](https://arxiv.org/abs/2507.01299) |
| **SCAP** | Per-layer statistical threshold | ⚠ Dynamic gating per token | ✅ (calib) | +0.5-1.5 tok/s | [arxiv 2412.07174](https://arxiv.org/abs/2412.07174); `repo-review/SCAP/` |
| **R-Sparse** | SVD decompose + rank-sparse branch | ❌ Dispatch overhead | ✅ | +1-2 tok/s | [arxiv 2504.19449](https://arxiv.org/abs/2504.19449); `repo-review/R-Sparse/` |
| **2:4 structured sparsity** | N:M pattern | ✅ (if NPU supports N:M) | ✅ | Unknown on ANE | NVIDIA Ampere introduced |
| **Wanda** | Magnitude × activation | ✅ | ✅ | ~+0.5 tok/s | [arxiv 2306.11695](https://arxiv.org/abs/2306.11695) |
| **SparseGPT** | Error-guided pruning | ✅ | ✅ | ~+1 tok/s | ICLR 2023 |
| **SliceGPT** | Structural rank reduction | ✅ (fold into weights) | ✅ | +1-2 tok/s | ICLR 2024 |

## 3. Key method deep-dives

### 3.1 SpinQuant (V6-6 on our roadmap)

**Paper:** Liu et al. May 2024. [arxiv 2405.16406](https://arxiv.org/abs/2405.16406).

**Mechanism:** Learn orthogonal rotation matrices R per-block that minimize post-quantization error. Rotations fold into surrounding weights (W_rotated = W @ R, and following layer unrotates via R^T @ Input). **Zero runtime overhead.**

**Headline numbers (paper):**
- LLaMA-7B: 2.9 PPL gap on W4A4KV4 (vs 1.8 PPL fp16).
- LLaMA-3-8B: 45% relative PPL-gap reduction vs QuaRot.

**ANE fit:** Perfect. Rotations fold into weights at conversion. Runtime graph identical.

**Effort on our stack:** 1-2 weeks.
- Calibration: ~12 hours on single A100 (corpus TBD; Wikitext-2 standard).
- Conversion: modify `conversion/build_w8a8_proper.py` or palettize path to multiply W by R.
- Validation: argmax agreement on test prompts.

### 3.2 QuaRot (V6-6 alt)

**Paper:** Ashkboos et al. Apr 2024. [arxiv 2404.00456](https://arxiv.org/abs/2404.00456).

**Mechanism:** Use random Hadamard matrices instead of learned rotations. Cheaper calibration (no training).

**Headline numbers:** W4A4 all-inclusive, 0.47 WikiText-2 PPL on LLaMA-2-70B. EuroSys 2025.

**Vs SpinQuant:** Cheaper (no training), slightly less accurate. Worth A/B compared to V6-6 SpinQuant.

### 3.3 COMPACT (ROUND7 R7-1, top pick)

**Paper:** Kwek & Yin Sep 2025. [arxiv 2509.06836](https://arxiv.org/abs/2509.06836).

**Mechanism:**
- Remove rare vocabulary tokens → shrink embedding + lm_head.
- Prune FFN intermediate channels using common-token-weighted activations.
- Result: standard Transformer with smaller `vocab_size` and `intermediate_size`.

**ANE fit:** ✅ Static reduced-dim weights. Compiles clean via our existing `build_*.py`.

**Our expected gains (from ROUND7):**
- chunk4 (lm_head-dominated): vocab 20-40% cut → ~0.5-1 tok/s @ 2K.
- chunk2/3 (FFN-heavy): 20% intermediate cut → ~1.5-2 tok/s @ 2K.
- Combined: **+2-3 tok/s @ 2K, +1.5-2 @ 8K**.

**Effort:** 3-5 days. Offline calibration + 4-chunk re-conversion.

**Top risk:** Repo not yet public. May need re-implementing calibration scoring from paper. Agent verification for Gemma 3 testing confirmed but E2B-specific not guaranteed.

### 3.4 LaRoSA (R7-3)

**Paper:** Liu et al. Jul 2025. [arxiv 2507.01299](https://arxiv.org/abs/2507.01299).

**Mechanism:** Per-layer orthogonal rotations (offline, fold into weights, like SpinQuant) + runtime top-k sparsity on rotated activations.

**ANE fit:** ⚠ Offline rotation clean, but runtime top-k selection is dynamic → CoreML compiles as dense-then-gather (ANE-hostile). Needs **static-mask variant** (calibration-time fixed sparsity pattern).

**Expected gain:** Paper 1.30× GPU wall-clock on LLaMA-2-7B @ 40%. ANE static-mask variant: **+1.5-2.5 tok/s @ 2K** at 35% static sparsity.

**Effort:** 4-5 days. Compose with V6-6 (shared rotation infrastructure).

**Risk:** Static-mask quality drop on Gemma 4 (untested). 1-day calibration PPL sweep first.

### 3.5 SCAP (R7-4)

**Paper:** Chua/Pan/Jain Dec 2024. [arxiv 2412.07174](https://arxiv.org/abs/2412.07174). NeurIPS 2024 ENLSP-IV Workshop (NOT main track).

**Source:** `repo-review/SCAP/`

**Mechanism:** Per-tensor (not per-channel) Mode-Centering. Compute mode of activation distribution (zero / median / KDE-peak), shift so sparsity threshold lands in high-density region.

**Code status (from source read):**
- 1 commit (May 2025): "update citation"
- No GeGLU test, no Gemma test
- Inference kernel `flash_gemv` assumes float32 — W4A8 integration undefined
- 64 C4 samples default calibration, max 256 tokens
- `SCAPLinearRealSparse` module: frozen thresholds per-layer, runtime `x_shifted.abs() > threshold` gating
- README claim: "1.5× over CATS at iso quality" — **no numbers in repo results**

**Realistic effort:** 2.5-3 days (+0.5-1 day GeGLU integration testing).

**ROUND7 claim +0.5-1.5 tok/s** is reasonable given Mode-Centering's modest effect on top of our existing INT4.

### 3.6 R-Sparse (R7-2)

**Paper:** Zhang et al. ICLR 2025. [arxiv 2504.19449](https://arxiv.org/abs/2504.19449).

**Source:** `repo-review/R-Sparse/`

**Mechanism:** SVD decomposes each linear as `W = U·S·V^T`. At inference, activations split into:
- **Sparse path:** top-k channels by magnitude scaled by singular values → direct linear
- **Low-rank path:** remaining channels → V·diag(S)·U^T matmul with rank truncation

**Code status:**
- 1 commit (Apr 2025): "v1"
- Triton kernel TODO; no public GPU impl
- No GeGLU test
- Config shows 1% threshold (NOT 50% sparsity as paper implies)
- Example script runs only PIQA task (no full benchmark)
- Assumes float32 inference

**ANE fit:** ❌ Runtime dynamic sparsity + 2 matmuls per FFN layer = dispatch overhead likely exceeds FLOP savings on our dispatch-bound ANE pipeline.

**Realistic effort:** 4-5 days (+1 day prototype-one-layer risk check first).

## 4. Our quantization roadmap reconciled

Per `docs/experiments/ROUND7_FINDINGS.md:238-261`:

### Proposed execution order

1. **V6-6 SpinQuant or QuaRot** (1-2 weeks) — learned rotation pre-quant. Identity runtime. +2% accuracy on W4.
2. **R7-1 COMPACT POC** (3-5 days) — post-training vocab + FFN pruning. +2-3 tok/s @ 2K.
3. **R7-4 SCAP audit** (2-3 days) — only if R7-1 hasn't captured prunable neurons.
4. **R7-3 LaRoSA static-mask** (4-5 days) — composes with V6-6 rotation infra.
5. **R7-2 R-Sparse** (4-6 days) — last, highest infra risk.

### Aggregate if all land

- @ 2K: 31 → **35-37 tok/s** (+4-6)
- @ 8K: 15 → **17-19 tok/s** (+2-4)

Still short of LiteRT 56. Ceiling under "no Metal port, no fine-tune, no new drafter": ~40 tok/s @ 2K.

## 5. iOS 18 quantization features (from `COREMLTOOLS_AND_IOS18.md`)

- **Per-block INT4 palettization** (block_size=32). Better for large layers (lm_head, embedding).
- **Joint compression** (prune + palettize) via `joint_compression=True`. Stacking sparsity + LUT-4.
- **Multifunction PTQ gap:** `@_multifunction_unsupported` blocks our multifunction packages. Workaround: pre-export torch-side quant.

## 6. What NOT to pursue

- **LUT-GEMM (ELUTQ, Vec-LUT):** No ANE LUT primitive. CPU/FPGA-oriented. Rejected ROUND7.
- **2-bit QuIP#:** Vector LUT gather is ANE-hostile. Research-grade, not mobile production.
- **FP4 / MXFP4:** Newer hardware-specific formats. ANE doesn't support natively.
- **Online SVD methods (xKV, expected attention):** Cannot compile as static CoreML graph. Rejected ROUND7.

## 7. Unknowns worth resolving

1. **Does our W4A8 stack actually hit the INT8 activation path or fall back to FP16 activation?** `build_w8a8_proper.py:186-196` uses `linear_quantize_activations` — verify runtime actually runs INT8 GEMM on ANE, not INT8→FP16 conversion pre-matmul.

2. **Per-channel vs per-tensor on our INT4 palettize:** We use `per_grouped_channel`, `group_size=32`. Is per-channel (no group) significantly better for outlier-heavy weights (lm_head)? 1-day sweep.

3. **Embedding quantization vs lm_head quantization asymmetry:** We INT8-quantize embeddings (external, CPU gather) but INT4-palettize lm_head (in-graph). Is this optimal? ExecuTorch defaults to 4-bit embeddings; try matching + compare.

4. **SmoothQuant applicability:** Our existing W8A8 experiment may already capture SmoothQuant's activation-scale migration implicitly via calibration. Explicit SmoothQuant could help if outliers remain.
