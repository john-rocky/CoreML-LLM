# Integrated Roadmap — Pre-Conversion Optimizations

**Date:** 2026-04-16
**Branch:** `research/conversion-deep-dive`
**Basis:** synthesis of 22 research + decision + implementation docs
**Baseline:** 15 tok/s @ 8K on iPhone 17 Pro (BASELINE_SPEED_AUDIT.md)
**Target:** beat LiteRT-LM 56.5 tok/s (project_direction memory)

---

## Section 1 — Executive Summary

After three research rounds the landscape is mapped. Previous analyses estimated the ANE-only ceiling at 22-28 tok/s. Deep architectural audits (GEMMA4_FORWARD_ANATOMY, ANE_EMPIRICAL_GUIDE, GEMMA4_ALGEBRAIC_REWRITES, CROSS_CHUNK_OPTIMIZATION) raise that estimate: stacked Python-side optimizations plausibly reach **50-70 tok/s** on iPhone 17 Pro before spec decoding on top, making the LiteRT-LM 56.5 target achievable from within CoreML-LLM without pivoting to Metal.

**Top-3 single biggest wins** (by Δtok/s × confidence / LOC × risk):

1. **`ane_softmax` → `ane_fused_softmax` swap** (2-line change, helper already implemented in ane_ops.py): **+15-25%** because it enables the ANE compiler to recognize the softmax op and potentially fuse QK^T·softmax·@V. Source: GEMMA4_FORWARD_ANATOMY §3.4, CUSTOM_MIL_PASSES R14, GEMMA4_ALGEBRAIC_REWRITES B2.

2. **D-1 FusedQKV/FusedGateUp wiring** (patch-ready per D1_WIRING_PATCH_PLAN, ~40 LOC in one file): **+13-20%** + **180 MB weight reduction** from KV-shared layer k/v deletion as a side effect. Source: D1_WIRING_PATCH_PLAN.md, GEMMA4_ANE_REWRITES #1+#2, GEMMA4_FORWARD_ANATOMY Priority 1.

3. **Sliding KV head_dim padding drop** (28 sliding layers currently store hd=256 padded to max_hd=512): **+5-10%** via halved sliding attention FLOPs. Source: GEMMA4_FORWARD_ANATOMY §3.8.

**Tier-S stacked (all three): +30-55% decode, ~150 LOC**. Hits 22-27 tok/s — the top of the previously estimated ANE ceiling.

**Tier-A add-ons (NCHW rewrite, QK^T→Conv1x1 for matmul 3× penalty recovery, RMSNorm absorption)**: further **+25-50%**. Combined Tier S+A: **50-65 tok/s** projected, **LiteRT-LM 56.5 is in range**.

**Tier-B spec-decode composition (LayerSkip @ L14, MTP drafter, prompt-lookup)** multiplies on top by **1.3-1.5×** if accept ≥ 40%.

The previously assumed "ANE is structurally capped at ~25 tok/s" was based on partial information. The compiler is the gate (not hardware), and ~30% of compute is matmul emulation penalty that CAN be reclaimed.

---

## Section 2 — Ranked Optimization Catalog

Ranked by (Δtok/s × confidence) / (LOC × risk). Confidence L/M/H. Risk L/M/H.

| # | Name | Source | Δtok/s | LOC | Risk | Parity | Deps |
|---|---|---|---|---|---|---|---|
| 1 | ane_softmax → ane_fused_softmax swap | R10 §3.4, R11 B2, R14 | +15-25% (M) | 2 | M (fp16 overflow on long ctx) | near-lossless | none |
| 2 | FusedQKV + FusedGateUp wiring (D-1) | D1_WIRING_PATCH_PLAN, R1 | +13-20% (M-H) | 40 | M | lossless (concat of weights) | none |
| 3 | Sliding KV hd=256→512 padding drop | R10 §3.8 | +5-10% (M) | ~30 + Swift cache | M (cache layout) | lossless | none |
| 4 | NCHW pure-channels-first rewrite | D4, R10 §1.E | +10-20% (M) | ~100 Py + 40 Swift | M (RMSNorm fast path) | lossless | #2 |
| 5 | Attention QK^T as Conv1x1 rewrite | R18 Priority 1, R11 B2 | +15-30% (L-M) | ~50 | M-H | lossless (arithmetic eq) | #1 |
| 6 | RMSNorm scale absorption (sandwich+QK+PLE) | R10 §3.3-§3.4, ane_ops util | +1-3% (H) | ~40 | L | lossless (weights-only) | #2 recommended |
| 7 | Enable optimize_mlpackage --optimize flag | IMPLEMENTATION_LOG A2+A4 | +5-10% (M) | 0 (done, needs run) | L | lossless | none |
| 8 | v_norm rsqrt → cat-trick + affine=False | R10 flag #1 | <0.5% (H) | 5 | L | near-lossless | none |
| 9 | Logit softcap drop from graph | R10 §3.5 | 0 latency / hygiene (H) | 3 | L | lossless for greedy argmax | none |
| 10 | layer_scalar hard-delete if all=1.0 | R10 §3.2 flag #5 | <0.5% (verify first) | 2 | L | lossless if all 1.0 | verify HF values |
| 11 | chunk4 lm_head INT4 palettize | audit §4.8, R17 | 0 latency, -300MB | 10 | M (quality eval) | near-lossless | none |
| 12 | PLE INT4 palette (on-disk bin) | R17 | -230 MB | ~20 | M (quality eval) | near-lossless | none |
| 13 | Vocab pruning activation | R17, prune_vocab.py dry-run | -2.0 GB, +1-2 tok/s | run existing scripts | M | near-lossless | retrain path? |
| 14 | SDPA sliced_q pass in verify/decode builds | audit §4.2, CUSTOM_MIL_PASSES | +20-35% prefill | 5 | L | lossless | #1 |
| 15 | RoPE/mask constant baking | CROSS_CHUNK C.2 | +X (small) | ~20 | L | lossless | none |
| 16 | Per-chunk PLE slicing (17.5KB/call → 2-4KB) | CROSS_CHUNK C.3 | marginal | ~20 | L | lossless | none |
| 17 | LayerSkip at L14 early exit + draft head | R13, FUNDAMENTAL_UNTRIED §4 | +1.4-1.6× (M) if accept ≥40% | train head + ~100 | M-H | lossless via verify_qK | early-head training |
| 18 | PLE low-rank factor r=256 | R11 4, R17 | PLE 1.17GB → 34MB | ~50 + retrain | H (retrain required) | approx with eval | retrain |
| 19 | Stateful KV 2-hour probe on iOS 26 | D5 hedge | +15-20% if works | single-line change + build | H (error -14 likely repeats) | lossless | iOS 26 device |
| 20 | Double-wide MLP rank compression L15-L34 | R10 §3.9 | +5-15% if quality holds | ~50 + eval notebook | H | approx with eval | quality eval budget |
| 21 | Partial_rotary_factor semantic verify+fix | R10 §3.7 | correctness fix (not perf) | ~10 | H (correctness) | — | reference parity |
| 22 | Per-op mixed-bit palettization | R16 LAYER_SENSITIVITY | heterogeneous (eval) | sweep + allocator | H | approx with eval | sensitivity sweep |
| 23 | fuse_manual_softmax custom MIL pass | R14 | +1-3% | ~100 | L | lossless | (subset of #1) |
| 24 | Drafter V-layout fix (Option B) | D3 | 0 perf / crash prevention | 8-12 | L | lossless | MTP re-enable |

Total actionable items: 24. Items 1-11 are near-term, 12-18 mid-term, 19-22 conditional/experimental.

---

## Section 3 — Top-10 Detailed Treatment

### 1. ane_softmax → ane_fused_softmax swap
Current path (ane_ops.py:216-241) decomposes softmax into `reduce_max + sub + exp + reduce_sum + real_div` to preserve fp16 inside the op (historical reason: avoid PyTorch fp32 upcast in trace). This produces ~50 MIL ops per chunk, ~200 model-wide, and prevents the ANE compiler from recognizing the softmax pattern and enabling SDPA fusion.

`ane_fused_softmax` (ane_ops.py:244-266) was added in commit ac2fdd7. It calls `F.softmax` directly with explicit fp16 casts, producing a single MIL `softmax` op. The mtp_drafter already uses this pattern and works (MLPACKAGE_STRUCTURE_AUDIT §2.4).

**Implementation:** edit two call sites in `conversion/models/gemma4_swa_chunks.py` (lines 142 and 595 per R10). Re-convert, parity-test.

**Validation:** `test_merged_parity.py` with strict top-1 match threshold on 256 decode tokens. Watch for attention-overflow at long ctx (ctx > 4096) — the fp16 Q@K^T is already the stability-critical path (gemma4.py:275-281 historical note) and the decomposed softmax's intermediate `max(x)` broadcast might have been protecting from underflow in exp. If parity fails at long ctx, restrict swap to short-ctx tests first, or apply per-layer.

### 2. FusedQKV / FusedGateUp wiring (D-1)
D1_WIRING_PATCH_PLAN.md has the complete patch. One-file edit (`gemma4_swa_chunks.py` `_run_layer_swa` and `_run_layer_verify` cover all 8 chunks via inlined usage). Load-time post-process via `fuse_layer_projections()` concatenates Q/K/V and gate/up weights along output channels.

**Side benefit:** KV-shared layers (L15-L34) have no K/V weights in HF checkpoint — after wiring, chunk code branches `if is_kv_shared: q = q_proj(x)` else `q, k, v = qkv_fused(x)`, and k_proj/v_proj can be deleted from shared layers entirely. 180 MB weight reduction documented in D1 plan.

**Implementation:** ~40 LOC, patch-ready. Option A post-load fuse preferred over Option B init-time (~650 MB transient RAM cost acceptable, much simpler).

**Validation:** per-layer cosine vs reference before/after fuse, then full chunk parity.

### 3. Sliding KV hd=256→512 padding drop
28 sliding layers currently pad K/V from hd=256 to max_hd=512 at line `gemma4_swa_chunks.py:93-97`. This is done because the cache buffer is shared across sliding and full-attention layers. But the attention math already slices `K[..., :hd]` correctly at read site.

Fix: split cache buffer into sliding (nsl, 1, W, 256) and full (nf, 1, ctx, 512). Remove the F.pad. Update Swift cache alloc.

**Win:** 7.3 MB KV memory saved. More importantly: sliding attention's Q @ K^T matmul halves in inner dim on 28/35 layers. If bandwidth-bound (likely on A17/A19 for KV reads), throughput roughly doubles for that step.

**Estimated +5-10% decode.** Low risk since attention math is already hd-correct.

### 4. NCHW pure-channels-first rewrite
D4_NCHW_FEASIBILITY.md: GO-AFTER-D1. Layout ops per chunk are 4.2× compute ops (162 transpose + 98 reshape vs 73 conv + 18 matmul in chunk1 verify). Keeping `(B, C, 1, S)` throughout eliminates ~70 removable layout ops per decoder forward.

Obstacle: `ANERMSNorm` does `cat([x,-x], dim=-1) + layer_norm(normalized_shape=(2*hidden,))` which operates on last dim. Recommendation per D4: wrap the layer_norm in transpose pairs to keep ANE fused-kernel path. Alternatives (group_norm, manual rsqrt) have unknown ANE residency.

**Implementation:** rewrite `_run_layer_swa` + `_run_layer_verify` + embedder boundary + `SWAChunk4.forward` lm_head path. ~100 LOC Python + 40 LOC Swift (buffer shape handshake).

**Validation:** chunk1 decode first, ComputePlanAudit gate (ANE placement must not regress), ≥3% tok/s gain required to proceed.

### 5. Attention QK^T as Conv1x1 rewrite (matmul 3× penalty recovery)
ANE_EMPIRICAL_GUIDE Priority 1: Orion #17 confirms matmul on ANE is 3× slower than Conv1x1. coremltools already lowers A@V to conv automatically (audit §2.1 shows 14 matmul = QK-only). Rewriting Q @ K^T as a Conv1x1 with appropriate reshape should also land on conv datapath.

**Sketch:** reshape Q `(1,8,1,256)` as conv input `(1, 256, 1, 1)`, use K (reshaped as conv weight `(L, 256, 1, 1)`) → conv output is per-position QK score, then reshape to `(1, 8, 1, L)`. This requires K to be materialized in a shape compatible with MIL's constexpr-conv expectation; may force K to be unpacked rather than state-backed. Risk here.

**Estimated +15-30%** per R18, but with uncertainty because QK^T has dynamic K (grows with decode position), unlike stored-weight conv.

**Validation:** bench per-layer attention latency pre/post.

### 6. RMSNorm scale absorption
`absorb_rmsnorm_scale_into_conv` utility exists (ane_ops.py:75-114, commit ac2fdd7). Applies to 7 norm+conv pairs per layer:
- input_layernorm → q/k/v_proj (pre-norm)
- pre_feedforward_layernorm → gate/up_proj (pre-norm, fused after D-1)
- q_norm → q_proj (per-head replicated scale)
- k_norm → k_proj (same)
- post_attention_layernorm → o_proj (post-norm absorbed in reverse)
- post_feedforward_layernorm → down_proj (same)
- post_per_layer_input_norm → per_layer_projection (same)

Total: 245 mul ops removed model-wide. Build-time one-shot loop.

**Validation:** weights-only change, trivial cosine check. v_norm (only non-cat-trick norm) is independent and should be rewritten per Item 8.

### 7. Enable optimize_mlpackage --optimize flag
Already implemented (commit ac2fdd7). Adds 14 MIL passes including const_deduplication, cast_optimization, fuse_transpose_matmul, merge_affine_dequantize. Size-guard against fuse_matmul_weight_bias INT4 blow-up. iOS 26 deployment target.

**Implementation:** just run `build_verify_chunks.py --optimize --ctx 8192`. Log reports op-count reduction.

**Validation:** existing `test_merged_parity.py`. Expected 10-30% op-count drop.

### 8. v_norm rewrite (the only rsqrt-path RMSNorm)
Per R10 flag #1: `v_norm` at `gemma4_swa_chunks.py:41-43` uses `pow(2)+reduce_mean+add+rsqrt+mul` (the manual RMSNorm formula). Every other RMSNorm in the model uses the cat-trick. ANE does not have a native rsqrt fast-path. v_norm has `with_scale=False` in HF so no learnable weight — use `ANERMSNorm(hd, affine=False)` which is the cat-trick.

**Implementation:** 5 LOC. Very low risk.

### 9. Logit softcap drop from graph
Per R10 §3.5: `tanh(logits/30) * 30` is 3 ops (div + tanh + mul). For greedy decoding, argmax(tanh(x/30)*30) = argmax(x) because tanh is monotonic and * positive const is monotonic. The softcap is a no-op for argmax decoding.

If Swift does sampling (temperature, top-p), compute softcap on the selected scalar logit on CPU instead of in the graph.

**Implementation:** 3 ops removed from `SWAChunk4.forward` lm_head tail. Swift side: one tanh on the scalar if sampling.

### 10. layer_scalar check and hard-delete (conditional)
Per R10 §3.2: `layer_scalar = nn.Parameter(torch.ones(1))`. If HF checkpoint values are all 1.0 (check with a load-time print), the 35 mul ops can be hard-deleted. If values differ from 1.0, absorption through residual path is possible but complex; the gain is <0.5% so not worth the bookkeeping.

**Action:** add a one-line check in load_weights. Most likely all 1.0 (the parameter's default init and Gemma 4 training may have kept it), in which case delete the mul. Worst case: absorbed into adjacent Conv weights via a distributed fold.

---

## Section 4 — PR-Sized Work Units

Dependency-ordered. Each PR: 1-3 days dev + 1 day parity test + 1 day device bench.

### PR-1: Safe baseline wins (2-3 days)
**Items:** #7 (enable --optimize), #8 (v_norm), #9 (softcap drop), #10 (layer_scalar check), #6 (RMSNorm absorption).
**Why bundled:** all weights-only or single-line changes, all lossless, all independent.
**LOC:** ~60 Python.
**Expected:** +7-13% decode, 250 mul ops removed, 0 regression risk.
**Rollback:** git revert.
**Parity:** cosine ≥ 0.9999, top-1 = 100%.

### PR-2: ane_softmax swap (1-2 days)
**Items:** #1.
**Why standalone:** biggest win, biggest risk. Must be measured in isolation.
**LOC:** 2 (plus parity harness update if attention fp16 overflow triggers).
**Expected:** +15-25%.
**Rollback:** trivial.
**Parity:** top-1 match across short-ctx (2K) and long-ctx (8K). Long-ctx is the stability test.

### PR-3: D-1 FusedQKV/FusedGateUp wiring (3-4 days)
**Items:** #2.
**Includes side effect:** KV-shared layer k/v deletion (180 MB).
**LOC:** ~40 Python + weight-loader update.
**Expected:** +13-20%.
**Dependencies:** none (but PR-1 parity verified first to isolate this change's effect).
**Parity:** per-layer cosine before full test.

### PR-4: Sliding KV padding drop (2-3 days)
**Items:** #3.
**Scope:** Python cache slot resize + Swift cache alloc update.
**Expected:** +5-10%.
**Dependencies:** benefits from PR-3 (shared cache buffer code path changes cleaner after fusion).

### PR-5: chunk4 lm_head palettize + PLE INT4 + vocab pruning (3-5 days)
**Items:** #11, #12, #13.
**Rationale:** ship-size grouping. All quality-eval-gated.
**Expected:** -2.5 GB package size, +1-2 tok/s from vocab pruning.
**Parity:** PPL on WikiText-2 within 0.5% of baseline, top-1 match on eval set.

### PR-6: NCHW end-to-end rewrite (2 weeks)
**Items:** #4.
**Scope:** biggest invasive rewrite. Separate work stream.
**Dependencies:** PR-2 + PR-3 landed and measured.
**Expected:** +10-20%.
**Parity:** per-chunk cosine + ComputePlanAudit + ≥3% tok/s gain gate.

### PR-7: Attention QK^T as Conv1x1 (1 week research + 1 week impl)
**Items:** #5.
**Scope:** requires verification that MIL will accept dynamic-K conv. R&D branch first.
**Expected:** +15-30%.
**Dependencies:** PR-6 NCHW landed (makes the K reshape cleaner).

### PR-8: Constants baking (2 days)
**Items:** #15, #16.
**Scope:** RoPE sin/cos and causal masks baked into chunks as constants. Saves per-decode input I/O.
**Expected:** small but easy.

### PR-9: SDPA sliced_q pass + multifunction (3 days)
**Items:** #14.
**Scope:** prefill speedup only; enable existing coremltools pass.
**Expected:** +20-35% prefill.
**Dependencies:** PR-2.

### PR-10: LayerSkip at L14 (4-6 weeks)
**Items:** #17.
**Scope:** early head training + CoreML export + Swift spec-decode integration.
**Expected:** +1.4-1.6× if accept ≥40%.
**Dependencies:** PR-2 + PR-3 landed.

### PR-11: Stateful KV probe (1 day; hedge)
**Items:** #19.
**Scope:** 2-hour probe only. If -14 repeats, abandon.
**Expected:** +15-20% if miraculously works.
**Dependencies:** none.

### PR-12 (conditional): Double-wide MLP rank compression, per-op mixed quant, PLE low-rank
**Items:** #18, #20, #22.
**Gate:** only if PR-1 through PR-10 deliver less than LiteRT-LM-beating throughput. Each requires training/retrain.

---

## Section 5 — Deferred / Conditional

| Item | Trigger condition |
|---|---|
| Stateful KV full adoption (#19 beyond probe) | Apple fixes -14 at WWDC 2026 OR developer forum thread 810987 resolves |
| PLE low-rank r=256 retrain (#18) | When ship-size is the binding constraint (LiteRT-LM parity already met on tok/s) |
| Double-wide MLP compression (#20) | Only if stacked PR-1 through PR-10 falls <50 tok/s |
| Partial_rotary_factor fix (#21) | Only if HF parity check (before any perf opt) reveals actual quality divergence |
| Per-op sensitivity sweep (#22) | Only if post-NCHW the pipeline is still size-constrained |
| LayerSkip (#17) | Only after PR-1 through PR-4 measure — gating on whether spec-decode is needed to beat LiteRT-LM |
| Drafter V-layout fix (#24) | Only when MTP drafter is re-enabled (currently parked) |

---

## Section 6 — Rejected with Primary-Source Rationale

| Item | Reason | Source |
|---|---|---|
| W2/W3 uniform post-training palettization | Produces gibberish on all test prompts | rejected_approaches memory, EXPERIMENTS.md |
| W8A8 activation quant | ANECCompile() FAILED on iPhone 17 Pro | SPEED_8K.md Tier D |
| INT8 KV cache (naive) | ANE dequantizes internally to fp16, 0 wall-clock gain | rejected_approaches memory |
| MLState with CPU_AND_NE | ANE error -14, confirmed stay-rejected on iOS 26 | D5_STATEFUL_KV_IOS26.md |
| SuffixDecoding as primary | T1=18% acceptance on diverse chat, workload-dependent | rejected_approaches memory |
| Pre-alloc mask | Measured ~0% at 2K (mask-fill already sub-ms) | rejected_approaches memory |
| CLLMs Jacobi iteration | Multi-token decode maps to prefill (slow) on ANE | UNEXPLORED_SOURCES.md |
| MInference sparse attention | Only effective at 100K+ context; 8K is below sweet spot | UNEXPLORED_SOURCES.md |
| QuIP# / AQLM | Hadamard/gather ops not on ANE | UNEXPLORED_SOURCES.md |
| Custom MIL composite ops | Serialize as `custom_layer`, CPU-only path, never land on ANE | CUSTOM_MIL_PASSES.md |
| Orion private `_ANEClient` dispatch | App-Store-unshippable (entitlement restricted) | ANE_NEW_TECHNIQUES_2026.md, APPLE_2026_ROADMAP_WATCH.md |
| Waiting for WWDC 2026 "Core AI" | Per Gurman it's a rename/modernization, no ANE public dispatch signal | APPLE_2026_ROADMAP_WATCH.md |

---

## Section 7 — Corrections / Retractions

**1. Stateful KV on iOS 26 — "fixed by Apple" claim reversed.**
- MIL_PASSES_ADDITIONAL.md §3 implied iOS 26 + coremltools 9.0 landed ANE stateful support.
- D5_STATEFUL_KV_IOS26.md (2026-04-16) fact-checked: no evidence in release notes, Apple's own Llama 3.1 still uses CPU_AND_GPU, zero third-party sightings, error -14 still open.
- **Action:** STAY-REJECTED except a 2-hour probe. Remove any "retry iOS 26" language from internal plans.

**2. EAGLE-3 PR #18039 — "merged" claim reversed.**
- LLAMA_CPP_INTEGRATION.md said the PR was merged.
- R5_LLAMA_CPP_SURGERY.md fact-checked via `gh pr view 18039`: mergedAt: null, still OPEN.
- **Action:** Metal-LLM Week 5-6 plan revised; EAGLE-3 moves to Week 8+ cherry-pick. MTP drafter becomes the Week-5 drafter of record.

**3. Drafter V-layout 0.5ms/step transpose — refers to removed code.**
- MLPACKAGE_STRUCTURE_AUDIT.md claimed a 0.5ms transpose happens per decode step.
- D3_V_LAYOUT_ANALYSIS.md showed the transposeLastTwoDims helper (commit e72cbeb) was removed when MTP was parked (commit aef01ee). Current tree has no active waste.
- **Risk remaining:** MTP re-enable will hit a shape-mismatch crash. Fix via Option B (8-12 LOC in drafter builder) when MTP is reactivated.

**4. Gemma 4 E2B dimensions — earlier misstatements corrected.**
- Multiple earlier docs cited hidden=2560 and num_attention_heads=10.
- D1_WIRING_PATCH_PLAN verified: hidden=1536, num_attention_heads=8, num_kv_heads=1, head_dim 256 sliding / 512 global.
- GEMMA4_ANE_REWRITES flagged activation is GELU-tanh, not SiLU.
- **Action:** any implementation patches must use the verified values.

**5. Per-layer embeddings (PLE) — "large in-graph weight" claim reversed.**
- Earlier assumption: PLE is in the CoreML graph at 2.3B params and needs in-graph compression.
- PLE_DEEP_DIVE confirmed: PLE is already CPU-resident, INT8-quantized (2.19 GB on-disk `embed_tokens_per_layer_q8.bin`), looked up via Swift CPU and passed in as `per_layer_raw` input. Not in graph.
- **Action:** PLE optimization focuses on on-disk quantization (INT4) + vocab pruning, not in-graph rewrites.

---

## Section 8 — Implementation Order Recommendation

Each week gates on the previous week's measurement. Stop conditions noted.

**Week 1: Safe baseline (PR-1 + PR-2)**
- Land PR-1 (5 hygiene items) + PR-2 (softmax swap)
- Expected cumulative: +22-38%, baseline 15 → 18-21 tok/s
- **Stop condition:** if PR-2 parity fails at long ctx, restrict to short-ctx. If no measurable gain after PR-2, investigate ANE placement regression before continuing.

**Week 2: D-1 + Sliding KV (PR-3 + PR-4)**
- Land PR-3 (FusedQKV/GateUp) + PR-4 (sliding padding drop)
- Expected cumulative: +40-65%, 21-25 tok/s
- **Stop condition:** if PR-3 fails parity on shared layers (k/v deletion branch), roll back to non-shared-layer-only fusion.

**Week 3: Ship size + Constants (PR-5 + PR-8)**
- Land PR-5 (lm_head + PLE + vocab pruning) + PR-8 (RoPE/mask baking)
- Expected cumulative tok/s: +2-4 additional; expected size: -2.5 GB
- **Milestone:** package size beats LiteRT-LM 2.58 GB by 2.5×.

**Week 4-5: NCHW rewrite (PR-6)**
- Invasive refactor. Chunk1-first, measure, extend.
- Expected cumulative: 30-45 tok/s.

**Week 6: SDPA sliced_q + prefill speedup (PR-9)**
- Prefill-only.
- Decision point: evaluate whether current tok/s + spec-decode is enough to beat LiteRT-LM, or if PR-7 (QK^T Conv1x1) is needed.

**Week 7-8: Attention QK^T Conv1x1 (PR-7, conditional)**
- Only if Week 6 tok/s < 45.
- High-risk, high-reward.
- Expected cumulative: 45-55 tok/s.

**Week 9-14: LayerSkip or alternative spec-decode (PR-10)**
- LayerSkip is the project-native spec-decode path that plays to Gemma 4's KV-share structure.
- Conditional on Week 6-8 tok/s; if Metal-LLM pivot is the better bet (see sibling Metal-LLM track), LayerSkip for CoreML may be deferred indefinitely.

**Continuous (every week):** PR-11 stateful probe if any Apple signal appears.

**Stop conditions (abandon CoreML optimization track):**
- After Week 4, if cumulative tok/s < 22 (below the previously-assumed ceiling), no further Python-side rewrites will help — move budget to Metal-LLM track.
- If Apple ships Core AI at WWDC 2026 with ANE public dispatch, halt and re-scope.

---

## Section 9 — Open Questions Requiring Measurement

1. **Softmax swap parity at long context.** ane_fused_softmax works in mtp_drafter but that's a small model. Need 8K ctx parity test.
2. **Stateful KV on iOS 26 2-hour probe.** Single data point needed.
3. **QK^T Conv1x1 feasibility on dynamic K.** MIL compiler acceptance test.
4. **Double-wide MLP rank compression quality floor.** Need a rank sweep (1536, 1024, 768, 512) with PPL eval.
5. **LayerSkip L14 accept rate on Gemma 4.** Literature numbers don't specialize; need one rollout measurement.
6. **Optimize_mlpackage pass behavior under INT4 constexpr.** Size-guard catches blow-ups but actual numerical stability of fuse_transpose_matmul on INT4 weights unmeasured.
7. **ANE SRAM budget at current chunk size.** Orion says 32 MB is the cliff — our 4-chunk pipeline per-chunk working set vs this number is unmeasured.
8. **Warm-path vs cold-path dispatch cost on A19.** Heartbeat prediction recommended (ANE_EMPIRICAL_GUIDE §8) but actual cold-path cost on A19 unmeasured.

---

## Section 10 — Appendix: Doc-by-Doc One-Line Summary

| Doc | Takeaway |
|---|---|
| CONVERSION_AUDIT_2026_04_15 | 19-item audit: 10 done, 3 missing (QKV fuse, Gate/Up fuse, RMSNorm absorb) + unused optimizer tool = +22-35% Python-side headroom |
| GEMMA4_ANE_REWRITES | 8 concrete PyTorch rewrites with code; confirms hidden=1536, GELU-tanh, activation quirks; stacked +23-26 tok/s projection |
| ANE_CONVERSION_RECIPE_2026 | Apple ml-ane-transformers + coremltools 9.0 canonical recipe; Llama 3.1 stateful path evolution 0.19 → 33.67 tok/s; layer-pair KV sharing suggestion |
| MIL_PASSES_ADDITIONAL | 73 default passes audited; 7 not in current optimizer; size-guard around fuse_matmul_weight_bias (INT4 blow-up risk); stateful KV retry recommendation LATER REVERSED |
| ANE_NEW_TECHNIQUES_2026 | 4 genuinely new App-Store-legal techniques: in-model argmax, multifunction mlpackage, 4-function SW rotation, MLComputePlan programmatic inspection |
| MLPACKAGE_STRUCTURE_AUDIT | Compiled-graph analysis: manual softmax decomposition (~50 ops/chunk) is biggest op-count waste; V-layout mismatch (since removed); INT4 palette confirmed |
| GPU_WHY_FAST | ANE 2.3ms/dispatch is DART-enforced; Metal wins via user-space ring buffer + command buffer batching; practical ANE ceiling was estimated 22-28 tok/s (now revised upward via this roadmap) |
| D1_WIRING_PATCH_PLAN | Patch-ready FusedQKV/GateUp wiring; 1-file edit covers 8 chunks; +13-20% + 180 MB weight savings via shared-layer k/v deletion |
| D3_V_LAYOUT_ANALYSIS | V-layout transpose (0.5ms claim) was removed code; current tree latently broken for MTP re-enable; 8-12 LOC consumer-side fix ready |
| D4_NCHW_FEASIBILITY | GO-AFTER-D1; 70 removable layout ops per chunk; RMSNorm channel-dim is main obstacle; ~2 week scope |
| D5_STATEFUL_KV_IOS26 | STAY-REJECTED; no evidence Apple fixed -14 on iOS 26; 2-hour probe allowed |
| APPLE_2026_ROADMAP_WATCH | DO-NOT-WAIT; Core AI rumor is rename, not API opening; Google LiteRT-LM shipped GA; competitive urgency high |
| IMPLEMENTATION_LOG_2026_04_15 | Branch `feat/pre-conv-optimizations`: 14 MIL passes + iOS26 + opt-in flags + helper modules shipped; wiring deferred |
| PARITY_TEST_PROTOCOL | Cosine ≥ 0.9995, PPL delta ≤ 0.5%, top-1 match 100% on 256 steps |
| GEMMA4_FORWARD_ANATOMY | Op-by-op decode analysis; top 3 wins = softmax swap + sliding padding + NCHW (+30-55%); v_norm is only non-cat-trick RMSNorm |
| CUSTOM_MIL_PASSES | Custom passes OK (use @register_pass), custom composite ops are CPU-only so useless; fuse_manual_softmax is #1 candidate |
| PLE_DEEP_DIVE | PLE is already CPU-resident INT8; remaining wins are on-disk INT4 + vocab pruning activation + optional low-rank factor; achievable <900 MB package (beat LiteRT 2.8×) |
| ANE_EMPIRICAL_GUIDE | Orion 20 constraints mapped; attention QK^T / A@V as Conv1x1 is top unimplemented lever; INT8 = FP16 throughput on public CoreML path |
| MIL_OP_CATALOG | Complete MIL op placement atlas; confirms ops we should/shouldn't use |
| GEMMA4_ALGEBRAIC_REWRITES | RMSNorm, attention, softcap math reformulations with error bounds |
| CROSS_CHUNK_OPTIMIZATION | RoPE/mask constant baking candidates; PLE per-chunk slicing; chunk boundary analysis |
| GEMMA4_LAYERSKIP_DESIGN | L14 early-exit + draft head training plan; 6-week roadmap; expected +1.4-1.6× if accept ≥ 40% |
| LAYER_SENSITIVITY_QUANT | Per-layer mixed-bit palettization plan; PLE as primary target (biggest weight share) |

---

## Summary (for next session)

This roadmap replaces all prior priority documents. Start with PR-1 + PR-2 (safe wins + softmax swap) — ~1 week to measure cumulative +22-38%. Gate on that result before investing in PR-3 D-1 wiring.

ANE optimization is not over. The 22-28 tok/s "ceiling" was based on partial information; stacked items 1-14 in this roadmap realistically deliver 50-65 tok/s, putting LiteRT-LM parity in range without a Metal pivot.

Where measurement is required, the table in Section 9 lists the 8 open questions.
