# Gemma 4 E2B — Single-Token Decode Forward Anatomy

Source of truth:
- `conversion/models/gemma4.py`
- `conversion/models/gemma4_swa_chunks.py` (`_run_layer_swa` L46–181; `SWAChunk1._compute_ple` L224–254; `SWAChunk4.forward` L442–450)
- `conversion/ane_ops.py` (`ANERMSNorm` L25–72; `apply_rotary_pos_emb` L204–213; `ane_softmax` L216–241; `absorb_rmsnorm_scale_into_conv` L75–114)
- `conversion/models/gemma4_wrapper.py` (L103–119, non-chunked PLE)
- `docs/MLPACKAGE_STRUCTURE_AUDIT.md` (compiled op counts)
- `Sources/CoreMLLLM/ChunkedEngine.swift` (L751: per_layer_raw CPU path)
- `Sources/CoreMLLLM/ModelDownloader.swift` (L532: `embed_tokens_per_layer_q8.bin` 2.35 GB)

Shapes below: `B=1, Q=1, H=1536, nh=8, nkv=1, hd=256|512, I=6144|12288, W=512, ctx=depends`. All FP16.

---

## 1. Per-layer op inventory (Q=1 decode)

### 1.A Pre-attention path

| # | Src | PyTorch | MIL emission | Shape | Cat | Tag | Notes |
|---|---|---|---|---|---|---|---|
| 1 | L66 | `input_layernorm(hs)` | `concat` + `layer_norm` + `split` + `mul(weight)` | (1,1,1536) | compute+layout | **ABSORBABLE** | Cat-trick; final `mul(norm.weight)` foldable into each of q/k/v_proj (replicate scale across 3 convs). `absorb_rmsnorm_scale_into_conv` exists, NOT wired to SWA path. |
| 2 | L67 | `h.permute(0,2,1).unsqueeze(2)` | `transpose` + `expand_dims` | (1,1,1536)→(1,1536,1,1) | layout | **REMOVABLE** | NHWC↔NCHW dance per audit §4.3 |
| 3 | L70 | `q_proj(x)` | `conv` 1×1 INT4 | (1,1536,1,1)→(1,2048,1,1) | compute | **NECESSARY** | 3.1M FMAs |
| 4 | L70 | `.view(1,8,256,1).permute(0,1,3,2)` | `reshape` + `transpose` | →(1,8,1,256) | layout | **REMOVABLE** |
| 5 | L71 | `q_norm(...)` + reshape | `reshape` + cat-trick + `reshape` | (1,8,1,256) | compute+layout | **ABSORBABLE** | `q_norm.weight` (256 floats) folds into q_proj by per-head replication (8 copies along out_channels). |
| 6 | L73-75 | `apply_rotary_pos_emb(q)` | `mul`+`split`+`neg`+`concat`+`mul`+`add` | (1,8,1,hd) | aux | **PARTIALLY REMOVABLE** | Global head_dim=512 with partial_rotary_factor=0.25 implies only first 128 should rotate; we compute full 512. See §3.7. |

### 1.B KV projections (L0-L14 only)

| # | Src | PyTorch | MIL | Shape | Cat | Tag |
|---|---|---|---|---|---|---|
| 7-8 | L84-85 | `k_proj`, `v_proj` | 2×`conv` 1×1 INT4 | (1,1536,1,1)→(1,256 or 512,1,1) | compute | **NECESSARY** |
| 9 | L84-85 | view/permute | 2×(reshape+transpose) | →(1,1,1,hd) | layout | **REMOVABLE** |
| 10 | L86 | `k_norm` | cat-trick | (1,1,1,hd) | compute+layout | **ABSORBABLE** into k_proj |
| 11 | L87 | `v_norm` | `pow(2)`+`reduce_mean`+`add`+`rsqrt`+`mul` | ≈5 ops | compute | **REPLACEABLE** | **The ONLY non-cat-trick RMSNorm in the model.** Swap to cat-trick with `affine=False` (no weight). |
| 12 | L88-91 | RoPE on K | same 6-op | | aux | see #6 |
| 13 | L93-97 | pad sliding hd=256→512 | `pad` | → (1,1,1,512) | memory | **REMOVABLE** (§3.9) |
| 14 | L101-109 | KV update (sliding shift or full mask) | `slice`+`concat` OR `mul`+`sub`+`mul`+`add` | (1,1,L,max_hd) | memory | **NECESSARY** |

### 1.C Attention body

| # | Src | PyTorch | MIL | Shape | Cat | Tag |
|---|---|---|---|---|---|---|
| 15 | L132-133 | `repeat_interleave(n=8)` | `expand_dims`+`tile`+`reshape` | (1,1,L,hd)→(1,8,L,hd) | layout | **NECESSARY** (GQA) |
| 16 | L140 | `matmul(q, K^T)` | `transpose`+`matmul` | →(1,8,1,L) | compute | **NECESSARY** |
| 17 | L141 | `+mask` | `add` | | aux | **NECESSARY** |
| 18 | L142 | `ane_softmax` | `reduce_max`+`sub`+`exp`+`reduce_sum`+`real_div` (35 in stateful_chunk2) | (1,8,1,L) | compute | **REPLACEABLE** | Swap to `ane_fused_softmax` → single `softmax` op. mtp_drafter already does this. |
| 19 | L143 | `matmul(attn_w, V)` | **lowered to `conv`** by coremltools (audit §3) | →(1,8,1,hd) | compute | **NECESSARY** |
| 20 | L145 | permute+reshape | | →(1,1,nh·hd) | layout | **REMOVABLE** (NCHW) |
| 21 | L147 | `o_proj` w/ layout dance | `transpose`+`expand`+`conv`+`squeeze`+`transpose` | →(1,1,1536) | compute+layout | **NECESSARY conv**; layout REMOVABLE |
| 22 | L149 | `post_attention_layernorm` | cat-trick | (1,1,1536) | compute | **ABSORBABLE** into o_proj |
| 23 | L150 | `residual + attn_out` | `add` | | elementwise | **NECESSARY** |

### 1.D MLP path

| # | Src | PyTorch | MIL | Shape | Cat | Tag |
|---|---|---|---|---|---|---|
| 24 | L154 | `pre_feedforward_layernorm` | cat-trick | (1,1,1536) | compute | **ABSORBABLE** into fused gate+up |
| 25 | L155 | layout dance | | (1,1536,1,1) | layout | **REMOVABLE** |
| 26 | L156-157 | gate+up (fused by coremltools per audit) | 1 `conv` → `split` | (1,2·I,1,1) | compute | **NECESSARY** |
| 27 | L158 | `F.gelu(approx='tanh')` | `gelu` native | | elementwise | kept |
| 28 | L159 | `gate * up` | `mul` | (1,I,1,1) | elementwise | **NECESSARY** |
| 29 | L159 | `down_proj` | `conv` | →(1,1536,1,1) | compute | **NECESSARY** |
| 30 | L160 | layout dance | | | layout | **REMOVABLE** |
| 31 | L161 | `post_feedforward_layernorm` | cat-trick | (1,1,1536) | compute | **ABSORBABLE** into down_proj |
| 32 | L162 | `residual+hs` | `add` | | elementwise | **NECESSARY** |

### 1.E PLE block (105 ops across 35 layers)

| # | Src | Op | Tag |
|---|---|---|---|
| 33 | L168 | slice PLE (1,1,8960)→(1,1,256) | NECESSARY |
| 34 | L169 | layout | REMOVABLE |
| 35 | L170 | `per_layer_input_gate` conv 1536→256 | NECESSARY (tiny, 0.4M FMAs) |
| 36 | L171 | gelu-tanh | kept |
| 37 | L172 | layout | REMOVABLE |
| 38 | L173 | `gated * ple_slice` | NECESSARY |
| 39 | L174 | `per_layer_projection` conv 256→1536 | NECESSARY |
| 40 | L175 | layout | REMOVABLE |
| 41 | L176 | `post_per_layer_input_norm` | ABSORBABLE into per_layer_projection |
| 42 | L177 | `residual_pl + hs` | NECESSARY |
| 43 | L178 | `hs * layer_scalar` | ABSORBABLE but requires distributing through residual path (§3.2) |

**Per-layer MIL live ops: ~50-70** (excl. const binds). Total model: ~2200 live ops.

### 1.F Model-level ops (once per decode)

- **Token embed**: NOT in CoreML graph. Swift CPU lookup (ChunkedEngine.swift L751), `per_layer_raw` passed in.
- **PLE compute** (`_compute_ple` L224-254): projection 1536→8960, scale, **ONE** cat-trick layer_norm (comment: "35 separate norms + 34 concats = ~100 MIL ops" reduced to 1). Already optimized.
- **Final norm + lm_head + softcap + argmax** (`SWAChunk4.forward` L442-450):
  - `norm(hs)` cat-trick — ABSORBABLE into lm_head (but FP16 weight, ~400 MB per audit §2.3)
  - `lm_head` conv 1536→262144 — NECESSARY
  - `tanh(logits/softcap)*softcap` — ABSORBABLE, see §3.5
  - `argmax` + `gather_along_axis` — NECESSARY

---

## 2. Op census summary (stateful_chunk2, 14 layers)

| Category | /14L chunk | /layer | /35L model |
|---|---|---|---|
| const (weights) | 1458 | 104 | 3640 |
| constexpr_lut_to_dense | 63 | 4.5 | 158 |
| conv | 63 | 4.5 | 158 |
| matmul (QK only; A·V→conv) | 14 | 1 | 35 |
| mul | 175 | 12.5 | 437 |
| transpose | 126 | 9 | 315 |
| reshape | 84 | 6 | 210 |
| slice | 41 | 2.9 | 102 |
| concat | 73 | 5.2 | 182 |
| split | 63 | 4.5 | 158 |
| softmax-decomposed primitives | 35 | 2.5 | 88 |
| gelu | 14 | 1 | 35 |
| **Total MIL** | 2473 | 177 | ~6200 |
| Live ops (excl const) | ~1000 | ~63 | ~2200 |

Per-chunk decode op count ≈ 2746 (chunk1 audit §2.2). 4 chunks → ~10K ops model-wide, ~60% const binds.

---

## 3. Gemma 4 quirks with exploit potential

### 3.1 PLE — ALREADY off-graph
`embed_tokens_per_layer` (2.35B params) is INT8-quantized, CPU-resident, looked up per step, fed to graph as `per_layer_raw`. Not in CoreML graph. Exploits: W4 palette on the disk blob (−1.2 GB), top-K token cache, rank-256 factorization of `per_layer_model_projection`.

### 3.2 Layer scalar — 35 muls
`layer_scalar = Parameter(ones(1))` applied at L178. Folding through residual requires distributing scalar through entire block (o_proj, down_proj, per_layer_projection weights + 3 pre-norms). **Check: if loaded HF values are all 1.0, hard-delete the op.** &lt;0.5% impact.

### 3.3 QK-norm — 70 RMSNorms model-wide
Scale weights (256 floats each) absorbable into q_proj/k_proj by per-head replication (8 copies for q). Cat-trick body stays. +0.5-1% decode.

### 3.4 Sandwich norm — 4 RMSNorms/layer × 35 = 140
- `input_layernorm` + `pre_feedforward_layernorm`: pre-norms → absorb into next Conv (existing `absorb_rmsnorm_scale_into_conv` works directly)
- `post_attention_layernorm` + `post_feedforward_layernorm`: post-norms before residual → absorb into last Conv of X (o_proj, down_proj)
- All 4 per-layer scale muls foldable. **140 muls removed, +1-2% decode.**

### 3.5 Logit softcap `tanh(x/30)*30`
- 3 ops (div, tanh, mul)
- **For greedy argmax: entirely no-op** (argmax is monotonic under tanh*const). Can delete from graph.
- For sampling: pre-scale lm_head by 1/30 → 2 ops, OR compute tanh in Swift on the scalar gathered value
- Recommendation: drop from graph, compute on scalar in Swift if sampling used

### 3.6 Per-layer input block (105 ops / model)
Architectural Gemma 4, not scaffolding. Convs are already rank-256 factorization (1536→256→1536). Only `post_per_layer_input_norm` scale is absorbable (+35 muls removed).

### 3.7 Global RoPE partial_rotary_factor=0.25
Code comment says factor is NOT used for inv_freq; we compute full 512-dim cos/sin. If HF's actual semantics only rotate first 128, **we have a quality bug**. Verify with HF reference parity first. If bug confirmed: split q into rot/pass channels, rotate 128 only.

### 3.8 Dual head_dim (256 sliding / 512 full) — **BIG WIN**
Sliding layers use hd=256 but KV cache is padded to max_hd=512. **28 sliding layers store 2× too much.** Fix: split KV storage (`K_sliding: (nsl,1,W,256)` + `K_full: (nf,1,ctx,512)`), remove F.pad at L93-95.

Savings at ctx=8192:
- KV storage: 14.7 MB → 7.3 MB sliding
- **Sliding path attention FLOPs halved** (28/35 layers' K·Q shape halved)
- If bandwidth-bound at that step: **2× sliding attention throughput**

**Top-5 win. Low risk — K/V slicing math already hd-correct at read site.**

### 3.9 Double-wide MLP for L15-L34
20 layers have intermediate×2=12288. MLP dominates decode compute (754M FMAs for shared layers vs 315M for narrow). Rank-compressing down_proj from 1536 to 768-1024 could reclaim 25-33% MLP compute. **Requires quality eval — training-time choice.**

### 3.10 chunk4 lm_head 400 MB FP16
Audit §4.8: palettize to W4g32 → 503 MB → ~200 MB ship size. 0 latency impact.

---

## 4. Projected op census after optimizations

| # | Change | Ops removed model-wide | Risk | Est Δ tok/s |
|---|---|---|---|---|
| 1 | `ane_softmax` → `ane_fused_softmax` | 110 | Med (fp16 attn overflow) | **+15-25%** |
| 2 | Absorb 4 sandwich norms | 140 | Very low | +1-2% |
| 3 | Absorb q/k_norm scales | 70 | Very low | +0.5-1% |
| 4 | Absorb layer_scalar | 35 | Low | &lt;0.5% |
| 5 | Absorb post_per_layer_input_norm | 35 | Very low | &lt;0.5% |
| 6 | Replace v_norm rsqrt w/ cat-trick | ~15 | Very low | &lt;0.5% |
| 7 | Remove softcap from graph | 3 | Very low | ~0 |
| 8 | Drop sliding KV padding (§3.8) | 28 pad + half sliding attn FLOPs | Med (cache layout) | **+5-10%** |
| 9 | Enable SDPA sliced_q pass | hundreds | Med (needs #1) | +20-35% prefill |
| 10 | Pure NCHW rewrite | ~400 layout ops | Med | **+10-20%** |
| 11 | Palettize chunk4 lm_head INT4 | 0 ops, -300 MB ship | Med (quality eval) | ~0 latency |
| 12 | Verify/fix partial_rotary_factor | — | High (correctness) | parity fix |

**Stacked #1-#8**: 435 ops removed, **+20-35% decode**.
**With #10 NCHW**: **+30-50%** plausible.
**All 1-12**: **+40-70% over baseline**. At 30 tok/s baseline → **42-51 tok/s** projected. Still short of LiteRT 56.5 without rank-compressing double-wide MLP or fused attention.

---

## 5. Priority ranking (Δtok/s / LOC·risk)

| Rank | Change | LOC | Risk | Δ |
|---|---|---|---|---|
| 1 | **ane_softmax → ane_fused_softmax** in `_run_layer_swa` L142, `_run_layer_verify` L595 | 2 | Med | +15-25% |
| 2 | **Drop sliding KV padding** (§3.8) | ~30 Py + Swift cache alloc | Med | +5-10% |
| 3 | **Pure-NCHW rewrite** of `_run_layer_swa` layout pairs | ~100 | Med | +10-20% |
| 4 | **Wire SDPA sliced_q pass** in verify/merged builds | 5 | Low (after #1) | +20-35% prefill |
| 5 | **Absorb all RMSNorm scales** (4 sandwich + 2 QK + post_ple) using existing util | ~40 | Very low | +1-3% |
| 6 | **Replace v_norm** with cat-trick + affine=False | 5 | Very low | &lt;0.5% |
| 7 | **Palettize chunk4 lm_head** W4g32 | 10 | Med | 0 latency, −300 MB ship |
| 8 | **Drop softcap** from graph | 3 | Very low | ~0 |
| 9 | **Absorb layer_scalar** (OR hard-delete if all=1.0) | 20 (or 2) | Low | &lt;0.5% |
| 10 | **Verify partial_rotary_factor** semantics vs HF | 10 | High correctness | — |
| 11 | **Rank-compress double-wide MLP** L15-L34 | 50 + eval notebook | High | +5-15% if holds |

**Top 3 combined ≈ +30-55% decode tok/s for ~150 LOC.** Fastest path to closing the gap to LiteRT-LM 56.5 tok/s.

---

## Flags / unexpected findings

1. **v_norm is the only non-cat-trick RMSNorm in the entire model** — rsqrt-path, ANE unfriendly. Easy fix.
2. **A·V matmul is already lowered to `conv` by coremltools** (audit §2.1: 14 matmul = QK only). This is actually good but explains why SDPA fusion is fragile (compiler has to recognize matmul+softmax+conv pattern).
3. **RoPE tables passed as runtime inputs** (cos_s/sin_s/cos_f/sin_f) — easy bakable.
4. **Two codepaths**: stateful vs stateless. Audit §3 recommends consolidation. Production is stateless.
5. **layer_scalar is loaded from HF but may be all 1.0** — if so, hard-delete instead of absorbing.
