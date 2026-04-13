# MTP Path A — LiteRT probe findings & integration design

**Status:** 2026-04-13 investigation. E2B MTP drafter EXTRACTED (44.3 MB). E2B-specific I/O confirmed. **L34 parity blocker was a FALSE ALARM — our Gemma4Model matches HF(cache=True) at all 35 layers (rel diff < 1e-5). No fix needed.** Ready for PyTorch reimplementation + CoreML conversion.

**TL;DR:** Google *does* ship Multi-Token Prediction as a separate TFLite `mtp_drafter` section inside `gemma-4-E2B-it.litertlm`. It is **not a multi-head MTP** as the V5 doc hypothesized — it is a **4-layer tiny transformer** that reads the target's hidden state and the target's shared KV caches (`kv_cache_k_13`/`kv_cache_v_13` for SWA, `kv_cache_k_14`/`kv_cache_v_14` for full attn — exactly our `kv13`/`kv14`). Structurally it behaves like EAGLE‑3 but is purpose-trained by Google against Google's reference forward, and is only 44.3 MB (vs our 47 M-param 188 MB EAGLE-3 draft). Extraction is done. Conversion to ANE CoreML is feasible (all ops map). **The previously suspected L34 parity blocker was an indexing artifact in the comparison harness** — `output_hidden_states[35]` is the post-norm output, not L34's raw output. Our forward is correct.

---

## 1. What Google ships

`litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm` is a `.litertlm` FlatBuffer container. Its `model.toml` manifest (confirmed via the E4B mirror) enumerates 11 sections. Section 11 is:

```toml
[[section]]
model_type = "mtp_drafter"
backend_constraint = "cpu"
section_type = "TFLiteModel"
data_path = "Section11_TFLiteModel_tf_lite_mtp_drafter.tflite"
```

`backend_constraint = "cpu"` is Google's own deployment choice (XNNPACK). It does **not** mean the drafter is incompatible with ANE — the ops all map (see §3).

Size: **45.1 MB** for E4B. E2B will be smaller (scales with hidden dim ratio 5120→2560, so ~20–30 MB expected).

HF discussion cites an official Google statement (google/gemma-4-E4B-it discussions/5):
> "The LiteRT exported models may include additional prediction heads for MTP. These are preserved in the exported graph because LiteRT runtime can leverage them for speculative/parallel decoding."

---

## 2. Architecture (reverse-engineered from E4B, ops + FC dims)

All dims below are E4B. E2B numbers will be confirmed once the download finishes.

```
Input tensors (from signature 'mtp_drafter'):
  activations          (1, 1, 5120)     fp32   ← target model's hidden state
  input_pos            (1,)             int32
  mask                 (1, 1, 1, 32003) bool
  param_tensor         (1, 1, 1, 7)     int32  ← rope position + flags
  kv_cache_k_22        (1, 2, 32003, 256) int8
  kv_cache_v_22        (1, 2, 256, 32003) int8
  kv_cache_k_23        (1, 2, 32003, 512) int8
  kv_cache_v_23        (1, 2, 512, 32003) int8

Output tensors:
  logits               (1, 1, 262144)   fp32   ← full next-token distribution
  projected_activations (1, 1, 2560)    fp32   ← carry state for K-step MTP
```

### Layer stack (FULLY_CONNECTED dims confirm)

```
mtp_pre_proj   Linear(5120 → 256)              ← project target hidden down

── layer_0  (SWA, uses kv_cache_*_22) ──
  RMSNorm
  pre_q       Linear(256 → 1024)              ← Q only (4 heads × 256)
  (attention reads K/V from cache_22, no k/v projection in drafter)
  post_q      Linear(1024 → 256)
  RMSNorm
  gate1       Linear(256 → 2048)
  gate2       Linear(256 → 2048)
  GeGLU: gelu(gate1) ⊙ gate2
  down        Linear(2048 → 256)

── layer_1  (SWA, kv_cache_*_22) ──  [same shape]
── layer_2  (SWA, kv_cache_*_22) ──  [same shape]
── layer_3  (FULL, kv_cache_*_23) ──
  pre_q       Linear(256 → 2048)              ← head_dim=512 (full-attn split)
  post_q      Linear(2048 → 256)
  (MLP same shape as layers 0–2)

── head ──
  RMSNorm
  embedder.decode  Linear(256 → 262144)       ← tied to target's embedding
  mtp_post_proj    Linear(256 → 2560)         ← produces projected_activations
```

**Composite ops used** (from TFLite subgraph names): `odml.rms_norm` ×21, `odml.runtime_bmm` ×8. That is: 5 RMSNorms per layer × 4 layers = 20, + final RMSNorm = 21; and attention BMMs. All lower to standard patterns we already handle in `conversion/ane_ops.py` (ANERMSNorm + Conv1×1 linears).

### E2B confirmed dimensions

E2B extraction complete (`output/mtp_probe/section_10.tflite`, 44.3 MB). Verified I/O signatures:

```
Input tensors (from E2B mtp_drafter signature):
  activations          (1, 1, 3072)     fp32   ← 2× E2B hidden_size (1536)
                                                  likely [hidden_state, embed_next_token]
                                                  or [hidden_state, projected_activations_prev]
  kv_cache_k_13        SWA, 256 head_dim  ← exactly our ChunkedEngine kv13
  kv_cache_v_13        SWA, 256 head_dim
  kv_cache_k_14        full, 512 head_dim ← exactly our ChunkedEngine kv14
  kv_cache_v_14        full, 512 head_dim

Output tensors:
  logits               (1, 1, 262144)   fp32   ← full next-token distribution
  projected_activations (1, 1, 1536)    fp32   ← same as E2B hidden_size
```

Key layer dimensions:
```
mtp_pre_proj    Linear(3072 → 256)
mtp_post_proj   Linear(256 → 1536)
embedder.decode Linear(256 → 262144)   ← LM head, likely weight-tied

layer_0  SWA  (q=1024, FFM=2048)
layer_1  SWA  (q=1024, FFM=2048)
layer_2  SWA  (q=1024, FFM=2048)
layer_3  full (q=2048, FFM=2048)
```

Notable: activations input is 3072, not 1536. This is 2× `hidden_size` — the drafter concatenates two 1536-dim vectors as input. The second half is either (a) the embedding of the draft token, or (b) the `projected_activations` output from the previous MTP step. Determining which is the **biggest remaining unknown** before PyTorch reimplementation.

### Why only 4 layers + KV-share

Layers 0–2 share `kv_cache_22`; layer 3 uses `kv_cache_23`. These are **not the drafter's own K/V** — there are *no* k_einsum / v_einsum projections anywhere in the drafter graph. The drafter **reads the target's KV caches directly**.

For E4B, indices 22/23 are the KV-producer layers (Gemma 4 E4B has more layers than E2B, so local/global split lands at 22/23). For **E2B the equivalent layers are L13 (sliding) and L14 (full)** — exactly the caches we already expose as `kv13` / `kv14` in `ChunkedEngine`.

This is the FUNDAMENTAL_UNTRIED.md §4 "LayerSkip on KV-share boundary" pattern. Google shipped it as MTP.

---

## 3. ANE compatibility audit

Ops used in the main subgraph (from tflite.Model inspection):

| TFLite op | ANE mapping |
|---|---|
| FULLY_CONNECTED | Conv1×1 (already standard pattern in this repo) |
| STABLEHLO_COMPOSITE `odml.rms_norm` | ANERMSNorm (cat+LayerNorm+slice, already implemented) |
| STABLEHLO_COMPOSITE `odml.runtime_bmm` | BMM — GQA broadcast matmul (Phase 1 item 4 pattern) |
| MUL, ADD, SUB, RESHAPE, SLICE, CONCATENATION | native |
| DEQUANTIZE (INT8 KV → fp16) | Handled by existing palettization/int8-kv patterns |
| QUANTIZE | Output-side, drop on ANE fp16 path |
| SOFTMAX | Native (explicit fp16 casts per CLAUDE.md) |
| SIN, COS | Precompute RoPE tables like we already do |
| GELU | Native |
| DYNAMIC_UPDATE_SLICE (KV write) | **Not needed on verify-path**: drafter is read-only during speculation |
| SELECT_V2, NOT_EQUAL, MAXIMUM | mask arithmetic, native |

**No blocker ops.** The `DYNAMIC_UPDATE_SLICE` is what writes into the drafter's own KV cache slots during generation, but in our pipeline the drafter reads **the target's kv13/kv14** that ChunkedEngine already maintains — we don't need a separate drafter KV cache at all.

Weights are INT8 quantized in the TFLite file (77% of tensors INT8). We can:
- dequantize to fp16 at extract time and re-palettize to INT4 to match our target's weight compression, or
- keep INT8 palettization (weight-only) which the ANE handles.

---

## 4. Integration plan for this repo

### 4.1 Extraction

- [done] Confirm E4B section layout via shadowlilac mirror.
- [running] Download `gemma-4-E2B-it.litertlm` (2.58 GB, currently 1.1 GB / 43%).
- [pending] Parse `.litertlm` container. Options:
  1. Install LiteRT-LM python tool and run `litertlm_peek --litertlm_file X --dump_files_dir out/` — cleanest path, requires building the generated flatbuffers schema.
  2. Direct byte scan for TFLite magic `TFL3` at aligned offsets — 50-line script, no deps.

  We'll use option 2 to avoid the bazel-ish dependency chain.

- [pending] Verify E2B drafter signature matches the E4B template with expected dim swaps:
  - `activations` should be `(1, 1, 2560)` for E2B (hidden=2560)
  - `mtp_post_proj` output likely `(1, 1, ~1280)` or same 2560 depending on Google's choice
  - KV cache shapes should reference head_dim=256 and 512 (Gemma 4's scaled/unscaled split)
  - Context window for drafter: 32003 tokens or similar (Google's `max_ctx`).

### 4.2 TFLite → PyTorch

Write a PyTorch `MtpDrafterModel` that mirrors §2. Weight loading:

- The TFLite graph already carries human-readable names like `layer_0.pre_q/attn.pre_q/.../q_einsum`. Build a rename map:
  - `mtp_pre_project.weight` ← `MtpDrafterModel.mtp_pre_project/mtp_pre_proj/...`
  - `layers.{i}.q_proj.weight`, `o_proj.weight`, `gate1/gate2/down.weight`, RMSNorm scales.
  - `head.weight` (262144-d logits head) ← tied to target embedding in LiteRT; for our path we can either (a) re-tie to our existing target embedding matrix, or (b) extract the explicit weight if LiteRT stores a tied reference.

Validate parity on a small prompt: PyTorch drafter + our target's L34 hidden → argmax equals Google's LiteRT output within fp16 rounding.

### 4.3 PyTorch → CoreML

New converter `conversion/build_mtp_drafter.py` (mirrors `build_eagle3.py`):

- Use ANERMSNorm, Conv2d(1,1) instead of Linear, pre-computed RoPE tables.
- I/O:
  - `target_hidden` (1,1,H) where H = target hidden dim (2560 for E2B)
  - `kv13_k`, `kv13_v` — sliding K/V (same shape as ChunkedEngine consumes; ANE-resident IOSurface)
  - `kv14_k`, `kv14_v` — full K/V
  - `position`, `mask` (built Swift-side, same builders we use for verify chunks)
- Outputs:
  - `logits` (1,1,V) fp16 — but **replace Linear(256→262144) with in-model `topk(8)`** (Phase 2 item 8 we need anyway)
  - `projected_activations` (1,1,H') — carry state

Expected `.mlpackage` size at INT4 palettization: **~6–10 MB** (drafter body is ~2 M params body + lm-head we can keep int8/tied). Drastically smaller than our 188 MB EAGLE-3.

### 4.4 Swift integration

- New `MtpDraftTarget` conformance to `SpeculativeDraftSource` (protocol already scaffolded on `feature/eagle3-speculative`).
- Feeds on `hidden_at_L34` tap from `ChunkedEngine` (already exists for EAGLE-3).
- Carry state: keep the `projected_activations` between speculative steps to enable K-token lookahead in a loop *without* re-running target.
- Verifier: the existing `verify_chunk{1..4}` T=3 arch already accepts K draft tokens — directly reusable.
- Commit path: same Blocker 2 applies (current commit re-runs T=1 decode per accepted token). MTP doesn't fix Blocker 2; that needs the K/V direct-write Swift patch that's already scoped at `EAGLE3_INTEGRATION_STATE.md §Step 2b`.

### 4.5 ~~Gotcha: the L34-divergence blocker~~ — RESOLVED (false alarm)

The previously reported "94% relative diff at L34" (in `EAGLE3_INTEGRATION_STATE.md`) was an **indexing artifact**. HF's `output_hidden_states` has 36 entries = `[embed, L0_out, ..., L33_out, post_norm]`. Index 35 is the **post-norm output**, not L34's raw output. Our comparison was comparing index 35 of HF (post-norm = 78.3) with index 35 of custom (L34 raw = 22.5).

Verified via forward hook: L34's actual output (captured before `self.norm`) = 22.4566 in HF(cache=True), matching our custom forward exactly. Per-layer parity debug script (`debug_l34_parity.py`) confirms all 35 layers match with `rel_diff < 1e-5` in fp32.

**This eliminates the L34 blocker entirely. MTP integration can proceed without any parity fix.**

This also reframes EAGLE-3 Blocker 1: the draft/target mismatch was NOT caused by our Gemma4Model being wrong. The likely cause is that EAGLE-3 training collected hidden states using HF's `use_cache=False` mode, which does NOT perform KV-sharing for L15+ (each layer computes its own K/V). On-device inference uses KV-sharing. This forward-mode mismatch produced the observed acceptance drop.

---

## 5. Scope & risks (UPDATED — L34 blocker eliminated)

| Step | Effort | Confidence |
|---|---|---|
| ~~4.1 Extract E2B section~~ | ~~0.5 day~~ | **DONE** (44.3 MB, `output/mtp_probe/section_10.tflite`) |
| 4.2 PyTorch reimplementation + weight load | 1–2 days | medium (rename map from TFLite tensor names is mechanical; tied-embedding detail is the only unknown) |
| 4.3 CoreML conversion with ANE ops | 1 day | high (same patterns as existing chunks) |
| 4.4 Swift wiring | 0.5 day | high (protocol exists) |
| ~~4.5 Fix Gemma4Model L34 parity~~ | ~~1–2 days~~ | **NOT NEEDED** (our forward matches HF exactly) |
| On-device bench | 0.5 day | — |

**Total: 3–4 days** to a measurable acceptance number on-device.

Primary risk: the drafter's `embedder.decode` head (Linear 256→262144) may be **tied weight-sharing** with the target's token embedding. If so, we must re-tie to our existing embedder in `EmbeddingLookup.swift` rather than extracting a full 0.25 GB head matrix. This is actually a size win but requires careful weight-name resolution.

---

## 6. Comparison to the EAGLE-3 status quo

| Axis | Our EAGLE-3 | Google's MTP drafter |
|---|---|---|
| Size | 188 MB (47 M params) | ~20–30 MB estimated (E2B) |
| Training status | Our custom train, Blocker 1 | Purpose-trained by Google, no training needed |
| Acceptance (expected) | 0% on-device (distribution mismatch) | unknown, **but Google trained against reference** so should clear 50% once L34 parity is fixed |
| KV dependence | Own decode chunks, needs `hidden_at_L{8,17,34}` taps | Reads target `kv13`/`kv14` directly — even tighter coupling |
| Verifier reuse | Built `verify_chunk{1..4}` T=3 | Same verifier works unchanged |

**If L34 parity lands first, MTP can probably reach the EAGLE-3 ceiling without the Colab retraining Blocker 1 demands.** That makes Path A plausibly faster-to-ship than finishing EAGLE-3 retrain.

---

## 7. Recommended next move

**L34 blocker: ELIMINATED.** E2B extraction: **DONE.** Both former prerequisites are cleared. This is a session-boundary-appropriate stopping point.

**Next session starts directly with PyTorch reimplementation (step 4.2).**

Remaining sequence:

| Step | Effort |
|---|---|
| 4.2 PyTorch reimplementation + weight load | 1–2 days |
| 4.3 CoreML conversion with ANE ops | 1 day |
| 4.4 Swift wiring | 0.5 day |
| On-device bench | 0.5 day |
| **Total** | **3–4 days** |

**Biggest unknown:** the `activations` input dimension is 3072 = 2× E2B `hidden_size` (1536). The drafter concatenates two 1536-dim vectors. The second half is either:
1. The **embedding of the draft token** (token embedding looked up for the predicted next token), or
2. The **`projected_activations` from the previous MTP step** (the drafter's own carry state fed back in).

This must be determined at the start of PyTorch reimplementation — it dictates the drafter's input wiring and whether the first MTP step needs a special "no previous activations" initialization path. Inspecting the TFLite graph's input naming or running a probe with known inputs should resolve it quickly.
