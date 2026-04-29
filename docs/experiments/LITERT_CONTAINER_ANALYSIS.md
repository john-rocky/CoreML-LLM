# LiteRT Container Analysis — gemma-4-E2B-it.litertlm

**Date:** 2026-04-14
**Source:** `litert-community/gemma-4-E2B-it-litert-lm` (2.58 GB)

---

## Container Layout

10 TFLite sections extracted from the `.litertlm` FlatBuffer container:

| Section | Size (MB) | Type | Description |
|---------|-----------|------|-------------|
| 0 | 103.8 | Embedder | `token_ids (1,1) → embeddings (1,1,1536)` — token embedding lookup |
| 1 | 1284.5 | Per-layer embedder | `token_ids (1,1) → embeddings (1,1,35,256)` — per-layer embedding (PLE) |
| 2 | 94.1 | Vision encoder | `src_inputs (1,1,816,128) → features (1,204,1536)` — SigLIP image encoder |
| 3 | 9.5 | Vision post-proc | `features (1,204,1536) → output_0 (1,204,1536)` — post-encoder projection |
| 4 | 0.0 | Empty | Placeholder/separator |
| 5 | 219.1 | Vision preprocessor | `images (1,2520,768) → features (1,280,768)` — image feature extraction |
| 6 | 4.7 | MM projector | `soft_tokens (1,280,768) → mm_embedding (1,280,1536)` — multimodal projection |
| 7 | 0.0 | Empty | Placeholder/separator |
| 8 | 818.3 | **Main LLM** | Full 35-layer transformer with 4 signatures (decode/prefill/verify) |
| 9 | 44.3 | **MTP drafter** | 4-layer mini-transformer for speculative decoding |

## Main LLM (Section 8) — Key Details

**4 signatures with EnumeratedShapes:**
- `decode`: T=1, single token decode
- `prefill_1024`: T=1024, large batch prefill
- `prefill_128`: T=128, medium batch prefill
- `verify_4`: T=4, verify 4 draft tokens (MTP K=3 + 1 extra)

**KV Caches:**
- 15 layers with own K/V: `kv_cache_k_0` through `kv_cache_k_14`
- Sliding layers (head_dim=256): L0-3, L5-8, L10-13
- Full attention layers (head_dim=512): L4, L9, L14
- Context window: 32003 tokens
- KV dtype: INT8 quantized (reported as bool in type enum)
- K layout: `(1, 1, 32003, head_dim)` — seq before head_dim
- V layout: `(1, 1, head_dim, 32003)` — transposed (head_dim before seq)

**Key insight — Google chunks the LLM as a SINGLE section** (818 MB), not split
into multiple chunks like our CoreML pipeline. They use EnumeratedShapes for
variable sequence lengths instead. Their section 8 has 976 subgraphs and
2703 tensors — the entire model in one TFLite.

**Outputs include `activations`** — the last hidden state:
- Decode: `activations (1,1,1536)` — fed to MTP drafter
- Verify: `activations (1,4,1536)` — hidden states at all verify positions

## MTP Drafter (Section 9) — Detailed Architecture

**I/O Signature:**
```
Inputs:
  activations        (1, 1, 3072)     fp32   ← concat(embed, projected_activations)
  input_pos          (1,)             int32
  kv_cache_k_13      (1, 1, 32003, 256) int8  ← target's sliding KV
  kv_cache_k_14      (1, 1, 32003, 512) int8  ← target's full-attn KV
  kv_cache_v_13      (1, 1, 256, 32003) int8
  kv_cache_v_14      (1, 1, 512, 32003) int8
  mask               (1, 1, 1, 32003)  bool
  param_tensor       (1, 1, 1, 7)     int32   ← RoPE position + flags

Outputs:
  logits             (1, 1, 262144)   fp32
  projected_activations (1, 1, 1536)  fp32   ← carry state for next step
```

**Activations input semantics** (confirmed from LiteRT C++ runtime source):
- `activations[0:1536]` = raw embedding of input token (NO sqrt(H) scaling)
- `activations[1536:3072]` = projected_activations from previous step
  (or base model's last hidden state for step 0)

**Layer stack (267 tensors, 194 operators, 30 subgraphs):**
```
mtp_pre_proj    Linear(3072 → 256)    INT8

layer_0  (SWA, reads kv_cache_13):
  RMSNorm(256)
  q_proj    Linear(256 → 1024)        INT8    [4 heads × 256 head_dim]
  [attention: Q-only, K/V from kv_cache_13]
  o_proj    Linear(1024 → 256)        INT8
  RMSNorm(256)
  gate1     Linear(256 → 2048)        INT4
  gate2     Linear(256 → 2048)        INT4
  GeGLU: gelu(gate1) * gate2
  down      Linear(2048 → 256)        INT4

layer_1  (SWA, kv_cache_13) — same dims as layer_0
layer_2  (SWA, kv_cache_13) — same dims as layer_0

layer_3  (Full attention, reads kv_cache_14):
  RMSNorm(256)
  q_proj    Linear(256 → 2048)        INT8    [4 heads × 512 head_dim]
  [attention: Q-only, K/V from kv_cache_14]
  o_proj    Linear(2048 → 256)        INT8
  RMSNorm(256)
  gate1     Linear(256 → 2048)        INT4
  gate2     Linear(256 → 2048)        INT4
  GeGLU: gelu(gate1) * gate2
  down      Linear(2048 → 256)        INT4

RMSNorm(256)                          ← final norm
embedder.decode  Linear(256 → 262144) INT4    [LM head, tied to target embedding]
mtp_post_proj    Linear(256 → 1536)   INT8    [→ projected_activations]
```

**Weight quantization breakdown:**
- INT8: mtp_pre_proj, q_proj ×4, o_proj ×4, mtp_post_proj (10 tensors)
- INT4: gate1 ×4, gate2 ×4, down ×4, embedder.decode (13 tensors)
- FP32: RMSNorm scales (21 tensors), RoPE tables, misc constants

**RoPE implementation:**
- SWA layers (0-2): precomputed `[1, 1, 128]` table (head_dim/2 = 128)
- Full layer (3): precomputed `[1, 1, 256]` table (head_dim/2 = 256)
- RoPE is applied via sin/cos tables baked into the graph, same theta as target

## Implications for CoreML Conversion

1. **Single-model vs chunked**: Google uses 1 TFLite for the full LLM; we split
   into 4 chunks for ANE dispatch. This is fine — our chunking is an ANE-specific
   optimization that Google doesn't need for XNNPACK CPU.

2. **KV layout matches**: K=(1,1,ctx,hd), V=(1,1,hd,ctx) — the V-transposed
   layout matches what we already use in ChunkedEngine.

3. **Verify signature T=4**: Google verifies 4 tokens at once (3 draft + 1).
   Our Q=K verifier needs to support at least K=3.

4. **Mixed INT8/INT4 quantization**: Attention in INT8, MLP in INT4. We can
   replicate this in CoreML via per-op palettization config.

5. **param_tensor**: Contains RoPE position info in a packed 7-element int32
   tensor. We'll replace this with our existing precomputed RoPE tables.

6. **No drafter KV cache**: The drafter reads target's kv13/kv14 directly.
   No separate cache needed — matches our architecture.
