# Gemma 4 E2B / E4B — source-verified architecture

**Date:** 2026-04-22
**Sources:** our `conversion/models/gemma4.py`, HF `transformers/models/gemma4/`, `llama.cpp/src/models/gemma4-iswa.cpp`. Google's `gemma.cpp` and `gemma_pytorch` do NOT ship Gemma 4 (only Gemma 1-3).

## 0. TL;DR canonical spec

| | E2B | E4B |
|---|---|---|
| `hidden_size` | **1536** | 2560 |
| `num_hidden_layers` | **35** | 42 |
| `num_attention_heads` | 8 | - |
| `num_key_value_heads` | 1 | 2 |
| `head_dim` (sliding) | 256 | 256 |
| `global_head_dim` (full) | **512** | 512 |
| `intermediate_size` (FFN) | 6144 | - |
| `use_double_wide_mlp` for KV-shared | True → 12288 | True |
| `vocab_size` | 262144 | 262144 |
| `sliding_window` | 512 | 512 |
| `final_logit_softcapping` | **30.0** (tanh) | 30.0 |
| `num_kv_shared_layers` | 20 (last 20) | 20 (last 20?) |
| `hidden_size_per_layer_input` (PLE) | 256 | 256 |
| `hidden_activation` | `gelu_pytorch_tanh` | same |
| `tie_word_embeddings` | True | True |
| `attention_bias` | False | False |

**Warning on HF defaults:** `transformers/models/gemma4/configuration_gemma4.py:Gemma4TextConfig.__init__` defaults to `hidden_size=2304, intermediate_size=9216, num_hidden_layers=30` which **do NOT match E2B or E4B**. Those are "default" values for unnamed variants; E2B and E4B override via HF config.json. Use the actual HF model config.json at `google/gemma-4-E2B-it/config.json` or our `conversion/models/gemma4.py` values — both agree.

---

## 1. Layer-type pattern

`conversion/models/gemma4.py:70-78`:

```python
for i in range(num_hidden_layers):
    if (i + 1) % 5 == 0:
        layer_types.append("full_attention")
    else:
        layer_types.append("sliding_attention")
```

E2B (35 layers): full_attention at L4, L9, L14, L19, L24, L29, L34. Sliding elsewhere.

This is **5:1 interleave** (5 local + 1 global). HF transformers has `sliding_window_pattern=6` default which generates the same pattern.

## 2. KV-sharing layout (the Gemma 4 signature feature)

`conversion/models/gemma4.py:95-98`:
```python
def is_kv_shared(self, layer_idx):
    kv_start = self.num_hidden_layers - self.num_kv_shared_layers
    return layer_idx >= kv_start
```

For E2B: `kv_start = 35 - 20 = 15`. **L15-34 are KV-shared** (do not compute K/V). They reuse the most recent KV from the earlier KV-producing layers.

Per `llama.cpp/src/models/gemma4-iswa.cpp:79-110`:
```cpp
if (hparams.has_kv(il)) {
    Kcur = build_lora_mm(model.layers[il].wk, cur);
    Vcur = build_lora_mm(model.layers[il].wv, cur);
} else {
    cur = build_attn(inp_attn, ..., Qcur, nullptr, nullptr, ...);
}
```

The boundary is encoded in GGUF metadata on llama.cpp's side; in our path it's explicit in the Python model config.

**KV producers that downstream layers read:**
- For the sliding path: the last sliding-layer K/V in the nearest prior chunk, i.e. L13 (sliding) in our chunk 2.
- For the full-attention path: the last full-attention K/V, i.e. L14 in our chunk 2.

Our code names these as `kv13_k, kv13_v, kv14_k, kv14_v` as aliases (`conversion/models/gemma4_swa_chunks.py:112-121`); actual producers are `config`-driven (E2B uses L13/L14; E4B uses L22/L23).

**Why "Q-only" matters:** for 20 of 35 layers (57%), we skip K-proj and V-proj entirely. ~40% of attn GEMM ops are eliminated. This is a structural advantage baked into the model; cannot be retrofitted elsewhere.

## 3. Per-Layer Embeddings (PLE)

`conversion/models/gemma4.py:55-57`, `gemma4_swa_chunks.py:164-177`:

**Two components:**
1. `embed_tokens_per_layer` — `[vocab_size, num_layers × hidden_size_per_layer_input] = [262144, 35×256] = [262144, 8960]`. Large (9 GB fp16, ~2.3 GB INT8). Lives on flash/CPU.
2. `per_layer_model_projection` — projects main hidden state to `[num_layers × 256]` per-layer signal.

**Per-layer injection:**
```python
# For each decoder layer i:
per_layer_input = per_layer_tokens_slice + per_layer_proj_slice    # add PLE components
gate = per_layer_input_gate(hidden)                                 # sigmoid gate
scaled = gate * per_layer_input                                     # element-wise
residual += layer_scalar * per_layer_proj_back(scaled)              # add to residual
```

**Implication for our runtime:**
- PLE is **not** a simple embedding lookup. It's a learned skip-connection-like augmentation at every layer.
- Our `EmbeddingLookup` handles the token-identity component (gather from INT8 table) on CPU.
- The projection + gate + residual math happens inside each chunk in CoreML.

**Memory:**
- INT8 per-layer embed table: ~2.3 GB (fits on flash, streamed lazy).
- fp16 per-layer proj weights: small (1536 × 256 × 35 = ~21 MB).

**Implication for effective vs total params:**
- "E2B" = effective 2B params counting active VRAM footprint.
- Total parameter count inflated for marketing; memory ceiling is PLE's flash footprint, not VRAM.

## 4. Attention details

### 4.1 Dual head_dim

- Sliding layers: `head_dim = 256`
- Full-attention layers: `global_head_dim = 512` **(double!)**

Rationale: full-attention layers carry more context (whole sequence) and benefit from wider heads.

**Implication for kernel port:** FlashAttention kernel needs both `dk256_dv256` and `dk512_dv512` templates. `llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5628-6512` has `dk256_dv256` (E2B sliding match) and `dk512_dv512` is typically covered but verify.

### 4.2 QK-norm

Per-head RMSNorm on Q and K **after projection, before RoPE**.

`conversion/models/gemma4.py:` (inferred via ANERMSNorm import) and `llama.cpp/src/models/gemma4-iswa.cpp:70-71`:
```cpp
Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
```

HF transformers also applies V-norm without scale: `Gemma4RMSNorm(head_dim, with_scale=False)` for V.

**Why it matters for quantization:** Q/K pre-norm reduces dynamic range, making Q, K values more quantization-friendly. ~2-3% accuracy advantage for INT4/INT8 vs un-normed.

### 4.3 RoPE variants

- Sliding layers: `rope_theta = 10_000.0`, default rope_type.
- Full-attention layers: `rope_theta = 1_000_000.0`, `partial_rotary_factor = 0.25`.

Separate cos/sin tables per type:
- Sliding: head_dim=256 → 128 pairs.
- Full: head_dim=512 → 256 pairs, 25% rotated = 128 pairs rotated.

Our runtime builds both tables (`ChunkedEngine.swift:810-822`).

### 4.4 Causal + sliding mask

Two masks per chunk:
- Full causal: `[1, 1, ctx, K]` for full-attention layers.
- Sliding: `[1, 1, ctx, K]` with window 512 (only allows attention to last 512 positions).

Our stack builds them per position (`ChunkedEngine.swift:798-804`).

## 5. FFN

GeGLU:
```
gate = gate_proj(x)            # [hidden] → [intermediate]
up   = up_proj(x)              # [hidden] → [intermediate]
fused = GELU(gate) * up
out  = down_proj(fused)        # [intermediate] → [hidden]
```

Activation: `gelu_pytorch_tanh` (`0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`).

**Double-wide MLP for KV-shared layers:** L15-34 have `intermediate_size = 12288` (2× regular). Compensates compute-wise for skipping K/V projections. Our conversion handles this automatically via `use_double_wide_mlp=True`.

**Fusion status:**
- llama.cpp Metal fuses via `ggml_geglu_split` → single op (`src/llama-graph.cpp:1227-1231`).
- Our CoreML path: separate gate/up/down Convs. CoreML converter may or may not fuse at MIL level (inspection TBD).

## 6. Sandwich norms (4 RMSNorms per layer)

Per `conversion/models/gemma4.py` header comment:

```
pre_attn_norm → attn → post_attn_norm → residual_add
pre_ffn_norm  → ffn  → post_ffn_norm  → residual_add
```

All four are RMSNorm (eps=1e-6). **Post-attention norm is unique to Gemma 4** (not in Gemma 2/3). See `llama.cpp/src/models/gemma4-iswa.cpp:117-120`.

## 7. Logit softcap

`final_logit_softcapping = 30.0`. Applied as:
```python
logits = tanh(logits / 30.0) * 30.0
```

Squashes extreme logits, stabilizes sampling numerics on 262K vocab. Critical for FP16 inference.

## 8. Tokenizer

SentencePiece, vocab_size=262144. Special tokens: BOS=2, EOS=1. Secondary EOS=106 (per gemma.cpp for Gemma 3, likely same here). We load via swift-transformers `AutoTokenizer` (`CoreMLLLM.swift:180-182`).

## 9. Differences from Gemma 3

| Feature | Gemma 3 | Gemma 4 |
|---|---|---|
| Layer count | 26/34/48/62 (1B/4B/12B/27B) | 35 (E2B), 42 (E4B) |
| Hidden size | 1152/2560/3840/5376 | 1536 (E2B), 2560 (E4B) |
| KV-sharing Q-only | ❌ (uses GQA only) | ✅ L15-34 Q-only |
| Per-Layer Embeddings | ❌ | ✅ |
| Post-attention norm | ❌ | ✅ |
| Global head_dim (full attn) | = head_dim | 2× head_dim (512 vs 256) |
| FFN double-wide on KV-shared | ❌ | ✅ (12288 for L15-34 E2B) |
| Logit softcap | disabled (0.0) | 30.0 |
| Sliding window | 4096 | 512 |
| RoPE per layer-type | single θ | dual: sliding 10K, full 1M |

These are the "structural advantages" Gemma 4 provides. None can be retrofitted into Gemma 3.

## 10. Validation against external refs

- **HF transformers gemma4**: matches (after overriding the configuration_gemma4.py __init__ defaults with our config.json).
- **llama.cpp gemma4-iswa.cpp**: matches on architecture. Ships with 3 `TODO @ngxson` comments (see `METAL_PORT_REFERENCE.md`).
- **Google gemma.cpp / gemma_pytorch**: Gemma 4 NOT PRESENT. Only Gemma 1-3. Our implementation is ahead of Google's public C++/PyTorch reference.
- **mlx-swift-examples LLMRegistry**: has `gemma3n_E2B_it_lm_4bit` (note: **3n not 4**). Confirms a Gemma 3n E2B exists separately; our Gemma 4 E2B is a different model. Don't confuse.

## 11. Open architectural questions

1. **Which earlier layer's KV does each Q-only layer actually read?** Our code aliases `kv13_k, kv13_v, kv14_k, kv14_v` but the mapping from Q-only layer → producer layer needs validation against HF's `num_kv_shared_layers` semantics.

2. **RoPE `partial_rotary_factor=0.25` for full-attention layers** — only 128 of 512 dims are RoPE'd; remaining 384 are plain. Does our code handle this correctly?

3. **Attention K-V aliasing (`attention_k_eq_v`)** — **RESOLVED 2026-04-24: FALSE for E2B.** HF config.json sets `attention_k_eq_v: false`. Per `modular_gemma4.py:911` (`use_alternative_attention = attention_k_eq_v and not is_sliding`) and `:943-947` (`v_proj = None` when K=V mode on), the K=V path requires the flag to be true AND drops `v_proj` entirely, reusing K as V. E2B has an independent `v_proj` with learned weights, so K ≠ V and no export-level alias is possible. **Implication:** the "global-layer K=V alias" optimization proposed in `SESSION_2026_04_23.md` #2 is not applicable to E2B. Applies to E4B only if its config.json also flips this flag — not checked yet.

4. **Embedding-head tie:** `tie_word_embeddings=True` but we run external INT8 embedding lookup. The lm_head in chunk4 still needs a weight. Is it a copy or aliased?

5. **MoE flag:** `enable_moe_block: False` default. E2B doesn't use MoE, but E4B might? HF `Gemma4TextConfig` has full MoE infrastructure. Check E4B's config.json.

Each of these is a 1-2 hour investigation against HF + our model's `config.json`.

## 12. References

- `conversion/models/gemma4.py:1-100` — our canonical config class
- `conversion/models/gemma4_swa_chunks.py:100-200` — chunk boundary, KV output aliasing
- `conversion/config.py:37-44` — E2B / E4B registry
- `llama.cpp/src/models/gemma4-iswa.cpp:1-300` — llama.cpp reference
- HF: `transformers/src/transformers/models/gemma4/configuration_gemma4.py` (class `Gemma4TextConfig`)
- HF: `transformers/src/transformers/models/gemma4/modular_gemma4.py` (lines 904-1025: attention; 861-868: MLP; 1238-1343: PLE)
