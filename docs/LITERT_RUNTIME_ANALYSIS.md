# LiteRT-LM Runtime & Model Analysis

**Date:** 2026-04-14
**Sources:** LiteRT-LM runtime (github.com/google-ai-edge/LiteRT-LM), ai-edge-torch export pipeline (github.com/google-ai-edge/ai-edge-torch), gemma-4-E2B-it.litertlm model file (2.58 GB)
**Branch:** research/litert-runtime-analysis

---

## Executive Summary

Five findings that directly change our roadmap:

1. **Blocker 2 is solved by design.** Google's verify pass runs ALL K+1 tokens through
   the full model in a single forward pass. The model's internal `dynamic_update_slice`
   writes KV for all positions (accepted AND rejected). Rejected tokens' KV entries
   are harmlessly masked out by the attention mask. **No separate KV commit step is
   needed.** Our existing verify_chunk T=3 architecture already does this — we just
   need to advance the position counter by `num_accepted + 1` after verification.
   (See B1.3)

2. **K=3 is fixed, not dynamic.** Google generates exactly 3 draft tokens per MTP cycle
   (derived from the verify signature's input_pos shape = 4 = K+1). No adaptive
   speculation, no early-exit from the drafting loop. Simple greedy argmax comparison
   for acceptance. This is much simpler than DISCO or other dynamic-K schemes.
   (See B1.1, B1.2)

3. **KV cache architecture: slices + update model on NPU, in-place on CPU/GPU.**
   The NPU executor externalizes KV cache updates into a separate TFLite model
   (cache_update signatures) because NPU can't do dynamic_update_slice efficiently.
   For CPU/GPU, the model does in-place updates via dynamic_update_slice — identical
   to our CoreML chunks. **Our ANE path needs no restructuring.** (See B1.3, B2)

4. **No MTP in the export pipeline.** The ai-edge-torch repo has zero references to
   MTP, drafter, or speculative decoding. The MTP drafter is trained/exported
   separately (likely Google-internal). The export pipeline for the base model uses
   standard HuggingFace loading with minimal patches (RMSNorm → HLFB composite only).
   (See C1)

5. **UINT2 quantization is real — Google uses 2-bit weights for 60% of FFN layers.**
   Deep graph inspection reveals a split architecture: L0-14 use INT4 everywhere,
   but L15-34 use **UINT2 weights with 2× wider FFN** (12288 vs 6144 intermediate).
   Same byte budget, more parameters. The LM head is also UINT2. Activations are
   INT8 per-tensor throughout (W4A8 for L0-14, W2A8 for L15-34). This is far more
   aggressive than our uniform INT4 palettization. (See A2)

6. **20 layers share a single KV cache.** L15-34 have Q-only attention — they only
   project queries, reading K/V from kv_cache_13 (SWA) and kv_cache_14 (full attn).
   This is how Gemma 4 achieves 35 layers with only 15 KV cache entries. (See A3)

---

## Part A — Model Graph

### A1. Section Inventory & Chunk Boundaries

(From LITERT_CONTAINER_ANALYSIS.md, confirmed)

| Section | Size (MB) | Type | Description |
|---------|-----------|------|-------------|
| 0 | 103.8 | Embedder | `token_ids (1,1) → embeddings (1,1,1536)` |
| 1 | 1284.5 | Per-layer embedder | `token_ids (1,1) → embeddings (1,1,35,256)` PLE |
| 2 | 94.1 | Vision encoder | SigLIP image encoder |
| 3 | 9.5 | Vision post-proc | Post-encoder projection |
| 4 | 0.0 | Empty | Placeholder |
| 5 | 219.1 | Vision preprocessor | Image feature extraction |
| 6 | 4.7 | MM projector | Multimodal projection |
| 7 | 0.0 | Empty | Placeholder |
| 8 | 818.3 | **Main LLM** | Full 35-layer transformer, 4 signatures |
| 9 | 44.3 | **MTP drafter** | 4-layer mini-transformer |

**Critical difference from our approach:** Google ships the LLM as a **single 818 MB
TFLite** (section 8) with 4 signatures (decode, prefill_128, prefill_1024, verify_4).
We split into 4 chunks for ANE dispatch. This is an ANE-specific optimization — Google
doesn't need it for XNNPACK CPU.

**Chunk boundaries (ours vs Google's):**

| Our chunks | Layers | Google |
|---|---|---|
| chunk1 (L0-7, SWA) | 8 layers | All in section 8 |
| chunk2 (L8-14, full-attn) | 7 layers | All in section 8 |
| chunk3 (L15-24, SWA) | 10 layers | All in section 8 |
| chunk4 (L25-34, SWA + LM head) | 10 layers | All in section 8 |

**LLM Signatures (section 8):**
- `decode`: T=1, single token decode
- `prefill_1024`: T=1024, large batch prefill
- `prefill_128`: T=128, medium batch prefill
- `verify_4`: T=4, verify 3 draft tokens + current token

### A2. Quantization Strategy

**MAJOR FINDING: Split architecture with UINT2 for 60% of FFN layers.**

Deep TFLite graph inspection of section 8 reveals two distinct layer groups with
dramatically different quantization:

**Layers 0-14 (15 layers with own KV cache — "KV layers"):**

| Component | Bit-width | Shape | Notes |
|---|---|---|---|
| q_proj | INT4 per-channel | [2048,1536] (SWA) / [4096,1536] (full) | 8 heads |
| k_proj | INT4 per-channel | [256,1536] (SWA) / [512,1536] (full) | 1 KV head |
| v_proj | INT4 per-channel | [256,1536] / [512,1536] | 1 KV head |
| o_proj | INT4 per-channel | [1536,2048] / [1536,4096] | |
| gate1/gate2 | INT4 per-channel | [6144,1536] | intermediate=6144 |
| down_proj | INT4 per-channel | [1536,6144] | |

**Layers 15-34 (20 layers sharing KV — "Q-only layers"):**

| Component | Bit-width | Shape | Notes |
|---|---|---|---|
| q_proj | INT4 per-channel | [2048,1536] | Q-only, no K/V projections |
| k_proj | **NONE** | — | Reads from kv_cache_13/14 |
| v_proj | **NONE** | — | Reads from kv_cache_13/14 |
| o_proj | INT4 per-channel | [1536,2048] | |
| gate1/gate2 | **UINT2** per-channel | [**12288**,1536] | **2× wider FFN** |
| down_proj | **UINT2** per-channel | [1536,**12288**] | **2× wider FFN** |

**LM head:** UINT2 per-channel, [262144,1536], 96.0 MB. Logit softcapping at 30.0.

**PLE (Per-Layer Embedding) weights:** INT8 gate [256,1536] + INT8 projection [1536,256]

**Width-for-bits trade:** L15-34 use UINT2 (2-bit) with 2× intermediate size (12288 vs
6144). The byte budget is identical to INT4 with 6144, but the model gets more
parameters through wider networks. This is "overparameterize then compress."

**Activation quantization:** Per-tensor INT8 throughout. Every FC input goes through
FP32→INT8 quantization, every FC output through INT8→FP32 dequantization. This
makes the pipeline **W4A8 for L0-14** and **W2A8 for L15-34**.

**All quantization is symmetric** (zero_point=0) and **per-channel** (one scale per
output channel).

**Comparison to ours:** We use uniform INT4 palettization across all weights. Google's
approach is far more aggressive — UINT2 for the majority of FFN compute. CoreML's
`coremltools` does not support UINT2, but we could approximate with INT4 at 2× width
or explore custom quantization tables. The key question is whether ANE supports UINT2
decompression efficiently.

**Weight size breakdown:**

| Category | Size (MB) |
|---|---|
| Attention (q/k/v/o) all layers | 132.8 |
| FFN (gate1/gate2/down) all layers | 472.5 |
| PLE weights | 26.2 |
| LM head | 96.0 |
| Norms/constants | 1.3 |
| **Total** | **~729** |

### A3. Attention Implementation

**GQA: 8:1 ratio, no explicit repeat.**
- SWA layers: Q=[1,1,8,256] (8 heads, head_dim=256), K/V=[1,1,1,256] (1 KV head)
- Full layers: Q=[1,1,8,512] (8 heads, head_dim=512), K/V=[1,1,1,512] (1 KV head)
- **No explicit repeat_interleave** — the matmul handles GQA broadcasting implicitly
  via composite subgraph calls (`STABLEHLO_SHIFT_RIGHT_LOGICAL` dispatches to
  `runtime_bmm.impl_*` subgraphs)

**Q·K attention matmul — two distinct implementations:**

*SWA Q·K (subgraph `runtime_bmm.impl_0`):*
- Extracts a **1535-element sliding window slice** from the INT8 KV cache
- Dequantizes the slice INT8→FP32
- Performs batch matmul
- Writes result into padded [1,1,8192,32003] via `DYNAMIC_UPDATE_SLICE`
- 13 ops total: compute window bounds, slice, dequant, matmul, scatter

*Full attention Q·K (subgraph `runtime_bmm.impl_1042_0`):*
- Dequantizes the **entire** K cache INT8→FP32
- Full matmul against all context positions
- Only 2 ops: DEQUANTIZE + matmul

**Softmax:** Standard SOFTMAX op (code 25). **No exp2 optimization.** No fused SDPA.

**RoPE:** Precomputed frequency tables baked as FP32 constants:
- SWA: `[1,1,128]` table (head_dim/2=128)
- Full: `[1,1,256]` table (head_dim/2=256)
- At runtime: `input_pos * freq_table` → sin/cos → apply to Q halves
- NPU path: separate auxiliary model with `decode_rope`/`prefill_rope_*` signatures

**KV cache layout (matches ours):**
- K: `(1, 1, 32003, head_dim)` — seq before head_dim
- V: `(1, 1, head_dim, 32003)` — transposed, head_dim before seq
- INT8 quantized, per-tensor scale, ZP=0
- V-transposed layout avoids runtime transposes for attn·V

**SWA vs full attention differentiation:**
- SWA layers (L0-3, L5-8, L10-13): head_dim=256, own kv_cache entries
- Full layers (L4, L9, L14): head_dim=512, own kv_cache entries
- SWA mask: dynamically computed range check `[pos - window, pos)`
- Full mask: externally provided decode_mask broadcast to all 8 heads

**KV sharing (L15-34):** These 20 layers have **Q-only attention** — they project
queries but read K/V directly from `kv_cache_k_13` (SWA) and `kv_cache_k_14`
(full). No K/V projections exist in these layers. This is how Gemma 4 achieves
35 layers of context with only 15 KV cache entries.

**Sparse attention:** No sparse attention patterns exist for full-attention layers.
The full-attention Q·K subgraph (`runtime_bmm.impl_1042_0`) dequantizes the
**entire** K cache and performs a dense matmul against all 32003 context positions.
No block-sparse, top-k retrieval, or TriForce-style eviction is applied.

**Op code remapping (Google-internal):** Google repurposes TFLite op codes:
- `STABLEHLO_SHIFT_RIGHT_LOGICAL` → composite subgraph call dispatcher
- `MEAN` → sin, `TANH` → cos (for RoPE)
- `AVERAGE_POOL_2D` → concatenation
- `SQUARED_DIFFERENCE` → slice
These are dispatched to custom subgraph implementations at runtime.

### A4. FFN / MLP Implementation

**GeGLU activation lowering (L0-14, INT4, intermediate=6144):**
```
FC:         INT8[1,1,1536] * INT4[6144,1536] → INT8[1,1,6144]   # gate1
DEQUANT:    INT8[1,1,6144] → FP32[1,1,6144]
FC:         INT8[1,1,1536] * INT4[6144,1536] → INT8[1,1,6144]   # gate2
DEQUANT:    INT8[1,1,6144] → FP32[1,1,6144]
GELU:       FP32[1,1,6144] → FP32[1,1,6144]                     # gelu(gate1)
MUL:        FP32 * FP32 → FP32                                   # gelu(gate1)*gate2
QUANT:      FP32 → INT8 (per-tensor)                             # re-quantize
FC:         INT8[1,1,6144] * INT4[1536,6144] → INT8[1,1,1536]   # down_proj
DEQUANT:    INT8[1,1,1536] → FP32[1,1,1536]
```

**GeGLU activation lowering (L15-34, UINT2, intermediate=12288):**
```
FC:         INT8[1,1,1536] * UINT2[12288,1536] → INT8[1,1,12288]  # gate1
FC:         INT8[1,1,1536] * UINT2[12288,1536] → INT8[1,1,12288]  # gate2
GELU + MUL
FC:         INT8[1,1,12288] * UINT2[1536,12288] → INT8[1,1,1536]  # down
```

Key observations:
- **No fused GeGLU op** — GELU and MUL are separate elementwise operations
- TFLite has a **native GELU op** (code 150) — not decomposed to tanh approximation
- **Activation quantization at every FC boundary:** FP32→INT8 (per-tensor, scale, ZP=0)
  before each FC, INT8→FP32 after. This is full W4A8/W2A8 dynamic quantization.
- **No tiling tricks** (no reshape to (B,C,8,8) pattern)

**PLE (Per-Layer Embedding) sub-block — present in EVERY layer:**
```
FC:   INT8[1,1,1536] * INT8[256,1536] → INT8[1,1,256]     # PLE gate
GELU: gelu(gate_output)
MUL:  gelu_output * per_layer_embedding[layer_i]            # element-wise gate
FC:   INT8[1,1,256] * INT8[1536,256] → INT8[1,1,1536]     # PLE projection
ADD:  residual + PLE_output
```
This is the Gemma 4 Per-Layer Embedding mechanism — each layer gets a gated
injection of its own learned embedding. PLE runs AFTER attention, BEFORE FFN.

**RMSNorm fusion:** `odml.rms_norm` StableHLO composite wraps pow+mean+rsqrt+mul.

### A5. LM Head

- **Location:** Inside section 8 (main LLM), last 4 ops of decode subgraph.
- **Quantization:** **UINT2** per-channel, [262144, 1536], 96.0 MB.
  262,144 quantization scales (one per vocab entry). Symmetric (ZP=0).
- **Output:** Full logit vector `(1, T, 262144)` as FP32.
  **No in-model argmax/topk** — sampling done externally by Sampler.
- **Logit softcapping:** `MUL(1/30)` → FC → `MUL(30)` — softcapping at 30.0.
  The LM head is the **only FC that takes FP32 activations directly** (no INT8
  quantization of the input), preserving precision for the final projection.
- **Weight tying: CONFIRMED.** Byte-identical data between embedding table
  (section 0) and LM head (section 8). Same UINT2 dtype, same per-channel
  scales. Stored as separate copies (no deduplication in the container).
- **Verify output:** `(1, 4, 262144)` logits + `(1, 4, 1536)` activations
  for all 4 positions. The activations output is the pre-norm hidden state
  (same tensor that feeds the MTP drafter).

---

## Part B — Runtime

### B1. Speculative Loop & KV Commit (CRITICAL)

**Overview of the MTP speculative cycle:**

```
Step 1: [First cycle only] Run normal decode (T=1)
        → get token + activations from last layer
        
Step 2: Run drafter 3× sequentially
        → Each step: embed(token) ++ projected_activations → drafter model
        → Get 3 draft tokens [d1, d2, d3]

Step 3: Run verifier (T=4) with [good_token, d1, d2, d3]
        → Full model forward on all 4 tokens
        → Get logits[4] + activations[4] + KV written for all 4 positions

Step 4: Greedy rejection sampling
        → Compare verifier argmax[i] vs draft[i] for i=0..2
        → Accept prefix, bonus = first mismatch or verifier[3]

Step 5: KV commit (NPU only): run cache_update model
        → CPU/GPU: no-op (verify already wrote KV via dynamic_update_slice)

Step 6: Save activation at last_accepted position → carry state for next cycle
        → Next cycle can skip Step 1 entirely
```

**File references:**
- GPU MTP drafter: `runtime/executor/llm_litert_mtp_drafter.cc`
- NPU MTP loop: `runtime/executor/llm_litert_npu_compiled_model_executor.cc:1556-1691`
- Rejection sampling: `llm_litert_npu_compiled_model_executor.cc:2422-2462`
- KV commit: `llm_litert_npu_compiled_model_executor.cc:2464-2491`

#### B1.1 Draft Token Count K

**K=3 is fixed**, derived from the verify signature's `input_pos` tensor shape:
```cpp
// llm_litert_mtp_drafter.cc:240
num_draft_steps = input_pos_dims[0] - 1;  // verify_4 → input_pos=[4] → K=3
```

No dynamic K. No early-exit from the drafting loop. No fallback when acceptance
rate drops. The loop always runs exactly 3 draft steps.

**The 56.5 tok/s figure at ~60% acceptance rate means:**
- Base decode: ~20 tok/s (estimated)
- With MTP K=3: expected speedup = 1/(1 + 1/3) * (1 + 0.6*3) ≈ 2.1× to 2.8×
- 20 × 2.8 ≈ 56 tok/s — consistent with observed performance

#### B1.2 Verify → Accept → Reject Flow

**GPU path** (`llm_litert_mtp_drafter.cc:486-516`):
```cpp
for (int i = 0; i < num_draft_steps_; ++i) {
    if (verifier_id_vector[i] != drafted_tokens[i]) {
        bonus_token = verifier_id_vector[i];
        break;
    }
    num_correct_tokens++;
}
if (bonus_token == -1) {
    bonus_token = verifier_id_vector[num_draft_steps_];
}
```

Simple greedy argmax comparison — NOT probabilistic rejection sampling. This is
deterministic and produces identical results to standard autoregressive decoding
(for greedy/argmax sampling).

**NPU path** (`llm_litert_npu_compiled_model_executor.cc:2438-2461`):
Identical logic using `GetLogitsAtBatchIndex` to sample from batch logits.

#### B1.3 KV Cache Update After Acceptance — BLOCKER 2 RESOLUTION

**THIS IS THE KEY FINDING.**

Two different mechanisms depending on backend:

**GPU/CPU path (our equivalent):**
The verify signature runs ALL K+1 tokens through the full model. The model contains
`dynamic_update_slice` ops that write KV entries at positions [P, P+1, P+2, P+3].
**ALL positions are written, including rejected ones.** After partial acceptance:
- Position counter advances by `num_accepted + 1` (for bonus token)
- Stale KV entries at positions beyond the counter are masked out by the attention mask
- No cleanup, no rollback, no separate commit step

**This means our CoreML verify chunks already solve Blocker 2!** The verify_chunk
models with T=3 run the full model on K+1 tokens, which writes KV via their internal
`dynamic_update_slice`. After acceptance, we simply advance the position counter.
The stale entries are invisible to future attention computations because the mask
only covers positions ≤ current_step.

**NPU path (not relevant to us but documented for completeness):**
The NPU can't efficiently do `dynamic_update_slice`, so the LLM outputs KV "slices"
(small tensors for current tokens only). A separate `cache_update` auxiliary model
copies these slices into the full KV cache. The `CommitVerifiedKVCache` function
(`llm_litert_npu_compiled_model_executor.cc:2464-2491`) runs this auxiliary model
with position indices `[start_step, start_step+1, ..., start_step+T-1]`.

**Implication:** Our CoreML/ANE pipeline behaves like Google's CPU/GPU path — the
model does in-place KV updates. No restructuring needed. Blocker 2 is dissolved.

#### B1.4 Carry State (Activations Bootstrap)

The verifier outputs `activations` at all K+1 positions. After acceptance, the
activation at the last accepted position is extracted and used as the carry state
for the next MTP cycle:

```cpp
// llm_litert_npu_compiled_model_executor.cc:1646-1661
size_t hidden_size_in_bytes = full_activations.size() / dratfer_seq_len;
memcpy(last_verify_activations_.data(),
       full_activations.data() + rs_result.num_accepted * hidden_size_in_bytes,
       hidden_size_in_bytes);
has_valid_verify_activations_ = true;
```

On the next cycle, this skips the initial decode step entirely — the drafter
starts from the saved activation. Our verify chunks already output
`activations` (the last hidden state) which can be used identically.

**For the MTP drafter's input:** `activations = concat(embed(token), projected_activations)`
- First half (1536 dims): raw embedding of the token (NO sqrt(H) scaling)
- Second half (1536 dims): either the target's last hidden state (step 0) or
  the drafter's own `projected_activations` output (steps 1+)

### B2. KV Cache Management

**GPU/CPU: Double-buffer with pointer swap.**
Two `flat_hash_map<string_view, TensorBuffer>` maps (`kv_cache_buffers_1_`,
`kv_cache_buffers_2_`) with `input_kv_cache_buffers_` and `output_kv_cache_buffers_`
pointer aliases. After each model run:
```cpp
// llm_litert_compiled_model_executor.cc:766
std::swap(input_kv_cache_buffers_, output_kv_cache_buffers_);
```

**GPU single-buffer optimization:** When `gpu_optimized_single_buffer_cache_` is true,
the same buffer serves as both input and output. A `param_tensor` with
`[start_index, end_index, end_index]` tells the kernel where to write. No swap needed.

**NPU: Full cache + slices + cache_update model.**
LLM outputs small KV slice tensors. Separate `cache_update` signatures
(`decode_cache_update`, `prefill_cache_update_128`, `verify_cache_update`) copy
slices into the full cache at specified positions.

**KV sharing with drafter:** The MTP drafter reads the target model's `kv_cache_k_13`
and `kv_cache_k_14` directly via `TensorBuffer::Duplicate()` (shared memory handles).
No separate drafter KV cache.

**Sliding window eviction:** StreamingLLM-style token deletion via
`DeleteTokensFromKvCache`:
- Drops `num_tokens_to_drop` tokens while retaining `init_tokens_to_retain` initial tokens
- K axis 2, V axis 3 (consistent with transposed-V layout)
- No ring buffer — simple shift/drop operation

**Full-attention KV eviction:** No eviction or compression strategy exists for
full-attention layers. The cache shape is hard-capped at 32003 positions. The
`DeleteTokensFromKvCache` function applies only to SWA cache layers. Full-attention
layers (L4, L9, L14) grow monotonically until the 32003 cap is reached.
At max context, the runtime does not evict, compress, or page full-attention KV —
inference simply operates with the full cache.

**INT8 KV cache:** All KV caches are INT8 quantized. The dequantization is handled
at the model level (DEQUANTIZE op in the TFLite graph).

**Memory reuse between prefill and decode:** Prefill and decode share the same
physical KV cache buffer allocations — both signatures reference the same
`input_kv_cache_buffers` / `output_kv_cache_buffers` maps. No buffer reallocation
or memory reclamation occurs at the prefill→decode transition. The double-buffer
swap is a pointer swap, not a copy. For the dynamic executor, KV cache grows
incrementally by `kv_increment_size` (default 16 tokens) via `ResizeTensorBuffer`.

### B3. Prefill vs Decode

**Separate signatures, not parameterized:**
- `prefill_128` / `prefill_1024`: Q=128/1024 tokens
- `decode`: Q=1 token
- `verify_4`: Q=4 tokens (K+1)

**Chunked prefill:** Yes. The static executor splits prefill into chunks matching
available signatures using `GetOptimizedPrefillWorkGroups`. Multiple chunks are
run sequentially. The dynamic executor supports `prefill_chunk_size` parameter.

**Backend consistency:** Same compiled model for all signatures (prefill, decode,
verify). No backend switching between phases.

**NPU decode pipeline (per token):**
1. Embedder (token → embedding)
2. Per-Layer Embedder (token → per-layer embedding) [Gemma 4 specific]
3. RoPE computation (auxiliary model)
4. Mask computation (auxiliary model)
5. LLM forward (main model, decode signature)
6. Cache update (auxiliary model)
7. Sampling

Each step is a separate TFLite model invocation. This decomposition is NPU-specific;
the CPU/GPU path runs embedder + LLM as two models with in-model RoPE/mask.

### B4. Model Loading & Session Management

**Cold start optimization:**
- Compiled model caching: GPU backends cache compiled shaders/kernels
  (`cache_dir` + `cache_compiled_shaders_only`)
- Warmup inference: explicit `WarmupInference` and `WarmupDrafterInference` methods
  run dummy inference on all signatures before actual use
- Weight upload: GPU has `convert_weights_on_gpu` + `wait_for_weight_uploads` options

**Section loading:** Lazy via `ModelResources::GetTFLiteModel(ModelType)`. Models are
memory-mapped (physical memory allocated on demand). Model types:
- `kTfLitePrefillDecode` (main LLM)
- `kTfLiteEmbedder` (token embedding)
- `kTfLitePerLayerEmbedder` (PLE)
- `kTfLiteAux` (RoPE, mask, cache update)
- `kTfLiteMtpDrafter` (MTP drafter)
- `kTfLiteMtpAux` (drafter's RoPE/mask)

**Session state:**
- One session per executor (enforced via `occupied_executors_` set)
- Checkpoint save/restore for KV cache rewind
- Clone for branching conversations (deep copy KV cache)

### B5. Thermal / Performance Management

**No runtime thermal management.** Zero throttling, adaptive behavior, token rate
monitoring, or backend switching under load in the open-source runtime code.

**GPU priority control:** Two flags for UI smoothness (not thermal):
- `gpu_context_low_priority`: Low-priority GPU context
- `hint_kernel_batch_size`: Periodic GPU command flush (e.g., every 4 ops)

**MTP statistics tracked:**
```cpp
struct LatencyStats {
    int mtp_num_draft_tokens = 0;
    int mtp_num_accepted_tokens = 0;
};
```

On executor destruction, acceptance rate is logged:
```cpp
ABSL_LOG(INFO) << "Success rate: " << (double)num_verified / num_drafted;
```

Purely observational — no runtime adaptation based on these metrics.

---

## Part C — Export Pipeline

### C1. Gemma 4 Model Definition

**Key finding: Google does NOT re-author Gemma 4.** They load the HuggingFace
`transformers` model directly and monkey-patch specific components:

```python
# export_lib.py:114-192
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
config._attn_implementation = 'lrt_transposed_attention'
```

**Gemma 4 patches (`model_ext/gemma4/patch.py`):**
Only one patch: Replace `Gemma4RMSNorm` with a version using StableHLO composite
HLFB (`odml.rms_norm`). This wraps the entire norm into a single fusible op.

**SWA / Full attention interleaving:** Handled by HuggingFace's `config.layer_types`
(values: `"sliding_attention"`, `"full_attention"`). The export pipeline reads this
to create separate masks and cache configurations per layer type.

**KV cache sharing (`model_ext/gemma4/cache.py:92-130`):**
Gemma 4 has `num_kv_shared_layers` where later layers share KV with earlier layers.
The `LiteRTCacheForGemma4` class:
- Creates only `num_hidden_layers - num_kv_shared_layers` unique cache layers
- Inserts dummy references for shared layers before inference
- Strips back to unique layers for I/O

**Only unique KV layers appear in model I/O.** Shared layers point to the same cache.

**Per-Layer Embedding (PLE):** Exported as a separate TFLite model (section 1).
Shape: `[batch, seq_len, num_hidden_layers, hidden_size_per_layer_input]`.
This is a Gemma 4-specific feature — per-layer input modulation.

**MTP drafter:** NOT in the ai-edge-torch export pipeline. Zero references to
MTP, drafter, or speculative decoding. The drafter is trained/exported separately
(likely Google-internal tooling).

### C2. Quantization Recipes

**Default recipe: `dynamic_wi8_afp32`**
- INT8 weight-only, FP32 activations, dynamic range
- Per-channel (channelwise) quantization
- Applied post-conversion via `ai_edge_quantizer`
- No calibration data — purely min/max based

**Mixed-bit per component (via `GenerativeQuantRecipe`):**
```python
GenerativeQuantRecipe(
    default=create_layer_quant_fp16(),
    attention=create_layer_quant_dynamic(Dtype.INT8, Granularity.CHANNELWISE),
    feedforward=create_layer_quant_dynamic(Dtype.INT4, Granularity.BLOCKWISE_128),
)
```

Supported granularities for INT4: BLOCKWISE_32, BLOCKWISE_64, BLOCKWISE_128, BLOCKWISE_256.

**Per-component regex matching:**
- Attention: `transformer_block/litert_torch.generative.layers.attention`
- FFN: `transformer_block/litert_torch.generative.layers.feed_forward`
- Per-block overrides supported via `{idx: LayerQuantRecipe}` dict

### C3. Conversion Passes

**Pipeline (`export_lib.py:234-356`):**
1. `torch.export.export()` — PyTorch 2 export
2. Decomposition passes (pre-lower + TFLite-specific)
3. `converter.convert()` — FX graph → TFLite via StableHLO
4. `mu_pass_lib.update_model()` — fuses sum+mul → mean
5. Mixed precision: FP16 conversion, keeping RMSNorm + Add in FP32
6. Quantization: post-conversion via `ai_edge_quantizer`

**StableHLO composites (fused ops):**

| Composite | Usage | Notes |
|---|---|---|
| `odml.rms_norm` | All RMSNorm layers | pow+mean+rsqrt+mul → single op |
| `odml.scaled_dot_product_attention` | Available but NOT used | HF path uses transposed attention instead |
| `litert_torch::bmm_4d` | Q·K and attn·V | `stablehlo.dot_general` with batch dims [0,1] |

**KV cache as model I/O:** Flattened into model inputs/outputs via PyTorch pytree
registration. Each layer's K and V are separate tensors (`k_0`, `v_0`, `k_1`, `v_1`...).
Cache updates use `tfl.dynamic_update_slice`. Stateful I/O, not internal state.

**Transposed KV layout optimization:**
- K: `[1, BK, S, H]` (seq at dim 2) — natural for Q·K^T
- V: `[1, BK, H, S]` (seq at dim 3) — natural for attn·V
- Avoids all runtime transposes. Matches our existing layout.

**EnumeratedShapes:** Multiple fixed prefill lengths (`prefill_256`, `prefill_512`,
etc.) as separate signatures. Decode always has input length 1. Verify has fixed
length K+1=4.

---

## Implications for CoreML-LLM

### Blocker 2: DISSOLVED

**What we thought:** After accepting draft tokens, we need to "commit" their KV
entries without re-running the full model. We assumed this required a separate
KV write mechanism.

**What Google does:** The verify pass IS the commit. The model runs all K+1 tokens
through the full forward pass, which naturally computes and writes KV for all
positions via `dynamic_update_slice`. After partial acceptance, rejected tokens'
KV entries sit in the cache at positions beyond `current_step`, invisible to
the attention mask.

**What we need to do:** Nothing architectural. Our verify_chunk T=3 models already
run K+1 tokens through the full pipeline. After verification:
1. Advance `current_step` by `num_accepted + 1`
2. Extract `activations` at the last accepted position for carry state
3. Done. Stale KV entries are masked out.

**Reference:** `EAGLE3_INTEGRATION_STATE.md` §Blockers — Blocker 2 ("commit re-runs
decode per accepted token") is no longer a blocker. The verify pass already writes
all needed KV.

### MTP vs EAGLE-3: Strategy Decision

| Factor | Google MTP | Our EAGLE-3 |
|---|---|---|
| Drafter size | 44 MB (4 layers) | 188 MB (47M params) |
| Training | Pre-trained by Google | Custom train, Blocker 1 (distribution mismatch) |
| Acceptance | Unknown (est. 60%+ from 56.5 tok/s) | 0% on-device (forward-mode mismatch) |
| KV dependence | Reads target kv13/kv14 directly | Needs hidden taps at L8/L17/L34 |
| Runtime | 3 sequential drafter calls + 1 verify | 1 draft call + 1 verify |
| Verifier | Same verify chunks, reusable | Same verify chunks, reusable |

**Recommendation:** Proceed with MTP Path A (extract Google's drafter). It's smaller,
pre-trained against Google's reference forward (no Blocker 1), and the verify
infrastructure is already built.

### Quantization: UINT2 Width-for-Bits Trade

Google's actual model is far more aggressive than we assumed:
- L0-14: INT4 attention + INT4 FFN (intermediate=6144)
- L15-34: INT4 attention + **UINT2 FFN** (intermediate=**12288**)
- LM head: **UINT2** [262144,1536]
- Activations: **INT8 per-tensor** at every FC boundary (W4A8 / W2A8)

The width-for-bits trade (2× wider at UINT2 = same bytes as 1× at INT4) is a
novel compression strategy. CoreML's `coremltools` does not support UINT2 natively.

**Options for our pipeline:**
1. **INT4 at 2× width** — approximate the W2A8 layers by using INT4 with expanded
   intermediate size (12288). This doubles model size for those layers but may
   improve quality. ANE should handle the wider matmuls.
2. **Custom 2-bit lookup tables** — palettize to 4 values (2 bits), which
   `coremltools` palettization supports via `nbits=2` in `OpPalettizerConfig`.
3. **Keep uniform INT4** — simpler, proven to work on ANE, and Google's UINT2
   quality may not transfer to our pipeline without their training recipe.

**Recommended first step:** Test 2-bit palettization on L15-34 FFN weights using
`coremltools.optimize.coreml.palettize_weights(nbits=2)` and measure perplexity.

### What Confirms Our Approach

1. **KV cache layout** — K=(1,heads,ctx,hd), V=(1,heads,hd,ctx) matches exactly
2. **Transposed V** — Google uses the same V-transposed layout we adopted
3. **INT8 KV cache** — Google quantizes KV cache to INT8
4. **Separate embedder** — externalized embedding lookup, same as our `EmbeddingLookup.swift`
5. **Chunked prefill** — breaking long prompts into fixed-size chunks
6. **No SDPA fusion** — manual attention, same as us
7. **Precomputed RoPE** — baked into model or separate computation, not runtime

### What We Should Adopt

1. **Per-Layer Embedding (PLE)** — Present in EVERY layer of Google's model, with
   gated injection of per-layer learned embeddings (INT8, 256-dim bottleneck).
   PLE is a 1284 MB section (section 1) — larger than the LLM itself. This
   suggests it's architecturally important, not optional. **Must investigate
   whether correct inference requires PLE or if it's quality-only.**

2. **Warmup inference** — Google explicitly warms up all models before actual inference.
   Maps to our `ANE pipeline prewarming` (Priority 0b). Should be 4× dummy predictions.

3. **Verify activation extraction** — Save the verifier's hidden state at the last
   accepted position to bootstrap the next MTP cycle without re-running decode.
   This is ~free once we have the verify output.

4. **W4A8 / activation quantization** — Google quantizes activations to INT8 at every
   FC boundary (per-tensor, symmetric). On XNNPACK this enables INT8×INT4 kernels.
   On ANE, `coremltools` W8A8 was rejected (compile fails), but W4A16 with FP16
   activations is our current path. Worth revisiting if Apple adds W4A8 support.

### What Invalidates an Assumption

1. **No dynamic K** — We assumed more complex speculation strategies (DISCO) might be
   needed. Google's fixed K=3 with simple greedy acceptance achieves 56.5 tok/s.
   Simplicity wins.

2. **No thermal adaptation** — We assumed Google had runtime thermal management
   contributing to sustained performance. They don't. Their 56.5 tok/s is raw
   compute throughput from efficient XNNPACK CPU kernels + MTP speculation.

3. **EAGLE-3 Blocker 1 cause confirmed** — The export pipeline uses
   `_attn_implementation = 'lrt_transposed_attention'` which differs from HF's
   default. EAGLE-3 was trained against HF's default attention, explaining the
   distribution mismatch.

4. **Quantization is far more aggressive than we assumed.** We thought Google used
   INT8 everywhere. They actually use UINT2 for 60% of FFN compute and the LM head.
   The "width-for-bits" trade (2× wider intermediate at UINT2) is a novel compression
   strategy we hadn't considered. This may explain part of the speed advantage —
   UINT2 weights mean half the memory bandwidth vs INT4.

5. **PLE is not optional.** Every layer in the model has a Per-Layer Embedding
   sub-block. We haven't implemented PLE, which means our current Gemma 4 forward
   may be missing this component entirely. Need to check if HuggingFace's default
   forward includes PLE or if it's LiteRT-specific.

---

## Raw Notes

### Critical File References (LiteRT-LM Runtime)

| File | Key Content | Lines |
|---|---|---|
| `runtime/executor/llm_litert_mtp_drafter.h` | MTP drafter class, `Draft()` API | 39-195 |
| `runtime/executor/llm_litert_mtp_drafter.cc` | Full MTP implementation: draft loop, verify, accept | 1-519 |
| `runtime/executor/llm_litert_npu_compiled_model_executor.h` | NPU executor with speculative decoding | 48-593 |
| `runtime/executor/llm_litert_npu_compiled_model_executor.cc` | NPU MTP loop, KV commit, rejection sampling | 1556-2491 |
| `runtime/executor/llm_litert_compiled_model_executor.h` | CPU/GPU executor with MTP drafter integration | 140-332 |
| `runtime/executor/llm_litert_compiled_model_executor.cc` | CPU MTP call + KV buffer swap | 1030-1135, 766 |
| `runtime/components/model_resources.h` | ModelType enum (kTfLiteMtpDrafter = 13) | 55-56 |
| `schema/py/litertlm_builder.py` | MTP_DRAFTER = "tf_lite_mtp_drafter" | 113-114 |

### Critical File References (ai-edge-torch)

| File | Key Content |
|---|---|
| `litert_torch/generative/export_hf/model_ext/gemma4/cache.py` | KV sharing, per-layer-type head dims |
| `litert_torch/generative/export_hf/model_ext/gemma4/exportable_module.py` | PLE embedder, prefill/decode |
| `litert_torch/generative/export_hf/model_ext/gemma4/patch.py` | RMSNorm → HLFB composite |
| `litert_torch/generative/export_hf/core/export_lib.py` | Main export pipeline |
| `litert_torch/generative/export_hf/core/attention.py` | Transposed attention |
| `litert_torch/generative/quantize/quant_recipe.py:87-160` | Per-component quantization recipes |
| `litert_torch/generative/quantize/supported_schemes.py:28-36` | Supported quant schemes |
| `litert_torch/generative/export_hf/core/mu/mixed_precision.py` | FP16 + selective FP32 |
| `litert_torch/generative/export_hf/core/exportable_module_config.py:42` | Default: `dynamic_wi8_afp32` |

### Key Graph Structure (Section 8 TFLite)

| Property | Value |
|---|---|
| Total operators | 2068 (decode), ~9000+ across all signatures |
| Total tensors | 2703 |
| Subgraphs | 976 (decode=1 main + composite impls) |
| Signatures | 4: decode, prefill_128, prefill_1024, verify_4 |
| KV cache layers | 15 unique (L0-14), shared by L15-34 |
| GQA ratio | 8:1 (8 query heads, 1 KV head) |
| Context window | 32003 tokens |
| Vocab size | 262144 |

### Key Code Snippets

**MTP drafting loop (embed + concat):**
```cpp
// llm_litert_mtp_drafter.cc:326-358
for (int i = 0; i < num_draft_steps_; ++i) {
    // Concat: embed(token) + projected_activations → activations[B,1,3072]
    RETURN_IF_ERROR(embedding_manager_.LookupDecode(last_drafted_token_id, embedding_vector));
    RETURN_IF_ERROR(ConcatenateEmbeddingsAndActivations(
        embedding_vector, *activations_ptr, *drafter_activations_buffer));
    // Run drafter
    LITERT_RETURN_IF_ERROR(mtp_drafter_model_.RunAsync(...));
    // Sample
    RETURN_IF_ERROR(drafter_sampler_->SampleToIdAndScoreBuffer(
        mtp_drafter_output_buffers["logits"], drafter_id_tensor, nullptr));
    // Carry state for next step
    activations_ptr = &mtp_drafter_output_buffers["projected_activations"];
}
```

**Rejection sampling (greedy argmax comparison):**
```cpp
// llm_litert_npu_compiled_model_executor.cc:2438-2451
for (int i = 0; i < draft_tokens.size(); ++i) {
    int sampled_verifier_token = all_verifier_sampled[i];
    if (sampled_verifier_token == draft_tokens[i]) {
        num_accepted++;
    } else {
        bonus_token_id = sampled_verifier_token;
        break;
    }
}
```

**KV commit (NPU - runs auxiliary cache_update model):**
```cpp
// llm_litert_npu_compiled_model_executor.cc:2464-2491
absl::Status CommitVerifiedKVCache(int start_step) {
    // Set positions [start_step, start_step+1, ..., start_step+tensor_size-1]
    for (int i = 0; i < tensor_size; ++i) {
        pos_ptr[i] = start_step + i;
    }
    // Run cache update model
    npu_auxiliary_context_.npu_auxiliary_compiled_model.Run(
        CacheUpdateSignatures::kVerifyCacheUpdate,
        cache_update_inference_context_.verify_input_buffers,
        cache_update_inference_context_.verify_output_buffers);
}
```

**Activation carry state extraction:**
```cpp
// llm_litert_npu_compiled_model_executor.cc:1646-1661
size_t hidden_size_in_bytes = full_activations.size() / dratfer_seq_len;
last_verify_activations_.resize(hidden_size_in_bytes);
memcpy(last_verify_activations_.data(),
       full_activations.data() + rs_result.num_accepted * hidden_size_in_bytes,
       hidden_size_in_bytes);
has_valid_verify_activations_ = true;
```

---

## Appendix: ANE-Specific Optimization Opportunities

Additional findings from deep-dive into LiteRT patterns mapped to Apple Neural
Engine constraints. Ordered by estimated ROI.

### Tier S — High Impact, Implementable Now

#### S1. INT8 KV Cache (Bandwidth 50% Reduction)

**iOS 26 natively supports INT8 model I/O.** Verified via coremltools 9.0+:
`adjust_io_to_supported_types.py` adds `types.int8` to supported I/O types for
`target.iOS26`. iPhone 17 Pro runs iOS 26.

**Bandwidth savings:**

| Component | FP16 (current) | INT8 | Saving |
|---|---|---|---|
| chunk1 KV (7 SWA + 1 full) | 26.1 MB | 13.0 MB | 13.0 MB |
| chunk2 KV (5 SWA + 2 full) | 38.8 MB | 19.4 MB | 19.4 MB |
| Shared KV13 (read 2x) | 3.0 MB | 1.5 MB | 1.5 MB |
| Shared KV14 (read 2x) | 31.3 MB | 15.6 MB | 15.6 MB |
| **Total per decode step** | **99.1 MB** | **49.6 MB** | **49.5 MB (50%)** |

**Implementation path:**
1. Target `ct.target.iOS26` in conversion
2. Add MIL `dequantize(input=kv_int8, scale=calibrated_scale)` before attention matmul
3. Add MIL `quantize(input=new_kv, scale=calibrated_scale, output_dtype='int8')`
   before KV output
4. Scale is compile-time constant (per-head or per-layer, calibrated offline)
5. Swift: allocate KV cache as `MLMultiArray(dataType: .int8)` instead of `.float16`
6. IOSurface: use `kCVPixelFormatType_OneComponent8` (unsigned) with ZP=128, or
   standard `MLMultiArray(.int8)` (may lose zero-copy)

**Constraints:**
- `scale` must be compile-time constant (no adaptive per-token quantization)
- IOSurface zero-copy for int8 needs `kCVPixelFormatType_OneComponent8` (unsigned),
  signed int8 may require standard MLMultiArray fallback
- Google ships Gemma 4 with INT8 KV -- quality impact is acceptable

**Expected impact:** chunk2 (bottleneck) drops from 20.7ms to ~13ms. Total decode
from 67ms to ~52ms -> **~19 tok/s** (vs 14.9 baseline).

#### S2. Chunk3+4 Merge (Dispatch 4->3)

L15-34 are all Q-only KV-shared layers (no K/V projections). They are
computationally lighter per layer than L0-14. A 20-layer merged chunk may
compile on ANE despite the ~15-layer empirical threshold.

**Why this merge is safe:**
- No K/V projections -> fewer ops per layer
- Q-only attention reads shared kv13/kv14 (no per-layer KV I/O)
- Eliminates kv13/kv14 roundtrip between chunk3 and chunk4
- Prototype exists: `gemma4_lite_chunks.py` LiteChunk2 (L15-34)

**Current chunking reason:** ANE compiler hangs with 15+ layers per chunk
(`gemma4_stateless_chunks.py` line 5). But Q-only layers generate fewer MIL ops
than full KV layers, so the effective threshold may be higher.

| Config | Dispatches | Estimated saving |
|---|---|---|
| Current (4 chunks) | 4 | baseline |
| Merge chunk3+4 | 3 | ~5-10ms |
| LiteChunk1+2 (2 chunks) | 2 | ~15-25ms (risky: 15 layers in chunk1) |

**Expected impact:** ~5-10ms -> **~17 tok/s**.

### Tier A — Medium Impact, Requires Implementation

#### A1. Dispatch Pipelining (Pre-prepare Next Step Inputs)

Google's "sampler-driven pipeline" pattern: the sampler prepares next step's
input tensors (positions, masks) AND kicks off the next inference call, all
without a CPU round-trip.

**Current flow:**
```
ANE done -> CPU argmax -> prepare position/mask -> ANE dispatch (idle gap)
```

**Optimized flow:**
```
ANE done -> CPU argmax + parallel next-step input prep -> immediate ANE dispatch
```

Position is predictable (`current_step + 1`), mask is deterministic. These can
be pre-computed during the current ANE execution. Only the token embedding
depends on argmax output.

**Implementation:** In `ChunkedEngine.swift`, pre-compute next step's position
and mask tensors during the current decode. After argmax, only the embedding
lookup is needed before the next dispatch.

**Expected impact:** ~2-5ms/step.

#### A2. Single-Buffer KV Cache (Eliminate copyBack)

Google's `gpu_optimized_single_buffer_cache` pattern: same buffer for KV
input and output, with `param_tensor` indicating write position. Eliminates
the buffer swap and copy overhead.

**Current overhead:** `ChunkedEngine.swift` does `copyBack` (memcpy) of the
entire KV buffer after each chunk execution. At 8K context, this is ~99 MB
of copies per step.

**CoreML approach:** If the model uses `scatter_nd` or `dynamic_update_slice`
to write KV in-place, the same `MLMultiArray` can serve as both input and
output. No `copyBack` needed.

**Risk:** CoreML's ANE backend may require separate input/output buffers
(MLState was rejected for ANE -- error -14). Needs empirical testing.

#### A3. Async ANE Dispatch

Google uses event-based synchronization: all inference is async, CPU only
blocks when reading output for sampling.

**Verify our dispatch is truly async:** If `ChunkedEngine` calls
`prediction(from:)` synchronously, the CPU thread blocks for the entire ANE
execution. Using `prediction(from:options:completionHandler:)` or
async/await allows CPU work (next-step prep) to overlap with ANE execution.

### Tier B — Confirmed Not Viable or Already Done

#### B1. 2-bit Palettization

**Not viable without QAT.** Post-training 2-bit = quality collapse (proven
in `EXPERIMENTS.md`). Apple's own benchmarks show 2-bit barely faster than
4-bit on ANE (LUT decompression overhead). QAT requires multi-day GPU training.

#### B2. SWA Window-Only KV Read

**Already implemented.** SWA layers receive W=512-sized KV tensors, not full
context. The bandwidth bottleneck is full-attention layers only -> addressed
by INT8 KV (S1).

#### B3. Compiled Model Caching

**Verify `.mlmodelc` persistence.** Google caches compiled models (`.xnnpack_cache`).
CoreML equivalently compiles to `.mlmodelc`. Ensure this is persisted across
app launches -- recompiling from `.mlpackage` every time adds seconds to cold start.

### Combined Impact Estimate

```
Baseline:                    67.0 ms/tok = 14.9 tok/s @8K

+ INT8 KV (S1):             ~52 ms = ~19 tok/s  (+28%)
+ chunk3+4 merge (S2):      ~47 ms = ~21 tok/s  (+41%)
+ dispatch pipeline (A1):   ~44 ms = ~23 tok/s  (+54%)
+ MTP K=3 @60% acceptance:  44ms / 2.8 tok = 15.7 ms/tok = ~64 tok/s

Target: LiteRT-LM 56.5 tok/s -- achievable with S1 + S2 + MTP.
```

### Recommended Action Priority

| # | Action | Gain | Effort | Dependency |
|---|---|---|---|---|
| 1 | INT8 KV cache prototype | +28% | 2-3 days (conversion + Swift) | iOS 26 target |
| 2 | chunk3+4 merge on-device bench | +10% | 1 day (LiteChunk2 exists) | None |
| 3 | MTP drafter integration | x2.8 | 3-4 days (Path A) | Blocker 2 resolved |
| 4 | Dispatch pipelining | +5% | 0.5 day Swift | None |

---

## 2026-04-18 addendum — runtime code re-audit

**Scope:** Full re-read of `/tmp/LiteRT-LM/runtime/` (cloned fresh 2026-04-18,
~10K lines across executor / kv_cache / mtp_drafter / session / pipeline /
threadpool) with a focused lens on speed optimizations not already captured
above.

### New finding (top priority)

**Pending-Token Skip** — the single new actionable optimization from this pass.

Code: `/tmp/LiteRT-LM/runtime/executor/llm_litert_compiled_model_executor.cc:1797-1802`
and `llm_executor_processed_tokens.h:114-116`.

Mechanism: at the end of prefill, if exactly 1 unprefilled token remains,
LiteRT-LM does **not** run the graph — it stashes the token as a "pending
input token" on the executor. That token is consumed on the first decode
step as its input, so the normal `tokenize → feed → full-forward → sample`
chain loses one forward pass.

Why this matters for us: our current handoff runs one extra ANE dispatch
(~13 ms) at the prefill→decode seam. Skipping it is a lossless TTFT win.
It is **orthogonal** to Phase 0 item `0d` (prefill bypass of chunk3+4 for
N-1 prompt tokens): `0d` trims chunks 3+4 for all non-last tokens, `0j`
skips the whole graph for the single last token. Stacked they replicate
LiteRT-LM's exact handoff.

Tracked as Phase 0 item **0j** in `docs/PRIORITY_ROADMAP.md` (added
2026-04-18, ~1 day Swift, no reconversion needed).

### Re-confirmed (not new)

The re-read confirmed the findings in this doc's original 2026-04-14 body:
- Single-forward verify with internal `dynamic_update_slice`, stale KV
  masked out via position counter + causal mask (§B1.3 above)
- KV sharing base ↔ drafter via handle duplication, not memory isolation
  (§B2 — see correction below for why the `.Duplicate()` semantics matter)
- Orthogonal model decomposition on the NPU path (§B3 #463-473)
- Session clone / checkpoint for KV rewind (§B4 #496)
- Chunked prefill via `GetOptimizedPrefillWorkGroups` (§B3 #456-458)
- Warmup inference + GPU weight-upload staging (§B4 #480-483)

### Retraction: initial "MTP retry via Duplicate()" hypothesis

The first pass of this re-audit (before cross-checking the LiteRT C API)
produced a hypothesis that LiteRT-LM's MTP drafter isolates draft/verify
KV via `TensorBuffer::Duplicate()` — implying a mechanical workaround for
our verify-chunk write-through issue (item 11c in PRIORITY_ROADMAP).

**That hypothesis is wrong.** Verified via the LiteRT C API header
`litert/c/litert_tensor_buffer.h`:

```c
// Create a duplicate of the current tensor buffer. It will increase the
// reference count of a managed tensor buffer. And the number decreases
// when LiteRtDestroyTensorBuffer() is called.
LiteRtStatus LiteRtDuplicateTensorBuffer(LiteRtTensorBuffer tensor_buffer);
```

Runtime implementation in `litert/runtime/tensor_buffer.h`:

```cpp
// Used to duplicate the current tensor buffer. Internally it increases
// reference count to the underlying buffer.
void Duplicate() const { Ref(); }
```

`Duplicate()` is a **ref-count increment on a single underlying buffer**,
not an allocate-and-copy. The `.Duplicate()` calls in
`llm_litert_mtp_drafter.cc:285-286, 396, 411` therefore do **not** isolate
draft/verify KV memory — they only extend handle lifetime across async
invocations. LiteRT-LM's actual mechanism is exactly what §B1.3 of this
doc already described (mask + position advance).

Implication: LiteRT-LM's design does **not** provide a new path around
item 11c. Our verify-chunk write-through semantics are not the delta. The
blocker remains drafter-quality / numerical compatibility between the
drafter's training target and our HF trunk, same as documented in
`MTP_INVESTIGATION_LITERT.md` (2026-04-17) and the Path A/C autopsies.

### Minor items worth a profile pass but not promoted to Phase 0

Not promoted because gains are speculative without measurement:
- **CPU-ANE stage overlap via callback chaining** —
  `session_basic.cc:593-605` and `tasks.cc:299-340`. Prefill-complete
  callback immediately schedules `RunDecodeAsync`; tokenization runs on
  a worker thread while the previous forward pass is in-flight. If our
  Swift runtime serializes any of {tokenize, sample, constraint-mask}
  against the ANE dispatch, hiding them behind it yields up to ~5% decode
  throughput. Profile first; only implement if profiler shows gap.
- **Dynamic-KV growth by `kv_increment_size`** — `kv_cache.cc:424-452`.
  Grows KV via `ExpandBuffer()` in fixed increments rather than
  pre-allocating max context. Possible memory-use improvement if we care
  about idle memory; no throughput impact.

### Verify sizing divergence from LiteRT-LM (our K vs their K+1)

Re-audit surfaced a **structural divergence** between LiteRT-LM's
`verify_4` signature and our verify chunks that is worth making explicit
alongside §B1.3:

| | LiteRT-LM `verify_4` | Our verify chunks |
|---|---|---|
| verify input length | **K+1 = 4** (seed + K drafts) | **K** (seed + K−1 drafts) |
| KV write positions per cycle | K+1 | K |
| Counter advance on full-accept | num_accepted + 1 ≤ K+1 | matchCount + 1 ≤ K |
| "All-accept KV hole" risk | **impossible by construction** | **dissolved by engine-side cap** |

Google sized verify to K+1 so every input position gets a KV slot and
the counter can advance up to K+1 safely. Our verify is sized to K, so
advancing by K+1 on full-accept would leave position P+K without a KV
write — the "all-accept KV hole" that was reported as a bug against
`MtpSpeculativeEngine.swift` and `CrossVocabSpeculativeEngine.swift`.

**Status (verified 2026-04-18):** the hole is already closed in-tree via
commit `1d9c990` ("fix: commit at most K positions per speculative
cycle"). Two coordinated changes:

1. **Engine side** — `MtpSpeculativeEngine.swift:135` and
   `CrossVocabSpeculativeEngine.swift:128` cap verified drafts to `K−1`
   (`let useProps = Array(proposals.prefix(K - 1))`), so `compareLen =
   K−1`, `matchCount ≤ K−1`, `committed = matchCount + 1 ≤ K`. Never
   over-commits past what was written. The inline comment at
   `MtpSpeculativeEngine.swift:130-134` explicitly names the phenomenon
   ("the all-accept KV hole that lets the next burst read garbage").
2. **Qwen-side `applyCommit`** — `CrossVocabDraft.swift:306-325` handles
   the full-accept case by explicitly issuing
   `stepQwen(lastQwenProposal, position: P+K)` to write the trailing
   Qwen KV slot, then sets `committedPosition = P + K + 1`. Do **not**
   remove this write: the Qwen drafter maintains its own KV separately
   from the target KV, and skipping the write here would create the
   same hole on the Qwen side.

No code change is needed. The bug reports calling for `committed =
min(matchCount + 1, K)` are stale — the K−1 cap already guarantees
that bound, and the proposed change to remove the `stepQwen(...,
position: P+K)` call in `applyCommit` would regress the Qwen-side hole.

**Performance implication (worth tracking).** The K−1 cap is a no-model-
change fix but costs us **one draft slot per cycle**. LiteRT-LM's
theoretical per-cycle headroom is K+1 accepted tokens; ours is K. If we
ever re-export the verify chunks from K to K+1 input positions, we
recover that slot and match LiteRT-LM's upper bound. That re-export
belongs under `PRIORITY_ROADMAP.md` item 11c's "(b) verify-protocol
redesign" track — multi-week due to chunk re-export plus Swift engine
re-wiring, so not on the critical path until speculative decoding is
otherwise unblocked.

### Audit method note

The audit used five parallel Explore agents (NPU executor / main executor
+ KV cache / MTP drafter / pipeline+session+threading / prefill+components
+ magic-numbers) to cover the 10K-line scope without blowing the main
context. The agents correctly surfaced the findings above, but one
reported `.Duplicate()` as "buffer isolation" which was only caught by
verifying the C API definition afterwards. Future runtime audits should
verify any claimed memory-isolation primitive against the implementing
C API before acting on it.
