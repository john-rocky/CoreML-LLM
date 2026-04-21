# Qwen3.5-0.8B Integration Roadmap

Long-running effort to add `Qwen/Qwen3.5-0.8B` to the CoreML-LLM collection. Kicked off 2026-04-20. This doc is the source of truth; update as Phase checkpoints land.

## What we're actually building

`Qwen/Qwen3.5-0.8B` is **Gated DeltaNet** (Yang et al. ICLR 2025, [arXiv 2412.06464](https://arxiv.org/pdf/2412.06464)) — *not* plain Mamba. It combines Mamba-2 gated decay with the DeltaNet delta rule. Architecture:

- 24 decoder layers, pattern `[linear, linear, linear, full]` × 6 → 18 linear-attention + 6 full-attention
- `full_attention_interval=4`
- Linear layers: `Qwen3_5GatedDeltaNet` (inherits `Qwen3NextGatedDeltaNet`)
- Full layers: `Qwen3_5Attention` with **attention-output gate** (post-SDPA, pre-`o_proj` sigmoid gate — NeurIPS 2025 best-paper pattern, folded into `q_proj` which outputs `2*heads*head_dim`)
- Vision (`Qwen3_5ForConditionalGeneration`) + MTP drafter are **orthogonal, skip for v0**
- `hidden=2048`, `num_heads=16`, `head_dim=256`, `vocab=248320`, `ctx=262144` (clip to 2048)
- `mamba_ssm_dtype=float32` — **scan accumulator must run in fp32**

## Why no prior art does this on CoreML

- No CoreML conversion of Mamba-1/2/DeltaNet/Gated-DeltaNet exists. We'd be first.
- `mlx-lm` Qwen3-Next uses a **custom Metal kernel** (not pure MLX ops). Not portable to CoreML.
- `mamba.py` MLX pscan (Blelloch in pure ops) is documented as *slower than sequential* on M2 Pro. Cautionary tale: don't follow the pure-MIL parallel-scan path naively.
- llama.cpp Mamba is CPU-only — caused the known 14× Qwen3.5 regression on Vulkan (#19957). No Metal `ggml_ssm_scan` kernel.
- **Scan-free reference is `ssd_minimal.py`** (Mamba-2 SSD paper) + HF `torch_chunk_gated_delta_rule` for our exact model. Both use only `cumsum, exp, einsum, matmul, tril_mask, pad, cat`.

## Green lights (confirmed safe)

- **head_dim=256 on ANE**: Gemma 4 E2B ships with `head_dim=256` at 99.78% ANE; global-attention layers use `head_dim=512`. The "4× Qwen2" concern is cancelled.
- **Text-only extraction is trivial**: `Qwen3_5ForCausalLM(config=Qwen3_5TextConfig.from_pretrained(...))` wraps `Qwen3_5TextModel` directly. Vision is never instantiated; `_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]` drops MTP silently. No patching.
- **Tokenizer reuses existing Qwen2 path** (`Qwen2Tokenizer`, ByteLevel BPE, `<|im_start|>`/`<|im_end|>`). Vocab extends to 248k for multimodal tokens that simply never fire in text-only mode.
- **MRoPE collapses to plain RoPE in text-only mode**: `partial_rotary_factor=0.25` rotates only the first 64 of 256 head dims; `mrope_section=[11,11,10]` interleaves T/H/W but for text-only T=H=W so it's standard RoPE on 64 dims with `rope_theta=10_000_000`. Gemma 4's `_build_rope_caches` + 3-line prefix-split patch suffices.
- **Attention output gate is trivial**: element-wise `sigmoid(gate) * attn_out` before `o_proj`. The gate is folded into `q_proj.weight` (second half of the 2×head_dim output). Does not break iOS 18 fused SDPA — apply after SDPA.
- **conv1d depthwise kernel=4 is ANE-native** (conv is the one true ANE-native op per `docs/MIL_OP_CATALOG.md §2.3`).
- **Decode path is trivial**: 1-token `torch_recurrent_gated_delta_rule` has no cumsum, no tril, no while_loop — just fp16 matmul + outer-product state update. This is Phase 2's happy path.

## Red lights (must engineer around)

- **`mb.cumsum` placement on ANE is uncertain** (`docs/MIL_OP_CATALOG.md:129` — "Mostly CPU"). Fallback: **`x @ tril(ones(L,L))`** — a single ANE-native matmul with a constexpr lower-triangular mask. O(S²) flops but ANE-resident; at S=2048 the mask is ~8MB fp16, cheap.
- **`mb.cumprod` does not exist**. The torch frontend unrolls it into `S` scatter ops — catastrophic for S=2048. Express every cumulative product as `exp(cumsum(log(·)))`. Aligns naturally with DeltaNet's log-space decay `g = -exp(A_log) * softplus(a + dt_bias)`.
- **`mb.while_loop` kills ANE placement** across the whole graph (empirical, apple/coremltools#2004). The 32 outer-chunk iterations (seq 2048 / chunk 64) and the 64 intra-chunk UT-substitution steps **must be unrolled at trace time**. Graph gets big but stays ANE-eligible.
- **MLState still fails on iOS 26 with `-14`** (`docs/D5_STATEFUL_KV_IOS26.md` STAY-REJECTED). SSM state goes in explicit I/O tensors (batch, num_v_heads, k_head_dim, v_head_dim) like KV cache, same design as Gemma 4.
- **fp32 accumulator** (`mamba_ssm_dtype=float32`) vs ANE's fp16-only world: split fp32 decay/segsum to **GPU**, keep fp16 matmuls on **ANE**. Mirrors b1_prefill_bypass's GPU-epilogue pattern.

## Revised odds

| Goal | Odds | Reasoning |
|---|---:|---|
| Text-only parity cos≥0.998 vs HF | 60-70% | Math is fully scan-free; reference impl in HF is 80 lines |
| ≥95% ANE placement on decode | 60-70% | Recurrent form is fp16 matmul-only; no cumsum/cumprod in decode |
| ≥95% ANE placement on prefill | 25-35% | Chunked algorithm pushes cumsum+unrolled UT loop; fp32 scan forces GPU split |
| Beat Gemma 4 E2B decode (31 tok/s) on iPhone 17 Pro | 25-35% | SSM decode step is ~5 small matmuls/layer, state doesn't grow with context; but 18 linear + 6 full = more layer boundaries than Gemma's chunked model |

Fallback trigger: if by end of Phase 2 step 2 the decode path can't stay ≥80% ANE, pivot to **Qwen3-0.6B** (plain transformer, ~80% success probability for full integration).

## Phased plan

### Phase 0 — ANE primitive probe (blocked on eagle3 freeing ANE)
- `conversion/probe_scan_ops_ane.py` tests: cumsum, depthwise conv1d k=4, lower-triangular matmul, SDPA head_dim=256, large SSM state tensor update.
- Seq lengths 128/512/2048. Output: per-op device placement table.
- **Gating decisions out of this probe:**
  - cumsum falls off ANE? → use tril-matmul trick (confirmed ANE in probe 3)
  - conv1d k=4 on ANE? (expected yes) → use as-is for Mamba conv state
  - SDPA head_dim=256 fused on ANE? (expected yes per Gemma 4) → use `mb.scaled_dot_product_attention` iOS 18+
  - state shape (1, 1024, 128) accepted? → layout decision for SSM I/O tensors

### Phase 1 — PyTorch reference & parity oracle
- `Qwen3_5ForCausalLM` + `Qwen3_5TextConfig`. CPU or MPS.
- Fixed 10-prompt set, capture logits and top-10 tokens per position. Save as `conversion/qwen3_5_reference_logits.pt`.
- Sanity: sample decode with tokenizer, confirm coherent English.
- Also validate both reference implementations: `torch_chunk_gated_delta_rule` (prefill) and `torch_recurrent_gated_delta_rule` (decode) produce identical logits on seq=8.
- **Does not need ANE. CPU is free, runs in parallel with eagle3.**

### Phase 2 — MIL hybrid block (split into decode-first, prefill-second)

**Phase 2a — Decode path (easy half)**
- `conversion/models/qwen3_5.py` — skeleton file.
- Implement **recurrent** Gated DeltaNet: fp16 matmul + outer-product + state carry. 1-token forward, SSM state as I/O tensor.
- Implement `Qwen3_5Attention` with output gate (post-SDPA sigmoid mul).
- Implement `apply_interleaved_mrope` → simplified to prefix-64-dim RoPE.
- Layer-type dispatch per `layer_types`.
- Parity gate: cos≥0.999 vs Phase 1 recurrent reference, 1 token at a time over 32 tokens.

**Phase 2b — Prefill path (hard half)**
- Implement `torch_chunk_gated_delta_rule` in MIL with unrolled outer (32 chunks) and intra (64 UT steps) loops.
- Use `cumsum` if probe green, else tril-matmul.
- fp32 accumulator regions marked for GPU placement explicitly.
- Parity gate: cos≥0.998 vs Phase 1 chunk reference at seq 512 and 2048.

### Phase 3 — Chunked INT4 conversion
- Apply Gemma 4 chunk pattern, splitting at full-attention boundaries (every 4 layers).
- INT4 palettization per existing recipe.
- **V-weight reordering trick from llama.cpp PR #19468** at conversion time — avoids runtime `repeat_interleave`.
- Upload to `mlboydaisuke/qwen3.5-0.8b-coreml` with `text-only` tag.

### Phase 4 — Swift/iOS integration
- `ModelDownloader.ModelInfo.defaults` entry.
- iOS `CoreMLLLMChat` picker case.
- README row under Pre-converted Models.
- On-device tok/s + thermal + ComputePlanAudit per standard validation.

### Phase 5+ — Deferred
- Re-enable MTP drafter (single-layer, `_keys_to_ignore_on_load_unexpected` stops silently ignoring it).
- Vision encoder.
- Reasoning `<think>` tags UI support.

## CLI registration anchors (Phase 2 lands these)

- `conversion/config.py` `MODEL_REGISTRY`: map `qwen3.5-0.8b` → `Qwen/Qwen3.5-0.8B`, arch `qwen3_5`.
- `conversion/convert.py` `_detect_architecture()` / `_detect_architecture_from_path()`: route `qwen3_5`/`qwen3.5` before the generic `qwen3` substring check.
- `conversion/convert.py` `_get_model_class()`: `if architecture == "qwen3_5": from models.qwen3_5 import Qwen35Model`.
- `conversion/models/__init__.py`: export `Qwen35Model`.

## Reference files to read in full before Phase 2

1. `transformers/models/qwen3_5/modular_qwen3_5.py` — short, authoritative, shows what's new vs inherited.
2. `transformers/models/qwen3_next/modeling_qwen3_next.py` — `Qwen3NextGatedDeltaNet` chunked + recurrent kernels, state shapes, conv1d. **This is what we reimplement in MIL.**
3. `transformers/models/qwen3_5/modeling_qwen3_5.py` lines 261–303 (MRoPE) + 863–988 (attention + gate).
4. `state-spaces/mamba/mamba_ssm/modules/ssd_minimal.py` — 48-line cumsum-only reference for the chunked algorithm.
5. `llama.cpp` PR #19468 — GGUF conversion handler; copy the V-weight reordering.
6. Songlin Yang, "DeltaNet Explained Part II" — UT-form recurrence reference.

## Known issues to watch across the ecosystem

- transformers.js #1574: Qwen3.5 ONNX conversion unsolved
- optimum #2351: Qwen3 ONNX export fails on dynamic-shape bool-tracer
- sglang #20774: Mamba conv_states dtype mismatch — same trap we must avoid
- vllm #39273: ngram spec decoding corrupts hybrid-GDN output
- llama.cpp #20222: hybrid attention chunk restore bug under parallel load
- mlx-lm #1136: closed without native Qwen3.5 support

No CoreML-specific issues yet — we'd be the first public data point.
