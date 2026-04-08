# CoreML LLM Conversion Guide

## Quick Start

```bash
cd conversion
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn  # for int4 quantization

# Qwen2.5-0.5B
python convert.py --model qwen2.5-0.5b --context-length 2048 --output ./output/qwen2.5-0.5b

# Gemma 4 E2B (multimodal text decoder)
python convert.py --model gemma4-e2b --context-length 512 --output ./output/gemma4-e2b
```

## Supported Models

| Model | Architecture | HF Repo | Context | Notes |
|-------|-------------|---------|---------|-------|
| Qwen2.5-0.5B | qwen2 | Qwen/Qwen2.5-0.5B-Instruct | 2048 | Simplest, good for validation |
| Qwen2.5-1.5B | qwen2 | Qwen/Qwen2.5-1.5B-Instruct | 2048 | Better quality |
| Gemma 4 E2B | gemma4 | google/gemma-4-E2B-it | 512 | Complex, multimodal text decoder |

## Adding a New Model

### Step 1: Check Architecture Compatibility

Required properties for CoreML/ANE conversion:
- Decoder-only transformer
- RMSNorm (we use ANE-optimized `[x,-x]` trick)
- RoPE (precomputed cos/sin tables)
- GQA or MHA (not MoE — MoE requires special handling)
- Standard SiLU/GELU activation

### Step 2: Create Model File

Create `conversion/models/<arch>.py` inheriting patterns from existing models.

Key components:
1. **Config class**: Parse HF `config.json`
2. **Model class**: `nn.Module` with Conv2d layers, ANERMSNorm, KV cache buffer
3. **Weight loading**: Map HF weight names to local parameter names, reshape `(out, in)` → `(out, in, 1, 1)` for Conv2d

### Step 3: Create Wrapper (if architecture is complex)

For models with non-standard features (e.g., Gemma 4), create a dedicated wrapper in `models/<arch>_wrapper.py`.

### Step 4: Register in config.py

```python
MODEL_REGISTRY["model-name"] = ConversionConfig(
    hf_repo="org/model-name",
    architecture="arch",
    default_context_length=2048,
)
```

### Step 5: Update convert.py

Add architecture detection and model class import.

## Architecture-Specific Notes

### Qwen2/2.5 (Simple)

- Standard GQA (14 heads, 2 KV heads for 0.5B)
- Attention bias: True
- SiLU activation
- Tied word embeddings
- RoPE theta: 1,000,000

No special handling needed. The base `MonolithicWrapper` in `exporter.py` works directly.

### Gemma 4 E2B (Complex)

This model required extensive debugging. Key lessons:

#### 1. Attention Scale = 1.0 (NOT 1/sqrt(head_dim))

**This was the root cause of incorrect output.** Gemma 4 uses QK normalization (RMSNorm on Q and K before attention), which normalizes the vectors. The traditional `1/sqrt(d)` scaling is therefore unnecessary and must be `1.0`.

```python
# WRONG:
scale = 1.0 / (head_dim ** 0.5)  # 0.0625 for head_dim=256

# CORRECT:
scale = 1.0  # QK norm handles scaling
```

**How to detect**: Check `model.layers[0].self_attn.scaling` in HuggingFace. If it's `1.0`, the model uses QK norm and doesn't need additional scaling.

#### 2. Dual Attention (Sliding + Full)

Every 5th layer uses full attention (head_dim=512), others use sliding attention (head_dim=256).

- **Different RoPE**: sliding uses theta=10,000; full uses theta=1,000,000
- **Full attention RoPE**: Uses `global_head_dim=512` for inv_freq computation (NOT `partial_rotary_factor * global_head_dim`)
- **KV cache**: Padded to max(head_dim)=512 for unified storage; trimmed to actual head_dim for attention

#### 3. KV Cache Sharing

Layers 15-34 share KV from layers 13 (sliding) and 14 (full):
- Sliding KV-shared layers → layer 13's KV
- Full attention KV-shared layers → layer 14's KV

This means KV-shared layers **do not compute their own K/V projections** during inference with KV cache.

```python
# Check sharing config:
layer.self_attn.is_kv_shared_layer  # True for layers 15-34
layer.self_attn.kv_shared_layer_index  # 13 or 14
layer.self_attn.store_full_length_kv  # True for layers 13, 14
```

#### 4. Value Normalization (v_norm)

Gemma 4 applies RMSNorm (without learnable scale) to value states before attention:
```python
value_states = v_norm(value_states)  # RMSNorm without weight
```

#### 5. Per-Layer Embeddings

Each layer gets additional token information from a separate embedding table:
1. `embed_tokens_per_layer(input_ids)` → scaled by `sqrt(256)`
2. `per_layer_model_projection(inputs_embeds)` → scaled by `hidden_size^-0.5`
3. Projection is norm'd per-layer-slice, then combined with raw per-layer embedding
4. In each layer: gate → GELU → multiply with per-layer input → project → norm → residual

#### 6. Sandwich Norm (4 norms per layer)

- `input_layernorm` (before attention)
- `post_attention_layernorm` (after attention, before residual)
- `pre_feedforward_layernorm` (before MLP)
- `post_feedforward_layernorm` (after MLP, before residual)

#### 7. Layer Scalar

Each layer has a learnable `layer_scalar` parameter (typically 0.01-0.8) that scales the entire layer output including residual.

#### 8. Logit Softcapping

Output logits are capped: `tanh(logits / 30) * 30`

#### 9. GELU Activation (not SiLU)

Both MLP and per-layer gate use `gelu_pytorch_tanh`.

## CoreML/ANE Optimization Techniques

### ANERMSNorm (`[x, -x]` trick)

Standard RMSNorm uses `rsqrt(mean(x^2))` which ANE can't accelerate. The trick:
1. `cat([x, -x])` → mean becomes 0
2. `LayerNorm([x, -x])` → uses ANE's optimized LayerNorm kernel (equivalent to RMSNorm when mean=0)
3. Take first half → correct RMS-normalized values

**Critical**: Must compute in float32 internally (cast input to float32, compute, cast back). Matches HF's `Gemma4RMSNorm` which also computes in float32.

### Conv2d Linear

Replace `nn.Linear(in, out)` with `nn.Conv2d(in, out, kernel_size=1)`. ANE processes Conv2d natively.

Weight reshape: `(out, in)` → `(out, in, 1, 1)`
Input layout: `(batch, seq, features)` → `(batch, features, 1, seq)` for Conv2d

**Proven**: Conv2d and nn.Linear produce identical results (diff=0.0) for the same weights in float16.

### In-Model Argmax

Compute argmax inside the CoreML graph. Outputs only token ID + logit value instead of full vocabulary logits (150K+ values). Dramatically reduces ANE → CPU data transfer.

### Stateful KV Cache (MLState)

Uses Apple's `MLState` API (iOS 18+) for persistent KV cache across predictions. 13x faster than passing KV cache as input/output tensors.

### Mask-Based Cache Update

For `torch.jit.trace` compatibility, KV cache updates use:
```python
cache_new = cache * (1 - update_mask) + new_value * update_mask
```
Instead of direct index assignment which creates untraceable `int` ops.

## Debugging Precision Issues

### Methodology

1. **Single token test**: Verify 1-token prediction matches HF
2. **2-token test**: If single token matches but 2 tokens diverge, the issue is in KV cache interaction
3. **Layer-by-layer comparison**: Hook into HF layers, compare outputs at each layer
4. **Component isolation**: Test each component (embedding, norm, QKV, attention, MLP) independently

### Common Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong attention scale | Output is English but wrong content | Check `self_attn.scaling` in HF |
| Python float vs tensor multiplication | 0.004 diff in embeddings | Use `torch.tensor(scale, dtype=float16)` |
| Missing v_norm | Divergence at layer boundaries | Check if model has `v_norm` |
| Missing KV sharing | Divergence at layer 15+ | Check `is_kv_shared_layer` |
| `aten::Int` in trace | CoreML conversion fails | Use `torch.chunk` instead of shape indexing, `torch.index_select` instead of tensor indexing |
| `rotate_half` with `x.shape[-1]//2` | `aten::Int` op | Use `torch.chunk(x, 2, dim=-1)` |
| Rank-0 tensor input | CoreML rejects scalar | Use shape `(1,)` instead of `()` |
| Dynamic shape in `repeat_kv` | `aten::Int` from shape unpacking | Use `repeat_interleave(n, dim=1)` |

### Float16 Precision Checklist

- [ ] RMSNorm computes internally in float32 (cast input, compute, cast back)
- [ ] Attention matmul in float32 (Q, K cast to float32 before matmul)
- [ ] Scale constants as tensors, not Python floats
- [ ] Softmax in float32 before casting back
