# Adding New Models to CoreML-LLM

## Before You Start

Check these properties of the target model:

```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("org/model-name")

# Key things to check:
print(config.architectures)          # Model class name
print(config.num_hidden_layers)      # Layer count
print(config.hidden_size)            # Hidden dimension
print(config.num_attention_heads)    # Query heads
print(config.num_key_value_heads)    # KV heads (GQA)
print(config.head_dim)               # Head dimension
print(config.vocab_size)             # Vocabulary size
print(config.tie_word_embeddings)    # Tied embeddings?
print(config.attention_bias)         # Attention has bias?

# Check for special features:
# - QK norm → attention scale might be 1.0
# - Sliding window → needs window-aware masking
# - MoE → needs expert routing
# - Per-layer embeddings → extra embedding table
# - Dual attention types → variable head dimensions
```

## Step-by-Step

### 1. Identify Architecture Differences

Compare with existing supported architectures:

| Feature | Qwen2 | Gemma 4 | Your Model |
|---------|-------|---------|------------|
| Attention | GQA | Dual (sliding/full) | ? |
| Attention bias | Yes | No | ? |
| Attention scale | 1/sqrt(d) | 1.0 (QK norm) | ? |
| Activation | SiLU | GELU | ? |
| Norm type | RMSNorm | RMSNorm + sandwich | ? |
| Norms per layer | 2 | 4 | ? |
| RoPE | Standard | Dual (different theta) | ? |
| KV sharing | No | Yes (layers 15-34) | ? |
| Per-layer embed | No | Yes | ? |
| Layer scalar | No | Yes | ? |
| Logit softcap | No | Yes (30.0) | ? |
| v_norm | No | Yes (scaleless) | ? |

### 2. Create the Model File

```
conversion/models/<arch>.py
```

Start from the simpler template (`qwen2.py`) unless your model needs Gemma 4-level complexity.

Key methods:
- `from_pretrained(model_path, context_length)` — load weights
- `load_weights(model_path)` — map HF names → local names, reshape for Conv2d
- `_map_weight_name(hf_name)` — weight name mapping

### 3. Handle Weight Name Mapping

Print HF weight names to find the pattern:
```python
import safetensors.torch
st = safetensors.torch.load_file("model.safetensors")
for name in sorted(st.keys())[:20]:
    print(f"{name}: {st[name].shape}")
```

Common patterns:
- `model.layers.{i}.self_attn.q_proj.weight` → `layers.{i}.self_attn.q_proj.weight` (Conv2d: add unsqueeze)
- `model.embed_tokens.weight` → `embed_tokens.weight` (no reshape)
- `lm_head.weight` → `lm_head.weight` (Conv2d: add unsqueeze, or tied to embed)

### 4. Attention Scale — Critical Check

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("org/model-name")
print(model.model.layers[0].self_attn.scaling)
```

- If `scaling = 1/sqrt(head_dim)` → use `scale = 1.0 / (head_dim ** 0.5)`
- If `scaling = 1.0` → model uses QK norm, use `scale = 1.0`

**Getting this wrong produces coherent but completely wrong text.**

### 4.5 Decode-Time KV State Layout — Critical for ANE

Before writing the wrapper, decide the decode-path state layout.

- **Monolithic INT4/INT8 > ~1.4 GB silently falls back to GPU on ANE.**
  Plan chunking upfront if your param count × nbits/8 exceeds this.
- **Default to mask-based state writes, not shift-based `cat`.** ANEC
  rejects shift-based `cat([K[:,:,1:,:], k], dim=2)` for Qwen3 + Stateful
  + tied-embedding combinations (error code -14). Mask-based rotating
  buffer works on the same arch patterns that ship today plus the ones
  that fail. See `docs/DECODE_STATE_LAYOUTS.md` §3 for the pattern.
- **Per-step cost is `O(state_length)` on ANE.** Start with
  `context_length=1024`, measure, then decide. For longer effective ctx,
  use sliding-window attention (mask-based rotating, W=1024 default).
- **Palettize with `mode="kmeans"` first.** Linear INT8 div-by-zero on
  sparse tensors; kmeans is the safer default. See `docs/DECODE_STATE_LAYOUTS.md` §4.
- **Parity test before CoreML conversion** (PyTorch HF vs your ANE model).
  Pattern: `conversion/experiments/bonsai/bonsai_reference_oracle.py`.

Full checklist: `docs/DECODE_STATE_LAYOUTS.md` §7.

### 5. Create Wrapper (if needed)

If the model uses the same structure as Qwen2 (standard GQA, 2 norms, SiLU, no special features), the default `MonolithicWrapper` in `exporter.py` works.

If the model has special features, create `models/<arch>_wrapper.py`. Use `gemma4_wrapper.py` as reference.

### 6. Test Before CoreML Conversion

```python
# Quick validation: does PyTorch output match HF?
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_model = AutoModelForCausalLM.from_pretrained("org/model-name")
tokenizer = AutoTokenizer.from_pretrained("org/model-name")

# HF reference
outputs = hf_model.generate(tokenizer("Hello", return_tensors="pt").input_ids, max_new_tokens=10)
print(tokenizer.decode(outputs[0]))

# Your model
from models.<arch> import YourModel
model = YourModel.from_pretrained("./path/to/hf_model")
# ... run forward and compare
```

### 7. Trace and Convert

Ensure no `aten::Int` ops in the traced graph:
```python
traced = torch.jit.trace(wrapper, sample_inputs)
graph = str(traced.graph)
int_ops = [l for l in graph.split('\n') if 'aten::Int' in l]
print(f"Int ops: {len(int_ops)}")  # Must be 0
```

Common fixes for `aten::Int`:
- `x[..., :x.shape[-1]//2]` → `torch.chunk(x, 2, dim=-1)`
- `x.view(batch, -1, dim)` → `x.view(1, num_heads, dim)` (explicit constants)
- `repeat_kv` with shape unpacking → `repeat_interleave(n, dim=1)`

## Verification Checklist

- [ ] Single-token prediction matches HF
- [ ] Multi-token generation matches HF (at least first 5 tokens)
- [ ] CoreML conversion completes without errors
- [ ] CoreML model loads and produces same predictions
- [ ] Int4 quantized model still produces reasonable output

## Worked examples

- **Gemma 3 / FunctionGemma-270M** (causal decoder, rides the existing
  `exporter.py` path with a lean wrapper): see `docs/FUNCTIONGEMMA.md`.
- **EmbeddingGemma-300M** (bidirectional Gemma-3 encoder with mean pool +
  2 dense + L2; stateless, no KV cache, fixed trace-time sequence length):
  see `docs/EMBEDDINGGEMMA.md`.
