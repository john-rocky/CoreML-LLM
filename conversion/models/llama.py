"""Llama / SmolLM2 model implementation for CoreML-LLM.

Supports the upstream HuggingFace `LlamaForCausalLM` architecture as used
by Llama 2/3, SmolLM 1/2, OpenLlama, TinyLlama, etc.

Why this is a thin subclass of Qwen2Model:
  * Weight names are byte-identical (HF uses `model.layers.{i}.self_attn.{...}`
    and `model.layers.{i}.mlp.{...}` for both Llama and Qwen2)
  * Both use SwiGLU MLP (gate_proj/up_proj/down_proj), RMSNorm, RoPE, GQA
  * The only architectural differences are at the *config* level:
    - Llama defaults `attention_bias=False` (Qwen2.5 defaults `True`)
    - Llama uses smaller `rope_theta` (typical 10k–500k, vs Qwen2's 1M)
    - Llama is more likely to tie `lm_head` to `embed_tokens`

  All three are read from the HF config and handled at runtime by
  Qwen2Model's `weight_map()` / `load_weights()`, so no override is
  needed. This subclass exists solely to give the converter a clean
  architecture key (`llama`) and to make the dispatch path obvious.

Tested with:
  * `HuggingFaceTB/SmolLM2-135M-Instruct` (49152 vocab, 30 layers,
    Llama-style with GQA Q=9/KV=3, head_dim=64, rope_theta=100k,
    tied embeddings, no attention bias)
  * `HuggingFaceTB/SmolLM2-360M-Instruct` (49152 vocab, 32 layers,
    Q=15/KV=5)
"""

from __future__ import annotations

from .qwen2 import Qwen2Model


class LlamaModel(Qwen2Model):
    """Llama-family decoder-only language model.

    Inherits the entire weight loading / forward path from `Qwen2Model`.
    See file docstring for the reasoning.
    """
    pass
