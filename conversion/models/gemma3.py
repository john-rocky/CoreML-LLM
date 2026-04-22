"""Gemma 3 text decoder (ANE-optimized) for CoreML-LLM.

Used by FunctionGemma-270M-IT and any future vanilla Gemma 3 text decoders.
Architectural subset of Gemma 4 (see models/gemma4.py): same sandwich norms,
QK norm, GELU MLP, GQA, tied embeddings, dual RoPE (global/local θ) following
sliding_window_pattern, but WITHOUT:
- per-layer embeddings (PLE) / per_layer_model_projection
- layer_scalar
- v_norm (RMSNorm without scale on values)
- KV sharing across layers
- dual head_dim (single head_dim throughout)
- double-wide MLP

Weights live under `model.` on HuggingFace (Gemma 3 text is monomodal — there
is no `model.language_model.` wrapper the way multimodal Gemma 4 has).

ANE layout follows docs/ANE_OPTIMIZATION_SURVEY.md + conversion/ane_ops.py:
- all projections as Conv2d(1x1) with (B,C,1,S) input
- ANERMSNorm via cat([x,-x])→LayerNorm trick
- per-token Q/K/V shape (1, num_heads, 1, head_dim)
- GQA expansion via reshape+repeat+view (repeat_kv_ane), NOT repeat_interleave
"""

from __future__ import annotations

import gc
import json
import os

import safetensors.torch
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ane_ops import MODEL_DTYPE, ANERMSNorm, InModelArgmax


class Gemma3Config:
    """Gemma 3 text decoder configuration (read from HuggingFace config.json)."""

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 640)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 18)
        self.num_attention_heads = kwargs.get("num_attention_heads", 4)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 1)
        self.head_dim = kwargs.get("head_dim", 256)
        self.intermediate_size = kwargs.get("intermediate_size", 2048)
        self.vocab_size = kwargs.get("vocab_size", 262144)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.hidden_activation = kwargs.get("hidden_activation", "gelu_pytorch_tanh")
        self.attention_bias = kwargs.get("attention_bias", False)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.bos_token_id = kwargs.get("bos_token_id", 2)
        self.eos_token_id = kwargs.get("eos_token_id", 1)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.sliding_window = kwargs.get("sliding_window", 512)
        self.sliding_window_pattern = kwargs.get("sliding_window_pattern", 6)
        # Gemma 3 270M does not use softcapping; larger variants may.
        self.final_logit_softcapping = kwargs.get("final_logit_softcapping", 0.0) or 0.0
        self.attn_logit_softcapping = kwargs.get("attn_logit_softcapping", 0.0) or 0.0

        # RoPE: Gemma 3 uses two thetas following the sliding/full pattern.
        self.rope_theta = kwargs.get("rope_theta", 1_000_000.0)  # global / full-attn
        self.rope_local_base_freq = kwargs.get(
            "rope_local_base_freq", kwargs.get("rope_local_theta", 10_000.0)
        )
        self.query_pre_attn_scalar = kwargs.get("query_pre_attn_scalar", None)

        # Layer types: list of "sliding_attention" / "full_attention".
        self.layer_types = kwargs.get("layer_types", [])
        if not self.layer_types:
            # Same rule as Gemma 3 modeling code: every sliding_window_pattern-th
            # layer (1-indexed) is full_attention, others are sliding.
            self.layer_types = [
                "full_attention" if (i + 1) % self.sliding_window_pattern == 0
                else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]

        # Context length (trace-time fixed; decode target).
        self.context_length = kwargs.get("context_length", 2048)
        self.state_length = kwargs.get("state_length", self.context_length)

    @classmethod
    def from_json(cls, path: str) -> Gemma3Config:
        with open(path) as f:
            d = json.load(f)
        if "text_config" in d:
            d = d["text_config"]
        return cls(**d)

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"


class Gemma3DecoderLayer(nn.Module):
    """Single Gemma 3 decoder layer (ANE layout).

    Matches HF `Gemma3DecoderLayer`:
      - input_layernorm            (pre-attn)
      - self_attn (q/k/v/o + q_norm/k_norm)
      - post_attention_layernorm   (post-attn, before residual add)
      - pre_feedforward_layernorm  (pre-MLP)
      - mlp (gate/up/down, GeGLU with tanh-approx GELU)
      - post_feedforward_layernorm (post-MLP, before residual add)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        eps: float,
        is_full_attention: bool,
        has_bias: bool,
    ):
        super().__init__()
        self.is_full_attention = is_full_attention
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        # Attention projections as Conv2d(1x1) for ANE (3x throughput vs Linear).
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Conv2d(hidden_size, q_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "k_proj": nn.Conv2d(hidden_size, kv_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "v_proj": nn.Conv2d(hidden_size, kv_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "o_proj": nn.Conv2d(q_dim, hidden_size, 1, bias=False, dtype=MODEL_DTYPE),
            "q_norm": ANERMSNorm(head_dim, eps=eps),
            "k_norm": ANERMSNorm(head_dim, eps=eps),
        })

        # MLP (GELU tanh-approx, matches Gemma 3's `gelu_pytorch_tanh`).
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Conv2d(hidden_size, intermediate_size, 1, bias=False, dtype=MODEL_DTYPE),
            "up_proj": nn.Conv2d(hidden_size, intermediate_size, 1, bias=False, dtype=MODEL_DTYPE),
            "down_proj": nn.Conv2d(intermediate_size, hidden_size, 1, bias=False, dtype=MODEL_DTYPE),
        })

        # Sandwich norms (4 per layer).
        self.input_layernorm = ANERMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = ANERMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = ANERMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = ANERMSNorm(hidden_size, eps=eps)


class Gemma3Model(nn.Module):
    """Gemma 3 text decoder with ANE-optimized layers."""

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config

        # Token embeddings.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers.
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = Gemma3DecoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                intermediate_size=config.intermediate_size,
                eps=config.rms_norm_eps,
                is_full_attention=config.is_full_attention(i),
                has_bias=config.attention_bias,
            )
            self.layers.append(layer)

        # Final norm.
        self.norm = ANERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head (tied to embed_tokens after load).
        self.lm_head = nn.Conv2d(
            config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE
        )

        # In-model argmax to avoid shipping full (1, vocab) logits back to host.
        self.argmax = InModelArgmax()

        # Final logit softcap (0.0 = disabled).
        self.softcap = config.final_logit_softcapping

        # Unified KV cache buffer: (2*num_layers, num_kv_heads, state_len, head_dim).
        cache_shape = (
            2 * config.num_hidden_layers,
            config.num_key_value_heads,
            config.state_length,
            config.head_dim,
        )
        self.register_buffer("kv_cache_0", torch.zeros(cache_shape, dtype=MODEL_DTYPE))

        # Dual RoPE tables (local θ for sliding, global θ for full attention).
        self._build_rope_caches(config)

    def _build_rope_caches(self, config: Gemma3Config) -> None:
        head_dim = config.head_dim
        max_len = config.context_length * 2
        t = torch.arange(max_len).float()

        # Local (sliding_attention) RoPE.
        inv_freq_local = 1.0 / (
            config.rope_local_base_freq
            ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        freqs_local = torch.einsum("i,j->ij", t, inv_freq_local)
        emb_local = torch.cat((freqs_local, freqs_local), dim=-1)
        self.register_buffer("cos_sliding", emb_local.cos().to(MODEL_DTYPE))
        self.register_buffer("sin_sliding", emb_local.sin().to(MODEL_DTYPE))

        # Global (full_attention) RoPE.
        inv_freq_global = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        freqs_global = torch.einsum("i,j->ij", t, inv_freq_global)
        emb_global = torch.cat((freqs_global, freqs_global), dim=-1)
        self.register_buffer("cos_full", emb_global.cos().to(MODEL_DTYPE))
        self.register_buffer("sin_full", emb_global.sin().to(MODEL_DTYPE))

    @classmethod
    def from_pretrained(cls, model_path: str, context_length: int = 2048) -> Gemma3Model:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        if "text_config" in config_dict:
            text_dict = config_dict["text_config"]
        else:
            text_dict = config_dict

        text_dict["context_length"] = context_length
        text_dict["state_length"] = context_length
        config = Gemma3Config(**text_dict)

        if config.attn_logit_softcapping:
            raise NotImplementedError(
                f"attn_logit_softcapping={config.attn_logit_softcapping} is not supported; "
                "FunctionGemma-270M / Gemma 3 270M do not use it. If you hit this, you're "
                "converting a Gemma 3 variant that requires attention softcap plumbing."
            )

        model = cls(config)
        model.load_weights(model_path)
        return model

    def load_weights(self, model_path: str) -> None:
        """Load HuggingFace safetensors weights (Gemma 3 text-only prefix = `model.`)."""
        st_files = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No .safetensors in {model_path}")

        loaded = 0
        skipped_shape = 0

        for st_file in st_files:
            filepath = os.path.join(model_path, st_file)
            state_dict = safetensors.torch.load_file(filepath)

            for hf_name, tensor in state_dict.items():
                local_name = self._map_weight_name(hf_name)
                if local_name is None:
                    continue

                tensor = tensor.to(MODEL_DTYPE)

                try:
                    parts = local_name.split(".")
                    target = self
                    for p in parts[:-1]:
                        target = getattr(target, p)
                    param_name = parts[-1]
                    param = getattr(target, param_name)

                    # Conv2d stores (out, in, 1, 1) — HF stores (out, in) — unsqueeze.
                    if param.dim() == 4 and tensor.dim() == 2:
                        tensor = tensor.unsqueeze(-1).unsqueeze(-1)

                    if param.shape != tensor.shape:
                        skipped_shape += 1
                        print(f"  Skip {hf_name}: shape mismatch {param.shape} vs {tensor.shape}")
                        continue

                    with torch.no_grad():
                        param.copy_(tensor)
                    loaded += 1
                except (AttributeError, RuntimeError):
                    pass

            del state_dict
            gc.collect()

        # Tie lm_head ↔ embed_tokens (Gemma 3 always has tied embeddings).
        if self.config.tie_word_embeddings:
            embed_w = self.embed_tokens.weight.data.to(MODEL_DTYPE)
            self.lm_head.weight.data = embed_w.unsqueeze(-1).unsqueeze(-1)
            print("Tied lm_head weights to embed_tokens")

        print(f"Loaded {loaded} weight tensors (skipped {skipped_shape} on shape)")

    def _map_weight_name(self, hf_name: str) -> str | None:
        """Map HuggingFace weight name → local parameter name.

        Gemma 3 text weights live under `model.` (not `model.language_model.` like
        multimodal Gemma 4). `lm_head.weight` is top-level (or tied).
        """
        # lm_head is top-level on HF (no `model.` prefix).
        if hf_name == "lm_head.weight":
            return "lm_head.weight"

        # Everything else under `model.`.
        if not hf_name.startswith("model."):
            return None
        name = hf_name[len("model."):]

        if name == "embed_tokens.weight":
            return "embed_tokens.weight"
        if name == "norm.weight":
            return "norm.weight"

        if name.startswith("layers."):
            parts = name.split(".")
            layer_idx = int(parts[1])
            rest = ".".join(parts[2:])

            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if rest == f"self_attn.{proj}.weight":
                    return f"layers.{layer_idx}.self_attn.{proj}.weight"

            if rest == "self_attn.q_norm.weight":
                return f"layers.{layer_idx}.self_attn.q_norm.weight"
            if rest == "self_attn.k_norm.weight":
                return f"layers.{layer_idx}.self_attn.k_norm.weight"

            for proj in ("gate_proj", "up_proj", "down_proj"):
                if rest == f"mlp.{proj}.weight":
                    return f"layers.{layer_idx}.mlp.{proj}.weight"

            for norm_name in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                if rest == f"{norm_name}.weight":
                    return f"layers.{layer_idx}.{norm_name}.weight"

        return None


def register_gemma3() -> None:
    """Register Gemma 3 / FunctionGemma in the model registry (called on import)."""
    from config import MODEL_REGISTRY, ConversionConfig
    MODEL_REGISTRY.setdefault(
        "functiongemma-270m",
        ConversionConfig(
            hf_repo="google/functiongemma-270m-it",
            architecture="gemma3",
            default_context_length=2048,
            max_context_length=32768,
            description="FunctionGemma 270M - Gemma 3 decoder fine-tuned for function calling",
        ),
    )
