"""Gemma 4 E2B text decoder implementation for CoreML-LLM.

Supports the text decoder portion of Gemma4ForConditionalGeneration.
Complex architecture with:
- Dual attention: sliding_attention (head_dim=256) + full_attention (head_dim=512)
- KV cache sharing: last 20 layers share KV with earlier layers
- Double-wide MLP for KV-shared layers (intermediate_size * 2)
- Per-layer input embeddings
- QK normalization (RMSNorm on Q and K)
- Logit softcapping (tanh-based, factor=30.0)
- GELU activation (not SiLU)
- Sandwich norm (4 RMSNorms per layer)
- No attention bias
"""

from __future__ import annotations

import gc
import json
import math
import os

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ane_ops import MODEL_DTYPE, ANERMSNorm, InModelArgmax, apply_rotary_pos_emb
from base_model import ModelConfig


class Gemma4Config:
    """Gemma 4 E2B text decoder configuration."""

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 1536)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 35)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 1)
        self.head_dim = kwargs.get("head_dim", 256)
        self.global_head_dim = kwargs.get("global_head_dim", 512)
        self.intermediate_size = kwargs.get("intermediate_size", 6144)
        self.vocab_size = kwargs.get("vocab_size", 262144)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.hidden_activation = kwargs.get("hidden_activation", "gelu_pytorch_tanh")
        self.attention_bias = kwargs.get("attention_bias", False)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.bos_token_id = kwargs.get("bos_token_id", 2)
        self.eos_token_id = kwargs.get("eos_token_id", 1)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 131072)
        self.sliding_window = kwargs.get("sliding_window", 512)
        self.final_logit_softcapping = kwargs.get("final_logit_softcapping", 30.0)
        self.num_kv_shared_layers = kwargs.get("num_kv_shared_layers", 20)
        self.use_double_wide_mlp = kwargs.get("use_double_wide_mlp", True)
        self.hidden_size_per_layer_input = kwargs.get("hidden_size_per_layer_input", 256)
        self.vocab_size_per_layer_input = kwargs.get("vocab_size_per_layer_input", 262144)

        # RoPE params
        rope_params = kwargs.get("rope_parameters", {})
        sliding_rope = rope_params.get("sliding_attention", {})
        full_rope = rope_params.get("full_attention", {})
        self.sliding_rope_theta = sliding_rope.get("rope_theta", 10000.0)
        self.full_rope_theta = full_rope.get("rope_theta", 1000000.0)
        self.full_partial_rotary_factor = full_rope.get("partial_rotary_factor", 0.25)

        # Layer types
        self.layer_types = kwargs.get("layer_types", [])
        if not self.layer_types:
            self.layer_types = []
            for i in range(self.num_hidden_layers):
                if (i + 1) % 5 == 0:
                    self.layer_types.append("full_attention")
                else:
                    self.layer_types.append("sliding_attention")

        # Context length (for conversion)
        self.context_length = kwargs.get("context_length", 512)
        self.state_length = kwargs.get("state_length", 512)

    @classmethod
    def from_json(cls, path: str) -> Gemma4Config:
        with open(path) as f:
            d = json.load(f)
        # Handle nested text_config
        if "text_config" in d:
            d = d["text_config"]
        return cls(**d)

    def is_full_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"

    def is_kv_shared(self, layer_idx: int) -> bool:
        kv_start = self.num_hidden_layers - self.num_kv_shared_layers
        return layer_idx >= kv_start

    def get_head_dim(self, layer_idx: int) -> int:
        return self.global_head_dim if self.is_full_attention(layer_idx) else self.head_dim

    def get_intermediate_size(self, layer_idx: int) -> int:
        if self.is_kv_shared(layer_idx) and self.use_double_wide_mlp:
            return self.intermediate_size * 2
        return self.intermediate_size

    @property
    def kv_sliding_producer(self) -> int:
        """Last non-shared sliding_attention layer; its KV is read by all shared sliding layers."""
        for i in range(self.num_hidden_layers - 1, -1, -1):
            if not self.is_kv_shared(i) and self.layer_types[i] == "sliding_attention":
                return i
        raise ValueError("No non-shared sliding_attention layer found")

    @property
    def kv_full_producer(self) -> int:
        """Last non-shared full_attention layer; its KV is read by all shared full layers."""
        for i in range(self.num_hidden_layers - 1, -1, -1):
            if not self.is_kv_shared(i) and self.layer_types[i] == "full_attention":
                return i
        raise ValueError("No non-shared full_attention layer found")


class Gemma4Model(nn.Module):
    """Gemma 4 E2B text decoder with ANE-optimized layers."""

    def __init__(self, config: Gemma4Config) -> None:
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Per-layer token embeddings (262144 x 8960, each layer uses a 256-dim slice)
        per_layer_dim = config.hidden_size_per_layer_input  # 256
        total_per_layer_dim = per_layer_dim * config.num_hidden_layers  # 8960
        self.embed_tokens_per_layer = nn.Embedding(config.vocab_size, total_per_layer_dim)
        # Conv2d for ANE efficiency (3x matmul throughput)
        self.per_layer_model_projection = nn.Conv2d(config.hidden_size, total_per_layer_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.per_layer_model_projection_scale = config.hidden_size ** -0.5
        self.per_layer_input_scale = 2.0 ** -0.5
        self.per_layer_embed_scale = per_layer_dim ** 0.5
        self.per_layer_projection_norm = ANERMSNorm(per_layer_dim, eps=config.rms_norm_eps)

        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            hd = config.get_head_dim(i)
            inter_size = config.get_intermediate_size(i)
            is_full = config.is_full_attention(i)

            layer = Gemma4DecoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=hd,
                intermediate_size=inter_size,
                eps=config.rms_norm_eps,
                is_full_attention=is_full,
                has_bias=config.attention_bias,
                per_layer_dim=config.hidden_size_per_layer_input,
            )
            self.layers.append(layer)

        # Final norm
        self.norm = ANERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM Head (tied with embeddings)
        self.lm_head = nn.Conv2d(
            config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE
        )

        # Argmax
        self.argmax = InModelArgmax()

        # Softcapping
        self.softcap = config.final_logit_softcapping

        # KV cache: separate for sliding and full attention
        # For simplicity, use unified cache with max head_dim
        max_hd = max(config.head_dim, config.global_head_dim)
        cache_shape = (
            2 * config.num_hidden_layers,
            config.num_key_value_heads,
            config.state_length,
            max_hd,
        )
        self.register_buffer("kv_cache_0", torch.zeros(cache_shape, dtype=MODEL_DTYPE))

        # Precompute RoPE for both sliding and full attention
        self._build_rope_caches(config)

    def _build_rope_caches(self, config):
        """Precompute RoPE cos/sin for both attention types."""
        max_len = config.context_length * 2

        # Sliding attention RoPE (full rotation, theta=10000)
        hd_s = config.head_dim
        inv_freq_s = 1.0 / (config.sliding_rope_theta ** (torch.arange(0, hd_s, 2).float() / hd_s))
        t = torch.arange(max_len).float()
        freqs_s = torch.einsum("i,j->ij", t, inv_freq_s)
        emb_s = torch.cat((freqs_s, freqs_s), dim=-1)
        self.register_buffer("cos_sliding", emb_s.cos().to(MODEL_DTYPE))
        self.register_buffer("sin_sliding", emb_s.sin().to(MODEL_DTYPE))

        # Full attention RoPE (proportional, theta=1M, head_dim=global_head_dim=512)
        # HF uses global_head_dim for inv_freq: 512/2=256 frequencies
        # partial_rotary_factor is NOT used for inv_freq computation in proportional mode
        hd_f = config.global_head_dim  # 512
        inv_freq_f = 1.0 / (config.full_rope_theta ** (torch.arange(0, hd_f, 2).float() / hd_f))
        freqs_f = torch.einsum("i,j->ij", t, inv_freq_f)
        emb_f = torch.cat((freqs_f, freqs_f), dim=-1)  # (max_len, 512)
        self.register_buffer("cos_full", emb_f.cos().to(MODEL_DTYPE))
        self.register_buffer("sin_full", emb_f.sin().to(MODEL_DTYPE))
        self.full_rotary_dim = hd_f

    @classmethod
    def from_pretrained(cls, model_path: str, context_length: int = 512) -> Gemma4Model:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        if "text_config" in config_dict:
            text_dict = config_dict["text_config"]
        else:
            text_dict = config_dict

        text_dict["context_length"] = context_length
        text_dict["state_length"] = context_length
        config = Gemma4Config(**text_dict)

        model = cls(config)
        model.load_weights(model_path)
        return model

    def load_weights(self, model_path: str) -> None:
        """Load weights from HuggingFace safetensors, extracting text decoder only."""
        st_files = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No .safetensors in {model_path}")

        loaded = 0

        for st_file in st_files:
            filepath = os.path.join(model_path, st_file)
            state_dict = safetensors.torch.load_file(filepath)

            for hf_name, tensor in state_dict.items():
                # Only load language model weights
                if not hf_name.startswith("model.language_model."):
                    continue

                # Map HF name to local name
                local_name = self._map_weight_name(hf_name)
                if local_name is None:
                    continue

                tensor = tensor.to(MODEL_DTYPE)

                # Navigate to target parameter
                try:
                    parts = local_name.split(".")
                    target = self
                    for p in parts[:-1]:
                        target = getattr(target, p)
                    param_name = parts[-1]

                    # Reshape for Conv2d
                    param = getattr(target, param_name)
                    if param.dim() == 4 and tensor.dim() == 2:
                        tensor = tensor.unsqueeze(-1).unsqueeze(-1)

                    if param.shape != tensor.shape:
                        print(f"  Skip {hf_name}: shape mismatch {param.shape} vs {tensor.shape}")
                        continue

                    with torch.no_grad():
                        param.copy_(tensor)
                    loaded += 1
                except (AttributeError, RuntimeError) as e:
                    pass

            del state_dict
            gc.collect()

        # Handle tied embeddings
        if self.config.tie_word_embeddings:
            embed_w = self.embed_tokens.weight.data.to(MODEL_DTYPE)
            self.lm_head.weight.data = embed_w.unsqueeze(-1).unsqueeze(-1)
            print("Tied lm_head weights to embed_tokens")

        # NOTE: We previously pre-scaled q_norm.weight by sqrt(head_dim) so
        # Fused SDPA's internal /sqrt(d) would cancel and recover Gemma 4's
        # effective scale=1.0 attention. However, in fp16, scaling Q by sqrt(d)
        # causes the (Q @ K^T) intermediate to overflow before /sqrt(d) is
        # applied (PyTorch reproduces NaN, CoreML produces finite garbage).
        # Reverted to scale=1.0 attention with manual softmax — same math
        # as v0.2.0. Costs SDPA fusion (more ops) but preserves correctness.

        print(f"Loaded {loaded} weight tensors")

    def _map_weight_name(self, hf_name: str) -> str | None:
        """Map HuggingFace weight name to local parameter name."""
        # Strip model.language_model prefix
        name = hf_name
        if name.startswith("model.language_model."):
            name = name[len("model.language_model."):]
        else:
            return None

        # Per-layer embeddings (model-level)
        if name == "embed_tokens_per_layer.weight":
            return "embed_tokens_per_layer.weight"
        if name == "per_layer_model_projection.weight":
            return "per_layer_model_projection.weight"
        if name == "per_layer_projection_norm.weight":
            return "per_layer_projection_norm.weight"

        # embed_tokens
        if name == "embed_tokens.weight":
            return "embed_tokens.weight"

        # Final norm
        if name == "norm.weight":
            return "norm.weight"

        # LM head
        if name == "lm_head.weight":
            return "lm_head.weight"

        # Layer weights
        if name.startswith("layers."):
            parts = name.split(".")
            layer_idx = int(parts[1])
            rest = ".".join(parts[2:])

            # Attention projections
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if rest == f"self_attn.{proj}.weight":
                    return f"layers.{layer_idx}.self_attn.{proj}.weight"

            # QK norms
            if rest == "self_attn.q_norm.weight":
                return f"layers.{layer_idx}.self_attn.q_norm.weight"
            if rest == "self_attn.k_norm.weight":
                return f"layers.{layer_idx}.self_attn.k_norm.weight"

            # MLP projections
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                if rest == f"mlp.{proj}.weight":
                    return f"layers.{layer_idx}.mlp.{proj}.weight"

            # Sandwich norms (4 per layer)
            for norm_name in ["input_layernorm", "post_attention_layernorm",
                              "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
                if rest == f"{norm_name}.weight":
                    return f"layers.{layer_idx}.{norm_name}.weight"

            # Layer scalar
            if rest == "layer_scalar":
                return f"layers.{layer_idx}.layer_scalar"

            # Per-layer input processing
            if rest == "per_layer_input_gate.weight":
                return f"layers.{layer_idx}.per_layer_input_gate.weight"
            if rest == "per_layer_projection.weight":
                return f"layers.{layer_idx}.per_layer_projection.weight"
            if rest == "post_per_layer_input_norm.weight":
                return f"layers.{layer_idx}.post_per_layer_input_norm.weight"

        return None


class Gemma4DecoderLayer(nn.Module):
    """Single Gemma 4 decoder layer with sandwich norm and dual attention support."""

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
        per_layer_dim: int = 256,
    ):
        super().__init__()
        self.is_full_attention = is_full_attention

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        # Attention
        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Conv2d(hidden_size, q_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "k_proj": nn.Conv2d(hidden_size, kv_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "v_proj": nn.Conv2d(hidden_size, kv_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "o_proj": nn.Conv2d(q_dim, hidden_size, 1, bias=False, dtype=MODEL_DTYPE),
            "q_norm": ANERMSNorm(head_dim, eps=eps),
            "k_norm": ANERMSNorm(head_dim, eps=eps),
        })
        # v_norm: RMSNorm without learnable scale (with_scale=False in HF)
        # We use a plain RMSNorm but don't load weights (there are none)
        self.v_norm_eps = eps
        self.head_dim_val = head_dim  # for v_norm computation

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # MLP (GELU)
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Conv2d(hidden_size, intermediate_size, 1, bias=False, dtype=MODEL_DTYPE),
            "up_proj": nn.Conv2d(hidden_size, intermediate_size, 1, bias=False, dtype=MODEL_DTYPE),
            "down_proj": nn.Conv2d(intermediate_size, hidden_size, 1, bias=False, dtype=MODEL_DTYPE),
        })

        # Sandwich norms (4 per layer)
        self.input_layernorm = ANERMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = ANERMSNorm(hidden_size, eps=eps)
        self.pre_feedforward_layernorm = ANERMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = ANERMSNorm(hidden_size, eps=eps)

        # Layer scalar (learnable scaling factor)
        self.layer_scalar = nn.Parameter(torch.ones(1))

        # Per-layer input processing (Conv2d for ANE efficiency — 3x throughput vs Linear)
        self.per_layer_input_gate = nn.Conv2d(hidden_size, per_layer_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.per_layer_projection = nn.Conv2d(per_layer_dim, hidden_size, 1, bias=False, dtype=MODEL_DTYPE)
        self.post_per_layer_input_norm = ANERMSNorm(hidden_size, eps=eps)


# Export registry
def register_gemma4():
    """Register Gemma 4 in the model config."""
    from config import MODEL_REGISTRY, ConversionConfig
    MODEL_REGISTRY["gemma4-e2b"] = ConversionConfig(
        hf_repo="google/gemma-4-E2B-it",
        architecture="gemma4",
        default_context_length=512,
        max_context_length=131072,
        description="Gemma 4 E2B Instruct - Google's smallest Gemma 4 text decoder",
    )
