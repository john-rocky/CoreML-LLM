"""LFM2 / LFM2.5 (Liquid Foundation Model 2) for CoreML-LLM.

Hybrid architecture: every layer is either
  - "full_attention": GQA + RoPE + QK-RMSNorm (Llama-style, no attn bias)
  - "conv":           Lfm2ShortConv — depthwise causal Conv1d (kernel=L_cache)
                      with gated in/out projections (in_proj 3x, B*x then C*conv).

LFM2.5-350M layout (16 layers):
  conv conv attn conv conv attn conv conv attn conv attn conv attn conv attn conv

This file only defines weight-loading and module structure.  The actual
inference graph (KV cache + conv-state cache + per-layer dispatch) lives in
``conversion.models.lfm2_wrapper.Lfm2MonolithicWrapper`` to keep the export
glue isolated, mirroring how gemma3/gemma4 split decoder vs. wrapper.
"""

from __future__ import annotations

import gc
import json
import os
import sys

import safetensors.torch
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ane_ops import MODEL_DTYPE, ANERMSNorm
from base_model import ModelConfig


def _adjust_intermediate_size(config_dict: dict) -> int:
    """Reproduce the HF Lfm2MLP intermediate_size adjustment.

    The HF MLP normally takes ``intermediate_size`` directly, but when
    ``block_auto_adjust_ff_dim`` is True (LFM2.5 default) it shrinks by 2/3
    and rounds up to a multiple of ``block_multiple_of``.  Without this we
    end up with a 6656-wide MLP instead of the trained 4608.
    """
    inter = int(config_dict.get("intermediate_size") or config_dict.get("block_ff_dim"))
    if not config_dict.get("block_auto_adjust_ff_dim", True):
        return inter
    inter = int(2 * inter / 3)
    mult = float(config_dict.get("block_ffn_dim_multiplier", 1.0))
    inter = int(mult * inter)
    multiple_of = int(config_dict.get("block_multiple_of", 256))
    inter = multiple_of * ((inter + multiple_of - 1) // multiple_of)
    return inter


class Lfm2Attention(nn.Module):
    """LFM2 attention block.

    Differences vs the generic ANEAttention in base_model.py:
      - output projection is ``out_proj`` (not ``o_proj``), no bias
      - Q/K each pass through a per-head RMSNorm (``q_layernorm`` /
        ``k_layernorm``) BEFORE RoPE
      - QKV projections have no bias
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        h = config.hidden_size

        self.q_proj = nn.Conv2d(h, q_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.k_proj = nn.Conv2d(h, kv_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.v_proj = nn.Conv2d(h, kv_dim, 1, bias=False, dtype=MODEL_DTYPE)
        self.out_proj = nn.Conv2d(q_dim, h, 1, bias=False, dtype=MODEL_DTYPE)
        self.q_layernorm = ANERMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = ANERMSNorm(self.head_dim, eps=config.rms_norm_eps)


class Lfm2ShortConv(nn.Module):
    """Gated short-conv block.

    HF reference: in_proj produces ``[B, C, x]`` along the channel axis (3·H
    out_features), the gated branch is ``Bx = B * x`` then a depthwise causal
    Conv1d (kernel=L_cache, groups=H), then ``y = C * conv(Bx)``, then
    out_proj back to H.

    For the ANE export we keep the two linears as Conv2d(1,1) (matches every
    other linear in this stack) and represent the depthwise conv as a 4D
    Conv2d(kernel=(1, L_cache), groups=H).  The trace receives the cached
    last L_cache-1 timesteps as a state input — see ``lfm2_wrapper.py``.
    """

    def __init__(self, config: ModelConfig, conv_l_cache: int, conv_bias: bool) -> None:
        super().__init__()
        h = config.hidden_size
        self.l_cache = conv_l_cache
        self.in_proj = nn.Conv2d(h, 3 * h, 1, bias=conv_bias, dtype=MODEL_DTYPE)
        self.out_proj = nn.Conv2d(h, h, 1, bias=conv_bias, dtype=MODEL_DTYPE)
        self.conv = nn.Conv2d(
            h, h, kernel_size=(1, conv_l_cache), groups=h,
            bias=conv_bias, dtype=MODEL_DTYPE,
        )


class Lfm2MLP(nn.Module):
    """SwiGLU MLP using HF naming (w1=gate, w3=up, w2=down)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        h = config.hidden_size
        i = config.intermediate_size
        self.w1 = nn.Conv2d(h, i, 1, bias=False, dtype=MODEL_DTYPE)
        self.w3 = nn.Conv2d(h, i, 1, bias=False, dtype=MODEL_DTYPE)
        self.w2 = nn.Conv2d(i, h, 1, bias=False, dtype=MODEL_DTYPE)


class Lfm2DecoderLayer(nn.Module):
    """One LFM2 layer.  ``is_attention_layer`` selects which operator is
    instantiated; the unused name simply doesn't exist on this module."""

    def __init__(
        self,
        config: ModelConfig,
        is_attention: bool,
        conv_l_cache: int,
        conv_bias: bool,
    ) -> None:
        super().__init__()
        self.is_attention_layer = is_attention
        if is_attention:
            self.self_attn = Lfm2Attention(config)
        else:
            self.conv = Lfm2ShortConv(config, conv_l_cache, conv_bias)
        self.feed_forward = Lfm2MLP(config)
        self.operator_norm = ANERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = ANERMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Lfm2Model(nn.Module):
    """LFM2 / LFM2.5 model container — weights only, no forward.

    The forward graph used for tracing is built in ``Lfm2MonolithicWrapper``,
    which reads the parameters off this module.  This lets us keep the
    weight-loading logic separate from the export-specific reshapes/states.
    """

    def __init__(self, config: ModelConfig, layer_types: list[str],
                 conv_l_cache: int, conv_bias: bool) -> None:
        super().__init__()
        self.config = config
        self.layer_types = layer_types
        self.conv_l_cache = conv_l_cache
        self.conv_bias = conv_bias

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Lfm2DecoderLayer(
                config,
                is_attention=(t == "full_attention"),
                conv_l_cache=conv_l_cache,
                conv_bias=conv_bias,
            )
            for t in layer_types
        ])
        self.embedding_norm = ANERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Conv2d(
            config.hidden_size, config.vocab_size, 1, bias=False, dtype=MODEL_DTYPE,
        )

        # Cached per-layer "is attn?" flags + slot indices (precomputed for the
        # wrapper to avoid Python-level branching inside the traced forward).
        self.attn_layer_indices: list[int] = [
            i for i, t in enumerate(layer_types) if t == "full_attention"
        ]
        self.conv_layer_indices: list[int] = [
            i for i, t in enumerate(layer_types) if t == "conv"
        ]

    @classmethod
    def from_pretrained(cls, model_path: str, context_length: int = 2048) -> Lfm2Model:
        with open(os.path.join(model_path, "config.json")) as f:
            cfg_dict = json.load(f)

        adjusted_inter = _adjust_intermediate_size(cfg_dict)

        rope_params = cfg_dict.get("rope_parameters") or {}
        rope_theta = float(rope_params.get("rope_theta", cfg_dict.get("rope_theta", 1_000_000.0)))

        head_dim = cfg_dict.get("head_dim") or (
            cfg_dict["hidden_size"] // cfg_dict["num_attention_heads"]
        )

        config = ModelConfig(
            architectures=cfg_dict.get("architectures", ["Lfm2ForCausalLM"]),
            hidden_size=cfg_dict["hidden_size"],
            num_hidden_layers=cfg_dict["num_hidden_layers"],
            num_attention_heads=cfg_dict["num_attention_heads"],
            num_key_value_heads=cfg_dict["num_key_value_heads"],
            intermediate_size=adjusted_inter,
            vocab_size=cfg_dict["vocab_size"],
            head_dim=head_dim,
            rms_norm_eps=float(cfg_dict.get("norm_eps", 1e-5)),
            rope_theta=rope_theta,
            max_position_embeddings=cfg_dict.get("max_position_embeddings", 128_000),
            tie_word_embeddings=bool(
                cfg_dict.get("tie_embedding", cfg_dict.get("tie_word_embeddings", True))
            ),
            attention_bias=False,
            hidden_act="silu",
            bos_token_id=cfg_dict.get("bos_token_id", 1),
            eos_token_id=(
                cfg_dict["eos_token_id"]
                if isinstance(cfg_dict.get("eos_token_id"), int)
                else 7
            ),
            context_length=context_length,
            state_length=context_length,
        )

        layer_types = cfg_dict.get("layer_types")
        if layer_types is None:
            full_attn_idxs = cfg_dict.get("full_attn_idxs", list(range(config.num_hidden_layers)))
            layer_types = [
                "full_attention" if i in full_attn_idxs else "conv"
                for i in range(config.num_hidden_layers)
            ]
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                f"layer_types length {len(layer_types)} != num_hidden_layers "
                f"{config.num_hidden_layers}"
            )

        conv_l_cache = int(cfg_dict.get("conv_L_cache", 3))
        conv_bias = bool(cfg_dict.get("conv_bias", False))

        model = cls(config, layer_types, conv_l_cache, conv_bias)
        model.load_weights(model_path)
        return model

    def weight_map(self) -> dict[str, str]:
        m: dict[str, str] = {}
        m["model.embed_tokens.weight"] = "embed_tokens.weight"
        m["model.embedding_norm.weight"] = "embedding_norm.weight"
        m["lm_head.weight"] = "lm_head.weight"  # ignored if tied

        for i, t in enumerate(self.layer_types):
            hf = f"model.layers.{i}"
            local = f"layers.{i}"
            m[f"{hf}.operator_norm.weight"] = f"{local}.operator_norm.weight"
            m[f"{hf}.ffn_norm.weight"] = f"{local}.ffn_norm.weight"
            for proj in ("w1", "w2", "w3"):
                m[f"{hf}.feed_forward.{proj}.weight"] = f"{local}.feed_forward.{proj}.weight"
            if t == "full_attention":
                for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    m[f"{hf}.self_attn.{proj}.weight"] = f"{local}.self_attn.{proj}.weight"
                m[f"{hf}.self_attn.q_layernorm.weight"] = f"{local}.self_attn.q_layernorm.weight"
                m[f"{hf}.self_attn.k_layernorm.weight"] = f"{local}.self_attn.k_layernorm.weight"
            else:
                for proj in ("in_proj", "out_proj"):
                    m[f"{hf}.conv.{proj}.weight"] = f"{local}.conv.{proj}.weight"
                m[f"{hf}.conv.conv.weight"] = f"{local}.conv.conv.weight"
                if self.conv_bias:
                    for proj in ("in_proj", "out_proj", "conv"):
                        m[f"{hf}.conv.{proj}.bias"] = f"{local}.conv.{proj}.bias"
        return m

    def load_weights(self, model_path: str) -> None:
        st_files = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No .safetensors in {model_path}")

        wmap = self.weight_map()
        loaded: set[str] = set()

        for st_file in st_files:
            sd = safetensors.torch.load_file(os.path.join(model_path, st_file))
            for hf_name, tensor in sd.items():
                if hf_name not in wmap:
                    continue
                local = wmap[hf_name]
                loaded.add(hf_name)
                tensor = tensor.to(MODEL_DTYPE)

                # Walk to the parent module.
                parts = local.split(".")
                target = self
                for p in parts[:-1]:
                    target = getattr(target, p)
                pname = parts[-1]
                param = getattr(target, pname)

                # Reshape rules:
                #   Embedding weight stays 2D.
                #   ANERMSNorm.weight stays 1D.
                #   Conv2d 1x1 linear: HF ships (out, in) → we want (out, in, 1, 1).
                #   Depthwise short conv: HF ships (out, 1, kernel) → (out, 1, 1, kernel).
                if param.dim() == 4:
                    if tensor.dim() == 2:
                        tensor = tensor.unsqueeze(-1).unsqueeze(-1)
                    elif tensor.dim() == 3:
                        tensor = tensor.unsqueeze(2)  # (out, 1, k) → (out, 1, 1, k)

                if param.shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {hf_name} → {local}: "
                        f"want {tuple(param.shape)}, got {tuple(tensor.shape)}"
                    )
                with torch.no_grad():
                    param.copy_(tensor)
            del sd
            gc.collect()

        # Tied embeddings: reuse embed_tokens for lm_head if HF didn't ship one.
        if self.config.tie_word_embeddings and "lm_head.weight" not in loaded:
            ew = self.embed_tokens.weight.data.to(MODEL_DTYPE)
            self.lm_head.weight.data = ew.unsqueeze(-1).unsqueeze(-1)
            print("Tied lm_head weights to embed_tokens")

        missing = set(wmap.keys()) - loaded
        if self.config.tie_word_embeddings:
            missing.discard("lm_head.weight")
        if missing:
            print(f"Warning: {len(missing)} weights missing — first 5: {sorted(missing)[:5]}")
        print(f"Loaded {len(loaded)} tensors from {len(st_files)} file(s)")
