"""Qwen3 model implementation for CoreML-LLM.

Supports Qwen3ForCausalLM architecture (Qwen3-1.7B / 4B / 8B, Bonsai-1.7B family).

Architecture specifics (vs Qwen2):
- No attention bias (attention_bias: False)
- Per-head RMSNorm on Q and K before RoPE (QK-norm)
- Otherwise identical: GQA, SwiGLU MLP, RoPE, RMSNorm, tied word embeddings
"""

from __future__ import annotations

import gc
import json
import os

import safetensors.torch
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ane_ops import MODEL_DTYPE
from base_model import ANETransformerModel, ModelConfig


class Qwen3Model(ANETransformerModel):
    """Qwen3 model with ANE-optimized layers + QK-norm."""

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        context_length: int = 2048,
    ) -> Qwen3Model:
        """Load a Qwen3 model from a HuggingFace model directory."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        config_dict["context_length"] = context_length
        config_dict["state_length"] = context_length
        config_dict["has_qk_norm"] = True
        # Qwen3 config has attention_bias=false; be explicit in case we get an older dump
        config_dict["attention_bias"] = bool(config_dict.get("attention_bias", False))
        config = ModelConfig.from_dict(config_dict)

        model = cls(config)
        model.load_weights(model_path)
        return model

    def weight_map(self) -> dict[str, str]:
        """Map HuggingFace weight names to local parameter names.

        HF format: model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
                   model.layers.{i}.self_attn.{q,k}_norm.weight      (Qwen3-specific)
        Local:     layers.{i}.self_attn.{q,k,v,o}_proj.weight
                   layers.{i}.self_attn.{q,k}_norm.weight
        """
        mapping: dict[str, str] = {}

        mapping["model.embed_tokens.weight"] = "embed_tokens.weight"
        mapping["model.norm.weight"] = "norm.weight"
        mapping["lm_head.weight"] = "lm_head.weight"

        for i in range(self.config.num_hidden_layers):
            prefix_hf = f"model.layers.{i}"
            prefix_local = f"layers.{i}"

            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                mapping[f"{prefix_hf}.self_attn.{proj}.weight"] = (
                    f"{prefix_local}.self_attn.{proj}.weight"
                )

            # Qwen3-specific: per-head RMSNorm on Q and K (applied before RoPE)
            mapping[f"{prefix_hf}.self_attn.q_norm.weight"] = (
                f"{prefix_local}.self_attn.q_norm.weight"
            )
            mapping[f"{prefix_hf}.self_attn.k_norm.weight"] = (
                f"{prefix_local}.self_attn.k_norm.weight"
            )

            for proj in ["gate_proj", "up_proj", "down_proj"]:
                mapping[f"{prefix_hf}.mlp.{proj}.weight"] = (
                    f"{prefix_local}.mlp.{proj}.weight"
                )

            mapping[f"{prefix_hf}.input_layernorm.weight"] = (
                f"{prefix_local}.input_layernorm.weight"
            )
            mapping[f"{prefix_hf}.post_attention_layernorm.weight"] = (
                f"{prefix_local}.post_attention_layernorm.weight"
            )

        return mapping

    def load_weights(self, model_path: str) -> None:
        """Load HuggingFace weights, reshaping Linear -> Conv2d for projections."""
        st_files = sorted(
            f for f in os.listdir(model_path) if f.endswith(".safetensors")
        )
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        wmap = self.weight_map()
        loaded_keys: set[str] = set()

        # Names that must be reshaped (out, in) -> (out, in, 1, 1) to feed Conv2d.
        # Everything else (embed_tokens, RMSNorm weights, q_norm, k_norm) stays 1D or 2D.
        conv2d_suffixes = (
            ".q_proj.weight",
            ".k_proj.weight",
            ".v_proj.weight",
            ".o_proj.weight",
            ".gate_proj.weight",
            ".up_proj.weight",
            ".down_proj.weight",
        )

        for st_file in st_files:
            filepath = os.path.join(model_path, st_file)
            state_dict = safetensors.torch.load_file(filepath)

            for hf_name, tensor in state_dict.items():
                if hf_name not in wmap:
                    continue

                local_name = wmap[hf_name]
                loaded_keys.add(hf_name)

                parts = local_name.split(".")
                target = self
                for p in parts[:-1]:
                    target = getattr(target, p)
                param_name = parts[-1]

                tensor = tensor.to(MODEL_DTYPE)

                if local_name == "lm_head.weight":
                    if tensor.dim() == 2:
                        tensor = tensor.unsqueeze(-1).unsqueeze(-1)
                elif any(local_name.endswith(suf) for suf in conv2d_suffixes):
                    if tensor.dim() == 2:
                        tensor = tensor.unsqueeze(-1).unsqueeze(-1)

                param = getattr(target, param_name)
                if param.shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {hf_name} -> {local_name}: "
                        f"expected {param.shape}, got {tensor.shape}"
                    )

                with torch.no_grad():
                    param.copy_(tensor)

            del state_dict
            gc.collect()

        if self.config.tie_word_embeddings and "lm_head.weight" not in loaded_keys:
            embed_weight = self.embed_tokens.weight.data.to(MODEL_DTYPE)
            self.lm_head.weight.data = embed_weight.unsqueeze(-1).unsqueeze(-1)
            print("Tied lm_head weights to embed_tokens")

        missing = set(wmap.keys()) - loaded_keys
        if self.config.tie_word_embeddings:
            missing.discard("lm_head.weight")
        if missing:
            print(f"Warning: {len(missing)} weights not found: {sorted(missing)[:5]}...")

        print(f"Loaded {len(loaded_keys)} weight tensors from {len(st_files)} file(s)")
