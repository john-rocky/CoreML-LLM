"""Qwen2/2.5 model implementation for CoreML-LLM.

Supports Qwen2ForCausalLM architecture (Qwen2.5-0.5B, 1.5B, 3B, 7B, etc.).

Architecture specifics:
- GQA: 14 query heads, 2 KV heads (for 0.5B), head_dim=64
- Attention bias: True (unlike LLaMA)
- RoPE theta: 1,000,000
- SiLU activation with gate/up/down projections
- Tied word embeddings (lm_head = embed_tokens)
- RMSNorm epsilon: 1e-6
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

from ane_ops import MODEL_DTYPE
from base_model import ANETransformerModel, ModelConfig


class Qwen2Model(ANETransformerModel):
    """Qwen2/2.5 model with ANE-optimized layers."""

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        context_length: int = 2048,
    ) -> Qwen2Model:
        """Load a Qwen2 model from a HuggingFace model directory.

        Args:
            model_path: Path to directory with config.json and *.safetensors
            context_length: Maximum context length for KV cache
        """
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)

        config_dict["context_length"] = context_length
        config_dict["state_length"] = context_length
        config = ModelConfig.from_dict(config_dict)

        model = cls(config)
        model.load_weights(model_path)
        return model

    def weight_map(self) -> dict[str, str]:
        """Map HuggingFace weight names to local parameter names.

        HF format: model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}
        Local:      layers.{i}.self_attn.{q,k,v,o}_proj.conv.{weight,bias}
        """
        mapping = {}

        # Embeddings
        mapping["model.embed_tokens.weight"] = "embed_tokens.weight"

        # Final norm
        mapping["model.norm.weight"] = "norm.weight"

        # LM head (may be tied to embeddings)
        mapping["lm_head.weight"] = "lm_head.conv.weight"

        # Per-layer mappings
        for i in range(self.config.num_hidden_layers):
            prefix_hf = f"model.layers.{i}"
            prefix_local = f"layers.{i}"

            # Attention projections
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                mapping[f"{prefix_hf}.self_attn.{proj}.weight"] = (
                    f"{prefix_local}.self_attn.{proj}.weight"
                )
                if self.config.attention_bias and proj != "o_proj":
                    mapping[f"{prefix_hf}.self_attn.{proj}.bias"] = (
                        f"{prefix_local}.self_attn.{proj}.bias"
                    )

            # MLP projections
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                mapping[f"{prefix_hf}.mlp.{proj}.weight"] = (
                    f"{prefix_local}.mlp.{proj}.weight"
                )

            # Layer norms
            mapping[f"{prefix_hf}.input_layernorm.weight"] = (
                f"{prefix_local}.input_layernorm.weight"
            )
            mapping[f"{prefix_hf}.post_attention_layernorm.weight"] = (
                f"{prefix_local}.post_attention_layernorm.weight"
            )

        return mapping

    def load_weights(self, model_path: str) -> None:
        """Load HuggingFace weights, reshaping Linear -> Conv2d."""
        # Find all safetensors files
        st_files = sorted(
            f
            for f in os.listdir(model_path)
            if f.endswith(".safetensors")
        )

        if not st_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {model_path}"
            )

        wmap = self.weight_map()
        loaded_keys = set()

        for st_file in st_files:
            filepath = os.path.join(model_path, st_file)
            state_dict = safetensors.torch.load_file(filepath)

            for hf_name, tensor in state_dict.items():
                if hf_name not in wmap:
                    continue

                local_name = wmap[hf_name]
                loaded_keys.add(hf_name)

                # Get target parameter
                parts = local_name.split(".")
                target = self
                for p in parts[:-1]:
                    target = getattr(target, p)
                param_name = parts[-1]

                tensor = tensor.to(MODEL_DTYPE)

                # Reshape for Conv2d: (out, in) -> (out, in, 1, 1)
                if local_name.endswith(".conv.weight") or (
                    "proj.weight" in local_name
                    and hasattr(target, "weight")
                    and target.weight.dim() == 4
                ):
                    if tensor.dim() == 2:
                        tensor = tensor.unsqueeze(-1).unsqueeze(-1)

                param = getattr(target, param_name)
                if param.shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {hf_name}: "
                        f"expected {param.shape}, got {tensor.shape}"
                    )

                with torch.no_grad():
                    param.copy_(tensor)

            del state_dict
            gc.collect()

        # Handle tied embeddings
        if self.config.tie_word_embeddings and "lm_head.weight" not in loaded_keys:
            embed_weight = self.embed_tokens.weight.data.to(MODEL_DTYPE)
            self.lm_head.weight.data = embed_weight.unsqueeze(-1).unsqueeze(-1)
            print("Tied lm_head weights to embed_tokens")

        missing = set(wmap.keys()) - loaded_keys
        if self.config.tie_word_embeddings:
            missing.discard("lm_head.weight")
        if missing:
            print(f"Warning: {len(missing)} weights not found: {list(missing)[:5]}...")

        print(f"Loaded {len(loaded_keys)} weight tensors from {len(st_files)} file(s)")
