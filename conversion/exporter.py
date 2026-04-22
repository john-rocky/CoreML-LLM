"""CoreML export pipeline: trace -> convert -> quantize -> save.

Converts an ANE-optimized PyTorch model into a monolithic CoreML .mlpackage
with stateful KV cache via Apple's MLState API (iOS 18+ / macOS 15+).

Monolithic approach: embed + transformer + lm_head in a single CoreML model.
This is simpler to trace and matches ANEMLL's proven pattern.

Inputs:  input_ids (1,1), position_ids (1,), causal_mask (1,1,1,ctx), current_pos (1,)
Outputs: token_id (1,), token_logit (1,)
State:   kv_cache_0 (2*num_layers, num_kv_heads, ctx, head_dim)
"""

from __future__ import annotations

import json
import os
import shutil

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ane_ops import (
    MODEL_DTYPE,
    ANERMSNorm,
    InModelArgmax,
    apply_rotary_pos_emb,
    rotate_half,
)
from base_model import ANETransformerModel, ModelConfig


class MonolithicWrapper(nn.Module):
    """Single traced wrapper: input_ids -> token_id + token_logit.

    All position-dependent operations use tensor inputs and CoreML-compatible
    ops (index_select, mask-based update) so the model traces correctly.
    """

    def __init__(self, model: ANETransformerModel) -> None:
        super().__init__()
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers
        self.norm = model.norm
        self.lm_head = model.lm_head
        self.argmax = model.argmax
        self.config = model.config

        # Register KV cache as buffer for CoreML state
        self.register_buffer("kv_cache_0", model.kv_cache_0.clone())

        # Precompute RoPE cos/sin for all positions: (max_len, head_dim)
        head_dim = model.config.head_dim
        base = model.config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        max_len = model.config.context_length * 2
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(MODEL_DTYPE))
        self.register_buffer("sin_cached", emb.sin().to(MODEL_DTYPE))

    def forward(
        self,
        input_ids: torch.Tensor,     # (1, 1) int32
        position_ids: torch.Tensor,  # (1,) int32 — position for RoPE
        causal_mask: torch.Tensor,   # (1, 1, 1, context_length) float16
        update_mask: torch.Tensor,   # (1, 1, context_length, 1) float16 — 1.0 at current_pos
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        n_rep = num_heads // num_kv_heads
        scale = 1.0 / (head_dim ** 0.5)

        # --- Embedding ---
        hidden_states = self.embed_tokens(input_ids).to(MODEL_DTYPE)  # (1, 1, hidden)

        # --- RoPE for current position via index_select ---
        cos = torch.index_select(self.cos_cached, 0, position_ids).view(1, 1, 1, head_dim)
        sin = torch.index_select(self.sin_cached, 0, position_ids).view(1, 1, 1, head_dim)

        # --- Transformer layers ---
        for layer_idx in range(num_layers):
            layer = self.layers[layer_idx]
            residual = hidden_states

            # Pre-norm
            hidden_states = layer.input_layernorm(hidden_states)

            # QKV projection via Conv2d
            x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
            q = layer.self_attn.q_proj(x).view(1, num_heads, head_dim, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
            k = layer.self_attn.k_proj(x).view(1, num_kv_heads, head_dim, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
            v = layer.self_attn.v_proj(x).view(1, num_kv_heads, head_dim, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)

            # Apply RoPE
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Update KV cache using mask-based write (CoreML-compatible)
            # update_mask: (1, 1, ctx, 1) — 1.0 at write position, 0.0 elsewhere
            # new_k: (1, kv_heads, 1, head_dim) -> broadcast to (1, kv_heads, ctx, head_dim)
            k_idx = layer_idx
            v_idx = num_layers + layer_idx

            # Read current cache slice
            K_cache = self.kv_cache_0[k_idx].unsqueeze(0)  # (1, kv_heads, ctx, dim)
            V_cache = self.kv_cache_0[v_idx].unsqueeze(0)

            # Mask-based update: cache = cache * (1 - mask) + new_val * mask
            k_broadcast = k.expand_as(K_cache)  # (1, kv_heads, ctx, head_dim)
            v_broadcast = v.expand_as(V_cache)
            K_new = K_cache * (1 - update_mask) + k_broadcast * update_mask
            V_new = V_cache * (1 - update_mask) + v_broadcast * update_mask

            # Write back to state
            self.kv_cache_0[k_idx] = K_new.squeeze(0)
            self.kv_cache_0[v_idx] = V_new.squeeze(0)

            # Expand KV for GQA using repeat_interleave (no shape unpacking)
            K_expanded = K_new.repeat_interleave(n_rep, dim=1)
            V_expanded = V_new.repeat_interleave(n_rep, dim=1)

            # Attention in fp32 for numerical stability
            q_f = q.to(torch.float32)
            k_f = K_expanded.to(torch.float32)
            attn_weights = torch.matmul(q_f, k_f.transpose(-1, -2)) * scale
            attn_weights = attn_weights + causal_mask.to(torch.float32)
            attn_weights = torch.softmax(attn_weights, dim=-1).to(MODEL_DTYPE)
            attn_output = torch.matmul(
                attn_weights.to(torch.float32), V_expanded.to(torch.float32)
            ).to(MODEL_DTYPE)

            # Output projection
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
            attn_output = layer.self_attn.o_proj(
                attn_output.permute(0, 2, 1).unsqueeze(2)
            ).squeeze(2).permute(0, 2, 1)

            hidden_states = residual + attn_output

            # Post-norm + MLP
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

        # --- Final norm ---
        hidden_states = self.norm(hidden_states)

        # --- LM Head + Argmax ---
        x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)  # (1, 1, vocab)
        token_id, token_logit = self.argmax(logits.squeeze(0))  # (1,), (1,)

        return token_id, token_logit


class CoreMLExporter:
    """Exports an ANE-optimized model to a CoreML .mlpackage file."""

    def __init__(self, model: ANETransformerModel) -> None:
        self.model = model
        self.config = model.config

    def export(
        self,
        output_dir: str,
        quantize: str | None = "int4",
        compute_units: str = "ALL",
        compute_precision: str | None = None,
    ) -> None:
        """Run the full export pipeline.

        compute_precision: None → coremltools default (fp16 for iOS 16+).
                           "fp32" → force fp32 activations (used by Gemma 3 to
                           keep its fp32 residual stream alive — fp16 lowering
                           collapses the casts and the model overflows).
        """
        os.makedirs(output_dir, exist_ok=True)

        print("=" * 60)
        print(f"Exporting model to {output_dir}")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Layers: {self.config.num_hidden_layers}")
        print(f"  Context length: {self.config.context_length}")
        print(f"  Quantization: {quantize or 'fp16'}")
        print(f"  Compute precision: {compute_precision or 'default (fp16)'}")
        print("=" * 60)

        self._export_monolithic(output_dir, quantize, compute_precision)
        self._write_config(output_dir, quantize, compute_units)

        print(f"\nExport complete! Files in {output_dir}")

    def _export_monolithic(self, output_dir: str, quantize: str | None,
                           compute_precision: str | None = None) -> None:
        """Export as a single monolithic CoreML model."""
        print("\nBuilding monolithic wrapper...")

        # Use architecture-specific wrapper if available
        cls_name = self.model.__class__.__name__
        if 'Gemma4' in cls_name:
            from models.gemma4_wrapper import Gemma4MonolithicWrapper
            wrapper = Gemma4MonolithicWrapper(self.model)
        elif 'Gemma3' in cls_name:
            from models.gemma3_wrapper import Gemma3MonolithicWrapper
            wrapper = Gemma3MonolithicWrapper(self.model)
        else:
            wrapper = MonolithicWrapper(self.model)
        wrapper.eval()

        ctx = self.config.context_length

        # Sample inputs for tracing
        sample_input_ids = torch.zeros((1, 1), dtype=torch.int32)
        sample_position_ids = torch.zeros((1,), dtype=torch.int32)
        sample_causal_mask = torch.zeros((1, 1, 1, ctx), dtype=torch.float16)
        # update_mask: 1.0 at current position, 0.0 elsewhere
        sample_update_mask = torch.zeros((1, 1, ctx, 1), dtype=torch.float16)
        sample_update_mask[0, 0, 0, 0] = 1.0

        print(f"  input_ids: {sample_input_ids.shape}")
        print(f"  position_ids: {sample_position_ids.shape}")
        print(f"  causal_mask: {sample_causal_mask.shape}")
        print(f"  update_mask: {sample_update_mask.shape}")

        # Reset KV cache before tracing
        with torch.no_grad():
            wrapper.kv_cache_0.zero_()

        print("Tracing model...")
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper,
                (sample_input_ids, sample_position_ids, sample_causal_mask, sample_update_mask),
            )

        # KV cache state shape - use the wrapper's actual buffer shape
        cache_shape = tuple(wrapper.kv_cache_0.shape)

        print("Converting to CoreML...")
        convert_kwargs = dict(
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
                ct.TensorType(name="position_ids", shape=(1,), dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=(1, 1, 1, ctx), dtype=np.float16),
                ct.TensorType(name="update_mask", shape=(1, 1, ctx, 1), dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="token_id", dtype=np.int32),
                ct.TensorType(name="token_logit", dtype=np.float16),
            ],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16),
                    name="kv_cache_0",
                ),
            ],
            compute_units=ct.ComputeUnit.ALL,
        )
        if compute_precision == "fp32":
            convert_kwargs["compute_precision"] = ct.precision.FLOAT32
        # Bump to iOS 26 target — its fp16 lowering path differs from iOS 18
        # and is what we validated EmbeddingGemma against.
        convert_kwargs["minimum_deployment_target"] = ct.target.iOS26
        mlmodel = ct.convert(traced, **convert_kwargs)

        if quantize:
            mlmodel = self._quantize_model(mlmodel, quantize)

        path = os.path.join(output_dir, "model.mlpackage")
        if os.path.exists(path):
            shutil.rmtree(path)
        mlmodel.save(path)
        size_mb = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(path)
            for f in fns
        ) / 1024 / 1024
        print(f"  Saved {path} ({size_mb:.1f} MB)")

    def _quantize_model(self, mlmodel, mode: str):
        """Apply quantization to a CoreML model."""
        if mode == "int4":
            op_config = ct.optimize.coreml.OpPalettizerConfig(
                nbits=4,
                granularity="per_grouped_channel",
                group_size=32,
            )
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config)
            print("  Applied int4 palettization (group_size=32)")

        elif mode == "int8":
            op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric",
                dtype="int8",
            )
            config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
            mlmodel = ct.optimize.coreml.linear_quantize_weights(mlmodel, config)
            print("  Applied int8 linear symmetric quantization")

        return mlmodel

    def _write_config(
        self,
        output_dir: str,
        quantize: str | None,
        compute_units: str,
    ) -> None:
        """Write model_config.json for the Swift inference engine."""
        config = {
            "model_name": "",
            "architecture": "",
            "hidden_size": self.config.hidden_size,
            "num_hidden_layers": self.config.num_hidden_layers,
            "num_attention_heads": self.config.num_attention_heads,
            "num_key_value_heads": self.config.num_key_value_heads,
            "head_dim": self.config.head_dim,
            "vocab_size": self.config.vocab_size,
            "context_length": self.config.context_length,
            "rms_norm_eps": self.config.rms_norm_eps,
            "bos_token_id": self.config.bos_token_id,
            "eos_token_id": self.config.eos_token_id,
            "quantization": quantize or "fp16",
            "compute_units": compute_units,
            "parts": {
                "model": "model.mlpackage",
            },
            "tokenizer_repo": "",
        }

        path = os.path.join(output_dir, "model_config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved {path}")
