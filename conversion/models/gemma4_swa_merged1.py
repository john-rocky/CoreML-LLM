"""Gemma 4 SWA 1-chunk decode: all 35 layers fused into one ANE dispatch.

This is the most aggressive dispatch-overhead reduction possible: a single
round-trip per decode step. If it stays on ANE the theoretical ceiling is
~3 tok/s above the 2-chunk variant at 2K (one fewer 2.3 ms dispatch),
plus whatever the ANE scheduler earns from planning a single graph.

Risk / known gotcha
-------------------
The ANE compiler has a historical stability ceiling around ~15 layers per
function (docs/EXPERIMENTS.md). 35 layers is well past that. Expected
failure modes, in order of likelihood:

  1. Compile succeeds, placement silently spills to CPU/GPU for a subset
     of layers. Detect with ComputePlanAudit.
  2. Compile OOMs on-device. Visible at load time.
  3. Compile succeeds but latency regresses vs 2-chunk due to worse
     scheduling or weight residency eviction.

Ship only if ComputePlanAudit reports 0% non-ANE ops AND tok/s beats
the 2-chunk variant on a fresh device boot (cold cache).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE

from .gemma4 import Gemma4Model
from .gemma4_swa_chunks import _run_layer_swa, _layer_kv_map


class MergedChunk1(nn.Module):
    """All 35 layers + PLE + final norm + lm_head + argmax in one graph.

    Inputs mirror the old 4-chunk pipeline's concatenated KV surface:
      K_sliding_in / V_sliding_in: (12, 1, W, max_hd)
      K_full_in    / V_full_in   : (3,  1, ctx, max_hd)

    kv13 / kv14 are produced by L13 / L14 and consumed by L15-34 WITHOUT
    leaving the graph, so the Swift runtime never materialises them. This
    is the same trick as MergedChunk12 but extended across the full model.
    """
    START, END = 0, 35

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        # Only L0-14 have an owned-KV slot; L15-34 are shared.
        self.sliding_map, self.full_map = _layer_kv_map(0, 15, model.config)
        self.num_sliding = len(self.sliding_map)  # 12
        self.num_full = len(self.full_map)        # 3

        # PLE modules
        self.per_layer_model_projection = model.per_layer_model_projection
        self.per_layer_projection_norm = model.per_layer_projection_norm
        self.per_layer_model_projection_scale = model.per_layer_model_projection_scale
        self.per_layer_input_scale = model.per_layer_input_scale
        self.per_layer_dim = model.config.hidden_size_per_layer_input
        self.num_layers_total = model.config.num_hidden_layers

        # Final norm + LM head
        self.norm = model.norm
        self.lm_head = nn.Conv2d(model.lm_head.in_channels, model.lm_head.out_channels,
                                  kernel_size=1, bias=False)
        self.lm_head.weight.data = model.lm_head.weight.data.clone()
        self.argmax = model.argmax
        self.softcap = model.softcap

    def _compute_ple(self, hidden_states, per_layer_raw):
        h_conv = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        proj = self.per_layer_model_projection(h_conv) * self.per_layer_model_projection_scale
        proj = proj.squeeze(2).permute(0, 2, 1)
        proj_grouped = proj.view(1, self.num_layers_total, self.per_layer_dim)

        norm_w = self.per_layer_projection_norm.weight
        eps = float(self.per_layer_projection_norm.eps)
        doubled = torch.cat([proj_grouped, -proj_grouped], dim=-1)
        normed = F.layer_norm(doubled, normalized_shape=(2 * self.per_layer_dim,),
                              weight=None, bias=None, eps=eps)
        normed, _ = torch.chunk(normed, 2, dim=-1)
        proj_normed = (normed * norm_w).view(1, 1, self.num_layers_total * self.per_layer_dim)
        return (proj_normed + per_layer_raw) * self.per_layer_input_scale

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_raw, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        per_layer_combined = self._compute_ple(hidden_states, per_layer_raw)

        kv13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        K_sliding_outs = []
        V_sliding_outs = []
        K_full_outs = []
        V_full_outs = []

        # --- L0-14: own KV ---
        for layer_idx in range(0, 15):
            is_full = config.is_full_attention(layer_idx)
            if is_full:
                fi = self.full_map[layer_idx]
                K_full_slot = K_full_in[fi].unsqueeze(0)
                V_full_slot = V_full_in[fi].unsqueeze(0)
                K_sliding_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
                V_sliding_slot = K_sliding_slot
            else:
                si = self.sliding_map[layer_idx]
                K_sliding_slot = K_sliding_in[si].unsqueeze(0)
                V_sliding_slot = V_sliding_in[si].unsqueeze(0)
                K_full_slot = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
                V_full_slot = K_full_slot

            (hidden_states, Kso, Vso, Kfo, Vfo,
             kv13_k, kv13_v, kv14_k, kv14_v) = _run_layer_swa(
                self.layers[layer_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                K_sliding_slot, V_sliding_slot, K_full_slot, V_full_slot,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )
            if is_full:
                K_full_outs.append(Kfo.squeeze(0))
                V_full_outs.append(Vfo.squeeze(0))
            else:
                K_sliding_outs.append(Kso.squeeze(0))
                V_sliding_outs.append(Vso.squeeze(0))

        # --- L15-34: all shared, kv13/kv14 stay internal ---
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K
        for layer_idx in range(15, 35):
            hidden_states, *_ = _run_layer_swa(
                self.layers[layer_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                dummy_K, dummy_V, dummy_K, dummy_V,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )

        K_sliding_out = torch.stack(K_sliding_outs, dim=0)
        V_sliding_out = torch.stack(V_sliding_outs, dim=0)
        K_full_out = torch.stack(K_full_outs, dim=0)
        V_full_out = torch.stack(V_full_outs, dim=0)

        normed = self.norm(hidden_states)
        x = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))

        return (token_id, token_logit, normed,
                K_sliding_out, V_sliding_out, K_full_out, V_full_out)
