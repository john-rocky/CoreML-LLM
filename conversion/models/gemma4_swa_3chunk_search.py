"""Topology-I 3-chunk split for Gemma 4: chunk1 absorbs chunk2 (L0-14),
then shared chunks run unchanged.

Layout (E2B):
    BigChunk1     L0-14   15 layers own-KV + PLE + emits kv13/kv14
    SWAChunk3     L15-24  10 layers KV-shared  (reused from gemma4_swa_chunks)
    SWAChunk4     L25-34  10 layers KV-shared + norm + lm_head + argmax (reused)

Under the 17-layer ANE compile threshold, three candidate topologies for
E2B are valid (producer L13 sliding / L14 full):

    Topology I   (this file): (0,15) + (15,25) + (25,35)        — c1 absorbs c2
    Topology II  (shipped):   (0,8)  + (8,25)  + (25,35)        — c2 absorbs c3
    Topology V   (future):    (0,15) + (15,30) + (30,35)        — middle shared fatter

Rationale: the shipped Topology II merges chunk2+chunk3 into a 17-layer
block.  Whether the ANE is faster per-layer at 15 layers (BigChunk1) than
at 17 layers (merged chunk2) is the open question this module addresses.
Per-layer cost on iPhone 17 Pro for the Topology II 17-layer chunk was
13.0 ms / 17 = 0.76 ms/layer.  If 15 layers stays under 11.4 ms, the
residual savings versus Topology II's chunk1 (5.8 ms, 8 layers = 0.73
ms/layer) are chunk-boundary overhead only — the sum c1+c2 should drop
from the current 5.8 + 13.0 = 18.8 ms to roughly 15 × 0.75 = 11.25 ms
in a best case, giving back ~7 ms/step and ~38 tok/s.  Compile and
real-ANE-layout constraints will likely shrink that, but the probe is
the only way to find out.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE

from .gemma4 import Gemma4Model
from .gemma4_swa_chunks import _run_layer_swa, _layer_kv_map, SWAChunk1


class BigChunk1(nn.Module):
    """Merged chunk covering all own-KV layers (L0..kv_full_producer) plus
    PLE computation plus kv13/kv14 emission.

    For E2B: L0-14 = 15 layers (13 sliding + 2 full, includes producers
    L13/L14).  Output contract matches SWAChunk1+SWAChunk2 combined:

        hidden_states_out
        K_sliding_out       (num_sliding=13, 1, W, max_hd)    — L0-13 raw
        V_sliding_out
        K_full_out          (num_full=2,    1, ctx, max_hd)   — L4, L14 raw
        V_full_out
        per_layer_combined_out
        kv13_k, kv13_v      (1, 1, W,   hd_s)   — L13 alias, what shared layers read
        kv14_k, kv14_v      (1, 1, ctx, hd_f)   — L14 alias
    """

    def __init__(self, model: Gemma4Model, start: int = 0, end: int = 15):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
        self.sliding_map, self.full_map = _layer_kv_map(start, end, model.config)
        self.num_sliding = len(self.sliding_map)
        self.num_full = len(self.full_map)
        # PLE computation reused verbatim from SWAChunk1
        self.per_layer_model_projection = model.per_layer_model_projection
        self.per_layer_projection_norm = model.per_layer_projection_norm
        self.per_layer_model_projection_scale = model.per_layer_model_projection_scale
        self.per_layer_input_scale = model.per_layer_input_scale
        self.per_layer_dim = model.config.hidden_size_per_layer_input
        self.num_layers_total = model.config.num_hidden_layers

    def _compute_ple(self, hidden_states, per_layer_raw):
        # Identical to SWAChunk1._compute_ple (not shared via import because
        # the attribute access on `self` matters — we own the same modules).
        import torch.nn.functional as F
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
        proj_normed = (normed * norm_w).view(1, 1,
                                             self.num_layers_total * self.per_layer_dim)

        return (proj_normed + per_layer_raw) * self.per_layer_input_scale

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_raw, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        per_layer_combined = self._compute_ple(hidden_states, per_layer_raw)

        # Layer-local kv13/kv14 stores; populated when producer layers run.
        kv13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)

        K_sliding_outs = []
        V_sliding_outs = []
        K_full_outs = []
        V_full_outs = []

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
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
                self.layers[local_idx], layer_idx, hidden_states,
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

        K_sliding_out = torch.stack(K_sliding_outs, dim=0)
        V_sliding_out = torch.stack(V_sliding_outs, dim=0)
        K_full_out = torch.stack(K_full_outs, dim=0)
        V_full_out = torch.stack(V_full_outs, dim=0)

        return (hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out,
                per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v)
