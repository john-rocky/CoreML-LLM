"""Merged chunk2+chunk3 (L8-24): keeps kv14 internal to eliminate CPU↔ANE roundtrip.

Standard: chunk2 outputs kv14 → CPU → chunk3 input (32MB I/O overhead)
Merged: kv14 computed and consumed within same model → zero external I/O for kv14

This chunk has 17 layers (5 sliding + 2 full own-KV + 10 KV-shared).
The shared layers read kv13/kv14 directly from L13/L14's computation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, ane_softmax

from .gemma4 import Gemma4Model
from .gemma4_swa_chunks import _run_layer_swa, _layer_kv_map


class MergedChunk23(nn.Module):
    """Layers 8-24: chunk2 (L8-14) + chunk3 (L15-24) merged.

    Own KV: L8-14 (5 sliding + 2 full). Shared KV: L15-24 (all shared from L13/L14).
    kv13/kv14 stay internal — never leave the ANE.

    Outputs: hidden_states, K/V for L8-14, BUT NOT kv13/kv14 (internal).
    chunk4 still needs kv14 → output it for chunk4.
    """
    START_C2, END_C2 = 8, 15  # chunk2 layers
    START_C3, END_C3 = 15, 25  # chunk3 layers

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        # All layers 8-24
        self.layers_c2 = nn.ModuleList([model.layers[i] for i in range(self.START_C2, self.END_C2)])
        self.layers_c3 = nn.ModuleList([model.layers[i] for i in range(self.START_C3, self.END_C3)])
        self.sliding_map, self.full_map = _layer_kv_map(self.START_C2, self.END_C2, model.config)
        self.num_sliding = len(self.sliding_map)
        self.num_full = len(self.full_map)

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                K_sliding_in, V_sliding_in, K_full_in, V_full_in):
        config = self.config
        kv13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        # --- chunk2 part (L8-14): own KV, produces kv13/kv14 ---
        K_sliding_outs = []
        V_sliding_outs = []
        K_full_outs = []
        V_full_outs = []

        for local_idx in range(self.END_C2 - self.START_C2):
            layer_idx = self.START_C2 + local_idx
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
                self.layers_c2[local_idx], layer_idx, hidden_states,
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

        # --- chunk3 part (L15-24): all KV-shared, uses kv13/kv14 INTERNALLY ---
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K

        for local_idx in range(self.END_C3 - self.START_C3):
            layer_idx = self.START_C3 + local_idx
            hidden_states, *_ = _run_layer_swa(
                self.layers_c3[local_idx], layer_idx, hidden_states,
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

        # Output kv14 for chunk4 (still needs it)
        return (hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out,
                kv13_k, kv13_v, kv14_k, kv14_v)
