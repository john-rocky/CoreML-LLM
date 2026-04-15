"""Gemma 4 SWA 2-chunk decode: merged (L0-14) + merged (L15-34 + LM head).

Motivation
----------
Per `docs/BASELINE_SPEED_AUDIT.md`, the 4-chunk decode wall-clock is dominated
by the serial chain c1 -> c2 -> c3 -> c4. Each chunk round-trip has an ANE
dispatch overhead measured at ~2.3 ms on iPhone 17 Pro. Halving the number
of chunks roughly halves that overhead: ~9 ms -> ~4.6 ms per step.

Risk
----
A merged chunk has more layers (15 or 20). The ANE compiler is known to
hit stability ceilings around ~15 layers per chunk (see docs/EXPERIMENTS.md
"Merged chunk2+chunk3" prototype note). If that happens on iPhone 17 Pro
the chunk falls off ANE onto CPU / GPU, which regresses throughput.
`ComputePlanAudit` is used to detect silent fallback before shipping.

Layout
------
  merged_chunk1 (L0-14): 12 sliding + 3 full, owns KV, produces kv13/kv14
  merged_chunk2 (L15-34): 20 layers all KV-shared, + norm + lm_head + argmax

The merged KV cache input/output surface mirrors the sum of the original
SWAChunk1 + SWAChunk2 surfaces (12 sliding slots + 3 full slots). Swift
keeps one pair of caches per merged chunk instead of two.

Unused in verify/prefill paths — those keep the 4-chunk layout because
(a) they are run rarely, (b) their shapes diverge from decode, and (c)
the ANE-dispatch saving there is a smaller fraction of the per-call cost.
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


class MergedChunk12(nn.Module):
    """Layers 0-14 fused. Owns KV for L0-14 and produces kv13/kv14
    for the shared-KV half. Also computes PLE internally from per_layer_raw,
    identical to the original SWAChunk1 PLE path.

    Input KV layout (same as SWAChunk1 + SWAChunk2 stacked):
      K_sliding_in / V_sliding_in: (num_sliding, 1, W, max_hd)  — 12 slots
      K_full_in    / V_full_in   : (num_full,    1, ctx, max_hd) — 3 slots

    The per-layer slot maps (sliding_map / full_map) are computed over the
    full 0..14 range, matching the input tensor ordering.
    """
    START, END = 0, 15

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        self.sliding_map, self.full_map = _layer_kv_map(self.START, self.END, model.config)
        self.num_sliding = len(self.sliding_map)  # 12 (7 from c1 + 5 from c2)
        self.num_full = len(self.full_map)        # 3  (1 from c1 + 2 from c2)

        # PLE computation modules (same as SWAChunk1)
        self.per_layer_model_projection = model.per_layer_model_projection
        self.per_layer_projection_norm = model.per_layer_projection_norm
        self.per_layer_model_projection_scale = model.per_layer_model_projection_scale
        self.per_layer_input_scale = model.per_layer_input_scale
        self.per_layer_dim = model.config.hidden_size_per_layer_input
        self.num_layers_total = model.config.num_hidden_layers

    def _compute_ple(self, hidden_states, per_layer_raw):
        """Decode-step PLE (Q=1). Byte-identical to SWAChunk1._compute_ple."""
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

        # kv13/kv14 are produced at L13/L14 within this chunk and passed
        # to the downstream merged chunk.
        kv13_k = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)

        K_sliding_outs = []
        V_sliding_outs = []
        K_full_outs = []
        V_full_outs = []

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
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

        # per_layer_combined is still output for downstream reuse (same wire
        # format as 4-chunk pipeline's chunk1 -> chunk2 handoff).
        return (hidden_states, K_sliding_out, V_sliding_out, K_full_out, V_full_out,
                kv13_k, kv13_v, kv14_k, kv14_v, per_layer_combined)


class MergedChunk34(nn.Module):
    """Layers 15-34 fused + final norm + lm_head + argmax.

    All 20 layers are KV-shared, so no cache writes — only reads of the
    incoming kv13 (W-sized sliding) and kv14 (ctx-sized full).
    """
    START, END = 15, 35

    def __init__(self, model: Gemma4Model):
        super().__init__()
        self.config = model.config
        self.layers = nn.ModuleList([model.layers[i] for i in range(self.START, self.END)])
        self.norm = model.norm
        self.lm_head = nn.Conv2d(model.lm_head.in_channels, model.lm_head.out_channels,
                                  kernel_size=1, bias=False)
        self.lm_head.weight.data = model.lm_head.weight.data.clone()
        self.argmax = model.argmax
        self.softcap = model.softcap

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding, update_mask,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        dummy_K = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_V = dummy_K

        for local_idx in range(self.END - self.START):
            layer_idx = self.START + local_idx
            hidden_states, *_ = _run_layer_swa(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding, update_mask,
                dummy_K, dummy_V, dummy_K, dummy_V,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )

        normed = self.norm(hidden_states)
        x = normed.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
        # Output normed hidden for drafter carry state — same as SWAChunk4.
        return token_id, token_logit, normed
