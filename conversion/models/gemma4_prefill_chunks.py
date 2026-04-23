"""Gemma 4 Prefill chunks: process N tokens in batch (single call).

For initial prompt processing, much faster than per-token decode.
- N = PREFILL_N (module-level, swapped per variant by build scripts)
- Single prefill call: handles prompts up to N tokens
- Output: K/V for the N positions, written into persistent cache by Swift
- Then decode model (SWA) takes over for token-by-token generation

Architecture differences from decode:
- hidden_states: (1, N, hidden) instead of (1, 1, hidden)
- Q/K/V: (1, num_heads, N, hd) instead of (1, num_heads, 1, hd)
- attn: (1, num_heads, N, N) — within-batch only (no existing cache)
- Output K/V per layer: (1, num_kv_heads, N, hd) — to be written by Swift

Chunk boundaries and KV producer indices are derived from the Gemma4 config
via `compute_chunk_boundaries` / `config.kv_{sliding,full}_producer` —
drop-in for both E2B (35 layers, L13/L14 producers) and E4B (42 layers,
L22/L23 producers).

Assumes pos_offset = 0 (first prefill call only). Multi-call prefill is
a future enhancement.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, ane_softmax

from .gemma4 import Gemma4Model
from .gemma4_swa_chunks import compute_chunk_boundaries

PREFILL_N = 1024


def chunk_kv_layout(start: int, end: int, config) -> tuple[list[int], int | None, int | None]:
    """Classify layers in [start, end) into (non_producer_layers, sliding_prod, full_prod).

    - non_producer_layers: absolute indices for which the chunk emits K{local}/V{local}
      (local index = position within non_producer_layers)
    - sliding_prod / full_prod: absolute index if the producer falls in this chunk,
      else None. If present the chunk emits kv13_k/kv13_v (sliding) and/or
      kv14_k/kv14_v (full) opaque-aliased outputs.

    KV-shared layers are filtered out: they don't emit their own K/V.
    """
    non_prod: list[int] = []
    sliding_prod = None
    full_prod = None
    for i in range(start, end):
        if config.is_kv_shared(i):
            continue
        if i == config.kv_sliding_producer:
            sliding_prod = i
        elif i == config.kv_full_producer:
            full_prod = i
        else:
            non_prod.append(i)
    return non_prod, sliding_prod, full_prod


def chunk_output_names(chunk_idx: int, start: int, end: int, config) -> list[str]:
    """Model output names for prefill_chunk{chunk_idx}, in declaration order.

    chunk1: hidden_states_out, per_layer_combined_out, then K{i}/V{i} for each
            non-producer own-KV layer.
    chunk2: hidden_states_out, then K{i}/V{i} for non-producers, then
            kv13_k/v and/or kv14_k/v where producers fall in the range.
    chunk3: hidden_states_out only.
    chunk4: token_id, token_logit.
    """
    if chunk_idx == 3:
        return ["hidden_states_out"]
    if chunk_idx == 4:
        return ["token_id", "token_logit"]

    non_prod, sliding_prod, full_prod = chunk_kv_layout(start, end, config)
    names: list[str] = ["hidden_states_out"]
    if chunk_idx == 1:
        names.append("per_layer_combined_out")
    for local, _ in enumerate(non_prod):
        names.extend([f"K{local}", f"V{local}"])
    if sliding_prod is not None:
        names.extend(["kv13_k", "kv13_v"])
    if full_prod is not None:
        names.extend(["kv14_k", "kv14_v"])
    return names


def v_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean_sq = x.pow(2).mean(-1, keepdim=True) + eps
    return x * torch.rsqrt(mean_sq)


def _run_layer_prefill(
    layer, layer_idx, hidden_states,  # (1, N, hidden)
    cos_s, sin_s, cos_f, sin_f,  # (1, 1, N, dim) — pre-computed for N positions
    causal_mask,  # (1, 1, N, N) — within-batch causal
    config, per_layer_combined,  # (1, N, total_pld)
    kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,  # for sharing
):
    """Run one layer for batched prefill input.

    Returns hidden_states (1, N, hidden), K_new (1, num_kv_heads, N, hd),
    V_new (1, num_kv_heads, N, hd), and updated kv stores.
    """
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    is_full = config.is_full_attention(layer_idx)
    hd = config.get_head_dim(layer_idx)
    is_kv_shared = config.is_kv_shared(layer_idx)
    N = PREFILL_N

    residual = hidden_states
    h = layer.input_layernorm(hidden_states)  # (1, N, hidden)
    # Conv2d input format: (1, hidden, 1, N)
    x = h.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

    # Q projection: (1, hidden, 1, N) → (1, num_heads*hd, 1, N)
    q_raw = layer.self_attn["q_proj"](x)
    # Reshape to per-head: (1, num_heads, hd, N) → permute to (1, num_heads, N, hd)
    q = q_raw.view(1, num_heads, hd, N).permute(0, 1, 3, 2).to(MODEL_DTYPE)
    # q_norm per token (normalize over hd)
    # q is (1, num_heads, N, hd) — view as (N, num_heads, hd) for per-token norm... or just apply
    # Actually q_norm expects (1, num_heads, hd) input. We have (1, num_heads, N, hd).
    # Apply per-token: reshape to (N, num_heads, hd), norm, reshape back.
    q = q.permute(0, 2, 1, 3).contiguous().view(N, num_heads, hd)
    q = layer.self_attn["q_norm"](q)  # (N, num_heads, hd)
    q = q.view(1, N, num_heads, hd).permute(0, 2, 1, 3)  # (1, num_heads, N, hd)

    # Apply RoPE — cos/sin shape is (1, 1, N, dim). Need to broadcast to (1, num_heads, N, dim).
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    K_new = None
    V_new = None

    if not is_kv_shared:
        k_raw = layer.self_attn["k_proj"](x)  # (1, num_kv_heads*hd, 1, N)
        v_raw = layer.self_attn["v_proj"](x)
        k = k_raw.view(1, num_kv_heads, hd, N).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = v_raw.view(1, num_kv_heads, hd, N).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        # k_norm per token
        k = k.permute(0, 2, 1, 3).contiguous().view(N, num_kv_heads, hd)
        k = layer.self_attn["k_norm"](k)
        k = k.view(1, N, num_kv_heads, hd).permute(0, 2, 1, 3)
        v = v_norm(v)

        if is_full:
            _, k = apply_rotary_pos_emb(k, k, cos_f, sin_f)
        else:
            _, k = apply_rotary_pos_emb(k, k, cos_s, sin_s)

        K_new = k  # (1, num_kv_heads, N, hd)
        V_new = v
        K_for_attn = k
        V_for_attn = v

        # Store producer-layer KVs for sharing (read by shared layers in
        # chunks 3/4). Output names stay kv13/kv14 for backward compatibility
        # — they are opaque aliases (E2B: L13/L14, E4B: L22/L23).
        if layer_idx == config.kv_sliding_producer:
            kv_store_13_k = k
            kv_store_13_v = v
        elif layer_idx == config.kv_full_producer:
            kv_store_14_k = k
            kv_store_14_v = v
    else:
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    # GQA expand
    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)  # (1, num_heads, N, hd)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)

    # Manual attention with scale=1.0 (Gemma 4 uses pre-normalized Q/K).
    # SDPA fusion tested with both attn_mask and is_causal approaches but
    # coremltools' SDPA decomposition produces subtly different results from
    # manual attention in CoreML (not a mask issue — precision difference in
    # the decomposed op sequence). Keeping manual attention for correctness.
    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
    attn_weights = attn_weights + causal_mask
    attn_weights = ane_softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V_expanded)

    # Back to (1, hidden, 1, N) format for o_proj
    # (1, num_heads, N, hd) → (1, N, num_heads, hd) → (1, N, num_heads*hd) → (1, num_heads*hd, 1, N)
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, N, num_heads * hd)
    attn_output = attn_output.permute(0, 2, 1).unsqueeze(2)  # (1, num_heads*hd, 1, N)
    attn_output = layer.self_attn["o_proj"](attn_output)  # (1, hidden, 1, N)
    # Back to (1, N, hidden) for residual add
    attn_output = attn_output.squeeze(2).permute(0, 2, 1)
    attn_output = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + attn_output

    # MLP
    residual = hidden_states
    h = layer.pre_feedforward_layernorm(hidden_states)
    x_mlp = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # (1, hidden, 1, N)
    gate = layer.mlp["gate_proj"](x_mlp)
    up = layer.mlp["up_proj"](x_mlp)
    gate = F.gelu(gate, approximate="tanh")
    mlp_out = layer.mlp["down_proj"](gate * up)  # (1, hidden, 1, N)
    hidden_states_mlp = mlp_out.squeeze(2).permute(0, 2, 1)  # (1, N, hidden)
    hidden_states_mlp = layer.post_feedforward_layernorm(hidden_states_mlp)
    hidden_states = residual + hidden_states_mlp

    # Per-layer input
    residual_pl = hidden_states
    s = layer_idx * config.hidden_size_per_layer_input
    e = s + config.hidden_size_per_layer_input
    per_layer_slice = per_layer_combined[:, :, s:e]  # (1, N, pld)
    # Conv2d layout
    hs_conv = hidden_states.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # (1, hidden, 1, N)
    gated = layer.per_layer_input_gate(hs_conv)  # (1, pld, 1, N)
    gated = F.gelu(gated, approximate="tanh")
    per_layer_slice_conv = per_layer_slice.permute(0, 2, 1).unsqueeze(2)  # (1, pld, 1, N)
    gated = gated * per_layer_slice_conv
    gated = layer.per_layer_projection(gated)  # (1, hidden, 1, N)
    gated = gated.squeeze(2).permute(0, 2, 1)  # (1, N, hidden)
    hidden_states = layer.post_per_layer_input_norm(gated)
    hidden_states = residual_pl + hidden_states
    hidden_states = hidden_states * layer.layer_scalar.to(MODEL_DTYPE)

    return hidden_states, K_new, V_new, kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v


def _process_layers_prefill(layers, start, end, hidden_states, causal_mask, per_layer_combined,
                              cos_s, sin_s, cos_f, sin_f, config,
                              kv13_k, kv13_v, kv14_k, kv14_v):
    """Run a range of layers in prefill mode. Returns hidden_states + K/V outputs + kv stores."""
    K_outs = []
    V_outs = []
    for local_idx, layer in enumerate(layers):
        layer_idx = start + local_idx
        (hidden_states, K_new, V_new,
         kv13_k, kv13_v, kv14_k, kv14_v) = _run_layer_prefill(
            layer, layer_idx, hidden_states,
            cos_s, sin_s, cos_f, sin_f, causal_mask,
            config, per_layer_combined,
            kv13_k, kv13_v, kv14_k, kv14_v,
        )
        K_outs.append(K_new)
        V_outs.append(V_new)
    return hidden_states, K_outs, V_outs, kv13_k, kv13_v, kv14_k, kv14_v


class PrefillChunk1(nn.Module):
    """First prefill chunk. Computes PLE inside, emits K/V per own-KV layer.

    Boundaries are config-driven (see compute_chunk_boundaries):
      E2B: L0-7  (7 sliding + 1 full)  → K0..K7/V0..V7
      E4B: L0-11 (10 sliding + 2 full) → K0..K11/V0..V11
    """

    def __init__(self, model: Gemma4Model, start: int = 0, end: int = 8):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
        self.per_layer_model_projection = model.per_layer_model_projection
        self.per_layer_projection_norm = model.per_layer_projection_norm
        self.per_layer_model_projection_scale = model.per_layer_model_projection_scale
        self.per_layer_input_scale = model.per_layer_input_scale
        self.per_layer_dim = model.config.hidden_size_per_layer_input
        self.num_layers_total = model.config.num_hidden_layers
        # Producer layers live in chunk2 for both E2B and E4B, but guard
        # anyway so a future variant with producers in chunk1 is handled.
        self.non_prod, self.sliding_prod, self.full_prod = chunk_kv_layout(
            start, end, model.config)

    def _compute_ple_batch(self, hidden_states, per_layer_raw):
        """Compute per_layer_combined for batched input (1, N, hidden).
        Original loop version (one ANERMSNorm per layer slice)."""
        h_conv = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        proj = self.per_layer_model_projection(h_conv) * self.per_layer_model_projection_scale
        proj = proj.squeeze(2).permute(0, 2, 1)  # (1, N, total_pld)
        normed_slices = []
        for li in range(self.num_layers_total):
            s = li * self.per_layer_dim
            e = s + self.per_layer_dim
            normed_slices.append(self.per_layer_projection_norm(proj[:, :, s:e]))
        proj_normed = torch.cat(normed_slices, dim=-1)
        return (proj_normed + per_layer_raw) * self.per_layer_input_scale

    def forward(self, hidden_states, causal_mask, per_layer_raw,
                cos_s, sin_s, cos_f, sin_f):
        """
        Inputs:
          hidden_states: (1, N, hidden)
          causal_mask: (1, 1, N, N) — within-batch causal
          per_layer_raw: (1, N, total_pld) — per-token raw embedding
          cos_s, sin_s, cos_f, sin_f: (1, 1, N, dim) — RoPE for N positions
        Outputs:
          hidden_states_out, per_layer_combined_out, then K{i}/V{i} pairs
          for each own-KV non-producer layer in [start, end). Variable-width
          tuple — the build script's `chunk_output_names(1, start, end, cfg)`
          enumerates the names in matching order.
        """
        config = self.config
        per_layer_combined = self._compute_ple_batch(hidden_states, per_layer_raw)

        dummy_13_k = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_13_v = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_14_k = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        dummy_14_v = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)

        non_prod_K: dict[int, torch.Tensor] = {}
        non_prod_V: dict[int, torch.Tensor] = {}
        sliding_prod_k = sliding_prod_v = None
        full_prod_k = full_prod_v = None

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, K_new, V_new,
             dummy_13_k, dummy_13_v, dummy_14_k, dummy_14_v) = _run_layer_prefill(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f, causal_mask,
                config, per_layer_combined,
                dummy_13_k, dummy_13_v, dummy_14_k, dummy_14_v,
            )
            if config.is_kv_shared(layer_idx):
                continue
            if layer_idx == config.kv_sliding_producer:
                sliding_prod_k, sliding_prod_v = K_new, V_new
            elif layer_idx == config.kv_full_producer:
                full_prod_k, full_prod_v = K_new, V_new
            else:
                non_prod_K[layer_idx] = K_new
                non_prod_V[layer_idx] = V_new

        outputs: list[torch.Tensor] = [hidden_states, per_layer_combined]
        for abs_idx in self.non_prod:
            outputs.append(non_prod_K[abs_idx])
            outputs.append(non_prod_V[abs_idx])
        if self.sliding_prod is not None:
            outputs.extend([sliding_prod_k, sliding_prod_v])
        if self.full_prod is not None:
            outputs.extend([full_prod_k, full_prod_v])
        return tuple(outputs)


class PrefillChunk2(nn.Module):
    """Second prefill chunk. Emits K/V for own-KV non-producer layers, plus
    kv13/kv14 for the sliding/full producers (consumed by chunks 3, 4).

    Boundaries are config-driven:
      E2B: L8-14  (L13/L14 producers) → K0..K4 + kv13_* + kv14_*
      E4B: L12-23 (L22/L23 producers) → K0..K9 + kv13_* + kv14_*
    """

    def __init__(self, model: Gemma4Model, start: int = 8, end: int = 15):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
        self.non_prod, self.sliding_prod, self.full_prod = chunk_kv_layout(
            start, end, model.config)

    def forward(self, hidden_states, causal_mask, per_layer_combined,
                cos_s, sin_s, cos_f, sin_f):
        config = self.config
        kv13_k = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)

        non_prod_K: dict[int, torch.Tensor] = {}
        non_prod_V: dict[int, torch.Tensor] = {}

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, K_new, V_new,
             kv13_k, kv13_v, kv14_k, kv14_v) = _run_layer_prefill(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f, causal_mask,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )
            if config.is_kv_shared(layer_idx):
                continue
            if layer_idx == config.kv_sliding_producer:
                continue  # captured via kv13_* below
            if layer_idx == config.kv_full_producer:
                continue  # captured via kv14_* below
            non_prod_K[layer_idx] = K_new
            non_prod_V[layer_idx] = V_new

        outputs: list[torch.Tensor] = [hidden_states]
        for abs_idx in self.non_prod:
            outputs.append(non_prod_K[abs_idx])
            outputs.append(non_prod_V[abs_idx])
        if self.sliding_prod is not None:
            outputs.extend([kv13_k, kv13_v])
        if self.full_prod is not None:
            outputs.extend([kv14_k, kv14_v])
        return tuple(outputs)


class PrefillChunk3(nn.Module):
    """Third prefill chunk. All layers KV-shared; reads kv13/kv14.

      E2B: L15-24 (10 shared). E4B: L24-32 (9 shared).
    """

    def __init__(self, model: Gemma4Model, start: int = 15, end: int = 25):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])

    def forward(self, hidden_states, causal_mask, per_layer_combined,
                cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        hidden_states, _, _, _, _, _, _ = _process_layers_prefill(
            self.layers, self.start, self.end, hidden_states, causal_mask, per_layer_combined,
            cos_s, sin_s, cos_f, sin_f, self.config,
            kv13_k, kv13_v, kv14_k, kv14_v,
        )
        return hidden_states


class PrefillChunk4(nn.Module):
    """Final prefill chunk + norm + lm_head. Outputs LAST token's id.

      E2B: L25-34 (10 shared). E4B: L33-41 (9 shared).
    """

    def __init__(self, model: Gemma4Model, start: int = 25, end: int = 35):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
        self.norm = model.norm
        self.lm_head = nn.Conv2d(model.lm_head.in_channels, model.lm_head.out_channels,
                                  kernel_size=1, bias=False)
        self.lm_head.weight.data = model.lm_head.weight.data.clone()
        self.argmax = model.argmax
        self.softcap = model.softcap

    def forward(self, hidden_states, causal_mask, per_layer_combined,
                cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v,
                last_position_mask):  # (1, N, 1) — 1 at real last position, 0 elsewhere
        hidden_states, _, _, _, _, _, _ = _process_layers_prefill(
            self.layers, self.start, self.end, hidden_states, causal_mask, per_layer_combined,
            cos_s, sin_s, cos_f, sin_f, self.config,
            kv13_k, kv13_v, kv14_k, kv14_v,
        )
        hidden_states = self.norm(hidden_states)  # (1, N, hidden)
        # Select the real last token's hidden state via masked sum (ANE-friendly)
        last = (hidden_states * last_position_mask).sum(dim=1, keepdim=True)  # (1, 1, hidden)
        x = last.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
        return token_id, token_logit
