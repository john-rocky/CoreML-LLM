"""Gemma 4 SWA chunks — stateful (MLState + slice_update) variant.

Forks `gemma4_swa_chunks.py` to use coremltools `StateType` for KV
instead of full-sequence in/out tensors. Two state buffers per
producing chunk (chunk1 + chunk2):

  kv_cache_sliding  (2*num_sliding, HKV, W, max_hd)   — ring write @ pos%W
  kv_cache_full     (2*num_full,    HKV, ctx, max_hd) — linear write @ pos

Sliding cache is a ring: write happens at slot `ring_pos = pos % W`,
which Swift precomputes and passes as `ring_pos` (int32). RoPE bakes
position into K at write time, so the order of slots in the ring does
not matter for correctness — the mask just declares "first (pos+1)
slots valid for pos<W, all W slots valid for pos>=W" (LEFT-aligned;
the recurrent shift build was right-aligned, so the Swift mask
builder needs the matching update).

Chunks 3 and 4 are STATELESS — they only read `kv13/kv14` (the
producer aliases emitted by chunk2) and own no KV slots.

T=1 only in this Phase 1 file. Multifunction prefill_bN is a follow-up
(mirrors the Qwen3-VL Phase 1 → v1.5.0 progression).

`update_mask` input is REMOVED — the full-layer mask-based write
trick was an ANE-compat workaround for ios17 that ios18.slice_update
makes unnecessary. One fewer input on every step.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, ane_softmax

from .gemma4 import Gemma4Model
from .gemma4_swa_chunks import (
    compute_chunk_boundaries, _layer_kv_map, v_norm,
)


# ============================================================
# Linear-vs-Conv2d projection helpers (cml9 PR #2577 follow-up)
# ============================================================
#
# Default pipeline uses Conv2d(1×1) wrappers around all projections
# because pre-cml9 the activation-quant path only matched conv ops. cml9
# extended `linear_quantize_activations` to native `linear` op (PR #2577),
# so we can drop the wrapper. MBA 5-layer + W4 PoC (commit 72d30b3) showed
# nn.Linear stays 100% on ANE post-W4 but Mac latency was +21% vs the
# wrapper path — likely a Mac-ANE scheduler quirk on linear+constexpr_lut
# fusion. iPhone re-measurement gates the production migration.
#
# `--linear-projections` build flag walks each chunk's layers and swaps
# every Conv2d(in, out, 1, ...) for a shape-equivalent nn.Linear(in, out),
# weights reshaped from (out, in, 1, 1) to (out, in). The forward path
# uses `_project` to dispatch on type so the same function works for
# both — the Linear branch skips the permute/unsqueeze that ANE has to
# undo at compile time.

def _replace_conv2d_with_linear(module: nn.Module) -> nn.Module:
    """Return an nn.Linear equivalent to a Conv2d(in, out, 1, ...) module.
    Pass-through for any non-Conv2d input.
    """
    if not isinstance(module, nn.Conv2d):
        return module
    in_ch = module.in_channels
    out_ch = module.out_channels
    has_bias = module.bias is not None
    lin = nn.Linear(in_ch, out_ch, bias=has_bias, dtype=module.weight.dtype)
    with torch.no_grad():
        lin.weight.copy_(module.weight.data.squeeze(-1).squeeze(-1))
        if has_bias:
            lin.bias.copy_(module.bias.data)
    return lin


def _swap_chunk_projections_to_linear(layers, *, also_swap_per_layer_model=None):
    """In-place: walk each Gemma4DecoderLayer in `layers` and swap every
    Conv2d-1×1 projection (q/k/v/o + gate/up/down + per_layer_input_gate
    + per_layer_projection) for a Linear with reshaped weights.

    Pass `also_swap_per_layer_model=model.per_layer_model_projection_holder`
    to also swap the model-level projection used by chunk1's PLE compute.
    The argument is the holder (a 1-tuple lambda or list); we mutate
    `holder[0]` because `model.per_layer_model_projection` rebinding from
    here doesn't propagate to the caller's reference.
    """
    for layer in layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if hasattr(layer, "self_attn") and name in layer.self_attn:
                layer.self_attn[name] = _replace_conv2d_with_linear(
                    layer.self_attn[name])
        for name in ("gate_proj", "up_proj", "down_proj"):
            if hasattr(layer, "mlp") and name in layer.mlp:
                layer.mlp[name] = _replace_conv2d_with_linear(layer.mlp[name])
        for name in ("per_layer_input_gate", "per_layer_projection"):
            if hasattr(layer, name):
                setattr(layer, name,
                        _replace_conv2d_with_linear(getattr(layer, name)))


def _project(layer_proj: nn.Module, x_3d: torch.Tensor) -> torch.Tensor:
    """Run a (B, S, in)-shaped tensor through a Conv2d(1×1) or Linear
    projection and return (B, S, out). Conv2d path adds the permute/
    unsqueeze wrap; Linear path is a direct call. Trace specializes on
    the isinstance branch so the active variant ends up in the MIL graph.
    """
    if isinstance(layer_proj, nn.Linear):
        return layer_proj(x_3d)
    # Conv2d 1×1 path: (B, S, in) → (B, in, 1, S) → conv → (B, S, out)
    x_4d = x_3d.permute(0, 2, 1).unsqueeze(2)
    out = layer_proj(x_4d)
    return out.squeeze(2).permute(0, 2, 1)


def _run_layer_swa_stateful(
    layer, layer_idx, hidden_states,
    cos_s, sin_s, cos_f, sin_f,
    causal_mask_full, causal_mask_sliding,
    current_pos,        # int32 (1,) — write index for full layers
    ring_pos,           # int32 (1,) — current_pos % W, sliding write index
    kv_sliding_state,   # (2*num_sliding, HKV, W, max_hd) — chunk-owned MLState
    kv_full_state,      # (2*num_full,    HKV, ctx, max_hd)
    sliding_map, full_map,
    config, per_layer_combined,
    kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
):
    """One layer of stateful SWA decode. Mirrors `_run_layer_swa`
    semantically, but writes K/V via slice_update into MLState rather
    than returning new K/V tensors. Returns (hidden, alias_kv...)
    only — KV mutation is in-place on the state buffer.
    """
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    max_hd = config.global_head_dim
    is_full = config.is_full_attention(layer_idx)
    hd = config.get_head_dim(layer_idx)
    is_kv_shared = config.is_kv_shared(layer_idx)

    residual = hidden_states
    h = layer.input_layernorm(hidden_states).to(MODEL_DTYPE)  # (1, 1, hidden)

    # Q (via _project helper — Conv2d/Linear branch specialized at trace time)
    q = _project(layer.self_attn["q_proj"], h).view(
        1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
    q = layer.self_attn["q_norm"](q.reshape(1, num_heads, hd)).view(1, num_heads, 1, hd)
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    if not is_kv_shared:
        # Compute K/V for the new token
        k = _project(layer.self_attn["k_proj"], h).view(
            1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = _project(layer.self_attn["v_proj"], h).view(
            1, num_kv_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        k = layer.self_attn["k_norm"](k.reshape(1, num_kv_heads, hd)).view(1, num_kv_heads, 1, hd)
        v = v_norm(v)
        if is_full:
            _, k = apply_rotary_pos_emb(k, k, cos_f, sin_f)
        else:
            _, k = apply_rotary_pos_emb(k, k, cos_s, sin_s)

        if hd < max_hd:
            k_padded = F.pad(k, (0, max_hd - hd))
            v_padded = F.pad(v, (0, max_hd - hd))
        else:
            k_padded, v_padded = k, v
        # k_padded/v_padded shape: (1, HKV, 1, max_hd)

        if is_full:
            fi = full_map[layer_idx]
            # Linear write @ current_pos. State shape (2*num_full, HKV, ctx, max_hd):
            #   slot 2*fi   = K, slot 2*fi+1 = V (interleaved like Qwen3-VL).
            kv_full_state[2*fi:2*fi+1, :, current_pos:current_pos+1, :] = k_padded
            kv_full_state[2*fi+1:2*fi+2, :, current_pos:current_pos+1, :] = v_padded
            K_full_slice = kv_full_state[2*fi:2*fi+1, :, :, :hd]   # (1, HKV, ctx, hd)
            V_full_slice = kv_full_state[2*fi+1:2*fi+2, :, :, :hd]
            K_for_attn = K_full_slice
            V_for_attn = V_full_slice
        else:
            si = sliding_map[layer_idx]
            # Ring write @ ring_pos. State shape (2*num_sliding, HKV, W, max_hd).
            kv_sliding_state[2*si:2*si+1, :, ring_pos:ring_pos+1, :] = k_padded
            kv_sliding_state[2*si+1:2*si+2, :, ring_pos:ring_pos+1, :] = v_padded
            K_sliding_slice = kv_sliding_state[2*si:2*si+1, :, :, :hd]   # (1, HKV, W, hd)
            V_sliding_slice = kv_sliding_state[2*si+1:2*si+2, :, :, :hd]
            K_for_attn = K_sliding_slice
            V_for_attn = V_sliding_slice

        # Producer alias outputs — same kv13/kv14 naming as the recurrent
        # build so chunks 3/4 see no input-name change. These are slice
        # views over the producer's just-updated state buffer.
        if layer_idx == config.kv_sliding_producer:
            kv_store_13_k = K_for_attn[..., :config.head_dim]
            kv_store_13_v = V_for_attn[..., :config.head_dim]
        elif layer_idx == config.kv_full_producer:
            kv_store_14_k = K_for_attn[..., :config.global_head_dim]
            kv_store_14_v = V_for_attn[..., :config.global_head_dim]
    else:
        # Shared layer: read producer KV from the alias inputs.
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    # GQA expansion + manual attention (matches recurrent path's scale=1.0)
    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)
    mask = causal_mask_full if is_full else causal_mask_sliding
    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
    attn_weights = attn_weights + mask
    attn_weights = ane_softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V_expanded)

    # attn_output: (1, num_heads, 1, hd) → (1, 1, num_heads*hd)
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
    attn_output = _project(layer.self_attn["o_proj"], attn_output)
    attn_output = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + attn_output

    # MLP
    residual = hidden_states
    h = layer.pre_feedforward_layernorm(hidden_states).to(MODEL_DTYPE)
    gate = _project(layer.mlp["gate_proj"], h)
    up = _project(layer.mlp["up_proj"], h)
    gate = F.gelu(gate, approximate="tanh")
    mlp_out = _project(layer.mlp["down_proj"], gate * up)
    hidden_states = layer.post_feedforward_layernorm(mlp_out)
    hidden_states = residual + hidden_states

    # Per-layer input
    residual_pl = hidden_states
    s = layer_idx * config.hidden_size_per_layer_input
    e = s + config.hidden_size_per_layer_input
    per_layer_slice = per_layer_combined[:, :, s:e]
    h_pl = hidden_states.to(MODEL_DTYPE)
    gated = _project(layer.per_layer_input_gate, h_pl)
    gated = F.gelu(gated, approximate="tanh")
    gated = gated * per_layer_slice
    gated = _project(layer.per_layer_projection, gated)
    hidden_states = layer.post_per_layer_input_norm(gated)
    hidden_states = residual_pl + hidden_states
    hidden_states = hidden_states * layer.layer_scalar.to(MODEL_DTYPE)

    return (hidden_states,
            kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v)


# ============================================================
# Chunk modules (stateful)
# ============================================================


class _StatefulChunkBase(nn.Module):
    """Common: sliding/full slot maps, KV state buffers."""

    def __init__(self, model: Gemma4Model, start: int, end: int, ctx: int):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
        self.sliding_map, self.full_map = _layer_kv_map(start, end, model.config)
        self.num_sliding = len(self.sliding_map)
        self.num_full = len(self.full_map)
        self.ctx = ctx
        self.W = model.config.sliding_window

        max_hd = model.config.global_head_dim
        HKV = model.config.num_key_value_heads
        # K/V interleaved: even slots = K, odd = V (matches Qwen3-VL stateful layout).
        # Allocate at least size-1 even when this chunk has zero
        # sliding/full layers, so coremltools doesn't choke on a
        # zero-sized state. Empty buffers just go unused.
        ns = max(self.num_sliding, 1)
        nf = max(self.num_full, 1)
        self.register_buffer(
            "kv_cache_sliding",
            torch.zeros(2 * ns, HKV, self.W, max_hd, dtype=MODEL_DTYPE),
        )
        self.register_buffer(
            "kv_cache_full",
            torch.zeros(2 * nf, HKV, ctx, max_hd, dtype=MODEL_DTYPE),
        )


class SWAStatefulChunk1(_StatefulChunkBase):
    """First decode chunk. Owns KV state. Computes per_layer_combined
    from raw and emits it for chunks 2-4 to consume.

    For E2B (L0-7): 7 sliding + 1 full.
    For E4B (L0-11): 10 sliding + 2 full.
    """

    def __init__(self, model: Gemma4Model, start: int = 0, end: int = 8, ctx: int = 2048,
                 use_linear: bool = False):
        super().__init__(model, start, end, ctx)
        self.per_layer_model_projection = model.per_layer_model_projection
        self.per_layer_projection_norm = model.per_layer_projection_norm
        self.per_layer_model_projection_scale = model.per_layer_model_projection_scale
        self.per_layer_input_scale = model.per_layer_input_scale
        self.per_layer_dim = model.config.hidden_size_per_layer_input
        self.num_layers_total = model.config.num_hidden_layers
        if use_linear:
            _swap_chunk_projections_to_linear(self.layers)
            self.per_layer_model_projection = _replace_conv2d_with_linear(
                self.per_layer_model_projection)

    def _compute_ple(self, hidden_states, per_layer_raw):
        """Reuses the cat-trick RMSNorm pattern from the recurrent build
        (one layer_norm over (1, num_layers, 256) instead of 35 separate
        norms + 34 concats). Identical to gemma4_swa_chunks.SWAChunk1."""
        h = hidden_states.to(MODEL_DTYPE)
        proj = _project(self.per_layer_model_projection, h) \
               * self.per_layer_model_projection_scale
        proj_grouped = proj.view(1, self.num_layers_total, self.per_layer_dim)

        norm_w = self.per_layer_projection_norm.weight
        eps = float(self.per_layer_projection_norm.eps)
        doubled = torch.cat([proj_grouped, -proj_grouped], dim=-1)
        normed = F.layer_norm(doubled, normalized_shape=(2 * self.per_layer_dim,),
                              weight=None, bias=None, eps=eps)
        normed, _ = torch.chunk(normed, 2, dim=-1)
        proj_normed = (normed * norm_w).view(
            1, 1, self.num_layers_total * self.per_layer_dim)

        return (proj_normed + per_layer_raw) * self.per_layer_input_scale

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_raw, cos_s, sin_s, cos_f, sin_f,
                current_pos, ring_pos):
        config = self.config
        per_layer_combined = self._compute_ple(hidden_states, per_layer_raw)

        # Producer alias placeholders — only chunk2 actually emits these,
        # but _run_layer_swa_stateful's signature wants them. Constants
        # are fine here because no layer in chunk1 is the producer.
        dummy_13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        dummy_13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        dummy_14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        dummy_14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, dummy_13_k, dummy_13_v, dummy_14_k, dummy_14_v
             ) = _run_layer_swa_stateful(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                current_pos, ring_pos,
                self.kv_cache_sliding, self.kv_cache_full,
                self.sliding_map, self.full_map,
                config, per_layer_combined,
                dummy_13_k, dummy_13_v, dummy_14_k, dummy_14_v,
            )
        return hidden_states, per_layer_combined


class SWAStatefulChunk2(_StatefulChunkBase):
    """Second decode chunk. Owns KV state. Ends at `kv_full_producer+1`
    so it emits the sliding (kv13_*) and full (kv14_*) producer alias
    outputs that chunks 3/4 read.

    For E2B (L8-14): 5 sliding + 2 full. For E4B (L12-23): 10 sliding + 2 full.
    """

    def __init__(self, model: Gemma4Model, start: int = 8, end: int = 15, ctx: int = 2048,
                 use_linear: bool = False):
        super().__init__(model, start, end, ctx)
        if use_linear:
            _swap_chunk_projections_to_linear(self.layers)

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                current_pos, ring_pos):
        config = self.config
        kv13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                current_pos, ring_pos,
                self.kv_cache_sliding, self.kv_cache_full,
                self.sliding_map, self.full_map,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )

        return hidden_states, kv13_k, kv13_v, kv14_k, kv14_v


class SWAStatefulChunk3(nn.Module):
    """Third decode chunk. Stateless. All layers are KV-shared and read
    kv13 (W-sized) / kv14 (ctx-sized) from inputs.

    For E2B: L15-24 (10 shared). For E4B: L24-32 (9 shared).
    """

    def __init__(self, model: Gemma4Model, start: int = 15, end: int = 25,
                 use_linear: bool = False):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
        if use_linear:
            _swap_chunk_projections_to_linear(self.layers)

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        # Shared layers don't write KV — pass dummies for state args and
        # zero current_pos/ring_pos (unused on the is_kv_shared path).
        zero_idx = torch.zeros(1, dtype=torch.int32)
        dummy_state = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                zero_idx, zero_idx,
                dummy_state, dummy_state,
                {}, {},
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )
        return hidden_states


class SWAStatefulChunk4(nn.Module):
    """Final decode chunk. Stateless. KV-shared + final norm + lm_head + argmax.
    For E2B: L25-34 (10 shared). For E4B: L33-41 (9 shared).
    """

    def __init__(self, model: Gemma4Model, start: int = 25, end: int = 35,
                 use_linear: bool = False):
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
        if use_linear:
            _swap_chunk_projections_to_linear(self.layers)
            self.lm_head = _replace_conv2d_with_linear(self.lm_head)

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        zero_idx = torch.zeros(1, dtype=torch.int32)
        dummy_state = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                zero_idx, zero_idx,
                dummy_state, dummy_state,
                {}, {},
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )

        normed = self.norm(hidden_states)
        logits = _project(self.lm_head, normed.to(MODEL_DTYPE))
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
        return token_id, token_logit, normed
