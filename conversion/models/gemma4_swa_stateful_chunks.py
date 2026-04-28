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


# ============================================================
# T=N prefill variants — Phase 2b multifunction (`prefill_bN`)
# ============================================================
#
# Mirrors the Qwen3-VL stateful T=1 → T=N progression. Each prefill
# chunk takes T tokens at once, writes T contiguous KV slots via
# slice_update, and returns hidden states for all T positions (chunk4
# argmaxes the LAST row only). Engine-side dispatch guarantees
# `current_pos + T <= ctx` and `ring_pos + T <= W` (no mid-batch wrap)
# so the slice_update remains a contiguous slice. The tail of any
# prompt where these constraints fail goes through the T=1 path.


def _run_layer_swa_stateful_prefill(
    layer, layer_idx, hidden_states,
    cos_s, sin_s, cos_f, sin_f,
    causal_mask_full, causal_mask_sliding,
    current_pos, ring_pos,
    kv_sliding_state, kv_full_state,
    sliding_map, full_map,
    config, per_layer_combined,
    kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
    T,
):
    """T=N variant of _run_layer_swa_stateful. Shapes:
      hidden_states:       (1, T, hidden)
      cos_*/sin_*:         (1, 1, T, hd_*)
      causal_mask_full:    (1, 1, T, ctx)
      causal_mask_sliding: (1, 1, T, W)
      per_layer_combined:  (1, T, num_layers*pld)
      kv_*_state:          unchanged from T=1 build; written T slots at
                            current_pos:current_pos+T (full) or
                            ring_pos:ring_pos+T (sliding).
    """
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    max_hd = config.global_head_dim
    is_full = config.is_full_attention(layer_idx)
    hd = config.get_head_dim(layer_idx)
    is_kv_shared = config.is_kv_shared(layer_idx)

    residual = hidden_states
    h = layer.input_layernorm(hidden_states).to(MODEL_DTYPE)  # (1, T, hidden)

    # Q: (1, T, num_heads*hd) → (1, num_heads, T, hd)
    q = _project(layer.self_attn["q_proj"], h).view(
        1, T, num_heads, hd).permute(0, 2, 1, 3).to(MODEL_DTYPE)
    q = layer.self_attn["q_norm"](q.reshape(1, num_heads, T, hd))
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    if not is_kv_shared:
        k = _project(layer.self_attn["k_proj"], h).view(
            1, T, num_kv_heads, hd).permute(0, 2, 1, 3).to(MODEL_DTYPE)
        v = _project(layer.self_attn["v_proj"], h).view(
            1, T, num_kv_heads, hd).permute(0, 2, 1, 3).to(MODEL_DTYPE)
        k = layer.self_attn["k_norm"](k.reshape(1, num_kv_heads, T, hd))
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
        # k_padded/v_padded shape: (1, HKV, T, max_hd)

        if is_full:
            fi = full_map[layer_idx]
            kv_full_state[2*fi:2*fi+1, :, current_pos:current_pos+T, :] = k_padded
            kv_full_state[2*fi+1:2*fi+2, :, current_pos:current_pos+T, :] = v_padded
            K_full_slice = kv_full_state[2*fi:2*fi+1, :, :, :hd]
            V_full_slice = kv_full_state[2*fi+1:2*fi+2, :, :, :hd]
            K_for_attn = K_full_slice
            V_for_attn = V_full_slice
        else:
            si = sliding_map[layer_idx]
            kv_sliding_state[2*si:2*si+1, :, ring_pos:ring_pos+T, :] = k_padded
            kv_sliding_state[2*si+1:2*si+2, :, ring_pos:ring_pos+T, :] = v_padded
            K_sliding_slice = kv_sliding_state[2*si:2*si+1, :, :, :hd]
            V_sliding_slice = kv_sliding_state[2*si+1:2*si+2, :, :, :hd]
            K_for_attn = K_sliding_slice
            V_for_attn = V_sliding_slice

        if layer_idx == config.kv_sliding_producer:
            kv_store_13_k = K_for_attn[..., :config.head_dim]
            kv_store_13_v = V_for_attn[..., :config.head_dim]
        elif layer_idx == config.kv_full_producer:
            kv_store_14_k = K_for_attn[..., :config.global_head_dim]
            kv_store_14_v = V_for_attn[..., :config.global_head_dim]
    else:
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    # GQA expansion + manual attention. q (1, H, T, hd), K (1, HKV, S, hd)
    # → expand HKV to H, attn (1, H, T, S). Mask shape (1, 1, T, S)
    # broadcasts across heads.
    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)
    mask = causal_mask_full if is_full else causal_mask_sliding
    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
    attn_weights = attn_weights + mask
    attn_weights = ane_softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V_expanded)

    # (1, num_heads, T, hd) → (1, T, num_heads*hd)
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, T, -1)
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

    # Per-layer input — slice along the trailing (num_layers*pld) dim,
    # which lives on dim=-1 for both T=1 (1, 1, *) and T>1 (1, T, *).
    residual_pl = hidden_states
    s = layer_idx * config.hidden_size_per_layer_input
    e = s + config.hidden_size_per_layer_input
    per_layer_slice = per_layer_combined[:, :, s:e]   # (1, T, pld)
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


class SWAStatefulChunk1Prefill(SWAStatefulChunk1):
    """T=N prefill variant of SWAStatefulChunk1. Shares the parent's
    layers, KV state buffers, and PLE projection — coremltools merges
    by state name when packaged as a multifunction mlpackage."""

    def __init__(self, model, start=0, end=8, ctx=2048,
                 use_linear=False, T: int = 8):
        super().__init__(model, start, end, ctx, use_linear=use_linear)
        self.T = T

    def _compute_ple_prefill(self, hidden_states, per_layer_raw):
        """T-aware reshape of the cat-trick RMSNorm. hidden_states is
        (1, T, hidden); output is (1, T, num_layers*pld)."""
        T = self.T
        h = hidden_states.to(MODEL_DTYPE)
        proj = _project(self.per_layer_model_projection, h) \
               * self.per_layer_model_projection_scale
        # (1, T, num_layers*pld) → (1, T, num_layers, pld)
        proj_grouped = proj.view(1, T, self.num_layers_total, self.per_layer_dim)

        norm_w = self.per_layer_projection_norm.weight
        eps = float(self.per_layer_projection_norm.eps)
        doubled = torch.cat([proj_grouped, -proj_grouped], dim=-1)
        normed = F.layer_norm(doubled, normalized_shape=(2 * self.per_layer_dim,),
                              weight=None, bias=None, eps=eps)
        normed, _ = torch.chunk(normed, 2, dim=-1)
        proj_normed = (normed * norm_w).view(
            1, T, self.num_layers_total * self.per_layer_dim)

        return (proj_normed + per_layer_raw) * self.per_layer_input_scale

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_raw, cos_s, sin_s, cos_f, sin_f,
                current_pos, ring_pos):
        config = self.config
        T = self.T
        per_layer_combined = self._compute_ple_prefill(hidden_states, per_layer_raw)

        dummy_13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        dummy_13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        dummy_14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        dummy_14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, dummy_13_k, dummy_13_v, dummy_14_k, dummy_14_v
             ) = _run_layer_swa_stateful_prefill(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                current_pos, ring_pos,
                self.kv_cache_sliding, self.kv_cache_full,
                self.sliding_map, self.full_map,
                config, per_layer_combined,
                dummy_13_k, dummy_13_v, dummy_14_k, dummy_14_v,
                T,
            )
        return hidden_states, per_layer_combined


class SWAStatefulChunk2Prefill(SWAStatefulChunk2):
    """T=N prefill variant of SWAStatefulChunk2."""

    def __init__(self, model, start=8, end=15, ctx=2048,
                 use_linear=False, T: int = 8):
        super().__init__(model, start, end, ctx, use_linear=use_linear)
        self.T = T

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                current_pos, ring_pos):
        config = self.config
        T = self.T
        kv13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_prefill(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                current_pos, ring_pos,
                self.kv_cache_sliding, self.kv_cache_full,
                self.sliding_map, self.full_map,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )

        return hidden_states, kv13_k, kv13_v, kv14_k, kv14_v


class SWAStatefulChunk3Prefill(SWAStatefulChunk3):
    """T=N prefill variant of SWAStatefulChunk3 (stateless, KV-shared)."""

    def __init__(self, model, start=15, end=25,
                 use_linear=False, T: int = 8):
        super().__init__(model, start, end, use_linear=use_linear)
        self.T = T

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        T = self.T
        zero_idx = torch.zeros(1, dtype=torch.int32)
        dummy_state = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_prefill(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                zero_idx, zero_idx,
                dummy_state, dummy_state,
                {}, {},
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )
        return hidden_states


class SWAStatefulChunk4Prefill(SWAStatefulChunk4):
    """T=N prefill variant of SWAStatefulChunk4. Final norm + lm_head on
    the LAST position only (chunk4 produces a single token regardless
    of T)."""

    def __init__(self, model, start=25, end=35,
                 use_linear=False, T: int = 8):
        super().__init__(model, start, end, use_linear=use_linear)
        self.T = T

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                kv13_k, kv13_v, kv14_k, kv14_v):
        config = self.config
        T = self.T
        zero_idx = torch.zeros(1, dtype=torch.int32)
        dummy_state = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)

        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_prefill(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                zero_idx, zero_idx,
                dummy_state, dummy_state,
                {}, {},
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )

        # Take last position only for argmax; the (T-1) positions before
        # are intermediate prefill outputs that the runtime discards.
        last = hidden_states[:, T-1:T, :]   # (1, 1, hidden)
        normed = self.norm(last)
        logits = _project(self.lm_head, normed.to(MODEL_DTYPE))
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
        return token_id, token_logit, normed


# ============================================================
# 3-chunk variant: merged middle chunk (L8-24)
# ============================================================
#
# 4-chunk → 3-chunk consolidation. Mirrors the recurrent
# `MergedChunk23` (gemma4_swa_merged.py) but stateful: the merged
# middle chunk owns KV state for L8-14 (sliding + full) and runs
# L15-24 (KV-shared) internally using the kv13/kv14 producer aliases
# WITHOUT round-tripping them through chunk boundaries. The final
# chunk_3 (= old chunk_4 with lm_head + argmax) still needs the
# kv13/kv14 alias inputs, so the merged chunk emits them as outputs.
#
# Layer split for E2B:
#   chunk_1 :   L0-7    (own KV, computes PLE)        — same as 4-chunk
#   chunk_2m :  L8-24   (merged: own KV L8-14 + KV-shared L15-24)
#   chunk_3 :   L25-34  (KV-shared, lm_head, argmax)  — same as 4-chunk's chunk_4


class SWAStatefulMergedChunk23(_StatefulChunkBase):
    """Merged stateful chunk that owns the lower-half KV span and runs
    the upper-half KV-shared internally. Eliminates the 4-chunk's
    chunk_2 → chunk_3 hidden-state round-trip (~+5-10% Mac decode).

    Boundaries default to E2B (own=L8-14, shared=L15-24). For E4B pass
    own_range / shared_range derived from compute_chunk_boundaries(cfg)
    (E4B: own=L12-23, shared=L24-32).
    """
    DEFAULT_OWN = (8, 15)       # E2B own-KV layers (= old chunk_2)
    DEFAULT_SHARED = (15, 25)   # E2B KV-shared layers (= old chunk_3)

    def __init__(self, model: Gemma4Model, ctx: int = 2048,
                 use_linear: bool = False,
                 own_range: tuple[int, int] | None = None,
                 shared_range: tuple[int, int] | None = None):
        own = own_range if own_range is not None else self.DEFAULT_OWN
        shared = shared_range if shared_range is not None else self.DEFAULT_SHARED
        self.START_OWN, self.END_OWN = own
        self.START_SHARED, self.END_SHARED = shared
        # Init base with the OWN-KV span so the kv_cache_* buffers size
        # to chunk_2 only. KV-shared layers don't need state slots.
        super().__init__(model, self.START_OWN, self.END_OWN, ctx)
        self.layers_shared = nn.ModuleList([
            model.layers[i] for i in range(self.START_SHARED, self.END_SHARED)
        ])
        if use_linear:
            _swap_chunk_projections_to_linear(self.layers)
            _swap_chunk_projections_to_linear(self.layers_shared)

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                current_pos, ring_pos):
        config = self.config
        kv13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)

        # OWN-KV span (L8-14): writes state, produces kv13/kv14 at
        # L13/L14 internally.
        for local_idx in range(self.END_OWN - self.START_OWN):
            layer_idx = self.START_OWN + local_idx
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

        # KV-SHARED span (L15-24): consumes kv13/kv14 internally.
        zero_idx = torch.zeros(1, dtype=torch.int32)
        dummy_state = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        for local_idx in range(self.END_SHARED - self.START_SHARED):
            layer_idx = self.START_SHARED + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful(
                self.layers_shared[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                zero_idx, zero_idx,
                dummy_state, dummy_state,
                {}, {},
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )

        return hidden_states, kv13_k, kv13_v, kv14_k, kv14_v


class SWAStatefulMergedChunk23Prefill(SWAStatefulMergedChunk23):
    """T=N prefill variant of the merged middle chunk."""

    def __init__(self, model, ctx=2048, use_linear=False, T: int = 8,
                 own_range: tuple[int, int] | None = None,
                 shared_range: tuple[int, int] | None = None):
        super().__init__(model, ctx, use_linear=use_linear,
                         own_range=own_range, shared_range=shared_range)
        self.T = T

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                current_pos, ring_pos):
        config = self.config
        T = self.T
        kv13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)

        for local_idx in range(self.END_OWN - self.START_OWN):
            layer_idx = self.START_OWN + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_prefill(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                current_pos, ring_pos,
                self.kv_cache_sliding, self.kv_cache_full,
                self.sliding_map, self.full_map,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )

        zero_idx = torch.zeros(1, dtype=torch.int32)
        dummy_state = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        for local_idx in range(self.END_SHARED - self.START_SHARED):
            layer_idx = self.START_SHARED + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_prefill(
                self.layers_shared[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                zero_idx, zero_idx,
                dummy_state, dummy_state,
                {}, {},
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )

        return hidden_states, kv13_k, kv13_v, kv14_k, kv14_v


# ============================================================
# Single-buffer state probe (iPhone ANE multifunction workaround)
# ============================================================
#
# Hypothesis: iPhone ANE rejects the dual-state (kv_cache_sliding +
# kv_cache_full) + multifunction T>1 combination. Collapsing the two
# state buffers into ONE unified KV cache may bypass the limit.
#
# Layout: kv_cache_unified shape (2*num_own, HKV, ctx, max_hd)
#   - axis 0: K at slot 2*oi, V at slot 2*oi+1, where `oi` is the
#     own-KV layer index within the chunk (0..num_own-1)
#   - sliding layers waste positions [W, ctx) of their slot — only
#     positions [0, W) are written/read (write at ring_pos, read
#     first-W slice). Compute cost identical to dual.
#   - full layers use all ctx positions (write at current_pos, read
#     full slot). Same as dual.
#
# Memory cost: chunks 1+2 of E2B → +19 MB vs dual (negligible).
# Compute cost: zero (slice-at-forward keeps sliding-layer attention
# scan at W positions, not ctx).


def _own_layer_map(start: int, end: int, config):
    """Return {layer_idx: own_idx} for own-KV layers in [start, end).
    own_idx is the in-order index across both sliding and full layers,
    which is the layer's slot in the unified state buffer."""
    own_map = {}
    oi = 0
    for i in range(start, end):
        own_map[i] = oi
        oi += 1
    return own_map


def _run_layer_swa_stateful_single(
    layer, layer_idx, hidden_states,
    cos_s, sin_s, cos_f, sin_f,
    causal_mask_full, causal_mask_sliding,
    current_pos, ring_pos,
    kv_cache_unified,
    own_map,
    config, per_layer_combined,
    kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
):
    """T=1 forward layer with unified state buffer. Drop-in replacement
    for `_run_layer_swa_stateful` but reads/writes a single MLState."""
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    max_hd = config.global_head_dim
    is_full = config.is_full_attention(layer_idx)
    hd = config.get_head_dim(layer_idx)
    is_kv_shared = config.is_kv_shared(layer_idx)
    W = config.sliding_window

    residual = hidden_states
    h = layer.input_layernorm(hidden_states).to(MODEL_DTYPE)

    q = _project(layer.self_attn["q_proj"], h).view(
        1, num_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
    q = layer.self_attn["q_norm"](q.reshape(1, num_heads, hd)).view(1, num_heads, 1, hd)
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    if not is_kv_shared:
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

        oi = own_map[layer_idx]
        if is_full:
            # Linear write at current_pos in full ctx range.
            kv_cache_unified[2*oi:2*oi+1, :, current_pos:current_pos+1, :] = k_padded
            kv_cache_unified[2*oi+1:2*oi+2, :, current_pos:current_pos+1, :] = v_padded
            K_for_attn = kv_cache_unified[2*oi:2*oi+1, :, :, :hd]
            V_for_attn = kv_cache_unified[2*oi+1:2*oi+2, :, :, :hd]
        else:
            # Ring write at ring_pos in [0, W) range. Slice attention
            # to first W slots so compute cost = dual-buffer path.
            kv_cache_unified[2*oi:2*oi+1, :, ring_pos:ring_pos+1, :] = k_padded
            kv_cache_unified[2*oi+1:2*oi+2, :, ring_pos:ring_pos+1, :] = v_padded
            K_for_attn = kv_cache_unified[2*oi:2*oi+1, :, :W, :hd]
            V_for_attn = kv_cache_unified[2*oi+1:2*oi+2, :, :W, :hd]

        if layer_idx == config.kv_sliding_producer:
            kv_store_13_k = K_for_attn[..., :config.head_dim]
            kv_store_13_v = V_for_attn[..., :config.head_dim]
        elif layer_idx == config.kv_full_producer:
            kv_store_14_k = K_for_attn[..., :config.global_head_dim]
            kv_store_14_v = V_for_attn[..., :config.global_head_dim]
    else:
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)
    mask = causal_mask_full if is_full else causal_mask_sliding
    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
    attn_weights = attn_weights + mask
    attn_weights = ane_softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V_expanded)

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
    attn_output = _project(layer.self_attn["o_proj"], attn_output)
    attn_output = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + attn_output

    residual = hidden_states
    h = layer.pre_feedforward_layernorm(hidden_states).to(MODEL_DTYPE)
    gate = _project(layer.mlp["gate_proj"], h)
    up = _project(layer.mlp["up_proj"], h)
    gate = F.gelu(gate, approximate="tanh")
    mlp_out = _project(layer.mlp["down_proj"], gate * up)
    hidden_states = layer.post_feedforward_layernorm(mlp_out)
    hidden_states = residual + hidden_states

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


def _run_layer_swa_stateful_prefill_single(
    layer, layer_idx, hidden_states,
    cos_s, sin_s, cos_f, sin_f,
    causal_mask_full, causal_mask_sliding,
    current_pos, ring_pos,
    kv_cache_unified,
    own_map,
    config, per_layer_combined,
    kv_store_13_k, kv_store_13_v, kv_store_14_k, kv_store_14_v,
    T,
):
    """T=N prefill variant with unified state buffer."""
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    max_hd = config.global_head_dim
    is_full = config.is_full_attention(layer_idx)
    hd = config.get_head_dim(layer_idx)
    is_kv_shared = config.is_kv_shared(layer_idx)
    W = config.sliding_window

    residual = hidden_states
    h = layer.input_layernorm(hidden_states).to(MODEL_DTYPE)

    q = _project(layer.self_attn["q_proj"], h).view(
        1, T, num_heads, hd).permute(0, 2, 1, 3).to(MODEL_DTYPE)
    q = layer.self_attn["q_norm"](q.reshape(1, num_heads, T, hd))
    if is_full:
        q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
    else:
        q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

    if not is_kv_shared:
        k = _project(layer.self_attn["k_proj"], h).view(
            1, T, num_kv_heads, hd).permute(0, 2, 1, 3).to(MODEL_DTYPE)
        v = _project(layer.self_attn["v_proj"], h).view(
            1, T, num_kv_heads, hd).permute(0, 2, 1, 3).to(MODEL_DTYPE)
        k = layer.self_attn["k_norm"](k.reshape(1, num_kv_heads, T, hd))
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

        oi = own_map[layer_idx]
        if is_full:
            kv_cache_unified[2*oi:2*oi+1, :, current_pos:current_pos+T, :] = k_padded
            kv_cache_unified[2*oi+1:2*oi+2, :, current_pos:current_pos+T, :] = v_padded
            K_for_attn = kv_cache_unified[2*oi:2*oi+1, :, :, :hd]
            V_for_attn = kv_cache_unified[2*oi+1:2*oi+2, :, :, :hd]
        else:
            kv_cache_unified[2*oi:2*oi+1, :, ring_pos:ring_pos+T, :] = k_padded
            kv_cache_unified[2*oi+1:2*oi+2, :, ring_pos:ring_pos+T, :] = v_padded
            K_for_attn = kv_cache_unified[2*oi:2*oi+1, :, :W, :hd]
            V_for_attn = kv_cache_unified[2*oi+1:2*oi+2, :, :W, :hd]

        if layer_idx == config.kv_sliding_producer:
            kv_store_13_k = K_for_attn[..., :config.head_dim]
            kv_store_13_v = V_for_attn[..., :config.head_dim]
        elif layer_idx == config.kv_full_producer:
            kv_store_14_k = K_for_attn[..., :config.global_head_dim]
            kv_store_14_v = V_for_attn[..., :config.global_head_dim]
    else:
        if is_full:
            K_for_attn = kv_store_14_k
            V_for_attn = kv_store_14_v
        else:
            K_for_attn = kv_store_13_k
            V_for_attn = kv_store_13_v

    K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
    V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)
    mask = causal_mask_full if is_full else causal_mask_sliding
    attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
    attn_weights = attn_weights + mask
    attn_weights = ane_softmax(attn_weights, dim=-1)
    attn_output = torch.matmul(attn_weights, V_expanded)

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, T, -1)
    attn_output = _project(layer.self_attn["o_proj"], attn_output)
    attn_output = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + attn_output

    residual = hidden_states
    h = layer.pre_feedforward_layernorm(hidden_states).to(MODEL_DTYPE)
    gate = _project(layer.mlp["gate_proj"], h)
    up = _project(layer.mlp["up_proj"], h)
    gate = F.gelu(gate, approximate="tanh")
    mlp_out = _project(layer.mlp["down_proj"], gate * up)
    hidden_states = layer.post_feedforward_layernorm(mlp_out)
    hidden_states = residual + hidden_states

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


class _StatefulSingleChunkBase(nn.Module):
    """Common: own-KV layer map + unified state buffer."""

    def __init__(self, model: Gemma4Model, start: int, end: int, ctx: int):
        super().__init__()
        self.config = model.config
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([model.layers[i] for i in range(start, end)])
        self.own_map = _own_layer_map(start, end, model.config)
        self.num_own = len(self.own_map)
        self.ctx = ctx
        self.W = model.config.sliding_window

        max_hd = model.config.global_head_dim
        HKV = model.config.num_key_value_heads
        self.register_buffer(
            "kv_cache_unified",
            torch.zeros(2 * max(self.num_own, 1), HKV, ctx, max_hd,
                         dtype=MODEL_DTYPE),
        )


class SWAStatefulChunk1Single(_StatefulSingleChunkBase):
    """T=1 first chunk with unified state buffer."""

    def __init__(self, model, start=0, end=8, ctx=2048,
                 use_linear=False):
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
        d_13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        d_13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        d_14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        d_14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, d_13_k, d_13_v, d_14_k, d_14_v
             ) = _run_layer_swa_stateful_single(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                current_pos, ring_pos,
                self.kv_cache_unified, self.own_map,
                config, per_layer_combined,
                d_13_k, d_13_v, d_14_k, d_14_v,
            )
        return hidden_states, per_layer_combined


class SWAStatefulChunk1PrefillSingle(SWAStatefulChunk1Single):
    """T=N prefill variant of SWAStatefulChunk1Single."""

    def __init__(self, model, start=0, end=8, ctx=2048,
                 use_linear=False, T: int = 8):
        super().__init__(model, start, end, ctx, use_linear=use_linear)
        self.T = T

    def _compute_ple_prefill(self, hidden_states, per_layer_raw):
        T = self.T
        h = hidden_states.to(MODEL_DTYPE)
        proj = _project(self.per_layer_model_projection, h) \
               * self.per_layer_model_projection_scale
        proj_grouped = proj.view(1, T, self.num_layers_total, self.per_layer_dim)
        norm_w = self.per_layer_projection_norm.weight
        eps = float(self.per_layer_projection_norm.eps)
        doubled = torch.cat([proj_grouped, -proj_grouped], dim=-1)
        normed = F.layer_norm(doubled, normalized_shape=(2 * self.per_layer_dim,),
                              weight=None, bias=None, eps=eps)
        normed, _ = torch.chunk(normed, 2, dim=-1)
        proj_normed = (normed * norm_w).view(
            1, T, self.num_layers_total * self.per_layer_dim)
        return (proj_normed + per_layer_raw) * self.per_layer_input_scale

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_raw, cos_s, sin_s, cos_f, sin_f,
                current_pos, ring_pos):
        config = self.config
        T = self.T
        per_layer_combined = self._compute_ple_prefill(hidden_states, per_layer_raw)
        d_13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        d_13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        d_14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        d_14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        for local_idx in range(self.end - self.start):
            layer_idx = self.start + local_idx
            (hidden_states, d_13_k, d_13_v, d_14_k, d_14_v
             ) = _run_layer_swa_stateful_prefill_single(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                current_pos, ring_pos,
                self.kv_cache_unified, self.own_map,
                config, per_layer_combined,
                d_13_k, d_13_v, d_14_k, d_14_v,
                T,
            )
        return hidden_states, per_layer_combined


class SWAStatefulMergedChunk23Single(_StatefulSingleChunkBase):
    """3-chunk merged middle with unified state buffer.
    Owns chunk_2 KV; runs chunk_3 KV-shared internally. Emits kv13/kv14
    aliases for the final chunk_3.

    Boundaries default to E2B (own=L8-14, shared=L15-24). For E4B pass
    own_range / shared_range from compute_chunk_boundaries(cfg)
    (E4B: own=L12-23, shared=L24-32)."""
    DEFAULT_OWN = (8, 15)
    DEFAULT_SHARED = (15, 25)

    def __init__(self, model, ctx=2048, use_linear=False,
                 own_range: tuple[int, int] | None = None,
                 shared_range: tuple[int, int] | None = None):
        own = own_range if own_range is not None else self.DEFAULT_OWN
        shared = shared_range if shared_range is not None else self.DEFAULT_SHARED
        self.START_OWN, self.END_OWN = own
        self.START_SHARED, self.END_SHARED = shared
        super().__init__(model, self.START_OWN, self.END_OWN, ctx)
        self.layers_shared = nn.ModuleList([
            model.layers[i] for i in range(self.START_SHARED, self.END_SHARED)
        ])
        if use_linear:
            _swap_chunk_projections_to_linear(self.layers)
            _swap_chunk_projections_to_linear(self.layers_shared)

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                current_pos, ring_pos):
        config = self.config
        kv13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        for local_idx in range(self.END_OWN - self.START_OWN):
            layer_idx = self.START_OWN + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_single(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                current_pos, ring_pos,
                self.kv_cache_unified, self.own_map,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )
        zero_idx = torch.zeros(1, dtype=torch.int32)
        dummy = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        for local_idx in range(self.END_SHARED - self.START_SHARED):
            layer_idx = self.START_SHARED + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_single(
                self.layers_shared[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                zero_idx, zero_idx,
                dummy, {},
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )
        return hidden_states, kv13_k, kv13_v, kv14_k, kv14_v


class SWAStatefulMergedChunk23PrefillSingle(SWAStatefulMergedChunk23Single):
    """T=N prefill variant of merged middle with unified state."""

    def __init__(self, model, ctx=2048, use_linear=False, T: int = 8,
                 own_range: tuple[int, int] | None = None,
                 shared_range: tuple[int, int] | None = None):
        super().__init__(model, ctx, use_linear=use_linear,
                         own_range=own_range, shared_range=shared_range)
        self.T = T

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_combined, cos_s, sin_s, cos_f, sin_f,
                current_pos, ring_pos):
        config = self.config
        T = self.T
        kv13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        for local_idx in range(self.END_OWN - self.START_OWN):
            layer_idx = self.START_OWN + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_prefill_single(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                current_pos, ring_pos,
                self.kv_cache_unified, self.own_map,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )
        zero_idx = torch.zeros(1, dtype=torch.int32)
        dummy = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        for local_idx in range(self.END_SHARED - self.START_SHARED):
            layer_idx = self.START_SHARED + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_prefill_single(
                self.layers_shared[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                zero_idx, zero_idx,
                dummy, {},
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )
        return hidden_states, kv13_k, kv13_v, kv14_k, kv14_v


# ============================================================
# 1-chunk all-in-one (Qwen3-VL pattern, single mlpackage)
# ============================================================
#
# Entire 35-layer Gemma 4 E2B (PLE compute + own-KV L0-14 + KV-shared
# L15-34 + final norm + lm_head + argmax) packaged as ONE mlpackage.
# Single MLState (kv_cache_unified) covers all 15 own-KV layers
# (L0-14, the sliding + full producers). KV-shared layers L15-34 read
# kv13/kv14 from the producers' own state slice internally — never
# leave the graph.
#
# Inputs (T=1 decode):
#   hidden_states (1, 1, hidden), per_layer_raw (1, 1, num_layers*pld),
#   causal_mask_full (1, 1, 1, ctx), causal_mask_sliding (1, 1, 1, W),
#   cos_s/sin_s (1, 1, 1, hd_s), cos_f/sin_f (1, 1, 1, hd_f),
#   current_pos (1,), ring_pos (1,)
# Outputs: token_id (1,), token_logit (1,), hidden_normed (1,1,hidden)
# State: kv_cache_unified (2*15, HKV, ctx, max_hd)
#
# Risks: Mac conversion time (35 layers in one MIL pipeline), Mac/
# iPhone ANE compile size limits, single mlpackage ~1.1 GB. Worth
# testing because Qwen3-VL ships this pattern (28 layers, single
# state) and our T=8 multifunction failure on iPhone may be cured by
# eliminating the chunk-level dual-state plumbing.


class SWAStatefulModel1Chunk(_StatefulSingleChunkBase):
    """All 35 layers + lm_head in one chunk. Own-KV L0-14, KV-shared
    L15-34. Output is a single token (last position only)."""
    OWN_START, OWN_END = 0, 15           # L0-14 own KV (sliding+full producers)
    SHARED_START, SHARED_END = 15, 35    # L15-34 KV-shared (internal alias)

    def __init__(self, model: Gemma4Model, ctx: int = 2048,
                 use_linear: bool = False):
        # Init base with own-KV span; layers list = own-KV layers only.
        super().__init__(model, self.OWN_START, self.OWN_END, ctx)
        self.layers_shared = nn.ModuleList([
            model.layers[i] for i in range(self.SHARED_START, self.SHARED_END)
        ])
        # PLE compute (was in chunk_1)
        self.per_layer_model_projection = model.per_layer_model_projection
        self.per_layer_projection_norm = model.per_layer_projection_norm
        self.per_layer_model_projection_scale = model.per_layer_model_projection_scale
        self.per_layer_input_scale = model.per_layer_input_scale
        self.per_layer_dim = model.config.hidden_size_per_layer_input
        self.num_layers_total = model.config.num_hidden_layers
        # Final norm + lm_head + argmax (was in chunk_4)
        self.norm = model.norm
        self.lm_head = nn.Conv2d(model.lm_head.in_channels,
                                  model.lm_head.out_channels,
                                  kernel_size=1, bias=False)
        self.lm_head.weight.data = model.lm_head.weight.data.clone()
        self.argmax = model.argmax
        self.softcap = model.softcap
        if use_linear:
            _swap_chunk_projections_to_linear(self.layers)
            _swap_chunk_projections_to_linear(self.layers_shared)
            self.per_layer_model_projection = _replace_conv2d_with_linear(
                self.per_layer_model_projection)
            self.lm_head = _replace_conv2d_with_linear(self.lm_head)

    def _compute_ple(self, hidden_states, per_layer_raw):
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
        kv13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        # Own-KV span
        for local_idx in range(self.OWN_END - self.OWN_START):
            layer_idx = self.OWN_START + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_single(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                current_pos, ring_pos,
                self.kv_cache_unified, self.own_map,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )
        # KV-shared span
        zero_idx = torch.zeros(1, dtype=torch.int32)
        dummy = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        for local_idx in range(self.SHARED_END - self.SHARED_START):
            layer_idx = self.SHARED_START + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_single(
                self.layers_shared[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                zero_idx, zero_idx,
                dummy, {},
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
            )
        # Final norm + lm_head + argmax
        normed = self.norm(hidden_states)
        logits = _project(self.lm_head, normed.to(MODEL_DTYPE))
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
        return token_id, token_logit, normed


class SWAStatefulModel1ChunkPrefill(SWAStatefulModel1Chunk):
    """T=N prefill variant — same all-in-one structure, lm_head on
    last position only (T-1 intermediates discarded)."""

    def __init__(self, model, ctx=2048, use_linear=False, T: int = 8):
        super().__init__(model, ctx, use_linear=use_linear)
        self.T = T

    def _compute_ple_prefill(self, hidden_states, per_layer_raw):
        T = self.T
        h = hidden_states.to(MODEL_DTYPE)
        proj = _project(self.per_layer_model_projection, h) \
               * self.per_layer_model_projection_scale
        proj_grouped = proj.view(1, T, self.num_layers_total, self.per_layer_dim)
        norm_w = self.per_layer_projection_norm.weight
        eps = float(self.per_layer_projection_norm.eps)
        doubled = torch.cat([proj_grouped, -proj_grouped], dim=-1)
        normed = F.layer_norm(doubled, normalized_shape=(2 * self.per_layer_dim,),
                              weight=None, bias=None, eps=eps)
        normed, _ = torch.chunk(normed, 2, dim=-1)
        proj_normed = (normed * norm_w).view(
            1, T, self.num_layers_total * self.per_layer_dim)
        return (proj_normed + per_layer_raw) * self.per_layer_input_scale

    def forward(self, hidden_states, causal_mask_full, causal_mask_sliding,
                per_layer_raw, cos_s, sin_s, cos_f, sin_f,
                current_pos, ring_pos):
        config = self.config
        T = self.T
        per_layer_combined = self._compute_ple_prefill(hidden_states, per_layer_raw)
        kv13_k = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, 1, config.head_dim, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, 1, config.global_head_dim, dtype=MODEL_DTYPE)
        for local_idx in range(self.OWN_END - self.OWN_START):
            layer_idx = self.OWN_START + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_prefill_single(
                self.layers[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                current_pos, ring_pos,
                self.kv_cache_unified, self.own_map,
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )
        zero_idx = torch.zeros(1, dtype=torch.int32)
        dummy = torch.zeros(1, 1, 1, 1, dtype=MODEL_DTYPE)
        for local_idx in range(self.SHARED_END - self.SHARED_START):
            layer_idx = self.SHARED_START + local_idx
            (hidden_states, kv13_k, kv13_v, kv14_k, kv14_v
             ) = _run_layer_swa_stateful_prefill_single(
                self.layers_shared[local_idx], layer_idx, hidden_states,
                cos_s, sin_s, cos_f, sin_f,
                causal_mask_full, causal_mask_sliding,
                zero_idx, zero_idx,
                dummy, {},
                config, per_layer_combined,
                kv13_k, kv13_v, kv14_k, kv14_v,
                T,
            )
        # lm_head on LAST position only.
        last = hidden_states[:, T-1:T, :]
        normed = self.norm(last)
        logits = _project(self.lm_head, normed.to(MODEL_DTYPE))
        if self.softcap > 0:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits.squeeze(0))
        return token_id, token_logit, normed
