"""Monolithic 1-token decode wrapper for LFM2 / LFM2.5.

Two MLState buffers:
  kv_cache_0   (2*n_attn_layers,    num_kv_heads, ctx,     head_dim)  fp16
  conv_cache_0 (n_conv_layers,      hidden,                L_cache)   fp16

Inputs:
  input_ids    (1, 1)             int32
  position_ids (1,)                int32   — RoPE position lookup
  causal_mask  (1, 1, 1, ctx)      fp16    — used by attention layers only
  update_mask  (1, 1, ctx, 1)      fp16    — KV write position selector

Outputs:
  token_id     (1,)                int32
  token_logit  (1,)                fp16

The wrapper mirrors the existing Qwen2 ``MonolithicWrapper`` but with two
extra pieces:

  1. Per-layer dispatch on ``is_attention_layer``.  Attention layers behave
     exactly like Qwen2 (mask-based KV write).  Conv layers shift the
     conv-state buffer left by one and append the current ``Bx`` at the
     rightmost slot, then do a depthwise Conv2d (kernel=(1, L_cache)) on
     the resulting window.  This handles cache_position==0 cleanly because
     the buffer is initialised to zero.

  2. The ``out_proj`` name (LFM2) instead of ``o_proj``, no attention bias,
     and Q/K-norm before RoPE.  Logit softcap is NOT applied (LFM2 has none).

LFM2.5-350M tied embeddings, no logit softcap, no embedding scale.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb


class Lfm2MonolithicWrapper(nn.Module):
    """1-token decode wrapper used by ``CoreMLExporter`` for LFM2 models."""

    def __init__(self, model) -> None:
        super().__init__()
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers
        self.embedding_norm = model.embedding_norm
        self.lm_head = model.lm_head
        self.config = model.config
        self.layer_types = list(model.layer_types)
        self.attn_layer_indices = list(model.attn_layer_indices)
        self.conv_layer_indices = list(model.conv_layer_indices)
        self.conv_l_cache = int(model.conv_l_cache)

        n_attn = len(self.attn_layer_indices)
        n_conv = len(self.conv_layer_indices)
        # global layer idx -> position inside its respective state tensor
        self._layer_to_attn_slot = {
            li: si for si, li in enumerate(self.attn_layer_indices)
        }
        self._layer_to_conv_slot = {
            li: si for si, li in enumerate(self.conv_layer_indices)
        }

        kv_shape = (
            2 * n_attn,
            self.config.num_key_value_heads,
            self.config.context_length,
            self.config.head_dim,
        )
        # KV cache: one rank-4 buffer slot-indexed by layer (works on ANE —
        # this is the same pattern Qwen2/FunctionGemma use successfully).
        self.register_buffer("kv_cache_0", torch.zeros(kv_shape, dtype=MODEL_DTYPE))

        # Conv state passed as INPUT/OUTPUT tensor (not MLState).
        #
        # Why not MLState: the M-series ANE planner rejects the dual-state
        # combination (kv_cache_0 + conv_cache_0) with status=0x1d.  We
        # bisected that — declaring the second state buffer + reading it
        # from the graph is what trips the runtime, regardless of rank,
        # innermost size, or update pattern.  The repo already documents
        # the same hazard for gemma4's dual KV cache (sliding/full) —
        # see ``conversion/models/gemma4_swa_stateful_chunks.py:961``.
        #
        # Workaround: keep the conv state OUT of MLState.  It's small
        # (n_conv × hidden × L_pad × 2 = 60–320 KB) so the CPU↔ANE
        # round-trip per decode step is negligible.  The runtime feeds
        # ``conv_state_in`` from the previous step's ``conv_state_out``.
        if os.environ.get("LFM2_PROBE_SKIP_CONV") != "1" and os.environ.get("LFM2_PROBE_NO_CONV_STATE") != "1":
            l_pad = int(os.environ.get("LFM2_CONV_L_PAD", "16"))
            assert l_pad >= self.conv_l_cache
            self.conv_l_padded = l_pad
            self.uses_conv_io = True

            # Slide via fixed shift matmul + one-hot Bx insert, both built
            # against the padded width.  Live taps are the first L_cache
            # columns; everything past that is zero.
            S = torch.zeros(l_pad, l_pad, dtype=MODEL_DTYPE)
            for i in range(self.conv_l_cache - 1):
                S[i + 1, i] = 1.0
            self.register_buffer("conv_shift", S)
            e = torch.zeros(1, 1, 1, l_pad, dtype=MODEL_DTYPE)
            e[..., self.conv_l_cache - 1] = 1.0
            self.register_buffer("conv_last_slot", e)

            # Replace each conv layer's depthwise kernel with a padded one so
            # it reads the full L_pad window with zeros in padded slots —
            # semantically identical to the original L_cache kernel.
            for li, t in enumerate(model.layer_types):
                if t != "conv":
                    continue
                old = model.layers[li].conv.conv  # (h, 1, 1, L_cache)
                pad_conv = nn.Conv2d(
                    self.config.hidden_size, self.config.hidden_size,
                    kernel_size=(1, l_pad), groups=self.config.hidden_size,
                    bias=(old.bias is not None), dtype=MODEL_DTYPE,
                )
                with torch.no_grad():
                    new_w = torch.zeros_like(pad_conv.weight)
                    new_w[..., : self.conv_l_cache] = old.weight
                    pad_conv.weight.copy_(new_w)
                    if old.bias is not None:
                        pad_conv.bias.copy_(old.bias)
                model.layers[li].conv.conv = pad_conv
        else:
            self.uses_conv_io = False

            # Shift matrix S (L_pad, L_pad) such that ``state @ S`` shifts the
            # (0..L_cache-1) live window left by one.  Outside the live block
            # all entries are zero.
            S = torch.zeros(l_pad, l_pad, dtype=MODEL_DTYPE)
            for i in range(self.conv_l_cache - 1):
                S[i + 1, i] = 1.0
            self.register_buffer("conv_shift", S)
            # One-hot selector — 1 at the live rightmost slot (L_cache - 1).
            e = torch.zeros(1, 1, 1, l_pad, dtype=MODEL_DTYPE)
            e[..., self.conv_l_cache - 1] = 1.0
            self.register_buffer("conv_last_slot", e)

            # Replace each conv layer's depthwise conv with a padded variant
            # so the kernel matches the padded L width.  Live taps go in the
            # first L_cache positions; padded positions are zero.
            for li, t in enumerate(model.layer_types):
                if t != "conv":
                    continue
                old = model.layers[li].conv.conv  # (h, 1, 1, L_cache)
                pad = nn.Conv2d(
                    self.config.hidden_size, self.config.hidden_size,
                    kernel_size=(1, l_pad), groups=self.config.hidden_size,
                    bias=(old.bias is not None), dtype=MODEL_DTYPE,
                )
                with torch.no_grad():
                    new_w = torch.zeros_like(pad.weight)
                    new_w[..., : self.conv_l_cache] = old.weight
                    pad.weight.copy_(new_w)
                    if old.bias is not None:
                        pad.bias.copy_(old.bias)
                model.layers[li].conv.conv = pad

        # RoPE tables (same recipe as exporter.MonolithicWrapper).
        head_dim = self.config.head_dim
        base = self.config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        max_len = max(self.config.context_length * 2, 128)
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(MODEL_DTYPE))
        self.register_buffer("sin_cached", emb.sin().to(MODEL_DTYPE))

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _rmsnorm(
        weight: torch.Tensor, eps: float, x: torch.Tensor, normalized_size: int,
    ) -> torch.Tensor:
        """Inline ANERMSNorm: cat([x, -x]) -> LayerNorm -> first half * weight.

        ``normalized_size`` is the doubled feature dim (i.e. 2 * hidden) and is
        passed in as a Python int so torch.jit.trace doesn't fall back to
        ``aten::Int(tensor.size(-1))`` — coremltools rejects that.
        """
        doubled = torch.cat([x, -x], dim=-1)
        normed = nn.functional.layer_norm(
            doubled,
            normalized_shape=(normalized_size,),
            weight=None,
            bias=None,
            eps=float(eps),
        )
        first, _ = torch.chunk(normed, 2, dim=-1)
        return first * weight

    def _attention_layer(
        self,
        layer,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
        update_mask: torch.Tensor,
        attn_slot: int,
    ) -> torch.Tensor:
        cfg = self.config
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        hd = cfg.head_dim
        n_rep = n_heads // n_kv
        scale = 1.0 / (hd ** 0.5)
        n_attn = len(self.attn_layer_indices)

        # (1, 1, hidden) -> (1, hidden, 1, 1) for Conv2d projections
        x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

        # QKV
        q = layer.self_attn.q_proj(x).view(1, n_heads, hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        k = layer.self_attn.k_proj(x).view(1, n_kv,    hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        v = layer.self_attn.v_proj(x).view(1, n_kv,    hd, 1).permute(0, 1, 3, 2).to(MODEL_DTYPE)

        # QK-norm (per head_dim, applied BEFORE RoPE — matches HF Lfm2Attention)
        q = self._rmsnorm(
            layer.self_attn.q_layernorm.weight,
            layer.self_attn.q_layernorm.eps,
            q,
            normalized_size=2 * hd,
        )
        k = self._rmsnorm(
            layer.self_attn.k_layernorm.weight,
            layer.self_attn.k_layernorm.eps,
            k,
            normalized_size=2 * hd,
        )

        # RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Stateful KV write (mask-based, matches Qwen2 monolithic).
        k_idx = attn_slot
        v_idx = n_attn + attn_slot
        K_cache = self.kv_cache_0[k_idx].unsqueeze(0)
        V_cache = self.kv_cache_0[v_idx].unsqueeze(0)
        k_b = k.expand_as(K_cache)
        v_b = v.expand_as(V_cache)
        K_new = K_cache * (1 - update_mask) + k_b * update_mask
        V_new = V_cache * (1 - update_mask) + v_b * update_mask
        self.kv_cache_0[k_idx] = K_new.squeeze(0)
        self.kv_cache_0[v_idx] = V_new.squeeze(0)

        # GQA expand
        K_e = K_new.repeat_interleave(n_rep, dim=1)
        V_e = V_new.repeat_interleave(n_rep, dim=1)

        # Attention in fp32 (matches existing stack convention)
        q_f = q.to(torch.float32)
        k_f = K_e.to(torch.float32)
        attn = torch.matmul(q_f, k_f.transpose(-1, -2)) * scale
        attn = attn + causal_mask.to(torch.float32)
        attn = torch.softmax(attn, dim=-1).to(MODEL_DTYPE)
        out = torch.matmul(attn.to(torch.float32), V_e.to(torch.float32)).to(MODEL_DTYPE)

        # output projection (LFM2 calls this `out_proj`)
        out = out.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
        out = layer.self_attn.out_proj(
            out.permute(0, 2, 1).unsqueeze(2)
        ).squeeze(2).permute(0, 2, 1)
        return out

    def _conv_layer(
        self,
        layer,
        hidden_states: torch.Tensor,
        conv_slot: int,
        conv_state_in: torch.Tensor,  # (n_conv, hidden, L_pad)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, new_state_slot_for_this_conv_layer).

        ``new_state_slot`` shape: (hidden, L_pad).  Caller stacks the slots
        from all conv layers and returns them as the model's conv_state_out.
        """
        h = self.config.hidden_size

        # (1, 1, h) -> (1, h, 1, 1) Conv2d layout
        x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

        # in_proj: (1, h, 1, 1) -> (1, 3h, 1, 1)
        BCx = layer.conv.in_proj(x)
        # Split along the channel axis into B, C, x_b each (1, h, 1, 1)
        B, C, x_b = torch.chunk(BCx, 3, dim=1)
        Bx = B * x_b  # (1, h, 1, 1)

        if os.environ.get("LFM2_PROBE_NO_CONV_STATE") == "1":
            # PROBE: bypass the stateful slide.  Run the depthwise conv on a
            # zero-padded window with only the current Bx.  Numerically wrong
            # past pos=0, but isolates whether the conv-state read/update
            # pattern is the ANE blocker.
            zeros = torch.zeros_like(Bx).repeat(1, 1, 1, self.conv_l_cache - 1)
            new_state = torch.cat([zeros, Bx], dim=-1)
        elif os.environ.get("LFM2_PROBE_READONLY_STATE") == "1":
            # PROBE: read state but never write — pinpoints whether the state
            # WRITE is what's tripping ANE inference.
            slot = self.conv_cache_0[conv_slot].unsqueeze(0)  # (1, h, 1, L)
            shifted = torch.matmul(slot, self.conv_shift)
            new_state = shifted + Bx * self.conv_last_slot
        else:
            # Slide the live (L_cache=3) window with slice + cat — no
            # reductions, so fp16 is bit-stable.  ``state[..., 1:L_cache]``
            # picks the trailing two live taps; cat with the new ``Bx``
            # gives a 3-element live window.  We then re-pad with the
            # already-zeroed tail so the buffer keeps its ``L_pad``
            # innermost dim (so the depthwise conv kernel matches).
            slot = conv_state_in[conv_slot].unsqueeze(0).unsqueeze(2)  # (1, h, 1, L_pad)
            live_tail = slot[..., 1:self.conv_l_cache]                  # (1, h, 1, L_cache-1)
            live = torch.cat([live_tail, Bx], dim=-1)                   # (1, h, 1, L_cache)
            if self.conv_l_padded > self.conv_l_cache:
                pad = torch.zeros(
                    1, h, 1, self.conv_l_padded - self.conv_l_cache,
                    dtype=MODEL_DTYPE, device=Bx.device,
                )
                new_state = torch.cat([live, pad], dim=-1)
            else:
                new_state = live
            new_state_slot = new_state.squeeze(0).squeeze(1)

        # Depthwise causal conv over the L-window. groups=h so it's per-channel.
        conv_out = layer.conv.conv(new_state)  # (1, h, 1, 1)

        # Gated branch: y = C * conv_out
        y = C * conv_out  # (1, h, 1, 1)
        y = layer.conv.out_proj(y)  # (1, h, 1, 1)

        # back to (1, 1, h) ; also return the slot for stacking into out state
        return y.squeeze(2).permute(0, 2, 1), new_state_slot

    # --- forward --------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,    # (1, 1) int32
        position_ids: torch.Tensor, # (1,) int32
        causal_mask: torch.Tensor,  # (1, 1, 1, ctx) fp16
        update_mask: torch.Tensor,  # (1, 1, ctx, 1) fp16
        conv_state_in: torch.Tensor,  # (n_conv, hidden, L_pad) fp16 input
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        hd = cfg.head_dim

        # Embedding
        hidden_states = self.embed_tokens(input_ids).to(MODEL_DTYPE)

        # RoPE for current position
        cos = torch.index_select(self.cos_cached, 0, position_ids).view(1, 1, 1, hd)
        sin = torch.index_select(self.sin_cached, 0, position_ids).view(1, 1, 1, hd)

        # Layers
        h = self.config.hidden_size
        norm_size = 2 * h
        n_conv = len(self.conv_layer_indices)
        # Collect new conv state slots in order of conv_slot 0..n_conv-1.
        new_conv_slots = [None] * n_conv

        for li, ltype in enumerate(self.layer_types):
            layer = self.layers[li]
            residual = hidden_states

            # Pre-norm (operator_norm in HF naming)
            normed = self._rmsnorm(
                layer.operator_norm.weight, layer.operator_norm.eps, hidden_states,
                normalized_size=norm_size,
            )
            if ltype == "full_attention":
                op_out = self._attention_layer(
                    layer, normed, cos, sin, causal_mask, update_mask,
                    attn_slot=self._layer_to_attn_slot[li],
                )
                hidden_states = residual + op_out
            elif os.environ.get("LFM2_PROBE_SKIP_CONV") == "1":
                # PROBE: skip the entire conv block.  Used to bisect ANE
                # inference failure — does it survive without the conv state?
                hidden_states = residual
            else:
                conv_slot = self._layer_to_conv_slot[li]
                op_out, new_slot = self._conv_layer(
                    layer, normed,
                    conv_slot=conv_slot,
                    conv_state_in=conv_state_in,
                )
                new_conv_slots[conv_slot] = new_slot
                hidden_states = residual + op_out

            # FFN
            residual = hidden_states
            normed = self._rmsnorm(
                layer.ffn_norm.weight, layer.ffn_norm.eps, hidden_states,
                normalized_size=norm_size,
            )
            xn = normed.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
            a = layer.feed_forward.w1(xn)
            b = layer.feed_forward.w3(xn)
            mlp = layer.feed_forward.w2(nn.functional.silu(a) * b)
            mlp = mlp.squeeze(2).permute(0, 2, 1)
            hidden_states = residual + mlp

        # Final embedding_norm (LFM2 has its own — distinct from layer norms)
        hidden_states = self._rmsnorm(
            self.embedding_norm.weight, self.embedding_norm.eps, hidden_states,
            normalized_size=norm_size,
        )

        # LM head + in-graph argmax (matches InModelArgmax used by Qwen2 path)
        x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)  # (1, 1, vocab)
        logits_2d = logits.squeeze(0)  # (1, vocab)
        token_id = torch.argmax(logits_2d, dim=-1)                          # (1,)
        token_logit = logits_2d.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)

        # Stack the per-layer conv state slots back into one tensor.
        if any(s is None for s in new_conv_slots):
            # PROBE mode skipped some conv layers — fall back to passing the
            # input state straight through for those slots.
            new_conv_slots = [
                conv_state_in[i] if s is None else s
                for i, s in enumerate(new_conv_slots)
            ]
        conv_state_out = torch.stack(new_conv_slots, dim=0)
        return token_id, token_logit, conv_state_out
