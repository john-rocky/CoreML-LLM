#!/usr/bin/env python3
"""Compare per-layer hidden states: HF Gemma4 vs our custom Gemma4MonolithicWrapper.

Loads models SEQUENTIALLY to fit in 16GB RAM.
"""
from __future__ import annotations
import gc, os, sys, torch
import torch.nn.functional as F

HF_DIR = os.environ.get(
    "HF_DIR",
    os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"),
)

def main():
    device = "cpu"
    prompt_ids = [2]  # single BOS token — simplest case

    ids_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    N = ids_t.shape[1]  # 1

    # ========== Phase 1: HF forward ==========
    print(f"=== Phase 1: HF forward ===")
    from transformers import Gemma4ForConditionalGeneration
    hf_model = Gemma4ForConditionalGeneration.from_pretrained(
        HF_DIR, torch_dtype=torch.float32, device_map=device,
    )
    hf_model.eval()
    hf_lm = hf_model.model.language_model

    # Collect layer_scalar values before freeing
    hf_layer_scalars = [hf_lm.layers[i].layer_scalar.item() for i in range(len(hf_lm.layers))]

    with torch.no_grad():
        hf_out = hf_lm(input_ids=ids_t, output_hidden_states=True, use_cache=False)

    hf_hidden = [h[0, -1].float().clone() for h in hf_out.hidden_states]
    hf_last = hf_out.last_hidden_state[0, -1].float().clone()
    print(f"  Collected {len(hf_hidden)} hidden states (last token)")
    print(f"  HF last_hidden norm: {hf_last.norm().item():.4f}")

    del hf_model, hf_lm, hf_out
    gc.collect()
    print("  HF model freed.\n")

    # ========== Phase 2: Custom forward ==========
    print(f"=== Phase 2: Custom forward ===")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models.gemma4 import Gemma4Model
    from models.gemma4_wrapper import Gemma4MonolithicWrapper, v_norm
    from ane_ops import apply_rotary_pos_emb

    custom_base = Gemma4Model.from_pretrained(HF_DIR, context_length=32)
    custom = Gemma4MonolithicWrapper(custom_base)
    custom = custom.to(torch.float32).to(device)
    custom.eval()
    del custom_base
    gc.collect()

    config = custom.config
    num_layers = config.num_hidden_layers
    max_hd = config.global_head_dim

    custom_hiddens = []

    with torch.no_grad():
        # Embedding
        text_embedding = custom.embed_tokens(ids_t).to(torch.float32)
        text_embedding = text_embedding * (config.hidden_size ** 0.5)
        hidden_states = text_embedding.clone()

        # PLE
        per_layer_raw = custom.embed_tokens_per_layer(ids_t).to(torch.float32) * custom.per_layer_embed_scale
        per_layer_proj = custom.per_layer_model_projection(
            hidden_states.permute(0, 2, 1).unsqueeze(2).to(torch.float32)
        ) * custom.per_layer_model_projection_scale
        per_layer_proj = per_layer_proj.squeeze(2).permute(0, 2, 1)

        normed_slices = []
        for li in range(custom.num_layers):
            s = li * custom.per_layer_dim
            e = s + custom.per_layer_dim
            normed_slices.append(custom.per_layer_projection_norm(per_layer_proj[:, :, s:e]))
        per_layer_proj_normed = torch.cat(normed_slices, dim=-1)
        per_layer_combined = (per_layer_proj_normed + per_layer_raw) * custom.per_layer_input_scale

        # RoPE
        positions = torch.arange(N, device=device)
        cos_s = custom.cos_sliding[positions].unsqueeze(0).unsqueeze(0).to(torch.float32)
        sin_s = custom.sin_sliding[positions].unsqueeze(0).unsqueeze(0).to(torch.float32)
        cos_f = custom.cos_full[positions].unsqueeze(0).unsqueeze(0).to(torch.float32)
        sin_f = custom.sin_full[positions].unsqueeze(0).unsqueeze(0).to(torch.float32)

        # Masks
        ctx = config.context_length
        causal_mask = torch.full((1, 1, N, ctx), -1e9, device=device, dtype=torch.float32)
        for i in range(N):
            causal_mask[0, 0, i, :i+1] = 0.0

        update_mask = torch.zeros(1, 1, ctx, 1, device=device, dtype=torch.float32)
        update_mask[:, :, :N, :] = 1.0

        # KV stores
        kv_store_13_k = torch.zeros(1, 1, ctx, 256, dtype=torch.float32, device=device)
        kv_store_13_v = torch.zeros(1, 1, ctx, 256, dtype=torch.float32, device=device)
        kv_store_14_k = torch.zeros(1, 1, ctx, 512, dtype=torch.float32, device=device)
        kv_store_14_v = torch.zeros(1, 1, ctx, 512, dtype=torch.float32, device=device)

        custom_hiddens.append(hidden_states[0, -1].clone())  # post-embedding

        for layer_idx in range(num_layers):
            layer = custom.layers[layer_idx]
            is_full = config.is_full_attention(layer_idx)
            hd = config.get_head_dim(layer_idx)
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            n_rep = num_heads // num_kv_heads
            is_kv_shared = config.is_kv_shared(layer_idx)

            residual = hidden_states
            h = layer.input_layernorm(hidden_states)
            x = h.permute(0, 2, 1).unsqueeze(2)

            # Q
            q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, N).permute(0, 1, 3, 2)
            q = layer.self_attn["q_norm"](q.reshape(num_heads, N, hd)).view(1, num_heads, N, hd)
            if is_full:
                q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
            else:
                q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)

            if not is_kv_shared:
                k = layer.self_attn["k_proj"](x).view(1, num_kv_heads, hd, N).permute(0, 1, 3, 2)
                k = layer.self_attn["k_norm"](k.reshape(num_kv_heads, N, hd)).view(1, num_kv_heads, N, hd)
                v = layer.self_attn["v_proj"](x).view(1, num_kv_heads, hd, N).permute(0, 1, 3, 2)
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

                K_cache = custom.kv_cache_0[layer_idx].unsqueeze(0).to(torch.float32)
                V_cache = custom.kv_cache_0[num_layers + layer_idx].unsqueeze(0).to(torch.float32)
                K_new = K_cache * (1 - update_mask) + k_padded.expand_as(K_cache) * update_mask
                V_new = V_cache * (1 - update_mask) + v_padded.expand_as(V_cache) * update_mask
                custom.kv_cache_0.data[layer_idx] = K_new.squeeze(0).to(custom.kv_cache_0.dtype)
                custom.kv_cache_0.data[num_layers + layer_idx] = V_new.squeeze(0).to(custom.kv_cache_0.dtype)
                K_for_attn = K_new[..., :hd]
                V_for_attn = V_new[..., :hd]

                if layer_idx == 13:
                    kv_store_13_k = K_new[..., :256]
                    kv_store_13_v = V_new[..., :256]
                elif layer_idx == 14:
                    kv_store_14_k = K_new[..., :512]
                    kv_store_14_v = V_new[..., :512]
            else:
                if is_full:
                    K_for_attn = kv_store_14_k[..., :hd]
                    V_for_attn = kv_store_14_v[..., :hd]
                else:
                    K_for_attn = kv_store_13_k[..., :hd]
                    V_for_attn = kv_store_13_v[..., :hd]

            K_expanded = K_for_attn.repeat_interleave(n_rep, dim=1)
            V_expanded = V_for_attn.repeat_interleave(n_rep, dim=1)

            attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
            attn_weights = attn_weights + causal_mask[:, :, :N, :]
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, V_expanded)

            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, N, -1)
            attn_output = layer.self_attn["o_proj"](
                attn_output.permute(0, 2, 1).unsqueeze(2)
            ).squeeze(2).permute(0, 2, 1)
            attn_output = layer.post_attention_layernorm(attn_output)
            hidden_states = residual + attn_output

            # MLP
            residual = hidden_states
            h = layer.pre_feedforward_layernorm(hidden_states)
            x_mlp = h.permute(0, 2, 1).unsqueeze(2)
            gate = layer.mlp["gate_proj"](x_mlp)
            up = layer.mlp["up_proj"](x_mlp)
            gate = F.gelu(gate, approximate="tanh")
            mlp_out = layer.mlp["down_proj"](gate * up)
            hidden_states = mlp_out.squeeze(2).permute(0, 2, 1)
            hidden_states = layer.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states

            # PLE
            residual_pl = hidden_states
            s = layer_idx * custom.per_layer_dim
            e = s + custom.per_layer_dim
            per_layer_slice = per_layer_combined[:, :, s:e]
            hs_conv = hidden_states.permute(0, 2, 1).unsqueeze(2)
            gated = layer.per_layer_input_gate(hs_conv)
            gated = F.gelu(gated, approximate="tanh")
            per_layer_slice_conv = per_layer_slice.permute(0, 2, 1).unsqueeze(2)
            gated = gated * per_layer_slice_conv
            gated = layer.per_layer_projection(gated)
            gated = gated.squeeze(2).permute(0, 2, 1)
            hidden_states = layer.post_per_layer_input_norm(gated)
            hidden_states = residual_pl + hidden_states
            hidden_states = hidden_states * layer.layer_scalar.to(torch.float32)

            custom_hiddens.append(hidden_states[0, -1].clone())

        custom_final = custom.norm(hidden_states)

    # ========== Phase 3: Compare ==========
    print(f"\n{'='*80}")
    print(f"{'Layer':>6} | {'HF norm':>12} | {'Custom norm':>12} | {'Rel diff':>12} | {'Max abs':>12}")
    print(f"{'-'*80}")
    for i in range(min(len(hf_hidden), len(custom_hiddens))):
        hf_h = hf_hidden[i]
        cu_h = custom_hiddens[i]
        hf_n = hf_h.norm().item()
        cu_n = cu_h.norm().item()
        rel = (hf_h - cu_h).norm().item() / max(hf_n, 1e-12)
        maxabs = (hf_h - cu_h).abs().max().item()
        label = f"L{i-1}" if i > 0 else "embed"
        marker = " ***" if rel > 0.01 else ""
        print(f"  {label:>6} | {hf_n:12.4f} | {cu_n:12.4f} | {rel:12.6f} | {maxabs:12.6f}{marker}")

    hf_fn = hf_last.norm().item()
    cu_fn = custom_final[0, -1].float().norm().item()
    diff_fn = (hf_last - custom_final[0, -1].float()).norm().item()
    print(f"\n  Final (post-norm): HF={hf_fn:.4f} Custom={cu_fn:.4f} rel_diff={diff_fn/max(hf_fn,1e-12):.6f}")

    # layer_scalar check
    print(f"\n=== layer_scalar values (only non-1.0 or mismatched) ===")
    for i in range(num_layers):
        cu_ls = custom.layers[i].layer_scalar.item()
        hf_ls = hf_layer_scalars[i]
        if abs(cu_ls - hf_ls) > 1e-4:
            print(f"  Layer {i}: custom={cu_ls:.6f} HF={hf_ls:.6f} ** MISMATCH **")
        elif abs(cu_ls - 1.0) > 0.005:
            print(f"  Layer {i}: {cu_ls:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
