#!/usr/bin/env python3
"""AWQ-style activation-aware smoothing on Gemma4Model.

Pipeline (per AWQ paper, alpha=0.5 by default):
  1. Run Gemma4Model on calibration data (chat-templated prompts).
  2. Hook per-Linear/Conv input, collect per-channel max abs activation.
  3. For each (norm → linear-group) pair, compute per-input-channel scale
        s_i = act_max[i] ** alpha  / weight_max[i] ** (1 - alpha)
     clamped to [eps, large].
  4. Apply equivalent transform: weight *= s (along input-dim),
     preceding norm.weight /= s.
  5. Save modified state dict for build_verify_chunks.py to load.

Math is bit-equivalent (modulo fp16 rounding) — activations get
divided by s before the linear, weights get multiplied by s, so
linear output is unchanged. After this transform, kmeans palettize
clusters land on weight values whose dynamic range is more uniform,
reducing per-channel quantization error.

Usage:
    .mtp_venv/bin/python conversion/awq_smooth_gemma4.py \\
        --hf-dir <gemma-4-E2B-it snapshot> \\
        --out-state /tmp/gemma4_e2b_awq_state.pt \\
        --n-calib 16 --alpha 0.5
"""
from __future__ import annotations
import argparse
import os
import sys
import torch
import torch.nn as nn
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gemma4 import Gemma4Model

CALIB_PROMPTS = [
    "Write a Python function that computes the Fibonacci sequence iteratively.",
    "Explain quantum entanglement in two sentences.",
    "List the capitals of France, Germany, Italy, and Spain.",
    "def quicksort(arr):",
    "Translate to Japanese: Hello, how are you today?",
    "Summarize the plot of Hamlet in 50 words.",
    "What is the boiling point of water at sea level?",
    "Write a JSON object describing a person with name, age, email.",
    "Compose a haiku about autumn leaves.",
    "Solve: 2x + 7 = 23. What is x?",
    "Compare HTTP/1.1 and HTTP/2 in three bullets.",
    "Refactor this code to use list comprehension: result = []\\nfor x in nums: result.append(x*2)",
    "Describe the structure of DNA in one paragraph.",
    "What are the ACID properties of databases?",
    "Generate a regex matching valid email addresses.",
    "Write a simple Express.js GET endpoint.",
]


def _hook_activation_stats(model):
    """Register forward hooks on every Linear/Conv2d, return dict of per-channel max abs."""
    act_max = defaultdict(lambda: None)

    def make_hook(name):
        def hook(module, inputs, output):
            x = inputs[0]
            # Conv2d input is (N, C, H, W); Linear input is (..., C). Per-input-channel max.
            if x.ndim == 4:
                # NCHW: per-C max abs
                amax = x.detach().abs().amax(dim=(0, 2, 3))
            elif x.ndim >= 2:
                amax = x.detach().abs().amax(
                    dim=tuple(range(x.ndim - 1)))  # all but last dim
            else:
                return
            amax = amax.float().cpu()
            if act_max[name] is None:
                act_max[name] = amax
            else:
                # take running max across calibration samples
                if amax.shape == act_max[name].shape:
                    act_max[name] = torch.maximum(act_max[name], amax)
                else:
                    pass  # shape changes — skip
        return hook

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)) and m.weight.dim() in (2, 4):
            handles.append(m.register_forward_hook(make_hook(name)))
    return act_max, handles


def _attention_layer_pairs(model):
    """For each decoder layer, yield (norm_module, [linear_modules], group_label).

    Gemma4DecoderLayer has 4 norm→linear groups:
      input_layernorm        → [q_proj, k_proj, v_proj]
      pre_feedforward_layernorm → [gate_proj, up_proj]
      post_per_layer_input_norm — separate path; harder to smooth
    Per-layer-input-gate path is also smoothing candidate but skip for v1.
    """
    for li, layer in enumerate(model.layers):
        # Attention norm → q,k,v
        if hasattr(layer, "input_layernorm"):
            qkv = []
            sa = layer.self_attn
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                if proj_name in sa:
                    qkv.append(sa[proj_name])
            if qkv:
                yield layer.input_layernorm, qkv, f"L{li}.attn"

        # Pre-FFN norm → gate, up
        if hasattr(layer, "pre_feedforward_layernorm"):
            mlp_in = []
            for proj_name in ("gate_proj", "up_proj"):
                if proj_name in layer.mlp:
                    mlp_in.append(layer.mlp[proj_name])
            if mlp_in:
                yield layer.pre_feedforward_layernorm, mlp_in, f"L{li}.mlp"


def _conv_or_linear_input_max(linear_module: nn.Module, x_max: torch.Tensor) -> torch.Tensor:
    """Reshape x_max to match weight's input-channel dim."""
    return x_max


def _compute_smooth_scales(act_max, model, alpha=0.5, eps=1e-5):
    """For each (norm, linears) pair, compute per-input-channel scale.

    Per AWQ: s_i = clip(act_max[i]^alpha / weight_max[i]^(1 - alpha))
    Combined linears (q,k,v) share the same input → take max of their weight maxes.
    """
    scales = {}  # group_label -> Tensor[input_dim]
    name_lookup = {id(m): n for n, m in model.named_modules()}

    for norm, linears, label in _attention_layer_pairs(model):
        norm_name = name_lookup.get(id(norm))
        if norm_name is None:
            continue
        # Find a hooked linear under this group; act_max key uses linear's name.
        first_lin_name = name_lookup.get(id(linears[0]))
        if first_lin_name is None or act_max.get(first_lin_name) is None:
            continue
        x_max = act_max[first_lin_name]  # per input channel
        # Combined weight max across linears (input-channel-wise).
        weight_maxes = []
        for lin in linears:
            w = lin.weight.detach().float()
            if w.dim() == 4:  # Conv2d (out, in, 1, 1)
                wmax = w.squeeze(-1).squeeze(-1).abs().amax(dim=0)
            else:  # Linear (out, in)
                wmax = w.abs().amax(dim=0)
            weight_maxes.append(wmax)
        w_max = torch.stack(weight_maxes).amax(dim=0).float().cpu()
        # AWQ scale formula
        s = (x_max.clamp_min(eps) ** alpha) / (w_max.clamp_min(eps) ** (1.0 - alpha))
        s = s.clamp_min(eps)
        scales[label] = (norm, linears, s)
    return scales


def _apply_smooth(scales):
    """In-place: norm.weight /= s, weight *= s (along input dim)."""
    for label, (norm, linears, s) in scales.items():
        s = s.to(norm.weight.dtype).to(norm.weight.device)
        with torch.no_grad():
            norm.weight.data /= s
            for lin in linears:
                w = lin.weight.data
                if w.dim() == 4:
                    # Conv2d: (out, in, 1, 1). Scale along dim=1 (in).
                    w *= s.view(1, -1, 1, 1)
                else:
                    # Linear: (out, in). Scale along dim=1.
                    w *= s.view(1, -1)
        # bias is unaffected
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default=os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/"
        "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"))
    ap.add_argument("--out-state", required=True)
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--ctx", type=int, default=64)
    args = ap.parse_args()

    print(f"Loading Gemma4Model from {args.hf_dir}")
    model = Gemma4Model.from_pretrained(args.hf_dir, context_length=args.ctx).eval()
    # Calibration in fp32 to avoid fp16 overflow in residual stream.
    # We snapshot fp16 weights first so the saved state dict goes back to
    # the production fp16 dtype after smoothing.
    print("  Snapshotting fp16 weights, then casting model to fp32 for calibration")
    fp16_weights = {n: p.data.clone() for n, p in model.named_parameters()
                    if p.dtype == torch.float16}
    fp16_buffers = {n: b.data.clone() for n, b in model.named_buffers()
                    if b.dtype == torch.float16}
    for p in model.parameters():
        p.data = p.data.float()
    for n, b in list(model.named_buffers()):
        if b.dtype == torch.float16:
            b.data = b.data.float()
    model_dtype = torch.float32
    print(f"  model dtype = {model_dtype} (calibration); will restore fp16 on save")

    print(f"Loading tokenizer (HF cache)")
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

    # Tokenize calibration prompts.
    prompts = CALIB_PROMPTS[: args.n_calib]
    print(f"Tokenizing {len(prompts)} calibration prompts")
    chat_inputs = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        # Truncate to ctx if needed
        ids = ids[:, : args.ctx]
        chat_inputs.append(ids)

    print("Hooking activations + running calibration forward passes")
    act_max, handles = _hook_activation_stats(model)

    # Stat-collection forward: bypass exact attention math (we only need
    # per-input-channel MAX magnitude entering each Linear, which depends on
    # the LayerNorm output distribution — that propagates correctly through
    # residual stream alone. We approximate attention output as zero so the
    # residual path carries the embed forward; gate_proj/up_proj/o_proj/etc.
    # see correct distributional inputs from their preceding LayerNorm.
    cfg = model.config
    if not hasattr(cfg, "is_full_attention"):
        cfg.is_full_attention = lambda i: cfg.layer_types[i] == "full_attention"
    if not hasattr(cfg, "is_kv_shared"):
        first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers
        cfg.is_kv_shared = lambda i, fs=first_shared: i >= fs
    if not hasattr(cfg, "get_head_dim"):
        cfg.get_head_dim = lambda i: cfg.global_head_dim if cfg.is_full_attention(i) else cfg.head_dim

    with torch.no_grad():
        for i, ids in enumerate(chat_inputs):
            # Single-token forward at last position is sufficient for activation-stat
            # collection (we only care about MAX magnitudes).
            tok_id = int(ids[0, -1].item())
            embed = model.embed_tokens(torch.tensor([[tok_id]])).to(model_dtype)
            embed_scaled = embed * (cfg.hidden_size ** 0.5)
            pl = model.embed_tokens_per_layer(torch.tensor([[tok_id]])).to(model_dtype)
            pl_scaled = pl * (cfg.hidden_size_per_layer_input ** 0.5)
            # Compute per_layer_combined via the model's PLE projection.
            h_conv = embed_scaled.permute(0, 2, 1).unsqueeze(2)
            proj = model.per_layer_model_projection(h_conv) * model.per_layer_model_projection_scale
            proj = proj.squeeze(2).permute(0, 2, 1)
            proj_grouped = proj.view(1, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input)
            doubled = torch.cat([proj_grouped, -proj_grouped], dim=-1)
            normed = torch.nn.functional.layer_norm(
                doubled, normalized_shape=(2 * cfg.hidden_size_per_layer_input,),
                weight=None, bias=None, eps=cfg.rms_norm_eps)
            normed, _ = torch.chunk(normed, 2, dim=-1)
            proj_normed = (normed * model.per_layer_projection_norm.weight).view(
                1, 1, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input)
            per_layer_combined = (proj_normed + pl_scaled) * model.per_layer_input_scale
            # Now drive through layers with synthetic K/V slots (zeros = empty cache).
            hidden = embed_scaled
            zd = model_dtype
            from ane_ops import apply_rotary_pos_emb
            num_heads = cfg.num_attention_heads
            num_kv = cfg.num_key_value_heads
            n_rep = num_heads // num_kv
            # Per-layer KV producer cache for shared layers.
            cached_swa_kv = {"k": None, "v": None, "hd": cfg.head_dim}
            cached_full_kv = {"k": None, "v": None, "hd": cfg.global_head_dim}
            for li in range(cfg.num_hidden_layers):
                layer = model.layers[li]
                is_full = cfg.is_full_attention(li)
                hd = cfg.get_head_dim(li)
                is_kv_shared = cfg.is_kv_shared(li)
                # === Pre-attention norm + Q/K/V proj ===
                residual = hidden
                h = layer.input_layernorm(hidden)
                x = h.permute(0, 2, 1).unsqueeze(2).to(zd)
                q = layer.self_attn["q_proj"](x).view(1, num_heads, hd, 1).permute(0, 1, 3, 2).to(zd)
                q = layer.self_attn["q_norm"](q.reshape(1, num_heads, hd)).view(1, num_heads, 1, hd)
                # Position 0: cos=1, sin=0 → q unchanged after RoPE
                if not is_kv_shared:
                    k = layer.self_attn["k_proj"](x).view(1, num_kv, hd, 1).permute(0, 1, 3, 2).to(zd)
                    v = layer.self_attn["v_proj"](x).view(1, num_kv, hd, 1).permute(0, 1, 3, 2).to(zd)
                    k = layer.self_attn["k_norm"](k.reshape(1, num_kv, hd)).view(1, num_kv, 1, hd)
                    v = v / (v.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
                    if is_full:
                        cached_full_kv["k"] = k
                        cached_full_kv["v"] = v
                    else:
                        cached_swa_kv["k"] = k
                        cached_swa_kv["v"] = v
                # Use cached KV (own or shared); for li=0 sliding, cached_swa_kv may be None.
                if is_full:
                    K_for = cached_full_kv["k"] if cached_full_kv["k"] is not None else q[:, :num_kv]
                    V_for = cached_full_kv["v"] if cached_full_kv["v"] is not None else q[:, :num_kv]
                else:
                    K_for = cached_swa_kv["k"] if cached_swa_kv["k"] is not None else q[:, :num_kv]
                    V_for = cached_swa_kv["v"] if cached_swa_kv["v"] is not None else q[:, :num_kv]
                # GQA expand
                K_exp = K_for.repeat_interleave(n_rep, dim=1)
                V_exp = V_for.repeat_interleave(n_rep, dim=1)
                # Single-position attention: weights all = softmax(q·k^T) over single key = 1.
                attn_out = V_exp  # (1, num_heads, 1, hd)
                attn_out_proxy = attn_out.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
                attn_out_proxy_c = attn_out_proxy.permute(0, 2, 1).unsqueeze(2).to(zd)
                o_out = layer.self_attn["o_proj"](attn_out_proxy_c).squeeze(2).permute(0, 2, 1)
                attn_post = layer.post_attention_layernorm(o_out)
                hidden = residual + attn_post
                # === Pre-FFN: gate/up/down ===
                residual2 = hidden
                h2 = layer.pre_feedforward_layernorm(hidden)
                x2 = h2.permute(0, 2, 1).unsqueeze(2).to(zd)
                gate_out = torch.nn.functional.gelu(layer.mlp["gate_proj"](x2), approximate="tanh")
                up_out = layer.mlp["up_proj"](x2)
                down_out = layer.mlp["down_proj"](gate_out * up_out)
                ffn_out = down_out.squeeze(2).permute(0, 2, 1)
                ffn_post = layer.post_feedforward_layernorm(ffn_out)
                hidden = residual2 + ffn_post
                # === Per-layer input gate path ===
                hs_conv = hidden.permute(0, 2, 1).unsqueeze(2).to(zd)
                gated = torch.nn.functional.gelu(
                    layer.per_layer_input_gate(hs_conv), approximate="tanh")
                # per_layer_combined slice for this layer
                s = li * cfg.hidden_size_per_layer_input
                e = s + cfg.hidden_size_per_layer_input
                pl_slice = per_layer_combined[:, :, s:e].permute(0, 2, 1).unsqueeze(2)
                gated = gated * pl_slice
                pl_proj_out = layer.per_layer_projection(gated)
                pl_proj_out = pl_proj_out.squeeze(2).permute(0, 2, 1)
                pl_post = layer.post_per_layer_input_norm(pl_proj_out)
                hidden = hidden + pl_post
                hidden = hidden * layer.layer_scalar.to(zd)
            print(f"  calib {i+1}/{len(chat_inputs)}: hidden norm={hidden.float().norm():.2f}")

    for h in handles:
        h.remove()

    # Filter: only keep entries with non-None act_max
    populated = {k: v for k, v in act_max.items() if v is not None}
    print(f"\nCollected activation stats for {len(populated)} modules")

    print(f"\nComputing AWQ scales (alpha={args.alpha})")
    scales = _compute_smooth_scales(populated, model, alpha=args.alpha)
    print(f"Computed scales for {len(scales)} (norm, linears) groups")

    print("Applying smoothing transform (still fp32)...")
    _apply_smooth(scales)

    # Cast smoothed weights back to their original fp16 dtype before saving
    # (Conv2d expects Half in build_verify_chunks).
    print("Casting smoothed weights back to fp16")
    for n, p in model.named_parameters():
        if n in fp16_weights:
            p.data = p.data.half()
    for n, b in list(model.named_buffers()):
        if n in fp16_buffers:
            b.data = b.data.half()

    # Save state dict for build_verify_chunks to use. Filter out
    # context-length-dependent buffers (kv_cache_*, cos/sin_*) — these get
    # rebuilt fresh by Gemma4Model.from_pretrained at the production ctx.
    out_dir = os.path.dirname(args.out_state) or "."
    os.makedirs(out_dir, exist_ok=True)
    SKIP_PREFIXES = ("kv_cache_", "cos_", "sin_")
    state = {k: v.clone() for k, v in model.state_dict().items()
             if not any(k.startswith(p) for p in SKIP_PREFIXES)}
    torch.save(state, args.out_state)
    print(f"\nSaved smoothed state to {args.out_state} ({len(state)} tensors, "
          f"{sum(v.numel() for v in state.values()):,} params)")
    print(f"  Approx file size: {sum(v.numel() * 2 for v in state.values()) / 1e9:.2f} GB (fp16-equiv)")


if __name__ == "__main__":
    main()
