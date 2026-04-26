"""SmoothQuant-style activation-weight smoothing for Gemma 4 chunks.

Stage 1 v3: redistribute activation outliers into weights pre-quantization
so cml9 INT8 activation quant has a tighter range to fit.

Mathematical setup (per linear op):
  y = W @ x
  y = (W * s) @ (x / s)   for any positive s ∈ R^in
  ↓ apply scaling
  W' = W * s
  x' = x / s

For Gemma 4's RMSNorm-fed linears, the (1/s) factor folds into the
preceding norm's per-channel weight:
  RMSNorm.weight ← RMSNorm.weight / s
  linear.weight  ← linear.weight * s

Choice of s per AWQ (Lin et al., 2023):
  s_i = activation_max[i]^alpha  /  weight_max[i]^(1 - alpha)

with α ∈ [0, 1]. α = 0 leaves activations untouched, α = 1 fully migrates
outlier difficulty into the weights. AWQ's empirical sweet spot is
α ≈ 0.5 for transformer activations; we expose `--awq-alpha` to tune.

Smoothing targets (per Gemma 4 layer):
  - q_proj / k_proj / v_proj: input is `input_layernorm(x)`
  - gate_proj / up_proj:      input is `pre_feedforward_layernorm(x)`
  - o_proj / down_proj:       input is intermediate (no preceding norm) — SKIPPED.
    These two linears stay unsmoothed; their per-op INT8 ε is unchanged.
    Coverage = 5/7 linears per layer × 8 layers = 40/56 = 71 %.

Usage:
  from awq_smoothing import apply_awq_to_chunk
  apply_awq_to_chunk(chunk_pytorch_module, calib_npz_path, alpha=0.5)

Operates IN-PLACE on the chunk's layer parameters (which alias `base.layers`).
After this call, the chunk produces mathematically-equivalent fp16 output
but redistributed weight/activation magnitudes — quantization downstream
should then preserve more cosine similarity vs the W4 fp16 baseline.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


# Same input names as gen_calib_data_real.py / chunk_1's forward signature.
_CHUNK1_INPUT_ORDER = (
    "hidden_states",
    "causal_mask_full",
    "causal_mask_sliding",
    "per_layer_raw",
    "cos_s",
    "sin_s",
    "cos_f",
    "sin_f",
    "current_pos",
    "ring_pos",
)


def _load_npz_samples(path: str) -> list[dict]:
    data = np.load(path)
    n = int(data["_meta_num_samples"][0])
    out = []
    for i in range(n):
        d = {}
        for name in _CHUNK1_INPUT_ORDER:
            key = f"sample_{i:03d}__{name}"
            if key in data.files:
                d[name] = data[key]
        out.append(d)
    return out


def _per_channel_input_max(x: torch.Tensor, hidden: int) -> torch.Tensor:
    """Reduce a hook-captured input tensor to a per-input-channel max-abs
    of length `hidden`. Handles both Linear (1, S, hidden) and Conv2d
    (1, hidden, H, W) input layouts.
    """
    a = x.abs()
    if a.dim() == 3 and a.shape[-1] == hidden:
        return a.amax(dim=tuple(range(a.dim() - 1)))
    if a.dim() == 4 and a.shape[1] == hidden:
        return a.amax(dim=(0, 2, 3))
    raise RuntimeError(f"unexpected hook tensor shape {tuple(x.shape)} "
                       f"for hidden={hidden}")


def _weight_per_in_channel_max(weights: list[torch.Tensor]) -> torch.Tensor:
    """For a list of [out, in, ...] weights, return per-input-channel max-abs
    aggregated across all weights. Conv2d weights (out, in, 1, 1) are
    squeezed to (out, in) first."""
    flat = []
    for w in weights:
        if w.dim() == 4:
            flat.append(w.squeeze(-1).squeeze(-1))
        else:
            flat.append(w)
    return torch.cat(flat, dim=0).abs().amax(dim=0)


def _scale_weight_in_dim(w: nn.Parameter, s: torch.Tensor) -> None:
    """w *= s along the input-channel axis. Handles Conv2d (out, in, 1, 1)
    and Linear (out, in)."""
    if w.dim() == 4:
        w.mul_(s.view(1, -1, 1, 1).to(w.dtype))
    elif w.dim() == 2:
        w.mul_(s.view(1, -1).to(w.dtype))
    else:
        raise RuntimeError(f"unexpected weight dim {w.dim()}")


def collect_chunk1_activation_stats(
    chunk: nn.Module, samples: Iterable[dict],
) -> dict[tuple[int, str], torch.Tensor]:
    """Run chunk_1 forward on calibration samples with hooks on each
    layer's q_proj and gate_proj. Returns a dict mapping
    (local_layer_idx, 'qkv' | 'gateup') → per-input-channel max-abs."""
    stats: dict[tuple[int, str], torch.Tensor] = {}
    handles = []
    hidden = chunk.config.hidden_size

    def make_hook(idx: int, kind: str):
        def hook(_module, args):
            x = args[0]
            m = _per_channel_input_max(x, hidden)
            key = (idx, kind)
            if key in stats:
                stats[key] = torch.maximum(stats[key], m.detach())
            else:
                stats[key] = m.detach().clone()
        return hook

    for local_idx, layer in enumerate(chunk.layers):
        handles.append(layer.self_attn["q_proj"].register_forward_pre_hook(
            make_hook(local_idx, "qkv")))
        handles.append(layer.mlp["gate_proj"].register_forward_pre_hook(
            make_hook(local_idx, "gateup")))

    try:
        with torch.no_grad():
            for sample in samples:
                args = []
                for name in _CHUNK1_INPUT_ORDER:
                    arr = sample[name]
                    t = torch.from_numpy(np.asarray(arr))
                    if t.dtype == torch.float16 or t.dtype == torch.float32:
                        t = t.to(torch.float16)
                    args.append(t)
                chunk(*args)
    finally:
        for h in handles:
            h.remove()
    return stats


def smooth_layer_qkv(layer: nn.Module, x_max: torch.Tensor,
                     alpha: float = 0.5, eps: float = 1e-5) -> torch.Tensor:
    """Apply AWQ scaling to q/k/v at input_layernorm. Returns the scale
    s actually used (for logging / debug)."""
    qproj = layer.self_attn["q_proj"]
    kproj = layer.self_attn["k_proj"]
    vproj = layer.self_attn["v_proj"]

    x_max_f = x_max.float().clamp(min=eps)
    w_max = _weight_per_in_channel_max(
        [p.weight for p in (qproj, kproj, vproj)]).float().clamp(min=eps)
    s = (x_max_f.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=eps)

    with torch.no_grad():
        layer.input_layernorm.weight.div_(s.to(layer.input_layernorm.weight.dtype))
        for proj in (qproj, kproj, vproj):
            _scale_weight_in_dim(proj.weight, s)
    return s


def smooth_layer_gateup(layer: nn.Module, x_max: torch.Tensor,
                         alpha: float = 0.5, eps: float = 1e-5) -> torch.Tensor:
    """Apply AWQ scaling to gate/up at pre_feedforward_layernorm."""
    gproj = layer.mlp["gate_proj"]
    uproj = layer.mlp["up_proj"]

    x_max_f = x_max.float().clamp(min=eps)
    w_max = _weight_per_in_channel_max(
        [p.weight for p in (gproj, uproj)]).float().clamp(min=eps)
    s = (x_max_f.pow(alpha) / w_max.pow(1.0 - alpha)).clamp(min=eps)

    with torch.no_grad():
        layer.pre_feedforward_layernorm.weight.div_(
            s.to(layer.pre_feedforward_layernorm.weight.dtype))
        for proj in (gproj, uproj):
            _scale_weight_in_dim(proj.weight, s)
    return s


def apply_awq_to_chunk(chunk: nn.Module, calib_npz_path: str,
                       alpha: float = 0.5) -> dict:
    """Top-level entry point. Mutates chunk's layers in-place. Returns a
    summary dict with per-layer scale stats for the converter to log.
    """
    if not os.path.exists(calib_npz_path):
        raise FileNotFoundError(f"calibration data not found: {calib_npz_path}")

    samples = _load_npz_samples(calib_npz_path)
    if not samples:
        raise RuntimeError("empty calibration samples")

    print(f"  AWQ: collecting activation stats from {len(samples)} samples "
          f"(alpha={alpha})...")
    stats = collect_chunk1_activation_stats(chunk, samples)
    print(f"  AWQ: collected stats for {len(stats)} (layer, op) pairs")

    summary = {}
    for local_idx, layer in enumerate(chunk.layers):
        layer_summary = {}
        if (local_idx, "qkv") in stats:
            s = smooth_layer_qkv(layer, stats[(local_idx, "qkv")], alpha=alpha)
            layer_summary["qkv_scale_min"] = float(s.min())
            layer_summary["qkv_scale_max"] = float(s.max())
            layer_summary["qkv_scale_median"] = float(s.median())
        if (local_idx, "gateup") in stats:
            s = smooth_layer_gateup(layer, stats[(local_idx, "gateup")], alpha=alpha)
            layer_summary["gateup_scale_min"] = float(s.min())
            layer_summary["gateup_scale_max"] = float(s.max())
            layer_summary["gateup_scale_median"] = float(s.median())
        summary[local_idx] = layer_summary

    print(f"  AWQ: smoothed {len(summary)} layers; per-layer scale ranges:")
    for idx in sorted(summary):
        ls = summary[idx]
        if "qkv_scale_min" in ls:
            print(f"    L{idx}: qkv s∈[{ls['qkv_scale_min']:.3f}, "
                  f"{ls['qkv_scale_max']:.3f}] med={ls['qkv_scale_median']:.3f}; "
                  f"gateup s∈[{ls['gateup_scale_min']:.3f}, "
                  f"{ls['gateup_scale_max']:.3f}] med={ls['gateup_scale_median']:.3f}")
    return summary
