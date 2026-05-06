#!/usr/bin/env python3
"""Generate RoPE cos/sin lookup tables for Gemma 4 SWA chunked models.

The chunked model takes cos/sin as inputs (not baked into the graph),
so we pre-compute them and save as .npy files.

Usage:
    python generate_rope.py --max-positions 32768 --output ./output/gemma4-swa/

Generates:
    cos_sliding.npy  (max_pos, 256) float16  — sliding attention, theta=10000
    sin_sliding.npy  (max_pos, 256) float16
    cos_full.npy     (max_pos, 512) float16  — full attention, theta=1000000
    sin_full.npy     (max_pos, 512) float16
"""

import argparse
import os

import numpy as np
import torch


def generate_rope_tables(
    max_positions: int = 32768,
    sliding_head_dim: int = 256,
    full_head_dim: int = 512,
    sliding_theta: float = 10000.0,
    full_theta: float = 1000000.0,
    full_partial_rotary_factor: float = 0.25,
    sliding_partial_rotary_factor: float = 1.0,
    dtype=torch.float16,
):
    """Generate RoPE cos/sin tables matching Gemma 4 config.

    HF Gemma 4 E2B applies `partial_rotary_factor=0.25` (rope_type=proportional)
    on the full-attention layer per `text_config.rope_parameters.full_attention`,
    i.e. only the first `0.25 * head_dim` channels rotate; the remainder use
    cos=1, sin=0. Earlier versions of this script defaulted to factor=1.0
    (full rotation), producing K caches that diverged from HF spec; the
    deviation was undetected because the model still produced coherent text.
    """
    t = torch.arange(max_positions).float()

    def _make(theta: float, head_dim: int, partial: float):
        rope_angles = int(partial * head_dim // 2)
        inv_rot = 1.0 / (
            theta ** (torch.arange(0, 2 * rope_angles, 2,
                                   dtype=torch.float32) / head_dim))
        nope = head_dim // 2 - rope_angles
        if nope > 0:
            inv_freq = torch.cat([inv_rot, torch.zeros(nope, dtype=torch.float32)])
        else:
            inv_freq = inv_rot
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

    emb_s = _make(sliding_theta, sliding_head_dim, sliding_partial_rotary_factor)
    emb_f = _make(full_theta, full_head_dim, full_partial_rotary_factor)

    return {
        "cos_sliding": emb_s.cos().to(dtype).numpy(),
        "sin_sliding": emb_s.sin().to(dtype).numpy(),
        "cos_full": emb_f.cos().to(dtype).numpy(),
        "sin_full": emb_f.sin().to(dtype).numpy(),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate RoPE lookup tables for Gemma 4")
    parser.add_argument("--max-positions", type=int, default=32768,
                        help="Number of positions to pre-compute (default: 32768)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for .npy files")
    parser.add_argument("--full-partial-rotary-factor", type=float, default=0.25,
                        help="HF Gemma 4 spec: 0.25 (proportional rope on full "
                             "attention). Earlier builds shipped 1.0 (full "
                             "rotation) — set explicitly only for legacy parity.")
    parser.add_argument("--sliding-partial-rotary-factor", type=float, default=1.0,
                        help="Sliding layers use full rotation (factor=1.0).")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Generating RoPE tables for {args.max_positions} positions "
          f"(full_partial_rotary_factor={args.full_partial_rotary_factor})...")
    tables = generate_rope_tables(
        max_positions=args.max_positions,
        full_partial_rotary_factor=args.full_partial_rotary_factor,
        sliding_partial_rotary_factor=args.sliding_partial_rotary_factor)

    total_bytes = 0
    for name, arr in tables.items():
        path = os.path.join(args.output, f"{name}.npy")
        np.save(path, arr)
        size = os.path.getsize(path)
        total_bytes += size
        print(f"  {name}.npy: shape={arr.shape}, {size / 1024 / 1024:.1f} MB")

    print(f"  Total: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Supports context lengths up to {args.max_positions}")


if __name__ == "__main__":
    main()
