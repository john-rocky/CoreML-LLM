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
    dtype=torch.float16,
):
    """Generate RoPE cos/sin tables matching Gemma 4 config."""
    t = torch.arange(max_positions).float()

    # Sliding attention RoPE (full rotation, theta=10000, head_dim=256)
    hd_s = sliding_head_dim
    inv_freq_s = 1.0 / (sliding_theta ** (torch.arange(0, hd_s, 2).float() / hd_s))
    freqs_s = torch.einsum("i,j->ij", t, inv_freq_s)
    emb_s = torch.cat((freqs_s, freqs_s), dim=-1)  # (max_pos, 256)

    # Full attention RoPE (proportional, theta=1M, head_dim=512)
    hd_f = full_head_dim
    inv_freq_f = 1.0 / (full_theta ** (torch.arange(0, hd_f, 2).float() / hd_f))
    freqs_f = torch.einsum("i,j->ij", t, inv_freq_f)
    emb_f = torch.cat((freqs_f, freqs_f), dim=-1)  # (max_pos, 512)

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
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Generating RoPE tables for {args.max_positions} positions...")
    tables = generate_rope_tables(max_positions=args.max_positions)

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
