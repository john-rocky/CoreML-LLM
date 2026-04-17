#!/usr/bin/env python3
"""Recompute tok_tgt.dat for an existing memmap-v1 dataset in fp32.

Root cause: the collector's GPU argmax patch (commit fd12229) computed
`F.linear(hiddens_B[:, 1:], lm_head_weight_gpu)` in fp16 with vocab=262144
and hidden=1536, which overflows ~some columns of the output to Inf/NaN
under fp16. The first index (usually token 0 = pad) consistently wins
the resulting argmax because the other columns become -Inf. All pairs in
the dataset end up labeled 0.

h_tgt.dat is intact (the forward pass itself is fine; only the final
vocab matmul overflowed). This script reads h_tgt in chunks, applies the
lm_head in fp32 on GPU, argmaxes, and rewrites tok_tgt.dat in place.

No change to h_tgt, e_in, fusion_L*, or the manifest. Takes ~5-10 min on
an A100 for a 7.7M-pair dataset.

Usage:
    python fix_tok_tgt.py --data /content/training_data.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _resolve_data_dir(args_data, manifest):
    sibling = args_data[:-3] + ".data" if args_data.endswith(".pt") else args_data + ".data"
    manifest_dir = manifest["data_dir"]
    for d in (sibling, manifest_dir):
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "h_tgt.dat")):
            return d
    raise SystemExit(f"ERROR: data dir not found (tried sibling={sibling}, manifest={manifest_dir})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch", type=int, default=4096,
                        help="Rows per GPU chunk (4096 × 1536 × fp32 ≈ 25 MB "
                             "intermediate, × 262144 vocab ≈ 4.3 GB logits).")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading manifest: {args.data}")
    raw = torch.load(args.data, map_location="cpu")
    if raw.get("format") != "memmap-v1":
        raise SystemExit("ERROR: manifest format is not memmap-v1.")

    data_dir = _resolve_data_dir(args.data, raw)
    print(f"  data_dir: {data_dir}")

    shapes = raw["shapes"]
    total_pairs = int(raw["total_pairs"])
    hidden_size = int(raw["hidden_size"])

    h_tgt_path = os.path.join(data_dir, "h_tgt.dat")
    tok_tgt_path = os.path.join(data_dir, "tok_tgt.dat")

    # h_tgt read-only, tok_tgt read-write (will overwrite)
    h_tgt_mm = np.memmap(h_tgt_path, dtype=np.float16, mode="r",
                         shape=shapes["h_tgt"])
    tok_tgt_mm = np.memmap(tok_tgt_path, dtype=np.int64, mode="r+",
                           shape=shapes["tok_tgt"])

    # Sanity check on broken state
    sample = tok_tgt_mm[:10000]
    frac_zero = (sample == 0).mean()
    print(f"  pre-fix % zero in first 10k: {frac_zero*100:.1f}%")

    # LM head weight (vocab, hidden), upcast to fp32 for the matmul
    lm_head = raw["lm_head_weight"].float().to(args.device)
    print(f"  lm_head shape: {tuple(lm_head.shape)}, dtype: {lm_head.dtype}")

    print(f"\nRecomputing tok_tgt for {total_pairs:,} pairs in chunks of {args.batch}...")
    t0 = 0
    import time
    start = time.time()
    for i in tqdm(range(0, total_pairs, args.batch)):
        end = min(i + args.batch, total_pairs)
        # Read chunk of h_tgt from memmap → fp32 GPU tensor
        h = torch.from_numpy(np.array(h_tgt_mm[i:end], copy=True)).to(args.device).float()
        # fp32 matmul
        logits = F.linear(h, lm_head)  # (chunk, vocab)
        tok = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
        tok_tgt_mm[i:end] = tok

    tok_tgt_mm.flush()
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s. Verifying...")

    # Post-fix verification
    sample = tok_tgt_mm[:10000]
    uniq = np.unique(sample)
    frac_zero = (sample == 0).mean()
    print(f"  post-fix unique values (first 10k): {len(uniq)}")
    print(f"  post-fix % zero: {frac_zero*100:.2f}%")
    print(f"  sample tokens: {sample[:20]}")

    if frac_zero > 0.9:
        raise SystemExit("ERROR: still mostly zeros after fix — inspect h_tgt / lm_head_weight.")

    print(f"\ntok_tgt.dat rewritten. Now re-run TTT training:")
    print(f"  python train_eagle3_ttt.py --data {args.data} --save-dir <dir> --epochs 2 --preload")


if __name__ == "__main__":
    raise SystemExit(main())
