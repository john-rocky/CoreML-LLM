#!/usr/bin/env python3
"""Measure how much the intermediate hidden states at fusion layers
(L8, L17, L34) diverge between bf16 HF Gemma-4-E2B and the W4A8
CoreML-deployed chunk{1..4}.mlpackage.

Per-position per-layer:
  cos_sim  — directional similarity (1 = identical direction)
  rel_err  — ||bf16 - w4a8|| / ||bf16||
  ||h||    — raw norms

Also compares the final argmax (same as compare_bf16_vs_w4a8.py) to
check that even when argmax agrees, the hiddens that FEED the draft
may already be far apart — which is what would kill a draft trained
on bf16 hiddens.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from collect_eagle_hidden_states_w4a8 import (  # noqa: E402
    ChunkRunner, QuantEmbed, PerLayerRawEmbed,
    load_rope_table, rope_row,
    HIDDEN, PLD, NUM_LAYERS, VOCAB, EMBED_SCALE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--max-pos", type=int, default=96)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(str(args.assets / "hf_model"))
    if args.corpus:
        with open(args.corpus) as f:
            text = json.loads(f.readline())["text"]
    else:
        text = args.prompt or "Tell me about the history of Japan briefly. It has many fascinating eras."
    ids = tokenizer.encode(text)[:args.max_pos]
    N = len(ids)
    print(f"[Prompt] {N} tokens")

    # ── HF bf16 forward with hidden_states output ──
    print(f"\n[HF] Loading bf16 from {args.hf}")
    t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(str(args.hf), dtype=torch.float16)
    hf.eval().to("cpu")
    print(f"[HF] loaded in {time.time()-t0:.1f}s")

    with torch.no_grad():
        out = hf(input_ids=torch.tensor(ids).unsqueeze(0),
                 output_hidden_states=True, use_cache=False)
        # all_h[0] = post-embed; all_h[i+1] = output of layer i.
        all_h = out.hidden_states
        # Fusion layers are [8, 17, 34] → indexes 9, 18, 35.
        bf16_L8  = all_h[9][0].float().numpy()   # (N, H)
        bf16_L17 = all_h[18][0].float().numpy()
        bf16_L34 = all_h[35][0].float().numpy()
        bf16_argmax = out.logits[0].argmax(dim=-1).cpu().numpy().astype(np.int64)
    del hf
    print(f"[HF] captured hiddens  L8={bf16_L8.shape}  L17={bf16_L17.shape}  L34={bf16_L34.shape}")

    # ── W4A8 CoreML forward via ChunkRunner ──
    print(f"\n[W4A8] Loading CoreML chunks")
    decoder = ChunkRunner(args.chunks, ctx=2048, W=512)
    embed = QuantEmbed(
        args.assets / "embed_tokens_q8.bin",
        args.assets / "embed_tokens_scales.bin",
        vocab=VOCAB, dim=HIDDEN, embed_scale=EMBED_SCALE)
    ple = PerLayerRawEmbed(
        args.assets / "embed_tokens_per_layer_q8.bin",
        args.assets / "embed_tokens_per_layer_scales.bin",
        vocab=VOCAB, per_layer_dim=PLD, num_layers=NUM_LAYERS)
    cos_s_tbl = load_rope_table(args.assets / "cos_sliding.npy")
    sin_s_tbl = load_rope_table(args.assets / "sin_sliding.npy")
    cos_f_tbl = load_rope_table(args.assets / "cos_full.npy")
    sin_f_tbl = load_rope_table(args.assets / "sin_full.npy")

    w4a8_L8  = np.empty((N, HIDDEN), dtype=np.float32)
    w4a8_L17 = np.empty((N, HIDDEN), dtype=np.float32)
    w4a8_L34 = np.empty((N, HIDDEN), dtype=np.float32)
    w4a8_argmax = np.empty(N, dtype=np.int64)

    t0 = time.time()
    for pos, tok in enumerate(ids):
        hid = embed.lookup(int(tok)).reshape(1, 1, HIDDEN).astype(np.float16)
        plr = ple.lookup(int(tok)).reshape(1, 1, -1).astype(np.float16)
        cos_s = rope_row(cos_s_tbl, pos, dim=256)
        sin_s = rope_row(sin_s_tbl, pos, dim=256)
        cos_f = rope_row(cos_f_tbl, pos, dim=512)
        sin_f = rope_row(sin_f_tbl, pos, dim=512)
        h_L8, h_L17, h_L34, argmax = decoder.step(
            hidden_states=hid, per_layer_raw=plr, position=pos,
            cos_s=cos_s, sin_s=sin_s, cos_f=cos_f, sin_f=sin_f)
        w4a8_L8[pos]  = h_L8.astype(np.float32)
        w4a8_L17[pos] = h_L17.astype(np.float32)
        w4a8_L34[pos] = h_L34.astype(np.float32)
        w4a8_argmax[pos] = argmax
    print(f"[W4A8] {N} tokens in {time.time()-t0:.1f}s")

    # ── Compare ──
    def summarize(name, a, b):
        # a, b: (N, H) float32
        cos = (a * b).sum(axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
        rel = np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(a, axis=-1) + 1e-9)
        na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        print(f"\n[{name}]")
        print(f"  ||bf16||   mean={na.mean():7.2f}  min={na.min():7.2f}  max={na.max():7.2f}")
        print(f"  ||w4a8||   mean={nb.mean():7.2f}  min={nb.min():7.2f}  max={nb.max():7.2f}")
        print(f"  cos_sim    mean={cos.mean():.4f}  min={cos.min():.4f}  max={cos.max():.4f}  p10={np.percentile(cos, 10):.4f}  p50={np.percentile(cos, 50):.4f}")
        print(f"  rel_err    mean={rel.mean():.4f}  min={rel.min():.4f}  max={rel.max():.4f}  p90={np.percentile(rel, 90):.4f}")

    summarize("L8  (chunk2 hidden_at_L8)", bf16_L8, w4a8_L8)
    summarize("L17 (chunk3 hidden_at_L17)", bf16_L17, w4a8_L17)
    summarize("L34 (chunk4 hidden_at_L34 pre-norm)", bf16_L34, w4a8_L34)

    agree = (bf16_argmax == w4a8_argmax).mean()
    print(f"\n[argmax] agreement: {agree*100:.1f}%")

    # Sanity: if cos_sim >> 0.95 at L8 but collapses by L34, the issue
    # is downstream quantization; if it's already low at L8, error is
    # injected near the embedding or early layers.


if __name__ == "__main__":
    main()
