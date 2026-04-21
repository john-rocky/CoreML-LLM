#!/usr/bin/env python3
"""Compare HF bf16 Gemma-4 E2B argmax vs deployed W4A8 mlpackage chunks
argmax on the SAME input sequence. Used to decide whether retraining the
EAGLE-3 draft against W4A8 targets will actually help.

If agreement is low (<50%), the 0% accept rate on iPhone is driven by
quantization divergence — retraining makes sense. If agreement is high
(>80%), retraining won't help; look elsewhere (draft bug, fusion input
staleness, speculative loop semantics).

Usage:
    python conversion/compare_bf16_vs_w4a8.py \
        --hf /path/to/hf_model \
        --chunks /path/to/output/eagle3-chunks \
        --assets /path/to/sideload-bundle \
        --prompt "The capital of Japan is" \
        --max-pos 128
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the ChunkRunner from the collector.
sys.path.insert(0, str(Path(__file__).parent))
from collect_eagle_hidden_states_w4a8 import (  # noqa: E402
    ChunkRunner, QuantEmbed, PerLayerRawEmbed,
    load_rope_table, rope_row,
    HIDDEN, PLD, NUM_LAYERS, VOCAB, EMBED_SCALE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, required=True,
                    help="HF Gemma-4 E2B checkpoint dir")
    ap.add_argument("--chunks", type=Path, required=True,
                    help="Dir with chunk{1..4}.mlpackage")
    ap.add_argument("--assets", type=Path, required=True,
                    help="Dir with embed_tokens_*.bin, cos/sin_*.npy, hf_model/")
    ap.add_argument("--prompt", type=str, default=None,
                    help="Text to tokenize + compare. Prefer --corpus for chat-formatted input.")
    ap.add_argument("--corpus", type=Path, default=None,
                    help="JSONL corpus; uses the first entry's text as the prompt.")
    ap.add_argument("--max-pos", type=int, default=128,
                    help="Stop comparing after this many positions")
    ap.add_argument("--device", type=str, default="cpu",
                    help="PyTorch device for HF model (cpu / mps / cuda)")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(str(args.assets / "hf_model"))
    if args.corpus is not None:
        import json
        with open(args.corpus) as f:
            prompt_text = json.loads(f.readline())["text"]
    elif args.prompt is not None:
        prompt_text = args.prompt
    else:
        prompt_text = "The capital of Japan is Tokyo. It is a bustling metropolis with over 13 million residents."
    ids = tokenizer.encode(prompt_text, return_tensors="pt")[0].tolist()
    N = min(len(ids), args.max_pos)
    ids = ids[:N]
    print(f"[Prompt] {N} tokens from {'corpus' if args.corpus else 'prompt arg'}")

    # ── HF bf16 forward ──────────────────────────────────────────
    print(f"\n[HF] Loading bf16 model from {args.hf} ({args.device}) ...")
    t0 = time.time()
    hf = AutoModelForCausalLM.from_pretrained(str(args.hf), dtype=torch.float16)
    hf.eval().to(args.device)
    print(f"[HF] loaded in {time.time()-t0:.1f}s")

    with torch.no_grad():
        t0 = time.time()
        out = hf(input_ids=torch.tensor(ids, device=args.device).unsqueeze(0),
                 use_cache=False)
        # logits shape (1, N, vocab). argmax at each position = prediction for
        # the NEXT position.
        hf_argmax = out.logits[0].argmax(dim=-1).cpu().numpy().astype(np.int64)
        print(f"[HF] forward {N} tokens in {time.time()-t0:.1f}s")
    del hf
    torch.cuda.empty_cache() if args.device == "cuda" else None

    # ── W4A8 mlpackage forward via ChunkRunner ───────────────────
    print(f"\n[W4A8] Loading deployed chunks ...")
    runner = ChunkRunner(args.chunks, ctx=2048, W=512)
    embed = QuantEmbed(
        args.assets / "embed_tokens_q8.bin",
        args.assets / "embed_tokens_scales.bin",
        vocab=VOCAB, dim=HIDDEN, embed_scale=EMBED_SCALE)
    ple = PerLayerRawEmbed(
        args.assets / "embed_tokens_per_layer_q8.bin",
        args.assets / "embed_tokens_per_layer_scales.bin",
        vocab=VOCAB, per_layer_dim=PLD, num_layers=NUM_LAYERS)
    cos_s_table = load_rope_table(args.assets / "cos_sliding.npy")
    sin_s_table = load_rope_table(args.assets / "sin_sliding.npy")
    cos_f_table = load_rope_table(args.assets / "cos_full.npy")
    sin_f_table = load_rope_table(args.assets / "sin_full.npy")

    w4a8_argmax = np.empty(N, dtype=np.int64)
    t0 = time.time()
    for pos, tok in enumerate(ids):
        hid = embed.lookup(tok).reshape(1, 1, HIDDEN).astype(np.float16)
        plr = ple.lookup(tok).reshape(1, 1, -1).astype(np.float16)
        cos_s = rope_row(cos_s_table, pos, dim=256)
        sin_s = rope_row(sin_s_table, pos, dim=256)
        cos_f = rope_row(cos_f_table, pos, dim=512)
        sin_f = rope_row(sin_f_table, pos, dim=512)
        _, _, _, argmax = runner.step(
            hidden_states=hid, per_layer_raw=plr, position=pos,
            cos_s=cos_s, sin_s=sin_s, cos_f=cos_f, sin_f=sin_f)
        w4a8_argmax[pos] = argmax
    print(f"[W4A8] {N} tokens in {time.time()-t0:.1f}s ({N/max(time.time()-t0,1e-6):.1f} tok/s)")

    # ── Compare ──────────────────────────────────────────────────
    print("\n── Compare ──")
    match = hf_argmax == w4a8_argmax
    agree = float(match.mean())
    print(f"Agreement: {match.sum()}/{N} ({agree*100:.1f}%)")

    # First disagreement
    diff_idx = np.where(~match)[0]
    if len(diff_idx) > 0:
        print(f"First disagreement at position {diff_idx[0]}:")
        for i in diff_idx[:8]:
            tok_input_str = tokenizer.decode([ids[i]]).replace("\n", "\\n")
            tok_hf_str = tokenizer.decode([int(hf_argmax[i])]).replace("\n", "\\n")
            tok_w4a8_str = tokenizer.decode([int(w4a8_argmax[i])]).replace("\n", "\\n")
            print(f"  pos {i} (input={tok_input_str!r}): "
                  f"HF argmax={hf_argmax[i]} ({tok_hf_str!r}) vs "
                  f"W4A8 argmax={w4a8_argmax[i]} ({tok_w4a8_str!r})")

    print("\n── Verdict ──")
    if agree >= 0.80:
        print(f"HIGH agreement ({agree*100:.0f}%): retraining probably won't "
              f"fix 0% accept. Look elsewhere (draft code, fusion timing, "
              f"speculative loop semantics).")
    elif agree >= 0.50:
        print(f"MEDIUM agreement ({agree*100:.0f}%): quantization causes some "
              f"argmax flips; retraining will help partially but may not hit "
              f"target 40-50% on-device accept.")
    else:
        print(f"LOW agreement ({agree*100:.0f}%): quantization divergence is "
              f"severe — retraining against W4A8 should close the gap.")


if __name__ == "__main__":
    main()
