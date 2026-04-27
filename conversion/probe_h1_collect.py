#!/usr/bin/env python3
"""H1 probe: collect HF Gemma 4 E2B hidden states for K-future prediction analysis.

Captures hidden_states at layers 13 (sliding KV producer), 14 (full KV producer),
34 (LM head input), aligned with tokens at position+0..+3, for downstream linear
probe training. Goal: determine whether HF base hidden state encodes K=2/K=3
future-token information, distinguishing MTP-aware (LiteRT-style) vs. scrubbed.

Usage:
    conversion/.venv/bin/python conversion/probe_h1_collect.py \
        --hf-dir output/gemma4-e2b/hf_model \
        --output /tmp/h1_probe/data.npz \
        --num-tokens 50000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


CORPUS_FILES_GLOB = ["docs/*.md", "README.md", "*.md"]


def gather_corpus_repo(repo_root: str, target_chars: int = 1_500_000) -> str:
    """Concatenate repository markdown for offline corpus (fallback / heterogeneous)."""
    chunks = []
    total = 0
    for pat in CORPUS_FILES_GLOB:
        for p in Path(repo_root).glob(pat):
            try:
                t = p.read_text(encoding="utf-8")
            except Exception:
                continue
            chunks.append(t)
            total += len(t)
            if total >= target_chars:
                break
        if total >= target_chars:
            break
    return "\n\n".join(chunks)


def gather_corpus_wikitext(target_chars: int = 1_500_000) -> str:
    """Natural-English baseline corpus from cached wikitext-2."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1",
                      split="train", download_mode="reuse_cache_if_exists")
    chunks = []
    total = 0
    for row in ds:
        t = row["text"]
        if len(t) < 32:
            continue
        chunks.append(t)
        total += len(t)
        if total >= target_chars:
            break
    return "\n".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="output/gemma4-e2b/hf_model")
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-tokens", type=int, default=50_000)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--device", default="cpu",
                    help="cpu / mps. fp16 on cpu is slower but stable.")
    ap.add_argument("--layers", default="13,14,34",
                    help="Layer indices to capture (post-layer hidden_state).")
    ap.add_argument("--corpus", default="repo", choices=["repo", "wikitext"])
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layers_to_capture = [int(x) for x in args.layers.split(",")]
    print(f"Capturing layers: {layers_to_capture}")

    print(f"\nLoading HF Gemma 4 E2B from {args.hf_dir}...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.hf_dir)
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, torch_dtype=dtype,
        attn_implementation="eager",
    )
    model.eval()
    if args.device == "mps":
        model = model.to("mps")
    print(f"  loaded in {time.time()-t0:.1f}s, dtype={dtype}, device={args.device}")

    text_model = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    hidden_size = text_model.config.hidden_size
    num_layers = text_model.config.num_hidden_layers
    vocab_size = text_model.config.vocab_size
    print(f"  hidden_size={hidden_size}, num_layers={num_layers}, vocab={vocab_size}")
    for li in layers_to_capture:
        assert 0 <= li < num_layers, f"layer {li} out of range [0, {num_layers})"

    # Build corpus
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_chars = args.num_tokens * 6
    if args.corpus == "wikitext":
        print(f"\nGathering wikitext-2 ({target_chars} chars target)...")
        corpus_text = gather_corpus_wikitext(target_chars=target_chars)
    else:
        print(f"\nGathering corpus from {repo_root}...")
        corpus_text = gather_corpus_repo(repo_root, target_chars=target_chars)
    ids = tok.encode(corpus_text, add_special_tokens=False)
    print(f"  corpus = {len(corpus_text)} chars → {len(ids)} tokens")

    # Trim / extend to ceil(num_tokens / seq_len) sequences of seq_len each
    target_seqs = (args.num_tokens + args.seq_len - 1) // args.seq_len
    needed_tokens = target_seqs * args.seq_len
    if len(ids) < needed_tokens:
        # Repeat the corpus
        rep = (needed_tokens + len(ids) - 1) // len(ids)
        ids = (ids * rep)[:needed_tokens]
    else:
        ids = ids[:needed_tokens]
    print(f"  using {target_seqs} sequences × {args.seq_len} = {needed_tokens} tokens")

    # Storage. One row per position. K_FUTURE=3 means we drop the last 3 positions.
    K_FUTURE = 3
    rows_per_seq = args.seq_len - K_FUTURE
    total_rows = target_seqs * rows_per_seq
    print(f"  effective probe positions: {total_rows} (= {target_seqs} × {rows_per_seq})")

    hiddens = {li: np.zeros((total_rows, hidden_size), dtype=np.float16)
               for li in layers_to_capture}
    targets = np.zeros((total_rows, K_FUTURE + 1), dtype=np.int32)  # token at +0..+K
    row = 0

    captured = {}
    hooks = []
    for li in layers_to_capture:
        layer_module = text_model.layers[li]
        def make_hook(layer_idx):
            def hook(module, inputs, output):
                # Output is tuple in some HF impls; first element is hidden_states.
                hs = output[0] if isinstance(output, (tuple, list)) else output
                captured[layer_idx] = hs.detach()
            return hook
        hooks.append(layer_module.register_forward_hook(make_hook(li)))

    print("\nRunning forwards...")
    t0 = time.time()
    with torch.no_grad():
        for s in range(target_seqs):
            seq = ids[s * args.seq_len:(s + 1) * args.seq_len]
            input_ids = torch.tensor([seq], dtype=torch.long)
            if args.device == "mps":
                input_ids = input_ids.to("mps")

            captured.clear()
            _ = model(input_ids=input_ids)

            seq_arr = np.asarray(seq, dtype=np.int32)
            for k_off in range(K_FUTURE + 1):
                targets[row:row + rows_per_seq, k_off] = seq_arr[k_off:k_off + rows_per_seq]
            for li in layers_to_capture:
                h = captured[li][0].cpu().to(torch.float16).numpy()  # (T, hidden)
                hiddens[li][row:row + rows_per_seq] = h[:rows_per_seq]
            row += rows_per_seq

            if (s + 1) % 5 == 0 or s == target_seqs - 1:
                el = time.time() - t0
                print(f"  seq {s+1}/{target_seqs}  rows={row}  elapsed={el:.1f}s  "
                      f"({row/el:.0f} rows/s)")

    for h in hooks:
        h.remove()

    print(f"\nSaving to {args.output}...")
    np.savez_compressed(args.output,
                        targets=targets[:row],
                        **{f"hidden_L{li}": hiddens[li][:row] for li in layers_to_capture},
                        layers=np.array(layers_to_capture, dtype=np.int32),
                        hidden_size=np.int32(hidden_size),
                        vocab_size=np.int32(vocab_size))
    sz = os.path.getsize(args.output) / 1e6
    print(f"  saved {sz:.1f} MB ({row} rows)")
    print(f"  total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
