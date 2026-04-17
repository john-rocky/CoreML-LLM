#!/usr/bin/env python3
"""Augment an existing memmap-v1 manifest with per-sequence boundary metadata.

The streaming collector writes pair-flat memmap data (one row per
(h_in, e_in, h_tgt, tok_tgt, fusion_L*) pair) but does not record
which pairs belong to the same source sequence. TTT training needs
that grouping: to train step k, the loader must fetch K+1 consecutive
pairs from the SAME sequence.

This utility re-tokenizes the corpus deterministically using the same
tokenizer / truncation / min-length rules as the collector, counts how
many valid pairs each sample contributed, and writes a `seq_starts`
cumulative-sum array back into the manifest .pt. The large .data/
tensors are NOT touched, so no re-collection is required.

Usage (Colab, after the collector has run and produced a memmap-v1
manifest):

    python augment_seq_metadata.py \
        --data /content/training_data.pt \
        --corpus /content/drive/MyDrive/eagle3_retrain_20260417/eagle_corpus.jsonl \
        --num-samples 30000 --seq-len 512

Adds:
    raw["seq_starts"]   : int64 tensor, shape (num_sequences + 1,)
    raw["num_sequences"]: int, count of sequences that contributed pairs

Checks `seq_starts[-1] == raw["total_pairs"]` — if this fails the
manifest was produced by a different collector version / run and
re-collection is required.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm


MIN_SEQ_TOKENS = 32  # must match collector's "if N < 32: continue"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to memmap-v1 manifest .pt to augment.")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Path to the eagle_corpus.jsonl used at collection.")
    parser.add_argument("--hf-dir", type=str, default=None,
                        help="HF model dir (for tokenizer). Auto-downloads "
                             "google/gemma-4-E2B-it if unset.")
    parser.add_argument("--num-samples", type=int, default=30000,
                        help="Must match the --num-samples used at collection.")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Must match the --seq-len used at collection.")
    args = parser.parse_args()

    print(f"Loading manifest: {args.data}")
    raw = torch.load(args.data, map_location="cpu")
    if raw.get("format") != "memmap-v1":
        raise SystemExit("ERROR: manifest format is not memmap-v1; re-run "
                         "the streaming collector first.")
    total_pairs_expected = int(raw["total_pairs"])
    print(f"  total_pairs in manifest: {total_pairs_expected:,}")

    if "seq_starts" in raw:
        print("  seq_starts already present — overwriting.")

    # Load tokenizer (identical to collector)
    hf_dir = args.hf_dir
    if hf_dir is None:
        from huggingface_hub import snapshot_download
        print("Downloading tokenizer from google/gemma-4-E2B-it...")
        hf_dir = snapshot_download("google/gemma-4-E2B-it",
                                   allow_patterns=["tokenizer*", "*.json"])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    print(f"  tokenizer: fast={getattr(tokenizer, 'is_fast', False)}")

    # Load corpus in the same order the collector consumed it
    print(f"Loading corpus: {args.corpus}")
    texts = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print(f"  {len(texts)} sequences in corpus")

    # Replicate the collector's per-sample logic to compute (N - 1) valid
    # pairs for each accepted sequence.
    num = min(args.num_samples, len(texts))
    seq_lengths = []
    skipped = 0

    for text in tqdm(texts[:num], desc="Tokenize"):
        ids = tokenizer.encode(text, return_tensors="pt",
                               truncation=True, max_length=args.seq_len)
        N = int(ids.shape[1])
        if N < MIN_SEQ_TOKENS:
            skipped += 1
            continue
        seq_lengths.append(N - 1)  # pairs per sequence

    num_seqs = len(seq_lengths)
    total_pairs = sum(seq_lengths)
    print(f"\nCounted:")
    print(f"  num_sequences: {num_seqs:,}")
    print(f"  total_pairs  : {total_pairs:,} (expected {total_pairs_expected:,})")
    print(f"  skipped      : {skipped:,} (N < {MIN_SEQ_TOKENS})")

    if total_pairs != total_pairs_expected:
        raise SystemExit(
            f"\nERROR: pair count mismatch.\n"
            f"  Recomputed total_pairs = {total_pairs}\n"
            f"  Manifest   total_pairs = {total_pairs_expected}\n"
            f"This means the corpus/tokenizer/args used here differ from\n"
            f"what the collector used. Re-collection is required, or fix\n"
            f"the --corpus / --num-samples / --seq-len / --hf-dir flags."
        )

    # Build cumulative index. seq_starts[i] = cursor at which seq i begins.
    # seq_starts[-1] = total_pairs (one-past-end).
    seq_starts = torch.zeros(num_seqs + 1, dtype=torch.int64)
    cum = 0
    for i, n_pairs in enumerate(seq_lengths):
        seq_starts[i] = cum
        cum += n_pairs
    seq_starts[num_seqs] = cum

    raw["seq_starts"] = seq_starts
    raw["num_sequences"] = num_seqs

    # Atomic write: save to a sibling .tmp then rename
    tmp_path = args.data + ".tmp"
    torch.save(raw, tmp_path)
    os.replace(tmp_path, args.data)

    size_mb = os.path.getsize(args.data) / 1e6
    print(f"\nManifest updated: {args.data} ({size_mb:.1f} MB)")
    print(f"  Added seq_starts (int64, len {num_seqs + 1})")
    print(f"  Ready for TTT training with train_eagle3_ttt.py")


if __name__ == "__main__":
    raise SystemExit(main())
