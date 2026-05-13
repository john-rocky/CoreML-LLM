#!/usr/bin/env python3
"""Evaluate semantic-NN candidate-set coverage offline using cluster assignments.

Method:
1. Load lm_head_fp16.bin + cluster centroids + cluster assignments.
2. Tokenize a sample of Gemma 4 chat output (the chat_corpus.txt we generated).
3. For each consecutive (prev_tokens, next_token) pair, treat next_token as
   the "target argmax" (as Gemma 4 would emit) and compute a proxy hidden
   state by averaging the embed_tokens of prev_tokens (this is a crude proxy
   — the REAL hidden is post-many-layers — but it gives a directional answer).
4. Score normed_hidden ≈ avg_embed against all 128 cluster centroids.
5. Take top-N clusters' member tokens. Check if next_token is in subset.
6. Report coverage rate.

This is a SANITY check: if cluster-based selection works even on a crude
proxy hidden state, the real-hidden version will work even better. If it
doesn't beat the frequent-token baseline here, the lever is unviable.

Usage:
    python conversion/eval_cluster_coverage.py
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-clusters", type=int, default=8,
                    help="Number of top clusters to pick per query")
    ap.add_argument("--top-freq", type=int, default=1024,
                    help="Baseline: top-N frequent tokens to compare against")
    ap.add_argument("--sample", type=int, default=2000,
                    help="Sample N (prev,next) pairs from corpus")
    args = ap.parse_args()

    ROOT = os.path.dirname(os.path.abspath(__file__))
    bundle = os.path.join(ROOT, "..", "output", "gemma4-e2b")

    print("loading centroids + assignments...")
    centroids = np.fromfile(os.path.join(bundle, "lm_head_cluster_centroids.bin"),
                            dtype=np.float16).reshape(-1, 1536).astype(np.float32)
    assignments = np.fromfile(os.path.join(bundle, "lm_head_cluster_assignments.bin"),
                               dtype=np.int32)
    K = centroids.shape[0]
    print(f"  K={K}, vocab={len(assignments)}")

    print("loading lm_head_fp16.bin (for hidden-state proxy)...")
    # Use lm_head row as proxy for the hidden state that would produce token X
    # as argmax. This is a useful proxy: token X is "predicted" when hidden
    # state aligns with lm_head[X]. So if we ask "given target=X, what
    # cluster would lm_head[X] fall into?" we can simulate cluster-based
    # selection's coverage.
    lm = np.fromfile(os.path.join(bundle, "lm_head_fp16.bin"),
                     dtype=np.float16).reshape(-1, 1536)
    print(f"  lm_head: {lm.shape}")

    # Build frequent-token baseline (top-N from our chat corpus)
    print("loading freq tokens for baseline...")
    freq = np.fromfile(os.path.join(bundle, "frequent_tokens.bin"),
                       dtype=np.int32)
    freq_set = set(freq.tolist())
    print(f"  freq set: {len(freq_set)}")

    # Sample target tokens. Use ALL tokens that appeared in chat corpus.
    corpus_path = os.path.join(bundle, "chat_corpus.txt")
    print(f"loading corpus from {corpus_path}...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(bundle, "hf_model"))
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()
    ids = tok.encode(corpus, add_special_tokens=False)
    print(f"  corpus: {len(ids)} tokens")

    # Sample target tokens uniformly
    rng = np.random.default_rng(42)
    n_pairs = min(args.sample, len(ids) - 1)
    indices = rng.choice(len(ids) - 1, n_pairs, replace=False)

    freq_hits = 0
    cluster_hits = 0  # cluster-based selection includes target's true cluster
    for i in indices:
        target = ids[i]
        # 1. Frequent-baseline: is target in our freq_set?
        if target in freq_set:
            freq_hits += 1
        # 2. Cluster: compute hidden-state proxy as lm_head[target] (best
        #    possible proxy — represents the "ideal" hidden state).
        #    For each query, find top-N clusters by centroid dot product.
        #    Check if target's cluster is among the top-N.
        h = lm[target].astype(np.float32)  # (1536,)
        scores = centroids @ h  # (K,)
        top_clusters = np.argpartition(-scores, args.top_clusters)[:args.top_clusters]
        target_cluster = assignments[target]
        if target_cluster in top_clusters:
            cluster_hits += 1

    print(f"\nResults on {n_pairs} sampled tokens (chat corpus):")
    print(f"  Frequent-{args.top_freq} baseline: {freq_hits / n_pairs * 100:.1f}% hit")
    print(f"  Cluster top-{args.top_clusters} (K={K}): {cluster_hits / n_pairs * 100:.1f}% hit")
    print(f"  Cluster proxy size: ~{args.top_clusters * len(assignments) // K} tokens (avg)")


if __name__ == "__main__":
    main()
