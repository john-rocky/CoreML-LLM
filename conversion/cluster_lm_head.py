#!/usr/bin/env python3
"""Cluster Gemma 4 lm_head rows for semantic-NN candidate selection.

L12 Phase 1 ran into a structural coverage ceiling (~50% per-position miss
rate) because corpus-derived frequent-token lists capture token GLOBAL
frequency, not the SEMANTIC NEIGHBORHOOD of the current hidden state. With
clusters, we can pick top-N clusters by hidden-state similarity at runtime
→ ~4096 candidates from the right semantic ballpark → expected miss rate
<10%.

Algorithm:
1. Load lm_head_fp16.bin (V=262144 × H=1536 row-major fp16) as fp32 numpy.
2. MiniBatchKMeans (k=128) on the rows.
3. Save:
   - lm_head_cluster_centroids.bin (k × H × fp16, row-major)
   - lm_head_cluster_assignments.bin (V × int32, cluster id per token)
   - lm_head_cluster_members.bin (offsets table + token IDs sorted by cluster)

Swift runtime: matmul normed_hidden × centroids → top-N clusters → union of
member token IDs → standard L12 sparse matmul on subset.

Usage:
    python conversion/cluster_lm_head.py --k 128
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=128,
                    help="Number of clusters (k-means k)")
    ap.add_argument("--input", default=None,
                    help="Path to lm_head_fp16.bin (default: ../output/gemma4-e2b/lm_head_fp16.bin)")
    ap.add_argument("--output-dir", default=None,
                    help="Output dir for cluster files (default: same dir as input)")
    ap.add_argument("--vocab", type=int, default=262144)
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--max-iter", type=int, default=20,
                    help="MiniBatchKMeans max_iter")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--use-faiss", action="store_true",
                    help="Use FAISS instead of sklearn (faster on big data)")
    args = ap.parse_args()

    ROOT = os.path.dirname(os.path.abspath(__file__))
    args.input = args.input or os.path.join(
        ROOT, "..", "output", "gemma4-e2b", "lm_head_fp16.bin")
    args.output_dir = args.output_dir or os.path.dirname(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    V, H = args.vocab, args.hidden

    print(f"loading {args.input} ({os.path.getsize(args.input) / 1024 / 1024:.0f} MB)")
    t0 = time.time()
    arr_f16 = np.fromfile(args.input, dtype=np.float16).reshape(V, H)
    arr_f32 = arr_f16.astype(np.float32)
    print(f"  loaded {arr_f32.shape} in {time.time() - t0:.1f}s")

    # IMPORTANT: cluster in L2-NORMALIZED space (spherical k-means).
    # Inference scores tokens by `hidden · lm_head[t]` (inner product).
    # Standard k-means uses Euclidean distance, which gives a DIFFERENT
    # ranking from inner product unless rows are unit-norm. By L2-normalizing
    # the LM head rows before k-means, k-means becomes spherical k-means
    # (= cosine clustering) and the cluster centroid that gives max
    # dot product to a normalized query is also the closest cluster in
    # Euclidean terms.
    norms = np.linalg.norm(arr_f32, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0  # avoid div-by-zero
    arr_f32 = arr_f32 / norms
    print(f"  L2-normalized rows (mean orig norm: {norms.mean():.4f})")

    print(f"\nrunning k-means k={args.k} on {V} rows of dim {H}...")
    t0 = time.time()
    if args.use_faiss:
        try:
            import faiss
        except ImportError:
            print("  FAISS not installed; falling back to sklearn")
            args.use_faiss = False
    if args.use_faiss:
        import faiss  # type: ignore
        d = H
        kmeans = faiss.Kmeans(d, args.k, niter=args.max_iter,
                              verbose=True, gpu=False, seed=42)
        kmeans.train(arr_f32)
        centroids = kmeans.centroids
        _, assignments = kmeans.index.search(arr_f32, 1)
        assignments = assignments.reshape(-1)
    else:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=args.k, batch_size=args.batch_size,
                              max_iter=args.max_iter, n_init=4, random_state=42,
                              verbose=1)
        km.fit(arr_f32)
        centroids = km.cluster_centers_.astype(np.float32)
        assignments = km.labels_.astype(np.int32)
    print(f"  kmeans done in {time.time() - t0:.1f}s")
    print(f"  centroids: {centroids.shape}")
    print(f"  assignments: {assignments.shape}")

    # Save centroids fp16 row-major
    centroids_path = os.path.join(args.output_dir, "lm_head_cluster_centroids.bin")
    centroids.astype(np.float16).tofile(centroids_path)
    print(f"  wrote {centroids_path} ({os.path.getsize(centroids_path)} bytes)")

    # Save assignments as int32 array indexed by token ID
    asg_path = os.path.join(args.output_dir, "lm_head_cluster_assignments.bin")
    assignments.astype(np.int32).tofile(asg_path)
    print(f"  wrote {asg_path} ({os.path.getsize(asg_path)} bytes)")

    # Save members: offsets table (k+1 int32) + sorted token IDs (V int32)
    # For each cluster, find tokens. Build flat array sorted by cluster id.
    order = np.argsort(assignments, kind="stable")
    sorted_ids = order.astype(np.int32)
    offsets = np.zeros(args.k + 1, dtype=np.int32)
    counts = np.bincount(assignments, minlength=args.k)
    offsets[1:] = np.cumsum(counts)
    members_path = os.path.join(args.output_dir, "lm_head_cluster_members.bin")
    with open(members_path, "wb") as f:
        f.write(offsets.tobytes())
        f.write(sorted_ids.tobytes())
    print(f"  wrote {members_path} ({os.path.getsize(members_path)} bytes)")

    # Stats
    print(f"\ncluster stats:")
    print(f"  k={args.k}")
    print(f"  size min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")
    print(f"  top-5 cluster sizes: {sorted(counts, reverse=True)[:5]}")
    print(f"  bottom-5 cluster sizes: {sorted(counts)[:5]}")


if __name__ == "__main__":
    main()
