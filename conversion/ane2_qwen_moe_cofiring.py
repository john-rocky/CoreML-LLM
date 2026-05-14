#!/usr/bin/env python3
"""ANE route, angle 2 — Qwen MoE expert co-firing clustering analysis.

The ANE wall for MoE is the per-layer routed-expert dispatch count
(Qwen1.5-MoE = 4 active of 60, every layer). IF the 60 experts cluster
into groups that tend to co-fire (get selected together for the same
token), we could re-map routing to GROUP granularity — fewer, larger
dispatches that ANE amortises better.

This script monkey-patches the MLX Qwen2MoE MoE block to record which
4-of-60 experts fire per token across a mixed corpus, builds a
per-layer co-firing matrix, and checks for cluster structure
(spectral clustering + silhouette). Reuses the methodology from the
Gemma 3n stage1 co-firing work.

Output: /tmp/qwen_moe_cofiring.json + console summary.

Usage:
  pyenv shell lama-cml
  python conversion/ane2_qwen_moe_cofiring.py --model /tmp/qwen_moe_3bit
"""
from __future__ import annotations
import argparse
import json

import numpy as np


CORPUS = """
The history of computing is filled with paradigm shifts. Apple's Neural
Engine occupies an interesting niche. Sparse activation patterns are an
opportunity for on-device inference.

class BinarySearchTree:
    def __init__(self):
        self.root = None
    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
            return
        node = self.root
        while True:
            if value < node.value:
                node = node.left
            else:
                node = node.right

function quicksort(arr) {
    if (arr.length <= 1) return arr;
    const pivot = arr[Math.floor(arr.length / 2)];
    return [...quicksort(left), ...middle, ...quicksort(right)];
}

Compute the derivative of f(x) = x^3 - 2x^2 + 5x - 7. Apply the power
rule term by term: 3x^2 - 4x + 5.

Say yes ten times: yes yes yes yes yes yes yes yes yes yes. Count to
ten: one two three four five six seven eight nine ten.

The transformer architecture replaced recurrence with self-attention.
Each layer applies multi-head attention then a position-wise feed
forward network. Residual connections and layer normalisation
stabilise training.

User: Could you summarise speculative decoding? Assistant: A small
drafter model proposes tokens, the large target verifies them in
parallel. Speedup depends on the drafter's accept rate.
""".strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/tmp/qwen_moe_3bit")
    p.add_argument("--num-groups", type=int, default=16,
                   help="target groups for the clustering test")
    p.add_argument("--out", default="/tmp/qwen_moe_cofiring.json")
    args = p.parse_args()

    import mlx.core as mx
    import mlx_lm
    from mlx_lm.models import qwen2_moe as qm

    print(f"[load] {args.model}")
    model, tok = mlx_lm.load(args.model)
    num_layers = len(model.model.layers)
    num_experts = model.model.layers[0].mlp.num_experts
    top_k = model.model.layers[0].mlp.top_k
    print(f"[cfg] layers={num_layers} experts={num_experts} top_k={top_k}")

    # Monkey-patch the MoE block __call__ to record selected indices.
    recorded: dict[int, list] = {i: [] for i in range(num_layers)}
    # Map each block instance to its layer index.
    block_to_layer = {}
    for i, layer in enumerate(model.model.layers):
        block_to_layer[id(layer.mlp)] = i

    orig_call = qm.Qwen2MoeSparseMoeBlock.__call__

    def patched_call(self, x):
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
        li = block_to_layer.get(id(self))
        if li is not None:
            recorded[li].append(np.array(inds).reshape(-1, k))
        return orig_call(self, x)

    qm.Qwen2MoeSparseMoeBlock.__call__ = patched_call

    # Forward the corpus (single pass; we only need routing decisions).
    ids = tok.encode(CORPUS)
    print(f"[run] corpus {len(ids)} tokens")
    x = mx.array([ids])
    _ = model(x)
    mx.eval(_)

    qm.Qwen2MoeSparseMoeBlock.__call__ = orig_call  # restore

    # Build per-layer co-firing matrices.
    report = {"model": args.model, "num_layers": num_layers,
              "num_experts": num_experts, "top_k": top_k, "layers": {}}
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_score

    global_cofire = np.zeros((num_experts, num_experts), dtype=np.float64)
    global_usage = np.zeros(num_experts, dtype=np.float64)
    total_tokens = 0

    for li in range(num_layers):
        chunks = recorded[li]
        if not chunks:
            continue
        sel = np.concatenate(chunks, axis=0)  # (T, top_k)
        T = sel.shape[0]
        total_tokens = max(total_tokens, T)
        # one-hot firing (T, num_experts)
        fire = np.zeros((T, num_experts), dtype=np.float32)
        rows = np.arange(T)[:, None]
        fire[rows, sel] = 1.0
        usage = fire.mean(axis=0)
        global_usage += fire.sum(axis=0)
        # co-firing: C[i,j] = P(j selected | i selected)
        cofire = fire.T @ fire  # (E, E) counts
        global_cofire += cofire
        # normalise to conditional prob
        diag = np.clip(np.diag(cofire), 1, None)
        cond = cofire / diag[:, None]
        np.fill_diagonal(cond, 0.0)
        report["layers"][str(li)] = {
            "tokens": int(T),
            "max_usage": float(usage.max()),
            "min_usage": float(usage.min()),
            "n_dead": int((usage == 0).sum()),
            "mean_offdiag_cofire_prob": float(cond.mean()),
            "max_offdiag_cofire_prob": float(cond.max()),
        }

    # Global clustering test on the aggregate co-firing matrix.
    diag = np.clip(np.diag(global_cofire), 1, None)
    cond = global_cofire / diag[:, None]
    cond = (cond + cond.T) / 2.0  # symmetrise for spectral
    np.fill_diagonal(cond, 0.0)
    affinity = np.clip(cond, 0.0, None)

    print(f"\n=== Global co-firing structure ({total_tokens} tokens) ===")
    print(f"mean off-diag co-fire prob: {affinity[affinity>0].mean():.4f}")
    print(f"max  off-diag co-fire prob: {affinity.max():.4f}")
    print(f"p99  off-diag co-fire prob: {np.percentile(affinity[affinity>0], 99):.4f}")
    # If experts were perfectly uniform/independent, co-fire prob ~ top_k/num_experts
    uniform_baseline = top_k / num_experts
    print(f"uniform-independent baseline: {uniform_baseline:.4f}")

    cluster_results = {}
    for ng in [8, 16, 32]:
        try:
            sc = SpectralClustering(n_clusters=ng, affinity="precomputed",
                                    assign_labels="kmeans", random_state=0,
                                    n_init=5)
            labels = sc.fit_predict(affinity)
            sizes = np.bincount(labels, minlength=ng).tolist()
            # silhouette on the affinity (convert to distance)
            dist = 1.0 - (affinity / max(affinity.max(), 1e-9))
            np.fill_diagonal(dist, 0.0)
            try:
                sil = float(silhouette_score(dist, labels, metric="precomputed"))
            except Exception:
                sil = float("nan")
            print(f"  groups={ng}: sizes min {min(sizes)} max {max(sizes)} "
                  f"silhouette {sil:.3f}")
            cluster_results[str(ng)] = {"sizes": sizes, "silhouette": sil}
        except Exception as e:
            print(f"  groups={ng}: ERROR {e}")

    report["global"] = {
        "total_tokens": total_tokens,
        "mean_offdiag_cofire": float(affinity[affinity > 0].mean()),
        "max_offdiag_cofire": float(affinity.max()),
        "uniform_baseline": float(uniform_baseline),
        "clustering": cluster_results,
    }

    # Verdict
    print(f"\n=== Verdict ===")
    sig = affinity[affinity > 0].mean() / uniform_baseline
    print(f"co-firing signal ratio (mean / uniform baseline): {sig:.2f}x")
    best_sil = max((v["silhouette"] for v in cluster_results.values()
                    if v["silhouette"] == v["silhouette"]), default=0.0)
    if sig > 1.5 and best_sil > 0.15:
        print("PROMISING — experts show co-firing structure; group-routing "
              "could cut dispatch count. Worth a grouped-multifunction build.")
        report["verdict"] = "PROMISING"
    elif sig > 1.2:
        print("WEAK SIGNAL — some structure but clusters not clean. "
              "Group-routing would lose accuracy.")
        report["verdict"] = "WEAK"
    else:
        print("FLAT — experts fire near-independently. Group-routing "
              "won't reduce dispatch count without quality loss.")
        report["verdict"] = "FLAT"

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
