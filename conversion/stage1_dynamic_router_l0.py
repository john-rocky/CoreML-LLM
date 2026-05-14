#!/usr/bin/env python3
"""Phase β-1 Stage 1 — Single-layer dynamic top-K routing feasibility.

End-to-end Python prototype that:

  1. Loads Gemma 3n E2B
  2. Captures per-token (FFN input X, SwiGLU intermediate Y) on layer 0
     across a mixed code+narrative+pattern corpus
  3. Slices the 8192-wide intermediate into N buckets (default 16x512)
  4. Trains a tiny linear router (2048 -> N) supervised by which K slices
     contain the top-K firing neurons per token
  5. Evaluates cos sim of FFN block output for three policies:
       * oracle  — pick true top-K slices (upper bound)
       * router  — pick predicted top-K slices
       * random  — pick K random slices (lower bound)
  6. Writes a JSON report and prints a summary table

Acceptance gate: router cos sim > 0.98 at K=2 of N=16. If under, the
dynamic-routing approach as proposed is dead and we should stop before
spending CoreML effort.

Usage:
  pyenv shell lama-cml
  python conversion/stage1_dynamic_router_l0.py \
    --model /tmp/gemma3n-e2b \
    --layer 0 \
    --num-slices 16 \
    --top-k 2 \
    --tokens 4096 \
    --out /tmp/stage1_router_l0.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Optional

# Wandb stub (same incantation as calibrate_ffn_sparsity.py).
import sys as _sys
import types as _types
import importlib.machinery as _machinery
if "wandb" not in _sys.modules:
    _w = _types.ModuleType("wandb")
    _w.__path__ = []  # type: ignore[attr-defined]
    _w.__spec__ = _machinery.ModuleSpec("wandb", loader=None, is_package=True)
    _sys.modules["wandb"] = _w

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Calibration corpus (mixed code + narrative + repetitive pattern)
# ---------------------------------------------------------------------------

NARRATIVE = """
The history of computing is filled with paradigm shifts. Early mechanical
calculators gave way to vacuum tube computers, then transistors, then
integrated circuits. Each revolution brought orders of magnitude in speed,
miniaturisation, and accessibility. Today the dominant trend is massively
parallel computation, and large language models have become the flagship
workload of this era. Running these models on personal devices remains a
challenge — a modern smartphone has perhaps eight gigabytes of unified
memory and a power budget measured in single-digit watts. Bandwidth
between DRAM and the compute units is often the binding constraint rather
than compute throughput itself.

Apple's Neural Engine occupies an interesting niche in this landscape. It
was originally designed for small computer-vision models that fit
comfortably in its on-chip SRAM. The introduction of the Foundation Models
framework in iOS 26 pushed it into territory it was not built for, and
developers have been discovering the boundaries ever since. Recent reverse
engineering work has illuminated much of the previously opaque
architecture, revealing both constraints and opportunities.

Sparse activation patterns are one such opportunity. When a feed forward
network applies its first linear layer, only a small fraction of the
resulting intermediate neurons fire strongly. The rest contribute
negligibly to the output. If we could predict which neurons will fire
before computing the layer, we would avoid reading the weights of the
dormant ones, saving bandwidth proportional to the dormancy rate.
""".strip()

CODE = """
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
                if node.left is None:
                    node.left = Node(value)
                    return
                node = node.left
            else:
                if node.right is None:
                    node.right = Node(value)
                    return
                node = node.right

    def search(self, value):
        node = self.root
        while node is not None:
            if value == node.value:
                return True
            if value < node.value:
                node = node.left
            else:
                node = node.right
        return False

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

import json
import os
from collections import defaultdict

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def save_results(results: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
""".strip()

PATTERN = """
Say yes ten times. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes. Yes.
Count to ten. One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten.
Repeat after me: hello hello hello world world world test test test.
The quick brown fox jumps over the lazy dog. The quick brown fox jumps
over the lazy dog. The quick brown fox jumps over the lazy dog.
""".strip()

MIXED_CORPUS = NARRATIVE + "\n\n" + CODE + "\n\n" + PATTERN


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_layer(model, layer_idx: int):
    """Return the decoder layer at index `layer_idx` for Gemma 3n / Gemma 4."""
    root = getattr(model, "model", None) or model
    text_model = getattr(root, "language_model", None) or root
    layers = getattr(text_model, "layers", None)
    if layers is None:
        raise SystemExit(f"cannot find layers under {type(model).__name__}")
    if layer_idx >= len(layers):
        raise SystemExit(f"layer {layer_idx} out of range (have {len(layers)})")
    return layers[layer_idx]


def capture_activations(model_path: str, layer_idx: int, num_tokens: int,
                        cache_path: Optional[str] = None
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a calibration forward and capture (X, Y, W_down) for one layer.

    X: (T, hidden) FFN block input per token
    Y: (T, intermediate) SwiGLU intermediate per token (input to down_proj)
    W_down: (hidden, intermediate) down_proj weight (fp32, on CPU)
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[capture] loading cached activations from {cache_path}")
        z = np.load(cache_path)
        return z["X"], z["Y"], z["W_down"]

    device = get_device()
    print(f"[capture] device={device} model={model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    print(f"[capture] model loaded, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    layer = find_layer(model, layer_idx)
    mlp = layer.mlp
    print(f"[capture] layer {layer_idx} mlp class: {type(mlp).__name__}")
    # Down_proj weight is (hidden, intermediate).
    W_down = mlp.down_proj.weight.detach().to(torch.float32).cpu().numpy()
    print(f"[capture] down_proj.weight shape: {W_down.shape}")

    captured_X: list[np.ndarray] = []
    captured_Y: list[np.ndarray] = []

    def mlp_pre_hook(_mod, inputs):
        x = inputs[0].detach()
        if x.dim() == 3:
            flat = x.reshape(-1, x.shape[-1])
        else:
            flat = x.reshape(-1, x.shape[-1])
        captured_X.append(flat.to(torch.float32).cpu().numpy())

    def down_pre_hook(_mod, inputs):
        y = inputs[0].detach()
        if y.dim() == 3:
            flat = y.reshape(-1, y.shape[-1])
        else:
            flat = y.reshape(-1, y.shape[-1])
        captured_Y.append(flat.to(torch.float32).cpu().numpy())

    h1 = mlp.register_forward_pre_hook(mlp_pre_hook)
    h2 = mlp.down_proj.register_forward_pre_hook(down_pre_hook)

    enc = tok(MIXED_CORPUS, return_tensors="pt",
              truncation=True, max_length=num_tokens)
    input_ids = enc["input_ids"].to(device)
    print(f"[capture] corpus tokens: {input_ids.shape[1]}")

    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    print(f"[capture] forward in {time.time() - t0:.1f}s")

    h1.remove()
    h2.remove()

    X = np.concatenate(captured_X, axis=0)
    Y = np.concatenate(captured_Y, axis=0)
    print(f"[capture] X shape {X.shape}, Y shape {Y.shape}")

    if cache_path:
        np.savez(cache_path, X=X, Y=Y, W_down=W_down)
        print(f"[capture] cached to {cache_path}")

    # Free the model from device memory before returning.
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return X, Y, W_down


# ---------------------------------------------------------------------------
# Slicing + targets
# ---------------------------------------------------------------------------


def compute_slice_targets(Y: np.ndarray, num_slices: int,
                          top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """For each token, score each slice by L2-energy and pick top-K.

    Returns:
      targets   : (T, num_slices) multi-hot (1.0 for top-K, else 0.0)
      slice_e   : (T, num_slices) per-slice L2 energy (fp32)
    """
    T, inter = Y.shape
    assert inter % num_slices == 0, (
        f"intermediate size {inter} not divisible by num_slices {num_slices}")
    per_slice = inter // num_slices
    # (T, num_slices, per_slice) -> energy per (T, num_slices)
    Y_sliced = Y.reshape(T, num_slices, per_slice)
    slice_e = (Y_sliced * Y_sliced).sum(axis=-1)  # L2 squared per slice
    # Per-token top-K slice indices.
    topk_idx = np.argpartition(-slice_e, kth=top_k - 1, axis=-1)[:, :top_k]
    targets = np.zeros((T, num_slices), dtype=np.float32)
    rows = np.arange(T)[:, None]
    targets[rows, topk_idx] = 1.0
    return targets, slice_e


# ---------------------------------------------------------------------------
# Router training (linear, multi-label, BCE)
# ---------------------------------------------------------------------------


def train_router(X_train: np.ndarray, targets_train: np.ndarray,
                 X_val: np.ndarray, targets_val: np.ndarray,
                 hidden: int = 0, epochs: int = 200,
                 lr: float = 1e-2) -> dict:
    """Train a tiny router. If hidden==0, the router is a single linear
    layer (X -> num_slices). If hidden>0 a (hidden_size, hidden, num_slices)
    MLP is used. Multi-label sigmoid BCE.
    """
    device = torch.device("cpu")  # tiny model, CPU is fine
    Xt = torch.from_numpy(X_train).to(device)
    Yt = torch.from_numpy(targets_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    Yv = torch.from_numpy(targets_val).to(device)

    in_dim = X_train.shape[1]
    out_dim = targets_train.shape[1]
    if hidden > 0:
        router = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, out_dim),
        )
        kind = f"mlp_{hidden}"
    else:
        router = torch.nn.Linear(in_dim, out_dim)
        kind = "linear"

    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()

    history = []
    best_val = -1.0
    best_state = None
    for ep in range(epochs):
        router.train()
        logits = router(Xt)
        loss = bce(logits, Yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            router.eval()
            with torch.no_grad():
                vl = router(Xv)
                v_loss = bce(vl, Yv).item()
                # Top-K agreement: for each token, do predicted topK match true topK?
                k = int(Yv.sum(dim=-1).max().item())
                pred_top = vl.topk(k, dim=-1).indices
                true_top = Yv.topk(k, dim=-1).indices
                # Set overlap per token (Jaccard would be /k+k-overlap; we use
                # plain overlap / k).
                overlap = 0.0
                for i in range(Xv.shape[0]):
                    p = set(pred_top[i].tolist())
                    t = set(true_top[i].tolist())
                    overlap += len(p & t) / k
                overlap /= Xv.shape[0]
            history.append({"epoch": ep + 1, "train_loss": float(loss.item()),
                            "val_loss": float(v_loss), "val_top_k_overlap": overlap})
            print(f"[router {kind}] ep {ep+1:>3} train {loss.item():.4f} "
                  f"val {v_loss:.4f} top-K overlap {overlap:.3f}")
            if overlap > best_val:
                best_val = overlap
                best_state = {k: v.detach().clone() for k, v in router.state_dict().items()}

    if best_state is not None:
        router.load_state_dict(best_state)
    return {"router": router, "kind": kind, "history": history,
            "best_val_overlap": best_val}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def compute_ffn_output(Y: np.ndarray, W_down: np.ndarray) -> np.ndarray:
    """Dense FFN block output: W_down @ Y.T -> (hidden, T) -> (T, hidden)."""
    return Y @ W_down.T  # (T, intermediate) @ (intermediate, hidden) = (T, hidden)


def compute_sparse_output(Y: np.ndarray, W_down: np.ndarray,
                          selected: np.ndarray, num_slices: int) -> np.ndarray:
    """Sparse FFN block output, zeroing slices not in `selected`.

    selected: (T, num_slices) multi-hot mask.
    """
    T, inter = Y.shape
    per_slice = inter // num_slices
    # Expand mask to neuron-level: (T, num_slices, 1) -> tile -> reshape (T, inter)
    mask = np.repeat(selected, per_slice, axis=1)
    Y_masked = Y * mask
    return Y_masked @ W_down.T


def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity. Returns shape (T,)."""
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9
    return num / den


def predict_router(router: torch.nn.Module, X: np.ndarray,
                   top_k: int) -> np.ndarray:
    """Run router on X (CPU) and return multi-hot top-K mask."""
    router.eval()
    with torch.no_grad():
        logits = router(torch.from_numpy(X))
        pred_top = logits.topk(top_k, dim=-1).indices.numpy()
    mask = np.zeros((X.shape[0], logits.shape[-1]), dtype=np.float32)
    rows = np.arange(X.shape[0])[:, None]
    mask[rows, pred_top] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--num-slices", type=int, default=16)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--mlp-hidden", type=int, default=0,
                   help="if >0, router is MLP with this many hidden units")
    p.add_argument("--cache", default="/tmp/l0_activations.npz",
                   help="path to cache captured activations (skip if exists)")
    p.add_argument("--out", default="/tmp/stage1_router_l0.json")
    args = p.parse_args()

    print(f"=== Stage 1: dynamic top-{args.top_k} of {args.num_slices} "
          f"on layer {args.layer} ===")

    X, Y, W_down = capture_activations(args.model, args.layer, args.tokens,
                                       cache_path=args.cache)

    # Train/val/test split: 70 / 15 / 15
    T = X.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(T)
    n_train = int(T * 0.70)
    n_val = int(T * 0.15)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    print(f"[split] train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")

    targets, slice_energy = compute_slice_targets(Y, args.num_slices, args.top_k)

    # ---------- Train router ----------
    train_res = train_router(
        X[train_idx], targets[train_idx],
        X[val_idx], targets[val_idx],
        hidden=args.mlp_hidden,
    )
    router = train_res["router"]
    router_kind = train_res["kind"]

    # ---------- Evaluate on test set ----------
    X_te = X[test_idx]
    Y_te = Y[test_idx]
    true_targets = targets[test_idx]
    # Dense reference output
    out_dense = compute_ffn_output(Y_te, W_down)
    # Oracle
    out_oracle = compute_sparse_output(Y_te, W_down, true_targets,
                                       args.num_slices)
    # Router
    router_mask = predict_router(router, X_te, args.top_k)
    out_router = compute_sparse_output(Y_te, W_down, router_mask,
                                       args.num_slices)
    # Random
    rng2 = np.random.default_rng(42)
    random_mask = np.zeros_like(true_targets)
    for i in range(X_te.shape[0]):
        idx = rng2.choice(args.num_slices, args.top_k, replace=False)
        random_mask[i, idx] = 1.0
    out_random = compute_sparse_output(Y_te, W_down, random_mask,
                                       args.num_slices)

    cs_oracle = cos_sim(out_dense, out_oracle)
    cs_router = cos_sim(out_dense, out_router)
    cs_random = cos_sim(out_dense, out_random)

    # Magnitude-coverage stats: how much of total |Y| energy do the K
    # selected slices cover under each policy?
    Y_te_e = (Y_te * Y_te).sum(axis=-1)  # (T,)
    def cov(mask):
        return ((Y_te * Y_te) * np.repeat(mask, Y_te.shape[1] // args.num_slices,
                                          axis=1)).sum(axis=-1) / Y_te_e
    cov_oracle = cov(true_targets)
    cov_router = cov(router_mask)
    cov_random = cov(random_mask)

    print("\n=== Test-set results ===")
    print(f"{'policy':>8} | cos sim mean | cos sim p10 | mag cov mean")
    for name, cs, cv in [("oracle", cs_oracle, cov_oracle),
                          ("router", cs_router, cov_router),
                          ("random", cs_random, cov_random)]:
        print(f"{name:>8} | "
              f"{cs.mean():.4f}      | "
              f"{np.percentile(cs, 10):.4f}      | "
              f"{cv.mean():.4f}")

    # Acceptance gate
    router_cs_mean = float(cs_router.mean())
    gate_passed = router_cs_mean >= 0.98
    print(f"\n=== Acceptance gate: cos sim >= 0.98 on router ===")
    print(f"router cos sim mean = {router_cs_mean:.4f}   "
          f"{'PASS' if gate_passed else 'FAIL'}")

    report = {
        "args": vars(args),
        "num_train_tokens": int(len(train_idx)),
        "num_val_tokens": int(len(val_idx)),
        "num_test_tokens": int(len(test_idx)),
        "router_kind": router_kind,
        "router_history": train_res["history"],
        "best_val_top_k_overlap": float(train_res["best_val_overlap"]),
        "cos_sim": {
            "oracle_mean": float(cs_oracle.mean()),
            "oracle_p10": float(np.percentile(cs_oracle, 10)),
            "router_mean": float(cs_router.mean()),
            "router_p10": float(np.percentile(cs_router, 10)),
            "router_p1":  float(np.percentile(cs_router, 1)),
            "random_mean": float(cs_random.mean()),
            "random_p10": float(np.percentile(cs_random, 10)),
        },
        "magnitude_coverage": {
            "oracle_mean": float(cov_oracle.mean()),
            "router_mean": float(cov_router.mean()),
            "random_mean": float(cov_random.mean()),
        },
        "gate_passed": gate_passed,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[main] wrote {args.out}")

    sys.exit(0 if gate_passed else 2)


if __name__ == "__main__":
    main()
