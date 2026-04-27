#!/usr/bin/env python3
"""H1 probe: train linear future-projection probes on captured hidden states.

Reads hidden states + targets from probe_h1_collect.py output, trains a small
Linear(hidden, hidden) per (layer L, future K) feeding the frozen LM head, and
reports top-1 accuracy on the held-out split.

Architecture per probe:
    hidden_L  →  Linear(H, H) [trainable]  →  final_norm  →  LM_head [frozen]  →  logits
    loss: CE(logits, token_at_position+K)
    K=0/+1/+2/+3 are all trained; K=1 with L34 ≈ HF baseline (~0.3-0.4 acc).

Verdict thresholds (K=2 / K=3 acc on best layer):
    STRONG  ≥ 0.25 / 0.20  →  HF base retains MTP-aware signal, MTP fine-tune go
    WEAK    ≥ 0.10 / 0.05  →  partial signal, marginal value
    SCRUBBED < 0.10 / 0.05  →  Google scrubbed, MTP arch dead

Usage:
    conversion/.venv/bin/python conversion/probe_h1_train.py \
        --data /tmp/h1_probe/data.npz \
        --hf-dir output/gemma4-e2b/hf_model \
        --epochs 4 --batch 64
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_lm_head_and_norm(hf_dir: str):
    """Load frozen LM head weight + final norm gain from HF Gemma 4 E2B."""
    from transformers import AutoModelForCausalLM
    print(f"Loading frozen LM head + final norm from {hf_dir}...")
    m = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32,
                                              attn_implementation="eager")
    text = m.get_decoder() if hasattr(m, "get_decoder") else m.model
    lm_head_w = m.lm_head.weight.detach().float().clone()  # (vocab, hidden)
    norm_gain = text.norm.weight.detach().float().clone()  # (hidden,)
    eps = getattr(text.norm, "eps", 1e-6)
    softcap = getattr(m.config, "final_logit_softcapping", None)
    if softcap is None:
        softcap = getattr(text.config, "final_logit_softcapping", None)
    print(f"  lm_head: {lm_head_w.shape}  norm_gain: {norm_gain.shape}  "
          f"eps={eps}  softcap={softcap}")
    del m
    return lm_head_w, norm_gain, float(eps), softcap


class FutureProjection(nn.Module):
    """Linear(H, H) initialized to identity. Trained per (layer, K)."""
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden))

    def forward(self, x):
        return self.linear(x)


class MLPProjection(nn.Module):
    """2-layer MLP (H -> 4*H -> H) with GELU. Approximates non-linear drafter capacity.
    Init: small random + skip connection so init behavior matches linear-identity."""
    def __init__(self, hidden: int, expand: int = 4):
        super().__init__()
        self.up = nn.Linear(hidden, hidden * expand, bias=True)
        self.down = nn.Linear(hidden * expand, hidden, bias=True)
        self.act = nn.GELU()
        # Small init so the residual identity dominates initially.
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)

    def forward(self, x):
        return x + self.down(self.act(self.up(x)))


def rms_norm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    # Gemma RMSNorm: x / sqrt(mean(x^2) + eps) * (1 + gain). HF Gemma 4 follows this.
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return x * rms * (1.0 + gain)


def evaluate(proj, hidden_eval, tokens_eval, lm_head_w, norm_gain, eps, softcap,
             batch: int = 64) -> tuple[float, float, float]:
    proj.eval()
    n = hidden_eval.shape[0]
    correct1 = 0
    correct5 = 0
    correct10 = 0
    with torch.no_grad():
        for i in range(0, n, batch):
            h = hidden_eval[i:i+batch]
            t = tokens_eval[i:i+batch]
            x = proj(h)
            x = rms_norm(x, norm_gain, eps)
            logits = F.linear(x, lm_head_w)
            if softcap is not None and softcap > 0:
                logits = torch.tanh(logits / softcap) * softcap
            top10 = logits.topk(10, dim=-1).indices  # (B, 10)
            correct1 += (top10[:, 0] == t).sum().item()
            correct5 += (top10[:, :5] == t.unsqueeze(-1)).any(-1).sum().item()
            correct10 += (top10 == t.unsqueeze(-1)).any(-1).sum().item()
    return correct1 / n, correct5 / n, correct10 / n


def train_one_probe(hidden, tokens, hidden_eval, tokens_eval,
                    lm_head_w, norm_gain, eps, softcap,
                    epochs: int, batch: int, lr: float, label: str,
                    probe_type: str = "linear") -> dict:
    H = hidden.shape[1]
    device = hidden.device
    if probe_type == "mlp":
        proj = MLPProjection(H, expand=4).to(device)
    else:
        proj = FutureProjection(H).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=lr, weight_decay=0.0)

    print(f"\n[{label}] training {epochs} epochs, "
          f"train={hidden.shape[0]}  val={hidden_eval.shape[0]}")

    # Initial eval (identity projection — for L34 K=1 this should already be OK)
    a1, a5, a10 = evaluate(proj, hidden_eval, tokens_eval,
                            lm_head_w, norm_gain, eps, softcap, batch)
    print(f"  init     top1={a1:.4f}  top5={a5:.4f}  top10={a10:.4f}")

    n = hidden.shape[0]
    for ep in range(epochs):
        proj.train()
        perm = torch.randperm(n)
        loss_sum = 0.0
        steps = 0
        t0 = time.time()
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            h = hidden[idx]
            t = tokens[idx]
            x = proj(h)
            x = rms_norm(x, norm_gain, eps)
            logits = F.linear(x, lm_head_w)
            if softcap is not None and softcap > 0:
                logits = torch.tanh(logits / softcap) * softcap
            loss = F.cross_entropy(logits, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            steps += 1
        a1, a5, a10 = evaluate(proj, hidden_eval, tokens_eval,
                                lm_head_w, norm_gain, eps, softcap, batch)
        print(f"  epoch {ep+1}  loss={loss_sum/steps:.4f}  "
              f"top1={a1:.4f}  top5={a5:.4f}  top10={a10:.4f}  "
              f"({time.time()-t0:.1f}s)")
    return {"top1": a1, "top5": a5, "top10": a10}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hf-dir", default="output/gemma4-e2b/hf_model")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--probe-type", default="linear", choices=["linear", "mlp"])
    args = ap.parse_args()

    print(f"Loading {args.data}...")
    d = np.load(args.data)
    layers = d["layers"].tolist()
    targets = d["targets"]  # (N, K_FUTURE+1) = (N, 4)
    N = targets.shape[0]
    print(f"  layers: {layers}, N={N} positions, probe={args.probe_type}")

    lm_head_w, norm_gain, eps, softcap = load_lm_head_and_norm(args.hf_dir)
    device = torch.device(args.device)
    lm_head_w = lm_head_w.to(device)
    norm_gain = norm_gain.to(device)

    # Train/val split (sequential so consecutive positions stay together)
    n_val = int(N * args.val_frac)
    n_train = N - n_val
    print(f"  train={n_train}  val={n_val}")

    targets_t = torch.from_numpy(targets).long()

    results = {}
    for li in layers:
        h_all = torch.from_numpy(d[f"hidden_L{li}"]).float().to(device)
        h_train = h_all[:n_train]
        h_val = h_all[n_train:]
        for k in (0, 1, 2, 3):
            t_train = targets_t[:n_train, k].to(device)
            t_val = targets_t[n_train:, k].to(device)
            label = f"L{li} K={k}"
            r = train_one_probe(h_train, t_train, h_val, t_val,
                                lm_head_w, norm_gain, eps, softcap,
                                epochs=args.epochs, batch=args.batch,
                                lr=args.lr, label=label,
                                probe_type=args.probe_type)
            results[(li, k)] = r

    # Summary
    print("\n" + "=" * 70)
    print("H1 PROBE RESULTS (top-1 / top-5 / top-10 accuracy on val split)")
    print("=" * 70)
    print(f"{'Layer':<8}{'K=0':<22}{'K=1':<22}{'K=2':<22}{'K=3':<22}")
    for li in layers:
        row = f"L{li:<7}"
        for k in (0, 1, 2, 3):
            r = results[(li, k)]
            row += f"{r['top1']:.3f}/{r['top5']:.3f}/{r['top10']:.3f}  "
        print(row)
    print("=" * 70)

    # Verdict
    best_k2 = max(results[(li, 2)]["top1"] for li in layers)
    best_k3 = max(results[(li, 3)]["top1"] for li in layers)
    print(f"\nbest K=2 top-1: {best_k2:.4f}")
    print(f"best K=3 top-1: {best_k3:.4f}")
    if best_k2 >= 0.25 and best_k3 >= 0.20:
        verdict = "STRONG — MTP-aware signal present, MTP fine-tune is high-EV"
    elif best_k2 >= 0.10 and best_k3 >= 0.05:
        verdict = "WEAK — partial signal, MTP fine-tune marginal"
    else:
        verdict = "SCRUBBED — MTP architecture path dead, drafter ceiling ~22% per-position"
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()
