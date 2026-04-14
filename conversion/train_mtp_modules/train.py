#!/usr/bin/env python3
"""Path C Phase 2: train MTP modules from precomputed L34 hiddens.

Goals:
  - Minimize wasted GPU time: train module_1 to gate metric first, then decide
    whether to train module_2 (and/or expand data).
  - Every ~500 steps: eval on held-out shard, print top-1 accuracy and CE loss
    per module. Save checkpoint.
  - Early-exit if module_1 top-1 > 80% after 1 epoch (already excellent).

Expected A100 time: ~4-8 hours for 5M tokens, 3 epochs, K=2.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_mtp_modules.mtp_modules import MtpStack, MtpModuleConfig, count_params
from train_mtp_modules.data import PrecomputedShardsDataset, collate_mtp


def load_trunk_tied_weights(hf_dir: str, device: str, dtype):
    """Return (embed_weight, lm_head_weight) from HF Gemma 4.

    For Gemma 4 lm_head is tied with embed_tokens — they share the same matrix.
    """
    from transformers import Gemma4ForConditionalGeneration
    hf = Gemma4ForConditionalGeneration.from_pretrained(
        hf_dir, torch_dtype=dtype, device_map="cpu"
    )
    lm = hf.model.language_model
    embed = lm.embed_tokens.weight.detach().clone()  # (V, H)
    # Gemma 4 ties lm_head.weight = embed_tokens.weight
    return embed, embed.clone()


def compute_losses_and_acc(stack, batch, device, k_depth):
    """Forward + per-module CE loss + top-1 accuracy."""
    l34 = batch["l34_hidden"].to(device)  # (B, T_eff, H)
    tokens = batch["input_tokens"].to(device)  # (B, T_eff + K)
    B, T_eff, H = l34.shape

    logits_list = stack(l34, tokens)  # list of (B, T_eff, V)

    losses = []
    accuracies = []
    for k in range(k_depth):
        target = tokens[:, k + 1:k + 1 + T_eff]  # (B, T_eff) — token at t+k+1
        logits = logits_list[k]  # (B, T_eff, V)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="mean",
        )
        acc = (logits.argmax(dim=-1) == target).float().mean()
        losses.append(loss)
        accuracies.append(acc.item())

    return losses, accuracies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=str, default="./output/mtp_train_cache",
                    help="Directory with precomputed shards.")
    ap.add_argument("--hf-dir", type=str,
                    default=os.path.expanduser(
                        "~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/"
                        "4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"))
    ap.add_argument("--k-depth", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-epochs", type=int, default=3)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--save-dir", type=str, default="./output/mtp_train_ckpt")
    ap.add_argument("--loss-weights", type=float, nargs="+", default=[1.0, 0.8],
                    help="Per-module loss weights (length must equal k_depth).")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--early-exit-acc", type=float, default=0.85,
                    help="If module_1 top-1 exceeds this on eval, end training early.")
    ap.add_argument("--resume", type=str, default=None,
                    help="Resume from checkpoint path.")
    args = ap.parse_args()

    assert len(args.loss_weights) == args.k_depth, \
        "loss_weights length must equal k_depth"

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = args.device

    # Data
    print("Loading precomputed shards...")
    full_ds = PrecomputedShardsDataset(args.cache_dir, k_depth=args.k_depth)
    # Val split: 2% or 32 seqs, whichever is smaller; ensure train has >= 1 seq
    val_size = max(1, min(32, int(len(full_ds) * 0.1)))
    if val_size >= len(full_ds):
        val_size = max(1, len(full_ds) // 5)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train: {len(train_ds)} seqs  Val: {len(val_ds)} seqs")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
        collate_fn=lambda b: collate_mtp(b, args.k_depth),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1,
        collate_fn=lambda b: collate_mtp(b, args.k_depth),
    )

    # Model
    print(f"Loading trunk tied weights from {args.hf_dir}...")
    embed_w, lm_head_w = load_trunk_tied_weights(args.hf_dir, device, torch.float32)
    print(f"  embed weight: {tuple(embed_w.shape)}, norm={embed_w.norm().item():.2f}")

    cfg = MtpModuleConfig(num_modules=args.k_depth)
    stack = MtpStack(cfg, lm_head_w.to(dtype), embed_w.to(dtype)).to(device)
    # Move only trainable params to working dtype; tied weights stay as buffers
    stack.modules_list = stack.modules_list.to(dtype)
    stack.train()

    trainable = sum(p.numel() for p in stack.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    # Optimizer
    opt = torch.optim.AdamW(
        [p for p in stack.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.num_epochs
    print(f"Steps per epoch: {steps_per_epoch}  Total steps: {total_steps}")

    # Cosine LR with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        stack.load_state_dict(ckpt["model"], strict=False)
        opt.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"]

    # Training
    print("\n=== Training ===")
    step = start_step
    ema_losses = [None] * args.k_depth
    ema_accs = [None] * args.k_depth
    ema_alpha = 0.02

    for epoch in range(args.num_epochs):
        for batch in train_loader:
            t0 = time.time()
            losses, accs = compute_losses_and_acc(stack, batch, device, args.k_depth)
            weighted_loss = sum(w * l for w, l in zip(args.loss_weights, losses))

            opt.zero_grad(set_to_none=True)
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(stack.parameters(), 1.0)
            opt.step()
            scheduler.step()
            step_time = time.time() - t0

            for k in range(args.k_depth):
                loss_val = losses[k].item()
                ema_losses[k] = loss_val if ema_losses[k] is None else \
                    ema_alpha * loss_val + (1 - ema_alpha) * ema_losses[k]
                ema_accs[k] = accs[k] if ema_accs[k] is None else \
                    ema_alpha * accs[k] + (1 - ema_alpha) * ema_accs[k]

            if step % args.log_interval == 0:
                lr = opt.param_groups[0]["lr"]
                loss_str = "  ".join([f"L{k+1}={ema_losses[k]:.3f} acc={ema_accs[k]*100:.1f}%"
                                      for k in range(args.k_depth)])
                print(f"[{step}/{total_steps}] lr={lr:.2e}  {loss_str}  "
                      f"step_time={step_time*1000:.0f}ms")

            if step > 0 and step % args.eval_interval == 0:
                print("  [eval]", end=" ")
                stack.eval()
                with torch.inference_mode():
                    eval_losses = [0.0] * args.k_depth
                    eval_accs = [0.0] * args.k_depth
                    n = 0
                    for batch in val_loader:
                        losses, accs = compute_losses_and_acc(
                            stack, batch, device, args.k_depth)
                        for k in range(args.k_depth):
                            eval_losses[k] += losses[k].item()
                            eval_accs[k] += accs[k]
                        n += 1
                    eval_losses = [l / n for l in eval_losses]
                    eval_accs = [a / n for a in eval_accs]
                stack.train()
                eval_str = "  ".join([
                    f"L{k+1}={eval_losses[k]:.3f} acc={eval_accs[k]*100:.1f}%"
                    for k in range(args.k_depth)
                ])
                print(eval_str)

                # Save checkpoint
                ckpt_path = save_dir / f"mtp_step_{step}.pt"
                torch.save({
                    "step": step,
                    "model": stack.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": scheduler.state_dict(),
                    "config": vars(args),
                    "eval_loss": eval_losses,
                    "eval_acc": eval_accs,
                }, ckpt_path)
                print(f"  Saved {ckpt_path}")

                # Early exit gate
                if eval_accs[0] >= args.early_exit_acc:
                    print(f"\n*** Early exit: module_1 acc {eval_accs[0]*100:.1f}% "
                          f">= {args.early_exit_acc*100:.0f}% threshold ***")
                    return

            step += 1

    # Final save
    ckpt_path = save_dir / "mtp_final.pt"
    torch.save({"step": step, "model": stack.state_dict(), "config": vars(args)},
               ckpt_path)
    print(f"\nFinal checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
