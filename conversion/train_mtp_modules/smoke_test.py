#!/usr/bin/env python3
"""Local smoke test — validates the full training pipeline without HF model.

Uses dummy embed_weight + random-generated L34 hiddens + random token IDs.
Verifies:
  1. MtpStack forward runs with K=2
  2. Loss computation works
  3. Backward + optim step runs
  4. Loss goes DOWN over 20 steps (overfitting on tiny batch)

If this fails, don't waste A100 time.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_mtp_modules.mtp_modules import MtpStack, MtpModuleConfig
from train_mtp_modules.data import PrecomputedShardsDataset, collate_mtp


def make_fake_shards(out_dir: str, n_seqs: int = 32, seq_len: int = 64,
                     hidden_size: int = 1536, vocab_size: int = 262144):
    """Create fake shards for testing."""
    # Make sequences with a simple pattern that the model can learn:
    # token at position t depends on hidden[t] linearly.
    # This ensures training loss goes down.
    rng = np.random.default_rng(42)
    hiddens = rng.standard_normal((n_seqs, seq_len, hidden_size)).astype(np.float32) * 0.1
    # Tokens: use a deterministic function of hiddens so training can fit
    # Take a fixed linear projection to vocab space for the "target"
    proj = rng.standard_normal((hidden_size, vocab_size)).astype(np.float32) * 0.01
    logits = hiddens @ proj  # (n_seqs, seq_len, vocab_size)
    tokens = np.argmax(logits, axis=-1).astype(np.int32)
    tokens = tokens % vocab_size  # safety

    # Save shard
    np.save(os.path.join(out_dir, "shard_0000.tokens.npy"), tokens)
    np.save(os.path.join(out_dir, "shard_0000.hidden.npy"), hiddens.astype(np.float16))
    import json
    with open(os.path.join(out_dir, "shard_0000.meta.json"), "w") as f:
        json.dump({"shard_idx": 0, "num_sequences": n_seqs,
                   "seq_len": seq_len, "hidden_size": hidden_size}, f)
    print(f"Created fake shard: {n_seqs} × {seq_len} × {hidden_size}")


def main():
    # 1. Create fake data
    tmp = tempfile.mkdtemp(prefix="mtp_smoke_")
    print(f"Temp dir: {tmp}")
    make_fake_shards(tmp, n_seqs=32, seq_len=64)

    # 2. Load via dataset
    ds = PrecomputedShardsDataset(tmp, k_depth=2)
    print(f"Dataset: {len(ds)} sequences")

    # 3. Build MtpStack with random embed/lm_head (small vocab for smoke)
    cfg = MtpModuleConfig(num_modules=2)
    # Use random tied weights (no HF model needed for smoke)
    embed_w = torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.02
    lm_head_w = embed_w.clone()

    stack = MtpStack(cfg, lm_head_w, embed_w)
    stack.train()
    print(f"Trainable params: {sum(p.numel() for p in stack.parameters() if p.requires_grad):,}")

    # 4. Make a tiny batch
    batch = collate_mtp([ds[0], ds[1], ds[2], ds[3]], k_depth=2)
    print(f"Batch: l34={tuple(batch['l34_hidden'].shape)}, "
          f"tokens={tuple(batch['input_tokens'].shape)}")

    # 5. Forward + loss
    opt = torch.optim.AdamW(stack.parameters(), lr=1e-3)

    print("\nTraining 20 steps on tiny batch (should overfit, loss ↓):")
    initial_losses = None
    for step in range(20):
        l34 = batch["l34_hidden"]
        tokens = batch["input_tokens"]
        B, T_eff, H = l34.shape

        logits_list = stack(l34, tokens)
        losses = []
        for k in range(cfg.num_modules):
            # DeepSeek V3: module_k target is tokens[t+k+2]
            target = tokens[:, k + 2:k + 2 + T_eff]
            loss = F.cross_entropy(
                logits_list[k].reshape(-1, cfg.vocab_size),
                target.reshape(-1),
            )
            losses.append(loss)

        total_loss = losses[0] + 0.8 * losses[1]
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if initial_losses is None:
            initial_losses = [l.item() for l in losses]
        if step % 5 == 0 or step == 19:
            print(f"  step {step:2d}: L1={losses[0].item():.3f}  L2={losses[1].item():.3f}")

    final_losses = [l.item() for l in losses]
    print(f"\nInitial L1={initial_losses[0]:.3f} → Final L1={final_losses[0]:.3f} "
          f"({'PASS' if final_losses[0] < initial_losses[0] * 0.9 else 'FAIL'})")
    print(f"Initial L2={initial_losses[1]:.3f} → Final L2={final_losses[1]:.3f} "
          f"({'PASS' if final_losses[1] < initial_losses[1] * 0.9 else 'FAIL'})")

    # Cleanup
    import shutil
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
