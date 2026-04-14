"""Dataset for precomputed (token_ids, L34_hidden) shards."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PrecomputedShardsDataset(Dataset):
    """Loads shards produced by precompute.py.

    Each item = one (token_ids, L34_hidden) sequence.
    L34_hidden shape: (seq_len, hidden_size). token_ids shape: (seq_len,).

    For K-depth MTP training, the dataloader must provide token_ids of length
    seq_len + K so that module_k can be supervised with target at position t+k+1.
    We handle this by slicing: __getitem__ returns full seq, the trainer
    builds teacher targets from token_ids[k:seq_len+k].
    """

    def __init__(self, cache_dir: str, k_depth: int = 2):
        self.cache_dir = Path(cache_dir)
        self.k_depth = k_depth
        self.shards = []  # list of (tokens_path, hiddens_path, num_sequences)

        shard_files = sorted(glob.glob(str(self.cache_dir / "shard_*.tokens.npy")))
        for tf in shard_files:
            hf = tf.replace(".tokens.npy", ".hidden.npy")
            if os.path.exists(hf):
                # mmap to avoid loading everything
                tokens_mm = np.load(tf, mmap_mode="r")
                hiddens_mm = np.load(hf, mmap_mode="r")
                assert tokens_mm.shape[0] == hiddens_mm.shape[0]
                assert tokens_mm.shape[1] == hiddens_mm.shape[1]
                self.shards.append({
                    "tokens": tokens_mm,
                    "hiddens": hiddens_mm,
                    "count": tokens_mm.shape[0],
                    "seq_len": tokens_mm.shape[1],
                    "hidden_size": hiddens_mm.shape[-1],
                })

        # Build global index
        self.offsets = []
        cumulative = 0
        for shard in self.shards:
            self.offsets.append(cumulative)
            cumulative += shard["count"]
        self.total = cumulative

        if self.total == 0:
            raise RuntimeError(f"No shards found in {cache_dir}")

        self.seq_len = self.shards[0]["seq_len"]
        self.hidden_size = self.shards[0]["hidden_size"]

        print(f"[data] Loaded {self.total} sequences from {len(self.shards)} shards, "
              f"seq_len={self.seq_len}, hidden_size={self.hidden_size}")

    def __len__(self):
        # Truncate by k_depth to ensure we can read future tokens
        # Each sequence yields (seq_len - k_depth) valid training positions
        # But we return the full sequence; trainer handles the slicing
        return self.total

    def __getitem__(self, idx):
        # Binary search for shard
        shard_i = 0
        for i, off in enumerate(self.offsets):
            if idx < off + self.shards[i]["count"]:
                shard_i = i
                idx_in_shard = idx - off
                break
        else:
            raise IndexError(idx)

        shard = self.shards[shard_i]
        tokens = np.asarray(shard["tokens"][idx_in_shard])  # (seq_len,)
        hiddens = np.asarray(shard["hiddens"][idx_in_shard])  # (seq_len, hidden_size)

        return {
            "tokens": torch.from_numpy(tokens).long(),
            "hiddens": torch.from_numpy(hiddens.astype(np.float32)),
        }


def collate_mtp(batch, k_depth: int = 2):
    """Collate MTP training batch.

    Input: list of {tokens: (T,), hiddens: (T, H)}
    Output:
      l34_hidden:  (B, T - K, H)           — input to modules
      input_tokens: (B, T - K + K, )       — embedding input (current + K draft targets)
      target_tokens: list of (B, T - K)    — CE target for each module k
    """
    B = len(batch)
    T_full = batch[0]["tokens"].shape[0]
    # DeepSeek V3: module_k needs embed(tokens[t+k+1]) and target tokens[t+k+2].
    # For k = K-1 (last module): needs tokens[t+K+1]. So need tokens up to
    # T_eff + K + 1 total. T_eff = T_full - K - 1.
    T_eff = T_full - k_depth - 1

    tokens = torch.stack([b["tokens"] for b in batch], dim=0)       # (B, T_full)
    hiddens = torch.stack([b["hiddens"] for b in batch], dim=0)     # (B, T_full, H)

    # DeepSeek V3 MTP indexing (K=2 example):
    #   l34_hidden[t]  (t in [0, T_eff))  — trunk hidden AT position t
    #   module_0 input: hidden_prev = L34[t], embed(tokens[t+1])  → tokens[t+2]
    #   module_1 input: hidden_prev = h_0[t], embed(tokens[t+2])  → tokens[t+3]
    #
    # Deployment analog: currentPos = P, carry = L34[P-1] (approx), nextID = tokens[P].
    #   module_0(L34[P-1], embed(tokens[P] = nextID)) → tokens[P+1] = d_0
    #   module_1(h_0,      embed(d_0))                → tokens[P+2] = d_1

    return {
        "l34_hidden": hiddens[:, :T_eff, :],              # (B, T_eff, H)
        "input_tokens": tokens[:, :T_eff + k_depth + 1],  # (B, T_eff + K + 1)
    }
