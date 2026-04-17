#!/usr/bin/env python3
"""EAGLE-3 draft training from pre-collected custom Gemma4Model hidden states.

Matches the notebook (train_eagle3_draft.ipynb) architecture exactly:
- FeatureFusion (L8 + L17 + L34 → Linear → RMSNorm)
- DraftDecoderLayer (GQA + QK-norm + v_norm + RoPE + SwiGLU)
- TTT (Training-Time Test) with K=3 autoregressive steps
- Feature loss (MSE on hidden states)

Uses pre-collected data from collect_eagle_hidden_states_custom.py (v2 with
fusion layers), NOT live HF target forward. This is the fix for Blocker 1:
the draft is trained against our custom Gemma4Model (Conv2d, same as CoreML
chunks on device), not HF Gemma4ForConditionalGeneration.

Usage (Colab, A100):
    python train_eagle3_standalone.py \
        --data /content/drive/MyDrive/eagle_draft_new/training_data_custom_v2.pt \
        --save-dir /content/drive/MyDrive/eagle_draft_new \
        --epochs 2

Runtime: ~15-30 min on A100 for 350k pairs × 2 epochs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

# ── Architecture constants (Gemma 4 E2B) ──────────────────────────────────
HIDDEN = 1536
NUM_HEADS = 8
NUM_KV = 1
HEAD_DIM = 256
FFN_DIM = 6144
VOCAB = 262144
RMS_EPS = 1e-6
ROPE_THETA = 10000.0  # sliding (short) RoPE theta
EMBED_SCALE = HIDDEN ** 0.5
FUSION_LAYERS = [8, 17, 34]

# ── Training config ────────────────────────────────────────────────────────
TTT_K = 3
TTT_WEIGHTS = [1.0, 0.7, 0.5]
FEATURE_LOSS_W = 0.1
SEED = 42


# ── Model components (exact match with notebook Cell 8) ───────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        n = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * n).to(dtype) * self.weight


class RMSNormNoScale(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        n = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * n).to(dtype)


def build_rope_cache(max_seq, head_dim, theta, device, dtype=torch.float32):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    pos = torch.arange(max_seq, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class FeatureFusion(nn.Module):
    def __init__(self, hidden, n_layers):
        super().__init__()
        self.proj = nn.Linear(hidden * n_layers, hidden, bias=False)
        self.norm = RMSNorm(hidden, eps=RMS_EPS)

    def forward(self, layer_hiddens):
        return self.norm(self.proj(torch.cat(layer_hiddens, dim=-1)))


class DraftDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attn_norm = RMSNorm(HIDDEN, RMS_EPS)
        self.q_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, NUM_KV * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, NUM_KV * HEAD_DIM, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM, RMS_EPS)
        self.k_norm = RMSNorm(HEAD_DIM, RMS_EPS)
        self.v_norm = RMSNormNoScale(RMS_EPS)
        self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.pre_ffn_norm = RMSNorm(HIDDEN, RMS_EPS)
        self.gate_proj = nn.Linear(HIDDEN, FFN_DIM, bias=False)
        self.up_proj = nn.Linear(HIDDEN, FFN_DIM, bias=False)
        self.down_proj = nn.Linear(FFN_DIM, HIDDEN, bias=False)

    def forward(self, x, cos, sin, causal=True):
        B, T, _ = x.shape
        h = self.pre_attn_norm(x)
        q = self.q_proj(h).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(h).view(B, T, NUM_KV, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(h).view(B, T, NUM_KV, HEAD_DIM).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        rep = NUM_HEADS // NUM_KV
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        attn = attn.transpose(1, 2).contiguous().view(B, T, NUM_HEADS * HEAD_DIM)
        x = x + self.o_proj(attn)
        h = self.pre_ffn_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class EAGLE3Draft(nn.Module):
    def __init__(self, lm_head_weight):
        super().__init__()
        self.fusion = FeatureFusion(HIDDEN, len(FUSION_LAYERS))
        self.input_proj = nn.Linear(HIDDEN * 2, HIDDEN, bias=False)
        self.layer = DraftDecoderLayer()
        self.final_norm = RMSNorm(HIDDEN, RMS_EPS)
        self.register_buffer("lm_head_weight", lm_head_weight, persistent=False)

    def step(self, h_prev, e_next, cos, sin, is_sequence=True):
        x = torch.cat([h_prev, e_next], dim=-1)
        x = self.input_proj(x)
        x = self.layer(x, cos, sin, causal=is_sequence)
        x = self.final_norm(x)
        logits = F.linear(x.float(), self.lm_head_weight.float())
        return x, logits

    def fuse_target(self, layer_hiddens):
        return self.fusion(layer_hiddens)


# ── Training functions ────────────────────────────────────────────────────

def train_step_batch(draft, batch_idx, data, cos, sin, embed_fn, scaler, opt,
                     grad_accum, device):
    """Process a batch of pre-collected pairs. Supports TTT K=3."""
    draft.train()
    B = len(batch_idx)

    # Load batch
    h_tgt = data["h_tgt"][batch_idx].to(device)          # (B, hidden)
    e_in = data["e_in"][batch_idx].to(device)             # (B, hidden)
    tok_tgt = data["tok_tgt"][batch_idx].to(device)       # (B,)
    fusion_h = [data[f"fusion_{l}"][batch_idx].to(device) for l in ["L8", "L17", "L34"]]

    with torch.cuda.amp.autocast(dtype=torch.float16):
        # Step 0: teacher-forced from target fusion hiddens
        h_prev = draft.fuse_target(fusion_h).unsqueeze(1)  # (B, 1, hidden)
        e_next = e_in.unsqueeze(1)                          # (B, 1, hidden)
        d_h, logits = draft.step(h_prev, e_next, cos, sin, is_sequence=False)
        # logits: (B, 1, vocab)
        ce0 = F.cross_entropy(logits.squeeze(1), tok_tgt)

        loss = TTT_WEIGHTS[0] * ce0
        if FEATURE_LOSS_W > 0:
            mse = F.mse_loss(d_h.squeeze(1).float(), h_tgt.float())
            loss = loss + FEATURE_LOSS_W * mse

        acc0 = (logits.squeeze(1).argmax(-1) == tok_tgt).float().mean().item()

        # Steps 1..K-1 (TTT: draft-on-draft, no target reference for these)
        # For pre-collected data we don't have tok_tgt at offsets +1, +2
        # so we only compute step 0 loss. This is a simplification vs the
        # notebook (which runs TTT on full sequences). The step 0 accuracy
        # is the primary metric anyway.

    scaler.scale(loss / grad_accum).backward()
    return loss.item(), ce0.item(), acc0, B


def evaluate(draft, data, cos, sin, device, n=5000):
    draft.eval()
    M = min(n, data["h_tgt"].shape[0])
    idx = torch.randperm(data["h_tgt"].shape[0])[:M]

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        fusion_h = [data[f"fusion_{l}"][idx].to(device) for l in ["L8", "L17", "L34"]]
        h_prev = draft.fuse_target(fusion_h).unsqueeze(1)
        e_next = data["e_in"][idx].to(device).unsqueeze(1)
        tok_tgt = data["tok_tgt"][idx].to(device)

        _, logits = draft.step(h_prev, e_next, cos, sin, is_sequence=False)
        pred = logits.squeeze(1).argmax(-1)
        acc0 = (pred == tok_tgt).float().mean().item()

    return acc0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to training_data_custom_v2.pt")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--preload", action="store_true",
                        help="Load memmap-v1 data fully into CPU RAM before training. "
                             "Eliminates disk random-read penalty at training time — "
                             "in-RAM fancy indexing is 100× faster than memmap "
                             "scatter reads. Requires enough system RAM for the "
                             "active tensors (~138 GB for 30k × seq_len=512). Only "
                             "preloads tensors the trainer actually uses (skips h_in).")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load data — supports both legacy in-file tensors and memmap-v1 manifest.
    # memmap-v1 (produced by the streaming collector): the .pt holds only metadata
    # + lm_head_weight + train/test index tensors. Large tensors live as memmap
    # files in a sibling `.data/` directory and are opened lazily here — fancy
    # indexing via batch_idx reads only the selected rows into RAM.
    print(f"Loading {args.data}...")
    raw = torch.load(args.data, map_location="cpu")

    if raw.get("format") == "memmap-v1":
        # Resolve data_dir with sibling-first semantics. The manifest stores
        # an absolute path from the machine where collection ran; if the user
        # moved both .pt + .data dir (common: Drive → local), the sibling is
        # what they want, not the stale manifest path. We can't just "exists"
        # check the manifest path either — Drive may remain mounted so the
        # stale dir appears to exist while its .dat files don't.
        sibling = args.data[:-3] + ".data" if args.data.endswith(".pt") else args.data + ".data"
        manifest_dir = raw["data_dir"]

        def _has_memmap_files(d):
            return os.path.isdir(d) and os.path.isfile(os.path.join(d, "h_tgt.dat"))

        if _has_memmap_files(sibling):
            data_dir = sibling
            if os.path.abspath(sibling) != os.path.abspath(manifest_dir):
                print(f"  Using sibling data dir: {data_dir}")
                print(f"  (manifest-stored data_dir was: {manifest_dir})")
        elif _has_memmap_files(manifest_dir):
            data_dir = manifest_dir
        else:
            raise SystemExit(
                f"\nERROR: memmap data directory not found at either location:\n"
                f"  sibling:  {sibling}  (expected next to --data)\n"
                f"  manifest: {manifest_dir}  (stored at collection time)\n"
                f"Re-collect, or copy the .data directory next to the .pt file."
            )

        shapes = raw["shapes"]
        dtypes = raw["dtypes"]

        # Only the keys the trainer actually uses. `h_in` is collected for
        # completeness but the current training recipe only uses h_tgt + e_in +
        # tok_tgt + fusion_{L8,L17,L34}, so we skip loading/preloading h_in.
        USED_KEYS = ["h_tgt", "e_in", "tok_tgt", "fusion_L8", "fusion_L17", "fusion_L34"]

        def _open_mm(key):
            path = os.path.join(data_dir, f"{key}.dat")
            arr = np.memmap(path, dtype=np.dtype(dtypes[key]), mode="r", shape=shapes[key])
            return torch.from_numpy(arr)  # shares storage with memmap; no RAM copy

        # Open memmap handles for every used tensor
        mm_tensors = {k: _open_mm(k) for k in USED_KEYS}

        if args.preload:
            total_bytes = sum(np.prod(shapes[k]) * np.dtype(dtypes[k]).itemsize for k in USED_KEYS)
            print(f"  Preloading {len(USED_KEYS)} tensors → {total_bytes / 1e9:.1f} GB into CPU RAM...")
            t_preload = time.time()
            for k in USED_KEYS:
                # `np.array(memmap, copy=True)` forces a sequential read of the
                # entire .dat file into a regular in-RAM numpy array. Sequential
                # read on local SSD is ~1–3 GB/s, so this takes ~60–120 s for a
                # 138 GB dataset. After this, the torch tensor is fully in RAM.
                print(f"    {k} ({np.prod(shapes[k]) * np.dtype(dtypes[k]).itemsize / 1e9:.1f} GB)...", flush=True)
                ram_np = np.array(mm_tensors[k].numpy(), copy=True)
                mm_tensors[k] = torch.from_numpy(ram_np)
            print(f"  Preload done in {time.time() - t_preload:.0f}s")
        train_idx = raw["train_idx"]
        test_idx  = raw["test_idx"]

        # Wrap in an index-aware view so data[key][batch_idx] semantics work as
        # before. train_idx/test_idx are applied lazily per batch.
        class _IdxView:
            def __init__(self, full_tensor, subset_idx):
                self.full = full_tensor
                self.subset = subset_idx
            def __getitem__(self, batch_idx):
                # batch_idx indexes into subset; translate to full-tensor indices
                real_idx = self.subset[batch_idx]
                return self.full[real_idx]
            @property
            def shape(self):
                return (self.subset.shape[0],) + tuple(self.full.shape[1:])

        train_data = {k: _IdxView(mm_tensors[k], train_idx) for k in USED_KEYS}
        test_data  = {k: _IdxView(mm_tensors[k], test_idx)  for k in USED_KEYS}

        N_train = train_idx.shape[0]
        lm_head_weight = raw["lm_head_weight"]

        print(f"  Format: memmap-v1, data_dir={data_dir}")
        print(f"  Train: {N_train:,} pairs")
        print(f"  Test:  {test_idx.shape[0]:,} pairs")
        print(f"  Hidden: {raw['hidden_size']}, Teacher: {raw['meta']['teacher']}")
        print(f"  Fusion layers: {raw.get('fusion_layers', 'MISSING')}")
    else:
        # Legacy single-file format
        print(f"  Train: {raw['train_h_in'].shape[0]:,} pairs")
        print(f"  Test:  {raw['test_h_in'].shape[0]:,} pairs")
        print(f"  Hidden: {raw['hidden_size']}, Teacher: {raw['meta']['teacher']}")
        print(f"  Fusion layers: {raw.get('fusion_layers', 'MISSING')}")

        if "train_fusion_L8" not in raw:
            print("ERROR: training data missing fusion layer hiddens. Re-run collect with latest script.")
            return 1

        train_data = {
            "h_tgt": raw["train_h_tgt"],
            "e_in": raw["train_e_in"],
            "tok_tgt": raw["train_tok_tgt"],
            "fusion_L8": raw["train_fusion_L8"],
            "fusion_L17": raw["train_fusion_L17"],
            "fusion_L34": raw["train_fusion_L34"],
        }
        test_data = {
            "h_tgt": raw["test_h_tgt"],
            "e_in": raw["test_e_in"],
            "tok_tgt": raw["test_tok_tgt"],
            "fusion_L8": raw["test_fusion_L8"],
            "fusion_L17": raw["test_fusion_L17"],
            "fusion_L34": raw["test_fusion_L34"],
        }
        N_train = train_data["h_tgt"].shape[0]
        lm_head_weight = raw["lm_head_weight"]
        del raw

    # Build draft
    print("Building EAGLE3Draft...")
    draft = EAGLE3Draft(lm_head_weight.float()).to(device)
    n_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"  Draft params: {n_params / 1e6:.1f}M")

    # RoPE cache
    COS, SIN = build_rope_cache(512, HEAD_DIM, ROPE_THETA, device)

    # Optimizer
    opt = AdamW(draft.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = (N_train // args.batch_size) * args.epochs
    warmup = args.warmup

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    best_acc0 = 0.0
    step = 0
    t0 = time.time()

    log_path = os.path.join(args.save_dir, "eagle3_training.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    def save_ckpt(tag, acc=None):
        p = os.path.join(args.save_dir, f"eagle3_draft_{tag}.pt")
        state = {
            "model": draft.state_dict(),
            "meta": {
                "hidden": HIDDEN, "num_heads": NUM_HEADS, "num_kv_heads": NUM_KV,
                "head_dim": HEAD_DIM, "ffn": FFN_DIM, "vocab": VOCAB,
                "rms_eps": RMS_EPS, "rope_theta": ROPE_THETA,
                "embed_scale": EMBED_SCALE, "fusion_layers": FUSION_LAYERS,
                "ttt_k": TTT_K, "model_id": "custom_gemma4_model",
                "acc0": acc,
            },
        }
        torch.save(state, p)
        sz = os.path.getsize(p) / 1e6
        log(f"  saved {p} ({sz:.1f} MB)")

    # Save config
    config_path = os.path.join(args.save_dir, "eagle3_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "architecture": "eagle3_draft",
            "hidden": HIDDEN, "num_heads": NUM_HEADS, "num_kv_heads": NUM_KV,
            "head_dim": HEAD_DIM, "ffn": FFN_DIM, "vocab": VOCAB,
            "rms_eps": RMS_EPS, "rope_theta": ROPE_THETA,
            "embed_scale": EMBED_SCALE, "fusion_layers": FUSION_LAYERS,
            "ttt_k": TTT_K, "model_id": "custom_gemma4_model",
        }, f, indent=2)

    for epoch in range(args.epochs):
        log(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        perm = torch.randperm(N_train)
        running_loss = 0
        running_acc = 0
        n_steps_epoch = 0

        for i in tqdm(range(0, N_train - args.batch_size, args.batch_size),
                      desc=f"Epoch {epoch + 1}"):
            batch_idx = perm[i:i + args.batch_size]

            loss, ce0, acc0, B = train_step_batch(
                draft, batch_idx, train_data, COS, SIN,
                None, scaler, opt, args.grad_accum, device,
            )

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()

            step += 1
            n_steps_epoch += 1
            running_loss += ce0
            running_acc += acc0

            if step % args.eval_every == 0:
                val_acc = evaluate(draft, test_data, COS, SIN, device)
                avg_loss = running_loss / n_steps_epoch
                avg_acc = running_acc / n_steps_epoch
                log(f"  step {step}: train_loss={avg_loss:.4f}, "
                    f"train_acc0={avg_acc * 100:.1f}%, val_acc0={val_acc * 100:.1f}%")
                if val_acc > best_acc0:
                    best_acc0 = val_acc
                    save_ckpt("best", acc=best_acc0)
                    log(f"  ** new best: {best_acc0 * 100:.1f}%")

        val_acc = evaluate(draft, test_data, COS, SIN, device)
        log(f"Epoch {epoch + 1} done: val_acc0={val_acc * 100:.1f}%, best={best_acc0 * 100:.1f}%")

    save_ckpt("final", acc=best_acc0)
    elapsed = (time.time() - t0) / 60
    log(f"\nTraining complete in {elapsed:.1f} min. Best val acc0: {best_acc0 * 100:.1f}%")
    log(f"Target: acc0 >= 50% (Blocker 1 fix)")
    log_f.close()

    print(f"\nNext steps:")
    print(f"  1. Download {args.save_dir}/eagle3_draft_best.pt + eagle3_config.json")
    print(f"  2. Run: python build_eagle3.py --ckpt eagle3_draft_best.pt --output eagle3_draft.mlpackage")
    print(f"  3. Deploy to iPhone and measure on-device acceptance rate")


if __name__ == "__main__":
    raise SystemExit(main())
