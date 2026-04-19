#!/usr/bin/env python3
"""Train the EAGLE-3 draft against a W4A8 target corpus with K=3 TTT rollout.

Pairs with `collect_eagle_hidden_states_w4a8.py`, which collected
`hidden_at_L{8,17,34}` + target argmax at every position by running the
deployed chunk{1..4}.mlpackage via coremltools on Mac ANE. That data
matches the on-device decoder's numerics (4-bit palettized + ANE fp16),
so training against it closes the accept-rate gap the bf16 PyTorch
collector caused.

Architecture must match `test_eagle3_infer.py` (draft) and the CoreML
export in `build_eagle3.py` so the trained checkpoint round-trips into
eagle3_draft.mlpackage / eagle3_fusion.mlpackage.

Usage:
    python conversion/train_eagle3_ttt.py \
        --data /path/to/training_data_w4a8 \
        --lm-head /path/to/lm_head_weight.bin  \
        --out-dir /path/to/eagle3_out \
        --epochs 5 --batch 32 --K 3

The `--lm-head` arg points to a (vocab, hidden) fp16 numpy dump of the
target LM head, used in the projection step of the draft's loss.
Generate it once from the HF model:
    python -c "import torch; from transformers import \
      AutoModelForCausalLM as M; \
      w = M.from_pretrained('google/gemma-4-E2B-it').lm_head.weight.half().cpu().numpy(); \
      w.tofile('lm_head_weight.bin')"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Draft architecture (must match test_eagle3_infer.py / build_eagle3.py) ─

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        n = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * n).to(dtype) * self.weight


class RMSNormNoScale(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        n = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * n).to(dtype)


def build_rope(max_seq: int, head_dim: int, theta: float,
               device: torch.device, dtype=torch.float32) -> tuple:
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    pos = torch.arange(max_seq, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", pos, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class FeatureFusion(nn.Module):
    def __init__(self, hidden: int, n_layers: int, rms_eps: float):
        super().__init__()
        self.proj = nn.Linear(hidden * n_layers, hidden, bias=False)
        self.norm = RMSNorm(hidden, eps=rms_eps)

    def forward(self, layer_hiddens: list[torch.Tensor]) -> torch.Tensor:
        return self.norm(self.proj(torch.cat(layer_hiddens, dim=-1)))


class DraftDecoderLayer(nn.Module):
    def __init__(self, hidden: int, num_heads: int, num_kv: int,
                 head_dim: int, ffn: int, rms_eps: float):
        super().__init__()
        self.H = hidden
        self.NH = num_heads
        self.NKV = num_kv
        self.HD = head_dim
        self.pre_attn_norm = RMSNorm(hidden, rms_eps)
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv    * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv    * head_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, rms_eps)
        self.k_norm = RMSNorm(head_dim, rms_eps)
        self.v_norm = RMSNormNoScale(rms_eps)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        self.pre_ffn_norm = RMSNorm(hidden, rms_eps)
        self.gate_proj = nn.Linear(hidden, ffn, bias=False)
        self.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                causal: bool = True) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.pre_attn_norm(x)
        q = self.q_proj(h).view(B, T, self.NH,  self.HD).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.NKV, self.HD).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.NKV, self.HD).transpose(1, 2)
        q = self.q_norm(q); k = self.k_norm(k); v = self.v_norm(v)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        rep = self.NH // self.NKV
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.NH * self.HD)
        x = x + self.o_proj(attn)
        h = self.pre_ffn_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class EAGLE3Draft(nn.Module):
    def __init__(self, cfg: dict, lm_head_weight: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.fusion = FeatureFusion(cfg["hidden"], len(cfg["fusion_layers"]), cfg["rms_eps"])
        self.input_proj = nn.Linear(cfg["hidden"] * 2, cfg["hidden"], bias=False)
        self.layer = DraftDecoderLayer(
            cfg["hidden"], cfg["num_heads"], cfg["num_kv"],
            cfg["head_dim"], cfg["ffn"], cfg["rms_eps"])
        self.final_norm = RMSNorm(cfg["hidden"], cfg["rms_eps"])
        # lm_head kept frozen, used for logit projection during training.
        self.register_buffer("lm_head_weight", lm_head_weight, persistent=False)

    def step(self, h_prev: torch.Tensor, e_next: torch.Tensor,
             cos: torch.Tensor, sin: torch.Tensor,
             is_sequence: bool = False) -> tuple:
        x = torch.cat([h_prev, e_next], dim=-1)
        x = self.input_proj(x)
        x = self.layer(x, cos, sin, causal=is_sequence)
        x = self.final_norm(x)
        logits = F.linear(x.float(), self.lm_head_weight.float())
        return x, logits

    def fuse_target(self, layer_hiddens: list[torch.Tensor]) -> torch.Tensor:
        return self.fusion(layer_hiddens)


# ── Dataset ────────────────────────────────────────────────────────────────

ROW_DTYPE = np.dtype([
    ("h_low",      np.float16, (1536,)),
    ("h_mid",      np.float16, (1536,)),
    ("h_high",     np.float16, (1536,)),
    ("tok_input",  np.int32),
    ("tok_argmax", np.int32),
])


class W4A8Dataset(torch.utils.data.Dataset):
    """Training samples are K+1 consecutive positions pulled from the memmap.

    Each sample:
      - layer hiddens at position t  → (h_low, h_mid, h_high)
      - embed inputs at t+1..t+K      → via tok_input (lookup table in __init__)
      - target argmaxes at t+1..t+K  → via tok_argmax

    We build a list of valid start indices (respecting sequence boundaries
    from the manifest) so we never cross a boundary within one sample.
    """

    def __init__(self, data_path: Path, manifest_path: Path, K: int,
                 embed_lookup: torch.Tensor, embed_scale: float):
        self.K = K
        self.data = np.memmap(data_path, dtype=ROW_DTYPE, mode="r")
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.embed_lookup = embed_lookup  # (vocab, hidden) fp16
        self.embed_scale = float(embed_scale)
        # Valid starts: every position t where t..t+K all lie in the same sequence.
        starts = self.manifest["seq_starts"]
        ends = starts[1:] + [self.manifest["num_rows"]]
        valid: list[int] = []
        for s, e in zip(starts, ends):
            # Positions s..e-1. Valid start t in [s, e-K-1].
            if e - s > K:
                valid.extend(range(s, e - K))
        self.valid_starts = np.asarray(valid, dtype=np.int64)
        print(f"[Data] rows={self.manifest['num_rows']} sequences={len(starts)} "
              f"valid_starts={len(self.valid_starts)} (K={K})")

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> dict:
        t = int(self.valid_starts[idx])
        rec = self.data[t]
        h_low  = torch.from_numpy(np.asarray(rec["h_low"]).copy())
        h_mid  = torch.from_numpy(np.asarray(rec["h_mid"]).copy())
        h_high = torch.from_numpy(np.asarray(rec["h_high"]).copy())
        # Tokens used as draft inputs (e_next at each unroll step) and as
        # training targets. At corpus position t:
        #   fused = fuse(h[t])  — context up to position t
        #   Step k ∈ 0..K-1:
        #     e_next   = embed(corpus token at t+1+k)  — teacher forces context
        #                                                up to position t+1+k
        #     target   = tok_argmax at t+1+k           — target's argmax at
        #                                                t+1+k = prediction
        #                                                for position t+2+k
        tok_inputs = np.empty(self.K, dtype=np.int64)
        tok_targets = np.empty(self.K, dtype=np.int64)
        for k in range(self.K):
            tok_inputs[k]  = self.data[t + 1 + k]["tok_input"]
            tok_targets[k] = self.data[t + 1 + k]["tok_argmax"]
        return {
            "h_low":  h_low,
            "h_mid":  h_mid,
            "h_high": h_high,
            "tok_inputs":  torch.from_numpy(tok_inputs),
            "tok_targets": torch.from_numpy(tok_targets),
        }


def collate(samples: list[dict]) -> dict:
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


# ── Train loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"[Train] device={device}")

    manifest_path = Path(str(args.data) + ".manifest.json")
    data_path = Path(str(args.data) + ".bin")

    # Config (matches eagle3_config.json format).
    cfg = {
        "hidden": 1536,
        "num_heads": 8,
        "num_kv": 1,
        "head_dim": 256,
        "ffn": 6144,
        "vocab": 262144,
        "rms_eps": 1e-6,
        "rope_theta": 10000.0,
        "embed_scale": 39.191835884530846,
        "fusion_layers": [8, 17, 34],
        "ttt_k": args.K,
    }

    # lm_head weight + embed table (both fp16 on disk, (vocab, hidden)).
    print(f"[Train] loading lm_head from {args.lm_head}")
    lm_head = np.fromfile(args.lm_head, dtype=np.float16)
    lm_head = lm_head.reshape(cfg["vocab"], cfg["hidden"])
    lm_head_t = torch.from_numpy(lm_head.copy()).to(device)
    print(f"        lm_head shape={lm_head_t.shape}")

    # For training we need the TOKEN embedding (same weights as lm_head for
    # tied-embedding Gemma 4). Create it directly from the same tensor.
    # This is the unscaled embedding; dataset multiplies by embed_scale.
    embed_table = lm_head_t  # (vocab, hidden) fp16

    # Dataset.
    ds = W4A8Dataset(data_path, manifest_path, K=args.K,
                     embed_lookup=embed_table, embed_scale=cfg["embed_scale"])
    n_total = len(ds)
    if n_total < 1000:
        print(f"[Train] warning: only {n_total} training samples — expect "
              f"poor generalisation; collect more data for a real run.")
    n_test = max(1, int(n_total * args.val_frac))
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
    val_idx = perm[:n_test]
    train_idx = perm[n_test:]
    train_ds = torch.utils.data.Subset(ds, train_idx.tolist())
    val_ds   = torch.utils.data.Subset(ds, val_idx.tolist())
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate,
        pin_memory=(device.type == "cuda"))
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=collate,
        pin_memory=(device.type == "cuda"))
    print(f"[Train] split train={len(train_ds)} val={len(val_ds)}")

    # Draft.
    draft = EAGLE3Draft(cfg, lm_head_t).to(device)
    # Optionally warm-start from prior checkpoint.
    if args.init_ckpt is not None:
        state = torch.load(args.init_ckpt, map_location=device)
        sd = state.get("model", state)
        missing, unexpected = draft.load_state_dict(sd, strict=False)
        print(f"[Train] init from {args.init_ckpt} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    n_params = sum(p.numel() for p in draft.parameters())
    print(f"[Train] draft params: {n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(draft.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = args.epochs * max(1, len(train_dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # RoPE for draft's single attention layer.
    cos_r, sin_r = build_rope(32, cfg["head_dim"], cfg["rope_theta"],
                              device=device, dtype=torch.float32)

    best_val = math.inf
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eagle3_config.json", "w") as f:
        json.dump({**cfg, "num_kv_heads": cfg["num_kv"],
                   "model_id": "w4a8_deployed",
                   "architecture": "eagle3_draft"}, f, indent=2)

    def run_epoch(dl, training: bool) -> tuple:
        total_loss = 0.0
        total_match = torch.zeros(args.K, device=device)
        total_cnt = 0
        (draft.train() if training else draft.eval())
        ctx = (torch.enable_grad() if training else torch.inference_mode())
        with ctx:
            for step, batch in enumerate(dl):
                # Cast hiddens to fp32 for training (draft params are fp32).
                h_low  = batch["h_low"].to(device, dtype=torch.float32)
                h_mid  = batch["h_mid"].to(device, dtype=torch.float32)
                h_high = batch["h_high"].to(device, dtype=torch.float32)
                tok_in = batch["tok_inputs"].to(device)   # (B, K)
                tok_tg = batch["tok_targets"].to(device)  # (B, K)

                B = h_low.shape[0]
                fused = draft.fuse_target([h_low.unsqueeze(1),
                                           h_mid.unsqueeze(1),
                                           h_high.unsqueeze(1)])
                # fused: (B, 1, hidden)
                d_h = fused
                loss = 0.0
                match = torch.zeros(args.K, device=device)
                for k in range(args.K):
                    e_next = (embed_table[tok_in[:, k]].unsqueeze(1).float()
                              * float(cfg["embed_scale"]))
                    d_h, d_logits = draft.step(d_h, e_next, cos_r, sin_r,
                                               is_sequence=False)
                    # d_logits: (B, 1, vocab)
                    logits_k = d_logits[:, -1, :]
                    loss_k = F.cross_entropy(logits_k, tok_tg[:, k])
                    loss = loss + loss_k
                    with torch.no_grad():
                        match[k] = (logits_k.argmax(-1) == tok_tg[:, k]).float().mean()

                loss = loss / args.K
                if training:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
                    opt.step()
                    sched.step()

                total_loss += float(loss.item()) * B
                total_match += match * B
                total_cnt += B

                if training and step % 50 == 0:
                    print(f"    step {step}/{len(dl)} loss={loss.item():.4f} "
                          f"match@k={match.tolist()}")
        return total_loss / total_cnt, (total_match / total_cnt).tolist()

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        tr_loss, tr_match = run_epoch(train_dl, training=True)
        vl_loss, vl_match = run_epoch(val_dl, training=False)
        avg_accept = sum(vl_match) / len(vl_match)
        print(f"  train loss={tr_loss:.4f} match={tr_match}")
        print(f"  val   loss={vl_loss:.4f} match={vl_match} "
              f"avg_accept={avg_accept*100:.1f}%")

        if vl_loss < best_val:
            best_val = vl_loss
            ckpt_path = out_dir / "eagle3_draft_best.pt"
            torch.save({
                "model": draft.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "val_loss": vl_loss,
                "val_match": vl_match,
            }, ckpt_path)
            print(f"  saved best → {ckpt_path}")
        # Always save latest as well for resume.
        torch.save({"model": draft.state_dict(), "cfg": cfg, "epoch": epoch},
                   out_dir / "eagle3_draft_latest.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="Prefix of <data>.bin + <data>.manifest.json produced "
                         "by collect_eagle_hidden_states_w4a8.py")
    ap.add_argument("--lm-head", type=Path, required=True,
                    help="Path to (vocab, hidden) fp16 lm_head weight dump")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--init-ckpt", type=Path, default=None,
                    help="Optional prior checkpoint (e.g. bf16-trained) to "
                         "warm-start the weights. Fine-tuning from this is "
                         "usually faster than from scratch.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str,
                    default=("mps" if torch.backends.mps.is_available()
                             else ("cuda" if torch.cuda.is_available() else "cpu")))
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
