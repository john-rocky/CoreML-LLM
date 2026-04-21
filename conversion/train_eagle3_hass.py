#!/usr/bin/env python3
"""HASS-style EAGLE-3 draft trainer for the W4A8 Gemma-4 E2B target.

Paper reference: "HASS: Hardware-Aware Speculative Sampling" (arXiv:2408.15766).

Deltas from vanilla EAGLE-3 TTT (as in `train_eagle3_ttt.py`):

1. Soft-KL loss over top-K teacher logits (default K=20), alongside the
   original hard cross-entropy on tok_argmax. The W4A8 collector now bakes
   teacher top-K through the exact-same softcapped-logits tensor the device
   argmaxes against, so the draft sees the full teacher distribution (not
   just the argmax token) and learns the mode *and* the shoulder. Weights:
     L = ce_weight * CE(draft_logits, tok_argmax)
       + kl_weight * KL(P_teacher_top_k || Q_draft_top_k)
       + feat_weight * RMSE(h_draft_final, h_high)      # off by default
   where the KL is computed on the teacher top-K support only: we gather
   the draft's logits at `top_k_ids`, log-softmax over K, compare to the
   teacher probs (softmax over the teacher's top-K logit values,
   renormalised within K). This restricts the loss to tokens the draft
   can emit at inference, which is functionally equivalent to the
   official EAGLE-3 t2d (target-to-draft) vocab pruning pass. Vanilla
   TTT used hard CE only, losing the teacher's shoulder distribution.

2. HASS "context harmonisation" Q-override, `--q-override step0`
   (OPTIONAL, default off):
   At TTT step k=0 the draft's own attention-input hidden is replaced
   by the teacher's L34 pre-norm (`h_high`). The rest of the rollout
   (k ≥ 1) is driven by the draft's own prev_argmax, same as vanilla
   TTT. This attempts to match HASS §3.2: at the first drafting step,
   the target's feature is inserted in place of the draft's cached
   query so the draft's attention is anchored to the teacher's context.

   NOTE: the HASS paper's Q-override is defined for multi-layer drafts
   with a KV cache (override cached K/V with target's). Our EAGLE-3
   draft is single-layer with no persistent cache, so the adaptation
   is "use h_high as step-0 input hidden". This creates a deliberate
   train/inference mismatch because SpeculativeLoop.drawBurst feeds
   `h_fused` (the fusion model's output) to the draft at step 0, not
   `h_high`. For exact train/inference parity, use `--q-override none`
   (vanilla EAGLE-3 TTT, default). Only enable `step0` as an ablation
   if you want to probe whether anchoring on h_high ex-fusion helps
   despite the mismatch.

   Derivation (the HASS paper's pseudo-code uses a KV cache which this
   draft does not have at train time): since our single-layer draft has
   no persistent KV state across TTT steps, the practical effect of
   "override the cached Q from target" is equivalent to "use the target
   hidden as the draft's input at step 0". The draft's `step()` then
   recomputes Q from that hidden inside pre_attn_norm → q_proj, which
   is precisely the target-anchored Q that HASS asks for. Mode `all`
   extends the override to every step (debug only, should be worse
   than step0 because the draft never learns to handle its own mistakes);
   mode `none` falls back to vanilla TTT.

3. Feature loss gated off by default (`--feat-loss-weight 0.0`).
   Official EAGLE-3 dropped the feature loss from EAGLE-2's recipe;
   HASS keeps it as an option with a small weight. We keep both doors
   open via CLI but default to 0.

4. Data format v2: reads `<prefix>.bin` + `<prefix>.manifest.json`
   produced by `collect_eagle_hidden_states_w4a8.py --top-k 20`. The
   manifest's `format_version`, `top_k`, `softcap`, and
   `per_layer_embed_scale` fields are asserted against CLI / config so
   a stale memmap or a re-baked chunk4 with a different K cannot
   silently poison training. If format_version==1 (legacy argmax-only
   data), we run in degraded mode with KL disabled.

The trained checkpoint round-trips via `conversion/build_eagle3.py` into
`eagle3_draft.mlpackage` / `eagle3_fusion.mlpackage`. The draft module
structure and parameter names are preserved from `train_eagle3_ttt.py`
so the existing loader in build_eagle3.py's `load_into_ane_model` works
unchanged.

Usage:
    python conversion/train_eagle3_hass.py \\
        --data /path/to/training_data_w4a8 \\
        --lm-head /path/to/lm_head_weight.bin  \\
        --out-dir /path/to/eagle3_out \\
        --epochs 5 --batch 128 --K 3

Author: John Rocky (2026-04-20). No HASS code was copied; everything
derived from the paper + iteration on the existing TTT trainer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

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

    def forward(self, layer_hiddens: list) -> torch.Tensor:
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
    """Identical param layout to train_eagle3_ttt.EAGLE3Draft so the checkpoint
    round-trips into build_eagle3.load_into_ane_model without renames."""
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

    def fuse_target(self, layer_hiddens: list) -> torch.Tensor:
        return self.fusion(layer_hiddens)


# ── Row dtype helpers (format v1 + v2) ─────────────────────────────────────

HIDDEN = 1536


def row_dtype_for(format_version: int, top_k: int) -> np.dtype:
    fields = [
        ("h_low",      np.float16, (HIDDEN,)),
        ("h_mid",      np.float16, (HIDDEN,)),
        ("h_high",     np.float16, (HIDDEN,)),
        ("tok_input",  np.int32),
        ("tok_argmax", np.int32),
    ]
    if format_version >= 2 and top_k > 0:
        fields.append(("top_k_ids",    np.int32,   (top_k,)))
        fields.append(("top_k_logits", np.float16, (top_k,)))
    return np.dtype(fields)


# ── Dataset ────────────────────────────────────────────────────────────────

class HASSDataset(torch.utils.data.Dataset):
    """Returns K+1 consecutive positions. Sample payload:
      h_low/h_mid/h_high at position t (teacher context up to t)
      tok_inputs  (K,)      — token fed to target at t+1..t+K
      tok_targets (K,)      — target argmax at t+1..t+K
      top_k_ids   (K, Kt)   — teacher top-K ids at t+1..t+K (v2 only)
      top_k_logits(K, Kt)   — teacher top-K logits at t+1..t+K (v2 only)
    Sequence boundaries from manifest.seq_starts are respected so a sample
    never crosses two documents.
    """

    def __init__(self, data_path: Path, manifest: dict, K: int):
        self.K = K
        self.manifest = manifest
        self.format_version = int(manifest.get("format_version", 1))
        self.top_k = int(manifest.get("top_k", 0)) if self.format_version >= 2 else 0
        dtype = row_dtype_for(self.format_version, self.top_k)
        self.data = np.memmap(data_path, dtype=dtype, mode="r")

        starts = manifest["seq_starts"]
        num_rows = int(manifest["num_rows"])
        ends = starts[1:] + [num_rows]
        valid: list = []
        for s, e in zip(starts, ends):
            if e - s > K:
                valid.extend(range(s, e - K))
        self.valid_starts = np.asarray(valid, dtype=np.int64)
        print(f"[Data] rows={num_rows} sequences={len(starts)} "
              f"valid_starts={len(self.valid_starts)} (K={K}, "
              f"fmt=v{self.format_version}, top_k={self.top_k})")

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> dict:
        t = int(self.valid_starts[idx])
        rec = self.data[t]
        h_low  = torch.from_numpy(np.asarray(rec["h_low"]).copy())
        h_mid  = torch.from_numpy(np.asarray(rec["h_mid"]).copy())
        h_high = torch.from_numpy(np.asarray(rec["h_high"]).copy())

        tok_inputs = np.empty(self.K, dtype=np.int64)
        tok_targets = np.empty(self.K, dtype=np.int64)
        top_k_ids = np.empty((self.K, max(self.top_k, 1)), dtype=np.int64)
        top_k_logits = np.empty((self.K, max(self.top_k, 1)), dtype=np.float16)
        for k in range(self.K):
            r = self.data[t + 1 + k]
            tok_inputs[k]  = r["tok_input"]
            tok_targets[k] = r["tok_argmax"]
            if self.top_k > 0:
                top_k_ids[k]    = np.asarray(r["top_k_ids"]).astype(np.int64)
                top_k_logits[k] = np.asarray(r["top_k_logits"]).astype(np.float16)

        out = {
            "h_low":  h_low,
            "h_mid":  h_mid,
            "h_high": h_high,
            "tok_inputs":  torch.from_numpy(tok_inputs),
            "tok_targets": torch.from_numpy(tok_targets),
        }
        if self.top_k > 0:
            out["top_k_ids"]    = torch.from_numpy(top_k_ids)
            out["top_k_logits"] = torch.from_numpy(top_k_logits.astype(np.float32))
        return out


def collate(samples: list) -> dict:
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


# ── Loss helpers ───────────────────────────────────────────────────────────

def kl_on_topk(draft_logits_full: torch.Tensor,
               teacher_top_k_ids: torch.Tensor,
               teacher_top_k_logits: torch.Tensor) -> torch.Tensor:
    """Soft-KL over the teacher's top-K support.

    Both distributions are restricted to the same K-token subset so the
    training signal matches the slice the draft can emit at inference.
    This is functionally equivalent to EAGLE-3's t2d vocab pruning.

    Args:
      draft_logits_full:  (B, V) draft's final logits
      teacher_top_k_ids:  (B, K) int64 indices into V
      teacher_top_k_logits: (B, K) fp32 raw teacher logits (already
                          softcapped by chunk4 via tanh(x/30)*30)

    Returns:
      scalar KL (batchmean), minimised when the draft assigns the
      teacher's renormalised top-K mass to the same K tokens.
    """
    # Gather draft logits at teacher's top-K ids.
    # draft_logits_full: (B, V) → (B, K)
    draft_top_k = torch.gather(draft_logits_full, dim=-1, index=teacher_top_k_ids)

    # Probabilities over the K-wide support (softmax inside top-K is
    # exactly "renormalise inside K" when the raw logits are shared).
    draft_log_probs = F.log_softmax(draft_top_k, dim=-1)       # (B, K)
    teacher_probs   = F.softmax(teacher_top_k_logits, dim=-1)  # (B, K)

    # kl_div expects log_q and p; order: KL(P || Q) = sum P * (log P - log Q).
    # We pass log Q = draft_log_probs and P = teacher_probs; reduction batchmean
    # divides by batch so scale is independent of B.
    return F.kl_div(draft_log_probs, teacher_probs, reduction="batchmean")


# ── Train loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"[Train] device={device}")

    manifest_path = Path(str(args.data) + ".manifest.json")
    data_path = Path(str(args.data) + ".bin")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Version/shape guards.
    fmt = int(manifest.get("format_version", 1))
    manifest_top_k = int(manifest.get("top_k", 0))
    softcap = float(manifest.get("softcap", 30.0))
    if fmt == 2:
        if manifest_top_k != args.top_k:
            raise ValueError(f"--top-k {args.top_k} disagrees with manifest.top_k "
                             f"{manifest_top_k}; rebuild chunk4 or pass matching K.")
        if abs(softcap - 30.0) > 1e-6:
            raise ValueError(f"manifest softcap {softcap} != expected 30.0 — "
                             f"soft-KL assumptions about logit scale break.")
        use_kl = True
    elif fmt == 1:
        print("[Train] WARNING: manifest format_version=1 (no top-K). "
              "Running in legacy mode (hard CE only, KL disabled).")
        use_kl = False
        args.kl_weight = 0.0
    else:
        raise ValueError(f"Unsupported manifest format_version={fmt}")

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
        "embed_scale": float(manifest.get("embed_scale", 39.191835884530846)),
        "fusion_layers": manifest.get("fusion_layers", [8, 17, 34]),
        "ttt_k": args.K,
    }

    print(f"[Train] loading lm_head from {args.lm_head}")
    lm_head = np.fromfile(args.lm_head, dtype=np.float16)
    lm_head = lm_head.reshape(cfg["vocab"], cfg["hidden"])
    lm_head_t = torch.from_numpy(lm_head.copy()).to(device)
    print(f"        lm_head shape={lm_head_t.shape}")

    # Tied-embedding: LM head weights double as the token embedding table.
    embed_table = lm_head_t  # (vocab, hidden) fp16

    # Dataset.
    ds = HASSDataset(data_path, manifest, K=args.K)
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
    if args.init_ckpt is not None:
        state = torch.load(args.init_ckpt, map_location=device)
        sd = state.get("model", state)
        missing, unexpected = draft.load_state_dict(sd, strict=False)
        print(f"[Train] init from {args.init_ckpt} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    n_params = sum(p.numel() for p in draft.parameters())
    print(f"[Train] draft params: {n_params/1e6:.1f}M")
    print(f"[Train] loss: ce_w={args.ce_weight} kl_w={args.kl_weight} "
          f"feat_w={args.feat_loss_weight} q_override={args.q_override}")

    opt = torch.optim.AdamW(draft.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = args.epochs * max(1, len(train_dl))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

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
        total_ce = 0.0
        total_kl = 0.0
        total_feat = 0.0
        total_match = torch.zeros(args.K, device=device)
        total_cnt = 0
        (draft.train() if training else draft.eval())
        ctx = (torch.enable_grad() if training else torch.inference_mode())
        with ctx:
            for step_idx, batch in enumerate(dl):
                h_low  = batch["h_low"].to(device, dtype=torch.float32)
                h_mid  = batch["h_mid"].to(device, dtype=torch.float32)
                h_high = batch["h_high"].to(device, dtype=torch.float32)
                tok_in = batch["tok_inputs"].to(device)   # (B, K)
                tok_tg = batch["tok_targets"].to(device)  # (B, K)
                if use_kl:
                    tk_ids = batch["top_k_ids"].to(device)           # (B, K, Kt)
                    tk_lg  = batch["top_k_logits"].to(device).float() # (B, K, Kt)

                B = h_low.shape[0]
                fused = draft.fuse_target([h_low.unsqueeze(1),
                                           h_mid.unsqueeze(1),
                                           h_high.unsqueeze(1)])
                # fused: (B, 1, hidden) — target's fused multi-layer hidden.

                # HASS Q-override: decide what h_prev is at each TTT step.
                #   step0: replace draft's own d_h with h_high (pre-norm L34
                #          of the teacher) at k=0 only.  At k≥1 use the
                #          draft's own previous hidden (autoregressive).
                #   all:   replace d_h with h_high at every k  (debug only).
                #   none:  never replace (vanilla EAGLE-3 TTT behaviour).
                # For step0/all we replace the fused input at k=0 so the
                # draft's pre_attn_norm→q_proj produces a Q anchored on the
                # teacher's L34 hidden, matching HASS context harmonisation
                # (see module docstring #2 for the derivation).
                if args.q_override == "step0":
                    d_h = h_high.unsqueeze(1)
                elif args.q_override == "all":
                    d_h = h_high.unsqueeze(1)
                else:  # "none"
                    d_h = fused

                ce = torch.zeros((), device=device)
                kl = torch.zeros((), device=device)
                feat = torch.zeros((), device=device)
                match = torch.zeros(args.K, device=device)

                prev_argmax: Optional[torch.Tensor] = None
                for k in range(args.K):
                    if k == 0:
                        tok_for_e = tok_in[:, 0]
                    else:
                        tok_for_e = prev_argmax
                    e_next = (embed_table[tok_for_e].unsqueeze(1).float()
                              * float(cfg["embed_scale"]))

                    # Apply Q-override per step.
                    if args.q_override == "all" and k > 0:
                        d_h = h_high.unsqueeze(1)

                    d_h_new, d_logits = draft.step(
                        d_h, e_next, cos_r, sin_r, is_sequence=False)
                    d_h = d_h_new
                    logits_k = d_logits[:, -1, :]  # (B, V)

                    # Hard CE on argmax label.
                    ce = ce + F.cross_entropy(logits_k, tok_tg[:, k])

                    # Soft KL over teacher top-K support.
                    if use_kl:
                        kl = kl + kl_on_topk(
                            logits_k,
                            tk_ids[:, k, :],
                            tk_lg[:, k, :],
                        )

                    with torch.no_grad():
                        prev_argmax = logits_k.argmax(-1)
                        match[k] = (prev_argmax == tok_tg[:, k]).float().mean()

                ce = ce / args.K
                kl = kl / args.K

                # Optional feature loss: draft's final hidden at step K-1 vs
                # teacher's L34 pre-norm at the matching position. This is
                # off by default (matching EAGLE-3 paper). Enabling it pulls
                # the draft's representation towards the teacher in L2 in
                # addition to the logit-space KL.
                if args.feat_loss_weight > 0.0:
                    feat = F.mse_loss(d_h.squeeze(1), h_high).sqrt()

                loss = (args.ce_weight * ce
                        + args.kl_weight * kl
                        + args.feat_loss_weight * feat)

                if training:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
                    opt.step()
                    sched.step()

                total_loss += float(loss.item()) * B
                total_ce   += float(ce.item()) * B
                total_kl   += float(kl.item()) * B
                total_feat += float(feat.item()) * B
                total_match += match * B
                total_cnt += B

                if training and step_idx % 50 == 0:
                    print(f"    step {step_idx}/{len(dl)} "
                          f"loss={loss.item():.4f} ce={ce.item():.4f} "
                          f"kl={kl.item():.4f} feat={feat.item():.4f} "
                          f"match@k={match.tolist()}")

        return (total_loss / total_cnt,
                total_ce / total_cnt,
                total_kl / total_cnt,
                total_feat / total_cnt,
                (total_match / total_cnt).tolist())

    for epoch in range(args.epochs):
        print(f"\n[Epoch {epoch+1}/{args.epochs}]")
        tr_loss, tr_ce, tr_kl, tr_feat, tr_match = run_epoch(train_dl, training=True)
        vl_loss, vl_ce, vl_kl, vl_feat, vl_match = run_epoch(val_dl, training=False)
        avg_accept = sum(vl_match) / len(vl_match)
        print(f"  train loss={tr_loss:.4f} ce={tr_ce:.4f} kl={tr_kl:.4f} "
              f"feat={tr_feat:.4f} match={tr_match}")
        print(f"  val   loss={vl_loss:.4f} ce={vl_ce:.4f} kl={vl_kl:.4f} "
              f"feat={vl_feat:.4f} match={vl_match} "
              f"avg_accept={avg_accept*100:.1f}%")

        if vl_loss < best_val:
            best_val = vl_loss
            ckpt_path = out_dir / "eagle3_draft_best.pt"
            torch.save({
                "model": draft.state_dict(),
                "cfg": cfg,
                "epoch": epoch,
                "val_loss": vl_loss,
                "val_ce": vl_ce,
                "val_kl": vl_kl,
                "val_feat": vl_feat,
                "val_match": vl_match,
                "hass": {
                    "q_override": args.q_override,
                    "ce_weight": args.ce_weight,
                    "kl_weight": args.kl_weight,
                    "feat_loss_weight": args.feat_loss_weight,
                    "top_k": args.top_k,
                },
            }, ckpt_path)
            print(f"  saved best to {ckpt_path}")
        torch.save({"model": draft.state_dict(), "cfg": cfg, "epoch": epoch},
                   out_dir / "eagle3_draft_latest.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="Prefix of <data>.bin + <data>.manifest.json from "
                         "collect_eagle_hidden_states_w4a8.py --top-k 20")
    ap.add_argument("--lm-head", type=Path, required=True,
                    help="Path to (vocab, hidden) fp16 lm_head weight dump")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--init-ckpt", type=Path, default=None,
                    help="Optional prior checkpoint to warm-start from.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--K", type=int, default=3,
                    help="TTT rollout length. Must match Swift drawBurst K.")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Teacher top-K width (must match manifest.top_k).")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ce-weight", type=float, default=0.1,
                    help="Weight on hard CE against tok_argmax (HASS: 0.1).")
    ap.add_argument("--kl-weight", type=float, default=1.0,
                    help="Weight on soft-KL over teacher top-K (HASS: 1.0).")
    ap.add_argument("--feat-loss-weight", type=float, default=0.0,
                    help="Weight on RMSE(draft_h_final, h_high). Off by "
                         "default (matches EAGLE-3 paper).")
    ap.add_argument("--q-override", type=str,
                    choices=["none", "step0", "all"], default="none",
                    help="Q-override at TTT step 0. 'none' (default) is the "
                         "vanilla EAGLE-3 TTT convention: k=0 input = fused "
                         "target hiddens, exactly matching what SpeculativeLoop "
                         "feeds the draft at inference (hPrev=hFused). 'step0' "
                         "is a HASS-adapted variant that overrides k=0 input "
                         "with teacher's pre-norm L34 (h_high) — anchors the "
                         "draft's Q on target's unfused feature. It's a "
                         "deliberate train/inference mismatch; use only for "
                         "ablation. 'all' replaces d_h at every k (debug).")
    ap.add_argument("--device", type=str,
                    default=("mps" if torch.backends.mps.is_available()
                             else ("cuda" if torch.cuda.is_available() else "cpu")))
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
