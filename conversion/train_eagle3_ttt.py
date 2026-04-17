#!/usr/bin/env python3
"""EAGLE-3 draft training with full Training-Time Test (TTT).

Training objective (per sample, K=3 steps):

  Step 0 — teacher-forced:
    h_prev ← draft.fuse_target(target_fusion_hiddens_L8/L17/L34)
    e_in   ← embed(target_token_at_t+1)
    d_h, logits ← draft.step(h_prev, e_in, is_sequence=True)
    CE loss vs target's next-token argmax at t+1.

  Step k (k = 1 .. K-1) — draft-on-draft autoregressive:
    pred_tok ← argmax(logits_{k-1})
    e_k      ← embed(pred_tok) * embed_scale
    d_h, logits ← draft.step(d_h, e_k, is_sequence=True)
    CE loss vs target's next-token argmax at t+1+k.

  Total loss = Σ_k TTT_WEIGHTS[k] * CE_k   +   FEATURE_LOSS_W * MSE(d_h_step0, h_tgt)

Why this matters: a draft trained only on step 0 (the standalone simplified
script) achieves trivial ~100% teacher-forced accuracy but gives ~0% accept
rate at steps 1/2 during inference because its distributional input there
(its own prior hidden state) was never seen in training. Full TTT rolls the
draft forward K times per sample during training, matching the inference
unroll exactly.

Data path: reuses the streaming collector's memmap-v1 manifest, augmented
with per-sequence boundaries by augment_seq_metadata.py. Each training
step reads one full sequence's worth of pairs from the memmap and applies
the K-step TTT loop with is_sequence=True so positions are parallelized
under the decoder's causal mask.

Usage (Colab, A100):
    # Prereq once: add seq boundaries to manifest (takes a few minutes,
    # does not touch the .data/ files):
    python augment_seq_metadata.py \
        --data /content/training_data.pt \
        --corpus /content/drive/MyDrive/eagle3_retrain_20260417/eagle_corpus.jsonl

    # Then train:
    python train_eagle3_ttt.py \
        --data /content/training_data.pt \
        --save-dir /content/drive/MyDrive/eagle3_retrain_20260417_ttt \
        --epochs 2 --preload

Outputs: eagle3_draft_best.pt, eagle3_draft_final.pt, eagle3_config.json,
eagle3_eval.json (acc[0..K-1] and expected-length metric), training log.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Reuse architecture + helpers from the standalone simplified trainer.
# This keeps EAGLE3Draft / RMSNorm / RoPE code identical across the two
# trainers so checkpoints are interchangeable with build_eagle3.py.
from train_eagle3_standalone import (
    EAGLE3Draft,
    build_rope_cache,
    HIDDEN,
    HEAD_DIM,
    VOCAB,
    ROPE_THETA,
    EMBED_SCALE,
    FUSION_LAYERS,
    RMS_EPS,
    TTT_K,
    TTT_WEIGHTS,
    FEATURE_LOSS_W,
    NUM_HEADS,
    NUM_KV,
    FFN_DIM,
    SEED,
)


# ─────────────────────────────────────────────────────────────────────────
# Data access — memmap-v1 with sequence boundaries
# ─────────────────────────────────────────────────────────────────────────

# Keys the TTT trainer uses. We intentionally skip h_in (unused), and we
# keep h_tgt for the step-0 feature-loss term.
USED_KEYS = ["h_tgt", "e_in", "tok_tgt", "fusion_L8", "fusion_L17", "fusion_L34"]


def _resolve_data_dir(args_data, manifest):
    """Sibling-first: prefer <args.data>.data if it has the tensor files."""
    sibling = args_data[:-3] + ".data" if args_data.endswith(".pt") else args_data + ".data"
    manifest_dir = manifest["data_dir"]

    def _ok(d):
        return os.path.isdir(d) and os.path.isfile(os.path.join(d, "h_tgt.dat"))

    if _ok(sibling):
        if os.path.abspath(sibling) != os.path.abspath(manifest_dir):
            print(f"  Using sibling data dir: {sibling}")
        return sibling
    if _ok(manifest_dir):
        return manifest_dir
    raise SystemExit(
        f"\nERROR: memmap data directory not found at either location:\n"
        f"  sibling : {sibling}\n"
        f"  manifest: {manifest_dir}\n"
        f"Re-collect, or place the .data dir next to the .pt."
    )


def _open_memmap_tensors(data_dir, manifest, preload):
    """Open all USED_KEYS as torch tensors. Preload into RAM if requested."""
    shapes = manifest["shapes"]
    dtypes = manifest["dtypes"]
    tensors = {}

    if preload:
        total_gb = sum(np.prod(shapes[k]) * np.dtype(dtypes[k]).itemsize
                       for k in USED_KEYS) / 1e9
        print(f"  Preloading {len(USED_KEYS)} tensors → {total_gb:.1f} GB into RAM...")

    t0 = time.time()
    for k in USED_KEYS:
        path = os.path.join(data_dir, f"{k}.dat")
        arr = np.memmap(path, dtype=np.dtype(dtypes[k]), mode="r", shape=shapes[k])
        if preload:
            print(f"    {k} ({np.prod(shapes[k]) * np.dtype(dtypes[k]).itemsize / 1e9:.1f} GB)...",
                  flush=True)
            arr = np.array(arr, copy=True)  # sequential read → RAM
        tensors[k] = torch.from_numpy(arr)

    if preload:
        print(f"  Preload done in {time.time() - t0:.0f}s")
    return tensors


# ─────────────────────────────────────────────────────────────────────────
# TTT training step (one sequence)
# ─────────────────────────────────────────────────────────────────────────

def ttt_step(
    draft,
    data,
    seq_start,
    seq_end,
    cos,
    sin,
    lm_head_weight_gpu,
    device,
    train=True,
):
    """Run K-step TTT on a single sequence's pairs.

    Returns (total_loss_tensor, per_step_correct_counts, n_valid_positions)
    OR None if the sequence is too short (< TTT_K + 1 pairs).
    """
    Ti = seq_end - seq_start
    if Ti < TTT_K + 1:
        return None

    # Fetch sequence slice from memmap (or RAM if preloaded)
    h_tgt      = data["h_tgt"][seq_start:seq_end].to(device)        # (Ti, H)
    e_in       = data["e_in"][seq_start:seq_end].to(device)          # (Ti, H)
    tok_tgt    = data["tok_tgt"][seq_start:seq_end].to(device).long()# (Ti,)
    fusion_L8  = data["fusion_L8"][seq_start:seq_end].to(device)
    fusion_L17 = data["fusion_L17"][seq_start:seq_end].to(device)
    fusion_L34 = data["fusion_L34"][seq_start:seq_end].to(device)

    # Valid positions j where we can train K steps: tok_tgt[j], tok_tgt[j+1],
    # ..., tok_tgt[j+K-1] must all exist, so j in [0, Ti - K).
    valid = Ti - TTT_K
    if valid <= 0:
        return None

    # Step 0 — teacher-forced from target fusion hiddens.
    # Shapes become (1, valid, H).
    h_prev = draft.fuse_target([
        fusion_L8[:valid],
        fusion_L17[:valid],
        fusion_L34[:valid],
    ]).unsqueeze(0)
    e_next = e_in[:valid].unsqueeze(0)
    d_h, logits = draft.step(h_prev, e_next, cos, sin, is_sequence=True)

    # Label for step 0: tok_tgt at the current pair index represents target's
    # argmax after processing up to the embedded token — i.e. the token draft
    # is trying to predict. See collector: tok_tgt[j] = argmax(LMhead(h[j+1])).
    label0 = tok_tgt[:valid].unsqueeze(0)  # (1, valid)
    ce0 = F.cross_entropy(logits.view(-1, VOCAB), label0.view(-1))

    # Feature loss on step 0 only (MSE between draft hidden and target hidden
    # at the same predicted position).
    loss = TTT_WEIGHTS[0] * ce0
    if FEATURE_LOSS_W > 0:
        mse = F.mse_loss(d_h.float(), h_tgt[:valid].unsqueeze(0).float())
        loss = loss + FEATURE_LOSS_W * mse

    correct = [0] * TTT_K
    correct[0] = (logits.argmax(-1) == label0).sum().item()

    # Steps 1..K-1 — draft-on-draft autoregressive
    for k in range(1, TTT_K):
        pred_tok = logits.argmax(-1)  # (1, valid) long
        # Tied embedding: lm_head_weight[tok] == embed_tokens(tok), modulo
        # the embed_scale. Multiplier applied per Gemma convention.
        e_k = lm_head_weight_gpu[pred_tok].to(d_h.dtype) * EMBED_SCALE

        d_h, logits = draft.step(d_h, e_k, cos, sin, is_sequence=True)
        label_k = tok_tgt[k:k + valid].unsqueeze(0)  # (1, valid)
        ce_k = F.cross_entropy(logits.view(-1, VOCAB), label_k.view(-1))
        loss = loss + TTT_WEIGHTS[k] * ce_k
        correct[k] = (logits.argmax(-1) == label_k).sum().item()

    return loss, correct, valid


@torch.no_grad()
def evaluate(draft, data, seq_starts, cos, sin, lm_head_weight_gpu, device, n=512):
    """Compute per-step acc[0..K-1] on n validation sequences."""
    draft.eval()
    total_valid = 0
    total_correct = [0] * TTT_K
    per_seq_pairs = 0

    # Sample last `n` sequences as a simple held-out set (seq_starts at the
    # tail of a shuffled corpus is random enough).
    num_seqs = seq_starts.shape[0] - 1
    eval_indices = list(range(max(0, num_seqs - n), num_seqs))

    for idx in eval_indices:
        start = int(seq_starts[idx])
        end = int(seq_starts[idx + 1])
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = ttt_step(draft, data, start, end, cos, sin,
                           lm_head_weight_gpu, device, train=False)
        if out is None:
            continue
        _, correct, valid = out
        total_valid += valid
        for k in range(TTT_K):
            total_correct[k] += correct[k]
        per_seq_pairs += end - start

    if total_valid == 0:
        return [0.0] * TTT_K, 0.0

    acc_per_step = [c / total_valid for c in total_correct]
    # Expected length = 1 + Σ_k Π_{j≤k} acc[j]. Rough ANE-independent proxy;
    # real accept rate depends on whether target's argmax matches draft's
    # under our CoreML chunks — but this number is how EAGLE-3 authors report.
    expL = 1.0
    p = 1.0
    for k in range(TTT_K):
        p *= acc_per_step[k]
        expL += p
    return acc_per_step, expL


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to memmap-v1 manifest with seq_starts "
                             "(run augment_seq_metadata.py first).")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=2000,
                        help="Training steps between eval passes.")
    parser.add_argument("--save-every", type=int, default=4000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--preload", action="store_true",
                        help="Load memmap tensors fully into CPU RAM. Required "
                             "for reasonable speed — sequence-level fetch on "
                             "Drive / SSD random-reads blocks the GPU.")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Optional: start from an existing draft checkpoint "
                             "(e.g. step-0-only best.pt). Can give ~1.5× faster "
                             "convergence.")
    parser.add_argument("--max-seqs", type=int, default=None,
                        help="Optional cap on training sequences per epoch "
                             "(debug; default uses all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a handful of steps on 10 sequences and exit. "
                             "Used for Mac-side syntax + loss-finite checks.")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # ─── Load manifest + open data ───────────────────────────────────────
    print(f"Loading manifest: {args.data}")
    raw = torch.load(args.data, map_location="cpu")
    if raw.get("format") != "memmap-v1":
        raise SystemExit("ERROR: manifest is not memmap-v1.")
    if "seq_starts" not in raw:
        raise SystemExit("ERROR: manifest has no seq_starts. Run "
                         "augment_seq_metadata.py first.")

    seq_starts = raw["seq_starts"]  # int64 tensor, shape (num_sequences + 1,)
    num_sequences = int(raw["num_sequences"])
    total_pairs = int(raw["total_pairs"])
    lm_head_weight = raw["lm_head_weight"]  # (vocab, hidden) fp16
    hidden_size = int(raw["hidden_size"])
    assert hidden_size == HIDDEN, f"hidden_size mismatch ({hidden_size} vs {HIDDEN})"

    print(f"  num_sequences: {num_sequences:,}")
    print(f"  total_pairs  : {total_pairs:,}")
    print(f"  hidden_size  : {hidden_size}")

    data_dir = _resolve_data_dir(args.data, raw)
    print(f"  data_dir     : {data_dir}")

    data = _open_memmap_tensors(data_dir, raw, preload=args.preload)

    # ─── Build model ─────────────────────────────────────────────────────
    print("Building EAGLE3Draft...")
    draft = EAGLE3Draft(lm_head_weight.float()).to(device)
    if args.init_from:
        print(f"  warm-start from {args.init_from}")
        ckpt = torch.load(args.init_from, map_location="cpu")
        draft.load_state_dict(ckpt["model"], strict=False)
    n_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"  params: {n_params / 1e6:.2f}M")

    # GPU-resident lm_head_weight for fast TTT step-k embedding lookup.
    lm_head_weight_gpu = lm_head_weight.to(device).to(torch.float16)

    COS, SIN = build_rope_cache(max(1024, 512), HEAD_DIM, ROPE_THETA, device)

    # ─── Optimizer ───────────────────────────────────────────────────────
    opt = AdamW(draft.parameters(), lr=args.lr, weight_decay=0.01,
                betas=(0.9, 0.95))

    # One training "step" = one sequence forward+backward (matches notebook's
    # MICRO_BATCH=1). Optimizer step happens every GRAD_ACCUM sequences.
    max_seqs_per_epoch = num_sequences if args.max_seqs is None else min(num_sequences, args.max_seqs)
    total_opt_steps = (max_seqs_per_epoch // args.grad_accum) * args.epochs

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, total_opt_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    # ─── Logging / checkpoint helpers ────────────────────────────────────
    log_path = os.path.join(args.save_dir, "eagle3_ttt_training.log")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    def save_ckpt(tag, acc=None, expL=None):
        p = os.path.join(args.save_dir, f"eagle3_draft_{tag}.pt")
        state = {
            "model": draft.state_dict(),
            "meta": {
                "hidden": HIDDEN, "num_heads": NUM_HEADS, "num_kv_heads": NUM_KV,
                "head_dim": HEAD_DIM, "ffn": FFN_DIM, "vocab": VOCAB,
                "rms_eps": RMS_EPS, "rope_theta": ROPE_THETA,
                "embed_scale": EMBED_SCALE, "fusion_layers": FUSION_LAYERS,
                "ttt_k": TTT_K, "model_id": "custom_gemma4_model",
                "acc": acc, "expL": expL,
                "training": "ttt",
            },
        }
        torch.save(state, p)
        sz = os.path.getsize(p) / 1e6
        log(f"  saved {p} ({sz:.1f} MB)")

    # Save config.json up-front
    with open(os.path.join(args.save_dir, "eagle3_config.json"), "w") as f:
        json.dump({
            "architecture": "eagle3_draft",
            "hidden": HIDDEN, "num_heads": NUM_HEADS, "num_kv_heads": NUM_KV,
            "head_dim": HEAD_DIM, "ffn": FFN_DIM, "vocab": VOCAB,
            "rms_eps": RMS_EPS, "rope_theta": ROPE_THETA,
            "embed_scale": EMBED_SCALE, "fusion_layers": FUSION_LAYERS,
            "ttt_k": TTT_K, "ttt_weights": TTT_WEIGHTS,
            "feature_loss_w": FEATURE_LOSS_W,
            "model_id": "custom_gemma4_model",
            "training": "ttt",
        }, f, indent=2)

    # ─── Training loop ───────────────────────────────────────────────────
    log(f"\nTTT training — K={TTT_K}, weights={TTT_WEIGHTS}, feature_w={FEATURE_LOSS_W}")
    log(f"  epochs={args.epochs}, grad_accum={args.grad_accum}, lr={args.lr}")
    log(f"  sequences/epoch={max_seqs_per_epoch}, opt_steps={total_opt_steps}")

    # Reserve last 512 sequences as validation.
    val_count = min(512, num_sequences // 10)
    train_seqs = list(range(num_sequences - val_count))
    print(f"  train sequences: {len(train_seqs):,}")
    print(f"  val sequences:   {val_count:,}")

    step = 0
    best_expL = 0.0
    t0 = time.time()

    dry_limit = 10 if args.dry_run else None

    for epoch in range(args.epochs):
        log(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        random.shuffle(train_seqs)

        running_ce = [0.0] * TTT_K
        running_correct = [0] * TTT_K
        running_valid = 0
        n_seqs_in_epoch = 0

        iterator = train_seqs[:max_seqs_per_epoch]
        if dry_limit is not None:
            iterator = iterator[:dry_limit]

        pbar = tqdm(iterator, desc=f"Epoch {epoch + 1}")
        for seq_idx in pbar:
            draft.train()
            start = int(seq_starts[seq_idx])
            end = int(seq_starts[seq_idx + 1])

            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = ttt_step(draft, data, start, end, COS, SIN,
                               lm_head_weight_gpu, device, train=True)
            if out is None:
                continue
            loss, correct, valid = out

            if not torch.isfinite(loss):
                log(f"  WARN: non-finite loss at seq {seq_idx} (skipping step)")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss / args.grad_accum).backward()
            running_valid += valid
            for k in range(TTT_K):
                running_correct[k] += correct[k]
            n_seqs_in_epoch += 1

            if (n_seqs_in_epoch % args.grad_accum) == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if running_valid > 0:
                    pbar.set_postfix(
                        step=step,
                        acc0=f"{running_correct[0] / running_valid * 100:.1f}%",
                        acc1=f"{running_correct[1] / running_valid * 100:.1f}%",
                        acc2=f"{running_correct[2] / running_valid * 100:.1f}%",
                    )

                if step % args.eval_every == 0:
                    acc_k, expL = evaluate(draft, data, seq_starts, COS, SIN,
                                           lm_head_weight_gpu, device,
                                           n=val_count)
                    msg = (f"  step {step}: "
                           f"train_acc[0..K-1]=" +
                           "/".join(f"{running_correct[k]/max(1,running_valid)*100:.1f}"
                                    for k in range(TTT_K)) +
                           f"  val_acc[0..K-1]=" +
                           "/".join(f"{a*100:.1f}" for a in acc_k) +
                           f"  val_expL={expL:.3f}")
                    log(msg)
                    if expL > best_expL:
                        best_expL = expL
                        save_ckpt("best", acc=acc_k, expL=expL)
                        log(f"  ** new best: expL={best_expL:.3f}")

                if step % args.save_every == 0:
                    save_ckpt(f"step{step}", acc=None, expL=None)

        # End-of-epoch evaluation + checkpoint
        acc_k, expL = evaluate(draft, data, seq_starts, COS, SIN,
                               lm_head_weight_gpu, device, n=val_count)
        log(f"Epoch {epoch + 1} end: "
            f"val_acc[0..K-1]=" +
            "/".join(f"{a*100:.1f}" for a in acc_k) +
            f"  val_expL={expL:.3f}  best_expL={best_expL:.3f}")
        if expL > best_expL:
            best_expL = expL
            save_ckpt("best", acc=acc_k, expL=expL)

    save_ckpt("final", acc=acc_k, expL=expL)
    elapsed = (time.time() - t0) / 60
    log(f"\nTraining complete in {elapsed:.1f} min. Best expL: {best_expL:.3f}")
    log(f"Target: expL >= 2.0 (≥ notebook baseline). acc[0] >= 0.7 is a good sign.")

    # Write eval summary JSON
    with open(os.path.join(args.save_dir, "eagle3_eval.json"), "w") as f:
        json.dump({
            "final_val_acc": acc_k,
            "final_val_expL": expL,
            "best_val_expL": best_expL,
            "training": "ttt",
            "ttt_k": TTT_K,
        }, f, indent=2)

    log_f.close()

    print(f"\nNext steps:")
    print(f"  1. Download {args.save_dir}/eagle3_draft_best.pt + eagle3_config.json")
    print(f"  2. Run: python build_eagle3.py --ckpt eagle3_draft_best.pt --output eagle3_draft.mlpackage")
    print(f"  3. Deploy to iPhone and measure on-device acceptance rate")


if __name__ == "__main__":
    raise SystemExit(main())
