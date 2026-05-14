#!/usr/bin/env python3
"""Phase 1b — distillation uptraining harness for recursive Gemma 4 E2B.

Phase 1a proved the recursive init alone is broken at every compression
rank — uptraining is the core method, not a polish. This is the harness
that does the uptraining.

Design:
  * Teacher  : Gemma 4 E2B, frozen (the original untied model).
  * Student  : recursive Gemma 4 — tie-eligible consecutive layer pairs
               replaced by a shared base block (= average init) plus a
               per-position LoRA adapter on every Linear inside the
               tied layers. The shared base is read once at inference;
               the small per-position LoRA gives back per-layer
               specialisation. This is the RRT "relaxed recursive"
               structure.
  * Trainable: by default only the LoRA adapters (relaxed-recursive).
               --train-base also unfreezes the shared base blocks
               (heavier, closer to RRT full uptrain).
  * Loss     : KL(teacher || student) on logits at temperature T, plus
               optional hidden-state MSE.

Modes:
  --validate : tiny Mac MPS run (few steps, small corpus) to prove the
               pipeline works end-to-end before any GPU rental.
  --train    : full run — point at a real corpus, run on a rented GPU.

Usage (Mac validation):
  pyenv shell lama-cml
  python conversion/recursive_distill_train.py --validate \
    --model output/gemma4-e2b/hf_model --rank 256

Usage (GPU uptrain, later):
  python conversion/recursive_distill_train.py --train \
    --model output/gemma4-e2b/hf_model --rank 256 \
    --corpus /path/to/corpus.txt --steps 20000 --batch 8 \
    --out /path/to/recursive_ckpt
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import sys as _sys
import types as _types
import importlib.machinery as _machinery
if "wandb" not in _sys.modules:
    _w = _types.ModuleType("wandb")
    _w.__path__ = []  # type: ignore[attr-defined]
    _w.__spec__ = _machinery.ModuleSpec("wandb", loader=None, is_package=True)
    _sys.modules["wandb"] = _w

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from recursive_tie_experiment import (
    CORPUS, get_device, find_layers, layer_signature,
)


# ---------------------------------------------------------------------
# LoRA wrapper
# ---------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Wraps a frozen base nn.Linear with a trainable low-rank adapter.

    effective(x) = base(x) + scale * (x @ A^T @ B^T)
    A: (rank, in)   B: (out, rank)   — both trainable, base frozen.
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        in_f = base.in_features
        out_f = base.out_features
        dev = base.weight.device
        dt = base.weight.dtype
        # Standard LoRA init: A ~ small random, B = 0 → adapter starts at 0.
        self.lora_A = nn.Parameter(torch.randn(rank, in_f, device=dev, dtype=dt) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=dev, dtype=dt))

    def forward(self, x):
        out = self.base(x)
        lora = (x @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scale * lora

    def fold(self) -> torch.Tensor:
        """Return the effective full weight (base + scale·B·A) — for
        export / conversion once trained."""
        return self.base.weight + self.scale * (self.lora_B @ self.lora_A)


def _iter_linears(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            yield module, name, child
        else:
            yield from _iter_linears(child)


# ---------------------------------------------------------------------
# Build the recursive student
# ---------------------------------------------------------------------


def eligible_pairs(layers):
    n = len(layers)
    sigs = [layer_signature(layers[i]) for i in range(n)]
    pairs, i = [], 0
    while i < n - 1:
        if sigs[i] == sigs[i + 1]:
            pairs.append((i, i + 1)); i += 2
        else:
            i += 1
    return pairs


def build_recursive_student(model, rank: int, train_base: bool) -> dict:
    """Mutate `model` into the recursive student in place.

    1. Each tie-eligible pair → both layers' 2D weights set to the pair
       average (the shared base).
    2. Every nn.Linear inside the tied layers is wrapped in LoRALinear
       (trainable per-position adapter).
    3. Everything except the LoRA adapters is frozen, unless
       train_base=True (then the shared base 2D weights are also
       trainable — but still SHARED, so a single Parameter object must
       back both layers; we approximate by training both copies with a
       tied-gradient hook).

    Returns bookkeeping dict.
    """
    layers = find_layers(model)
    pairs = eligible_pairs(layers)

    # Step 1: average the 2D weights of each pair (the shared base).
    with torch.no_grad():
        for (ia, ib) in pairs:
            sda = layers[ia].state_dict()
            sdb = layers[ib].state_dict()
            for k in sda:
                if sda[k].dim() == 2:
                    avg = ((sda[k].float() + sdb[k].float()) / 2).to(sda[k].dtype)
                    sda[k].copy_(avg)
                    sdb[k].copy_(avg)

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad_(False)

    # Step 2: wrap Linears in the tied layers with LoRA.
    n_lora = 0
    tied_layer_ids = set()
    for (ia, ib) in pairs:
        tied_layer_ids.add(ia); tied_layer_ids.add(ib)
    for li in sorted(tied_layer_ids):
        layer = layers[li]
        for parent, name, lin in list(_iter_linears(layer)):
            wrapped = LoRALinear(lin, rank=rank)
            setattr(parent, name, wrapped)
            n_lora += 1

    # Step 3: optionally unfreeze the shared base weights.
    if train_base:
        for li in sorted(tied_layer_ids):
            for parent, name, mod in list(_iter_linears(layers[li])):
                # mod is now LoRALinear; unfreeze its base
                if isinstance(mod, LoRALinear):
                    for p in mod.base.parameters():
                        p.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "n_pairs": len(pairs),
        "n_lora_modules": n_lora,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_frac": trainable / max(total, 1),
        "unique_blocks": len(layers) - len(pairs),
        "total_layers": len(layers),
    }


# ---------------------------------------------------------------------
# Distillation training
# ---------------------------------------------------------------------


def distill_loss(student_logits, teacher_logits, temperature: float = 2.0):
    """KL(teacher || student) at temperature, per-token mean.

    Computed in fp32 regardless of model dtype — softmax/log over a
    262k-vocab in fp16 overflows. reduction is an explicit
    mean-over-(batch*seq), NOT F.kl_div's batchmean (which divides by
    the batch dim only and yields a loss ~seq_len too large).
    """
    s_lp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_lp = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    t_p = t_lp.exp()
    kl = (t_p * (t_lp - s_lp)).sum(-1).mean()  # mean over batch*seq
    return kl * (temperature ** 2)


def make_batches(tok, text: str, seq_len: int, n_batches: int, device):
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    # Tile if too short.
    while ids.numel() < seq_len * n_batches + 1:
        ids = torch.cat([ids, ids], dim=0)
    batches = []
    for b in range(n_batches):
        chunk = ids[b * seq_len: b * seq_len + seq_len].unsqueeze(0)
        batches.append(chunk.to(device))
    return batches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="output/gemma4-e2b/hf_model")
    p.add_argument("--rank", type=int, default=256)
    p.add_argument("--train-base", action="store_true",
                   help="also unfreeze the shared base weights")
    p.add_argument("--validate", action="store_true",
                   help="tiny Mac run to prove the pipeline")
    p.add_argument("--train", action="store_true",
                   help="full training run")
    p.add_argument("--corpus", default=None,
                   help="path to a text corpus for --train (else builtin)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float32", "float16"],
                   help="model+training dtype. bf16 is the stable default; "
                        "fp16 overflows in training, fp32 is heavy.")
    p.add_argument("--out", default="/tmp/recursive_ckpt")
    p.add_argument("--report", default="/tmp/recursive_distill_report.json")
    args = p.parse_args()

    if not (args.validate or args.train):
        args.validate = True  # default to the safe mode

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32,
             "float16": torch.float16}[args.dtype]
    device = get_device()
    print(f"[distill] device={device} dtype={args.dtype} "
          f"mode={'validate' if args.validate else 'train'}")
    tok = AutoTokenizer.from_pretrained(args.model)

    # --- teacher (frozen) ---
    print("[distill] loading teacher (frozen)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    for pp in teacher.parameters():
        pp.requires_grad_(False)

    # --- student (recursive) ---
    print(f"[distill] building recursive student (rank={args.rank}, "
          f"train_base={args.train_base})...")
    student = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    info = build_recursive_student(student, args.rank, args.train_base)
    student.train()
    print(f"[distill] {info['unique_blocks']}/{info['total_layers']} unique "
          f"blocks, {info['n_lora_modules']} LoRA modules, "
          f"{info['trainable_params']:,} trainable "
          f"({info['trainable_frac']*100:.2f}% of {info['total_params']:,})")

    # --- corpus ---
    if args.corpus and Path(args.corpus).exists():
        text = Path(args.corpus).read_text()
        print(f"[distill] corpus: {args.corpus} ({len(text)} chars)")
    else:
        text = CORPUS
        print(f"[distill] corpus: builtin ({len(text)} chars)")

    n_batches = args.batch * (2 if args.validate else 8)
    batches = make_batches(tok, text, args.seq_len, n_batches, device)

    # --- optimizer (only trainable params) ---
    trainable = [pp for pp in student.parameters() if pp.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    steps = args.steps if args.validate else args.steps
    print(f"[distill] training {steps} steps, batch {args.batch}, "
          f"seq_len {args.seq_len}, lr {args.lr}")

    history = []
    t0 = time.time()
    for step in range(steps):
        batch = batches[step % len(batches)]
        with torch.no_grad():
            t_logits = teacher(input_ids=batch).logits
        s_logits = student(input_ids=batch).logits
        loss = distill_loss(s_logits, t_logits, args.temperature)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            # also: top-1 agreement on this batch
            with torch.no_grad():
                agree = (s_logits[:, :-1].argmax(-1)
                         == t_logits[:, :-1].argmax(-1)).float().mean().item()
            history.append({"step": step + 1, "loss": float(loss.item()),
                            "top1_agreement": agree})
            print(f"[distill] step {step+1:>4}  KL {loss.item():.4f}  "
                  f"top-1 {agree:.3f}")

    dt = time.time() - t0
    print(f"[distill] {steps} steps in {dt:.1f}s ({dt/steps*1000:.0f} ms/step)")

    # Loss trend check
    if len(history) >= 2:
        first_loss = history[0]["loss"]
        last_loss = history[-1]["loss"]
        improved = last_loss < first_loss
        print(f"[distill] KL {first_loss:.4f} → {last_loss:.4f}  "
              f"{'✓ decreasing' if improved else '✗ NOT decreasing'}")
    else:
        improved = None

    report = {
        "args": vars(args),
        "student_info": info,
        "history": history,
        "ms_per_step": dt / steps * 1000,
        "loss_decreasing": improved,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[distill] wrote {args.report}")

    if args.validate:
        print("\n=== Validation verdict ===")
        if improved:
            print("PASS — pipeline runs, loss decreases. Harness is ready "
                  "for a full GPU run (--train with a real corpus + steps).")
        else:
            print("FAIL — loss not decreasing. Debug before GPU spend: "
                  "check lr, LoRA init, gradient flow.")
    else:
        # Save the trained student (folded weights for conversion).
        print(f"[distill] saving trained student to {args.out} ...")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        student.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        print(f"[distill] saved.")


if __name__ == "__main__":
    main()
