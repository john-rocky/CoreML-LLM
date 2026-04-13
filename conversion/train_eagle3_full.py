#!/usr/bin/env python3
"""EAGLE-3 draft training — notebook-exact, custom Gemma4Model target.

This is a 1:1 translation of train_eagle3_draft.ipynb with ONE change:
the target model is our custom Gemma4Model (Conv2d, same forward path as
CoreML chunks on device) instead of HF Gemma4ForConditionalGeneration.

This fixes Blocker 1: HF target hidden states differ from custom target
(L34 norm 4.4× gap), causing ~0% on-device acceptance.

All architecture, training loop, TTT, eval, warm-start, and checkpoint
logic are identical to the notebook.

Usage (Colab, A100):
    !git clone -q -b claude/eagle3-full-retrain https://github.com/john-rocky/CoreML-LLM.git
    %cd CoreML-LLM/conversion
    !pip install -q safetensors transformers accelerate
    !python train_eagle3_full.py \
        --corpus /content/drive/MyDrive/eagle_corpus.jsonl \
        --save-dir /content/drive/MyDrive/eagle_draft_new \
        --num-samples 2000 --seq-len 256 --epochs 2

Runtime: ~1-2h on A100 (target forward dominates).
"""
from __future__ import annotations

import argparse
import gc
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

# ── Architecture constants (read from model after load) ───────────────────
# These are set in main() after loading the model config.
HIDDEN = NUM_HEADS = NUM_KV = HEAD_DIM = FFN_DIM = VOCAB = 0
EMBED_SCALE = RMS_EPS = ROPE_THETA = 0.0
FUSION_LAYERS = [8, 17, 34]

# ── Training config ───────────────────────────────────────────────────────
TTT_K = 3
TTT_WEIGHTS = [1.0, 0.7, 0.5]
FEATURE_LOSS_W = 0.1
INIT_FROM_TARGET_LAYER = 13


# ══════════════════════════════════════════════════════════════════════════
# Model components — EXACT copy from notebook Cell 8
# ══════════════════════════════════════════════════════════════════════════

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
        q = self.q_norm(q); k = self.k_norm(k); v = self.v_norm(v)
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


# ══════════════════════════════════════════════════════════════════════════
# Custom target forward — uses our Gemma4Model (Conv2d) via forward_batch
# This is the ONLY change from the notebook.
# ══════════════════════════════════════════════════════════════════════════

from models.gemma4 import Gemma4Model
from models.gemma4_swa_chunks import _run_layer_swa, v_norm
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, ane_softmax


def custom_target_forward(model, tokenizer, text, seq_len, device):
    """Run custom Gemma4Model on a text, return hidden states matching notebook format.

    Returns same dict as notebook's target_forward():
        layer_h: [3 × (T, H)] — fusion layer hiddens (L8, L17, L34)
        last_h:  (T, H) — final hidden (post-norm)
        embeds:  (T, H) — token embeddings × embed_scale
        tok_tgt: (T,) — argmax target tokens
    """
    ids = tokenizer.encode(text, return_tensors="pt",
                           truncation=True, max_length=seq_len).to(device)
    if ids.shape[1] < 32:
        return None

    config = model.config
    seq_len_actual = ids.shape[1]
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    nlayers = config.num_hidden_layers
    max_hd = config.global_head_dim
    pld = config.hidden_size_per_layer_input

    # Embedding
    embed_scale = config.hidden_size ** 0.5
    hidden = model.embed_tokens(ids).to(MODEL_DTYPE) * embed_scale

    # Per-layer embedding (matching forward_batch)
    pl_raw = model.embed_tokens_per_layer(ids).to(MODEL_DTYPE)
    pl_raw = pl_raw * (pld ** 0.5)
    h_conv = hidden.permute(0, 2, 1).unsqueeze(2)
    pl_proj = model.per_layer_model_projection(h_conv.to(MODEL_DTYPE))
    pl_proj = pl_proj.squeeze(2).permute(0, 2, 1)
    pl_proj = pl_proj * model.per_layer_model_projection_scale

    pl_combined_parts = []
    for i in range(nlayers):
        s = i * pld
        e = s + pld
        proj_normed = model.per_layer_projection_norm(pl_proj[:, :, s:e])
        combined = (proj_normed + pl_raw[:, :, s:e]) * model.per_layer_input_scale
        pl_combined_parts.append(combined)
    per_layer_combined = torch.cat(pl_combined_parts, dim=2)

    # RoPE
    cos_s = model.cos_sliding[:seq_len_actual].unsqueeze(0).unsqueeze(0)
    sin_s = model.sin_sliding[:seq_len_actual].unsqueeze(0).unsqueeze(0)
    cos_f = model.cos_full[:seq_len_actual].unsqueeze(0).unsqueeze(0)
    sin_f = model.sin_full[:seq_len_actual].unsqueeze(0).unsqueeze(0)

    # Causal mask
    causal_mask = torch.full((seq_len_actual, seq_len_actual), -65504.0,
                             dtype=MODEL_DTYPE, device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    # Run all layers, collect fusion hiddens
    kv13_k = kv13_v = kv14_k = kv14_v = None
    fusion_hiddens = {}

    for i in range(nlayers):
        is_full = config.is_full_attention(i)
        is_kv_shared = config.is_kv_shared(i)
        hd = config.get_head_dim(i)
        layer = model.layers[i]

        residual = hidden
        h = layer.input_layernorm(hidden)
        x = h.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)

        q = layer.self_attn["q_proj"](x)
        q = q.view(1, num_heads, hd, seq_len_actual).permute(0, 1, 3, 2).to(MODEL_DTYPE)
        q = layer.self_attn["q_norm"](q.reshape(1, num_heads * seq_len_actual, hd))
        q = q.view(1, num_heads, seq_len_actual, hd)

        if not is_kv_shared:
            k = layer.self_attn["k_proj"](x)
            k = k.view(1, num_kv_heads, hd, seq_len_actual).permute(0, 1, 3, 2).to(MODEL_DTYPE)
            k = layer.self_attn["k_norm"](k.reshape(1, num_kv_heads * seq_len_actual, hd))
            k = k.view(1, num_kv_heads, seq_len_actual, hd)
            v = layer.self_attn["v_proj"](x)
            v = v.view(1, num_kv_heads, hd, seq_len_actual).permute(0, 1, 3, 2).to(MODEL_DTYPE)
            v = v_norm(v)

            if is_full:
                q, k = apply_rotary_pos_emb(q, k, cos_f, sin_f)
            else:
                q, k = apply_rotary_pos_emb(q, k, cos_s, sin_s)

            if i == 13:
                kv13_k, kv13_v = k, v
            elif i == 14:
                kv14_k, kv14_v = k, v
        else:
            if is_full:
                q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f)
                k, v = kv14_k, kv14_v
            else:
                q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s)
                k, v = kv13_k, kv13_v

        K_expanded = k.repeat_interleave(n_rep, dim=1)
        V_expanded = v.repeat_interleave(n_rep, dim=1)

        attn_weights = torch.matmul(q, K_expanded.transpose(-1, -2))
        attn_weights = attn_weights + causal_mask
        attn_weights = ane_softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V_expanded)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, seq_len_actual, -1)
        attn_output = layer.self_attn["o_proj"](
            attn_output.permute(0, 2, 1).unsqueeze(2)
        ).squeeze(2).permute(0, 2, 1)
        attn_output = layer.post_attention_layernorm(attn_output)
        hidden = residual + attn_output

        residual = hidden
        h = layer.pre_feedforward_layernorm(hidden)
        x_mlp = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        gate = layer.mlp["gate_proj"](x_mlp)
        up = layer.mlp["up_proj"](x_mlp)
        gate = F.gelu(gate, approximate="tanh")
        mlp_out = layer.mlp["down_proj"](gate * up)
        hidden = mlp_out.squeeze(2).permute(0, 2, 1)
        hidden = layer.post_feedforward_layernorm(hidden)
        hidden = residual + hidden

        residual_pl = hidden
        s_idx = i * pld
        e_idx = s_idx + pld
        per_layer_slice = per_layer_combined[:, :, s_idx:e_idx]
        hs_conv = hidden.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        gated = layer.per_layer_input_gate(hs_conv)
        gated = F.gelu(gated, approximate="tanh")
        per_layer_slice_conv = per_layer_slice.permute(0, 2, 1).unsqueeze(2)
        gated = gated * per_layer_slice_conv
        gated = layer.per_layer_projection(gated)
        gated = gated.squeeze(2).permute(0, 2, 1)
        hidden = layer.post_per_layer_input_norm(gated)
        hidden = residual_pl + hidden
        hidden = hidden * layer.layer_scalar.to(MODEL_DTYPE)

        if i in FUSION_LAYERS:
            fusion_hiddens[i] = hidden[0].detach().half()

    hidden = model.norm(hidden)
    last_h = hidden[0].detach().half()

    # LM head (Conv2d → squeeze for logits)
    lm_w = model.lm_head.weight.data.squeeze(-1).squeeze(-1)
    logits = F.linear(last_h.float(), lm_w.float())
    tok_tgt = logits.argmax(dim=-1).detach()

    embeds = model.embed_tokens(ids)[0].detach().half() * embed_scale

    layer_h = [fusion_hiddens[l] for l in FUSION_LAYERS]

    return {
        "layer_h": layer_h,
        "last_h": last_h,
        "embeds": embeds,
        "tok_tgt": tok_tgt,
    }


# ══════════════════════════════════════════════════════════════════════════
# Warm-start — exact copy from notebook, adapted for custom model
# ══════════════════════════════════════════════════════════════════════════

def init_draft_from_custom_target(draft, target_model, layer_idx):
    """Warm-start draft from custom Gemma4Model layer weights."""
    if layer_idx is None:
        return
    try:
        tl = target_model.layers[layer_idx]
    except Exception as e:
        print(f"  warm-start skipped: {e}")
        return

    def get_w(mod, name):
        m = mod.get(name) if isinstance(mod, nn.ModuleDict) else getattr(mod, name, None)
        if m is not None and hasattr(m, "weight"):
            w = m.weight.data
            if w.dim() == 4:  # Conv2d → squeeze to 2D
                w = w.squeeze(-1).squeeze(-1)
            return w
        return None

    def copy_(dst, src):
        if src is None or dst.shape != src.shape:
            return False
        with torch.no_grad():
            dst.copy_(src.detach().float())
        return True

    attn = tl.self_attn
    mlp = tl.mlp
    ok = 0
    if copy_(draft.layer.q_proj.weight, get_w(attn, "q_proj")): ok += 1
    if copy_(draft.layer.k_proj.weight, get_w(attn, "k_proj")): ok += 1
    if copy_(draft.layer.v_proj.weight, get_w(attn, "v_proj")): ok += 1
    if copy_(draft.layer.o_proj.weight, get_w(attn, "o_proj")): ok += 1
    if copy_(draft.layer.q_norm.weight, get_w(attn, "q_norm")): ok += 1
    if copy_(draft.layer.k_norm.weight, get_w(attn, "k_norm")): ok += 1
    if copy_(draft.layer.gate_proj.weight, get_w(mlp, "gate_proj")): ok += 1
    if copy_(draft.layer.up_proj.weight, get_w(mlp, "up_proj")): ok += 1
    if copy_(draft.layer.down_proj.weight, get_w(mlp, "down_proj")): ok += 1
    print(f"  warm-start: copied {ok}/9 tensors from custom target layer {layer_idx}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    global HIDDEN, NUM_HEADS, NUM_KV, HEAD_DIM, FFN_DIM, VOCAB
    global EMBED_SCALE, RMS_EPS, ROPE_THETA

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--hf-dir", type=str, default=None,
                        help="HF model dir (auto-downloads if not set)")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load custom target model ──────────────────────────────────────
    hf_dir = args.hf_dir
    if hf_dir is None:
        from huggingface_hub import snapshot_download
        print("Downloading google/gemma-4-E2B-it...")
        hf_dir = snapshot_download("google/gemma-4-E2B-it")

    print(f"Loading custom Gemma4Model from {hf_dir}...")
    target = Gemma4Model.from_pretrained(hf_dir, context_length=args.seq_len)
    target = target.half().to(device).eval()
    for p in target.parameters():
        p.requires_grad = False

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)

    # Read config
    config = target.config
    HIDDEN = config.hidden_size
    NUM_HEADS = config.num_attention_heads
    NUM_KV = config.num_key_value_heads
    HEAD_DIM = config.head_dim
    FFN_DIM = config.intermediate_size
    VOCAB = config.vocab_size
    EMBED_SCALE = HIDDEN ** 0.5
    RMS_EPS = config.rms_norm_eps
    ROPE_THETA = getattr(config, "sliding_rope_theta", 10000.0)

    print(f"hidden={HIDDEN} heads={NUM_HEADS} kv={NUM_KV} head_dim={HEAD_DIM} ffn={FFN_DIM}")
    print(f"vocab={VOCAB} embed_scale={EMBED_SCALE:.3f} rope_theta={ROPE_THETA}")

    # ── Load corpus ───────────────────────────────────────────────────
    print(f"Loading corpus from {args.corpus}...")
    texts = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print(f"  {len(texts)} sequences")

    random.Random(args.seed).shuffle(texts)
    val_texts = texts[:args.val_samples]
    train_texts = texts[args.val_samples:args.val_samples + args.num_samples]
    print(f"  train: {len(train_texts)}  val: {len(val_texts)}")

    # Embedding function (for TTT steps 1-2)
    embed_fn = target.embed_tokens

    # ── Build draft ───────────────────────────────────────────────────
    lm_head_w = target.lm_head.weight.data.squeeze(-1).squeeze(-1).detach().clone()
    draft = EAGLE3Draft(lm_head_w.float()).to(device)
    init_draft_from_custom_target(draft, target, INIT_FROM_TARGET_LAYER)
    n_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"Draft params: {n_params / 1e6:.1f}M")

    COS, SIN = build_rope_cache(args.seq_len + 16, HEAD_DIM, ROPE_THETA, device)

    # ── Optimizer (exact notebook config) ─────────────────────────────
    opt = AdamW(draft.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = (len(train_texts) // args.micro_batch) * args.epochs

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    # ── target_forward wrapper ────────────────────────────────────────
    def target_forward(text):
        return custom_target_forward(target, tokenizer, text, args.seq_len, device)

    # ── train_step (exact notebook Cell 12) ───────────────────────────
    def train_step(batch_texts):
        draft.train()
        loss_total = 0.0
        loss_s0 = 0.0
        correct_s0 = 0
        n_pairs = 0
        for text in batch_texts:
            smp = target_forward(text)
            if smp is None:
                continue
            T = smp["last_h"].shape[0]
            if T < TTT_K + 2:
                continue
            valid = T - TTT_K
            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Step 0: teacher-forced from target fusion
                h_prev = draft.fuse_target(
                    [h[:valid] for h in smp["layer_h"]]
                ).unsqueeze(0)
                e_in = smp["embeds"][1:valid + 1].unsqueeze(0)
                label0 = smp["tok_tgt"][:valid].unsqueeze(0)
                d_h, logits = draft.step(h_prev, e_in, COS, SIN, is_sequence=True)
                ce0 = F.cross_entropy(logits.view(-1, VOCAB), label0.view(-1))
                if FEATURE_LOSS_W > 0:
                    mse = F.mse_loss(
                        d_h.float(),
                        smp["last_h"][1:valid + 1].unsqueeze(0).float(),
                    )
                    loss = TTT_WEIGHTS[0] * ce0 + FEATURE_LOSS_W * mse
                else:
                    loss = TTT_WEIGHTS[0] * ce0

                loss_s0 += ce0.item()
                n_pairs += valid
                correct_s0 += (logits.argmax(-1) == label0).sum().item()

                # Steps 1..K-1 (TTT: draft-on-draft)
                for k in range(1, TTT_K):
                    pred_tok = logits.argmax(-1)
                    e_k = embed_fn(pred_tok).to(d_h.dtype) * EMBED_SCALE
                    d_h, logits = draft.step(d_h, e_k, COS, SIN, is_sequence=True)
                    label_k = smp["tok_tgt"][k:k + valid].unsqueeze(0)
                    ce_k = F.cross_entropy(logits.view(-1, VOCAB), label_k.view(-1))
                    loss = loss + TTT_WEIGHTS[k] * ce_k

            loss_total += loss.item()
            scaler.scale(loss / max(1, args.grad_accum)).backward()

        return loss_total, loss_s0, correct_s0, n_pairs

    # ── evaluate (exact notebook Cell 14) ─────────────────────────────
    @torch.no_grad()
    def evaluate(n=None):
        draft.eval()
        n = n or len(val_texts)
        totals = [0] * TTT_K
        correct = [0] * TTT_K
        for text in val_texts[:n]:
            smp = target_forward(text)
            if smp is None:
                continue
            T = smp["last_h"].shape[0]
            if T < TTT_K + 2:
                continue
            valid = T - TTT_K
            with torch.cuda.amp.autocast(dtype=torch.float16):
                h_prev = draft.fuse_target(
                    [h[:valid] for h in smp["layer_h"]]
                ).unsqueeze(0)
                e_in = smp["embeds"][1:valid + 1].unsqueeze(0)
                d_h, logits = draft.step(h_prev, e_in, COS, SIN, is_sequence=True)
                label0 = smp["tok_tgt"][:valid].unsqueeze(0)
                correct[0] += (logits.argmax(-1) == label0).sum().item()
                totals[0] += valid
                for k in range(1, TTT_K):
                    pred_tok = logits.argmax(-1)
                    e_k = embed_fn(pred_tok).to(d_h.dtype) * EMBED_SCALE
                    d_h, logits = draft.step(d_h, e_k, COS, SIN, is_sequence=True)
                    label_k = smp["tok_tgt"][k:k + valid].unsqueeze(0)
                    correct[k] += (logits.argmax(-1) == label_k).sum().item()
                    totals[k] += valid
        accs = [c / max(1, t) for c, t in zip(correct, totals)]
        cum = 1.0
        exp_len = 1.0
        for a in accs:
            cum *= a
            exp_len += cum
        return accs, exp_len

    # ── Logging / checkpointing (exact notebook Cell 16) ──────────────
    log_f = open(os.path.join(args.save_dir, "eagle3_training.log"), "w")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    def save_ckpt(tag, extra=None):
        p = os.path.join(args.save_dir, f"eagle3_draft_{tag}.pt")
        state = {
            "model": draft.state_dict(),
            "meta": {
                "hidden": HIDDEN, "num_heads": NUM_HEADS, "num_kv": NUM_KV,
                "head_dim": HEAD_DIM, "ffn": FFN_DIM, "vocab": VOCAB,
                "rms_eps": RMS_EPS, "rope_theta": ROPE_THETA,
                "embed_scale": EMBED_SCALE, "fusion_layers": FUSION_LAYERS,
                "ttt_k": TTT_K, "ttt_weights": TTT_WEIGHTS,
                "model_id": "custom_gemma4_model",
            },
        }
        if extra:
            state["meta"].update(extra)
        torch.save(state, p)
        sz = os.path.getsize(p) / 1e6
        log(f"  checkpoint: {p} ({sz:.1f} MB)")

    with open(os.path.join(args.save_dir, "eagle3_config.json"), "w") as f:
        json.dump({
            "architecture": "eagle3_draft",
            "hidden": HIDDEN, "num_heads": NUM_HEADS, "num_kv_heads": NUM_KV,
            "head_dim": HEAD_DIM, "ffn": FFN_DIM, "vocab": VOCAB,
            "rms_eps": RMS_EPS, "rope_theta": ROPE_THETA,
            "embed_scale": EMBED_SCALE, "fusion_layers": FUSION_LAYERS,
            "ttt_k": TTT_K, "model_id": "custom_gemma4_model",
        }, f, indent=2)

    # ── Training loop (exact notebook Cell 16) ────────────────────────
    best_acc0 = 0.0
    step = 0
    seen = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        log(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        random.Random(args.seed + epoch).shuffle(train_texts)
        opt.zero_grad()
        pbar = tqdm(range(0, len(train_texts), args.micro_batch),
                    desc=f"epoch {epoch + 1}")
        loss_ema = None
        acc_ema = None

        for i in pbar:
            batch = train_texts[i:i + args.micro_batch]
            loss, loss0, corr0, pairs = train_step(batch)
            seen += 1

            if (seen % args.grad_accum) == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
                step += 1

            if pairs > 0:
                a0 = corr0 / pairs
                loss_ema = loss0 if loss_ema is None else 0.98 * loss_ema + 0.02 * loss0
                acc_ema = a0 if acc_ema is None else 0.98 * acc_ema + 0.02 * a0
                pbar.set_postfix({
                    "ce0": f"{loss_ema:.3f}",
                    "acc0": f"{acc_ema * 100:.1f}%",
                    "lr": f"{sched.get_last_lr()[0]:.2e}",
                })

            if step > 0 and (step % args.eval_every) == 0 and (seen % args.grad_accum) == 0:
                accs, exp_len = evaluate(n=64)
                log(f"  [eval @ step {step}] accs={[f'{a*100:.1f}' for a in accs]} expL={exp_len:.2f}")
                if accs[0] > best_acc0:
                    best_acc0 = accs[0]
                    save_ckpt("best", extra={
                        "val_accs": accs, "val_exp_len": exp_len, "step": step,
                    })

            if step > 0 and (step % args.save_every) == 0 and (seen % args.grad_accum) == 0:
                save_ckpt(f"step{step}")

    # Final eval
    accs, exp_len = evaluate()
    log(f"\nFinal eval: accs={[f'{a*100:.1f}' for a in accs]} expL={exp_len:.2f}")
    save_ckpt("final", extra={"val_accs": accs, "val_exp_len": exp_len})

    elapsed = (time.time() - t0) / 60
    log(f"Training done in {elapsed:.1f} min. Best val acc0: {best_acc0 * 100:.1f}%")
    log(f"Target: acc0 >= 50% against custom Gemma4Model target")
    log_f.close()

    # Save eval JSON
    with open(os.path.join(args.save_dir, "eagle3_eval.json"), "w") as f:
        json.dump({
            "accs": accs, "expected_length": exp_len,
            "base_tokps": 31, "est_tokps": 31 * exp_len,
        }, f, indent=2)

    print(f"\nNext: download eagle3_draft_best.pt + eagle3_config.json")
    print(f"  python build_eagle3.py --ckpt eagle3_draft_best.pt --output eagle3_draft.mlpackage")


if __name__ == "__main__":
    raise SystemExit(main())
