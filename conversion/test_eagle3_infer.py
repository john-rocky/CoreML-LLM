#!/usr/bin/env python3
"""Verify a trained EAGLE-3 draft end-to-end against Gemma 4 target (Colab / GPU).

Loads a checkpoint saved by conversion/train_eagle3_draft.ipynb and runs:
  1. Target-only greedy generation (reference)
  2. EAGLE-3 speculative generation (draft proposes K=3, target verifies)

Asserts token-by-token equality (speculative decoding is lossless by design when
verification is greedy argmax) and reports wall-clock speedup on the current GPU.

This is a PyTorch sanity check before CoreML conversion. It does NOT use ANE.

Usage (Colab after training):
    !python conversion/test_eagle3_infer.py \\
        --ckpt /content/drive/MyDrive/eagle3_draft/eagle3_draft_best.pt \\
        --prompt "The capital of Japan is" \\
        --max-new 64 \\
        --K 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Draft architecture (must match train_eagle3_draft.ipynb Cell 8) ─────────

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
    def __init__(self, hidden, n_layers, rms_eps):
        super().__init__()
        self.proj = nn.Linear(hidden * n_layers, hidden, bias=False)
        self.norm = RMSNorm(hidden, eps=rms_eps)
    def forward(self, layer_hiddens):
        return self.norm(self.proj(torch.cat(layer_hiddens, dim=-1)))


class DraftDecoderLayer(nn.Module):
    def __init__(self, hidden, num_heads, num_kv, head_dim, ffn, rms_eps):
        super().__init__()
        self.H = hidden; self.NH = num_heads; self.NKV = num_kv; self.HD = head_dim
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
        self.up_proj   = nn.Linear(hidden, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hidden, bias=False)
    def forward(self, x, cos, sin, causal=True):
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
    def __init__(self, cfg, lm_head_weight):
        super().__init__()
        self.cfg = cfg
        self.fusion     = FeatureFusion(cfg["hidden"], len(cfg["fusion_layers"]), cfg["rms_eps"])
        self.input_proj = nn.Linear(cfg["hidden"] * 2, cfg["hidden"], bias=False)
        self.layer      = DraftDecoderLayer(cfg["hidden"], cfg["num_heads"], cfg["num_kv"],
                                            cfg["head_dim"], cfg["ffn"], cfg["rms_eps"])
        self.final_norm = RMSNorm(cfg["hidden"], cfg["rms_eps"])
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


# ── Helpers ────────────────────────────────────────────────────────────────

@torch.no_grad()
def target_forward(target, ids, fusion_layers):
    out = target.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    all_h = out.hidden_states
    layer_h = [all_h[i + 1][0].detach() for i in fusion_layers]
    last_h = all_h[-1][0].detach()
    logits = F.linear(last_h.float(), target.lm_head.weight.float())
    return layer_h, last_h, logits


@torch.no_grad()
def greedy_target_only(target, tokenizer, prompt_ids, max_new, device):
    ids = prompt_ids.clone()
    t0 = time.time()
    for _ in range(max_new):
        out = target.model(input_ids=ids.unsqueeze(0), use_cache=False)
        last_h = out.last_hidden_state[0, -1]
        logit = F.linear(last_h.float(), target.lm_head.weight.float())
        nxt = logit.argmax(-1).unsqueeze(0)
        ids = torch.cat([ids, nxt], dim=0)
    dt = time.time() - t0
    new_tokens = ids[-max_new:].tolist()
    return new_tokens, dt


@torch.no_grad()
def greedy_speculative(target, draft, tokenizer, prompt_ids, max_new, K, cfg, device):
    """EAGLE-3 speculative decode with K draft tokens + batched target verify."""
    embed_fn = target.get_input_embeddings()
    embed_scale = cfg["embed_scale"]
    fusion_layers = cfg["fusion_layers"]
    cos, sin = build_rope_cache(max_new + prompt_ids.shape[0] + 16,
                                cfg["head_dim"], cfg["rope_theta"], device)

    ids = prompt_ids.clone()
    generated = 0
    total_accepts = 0; total_proposals = 0
    t0 = time.time()

    while generated < max_new:
        # 1) Target forward on current sequence to get multi-layer hiddens
        layer_h, last_h, logits = target_forward(target, ids.unsqueeze(0), fusion_layers)
        # Token the target would emit at the LAST position (if no speculation)
        t_tok_next = logits[-1].argmax(-1)

        # 2) Draft proposes K tokens autoregressively from position (len - 1)
        #    Seed: fused hidden at last position + embed(t_tok_next)
        fused_last = draft.fuse_target([h[-1:] for h in layer_h]).unsqueeze(0)   # (1,1,H)
        d_h = fused_last
        e_in = (embed_fn(t_tok_next.unsqueeze(0)).unsqueeze(0).to(d_h.dtype) * embed_scale)
        proposals = []
        for _ in range(K):
            d_h, d_logits = draft.step(d_h, e_in, cos, sin, is_sequence=False)
            pred = d_logits[0, -1].argmax(-1)
            proposals.append(pred.item())
            e_in = (embed_fn(pred.view(1, 1)).to(d_h.dtype) * embed_scale)

        # 3) Target verifies: feed [...prompt, t_tok_next, proposals[0..K-2]] and read argmax at each.
        #    Target's argmax at each position gives the "correct" next token for that prefix.
        verify_ids = torch.cat([ids, t_tok_next.unsqueeze(0),
                                torch.tensor(proposals[:-1], device=device)], dim=0)
        _, _, v_logits = target_forward(target, verify_ids.unsqueeze(0), fusion_layers)
        # v_logits[-K:]: target's predicted next token at each of the last K positions
        target_next = v_logits[-K:].argmax(-1).tolist()

        # 4) Accept t_tok_next (always accepted), then walk through K proposals;
        #    accept proposal[k] if it matches target_next[k], stop at first disagreement.
        accepted = [t_tok_next.item()]
        for k in range(K):
            if proposals[k] == target_next[k]:
                accepted.append(proposals[k])
            else:
                # On disagreement, take target's choice at that position and stop.
                accepted.append(target_next[k])
                break
        # If all K accepted (for-loop didn't break), no bonus token appended here —
        # on next iteration target_forward re-derives hiddens from the extended prefix.
        # This is slightly suboptimal vs textbook speculative decoding but guarantees
        # token-for-token equality with target-only generation.

        # Prevent over-generation
        accepted = accepted[: max_new - generated]
        total_accepts += len(accepted) - 1  # count draft acceptances (excluding the guaranteed t_tok_next)
        total_proposals += K

        ids = torch.cat([ids, torch.tensor(accepted, device=device)], dim=0)
        generated += len(accepted)

    dt = time.time() - t0
    new_tokens = ids[prompt_ids.shape[0]:][:max_new].tolist()
    accept_rate = total_accepts / max(1, total_proposals)
    return new_tokens, dt, accept_rate


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to eagle3_draft_best.pt")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to eagle3_config.json (defaults to sibling of ckpt)")
    ap.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it")
    ap.add_argument("--prompt", type=str, default="The capital of Japan is")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = args.device
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name()}")

    # Load config
    cfg_path = args.config or str(Path(args.ckpt).parent / "eagle3_config.json")
    with open(cfg_path) as f:
        raw = json.load(f)
    cfg = {
        "hidden":        raw["hidden"],
        "num_heads":     raw["num_heads"],
        "num_kv":        raw["num_kv_heads"],
        "head_dim":      raw["head_dim"],
        "ffn":           raw["ffn"],
        "vocab":         raw["vocab"],
        "rms_eps":       raw["rms_eps"],
        "rope_theta":    raw["rope_theta"],
        "embed_scale":   raw["embed_scale"],
        "fusion_layers": raw["fusion_layers"],
    }
    print(f"Config: {cfg}")

    # Load target
    print(f"\nLoading target {args.model_id}...")
    try:
        from transformers import Gemma4ForConditionalGeneration as TCls
    except Exception:
        from transformers import AutoModelForCausalLM as TCls
    from transformers import AutoTokenizer
    target = TCls.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map=device)
    target.eval()
    for p in target.parameters(): p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Build + load draft
    print(f"\nLoading draft from {args.ckpt}...")
    lm_head_weight = target.lm_head.weight.data.detach().clone().float()
    draft = EAGLE3Draft(cfg, lm_head_weight).to(device)
    state = torch.load(args.ckpt, map_location=device)
    sd = state.get("model", state)
    missing, unexpected = draft.load_state_dict(sd, strict=False)
    if missing:    print(f"  missing keys:    {missing[:5]} (total {len(missing)})")
    if unexpected: print(f"  unexpected keys: {unexpected[:5]} (total {len(unexpected)})")
    draft.eval()
    n = sum(p.numel() for p in draft.parameters())
    print(f"  draft params: {n/1e6:.1f}M")

    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt")[0].to(device)
    print(f"\nPrompt: {args.prompt!r} ({prompt_ids.shape[0]} tokens)")
    print(f"max_new: {args.max_new}, K: {args.K}")

    # Baseline: target only
    print("\n[1/2] target-only greedy generation...")
    for _ in range(1): target.model(input_ids=prompt_ids.unsqueeze(0), use_cache=False)  # warmup
    torch.cuda.synchronize() if device == "cuda" else None
    tgt_tokens, tgt_dt = greedy_target_only(target, tokenizer, prompt_ids, args.max_new, device)
    tgt_tps = args.max_new / tgt_dt
    print(f"  {tgt_dt*1000:.0f} ms total = {tgt_tps:.2f} tok/s")
    print(f"  output: {tokenizer.decode(tgt_tokens)!r}")

    # Speculative: EAGLE-3 draft + target verify
    print(f"\n[2/2] EAGLE-3 speculative (K={args.K})...")
    for _ in range(1): target.model(input_ids=prompt_ids.unsqueeze(0), use_cache=False)
    torch.cuda.synchronize() if device == "cuda" else None
    spec_tokens, spec_dt, accept_rate = greedy_speculative(
        target, draft, tokenizer, prompt_ids, args.max_new, args.K, cfg, device)
    spec_tps = args.max_new / spec_dt
    print(f"  {spec_dt*1000:.0f} ms total = {spec_tps:.2f} tok/s")
    print(f"  output: {tokenizer.decode(spec_tokens)!r}")
    print(f"  draft accept rate: {accept_rate*100:.1f}% of {args.K} proposals per step")

    # Losslessness check
    print("\n── Verdict ────────────────────────────────────────────")
    match = tgt_tokens == spec_tokens
    print(f"  outputs match: {match}")
    print(f"  speedup (wall-clock, GPU PyTorch): {spec_tps / tgt_tps:.2f}x")
    print(f"  note: GPU PyTorch numbers are a sanity check, NOT the ANE-deployed speed.")
    if not match:
        # Find first divergence
        for i, (a, b) in enumerate(zip(tgt_tokens, spec_tokens)):
            if a != b:
                print(f"  first divergence at token {i}: target={a} ({tokenizer.decode([a])!r}), "
                      f"spec={b} ({tokenizer.decode([b])!r})")
                break
        print("  → speculative decoding should be LOSSLESS; investigate verification logic.")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
