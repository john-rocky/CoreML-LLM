#!/usr/bin/env python3
"""End-to-end EAGLE-3 speculative test against the CUSTOM Gemma4Model target.

Unlike test_eagle3_infer.py which uses HF Gemma4ForConditionalGeneration
(which has a different forward path than our CoreML chunks on device), this
script uses the same custom Gemma4Model that the collector used to produce
training fusion hiddens. On-device CoreML chunks run the same forward path,
so this Mac-side accept-rate number should track what iPhone sees.

Runs on Mac (CPU or MPS) or Colab GPU. For Mac Studio with ≥ 20 GB RAM,
CPU is the safest: model is ~9 GB fp16, plus draft + KV cache. MPS may
hit memory-map limits on M3 16 GB.

Usage:
    python test_eagle3_custom_target.py \\
        --ckpt /Users/.../eagle3_draft_best.pt \\
        --prompt "The capital of Japan is" \\
        --max-new 64 --K 3 --device cpu

Reports: target-only vs speculative output match, per-step accept counts,
rolling accept rate, final token sequence.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Use bf16 for the target (same rationale as the bf16 collector — avoids
# fp16 overflow in unscaled attention at long N). The draft was trained
# against bf16-collected h_tgt, so this matches the training distribution.
import torch as _torch
import ane_ops as _ane_ops
import models.gemma4_swa_chunks as _swa_chunks
_ane_ops.MODEL_DTYPE = _torch.bfloat16
_swa_chunks.MODEL_DTYPE = _torch.bfloat16

from models.gemma4 import Gemma4Model
from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb, ane_softmax
from models.gemma4_swa_chunks import v_norm
from train_eagle3_standalone import (
    EAGLE3Draft,
    build_rope_cache,
    HIDDEN,
    HEAD_DIM,
    ROPE_THETA,
    EMBED_SCALE,
    FUSION_LAYERS,
)

MODEL_DTYPE = _torch.bfloat16  # local override


# ─────────────────────────────────────────────────────────────────────────
# Custom Gemma4Model forward (mirrors collector's forward_batch but returns
# fusion hiddens, last hidden, and logits together for speculative verify).
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def custom_target_forward(model, input_ids):
    """Return (fusion_list [3 × (B, T, H)], last_hidden (B, T, H), logits (B, T, V))."""
    config = model.config
    device = input_ids.device
    B, seq_len = input_ids.shape
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    n_rep = num_heads // num_kv_heads
    nlayers = config.num_hidden_layers

    hidden = model.embed_tokens(input_ids).to(MODEL_DTYPE) * EMBED_SCALE

    pl_raw = model.embed_tokens_per_layer(input_ids).to(MODEL_DTYPE)
    pl_raw = pl_raw * (config.hidden_size_per_layer_input ** 0.5)
    h_conv = hidden.permute(0, 2, 1).unsqueeze(2)
    pl_proj = model.per_layer_model_projection(h_conv.to(MODEL_DTYPE))
    pl_proj = pl_proj.squeeze(2).permute(0, 2, 1)
    pl_proj = pl_proj * model.per_layer_model_projection_scale
    pld = config.hidden_size_per_layer_input
    pl_parts = []
    for i in range(nlayers):
        s, e = i * pld, (i + 1) * pld
        proj_n = model.per_layer_projection_norm(pl_proj[:, :, s:e])
        pl_parts.append((proj_n + pl_raw[:, :, s:e]) * model.per_layer_input_scale)
    per_layer_combined = torch.cat(pl_parts, dim=2)

    cos_s = model.cos_sliding[:seq_len].unsqueeze(0).unsqueeze(0)
    sin_s = model.sin_sliding[:seq_len].unsqueeze(0).unsqueeze(0)
    cos_f = model.cos_full[:seq_len].unsqueeze(0).unsqueeze(0)
    sin_f = model.sin_full[:seq_len].unsqueeze(0).unsqueeze(0)
    causal = torch.full((seq_len, seq_len), -65504.0, dtype=MODEL_DTYPE, device=device)
    causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
    kv13_k = kv13_v = kv14_k = kv14_v = None
    fusion = {}

    for i in range(nlayers):
        is_full = config.is_full_attention(i)
        is_shared = config.is_kv_shared(i)
        hd = config.get_head_dim(i)
        layer = model.layers[i]

        residual = hidden
        h = layer.input_layernorm(hidden)
        x = h.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        q = layer.self_attn["q_proj"](x).view(B, num_heads, hd, seq_len).permute(0,1,3,2).to(MODEL_DTYPE)
        q = layer.self_attn["q_norm"](q.reshape(B, num_heads*seq_len, hd)).view(B, num_heads, seq_len, hd)
        if not is_shared:
            k = layer.self_attn["k_proj"](x).view(B, num_kv_heads, hd, seq_len).permute(0,1,3,2).to(MODEL_DTYPE)
            k = layer.self_attn["k_norm"](k.reshape(B, num_kv_heads*seq_len, hd)).view(B, num_kv_heads, seq_len, hd)
            v = layer.self_attn["v_proj"](x).view(B, num_kv_heads, hd, seq_len).permute(0,1,3,2).to(MODEL_DTYPE)
            v = v_norm(v)
            if is_full: q, k = apply_rotary_pos_emb(q, k, cos_f, sin_f)
            else:       q, k = apply_rotary_pos_emb(q, k, cos_s, sin_s)
            if i == 13: kv13_k, kv13_v = k, v
            elif i == 14: kv14_k, kv14_v = k, v
        else:
            if is_full: q, _ = apply_rotary_pos_emb(q, q, cos_f, sin_f); k, v = kv14_k, kv14_v
            else:       q, _ = apply_rotary_pos_emb(q, q, cos_s, sin_s); k, v = kv13_k, kv13_v

        Ke = k.repeat_interleave(n_rep, dim=1)
        Ve = v.repeat_interleave(n_rep, dim=1)
        w = torch.matmul(q, Ke.transpose(-1, -2)) + causal
        w = ane_softmax(w, dim=-1)
        a = torch.matmul(w, Ve).permute(0, 2, 1, 3).contiguous().view(B, seq_len, -1)
        a = layer.self_attn["o_proj"](a.permute(0, 2, 1).unsqueeze(2)).squeeze(2).permute(0, 2, 1)
        hidden = residual + layer.post_attention_layernorm(a)

        residual = hidden
        h = layer.pre_feedforward_layernorm(hidden)
        xm = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        g = F.gelu(layer.mlp["gate_proj"](xm), approximate="tanh")
        u = layer.mlp["up_proj"](xm)
        mlp_out = layer.mlp["down_proj"](g * u).squeeze(2).permute(0, 2, 1)
        hidden = residual + layer.post_feedforward_layernorm(mlp_out)

        residual_pl = hidden
        si, ei = i * pld, (i + 1) * pld
        hs = hidden.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        gated = F.gelu(layer.per_layer_input_gate(hs), approximate="tanh")
        gated = gated * per_layer_combined[:, :, si:ei].permute(0, 2, 1).unsqueeze(2)
        gated = layer.per_layer_projection(gated).squeeze(2).permute(0, 2, 1)
        hidden = residual_pl + layer.post_per_layer_input_norm(gated)
        hidden = hidden * layer.layer_scalar.to(MODEL_DTYPE)

        if i in FUSION_LAYERS:
            fusion[i] = hidden.clone()

    hidden = model.norm(hidden)
    lm_w = model.lm_head.weight.data.squeeze(-1).squeeze(-1)
    logits = F.linear(hidden.float(), lm_w.float())
    fusion_list = [fusion[l] for l in FUSION_LAYERS]
    return fusion_list, hidden, logits


def greedy_target_only(model, tokenizer, prompt_ids, max_new, device):
    """Reference: run custom target autoregressively, emit max_new tokens greedily."""
    ids = prompt_ids.to(device)
    t0 = time.time()
    for _ in range(max_new):
        _, _, logits = custom_target_forward(model, ids.unsqueeze(0))
        tok = logits[0, -1].argmax(-1).item()
        ids = torch.cat([ids, torch.tensor([tok], device=device, dtype=ids.dtype)])
    return ids, (time.time() - t0)


def greedy_speculative(model, draft, tokenizer, prompt_ids, max_new, K, device):
    """EAGLE-3 speculative: draft proposes K, target verifies batched."""
    COS, SIN = build_rope_cache(2048, HEAD_DIM, ROPE_THETA, device)
    ids = prompt_ids.to(device)
    total_acc = 0
    total_prop = 0
    t0 = time.time()

    while len(ids) - len(prompt_ids) < max_new:
        fusion_list, last_h, t_logits = custom_target_forward(model, ids.unsqueeze(0))
        t_last_tok = t_logits[0, -1].argmax(-1).item()

        # Draft K tokens autoregressively, seeded from target's fusion at LAST position.
        h_prev = draft.fuse_target([h[:, -1:] for h in fusion_list])  # (1, 1, H)
        e_next = model.embed_tokens(torch.tensor([[t_last_tok]], device=device, dtype=ids.dtype)).to(MODEL_DTYPE) * EMBED_SCALE
        proposals = []
        for _ in range(K):
            with torch.no_grad():
                d_h, d_logits = draft.step(h_prev, e_next, COS[:1], SIN[:1], is_sequence=False)
            pred = d_logits[0, 0].argmax(-1).item()
            proposals.append(pred)
            h_prev = d_h
            e_next = model.embed_tokens(torch.tensor([[pred]], device=device, dtype=ids.dtype)).to(MODEL_DTYPE) * EMBED_SCALE

        # Verify: run custom target on [ids, t_last_tok, proposals[:-1]].
        # Target produces argmax at each new position; accept prefix of matches.
        verify_ids = torch.cat([
            ids,
            torch.tensor([t_last_tok] + proposals[:-1], device=device, dtype=ids.dtype),
        ])
        _, _, v_logits = custom_target_forward(model, verify_ids.unsqueeze(0))
        # v_logits[-K:] predicts the token AFTER each of the last K input positions.
        t_preds = v_logits[0, -K:].argmax(-1).tolist()  # K target predictions

        # Accept prefix of matches between proposals and t_preds.
        accepted = []
        for k in range(K):
            if proposals[k] == t_preds[k]:
                accepted.append(proposals[k])
            else:
                accepted.append(t_preds[k])  # correction
                break
        total_acc += len(accepted) - (0 if proposals[len(accepted)-1] == t_preds[len(accepted)-1] else 1)
        total_prop += K

        ids = torch.cat([
            ids,
            torch.tensor([t_last_tok], device=device, dtype=ids.dtype),
            torch.tensor(accepted, device=device, dtype=ids.dtype),
        ])
        if len(ids) - len(prompt_ids) >= max_new:
            break

    return ids, (time.time() - t0), total_acc, total_prop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to eagle3_draft_best.pt")
    ap.add_argument("--hf-dir", default=None, help="HF model dir for custom Gemma4Model weights")
    ap.add_argument("--prompt", default="The capital of Japan is")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--device", default="cpu",
                    help="cpu (safest on Mac), mps (experimental), or cuda.")
    args = ap.parse_args()

    device = args.device
    print(f"Device: {device}")

    hf_dir = args.hf_dir
    if hf_dir is None:
        from huggingface_hub import snapshot_download
        print("Downloading google/gemma-4-E2B-it weights + tokenizer...")
        hf_dir = snapshot_download("google/gemma-4-E2B-it")

    print(f"Loading custom Gemma4Model from {hf_dir}...")
    model = Gemma4Model.from_pretrained(hf_dir, context_length=2048)
    model = model.to(MODEL_DTYPE).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)

    print(f"Loading draft from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    lm_w = model.lm_head.weight.data.squeeze(-1).squeeze(-1).float().to(device)
    draft = EAGLE3Draft(lm_w).to(device)
    draft.load_state_dict(ckpt["model"])
    draft.eval()
    for p in draft.parameters():
        p.requires_grad = False
    print(f"  draft params: {sum(p.numel() for p in draft.parameters()) / 1e6:.1f}M")
    print(f"  draft meta: acc={ckpt['meta'].get('acc')} expL={ckpt['meta'].get('expL')}")

    prompt_ids = tokenizer.encode(args.prompt, return_tensors="pt")[0].to(device)
    print(f"\nPrompt: {args.prompt!r} ({prompt_ids.shape[0]} tokens)")
    print(f"max_new: {args.max_new}, K: {args.K}\n")

    print("[1/2] target-only greedy (custom Gemma4Model) ...")
    ref_ids, ref_t = greedy_target_only(model, tokenizer, prompt_ids, args.max_new, device)
    ref_text = tokenizer.decode(ref_ids[prompt_ids.shape[0]:].cpu().tolist())
    print(f"  {ref_t*1000:.0f} ms total = {args.max_new/ref_t:.2f} tok/s")
    print(f"  output: {ref_text!r}")

    print("\n[2/2] EAGLE-3 speculative (custom target) ...")
    spec_ids, spec_t, total_acc, total_prop = greedy_speculative(
        model, draft, tokenizer, prompt_ids, args.max_new, args.K, device,
    )
    spec_text = tokenizer.decode(spec_ids[prompt_ids.shape[0]:].cpu().tolist())
    emitted = len(spec_ids) - len(prompt_ids)
    print(f"  {spec_t*1000:.0f} ms total = {emitted/spec_t:.2f} tok/s")
    print(f"  output: {spec_text!r}")
    print(f"  accept rate: {total_acc/total_prop*100:.1f}% ({total_acc}/{total_prop} of K={args.K} proposals)")

    match = ref_text[:50] == spec_text[:50]
    print(f"\n── Verdict ──")
    print(f"  outputs match (first 50 chars): {match}")
    print(f"  speedup (wall-clock, PyTorch {device}): {ref_t/spec_t:.2f}x")
    print(f"  note: PyTorch numbers are NOT iPhone ANE — use for correctness / accept rate only.")


if __name__ == "__main__":
    raise SystemExit(main())
