#!/usr/bin/env python3
"""Convert a trained EAGLE-3 draft to an ANE-friendly Core ML mlpackage.

Input:   eagle3_draft_best.pt  +  eagle3_config.json  (from train_eagle3_draft.ipynb)
Output:  eagle3_draft.mlpackage  (ANE-targeted, fp16, optionally INT4 palettized)

Design choices for ANE compatibility:
  - Inference is one-token-per-call (T=1). The draft is invoked K times
    autoregressively from Swift. This removes self-attention causal masking
    and makes RoPE a no-op (rotations of same-position Q,K cancel in Q·K).
  - Concat ops are eliminated: `Linear(cat([h_prev, e_next]))` is rewritten
    as `Linear_A(h_prev) + Linear_B(e_next)` (equivalent math, no concat).
    Same for the 3-way feature fusion.
  - RMSNorm is kept as-is; it maps to LayerNorm on ANE (via the established
    cat-trick in ane_ops when needed).
  - lm_head is applied in-graph with argmax to minimize output transfer,
    matching the pattern used by chunk4 in build_speculative.py.

Usage (Mac, after training in Colab + download of the ckpt):
    python conversion/build_eagle3.py \\
        --ckpt ./eagle3_draft/eagle3_draft_best.pt \\
        --output ./eagle3_draft.mlpackage \\
        --palettize-int4

TODOs (need coordination with bench session that owns build_speculative.py):
  - Emit `hidden_mid` outputs from target chunks at FUSION_LAYERS so the draft
    can consume them at inference time. Suggested: new conversion flag in
    build_speculative.py or a sibling script. Not done here.
  - Build `verify_chunk{1..4}_K3.mlpackage` with seq_dim EnumeratedShapes
    {1, 3} so Swift can switch between decode (K=1) and verify (K=3).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Matches train_eagle3_draft.ipynb Cell 8 (training architecture) ─────────

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
        super().__init__(); self.eps = eps
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        n = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * n).to(dtype)


# ── ANE-friendly T=1 inference graph ─────────────────────────────────────────

class EAGLE3DraftANE(nn.Module):
    """Single-step draft for ANE export.

    Inference contract: T=1 per call. For K>1 drafting, Swift calls this K times
    with the previous step's hidden as `h_prev` and the embed of the previous
    token as `e_next`. The first call uses the target's multi-layer fused hidden
    (produced outside this graph) as `h_prev` and embed(t_tok_next) as `e_next`.

    The multi-layer feature fusion is NOT in this graph — it is applied once
    per drafting burst, before the first call, by concatenating target's
    hidden_low/mid/high from the decode chunks and running through
    `fused_hidden.mlpackage` (or done in Swift with a single linear op).

    Shapes (B=1, T=1):
      h_prev:  (1, 1, H)  fp16
      e_next:  (1, 1, H)  fp16
      →
      h_out:   (1, 1, H)  fp16
      token:   scalar int (argmax of logits)
      logit:   scalar fp16 (value of argmax token, for verify)
    """

    def __init__(self, cfg):
        super().__init__()
        H   = cfg["hidden"]
        NH  = cfg["num_heads"]
        NKV = cfg["num_kv"]
        HD  = cfg["head_dim"]
        FFN = cfg["ffn"]
        eps = cfg["rms_eps"]
        self.H, self.NH, self.NKV, self.HD = H, NH, NKV, HD

        # input_proj: Linear(2H, H) split into two Linear(H, H) to avoid concat
        self.in_h = nn.Linear(H, H, bias=False)
        self.in_e = nn.Linear(H, H, bias=False)

        # attention
        self.pre_attn_norm = RMSNorm(H, eps)
        self.q_proj = nn.Linear(H, NH  * HD, bias=False)
        self.k_proj = nn.Linear(H, NKV * HD, bias=False)
        self.v_proj = nn.Linear(H, NKV * HD, bias=False)
        self.q_norm = RMSNorm(HD, eps)
        self.k_norm = RMSNorm(HD, eps)
        self.v_norm = RMSNormNoScale(eps)
        self.o_proj = nn.Linear(NH * HD, H, bias=False)

        # FFN
        self.pre_ffn_norm = RMSNorm(H, eps)
        self.gate_proj = nn.Linear(H, FFN, bias=False)
        self.up_proj   = nn.Linear(H, FFN, bias=False)
        self.down_proj = nn.Linear(FFN, H, bias=False)

        # final norm + lm_head (lm_head weight registered as buffer, non-learned)
        self.final_norm = RMSNorm(H, eps)
        self.register_buffer("lm_head_weight",
                             torch.zeros(cfg["vocab"], H, dtype=torch.float16),
                             persistent=False)

    def forward(self, h_prev, e_next):
        # input projection (decomposed, no concat)
        x = self.in_h(h_prev) + self.in_e(e_next)

        # attention (T=1: self-attn degenerates, RoPE rotates Q and K identically so Q·K is invariant)
        h = self.pre_attn_norm(x)
        q = self.q_proj(h).view(1, 1, self.NH,  self.HD)
        k = self.k_proj(h).view(1, 1, self.NKV, self.HD)
        v = self.v_proj(h).view(1, 1, self.NKV, self.HD)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        # For T=1: attn_weight = softmax(Q @ K^T / sqrt(d)) = 1 (single position).
        # attn_output = 1 * V = V. So we can skip the matmul/softmax entirely.
        # This is an exact simplification, not an approximation, for T=1 inputs.
        # Expand K/V heads via GQA (rep = NH / NKV). For the T=1 degeneracy the
        # output is V expanded to NH heads; equivalent to repeat_interleave.
        rep = self.NH // self.NKV
        attn = v.repeat_interleave(rep, dim=2)        # (1, 1, NH, HD)
        attn = attn.reshape(1, 1, self.NH * self.HD)
        x = x + self.o_proj(attn)

        # FFN
        h = self.pre_ffn_norm(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))

        # final norm + lm_head + argmax (in-graph)
        h_out = self.final_norm(x)                     # (1, 1, H)
        logits = F.linear(h_out.float(),
                          self.lm_head_weight.float()) # (1, 1, V)
        token = logits.argmax(-1)                      # (1, 1)
        logit = logits.gather(-1, token.unsqueeze(-1)).squeeze(-1)  # (1, 1)
        return h_out, token.to(torch.int32), logit.to(torch.float16)


# ── Checkpoint → ANE model weight mapping ────────────────────────────────────

def load_into_ane_model(ane: EAGLE3DraftANE, ckpt_state: dict, lm_head_weight: torch.Tensor):
    """Copy trained weights into the ANE-friendly model, splitting concat layers."""
    sd = ckpt_state.get("model", ckpt_state)

    def get(name):
        # Direct copy if present, else None
        return sd.get(name, None)

    def set_(dst: torch.Tensor, src):
        if src is None: raise KeyError(f"missing weight in ckpt")
        if dst.shape != src.shape:
            raise ValueError(f"shape mismatch: dst={tuple(dst.shape)} src={tuple(src.shape)}")
        with torch.no_grad():
            dst.copy_(src)

    H = ane.H
    # input_proj: W of shape (H, 2H). Split rows/cols.
    # In training: Linear(2H, H) weight is shape (H, 2H), out = W @ cat([h, e])
    # Equivalent: in_h with weight W[:, :H], in_e with weight W[:, H:]
    W_in = get("input_proj.weight")
    if W_in is None:
        raise KeyError("input_proj.weight missing in checkpoint")
    if W_in.shape != (H, 2 * H):
        raise ValueError(f"input_proj weight shape {tuple(W_in.shape)} != ({H}, {2*H})")
    set_(ane.in_h.weight, W_in[:, :H].contiguous())
    set_(ane.in_e.weight, W_in[:, H:].contiguous())

    # direct copies (attention + FFN + norms)
    mapping = {
        "pre_attn_norm.weight": "layer.pre_attn_norm.weight",
        "q_proj.weight":        "layer.q_proj.weight",
        "k_proj.weight":        "layer.k_proj.weight",
        "v_proj.weight":        "layer.v_proj.weight",
        "q_norm.weight":        "layer.q_norm.weight",
        "k_norm.weight":        "layer.k_norm.weight",
        "o_proj.weight":        "layer.o_proj.weight",
        "pre_ffn_norm.weight":  "layer.pre_ffn_norm.weight",
        "gate_proj.weight":     "layer.gate_proj.weight",
        "up_proj.weight":       "layer.up_proj.weight",
        "down_proj.weight":     "layer.down_proj.weight",
        "final_norm.weight":    "final_norm.weight",
    }
    for ane_name, ckpt_name in mapping.items():
        src = get(ckpt_name)
        if src is None:
            print(f"  WARN: {ckpt_name} missing, leaving {ane_name} at init")
            continue
        dst = dict(ane.named_parameters())[ane_name]
        set_(dst, src)

    # lm_head buffer
    with torch.no_grad():
        ane.lm_head_weight.copy_(lm_head_weight.to(torch.float16))

    print(f"  loaded {len(mapping) + 2} tensors into ANE draft model")


# ── Feature Fusion as a separate tiny mlpackage ─────────────────────────────

class FeatureFusionANE(nn.Module):
    """Three-way fusion: Linear(3H, H). Decomposed into 3 × Linear(H, H) to skip concat."""
    def __init__(self, H, n_layers, eps):
        super().__init__()
        self.n = n_layers
        self.ins = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in range(n_layers)])
        self.norm = RMSNorm(H, eps)

    def forward(self, h_low, h_mid, h_high):
        x = self.ins[0](h_low) + self.ins[1](h_mid) + self.ins[2](h_high)
        return self.norm(x)


def load_fusion(fus: FeatureFusionANE, ckpt_state: dict):
    sd = ckpt_state.get("model", ckpt_state)
    W = sd["fusion.proj.weight"]        # shape (H, n*H)
    H = fus.ins[0].weight.shape[0]
    assert W.shape == (H, fus.n * H), f"fusion weight shape {tuple(W.shape)}"
    with torch.no_grad():
        for i, layer in enumerate(fus.ins):
            layer.weight.copy_(W[:, i * H:(i + 1) * H].contiguous())
        fus.norm.weight.copy_(sd["fusion.norm.weight"])


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default=None,
                    help="defaults to sibling eagle3_config.json")
    ap.add_argument("--output", type=str, default="./eagle3_draft.mlpackage")
    ap.add_argument("--fusion-output", type=str, default="./eagle3_fusion.mlpackage")
    ap.add_argument("--palettize-int4", action="store_true",
                    help="apply per-grouped-channel INT4 palettization (group_size=32)")
    ap.add_argument("--lm-head", type=str, default=None,
                    help="path to a .pt tensor of lm_head weights. If omitted, loaded from ckpt.")
    ap.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it",
                    help="HF model id to read lm_head weights from if --lm-head is not provided")
    args = ap.parse_args()

    cfg_path = args.config or str(Path(args.ckpt).parent / "eagle3_config.json")
    with open(cfg_path) as f: raw = json.load(f)
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
    print(f"config: hidden={cfg['hidden']} heads={cfg['num_heads']} kv={cfg['num_kv']} "
          f"head_dim={cfg['head_dim']} ffn={cfg['ffn']} vocab={cfg['vocab']}")

    # Load ckpt (torch.save dict; weights_only=False to accept full pickle,
    # since the trainer saves {"model": state_dict, "cfg": ..., "epoch": ...}).
    print(f"loading ckpt: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Obtain lm_head weight. Support both:
    #   .bin — raw fp16 bytes from `arr.tofile(...)` (used by HASS trainer),
    #          load via np.fromfile and reshape to (vocab, hidden).
    #   .pt  — torch pickle (legacy), loaded via torch.load.
    if args.lm_head:
        print(f"loading lm_head from {args.lm_head}")
        p = str(args.lm_head)
        if p.endswith(".bin"):
            import numpy as np
            lm_head_np = np.fromfile(p, dtype=np.float16)
            # Shape is (vocab, hidden) = (262144, 1536) for Gemma 4 E2B.
            if lm_head_np.size % 1536 != 0:
                raise ValueError(
                    f"lm_head .bin size {lm_head_np.size} not divisible by hidden=1536")
            lm_head_weight = torch.from_numpy(
                lm_head_np.reshape(lm_head_np.size // 1536, 1536).copy())
        else:
            lm_head_weight = torch.load(p, map_location="cpu", weights_only=False)
    else:
        print(f"loading lm_head from HF {args.model_id} (will cache)")
        try:
            from transformers import Gemma4ForConditionalGeneration as TCls
        except Exception:
            from transformers import AutoModelForCausalLM as TCls
        tgt = TCls.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="cpu")
        lm_head_weight = tgt.lm_head.weight.data.detach().clone().to(torch.float16)
        del tgt
    print(f"  lm_head: {tuple(lm_head_weight.shape)} {lm_head_weight.dtype}")

    # Build + load ANE draft
    ane = EAGLE3DraftANE(cfg).to(torch.float16).eval()
    load_into_ane_model(ane, state, lm_head_weight)

    # Build + load fusion
    fus = FeatureFusionANE(cfg["hidden"], len(cfg["fusion_layers"]), cfg["rms_eps"]).to(torch.float16).eval()
    load_fusion(fus, state)

    # Sanity: forward pass shape check
    H = cfg["hidden"]
    dummy_h = torch.zeros((1, 1, H), dtype=torch.float16)
    dummy_e = torch.zeros((1, 1, H), dtype=torch.float16)
    with torch.no_grad():
        h_out, tok, lg = ane(dummy_h, dummy_e)
    print(f"  ane draft forward OK: h_out={tuple(h_out.shape)} tok={tuple(tok.shape)} lg={tuple(lg.shape)}")

    # Convert to Core ML
    print("\nconverting to Core ML...")
    import coremltools as ct

    # Draft mlpackage
    traced = torch.jit.trace(ane, (dummy_h, dummy_e), strict=False)
    mlm = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="h_prev", shape=(1, 1, H), dtype=ct.converters.mil.mil.types.fp16),
            ct.TensorType(name="e_next", shape=(1, 1, H), dtype=ct.converters.mil.mil.types.fp16),
        ],
        outputs=[
            ct.TensorType(name="h_out"),
            ct.TensorType(name="token"),
            ct.TensorType(name="logit"),
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    if args.palettize_int4:
        print("  palettizing weights INT4 (group_size=32)...")
        import coremltools.optimize.coreml as cto
        cfg_q = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=32)
        )
        mlm = cto.palettize_weights(mlm, cfg_q)

    mlm.save(args.output)
    size_mb = sum(f.stat().st_size for f in Path(args.output).rglob("*") if f.is_file()) / 1e6
    print(f"  saved: {args.output} ({size_mb:.1f} MB)")

    # Fusion mlpackage
    print("\nconverting fusion to Core ML...")
    dummy_l = torch.zeros((1, 1, H), dtype=torch.float16)
    traced_f = torch.jit.trace(fus, (dummy_l, dummy_l, dummy_l), strict=False)
    fmlm = ct.convert(
        traced_f,
        inputs=[
            ct.TensorType(name="h_low",  shape=(1, 1, H), dtype=ct.converters.mil.mil.types.fp16),
            ct.TensorType(name="h_mid",  shape=(1, 1, H), dtype=ct.converters.mil.mil.types.fp16),
            ct.TensorType(name="h_high", shape=(1, 1, H), dtype=ct.converters.mil.mil.types.fp16),
        ],
        outputs=[ct.TensorType(name="h_fused")],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )
    fmlm.save(args.fusion_output)
    fsize_mb = sum(f.stat().st_size for f in Path(args.fusion_output).rglob("*") if f.is_file()) / 1e6
    print(f"  saved: {args.fusion_output} ({fsize_mb:.1f} MB)")

    print("\nnext steps:")
    print("  1. Add hidden_mid outputs to target chunks at layers "
          f"{cfg['fusion_layers']} (coordinate with build_speculative.py).")
    print("  2. Build verify_chunk{1..4}_K3.mlpackage via EnumeratedShapes for seq_dim in {1, 3}.")
    print("  3. Swift: 3-tier call loop — fusion ⟶ draft×K ⟶ verify_chunks.")


if __name__ == "__main__":
    main()
