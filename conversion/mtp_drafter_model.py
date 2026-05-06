#!/usr/bin/env python3
"""PyTorch reimplementation of Google's official Gemma 4 MTP drafter.

Source: HF `google/gemma-4-E2B-it-assistant` (BF16 safetensors, Apache 2.0).

Architecture (matches transformers.models.gemma4_assistant.modeling_gemma4_assistant
in transformers >= 5.7.0):

  pre_projection   Linear(2 * 1536 → 256)         # = mtp_pre_proj
  layers[0..2]     Gemma 4 SWA layer (head_dim=256, theta=10k, KV-shared)
  layers[3]        Gemma 4 full layer (head_dim=512, theta=1M, partial=0.25, KV-shared)
                   each layer has a scalar `layer_scalar` applied at the very end
  norm             RMSNorm(256)                   # = final_norm
  post_projection  Linear(256 → 1536)             # = mtp_post_proj
  lm_head          weight tied to model.embed_tokens.weight (262144, 256)

Differences vs the now-obsolete LiteRT W4A8 drafter:
  * BF16 weights (no quant drift)
  * Per-layer learned scalar (`layer_scalar`) on the residual output
  * No final logit softcapping (`final_logit_softcapping=null`)
  * Full-attention RoPE: theta=1e6, partial_rotary_factor=0.25 (192 nope angles)
  * lm_head tied to embed_tokens (no separate weight)
  * Optional `masked_embedding` centroid lookup (not used at inference for top-K
    argmax — full lm_head is computed instead; the masked path is purely a
    decoding speedup that doesn't change the top-1 in practice)

Q-only attention: every layer reads K/V directly from the target's
shared_kv_states (kv13 sliding, kv14 full) — `num_kv_shared_layers=4`.

Usage:
    python conversion/mtp_drafter_model.py \\
        --hf-repo google/gemma-4-E2B-it-assistant \\
        --output output/mtp_probe/mtp_drafter.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class MtpDrafterConfig:
    """Default = E2B drafter geometry. Use `MtpDrafterConfig.e4b()` for E4B.

    The two variants differ only in `target_hidden` (E2B: 1536, E4B: 2560);
    everything else (drafter internal hidden=256, 4 layers, num_centroids,
    centroid top-k, vocab) is identical between the two official drafters.
    """
    hidden_size: int = 256          # internal dim
    input_size: int = 3072          # 2 × target hidden (1536)
    target_hidden: int = 1536       # target model hidden_size
    num_layers: int = 4
    layer_types = ("sliding_attention",) * 3 + ("full_attention",)

    @classmethod
    def for_target(cls, target_hidden: int) -> "MtpDrafterConfig":
        cfg = cls()
        cfg.target_hidden = target_hidden
        cfg.input_size = 2 * target_hidden
        return cfg

    @classmethod
    def e4b(cls) -> "MtpDrafterConfig":
        return cls.for_target(2560)

    # SWA layers (0-2)
    swa_num_heads: int = 4
    swa_head_dim: int = 256
    swa_q_dim: int = 1024            # 4 × 256
    swa_rope_theta: float = 10000.0
    swa_partial_rotary: float = 1.0

    # Full layer (3)
    full_num_heads: int = 4
    full_head_dim: int = 512
    full_q_dim: int = 2048           # 4 × 512
    full_rope_theta: float = 1_000_000.0
    full_partial_rotary: float = 0.25

    # MLP
    ffn_dim: int = 2048

    # Vocab + projections
    vocab_size: int = 262144

    # Misc
    rms_eps: float = 1e-6
    max_position_embeddings: int = 131072


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Gemma 4 RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # `pow(.., -0.5)` mirrors upstream — keeps Torch/JAX bit-exact.
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).pow(-0.5)
        return (x * rms * self.weight.float()).to(dtype)


def _precompute_rope(head_dim: int, max_pos: int, theta: float,
                     partial_rotary_factor: float = 1.0):
    """Precompute cos/sin tables for RoPE in Gemma-4 layout.

    Returns cos/sin of shape (max_pos, head_dim). For partial_rotary < 1,
    the first `2 * rope_angles` channels carry rotation; the remainder
    has inv_freq = 0 (cos=1, sin=0) which is identity for the rotated form.
    The half/half layout matches `torch.cat((freqs, freqs), dim=-1)` from
    the upstream rope module — i.e. cos/sin both have head_dim entries
    that mirror their first half.
    """
    rope_angles = int(partial_rotary_factor * head_dim) // 2
    inv_rot = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2,
                                            dtype=torch.float32) / head_dim))
    nope = head_dim // 2 - rope_angles
    if nope > 0:
        inv_freq = torch.cat([inv_rot, torch.zeros(nope, dtype=torch.float32)])
    else:
        inv_freq = inv_rot
    # inv_freq has length head_dim/2. Build (max_pos, head_dim/2) freqs,
    # then duplicate along last dim → (max_pos, head_dim).
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_pos, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope_q(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to Q. q: (B, H, 1, D); cos/sin: (1, D) → broadcast to (1,1,1,D)."""
    cos = cos.view(1, 1, 1, -1)
    sin = sin.view(1, 1, 1, -1)
    return q * cos + _rotate_half(q) * sin


class DrafterAttention(nn.Module):
    """Q-only attention that reads target's shared K/V cache."""

    def __init__(self, hidden: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)
        # Gemma 4 uses scaling = 1.0 (not 1/sqrt(head_dim)); QK_norm absorbs scale.
        self.scale = 1.0

    def forward(
        self,
        x: torch.Tensor,           # (B, 1, hidden)
        kv_k: torch.Tensor,        # (B, num_kv, ctx, head_dim)
        kv_v: torch.Tensor,        # (B, num_kv, head_dim, ctx)  — pre-transposed
        cos: torch.Tensor,         # (1, head_dim) RoPE for current pos
        sin: torch.Tensor,
        mask: torch.Tensor,        # (B, 1, 1, ctx) additive (-inf masked)
    ) -> torch.Tensor:
        B = x.shape[0]
        q = self.q_proj(x)                                     # (B, 1, nH*hd)
        q = q.view(B, 1, self.num_heads, self.head_dim)        # (B, 1, nH, hd)
        q = self.q_norm(q)                                     # per-head RMSNorm
        q = q.permute(0, 2, 1, 3)                              # (B, nH, 1, hd)
        q = _apply_rope_q(q, cos, sin)

        # K^T: (B, num_kv, hd, ctx). Q @ K^T → (B, nH, 1, ctx) with broadcast.
        k_t = kv_k.transpose(-2, -1)
        attn = torch.matmul(q.float(), k_t.float()) * self.scale
        attn = attn + mask.float()
        attn = F.softmax(attn, dim=-1).to(x.dtype)

        # V is stored as (B, num_kv, hd, ctx). attn @ V^T  where V^T: (B,num_kv,ctx,hd).
        v_t = kv_v.transpose(-2, -1)
        out = torch.matmul(attn.float(), v_t.float()).to(x.dtype)  # (B, nH, 1, hd)
        out = out.permute(0, 2, 1, 3).reshape(B, 1, self.num_heads * self.head_dim)
        return self.o_proj(out)


class DrafterMLP(nn.Module):
    """GeGLU MLP using Gemma 4 names: gate_proj, up_proj, down_proj."""

    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, ffn, bias=False)
        self.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class DrafterLayer(nn.Module):
    """Single drafter layer with Gemma 4 sandwich norms + per-layer scalar."""

    def __init__(self, hidden: int, num_heads: int, head_dim: int,
                 ffn: int, eps: float):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden, eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps)
        self.pre_feedforward_layernorm = RMSNorm(hidden, eps)
        self.post_feedforward_layernorm = RMSNorm(hidden, eps)
        self.self_attn = DrafterAttention(hidden, num_heads, head_dim, eps)
        self.mlp = DrafterMLP(hidden, ffn)
        # Per-layer learned scalar applied at the very end of forward().
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, kv_k, kv_v, cos, sin, mask):
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, kv_k, kv_v, cos, sin, mask)
        h = self.post_attention_layernorm(h)
        x = residual + h

        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = residual + h

        return x * self.layer_scalar


class MtpDrafterModel(nn.Module):
    """Google's official Gemma 4 MTP drafter (PyTorch port)."""

    def __init__(self, cfg: MtpDrafterConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = MtpDrafterConfig()
        self.cfg = cfg

        # Pre-projection: concat(embed, proj_act) → hidden.
        self.pre_projection = nn.Linear(cfg.input_size, cfg.hidden_size, bias=False)

        self.layers = nn.ModuleList()
        for i in range(cfg.num_layers):
            if cfg.layer_types[i] == "sliding_attention":
                self.layers.append(DrafterLayer(
                    cfg.hidden_size, cfg.swa_num_heads, cfg.swa_head_dim,
                    cfg.ffn_dim, cfg.rms_eps))
            else:
                self.layers.append(DrafterLayer(
                    cfg.hidden_size, cfg.full_num_heads, cfg.full_head_dim,
                    cfg.ffn_dim, cfg.rms_eps))

        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.post_projection = nn.Linear(cfg.hidden_size, cfg.target_hidden, bias=False)

        # lm_head is tied to embed_tokens at load time.
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # MaskedEmbedder sub-module — buffers only, not consumed by this
        # reference forward. build_mtp_drafter.py reads `state_dict()`
        # entries `masked_embedding.centroids.weight` and
        # `masked_embedding.token_ordering` from this module to copy into
        # the ANE-targeted MaskedEmbedderANE buffers.
        self.masked_embedding = nn.Module()
        self.masked_embedding.centroids = nn.Linear(cfg.hidden_size, 2048, bias=False)
        # persistent=True so it round-trips via state_dict() into load_ane_weights.
        self.masked_embedding.register_buffer(
            "token_ordering", torch.zeros(cfg.vocab_size, dtype=torch.int32),
            persistent=True)

        # RoPE tables, precomputed at init (deployment-time recompute is also fine).
        swa_cos, swa_sin = _precompute_rope(
            cfg.swa_head_dim, cfg.max_position_embeddings,
            cfg.swa_rope_theta, cfg.swa_partial_rotary)
        full_cos, full_sin = _precompute_rope(
            cfg.full_head_dim, cfg.max_position_embeddings,
            cfg.full_rope_theta, cfg.full_partial_rotary)
        self.register_buffer("swa_cos", swa_cos, persistent=False)
        self.register_buffer("swa_sin", swa_sin, persistent=False)
        self.register_buffer("full_cos", full_cos, persistent=False)
        self.register_buffer("full_sin", full_sin, persistent=False)

    def forward(
        self,
        activations: torch.Tensor,      # (B, 1, 3072) = concat(embed, proj_act)
        input_pos: torch.Tensor,         # (B,) int position
        kv13_k: torch.Tensor,            # (B, num_kv_swa, ctx, 256)
        kv13_v: torch.Tensor,            # (B, num_kv_swa, 256, ctx) pre-transposed
        kv14_k: torch.Tensor,            # (B, num_kv_full, ctx, 512)
        kv14_v: torch.Tensor,            # (B, num_kv_full, 512, ctx) pre-transposed
        mask_swa: torch.Tensor,          # (B, 1, 1, ctx)
        mask_full: torch.Tensor,
    ):
        x = self.pre_projection(activations)

        pos = int(input_pos[0].item())
        swa_cos = self.swa_cos[pos:pos + 1]                  # (1, head_dim)
        swa_sin = self.swa_sin[pos:pos + 1]
        full_cos = self.full_cos[pos:pos + 1]
        full_sin = self.full_sin[pos:pos + 1]

        for i, layer in enumerate(self.layers):
            if self.cfg.layer_types[i] == "sliding_attention":
                x = layer(x, kv13_k, kv13_v, swa_cos, swa_sin, mask_swa)
            else:
                x = layer(x, kv14_k, kv14_v, full_cos, full_sin, mask_full)

        h = self.norm(x)
        logits = self.lm_head(h.float()).to(h.dtype)
        proj_act = self.post_projection(h)
        return logits, proj_act


# ---------------------------------------------------------------------------
# Weight loading from HF safetensors
# ---------------------------------------------------------------------------

def load_from_hf(model: MtpDrafterModel, hf_repo_or_path: str) -> list[str]:
    """Load weights from a HF safetensors file into MtpDrafterModel.

    `hf_repo_or_path` may be a HF repo id (e.g. 'google/gemma-4-E2B-it-assistant')
    or a local directory containing model.safetensors + config.json.
    """
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    p = Path(hf_repo_or_path)
    if p.is_dir():
        safetensors_path = str(p / "model.safetensors")
    elif p.is_file():
        safetensors_path = str(p)
    else:
        safetensors_path = hf_hub_download(hf_repo_or_path, "model.safetensors")

    print(f"Loading HF safetensors: {safetensors_path}")
    with safe_open(safetensors_path, framework="pt") as f:
        hf_keys = set(f.keys())

        def get(k):
            return f.get_tensor(k).float()

        sd = model.state_dict()
        matched = []

        # Pre/post projections.
        sd["pre_projection.weight"] = get("pre_projection.weight")
        matched.append("pre_projection.weight")
        sd["post_projection.weight"] = get("post_projection.weight")
        matched.append("post_projection.weight")

        # Final norm.
        sd["norm.weight"] = get("model.norm.weight")
        matched.append("norm.weight")

        # lm_head: tied to embed_tokens at inference. Use embed_tokens.weight directly.
        sd["lm_head.weight"] = get("model.embed_tokens.weight")
        matched.append("lm_head.weight")

        # MaskedEmbedder (centroid lm-head) — load when present so the ANE
        # build can opt into the cluster-routed lm-head path.
        if "masked_embedding.centroids.weight" in hf_keys:
            sd["masked_embedding.centroids.weight"] = get(
                "masked_embedding.centroids.weight")
            sd["masked_embedding.token_ordering"] = (
                f.get_tensor("masked_embedding.token_ordering").to(torch.int32))
            matched.extend(["masked_embedding.centroids.weight",
                            "masked_embedding.token_ordering"])

        # Per-layer.
        for i in range(model.cfg.num_layers):
            base = f"model.layers.{i}"
            sd[f"layers.{i}.input_layernorm.weight"] = get(f"{base}.input_layernorm.weight")
            sd[f"layers.{i}.post_attention_layernorm.weight"] = get(
                f"{base}.post_attention_layernorm.weight")
            sd[f"layers.{i}.pre_feedforward_layernorm.weight"] = get(
                f"{base}.pre_feedforward_layernorm.weight")
            sd[f"layers.{i}.post_feedforward_layernorm.weight"] = get(
                f"{base}.post_feedforward_layernorm.weight")

            sd[f"layers.{i}.self_attn.q_proj.weight"] = get(f"{base}.self_attn.q_proj.weight")
            sd[f"layers.{i}.self_attn.q_norm.weight"] = get(f"{base}.self_attn.q_norm.weight")
            sd[f"layers.{i}.self_attn.o_proj.weight"] = get(f"{base}.self_attn.o_proj.weight")

            sd[f"layers.{i}.mlp.gate_proj.weight"] = get(f"{base}.mlp.gate_proj.weight")
            sd[f"layers.{i}.mlp.up_proj.weight"] = get(f"{base}.mlp.up_proj.weight")
            sd[f"layers.{i}.mlp.down_proj.weight"] = get(f"{base}.mlp.down_proj.weight")

            sd[f"layers.{i}.layer_scalar"] = get(f"{base}.layer_scalar")

            for k in (
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                "self_attn.q_proj.weight", "self_attn.q_norm.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                "layer_scalar",
            ):
                matched.append(f"layers.{i}.{k}")

        skipped = [k for k in hf_keys if k not in {
            "pre_projection.weight", "post_projection.weight",
            "model.norm.weight", "model.embed_tokens.weight",
            *(f"model.layers.{i}.{leaf}" for i in range(model.cfg.num_layers) for leaf in (
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
                "self_attn.q_proj.weight", "self_attn.q_norm.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                "layer_scalar",
            ))
        }]
        # `masked_embedding.*` is intentionally skipped — see file docstring.
        if skipped:
            print(f"  skipped {len(skipped)} HF tensors (masked_embedding etc.):")
            for k in skipped:
                print(f"    {k}")

    model.load_state_dict(sd, strict=False)
    print(f"  loaded {len(matched)} tensors")
    return matched


# ---------------------------------------------------------------------------
# Main: load + smoke forward
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", type=str, default="google/gemma-4-E2B-it-assistant",
                    help="HF repo id or local path containing model.safetensors")
    ap.add_argument("--output", type=str, default="output/mtp_probe/mtp_drafter.pt")
    args = ap.parse_args()

    cfg = MtpDrafterConfig()
    model = MtpDrafterModel(cfg).float().eval()
    print(f"MtpDrafterModel params: {sum(p.numel() for p in model.parameters()):,}")

    load_from_hf(model, args.hf_repo)

    # Save checkpoint (state_dict only — RoPE tables are recomputed at load).
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v for k, v in model.state_dict().items()
          if not k.startswith(("swa_", "full_"))}
    torch.save(sd, args.output)
    print(f"Saved: {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")

    print("\nForward smoke test (zeros)...")
    with torch.no_grad():
        B = 1
        ctx_swa = 512
        ctx_full = 4096
        act = torch.randn(B, 1, cfg.input_size) * 0.02
        pos = torch.tensor([10], dtype=torch.int32)
        kv13_k = torch.zeros(B, 1, ctx_swa, cfg.swa_head_dim)
        kv13_v = torch.zeros(B, 1, cfg.swa_head_dim, ctx_swa)
        kv14_k = torch.zeros(B, 1, ctx_full, cfg.full_head_dim)
        kv14_v = torch.zeros(B, 1, cfg.full_head_dim, ctx_full)
        mask_swa = torch.zeros(B, 1, 1, ctx_swa)
        mask_swa[:, :, :, pos.item() + 1:] = -float("inf")
        mask_full = torch.zeros(B, 1, 1, ctx_full)
        mask_full[:, :, :, pos.item() + 1:] = -float("inf")

        logits, proj_act = model(act, pos, kv13_k, kv13_v, kv14_k, kv14_v,
                                 mask_swa, mask_full)
        print(f"  logits:   {tuple(logits.shape)} argmax={int(logits.argmax(-1).item())}"
              f" max={logits.max().item():.2f}")
        print(f"  proj_act: {tuple(proj_act.shape)} norm={proj_act.norm().item():.3f}")
        print("  Forward OK")


if __name__ == "__main__":
    main()
