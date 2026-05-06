#!/usr/bin/env python3
"""Convert Google's official Gemma 4 MTP drafter to ANE-targeted CoreML.

Source: HF `google/gemma-4-E2B-it-assistant` (BF16 safetensors, Apache 2.0).

Pipeline:
    1. Build PyTorch port (`mtp_drafter_model.MtpDrafterModel`)
    2. Load HF safetensors via `mtp_drafter_model.load_from_hf`
    3. Wrap into ANE-friendly module (Linear -> Conv2d(1,1), ANERMSNorm,
       in-model top-k, no internal RoPE precompute — cos/sin are inputs)
    4. coremltools convert to mlprogram (fp16) and optionally INT4 palettize

Architecture differences from the now-obsolete LiteRT W4A8 build:
  * Per-layer learned `layer_scalar` applied at the end of each layer
  * No final logit softcapping (`final_logit_softcapping=null`)
  * MLP names: gate_proj / up_proj / down_proj (Gemma 4 convention)
  * lm_head tied to embed_tokens (we ship the lm_head matrix as a buffer)
  * Masked-centroid logit path (`Gemma4AssistantMaskedEmbedder`) IS
    implemented when --centroid-lm-head is set: 2048-centroid cluster
    routing → top-32 clusters → 4096 candidate tokens → top-K argmax.
    The drafter network was trained against this restricted vocab path,
    so full lm_head argmax can give different (untrained) top-1 tokens.

I/O contract (UNCHANGED from the previous build — Swift caller stays as-is):
  Inputs:
    embed_token  (1, 1, 1536) fp16
    proj_act     (1, 1, 1536) fp16     # last_hidden_state from prev step
                                        # (or target's last hidden for step 0)
    kv13_k       (1, 1, W, 256) fp16
    kv13_v       (1, 1, 256, W) fp16   # pre-transposed (head_dim, seq)
    kv14_k       (1, 1, C, 512) fp16
    kv14_v       (1, 1, 512, C) fp16   # pre-transposed
    cos_swa      (1, 128) fp16          # SWA RoPE cos (first half, theta=10k)
    sin_swa      (1, 128) fp16
    cos_full     (1, 256) fp16          # full RoPE cos (first half)
    sin_full     (1, 256) fp16
    mask_swa     (1, 1, 1, W) fp16
    mask_full    (1, 1, 1, C) fp16

  RoPE convention: standard mirror-duplicated cos/sin layout — `cos_full` of
  length head_dim is `cat([cos_first_half, cos_first_half], dim=-1)` — so the
  caller sends only the first half (length head_dim/2) and the model concats
  it back internally. Matches `reshapeRoPEForDrafter` in MtpSpeculativeEngine.

  Outputs:
    top_k_indices  (8,) int32
    top_k_values   (8,) fp16
    proj_act_out   (1, 1, 1536) fp16

Usage:
    python conversion/build_mtp_drafter.py \\
        --hf-repo google/gemma-4-E2B-it-assistant \\
        --output mtp_drafter.mlpackage \\
        --palettize-int4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _build_centroid_pass_pipeline():
    """Default pipeline minus the int16 demotion pass that truncates the
    drafter's token-id outputs (vocab=262144 doesn't fit in uint16/int16).
    """
    import coremltools as ct
    pipeline = ct.PassPipeline.DEFAULT
    pipeline.remove_passes({"common::add_int16_cast"})
    return pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ane_ops import MODEL_DTYPE, ANERMSNorm
from mtp_drafter_model import MtpDrafterModel, MtpDrafterConfig, load_from_hf


# ---------------------------------------------------------------------------
# ANE-friendly drafter
# ---------------------------------------------------------------------------

class MtpDrafterANE(nn.Module):
    """ANE-friendly version of the official Gemma 4 MTP drafter.

    All Linear layers are 1x1 Conv2d for ANE throughput. RMSNorm uses
    ANERMSNorm (cat([x,-x]) + LayerNorm + slice trick). RoPE cos/sin are
    inputs (no internal precompute) so the Swift caller can pass the
    target's existing RoPE tables verbatim.
    """

    def __init__(self, cfg: MtpDrafterConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        TH = cfg.target_hidden
        FFN = cfg.ffn_dim
        V = cfg.vocab_size
        eps = cfg.rms_eps

        # Pre-projection (decomposed concat: in_e(embed) + in_p(proj_act)).
        self.in_e = nn.Conv2d(TH, H, 1, bias=False, dtype=MODEL_DTYPE)
        self.in_p = nn.Conv2d(TH, H, 1, bias=False, dtype=MODEL_DTYPE)

        self.layers = nn.ModuleList()
        for i in range(cfg.num_layers):
            if cfg.layer_types[i] == "sliding_attention":
                self.layers.append(
                    MtpLayerANE(H, cfg.swa_num_heads, cfg.swa_head_dim, FFN, eps))
            else:
                self.layers.append(
                    MtpLayerANE(H, cfg.full_num_heads, cfg.full_head_dim, FFN, eps))

        self.final_norm = ANERMSNorm(H, eps)
        # post_projection: hidden(256) -> target_hidden(1536) for next step.
        self.post_projection = nn.Conv2d(H, TH, 1, bias=False, dtype=MODEL_DTYPE)

        # lm_head as a buffer (tied to base embed_tokens at load time).
        self.register_buffer("lm_head_weight",
                             torch.zeros(V, H, dtype=MODEL_DTYPE), persistent=False)

        # Optional centroid (MaskedEmbedder) head: drafter network was trained
        # against this restricted vocab path. Top-32 clusters of 128 tokens →
        # 4096 candidates → top-K. Set via --centroid-lm-head.
        self.centroid_lm_head = bool(getattr(cfg, "centroid_lm_head", False))
        self.num_centroids = 2048
        self.top_k_centroids = 32
        self.vocab_size_per_centroid = V // self.num_centroids   # = 128
        self.centroids = nn.Conv2d(H, self.num_centroids, 1, bias=False, dtype=MODEL_DTYPE)
        # token_ordering: int32 (matches token-id range). gather of an int32
        # buffer gives int32 output; downstream gather into lm_head_weight has
        # dim_size=vocab_size > 32767, which triggers coremltools' gather
        # guard and prevents int16 demotion of the indices.
        self.register_buffer("token_ordering",
                             torch.zeros(V, dtype=torch.int32), persistent=False)

    def forward(
        self,
        embed_token,    # (1, 1, TH)  BSH
        proj_act,       # (1, 1, TH)
        kv13_k,         # (1, 1, W, 256)
        kv13_v,         # (1, 1, 256, W)
        kv14_k,         # (1, 1, C, 512)
        kv14_v,         # (1, 1, 512, C)
        cos_swa,        # (1, 256)
        sin_swa,        # (1, 256)
        cos_full,       # (1, 512)
        sin_full,       # (1, 512)
        mask_swa,       # (1, 1, 1, W)
        mask_full,      # (1, 1, 1, C)
    ):
        e_nchw = embed_token.permute(0, 2, 1).unsqueeze(2)
        p_nchw = proj_act.permute(0, 2, 1).unsqueeze(2)
        x_nchw = self.in_e(e_nchw) + self.in_p(p_nchw)
        x = x_nchw.squeeze(2).permute(0, 2, 1)  # (1, 1, H) BSH

        for i, layer in enumerate(self.layers):
            if self.cfg.layer_types[i] == "sliding_attention":
                x = layer(x, kv13_k, kv13_v, cos_swa, sin_swa, mask_swa)
            else:
                x = layer(x, kv14_k, kv14_v, cos_full, sin_full, mask_full)

        h = self.final_norm(x)  # (1, 1, H) BSH

        if self.centroid_lm_head:
            # Centroid (MaskedEmbedder) path — matches HF Gemma4Assistant.
            # Token IDs flow as int32 throughout. The lm_head gather has
            # vocab > 32767, so coremltools' gather guard prevents int16
            # demotion of indices. The final gather of int32 token IDs at
            # topk positions returns int32, no precision loss.
            h_nchw_for_centroids = h.permute(0, 2, 1).unsqueeze(2)
            c_logits_nchw = self.centroids(h_nchw_for_centroids)
            c_logits = c_logits_nchw.squeeze(-1).squeeze(-1).squeeze(0)  # (2048,)
            _, top_clusters = torch.topk(c_logits.float(), self.top_k_centroids)
            ordering_2d = self.token_ordering.view(self.num_centroids,
                                                    self.vocab_size_per_centroid)
            selected_canonical = ordering_2d.index_select(0, top_clusters.to(torch.int64))
            selected_canonical_flat = selected_canonical.reshape(-1)  # (4096,) int32
            # Gather embeddings — lm_head_weight dim 0 = vocab_size > 32767 →
            # coremltools' gather guard keeps these indices int32.
            sel_emb = self.lm_head_weight.index_select(
                0, selected_canonical_flat.long())
            h_2d = h.reshape(1, -1)
            selected_logits = (h_2d.float() @ sel_emb.float().transpose(0, 1)).squeeze(0)
            # Top-K over the 4096-element selected vocab.
            top_k_vals_sel, top_k_pos_sel = torch.topk(selected_logits, k=8)
            # Final gather: int32 token IDs at the K positions.
            top_k_ids = selected_canonical_flat.index_select(
                0, top_k_pos_sel.to(torch.int64))
        else:
            # Full vocab argmax (legacy path).
            logits = F.linear(h.float(), self.lm_head_weight.float())
            logits = logits.squeeze(0).squeeze(0)  # (V,)
            top_k_vals_sel, top_k_ids = torch.topk(logits, k=8)

        h_nchw = h.permute(0, 2, 1).unsqueeze(2)
        proj_out_nchw = self.post_projection(h_nchw)
        proj_out = proj_out_nchw.squeeze(2).permute(0, 2, 1)

        return top_k_ids.to(torch.int32), top_k_vals_sel.to(MODEL_DTYPE), proj_out


class MtpLayerANE(nn.Module):
    """Drafter layer with sandwich norms + per-layer scalar (Gemma 4 spec)."""

    def __init__(self, H: int, nh: int, hd: int, ffn: int, eps: float):
        super().__init__()
        self.nh = nh
        self.hd = hd

        self.input_layernorm = ANERMSNorm(H, eps)
        self.post_attention_layernorm = ANERMSNorm(H, eps)
        self.pre_feedforward_layernorm = ANERMSNorm(H, eps)
        self.post_feedforward_layernorm = ANERMSNorm(H, eps)

        # Q-only attention.
        self.q_proj = nn.Conv2d(H, nh * hd, 1, bias=False, dtype=MODEL_DTYPE)
        self.q_norm = ANERMSNorm(hd, eps)
        self.o_proj = nn.Conv2d(nh * hd, H, 1, bias=False, dtype=MODEL_DTYPE)

        # GeGLU MLP.
        self.gate_proj = nn.Conv2d(H, ffn, 1, bias=False, dtype=MODEL_DTYPE)
        self.up_proj = nn.Conv2d(H, ffn, 1, bias=False, dtype=MODEL_DTYPE)
        self.down_proj = nn.Conv2d(ffn, H, 1, bias=False, dtype=MODEL_DTYPE)

        # Per-layer scalar applied at end of forward.
        self.register_buffer("layer_scalar",
                             torch.ones(1, dtype=MODEL_DTYPE), persistent=False)

    def forward(self, x, kv_k, kv_v, cos, sin, mask):
        # x: (1, 1, H) BSH; cos/sin: (1, hd/2) — caller-supplied first half.

        # === Attention (Q-only, reads target K/V) ===
        residual = x
        h = self.input_layernorm(x)

        h_nchw = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)  # (1, H, 1, 1)
        q = self.q_proj(h_nchw)                                  # (1, nh*hd, 1, 1)

        # Per-head Q-norm: (1, nh, 1, hd) → norm last dim.
        q = q.view(1, self.nh, self.hd, 1).permute(0, 1, 3, 2)
        q_flat = q.view(self.nh, 1, self.hd)
        q_normed = self.q_norm(q_flat)
        q = q_normed.view(1, self.nh, 1, self.hd)

        # Mirror-duplicate cos/sin to (1, hd) since `cos[:half] == cos[half:]`
        # for both standard and partial-rotary RoPE (partial just zeros out
        # high-index inv_freq, but the second half still mirrors the first).
        cos_full = torch.cat([cos, cos], dim=-1).view(1, 1, 1, self.hd)
        sin_full = torch.cat([sin, sin], dim=-1).view(1, 1, 1, self.hd)
        half = self.hd // 2
        q_front, q_back = q[..., :half], q[..., half:]
        rot = torch.cat([-q_back, q_front], dim=-1)  # rotate_half(q)
        q = (q * cos_full) + (rot * sin_full)
        q = q.to(MODEL_DTYPE)

        # Q @ K^T:  Q (1,nh,1,hd) @ K^T (1,1,hd,ctx) → (1,nh,1,ctx).
        k_t = kv_k.transpose(-2, -1).to(MODEL_DTYPE)
        attn = torch.matmul(q.float(), k_t.float())
        attn = attn + mask.float()
        attn = F.softmax(attn, dim=-1).to(MODEL_DTYPE)

        # Attn @ V^T: V is stored (1,1,hd,ctx); V^T (1,1,ctx,hd).
        v_t = kv_v.transpose(-2, -1).to(MODEL_DTYPE)
        out = torch.matmul(attn.float(), v_t.float()).to(MODEL_DTYPE)  # (1,nh,1,hd)

        out_nchw = out.reshape(1, self.nh * self.hd, 1, 1)
        h_attn_nchw = self.o_proj(out_nchw)
        h_attn = h_attn_nchw.squeeze(2).permute(0, 2, 1)
        h_attn = self.post_attention_layernorm(h_attn)
        x = residual + h_attn

        # === MLP (sandwich norm) ===
        residual = x
        h = self.pre_feedforward_layernorm(x)
        h_nchw = h.to(MODEL_DTYPE).permute(0, 2, 1).unsqueeze(2)
        g = self.gate_proj(h_nchw)
        u = self.up_proj(h_nchw)
        # Gemma 4 hidden_activation = gelu_pytorch_tanh.
        mlp_nchw = self.down_proj(F.gelu(g, approximate="tanh") * u)
        mlp_out = mlp_nchw.squeeze(2).permute(0, 2, 1)
        mlp_out = self.post_feedforward_layernorm(mlp_out)
        x = residual + mlp_out

        return x * self.layer_scalar


# ---------------------------------------------------------------------------
# Weight transfer: PyTorch reference -> ANE module
# ---------------------------------------------------------------------------

def _to_conv(w: torch.Tensor) -> torch.Tensor:
    """Linear weight (out, in) -> Conv2d weight (out, in, 1, 1)."""
    return w.unsqueeze(-1).unsqueeze(-1).to(MODEL_DTYPE)


def load_ane_weights(ane: MtpDrafterANE, ref: MtpDrafterModel):
    """Copy weights from reference model into the ANE-friendly module."""
    sd = ane.state_dict()
    rs = ref.state_dict()
    cfg = ane.cfg

    # pre_projection: split (256, 3072) into (256, 1536) embed half + proj_act half.
    W_pre = rs["pre_projection.weight"]
    sd["in_e.weight"] = _to_conv(W_pre[:, :cfg.target_hidden])
    sd["in_p.weight"] = _to_conv(W_pre[:, cfg.target_hidden:])

    for i in range(cfg.num_layers):
        pfx_a = f"layers.{i}"  # ANE
        pfx_r = f"layers.{i}"  # ref

        for nm in ("input_layernorm", "post_attention_layernorm",
                   "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            sd[f"{pfx_a}.{nm}.weight"] = rs[f"{pfx_r}.{nm}.weight"].to(MODEL_DTYPE)

        sd[f"{pfx_a}.q_proj.weight"] = _to_conv(rs[f"{pfx_r}.self_attn.q_proj.weight"])
        sd[f"{pfx_a}.q_norm.weight"] = rs[f"{pfx_r}.self_attn.q_norm.weight"].to(MODEL_DTYPE)
        sd[f"{pfx_a}.o_proj.weight"] = _to_conv(rs[f"{pfx_r}.self_attn.o_proj.weight"])

        sd[f"{pfx_a}.gate_proj.weight"] = _to_conv(rs[f"{pfx_r}.mlp.gate_proj.weight"])
        sd[f"{pfx_a}.up_proj.weight"] = _to_conv(rs[f"{pfx_r}.mlp.up_proj.weight"])
        sd[f"{pfx_a}.down_proj.weight"] = _to_conv(rs[f"{pfx_r}.mlp.down_proj.weight"])

        # layer_scalar is a 1-element buffer.
        ls = rs[f"{pfx_r}.layer_scalar"].to(MODEL_DTYPE)
        # Cast through state_dict to keep registration paths consistent.
        getattr(ane.layers[i], "layer_scalar").copy_(ls)

    # final_norm + post_projection.
    sd["final_norm.weight"] = rs["norm.weight"].to(MODEL_DTYPE)
    sd["post_projection.weight"] = _to_conv(rs["post_projection.weight"])

    # lm_head buffer.
    with torch.no_grad():
        ane.lm_head_weight.copy_(rs["lm_head.weight"].to(MODEL_DTYPE))
        # Centroid (MaskedEmbedder) buffers — only populated when the path is
        # enabled. Reference state_dict has them under masked_embedding.* IF
        # the HF safetensors were loaded; otherwise they remain zero (and the
        # build will fall back to full lm_head argmax).
        if "masked_embedding.centroids.weight" in rs:
            sd["centroids.weight"] = _to_conv(rs["masked_embedding.centroids.weight"])
            ane.token_ordering.copy_(
                rs["masked_embedding.token_ordering"].to(torch.int32))

    ane.load_state_dict(sd, strict=False)


# ---------------------------------------------------------------------------
# Main: load -> trace -> coremltools convert -> palettize -> save
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", default="google/gemma-4-E2B-it-assistant",
                    help="HF repo id or local dir containing model.safetensors")
    ap.add_argument("--ckpt", default=None,
                    help="Optional pre-built reference checkpoint "
                         "(from `python conversion/mtp_drafter_model.py`)")
    ap.add_argument("--output", default="mtp_drafter.mlpackage")
    ap.add_argument("--palettize-int4", action="store_true",
                    help="INT4 palettize (group_size=32) — recommended for ANE.")
    ap.add_argument("--sliding-window", type=int, default=512,
                    help="Sliding-window K/V length (matches target's kv13).")
    ap.add_argument("--context-length", type=int, default=8192,
                    help="Full-attention K/V length (matches target's kv14).")
    ap.add_argument("--centroid-lm-head", action="store_true",
                    help="Use the official MaskedEmbedder cluster-routed lm "
                         "head (2048 centroids → top-32 → 4096 candidates → "
                         "top-K). Drafter network was trained against this "
                         "restricted vocab; full lm_head argmax can give "
                         "untrained top-1 tokens.")
    args = ap.parse_args()

    cfg = MtpDrafterConfig()
    cfg.centroid_lm_head = args.centroid_lm_head
    W = args.sliding_window
    C = args.context_length
    H = cfg.hidden_size
    TH = cfg.target_hidden

    # 1. Build PyTorch reference and load HF weights.
    print("Building PyTorch reference + loading HF weights ...")
    ref = MtpDrafterModel(cfg).float().eval()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        ref.load_state_dict(sd, strict=False)
    else:
        load_from_hf(ref, args.hf_repo)

    # 2. Build ANE module and copy weights.
    print(f"Building ANE module (W={W} C={C}) ...")
    ane = MtpDrafterANE(cfg).to(MODEL_DTYPE).eval()
    # Keep token_ordering int32 — the model-level .to(MODEL_DTYPE=fp16) above
    # tries to downcast all params/buffers to fp16, but fp16 can't represent
    # token ids > 2048 exactly. .to() is a no-op on integer buffers in PyTorch
    # if MODEL_DTYPE is float, but reassert int32 for safety.
    if hasattr(ane, "token_ordering"):
        ane.token_ordering.data = ane.token_ordering.data.to(torch.int32)
    print(f"  ANE params: {sum(p.numel() for p in ane.parameters()):,}")
    load_ane_weights(ane, ref)

    # 3. Smoke forward.
    print("\nForward smoke (zeros) ...")
    with torch.no_grad():
        embed = torch.zeros(1, 1, TH, dtype=MODEL_DTYPE)
        proj = torch.zeros(1, 1, TH, dtype=MODEL_DTYPE)
        kv13_k = torch.zeros(1, 1, W, cfg.swa_head_dim, dtype=MODEL_DTYPE)
        kv13_v = torch.zeros(1, 1, cfg.swa_head_dim, W, dtype=MODEL_DTYPE)
        kv14_k = torch.zeros(1, 1, C, cfg.full_head_dim, dtype=MODEL_DTYPE)
        kv14_v = torch.zeros(1, 1, cfg.full_head_dim, C, dtype=MODEL_DTYPE)
        # Caller passes only the first half of the mirror-duplicated cos/sin.
        cos_s = torch.ones(1, cfg.swa_head_dim // 2, dtype=MODEL_DTYPE)
        sin_s = torch.zeros(1, cfg.swa_head_dim // 2, dtype=MODEL_DTYPE)
        cos_f = torch.ones(1, cfg.full_head_dim // 2, dtype=MODEL_DTYPE)
        sin_f = torch.zeros(1, cfg.full_head_dim // 2, dtype=MODEL_DTYPE)
        mask_s = torch.zeros(1, 1, 1, W, dtype=MODEL_DTYPE)
        mask_f = torch.zeros(1, 1, 1, C, dtype=MODEL_DTYPE)
        ids, vals, proj_out = ane(embed, proj, kv13_k, kv13_v, kv14_k, kv14_v,
                                   cos_s, sin_s, cos_f, sin_f, mask_s, mask_f)
    print(f"  top_k_indices {tuple(ids.shape)} {ids.dtype}")
    print(f"  top_k_values  {tuple(vals.shape)} {vals.dtype}")
    print(f"  proj_act_out  {tuple(proj_out.shape)} {proj_out.dtype}")

    # 4. coremltools convert.
    print("\nConverting to CoreML ...")
    import coremltools as ct
    fp16 = ct.converters.mil.mil.types.fp16

    traced = torch.jit.trace(ane, (
        embed, proj, kv13_k, kv13_v, kv14_k, kv14_v,
        cos_s, sin_s, cos_f, sin_f, mask_s, mask_f,
    ), strict=False)

    mlm = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="embed_token", shape=(1, 1, TH), dtype=fp16),
            ct.TensorType(name="proj_act",    shape=(1, 1, TH), dtype=fp16),
            ct.TensorType(name="kv13_k",      shape=(1, 1, W, cfg.swa_head_dim), dtype=fp16),
            ct.TensorType(name="kv13_v",      shape=(1, 1, cfg.swa_head_dim, W), dtype=fp16),
            ct.TensorType(name="kv14_k",      shape=(1, 1, C, cfg.full_head_dim), dtype=fp16),
            ct.TensorType(name="kv14_v",      shape=(1, 1, cfg.full_head_dim, C), dtype=fp16),
            ct.TensorType(name="cos_swa",     shape=(1, cfg.swa_head_dim // 2), dtype=fp16),
            ct.TensorType(name="sin_swa",     shape=(1, cfg.swa_head_dim // 2), dtype=fp16),
            ct.TensorType(name="cos_full",    shape=(1, cfg.full_head_dim // 2), dtype=fp16),
            ct.TensorType(name="sin_full",    shape=(1, cfg.full_head_dim // 2), dtype=fp16),
            ct.TensorType(name="mask_swa",    shape=(1, 1, 1, W), dtype=fp16),
            ct.TensorType(name="mask_full",   shape=(1, 1, 1, C), dtype=fp16),
        ],
        outputs=[
            ct.TensorType(name="top_k_indices", dtype=np.int32),
            ct.TensorType(name="top_k_values"),
            ct.TensorType(name="proj_act_out"),
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        # When the centroid path is active, drop the `add_int16_cast` pass —
        # it auto-demotes topk's output indices to uint16 (max 65535) which
        # truncates token-id outputs (vocab=262144). Keeping the rest of the
        # default pipeline so weights stay fp16 (ANE-eligible).
        pass_pipeline=(_build_centroid_pass_pipeline()
                       if cfg.centroid_lm_head else ct.PassPipeline.DEFAULT),
        minimum_deployment_target=ct.target.iOS18,
    )

    if args.palettize_int4:
        print("  Palettizing weights INT4 (group_size=32) ...")
        import coremltools.optimize.coreml as cto
        pal_cfg = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                mode="kmeans", nbits=4,
                granularity="per_grouped_channel", group_size=32,
            )
        )
        mlm = cto.palettize_weights(mlm, pal_cfg)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    mlm.save(args.output)
    size_mb = sum(f.stat().st_size for f in Path(args.output).rglob("*") if f.is_file()) / 1e6
    print(f"\nSaved: {args.output} ({size_mb:.1f} MB)")
    print("\nNext steps:")
    print(f"  1. Drop {args.output} into the Gemma 4 bundle directory on iPhone.")
    print("  2. Existing MtpSpeculativeEngine + MtpDraftSource pick it up via "
          "`mtp_drafter.mlpackage` / `mtp_drafter.mlmodelc` (see CoreMLLLM.swift).")
    print("  3. Bench burst with [SpecDbg]; verify rolling acceptance ≥ 0.50.")


if __name__ == "__main__":
    main()
