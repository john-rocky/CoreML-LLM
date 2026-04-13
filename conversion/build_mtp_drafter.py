#!/usr/bin/env python3
"""Convert Google's E2B MTP drafter from TFLite to ANE-friendly CoreML.

Steps 4.2-4.3 of docs/MTP_PATH_A_FINDINGS.md.

Input:  output/mtp_probe/section_10.tflite  (E2B MTP drafter, 44.3 MB)
Output: mtp_drafter.mlpackage  (ANE-targeted, optionally INT4 palettized)

Architecture (reverse-engineered from TFLite, confirmed in MTP_PATH_A_FINDINGS.md S2):
  4-layer cross-attention transformer that reads the target model's KV caches.
  No K/V projections --- all K/V come from ChunkedEngine's kv13 (SWA) and kv14 (full).

  mtp_pre_proj   Linear(3072 -> 256)     cat(hidden_state, proj_act_prev) -> drafter dim
  layer 0-2      SWA cross-attn (Q-only, target kv13, head_dim=256)
  layer 3        Full cross-attn (Q-only, target kv14, head_dim=512)
  final_norm  -> embedder.decode Linear(256 -> 262144) -> softcap -> argmax
  mtp_post_proj  Linear(256 -> 1536)     carry state for next MTP step

Usage:
    # Probe TFLite to dump tensor names/shapes (run first to verify weight mapping)
    python conversion/build_mtp_drafter.py --probe

    # Full conversion
    python conversion/build_mtp_drafter.py \\
        --output output/mtp_drafter/mtp_drafter.mlpackage \\
        --palettize-int4

    # Custom context lengths matching ChunkedEngine
    python conversion/build_mtp_drafter.py \\
        --swa-context 512 --full-context 8192 --palettize-int4

    # Dry-run: build model + load weights, skip CoreML (quick parity check)
    python conversion/build_mtp_drafter.py --dry-run

    # Override auto weight mapping with manual JSON
    python conversion/build_mtp_drafter.py --weight-map my_map.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ane_ops import (
    MODEL_DTYPE,
    ANERMSNorm,
    InModelArgmax,
    repeat_kv_ane,
    rotate_half,
    stable_attention,
)


# ---------------------------------------------------------------------------
# E2B MTP drafter configuration (from MTP_PATH_A_FINDINGS.md S2, confirmed)
# ---------------------------------------------------------------------------

E2B_CONFIG = {
    "target_hidden": 1536,       # Gemma4 E2B hidden_size
    "drafter_hidden": 256,       # Internal drafter dim (mtp_pre_proj output)
    "num_heads": 4,              # Q heads in drafter (all layers)
    "num_kv_heads": 1,           # Target's KV heads (GQA)
    "swa_head_dim": 256,         # SWA K/V head dim (layers 0-2, target kv13)
    "full_head_dim": 512,        # Full-attn K/V head dim (layer 3, target kv14)
    "ffn_dim": 2048,             # GeGLU intermediate dim (all layers)
    "vocab_size": 262144,        # Shared with target
    "rms_eps": 1e-6,
    "swa_rope_theta": 10000.0,   # Sliding-window RoPE base
    "full_rope_theta": 1000000.0,  # Full-attention RoPE base
    "softcap_factor": 30.0,      # Logit softcapping (same as target)
}


# ---------------------------------------------------------------------------
# TFLite weight extraction
# ---------------------------------------------------------------------------

TFLITE_DTYPES = {
    0: np.float32, 1: np.float16, 2: np.int32,
    3: np.uint8, 6: np.bool_, 9: np.int8, 17: "int4",
}
TFLITE_TYPE_NAMES = {
    0: "f32", 1: "f16", 2: "i32", 3: "u8",
    6: "bool", 9: "i8", 17: "i4",
}


def _load_tflite_model(path: str):
    """Load TFLite FlatBuffer via the ``tflite`` pure-Python package."""
    try:
        import tflite
    except ImportError:
        raise ImportError(
            "The 'tflite' package is required for TFLite weight extraction.\n"
            "Install with:  pip install tflite"
        )
    with open(path, "rb") as f:
        buf = bytearray(f.read())
    return tflite.Model.GetRootAs(buf)


def extract_tflite_tensors(path: str) -> dict[str, np.ndarray]:
    """Extract all constant tensors from a TFLite file.

    INT8-quantized tensors are dequantized to float32 (per-tensor or
    per-channel, depending on the quantization metadata).
    """
    model = _load_tflite_model(path)
    sg = model.Subgraphs(0)
    out: dict[str, np.ndarray] = {}

    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        buf = model.Buffers(t.Buffer())
        if buf is None or buf.DataLength() == 0:
            continue  # activation placeholder, not a weight

        name = t.Name().decode("utf-8")
        shape = tuple(t.Shape(j) for j in range(t.ShapeLength()))
        dtype_key = TFLITE_DTYPES.get(t.Type(), np.float32)

        # INT4: two values packed per byte, needs unpacking
        if dtype_key == "int4":
            raw_bytes = buf.DataAsNumpy()
            packed = np.frombuffer(raw_bytes, dtype=np.uint8).copy()
            lo = (packed & 0x0F).astype(np.int8)
            hi = ((packed >> 4) & 0x0F).astype(np.int8)
            lo = np.where(lo >= 8, lo - 16, lo).astype(np.int8)
            hi = np.where(hi >= 8, hi - 16, hi).astype(np.int8)
            unpacked = np.empty(len(packed) * 2, dtype=np.int8)
            unpacked[0::2] = lo
            unpacked[1::2] = hi
            n_elements = 1
            for s in shape:
                n_elements *= s
            data = unpacked[:n_elements].reshape(shape)
            np_dtype = np.int8  # treat as int8 for dequantization
        else:
            np_dtype = dtype_key
            data = np.frombuffer(buf.DataAsNumpy(), dtype=np_dtype).copy().reshape(shape)

        # Dequantize INT8/INT4 weights
        quant = t.Quantization()
        if quant and quant.ScaleLength() > 0 and np_dtype == np.int8:
            scales = np.array([quant.Scale(j) for j in range(quant.ScaleLength())])
            zeros = np.array(
                [quant.ZeroPoint(j) for j in range(quant.ZeroPointLength())]
            )
            if quant.ScaleLength() == 1:
                # Per-tensor
                data = (data.astype(np.float32) - zeros[0]) * scales[0]
            else:
                # Per-channel
                axis = quant.QuantizedDimension()
                bcast = [1] * len(shape)
                bcast[axis] = -1
                data = (data.astype(np.float32) - zeros.reshape(bcast)) * scales.reshape(
                    bcast
                )

        out[name] = data.astype(np.float32)

    return out


def probe_tflite(path: str) -> None:
    """Print every tensor in the TFLite file for weight-mapping verification."""
    model = _load_tflite_model(path)

    # Signatures (I/O contract)
    for i in range(model.SignatureDefsLength()):
        sig = model.SignatureDefs(i)
        print(f"Signature: {sig.SignatureKey().decode()}")
        print("  Inputs:")
        for j in range(sig.InputsLength()):
            print(f"    {sig.Inputs(j).Name().decode()}")
        print("  Outputs:")
        for j in range(sig.OutputsLength()):
            print(f"    {sig.Outputs(j).Name().decode()}")

    sg = model.Subgraphs(0)
    print(f"\nTensors ({sg.TensorsLength()} total):\n")

    n_w = 0
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        name = t.Name().decode("utf-8")
        shape = tuple(t.Shape(j) for j in range(t.ShapeLength()))
        dtype_name = TFLITE_TYPE_NAMES.get(t.Type(), f"?{t.Type()}")
        buf = model.Buffers(t.Buffer())
        has_data = buf is not None and buf.DataLength() > 0
        sz = buf.DataLength() if has_data else 0

        quant_info = ""
        quant = t.Quantization()
        if quant and quant.ScaleLength() > 0:
            quant_info = (
                f"  Q(n={quant.ScaleLength()}, dim={quant.QuantizedDimension()})"
            )

        tag = "W" if has_data else "A"
        n_w += int(has_data)
        print(
            f"  [{tag}] {name:65s} {str(shape):25s} "
            f"{dtype_name:4s} {sz:>10,}B{quant_info}"
        )

    print(f"\n  Weight tensors: {n_w}")


# ---------------------------------------------------------------------------
# PyTorch model  (ANE-friendly, mirrors build_eagle3.py patterns)
# ---------------------------------------------------------------------------

class ANERMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale, ANE-friendly.

    Used for q_norm in SWA drafter layers (0-2) where Google's TFLite model
    has no learnable q_norm weight. Layer 3 (full attn) DOES have a weight.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(
            doubled, (2 * self.hidden_size,), eps=float(self.eps)
        )
        normed, _ = torch.chunk(normed, 2, dim=-1)
        return normed


class MtpDrafterLayerANE(nn.Module):
    """Single MTP drafter layer: Q-only cross-attention + GeGLU MLP.

    Follows Gemma4 sandwich-norm pattern:
      input_layernorm -> q_proj -> q_norm -> cross-attn(Q, target_K, target_V)
        -> o_proj -> post_attention_layernorm -> residual add
      pre_feedforward_layernorm -> GeGLU(gate, up, down)
        -> post_feedforward_layernorm -> residual add

    Layers 0-2 (SWA): q_norm has NO learnable weight (ANERMSNormNoScale).
    Layer 3 (full):    q_norm HAS a learnable weight (ANERMSNorm).
    This yields 4 norm weights/layer for SWA, 5 for full = 18 total + 1 final = 19.
    """

    def __init__(
        self,
        D: int,
        NH: int,
        NKV: int,
        HD: int,
        FFN: int,
        eps: float,
        has_q_norm_weight: bool = False,
    ):
        super().__init__()
        self.NH = NH
        self.NKV = NKV
        self.HD = HD
        self.gqa_rep = NH // NKV
        QD = NH * HD

        # Sandwich norms (4 per SWA layer, 5 per full-attn layer)
        self.input_layernorm = ANERMSNorm(D, eps)
        self.post_attention_layernorm = ANERMSNorm(D, eps)
        self.pre_feedforward_layernorm = ANERMSNorm(D, eps)
        self.post_feedforward_layernorm = ANERMSNorm(D, eps)

        # Q-only attention (Conv2d for ANE, 3x throughput over Linear)
        self.q_proj = nn.Conv2d(D, QD, 1, bias=False, dtype=MODEL_DTYPE)
        if has_q_norm_weight:
            self.q_norm = ANERMSNorm(HD, eps)
        else:
            self.q_norm = ANERMSNormNoScale(HD, eps)
        self.o_proj = nn.Conv2d(QD, D, 1, bias=False, dtype=MODEL_DTYPE)

        # GeGLU MLP (gelu(gate) * up -> down)
        self.gate_proj = nn.Conv2d(D, FFN, 1, bias=False, dtype=MODEL_DTYPE)
        self.up_proj = nn.Conv2d(D, FFN, 1, bias=False, dtype=MODEL_DTYPE)
        self.down_proj = nn.Conv2d(FFN, D, 1, bias=False, dtype=MODEL_DTYPE)

    def _to_conv(self, x: torch.Tensor) -> torch.Tensor:
        """(1,1,D) -> (1,D,1,1) for Conv2d."""
        return x.permute(0, 2, 1).unsqueeze(2)

    def _from_conv(self, x: torch.Tensor) -> torch.Tensor:
        """(1,D,1,1) -> (1,1,D) back to sequence format."""
        return x.squeeze(2).permute(0, 2, 1)

    def forward(
        self,
        x: torch.Tensor,       # (1, 1, D)
        cos: torch.Tensor,     # (1, 1, HD) RoPE cos at current position
        sin: torch.Tensor,     # (1, 1, HD) RoPE sin at current position
        mask: torch.Tensor,    # (1, 1, 1, CTX)  -inf=masked, 0=valid
        cache_k: torch.Tensor, # (1, NKV, CTX, HD) target K cache
        cache_v: torch.Tensor, # (1, NKV, CTX, HD) target V cache
        ctx_len: int,          # context length (explicit for repeat_kv_ane)
    ) -> torch.Tensor:
        residual = x

        # --- Attention ---
        h = self.input_layernorm(x)

        # Q projection: (1,1,D) -> Conv2d -> (1,1,QD) -> (1,NH,1,HD)
        q = self._from_conv(self.q_proj(self._to_conv(h)))
        q = q.view(1, 1, self.NH, self.HD)
        q = self.q_norm(q)                           # per-head along last dim
        q = q.permute(0, 2, 1, 3)                    # (1, NH, 1, HD)

        # RoPE on Q (target K in cache already has RoPE applied)
        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))

        # GQA-expand target K/V: (1,NKV,CTX,HD) -> (1,NH,CTX,HD)
        k = repeat_kv_ane(cache_k, self.gqa_rep, self.NKV, ctx_len, self.HD)
        v = repeat_kv_ane(cache_v, self.gqa_rep, self.NKV, ctx_len, self.HD)

        # Cross-attention: drafter Q vs target K/V
        # Gemma4 with qk_norm uses effective scale=1.0 --- norms control magnitude.
        # If parity fails, try head_dim**-0.5 (standard 1/sqrt(d)).
        a = stable_attention(q, k, v, 1.0, mask)     # (1, NH, 1, HD)

        # O projection
        a = a.permute(0, 2, 1, 3).reshape(1, 1, -1)  # (1, 1, QD)
        a = self._from_conv(self.o_proj(self._to_conv(a)))

        x = residual + self.post_attention_layernorm(a)

        # --- GeGLU MLP ---
        residual = x
        h = self.pre_feedforward_layernorm(x)
        hc = self._to_conv(h)                         # (1, D, 1, 1)
        ffn = self.down_proj(
            F.gelu(self.gate_proj(hc), approximate="tanh") * self.up_proj(hc)
        )
        x = residual + self.post_feedforward_layernorm(self._from_conv(ffn))

        return x


class MtpDrafterANE(nn.Module):
    """Google's MTP drafter for Gemma4 E2B --- ANE CoreML conversion target.

    4-layer cross-attention transformer that reads target's KV caches:
      Layers 0-2: SWA (kv13, head_dim=256, sliding_window context)
      Layer 3:    Full (kv14, head_dim=512, full context)

    Inputs:
      activations   (1, 1, 3072)            cat(target_hidden, proj_act_prev)
      position_ids  (1,)           int32     absolute decode position
      swa_mask      (1, 1, 1, SWA_CTX)      -inf=masked, 0=valid
      full_mask     (1, 1, 1, FULL_CTX)
      kv13_k/v      (1, 1, SWA_CTX, 256)    target's SWA KV cache
      kv14_k/v      (1, 1, FULL_CTX, 512)   target's full KV cache

    Outputs:
      token_id      (1,)   int32
      token_logit   (1,)   fp16
      projected_activations (1, 1, 1536) fp16   carry state for next MTP step
    """

    def __init__(self, cfg: dict, swa_ctx: int, full_ctx: int):
        super().__init__()
        D = cfg["drafter_hidden"]        # 256
        H_t = cfg["target_hidden"]       # 1536
        NH = cfg["num_heads"]            # 4
        NKV = cfg["num_kv_heads"]        # 1
        V = cfg["vocab_size"]            # 262144
        eps = cfg["rms_eps"]

        self.swa_ctx = swa_ctx
        self.full_ctx = full_ctx
        self.softcap = cfg["softcap_factor"]

        # Input projection: cat(hidden_1536, proj_act_1536) -> 256
        self.mtp_pre_proj = nn.Conv2d(
            2 * H_t, D, 1, bias=False, dtype=MODEL_DTYPE
        )

        # 4 transformer layers (3 SWA + 1 full)
        # Layers 0-2: SWA, no learnable q_norm weight
        # Layer 3: full attention, learnable q_norm weight (512-dim)
        self.layers = nn.ModuleList()
        for i in range(4):
            hd = cfg["full_head_dim"] if i == 3 else cfg["swa_head_dim"]
            self.layers.append(
                MtpDrafterLayerANE(
                    D, NH, NKV, hd, cfg["ffn_dim"], eps,
                    has_q_norm_weight=(i == 3),
                )
            )

        # Output
        self.final_norm = ANERMSNorm(D, eps)
        self.lm_head = nn.Conv2d(D, V, 1, bias=False, dtype=MODEL_DTYPE)
        self.mtp_post_proj = nn.Conv2d(D, H_t, 1, bias=False, dtype=MODEL_DTYPE)
        self.argmax = InModelArgmax()

        # Pre-computed RoPE tables (embedded as model buffers)
        max_len = max(swa_ctx, full_ctx) * 2
        for prefix, hd, theta in [
            ("swa", cfg["swa_head_dim"], cfg["swa_rope_theta"]),
            ("full", cfg["full_head_dim"], cfg["full_rope_theta"]),
        ]:
            inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
            freqs = torch.einsum(
                "i,j->ij", torch.arange(max_len).float(), inv_freq
            )
            emb = torch.cat((freqs, freqs), dim=-1)  # (max_len, hd)
            self.register_buffer(f"cos_{prefix}", emb.cos().to(MODEL_DTYPE))
            self.register_buffer(f"sin_{prefix}", emb.sin().to(MODEL_DTYPE))

    def forward(
        self,
        activations: torch.Tensor,    # (1, 1, 3072)
        position_ids: torch.Tensor,    # (1,)
        swa_mask: torch.Tensor,        # (1, 1, 1, SWA_CTX)
        full_mask: torch.Tensor,       # (1, 1, 1, FULL_CTX)
        kv13_k: torch.Tensor,          # (1, 1, SWA_CTX, 256)
        kv13_v: torch.Tensor,          # (1, 1, SWA_CTX, 256)
        kv14_k: torch.Tensor,          # (1, 1, FULL_CTX, 512)
        kv14_v: torch.Tensor,          # (1, 1, FULL_CTX, 512)
    ):
        # Input projection
        xc = activations.permute(0, 2, 1).unsqueeze(2)  # (1, 3072, 1, 1)
        xc = self.mtp_pre_proj(xc)                       # (1, 256, 1, 1)
        x = xc.squeeze(2).permute(0, 2, 1)               # (1, 1, 256)

        # RoPE lookups: (1, HD) -> (1, 1, HD) for broadcast
        cos_s = torch.index_select(self.cos_swa, 0, position_ids).unsqueeze(1)
        sin_s = torch.index_select(self.sin_swa, 0, position_ids).unsqueeze(1)
        cos_f = torch.index_select(self.cos_full, 0, position_ids).unsqueeze(1)
        sin_f = torch.index_select(self.sin_full, 0, position_ids).unsqueeze(1)

        # Layers 0-2: SWA cross-attention against target's kv13
        for i in range(3):
            x = self.layers[i](
                x, cos_s, sin_s, swa_mask, kv13_k, kv13_v, self.swa_ctx
            )

        # Layer 3: full cross-attention against target's kv14
        x = self.layers[3](
            x, cos_f, sin_f, full_mask, kv14_k, kv14_v, self.full_ctx
        )

        # Final norm
        h = self.final_norm(x)  # (1, 1, 256)

        # LM head + softcapping + argmax
        hc = h.permute(0, 2, 1).unsqueeze(2)             # (1, 256, 1, 1)
        logits = self.lm_head(hc).squeeze(2).permute(0, 2, 1)  # (1, 1, V)
        logits = torch.tanh(logits / self.softcap) * self.softcap
        token_id, token_logit = self.argmax(logits)

        # Projected activations (carry state for next MTP step)
        proj = self.mtp_post_proj(hc).squeeze(2).permute(0, 2, 1)  # (1,1,1536)

        return token_id, token_logit, proj


# ---------------------------------------------------------------------------
# Weight loading: TFLite -> PyTorch
# ---------------------------------------------------------------------------

# Explicit weight mapping derived from --probe output of section_10.tflite.
#
# FC weights have descriptive names from TFLite graph ops.
# Norm weights are jax2tf_arg_N (JAX tree traversal order, alphabetical):
#   - arg_0: final_norm (model-level, "final_norm" < "layers" alphabetically)
#   - Layers 0-2: 4 norms each (no learnable q_norm weight for SWA layers)
#     Order: post_attention_norm, post_ffw_norm, pre_attention_norm, pre_ffw_norm
#   - Layer 3: 5 norms (q_norm inside attn sub-module comes first: "attn" < "post")

_ARG = "jax2tf_arg_{}/ReadVariableOp;StatefulPartitionedCall"

# Per-layer norm order (alphabetical in Google's naming)
_NORM_ORDER_4 = [
    "post_attention_layernorm.weight",    # post_attention_norm
    "post_feedforward_layernorm.weight",  # post_ffw_norm
    "input_layernorm.weight",             # pre_attention_norm
    "pre_feedforward_layernorm.weight",   # pre_ffw_norm
]


def _build_explicit_map() -> dict[str, str]:
    """Build TFLite -> PyTorch weight name mapping for E2B MTP drafter."""
    m: dict[str, str] = {}

    # --- FC weights (descriptive TFLite names) ---
    m[
        "MtpDrafterModel.mtp_pre_project/mtp_pre_proj/"
        "btm,md->btd/dot_general"
    ] = "mtp_pre_proj.weight"

    for i in range(4):
        m[
            f"layer_{i}/layer_{i}.pre_q/attn.pre_q/attn._pre_attention_"
            f"query_fn/q_einsum/reshape;layer_{i}/layer_{i}.pre_q/attn."
            f"pre_q/attn._pre_attention_query_fn/q_einsum/btd,dH->btH/"
            f"dot_general"
        ] = f"layers.{i}.q_proj.weight"
        m[
            f"layer_{i}/layer_{i}.post_qkv/attn.post_qkv/attn_vec_einsum/"
            f"btH,Hd->btd/dot_general"
        ] = f"layers.{i}.o_proj.weight"
        m[
            f"layer_{i}/layer_{i}.post_qkv/mlp/gating_einsum1/"
            f"btd,df->btf/dot_general"
        ] = f"layers.{i}.gate_proj.weight"
        m[
            f"layer_{i}/layer_{i}.post_qkv/mlp/gating_einsum2/"
            f"btd,df->btf/dot_general"
        ] = f"layers.{i}.up_proj.weight"
        m[
            f"layer_{i}/layer_{i}.post_qkv/mlp/linear/"
            f"btf,fd->btd/dot_general"
        ] = f"layers.{i}.down_proj.weight"

    m[
        "MtpDrafterModel.decode_softmax/transformer.decode_softmax/"
        "embedder.decode/composite"
    ] = "lm_head.weight"
    m[
        "MtpDrafterModel.mtp_post_project/mtp_post_proj/"
        "btd,dm->btm/dot_general"
    ] = "mtp_post_proj.weight"

    # --- Norm weights (jax2tf_arg_N) ---
    m[_ARG.format(0)] = "final_norm.weight"

    # Layers 0-2: 4 sandwich norms each (no q_norm weight)
    swa_args = {0: [10, 11, 12, 13], 1: [21, 22, 23, 24], 2: [32, 33, 34, 35]}
    for layer_i, args in swa_args.items():
        for arg_n, suffix in zip(args, _NORM_ORDER_4):
            m[_ARG.format(arg_n)] = f"layers.{layer_i}.{suffix}"

    # Layer 3: q_norm first (inside attn, "attn" < "post"), then 4 sandwich norms
    m[_ARG.format(39)] = "layers.3.q_norm.weight"
    for arg_n, suffix in zip([43, 44, 45, 46], _NORM_ORDER_4):
        m[_ARG.format(arg_n)] = f"layers.3.{suffix}"

    return m


EXPLICIT_MAP = _build_explicit_map()


def _try_set_weight(
    pt_param: torch.Tensor,
    tfl_data: np.ndarray,
    tfl_name: str,
    pt_name: str,
) -> bool:
    """Copy TFLite tensor into PyTorch parameter.

    Handles Conv2d reshape (2D -> 4D) and transpose automatically.
    Returns True on success.
    """
    is_conv = pt_param.dim() == 4
    w = torch.from_numpy(tfl_data).to(MODEL_DTYPE)

    # Try direct, then transposed
    for _ in range(2):
        wc = w
        if is_conv and wc.dim() == 2:
            wc = wc.unsqueeze(-1).unsqueeze(-1)
        if wc.shape == pt_param.shape:
            with torch.no_grad():
                pt_param.copy_(wc)
            return True
        # Transpose 2D core and retry
        if w.dim() == 2:
            w = w.t().contiguous()
        else:
            break

    print(
        f"  SKIP {tfl_name} -> {pt_name}: "
        f"shape {tuple(torch.from_numpy(tfl_data).shape)} "
        f"vs {tuple(pt_param.shape)}"
    )
    return False


def load_tflite_into_model(
    model: MtpDrafterANE,
    tfl_tensors: dict[str, np.ndarray],
    weight_map_override: dict[str, str] | None = None,
) -> int:
    """Load TFLite weight tensors into PyTorch model.

    Uses EXPLICIT_MAP by default (derived from --probe analysis).
    If *weight_map_override* is given, it replaces the default map entirely.

    Returns number of successfully loaded parameters.
    """
    param_dict = dict(model.named_parameters())
    wmap = weight_map_override if weight_map_override else EXPLICIT_MAP

    count = 0
    for tfl_name, pt_name in sorted(wmap.items()):
        if tfl_name not in tfl_tensors:
            print(f"  MISS  {tfl_name}  (not in TFLite)")
            continue
        if pt_name not in param_dict:
            print(f"  MISS  {pt_name}  (not in model)")
            continue

        ok = _try_set_weight(
            param_dict[pt_name], tfl_tensors[tfl_name], tfl_name, pt_name
        )
        if ok:
            count += 1
            print(f"  OK  {pt_name}  {tuple(param_dict[pt_name].shape)}")

    # Report unmatched
    mapped_pt = set(wmap.values())
    mapped_tfl = set(wmap.keys())
    unmatched_pt = sorted(n for n in param_dict if n not in mapped_pt)
    unmatched_tfl = sorted(
        n for n in tfl_tensors
        if n not in mapped_tfl
        and tfl_tensors[n].ndim >= 1
        and tfl_tensors[n].size > 16  # skip tiny constants
    )

    if unmatched_pt:
        print(f"\n  Unloaded PyTorch params ({len(unmatched_pt)}):")
        for n in unmatched_pt:
            print(f"    {n}  {tuple(param_dict[n].shape)}")
    if unmatched_tfl:
        print(f"\n  Unmapped TFLite weights ({len(unmatched_tfl)}):")
        for n in unmatched_tfl:
            print(f"    {n}  {tfl_tensors[n].shape}")

    return count


# ---------------------------------------------------------------------------
# CoreML conversion
# ---------------------------------------------------------------------------

def convert_to_coreml(
    model: MtpDrafterANE,
    output_path: str,
    swa_ctx: int,
    full_ctx: int,
    palettize_int4: bool = False,
) -> None:
    """Trace PyTorch model and convert to CoreML .mlpackage."""
    import coremltools as ct

    cfg = E2B_CONFIG
    D = cfg["drafter_hidden"]
    H_t = cfg["target_hidden"]
    NKV = cfg["num_kv_heads"]
    swa_hd = cfg["swa_head_dim"]
    full_hd = cfg["full_head_dim"]

    # Fixed-shape sample inputs (batch=1, seq=1 for decode)
    sample = (
        torch.zeros(1, 1, 2 * H_t, dtype=MODEL_DTYPE),            # activations
        torch.zeros(1, dtype=torch.int32),                          # position_ids
        torch.zeros(1, 1, 1, swa_ctx, dtype=MODEL_DTYPE),         # swa_mask
        torch.zeros(1, 1, 1, full_ctx, dtype=MODEL_DTYPE),        # full_mask
        torch.zeros(1, NKV, swa_ctx, swa_hd, dtype=MODEL_DTYPE),  # kv13_k
        torch.zeros(1, NKV, swa_ctx, swa_hd, dtype=MODEL_DTYPE),  # kv13_v
        torch.zeros(1, NKV, full_ctx, full_hd, dtype=MODEL_DTYPE), # kv14_k
        torch.zeros(1, NKV, full_ctx, full_hd, dtype=MODEL_DTYPE), # kv14_v
    )

    # Sanity forward
    print("forward pass sanity check...")
    with torch.no_grad():
        tok, lg, proj = model(*sample)
    print(
        f"  token_id={tuple(tok.shape)} token_logit={tuple(lg.shape)} "
        f"projected_activations={tuple(proj.shape)}"
    )

    print("tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(model, sample, strict=False)

    print("converting to CoreML...")
    fp16_type = ct.converters.mil.mil.types.fp16
    mlm = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="activations",
                shape=(1, 1, 2 * H_t),
                dtype=fp16_type,
            ),
            ct.TensorType(
                name="position_ids", shape=(1,), dtype=np.int32
            ),
            ct.TensorType(
                name="swa_mask",
                shape=(1, 1, 1, swa_ctx),
                dtype=fp16_type,
            ),
            ct.TensorType(
                name="full_mask",
                shape=(1, 1, 1, full_ctx),
                dtype=fp16_type,
            ),
            ct.TensorType(
                name="kv13_k",
                shape=(1, NKV, swa_ctx, swa_hd),
                dtype=fp16_type,
            ),
            ct.TensorType(
                name="kv13_v",
                shape=(1, NKV, swa_ctx, swa_hd),
                dtype=fp16_type,
            ),
            ct.TensorType(
                name="kv14_k",
                shape=(1, NKV, full_ctx, full_hd),
                dtype=fp16_type,
            ),
            ct.TensorType(
                name="kv14_v",
                shape=(1, NKV, full_ctx, full_hd),
                dtype=fp16_type,
            ),
        ],
        outputs=[
            ct.TensorType(name="token_id"),
            ct.TensorType(name="token_logit"),
            ct.TensorType(name="projected_activations"),
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    if palettize_int4:
        print("  palettizing INT4 (group_size=32)...")
        import coremltools.optimize.coreml as cto

        opt_cfg = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                mode="kmeans",
                nbits=4,
                granularity="per_grouped_channel",
                group_size=32,
            )
        )
        mlm = cto.palettize_weights(mlm, opt_cfg)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    mlm.save(output_path)
    size_mb = (
        sum(f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file())
        / 1e6
    )
    print(f"  saved: {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_tflite = os.path.join(
        ROOT, "..", "output", "mtp_probe", "section_10.tflite"
    )
    default_output = os.path.join(
        ROOT, "..", "output", "mtp_drafter", "mtp_drafter.mlpackage"
    )

    ap = argparse.ArgumentParser(
        description="Convert Google's E2B MTP drafter (TFLite) to ANE CoreML"
    )
    ap.add_argument(
        "--tflite",
        type=str,
        default=default_tflite,
        help="Path to E2B MTP drafter .tflite (default: output/mtp_probe/section_10.tflite)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=default_output,
        help="Output .mlpackage path",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="Dump TFLite tensor info and exit (for weight-mapping verification)",
    )
    ap.add_argument(
        "--swa-context",
        type=int,
        default=512,
        help="Sliding-window context length (must match ChunkedEngine)",
    )
    ap.add_argument(
        "--full-context",
        type=int,
        default=8192,
        help="Full-attention context length (must match ChunkedEngine)",
    )
    ap.add_argument(
        "--palettize-int4",
        action="store_true",
        help="Apply INT4 palettization (group_size=32)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build model and load weights, skip CoreML conversion",
    )
    ap.add_argument(
        "--weight-map",
        type=str,
        default=None,
        help='JSON file mapping TFLite names -> PyTorch names (overrides auto-mapping)',
    )
    args = ap.parse_args()

    # --- Probe mode ---
    if args.probe:
        print(f"Probing: {args.tflite}\n")
        probe_tflite(args.tflite)
        print(
            "\nUse the tensor names above to verify or build a --weight-map JSON file."
        )
        return

    # --- Build model ---
    cfg = E2B_CONFIG
    print(
        f"MTP drafter config: D={cfg['drafter_hidden']} NH={cfg['num_heads']} "
        f"NKV={cfg['num_kv_heads']} swa_hd={cfg['swa_head_dim']} "
        f"full_hd={cfg['full_head_dim']} FFN={cfg['ffn_dim']} V={cfg['vocab_size']}"
    )
    print(f"Context: SWA={args.swa_context}  Full={args.full_context}")

    print("\nBuilding PyTorch model...")
    model = MtpDrafterANE(cfg, args.swa_context, args.full_context)
    model = model.to(MODEL_DTYPE).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters ({n_params * 2 / 1e6:.1f} MB at fp16)")

    # --- Load TFLite weights ---
    print(f"\nExtracting TFLite weights: {args.tflite}")
    tfl = extract_tflite_tensors(args.tflite)
    print(f"  {len(tfl)} weight tensors extracted")

    weight_map = None
    if args.weight_map:
        print(f"\nLoading weight map: {args.weight_map}")
        with open(args.weight_map) as f:
            weight_map = json.load(f)

    print("\nLoading weights into model...")
    n_loaded = load_tflite_into_model(model, tfl, weight_map)
    print(f"\n  {n_loaded} / {len(list(model.named_parameters()))} params loaded")

    # --- Forward sanity check ---
    print("\nForward pass sanity check...")
    with torch.no_grad():
        dummy = (
            torch.randn(1, 1, 2 * cfg["target_hidden"], dtype=MODEL_DTYPE),
            torch.tensor([42], dtype=torch.int32),
            torch.zeros(1, 1, 1, args.swa_context, dtype=MODEL_DTYPE),
            torch.zeros(1, 1, 1, args.full_context, dtype=MODEL_DTYPE),
            torch.randn(
                1, cfg["num_kv_heads"], args.swa_context, cfg["swa_head_dim"],
                dtype=MODEL_DTYPE,
            ),
            torch.randn(
                1, cfg["num_kv_heads"], args.swa_context, cfg["swa_head_dim"],
                dtype=MODEL_DTYPE,
            ),
            torch.randn(
                1, cfg["num_kv_heads"], args.full_context, cfg["full_head_dim"],
                dtype=MODEL_DTYPE,
            ),
            torch.randn(
                1, cfg["num_kv_heads"], args.full_context, cfg["full_head_dim"],
                dtype=MODEL_DTYPE,
            ),
        )
        tok, lg, proj = model(*dummy)
    print(
        f"  token_id={tok.item()} token_logit={lg.item():.4f} "
        f"proj_act shape={tuple(proj.shape)}"
    )

    if args.dry_run:
        print("\n--dry-run: skipping CoreML conversion")
        print(
            "\nNext: run --probe to verify weight mapping, "
            "then re-run without --dry-run."
        )
        return

    # --- CoreML conversion ---
    print("\n" + "=" * 60)
    convert_to_coreml(
        model,
        args.output,
        args.swa_context,
        args.full_context,
        palettize_int4=args.palettize_int4,
    )

    print("\nNext steps:")
    print(
        "  1. Validate parity: compare PyTorch drafter output vs TFLite interpreter"
    )
    print(
        "  2. If parity fails, check attention scale (1.0 vs head_dim**-0.5) "
        "and weight map"
    )
    print(
        "  3. Swift integration: MtpDraftTarget in ChunkedEngine "
        "(MTP_PATH_A_FINDINGS.md S4.4)"
    )


if __name__ == "__main__":
    main()
