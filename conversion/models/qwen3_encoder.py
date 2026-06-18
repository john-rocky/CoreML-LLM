"""Qwen3 bidirectional encoder (ANE-optimized) for pplx-embed.

Perplexity's `pplx-embed-v1-0.6b` (plain) and `pplx-embed-context-v1-0.6b` (late
chunking) are a **bidirectional** Qwen3-0.6B encoder (`PPLXQwen3Model`, see the HF
checkpoint's `modeling.py`): every token attends to every non-pad token, the model
returns `last_hidden_state`, and a downstream pooling + tanh-int8 head produces the
embedding.

This is the ANE port — templated on `models/gemma3_encoder.py` but with Qwen3 math:
  - single pre/post RMSNorm per layer (pre-norm; NOT Gemma's 4 sandwich norms)
  - plain-weight RMSNorm (`x*rsqrt(..)*w`, no +1 gain — matches the working Qwen3 decoder)
  - per-head QK-norm (RMSNorm over head_dim on Q and K, before RoPE)
  - single RoPE table, θ=1e6 (NOT Gemma's dual local/global)
  - SwiGLU MLP (silu), GQA 16 q / 8 kv heads, head_dim 128 (q proj = 2048)
  - full bidirectional attention (pad-mask only, no causal triangle, no sliding window)
  - NO embedding scaling (that is a Gemma-ism)

ANE layout (docs/ANE_OPTIMIZATION_SURVEY.md + conversion/ane_ops.py): all projections
are Conv2d(1×1) on (B, C, 1, S); RMSNorm uses cat([x,−x])→LayerNorm; GQA expansion uses
repeat_kv_ane; the residual stream is kept in fp32 (fp16 can overflow over 28 layers).
Fixed trace-time sequence length — variable length is handled by padding to a bucket.
"""

from __future__ import annotations

import gc
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ane_ops import (  # noqa: E402
    MODEL_DTYPE,
    ANERMSNorm,
    apply_rotary_pos_emb,
    stable_attention,
)

# Max chunks per document for the context (late-chunking) variant.
N_MAX_CHUNKS = 32


def _repeat_kv_b(x: torch.Tensor, n_rep: int, B: int, num_kv_heads: int,
                 seq_len: int, head_dim: int) -> torch.Tensor:
    """Batched GQA expansion: (B, kv, S, D) → (B, kv*n_rep, S, D), explicit shapes."""
    if n_rep == 1:
        return x
    x = x.unsqueeze(2).expand(B, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(B, num_kv_heads * n_rep, seq_len, head_dim)


class Qwen3RMSNorm(nn.Module):
    """Native RMSNorm `x * rsqrt(mean(x²)+eps) * w`, computed in fp32 (HF Qwen3 parity).

    A *local* A/B alternative to the shared `ane_ops.ANERMSNorm` cat([x,−x])→LayerNorm
    trick. That trick was chosen years ago because the ANE had a highly-optimized
    LayerNorm kernel and no native `rsqrt`; on current M4 Max / macOS 26 / coremltools 9
    that may no longer hold (see docs/PPLX_EMBED_GPU_RESIDENCY.md). This class lets the
    pplx-embed encoder switch the 5 norm sites to native RMSNorm and measure.

    It stores a 1-D fp16 weight exactly like `ANERMSNorm`, so weight loading is
    unchanged (both are a plain `.weight` of shape `(hidden,)`). The normalization is
    done in fp32 (fp16 `x²` can overflow for large activations) and returned in the
    input dtype, mirroring the HF Qwen3 RMSNorm the fp32 reference already matches.
    NB: coremltools lowers the whole graph to fp16 at convert time, so the fp32 here is
    a trace-time/fidelity nicety; on device the op runs in fp16 like the rest.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=MODEL_DTYPE))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.to(in_dtype) * self.weight


def make_norm(norm_impl: str, hidden_size: int, eps: float) -> nn.Module:
    """Select the RMSNorm implementation for the encoder's 5 local norm sites.

    "ane_cat" (default) → shared `ANERMSNorm` (cat/chunk LayerNorm trick, unchanged
    behavior). "native" → local `Qwen3RMSNorm` (native rsqrt). Both store a 1-D fp16
    weight, so swapping does not affect weight loading.
    """
    if norm_impl == "native":
        return Qwen3RMSNorm(hidden_size, eps=eps)
    if norm_impl == "ane_cat":
        return ANERMSNorm(hidden_size, eps=eps)
    raise ValueError(f"Unknown norm_impl '{norm_impl}'; expected 'ane_cat' or 'native'.")


class Qwen3EncoderConfig:
    """Qwen3 encoder config (read from the pplx-embed HF config.json)."""

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 1024)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 28)
        self.num_attention_heads = kwargs.get("num_attention_heads", 16)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 8)
        self.head_dim = kwargs.get("head_dim", 128)
        self.intermediate_size = kwargs.get("intermediate_size", 3072)
        self.vocab_size = kwargs.get("vocab_size", 151936)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.attention_bias = bool(kwargs.get("attention_bias", False))
        # rope_theta may live at top level or under rope_parameters.
        rp = kwargs.get("rope_parameters") or {}
        self.rope_theta = float(kwargs.get("rope_theta", rp.get("rope_theta", 1_000_000.0)))
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        # Trace-time fixed sequence length (the bucket).
        self.max_seq_len = kwargs.get("max_seq_len", 4096)
        # RMSNorm implementation for the 5 local norm sites: "native" (local
        # Qwen3RMSNorm, native rsqrt — the shipped default) or "ane_cat" (shared
        # ANERMSNorm cat/chunk LayerNorm trick). native is the default because the A/B
        # (experiment_ane_rmsnorm.py / docs/PPLX_EMBED_GPU_RESIDENCY.md follow-up) found
        # it 12.7% (L=256) / 21.5% (L=512) faster on the ANE at identical 99.81%
        # residency and cosine 0.99998 vs the fp32 oracle, on M4 Max / macOS 26 /
        # coremltools 9. (The cat/chunk trick predates a native ANE rsqrt.)
        self.norm_impl = kwargs.get("norm_impl", "native")

    @classmethod
    def from_json(cls, path: str, max_seq_len: int = 4096,
                  norm_impl: str = "native") -> "Qwen3EncoderConfig":
        with open(path) as f:
            d = json.load(f)
        d = d.get("text_config", d)
        d["max_seq_len"] = max_seq_len
        d["norm_impl"] = norm_impl
        return cls(**d)


class Qwen3EncoderLayer(nn.Module):
    """One bidirectional Qwen3 block (pre-norm, ANE layout)."""

    def __init__(self, config: Qwen3EncoderConfig):
        super().__init__()
        hidden = config.hidden_size
        head_dim = config.head_dim
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        inter = config.intermediate_size
        eps = config.rms_norm_eps
        has_bias = config.attention_bias
        norm_impl = config.norm_impl

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        self.self_attn = nn.ModuleDict({
            "q_proj": nn.Conv2d(hidden, q_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "k_proj": nn.Conv2d(hidden, kv_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "v_proj": nn.Conv2d(hidden, kv_dim, 1, bias=has_bias, dtype=MODEL_DTYPE),
            "o_proj": nn.Conv2d(q_dim, hidden, 1, bias=False, dtype=MODEL_DTYPE),
            # Qwen3 QK-norm: per-head RMSNorm over head_dim, plain weight.
            "q_norm": make_norm(norm_impl, head_dim, eps),
            "k_norm": make_norm(norm_impl, head_dim, eps),
        })
        self.mlp = nn.ModuleDict({
            "gate_proj": nn.Conv2d(hidden, inter, 1, bias=False, dtype=MODEL_DTYPE),
            "up_proj": nn.Conv2d(hidden, inter, 1, bias=False, dtype=MODEL_DTYPE),
            "down_proj": nn.Conv2d(inter, hidden, 1, bias=False, dtype=MODEL_DTYPE),
        })
        self.input_layernorm = make_norm(norm_impl, hidden, eps)
        self.post_attention_layernorm = make_norm(norm_impl, hidden, eps)

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.n_rep = num_heads // num_kv_heads
        self.scale = float(head_dim) ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,   # (1, L, H) fp32
        cos: torch.Tensor,             # (1, 1, L, head_dim)
        sin: torch.Tensor,             # (1, 1, L, head_dim)
        attention_mask: torch.Tensor,  # (1, 1, L, L) fp16 additive (0 / −1e4)
        seq_len: int,
    ) -> torch.Tensor:
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        B = hidden_states.shape[0]

        residual = hidden_states
        # Normalize in fp32 then downcast: the pre-norm residual can exceed fp16
        # max (65504) over 28 layers, so casting *before* the norm would inf.
        # RMSNorm is scale-invariant; its output is O(1) and fp16-safe.
        normed = self.input_layernorm(hidden_states).to(MODEL_DTYPE)

        # (B, H, 1, L) layout for Conv2d.
        x = normed.permute(0, 2, 1).unsqueeze(2)

        # Q/K/V: (B, q_dim, 1, L) → (B, heads, L, head_dim).
        q = self.self_attn["q_proj"](x).view(B, num_heads, head_dim, seq_len).permute(0, 1, 3, 2)
        k = self.self_attn["k_proj"](x).view(B, num_kv_heads, head_dim, seq_len).permute(0, 1, 3, 2)
        v = self.self_attn["v_proj"](x).view(B, num_kv_heads, head_dim, seq_len).permute(0, 1, 3, 2)

        # QK-norm per head, then RoPE.
        q = self.self_attn["q_norm"](q.reshape(B, num_heads, seq_len, head_dim))
        k = self.self_attn["k_norm"](k.reshape(B, num_kv_heads, seq_len, head_dim))
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA expansion (ANE-safe, batched).
        k = _repeat_kv_b(k, self.n_rep, B, num_kv_heads, seq_len, head_dim)
        v = _repeat_kv_b(v, self.n_rep, B, num_kv_heads, seq_len, head_dim)

        # Bidirectional attention (fp32), pad-mask only. scale = 1/sqrt(head_dim).
        attn_out = stable_attention(q, k, v, self.scale, attention_mask)

        # (B, heads, L, head_dim) → (B, L, q_dim) → Conv2d o_proj.
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, seq_len, num_heads * head_dim)
        attn_out = self.self_attn["o_proj"](
            attn_out.permute(0, 2, 1).unsqueeze(2)
        ).squeeze(2).permute(0, 2, 1)

        # fp32 residual add (attn_out is fp16 → upcast).
        hidden_states = residual + attn_out.to(torch.float32)

        # MLP: post_attention_layernorm → SwiGLU → residual.
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states).to(MODEL_DTYPE)
        x_mlp = normed.permute(0, 2, 1).unsqueeze(2)
        gate = self.mlp["gate_proj"](x_mlp)
        up = self.mlp["up_proj"](x_mlp)
        mlp_out = self.mlp["down_proj"](F.silu(gate) * up).squeeze(2).permute(0, 2, 1)
        hidden_states = residual + mlp_out.to(torch.float32)

        return hidden_states


class Qwen3Encoder(nn.Module):
    """Bidirectional Qwen3 encoder backbone → last_hidden_state (no pooling).

    Input:
        input_ids      (1, L) int32
        attention_mask (1, L) fp16 — 1.0 for valid tokens, 0.0 for pad
    Output:
        hidden_states  (1, L, hidden_size) fp32
    """

    NEG_INF = -1.0e4  # ANE-safe additive-mask value

    def __init__(self, config: Qwen3EncoderConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = make_norm(config.norm_impl, config.hidden_size, eps=config.rms_norm_eps)
        self._build_rope(config)

    def _build_rope(self, config: Qwen3EncoderConfig):
        head_dim = config.head_dim
        # RoPE table size is DECOUPLED from the bucket (max_seq_len). We always build
        # it to a single fixed length (max_position_embeddings, 32768) so the baked
        # cos/sin constants are byte-identical across every bucket — that makes the
        # whole CoreML weight.bin identical across buckets, so HF LFS / on-disk store
        # it once instead of one ~1.19 GB blob per L. forward() slices [:S] at trace
        # time; a runtime position_ids `gather` keeps that slice from being const-
        # folded back into a per-bucket [S, head_dim] constant (verified by sha256).
        L = config.max_position_embeddings
        t = torch.arange(L).float()
        inv = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.einsum("i,j->ij", t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(MODEL_DTYPE))
        self.register_buffer("sin_cached", emb.sin().to(MODEL_DTYPE))

    def _pad_mask(self, attention_mask: torch.Tensor, S: int) -> torch.Tensor:
        """(B, S) {1 valid, 0 pad} → (B, 1, S, S) additive fp16 (0 / −1e4), key-side."""
        B = attention_mask.shape[0]
        key_pad = (1.0 - attention_mask).to(MODEL_DTYPE) * self.NEG_INF
        return key_pad.view(B, 1, 1, S).expand(B, 1, S, S)

    def forward(
        self,
        input_ids: torch.Tensor,       # (1, S) int32
        attention_mask: torch.Tensor,  # (1, S) fp16
    ) -> torch.Tensor:
        # Derive the sequence length from the input, not config — this makes the
        # same graph serve both fixed buckets (S == bucket, static) and a flexible
        # RangeDim export (S dynamic, GPU).
        head_dim = self.config.head_dim
        S = input_ids.shape[1]

        # No embedding scaling (Qwen3). Keep residual stream in fp32.
        hidden = self.embed_tokens(input_ids).to(torch.float32)

        # RoPE: the cos/sin tables are built once to a FIXED length
        # (max_position_embeddings, 32768) so they are byte-identical across every
        # bucket — that makes the whole CoreML weight.bin identical across buckets.
        # A plain static `cos_cached[:S]` slice would be const-folded back into a
        # per-bucket [S, head_dim] constant (verified: it defeats the dedup). To keep
        # the slice fold-proof we GATHER rows [0..S-1] using position_ids derived from
        # a runtime input (attention_mask), so the indices are runtime-dependent and
        # coremltools cannot const-fold the gather. This needs NO new model input.
        position_ids = (
            torch.cumsum(torch.ones_like(attention_mask, dtype=torch.float32), dim=1) - 1.0
        ).to(torch.int32)                                  # (1, S) = [[0,1,…,S-1]]
        pos = position_ids[0]                              # (S,)
        cos = self.cos_cached.index_select(0, pos).view(1, 1, S, head_dim)
        sin = self.sin_cached.index_select(0, pos).view(1, 1, S, head_dim)
        mask = self._pad_mask(attention_mask, S)

        for layer in self.layers:
            hidden = layer(hidden, cos, sin, mask, S)

        return self.norm(hidden).to(MODEL_DTYPE)


class PplxEmbedModel(nn.Module):
    """Full pplx-embed plain forward: tokens → pooled embedding.

    output_mode:
      "pooled_fp16" — masked-mean → fp16 (readable from the Python bridge; for
                      fidelity iteration; quantize to int8 downstream).
      "int8"        — masked-mean → tanh → clamp(round(·*127), −128, 127) → int8
                      (the deliverable; native int8 output, read via the Swift harness).
    """

    def __init__(self, config: Qwen3EncoderConfig, output_mode: str = "pooled_fp16"):
        super().__init__()
        assert output_mode in ("pooled_fp16", "int8")
        self.encoder = Qwen3Encoder(config)
        self.output_mode = output_mode

    def _masked_mean(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(torch.float32).unsqueeze(-1)      # (1, L, 1)
        summed = (hidden.to(torch.float32) * mask).sum(dim=1)      # (1, H)
        denom = mask.sum(dim=1).clamp_min(1.0)                     # (1, 1)
        return summed / denom

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask)           # (1, L, H)
        pooled = self._masked_mean(hidden, attention_mask)         # (1, H) fp32
        if self.output_mode == "pooled_fp16":
            return pooled.to(MODEL_DTYPE)
        # int8 tanh head (matches st_quantize.py: torch.round, qmin=−128).
        q = torch.clamp(torch.round(torch.tanh(pooled) * 127.0), -128, 127)
        return q.to(torch.int8)


class PplxEmbedContextModel(nn.Module):
    """Context (late-chunking) forward: encode the whole window once, pool per chunk.

    Inputs:
        input_ids      (1, L) int32
        attention_mask (1, L) fp16  — 1.0 valid, 0.0 pad
        pool_matrix    (N_max, L) fp16 — row k = normalized mean weights over chunk k's
                       token span (1/n_k on the span, else 0); unused rows are all-zero.
    Output:
        chunk_embeddings (N_max, 1024) — int8 or fp16. Unused rows → 0 vector (skip them).

    Pooling is a single matmul `pool_matrix @ hidden`, so the same encoder serves plain
    (one row = 1/n over all valid tokens) and context. See the pool_matrix lesson.
    """

    def __init__(self, config: Qwen3EncoderConfig, output_mode: str = "pooled_fp16",
                 n_max: int = N_MAX_CHUNKS):
        super().__init__()
        assert output_mode in ("pooled_fp16", "int8")
        self.encoder = Qwen3Encoder(config)
        self.output_mode = output_mode
        self.n_max = n_max

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                pool_matrix: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask)           # (1, L, H)
        h = hidden.squeeze(0).to(torch.float32)                    # (L, H)
        pooled = pool_matrix.to(torch.float32) @ h                 # (N_max, H)
        if self.output_mode == "pooled_fp16":
            return pooled.to(MODEL_DTYPE)
        q = torch.clamp(torch.round(torch.tanh(pooled) * 127.0), -128, 127)
        return q.to(torch.int8)


# --------------------------------------------------------------------------- #
# Weight loading.
# --------------------------------------------------------------------------- #
_CONV2D_SUFFIXES = (
    ".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight",
    ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight",
)


def _map_weight(hf_name: str) -> str | None:
    """Map a pplx-embed checkpoint key → local Qwen3Encoder param name.

    The checkpoint is sentence-transformer style (no `model.` prefix); accept both.
    """
    name = hf_name[len("model."):] if hf_name.startswith("model.") else hf_name
    if name == "embed_tokens.weight":
        return "embed_tokens.weight"
    if name == "norm.weight":
        return "norm.weight"
    if name == "lm_head.weight":
        return None  # encoder needs no LM head (tied embeddings)
    if name.startswith("layers."):
        return name  # local layout mirrors HF layer naming
    return None


def load_encoder_weights(encoder: Qwen3Encoder, hf_dir: str) -> None:
    """Load pplx-embed weights into a Qwen3Encoder (reshaping projections to Conv2d)."""
    import safetensors.torch

    st_files = sorted(f for f in os.listdir(hf_dir) if f.endswith(".safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors in {hf_dir}")

    loaded = 0
    seen: set[str] = set()
    for st_file in st_files:
        state = safetensors.torch.load_file(os.path.join(hf_dir, st_file))
        for hf_name, tensor in state.items():
            local = _map_weight(hf_name)
            if local is None:
                continue
            tensor = tensor.to(MODEL_DTYPE)
            if any(local.endswith(suf) for suf in _CONV2D_SUFFIXES) and tensor.dim() == 2:
                tensor = tensor.unsqueeze(-1).unsqueeze(-1)
            parts = local.split(".")
            target = encoder
            for p in parts[:-1]:
                target = getattr(target, p)
            param = getattr(target, parts[-1])
            if param.shape != tensor.shape:
                raise ValueError(f"Shape mismatch {hf_name}->{local}: {param.shape} vs {tensor.shape}")
            with torch.no_grad():
                param.copy_(tensor)
            loaded += 1
            seen.add(local)
        del state
        gc.collect()
    print(f"  loaded {loaded} tensors into Qwen3Encoder from {len(st_files)} file(s)")
    return None


def apply_fp16_residual_rescale(encoder: Qwen3Encoder, K: float) -> None:
    """Shrink the residual stream by 1/K so fp16 lowering doesn't overflow.

    This 28-layer encoder's activations exceed fp16 max (65504) in deep layers —
    specifically the `down_proj` accumulation (3072→1024) infs out around layer 19.
    coremltools lowers float ops to fp16, so this bites on-device.

    Because Qwen3 is **pre-norm** and every sublayer input goes through a
    scale-invariant RMSNorm (and so does the final `norm`), scaling
        embed_tokens, every o_proj, every down_proj   by 1/K
    makes every stored residual and every down_proj accumulation exactly K×
    smaller while leaving the pooled embedding mathematically unchanged
    (the scale-invariant final norm cancels the 1/K factor).
    """
    inv = 1.0 / float(K)
    with torch.no_grad():
        encoder.embed_tokens.weight.mul_(inv)
        for layer in encoder.layers:
            layer.self_attn["o_proj"].weight.mul_(inv)
            layer.mlp["down_proj"].weight.mul_(inv)


__all__ = [
    "Qwen3EncoderConfig", "Qwen3EncoderLayer", "Qwen3Encoder",
    "Qwen3RMSNorm", "make_norm",
    "PplxEmbedModel", "PplxEmbedContextModel", "N_MAX_CHUNKS",
    "load_encoder_weights", "apply_fp16_residual_rescale",
]
