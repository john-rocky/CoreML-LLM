"""Ternary-Bonsai-1.7B decode: 2-chunk INT8 palettized build for iPhone ANE.

Why chunks: monolithic INT8 is ~1.9 GB — same class as Qwen3.5-2B, which jetsam-killed
on iPhone (see `docs/QWEN35_2B_CHUNKED_HANDOFF.md`). Per-mlpackage ANE compile budget
is ~1.4 GB; splitting gets us under it and also stops the silent GPU fallback that
kills throughput even on Mac.

Split (28 layers total):
  chunk_a: input_ids → embed → layers [0..14)              → hidden_out (fp16)
  chunk_b: hidden_in → layers [14..28) + norm + lm_head     → token_id, token_logit

Each chunk ships its own `kv_cache` StateType for its 14 layers.

Uses the same attention / KV-write pattern as `exporter.py::MonolithicWrapper`
(mask-based cache update, per-channel `index_select` for RoPE, Conv2d Q/K/V/O,
QK-norm before RoPE). That wrapper is parity-verified against HF.

Output layout (Swift-friendly):
  <out-dir>/bonsai_1_7b_decode_chunks/
      chunk_a.mlpackage
      chunk_b.mlpackage
      model_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ane_ops import MODEL_DTYPE, apply_rotary_pos_emb
from models.qwen3 import Qwen3Model


# ---- Shared forward helpers (match MonolithicWrapper) ---------------------


def _decode_layer_step(
    layer,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    causal_mask: torch.Tensor,
    update_mask: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    n_rep: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single decoder-layer decode step with mask-based KV write.

    Returns: (hidden_out, K_new, V_new) where K_new/V_new are the (1, kv, ctx, d)
    caches with the current position written in.
    """
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
    q = (
        layer.self_attn.q_proj(x)
        .view(1, num_heads, head_dim, 1)
        .permute(0, 1, 3, 2)
        .to(MODEL_DTYPE)
    )
    k = (
        layer.self_attn.k_proj(x)
        .view(1, num_kv_heads, head_dim, 1)
        .permute(0, 1, 3, 2)
        .to(MODEL_DTYPE)
    )
    v = (
        layer.self_attn.v_proj(x)
        .view(1, num_kv_heads, head_dim, 1)
        .permute(0, 1, 3, 2)
        .to(MODEL_DTYPE)
    )

    # Qwen3 QK-norm (per-head RMSNorm) before RoPE
    if getattr(layer.self_attn, "has_qk_norm", False):
        q = layer.self_attn.q_norm(q)
        k = layer.self_attn.k_norm(k)

    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # Mask-based KV write — broadcast current (1,1,head_dim) to (1,ctx,head_dim)
    # and blend with cache at `update_mask`-selected position.
    k_broadcast = k.expand_as(K_cache)
    v_broadcast = v.expand_as(V_cache)
    K_new = K_cache * (1 - update_mask) + k_broadcast * update_mask
    V_new = V_cache * (1 - update_mask) + v_broadcast * update_mask

    K_expanded = K_new.repeat_interleave(n_rep, dim=1)
    V_expanded = V_new.repeat_interleave(n_rep, dim=1)

    q_f = q.to(torch.float32)
    k_f = K_expanded.to(torch.float32)
    attn_weights = torch.matmul(q_f, k_f.transpose(-1, -2)) * scale
    attn_weights = attn_weights + causal_mask.to(torch.float32)
    attn_weights = torch.softmax(attn_weights, dim=-1).to(MODEL_DTYPE)
    attn_output = torch.matmul(
        attn_weights.to(torch.float32), V_expanded.to(torch.float32)
    ).to(MODEL_DTYPE)

    attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(1, 1, -1)
    attn_output = (
        layer.self_attn.o_proj(attn_output.permute(0, 2, 1).unsqueeze(2))
        .squeeze(2)
        .permute(0, 2, 1)
    )
    hidden_states = residual + attn_output

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, K_new, V_new


# SWA uses the exact same `_decode_layer_step` as full attention. The only
# differences are external: state buffer is sized W (not ctx), and the host
# passes an `update_mask` with the 1.0 at `pos % W` (circular slot), plus a
# causal_mask sized (1,1,1,W). This keeps the ops identical to the ANE-proven
# non-SWA path (mask-based blend + standard matmul attention) and avoids the
# ANEC -14 compile rejection that `cat([K[:,:,1:,:], k])` hits.
#
# Attention order invariance: softmax+weighted sum is permutation-invariant
# over the keys, so the scrambled slot order in the circular buffer is fine.
# RoPE is baked into K at write time, so positional information is preserved.


class ChunkBase(nn.Module):
    """Base class holding shared precomputed RoPE + KV cache state."""

    def __init__(self, config, layer_indices: list[int],
                 sliding_window: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.num_chunk_layers = len(layer_indices)
        self.sliding_window = sliding_window

        # KV cache size:
        #   full attention → (..., ctx, head_dim)
        #   SWA (shift-based rotating buffer) → (..., W, head_dim)
        state_len = sliding_window if sliding_window is not None else config.context_length
        cache_shape = (
            2 * self.num_chunk_layers,
            config.num_key_value_heads,
            state_len,
            config.head_dim,
        )
        self.register_buffer("kv_cache", torch.zeros(cache_shape, dtype=MODEL_DTYPE))

        # RoPE cos/sin (shared identical buffer across chunks — small, ~4 MB total)
        head_dim = config.head_dim
        base = config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        max_len = config.context_length * 2
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(MODEL_DTYPE))
        self.register_buffer("sin_cached", emb.sin().to(MODEL_DTYPE))


class ChunkA(ChunkBase):
    """Head chunk: input_ids → embed → layers [0, split) → hidden_out.

    Inputs are the same whether full-attention or SWA — the only difference is
    the state buffer size (ctx vs W) and the semantics of update_mask (absolute
    position vs `pos % W` slot).
    """

    def __init__(self, full_model: Qwen3Model, split_at: int,
                 sliding_window: int | None = None) -> None:
        super().__init__(full_model.config, list(range(split_at)),
                         sliding_window=sliding_window)
        self.embed_tokens = full_model.embed_tokens
        self.layers = nn.ModuleList([full_model.layers[i] for i in range(split_at)])

    def forward(self, input_ids, position_ids, causal_mask, update_mask):
        cfg = self.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        n_rep = num_heads // num_kv_heads
        scale = 1.0 / (head_dim ** 0.5)
        num_layers = self.num_chunk_layers

        hidden_states = self.embed_tokens(input_ids).to(MODEL_DTYPE)

        cos = torch.index_select(self.cos_cached, 0, position_ids).view(1, 1, 1, head_dim)
        sin = torch.index_select(self.sin_cached, 0, position_ids).view(1, 1, 1, head_dim)

        for i in range(num_layers):
            layer = self.layers[i]
            k_idx = i
            v_idx = num_layers + i
            K_cache = self.kv_cache[k_idx].unsqueeze(0)
            V_cache = self.kv_cache[v_idx].unsqueeze(0)

            hidden_states, K_new, V_new = _decode_layer_step(
                layer, hidden_states, cos, sin, causal_mask, update_mask,
                K_cache, V_cache, num_heads, num_kv_heads, head_dim, n_rep, scale,
            )

            self.kv_cache[k_idx] = K_new.squeeze(0)
            self.kv_cache[v_idx] = V_new.squeeze(0)

        return hidden_states


class ChunkB(ChunkBase):
    """Tail chunk: hidden_in → layers [split, end) → norm → lm_head → (token, logit)."""

    def __init__(self, full_model: Qwen3Model, split_at: int,
                 sliding_window: int | None = None) -> None:
        cfg = full_model.config
        tail_indices = list(range(split_at, cfg.num_hidden_layers))
        super().__init__(cfg, tail_indices, sliding_window=sliding_window)
        self.layers = nn.ModuleList([full_model.layers[i] for i in tail_indices])
        self.norm = full_model.norm
        self.lm_head = full_model.lm_head
        self.argmax = full_model.argmax

    def forward(self, hidden_in, position_ids, causal_mask, update_mask):
        cfg = self.config
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        n_rep = num_heads // num_kv_heads
        scale = 1.0 / (head_dim ** 0.5)
        num_layers = self.num_chunk_layers

        hidden_states = hidden_in.to(MODEL_DTYPE)

        cos = torch.index_select(self.cos_cached, 0, position_ids).view(1, 1, 1, head_dim)
        sin = torch.index_select(self.sin_cached, 0, position_ids).view(1, 1, 1, head_dim)

        for i in range(num_layers):
            layer = self.layers[i]
            k_idx = i
            v_idx = num_layers + i
            K_cache = self.kv_cache[k_idx].unsqueeze(0)
            V_cache = self.kv_cache[v_idx].unsqueeze(0)

            hidden_states, K_new, V_new = _decode_layer_step(
                layer, hidden_states, cos, sin, causal_mask, update_mask,
                K_cache, V_cache, num_heads, num_kv_heads, head_dim, n_rep, scale,
            )

            self.kv_cache[k_idx] = K_new.squeeze(0)
            self.kv_cache[v_idx] = V_new.squeeze(0)

        hidden_states = self.norm(hidden_states)
        x = hidden_states.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
        logits = self.lm_head(x).squeeze(2).permute(0, 2, 1)
        return self.argmax(logits.squeeze(0))


# ---- ANE placement audit --------------------------------------------------


def audit_ane(pkg_path: Path) -> float:
    """Print per-op device placement and return ANE percentage.

    Returns -1.0 if audit fails (e.g. the save-time ANEC warning left the model
    without a cached compiled path). This is a diagnostic; never fatal.
    """
    try:
        reloaded = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        compiled = reloaded.get_compiled_model_path()
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    except Exception as e:
        print(f"    ANE audit skipped: {e}")
        return -1.0
    dev = Counter()
    for fn in plan.model_structure.program.functions.values():
        for op in fn.block.operations:
            a = plan.get_compute_device_usage_for_mlprogram_operation(op)
            d = ("const" if (a is None and op.operator_name == "const")
                 else (a.preferred_compute_device.__class__.__name__ if a else "unknown"))
            dev[d] += 1
    total = sum(dev.values())
    compute = total - dev.get("const", 0)
    ane = dev.get("MLNeuralEngineComputeDevice", 0)
    pct = 100.0 * ane / compute if compute else 0.0
    print(f"    ANE placement: {ane}/{compute} ({pct:.1f}%)  "
          f"dev breakdown={dict(dev)}")
    return pct


# ---- Conversion -----------------------------------------------------------


def convert_chunk(
    chunk: nn.Module,
    ctx: int,
    hidden_size: int,
    cache_shape: tuple,
    out_path: Path,
    *,
    is_head: bool,
    sliding_window: int | None = None,
) -> ct.models.MLModel:
    """Trace + convert one chunk to fp16 mlpackage.

    If sliding_window is set, the chunk expects an extra (W,) int32 `gather_idx`
    input and the causal_mask shape is (1,1,1,W) instead of (1,1,1,ctx).
    """
    label = "chunk_a (head)" if is_head else "chunk_b (tail)"
    w = sliding_window if sliding_window is not None else ctx
    print(f"\n--- {label} → {out_path.name} "
          f"(ctx={ctx}, window={w}{' SWA' if sliding_window else ''}) ---")

    if is_head:
        sample_input = torch.zeros((1, 1), dtype=torch.int32)
        input_spec = ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32)
    else:
        sample_input = torch.zeros((1, 1, hidden_size), dtype=torch.float16)
        input_spec = ct.TensorType(
            name="hidden_in", shape=(1, 1, hidden_size), dtype=np.float16
        )

    # State buffer / mask length:
    #   full attention → buffer = ctx, update_mask over ctx, causal over ctx
    #   SWA           → buffer = W,   update_mask over W,   causal over W
    # This keeps the op pattern identical to the proven non-SWA path. SWA just
    # uses a smaller rotating buffer with `pos % W` slot selection on the host.
    sample_position = torch.zeros((1,), dtype=torch.int32)
    sample_causal = torch.zeros((1, 1, 1, w), dtype=torch.float16)
    sample_update = torch.zeros((1, 1, w, 1), dtype=torch.float16)
    sample_update[0, 0, 0, 0] = 1.0

    sample_args = [sample_input, sample_position, sample_causal, sample_update]

    inputs = [
        input_spec,
        ct.TensorType(name="position_ids", shape=(1,), dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, w), dtype=np.float16),
        ct.TensorType(name="update_mask", shape=(1, 1, w, 1), dtype=np.float16),
    ]

    with torch.no_grad():
        chunk.kv_cache.zero_()

    print("  tracing...")
    t0 = time.time()
    with torch.no_grad():
        # strict=False: the module mutates `kv_cache` buffer, so JIT's trace-
        # validation re-run sees different state and complains. The graph itself
        # is correct (mask-based or shift-based write is functional); this matches
        # the pattern in `build_qwen35_2b_decode_chunks.py`.
        traced = torch.jit.trace(chunk, tuple(sample_args), strict=False)
    print(f"    traced in {time.time()-t0:.1f}s")

    if is_head:
        outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
    else:
        outputs = [
            ct.TensorType(name="token_id", dtype=np.int32),
            ct.TensorType(name="token_logit", dtype=np.float16),
        ]

    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=cache_shape, dtype=np.float16),
            name="kv_cache",
        ),
    ]

    print("  converting to CoreML...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        states=states,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"    converted in {time.time()-t0:.1f}s")

    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6
    print(f"  saved fp16 {out_path.name} ({size_mb:.0f} MB)")
    return mlmodel


def palettize(
    src: Path,
    dst: Path,
    nbits: int,
    mode: str = "kmeans",
    granularity: str = "per_tensor",
    group_size: int | None = None,
) -> None:
    """Palettize weights via Core ML OpPalettizerConfig.

    Useful combos:
      • kmeans / per_tensor / nbits=4       — default lossy approximation
      • unique / per_grouped_channel / nbits=2 / group_size=128
          bit-exact for Bonsai-style ternary: each 128-group has only 3
          distinct values so nbits=2 palette is lossless. No quality drop.
    """
    label = f"{mode}-{granularity}"
    if group_size is not None:
        label += f"-g{group_size}"
    print(f"\n--- palettize INT{nbits} ({label}): {src.name} → {dst.name} ---")
    m = ct.models.MLModel(str(src))
    kwargs: dict = dict(mode=mode, granularity=granularity)
    # `unique` mode derives nbits itself from the unique-value count; passing
    # nbits is explicitly rejected by OpPalettizerConfig.
    if mode != "unique":
        kwargs["nbits"] = nbits
    if granularity == "per_grouped_channel" and group_size is not None:
        kwargs["group_size"] = group_size
    op_cfg = OpPalettizerConfig(**kwargs)
    opt = OptimizationConfig(global_config=op_cfg)
    t0 = time.time()
    m = palettize_weights(m, opt)
    print(f"  palettize in {time.time()-t0:.1f}s")
    if dst.exists():
        shutil.rmtree(dst)
    m.save(str(dst))
    src_mb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e6
    dst_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print(f"  {src_mb:.0f} MB (fp16) → {dst_mb:.0f} MB (nbits={nbits} {label}) "
          f"[{100*dst_mb/src_mb:.1f}%]")
    audit_ane(dst)


# Back-compat alias for earlier scripts that imported palettize_kmeans.
def palettize_kmeans(src: Path, dst: Path, nbits: int) -> None:
    palettize(src, dst, nbits, mode="kmeans", granularity="per_tensor")


# ---- Main -----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="HF Bonsai model dir (config.json + safetensors)")
    ap.add_argument("--output", required=True, help="Output root dir")
    ap.add_argument("--context-length", type=int, default=2048)
    ap.add_argument("--split-at", type=int, default=14,
                    help="Split layers before this index (14 → layers 0-13 / 14-27)")
    ap.add_argument("--quantize",
                    choices=["fp16", "int8", "int4", "ternary"],
                    default="int8",
                    help="Quantization preset. 'ternary' = nbits=2 + mode=unique + "
                         "per_grouped_channel + group_size=128, which is bit-exact for "
                         "Bonsai's {-s, 0, +s} per-128-group weights (see "
                         "`verify_bonsai_ternary.py`). Lossless vs INT8/INT4 kmeans "
                         "approximations.")
    ap.add_argument("--nbits", type=int, default=None,
                    help="Override quantize nbits (1,2,3,4,6,8). Takes precedence over --quantize")
    ap.add_argument("--palette-mode", default=None,
                    choices=[None, "kmeans", "uniform", "unique"],
                    help="Override palettization mode. Default depends on --quantize.")
    ap.add_argument("--palette-granularity", default=None,
                    choices=[None, "per_tensor", "per_grouped_channel"],
                    help="Override granularity. Default depends on --quantize.")
    ap.add_argument("--palette-group-size", type=int, default=None,
                    help="Group size for per_grouped_channel. Default 128 for ternary.")
    ap.add_argument("--keep-fp16", action="store_true",
                    help="Keep the fp16 intermediates under _fp16_intermediate/ for re-palettizing")
    ap.add_argument("--sliding-window", type=int, default=None,
                    help="Enable SWA decode: state buffer = context-length but per-step "
                         "attention is over the last W slots selected via gather_idx input. "
                         "Expected to preserve ~ctx=W speed while allowing ctx=context-length "
                         "prefill. Host computes gather_idx + windowed causal mask per step. "
                         "Set to e.g. 1024 while --context-length 4096.")
    args = ap.parse_args()

    out_root = Path(args.output).resolve()
    out_dir = out_root / "bonsai_1_7b_decode_chunks"
    tmp_dir = out_root / "_fp16_intermediate"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Qwen3Model from {args.model_path}")
    t0 = time.time()
    model = Qwen3Model.from_pretrained(args.model_path, context_length=args.context_length)
    model.eval()
    cfg = model.config
    print(f"  loaded in {time.time()-t0:.1f}s: {cfg.num_hidden_layers} layers, "
          f"hidden={cfg.hidden_size}, heads={cfg.num_attention_heads}/"
          f"{cfg.num_key_value_heads}, head_dim={cfg.head_dim}, "
          f"vocab={cfg.vocab_size}, tie_embed={cfg.tie_word_embeddings}")

    assert 0 < args.split_at < cfg.num_hidden_layers, \
        f"split_at must be in (0, {cfg.num_hidden_layers}), got {args.split_at}"

    if args.sliding_window is not None:
        assert 0 < args.sliding_window <= args.context_length, \
            f"--sliding-window must be in (0, {args.context_length}], got {args.sliding_window}"
        print(f"  SWA: state_buffer=W={args.sliding_window} (circular, pos % W slot). "
              f"context_length={args.context_length} only bounds RoPE max position.")
    chunk_a = ChunkA(model, args.split_at, sliding_window=args.sliding_window).eval()
    chunk_b = ChunkB(model, args.split_at, sliding_window=args.sliding_window).eval()
    chunk_a_shape = tuple(chunk_a.kv_cache.shape)
    chunk_b_shape = tuple(chunk_b.kv_cache.shape)
    print(f"  chunk_a: embed + layers [0..{args.split_at})   state {chunk_a_shape}")
    print(f"  chunk_b: layers [{args.split_at}..{cfg.num_hidden_layers}) + head  "
          f"state {chunk_b_shape}")

    # Free full-model param refs we don't need for tracing
    # (chunks hold direct references to the relevant submodules already).
    del model

    fp16_a = tmp_dir / "chunk_a.mlpackage"
    fp16_b = tmp_dir / "chunk_b.mlpackage"
    final_a = out_dir / "chunk_a.mlpackage"
    final_b = out_dir / "chunk_b.mlpackage"

    convert_chunk(
        chunk_a, args.context_length, cfg.hidden_size, chunk_a_shape,
        fp16_a, is_head=True, sliding_window=args.sliding_window,
    )
    audit_ane(fp16_a)
    del chunk_a

    convert_chunk(
        chunk_b, args.context_length, cfg.hidden_size, chunk_b_shape,
        fp16_b, is_head=False, sliding_window=args.sliding_window,
    )
    audit_ane(fp16_b)
    del chunk_b

    if args.quantize == "fp16" and args.nbits is None:
        if final_a.exists():
            shutil.rmtree(final_a)
        if final_b.exists():
            shutil.rmtree(final_b)
        shutil.copytree(fp16_a, final_a)
        shutil.copytree(fp16_b, final_b)
    else:
        # Defaults per --quantize preset
        preset_nbits = {"int8": 8, "int4": 4, "ternary": 2}
        preset_mode = {"int8": "kmeans", "int4": "kmeans", "ternary": "unique"}
        preset_granularity = {
            "int8": "per_tensor",
            "int4": "per_tensor",
            "ternary": "per_grouped_channel",
        }
        preset_group_size = {"ternary": 128}

        nbits = args.nbits if args.nbits is not None else preset_nbits[args.quantize]
        mode = args.palette_mode if args.palette_mode else preset_mode[args.quantize]
        granularity = (args.palette_granularity
                       if args.palette_granularity
                       else preset_granularity[args.quantize])
        group_size = (args.palette_group_size
                      if args.palette_group_size is not None
                      else preset_group_size.get(args.quantize))

        palettize(fp16_a, final_a, nbits, mode=mode,
                  granularity=granularity, group_size=group_size)
        palettize(fp16_b, final_b, nbits, mode=mode,
                  granularity=granularity, group_size=group_size)

    # Manifest for Swift
    manifest = {
        "architecture": "qwen3",
        "model": "ternary-bonsai-1.7b",
        "split_at": args.split_at,
        "context_length": args.context_length,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
        "quantization": args.quantize if args.nbits is None else f"int{args.nbits}",
        "palette_mode": args.palette_mode or (
            {"int8": "kmeans", "int4": "kmeans", "ternary": "unique"}.get(args.quantize)
        ),
        "palette_granularity": args.palette_granularity or (
            {"int8": "per_tensor", "int4": "per_tensor",
             "ternary": "per_grouped_channel"}.get(args.quantize)
        ),
        "palette_group_size": (args.palette_group_size if args.palette_group_size
                               else (128 if args.quantize == "ternary" else None)),
        "sliding_window": args.sliding_window,
        "parts": {
            "chunk_a": "chunk_a.mlpackage",
            "chunk_b": "chunk_b.mlpackage",
        },
    }
    with open(out_dir / "model_config.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if not args.keep_fp16:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n✓ shipping artifacts under {out_dir}")
    for p in sorted(out_dir.iterdir()):
        size = (
            sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
            if p.is_dir() else p.stat().st_size / 1e6
        )
        print(f"  {p.name}: {size:.0f} MB")


if __name__ == "__main__":
    main()
