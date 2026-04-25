"""Qwen3-VL 2B stateful multifunction (decode T=1 + prefill T=N).

Produces multifunction mlpackages where one .mlpackage carries two
functions sharing the same ct.StateType (kv_cache_0):

  infer       — T=1, identical to build_qwen3_vl_2b_stateful_chunks
  prefill_bN  — T=N, batched prefill for TTFT reduction

Swift selects the function via MLModelConfiguration.functionName.
A single MLState created from `infer` is reused for both calls;
prefill writes T slots, infer writes 1 slot, both via slice_update.

Tier-0: only the body chunks. chunk_head stays single-function (one
final-norm + lm_head sized to T=1 — the head only runs on the last
prefill token, never on a T=N input).
"""
from pathlib import Path
import argparse
import os
import shutil
import sys
import tempfile
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
)

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import (
    MODEL_DTYPE, ANERMSNorm, Conv2dLinear, InModelArgmax,
    apply_rotary_pos_emb, repeat_kv_ane, ane_softmax,
)
from build_qwen3_vl_2b_stateful_chunks import (
    load_text_config, load_text_backbone, _conv_from_linear,
    _norm_from_hf, ANEHeadChunk, _audit_ane, palettize_pkg,
    convert_body_stateful, convert_head, export_embed_fp16,
    EMBED_BIN_NAME, HEAD_CHUNK_NAME, _body_boundaries,
    ANEStatefulBodyChunk,
    _replace_conv2dlinear_with_linear, _swap_layer_projections_to_linear,
    _project,
)
from build_qwen3_vl_2b_stateful_chunks import MAX_SEQ as DEFAULT_MAX_SEQ


class ANEStatefulPrefillLayer(nn.Module):
    """T-batched stateful decoder layer. Reads/writes kv_cache_0 via
    slice_update over T consecutive slots."""

    def __init__(self, cfg, hf_layer, max_seq, layer_idx_in_chunk, T,
                 use_linear: bool = False):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_heads_per_kv = self.num_heads // self.num_kv_heads
        self.hidden_size = cfg.hidden_size
        self.max_seq = max_seq
        self.T = T
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.k_idx = 2 * layer_idx_in_chunk
        self.v_idx = 2 * layer_idx_in_chunk + 1

        attn = hf_layer.self_attn
        self.q_proj = _conv_from_linear(attn.q_proj)
        self.k_proj = _conv_from_linear(attn.k_proj)
        self.v_proj = _conv_from_linear(attn.v_proj)
        self.o_proj = _conv_from_linear(attn.o_proj)
        self.q_norm = _norm_from_hf(attn.q_norm.weight, cfg.rms_norm_eps, self.head_dim)
        self.k_norm = _norm_from_hf(attn.k_norm.weight, cfg.rms_norm_eps, self.head_dim)
        self.input_layernorm = _norm_from_hf(
            hf_layer.input_layernorm.weight, cfg.rms_norm_eps, self.hidden_size)
        self.post_attn_layernorm = _norm_from_hf(
            hf_layer.post_attention_layernorm.weight, cfg.rms_norm_eps,
            self.hidden_size)

        gate_w = hf_layer.mlp.gate_proj.weight
        up_w = hf_layer.mlp.up_proj.weight
        intermediate = gate_w.shape[0]
        stacked = torch.cat([gate_w, up_w], dim=0)
        self.gate_up_proj = Conv2dLinear(
            gate_w.shape[1], 2 * intermediate, bias=False, dtype=MODEL_DTYPE)
        self.gate_up_proj.conv.weight.data = stacked.detach().to(MODEL_DTYPE) \
            .unsqueeze(-1).unsqueeze(-1)
        self.intermediate_size = intermediate
        self.down_proj = _conv_from_linear(hf_layer.mlp.down_proj)

        if use_linear:
            _swap_layer_projections_to_linear(self)

    def _norm_in_conv_form(self, x_conv, norm):
        # (1, hidden, 1, T) → (1, T, hidden) → norm → (1, T, hidden) → (1, hidden, 1, T)
        T = self.T
        x = x_conv.permute(0, 2, 3, 1).reshape(1, T, self.hidden_size)
        x = norm(x)
        return x.reshape(1, 1, T, self.hidden_size).permute(0, 3, 1, 2)

    def forward(self, hidden_conv, cos, sin, causal_mask, current_pos,
                kv_cache):
        """
        hidden_conv: (1, hidden, 1, T) fp16
        cos, sin:    (1, T, head_dim)  fp16
        causal_mask: (1, 1, T, max_seq) fp16
        current_pos: int32 (1,) — write position in the state (start of T window)
        kv_cache:    (2*L, HKV, max_seq, head_dim) fp16
        """
        residual = hidden_conv
        h_conv = self._norm_in_conv_form(hidden_conv, self.input_layernorm)

        q = _project(self.q_proj, h_conv)  # (1, H*D, 1, T)
        k = _project(self.k_proj, h_conv)  # (1, HKV*D, 1, T)
        v = _project(self.v_proj, h_conv)

        H, HKV, D, T = self.num_heads, self.num_kv_heads, self.head_dim, self.T
        # (1, H*D, 1, T) → (1, H, D, T) → (1, H, T, D)
        q = q.view(1, H, D, T).permute(0, 1, 3, 2)
        k = k.view(1, HKV, D, T).permute(0, 1, 3, 2)
        v = v.view(1, HKV, D, T).permute(0, 1, 3, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        # cos/sin (1, T, D) → (1, 1, T, D) for broadcast over H
        cos_b = cos.unsqueeze(1)
        sin_b = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos_b, sin_b)

        # slice_update T slots: kv_cache[idx:idx+1, :, pos:pos+T, :] = k
        # k shape (1, HKV, T, D), target slice has same shape after the
        # leading-dim slice keeps size 1.
        k_write = k.to(kv_cache.dtype)
        v_write = v.to(kv_cache.dtype)
        kv_cache[self.k_idx:self.k_idx + 1, :, current_pos:current_pos + T, :] = k_write
        kv_cache[self.v_idx:self.v_idx + 1, :, current_pos:current_pos + T, :] = v_write

        k_full = kv_cache[self.k_idx:self.k_idx + 1, :, :, :]
        v_full = kv_cache[self.v_idx:self.v_idx + 1, :, :, :]

        k_rep = repeat_kv_ane(k_full, self.num_heads_per_kv, HKV, self.max_seq, D)
        v_rep = repeat_kv_ane(v_full, self.num_heads_per_kv, HKV, self.max_seq, D)

        # (1, H, T, D) @ (1, H, D, MAX_SEQ) → (1, H, T, MAX_SEQ)
        scores = torch.matmul(q, k_rep.transpose(-1, -2)) * self.scale
        scores = scores + causal_mask
        attn = ane_softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)  # (1, H, T, D)

        # (1, H, T, D) → (1, H*D, 1, T)
        out = out.permute(0, 1, 3, 2).reshape(1, H * D, 1, T)
        attn_out_conv = _project(self.o_proj, out)
        hidden_conv = residual + attn_out_conv

        residual = hidden_conv
        h_conv = self._norm_in_conv_form(hidden_conv, self.post_attn_layernorm)
        gate_up = _project(self.gate_up_proj, h_conv)
        gate, up = torch.split(gate_up, self.intermediate_size, dim=1)
        mlp_out = _project(self.down_proj, F.silu(gate) * up)
        hidden_conv = residual + mlp_out
        return hidden_conv


class ANEStatefulPrefillBodyChunk(nn.Module):
    def __init__(self, cfg, hf_layers, start, end, max_seq, T,
                 use_linear: bool = False):
        super().__init__()
        self.start = start
        self.end = end
        self.hidden_size = cfg.hidden_size
        self.T = T
        layers_in_chunk = end - start
        self.layers = nn.ModuleList([
            ANEStatefulPrefillLayer(cfg, hf_layers[start + li],
                                     max_seq, li, T,
                                     use_linear=use_linear)
            for li in range(layers_in_chunk)
        ])
        self.register_buffer(
            "kv_cache_0",
            torch.zeros(2 * layers_in_chunk, cfg.num_key_value_heads,
                        max_seq, cfg.head_dim, dtype=MODEL_DTYPE),
        )

    def forward(self, hidden_in, cos, sin, causal_mask, current_pos):
        # (1, T, hidden) → Conv2d (1, hidden, 1, T)
        T = self.T
        h = hidden_in.reshape(1, T, 1, self.hidden_size).permute(0, 3, 2, 1)
        for layer in self.layers:
            h = layer(h, cos, sin, causal_mask, current_pos, self.kv_cache_0)
        # (1, hidden, 1, T) → (1, T, hidden)
        return h.permute(0, 3, 2, 1).reshape(1, T, self.hidden_size)


def convert_body_prefill(chunk, cfg, start_layer, end_layer, max_seq, T,
                          out_path: Path):
    print(f"\n--- convert PREFILL T={T} body layers [{start_layer}, {end_layer}) ---")
    head_dim = cfg.head_dim
    layers_in_chunk = end_layer - start_layer

    example = (
        torch.zeros(1, T, cfg.hidden_size, dtype=MODEL_DTYPE),     # hidden_in
        torch.zeros(1, T, head_dim, dtype=MODEL_DTYPE),             # cos
        torch.zeros(1, T, head_dim, dtype=MODEL_DTYPE),             # sin
        torch.zeros(1, 1, T, max_seq, dtype=MODEL_DTYPE),           # causal_mask
        torch.zeros(1, dtype=torch.int32),                          # current_pos
    )
    t0 = time.time()
    traced = torch.jit.trace(chunk, example, strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, T, cfg.hidden_size),
                      dtype=np.float16),
        ct.TensorType(name="cos", shape=(1, T, head_dim), dtype=np.float16),
        ct.TensorType(name="sin", shape=(1, T, head_dim), dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, T, max_seq),
                      dtype=np.float16),
        ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
    ]
    ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
    state_shape = (2 * layers_in_chunk, cfg.num_key_value_heads,
                    max_seq, cfg.head_dim)
    ct_states = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=state_shape, dtype=np.float16),
            name="kv_cache_0",
        )
    ]

    t0 = time.time()
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs, states=ct_states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"  converted in {time.time()-t0:.1f}s")
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*')
                  if f.is_file()) / 1e6
    print(f"  saved fp16 {out_path.name} ({size_mb:.0f} MB)")
    _audit_ane(out_path)


def merge_into_multifunction(decode_pkg: Path, prefill_pkg: Path,
                              out_path: Path, prefill_T: int):
    """Merge decode (T=1) + prefill (T=N) into a single multifunction
    mlpackage. Both share weights and the kv_cache_0 state."""
    print(f"\n--- merging into multifunction → {out_path.name} ---")
    desc = MultiFunctionDescriptor()
    desc.add_function(str(decode_pkg), src_function_name="main",
                       target_function_name="infer")
    desc.add_function(str(prefill_pkg), src_function_name="main",
                       target_function_name=f"prefill_b{prefill_T}")
    desc.default_function_name = "infer"
    if out_path.exists():
        shutil.rmtree(out_path)
    save_multifunction(desc, str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*')
                  if f.is_file()) / 1e6
    print(f"  multifunction size: {size_mb:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    ap.add_argument("--num-chunks", type=int, default=4)
    ap.add_argument("--prefill-T", type=int, default=8,
                    help="T tokens per prefill forward (8 was v1.4.0 ship)")
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--keep-fp16", action="store_true")
    ap.add_argument("--linear-projections", action="store_true",
                    help="Plan-3 (cml9 PR #2577) variant: replace every "
                         "Conv2dLinear (1×1 wrapper) with nn.Linear in "
                         "both decode and prefill chunks. Multifunction "
                         "weights are shared, so both functions adopt the "
                         "same projection variant.")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    chunks_dir = out_root / "qwen3_vl_2b_stateful_chunks"
    fp16_dir = out_root / "_fp16_intermediate"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    print("loading Qwen3-VL 2B text backbone (fp32)...")
    cfg = load_text_config()
    print(f"  cfg: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"num_kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim}")
    text_model, lm_head = load_text_backbone()

    use_linear = args.linear_projections
    if use_linear:
        print("  projections: nn.Linear (cml9 PR #2577 variant)")
    else:
        print("  projections: Conv2dLinear 1×1 wrapper (current default)")

    export_embed_fp16(text_model.embed_tokens.weight,
                       chunks_dir / EMBED_BIN_NAME)
    head_module = ANEHeadChunk(cfg, text_model, lm_head,
                                cfg.tie_word_embeddings,
                                use_linear=use_linear).eval().to(MODEL_DTYPE)

    boundaries = _body_boundaries(cfg.num_hidden_layers, args.num_chunks)
    print(f"  body boundaries: {boundaries}")

    for ci, (start, end) in enumerate(boundaries):
        # Decode (T=1) module
        decode_mod = ANEStatefulBodyChunk(cfg, text_model.layers, start, end,
                                           args.max_seq,
                                           use_linear=use_linear
                                           ).eval().to(MODEL_DTYPE)
        decode_fp16 = fp16_dir / f"chunk_{ci}_decode.mlpackage"
        convert_body_stateful(decode_mod, cfg, start, end, args.max_seq, decode_fp16)
        del decode_mod

        # Prefill (T=N) module
        prefill_mod = ANEStatefulPrefillBodyChunk(
            cfg, text_model.layers, start, end, args.max_seq, args.prefill_T,
            use_linear=use_linear,
        ).eval().to(MODEL_DTYPE)
        prefill_fp16 = fp16_dir / f"chunk_{ci}_prefill.mlpackage"
        convert_body_prefill(prefill_mod, cfg, start, end, args.max_seq,
                              args.prefill_T, prefill_fp16)
        del prefill_mod

        # Optionally palettize each variant; multifunction merges
        # weights so per-variant palettize is fine.
        if args.nbits != 0:
            decode_int = fp16_dir / f"chunk_{ci}_decode_int{args.nbits}.mlpackage"
            prefill_int = fp16_dir / f"chunk_{ci}_prefill_int{args.nbits}.mlpackage"
            palettize_pkg(decode_fp16, decode_int, args.nbits)
            palettize_pkg(prefill_fp16, prefill_int, args.nbits)
            merge_src_decode = decode_int
            merge_src_prefill = prefill_int
        else:
            merge_src_decode = decode_fp16
            merge_src_prefill = prefill_fp16

        # Merge into multifunction
        final_path = chunks_dir / f"chunk_{ci}.mlpackage"
        merge_into_multifunction(merge_src_decode, merge_src_prefill,
                                  final_path, args.prefill_T)
        _audit_ane(final_path)

    del text_model, lm_head

    # Head stays single-function (T=1 only)
    head_fp16 = fp16_dir / f"{HEAD_CHUNK_NAME}.mlpackage"
    final_head = chunks_dir / f"{HEAD_CHUNK_NAME}.mlpackage"
    convert_head(head_module, cfg, head_fp16)
    if args.nbits == 0:
        if final_head.exists():
            shutil.rmtree(final_head)
        shutil.move(str(head_fp16), str(final_head))
    else:
        palettize_pkg(head_fp16, final_head, args.nbits)

    if not args.keep_fp16:
        shutil.rmtree(fp16_dir, ignore_errors=True)

    print(f"\nshipping artifacts under {chunks_dir}")
    for p in sorted(chunks_dir.iterdir()):
        size = p.stat().st_size / 1e6 if p.is_file() else \
            sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e6
        print(f"  {p.name}: {size:.0f} MB")


if __name__ == "__main__":
    main()
