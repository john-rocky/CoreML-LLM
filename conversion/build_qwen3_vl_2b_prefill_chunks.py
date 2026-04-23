"""Qwen3-VL 2B batched-prefill converter — T=32 tokens per forward.

Ports the Gemma-3 prefill-wrapper recipe (`gemma3_prefill_wrapper.py`)
to the Qwen3-VL 2B text backbone. One prefill forward consumes T prompt
tokens instead of 1, collapsing a 200-token prompt from ~200 sequential
decode steps to ~7 batched ones. ANE is memory-bandwidth-bound, so the
per-step cost barely grows with T.

Shipping bundle layout (under `qwen3_vl_2b_decode_chunks/`, alongside
the existing decode chunks):

    prefill_chunk_0..3.mlpackage     — T-batched body, same 7 layers
    prefill_chunk_0_vision.mlpackage — T-batched DeepStack-aware body
    (no prefill_chunk_head — Swift trims the tail by running the last
     few tokens through the T=1 decode path so we get `next_token`
     without needing a second head variant.)

KV-cache write under batching (same trick Gemma3PrefillWrapper uses):
  update_mask : (1, 1, max_seq, T) — column t has a single 1.0 at
                 position p+t. Zero elsewhere.
  K_new       : (1, kv_heads, T, hd)
  Want K_cache[:, :, p+t, :] = K_new[:, :, t, :]. Encoded as
      k_increment = matmul(update_mask, K_new) → (1, kv, max_seq, hd)
      write_any   = update_mask.sum(dim=-1)     → (1, 1, max_seq, 1)
      K_cache_new = K_cache * (1 - write_any) + k_increment
  ANE-safe: single matmul + two elementwise ops, no scatter_nd.

Attention mask per step is (1, 1, T, max_seq) — row t has -1e4 on any
past-slot column that has not yet been written *and* on any slot
strictly after p+t (future tokens in the same batch are also masked).

Hidden layout stays Conv2d-friendly `(B, hidden, 1, T)` throughout
each chunk, matching the decode recipe. Swift sees fp16 `(1, T, hidden)`
at chunk boundaries.

Usage:
  python build_qwen3_vl_2b_prefill_chunks.py --out-dir /tmp/qwen3_vl_2b
  # → /tmp/qwen3_vl_2b/qwen3_vl_2b_decode_chunks/prefill_chunk_*.mlpackage
"""
from pathlib import Path
import argparse
import shutil
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
)
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLTextConfig

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import (
    MODEL_DTYPE, ANERMSNorm, Conv2dLinear,
    apply_rotary_pos_emb, repeat_kv_ane, ane_softmax,
)

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MAX_SEQ = 2048
NUM_BODY_CHUNKS = 4
LAYERS_PER_CHUNK = 7
PREFILL_T = 32           # tokens per prefill forward
DEEPSTACK_LAYER_COUNT = 3  # DeepStack adds at text layers 0/1/2


def load_text_config() -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig.from_pretrained(MODEL_ID)


def load_text_backbone():
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    return full.model.language_model, full.lm_head


def _conv_from_linear(lin: nn.Linear) -> Conv2dLinear:
    out_features = lin.weight.shape[0]
    in_features = lin.weight.shape[1]
    c = Conv2dLinear(in_features, out_features,
                     bias=lin.bias is not None, dtype=MODEL_DTYPE)
    c.conv.weight.data = lin.weight.detach().to(MODEL_DTYPE) \
        .unsqueeze(-1).unsqueeze(-1)
    if lin.bias is not None:
        c.conv.bias.data = lin.bias.detach().to(MODEL_DTYPE)
    return c


def _norm_from_hf(weight: torch.Tensor, eps: float, hidden: int) -> ANERMSNorm:
    n = ANERMSNorm(hidden, eps=eps)
    n.weight.data = weight.detach().to(MODEL_DTYPE)
    return n


# ---- T-batched decoder layer ----------------------------------------------

class ANEPrefillLayer(nn.Module):
    """One decoder layer that takes T tokens per forward.

    Input hidden: `(1, hidden, 1, T)` Conv2d layout.
    KV cache:     `(1, num_kv_heads, max_seq, head_dim)`
    position_ids: `(T,)` fp32 — per-token absolute position
    cos / sin:    `(1, T, head_dim)` — per-token RoPE
    update_mask:  `(1, 1, max_seq, T)` — col-t one-hot at pos+t
    attn_mask:    `(1, 1, T, max_seq)` — -1e4 on "not yet written" slots
    """
    def __init__(self, cfg, hf_layer, max_seq, T):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_heads_per_kv = self.num_heads // self.num_kv_heads
        self.hidden_size = cfg.hidden_size
        self.max_seq = max_seq
        self.T = T
        self.scale = 1.0 / (self.head_dim ** 0.5)

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
            hf_layer.post_attention_layernorm.weight, cfg.rms_norm_eps, self.hidden_size)

        gate_w = hf_layer.mlp.gate_proj.weight
        up_w = hf_layer.mlp.up_proj.weight
        intermediate = gate_w.shape[0]
        stacked = torch.cat([gate_w, up_w], dim=0)
        self.gate_up_proj = Conv2dLinear(
            gate_w.shape[1], 2 * intermediate, bias=False, dtype=MODEL_DTYPE)
        self.gate_up_proj.conv.weight.data = \
            stacked.detach().to(MODEL_DTYPE).unsqueeze(-1).unsqueeze(-1)
        self.intermediate_size = intermediate
        self.down_proj = _conv_from_linear(hf_layer.mlp.down_proj)

    def _norm_batched(self, x_conv: torch.Tensor, norm: ANERMSNorm
                      ) -> torch.Tensor:
        # (1, hidden, 1, T) → (1, T, hidden) → norm → (1, hidden, 1, T)
        x = x_conv.permute(0, 2, 3, 1).reshape(1, self.T, self.hidden_size)
        x = norm(x)
        return x.reshape(1, self.T, 1, self.hidden_size).permute(0, 3, 2, 1)

    def forward(self, hidden_conv, cos, sin,
                update_mask, attn_mask,
                k_cache, v_cache):
        T = self.T
        H, HKV, D = self.num_heads, self.num_kv_heads, self.head_dim
        residual = hidden_conv

        # --- input_layernorm ---
        h_conv = self._norm_batched(hidden_conv, self.input_layernorm)

        # --- Q / K / V projections (Conv2d over the T axis) ---
        # Output: (1, H*D, 1, T)
        q = self.q_proj.forward_conv(h_conv)
        k = self.k_proj.forward_conv(h_conv)
        v = self.v_proj.forward_conv(h_conv)

        # → (1, H, T, D) for attention
        q = q.view(1, H, D, T).permute(0, 1, 3, 2)
        k = k.view(1, HKV, D, T).permute(0, 1, 3, 2)
        v = v.view(1, HKV, D, T).permute(0, 1, 3, 2)

        # Q/K norm + RoPE. cos/sin (1, T, D) → broadcast over heads.
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos_b = cos.unsqueeze(1)  # (1, 1, T, D)
        sin_b = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos_b, sin_b)

        # --- KV scatter-write via matmul-mask ---
        # update_mask: (1, 1, max_seq, T), k: (1, HKV, T, D)
        # matmul: (1, HKV, max_seq, T) @ (1, HKV, T, D) broadcast-ok
        # → (1, HKV, max_seq, D)
        k_inc = torch.matmul(update_mask, k)
        v_inc = torch.matmul(update_mask, v)
        write_any = update_mask.sum(dim=-1, keepdim=True)  # (1, 1, max_seq, 1)
        k_full = k_cache * (1.0 - write_any) + k_inc
        v_full = v_cache * (1.0 - write_any) + v_inc

        # GQA repeat
        k_rep = repeat_kv_ane(k_full, self.num_heads_per_kv, HKV, self.max_seq, D)
        v_rep = repeat_kv_ane(v_full, self.num_heads_per_kv, HKV, self.max_seq, D)

        # Attention: (1, H, T, D) @ (1, H, D, max_seq) → (1, H, T, max_seq)
        scores = torch.matmul(q, k_rep.transpose(-1, -2)) * self.scale
        scores = scores + attn_mask
        attn = ane_softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)  # (1, H, T, D)

        # Back to Conv2d layout: (1, H, T, D) → (1, H*D, 1, T)
        out = out.permute(0, 1, 3, 2).reshape(1, H * D, 1, T)
        attn_out_conv = self.o_proj.forward_conv(out)
        hidden_conv = residual + attn_out_conv

        # --- MLP ---
        residual = hidden_conv
        h_conv = self._norm_batched(hidden_conv, self.post_attn_layernorm)
        gate_up = self.gate_up_proj.forward_conv(h_conv)  # (1, 2*I, 1, T)
        gate, up = torch.split(gate_up, self.intermediate_size, dim=1)
        mlp_out = self.down_proj.forward_conv(F.silu(gate) * up)
        hidden_conv = residual + mlp_out

        return hidden_conv, k_full, v_full


class ANEPrefillBodyChunk(nn.Module):
    """Batched-T body chunk. I/O: Swift-visible `(1, T, hidden)` fp16."""
    def __init__(self, cfg, hf_layers, start, end, max_seq, T,
                 with_deepstack: bool = False):
        super().__init__()
        self.start = start
        self.end = end
        self.hidden_size = cfg.hidden_size
        self.T = T
        self.with_deepstack = with_deepstack
        self.layers = nn.ModuleList([
            ANEPrefillLayer(cfg, hf_layers[i], max_seq, T)
            for i in range(start, end)
        ])

    def forward(self, hidden_in, cos, sin, update_mask, attn_mask,
                 *extra):
        """
        hidden_in:  (1, T, hidden) fp16
        cos, sin:   (1, T, head_dim) fp16
        update_mask: (1, 1, max_seq, T) fp16
        attn_mask:   (1, 1, T, max_seq) fp16
        Without DeepStack: extra = (k_0, v_0, k_1, v_1, ...) fp16.
        With DeepStack:    extra = (ds_0, ds_1, ds_2, visual_active,
                                    k_0, v_0, k_1, v_1, ...)
        """
        h = hidden_in.reshape(1, self.T, 1, self.hidden_size).permute(0, 3, 2, 1)

        ds = None
        gate = None
        if self.with_deepstack:
            ds = [extra[0], extra[1], extra[2]]
            visual_active = extra[3]   # (T,) fp32
            # (T,) fp32 → (1, 1, 1, T) fp16 for conv-layout broadcast
            gate = visual_active.to(MODEL_DTYPE).view(1, 1, 1, self.T)
            kv_states = extra[4:]
        else:
            kv_states = extra

        new_states = []
        for local_i, layer in enumerate(self.layers):
            k = kv_states[2 * local_i]
            v = kv_states[2 * local_i + 1]
            h, k_new, v_new = layer(h, cos, sin, update_mask, attn_mask, k, v)
            new_states.append(k_new); new_states.append(v_new)
            if self.with_deepstack and local_i < DEEPSTACK_LAYER_COUNT:
                # ds[i]: (1, T, hidden) → (1, hidden, 1, T) conv layout
                ds_conv = ds[local_i].reshape(1, self.T, 1, self.hidden_size) \
                    .permute(0, 3, 2, 1)
                h = h + gate * ds_conv

        h_out = h.permute(0, 3, 2, 1).reshape(1, self.T, self.hidden_size)
        return (h_out, *new_states)


# ---- convert + palettize + audit ------------------------------------------

def _audit_ane(out_path: Path) -> float:
    reloaded = ct.models.MLModel(str(out_path),
                                  compute_units=ct.ComputeUnit.CPU_AND_NE)
    compiled = reloaded.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=str(compiled), compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
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
    cpu = dev.get("MLCPUComputeDevice", 0)
    gpu = dev.get("MLGPUComputeDevice", 0)
    pct = 100 * ane / compute if compute else 0.0
    print(f"    ANE placement: {ane}/{compute} ({pct:.1f}%) — CPU={cpu} GPU={gpu}")
    return pct


def _kv_shape(cfg, max_seq):
    return (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)


def convert_body(chunk, cfg, start_layer, end_layer, max_seq, T,
                  with_deepstack, out_path):
    tag = "chunk_0_vision" if with_deepstack else f"chunk[{start_layer},{end_layer})"
    print(f"\n--- convert prefill {tag} ---")
    head_dim = cfg.head_dim
    hidden = cfg.hidden_size

    example = [torch.zeros(1, T, hidden, dtype=MODEL_DTYPE)]          # hidden_in
    example.append(torch.zeros(1, T, head_dim, dtype=MODEL_DTYPE))    # cos
    example.append(torch.zeros(1, T, head_dim, dtype=MODEL_DTYPE))    # sin
    example.append(torch.zeros(1, 1, max_seq, T, dtype=MODEL_DTYPE))  # update_mask
    example.append(torch.zeros(1, 1, T, max_seq, dtype=MODEL_DTYPE))  # attn_mask
    if with_deepstack:
        example.append(torch.zeros(1, T, hidden, dtype=MODEL_DTYPE))  # ds_0
        example.append(torch.zeros(1, T, hidden, dtype=MODEL_DTYPE))  # ds_1
        example.append(torch.zeros(1, T, hidden, dtype=MODEL_DTYPE))  # ds_2
        example.append(torch.zeros(T, dtype=torch.float32))            # visual_active
    for _ in range(start_layer, end_layer):
        example.append(torch.zeros(*_kv_shape(cfg, max_seq), dtype=MODEL_DTYPE))
        example.append(torch.zeros(*_kv_shape(cfg, max_seq), dtype=MODEL_DTYPE))

    t0 = time.time()
    traced = torch.jit.trace(chunk, tuple(example), strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, T, hidden), dtype=np.float16),
        ct.TensorType(name="cos", shape=(1, T, head_dim), dtype=np.float16),
        ct.TensorType(name="sin", shape=(1, T, head_dim), dtype=np.float16),
        ct.TensorType(name="update_mask", shape=(1, 1, max_seq, T), dtype=np.float16),
        ct.TensorType(name="attn_mask", shape=(1, 1, T, max_seq), dtype=np.float16),
    ]
    if with_deepstack:
        ct_inputs += [
            ct.TensorType(name="ds_0", shape=(1, T, hidden), dtype=np.float16),
            ct.TensorType(name="ds_1", shape=(1, T, hidden), dtype=np.float16),
            ct.TensorType(name="ds_2", shape=(1, T, hidden), dtype=np.float16),
            ct.TensorType(name="visual_active", shape=(T,), dtype=np.float32),
        ]
    ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
    for i in range(start_layer, end_layer):
        ct_inputs.append(ct.TensorType(
            name=f"k_{i}", shape=_kv_shape(cfg, max_seq), dtype=np.float16))
        ct_inputs.append(ct.TensorType(
            name=f"v_{i}", shape=_kv_shape(cfg, max_seq), dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_k_{i}", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_v_{i}", dtype=np.float16))

    t0 = time.time()
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"  converted in {time.time()-t0:.1f}s")
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved fp16 {out_path.name} ({size_mb:.0f} MB)")
    _audit_ane(out_path)


def palettize_pkg(fp16_pkg: Path, out_pkg: Path, nbits: int):
    print(f"\n--- palettize INT{nbits}: {fp16_pkg.name} → {out_pkg.name} ---")
    m_in = ct.models.MLModel(str(fp16_pkg))
    op_cfg = OpPalettizerConfig(mode="kmeans", nbits=nbits,
                                  granularity="per_tensor")
    opt_cfg = OptimizationConfig(global_config=op_cfg)
    t0 = time.time()
    m_out = palettize_weights(m_in, opt_cfg)
    print(f"  palettize done in {time.time()-t0:.1f}s")
    m_out.save(str(out_pkg))
    src_mb = sum(f.stat().st_size for f in fp16_pkg.rglob('*') if f.is_file()) / 1e6
    dst_mb = sum(f.stat().st_size for f in out_pkg.rglob('*') if f.is_file()) / 1e6
    print(f"  bundle: {src_mb:.0f} MB (fp16) → {dst_mb:.0f} MB (int{nbits}) "
          f"[{100*dst_mb/src_mb:.1f}%]")
    _audit_ane(out_pkg)


def _body_boundaries(num_layers, num_chunks):
    assert num_layers % num_chunks == 0
    per = num_layers // num_chunks
    return [(i * per, (i + 1) * per) for i in range(num_chunks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--num-chunks", type=int, default=NUM_BODY_CHUNKS)
    ap.add_argument("--T", type=int, default=PREFILL_T,
                    help="tokens per prefill forward (default 32)")
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--keep-fp16", action="store_true")
    ap.add_argument("--skip-text", action="store_true",
                    help="only build prefill_chunk_0_vision")
    ap.add_argument("--only-chunk-0-vision", action="store_true",
                    help="alias for --skip-text")
    args = ap.parse_args()

    if args.only_chunk_0_vision:
        args.skip_text = True

    out_root = Path(args.out_dir).resolve()
    chunks_dir = out_root / "qwen3_vl_2b_decode_chunks"
    fp16_dir = out_root / "_fp16_prefill_intermediate"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading Qwen3-VL 2B text backbone (fp32)...")
    t0 = time.time()
    cfg = load_text_config()
    print(f"  text cfg: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"num_kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim}")
    text_model, _lm_head = load_text_backbone()
    print(f"  loaded in {time.time()-t0:.1f}s")

    boundaries = _body_boundaries(cfg.num_hidden_layers, args.num_chunks)
    print(f"  body boundaries: {boundaries}")

    T = args.T

    if not args.skip_text:
        for ci, (start, end) in enumerate(boundaries):
            name = f"prefill_chunk_{ci}"
            fp16_path = fp16_dir / f"{name}.mlpackage"
            final_path = chunks_dir / f"{name}.mlpackage"
            mod = ANEPrefillBodyChunk(
                cfg, text_model.layers, start, end, args.max_seq, T,
                with_deepstack=False).eval().to(MODEL_DTYPE)
            convert_body(mod, cfg, start, end, args.max_seq, T,
                          False, fp16_path)
            if args.nbits == 0:
                shutil.move(str(fp16_path), str(final_path))
            else:
                palettize_pkg(fp16_path, final_path, args.nbits)

    # prefill_chunk_0_vision: layers [0, LAYERS_PER_CHUNK) + DeepStack
    name = "prefill_chunk_0_vision"
    fp16_path = fp16_dir / f"{name}.mlpackage"
    final_path = chunks_dir / f"{name}.mlpackage"
    mod = ANEPrefillBodyChunk(
        cfg, text_model.layers, 0, LAYERS_PER_CHUNK, args.max_seq, T,
        with_deepstack=True).eval().to(MODEL_DTYPE)
    convert_body(mod, cfg, 0, LAYERS_PER_CHUNK, args.max_seq, T,
                  True, fp16_path)
    if args.nbits == 0:
        shutil.move(str(fp16_path), str(final_path))
    else:
        palettize_pkg(fp16_path, final_path, args.nbits)

    if not args.keep_fp16:
        shutil.rmtree(fp16_dir, ignore_errors=True)

    print(f"\n✓ shipping artifacts under {chunks_dir}")
    for p in sorted(chunks_dir.iterdir()):
        if not p.name.startswith("prefill_"):
            continue
        if p.is_file():
            size = p.stat().st_size / 1e6
        else:
            size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e6
        print(f"  {p.name}: {size:.0f} MB")


if __name__ == "__main__":
    main()
