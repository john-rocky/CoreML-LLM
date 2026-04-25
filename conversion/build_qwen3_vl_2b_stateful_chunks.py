"""Qwen3-VL 2B stateful decode converter — MLState + slice_update KV.

Phase 1 port of the ANEMLL recipe onto Qwen3-VL 2B's 28-layer text
backbone. This file is a fork of `build_qwen3_vl_2b_text_decode_chunks.py`
with two changes:

  1. **Unified KV state per chunk.** Each body chunk registers one
     `kv_cache_0` buffer of shape
     `(2 * layers_in_chunk, num_kv_heads, max_seq, head_dim)` (K and V
     interleaved). coremltools + `ct.StateType` lowers the Python
     slice-assign writes to `ios18.slice_update` inside the graph, so
     there is no per-step Swift↔ANE KV marshaling.
  2. **No `positions` broadcast mask.** The scatter-free
     `torch.where(positions.eq(pos), ...)` pattern that materialized
     `(1, HKV, max_seq, D)` per layer is replaced with
     `kv_cache[2*li:2*li+1, :, p:p+1, :] = k`. Swift builds the causal
     mask once per step and hands it in.

Chunk count stays at 4 (matches the 14-layer-per-chunk ANEMLL shipping
config is layer_count==2 — that lives behind --num-chunks=2 here).

Gate Zero (conversion/gate_zero_mlstate_stub.py) established the
(4, 8, 2048, 128) state shape compiles + runs clean on iPhone 17 Pro
ANE (4.1 ms, no -14, no MILCompilerForANE Error=(11)).
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
    MODEL_DTYPE, ANERMSNorm, Conv2dLinear, InModelArgmax,
    apply_rotary_pos_emb, repeat_kv_ane, ane_softmax,
)


MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MAX_SEQ = 2048
NUM_BODY_CHUNKS = 4  # 28 / 4 = 7 layers/chunk (first pass); --num-chunks=2 for 14/chunk.


def load_text_config() -> Qwen3VLTextConfig:
    return Qwen3VLTextConfig.from_pretrained(MODEL_ID)


def load_text_backbone():
    full = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    return full.model.language_model, full.lm_head


def _conv_from_linear(lin: nn.Linear) -> Conv2dLinear:
    c = Conv2dLinear(lin.in_features, lin.out_features,
                     bias=lin.bias is not None, dtype=MODEL_DTYPE)
    c.conv.weight.data = lin.weight.detach().to(MODEL_DTYPE) \
        .unsqueeze(-1).unsqueeze(-1)
    if lin.bias is not None:
        c.conv.bias.data = lin.bias.detach().to(MODEL_DTYPE)
    return c


def _norm_from_hf(weight: torch.Tensor, eps: float, hidden: int) -> ANERMSNorm:
    n = ANERMSNorm(hidden, eps=eps)
    n.weight.data = weight.detach().to(MODEL_DTYPE).clone()
    return n


# ============================================================
# Linear-vs-Conv2dLinear projection helpers (cml9 PR #2577 follow-up)
# ============================================================
#
# Default pipeline wraps every projection in `Conv2dLinear` (a 1×1
# nn.Conv2d) because pre-cml9 the activation-quant path only matched
# native conv ops. cml9's `linear_quantize_activations` recognises
# `linear` directly (PR #2577), so we can drop the wrapper. Mirrors the
# Gemma 4 Plan 3 migration in `models/gemma4_swa_stateful_chunks.py`.
#
# `--linear-projections` walks each chunk's layers and swaps every
# Conv2dLinear for a shape-equivalent nn.Linear; weights are reshaped
# from (out, in, 1, 1) back to (out, in). The forward path uses
# `_project` to dispatch on type so the same module works for both —
# the Linear branch skips the permute/unsqueeze that ANE has to undo at
# compile time.

def _replace_conv2dlinear_with_linear(module: nn.Module) -> nn.Module:
    """Return an nn.Linear equivalent to a Conv2dLinear (1×1 wrapper).
    Pass-through for any non-Conv2dLinear input."""
    if not isinstance(module, Conv2dLinear):
        return module
    has_bias = module.conv.bias is not None
    lin = nn.Linear(module.in_features, module.out_features,
                    bias=has_bias, dtype=module.conv.weight.dtype)
    with torch.no_grad():
        lin.weight.copy_(module.conv.weight.data.squeeze(-1).squeeze(-1))
        if has_bias:
            lin.bias.copy_(module.conv.bias.data)
    return lin


def _swap_layer_projections_to_linear(layer: nn.Module) -> None:
    """In-place: swap every Conv2dLinear projection on a single decoder
    layer (ANEStatefulDecoderLayer / ANEStatefulPrefillLayer) for an
    nn.Linear with reshaped weights."""
    for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_up_proj", "down_proj"):
        if hasattr(layer, name):
            setattr(layer, name,
                    _replace_conv2dlinear_with_linear(getattr(layer, name)))


def _project(proj: nn.Module, x_conv: torch.Tensor) -> torch.Tensor:
    """Run a (B, in, 1, T)-shaped Conv2d-layout tensor through either a
    Conv2dLinear (1×1) or nn.Linear projection and return (B, out, 1, T).
    Conv2dLinear stays in conv layout end-to-end; Linear path squeezes
    to (B, T, in), runs the matmul, and re-permutes back. Trace
    specializes on the isinstance branch so the active variant ends up
    in the MIL graph."""
    if isinstance(proj, nn.Linear):
        # (B, in, 1, T) → (B, 1, T, in) → squeeze(1) → (B, T, in)
        x_3d = x_conv.permute(0, 2, 3, 1).squeeze(1)
        out = proj(x_3d)  # (B, T, out)
        # (B, T, out) → (B, out, T) → (B, out, 1, T)
        return out.permute(0, 2, 1).unsqueeze(2)
    return proj.forward_conv(x_conv)


class ANEStatefulDecoderLayer(nn.Module):
    """One Qwen3-VL text decoder layer that reads/writes a unified KV
    cache buffer owned by the enclosing chunk. T=1 (decode step)."""

    def __init__(self, cfg, hf_layer, max_seq, layer_idx_in_chunk,
                 use_linear: bool = False):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.num_heads_per_kv = self.num_heads // self.num_kv_heads
        self.hidden_size = cfg.hidden_size
        self.max_seq = max_seq
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
        x = x_conv.permute(0, 2, 3, 1).reshape(1, 1, self.hidden_size)
        x = norm(x)
        return x.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)

    def forward(self, hidden_conv, cos, sin, causal_mask, current_pos,
                kv_cache):
        """
        hidden_conv: (1, hidden, 1, 1) fp16
        cos, sin:    (1, 1, head_dim)  fp16
        causal_mask: (1, 1, 1, max_seq) fp16 (-1e4 for future positions)
        current_pos: int32 (1,) — write position in the state
        kv_cache:    (2*L, HKV, max_seq, head_dim) fp16 — unified per-chunk
        Returns: (hidden_conv_out, kv_cache_out)
        """
        residual = hidden_conv
        h_conv = self._norm_in_conv_form(hidden_conv, self.input_layernorm)

        q = _project(self.q_proj, h_conv)
        k = _project(self.k_proj, h_conv)
        v = _project(self.v_proj, h_conv)

        H, HKV, D = self.num_heads, self.num_kv_heads, self.head_dim
        q = q.view(1, H, D, 1).permute(0, 1, 3, 2)
        k = k.view(1, HKV, D, 1).permute(0, 1, 3, 2)
        v = v.view(1, HKV, D, 1).permute(0, 1, 3, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        cos_b = cos.unsqueeze(1)
        sin_b = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos_b, sin_b)

        # slice-assign write: coremltools lowers to ios18.slice_update.
        # k/v shape (1, HKV, 1, D) → drop batch dim to (HKV, 1, D) for
        # the 3-D target slice `kv_cache[k_idx:k_idx+1, :, pos:pos+1, :]`
        # which is rank 4 (1, HKV, 1, D). The assignment matches after
        # k.squeeze(0) gives (HKV, 1, D) and kv_cache's leading dim has
        # size 1 after the slice.
        k_write = k.squeeze(0).to(kv_cache.dtype)  # (HKV, 1, D)
        v_write = v.squeeze(0).to(kv_cache.dtype)
        kv_cache[self.k_idx:self.k_idx + 1, :, current_pos:current_pos + 1, :] = \
            k_write.unsqueeze(0)
        kv_cache[self.v_idx:self.v_idx + 1, :, current_pos:current_pos + 1, :] = \
            v_write.unsqueeze(0)

        # Re-slice full layer K/V (after the write).
        k_full = kv_cache[self.k_idx:self.k_idx + 1, :, :, :]   # (1, HKV, MAX_SEQ, D)
        v_full = kv_cache[self.v_idx:self.v_idx + 1, :, :, :]

        k_rep = repeat_kv_ane(k_full, self.num_heads_per_kv, HKV, self.max_seq, D)
        v_rep = repeat_kv_ane(v_full, self.num_heads_per_kv, HKV, self.max_seq, D)

        scores = torch.matmul(q, k_rep.transpose(-1, -2)) * self.scale
        scores = scores + causal_mask
        attn = ane_softmax(scores, dim=-1)
        out = torch.matmul(attn, v_rep)  # (1, H, 1, D)

        out = out.permute(0, 1, 3, 2).reshape(1, H * D, 1, 1)
        attn_out_conv = _project(self.o_proj, out)
        hidden_conv = residual + attn_out_conv

        residual = hidden_conv
        h_conv = self._norm_in_conv_form(hidden_conv, self.post_attn_layernorm)
        gate_up = _project(self.gate_up_proj, h_conv)
        gate, up = torch.split(gate_up, self.intermediate_size, dim=1)
        mlp_out = _project(self.down_proj, F.silu(gate) * up)
        hidden_conv = residual + mlp_out
        return hidden_conv


class ANEStatefulBodyChunk(nn.Module):
    """Body chunk with unified kv_cache_0 buffer. One `forward` step
    consumes one token (T=1) and advances the state by one slot."""

    def __init__(self, cfg, hf_layers, start, end, max_seq,
                 use_linear: bool = False):
        super().__init__()
        self.start = start
        self.end = end
        self.hidden_size = cfg.hidden_size
        self.max_seq = max_seq
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

        layers_in_chunk = end - start
        self.layers = nn.ModuleList([
            ANEStatefulDecoderLayer(cfg, hf_layers[start + li], max_seq, li,
                                    use_linear=use_linear)
            for li in range(layers_in_chunk)
        ])

        # Unified K+V state. Buffer name must match the ct.StateType name.
        self.register_buffer(
            "kv_cache_0",
            torch.zeros(2 * layers_in_chunk, cfg.num_key_value_heads,
                        max_seq, cfg.head_dim, dtype=MODEL_DTYPE),
        )

    def forward(self, hidden_in, cos, sin, causal_mask, current_pos):
        h = hidden_in.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
        for layer in self.layers:
            h = layer(h, cos, sin, causal_mask, current_pos, self.kv_cache_0)
        return h.permute(0, 2, 3, 1).reshape(1, 1, self.hidden_size)


class ANEHeadChunk(nn.Module):
    def __init__(self, cfg, hf_text_model, lm_head, tie_word_embeddings,
                 use_linear: bool = False):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.final_norm = _norm_from_hf(
            hf_text_model.norm.weight, cfg.rms_norm_eps, cfg.hidden_size)
        lm_w = hf_text_model.embed_tokens.weight if tie_word_embeddings \
            else lm_head.weight
        self.lm_head = Conv2dLinear(cfg.hidden_size, cfg.vocab_size,
                                    bias=False, dtype=MODEL_DTYPE)
        self.lm_head.conv.weight.data = lm_w.detach().to(MODEL_DTYPE) \
            .unsqueeze(-1).unsqueeze(-1)
        self.argmax = InModelArgmax()
        if use_linear:
            self.lm_head = _replace_conv2dlinear_with_linear(self.lm_head)

    def forward(self, hidden_in):
        h_conv = hidden_in.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
        h_seq = h_conv.permute(0, 2, 3, 1).reshape(1, 1, self.hidden_size)
        h_seq = self.final_norm(h_seq)
        h_conv = h_seq.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
        logits_conv = _project(self.lm_head, h_conv)
        logits = logits_conv.squeeze(-1).squeeze(-1).unsqueeze(1)
        token_id, _ = self.argmax(logits)
        return token_id.to(torch.int32)


def export_embed_fp16(embed_weight: torch.Tensor, out_path: Path) -> None:
    w = embed_weight.detach().to(torch.float16).contiguous()
    buf = w.cpu().numpy().astype(np.float16).tobytes()
    out_path.write_bytes(buf)
    print(f"  wrote {out_path.name} ({len(buf)/1e6:.0f} MB)")


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
    pct = 100 * ane / compute if compute else 0.0
    print(f"    ANE placement: {ane}/{compute} ({pct:.1f}%)")
    return pct


def convert_body_stateful(chunk: ANEStatefulBodyChunk, cfg, start_layer,
                          end_layer, max_seq, out_path: Path):
    print(f"\n--- convert STATEFUL body layers [{start_layer}, {end_layer}) ---")
    head_dim = cfg.head_dim
    layers_in_chunk = end_layer - start_layer

    example = (
        torch.zeros(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE),      # hidden_in
        torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE),              # cos
        torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE),              # sin
        torch.zeros(1, 1, 1, max_seq, dtype=MODEL_DTYPE),            # causal_mask
        torch.zeros(1, dtype=torch.int32),                           # current_pos
    )
    t0 = time.time()
    traced = torch.jit.trace(chunk, example, strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, 1, cfg.hidden_size),
                      dtype=np.float16),
        ct.TensorType(name="cos", shape=(1, 1, head_dim), dtype=np.float16),
        ct.TensorType(name="sin", shape=(1, 1, head_dim), dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, max_seq),
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


def convert_head(chunk, cfg, out_path):
    print(f"\n--- convert head (final_norm + lm_head + argmax) ---")
    example = (torch.zeros(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE),)
    traced = torch.jit.trace(chunk, example, strict=False)
    ct_inputs = [ct.TensorType(name="hidden_in",
                                shape=(1, 1, cfg.hidden_size),
                                dtype=np.float16)]
    ct_outputs = [ct.TensorType(name="next_token", dtype=np.int32)]
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*')
                  if f.is_file()) / 1e6
    print(f"  saved fp16 {out_path.name} ({size_mb:.0f} MB)")
    _audit_ane(out_path)


def palettize_pkg(fp16_pkg: Path, out_pkg: Path, nbits: int):
    print(f"\n--- palettize INT{nbits}: {fp16_pkg.name} → {out_pkg.name} ---")
    m_in = ct.models.MLModel(str(fp16_pkg))
    op_cfg = OpPalettizerConfig(mode="kmeans", nbits=nbits,
                                granularity="per_tensor")
    opt_cfg = OptimizationConfig(global_config=op_cfg)
    m_out = palettize_weights(m_in, opt_cfg)
    m_out.save(str(out_pkg))
    src_mb = sum(f.stat().st_size for f in fp16_pkg.rglob('*')
                 if f.is_file()) / 1e6
    dst_mb = sum(f.stat().st_size for f in out_pkg.rglob('*')
                 if f.is_file()) / 1e6
    print(f"  bundle: {src_mb:.0f} MB (fp16) → {dst_mb:.0f} MB (int{nbits}) "
          f"[{100*dst_mb/src_mb:.1f}%]")
    _audit_ane(out_pkg)


EMBED_BIN_NAME = "embed_weight.bin"
HEAD_CHUNK_NAME = "chunk_head"


def _body_boundaries(num_layers, num_chunks):
    assert num_layers % num_chunks == 0, \
        f"num_layers={num_layers} not divisible by num_chunks={num_chunks}"
    per = num_layers // num_chunks
    return [(i * per, (i + 1) * per) for i in range(num_chunks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--num-chunks", type=int, default=NUM_BODY_CHUNKS,
                    help="4 (7 layers/chunk) or 2 (14 layers/chunk, ANEMLL).")
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--keep-fp16", action="store_true")
    ap.add_argument("--only-one-chunk", action="store_true",
                    help="Convert chunk 0 only — smoke test the stateful path "
                         "before committing to a full 4-chunk build.")
    ap.add_argument("--linear-projections", action="store_true",
                    help="Plan-3 (cml9 PR #2577) variant: replace every "
                         "Conv2dLinear (1×1 wrapper) projection with a "
                         "shape-equivalent nn.Linear. Drops the permute/"
                         "squeeze wrapper ops in the MIL graph; iPhone "
                         "parity validated for Gemma 4 stateful chunks "
                         "(commit 7c9cfea).")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    chunks_dir = out_root / "qwen3_vl_2b_stateful_chunks"
    fp16_dir = out_root / "_fp16_intermediate"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    print("loading Qwen3-VL 2B text backbone (fp32)...")
    t0 = time.time()
    cfg = load_text_config()
    print(f"  text cfg: layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} "
          f"num_kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim}")
    text_model, lm_head = load_text_backbone()
    print(f"  loaded in {time.time()-t0:.1f}s")

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

    body_modules = []
    for start, end in boundaries:
        m = ANEStatefulBodyChunk(cfg, text_model.layers, start, end, args.max_seq,
                                  use_linear=use_linear)
        body_modules.append(m.eval().to(MODEL_DTYPE))

    if args.only_one_chunk:
        boundaries = boundaries[:1]
        body_modules = body_modules[:1]

    del text_model, lm_head

    for ci, ((start, end), mod) in enumerate(zip(boundaries, body_modules)):
        name = f"chunk_{ci}"
        fp16_path = fp16_dir / f"{name}.mlpackage"
        final_path = chunks_dir / f"{name}.mlpackage"
        convert_body_stateful(mod, cfg, start, end, args.max_seq, fp16_path)
        if args.nbits == 0:
            if final_path.exists():
                shutil.rmtree(final_path)
            shutil.move(str(fp16_path), str(final_path))
        else:
            palettize_pkg(fp16_path, final_path, args.nbits)

    if not args.only_one_chunk:
        fp16_head = fp16_dir / f"{HEAD_CHUNK_NAME}.mlpackage"
        final_head = chunks_dir / f"{HEAD_CHUNK_NAME}.mlpackage"
        convert_head(head_module, cfg, fp16_head)
        if args.nbits == 0:
            if final_head.exists():
                shutil.rmtree(final_head)
            shutil.move(str(fp16_head), str(final_head))
        else:
            palettize_pkg(fp16_head, final_head, args.nbits)

    if not args.keep_fp16:
        shutil.rmtree(fp16_dir, ignore_errors=True)

    print(f"\nshipping artifacts under {chunks_dir}")
    for p in sorted(chunks_dir.iterdir()):
        size = p.stat().st_size / 1e6 if p.is_file() else \
            sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e6
        print(f"  {p.name}: {size:.0f} MB")


if __name__ == "__main__":
    main()
