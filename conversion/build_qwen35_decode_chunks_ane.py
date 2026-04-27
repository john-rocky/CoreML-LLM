"""Qwen3.5 decode in stateful ANE-optimized chunks. Drives both 0.8B
and 2B re-conversions; output layout matches the existing Swift loader
(`Qwen35Generator.swift` chunked path) so no Swift wiring change is
needed for shipping.

Structure (4 chunks total + 1 embed sidecar):
  embed_weight.bin — raw fp16, mmap'd by Swift (no CoreML residency).
  chunk_a..chunk_c — body-only chunks, ANE-recipe layers.
  chunk_d          — last 6 body layers + final_norm + Conv2d lm_head
                     emits fp32 logits (Swift-side argmax/sample).

Why 4 chunks (not 1, not 3-with-split-head):
  - Monolithic 0.8B + Gemma 4 ANE recipe was reverted at 5be231b/f3ec1ef
    because the 24-layer graph blew the iOS 26.1 BNNS/ANEF compiler
    ceiling. Per-chunk graphs (6 layers each) fit comfortably.
  - 4-chunk + chunk_d=body+head matches the layout the Swift loader
    already supports for 2B v1.1.0 (`Qwen35Generator.swift` lines
    200-220) — same code path serves both 0.8B and 2B once they ship
    with this layout. In-graph TopK + split LM head are saved for a
    follow-up "stretch" PR (1 MB/step transfer save).

ANE recipe applied (see conversion/qwen3_5_decode_layer_ane.py):
  - Conv2dLinear for every projection (SSM + full_attn + MLP + lm_head).
  - ANERMSNorm via cat([x,-x]) → LayerNorm identity.
  - ane_softmax (decomposed) for full_attention.
  - repeat_kv_ane for GQA.
  - KV cache update via where(range == position) — scatter-free.

Usage (0.8B):
  python build_qwen35_decode_chunks_ane.py --out-dir /tmp/qwen35_0_8b_ane

Usage (2B):
  python build_qwen35_decode_chunks_ane.py --out-dir /tmp/qwen35_2b_ane \\
      --model-id Qwen/Qwen3.5-2B \\
      --bundle-name qwen3_5_2b_decode_chunks
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

import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
)
from transformers import AutoModelForCausalLM, AutoConfig, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import MODEL_DTYPE, Conv2dLinear
from qwen3_5_decode_layer_ane import ANEDecoderDecodeLayer, _norm_from_hf
from test_qwen3_5_full_decode_trace import MAX_SEQ


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_BUNDLE_NAME = "qwen3_5_0_8b_decode_chunks"
DEFAULT_NUM_CHUNKS = 4     # 24 layers / 6 each — matches 2B v1.1.0 layout.
EMBED_BIN_NAME = "embed_weight.bin"
CHUNK_NAMES = ["chunk_a", "chunk_b", "chunk_c", "chunk_d",
               "chunk_e", "chunk_f"]


def load_text_config(model_id: str) -> Qwen3_5TextConfig:
    full_cfg = AutoConfig.from_pretrained(model_id)
    text_dict = (full_cfg.text_config.to_dict()
                 if hasattr(full_cfg, "text_config") else full_cfg.to_dict())
    return Qwen3_5TextConfig.from_dict(text_dict)


def load_text_backbone(model_id: str):
    full = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    assert hasattr(full, "model") and hasattr(full.model, "layers"), \
        f"expected .model.layers on {type(full).__name__}"
    return full


# ---- chunk modules --------------------------------------------------------


def export_embed_fp16(hf_model, out_path: Path) -> None:
    """Raw fp16 (vocab, hidden) row-major — mmap'd by Swift."""
    w = hf_model.model.embed_tokens.weight.detach().to(torch.float16).contiguous()
    vocab, hidden = w.shape
    print(f"\n--- export {out_path.name} ({vocab} × {hidden} fp16) ---")
    buf = w.cpu().numpy().astype(np.float16).tobytes()
    out_path.write_bytes(buf)
    mb = len(buf) / 1e6
    print(f"  wrote {out_path.name} ({mb:.0f} MB)")


class ANEBodyChunk(nn.Module):
    """Pure body chunk — runs ANEDecoderDecodeLayer for [start, end)."""
    def __init__(self, cfg, hf_model, start: int, end: int, max_seq: int):
        super().__init__()
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([
            ANEDecoderDecodeLayer(cfg, hf_model.model.layers[i], max_seq)
            for i in range(start, end)
        ])

    def forward(self, hidden_in, position, cos, sin, *states):
        hidden = hidden_in
        new_states = []
        for local_i, layer in enumerate(self.layers):
            sa, sb = states[2 * local_i], states[2 * local_i + 1]
            hidden, ns_a, ns_b = layer(hidden, position, cos, sin, sa, sb)
            new_states.append(ns_a); new_states.append(ns_b)
        return (hidden, *new_states)


class ANETailChunk(nn.Module):
    """Last chunk = body layers [start, num_layers) + final_norm + Conv2d
    lm_head. Emits fp32 logits (Swift-side argmax/sample) for drop-in
    compat with the existing chunked Qwen35Generator.swift loader.

    A future PR can split lm_head into a separate chunk_head with
    in-graph TopK (~1 MB/step ANE→Swift transfer save) — gated on the
    Swift loader picking up the new chunk_head signature.
    """
    def __init__(self, cfg, hf_model, start: int, max_seq: int):
        super().__init__()
        self.start = start
        self.end = cfg.num_hidden_layers
        self.layers = nn.ModuleList([
            ANEDecoderDecodeLayer(cfg, hf_model.model.layers[i], max_seq)
            for i in range(start, cfg.num_hidden_layers)
        ])
        self.final_norm = _norm_from_hf(
            hf_model.model.norm.weight, cfg.rms_norm_eps, cfg.hidden_size)
        # tie_word_embeddings=True on 2B, False on 0.8B — both handled.
        lm_w = (hf_model.model.embed_tokens.weight
                if cfg.tie_word_embeddings else hf_model.lm_head.weight)
        self.lm_head = Conv2dLinear(
            cfg.hidden_size, cfg.vocab_size, bias=False, dtype=MODEL_DTYPE)
        self.lm_head.conv.weight.data = (
            lm_w.detach().to(MODEL_DTYPE).unsqueeze(-1).unsqueeze(-1)
        )

    def forward(self, hidden_in, position, cos, sin, *states):
        hidden = hidden_in
        new_states = []
        for local_i, layer in enumerate(self.layers):
            sa, sb = states[2 * local_i], states[2 * local_i + 1]
            hidden, ns_a, ns_b = layer(hidden, position, cos, sin, sa, sb)
            new_states.append(ns_a); new_states.append(ns_b)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)        # (1, 1, V) fp16
        return (logits.float(), *new_states) # fp32 logits matches Swift expectation


# ---- shape helpers --------------------------------------------------------


def _layer_state_shapes(cfg, layer_idx: int, max_seq: int):
    lt = "linear_attention" if layer_idx % 4 != 3 else "full_attention"
    if lt == "linear_attention":
        conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2 \
            + cfg.linear_value_head_dim * cfg.linear_num_value_heads
        a = (1, conv_dim, cfg.linear_conv_kernel_dim)
        b = (1, cfg.linear_num_value_heads,
             cfg.linear_key_head_dim, cfg.linear_value_head_dim)
    else:
        a = (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)
        b = (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)
    return a, b


def _audit_ane(out_path: Path) -> float:
    reloaded = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
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


# ---- chunk conversion -----------------------------------------------------


def convert_chunk(chunk, cfg, start: int, end: int, max_seq: int,
                   out_path: Path, *, kind: str):
    """Trace and convert a body or tail chunk to mlpackage.

    kind = "body" → emits hidden (1,1,H) fp16 + state outputs.
    kind = "tail" → emits logits (1,1,V) fp32 + state outputs.
    """
    print(f"\n--- convert ANE {kind} chunk layers [{start}, {end}) ---")
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    pos_ids = torch.zeros(1, 1, dtype=torch.long)
    dummy = torch.zeros(1, 1, cfg.hidden_size)
    with torch.no_grad():
        cos_t, sin_t = rot(dummy, pos_ids)

    example = [torch.zeros(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE)]
    example.append(torch.zeros(1, dtype=torch.float32))
    example.append(cos_t.to(MODEL_DTYPE))
    example.append(sin_t.to(MODEL_DTYPE))
    for i in range(start, end):
        sa_shape, sb_shape = _layer_state_shapes(cfg, i, max_seq)
        example.append(torch.zeros(*sa_shape, dtype=MODEL_DTYPE))
        example.append(torch.zeros(*sb_shape, dtype=MODEL_DTYPE))

    t0 = time.time()
    traced = torch.jit.trace(chunk, tuple(example), strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, 1, cfg.hidden_size), dtype=np.float16),
        ct.TensorType(name="position", shape=(1,), dtype=np.float32),
        ct.TensorType(name="cos", shape=cos_t.shape, dtype=np.float16),
        ct.TensorType(name="sin", shape=sin_t.shape, dtype=np.float16),
    ]
    if kind == "tail":
        ct_outputs = [ct.TensorType(name="logits", dtype=np.float32)]
    else:
        ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
    for i in range(start, end):
        sa_shape, sb_shape = _layer_state_shapes(cfg, i, max_seq)
        ct_inputs.append(ct.TensorType(
            name=f"state_{i}_a", shape=sa_shape, dtype=np.float16))
        ct_inputs.append(ct.TensorType(
            name=f"state_{i}_b", shape=sb_shape, dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_state_{i}_a", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_state_{i}_b", dtype=np.float16))

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


def palettize_chunk(fp16_pkg: Path, out_pkg: Path, nbits: int):
    print(f"\n--- palettize INT{nbits}: {fp16_pkg.name} → {out_pkg.name} ---")
    m_in = ct.models.MLModel(str(fp16_pkg))
    op_cfg = OpPalettizerConfig(mode="kmeans", nbits=nbits, granularity="per_tensor")
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


# ---- main -----------------------------------------------------------------


def _chunk_boundaries(num_layers: int, num_chunks: int):
    assert num_layers % num_chunks == 0, \
        f"num_layers ({num_layers}) must divide num_chunks ({num_chunks})"
    per = num_layers // num_chunks
    return [(i * per, (i + 1) * per) for i in range(num_chunks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                    help=f"HF model id (default {DEFAULT_MODEL_ID}). "
                         "Use Qwen/Qwen3.5-2B for the 2B re-build.")
    ap.add_argument("--bundle-name", default=DEFAULT_BUNDLE_NAME,
                    help=f"Output bundle directory name (default "
                         f"{DEFAULT_BUNDLE_NAME}). Use "
                         "qwen3_5_2b_decode_chunks for 2B.")
    ap.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS,
                    help="number of chunks total (last is body+head). "
                         "Default 4 → 6 layers each, matches 2B v1.1.0.")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--keep-fp16", action="store_true")
    args = ap.parse_args()

    if args.num_chunks > len(CHUNK_NAMES):
        raise SystemExit(
            f"max {len(CHUNK_NAMES)} chunks; extend CHUNK_NAMES.")

    out_root = Path(args.out_dir).resolve()
    chunks_dir = out_root / args.bundle_name
    fp16_dir = out_root / "_fp16_intermediate"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model_id} (fp32)...")
    t0 = time.time()
    cfg = load_text_config(args.model_id)
    print(f"  text config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"vocab={cfg.vocab_size} tie_word_embeddings={cfg.tie_word_embeddings}")
    hf = load_text_backbone(args.model_id)
    print(f"  loaded in {time.time()-t0:.1f}s as {type(hf).__name__}")

    boundaries = _chunk_boundaries(cfg.num_hidden_layers, args.num_chunks)
    print(f"  chunk boundaries: {boundaries} (last = body+head; embed sidecar)")

    # Embed sidecar
    export_embed_fp16(hf, chunks_dir / EMBED_BIN_NAME)

    # Build modules: body chunks + tail chunk (body + final_norm + lm_head).
    modules = []
    for ci, (start, end) in enumerate(boundaries):
        is_last = ci == len(boundaries) - 1
        if is_last:
            m = ANETailChunk(cfg, hf, start, args.max_seq)
        else:
            m = ANEBodyChunk(cfg, hf, start, end, args.max_seq)
        modules.append(m.eval().to(MODEL_DTYPE))
    del hf  # free fp32 weights before ct.convert spikes RAM

    # Convert + palettize each chunk
    chunk_names = CHUNK_NAMES[:args.num_chunks]
    for ci, ((start, end), torch_m, name) in enumerate(
        zip(boundaries, modules, chunk_names)
    ):
        is_last = ci == len(boundaries) - 1
        kind = "tail" if is_last else "body"
        fp16_path = fp16_dir / f"{name}.mlpackage"
        final_path = chunks_dir / f"{name}.mlpackage"
        convert_chunk(torch_m, cfg, start, end, args.max_seq, fp16_path, kind=kind)
        if args.nbits == 0:
            shutil.move(str(fp16_path), str(final_path))
        else:
            palettize_chunk(fp16_path, final_path, args.nbits)

    if not args.keep_fp16:
        shutil.rmtree(fp16_dir, ignore_errors=True)

    print(f"\n✓ shipping artifacts under {chunks_dir}")
    for p in sorted(chunks_dir.iterdir()):
        if p.is_file():
            size = p.stat().st_size / 1e6
        else:
            size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e6
        print(f"  {p.name}: {size:.0f} MB")


if __name__ == "__main__":
    main()
