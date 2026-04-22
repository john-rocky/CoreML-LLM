"""Qwen3.5-2B decode split into 4 stateful chunks for iPhone shipping.

Why 4 chunks (not 2): iPhone ANE compile budget is per-mlprogram and
applies to the **fp16-dequantized** weight size, not disk size (lessons
§3.3: palettization shrinks disk, not ANE memory). Our 2-chunk attempt
put each chunk at ~1.1 GB INT8 → ~2 GB fp16 in ANE → silent GPU
fallback. 4-chunk (6 layers each) keeps each chunk under ~1 GB INT8 /
~1.8 GB fp16, matching the envelope Gemma 4 E4B (42 layers, 4 chunks,
~1.6 GB fp16 each) proves works on iPhone 17 Pro ANE.

Layer distribution (even split over 24 layers):
  chunk_a: embed_tokens + layers  0..5   — has embed (~1 GB fp16)
  chunk_b: layers  6..11                 — pure body
  chunk_c: layers 12..17                 — pure body
  chunk_d: layers 18..23 + final_norm + lm_head — has head (~1 GB fp16)

chunk_a carries `input_token` input, emits `hidden` fp16.
chunk_b/c receive `hidden_in` fp16, emit `hidden` fp16.
chunk_d receives `hidden_in` fp16, emits `logits` fp32.
Each chunk has its own 6-layer state slice (2 tensors per layer).

Usage:
  python build_qwen35_2b_decode_chunks.py --out-dir /tmp/qwen35_2b_chunks
  # → /tmp/qwen35_2b_chunks/qwen3_5_2b_decode_chunks/chunk_{a,b,c,d}.mlpackage
"""
from pathlib import Path
import argparse
import shutil
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
from transformers import AutoModelForCausalLM, AutoConfig, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from test_qwen3_5_full_decode_trace import (
    DecoderDecodeLayer, DecodeRMSNorm, MAX_SEQ, make_zero_states, cos_sim,
)

MODEL_ID = "Qwen/Qwen3.5-2B"
NUM_CHUNKS = 4  # 24 layers / 6 each — matches Gemma 4 E4B chunk pattern


# ---- 2B text config / backbone -------------------------------------------

def load_2b_text_config():
    full_cfg = AutoConfig.from_pretrained(MODEL_ID)
    text_dict = (full_cfg.text_config.to_dict()
                 if hasattr(full_cfg, "text_config") else full_cfg.to_dict())
    return Qwen3_5TextConfig.from_dict(text_dict)


def load_2b_text_backbone():
    full = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).eval()
    assert hasattr(full, "model") and hasattr(full.model, "layers"), \
        f"expected .model.layers on {type(full).__name__}"
    return full


# ---- chunk modules --------------------------------------------------------

def export_embed_fp16(hf_model, out_path: Path) -> None:
    """Export `embed_tokens.weight` as a raw fp16 binary file for
    Swift-side mmap lookup. Avoids the chunk_embed mlpackage path —
    CoreML would dequantize the palettized embed into ~1 GB of process
    memory at load time (gather op isn't ANE-native, so the chunk fell
    to CPU and its weights counted against phys_footprint). Shipping
    the raw fp16 bin lets Swift mmap it read-only so only touched rows
    (4 KB per step) page in and those pages remain 'clean' — they don't
    inflate the app's resident memory footprint.

    File layout: contiguous fp16, shape (vocab_size, hidden_size),
    row-major. Swift indexes as `fp16_ptr[token * hidden_size + i]`.
    """
    w = hf_model.model.embed_tokens.weight.detach().to(torch.float16).contiguous()
    vocab, hidden = w.shape
    print(f"\n--- export embed_weight.bin ({vocab} × {hidden} fp16) ---")
    buf = w.cpu().numpy().astype(np.float16).tobytes()
    out_path.write_bytes(buf)
    mb = len(buf) / 1e6
    print(f"  wrote {out_path.name} ({mb:.0f} MB)")


class DecodeChunkBody(nn.Module):
    """Middle chunk. Takes hidden from the previous chunk, runs layers
    [start, end), emits hidden + updated state slice."""
    def __init__(self, cfg, hf_model, start, end, max_seq):
        super().__init__()
        self.start = start
        self.end = end
        self.layers = nn.ModuleList([
            DecoderDecodeLayer(cfg, hf_model.model.layers[i], max_seq)
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


class DecodeChunkTail(nn.Module):
    """Last chunk. Layers [start, num_layers) + final_norm + lm_head.
    Emits logits + updated state slice."""
    def __init__(self, cfg, hf_model, start, max_seq):
        super().__init__()
        self.start = start
        self.end = cfg.num_hidden_layers
        self.layers = nn.ModuleList([
            DecoderDecodeLayer(cfg, hf_model.model.layers[i], max_seq)
            for i in range(start, cfg.num_hidden_layers)
        ])
        self.final_norm = DecodeRMSNorm(cfg.rms_norm_eps, hf_model.model.norm.weight)
        lm_w = (hf_model.model.embed_tokens.weight
                if cfg.tie_word_embeddings else hf_model.lm_head.weight)
        self.lm_head_w = nn.Parameter(lm_w.detach().clone(), requires_grad=False)

    def forward(self, hidden_in, position, cos, sin, *states):
        hidden = hidden_in
        new_states = []
        for local_i, layer in enumerate(self.layers):
            sa, sb = states[2 * local_i], states[2 * local_i + 1]
            hidden, ns_a, ns_b = layer(hidden, position, cos, sin, sa, sb)
            new_states.append(ns_a); new_states.append(ns_b)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.lm_head_w)
        return (logits, *new_states)


# ---- shape helpers --------------------------------------------------------

def _layer_state_shapes(cfg, layer_idx, max_seq):
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

def convert_chunk(
    chunk,
    cfg,
    start_layer: int,
    end_layer: int,
    max_seq: int,
    out_path: Path,
    *,
    kind: str,  # "body" | "tail"
):
    """Trace `chunk` and emit fp16 mlpackage at `out_path`. Embed is not
    traced — it's exported as raw fp16 via export_embed_fp16()."""
    print(f"\n--- convert chunk {kind!r} (layers [{start_layer}, {end_layer})) ---")

    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    pos_ids = torch.zeros(1, 1, dtype=torch.long)
    dummy = torch.zeros(1, 1, cfg.hidden_size)
    with torch.no_grad():
        cos_t, sin_t = rot(dummy, pos_ids)

    # Example inputs for tracing: body and tail chunks both take hidden_in.
    example = [torch.zeros(1, 1, cfg.hidden_size, dtype=torch.float32)]
    example.append(torch.zeros(1, dtype=torch.float32))  # position
    example.append(cos_t.float())
    example.append(sin_t.float())
    for i in range(start_layer, end_layer):
        sa_shape, sb_shape = _layer_state_shapes(cfg, i, max_seq)
        example.append(torch.zeros(*sa_shape))
        example.append(torch.zeros(*sb_shape))

    t0 = time.time()
    traced = torch.jit.trace(chunk, tuple(example), strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, 1, cfg.hidden_size), dtype=np.float16),
        ct.TensorType(name="position", shape=(1,), dtype=np.float32),
        ct.TensorType(name="cos", shape=cos_t.shape, dtype=np.float16),
        ct.TensorType(name="sin", shape=sin_t.shape, dtype=np.float16),
    ]
    ct_outputs = []
    if kind == "tail":
        ct_outputs.append(ct.TensorType(name="logits", dtype=np.float32))
    else:
        ct_outputs.append(ct.TensorType(name="hidden", dtype=np.float16))

    for i in range(start_layer, end_layer):
        sa_shape, sb_shape = _layer_state_shapes(cfg, i, max_seq)
        ct_inputs.append(ct.TensorType(
            name=f"state_{i}_a", shape=sa_shape, dtype=np.float16))
        ct_inputs.append(ct.TensorType(
            name=f"state_{i}_b", shape=sb_shape, dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_state_{i}_a", dtype=np.float16))
        ct_outputs.append(ct.TensorType(name=f"new_state_{i}_b", dtype=np.float16))

    t0 = time.time()
    ct_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=ct_inputs,
        outputs=ct_outputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    print(f"  converted in {time.time()-t0:.1f}s")
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*') if f.is_file()) / 1e6
    print(f"  saved fp16 {out_path.name} ({size_mb:.0f} MB)")
    _audit_ane(out_path)
    return ct_model


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
    """Evenly split num_layers across num_chunks. Returns list of
    (start, end) pairs, end-exclusive."""
    assert num_layers % num_chunks == 0, \
        f"num_layers ({num_layers}) must be divisible by num_chunks ({num_chunks})"
    per = num_layers // num_chunks
    return [(i * per, (i + 1) * per) for i in range(num_chunks)]


# Body chunks follow in order; the last body chunk is the tail with
# final_norm + lm_head. 24 layers split evenly across N body chunks.
# Token embedding is NOT an mlpackage — it's exported as a raw fp16
# binary so Swift can mmap it (clean pages, doesn't inflate
# phys_footprint the way CPU-resident CoreML weights do).
EMBED_BIN_NAME = "embed_weight.bin"
BODY_CHUNK_NAMES = ["chunk_a", "chunk_b", "chunk_c", "chunk_d", "chunk_e", "chunk_f"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-chunks", type=int, default=NUM_CHUNKS,
                    help="number of body/tail chunks (default 4). Embed "
                         "is exported as a separate raw fp16 .bin — not "
                         "counted here.")
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8],
                    help="0 = keep fp16 only; 4/8 = palettize")
    ap.add_argument("--keep-fp16", action="store_true")
    args = ap.parse_args()

    if args.num_chunks > len(BODY_CHUNK_NAMES):
        raise SystemExit(f"max {len(BODY_CHUNK_NAMES)} body chunks; add more "
                         f"names to BODY_CHUNK_NAMES list for larger splits")

    out_root = Path(args.out_dir).resolve()
    chunks_dir = out_root / "qwen3_5_2b_decode_chunks"
    fp16_dir = out_root / "_fp16_intermediate"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    print("loading Qwen3.5-2B (AutoModelForCausalLM, fp32)...")
    t0 = time.time()
    cfg = load_2b_text_config()
    print(f"  text config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"vocab={cfg.vocab_size} tie_word_embeddings={cfg.tie_word_embeddings}")
    hf = load_2b_text_backbone()
    print(f"  loaded in {time.time()-t0:.1f}s as {type(hf).__name__}")

    boundaries = _chunk_boundaries(cfg.num_hidden_layers, args.num_chunks)
    print(f"  body boundaries: {boundaries}  (+ embed_weight.bin sidecar)")

    # Export embed_tokens as raw fp16 — Swift mmaps this, skipping the
    # CoreML dequant-to-fp16-on-CPU path that inflated memory by ~1 GB.
    export_embed_fp16(hf, chunks_dir / EMBED_BIN_NAME)

    # Build body+tail modules (embed is handled in Swift, not traced)
    body_modules = []
    for ci, (start, end) in enumerate(boundaries):
        is_last = ci == len(boundaries) - 1
        m = (DecodeChunkTail(cfg, hf, start, args.max_seq)
             if is_last else
             DecodeChunkBody(cfg, hf, start, end, args.max_seq))
        body_modules.append(m.eval().float())
    del hf  # free 8 GB fp32 weights before ct.convert spikes RAM

    # Convert + palettize each body/tail chunk
    for ci, ((start, end), torch_m, name) in enumerate(
        zip(boundaries, body_modules, BODY_CHUNK_NAMES[:args.num_chunks])
    ):
        kind = "tail" if ci == len(boundaries) - 1 else "body"
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
        size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1e6
        print(f"  {p.name}: {size:.0f} MB")


if __name__ == "__main__":
    main()
