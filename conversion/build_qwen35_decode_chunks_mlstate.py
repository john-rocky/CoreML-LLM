"""Qwen3.5 decode in MLState + slice_update chunks. Variant of
build_qwen35_decode_chunks_ane.py that stores all KV / SSM state inside
the model via MLState instead of marshaling state I/O each step.

Per VL Phase 1 (qwen3vl_phase1_result memory), the MLState path gave
10→24.4 tok/s = 2.4× decode speedup on Qwen3-VL 2B and dropped
phys_footprint 1.7 GB → 264 MB. The win comes from eliminating ~50 KV
state arrays per step that would otherwise round-trip through the
Swift/Python feature dictionary.

Structure (4 chunks total + 1 embed sidecar):
  embed_weight.bin — raw fp16, mmap'd by Swift.
  chunk_a..chunk_c — body chunks; each owns its own KV / SSM states.
  chunk_d          — last 6 body layers + final_norm + Conv2d lm_head;
                     emits fp32 logits.

Each body chunk has up to three MLStates depending on which layer types
fall in its range:
  conv_state_<i> : (L_lin, conv_dim, K=4)            SSM conv1d states
  rec_state_<i>  : (L_lin, num_v, Dk, Dv)            SSM recurrent states
  kv_cache_<i>   : (2*L_full, num_kv_heads, S, head_dim)  full-attn KV

Where L_lin / L_full are the count of linear / full attention layers in
that chunk's range.

Usage:
  python build_qwen35_decode_chunks_mlstate.py --out-dir /tmp/qwen35_0_8b_mls
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
from qwen3_5_decode_layer_mlstate import (
    MLStateDecoderLayer, _norm_from_hf, is_full_attn,
)
from test_qwen3_5_full_decode_trace import MAX_SEQ


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_BUNDLE_NAME = "qwen3_5_0_8b_decode_chunks_mls"
DEFAULT_NUM_CHUNKS = 4
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


def export_embed_fp16(hf_model, out_path: Path) -> None:
    w = hf_model.model.embed_tokens.weight.detach().to(torch.float16).contiguous()
    vocab, hidden = w.shape
    print(f"\n--- export {out_path.name} ({vocab} × {hidden} fp16) ---")
    buf = w.cpu().numpy().astype(np.float16).tobytes()
    out_path.write_bytes(buf)
    mb = len(buf) / 1e6
    print(f"  wrote {out_path.name} ({mb:.0f} MB)")


def _ssm_state_shapes(cfg, l_lin: int):
    conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2 \
        + cfg.linear_value_head_dim * cfg.linear_num_value_heads
    # 4D state shape — ANE-friendly rank.  Layout: (L_lin, 1, C, K).
    conv_shape = (l_lin, 1, conv_dim, cfg.linear_conv_kernel_dim)
    rec_shape  = (l_lin, cfg.linear_num_value_heads,
                  cfg.linear_key_head_dim, cfg.linear_value_head_dim)
    return conv_shape, rec_shape


def _kv_state_shape(cfg, l_full: int, max_seq: int):
    return (2 * l_full, cfg.num_key_value_heads, max_seq, cfg.head_dim)


class MLStateBodyChunk(nn.Module):
    """N-layer body chunk owning per-state-type MLState buffers."""
    def __init__(self, cfg, hf_model, start: int, end: int, max_seq: int,
                 chunk_idx: int):
        super().__init__()
        self.start = start
        self.end = end
        self.hidden_size = cfg.hidden_size
        self.max_seq = max_seq

        layers = []
        lin_count = 0
        full_count = 0
        for i in range(start, end):
            full = is_full_attn(i)
            layers.append(MLStateDecoderLayer(
                cfg, hf_model.model.layers[i], max_seq,
                lin_idx_in_chunk=(0 if full else lin_count),
                full_idx_in_chunk=(full_count if full else 0),
            ))
            if full:
                full_count += 1
            else:
                lin_count += 1
        self.layers = nn.ModuleList(layers)
        self.lin_count = lin_count
        self.full_count = full_count

        # Register state buffers. The buffer name is what ct.StateType
        # binds to inside the converted graph.
        conv_shape, rec_shape = _ssm_state_shapes(cfg, max(lin_count, 1))
        kv_shape = _kv_state_shape(cfg, max(full_count, 1), max_seq)
        # Always register all three so trace captures the field; in
        # forward we only touch the ones that have layers.
        self.register_buffer(
            "conv_state",
            torch.zeros(*conv_shape, dtype=MODEL_DTYPE),
        )
        self.register_buffer(
            "rec_state",
            torch.zeros(*rec_shape, dtype=MODEL_DTYPE),
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(*kv_shape, dtype=MODEL_DTYPE),
        )

    def forward(self, hidden_in, cos, sin, causal_mask, current_pos):
        h = hidden_in
        for layer in self.layers:
            h = layer(h, cos, sin, causal_mask, current_pos,
                      self.conv_state, self.rec_state, self.kv_cache)
        return h


class MLStateTailChunk(MLStateBodyChunk):
    """Last chunk = body + final_norm + Conv2d lm_head + fp32 logits."""
    def __init__(self, cfg, hf_model, start: int, max_seq: int,
                 chunk_idx: int):
        super().__init__(cfg, hf_model, start, cfg.num_hidden_layers,
                          max_seq, chunk_idx)
        self.final_norm = _norm_from_hf(
            hf_model.model.norm.weight, cfg.rms_norm_eps, cfg.hidden_size)
        lm_w = (hf_model.model.embed_tokens.weight
                if cfg.tie_word_embeddings else hf_model.lm_head.weight)
        self.lm_head = Conv2dLinear(
            cfg.hidden_size, cfg.vocab_size, bias=False, dtype=MODEL_DTYPE)
        self.lm_head.conv.weight.data = (
            lm_w.detach().to(MODEL_DTYPE).unsqueeze(-1).unsqueeze(-1)
        )

    def forward(self, hidden_in, cos, sin, causal_mask, current_pos):
        h = super().forward(hidden_in, cos, sin, causal_mask, current_pos)
        h = self.final_norm(h)
        logits = self.lm_head(h)
        return logits.float()


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


def convert_chunk(chunk, cfg, start: int, end: int, max_seq: int,
                   out_path: Path, *, kind: str):
    print(f"\n--- convert MLState {kind} chunk layers [{start}, {end}) ---")
    rot = Qwen3_5TextRotaryEmbedding(cfg).eval()
    pos_ids = torch.zeros(1, 1, dtype=torch.long)
    dummy = torch.zeros(1, 1, cfg.hidden_size)
    with torch.no_grad():
        cos_t, sin_t = rot(dummy, pos_ids)

    head_dim = cfg.head_dim
    example = (
        torch.zeros(1, 1, cfg.hidden_size, dtype=MODEL_DTYPE),
        cos_t.to(MODEL_DTYPE),
        sin_t.to(MODEL_DTYPE),
        torch.zeros(1, 1, 1, max_seq, dtype=MODEL_DTYPE),  # causal_mask
        torch.zeros(1, dtype=torch.int32),                  # current_pos
    )

    t0 = time.time()
    traced = torch.jit.trace(chunk, example, strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, 1, cfg.hidden_size), dtype=np.float16),
        ct.TensorType(name="cos", shape=cos_t.shape, dtype=np.float16),
        ct.TensorType(name="sin", shape=sin_t.shape, dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, max_seq), dtype=np.float16),
        ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
    ]
    if kind == "tail":
        ct_outputs = [ct.TensorType(name="logits", dtype=np.float32)]
    else:
        ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]

    # State declarations — bind to the buffer names we registered.
    conv_shape, rec_shape = _ssm_state_shapes(cfg, max(chunk.lin_count, 1))
    kv_shape = _kv_state_shape(cfg, max(chunk.full_count, 1), max_seq)
    ct_states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=conv_shape, dtype=np.float16),
                     name="conv_state"),
        ct.StateType(wrapped_type=ct.TensorType(shape=rec_shape, dtype=np.float16),
                     name="rec_state"),
        ct.StateType(wrapped_type=ct.TensorType(shape=kv_shape, dtype=np.float16),
                     name="kv_cache"),
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


def _chunk_boundaries(num_layers: int, num_chunks: int):
    assert num_layers % num_chunks == 0
    per = num_layers // num_chunks
    return [(i * per, (i + 1) * per) for i in range(num_chunks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--bundle-name", default=DEFAULT_BUNDLE_NAME)
    ap.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    ap.add_argument("--max-seq", type=int, default=MAX_SEQ)
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--keep-fp16", action="store_true")
    args = ap.parse_args()

    if args.num_chunks > len(CHUNK_NAMES):
        raise SystemExit(f"max {len(CHUNK_NAMES)} chunks; extend CHUNK_NAMES.")

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

    export_embed_fp16(hf, chunks_dir / EMBED_BIN_NAME)

    modules = []
    for ci, (start, end) in enumerate(boundaries):
        is_last = ci == len(boundaries) - 1
        if is_last:
            m = MLStateTailChunk(cfg, hf, start, args.max_seq, ci)
        else:
            m = MLStateBodyChunk(cfg, hf, start, end, args.max_seq, ci)
        modules.append(m.eval().to(MODEL_DTYPE))
    del hf

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
