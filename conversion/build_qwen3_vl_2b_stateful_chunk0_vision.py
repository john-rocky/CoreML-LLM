"""Qwen3-VL 2B chunk_0 with DeepStack vision injection — stateful (Phase 1).

Combines:
  - the MLState + slice_update KV recipe from
    `build_qwen3_vl_2b_stateful_chunks.py`
  - the DeepStack injection recipe from
    `build_qwen3_vl_2b_chunk0_vision.py`

Produces `chunk_0_vision.mlpackage` that drops in alongside the
plain stateful `chunk_0.mlpackage`. The Swift generator routes through
this variant whenever an image is present, leaving every other chunk
unchanged.

Inputs (vs plain stateful chunk_0):
  hidden_in    (1, 1, hidden) fp16          — same
  cos, sin     (1, 1, head_dim) fp16        — same (Swift writes mRoPE
                                              for image tokens, 1D for text)
  causal_mask  (1, 1, 1, max_seq) fp16      — same
  current_pos  (1,) int32                   — same
  ds_0,ds_1,ds_2 (1, 1, hidden) fp16        — DeepStack features for layers 0,1,2
  visual_active  (1,) fp32                  — 1.0 on image-pad steps, 0.0 otherwise
  state kv_cache_0                          — same as stateful chunk_0
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

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import MODEL_DTYPE
from build_qwen3_vl_2b_stateful_chunks import (
    ANEStatefulDecoderLayer, load_text_config, load_text_backbone,
)


LAYERS_PER_CHUNK = 7
DEEPSTACK_LAYER_COUNT = 3


class DeepStackStatefulChunk0(nn.Module):
    """Stateful chunk_0 with DeepStack injection at layers 0/1/2."""
    def __init__(self, cfg, hf_layers, max_seq):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.max_seq = max_seq
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

        layers_in_chunk = LAYERS_PER_CHUNK
        self.layers = nn.ModuleList([
            ANEStatefulDecoderLayer(cfg, hf_layers[li], max_seq, li)
            for li in range(layers_in_chunk)
        ])

        # Same unified KV cache shape as plain stateful chunk_0
        # (so swapping in/out doesn't change the state shape — Swift
        # MLState handles created from chunk_0 are interchangeable).
        self.register_buffer(
            "kv_cache_0",
            torch.zeros(2 * layers_in_chunk, cfg.num_key_value_heads,
                        max_seq, cfg.head_dim, dtype=MODEL_DTYPE),
        )

    def forward(self, hidden_in, cos, sin, causal_mask, current_pos,
                ds_0, ds_1, ds_2, visual_active):
        # (1, 1, hidden) → Conv2d layout (1, hidden, 1, 1)
        h = hidden_in.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
        deepstack = [ds_0, ds_1, ds_2]
        # visual_active fp32 (1,) → fp16 (1, 1, 1, 1) gate
        gate = visual_active.to(MODEL_DTYPE).view(1, 1, 1, 1)
        for li, layer in enumerate(self.layers):
            h = layer(h, cos, sin, causal_mask, current_pos, self.kv_cache_0)
            if li < DEEPSTACK_LAYER_COUNT:
                ds = deepstack[li]
                ds_conv = ds.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
                h = h + gate * ds_conv
        return h.permute(0, 2, 3, 1).reshape(1, 1, self.hidden_size)


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


def convert(chunk, cfg, max_seq, out_path: Path):
    print(f"\n--- convert STATEFUL chunk_0_vision ---")
    head_dim = cfg.head_dim
    hidden = cfg.hidden_size

    example = (
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),       # hidden_in
        torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE),      # cos
        torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE),      # sin
        torch.zeros(1, 1, 1, max_seq, dtype=MODEL_DTYPE),    # causal_mask
        torch.zeros(1, dtype=torch.int32),                   # current_pos
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),        # ds_0
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),        # ds_1
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),        # ds_2
        torch.zeros(1, dtype=torch.float32),                 # visual_active
    )
    t0 = time.time()
    traced = torch.jit.trace(chunk, example, strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="cos", shape=(1, 1, head_dim), dtype=np.float16),
        ct.TensorType(name="sin", shape=(1, 1, head_dim), dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, max_seq), dtype=np.float16),
        ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
        ct.TensorType(name="ds_0", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="ds_1", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="ds_2", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="visual_active", shape=(1,), dtype=np.float32),
    ]
    ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
    state_shape = (2 * LAYERS_PER_CHUNK, cfg.num_key_value_heads,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True,
                    help="Parent dir; mlpackage lands under "
                         "qwen3_vl_2b_stateful_chunks/chunk_0_vision.mlpackage")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--keep-fp16", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    chunks_dir = out_root / "qwen3_vl_2b_stateful_chunks"
    fp16_dir = out_root / "_fp16_intermediate"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    print("loading Qwen3-VL 2B text backbone (fp32)...")
    cfg = load_text_config()
    text_model, _ = load_text_backbone()
    chunk0 = DeepStackStatefulChunk0(cfg, text_model.layers, args.max_seq) \
        .eval().to(MODEL_DTYPE)
    del text_model

    fp16_path = fp16_dir / "chunk_0_vision.mlpackage"
    final_path = chunks_dir / "chunk_0_vision.mlpackage"
    convert(chunk0, cfg, args.max_seq, fp16_path)
    if args.nbits == 0:
        if final_path.exists():
            shutil.rmtree(final_path)
        shutil.move(str(fp16_path), str(final_path))
    else:
        palettize_pkg(fp16_path, final_path, args.nbits)

    if not args.keep_fp16:
        shutil.rmtree(fp16_dir, ignore_errors=True)

    print(f"\n✓ shipping artifact: {final_path}")


if __name__ == "__main__":
    main()
