"""Qwen3-VL 2B chunk_0 with DeepStack vision injection.

Produces a replacement for the plain `chunk_0.mlpackage` that accepts
three extra deepstack hidden features (from the vision encoder's layers
5, 11, 17 after per-tap PatchMerger) and a gate scalar. When the gate
is 1.0, the chunk adds each deepstack feature to the output of the
corresponding text layer 0/1/2 — matching HF's
`Qwen3VLTextModel._deepstack_process`. When the gate is 0.0 the adds
are no-ops so the same chunk works for both text-only and image steps.

Layer layout (chunk_0 covers text layers [0, 7)):
    layer 0 → deepstack add with ds_0
    layer 1 → deepstack add with ds_1
    layer 2 → deepstack add with ds_2
    layer 3..6 → no injection

Additional inputs (beyond the plain chunk_0 signature):
    ds_0:           fp16 (1, 1, hidden) — vision deepstack slice for this
                    step's visual position, OR zero buffer when the step's
                    token is not a visual token.
    ds_1, ds_2:     same shape, same semantics.
    visual_active:  fp32 (1,) scalar. 1.0 on image-pad steps, 0.0 otherwise.

The Swift generator indexes the vision encoder's DeepStack outputs by
a running `imageTokenIdx` and pokes the current slice into the reusable
ds_N MLMultiArrays. Non-image steps get a zeroed ds + visual_active=0
so the chunk stays on the same fp16 ANE graph.
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

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import MODEL_DTYPE

# Reuse the 2B converter's layer + config helpers wholesale — only the
# chunk wrapper differs.
from build_qwen3_vl_2b_text_decode_chunks import (
    ANEDecoderLayer, load_text_config, load_text_backbone, MODEL_ID,
)


LAYERS_PER_CHUNK = 7
DEEPSTACK_LAYER_COUNT = 3


class DeepStackChunk0(nn.Module):
    """chunk_0 with DeepStack injection points at text layers 0, 1, 2."""
    def __init__(self, cfg, hf_layers, max_seq):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.layers = nn.ModuleList([
            ANEDecoderLayer(cfg, hf_layers[i], max_seq)
            for i in range(LAYERS_PER_CHUNK)
        ])

    def forward(self, hidden_in, position, cos, sin,
                ds_0, ds_1, ds_2, visual_active,
                *kv_states):
        # (1, 1, hidden) → Conv2d layout (1, hidden, 1, 1)
        h = hidden_in.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
        deepstack = [ds_0, ds_1, ds_2]
        # visual_active (1,) fp32 → (1, 1, 1, 1) fp16 so it broadcasts onto
        # the Conv2d-shaped hidden with minimal op count.
        gate = visual_active.to(MODEL_DTYPE).view(1, 1, 1, 1)
        new_states = []
        for local_i, layer in enumerate(self.layers):
            k = kv_states[2 * local_i]
            v = kv_states[2 * local_i + 1]
            h, k_new, v_new = layer(h, position, cos, sin, k, v)
            new_states.append(k_new); new_states.append(v_new)
            if local_i < DEEPSTACK_LAYER_COUNT:
                ds = deepstack[local_i]  # (1, 1, hidden) fp16
                ds_conv = ds.reshape(1, 1, 1, self.hidden_size).permute(0, 3, 1, 2)
                h = h + gate * ds_conv
        h_out = h.permute(0, 2, 3, 1).reshape(1, 1, self.hidden_size)
        return (h_out, *new_states)


def _kv_shape(cfg, max_seq):
    return (1, cfg.num_key_value_heads, max_seq, cfg.head_dim)


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


def convert_chunk0_vision(chunk, cfg, max_seq, out_path):
    print(f"\n--- convert chunk_0 (DeepStack-aware) ---")
    head_dim = cfg.head_dim
    hidden = cfg.hidden_size

    example = [torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE)]          # hidden_in
    example.append(torch.zeros(1, dtype=torch.float32))                # position
    example.append(torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE))     # cos
    example.append(torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE))     # sin
    example.append(torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE))       # ds_0
    example.append(torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE))       # ds_1
    example.append(torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE))       # ds_2
    example.append(torch.zeros(1, dtype=torch.float32))                # visual_active
    for _ in range(LAYERS_PER_CHUNK):
        example.append(torch.zeros(*_kv_shape(cfg, max_seq), dtype=MODEL_DTYPE))
        example.append(torch.zeros(*_kv_shape(cfg, max_seq), dtype=MODEL_DTYPE))

    t0 = time.time()
    traced = torch.jit.trace(chunk, tuple(example), strict=False)
    print(f"  traced in {time.time()-t0:.1f}s")

    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="position", shape=(1,), dtype=np.float32),
        ct.TensorType(name="cos", shape=(1, 1, head_dim), dtype=np.float16),
        ct.TensorType(name="sin", shape=(1, 1, head_dim), dtype=np.float16),
        ct.TensorType(name="ds_0", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="ds_1", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="ds_2", shape=(1, 1, hidden), dtype=np.float16),
        ct.TensorType(name="visual_active", shape=(1,), dtype=np.float32),
    ]
    ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
    for i in range(LAYERS_PER_CHUNK):
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
    t0 = time.time()
    m_out = palettize_weights(m_in, opt_cfg)
    print(f"  palettize done in {time.time()-t0:.1f}s")
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
                         "qwen3_vl_2b_decode_chunks/chunk_0_vision.mlpackage")
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--keep-fp16", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    chunks_dir = out_root / "qwen3_vl_2b_decode_chunks"
    fp16_dir = out_root / "_fp16_intermediate"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading Qwen3-VL 2B text backbone (fp32) for chunk_0...")
    t0 = time.time()
    cfg = load_text_config()
    text_model, _lm_head = load_text_backbone()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Grab just layers 0..6 for chunk_0.
    chunk0 = DeepStackChunk0(cfg, text_model.layers, args.max_seq) \
        .eval().to(MODEL_DTYPE)
    del text_model, _lm_head

    fp16_path = fp16_dir / "chunk_0_vision.mlpackage"
    final_path = chunks_dir / "chunk_0_vision.mlpackage"
    convert_chunk0_vision(chunk0, cfg, args.max_seq, fp16_path)
    if args.nbits == 0:
        shutil.move(str(fp16_path), str(final_path))
    else:
        palettize_pkg(fp16_path, final_path, args.nbits)

    if not args.keep_fp16:
        shutil.rmtree(fp16_dir, ignore_errors=True)

    print(f"\n✓ shipping artifact: {final_path}")


if __name__ == "__main__":
    main()
