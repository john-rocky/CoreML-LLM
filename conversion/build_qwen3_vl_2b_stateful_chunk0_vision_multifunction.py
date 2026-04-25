"""Qwen3-VL 2B chunk_0_vision multifunction (infer T=1 + prefill_b<N> T=N).

Drop-in replacement for `chunk_0_vision.mlpackage` that carries two
functions sharing one `kv_cache_0` ct.StateType:

  infer       — T=1 (matches build_qwen3_vl_2b_stateful_chunk0_vision)
  prefill_b<N> — T=N batched, used to fast-prefill the 196 image-pad
                  tokens at the start of a vision prompt.

DeepStack injection at layers 0/1/2: ds_0/1/2 are (1, T, hidden) fp16,
visual_active is fp32 (1,) and broadcasts onto the Conv2d-shaped
hidden so a single scalar gate gates all T positions of the batch.
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
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
from coremltools.optimize.coreml import (
    OpPalettizerConfig, OptimizationConfig, palettize_weights,
)

sys.path.insert(0, str(Path(__file__).parent))
from ane_ops import MODEL_DTYPE
from build_qwen3_vl_2b_stateful_chunks import (
    ANEStatefulDecoderLayer, load_text_config, load_text_backbone,
)
from build_qwen3_vl_2b_stateful_multifunction import (
    ANEStatefulPrefillLayer,
)
from build_qwen3_vl_2b_stateful_chunk0_vision import DeepStackStatefulChunk0


LAYERS_PER_CHUNK = 7
DEEPSTACK_LAYER_COUNT = 3


class DeepStackStatefulPrefillChunk0(nn.Module):
    """T-batched stateful chunk_0 with DeepStack injection at layers 0/1/2."""

    def __init__(self, cfg, hf_layers, max_seq, T):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.max_seq = max_seq
        self.T = T
        layers_in_chunk = LAYERS_PER_CHUNK
        self.layers = nn.ModuleList([
            ANEStatefulPrefillLayer(cfg, hf_layers[li], max_seq, li, T)
            for li in range(layers_in_chunk)
        ])
        self.register_buffer(
            "kv_cache_0",
            torch.zeros(2 * layers_in_chunk, cfg.num_key_value_heads,
                        max_seq, cfg.head_dim, dtype=MODEL_DTYPE),
        )

    def forward(self, hidden_in, cos, sin, causal_mask, current_pos,
                ds_0, ds_1, ds_2, visual_active):
        # (1, T, hidden) → Conv2d (1, hidden, 1, T)
        T = self.T
        h = hidden_in.reshape(1, T, 1, self.hidden_size).permute(0, 3, 2, 1)
        deepstack = [ds_0, ds_1, ds_2]
        gate = visual_active.to(MODEL_DTYPE).view(1, 1, 1, 1)
        for li, layer in enumerate(self.layers):
            h = layer(h, cos, sin, causal_mask, current_pos, self.kv_cache_0)
            if li < DEEPSTACK_LAYER_COUNT:
                ds = deepstack[li]
                # ds is (1, T, hidden) → conv form (1, hidden, 1, T)
                ds_conv = ds.reshape(1, T, 1, self.hidden_size).permute(0, 3, 2, 1)
                h = h + gate * ds_conv
        # (1, hidden, 1, T) → (1, T, hidden)
        return h.permute(0, 3, 2, 1).reshape(1, T, self.hidden_size)


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


def convert_decode(chunk, cfg, max_seq, out_path: Path):
    print(f"\n--- convert chunk_0_vision DECODE (T=1) ---")
    head_dim = cfg.head_dim
    hidden = cfg.hidden_size
    example = (
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),
        torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE),
        torch.zeros(1, 1, head_dim, dtype=MODEL_DTYPE),
        torch.zeros(1, 1, 1, max_seq, dtype=MODEL_DTYPE),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),
        torch.zeros(1, 1, hidden, dtype=MODEL_DTYPE),
        torch.zeros(1, dtype=torch.float32),
    )
    traced = torch.jit.trace(chunk, example, strict=False)
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
    ct_states = [ct.StateType(
        wrapped_type=ct.TensorType(shape=state_shape, dtype=np.float16),
        name="kv_cache_0",
    )]
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs, states=ct_states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    ct_model.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob('*')
                   if f.is_file()) / 1e6
    print(f"  saved fp16 {out_path.name} ({size_mb:.0f} MB)")
    _audit_ane(out_path)


def convert_prefill(chunk, cfg, max_seq, T, out_path: Path):
    print(f"\n--- convert chunk_0_vision PREFILL (T={T}) ---")
    head_dim = cfg.head_dim
    hidden = cfg.hidden_size
    example = (
        torch.zeros(1, T, hidden, dtype=MODEL_DTYPE),
        torch.zeros(1, T, head_dim, dtype=MODEL_DTYPE),
        torch.zeros(1, T, head_dim, dtype=MODEL_DTYPE),
        torch.zeros(1, 1, T, max_seq, dtype=MODEL_DTYPE),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, T, hidden, dtype=MODEL_DTYPE),
        torch.zeros(1, T, hidden, dtype=MODEL_DTYPE),
        torch.zeros(1, T, hidden, dtype=MODEL_DTYPE),
        torch.zeros(1, dtype=torch.float32),
    )
    traced = torch.jit.trace(chunk, example, strict=False)
    ct_inputs = [
        ct.TensorType(name="hidden_in", shape=(1, T, hidden), dtype=np.float16),
        ct.TensorType(name="cos", shape=(1, T, head_dim), dtype=np.float16),
        ct.TensorType(name="sin", shape=(1, T, head_dim), dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, T, max_seq), dtype=np.float16),
        ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
        ct.TensorType(name="ds_0", shape=(1, T, hidden), dtype=np.float16),
        ct.TensorType(name="ds_1", shape=(1, T, hidden), dtype=np.float16),
        ct.TensorType(name="ds_2", shape=(1, T, hidden), dtype=np.float16),
        ct.TensorType(name="visual_active", shape=(1,), dtype=np.float32),
    ]
    ct_outputs = [ct.TensorType(name="hidden", dtype=np.float16)]
    state_shape = (2 * LAYERS_PER_CHUNK, cfg.num_key_value_heads,
                    max_seq, cfg.head_dim)
    ct_states = [ct.StateType(
        wrapped_type=ct.TensorType(shape=state_shape, dtype=np.float16),
        name="kv_cache_0",
    )]
    ct_model = ct.convert(
        traced, convert_to="mlprogram",
        inputs=ct_inputs, outputs=ct_outputs, states=ct_states,
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


def merge(decode_pkg: Path, prefill_pkg: Path, out_path: Path, T: int):
    print(f"\n--- merging into multifunction → {out_path.name} ---")
    desc = MultiFunctionDescriptor()
    desc.add_function(str(decode_pkg), src_function_name="main",
                       target_function_name="infer")
    desc.add_function(str(prefill_pkg), src_function_name="main",
                       target_function_name=f"prefill_b{T}")
    desc.default_function_name = "infer"
    if out_path.exists():
        shutil.rmtree(out_path)
    save_multifunction(desc, str(out_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--prefill-T", type=int, default=8)
    ap.add_argument("--nbits", type=int, default=8, choices=[0, 4, 8])
    ap.add_argument("--keep-fp16", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    chunks_dir = out_root / "qwen3_vl_2b_stateful_chunks"
    fp16_dir = out_root / "_fp16_intermediate"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    fp16_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_text_config()
    text_model, _ = load_text_backbone()

    decode_mod = DeepStackStatefulChunk0(cfg, text_model.layers, args.max_seq) \
        .eval().to(MODEL_DTYPE)
    decode_fp16 = fp16_dir / "chunk_0_vision_decode.mlpackage"
    convert_decode(decode_mod, cfg, args.max_seq, decode_fp16)
    del decode_mod

    prefill_mod = DeepStackStatefulPrefillChunk0(
        cfg, text_model.layers, args.max_seq, args.prefill_T
    ).eval().to(MODEL_DTYPE)
    prefill_fp16 = fp16_dir / "chunk_0_vision_prefill.mlpackage"
    convert_prefill(prefill_mod, cfg, args.max_seq, args.prefill_T, prefill_fp16)
    del prefill_mod, text_model

    if args.nbits != 0:
        decode_int = fp16_dir / f"chunk_0_vision_decode_int{args.nbits}.mlpackage"
        prefill_int = fp16_dir / f"chunk_0_vision_prefill_int{args.nbits}.mlpackage"
        palettize_pkg(decode_fp16, decode_int, args.nbits)
        palettize_pkg(prefill_fp16, prefill_int, args.nbits)
        src_decode, src_prefill = decode_int, prefill_int
    else:
        src_decode, src_prefill = decode_fp16, prefill_fp16

    final_path = chunks_dir / "chunk_0_vision.mlpackage"
    merge(src_decode, src_prefill, final_path, args.prefill_T)
    _audit_ane(final_path)

    if not args.keep_fp16:
        shutil.rmtree(fp16_dir, ignore_errors=True)

    size_mb = sum(f.stat().st_size for f in final_path.rglob('*')
                   if f.is_file()) / 1e6
    print(f"\n✓ shipping artifact: {final_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
