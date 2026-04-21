#!/usr/bin/env python3
"""Dump all TFLite metadata for the MTP drafter: quantization scales,
tensor shapes, op types, any suspicious scalar parameters.

Hunt for:
  1. Scalar/small tensors (shape=(1,), (4,), etc.) that might be layer scalars
  2. Per-channel quantization granularity
  3. QUANTIZE/DEQUANTIZE ops around FC layers
  4. Any tensor we haven't loaded into PyTorch
"""
import sys
import os

TFL_PATH = "/Users/majimadaisuke/Downloads/CoreML-LLM/output/mtp_probe/section_9.tflite"

# Try ai_edge_litert (most complete), fall back to tflite package
try:
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=TFL_PATH)
    interp.allocate_tensors()
    tensor_details = interp.get_tensor_details()
    op_details = None
    try:
        op_details = interp._get_ops_details() if hasattr(interp, '_get_ops_details') else None
    except Exception:
        op_details = None
    print(f"# Loaded via ai_edge_litert, {len(tensor_details)} tensors")
except ImportError:
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=TFL_PATH)
    interp.allocate_tensors()
    tensor_details = interp.get_tensor_details()
    print(f"# Loaded via tensorflow, {len(tensor_details)} tensors")


# Categorize tensors
print("\n=== 1. SMALL/SCALAR TENSORS (candidates for hidden scale params) ===")
small_tensors = []
for t in tensor_details:
    shape = tuple(t["shape"].tolist()) if hasattr(t["shape"], "tolist") else tuple(t["shape"])
    if len(shape) == 0 or (len(shape) == 1 and shape[0] <= 8) or shape == (4,) or shape == (4, 1):
        small_tensors.append(t)

for t in sorted(small_tensors, key=lambda x: x.get("name", "")):
    shape = tuple(t["shape"].tolist()) if hasattr(t["shape"], "tolist") else tuple(t["shape"])
    dtype = t.get("dtype", "?")
    q = t.get("quantization_parameters", {})
    scales = q.get("scales", [])
    q_summary = f"quant_scales={scales[:3].tolist() if hasattr(scales, 'tolist') else list(scales)[:3]}" if len(scales) > 0 else ""
    name = t.get("name", "?")
    # Get actual values if possible
    try:
        val = interp.get_tensor(t["index"])
        val_str = f"val={val.tolist() if val.size <= 8 else val.tolist()[:4]}"
    except Exception as e:
        val_str = f"val=<err {e}>"
    print(f"  {str(shape):15s} {str(dtype):25s} {name:70s}")
    print(f"    {val_str}  {q_summary}")


print("\n=== 2. ALL TENSOR SHAPES (grouped by shape) ===")
from collections import defaultdict
by_shape = defaultdict(list)
for t in tensor_details:
    shape = tuple(t["shape"].tolist()) if hasattr(t["shape"], "tolist") else tuple(t["shape"])
    by_shape[shape].append(t.get("name", "?"))

# Print groups ordered by count
for shape in sorted(by_shape.keys(), key=lambda s: (len(s), s)):
    names = by_shape[shape]
    if len(names) <= 3:
        for n in names:
            print(f"  {str(shape):25s} {n}")
    else:
        print(f"  {str(shape):25s} [{len(names)} tensors, e.g.: {names[0]}, {names[1]}, {names[2]}...]")


print("\n=== 3. QUANTIZATION METADATA ===")
per_channel_count = 0
per_tensor_count = 0
no_quant_count = 0
for t in tensor_details:
    shape = tuple(t["shape"].tolist()) if hasattr(t["shape"], "tolist") else tuple(t["shape"])
    q = t.get("quantization_parameters", {})
    scales = q.get("scales", [])
    scales_list = scales.tolist() if hasattr(scales, "tolist") else list(scales)
    if len(scales_list) > 1:
        per_channel_count += 1
    elif len(scales_list) == 1:
        per_tensor_count += 1
    else:
        no_quant_count += 1

print(f"  Per-channel quant (>1 scale): {per_channel_count}")
print(f"  Per-tensor quant (1 scale):    {per_tensor_count}")
print(f"  No quantization:               {no_quant_count}")

# Show sample per-tensor quant scales (these would be the activation quant scales)
print("\n=== 4. SAMPLE PER-TENSOR QUANT SCALES (likely activation quant boundaries) ===")
samples_shown = 0
for t in tensor_details:
    q = t.get("quantization_parameters", {})
    scales = q.get("scales", [])
    scales_list = scales.tolist() if hasattr(scales, "tolist") else list(scales)
    if len(scales_list) == 1 and samples_shown < 20:
        zp = q.get("zero_points", [0])
        zp_list = zp.tolist() if hasattr(zp, "tolist") else list(zp)
        name = t.get("name", "?")
        shape = tuple(t["shape"].tolist()) if hasattr(t["shape"], "tolist") else tuple(t["shape"])
        print(f"  {name:70s} shape={str(shape):15s} scale={scales_list[0]:.6f} zp={zp_list[0]}")
        samples_shown += 1


print("\n=== 5. LM HEAD QUANTIZATION (what's used for logit computation) ===")
for t in tensor_details:
    name = t.get("name", "")
    if "embedder" in name or "lm_head" in name or "decode_softmax" in name:
        shape = tuple(t["shape"].tolist()) if hasattr(t["shape"], "tolist") else tuple(t["shape"])
        dtype = t.get("dtype", "?")
        q = t.get("quantization_parameters", {})
        scales = q.get("scales", [])
        scales_list = scales.tolist() if hasattr(scales, "tolist") else list(scales)
        zp = q.get("zero_points", [0])
        zp_list = zp.tolist() if hasattr(zp, "tolist") else list(zp)
        print(f"  {name}")
        print(f"    shape={shape} dtype={dtype} n_scales={len(scales_list)}")
        if scales_list:
            print(f"    scale_range=[{min(scales_list):.6f}, {max(scales_list):.6f}] zp_range=[{min(zp_list)}, {max(zp_list)}]")
