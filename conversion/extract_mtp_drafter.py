#!/usr/bin/env python3
"""Extract MTP drafter TFLite from .litertlm container.

Scans for TFL3 magic bytes at aligned offsets and extracts each TFLite
section. Also parses the TOML manifest if present.

Usage:
    python conversion/extract_mtp_drafter.py \
        --input ~/.cache/huggingface/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/*/gemma-4-E2B-it.litertlm \
        --output-dir output/mtp_probe/
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path


# TFLite FlatBuffer magic: offset 4-7 = "TFL3"
TFLITE_MAGIC = b"TFL3"
SCAN_ALIGNMENT = 4  # TFLite files are 4-byte aligned


def find_tflite_sections(data: bytes) -> list[tuple[int, int]]:
    """Find all TFLite model sections by scanning for TFL3 magic bytes.

    Returns list of (offset, size) tuples.
    """
    sections = []
    i = 0
    while i < len(data) - 8:
        # TFLite FlatBuffer: bytes 4-7 are the file identifier "TFL3"
        if data[i + 4 : i + 8] == TFLITE_MAGIC:
            # Read the root table offset at bytes 0-3 (little-endian uint32)
            root_offset = struct.unpack_from("<I", data, i)[0]

            # Heuristic for size: scan forward until the next TFL3 or end
            # Better: read the FlatBuffer size from the file
            # The actual file size is harder to determine from FlatBuffer alone,
            # so we find the next section boundary
            next_section = len(data)
            j = i + SCAN_ALIGNMENT
            while j < len(data) - 8:
                if data[j + 4 : j + 8] == TFLITE_MAGIC:
                    # Walk backwards from next TFL3 to find actual boundary
                    # (there may be padding/zeros between sections)
                    next_section = j
                    break
                j += SCAN_ALIGNMENT
            size = next_section - i
            sections.append((i, size))
            i = next_section
        else:
            i += SCAN_ALIGNMENT
    return sections


def extract_toml_manifest(data: bytes) -> str | None:
    """Try to find and extract a TOML manifest from the container."""
    # Look for [[section]] pattern which indicates TOML
    markers = [b"[[section]]", b"[model]", b"model_type"]
    for marker in markers:
        idx = data.find(marker)
        if idx >= 0:
            # Walk backwards to find start of text section
            start = idx
            while start > 0 and data[start - 1 : start] != b"\x00":
                start -= 1
            # Walk forward to find end
            end = idx
            while end < len(data) and data[end : end + 1] != b"\x00":
                end += 1
            return data[start:end].decode("utf-8", errors="replace")
    return None


def inspect_tflite_signature(filepath: str) -> dict:
    """Read TFLite file and extract input/output tensor info."""
    info = {"inputs": [], "outputs": []}

    with open(filepath, "rb") as f:
        buf = f.read()

    # Try using flatbuffers/tflite schema if available
    try:
        import tflite
        model = tflite.Model.GetRootAs(buf, 0)

        # Get signature defs
        for si in range(model.SignatureDefsLength()):
            sig = model.SignatureDefs(si)
            sig_key = sig.SignatureKey().decode() if sig.SignatureKey() else f"sig_{si}"

            # Inputs
            for ii in range(sig.InputsLength()):
                inp = sig.Inputs(ii)
                name = inp.Name().decode() if inp.Name() else f"input_{ii}"
                tensor_idx = inp.TensorIndex()

                # Get tensor details from subgraph
                subgraph = model.Subgraphs(sig.SubgraphIndex())
                tensor = subgraph.Tensors(tensor_idx)
                shape = [tensor.Shape(d) for d in range(tensor.ShapeLength())]
                dtype_map = {0: "fp32", 1: "fp16", 2: "int32", 3: "uint8",
                             7: "int8", 9: "bool", 15: "int16"}
                dtype = dtype_map.get(tensor.Type(), f"type_{tensor.Type()}")
                info["inputs"].append({
                    "name": name, "shape": shape, "dtype": dtype,
                    "signature": sig_key
                })

            # Outputs
            for oi in range(sig.OutputsLength()):
                out = sig.Outputs(oi)
                name = out.Name().decode() if out.Name() else f"output_{oi}"
                tensor_idx = out.TensorIndex()

                subgraph = model.Subgraphs(sig.SubgraphIndex())
                tensor = subgraph.Tensors(tensor_idx)
                shape = [tensor.Shape(d) for d in range(tensor.ShapeLength())]
                dtype_map = {0: "fp32", 1: "fp16", 2: "int32", 3: "uint8",
                             7: "int8", 9: "bool", 15: "int16"}
                dtype = dtype_map.get(tensor.Type(), f"type_{tensor.Type()}")
                info["outputs"].append({
                    "name": name, "shape": shape, "dtype": dtype,
                    "signature": sig_key
                })

        # Count ops and tensors
        if model.SubgraphsLength() > 0:
            sg = model.Subgraphs(0)
            info["num_tensors"] = sg.TensorsLength()
            info["num_operators"] = sg.OperatorsLength()
            info["num_subgraphs"] = model.SubgraphsLength()

            # Collect tensor names for weight mapping
            tensor_names = []
            for ti in range(sg.TensorsLength()):
                t = sg.Tensors(ti)
                tname = t.Name().decode() if t.Name() else f"tensor_{ti}"
                tshape = [t.Shape(d) for d in range(t.ShapeLength())]
                dtype_map = {0: "fp32", 1: "fp16", 2: "int32", 3: "uint8",
                             7: "int8", 9: "bool", 15: "int16"}
                tdtype = dtype_map.get(t.Type(), f"type_{t.Type()}")
                # Check if it has a buffer (i.e., is a weight, not an activation)
                buf_idx = t.Buffer()
                has_data = False
                if buf_idx > 0 and buf_idx < model.BuffersLength():
                    b = model.Buffers(buf_idx)
                    has_data = b.DataLength() > 0
                tensor_names.append({
                    "idx": ti, "name": tname, "shape": tshape,
                    "dtype": tdtype, "has_data": has_data
                })
            info["tensors"] = tensor_names

    except ImportError:
        print("  WARN: 'tflite' package not installed; skipping detailed inspection")
        print("  Install with: pip install tflite")

    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True,
                    help="Path to .litertlm container file")
    ap.add_argument("--output-dir", type=str, default="output/mtp_probe/")
    args = ap.parse_args()

    input_path = args.input
    # Expand glob if needed
    if "*" in input_path:
        import glob as gl
        matches = sorted(gl.glob(input_path))
        if not matches:
            print(f"ERROR: no files match {input_path}")
            return
        input_path = matches[0]

    print(f"Reading container: {input_path}")
    print(f"  Size: {os.path.getsize(input_path) / 1e9:.2f} GB")

    with open(input_path, "rb") as f:
        data = f.read()

    # Extract TOML manifest
    print("\n=== TOML Manifest ===")
    toml_text = extract_toml_manifest(data)
    if toml_text:
        print(toml_text[:3000])
        # Save manifest
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "manifest.toml"), "w") as f:
            f.write(toml_text)
        print(f"\n  Saved to {args.output_dir}/manifest.toml")
    else:
        print("  No TOML manifest found")

    # Find all TFLite sections
    print("\n=== TFLite Sections ===")
    sections = find_tflite_sections(data)
    print(f"  Found {len(sections)} TFLite sections\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for i, (offset, size) in enumerate(sections):
        size_mb = size / 1e6
        out_path = os.path.join(args.output_dir, f"section_{i}.tflite")

        print(f"  Section {i}: offset={offset:#010x} size={size_mb:.1f} MB")

        # Extract to file
        with open(out_path, "wb") as f:
            f.write(data[offset : offset + size])

        # Inspect if tflite package is available
        sig_info = inspect_tflite_signature(out_path)
        if sig_info.get("inputs"):
            for inp in sig_info["inputs"]:
                print(f"    Input:  {inp['name']:30s} {str(inp['shape']):20s} {inp['dtype']}")
            for out in sig_info["outputs"]:
                print(f"    Output: {out['name']:30s} {str(out['shape']):20s} {out['dtype']}")
            if "num_tensors" in sig_info:
                print(f"    Tensors: {sig_info['num_tensors']}, "
                      f"Operators: {sig_info['num_operators']}, "
                      f"Subgraphs: {sig_info['num_subgraphs']}")

            # Check if this is the mtp_drafter
            input_names = {inp["name"] for inp in sig_info["inputs"]}
            if "activations" in input_names:
                print(f"    >>> MTP DRAFTER DETECTED <<<")

                # Save weight tensor listing for rename map
                if "tensors" in sig_info:
                    weights = [t for t in sig_info["tensors"] if t["has_data"]]
                    print(f"    Weight tensors: {len(weights)}")
                    weight_path = os.path.join(args.output_dir, f"section_{i}_weights.txt")
                    with open(weight_path, "w") as wf:
                        for t in weights:
                            wf.write(f"{t['idx']:4d}  {t['name']:60s}  {str(t['shape']):20s}  {t['dtype']}\n")
                    print(f"    Weight listing saved to {weight_path}")

        print()


if __name__ == "__main__":
    main()
