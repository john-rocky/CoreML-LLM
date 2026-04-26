#!/usr/bin/env python3.12
"""Print a size breakdown of one or more CoreML mlpackage files.

Reports per-component bytes (proto vs weight.bin), tensor-level top-N,
and an attempt to bucket tensors into roles (embedding / lm_head /
attention / mlp / norm / rope / other) using name patterns common in
the Gemma 4 stateful chunks emitted by
`conversion/build_gemma4_e2b_stateful_chunks.py`.

Usage:
    python3.12 scripts/print_coreml_size_breakdown.py <mlpackage> [<mlpackage> ...]

Output: stdout, plain text. No files written.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import coremltools as ct


WEIGHT_HEADER_SENTINEL = 0xDEADBEEF
WEIGHT_HEADER_BYTES = 32

BUCKET_PATTERNS = [
    ("token_embedding",  re.compile(r"embed.*token|tok_embed|wte|input_embed")),
    ("lm_head",          re.compile(r"lm_head|output_proj_logits|logits_proj|to_logits")),
    ("per_layer_embed",  re.compile(r"per_layer|ple")),
    ("rope",             re.compile(r"rope|cos[_/]|sin[_/]")),
    ("attn_q_norm",      re.compile(r"q_norm")),
    ("attn_k_norm",      re.compile(r"k_norm")),
    ("rmsnorm",          re.compile(r"layernorm|rmsnorm|input_layernorm|post_attention_layernorm|post_feedforward_layernorm|pre_feedforward_layernorm|final_layernorm|model_norm|\bnorm\b")),
    ("attn_q",           re.compile(r"q_proj|to_q|wq|attn.*q\b")),
    ("attn_k",           re.compile(r"k_proj|to_k|wk|attn.*k\b")),
    ("attn_v",           re.compile(r"v_proj|to_v|wv|attn.*v\b")),
    ("attn_o",           re.compile(r"o_proj|out_proj|wo|attn.*output")),
    ("mlp_gate",         re.compile(r"gate_proj|gate1")),
    ("mlp_up",           re.compile(r"up_proj|gate2")),
    ("mlp_down",         re.compile(r"down_proj")),
    ("kv_palettize_lut", re.compile(r"lut|palettize|cluster|centroid")),
    ("mask_or_const",    re.compile(r"mask|update_indicator|causal")),
]


def bucket_for(name: str) -> str:
    nl = name.lower()
    for label, pat in BUCKET_PATTERNS:
        if pat.search(nl):
            return label
    return "other"


def fmt_bytes(n: int) -> str:
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n} B"


def read_blob_size(weights_dir: Path, fileName: str, offset: int) -> int:
    """Read the size of a weight blob given its 0xDEADBEEF-prefixed header offset."""
    rel = os.path.basename(fileName)
    path = weights_dir / rel
    with open(path, "rb") as f:
        f.seek(offset)
        h = f.read(WEIGHT_HEADER_BYTES)
    if len(h) < WEIGHT_HEADER_BYTES:
        return 0
    sentinel = int.from_bytes(h[0:4], "little")
    if sentinel != WEIGHT_HEADER_SENTINEL:
        return 0
    size = int.from_bytes(h[8:16], "little")
    return size


def collect_const_rows(spec, weights_dir: Path) -> list[tuple[str, str, str, str, int, set[str]]]:
    """Walk all weight references (const ops + constexpr inline blobs).

    Deduplicated by physical blob (fileName, offset) so weights shared
    between functions are counted once. Returns
    (any_function, op_type, output_name, role, byte_size, functions_set).
    """
    # blob_key -> (op_type, name, role, size, functions)
    seen: dict[tuple[str, int], list] = {}
    if spec.WhichOneof("Type") != "mlProgram":
        return []

    def record(fn_name, op_type, out_name, role, blob):
        rel = os.path.basename(blob.fileName)
        key = (rel, blob.offset)
        if key not in seen:
            sz = read_blob_size(weights_dir, blob.fileName, blob.offset)
            seen[key] = [op_type, out_name, role, sz, set()]
        seen[key][4].add(fn_name)

    for fn_name, fn in spec.mlProgram.functions.items():
        for bs_name, block in fn.block_specializations.items():
            for op in block.operations:
                out_name = op.outputs[0].name if op.outputs else "?"
                if op.type == "const":
                    if "val" in op.attributes:
                        v = op.attributes["val"]
                        if v.WhichOneof("value") == "blobFileValue":
                            record(fn_name, "const", out_name, "val", v.blobFileValue)
                elif op.type.startswith("constexpr_"):
                    for inp_name, arg_list in op.inputs.items():
                        for arg in arg_list.arguments:
                            if arg.WhichOneof("binding") != "value":
                                continue
                            inner = arg.value
                            if inner.WhichOneof("value") != "blobFileValue":
                                continue
                            record(fn_name, op.type, out_name, inp_name, inner.blobFileValue)

    rows: list[tuple[str, str, str, str, int, set[str]]] = []
    for (rel, off), (op_type, name, role, sz, fns) in seen.items():
        rows.append((rel, op_type, name, role, sz, fns))
    return rows


def analyse(mlpackage: Path):
    print(f"\n## {mlpackage}")
    if not mlpackage.exists():
        print("  (missing)")
        return

    weights_dir = mlpackage / "Data" / "com.apple.CoreML" / "weights"
    proto_path  = mlpackage / "Data" / "com.apple.CoreML" / "model.mlmodel"

    total = sum(p.stat().st_size for p in mlpackage.rglob("*") if p.is_file())
    proto_size = proto_path.stat().st_size if proto_path.exists() else 0
    weights_size = sum(p.stat().st_size for p in weights_dir.glob("weight*.bin")) if weights_dir.exists() else 0

    print(f"  total    : {fmt_bytes(total):>10s}")
    print(f"  proto    : {fmt_bytes(proto_size):>10s}")
    print(f"  weights  : {fmt_bytes(weights_size):>10s}")
    print()

    spec = ct.utils.load_spec(str(mlpackage))
    print(f"  specVer={spec.specificationVersion}  defaultFn={spec.description.defaultFunctionName}")
    fn_names = [d.name for d in spec.description.functions]
    print(f"  functions: {fn_names}")
    print()

    # Per-function I/O + state
    for fd in spec.description.functions:
        n_state = len(fd.state)
        print(f"  function `{fd.name}`  inputs={len(fd.input)}  outputs={len(fd.output)}  states={n_state}")

    # Const + constexpr tensors (deduplicated by physical blob)
    rows = collect_const_rows(spec, weights_dir)
    rows.sort(key=lambda r: -r[4])
    by_bucket: dict[str, int] = {}
    by_op_type: dict[str, int] = {}
    by_role: dict[str, int] = {}
    total_const = 0
    for rel, op_type, name, role, sz, fns in rows:
        by_bucket[bucket_for(name)] = by_bucket.get(bucket_for(name), 0) + sz
        by_op_type[op_type] = by_op_type.get(op_type, 0) + sz
        by_role[role] = by_role.get(role, 0) + sz
        total_const += sz

    print()
    print(f"  unique weight blobs: {len(rows)}, total bytes: {fmt_bytes(total_const)}")
    print(f"    (vs weight.bin file size: {fmt_bytes(weights_size)} — diff is alignment padding)")
    print()
    print(f"  by op type:")
    for op_t in sorted(by_op_type, key=lambda k: -by_op_type[k]):
        bb = by_op_type[op_t]
        pct = (100.0 * bb / total_const) if total_const else 0
        print(f"    {op_t:30s} {fmt_bytes(bb):>10s}  ({pct:5.1f}%)")
    print()
    print(f"  by argument role:")
    for r in sorted(by_role, key=lambda k: -by_role[k]):
        bb = by_role[r]
        pct = (100.0 * bb / total_const) if total_const else 0
        print(f"    {r:30s} {fmt_bytes(bb):>10s}  ({pct:5.1f}%)")
    print()
    print(f"  bucket breakdown (by tensor-name pattern):")
    for label in sorted(by_bucket, key=lambda k: -by_bucket[k]):
        bb = by_bucket[label]
        pct = (100.0 * bb / total_const) if total_const else 0
        print(f"    {label:30s} {fmt_bytes(bb):>10s}  ({pct:5.1f}%)")

    print()
    top_n = 15
    print(f"  top {top_n} weight blobs by size:")
    for rel, op_type, name, role, sz, fns in rows[:top_n]:
        fn_str = ",".join(sorted(fns))
        print(f"    {fmt_bytes(sz):>10s}  [{fn_str}]  {op_type}/{role}  {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlpackages", nargs="+")
    args = ap.parse_args()
    for p in args.mlpackages:
        analyse(Path(p))


if __name__ == "__main__":
    main()
