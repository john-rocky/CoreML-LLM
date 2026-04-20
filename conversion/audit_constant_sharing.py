#!/usr/bin/env python3
"""S4: audit large constants duplicated across decode chunks.

Walks each chunk{1..4}.mlpackage in a model directory, lists every
constant >= --min-bytes (default 1 MiB), groups by content hash, and
reports any constant that appears in two or more chunks. The intent is
to find candidates to externalise as shared inputs (RoPE tables,
embedding sub-tables, etc.) so that CoreML can map them once and keep
them resident in ANE SRAM across chunks.

Output:
  - human-readable summary on stdout
  - optional JSON report via --report-json

Does NOT mutate any mlpackage. Use the report to decide which constants
to lift into shared inputs in the next conversion pass.

See docs/LITERT_PERF_ADOPTIONS.md §S4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import coremltools as ct
    from coremltools.proto import MIL_pb2
except ImportError:
    print("ERROR: coremltools is required. `pip install coremltools`.", file=sys.stderr)
    sys.exit(2)


def _const_bytes(op) -> int:
    """Approximate the byte size of a const op's value tensor."""
    try:
        v = op.attributes["val"].immediateValue
        # Most numeric tensors land here as Tensor with raw bytes.
        if v.HasField("tensor"):
            t = v.tensor
            if t.bytes:
                return len(t.bytes.values)
            # Fall back to per-dtype element counting.
            shape_dims = list(t.dimension) if t.dimension else []
            count = 1
            for d in shape_dims:
                count *= max(d, 1)
            # Guess element size by which value list is populated.
            for fname in ("floats", "ints", "doubles", "bools"):
                if t.HasField(fname) or len(getattr(t, fname).values) > 0:
                    if fname == "floats":
                        return count * 4
                    if fname == "doubles":
                        return count * 8
                    if fname == "ints":
                        return count * 8
                    if fname == "bools":
                        return count
            return 0
    except Exception:
        pass
    return 0


def _const_digest(op) -> str | None:
    """SHA1 of the const tensor's raw bytes if available, else None."""
    try:
        v = op.attributes["val"].immediateValue
        if v.HasField("tensor") and v.tensor.bytes:
            return hashlib.sha1(v.tensor.bytes.values).hexdigest()
    except Exception:
        pass
    return None


def walk_block(block, on_op):
    for op in block.operations:
        on_op(op)
        for nested in op.blocks:
            walk_block(nested, on_op)


def collect_constants(mlpackage: Path, min_bytes: int) -> list[dict[str, Any]]:
    """Return a list of {name, op, bytes, digest} for every const >= min_bytes."""
    spec = ct.utils.load_spec(str(mlpackage))
    if not spec.HasField("mlProgram"):
        return []
    prog = spec.mlProgram
    out: list[dict[str, Any]] = []

    const_ops = {
        "const",
        "constexpr_lut_to_dense",
        "constexpr_affine_dequantize",
        "constexpr_blockwise_shift_scale",
        "constexpr_sparse_to_dense",
        "constexpr_cast",
    }

    def visit(op):
        if op.type not in const_ops:
            return
        nbytes = _const_bytes(op)
        if nbytes < min_bytes:
            return
        digest = _const_digest(op) or f"opaque:{op.outputs[0].name}"
        name = op.outputs[0].name if op.outputs else "?"
        out.append({"name": name, "op": op.type, "bytes": nbytes, "digest": digest})

    for fn in prog.functions.values():
        for blk in fn.block_specializations.values():
            walk_block(blk, visit)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True,
                    help="Directory containing chunk{1..4}.mlpackage")
    ap.add_argument("--chunks", default="chunk1,chunk2,chunk3,chunk4",
                    help="Comma-separated chunk basenames")
    ap.add_argument("--min-bytes", type=int, default=1 << 20,
                    help="Minimum constant size to consider (default 1 MiB)")
    ap.add_argument("--report-json", default=None,
                    help="Optional path to write the full report as JSON")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    chunk_names = [c.strip() for c in args.chunks.split(",") if c.strip()]

    per_chunk: dict[str, list[dict[str, Any]]] = {}
    for c in chunk_names:
        pkg = model_dir / f"{c}.mlpackage"
        if not pkg.exists():
            print(f"WARN: {pkg} not found, skipping")
            continue
        per_chunk[c] = collect_constants(pkg, args.min_bytes)
        print(f"  {c}: {len(per_chunk[c])} large const(s) >= {args.min_bytes} bytes")

    # Group by digest across chunks.
    by_digest: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for c, entries in per_chunk.items():
        for e in entries:
            by_digest[e["digest"]].append((c, e))

    duplicates = {d: items for d, items in by_digest.items() if len(items) >= 2}

    print()
    if not duplicates:
        print("No large constants are shared across chunks.")
    else:
        total_dup_bytes = 0
        print("Large constants present in >=2 chunks (candidates to externalise):")
        for d, items in sorted(duplicates.items(),
                               key=lambda kv: -kv[1][0][1]["bytes"]):
            sz = items[0][1]["bytes"]
            redundant_copies = len(items) - 1
            total_dup_bytes += sz * redundant_copies
            chunks = ", ".join(c for c, _ in items)
            sample_name = items[0][1]["name"]
            print(f"  digest={d[:12]} bytes={sz:>10}  copies={len(items)} "
                  f"chunks=[{chunks}] sample_name={sample_name}")
        print(f"\nTotal redundant bytes (if all duplicates externalised): "
              f"{total_dup_bytes:,} ({total_dup_bytes / (1<<20):.1f} MiB)")

    if args.report_json:
        report = {
            "model_dir": str(model_dir),
            "min_bytes": args.min_bytes,
            "per_chunk": per_chunk,
            "duplicates": [
                {
                    "digest": d,
                    "bytes": items[0][1]["bytes"],
                    "occurrences": [{"chunk": c, "name": e["name"], "op": e["op"]}
                                    for c, e in items],
                }
                for d, items in duplicates.items()
            ],
        }
        Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written to {args.report_json}")


if __name__ == "__main__":
    main()
