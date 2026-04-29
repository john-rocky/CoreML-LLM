"""Bit-exact ternary weight surgery for Bonsai Core ML mlpackages.

Replaces each FP16 weight const in a Core ML mlpackage with a two-op compressed
chain that exactly reproduces the {-s_{r,b}, 0, +s_{r,b}} per-128-block structure
of Bonsai's native 1.58-bit encoding:

    weight[r, b*128+k]
      = scale[r, b]  *  sign_codebook[ indices[r, b*128+k] ]
      = constexpr_blockwise_shift_scale(
            data  = constexpr_lut_to_dense(indices=uint2(...), lut=[0,+1,-1,0]),
            scale = fp16 per-(row,block) scale
        )

This keeps the sign LUT small and shared (1,1,..,4,1) — ANE-friendly — and
factors per-row scale into a separate blockwise op. `reorder_lut_per_channel_scale`
(the coremltools pass that moves scale post-matmul) will apply at compile time if
the downstream is a linear/matmul/conv, letting the ANE run the quantized matmul.

Usage:
    python ternary_surgery.py --src <fp16 mlpackage> --dst <output mlpackage>
    python ternary_surgery.py --src <bundle>/chunk_a.mlpackage \\
                               --dst <bundle>/chunk_a.mlpackage \\
                               --block-size 128
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.models.utils import _apply_graph_pass


SIGN_LUT_VALUES = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float16)


def encode_ternary(w: np.ndarray, block_size: int = 128):
    """Encode a 2D or 4D-with-trailing-1s weight as (indices uint8, scale fp16).

    Returns:
      indices: uint8 array same shape as `w`, values in {0,1,2,3}:
               0 → zero, 1 → +s, 2 → -s, 3 → unused.
      scale:   fp16 array shape (out, in/block_size) for 2D weights, or
               (out, in/block_size, 1, 1) for (out, in, 1, 1) 4D weights.
    Both in same numeric type as the data path uses.
    """
    orig_shape = w.shape
    if w.ndim == 4:
        assert orig_shape[-1] == 1 and orig_shape[-2] == 1, (
            f"only trailing-1 4D weights supported, got {orig_shape}"
        )
        w2 = w.reshape(orig_shape[0], orig_shape[1])
    elif w.ndim == 2:
        w2 = w
    else:
        raise ValueError(f"unsupported weight rank {w.ndim}: shape {orig_shape}")

    out_dim, in_dim = w2.shape
    if in_dim % block_size != 0:
        raise ValueError(
            f"in_dim {in_dim} not divisible by block_size {block_size}"
        )
    num_blocks = in_dim // block_size

    w_blocks = w2.reshape(out_dim, num_blocks, block_size).astype(np.float32)
    absval = np.abs(w_blocks)
    scale_2d = absval.max(axis=-1)  # (out, num_blocks)

    # Indices via sign + magnitude > 0.5 * scale
    safe_scale = np.where(scale_2d == 0.0, 1.0, scale_2d).reshape(out_dim, num_blocks, 1)
    normalized = w_blocks / safe_scale  # ∈ [-1, 1] approximately, with values {-1, 0, +1}
    # Threshold loosely to absorb any fp-noise.
    indices_b = np.where(
        normalized > 0.5, 1,
        np.where(normalized < -0.5, 2, 0),
    ).astype(np.uint8)  # (out, num_blocks, block_size)
    indices = indices_b.reshape(out_dim, in_dim)

    if w.ndim == 4:
        indices = indices.reshape(orig_shape)
        scale = scale_2d.reshape(orig_shape[0], num_blocks, 1, 1).astype(np.float16)
    else:
        scale = scale_2d.astype(np.float16)

    # Sanity: reconstruction round-trip
    recon = SIGN_LUT_VALUES[indices_b].astype(np.float32)  # (out, nb, bs)
    recon = recon * safe_scale
    recon = recon.reshape(out_dim, in_dim)
    diff = np.abs(recon - w2.astype(np.float32))
    max_diff = float(diff.max())

    return indices, scale, max_diff


def _is_target_weight(op, block_size: int = 128, min_numel: int = 1024) -> bool:
    """Is this `const` op a Bonsai weight tensor we can ternarize?"""
    if op.op_type != "const":
        return False
    # Access the materialized value via the op's output Var.
    arr = op.outputs[0].val
    if arr is None or not isinstance(arr, np.ndarray):
        return False
    if arr.dtype not in (np.float16, np.float32):
        return False
    if arr.ndim not in (2, 4):
        return False
    if arr.ndim == 4 and not (arr.shape[-1] == 1 and arr.shape[-2] == 1):
        return False
    # For a 2D or (out, in, 1, 1) 4D weight, axis=1 is the "in" axis along which
    # blocks run — same convention as Bonsai's per-128-block scales.
    in_dim = arr.shape[1] if arr.ndim >= 2 else 0
    if in_dim == 0 or in_dim % block_size != 0:
        return False
    if arr.size < min_numel:
        return False
    # Ternary probe: for ≥3 sampled rows, each 128-group must have
    # exactly {-s, 0, +s} structure (3 unique values with +s == -(-s)).
    flat = arr if arr.ndim == 2 else arr.reshape(arr.shape[0], arr.shape[1])
    out_dim = flat.shape[0]
    sample_rows = [0, out_dim // 2, out_dim - 1]
    for r in sample_rows:
        probe = flat[r, :block_size]
        u = np.unique(probe)
        # Allow all-zero group (padding) only if we have a positive example elsewhere
        nz = u[u != 0]
        if len(u) > 3:
            return False
        if len(nz) == 2:
            # Must be opposite-signed pair (ternary structure)
            if not np.isclose(nz[0], -nz[1], rtol=1e-3):
                return False
        elif len(nz) == 1:
            # Single non-zero means the whole block is {0, v} — still ternary-compatible
            pass
        elif len(nz) == 0:
            # All zeros — could be padding row of a real ternary weight or a non-weight table
            continue
        else:
            return False
    # At least one of the sampled rows must have a non-zero structure; otherwise
    # this is likely a trivial tensor (embed padding etc.) or a trig table.
    any_nonzero = False
    for r in sample_rows:
        if np.any(flat[r] != 0):
            # And a stronger check: the full row should be buildable from at most
            # 16 unique values (matches Bonsai per-row ≤ 16 block scales × 3).
            row_u = np.unique(flat[r])
            if len(row_u) <= 64:  # generous; ternary rows typically ~33-49 unique
                any_nonzero = True
                break
    return any_nonzero


def _make_uint2_indices(indices_uint8: np.ndarray):
    """Convert uint8 {0,1,2,3} indices to the coremltools uint2 numpy dtype."""
    uint2_dt = types.nptype_from_builtin(types.string_to_builtin("uint2"))
    return indices_uint8.astype(uint2_dt)


class TernaryPalettizePass(AbstractGraphPass):
    """Pass that replaces Bonsai weight consts with bit-exact ternary constexpr chains."""

    def __init__(self, block_size: int = 128, verbose: bool = True):
        self.block_size = block_size
        self.verbose = verbose
        self.replaced = 0
        self.skipped = 0
        self.max_max_diff = 0.0
        self.bytes_before = 0
        self.bytes_after = 0

    def apply(self, prog):
        # `mb.<op>()` requires a live block context; the decorator pushes each block
        # onto the Builder's stack so our in-place ops insert correctly.
        pass_self = self

        @block_context_manager
        def _visit_block(block):
            for op in list(block.operations):
                for nested in op.blocks:
                    _visit_block(nested)
                try:
                    if not _is_target_weight(op, pass_self.block_size):
                        continue
                    pass_self._replace_one(op)
                except Exception as e:
                    pass_self.skipped += 1
                    if pass_self.verbose:
                        print(f"  skip {op.name}: {type(e).__name__}: {e}")

        for func in prog.functions.values():
            _visit_block(func)

        if self.verbose:
            size_mb_before = self.bytes_before / 1e6
            size_mb_after = self.bytes_after / 1e6
            print(f"\nternary pass summary: replaced {self.replaced}, skipped {self.skipped}")
            print(f"  max reconstruction |diff|: {self.max_max_diff:.6f}")
            print(f"  weight bytes {size_mb_before:.0f} MB → {size_mb_after:.0f} MB "
                  f"({100 * size_mb_after / max(size_mb_before, 1):.1f}%)")

    def _replace_one(self, op):
        w = op.outputs[0].val
        self.bytes_before += w.nbytes

        indices, scale, max_diff = encode_ternary(w, self.block_size)
        if max_diff > 1e-3 and self.verbose:
            print(f"  {op.name}: shape={w.shape} max_diff={max_diff:.6f} "
                  f"w.max={np.abs(w).max():.6f}")
        self.max_max_diff = max(self.max_max_diff, max_diff)

        indices_u2 = _make_uint2_indices(indices)

        # Build a per-row-per-block LUT with the scale baked in, so each
        # (row, block) has its own 4-entry codebook = [0, +s, -s, 0]. This is
        # a single-op replacement (no constexpr_blockwise_shift_scale), which
        # avoids ANE compile rejection we hit with the 2-op chain.
        #
        # scale shape: (out, num_blocks) for 2D, (out, num_blocks, 1, 1) for 4D.
        # LUT rank = indices_rank + 2, with per-(row,block) palette:
        #   2D indices (out, in)       → lut (out, num_blocks, 4, 1)
        #   4D indices (out, in, 1, 1) → lut (out, num_blocks, 1, 1, 4, 1)
        if w.ndim == 2:
            s = scale.astype(np.float16)  # (out, num_blocks)
            out_dim, num_blocks = s.shape
            lut = np.zeros((out_dim, num_blocks, 4, 1), dtype=np.float16)
            lut[..., 1, 0] = s       # +s
            lut[..., 2, 0] = -s      # -s
            # entries 0 and 3 stay 0.0
        else:  # 4D (out, in, 1, 1)
            s2d = scale.reshape(scale.shape[0], scale.shape[1]).astype(np.float16)
            out_dim, num_blocks = s2d.shape
            lut = np.zeros((out_dim, num_blocks, 1, 1, 4, 1), dtype=np.float16)
            lut[..., 1, 0] = s2d.reshape(out_dim, num_blocks, 1, 1)
            lut[..., 2, 0] = -s2d.reshape(out_dim, num_blocks, 1, 1)

        new_var = mb.constexpr_lut_to_dense(
            indices=indices_u2, lut=lut,
            before_op=op, name=op.name + "_tern",
        )

        op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=op, old_var=op.outputs[0], new_var=new_var,
            no_check_var_types=True, force_replace=True,
        )
        op.enclosing_block.remove_ops([op])

        self.replaced += 1
        # bytes_after: uint2 indices + per-(row,block) fp16 LUT (4 entries of which
        # 2 are meaningful; still compact compared to fp16 weights).
        self.bytes_after += (indices.size * 2 + 7) // 8  # uint2
        self.bytes_after += lut.nbytes


def run(src: Path, dst: Path, block_size: int = 128) -> None:
    print(f"Loading {src}")
    m = ct.models.MLModel(str(src))

    pass_inst = TernaryPalettizePass(block_size=block_size, verbose=True)
    print("Running ternary MIL surgery...")
    t0 = time.time()
    out_model = _apply_graph_pass(
        m, pass_inst,
        skip_model_load=True,  # avoid forcing compile before save
    )
    print(f"  pass applied in {time.time() - t0:.1f}s")

    if dst.exists() and dst != src:
        shutil.rmtree(dst)
    elif dst == src:
        tmp = dst.with_suffix(".mlpackage.new")
        if tmp.exists():
            shutil.rmtree(tmp)
        out_model.save(str(tmp))
        shutil.rmtree(dst)
        shutil.move(str(tmp), str(dst))
        return
    out_model.save(str(dst))
    size_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
    print(f"Saved {dst.name} ({size_mb:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--block-size", type=int, default=128)
    args = ap.parse_args()
    run(Path(args.src), Path(args.dst), args.block_size)


if __name__ == "__main__":
    main()
