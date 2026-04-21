"""Phase 0 probe: ANE placement of primitives needed for Qwen3.5 Gated DeltaNet.

Tests (each at seq 128/512/2048 where applicable):
  1. cumsum           — expected CPU (docs/MIL_OP_CATALOG.md flags it). Fallback needed.
  2. cumprod          — mb.cumprod does not exist; skipped. Torch frontend unrolls to
                        N scatter ops — catastrophic. Must use exp(cumsum(log(·))).
  3. conv1d_k4        — depthwise conv1d kernel=4 along seq. Mamba-style conv state.
                        Expected ANE (conv is the ANE-native primitive).
  4. tril_matmul      — x @ L where L is a lower-triangular constant. Drop-in scan
                        replacement if cumsum falls off ANE. Expected ANE.
  5. sdpa_hd256       — scaled_dot_product_attention at head_dim=256. Baseline for the
                        6 full-attention layers. Expected ANE (Gemma 4 precedent).
  6. state_update     — elementwise update of a large SSM state tensor
                        (batch, d_inner, d_state) = (1, 1024, 128). Verifies no
                        ANE rejection on SSM-shape I/O tensors.

Run: python probe_scan_ops_ane.py
Gate: Do not run while ANE is busy (eagle3 data collection). The MLComputePlan
  inspection triggers compile + schedule, which touches ANE.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


SEQ_LENS = [128, 512, 2048]
FEATURE_DIM = 64


# ---------- Program builders ----------

def _prog_cumsum(seq_len: int):
    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, seq_len, FEATURE_DIM), dtype=types.fp16)],
        opset_version=ct.target.iOS17,
    )
    def prog(x):
        return mb.cumsum(x=x, axis=1)
    return prog


def _prog_conv1d_k4(seq_len: int):
    # Depthwise conv1d kernel=4 along seq axis. CoreML conv expects NCHW; we treat
    # seq as the W axis and feature as C, with H=1.
    c = 1024  # mimic Mamba d_inner
    kernel = np.ones((c, 1, 1, 4), dtype=np.float16) / 4.0

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, c, 1, seq_len), dtype=types.fp16)],
        opset_version=ct.target.iOS17,
    )
    def prog(x):
        return mb.conv(
            x=x,
            weight=kernel,
            strides=(1, 1),
            pad_type="same",
            dilations=(1, 1),
            groups=c,  # depthwise
        )
    return prog


def _prog_tril_matmul(seq_len: int):
    # x @ L where L is lower-triangular ones. Drop-in cumsum replacement.
    # ANE prefers rank-4 NCHW. x is (1, C, 1, S), L broadcast via matmul.
    L = np.tril(np.ones((seq_len, seq_len), dtype=np.float16))

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, FEATURE_DIM, 1, seq_len), dtype=types.fp16)],
        opset_version=ct.target.iOS17,
    )
    def prog(x):
        return mb.matmul(x=x, y=L)
    return prog


def _prog_sdpa_hd256(seq_len: int):
    num_heads = 8
    head_dim = 256

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, num_heads, seq_len, head_dim), dtype=types.fp16),
            mb.TensorSpec(shape=(1, num_heads, seq_len, head_dim), dtype=types.fp16),
            mb.TensorSpec(shape=(1, num_heads, seq_len, head_dim), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(q, k, v):
        return mb.scaled_dot_product_attention(query=q, key=k, value=v)

    return prog


def _prog_state_update():
    # Rank-4 (B, C, H, W) = (1, d_inner, 1, d_state) — ANE's preferred layout.
    d_inner = 1024
    d_state = 128

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, d_inner, 1, d_state), dtype=types.fp16),
            mb.TensorSpec(shape=(1, d_inner, 1, d_state), dtype=types.fp16),
            mb.TensorSpec(shape=(1, d_inner, 1, d_state), dtype=types.fp16),
            mb.TensorSpec(shape=(1, d_inner, 1, d_state), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS17,
    )
    def prog(prev, gate, coeff, x):
        decayed = mb.mul(x=prev, y=gate)
        update = mb.mul(x=coeff, y=x)
        return mb.add(x=decayed, y=update)

    return prog


def _prog_chunk_tril(chunk_size: int = 64):
    """Actual operation inside chunked DeltaNet: small tril matmul at chunk=64.
    Shape: (1, num_heads, chunk, chunk) — per-head intra-chunk causal mask apply."""
    num_heads = 16
    head_dim = 128
    L = np.tril(np.ones((chunk_size, chunk_size), dtype=np.float16))

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, num_heads, chunk_size, chunk_size), dtype=types.fp16)],
        opset_version=ct.target.iOS17,
    )
    def prog(x):
        return mb.matmul(x=x, y=L)

    return prog


def _prog_decode_block():
    """Realistic Gated DeltaNet decode-step skeleton (fp16 only).
    Checks whether the decode workload lands on ANE end-to-end."""
    H = 16   # num heads
    Dk = 128
    Dv = 128

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, H, Dk, Dv), dtype=types.fp16),  # prev_state
            mb.TensorSpec(shape=(1, H, 1, Dk),  dtype=types.fp16),  # q
            mb.TensorSpec(shape=(1, H, 1, Dk),  dtype=types.fp16),  # k
            mb.TensorSpec(shape=(1, H, 1, Dv),  dtype=types.fp16),  # v
            mb.TensorSpec(shape=(1, H, 1, 1),   dtype=types.fp16),  # g (per-head decay)
            mb.TensorSpec(shape=(1, H, 1, 1),   dtype=types.fp16),  # beta
        ],
        opset_version=ct.target.iOS18,
    )
    def prog(prev, q, k, v, g, beta):
        # state = g * prev + beta * (k^T @ v)
        kt = mb.transpose(x=k, perm=[0, 1, 3, 2])       # (1,H,Dk,1)
        outer = mb.matmul(x=kt, y=v)                     # (1,H,Dk,Dv)
        scaled = mb.mul(x=beta, y=outer)
        decayed = mb.mul(x=g, y=prev)
        new_state = mb.add(x=decayed, y=scaled)
        # y = q @ state
        y = mb.matmul(x=q, y=new_state)                  # (1,H,1,Dv)
        return y, new_state

    return prog


def _prog_cumsum_surrounded():
    """cumsum sandwiched between matmuls. Tests contamination of the graph."""
    S = 2048
    C = 64

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, C, 1, S), dtype=types.fp16)],
        opset_version=ct.target.iOS18,
    )
    def prog(x):
        W1 = np.random.randn(S, S).astype(np.float16) * 0.01
        y = mb.matmul(x=x, y=W1)
        z = mb.cumsum(x=y, axis=3)
        W2 = np.random.randn(S, S).astype(np.float16) * 0.01
        out = mb.matmul(x=z, y=W2)
        return out

    return prog


def _prog_outer_product_state_update():
    """Decode step state update: state += beta * v * k^T.
    Shape: state (1, H, Dk, Dv) + outer-product of k (1,H,1,Dk) and v (1,H,1,Dv)."""
    num_heads = 16
    dk = 128
    dv = 128

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, num_heads, dk, dv), dtype=types.fp16),  # state
            mb.TensorSpec(shape=(1, num_heads, dk, 1),  dtype=types.fp16),  # k
            mb.TensorSpec(shape=(1, num_heads, 1,  dv), dtype=types.fp16),  # v
        ],
        opset_version=ct.target.iOS17,
    )
    def prog(state, k, v):
        outer = mb.matmul(x=k, y=v)  # (1,H,Dk,Dv)
        return mb.add(x=state, y=outer)

    return prog


# ---------- Runner ----------

def _convert(prog, target=ct.target.iOS18) -> ct.models.MLModel:
    return ct.convert(
        prog,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=target,
    )


def _dispatch_summary(model: ct.models.MLModel, tmpdir: Path, tag: str):
    path = tmpdir / f"{tag}.mlpackage"
    model.save(str(path))

    # MLComputePlan.load_from_path needs a compiled .mlmodelc, not .mlpackage.
    # Reloading triggers compilation; grab the compiled path from the loaded model.
    try:
        reloaded = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        compiled_path = reloaded.get_compiled_model_path()
    except Exception as exc:  # noqa: BLE001
        return f"compile_error: {exc.__class__.__name__}: {exc}"

    try:
        plan = ct.models.compute_plan.MLComputePlan.load_from_path(
            path=str(compiled_path),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
    except Exception as exc:  # noqa: BLE001
        return f"compute_plan_error: {exc.__class__.__name__}: {exc}"

    device_counts: dict[str, int] = {}
    op_details: list[str] = []
    program = plan.model_structure.program
    if program is None:
        return "no_program"

    for func_name, func in program.functions.items():
        for op in func.block.operations:
            assignment = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = assignment.preferred_compute_device.__class__.__name__ if assignment else "unknown"
            device_counts[dev] = device_counts.get(dev, 0) + 1
            op_details.append(f"  {func_name}.{op.operator_name} -> {dev}")

    lines = [f"device_counts={device_counts}"]
    lines.extend(op_details)
    return "\n".join(lines)


PROBES = {
    "cumsum":             (_prog_cumsum,                 SEQ_LENS),
    "conv1d_k4":          (_prog_conv1d_k4,              SEQ_LENS),
    "tril_matmul":        (_prog_tril_matmul,            SEQ_LENS),
    "sdpa_hd256":         (_prog_sdpa_hd256,             SEQ_LENS),
    "state_update":       (_prog_state_update,           [None]),
    "chunk_tril":         (_prog_chunk_tril,             [None]),
    "outer_state_update": (_prog_outer_product_state_update, [None]),
    "decode_block":       (_prog_decode_block,           [None]),
    "cumsum_surrounded":  (_prog_cumsum_surrounded,      [None]),
}


def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="qwen35_probe_"))
    print(f"tmpdir: {tmpdir}")

    for probe_name, (builder, seqs) in PROBES.items():
        for seq in seqs:
            tag = probe_name if seq is None else f"{probe_name}_seq{seq}"
            print(f"\n=== {tag} ===")
            try:
                prog = builder() if seq is None else builder(seq)
                model = _convert(prog)
            except Exception as exc:  # noqa: BLE001
                print(f"  build/convert_error: {exc.__class__.__name__}: {exc}")
                continue
            print(_dispatch_summary(model, tmpdir, tag))


if __name__ == "__main__":
    main()
