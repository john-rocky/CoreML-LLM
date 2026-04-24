"""Gate Zero: minimal 2-layer MLState stub for Qwen3-VL 2B Phase 1.

Purpose: verify that Core ML + ANE accept the target stateful KV pattern
(ct.StateType + slice-assign writes) at the exact shape of the real
model's per-chunk KV cache, BEFORE we touch the full converter.

State shape per chunk for Qwen3-VL 2B:
    (2 * layers_in_chunk, num_kv_heads=8, max_seq=2048, head_dim=128) fp16

We use layers_in_chunk=2 for the stub, so the state is (4, 8, 2048, 128).

Success = converts + compiles + predicts on Mac ANE. Deploy to iPhone for
final Gate Zero sign-off. Failure mode to watch for: Core ML error -14 or
MILCompilerForANE Error=(11).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct


# Qwen3-VL 2B dims (from config.json)
HIDDEN = 2048
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
MAX_SEQ = 2048
LAYERS_IN_CHUNK = 2  # stub
KV_CACHE_LEN = 2 * LAYERS_IN_CHUNK  # K and V interleaved


class StubAttention(nn.Module):
    """Tiny attention-like block: Conv2d QKV proj + slice-write KV state +
    slice-read KV cache + matmul. Just enough to exercise slice_update.
    """

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.k_idx = 2 * layer_idx
        self.v_idx = 2 * layer_idx + 1

        self.q_proj = nn.Conv2d(HIDDEN, NUM_HEADS * HEAD_DIM, 1, bias=False)
        self.k_proj = nn.Conv2d(HIDDEN, NUM_KV_HEADS * HEAD_DIM, 1, bias=False)
        self.v_proj = nn.Conv2d(HIDDEN, NUM_KV_HEADS * HEAD_DIM, 1, bias=False)
        self.o_proj = nn.Conv2d(NUM_HEADS * HEAD_DIM, HIDDEN, 1, bias=False)
        self.scale = 1.0 / (HEAD_DIM ** 0.5)

    def forward(self, x, kv_cache, current_pos):
        # x: (1, HIDDEN, 1, 1) ANE layout
        # kv_cache: (2*LAYERS_IN_CHUNK, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM)
        # current_pos: int32 scalar tensor (just use .item-free slicing)

        q = self.q_proj(x).view(1, NUM_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        k = self.k_proj(x).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)
        v = self.v_proj(x).view(1, NUM_KV_HEADS, HEAD_DIM, 1).permute(0, 1, 3, 2)

        # slice-assign write: kv_cache[k_idx, :, pos:pos+1, :] = k
        # Use tensor indexing so trace captures it as slice_update.
        kv_cache[self.k_idx:self.k_idx + 1, :, current_pos:current_pos + 1, :] = \
            k.to(kv_cache.dtype)
        kv_cache[self.v_idx:self.v_idx + 1, :, current_pos:current_pos + 1, :] = \
            v.to(kv_cache.dtype)

        # Read back full K/V cache for this layer.
        k_full = kv_cache[self.k_idx:self.k_idx + 1].squeeze(0)  # (KV_H, MAX_SEQ, HEAD_DIM)
        v_full = kv_cache[self.v_idx:self.v_idx + 1].squeeze(0)

        # Repeat KV for MHA.
        n_rep = NUM_HEADS // NUM_KV_HEADS
        k_rep = k_full.unsqueeze(1).repeat(1, n_rep, 1, 1).view(
            1, NUM_HEADS, MAX_SEQ, HEAD_DIM
        )
        v_rep = v_full.unsqueeze(1).repeat(1, n_rep, 1, 1).view(
            1, NUM_HEADS, MAX_SEQ, HEAD_DIM
        )

        # attention in fp32 for stability
        q32 = q.to(torch.float32)
        k32 = k_rep.to(torch.float32)
        v32 = v_rep.to(torch.float32)
        scores = torch.matmul(q32, k32.transpose(-2, -1)) * self.scale
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v32).to(x.dtype)  # (1, H, 1, D)

        attn = attn.permute(0, 1, 3, 2).contiguous().view(1, NUM_HEADS * HEAD_DIM, 1, 1)
        return self.o_proj(attn)


class StubChunk(nn.Module):
    """2-layer residual stack that reads/writes one unified KV state tensor."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([StubAttention(i) for i in range(LAYERS_IN_CHUNK)])
        self.register_buffer(
            "kv_cache_0",
            torch.zeros(KV_CACHE_LEN, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM, dtype=torch.float16),
        )

    def forward(self, hidden, current_pos):
        x = hidden
        for layer in self.layers:
            x = x + layer(x, self.kv_cache_0, current_pos)
        return x


def convert_stub(out_path: Path, minimum_deployment_target):
    module = StubChunk().eval()
    for p in module.parameters():
        p.data = p.data.half()

    sample_hidden = torch.zeros(1, HIDDEN, 1, 1, dtype=torch.float16)
    sample_pos = torch.tensor(0, dtype=torch.int32)

    print("[stub] tracing...")
    traced = torch.jit.trace(module, (sample_hidden, sample_pos), check_trace=False)

    print("[stub] converting to mlprogram...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=sample_hidden.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(KV_CACHE_LEN, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM),
                    dtype=np.float16,
                ),
                name="kv_cache_0",
            )
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=minimum_deployment_target,
        convert_to="mlprogram",
        skip_model_load=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"[stub] saved to {out_path}")
    return mlmodel


def predict_once(mlmodel, label: str):
    """Run a single predict() to smoke-test actual ANE dispatch."""
    try:
        state = mlmodel.make_state()
    except Exception as e:
        print(f"[{label}] make_state() failed: {e}")
        raise

    hidden = np.zeros((1, HIDDEN, 1, 1), dtype=np.float16)
    hidden[0, 0, 0, 0] = 1.0
    pos = np.array([0], dtype=np.int32)

    t0 = time.perf_counter()
    out = mlmodel.predict(
        {"hidden_states": hidden, "current_pos": pos},
        state=state,
    )
    dt_ms = (time.perf_counter() - t0) * 1000
    h = out["output_hidden_states"]
    finite = np.isfinite(np.asarray(h, dtype=np.float32)).all()
    print(f"[{label}] predict OK in {dt_ms:.1f} ms, output finite={finite}, "
          f"norm={np.linalg.norm(h.astype(np.float32)):.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="conversion/out/gate_zero_stub.mlpackage",
        type=Path,
    )
    ap.add_argument(
        "--ios-target",
        choices=["iOS17", "iOS18"],
        default="iOS18",
        help="MLState requires iOS18+.",
    )
    ap.add_argument("--predict", action="store_true", help="Run Mac ANE predict after convert.")
    args = ap.parse_args()

    dt = (ct.target.iOS18 if args.ios_target == "iOS18" else ct.target.iOS17)
    mlmodel = convert_stub(args.out, dt)

    if args.predict:
        predict_once(mlmodel, "mac-ane")


if __name__ == "__main__":
    main()
