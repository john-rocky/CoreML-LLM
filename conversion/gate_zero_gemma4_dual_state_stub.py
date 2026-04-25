"""Gate Zero (Gemma 4): minimal 2-layer DUAL-MLState stub.

Purpose: confirm whether the dual non-uniform StateType pattern that
the 2026-04-13 / 04-15 Gemma 4 attempts hit error -14 with is still
rejected by the ANE compiler on current iOS / coremltools, BEFORE
committing Mac Studio time to a full E2B stateful build.

The stub mimics the structure of build_gemma4_e2b_stateful_chunks.py
at minimum scale:
  - 2 layers: 1 sliding (head_dim=256, W=512) + 1 full (head_dim=512, ctx=2048)
  - TWO ct.StateType per chunk: kv_cache_sliding + kv_cache_full
  - slice_update writes (sliding @ ring_pos, full @ current_pos)

If this stub converts AND mlmodel.predict() returns finite output on
Mac ANE without -14 / MILCompilerForANE Error=(11), the dual-state
design is viable and we proceed with the full converter on Mac Studio.

If it fails: fall back to unified single-StateType-per-chunk (pad
sliding to ctx, single (2*L, HKV, ctx, max_hd) buffer like Qwen3-VL).

Usage:
  python conversion/gate_zero_gemma4_dual_state_stub.py --predict
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


# Gemma 4 E2B-shaped dims
HIDDEN = 2560
NUM_HEADS = 8
NUM_KV_HEADS = 1            # E2B native
HEAD_DIM_SLIDING = 256
HEAD_DIM_FULL = 512
MAX_HD = 512                # max(sliding, full) — sliding zero-padded to this
W = 512                     # sliding window
CTX = 2048


class StubLayer(nn.Module):
    """Tiny attention-like block with slice_update KV write into ONE
    of the two state buffers. Sliding layer writes ring_pos, full
    writes current_pos."""

    def __init__(self, is_full: bool, slot_idx: int):
        super().__init__()
        self.is_full = is_full
        self.k_idx = 2 * slot_idx
        self.v_idx = 2 * slot_idx + 1
        self.head_dim = HEAD_DIM_FULL if is_full else HEAD_DIM_SLIDING

        self.q_proj = nn.Conv2d(HIDDEN, NUM_HEADS * self.head_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(HIDDEN, NUM_KV_HEADS * self.head_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(HIDDEN, NUM_KV_HEADS * self.head_dim, 1, bias=False)
        self.o_proj = nn.Conv2d(NUM_HEADS * self.head_dim, HIDDEN, 1, bias=False)
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(self, x, kv_sliding, kv_full, current_pos, ring_pos):
        # x: (1, HIDDEN, 1, 1)
        q = self.q_proj(x).view(1, NUM_HEADS, self.head_dim, 1).permute(0, 1, 3, 2)
        k = self.k_proj(x).view(1, NUM_KV_HEADS, self.head_dim, 1).permute(0, 1, 3, 2)
        v = self.v_proj(x).view(1, NUM_KV_HEADS, self.head_dim, 1).permute(0, 1, 3, 2)

        # Pad sliding's hd to max_hd so both states share the inner dim.
        if not self.is_full and self.head_dim < MAX_HD:
            k = torch.nn.functional.pad(k, (0, MAX_HD - self.head_dim))
            v = torch.nn.functional.pad(v, (0, MAX_HD - self.head_dim))

        if self.is_full:
            kv_full[self.k_idx:self.k_idx + 1, :, current_pos:current_pos + 1, :] = k.to(kv_full.dtype)
            kv_full[self.v_idx:self.v_idx + 1, :, current_pos:current_pos + 1, :] = v.to(kv_full.dtype)
            k_full = kv_full[self.k_idx:self.k_idx + 1, :, :, :self.head_dim].squeeze(0)
            v_full = kv_full[self.v_idx:self.v_idx + 1, :, :, :self.head_dim].squeeze(0)
        else:
            kv_sliding[self.k_idx:self.k_idx + 1, :, ring_pos:ring_pos + 1, :] = k.to(kv_sliding.dtype)
            kv_sliding[self.v_idx:self.v_idx + 1, :, ring_pos:ring_pos + 1, :] = v.to(kv_sliding.dtype)
            k_full = kv_sliding[self.k_idx:self.k_idx + 1, :, :, :self.head_dim].squeeze(0)
            v_full = kv_sliding[self.v_idx:self.v_idx + 1, :, :, :self.head_dim].squeeze(0)

        # Repeat-KV expansion for MHA attention.
        n_rep = NUM_HEADS // NUM_KV_HEADS
        k_rep = k_full.unsqueeze(1).repeat(1, n_rep, 1, 1).view(
            1, NUM_HEADS, k_full.shape[1], self.head_dim)
        v_rep = v_full.unsqueeze(1).repeat(1, n_rep, 1, 1).view(
            1, NUM_HEADS, v_full.shape[1], self.head_dim)

        q32 = q.to(torch.float32)
        scores = torch.matmul(q32, k_rep.to(torch.float32).transpose(-2, -1)) * self.scale
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v_rep.to(torch.float32)).to(x.dtype)

        attn = attn.permute(0, 1, 3, 2).contiguous().view(1, NUM_HEADS * self.head_dim, 1, 1)
        return self.o_proj(attn)


class DualStateChunk(nn.Module):
    """2 layers (1 sliding + 1 full), TWO ct.StateType buffers.
    This reproduces the exact non-uniform-dual-state pattern that
    failed in 2026-04 with error -14."""

    def __init__(self):
        super().__init__()
        self.sliding_layer = StubLayer(is_full=False, slot_idx=0)
        self.full_layer = StubLayer(is_full=True, slot_idx=0)
        self.register_buffer(
            "kv_cache_sliding",
            torch.zeros(2, NUM_KV_HEADS, W, MAX_HD, dtype=torch.float16),
        )
        self.register_buffer(
            "kv_cache_full",
            torch.zeros(2, NUM_KV_HEADS, CTX, MAX_HD, dtype=torch.float16),
        )

    def forward(self, hidden, current_pos, ring_pos):
        x = hidden
        x = x + self.sliding_layer(x, self.kv_cache_sliding, self.kv_cache_full,
                                    current_pos, ring_pos)
        x = x + self.full_layer(x, self.kv_cache_sliding, self.kv_cache_full,
                                current_pos, ring_pos)
        return x


def convert_stub(out_path: Path):
    module = DualStateChunk().eval()
    for p in module.parameters():
        p.data = p.data.half()

    sample_hidden = torch.zeros(1, HIDDEN, 1, 1, dtype=torch.float16)
    sample_pos = torch.zeros(1, dtype=torch.int32)
    sample_ring = torch.zeros(1, dtype=torch.int32)

    print("[stub] tracing 2-layer dual-state module...")
    traced = torch.jit.trace(module, (sample_hidden, sample_pos, sample_ring),
                              check_trace=False)

    print("[stub] converting to mlprogram (iOS18, CPU_AND_NE)...")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="hidden_states", shape=sample_hidden.shape, dtype=np.float16),
            ct.TensorType(name="current_pos",   shape=(1,),                dtype=np.int32),
            ct.TensorType(name="ring_pos",      shape=(1,),                dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(2, NUM_KV_HEADS, W, MAX_HD), dtype=np.float16),
                name="kv_cache_sliding",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(2, NUM_KV_HEADS, CTX, MAX_HD), dtype=np.float16),
                name="kv_cache_full",
            ),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"[stub] saved to {out_path}")
    return mlmodel


def predict_once(mlmodel, label: str):
    """Single predict() to smoke-test ANE dispatch. Watch stderr for
    'error code: -14' or 'MILCompilerForANE Error=(11)'."""
    try:
        state = mlmodel.make_state()
    except Exception as e:
        print(f"[{label}] make_state() failed: {e}")
        raise

    hidden = np.zeros((1, HIDDEN, 1, 1), dtype=np.float16)
    hidden[0, 0, 0, 0] = 1.0

    t0 = time.perf_counter()
    out = mlmodel.predict(
        {"hidden_states": hidden,
         "current_pos": np.array([0], dtype=np.int32),
         "ring_pos":    np.array([0], dtype=np.int32)},
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
        default="conversion/out/gate_zero_gemma4_dual_stub.mlpackage",
        type=Path,
    )
    ap.add_argument("--predict", action="store_true",
                    help="Run Mac ANE predict after convert (the actual gate).")
    args = ap.parse_args()

    mlmodel = convert_stub(args.out)
    if args.predict:
        predict_once(mlmodel, "mac-ane")


if __name__ == "__main__":
    main()
