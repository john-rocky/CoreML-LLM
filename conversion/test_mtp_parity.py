#!/usr/bin/env python3
"""Parity test: PyTorch MTP drafter vs TFLite interpreter.

Run with Python 3.12 (ai-edge-litert requires it):
    python3.12 conversion/test_mtp_parity.py

Compares argmax outputs and logit distributions between the two.
"""

from __future__ import annotations

import sys
import numpy as np


def run_tflite(tflite_path: str, activations: np.ndarray, input_pos: np.ndarray,
               kv13_k: np.ndarray, kv13_v: np.ndarray,
               kv14_k: np.ndarray, kv14_v: np.ndarray,
               mask: np.ndarray, param_tensor: np.ndarray):
    """Run TFLite interpreter and return (logits, projected_activations)."""
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=tflite_path)

    # Get signature runner
    signatures = interp.get_signature_list()
    sig_name = list(signatures.keys())[0]
    runner = interp.get_signature_runner(sig_name)

    # Run with named inputs
    outputs = runner(
        activations=activations,
        input_pos=input_pos,
        kv_cache_k_13=kv13_k,
        kv_cache_v_13=kv13_v,
        kv_cache_k_14=kv14_k,
        kv_cache_v_14=kv14_v,
        mask=mask,
        param_tensor=param_tensor,
    )

    return outputs["logits"], outputs["projected_activations"]


def run_pytorch(pt_path: str, activations: np.ndarray, input_pos: np.ndarray,
                kv13_k: np.ndarray, kv13_v: np.ndarray,
                kv14_k: np.ndarray, kv14_v: np.ndarray,
                mask_swa: np.ndarray, mask_full: np.ndarray):
    """Run PyTorch model and return (logits, projected_activations)."""
    import torch
    sys.path.insert(0, "conversion")
    from mtp_drafter_model import MtpDrafterModel, MtpDrafterConfig

    cfg = MtpDrafterConfig()
    model = MtpDrafterModel(cfg).float().eval()
    sd = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(sd)

    with torch.no_grad():
        logits, proj_act = model(
            torch.from_numpy(activations).float(),
            torch.from_numpy(input_pos).int(),
            torch.from_numpy(kv13_k).float(),
            torch.from_numpy(kv13_v).float(),
            torch.from_numpy(kv14_k).float(),
            torch.from_numpy(kv14_v).float(),
            torch.from_numpy(mask_swa).float(),
            torch.from_numpy(mask_full).float(),
        )

    return logits.numpy(), proj_act.numpy()


def main():
    tflite_path = "output/mtp_probe/section_9.tflite"
    pt_path = "output/mtp_probe/mtp_drafter.pt"

    ctx = 32003
    np.random.seed(42)

    pos = 10
    # Input tensors (random but consistent)
    activations = np.random.randn(1, 1, 3072).astype(np.float32) * 0.1
    input_pos = np.array([pos], dtype=np.int32)

    # KV caches: INT8 in TFLite, fp32 for PyTorch
    # For TFLite: use int8 zeros (empty cache)
    kv13_k_int8 = np.zeros((1, 1, ctx, 256), dtype=np.int8)
    kv13_v_int8 = np.zeros((1, 1, 256, ctx), dtype=np.int8)
    kv14_k_int8 = np.zeros((1, 1, ctx, 512), dtype=np.int8)
    kv14_v_int8 = np.zeros((1, 1, 512, ctx), dtype=np.int8)

    # For PyTorch: fp32 zeros (equivalent of empty cache)
    kv13_k_fp = kv13_k_int8.astype(np.float32)
    kv13_v_fp = kv13_v_int8.astype(np.float32)
    kv14_k_fp = kv14_k_int8.astype(np.float32)
    kv14_v_fp = kv14_v_int8.astype(np.float32)

    # Masks: TFLite uses bool, PyTorch uses -inf mask
    mask_tfl = np.ones((1, 1, 1, ctx), dtype=np.bool_)
    # Only unmask positions <= input_pos
    mask_tfl[:, :, :, :pos+1] = True
    mask_tfl[:, :, :, pos+1:] = False

    # PyTorch mask: 0 for valid, -inf for masked
    mask_swa = np.zeros((1, 1, 1, ctx), dtype=np.float32)
    mask_swa[:, :, :, pos+1:] = -float("inf")
    mask_full = mask_swa.copy()

    # param_tensor for TFLite
    param_tensor = np.zeros((1, 1, 1, 7), dtype=np.int32)
    param_tensor[0, 0, 0, 0] = pos  # position

    print("Running TFLite interpreter...")
    tfl_logits, tfl_proj = run_tflite(
        tflite_path, activations, input_pos,
        kv13_k_int8, kv13_v_int8, kv14_k_int8, kv14_v_int8,
        mask_tfl, param_tensor
    )
    print(f"  TFLite logits: {tfl_logits.shape}, argmax={tfl_logits.argmax()}")
    print(f"  TFLite proj:   {tfl_proj.shape}, norm={np.linalg.norm(tfl_proj):.4f}")

    print("\nRunning PyTorch model...")
    pt_logits, pt_proj = run_pytorch(
        pt_path, activations, input_pos,
        kv13_k_fp, kv13_v_fp, kv14_k_fp, kv14_v_fp,
        mask_swa, mask_full
    )
    print(f"  PyTorch logits: {pt_logits.shape}, argmax={pt_logits.argmax()}")
    print(f"  PyTorch proj:   {pt_proj.shape}, norm={np.linalg.norm(pt_proj):.4f}")

    # Compare
    print("\n=== Parity Check ===")
    tfl_argmax = tfl_logits.argmax()
    pt_argmax = pt_logits.argmax()
    argmax_match = tfl_argmax == pt_argmax
    print(f"  Argmax match: {argmax_match} (TFL={tfl_argmax}, PT={pt_argmax})")

    # Top-5 comparison
    tfl_top5 = np.argsort(tfl_logits.flatten())[-5:][::-1]
    pt_top5 = np.argsort(pt_logits.flatten())[-5:][::-1]
    top5_overlap = len(set(tfl_top5) & set(pt_top5))
    print(f"  Top-5 overlap: {top5_overlap}/5")
    print(f"    TFL top-5: {tfl_top5}")
    print(f"    PT  top-5: {pt_top5}")

    # Logit statistics
    tfl_flat = tfl_logits.flatten()
    pt_flat = pt_logits.flatten()
    cos_sim = np.dot(tfl_flat, pt_flat) / (np.linalg.norm(tfl_flat) * np.linalg.norm(pt_flat) + 1e-8)
    print(f"  Logit cosine similarity: {cos_sim:.6f}")

    # Projected activations comparison
    tfl_p = tfl_proj.flatten()
    pt_p = pt_proj.flatten()
    proj_cos = np.dot(tfl_p, pt_p) / (np.linalg.norm(tfl_p) * np.linalg.norm(pt_p) + 1e-8)
    print(f"  Proj activations cosine sim: {proj_cos:.6f}")

    max_diff = np.max(np.abs(tfl_flat - pt_flat))
    mean_diff = np.mean(np.abs(tfl_flat - pt_flat))
    print(f"  Logit max abs diff: {max_diff:.6f}")
    print(f"  Logit mean abs diff: {mean_diff:.6f}")

    # Verdict
    if argmax_match and cos_sim > 0.99:
        print("\n  PASS: parity confirmed")
    elif top5_overlap >= 4 and cos_sim > 0.95:
        print("\n  PASS (soft): near-parity, likely fp16 rounding")
    elif cos_sim > 0.8:
        print("\n  WARN: moderate correlation — check norm mapping or RoPE")
    else:
        print("\n  FAIL: significant divergence — investigate weight mapping")


if __name__ == "__main__":
    main()
