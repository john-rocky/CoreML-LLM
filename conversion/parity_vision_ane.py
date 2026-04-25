"""Real-input parity: our ANE CoreML build vs HF Gemma 4 vision forward.

The convert-time parity check used zero pixel values, which are
trivially equivalent under most ops. A sunflower image on the iPhone
produces garbled output ("human eye"), suggesting the features disagree
on real content. Compare per-patch cosine on a synthetic checker image
(deterministic, no external deps).
"""
from __future__ import annotations

import argparse
import numpy as np
import torch

import coremltools as ct
from transformers import Gemma4ForConditionalGeneration


MODEL_ID = "google/gemma-4-E2B-it"
GRID = 48
PATCHES = GRID * GRID
P = 16
PD = P * P * 3


def make_checker() -> np.ndarray:
    """48x48 grid, checker across the tile dim — fp16 (1, 2304, 768)."""
    rng = np.random.default_rng(42)
    pv = rng.standard_normal(size=(1, PATCHES, PD)).astype(np.float16) * 0.25
    return pv


def make_pid() -> np.ndarray:
    pid = np.zeros((1, PATCHES, 2), dtype=np.int32)
    k = 0
    for py in range(GRID):
        for px in range(GRID):
            pid[0, k, 0] = px
            pid[0, k, 1] = py
            k += 1
    return pid


def hf_forward(hf, pv: np.ndarray, pid: np.ndarray) -> np.ndarray:
    t_pv = torch.from_numpy(pv).to(torch.float32)
    t_pid = torch.from_numpy(pid).to(torch.long)
    with torch.no_grad():
        out = hf.model.get_image_features(
            pixel_values=t_pv.to(torch.float16),
            image_position_ids=t_pid,
            return_dict=True,
        )
    feat = out.pooler_output
    return feat.to(torch.float32).cpu().numpy()


def coreml_forward(path: str, pv: np.ndarray, pid: np.ndarray,
                    units: ct.ComputeUnit) -> np.ndarray:
    m = ct.models.MLModel(path, compute_units=units)
    r = m.predict({"pixel_values": pv, "pixel_position_ids": pid})
    return r["image_features"].astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def per_token_cos(a: np.ndarray, b: np.ndarray) -> list[float]:
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    out = []
    for i in range(a.shape[0]):
        na = np.linalg.norm(a[i]); nb = np.linalg.norm(b[i])
        out.append(float((a[i] * b[i]).sum() / (na * nb + 1e-9)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ane", required=True, help="vision.ane.mlpackage path")
    args = ap.parse_args()

    print("HF reference forward...")
    hf = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="cpu"
    ).eval()

    pv = make_checker()
    pid = make_pid()

    ref = hf_forward(hf, pv, pid)
    print(f"  ref shape: {ref.shape}  norm: {np.linalg.norm(ref):.2f}")

    for units, name in [
        (ct.ComputeUnit.CPU_AND_GPU, "CPU_AND_GPU"),
        (ct.ComputeUnit.CPU_AND_NE, "CPU_AND_NE"),
    ]:
        print(f"\nCoreML({name}) forward...")
        pred = coreml_forward(args.ane, pv, pid, units)
        print(f"  pred shape: {pred.shape}  norm: {np.linalg.norm(pred):.2f}")
        ref_t = ref
        if pred.ndim == 3 and ref.ndim == 2 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.shape != ref_t.shape:
            print(f"  !! SHAPE MISMATCH: ref {ref_t.shape} vs pred {pred.shape}")
            continue
        print(f"  overall cosine: {cosine(ref_t, pred):.4f}")
        tokens = per_token_cos(ref_t, pred)
        print(f"  per-token cos: min={min(tokens):.4f}  "
              f"median={sorted(tokens)[len(tokens)//2]:.4f}  "
              f"max={max(tokens):.4f}")
        print(f"  max_abs_diff: {np.abs(ref_t - pred).max():.4f}")


if __name__ == "__main__":
    main()
