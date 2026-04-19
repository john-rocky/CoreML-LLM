#!/usr/bin/env python3
"""Feed identical (h_prev, e_next) tensors to the PyTorch checkpoint draft
AND the CoreML-converted draft, compare output argmax + h_out norms.

If argmax diverges materially, the conversion corrupted the draft — that's
what causes on-device 0% accept, because the deployed draft isn't the
same function as the trained one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_eagle3_infer import EAGLE3Draft  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--coreml", type=Path, required=True,
                    help="eagle3_draft.mlpackage")
    ap.add_argument("--fusion-coreml", type=Path, default=None,
                    help="eagle3_fusion.mlpackage (optional cross-check)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20,
                    help="Random (h_prev, e_next) inputs to test")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load PT ckpt.
    with open(args.config) as f:
        raw = json.load(f)
    cfg = {
        "hidden":        raw["hidden"],
        "num_heads":     raw["num_heads"],
        "num_kv":        raw["num_kv_heads"],
        "head_dim":      raw["head_dim"],
        "ffn":           raw["ffn"],
        "vocab":         raw["vocab"],
        "rms_eps":       raw["rms_eps"],
        "rope_theta":    raw["rope_theta"],
        "embed_scale":   raw["embed_scale"],
        "fusion_layers": raw["fusion_layers"],
    }
    H = cfg["hidden"]
    # Load real lm_head weight so logits are meaningful.
    lm_head_path = Path("/Users/majimadaisuke/Downloads/lm_head_weight.bin")
    if lm_head_path.exists():
        lm_head_np = np.fromfile(lm_head_path, dtype=np.float16).reshape(cfg["vocab"], H)
        lm_head_weight = torch.from_numpy(lm_head_np.copy())
    else:
        raise FileNotFoundError(f"Need {lm_head_path}; generate via `python -c \"...\"` per docstring.")
    pt_draft = EAGLE3Draft(cfg, lm_head_weight).eval()
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model", state)
    missing, unexpected = pt_draft.load_state_dict(sd, strict=False)
    if missing:
        print(f"[PT] missing keys: {missing[:5]} ... ({len(missing)} total)")
    if unexpected:
        print(f"[PT] unexpected keys: {unexpected[:5]} ... ({len(unexpected)} total)")
    pt_draft = pt_draft.to(torch.float16)

    # Load CoreML.
    print(f"[CoreML] Loading {args.coreml}")
    cm = ct.models.MLModel(str(args.coreml), compute_units=ct.ComputeUnit.CPU_ONLY)

    # Build RoPE just once for PT (tiny).
    def build_rope():
        half = cfg["head_dim"] // 2
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, half, dtype=torch.float32) / half))
        pos = torch.arange(32, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", pos, inv)
        return freqs.cos(), freqs.sin()
    cos_r, sin_r = build_rope()

    pt_toks, cm_toks = [], []
    diff_norms = []
    for i in range(args.n):
        h_prev = rng.standard_normal((1, 1, H)).astype(np.float16) * np.float16(0.5)
        e_next = rng.standard_normal((1, 1, H)).astype(np.float16) * np.float16(1.0)

        # CoreML
        cm_out = cm.predict({
            "h_prev": h_prev, "e_next": e_next,
        })
        cm_tok = int(np.asarray(cm_out["token"]).reshape(-1)[0])
        cm_hout = np.asarray(cm_out["h_out"]).reshape(H).astype(np.float32)

        # PT
        with torch.no_grad():
            hp = torch.from_numpy(h_prev.copy())
            en = torch.from_numpy(e_next.copy())
            d_h, logits = pt_draft.step(hp, en, cos_r, sin_r, is_sequence=False)
            pt_tok = int(logits[0, -1].argmax(-1).item())
            pt_hout = d_h[0, 0].float().numpy()

        pt_toks.append(pt_tok)
        cm_toks.append(cm_tok)
        diff = np.linalg.norm(pt_hout - cm_hout) / (np.linalg.norm(pt_hout) + 1e-9)
        diff_norms.append(float(diff))

        print(f"  sample {i}:  PT_tok={pt_tok}   CoreML_tok={cm_tok}   "
              f"|h_out| diff_rel={diff:.4f}")

    match = sum(1 for a, b in zip(pt_toks, cm_toks) if a == b)
    print(f"\n── Summary ──")
    print(f"Tokens matched: {match}/{args.n}")
    print(f"h_out diff_rel mean: {np.mean(diff_norms):.4f}  "
          f"max: {np.max(diff_norms):.4f}")
    if match < args.n * 0.5:
        print("→ CoreML draft deviates substantially from PT. "
              "Check the conversion (build_eagle3.py).")
    elif match < args.n:
        print("→ CoreML draft is close but not identical — "
              "may be fp16/palettize quantization noise.")
    else:
        print("→ CoreML draft matches PT exactly.")


if __name__ == "__main__":
    main()
