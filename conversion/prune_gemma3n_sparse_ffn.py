#!/usr/bin/env python3
"""Static structural prune for Gemma 3n's sparse FFN layers.

Phase α-2 implementation of Path 1 from
`docs/GEMMA3N_SPARSITY_VALIDATED_2026_05_14.md`.

Concept:
  Gemma 3n's `activation_sparsity_pattern` declares 95% of FFN
  intermediate neurons dormant for layers 0-9. Our calibration
  (Phase α-2, `output/sparsity_gemma3n.json` regen) measured the
  per-layer cumulative magnitude curves. For each sparse layer, the
  top-K most-frequently-fired neurons cover 70-95% of magnitude (at
  K=20-30% of 8192 = 1638-2458 neurons).

  By pruning each sparse layer's FFN down to its top-K neurons —
  permanently dropping the 70-80% of low-frequency neurons — we get
  a smaller FFN that matches the trained sparsity pattern
  *structurally* instead of via runtime masking. CoreML static
  graphs handle this trivially: just smaller Linear layers.

  Estimated bandwidth win: ~21% of total model bytes (per the
  empirical calibration), translating to ~1.26× wall-clock on
  bandwidth-bound ANE decode.

Inputs:
  --src       HF model directory (safetensors + config.json)
  --calib     sparsity JSON (output of calibrate_ffn_sparsity.py)
  --keep-pct  fraction of intermediate neurons to retain in sparse
              layers (default 0.20 = 1638 of 8192). Higher than the
              5% trained pattern to give a quality margin.
  --dst       output HF directory (modified safetensors + config.json)

Output:
  * New safetensors with `mlp.gate_proj.weight`, `mlp.up_proj.weight`
    rows reduced to top-K (per layer) for sparse layers.
  * `mlp.down_proj.weight` columns reduced to top-K (per layer).
  * Other layers untouched.
  * Updated config.json with `intermediate_size: [K, K, ..., 8192,
    8192, ...]` reflecting the per-layer reduction.

HF Gemma3n's modeling code supports per-layer intermediate_size as a
list, so the pruned model loads with the stock loader.

After pruning, verify Mac output parity (same generation behavior on a
few prompts) before pursuing CoreML conversion.

Usage:
  pyenv shell lama-cml
  python conversion/calibrate_ffn_sparsity.py \
    --model /tmp/gemma3n-e2b \
    --tokens 4096 \
    --out /tmp/sparsity_gemma3n_4k.json
  python conversion/prune_gemma3n_sparse_ffn.py \
    --src /tmp/gemma3n-e2b \
    --calib /tmp/sparsity_gemma3n_4k.json \
    --keep-pct 0.25 \
    --dst /tmp/gemma3n-e2b-pruned

The pruned checkpoint is a drop-in replacement for HF
inference. CoreML conversion would then route through a Gemma 3n
wrapper (TBD — see ROADMAP) that respects the per-layer
intermediate sizes.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import types as _types
import importlib.machinery as _machinery

# wandb stub (Gemma 3n transformers loading needs it)
if "wandb" not in sys.modules:
    _w = _types.ModuleType("wandb")
    _w.__path__ = []  # type: ignore
    _w.__spec__ = _machinery.ModuleSpec("wandb", loader=None, is_package=True)
    sys.modules["wandb"] = _w

import safetensors.torch
import torch


def load_sparsity_pattern(config_path: str) -> list[float]:
    """Read the activation_sparsity_pattern from a Gemma 3n config."""
    with open(config_path) as f:
        c = json.load(f)
    t = c.get("text_config") or c
    sp = t.get("activation_sparsity_pattern")
    if not isinstance(sp, list):
        raise SystemExit("config has no activation_sparsity_pattern list")
    return sp


def load_topk_per_layer(calib_path: str, keep_pct: float
                         ) -> dict[int, list[int]]:
    """Return {layer_idx: [neuron_indices]} keeping top-K% per layer.

    Picks the closest pre-computed top-K bucket from the calibration
    JSON. The JSON stores top_10pct, top_20pct, top_30pct, top_50pct,
    top_70pct; we pick the smallest one ≥ requested keep_pct so the
    quality margin is conservative.
    """
    with open(calib_path) as f:
        report = json.load(f)
    layers = report["layers"]
    pct = int(round(keep_pct * 100))
    available = [10, 20, 30, 50, 70]
    chosen = next((p for p in available if p >= pct), 70)
    print(f"[prune] requested keep_pct={keep_pct:.2f} → using top_{chosen}pct"
          f" bucket from calibration")
    out: dict[int, list[int]] = {}
    for li, L in layers.items():
        idxs = L["top_k"].get(f"top_{chosen}pct")
        if not isinstance(idxs, list):
            raise SystemExit(f"layer {li} missing top_{chosen}pct in calib")
        out[int(li)] = sorted(idxs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF model directory")
    ap.add_argument("--calib", required=True, help="sparsity JSON")
    ap.add_argument("--keep-pct", type=float, default=0.25,
                    help="fraction of intermediate neurons to retain in "
                         "sparse layers (default 0.25)")
    ap.add_argument("--dst", required=True, help="output HF directory")
    args = ap.parse_args()

    src = args.src
    dst = args.dst

    if os.path.exists(dst):
        raise SystemExit(f"dst exists: {dst} (refusing to overwrite)")

    cfg_path = os.path.join(src, "config.json")
    sparsity = load_sparsity_pattern(cfg_path)
    print(f"[prune] sparsity pattern (first 30): {sparsity[:30]}")
    sparse_layer_idxs = [i for i, s in enumerate(sparsity) if s > 0]
    print(f"[prune] {len(sparse_layer_idxs)} sparse layers: "
          f"{sparse_layer_idxs}")

    topk = load_topk_per_layer(args.calib, args.keep_pct)
    print(f"[prune] calibration covers {len(topk)} layers")

    # Build new safetensors. Locate all *.safetensors files.
    st_files = sorted(
        f for f in os.listdir(src) if f.endswith(".safetensors")
    )
    if not st_files:
        raise SystemExit(f"no safetensors found in {src}")

    os.makedirs(dst, exist_ok=False)
    # Stage in-memory new tensors. Streaming would be more memory-friendly
    # but Gemma 3n E2B fits in RAM (10 GB).
    print(f"[prune] copying weights from {len(st_files)} shard(s)")
    new_state: dict[str, torch.Tensor] = {}
    per_layer_keep: dict[int, int] = {}

    for st_file in st_files:
        filepath = os.path.join(src, st_file)
        state = safetensors.torch.load_file(filepath, device="cpu")
        for name, tensor in state.items():
            new_state[name] = tensor

    # Process each sparse layer's MLP weights.
    for li in sparse_layer_idxs:
        if li not in topk:
            print(f"[prune] WARN layer {li} missing from calib, skipping")
            continue
        keep = topk[li]
        per_layer_keep[li] = len(keep)
        idx = torch.tensor(keep, dtype=torch.long)

        # HF Gemma3n weight name format:
        # `model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight`
        prefix_options = [
            f"model.language_model.layers.{li}.mlp",
            f"model.layers.{li}.mlp",  # fallback for monomodal variants
        ]
        prefix = None
        for p in prefix_options:
            if f"{p}.gate_proj.weight" in new_state:
                prefix = p
                break
        if prefix is None:
            print(f"[prune] WARN layer {li} mlp weights not found "
                  f"(tried {prefix_options})")
            continue

        gate = new_state[f"{prefix}.gate_proj.weight"]
        up = new_state[f"{prefix}.up_proj.weight"]
        down = new_state[f"{prefix}.down_proj.weight"]

        # Linear layer convention: weight is (out_features, in_features).
        # gate_proj: (intermediate, hidden) — keep rows = `idx`
        # up_proj:   (intermediate, hidden) — keep rows = `idx`
        # down_proj: (hidden, intermediate) — keep columns = `idx`
        new_state[f"{prefix}.gate_proj.weight"] = gate.index_select(0, idx)
        new_state[f"{prefix}.up_proj.weight"] = up.index_select(0, idx)
        new_state[f"{prefix}.down_proj.weight"] = down.index_select(1, idx)
        print(f"[prune]   layer {li:>2} {prefix.split('.')[-2]}: "
              f"{gate.shape[0]} → {len(keep)} neurons")

    # Save shards. Use a single shard for the pruned model if it fits.
    dst_st = os.path.join(dst, "model.safetensors")
    print(f"[prune] saving {len(new_state)} tensors to {dst_st}")
    safetensors.torch.save_file(new_state, dst_st)

    # Update config: turn intermediate_size into a per-layer list.
    with open(cfg_path) as f:
        c = json.load(f)
    t = c.get("text_config") or c
    orig_inter = t["intermediate_size"]
    if isinstance(orig_inter, int):
        orig_inter = [orig_inter] * t["num_hidden_layers"]
    new_inter = list(orig_inter)
    for li, k in per_layer_keep.items():
        new_inter[li] = k
    t["intermediate_size"] = new_inter
    if "text_config" in c:
        c["text_config"] = t
    else:
        c = t
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(c, f, indent=2)
    print(f"[prune] new intermediate_size: {new_inter[:30]}")

    # Copy other files (tokenizer, special tokens, etc.)
    for fname in os.listdir(src):
        if fname.endswith(".safetensors"):
            continue
        if fname == "config.json":
            continue
        src_p = os.path.join(src, fname)
        dst_p = os.path.join(dst, fname)
        if os.path.isdir(src_p):
            shutil.copytree(src_p, dst_p)
        else:
            shutil.copy2(src_p, dst_p)
    # Remove the safetensors index that referenced the original shards.
    idx_path = os.path.join(dst, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        os.remove(idx_path)

    print(f"[prune] done — pruned model at {dst}")
    print("[prune] next: smoke test with HF transformers; if quality OK,")
    print("              wire into CoreML conversion via Gemma 3n wrapper.")


if __name__ == "__main__":
    main()
