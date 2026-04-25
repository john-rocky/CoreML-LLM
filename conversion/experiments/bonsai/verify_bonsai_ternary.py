"""Verify that Bonsai's FP16 unpacked weights really are ternary per 128-group.

If yes → nbits=2 + mode="unique" + per_grouped_channel + group_size=128 gives
a bit-exact Core ML palettization. If the groups have float noise around the
3 ternary centroids, we need a custom LUT builder.

For each sampled weight tensor we report the distribution of unique-values-per-group
and the actual values in a few example groups.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import safetensors.torch


SAMPLE_TENSORS = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.5.mlp.gate_proj.weight",
    "model.layers.5.mlp.up_proj.weight",
    "model.layers.5.mlp.down_proj.weight",
    "model.layers.13.self_attn.q_proj.weight",
    "model.layers.27.self_attn.o_proj.weight",
    "model.layers.27.mlp.down_proj.weight",
]


def analyze_tensor(name: str, w: np.ndarray, group_size: int, sample_rows: int,
                   axis: int) -> None:
    print(f"\n=== {name}  shape={w.shape}  dtype={w.dtype}  "
          f"group along axis={axis} ===")
    if w.ndim != 2:
        print(f"  skipping ({w.ndim}d, only 2d supported)")
        return

    # Always group along axis; bring it to last for simplicity.
    if axis == 0:
        w2 = w.T
    else:
        w2 = w
    rows, cols = w2.shape
    if cols % group_size != 0:
        print(f"  skipping ({cols} % {group_size} != 0)")
        return
    n_groups_per_row = cols // group_size
    rows_to_sample = min(sample_rows, rows)

    uniq_counts: list[int] = []
    example_group_values: list[np.ndarray] = []

    for ri in np.linspace(0, rows - 1, rows_to_sample).astype(int):
        row = w2[ri]
        for gi in range(n_groups_per_row):
            group = row[gi * group_size : (gi + 1) * group_size]
            u = np.unique(group)
            uniq_counts.append(len(u))
            if len(example_group_values) < 3:
                example_group_values.append(u)

    cnt = np.array(uniq_counts)
    total = len(cnt)
    print(f"  sampled {total} groups across {rows_to_sample} rows × {n_groups_per_row} groups")
    # Distribution
    for k in [1, 2, 3, 4, 5, 6, 8, 16, 32, 64, 128]:
        n = int((cnt == k).sum())
        if n > 0:
            print(f"    exactly {k:>3} unique: {n:>6} ({100 * n / total:5.1f}%)")
    # Buckets for "weird" groups
    le3 = int((cnt <= 3).sum())
    le4 = int((cnt <= 4).sum())
    le8 = int((cnt <= 8).sum())
    more = int((cnt > 8).sum())
    print(f"  cumulative: ≤3 unique = {100 * le3 / total:.2f}%, "
          f"≤4 = {100 * le4 / total:.2f}%, "
          f"≤8 = {100 * le8 / total:.2f}%, "
          f">8 = {100 * more / total:.2f}%")
    print(f"  max unique in any group: {int(cnt.max())}, "
          f"min: {int(cnt.min())}, mean: {cnt.mean():.2f}")
    for i, u in enumerate(example_group_values):
        vals_str = ", ".join(f"{v:+.6f}" for v in u[: min(10, len(u))])
        if len(u) > 10:
            vals_str += ", ..."
        print(f"  example group {i}: {len(u)} unique — [{vals_str}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="../output/bonsai/hf_model")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--sample-rows", type=int, default=20,
                    help="rows sampled per tensor (cost vs coverage)")
    ap.add_argument("--axis", type=int, default=1, choices=[0, 1],
                    help="axis to group along; 1=last dim (in-channel), 0=first dim")
    args = ap.parse_args()

    weights_path = Path(args.model_path) / "model.safetensors"
    print(f"Loading {weights_path}")
    state = safetensors.torch.load_file(str(weights_path))
    print(f"  {len(state)} tensors loaded")

    for name in SAMPLE_TENSORS:
        if name not in state:
            print(f"\n=== {name}: NOT FOUND ===")
            continue
        t = state[name].float().cpu().numpy()
        analyze_tensor(name, t, args.group_size, args.sample_rows, args.axis)

    print("\n=== interpretation ===")
    print("  If most tensors show ≤3 unique values per group (>99%) → bit-exact")
    print("  palettization via nbits=2 + mode='unique' + per_grouped_channel +")
    print(f"  group_size={args.group_size} should work losslessly.")
    print("  If 4+ unique values appear often, the unpacked FP16 has numerical")
    print("  noise around the ternary centroids → needs custom LUT construction.")


if __name__ == "__main__":
    main()
