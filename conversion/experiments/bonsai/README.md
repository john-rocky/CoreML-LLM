# Bonsai (1.58-bit ternary) — Investigation Artifacts

These scripts attempted to bring `prism-ml/Ternary-Bonsai-1.7B` to Apple
Neural Engine via Core ML. **The investigation concluded that ANE cannot
faithfully run Bonsai's per-(row, block) ternary structure.** Apple's ANE
compiler rejects per-block LUT palettization (`error code: -14`), and
working around it (per-tensor / per-channel kmeans) collapses Bonsai's
core design — the per-block independent scales — into a rank-1 outer
product. At that point we'd be shipping "Qwen3-1.7B with palette quant",
not Bonsai. So we don't ship.

The full post-mortem and the path forward (MLX Swift for Bonsai-class
models) is in `docs/TERNARY_BONSAI.md`.

## What's here, briefly

| File | Purpose | Result |
|---|---|---|
| `bonsai_reference_oracle.py` | HF vs our `Qwen3Model` parity, 5-token greedy match | **Pass** — confirmed `models/qwen3.py` correctness |
| `build_bonsai_17b_decode_chunks.py` | 2-chunk INT4/INT8 + optional SWA decode build | **Pass** — produced ANE-running INT4 at 24 tok/s, but quality is approximate Qwen3, not faithful Bonsai |
| `verify_bonsai_ternary.py` | Validates per-128-block ternary structure of unpacked FP16 | **Pass** — 100% of sampled 128-groups have exactly 3 unique values |
| `ternary_surgery.py` | Custom MIL pass: per-(row, block) `constexpr_lut_to_dense` palettization | **Pass to save, fail at load** — ANE compiler -14 |
| `test_bonsai_inference.py`, `test_bonsai_chunks_inference.py` | Smoke + benchmark | Used during investigation |
| `compare_swa_vs_full.py`, `compare_swa_long_range.py` | SWA-vs-full divergence measurements | Found long-range recall regression with sinks=0 SWA |

## Reusable bits that escaped to `conversion/`

These are the parts of the work that landed in the main codebase:

- `models/qwen3.py` — Qwen3 architecture support (QK-norm, tied embed,
  no attention bias). Useful for Qwen3-1.7B / 4B / 8B and any QK-normed
  Qwen-family model.
- `base_model.py` — `ModelConfig.has_qk_norm` flag and conditional
  `q_norm` / `k_norm` modules in `ANEAttention`. Backward-compatible
  default (`has_qk_norm=False`) so Qwen2 / Gemma builds are unchanged.
- `exporter.py` — `MonolithicWrapper` applies QK-norm when the layer
  has `has_qk_norm=True`.
- `convert.py` — `qwen3` architecture routes to `Qwen3Model`.
- `docs/DECODE_STATE_LAYOUTS.md` — captured ANE decode-path lessons
  including the per-block palette finding.

## If you want to actually run Bonsai on Apple Silicon

Use MLX, not Core ML / ANE:

```bash
pip install mlx-lm
mlx_lm.generate \
  --model prism-ml/Ternary-Bonsai-1.7B-mlx-2bit \
  --prompt "..."
```

`mlx-lm` natively supports the 2-bit packed ternary format with `mx.quantized_matmul`,
preserving the per-block scale structure. Runs on Apple Silicon GPU at full fidelity.

For Swift integration, see [`mlx-swift-examples`](https://github.com/ml-explore/mlx-swift-examples).
