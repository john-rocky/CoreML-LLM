# Bonsai 1.58-bit on Core ML / ANE — Investigation Post-Mortem

**Status:** investigated and **not shipped**. Use MLX for Bonsai instead.

## Goal

Bring [`prism-ml/Ternary-Bonsai-1.7B`](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B)
to Apple Neural Engine (iPhone 17 Pro), preserving the author's 1.58-bit ternary
weight encoding so the model's structural compression advantage carries over
into Core ML.

## What we built

A complete Qwen3 conversion path that didn't exist before:

- `conversion/models/qwen3.py` — `Qwen3Model` (QK-norm, tied embed, no attention bias)
- `conversion/base_model.py` — optional QK-norm in `ANEAttention`, off by default
- `conversion/exporter.py` — `MonolithicWrapper` honors QK-norm
- `conversion/convert.py` — `qwen3` architecture routing
- `docs/DECODE_STATE_LAYOUTS.md` — generalized decode-path lessons that came out
  of this work (chunking, SWA, palette traps)

These all stay in the codebase; they're useful for any Qwen3 derivative.

## Bonsai-specific work, summarized

Verified the per-128-block structure of the unpacked FP16 weights — 100% of
sampled groups have exactly 3 unique values `{−s, 0, +s}`, scale `s` varies
per (row, block). Built three Core ML variants and measured them on Mac ANE:

| variant | size | speed | top-1 vs Bonsai/MLX | ANE OK |
|---|---|---|---|---|
| INT4 k-means per_tensor (chunked) | 1.0 GB | 24 tok/s | matches "Paris" but logits are coarse | yes |
| nbits=6 unique per-row palette | 1.7 GB | 11 tok/s | drifts (different output) | yes |
| **bit-exact per-(row,block) LUT** (custom MIL surgery) | **0.4 GB / 1.0 GB** | **N/A** | **would be exact** | **NO — ANEC error -14** |
| SWA mask-based rotating buffer at ctx=4096/W=1024 | 1.0 GB | 25 tok/s | matches at short ctx, forgets long-range | yes |

Reference build: `conversion/experiments/bonsai/build_bonsai_17b_decode_chunks.py`.
Bit-exact MIL surgery: `conversion/experiments/bonsai/ternary_surgery.py`.

## The blocking finding

Bonsai's compression depends on **per-(row, block) independent scales** — for
a (2048, 2048) layer, that's 32,768 distinct scales arranged as a (2048, 16)
matrix. Two ways to express this in Core ML:

1. **Single-op `constexpr_lut_to_dense`** with LUT shape `(2048, 16, 4, 1)`,
   where each (row, block) has its own 4-entry codebook `[0, +s, -s, 0]`.
2. **Two-op chain** — `constexpr_lut_to_dense` with shared sign codebook
   `[0, +1, -1, 0]` followed by `constexpr_blockwise_shift_scale` carrying
   the (2048, 16) scale matrix.

Both **load as MLModel and serialize fine**. Both **fail Apple's ANE compiler
(`error code: -14`)** when iOS tries to build the execution plan. The model
loads as a CPU-stub and `make_state()` throws "This model was not loaded
with the Core ML Framework."

The granularity ANE accepts in iOS18 is **per-tensor or per-grouped-channel
along a single axis** — there is no current ANE kernel that handles a
per-(row, block) palette layout. Until Apple adds support, Bonsai's structure
cannot be faithfully run on ANE.

## Why "approximate but ANE-running" doesn't work

The next-best approximation in stock coremltools is `nbits=2 per_grouped_channel
+ enable_per_channel_scale`: per-block LUT (16 LUTs, 4 codes each) plus a
per-row scale factor. This is rank-1: `s_{r,b} ≈ c_b · d_r`. It compiles for
ANE, but **discards the per-(row, block) scale independence** — the very thing
that justifies Bonsai's training procedure.

If you ship that, you're shipping "Qwen3-1.7B with structured palette quant",
not Bonsai. There is no point in pulling Bonsai's weights specifically; any
Qwen3-1.7B variant gives equivalent results through the same path. So we don't
ship it.

## What to do if you want Bonsai

Use [MLX](https://github.com/ml-explore/mlx) and `mlx-lm`. The published
`prism-ml/Ternary-Bonsai-1.7B-mlx-2bit` weights run on Apple Silicon GPU
via `mx.quantized_matmul` with native 2-bit packed ternary, preserving the
per-block scales. Reported speed: ~27 tok/s on iPhone 17 Pro Max for the 8B
class; the 1.7B should be substantially faster.

For Swift integration, [`mlx-swift-examples`](https://github.com/ml-explore/mlx-swift-examples)
provides drop-in patterns. ANE is not used; this is a GPU path.

## Knowledge harvested for the rest of the codebase

The Bonsai investigation produced reusable lessons recorded in
[`docs/DECODE_STATE_LAYOUTS.md`](DECODE_STATE_LAYOUTS.md):

- Mask-based circular rotating KV buffer (replaces shift-based `cat([K[:,1:], k])`
  which ANEC rejects on Qwen3 + Stateful + tied embed)
- ANE per-step decode cost is `O(state_length)`, not weight bandwidth, in this
  model class — context reduction beats more aggressive quant
- `mode="kmeans"` palettization is the safe default; `mode="unique"` falls back
  silently when global tensor uniqueness exceeds nbits range
- Trace-time `TracerWarning` on stateful modules is suppressed by `strict=False`
- `audit_ane` must wrap `get_compiled_model_path()` in try/except to survive
  ANEC -14 saves

## Files

- `conversion/experiments/bonsai/README.md` — manifest of the experiment scripts
- `conversion/experiments/bonsai/*` — full set of build, parity, and surgery
  scripts that we walked through during the investigation
- This doc — the human-readable summary
