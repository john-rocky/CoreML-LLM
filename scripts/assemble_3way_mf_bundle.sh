#!/usr/bin/env bash
# Compile the 3-chunk multifunction mlpackages from
# `output/<model>/chunks_3way_fp16kv_mf/` to .mlmodelc and assemble into a
# Mac-runnable / iPhone-pushable bundle.
#
# Inputs:
#   output/gemma4-e2b/chunks_3way_fp16kv_mf/chunk2_3way.mlpackage
#   output/gemma4-e2b/chunks_3way_fp16kv_mf/chunk3_3way.mlpackage
#   output/gemma4-e2b/bundle_diff_logits/  (source of chunk1, prefill_chunk*, embed/RoPE/config/tokenizer/drafter)
#
# Output:
#   output/gemma4-e2b/bundle_3way_mf/
#     chunk1.mlmodelc                 (symlink to bundle_diff_logits)
#     chunk2_3way.mlmodelc            (compiled fresh)
#     chunk3_3way.mlmodelc            (compiled fresh)
#     prefill_chunk{1..4}.mlmodelc    (symlink to bundle_diff_logits)
#     mtp_drafter.mlmodelc            (symlink — centroid drafter)
#     all .npy / .bin / .json / hf_model (symlink)
#
# Usage:
#   bash scripts/assemble_3way_mf_bundle.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-gemma4-e2b}"
SRC_BUNDLE="$ROOT/output/$MODEL/bundle_diff_logits"
MF_DIR="$ROOT/output/$MODEL/chunks_3way_fp16kv_mf"
OUT="$ROOT/output/$MODEL/bundle_3way_mf"

mkdir -p "$OUT"

# 1. Compile multifunction mlpackages to mlmodelc.
for pkg in chunk2_3way chunk3_3way; do
  src="$MF_DIR/$pkg.mlpackage"
  dst="$OUT/$pkg.mlmodelc"
  if [ ! -d "$src" ]; then
    echo "[assemble] ERROR: missing $src — run build_verify_chunks_3way.py first" >&2
    exit 1
  fi
  echo "[assemble] compiling $pkg ..."
  rm -rf "$dst"
  xcrun coremlcompiler compile "$src" "$OUT/" > /dev/null
  # `coremlcompiler compile` writes into <OUT>/<pkg>.mlmodelc when source
  # is named <pkg>.mlpackage, so $dst already points to the right path.
  test -d "$dst" || { echo "[assemble] FAILED to compile $pkg" >&2; exit 1; }
  size=$(du -sh "$dst" | awk '{print $1}')
  echo "[assemble]   → $dst ($size)"
done

# 2. Symlink chunk1 + prefill + supporting files from bundle_diff_logits.
for f in chunk1.mlmodelc \
         prefill_chunk1.mlmodelc prefill_chunk2.mlmodelc \
         prefill_chunk3.mlmodelc prefill_chunk4.mlmodelc \
         mtp_drafter.mlmodelc \
         cos_full.npy cos_sliding.npy sin_full.npy sin_sliding.npy \
         embed_tokens_q8.bin embed_tokens_scales.bin \
         embed_tokens_per_layer_q8.bin embed_tokens_per_layer_scales.bin \
         per_layer_projection.bin per_layer_norm_weight.bin \
         model_config.json hf_model; do
  src="$SRC_BUNDLE/$f"
  if [ ! -e "$src" ]; then
    echo "[assemble] WARN: $f not in $SRC_BUNDLE — skipping"
    continue
  fi
  rm -f "$OUT/$f"
  ln -s "$src" "$OUT/$f"
done

echo ""
echo "[assemble] bundle written to: $OUT"
echo ""
echo "Run Mac smoke:"
echo "  SPECULATIVE_PROFILE=1 MTP_FORCE_SPECULATE=1 MTP_MODE=mtp \\"
echo "    .build/debug/coreml-llm-smoke \"$OUT\" \\"
echo "    \"Write a Python class implementing a binary search tree.\" 256"
