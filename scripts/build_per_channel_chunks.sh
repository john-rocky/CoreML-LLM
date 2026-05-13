#!/usr/bin/env bash
# Build Gemma 4 E2B 3-chunk per-channel INT4 chunks (T7 lever).
#
# Per-channel INT4 (MobileLLM-Pro arxiv 2511.06719) beats group-wise on
# accelerator NPUs by avoiding LUT-decode stall. PPL hit 1.3% vs 0.4%
# (negligible for chat).
#
# Output: output/gemma4-e2b/chunks_3way_perch/  +  bundle_3way_perch/
#
# After this, run scripts/assemble_3way_mf_bundle.sh-equivalent + push.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

pyenv shell lama-cml 2>/dev/null || true

# Step 1: produce fp16 chunks via the existing 3way builder (no quantize).
FP16_DIR="$ROOT/output/gemma4-e2b/chunks_3way_fp16_unquant"
PERCH_DIR="$ROOT/output/gemma4-e2b/chunks_3way_perch"

if [ ! -d "$FP16_DIR/chunk2_3way.mlpackage" ]; then
  echo "[step 1/3] building fp16 unquantized 3-chunk variants → $FP16_DIR"
  python "$ROOT/conversion/build_gemma4_3way.py" \
    --model gemma4-e2b --ctx 2048 \
    --output "$FP16_DIR" --no-quantize
else
  echo "[step 1/3] fp16 chunks already present at $FP16_DIR — skipping"
fi

# Step 2: re-palettize per-channel.
if [ ! -d "$PERCH_DIR/chunk2_3way.mlpackage" ]; then
  echo "[step 2/3] re-palettize per-channel INT4 → $PERCH_DIR"
  python "$ROOT/conversion/rebuild_chunks_per_channel.py" \
    --src "$FP16_DIR" --dst "$PERCH_DIR" \
    --nbits 4 --granularity per_channel
else
  echo "[step 2/3] per-channel chunks already at $PERCH_DIR — skipping"
fi

# Step 3: compile to .mlmodelc.
BUNDLE_DIR="$ROOT/output/gemma4-e2b/bundle_3way_perch"
mkdir -p "$BUNDLE_DIR"
echo "[step 3/3] compiling .mlpackage → .mlmodelc in $BUNDLE_DIR"
for pkg in "$PERCH_DIR"/*.mlpackage; do
  name=$(basename "$pkg" .mlpackage)
  if [ ! -d "$BUNDLE_DIR/$name.mlmodelc" ]; then
    echo "  compiling $name ..."
    xcrun coremlcompiler compile "$pkg" "$BUNDLE_DIR/" > /dev/null
  fi
done

# Step 4: symlink the rest of bundle_diff_logits (chunk1, prefill, drafter,
# embed, tokenizer) so the resulting bundle is iPhone-pushable.
SRC_BUNDLE="$ROOT/output/gemma4-e2b/bundle_diff_logits"
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
  [ -e "$src" ] || continue
  rm -f "$BUNDLE_DIR/$f"
  ln -s "$src" "$BUNDLE_DIR/$f"
done

echo ""
echo "[done] bundle ready at $BUNDLE_DIR"
echo ""
echo "Mac smoke:"
echo "  MTP_MODE=t1 .build/debug/coreml-llm-smoke $BUNDLE_DIR \"Write a Python class.\" 128"
echo ""
echo "iPhone push (dereferenced — symlinks won't push as-is):"
echo "  # Reuse /tmp/push-bundle pattern from earlier, point at $BUNDLE_DIR"
