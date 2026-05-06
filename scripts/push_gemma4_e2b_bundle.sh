#!/usr/bin/env bash
# Push the full local Gemma 4 E2B bundle to iPhone 17 Pro.
# Includes: chunks 1-4 (production INT4), prefill chunks 1-4, mtp_drafter
# (centroid version overrides the production drafter), embed bins, RoPE
# tables, per-layer projection, model config, and tokenizer.
#
# Usage:
#   bash scripts/push_gemma4_e2b_bundle.sh [bundle-dir]
#
# Default bundle-dir: output/gemma4-e2b/bundle
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE="${1:-$ROOT/output/gemma4-e2b/bundle}"
CENTROID_DRAFTER="/tmp/mtp_drafter_centroid_out/mtp_drafter_centroid.mlmodelc"
BUNDLE_ID="com.example.CoreMLLLMChat"
REMOTE_DIR="Documents/Models/gemma4-e2b"

if [ ! -d "$BUNDLE" ]; then
  echo "missing $BUNDLE" >&2
  exit 1
fi

DEVICE=$(xcrun devicectl list devices 2>/dev/null \
  | grep "iPhone 17 Pro" | grep "connected" \
  | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}' \
  | head -1)
if [ -z "$DEVICE" ]; then
  echo "iPhone 17 Pro not connected" >&2
  xcrun devicectl list devices >&2
  exit 1
fi

echo "[push] device=$DEVICE  bundle=$BUNDLE  -> $REMOTE_DIR"
echo "[push] total bundle size: $(du -sh "$BUNDLE" | awk '{print $1}')"
echo ""

push_one() {
  local src="$1"
  local rel="$(basename "$src")"
  local size=$(du -sh "$src" 2>/dev/null | awk '{print $1}')
  echo "[push]   $rel ($size)..."
  xcrun devicectl device copy to \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" \
    --source "$src" \
    --destination "$REMOTE_DIR/$rel" 2>&1 \
    | grep -aE "Path:|Acquired tunnel|Error|fail" | head -2
}

# 1. Chunks (decode + prefill).
for f in chunk1.mlmodelc chunk2.mlmodelc chunk3.mlmodelc chunk4.mlmodelc \
         prefill_chunk1.mlmodelc prefill_chunk2.mlmodelc \
         prefill_chunk3.mlmodelc prefill_chunk4.mlmodelc; do
  push_one "$BUNDLE/$f"
done

# 2. Embed + per-layer + RoPE tables + config + tokenizer.
for f in embed_tokens_q8.bin embed_tokens_scales.bin \
         embed_tokens_per_layer_q8.bin embed_tokens_per_layer_scales.bin \
         per_layer_projection.bin per_layer_norm_weight.bin \
         cos_full.npy sin_full.npy cos_sliding.npy sin_sliding.npy \
         model_config.json hf_model; do
  if [ -e "$BUNDLE/$f" ]; then
    push_one "$BUNDLE/$f"
  fi
done

# 3. Drafter — push the centroid version if present, else fall back to
#    the bundle's drafter (older full-vocab build).
if [ -d "$CENTROID_DRAFTER" ]; then
  echo "[push] using centroid drafter from $CENTROID_DRAFTER"
  TMPDIR=$(mktemp -d)
  trap "rm -rf '$TMPDIR'" EXIT
  cp -R "$CENTROID_DRAFTER" "$TMPDIR/mtp_drafter.mlmodelc"
  push_one "$TMPDIR/mtp_drafter.mlmodelc"
else
  echo "[push] no centroid drafter at $CENTROID_DRAFTER; pushing bundle's drafter"
  push_one "$BUNDLE/mtp_drafter.mlmodelc"
fi

echo ""
echo "[push] done. Verifying..."
xcrun devicectl device info files \
  --device "$DEVICE" \
  --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" \
  --subdirectory "$REMOTE_DIR" 2>&1 \
  | awk 'NR>5 && $1 !~ /^-/ && NF>2 {print $1}' \
  | grep -aE "^(chunk[1-4]|prefill_chunk[1-4]|mtp_drafter|embed_tokens|per_layer|cos_|sin_|model_config|hf_model)" \
  | awk -F/ '{print $1}' | sort -u | head -20
