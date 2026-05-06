#!/usr/bin/env bash
# Push the centroid (MaskedEmbedder) MTP drafter (and optionally fresh
# verify chunks) to the connected iPhone for E2B or E4B.
#
# Usage:
#   bash scripts/push_centroid_drafter.sh [e2b|e4b] [drafter|chunks|all]
#
# Default: e2b drafter only (fastest path to test the drafter swap;
# production chunks already tolerate the centroid drafter).
#
# Pre-built artifacts expected under /tmp/:
#   /tmp/mtp_drafter_centroid_out/mtp_drafter_centroid.mlmodelc        (E2B)
#   /tmp/mtp_drafter_centroid_e4b_out/mtp_drafter_centroid_e4b.mlmodelc (E4B)
#   /tmp/gemma4_chunks_K3_fresh/chunk{1..4}.mlpackage  (E2B fresh INT4)
#   /tmp/gemma4_e4b_chunks_K3/chunk{1..4}.mlpackage    (E4B fresh INT4)
#
# Build commands (if missing):
#   E2B drafter:
#     ~/.pyenv/versions/lama-cml/bin/python conversion/build_mtp_drafter.py \
#       --hf-repo google/gemma-4-E2B-it-assistant \
#       --output /tmp/mtp_drafter_centroid.mlpackage \
#       --sliding-window 512 --context-length 2048 --centroid-lm-head
#     xcrun coremlcompiler compile /tmp/mtp_drafter_centroid.mlpackage /tmp/mtp_drafter_centroid_out
#
#   E4B drafter:
#     ~/.pyenv/versions/lama-cml/bin/python conversion/build_mtp_drafter.py \
#       --hf-repo google/gemma-4-E4B-it-assistant \
#       --output /tmp/mtp_drafter_centroid_e4b.mlpackage \
#       --sliding-window 512 --context-length 2048 --centroid-lm-head --target e4b
#     xcrun coremlcompiler compile /tmp/mtp_drafter_centroid_e4b.mlpackage /tmp/mtp_drafter_centroid_e4b_out

set -euo pipefail

VARIANT="${1:-e2b}"
SCOPE="${2:-drafter}"

case "$VARIANT" in
  e2b)
    REMOTE_DIR="Documents/Models/gemma4-e2b"
    DRAFTER_LOCAL="/tmp/mtp_drafter_centroid_out/mtp_drafter_centroid.mlmodelc"
    CHUNKS_PKG_DIR="/tmp/gemma4_chunks_K3_fresh"
    ;;
  e4b)
    REMOTE_DIR="Documents/Models/gemma4-e4b"
    DRAFTER_LOCAL="/tmp/mtp_drafter_centroid_e4b_out/mtp_drafter_centroid_e4b.mlmodelc"
    CHUNKS_PKG_DIR="/tmp/gemma4_e4b_chunks_K3"
    ;;
  *)
    echo "usage: $0 [e2b|e4b] [drafter|chunks|all]" >&2
    exit 2
    ;;
esac
case "$SCOPE" in
  drafter|chunks|all) ;;
  *)
    echo "scope must be drafter|chunks|all" >&2
    exit 2
    ;;
esac

BUNDLE_ID="com.example.CoreMLLLMChat"
# Match a 36-char UUID followed eventually by "connected" on iPhone 17 Pro.
DEVICE=$(xcrun devicectl list devices 2>/dev/null \
  | grep "iPhone 17 Pro" | grep "connected" \
  | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}' \
  | head -1)
if [ -z "$DEVICE" ]; then
  echo "iPhone 17 Pro not connected — please plug in + unlock and retry" >&2
  echo "Currently visible:" >&2
  xcrun devicectl list devices 2>&1 | sed 's/^/  /' >&2
  exit 1
fi
echo "[push] device=$DEVICE  variant=$VARIANT  scope=$SCOPE  remote=$REMOTE_DIR"

push_one() {
  local src="$1" dst="$2" label="$3"
  if [ ! -d "$src" ]; then
    echo "[push] missing $label at $src — skip" >&2
    return 1
  fi
  local size
  size=$(du -sh "$src" | awk '{print $1}')
  echo "[push]   $label ($size) -> $dst"
  xcrun devicectl device copy to \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" \
    --source "$src" \
    --destination "$dst"
}

# Drafter swap.
if [ "$SCOPE" = "drafter" ] || [ "$SCOPE" = "all" ]; then
  push_one "$DRAFTER_LOCAL" "$REMOTE_DIR/mtp_drafter.mlmodelc" \
           "$VARIANT mtp_drafter.mlmodelc"
fi

# Chunks swap (compile if needed).
if [ "$SCOPE" = "chunks" ] || [ "$SCOPE" = "all" ]; then
  for i in 1 2 3 4; do
    PKG="$CHUNKS_PKG_DIR/chunk${i}.mlpackage"
    if [ ! -d "$PKG" ]; then
      echo "[push] missing $PKG — build verify chunks first:" >&2
      echo "       ~/.pyenv/versions/lama-cml/bin/python \\" >&2
      echo "         conversion/build_verify_chunks.py --model gemma4-${VARIANT} \\" >&2
      echo "         --K 3 --output $CHUNKS_PKG_DIR --ctx 2048" >&2
      exit 1
    fi
    TMPDIR=$(mktemp -d)
    trap "rm -rf '$TMPDIR'" RETURN
    xcrun coremlcompiler compile "$PKG" "$TMPDIR" >/dev/null 2>&1
    push_one "$TMPDIR/chunk${i}.mlmodelc" "$REMOTE_DIR/chunk${i}.mlmodelc" \
             "$VARIANT chunk${i}.mlmodelc"
    rm -rf "$TMPDIR"
  done
fi

echo ""
echo "[push] verifying on-device files..."
xcrun devicectl device info files \
  --device "$DEVICE" \
  --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" \
  --subdirectory "$REMOTE_DIR" 2>&1 \
  | grep -aE "mtp_drafter|chunk[1-4]\.mlmodelc" | head -10

cat <<EOF

[push] done.

Run the app:
  - Set scheme env var SPECULATIVE_PROFILE=1 to print [SpecProfile mtp ...]
  - Optional: MTP_DRAFT_POS_MODE=constpm1 (default in latest binary)
  - Optional: MTP_DRAFTER_DEVICE=cpu / gpu / ane (default ane)

Watch for:
  [MTP] Drafter loaded (K=3)
  [SpecProfile mtp #N] draft=Xms verify=Yms accepted=A/B emitted=N rolling=Z

Expected vs prior full-vocab drafter on iPhone 17 Pro:
  E2B: per-slot accept ~0.13 -> 0.20-0.27, tok/s ~31 -> 36-42
  E4B: high-accept content +2x; chat code likely net regression
       (see docs/MTP_CENTROID_LM_HEAD_BREAKTHROUGH.md)

If accept = 0.00 or output is gibberish:
  - iPhone ANE 18 may demote ints differently than Mac.
    Try MTP_DRAFTER_DEVICE=cpu in scheme env to bypass ANE for the drafter.
  - Per memory: layout swaps require app delete + reinstall to avoid
    devicectl orphan-file silent overrides.
EOF
