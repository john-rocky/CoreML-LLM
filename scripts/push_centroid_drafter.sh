#!/usr/bin/env bash
# Push the centroid (MaskedEmbedder) MTP drafter to the connected iPhone.
# Replaces Documents/Models/gemma4-e2b/mtp_drafter.mlmodelc.
#
# Build first:
#   ~/.pyenv/versions/lama-cml/bin/python conversion/build_mtp_drafter.py \
#     --hf-repo google/gemma-4-E2B-it-assistant \
#     --output /tmp/mtp_drafter_centroid.mlpackage \
#     --sliding-window 512 --context-length 2048 \
#     --centroid-lm-head
#
#   xcrun coremlcompiler compile /tmp/mtp_drafter_centroid.mlpackage \
#     /tmp/mtp_drafter_centroid_out
#
# Then run this script.
set -euo pipefail

SRC="${1:-/tmp/mtp_drafter_centroid_out/mtp_drafter_centroid.mlmodelc}"
BUNDLE_ID="com.example.CoreMLLLMChat"
REMOTE_DIR="Documents/Models/gemma4-e2b"

if [ ! -d "$SRC" ]; then
    echo "missing $SRC — build + compile first (see docstring)" >&2
    exit 1
fi

DEVICE=$(xcrun devicectl list devices | awk '/connected/{print $3}' | head -1)
if [ -z "$DEVICE" ]; then
    echo "no connected device" >&2
    exit 1
fi
echo "pushing centroid drafter to $DEVICE"

size=$(du -sh "$SRC" | awk '{print $1}')
echo "  push mtp_drafter.mlmodelc ($size)..."
xcrun devicectl device copy to \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" \
    --source "$SRC" \
    --destination "$REMOTE_DIR/mtp_drafter.mlmodelc"

echo ""
echo "verifying on-device layout..."
xcrun devicectl device info files \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" \
    --subdirectory "$REMOTE_DIR" 2>&1 | grep -E "mtp_drafter" | head -5

echo ""
echo "done. Launch app with SPECULATIVE_PROFILE=1 set in scheme env, look for:"
echo "  [MTP] Drafter loaded (K=3)"
echo "  [SpecProfile mtp #NNNN] draft=X.Xms verify=Y.Yms accepted=A/B emitted=N rolling=Z.ZZZ"
echo ""
echo "Expected on iPhone 17 Pro vs prior full-vocab drafter:"
echo "  per-slot accept: 0.13 → 0.20-0.25 (target)"
echo "  tok/s:           ~31  → 36-42  (target)"
