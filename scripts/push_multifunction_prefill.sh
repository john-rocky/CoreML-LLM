#!/usr/bin/env bash
# Push the 4 compiled multifunction prefill mlmodelc directories to the
# connected iPhone, replacing the existing prefill_chunk{1..4}.mlmodelc
# in the app's Documents/Models/gemma4-e2b/ directory.
#
# Requires a prior backup (see docs/USB_MODEL_SIDELOAD.md). DO NOT use
# --remove-existing-content true on subfolder destinations — it wipes
# sibling files in the parent (chunks, embeddings, etc.).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPILED_DIR="${1:-$ROOT/output/gemma4-e2b/prefill_multifunction_compiled}"
BUNDLE_ID="com.example.CoreMLLLMChat"
REMOTE_DIR="Documents/Models/gemma4-e2b"

DEVICE=$(xcrun devicectl list devices | awk '/connected/{print $3}' | head -1)
if [ -z "$DEVICE" ]; then
    echo "no connected device" >&2
    exit 1
fi
echo "pushing to device $DEVICE"

for i in 1 2 3 4; do
    SRC="$COMPILED_DIR/prefill_chunk${i}.mlmodelc"
    DST="$REMOTE_DIR/prefill_chunk${i}.mlmodelc"
    if [ ! -d "$SRC" ]; then
        echo "missing $SRC — aborting"
        exit 1
    fi
    size=$(du -sh "$SRC" | awk '{print $1}')
    echo "  push prefill_chunk${i}.mlmodelc ($size)..."
    xcrun devicectl device copy to \
        --device "$DEVICE" \
        --domain-type appDataContainer \
        --domain-identifier "$BUNDLE_ID" \
        --source "$SRC" \
        --destination "$DST"
done

echo ""
echo "verifying on-device layout..."
xcrun devicectl device info files \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" \
    --subdirectory "$REMOTE_DIR" 2>&1 | grep -E "prefill_chunk" | head -20

echo ""
echo "done. Now launch the app with LLM_PREFILL_MULTIFUNCTION=1 set in"
echo "the Xcode scheme's Environment Variables. Look for:"
echo "  [Load] LLM_PREFILL_MULTIFUNCTION=1 — will discover variants b64,b128,b256"
echo "  [Load] prefill variant b64 loaded in X.Xs"
echo "  [Load] prefill variant b128 loaded in X.Xs"
echo "  [Load] prefill variant b256 loaded in X.Xs"
