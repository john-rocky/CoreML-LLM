#!/usr/bin/env bash
# Gate Zero: compile the MLState stub and sideload to the connected iPhone.
# Run: ./scripts/gate_zero_push.sh
#
# After push, rebuild+run CoreMLLLMChat in Xcode, open the Models tab,
# tap "Gate Zero (MLState stub)", tap "Predict on ANE", read result.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/conversion/out/gate_zero_stub.mlpackage"
MLC_DIR="$ROOT/conversion/out"
BUNDLE_ID="com.example.CoreMLLLMChat"

if [ ! -d "$PKG" ]; then
    echo "building stub via Python converter..."
    "$ROOT/conversion/.venv/bin/python" \
        "$ROOT/conversion/gate_zero_mlstate_stub.py" --predict
fi

echo "compiling to mlmodelc..."
xcrun coremlcompiler compile "$PKG" "$MLC_DIR"
MLC="$MLC_DIR/gate_zero_stub.mlmodelc"
if [ ! -d "$MLC" ]; then
    echo "compile failed: no $MLC" >&2
    exit 1
fi
echo "  $(du -sh "$MLC" | awk '{print $1}')  $MLC"

DEVICE=$(xcrun devicectl list devices | awk '/connected/{print $3}' | head -1)
if [ -z "$DEVICE" ]; then
    echo "no connected iOS device" >&2
    exit 1
fi
echo "pushing to $DEVICE → $BUNDLE_ID Documents/gate_zero_stub.mlmodelc"

xcrun devicectl device copy to \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" \
    --source "$MLC" \
    --destination "Documents/gate_zero_stub.mlmodelc"

echo ""
echo "done. In Xcode, build+run CoreMLLLMChat, open Models tab,"
echo "tap Gate Zero (MLState stub) → Predict on ANE. Expected:"
echo "  PASS — predict=X.Xms, ||out||=... ANE accepted MLState + slice_update."
