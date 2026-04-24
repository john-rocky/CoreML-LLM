#!/usr/bin/env bash
# Compile + sideload the Qwen3-VL 2B stateful chunks to the connected
# iPhone. Mirrors scripts/push_multifunction_prefill.sh pattern.
#
# Builds .mlmodelc from each .mlpackage under --src, copies to
# Documents/Models/qwen3-vl-2b-stateful/qwen3_vl_2b_stateful_chunks/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${1:-/tmp/qwen3vl_stateful/qwen3_vl_2b_stateful_chunks}"
BUNDLE_ID="com.example.CoreMLLLMChat"
REMOTE_DIR="Documents/Models/qwen3-vl-2b-stateful/qwen3_vl_2b_stateful_chunks"

if [ ! -d "$SRC_DIR" ]; then
    echo "source dir not found: $SRC_DIR" >&2
    echo "expected layout: $SRC_DIR/{chunk_0..3.mlpackage, chunk_head.mlpackage, embed_weight.bin}" >&2
    exit 1
fi

DEVICE=$(xcrun devicectl list devices | awk '/connected/{print $3}' | head -1)
if [ -z "$DEVICE" ]; then
    echo "no connected iOS device" >&2
    exit 1
fi
echo "target device $DEVICE"
echo "target bundle $BUNDLE_ID → $REMOTE_DIR"

COMPILE_DIR="$SRC_DIR/_mlmodelc"
mkdir -p "$COMPILE_DIR"

for pkg in "$SRC_DIR"/*.mlpackage; do
    name=$(basename "$pkg" .mlpackage)
    mlc="$COMPILE_DIR/$name.mlmodelc"
    if [ ! -d "$mlc" ] || [ "$pkg" -nt "$mlc" ]; then
        echo "compiling $name.mlpackage..."
        xcrun coremlcompiler compile "$pkg" "$COMPILE_DIR" >/dev/null
    fi
    size=$(du -sh "$mlc" | awk '{print $1}')
    echo "  push $name.mlmodelc ($size)..."
    xcrun devicectl device copy to \
        --device "$DEVICE" \
        --domain-type appDataContainer \
        --domain-identifier "$BUNDLE_ID" \
        --source "$mlc" \
        --destination "$REMOTE_DIR/$name.mlmodelc" > /dev/null
done

EMBED_BIN="$SRC_DIR/embed_weight.bin"
if [ -f "$EMBED_BIN" ]; then
    size=$(du -sh "$EMBED_BIN" | awk '{print $1}')
    echo "  push embed_weight.bin ($size)..."
    xcrun devicectl device copy to \
        --device "$DEVICE" \
        --domain-type appDataContainer \
        --domain-identifier "$BUNDLE_ID" \
        --source "$EMBED_BIN" \
        --destination "$REMOTE_DIR/embed_weight.bin" > /dev/null
fi

echo ""
echo "verifying layout on device..."
xcrun devicectl device info files \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" \
    --subdirectory "$REMOTE_DIR" 2>&1 | grep -E "chunk_|embed_weight" | head -30

echo ""
echo "done. In Xcode: rebuild+run CoreMLLLMChat → Models tab →"
echo "  'Stateful 64-token smoke test' → tap Run."
echo "Expected: decode tok/s ≥ 25 and phys_footprint < 500 MB."
