#!/usr/bin/env bash
# Generic iPhone bundle deref + push helper.
#
# Takes a bundle directory (which may contain symlinks pointing back to
# bundle_diff_logits/), produces a fully-dereferenced copy in /tmp/, and
# pushes to iPhone 17 Pro's Documents/Models/<remote-name>/ via
# devicectl.
#
# Usage:
#   bash scripts/push_bundle_to_iphone.sh <local-bundle-dir> [<remote-name>]
#
# Examples:
#   bash scripts/push_bundle_to_iphone.sh output/gemma4-e2b/bundle_3way_perch gemma4-e2b
#       → derefs bundle_3way_perch (which symlinks chunk1, prefill, drafter, etc
#         from bundle_diff_logits), pushes to Documents/Models/gemma4-e2b/
#         on iPhone — overwrites the default bundle.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:?usage: <local-bundle-dir> [<remote-name>]}"
REMOTE_NAME="${2:-gemma4-e2b}"
BUNDLE_ID="com.example.CoreMLLLMChat"
REMOTE_DIR="Documents/Models/$REMOTE_NAME"

if [ ! -d "$SRC" ]; then
  echo "missing source: $SRC" >&2
  exit 1
fi

DEVICE=$(xcrun devicectl list devices 2>/dev/null \
  | grep "iPhone 17 Pro" | grep "connected" \
  | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}' \
  | head -1)
if [ -z "$DEVICE" ]; then
  echo "iPhone 17 Pro not connected" >&2
  exit 1
fi

# Dereference: produce /tmp/push-bundle-<remote-name> with real files only.
DEREF="/tmp/push-bundle-$REMOTE_NAME"
echo "[deref] $SRC → $DEREF"
rm -rf "$DEREF"
mkdir -p "$DEREF"
for entry in "$SRC"/*; do
  name=$(basename "$entry")
  if [ -L "$entry" ]; then
    real=$(readlink "$entry")
    # readlink may be relative; resolve against SRC
    case "$real" in
      /*) abs="$real" ;;
      *)  abs="$SRC/$real" ;;
    esac
    cp -R "$abs" "$DEREF/$name"
  elif [ -e "$entry" ]; then
    cp -R "$entry" "$DEREF/$name"
  fi
done

echo "[deref] bundle size: $(du -sh "$DEREF" | awk '{print $1}')"

# Push.
echo "[push] device=$DEVICE  remote=$REMOTE_DIR"
for src in "$DEREF"/*; do
  rel=$(basename "$src")
  size=$(du -sh "$src" 2>/dev/null | awk '{print $1}')
  echo "[push]   $rel ($size)..."
  xcrun devicectl device copy to \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" \
    --source "$src" \
    --destination "$REMOTE_DIR/$rel" 2>&1 \
    | grep -aE "Path:|Acquired tunnel|Error|fail" | head -2 || true
done

echo ""
echo "[push] done. Bundle live at $REMOTE_DIR on device $DEVICE."
echo ""
echo "AutoBench (default 4 prompts):"
echo "  xcrun devicectl device process launch --device $DEVICE --console \\"
echo "    --environment-variables '{\"LLM_AUTOBENCH\": \"1\"}' \\"
echo "    com.example.CoreMLLLMChat"
