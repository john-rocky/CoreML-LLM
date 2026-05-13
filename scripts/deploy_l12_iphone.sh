#!/usr/bin/env bash
# Deploy L12 Subset LM Head to iPhone 17 Pro and prepare for bench.
#
# Usage:
#   bash scripts/deploy_l12_iphone.sh
#
# What it does:
#  1. Find the iPhone 17 Pro via devicectl
#  2. Install the freshly-built Release .app
#  3. Push chunk4_subset.mlmodelc + lm_head_fp16.bin + frequent_tokens.bin
#     to the app data container
#  4. (No env-var setup needed — iOS auto-enables subset when files present
#     via the iOS-conditional default in `MtpSpeculativeEngine.init`)
#
# After running, open the CoreMLLLMChat app and bench:
#   - "What is your favourite hobby and why?" (free-form English)
#   - "Say yes 5 times." (yes-yes regression test)
#   - "江戸時代について教えて。" (Japanese)
# Compare tok/s in UI vs current ship 1.16×.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="/tmp/CoreMLLLMChat-build/Build/Products/Release-iphoneos/CoreMLLLMChat.app"
BUNDLE_ID="com.example.CoreMLLLMChat"
REMOTE_DIR="Documents/Models/gemma4-e2b"

if [ ! -d "$APP" ]; then
  echo "missing Release .app at $APP — building..."
  cd "$ROOT/Examples/CoreMLLLMChat"
  xcodebuild -project CoreMLLLMChat.xcodeproj -scheme CoreMLLLMChat \
    -configuration Release -destination 'generic/platform=iOS' \
    -derivedDataPath /tmp/CoreMLLLMChat-build build > /tmp/build.log 2>&1
  cd "$ROOT"
fi

# Find connected iPhone 17 Pro
DEVICE=$(xcrun devicectl list devices 2>&1 \
  | grep "iPhone 17 Pro" \
  | grep -v unavailable \
  | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}' \
  | head -1)
if [ -z "$DEVICE" ]; then
  echo "iPhone 17 Pro not connected (or paired but not available)" >&2
  xcrun devicectl list devices >&2
  exit 1
fi
echo "[deploy] device=$DEVICE"

# 1. Install app
echo "[deploy] installing $APP ..."
xcrun devicectl device install app --device "$DEVICE" "$APP" 2>&1 \
  | grep -E "Acquired|Installation|Error" | head -5 || true

# 2. Push L12 artifacts
SUBSET="$ROOT/output/gemma4-e2b/chunks_subset/chunk4_subset.mlmodelc"
LMHEAD="$ROOT/output/gemma4-e2b/lm_head_fp16.bin"
FREQ="$ROOT/output/gemma4-e2b/frequent_tokens.bin"
push() {
  local src="$1"
  local rel="$(basename "$src")"
  local size=$(du -sh "$src" 2>/dev/null | awk '{print $1}')
  echo "[deploy]   $rel ($size)..."
  xcrun devicectl device copy to \
    --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" \
    --source "$src" \
    --destination "$REMOTE_DIR/$rel" 2>&1 \
    | grep -aE "Path:|Acquired|Error|fail" | head -2 || true
}
[ -d "$SUBSET" ] && push "$SUBSET"
[ -f "$LMHEAD" ] && push "$LMHEAD"
[ -f "$FREQ" ] && push "$FREQ"

echo ""
echo "[deploy] done. Open CoreMLLLMChat on the device and bench."
echo "[deploy] Settings: M=2048 floor=8 (iOS defaults, freq tokens from bundle)"
echo "[deploy] If something fails, see logs via xcrun devicectl list logs --device $DEVICE"
