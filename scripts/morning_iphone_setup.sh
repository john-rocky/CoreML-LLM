#!/usr/bin/env bash
# Morning iPhone setup — run this first thing.
#
# Steps:
# 1. Rebuild iPhone Release app (in case branch changed overnight).
# 2. Install on connected iPhone 17 Pro.
# 3. Confirm bundle present at Documents/Models/gemma4-e2b/.
# 4. Print ready-to-paste AutoBench commands for the lever queue.
#
# Pre-conditions:
# * iPhone 17 Pro unlocked + plugged in
# * iPhone at room temperature (chunk1 load < 0.5s in first bench)
#
# Usage:
#   bash scripts/morning_iphone_setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DEVICE=$(xcrun devicectl list devices 2>/dev/null \
  | grep "iPhone 17 Pro" | grep "connected" \
  | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}' \
  | head -1)
if [ -z "$DEVICE" ]; then
  echo "ERROR: iPhone 17 Pro not connected. Plug in and unlock first." >&2
  exit 1
fi

echo "[1/3] Building iPhone Release app..."
cd "$ROOT/Examples/CoreMLLLMChat"
xcodebuild -project CoreMLLLMChat.xcodeproj \
  -scheme CoreMLLLMChat \
  -configuration Release \
  -destination 'generic/platform=iOS' \
  -derivedDataPath /tmp/CoreMLLLMChat-build \
  build 2>&1 | tail -3
cd "$ROOT"

echo "[2/3] Installing on iPhone..."
xcrun devicectl device install app \
  --device "$DEVICE" \
  /tmp/CoreMLLLMChat-build/Build/Products/Release-iphoneos/CoreMLLLMChat.app \
  2>&1 | grep -aE "App installed|Error" | head -3

echo "[3/3] Verifying bundle on device..."
xcrun devicectl device info files \
  --device "$DEVICE" \
  --domain-type appDataContainer \
  --domain-identifier com.example.CoreMLLLMChat \
  --subdirectory Documents/Models/gemma4-e2b 2>&1 \
  | awk 'NR>5 {print $1}' | grep -E "^chunk[1-4]\.mlmodelc|^mtp_drafter" | sort -u

echo ""
echo "=========================================="
echo "Ready. Recommended bench order (cool iPhone, 10-min gaps):"
echo "=========================================="
echo ""
echo "# Round A — baseline reconfirm (5 min)"
echo "xcrun devicectl device process launch --device $DEVICE --console \\"
echo "  --environment-variables '{\"LLM_AUTOBENCH\": \"1\", \"LLM_AUTOBENCH_PROMPTS\": \"narrative,code\"}' \\"
echo "  com.example.CoreMLLLMChat > /tmp/round_a.log 2>&1"
echo ""
echo "# Round B — S1 MLComputePlan audit (find ANE/CPU fallback ops)"
echo "xcrun devicectl device process launch --device $DEVICE --console \\"
echo "  --environment-variables '{\"COMPUTE_PLAN_AUDIT\": \"1\", \"LLM_AUTOBENCH\": \"1\", \"LLM_AUTOBENCH_PROMPTS\": \"narrative\"}' \\"
echo "  com.example.CoreMLLLMChat > /tmp/round_b.log 2>&1"
echo ""
echo "# Round C — S2 ping-pong outputBackings"
echo "xcrun devicectl device process launch --device $DEVICE --console \\"
echo "  --environment-variables '{\"LLM_AUTOBENCH\": \"1\", \"LLM_AUTOBENCH_PROMPTS\": \"code\", \"MTP_VERIFY_BACKINGS_PING_PONG\": \"1\"}' \\"
echo "  com.example.CoreMLLLMChat > /tmp/round_c.log 2>&1"
echo ""
echo "# Round D — env sweep examples (one at a time, 10-min cool between):"
echo "bash scripts/iphone_autobench_sweep.sh fly_topk code         # K=8/16/24/32"
echo "bash scripts/iphone_autobench_sweep.sh pld_prefetch code     # +/- prefetch"
echo "bash scripts/iphone_autobench_sweep.sh ping_pong code        # S2 explicit sweep"
echo "bash scripts/iphone_autobench_sweep.sh bail_threshold narrative"
echo "bash scripts/iphone_autobench_sweep.sh decode_qos code"
echo ""
echo "# Day 2 — T7 per-channel INT4 (Mac build first, 30-90 min)"
echo "bash scripts/build_per_channel_chunks.sh"
echo "bash scripts/push_bundle_to_iphone.sh output/gemma4-e2b/bundle_3way_perch gemma4-e2b-perch"
echo "# then AutoBench with LLM_AUTOBENCH_MODEL=gemma4-e2b-perch"
echo ""
echo "Read: docs/RESEARCH_FINDINGS_2026_05_13_R2.md for the full lever rationale."
