#!/usr/bin/env bash
# Push Gemma 4 E2B stateful Linear bundle (Plan 3) to iPhone 17 Pro.
# Memory: Mac 34.6 tok/s; iPhone parity (project_stateful_plan3_phase2a).
#
# Source: /Users/majimadaisuke/Downloads/workspace/CoreML-LLM-stage3/
#         build/gemma4_stateful_3chunk_linear/gemma4_e2b_stateful_chunks/
# Target: Documents/Models/gemma4-e2b-stateful-linear/gemma4_e2b_stateful_chunks/
#
# After push, in CoreMLLLMChat picker → Gemma 4 E2B stateful (Linear).
# AutoBench can target via LLM_AUTOBENCH_MODEL=gemma4-e2b-stateful-linear
# (note: stateful path is T=1 only — no MTP, no FLy).

set -euo pipefail
SRC="${SRC:-/Users/majimadaisuke/Downloads/workspace/CoreML-LLM-stage3/build/gemma4_stateful_3chunk_linear/gemma4_e2b_stateful_chunks}"
BUNDLE_ID="com.example.CoreMLLLMChat"
REMOTE_PARENT="Documents/Models/gemma4-e2b-stateful-linear"
REMOTE_DIR="$REMOTE_PARENT/gemma4_e2b_stateful_chunks"

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

echo "[push] device=$DEVICE  src=$SRC"
echo "[push] total size: $(du -sh "$SRC" | awk '{print $1}')"

# Push entire stateful_chunks directory in one go (devicectl recurses).
xcrun devicectl device copy to \
  --device "$DEVICE" \
  --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" \
  --source "$SRC" \
  --destination "$REMOTE_DIR" 2>&1 \
  | grep -aE "Path:|Acquired tunnel|Error|fail" | head -3

echo ""
echo "[push] done. Pick 'Gemma 4 E2B stateful (Linear)' in CoreMLLLMChat."
echo ""
echo "AutoBench command (T=1 only, no MTP):"
echo "  xcrun devicectl device process launch --device $DEVICE --console \\"
echo "    --environment-variables '{\"LLM_AUTOBENCH\": \"1\", \"LLM_AUTOBENCH_MODEL\": \"gemma4-e2b-stateful-linear\", \"LLM_AUTOBENCH_PROMPTS\": \"narrative,code\"}' \\"
echo "    com.example.CoreMLLLMChat > /tmp/L9_stateful_linear.log 2>&1"
