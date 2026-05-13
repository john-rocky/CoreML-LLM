#!/usr/bin/env bash
# Sweep MTP env vars on iPhone via AutoBench. Captures tok/s per
# (prompt × knob) combination into a TSV.
#
# Usage:
#   bash scripts/iphone_autobench_sweep.sh <sweep_name> [<prompt_csv>]
#
# `<sweep_name>` controls which env grid is exercised. Edit the case
# block below to add more sweeps.
#
# Output: stdout TSV "sweep  knob  prompt  tok/s  acc?"  +
#         /tmp/iphone_sweep_<sweep_name>.tsv

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE=$(xcrun devicectl list devices 2>/dev/null \
  | grep "iPhone 17 Pro" | grep "connected" \
  | grep -oE '[A-F0-9]{8}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{12}' \
  | head -1)
if [ -z "$DEVICE" ]; then
  echo "iPhone 17 Pro not connected" >&2
  exit 1
fi

SWEEP="${1:?usage: <sweep_name> [<prompt_csv>]}"
PROMPTS="${2:-narrative,code}"

case "$SWEEP" in
  k_use)
    KEYS=("MTP_K_USE")
    VALUES=("1" "2" "3")
    ;;
  fly_topk)
    KEYS=("MTP_FLY_TOPK")
    VALUES=("8" "16" "24" "32")
    ;;
  bail_threshold)
    KEYS=("MTP_FALLBACK_THRESHOLD")
    VALUES=("0.20" "0.25" "0.30" "0.35" "0.40")
    ;;
  chunk_pipeline)
    KEYS=("LLM_CHUNK_PIPELINE")
    VALUES=("0" "1")
    ;;
  l5_async)
    KEYS=("MTP_L5_ASYNC_DISABLE")
    VALUES=("0" "1")
    ;;
  *)
    echo "Unknown sweep: $SWEEP (known: k_use, fly_topk, bail_threshold, chunk_pipeline, l5_async)" >&2
    exit 1
    ;;
esac

OUT="/tmp/iphone_sweep_${SWEEP}.tsv"
echo -e "sweep\tknob\tprompt\ttok/s" > "$OUT"

KEY="${KEYS[0]}"
for V in "${VALUES[@]}"; do
  echo "=== $SWEEP: $KEY=$V ==="
  LOG="/tmp/iphone_sweep_${SWEEP}_${V}.log"
  ENV_JSON=$(printf '{"LLM_AUTOBENCH": "1", "LLM_AUTOBENCH_MAX_TOKENS": "256", "LLM_AUTOBENCH_PROMPTS": "%s", "%s": "%s"}' \
                "$PROMPTS" "$KEY" "$V")
  xcrun devicectl device process launch \
    --device "$DEVICE" --console \
    --environment-variables "$ENV_JSON" \
    com.example.CoreMLLLMChat > "$LOG" 2>&1
  # Parse `[AutoBench] <label>: tokens=N wall=Xs tok/s=Y`
  grep -aE "^\[AutoBench\] [a-z]+: tokens=" "$LOG" | while read -r line; do
    label=$(echo "$line" | grep -oE '\] [a-z]+:' | tr -d '] :')
    tps=$(echo "$line" | grep -oE 'tok/s=[0-9.]+' | head -1 | cut -d= -f2)
    printf "%s\t%s=%s\t%s\t%s\n" "$SWEEP" "$KEY" "$V" "$label" "$tps" | tee -a "$OUT"
  done
done

echo ""
echo "=== Summary ==="
column -t -s$'\t' "$OUT"
