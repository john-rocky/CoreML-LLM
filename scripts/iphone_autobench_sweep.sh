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
  pld_prefetch)
    KEYS=("MTP_PLD_PREFETCH_ENABLE")
    VALUES=("0" "1")
    ;;
  self_bail)
    KEYS=("MTP_SELF_BAIL_DISABLE")
    VALUES=("0" "1")
    ;;
  fast_pred)
    KEYS=("LLM_FAST_PREDICTION")
    VALUES=("0" "1")
    ;;
  draft_pos_mode)
    KEYS=("MTP_DRAFT_POS_MODE")
    VALUES=("constpm1" "perstep" "constpos")
    ;;
  drafter_device)
    KEYS=("MTP_DRAFTER_DEVICE")
    VALUES=("ane" "cpu" "gpu")
    ;;
  ping_pong)
    # S2 — re-enable verify outputBackings with A/B alternation.
    KEYS=("MTP_VERIFY_BACKINGS_PING_PONG")
    VALUES=("0" "1")
    ;;
  lookahead)
    # L20 — Jacobi/Lookahead engine.
    KEYS=("LLM_LOOKAHEAD_ENABLE")
    VALUES=("0" "1")
    ;;
  prefix_cache)
    # L21 — multi-turn TTFT (note: only relevant for multi-turn).
    KEYS=("LLM_PREFIX_CACHE")
    VALUES=("0" "1")
    ;;
  sampling_temp)
    # L22 — rejection-sampling MTP path. 0 = greedy (default).
    KEYS=("MTP_TEMPERATURE")
    VALUES=("0.0" "0.5" "0.7" "1.0")
    ;;
  decode_qos)
    # L16/L17 — decode loop QoS (high=peak, utility=anti-thermal).
    KEYS=("LLM_DECODE_QOS")
    VALUES=("userinitiated" "high" "utility")
    ;;
  *)
    echo "Unknown sweep: $SWEEP" >&2
    echo "known: k_use, fly_topk, bail_threshold, chunk_pipeline, l5_async," >&2
    echo "       pld_prefetch, self_bail, fast_pred, draft_pos_mode, drafter_device," >&2
    echo "       ping_pong, lookahead, prefix_cache, sampling_temp, decode_qos" >&2
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
