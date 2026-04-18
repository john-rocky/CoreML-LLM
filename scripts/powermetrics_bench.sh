#!/usr/bin/env bash
# Tethered iPhone power measurement via macOS powermetrics.
#
# Splits CPU / GPU / ANE power so we can attribute heat sources during
# inference. Pair with the on-device PowerBench (see docs/POWER_BENCH.md)
# or just run during a normal generation in CoreMLLLMChat.
#
# Usage:
#   sudo ./scripts/powermetrics_bench.sh [duration_seconds] [interval_ms]
#
# Defaults: 60 seconds at 1000 ms intervals (one sample per second).
# Output: /tmp/powermetrics_<unix>.log + /tmp/powermetrics_<unix>.csv
#
# The CSV columns are: ts_sec, cpu_w, gpu_w, ane_w, combined_w
# powermetrics emits per-domain power; the script greps and reformats.

set -euo pipefail

DURATION="${1:-60}"
INTERVAL_MS="${2:-1000}"
SAMPLES=$(( DURATION * 1000 / INTERVAL_MS ))
TS=$(date +%s)
LOG="/tmp/powermetrics_${TS}.log"
CSV="/tmp/powermetrics_${TS}.csv"

if [[ $EUID -ne 0 ]]; then
    echo "powermetrics requires sudo. Re-run: sudo $0 $*" >&2
    exit 1
fi

echo "[powermetrics_bench] duration=${DURATION}s interval=${INTERVAL_MS}ms samples=${SAMPLES}"
echo "[powermetrics_bench] log=${LOG}"
echo "[powermetrics_bench] csv=${CSV}"
echo "[powermetrics_bench] Starting in 3s — start the iPhone bench now…"
sleep 3

powermetrics \
    --samplers cpu_power,gpu_power,ane_power \
    -i "${INTERVAL_MS}" \
    -n "${SAMPLES}" \
    > "${LOG}" 2>&1

echo "ts_sec,cpu_w,gpu_w,ane_w,combined_w" > "${CSV}"
awk '
/^\*\*\* Sampled system activity/ { ts = $NF }
/^CPU Power:/                     { cpu = $3 }
/^GPU Power:/                     { gpu = $3 }
/^ANE Power:/                     { ane = $3 }
/^Combined Power/                 {
    comb = $4
    printf "%s,%.3f,%.3f,%.3f,%.3f\n", ts, cpu/1000, gpu/1000, ane/1000, comb/1000
}
' "${LOG}" >> "${CSV}"

ROWS=$(( $(wc -l < "${CSV}") - 1 ))
echo "[powermetrics_bench] Wrote ${ROWS} rows to ${CSV}"

if [[ "${ROWS}" -gt 0 ]]; then
    echo
    echo "Mean power (W):"
    awk -F, 'NR>1 { c+=$2; g+=$3; a+=$4; b+=$5; n++ }
             END  { if (n>0) printf "  CPU=%.2f  GPU=%.2f  ANE=%.2f  Combined=%.2f\n",
                                    c/n, g/n, a/n, b/n }' "${CSV}"
fi

echo
echo "Tip: keep the device awake while tethered:  caffeinate -dimsu &"
