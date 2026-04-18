#!/usr/bin/env bash
# Mac-side A/B test for the CPU-bottleneck investigation.
#
# Runs the smoke CLI three times against the same model:
#   1. baseline                 — no env vars
#   2. double-buffer KV         — LLM_DOUBLE_BUFFER_KV=1
#   3. + utility QoS            — LLM_DOUBLE_BUFFER_KV=1 LLM_DECODE_QOS=utility
#
# For each run, captures the [Profile] / [ANE/CPU] log lines and the
# reported tok/s, then prints a side-by-side comparison.
#
# Usage:
#   ./scripts/test_double_buffer_locally.sh <model-dir> [maxTokens]
#
# Mac note: the ANE on Mac is much smaller than iPhone's, so absolute
# numbers won't transfer. The comparison ratios (copyBack drop, %CPU
# change) DO carry over. iPhone is still the authoritative measurement.

set -euo pipefail

MODEL_DIR="${1:-}"
MAX_TOKENS="${2:-64}"

if [[ -z "${MODEL_DIR}" || ! -d "${MODEL_DIR}" ]]; then
    echo "Usage: $0 <model-dir> [maxTokens]" >&2
    echo "  <model-dir> must contain chunk{1..4}.mlmodelc or .mlpackage" >&2
    exit 1
fi

if [[ ! -d "${MODEL_DIR}/chunk1.mlmodelc" && ! -d "${MODEL_DIR}/chunk1.mlpackage" ]]; then
    echo "Error: ${MODEL_DIR} doesn't contain chunk1.mlmodelc or chunk1.mlpackage" >&2
    exit 1
fi

cd "$(dirname "$0")/.."
echo "[bench] building release…"
swift build -c release 2>&1 | tail -1

PROMPT="Write three short sentences about the ocean."
LOG_DIR=$(mktemp -d -t coreml-llm-bench)
echo "[bench] logs → ${LOG_DIR}"

run_config() {
    local name="$1"
    local log="${LOG_DIR}/${name}.log"
    shift
    echo
    echo "════════════════════════════════════════════════════════"
    echo "[bench] ${name} :  $*"
    echo "════════════════════════════════════════════════════════"
    env "$@" LLM_PROFILE_EVERY_STEP=1 \
        ./.build/release/coreml-llm-smoke "${MODEL_DIR}" "${PROMPT}" "${MAX_TOKENS}" \
        2>&1 | tee "${log}" \
        || echo "[bench] (run failed; see ${log})"
}

run_config "baseline"        BASELINE_PLACEHOLDER=1
run_config "double_buffer"   LLM_DOUBLE_BUFFER_KV=1
run_config "doublebuf_utility" LLM_DOUBLE_BUFFER_KV=1 LLM_DECODE_QOS=utility

echo
echo "════════════════════════════════════════════════════════"
echo "[bench] SUMMARY"
echo "════════════════════════════════════════════════════════"
printf "%-22s | %-9s | %-22s | %-12s\n" "config" "tok/s" "ANE/copyBack/cpu(ms)" "%CPU"
printf "%-22s | %-9s | %-22s | %-12s\n" "----------------------" "---------" "----------------------" "------------"

extract() {
    local log="$1"
    local toks
    toks=$(grep -E "^\[smoke\] tok/s" "${log}" | tail -1 | awk '{print $NF}')
    local lastCPU
    lastCPU=$(grep -E "^\[ANE/CPU\]" "${log}" | tail -1 \
        | sed -E 's/.*ANE_wait=([0-9.]+)ms copyBack=([0-9.]+)ms cpu_active=([0-9.]+)ms \(([0-9]+)% CPU\).*/\1\/\2\/\3 \4%/')
    echo "${toks:-—} | ${lastCPU:-—}"
}

for name in baseline double_buffer doublebuf_utility; do
    line=$(extract "${LOG_DIR}/${name}.log")
    toks=${line%% |*}
    rest=${line##*| }
    nums=${rest%% *}
    pct=${rest##* }
    printf "%-22s | %-9s | %-22s | %-12s\n" "${name}" "${toks}" "${nums}" "${pct}"
done

echo
echo "Raw logs: ${LOG_DIR}"
echo "If 'double_buffer' shows copyBack near 0ms AND tok/s ≥ baseline,"
echo "outputBackings is honored — recommend shipping LLM_DOUBLE_BUFFER_KV=1 to iPhone."
