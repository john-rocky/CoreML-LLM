#!/usr/bin/env bash
# Mac A/B bench: 4-chunk (default) vs 3-chunk (LLM_3CHUNK=1) decode.
#
# Requires:
#   - output/gemma4-e2b/bundle/ populated by build_gemma4_bundle.py
#   - output/gemma4-e2b/bundle/chunk{2,3}_3way.mlmodelc from install_3way_bundle.py
#
# Runs the existing `coreml-llm-smoke` CLI twice back-to-back with the same
# prompt + max_tokens and greps tok/s out of both runs. Mac CPU/GPU/ANE mix
# is a sanity check, not an iPhone prediction — dispatch overhead on M-series
# ANE is ~3× lower than A-series so the 3-chunk win shrinks. iPhone still
# needs its own A/B.
set -euo pipefail

MODEL_DIR="${1:-output/gemma4-e2b/bundle}"
PROMPT="${2:-Say one short fact about the moon.}"
MAXT="${3:-64}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "bench: $MODEL_DIR not found" >&2
    exit 2
fi

echo "====================================================="
echo " Bench config"
echo "  model_dir : $MODEL_DIR"
echo "  prompt    : $PROMPT"
echo "  max_tokens: $MAXT"
echo "====================================================="

run_one() {
    local label="$1"
    local extra_env="$2"
    echo ""
    echo "----- $label -----"
    # shellcheck disable=SC2086
    env $extra_env LLM_PROFILE_EVERY_STEP=0 \
        swift run -c release coreml-llm-smoke "$MODEL_DIR" "$PROMPT" "$MAXT" 2>&1 \
        | tee "/tmp/bench_${label// /_}.log" \
        | grep -E "\[Load\]|\[Profile\]|\[ANE/CPU\]|tok/s|LLM_3CHUNK" || true
}

run_one "4chunk" "LLM_3CHUNK=0"
run_one "3chunk" "LLM_3CHUNK=1"

echo ""
echo "====================================================="
echo " Comparison"
echo "====================================================="
for label in "4chunk" "3chunk"; do
    log="/tmp/bench_${label}.log"
    tok=$(grep -oE "tok/s = [0-9.]+" "$log" | tail -1 | awk '{print $NF}')
    printf "  %-8s  tok/s = %s\n" "$label" "${tok:-(not found)}"
done
