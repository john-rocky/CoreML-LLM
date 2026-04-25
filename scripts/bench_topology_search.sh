#!/usr/bin/env bash
# Mac three-way bench: 4-chunk / 3-chunk Topology II / 3-chunk Topology I
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
    env $extra_env LLM_PROFILE_EVERY_STEP=0 \
        swift run -c release coreml-llm-smoke "$MODEL_DIR" "$PROMPT" "$MAXT" 2>&1 \
        | tee "/tmp/bench_${label// /_}.log" \
        | grep -E "\[Load\]|\[Profile\]|\[ANE/CPU\]|tok/s|LLM_3CHUNK|Topology" || true
}

run_one "4chunk"  "LLM_3CHUNK=0"
run_one "topoII"  "LLM_3CHUNK=1"
run_one "topoI"   "LLM_3CHUNK=1 LLM_3CHUNK_TOPO=I"

echo ""
echo "====================================================="
echo " Comparison (last tok/s line from each run)"
echo "====================================================="
for label in "4chunk" "topoII" "topoI"; do
    log="/tmp/bench_${label}.log"
    tok=$(grep -oE "tok/s = [0-9.]+" "$log" | tail -1 | awk '{print $NF}')
    printf "  %-8s  tok/s = %s\n" "$label" "${tok:-(not found)}"
done
