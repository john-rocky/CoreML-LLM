#!/usr/bin/env bash
# Bench 3-chunk + MTP after the build → compile → assemble pipeline.
#
# Prereqs:
#   1.  conversion/build_verify_chunks_3way.py finished cleanly →
#       output/gemma4-e2b/chunks_3way_fp16kv_mf/{chunk2_3way,chunk3_3way}.mlpackage
#   2.  scripts/assemble_3way_mf_bundle.sh ran →
#       output/gemma4-e2b/bundle_3way_mf/  (compiled + symlinked)
#   3.  swift build (debug) up to date
#
# A/B against 4-chunk + MTP baseline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B4="$ROOT/output/gemma4-e2b/bundle_diff_logits"
B3="$ROOT/output/gemma4-e2b/bundle_3way_mf"

PROMPTS=(
  "Write a detailed essay about the ocean and marine life. Include facts about coral reefs, deep sea ecosystems, and the impact of climate change."
  "Write a Python class implementing a binary search tree with insert, delete, and find methods."
  "List 30 facts about ancient Roman emperors with their reign dates."
  "Explain how a transformer neural network processes input tokens through self-attention, layer normalization, and feed-forward networks."
)
LABELS=("narrative" "code" "list" "technical")

bench_one() {
  local bundle="$1"
  local label="$2"
  local prompt="$3"
  local out
  out=$(SPECULATIVE_PROFILE=1 MTP_FORCE_SPECULATE=1 MTP_MODE=mtp \
    "$ROOT/.build/debug/coreml-llm-smoke" "$bundle" "$prompt" 256 2>&1 \
    | grep -aE "^\[smoke\] tok/s|^\[smoke\] mtp accept")
  echo "$out" | awk -v lbl="$label" 'BEGIN{tok="?";acc="?"} /tok\/s/{tok=$NF} /accept/{acc=$NF} END{printf "  %-12s tok/s=%-7s accept=%-5s\n", lbl, tok, acc}'
}

for i in "${!PROMPTS[@]}"; do
  echo "=== ${LABELS[i]} ==="
  echo "  4-chunk + MTP:"
  bench_one "$B4" "4-chunk" "${PROMPTS[i]}"
  echo "  3-chunk + MTP:"
  bench_one "$B3" "3-chunk" "${PROMPTS[i]}"
done
