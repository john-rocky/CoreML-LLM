#!/bin/bash
# Assemble the Gemma 4 E4B stateful bundle for iPhone sideload.
# Stage 2 sibling of assemble_gemma4_stateful_bundle.sh (which builds
# the E2B variant). Layout matches what Gemma4StatefulEngine expects:
#
#   build/gemma4_stateful_e4b/
#     gemma4_e2b_stateful_chunks/        # subdir name shared with E2B
#       chunk_{1..4}.mlmodelc            (from /tmp/gemma4-e4b-stateful)
#       embed_tokens_q8.bin              (E4B sidecars from output/)
#       embed_tokens_scales.bin
#       embed_tokens_per_layer_q8.bin
#       embed_tokens_per_layer_scales.bin
#       per_layer_projection.bin         (parity, not used by Engine)
#       per_layer_norm_weight.bin
#       cos_sliding.npy / sin_sliding.npy
#       cos_full.npy    / sin_full.npy
#       hf_model/                        (tokenizer files)
#       model_config.json                (E4B: hidden=2560, layers=42, HKV=2)
#
# Push:
#   xcrun devicectl device copy to --device <ID> \
#     --domain-type appDataContainer \
#     --domain-identifier com.example.CoreMLLLMChat \
#     --source build/gemma4_stateful_e4b \
#     --destination Documents/Models/gemma4-e4b-stateful
#
# Scheme: LLM_SHOW_EXPERIMENTAL=1 to reveal the picker entry.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_CHUNKS="${SRC_CHUNKS:-/tmp/gemma4-e4b-stateful}"
# E4B sidecars: the existing legacy 4-chunk E4B bundle in the sibling
# CoreML-LLM workspace already ships every sidecar we need (same names
# the E2B staging-2k-fast-prefill dir uses). Override via env if you
# moved the bundle.
SIDECARS="${SIDECARS:-/Users/majimadaisuke/Downloads/workspace/CoreML-LLM/output/gemma4-e4b/bundle}"
OUT_PARENT="${OUT_PARENT:-$ROOT/build/gemma4_stateful_e4b}"
OUT="$OUT_PARENT/gemma4_e2b_stateful_chunks"

for d in "$SRC_CHUNKS" "$SIDECARS"; do
    if [[ ! -d "$d" ]]; then
        echo "[error] missing $d" >&2
        exit 1
    fi
done
for c in chunk_1 chunk_2 chunk_3 chunk_4; do
    if [[ ! -d "$SRC_CHUNKS/${c}.mlpackage" && ! -d "$SRC_CHUNKS/${c}.mlmodelc" ]]; then
        echo "[error] $SRC_CHUNKS/${c}.{mlpackage,mlmodelc} missing — run build_gemma4_e2b_stateful_chunks.py --model gemma4-e4b first" >&2
        exit 1
    fi
done

rm -rf "$OUT_PARENT"
mkdir -p "$OUT"

# 1. Compile chunks .mlpackage → .mlmodelc into the bundle dir
for c in chunk_1 chunk_2 chunk_3 chunk_4; do
    echo "[compile] $c"
    if [[ -d "$SRC_CHUNKS/${c}.mlmodelc" ]]; then
        cp -R "$SRC_CHUNKS/${c}.mlmodelc" "$OUT/${c}.mlmodelc"
    else
        xcrun coremlcompiler compile \
            "$SRC_CHUNKS/${c}.mlpackage" "$OUT/" 2>&1 | tail -2
    fi
done

# 2. Copy sidecars from the E4B legacy bundle
SIDE_ITEMS=(
    "embed_tokens_q8.bin"
    "embed_tokens_scales.bin"
    "embed_tokens_per_layer_q8.bin"
    "embed_tokens_per_layer_scales.bin"
    "per_layer_projection.bin"
    "per_layer_norm_weight.bin"
    "cos_sliding.npy"
    "sin_sliding.npy"
    "cos_full.npy"
    "sin_full.npy"
    "hf_model"
    "model_config.json"
)
for item in "${SIDE_ITEMS[@]}"; do
    if [[ -e "$SIDECARS/$item" ]]; then
        echo "[copy] $item"
        cp -R "$SIDECARS/$item" "$OUT/"
    else
        echo "  [warn] missing $item"
    fi
done

echo ""
echo "=== assembled ==="
du -sh "$OUT_PARENT"
ls -la "$OUT/" | head -25

echo ""
echo "Push to iPhone:"
echo "  DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF"
echo "  xcrun devicectl device copy to --device \$DEVICE \\"
echo "    --domain-type appDataContainer \\"
echo "    --domain-identifier com.example.CoreMLLLMChat \\"
echo "    --source $OUT_PARENT --destination Documents/Models/gemma4-e4b-stateful"
