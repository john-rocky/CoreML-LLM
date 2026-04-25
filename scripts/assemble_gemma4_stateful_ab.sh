#!/bin/bash
# Assemble two sibling Gemma 4 E2B stateful bundles (A: Conv2d, B: Linear)
# for iPhone A/B comparison of cml9 PR #2577 native linear-op support.
#
# Layout per bundle (matches Gemma4StatefulEngine.load):
#   build/gemma4_stateful_ab/<variant>/gemma4_e2b_stateful_chunks/
#     chunk_{1..4}.mlmodelc            (variant-specific)
#     embed_tokens_q8.bin / scales     (shared from staging)
#     embed_tokens_per_layer_q8.bin / scales
#     per_layer_projection.bin / per_layer_norm_weight.bin
#     cos_sliding.npy / sin_sliding.npy / cos_full.npy / sin_full.npy
#     hf_model/ (tokenizer)
#     model_config.json
#
# Sources (must already exist):
#   /tmp/g4_chunk1_ab/conv/chunk_{1..4}.mlmodelc       (Conv2d-1×1 wrapper)
#   /tmp/g4_chunk1_ab/linear/chunk_{1..4}.mlmodelc     (nn.Linear native)
#   /Users/majimadaisuke/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b/
#     (sidecars)
#
# Push to iPhone:
#   DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF
#   xcrun devicectl device copy to --device $DEVICE \
#     --domain-type appDataContainer \
#     --domain-identifier com.example.CoreMLLLMChat \
#     --source build/gemma4_stateful_ab/conv \
#     --destination Documents/Models/gemma4-e2b-stateful
#   # repeat with --source build/gemma4_stateful_ab/linear
#   #          --destination Documents/Models/gemma4-e2b-stateful-linear
#
# In Xcode scheme: LLM_SHOW_EXPERIMENTAL=1 reveals both picker entries
# (gemma4e2bStateful + gemma4e2bStatefulLinear in ModelDownloader).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_BASE="/tmp/g4_chunk1_ab"
STAGING="/Users/majimadaisuke/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b"
OUT_BASE="$ROOT/build/gemma4_stateful_ab"

# Sidecar files copied into both A and B bundles unchanged.
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

# Sanity
[[ -d "$STAGING" ]] || { echo "[error] staging missing: $STAGING" >&2; exit 1; }
for variant in conv linear; do
    for c in chunk_1 chunk_2 chunk_3 chunk_4; do
        if [[ ! -d "$SRC_BASE/$variant/${c}.mlmodelc" ]]; then
            echo "[error] $SRC_BASE/$variant/${c}.mlmodelc missing — compile mlpackage first" >&2
            exit 1
        fi
    done
done

rm -rf "$OUT_BASE"

assemble() {
    local variant="$1"
    local out_root="$OUT_BASE/$variant"
    local out="$out_root/gemma4_e2b_stateful_chunks"
    mkdir -p "$out"
    echo ""
    echo "=== assembling $variant → $out ==="
    for c in chunk_1 chunk_2 chunk_3 chunk_4; do
        echo "  [chunk] $c"
        cp -R "$SRC_BASE/$variant/${c}.mlmodelc" "$out/"
    done
    for item in "${SIDE_ITEMS[@]}"; do
        if [[ -e "$STAGING/$item" ]]; then
            cp -R "$STAGING/$item" "$out/"
        else
            echo "  [warn] staging missing $item"
        fi
    done
    du -sh "$out_root"
}

assemble conv
assemble linear

echo ""
echo "=== summary ==="
du -sh "$OUT_BASE"/*/
echo ""
echo "Push commands:"
echo ""
echo "  DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF"
echo "  for variant in conv linear; do"
echo "    case \$variant in"
echo "      conv)   dst=gemma4-e2b-stateful ;;"
echo "      linear) dst=gemma4-e2b-stateful-linear ;;"
echo "    esac"
echo "    xcrun devicectl device copy to --device \$DEVICE \\"
echo "      --domain-type appDataContainer \\"
echo "      --domain-identifier com.example.CoreMLLLMChat \\"
echo "      --source $OUT_BASE/\$variant \\"
echo "      --destination Documents/Models/\$dst"
echo "  done"
echo ""
echo "Xcode env: LLM_SHOW_EXPERIMENTAL=1 + LLM_PROFILE_EVERY_STEP=1"
echo ""
echo "ModelDownloader.swift entries needed:"
echo "  - gemma4e2bStateful       (Conv2d, folderName: gemma4-e2b-stateful)"
echo "  - gemma4e2bStatefulLinear (Linear, folderName: gemma4-e2b-stateful-linear)"
