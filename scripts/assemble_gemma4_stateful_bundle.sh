#!/bin/bash
# Assemble the Gemma 4 E2B stateful bundle for iPhone sideload.
#
# Layout produced (matches Gemma4StatefulGenerator.resolveURLs):
#   build/gemma4_stateful/
#     gemma4_e2b_stateful_chunks/
#       chunk_1.mlmodelc   (compiled from /tmp/gemma4-e2b-stateful/chunk_1.mlpackage)
#       chunk_2.mlmodelc
#       chunk_3.mlmodelc
#       chunk_4.mlmodelc
#       embed_tokens_q8.bin            (from staging-2k-fast-prefill/gemma4-e2b)
#       embed_tokens_scales.bin
#       embed_tokens_per_layer_q8.bin
#       embed_tokens_per_layer_scales.bin
#       per_layer_projection.bin       (not used by Generator, present for parity)
#       per_layer_norm_weight.bin
#       cos_sliding.npy / sin_sliding.npy / cos_full.npy / sin_full.npy
#       hf_model/                      (tokenizer files)
#       model_config.json
#
# Total ~4 GB.
#
# Push with:
#   xcrun devicectl device copy to --device <ID> \
#     --domain-type appDataContainer \
#     --domain-identifier com.example.CoreMLLLMChat \
#     --source build/gemma4_stateful \
#     --destination Documents/Models/gemma4-e2b-stateful
#
# In Xcode scheme: LLM_SHOW_EXPERIMENTAL=1 reveals the picker entry.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_CHUNKS="${SRC_CHUNKS:-/tmp/gemma4-e2b-stateful}"
STAGING="${STAGING:-/Users/majimadaisuke/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b}"
OUT_PARENT="${OUT_PARENT:-$ROOT/build/gemma4_stateful}"
OUT="$OUT_PARENT/gemma4_e2b_stateful_chunks"

# Sanity
for d in "$SRC_CHUNKS" "$STAGING"; do
    if [[ ! -d "$d" ]]; then
        echo "[error] missing $d" >&2
        exit 1
    fi
done
# Auto-detect chunk count: 3-chunk merged vs 4-chunk default. Set
# CHUNKS env to override, e.g. `CHUNKS="chunk_1 chunk_2 chunk_3"`.
if [[ -z "${CHUNKS:-}" ]]; then
    if [[ -d "$SRC_CHUNKS/chunk_4.mlpackage" || -d "$SRC_CHUNKS/chunk_4.mlmodelc" ]]; then
        CHUNKS="chunk_1 chunk_2 chunk_3 chunk_4"
    else
        CHUNKS="chunk_1 chunk_2 chunk_3"
    fi
fi
echo "[info] chunks: $CHUNKS"
for c in $CHUNKS; do
    if [[ ! -d "$SRC_CHUNKS/${c}.mlpackage" && ! -d "$SRC_CHUNKS/${c}.mlmodelc" ]]; then
        echo "[error] $SRC_CHUNKS/${c}.{mlpackage,mlmodelc} missing" >&2
        exit 1
    fi
done

rm -rf "$OUT_PARENT"
mkdir -p "$OUT"

# 1. Compile chunks .mlpackage → .mlmodelc into the bundle dir
for c in $CHUNKS; do
    echo "[compile] $c"
    if [[ -d "$SRC_CHUNKS/${c}.mlmodelc" ]]; then
        cp -R "$SRC_CHUNKS/${c}.mlmodelc" "$OUT/${c}.mlmodelc"
    else
        xcrun coremlcompiler compile \
            "$SRC_CHUNKS/${c}.mlpackage" "$OUT/" 2>&1 | tail -2
    fi
done

# 2. Copy sidecars from staging
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
    if [[ -e "$STAGING/$item" ]]; then
        echo "[copy] $item"
        cp -R "$STAGING/$item" "$OUT/"
    else
        echo "  [warn] staging missing $item"
    fi
done

# 3. Stage 6: optional multimodal encoders (vision / video / audio).
#    MM_SRC defaults to a snapshot of `mlboydaisuke/gemma-4-E2B-coreml`
#    on disk (matches what `convert_gemma4_multimodal.py` produces). Set
#    MM_SRC=skip to opt out (text-only bundle, e.g. for Stage 3 bench).
MM_SRC="${MM_SRC:-/tmp/gemma4_mm_encoders}"
MM_ITEMS=(
    "vision.mlmodelc"
    "vision_video.mlmodelc"
    "audio.mlmodelc"
    "mel_filterbank.bin"
    "audio_config.json"
    "output_proj_weight.npy"
    "output_proj_bias.npy"
    "embed_proj_weight.npy"
)
if [[ "$MM_SRC" == "skip" ]]; then
    echo "[skip] multimodal encoders (MM_SRC=skip)"
elif [[ -d "$MM_SRC" ]]; then
    for item in "${MM_ITEMS[@]}"; do
        if [[ -e "$MM_SRC/$item" ]]; then
            echo "[mm-copy] $item"
            cp -R "$MM_SRC/$item" "$OUT/"
        else
            echo "  [warn] mm source missing $item"
        fi
    done
else
    echo "  [warn] MM_SRC=$MM_SRC not present — bundle stays text-only"
fi

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
echo "    --source $OUT_PARENT --destination Documents/Models/gemma4-e2b-stateful"
echo ""
echo "Xcode TODO before build:"
echo "  1. Add Examples/CoreMLLLMChat/CoreMLLLMChat/Gemma4StatefulGenerator.swift"
echo "     to the CoreMLLLMChat target (drag into project navigator → Add to Target)."
echo "  2. Set scheme env: LLM_SHOW_EXPERIMENTAL=1"
echo "  3. Wire ModelPickerView/LLMRunner to dispatch to Gemma4StatefulGenerator"
echo "     when picker selects 'Gemma 4 E2B (stateful, MLState)' (TBD)."
