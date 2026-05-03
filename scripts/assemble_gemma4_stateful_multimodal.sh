#!/bin/bash
# Assemble the Gemma 4 stateful + multimodal bundle for iPhone sideload.
# Stage 8: 3-chunk merged Linear decode + T=288 single-function prefill +
# vision / video / audio encoders. Drives Gemma4StatefulMultimodalEngine.
#
# Layout produced (matches LLMRunner detection — chunks + prefill_T288/
# subdir under gemma4_e2b_stateful_chunks/):
#
#   build/gemma4_stateful_multimodal_e{2,4}b/
#     gemma4_e2b_stateful_chunks/        # subdir name shared with E2B/E4B
#       chunk_{1..3}.mlmodelc            (3-chunk merged decode)
#       prefill_T288/
#         chunk_1_prefill_T288.mlmodelc
#         chunk_2_3way_prefill_T288.mlmodelc
#         chunk_3_prefill_T288.mlmodelc
#       embed_tokens_q8.bin              (sidecars from legacy bundle)
#       embed_tokens_scales.bin
#       embed_tokens_per_layer_q8.bin
#       embed_tokens_per_layer_scales.bin
#       per_layer_projection.bin
#       per_layer_norm_weight.bin
#       cos_sliding.npy / sin_sliding.npy / cos_full.npy / sin_full.npy
#       hf_model/                        (tokenizer files)
#       model_config.json
#       vision.mlmodelc                  (multimodal encoders, shared)
#       vision_video.mlmodelc
#       audio.mlmodelc
#       mel_filterbank.bin               (audio sidecars)
#       audio_config.json
#       output_proj_weight.npy
#       output_proj_bias.npy
#       embed_proj_weight.npy
#
# Usage:
#   MODEL=gemma4-e2b bash scripts/assemble_gemma4_stateful_multimodal.sh
#   MODEL=gemma4-e4b bash scripts/assemble_gemma4_stateful_multimodal.sh
#
# Inputs (overridable via env):
#   SRC_CHUNKS         /tmp/$MODEL-stateful-3chunk
#   SRC_PREFILL_T288   /tmp/$MODEL-singlefunc-prefill-T288
#   SIDECARS           legacy text-only bundle (embed/RoPE/tokenizer)
#   ENCODERS           legacy multimodal bundle (vision/audio mlmodelc)
#                       — vision/audio shared between E2B and E4B
#
# Push:
#   DEVICE=<id-from-devicectl-list>
#   xcrun devicectl device copy to --device $DEVICE \
#     --domain-type appDataContainer \
#     --domain-identifier com.example.CoreMLLLMChat \
#     --source build/gemma4_stateful_multimodal_e4b \
#     --destination Documents/Models/gemma4-e4b-stateful-multimodal
#
# Scheme: LLM_SHOW_EXPERIMENTAL=1 to reveal the picker entry.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-gemma4-e4b}"
case "$MODEL" in
    gemma4-e2b) SHORT=e2b ;;
    gemma4-e4b) SHORT=e4b ;;
    *) echo "[error] MODEL must be gemma4-e2b or gemma4-e4b" >&2; exit 1 ;;
esac

SRC_CHUNKS="${SRC_CHUNKS:-/tmp/$MODEL-stateful-3chunk}"
SRC_PREFILL_T288="${SRC_PREFILL_T288:-/tmp/$MODEL-singlefunc-prefill-T288}"
# Text-side sidecars (embed_tokens, RoPE, tokenizer, model_config). E2B
# defaults to the iphone_8k staging dir; E4B defaults to its own legacy
# bundle (text-only — vision/audio come from ENCODERS below).
if [[ "$MODEL" == "gemma4-e2b" ]]; then
    SIDECARS="${SIDECARS:-/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/output/iphone_8k}"
else
    SIDECARS="${SIDECARS:-/Users/majimadaisuke/Downloads/CoreML-LLM/output/$MODEL/bundle}"
fi
# Multimodal encoders + audio sidecars. Shared between E2B and E4B (same
# SigLIP + Conformer regardless of LM size).
ENCODERS="${ENCODERS:-/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/output/iphone_8k}"
# mel_filterbank.bin lives in a separate dir in some build trees; the
# script falls back to this path if it's not under ENCODERS.
MEL_FALLBACK="${MEL_FALLBACK:-/Users/majimadaisuke/Downloads/CoreML-LLM/conversion/output/audio}"

OUT_PARENT="${OUT_PARENT:-$ROOT/build/gemma4_stateful_multimodal_$SHORT}"
OUT="$OUT_PARENT/gemma4_e2b_stateful_chunks"
PREFILL_OUT="$OUT/prefill_T288"

# Set FOUR_CHUNK=1 to assemble the 4-chunk decode + 4-chunk prefill_T288
# variant (E4B fallback when 3-chunk merged trips iPhone ANE 18). Requires
# `--four-chunk` builds: SRC_CHUNKS holds chunk_{1..4}.mlpackage and
# SRC_PREFILL_T288 holds chunk_{1..4}_prefill_T288.mlpackage.
FOUR_CHUNK="${FOUR_CHUNK:-0}"
if [[ "$FOUR_CHUNK" == "1" ]]; then
    DECODE_CHUNKS=(chunk_1 chunk_2 chunk_3 chunk_4)
    PREFILL_CHUNKS=(chunk_1_prefill_T288 chunk_2_prefill_T288
                    chunk_3_prefill_T288 chunk_4_prefill_T288)
else
    DECODE_CHUNKS=(chunk_1 chunk_2 chunk_3)
    PREFILL_CHUNKS=(chunk_1_prefill_T288 chunk_2_3way_prefill_T288
                    chunk_3_prefill_T288)
fi

# ---- Sanity: required inputs ----
for d in "$SRC_CHUNKS" "$SRC_PREFILL_T288" "$SIDECARS" "$ENCODERS"; do
    if [[ ! -d "$d" ]]; then
        echo "[error] missing input dir: $d" >&2
        exit 1
    fi
done
for c in "${DECODE_CHUNKS[@]}"; do
    if [[ ! -d "$SRC_CHUNKS/${c}.mlpackage" && ! -d "$SRC_CHUNKS/${c}.mlmodelc" ]]; then
        echo "[error] $SRC_CHUNKS/${c}.{mlpackage,mlmodelc} missing — run build_gemma4_e2b_stateful_{,3}chunks.py first" >&2
        exit 1
    fi
done
for c in "${PREFILL_CHUNKS[@]}"; do
    if [[ ! -d "$SRC_PREFILL_T288/${c}.mlpackage" && ! -d "$SRC_PREFILL_T288/${c}.mlmodelc" ]]; then
        echo "[error] $SRC_PREFILL_T288/${c}.{mlpackage,mlmodelc} missing — run build_gemma4_stateful_singlefunc_prefill.py first" >&2
        exit 1
    fi
done

rm -rf "$OUT_PARENT"
mkdir -p "$OUT" "$PREFILL_OUT"

# ---- 1. Compile + place decode mlpackages ----
for c in "${DECODE_CHUNKS[@]}"; do
    echo "[compile decode] $c"
    if [[ -d "$SRC_CHUNKS/${c}.mlmodelc" ]]; then
        cp -R "$SRC_CHUNKS/${c}.mlmodelc" "$OUT/${c}.mlmodelc"
    else
        xcrun coremlcompiler compile \
            "$SRC_CHUNKS/${c}.mlpackage" "$OUT/" 2>&1 | tail -2
    fi
done

# ---- 2. Compile + place T=288 single-function prefill mlpackages ----
for c in "${PREFILL_CHUNKS[@]}"; do
    echo "[compile prefill_T288] $c"
    if [[ -d "$SRC_PREFILL_T288/${c}.mlmodelc" ]]; then
        cp -R "$SRC_PREFILL_T288/${c}.mlmodelc" "$PREFILL_OUT/${c}.mlmodelc"
    else
        xcrun coremlcompiler compile \
            "$SRC_PREFILL_T288/${c}.mlpackage" "$PREFILL_OUT/" 2>&1 | tail -2
    fi
done

# ---- 3. Copy text-side sidecars ----
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
        echo "[copy text sidecar] $item"
        cp -R "$SIDECARS/$item" "$OUT/"
    else
        echo "  [warn] missing text sidecar $item"
    fi
done

# ---- 4. Copy multimodal encoders + audio sidecars ----
ENC_ITEMS=(
    "vision.mlmodelc"
    "vision_video.mlmodelc"
    "audio.mlmodelc"
    "mel_filterbank.bin"
    "audio_config.json"
    "output_proj_weight.npy"
    "output_proj_bias.npy"
    "embed_proj_weight.npy"
)
for item in "${ENC_ITEMS[@]}"; do
    if [[ -e "$ENCODERS/$item" ]]; then
        echo "[copy encoder] $item"
        cp -R "$ENCODERS/$item" "$OUT/"
    elif [[ "$item" == "mel_filterbank.bin" && -e "$MEL_FALLBACK/$item" ]]; then
        echo "[copy encoder fallback] $item (from $MEL_FALLBACK)"
        cp -R "$MEL_FALLBACK/$item" "$OUT/"
    else
        echo "  [warn] missing encoder $item (engine treats as optional)"
    fi
done

echo ""
echo "=== assembled ==="
du -sh "$OUT_PARENT"
echo ""
echo "Top-level:"
ls -la "$OUT/" | head -30
echo ""
echo "prefill_T288/:"
ls -la "$PREFILL_OUT/"

echo ""
echo "Push to iPhone:"
echo "  DEVICE=\$(xcrun devicectl list devices --quiet | awk 'NR==2{print \$3}')"
echo "  xcrun devicectl device copy to --device \$DEVICE \\"
echo "    --domain-type appDataContainer \\"
echo "    --domain-identifier com.example.CoreMLLLMChat \\"
echo "    --source $OUT_PARENT \\"
echo "    --destination Documents/Models/$MODEL-stateful-multimodal"
echo ""
echo "Scheme: LLM_SHOW_EXPERIMENTAL=1 to reveal the picker entry."
