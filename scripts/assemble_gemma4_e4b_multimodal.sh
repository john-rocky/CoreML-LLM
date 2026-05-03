#!/bin/bash
# Assemble the Gemma 4 E4B multimodal CoreML bundle for iPhone sideload
# (or HF upload). Working configuration validated 2026-05-03 on iPhone 17
# Pro: text 15.7 tok/s + image / video / audio all functional.
#
# Layout produced:
#
#   build/gemma4-e4b-multimodal/
#     chunk1.mlmodelc          # legacy 4-chunk (also has prefill_b8 multifunction)
#     chunk2.mlmodelc          # legacy chunk2 — used as prefill_b8 only
#     chunk3.mlmodelc          # legacy chunk3 — used as prefill_b8 only
#     chunk4.mlmodelc          # legacy chunk4 — used as prefill_b8 only
#     chunk2_3way.mlmodelc     # Topology II decode (own L12-23 + shared L24-32 merged)
#     chunk3_3way.mlmodelc     # Topology II decode (shared L33-41 + lm_head)
#     vision.ane.mlmodelc      # E4B SigLIP encoder (output [1, 256, 2560])
#     audio.mlmodelc           # E4B Conformer encoder (output [1, 50, 1024])
#     audio_config.json
#     mel_filterbank.bin
#     output_proj_weight.npy   # Audio projection 1024 → 1536
#     output_proj_bias.npy
#     embed_proj_weight.npy    # Audio projection 1536 → 2560 (E4B-specific shape)
#     embed_tokens_q8.bin
#     embed_tokens_scales.bin
#     embed_tokens_per_layer_q8.bin
#     embed_tokens_per_layer_scales.bin
#     per_layer_projection.bin
#     per_layer_norm_weight.bin
#     cos_sliding.npy / sin_sliding.npy / cos_full.npy / sin_full.npy
#     hf_model/                (tokenizer)
#     model_config.json
#
# Total bundle size: ~7.6 GB.
#
# Engine routing (CoreMLLLM umbrella in Sources/CoreMLLLM/):
#   - decode = Topology II (chunk1 + chunk2_3way + chunk3_3way) — auto-detected
#     when chunk2_3way/chunk3_3way are present.
#   - prefill = legacy chunks 1/2/3/4 prefill_b8 multifunction. The newer
#     prefill_chunk{1..4}.mlmodelc separate-file path is INTENTIONALLY
#     omitted: it produces broken outputs on iPhone ANE 18 with E4B
#     (likely int4 quantization noise). Mac decodes E4B prefill_chunk*
#     fine — iPhone-specific issue.
#   - vision = vision.ane.mlmodelc when LLM_VISION_FORCE_ANE=1, else GPU
#     fallback. Built E4B-specific (output dim 2560 matches LM hidden).
#   - audio = audio.mlmodelc + Swift-side projection (AudioProcessor.swift,
#     embed_proj is non-square 1536 → 2560 for E4B).
#
# Usage:
#   bash scripts/assemble_gemma4_e4b_multimodal.sh
#
# Required input directories (override via env if non-default):
#   THREEWAY=/tmp/gemma4-e4b-3way               (build_gemma4_3way.py --model gemma4-e4b output)
#   VISION_ANE=/tmp/gemma4-e4b-vision-ane       (convert_gemma4_multimodal.py --vision-ane on E4B HF)
#   AUDIO=/tmp/gemma4-e4b-audio                 (convert_audio.py on E4B HF)
#   LEGACY=.../output/gemma4-e4b/bundle         (legacy 4-chunk text-only bundle, e.g. from HF
#                                                mlboydaisuke/gemma-4-E4B-coreml)
#   MEL_FALLBACK=.../conversion/output/audio    (mel_filterbank.bin source if missing from AUDIO)
#
# Push to iPhone (clean sandbox required — see docs/E4B_MULTIMODAL_BUILD.md):
#   xcrun devicectl device copy to --device <ID> \
#       --domain-type appDataContainer \
#       --domain-identifier com.example.CoreMLLLMChat \
#       --source build/gemma4-e4b-multimodal \
#       --destination Documents/Models/gemma4-e4b
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEGACY="${LEGACY:-/Users/$USER/Downloads/CoreML-LLM/output/gemma4-e4b/bundle}"
THREEWAY="${THREEWAY:-/tmp/gemma4-e4b-3way}"
VISION_ANE="${VISION_ANE:-/tmp/gemma4-e4b-vision-ane}"
AUDIO="${AUDIO:-/tmp/gemma4-e4b-audio}"
MEL_FALLBACK="${MEL_FALLBACK:-/Users/$USER/Downloads/CoreML-LLM/conversion/output/audio}"
OUT="${OUT:-$ROOT/build/gemma4-e4b-multimodal}"

# Sanity
for d in "$LEGACY" "$THREEWAY" "$VISION_ANE" "$AUDIO"; do
    if [[ ! -d "$d" ]]; then
        echo "[error] missing input dir: $d" >&2
        exit 1
    fi
done

mkdir -p "$OUT"
rm -rf "$OUT"/*

echo "[$(date)] === decode (chunk1 legacy + chunk{2,3}_3way Topology II) ==="
cp -R "$LEGACY/chunk1.mlmodelc" "$OUT/"
for c in chunk2_3way chunk3_3way; do
    if [[ -d "$THREEWAY/$c.mlmodelc" ]]; then
        cp -R "$THREEWAY/$c.mlmodelc" "$OUT/"
    elif [[ -d "$THREEWAY/$c.mlpackage" ]]; then
        echo "  compile $c"
        xcrun coremlcompiler compile "$THREEWAY/$c.mlpackage" "$OUT/" 2>&1 | tail -1
    else
        echo "[error] $c{,_3way}.{mlpackage,mlmodelc} missing in $THREEWAY" >&2
        exit 1
    fi
done

echo ""
echo "[$(date)] === legacy chunks 2/3/4 (prefill_b8 multifunction) ==="
for c in chunk2 chunk3 chunk4; do
    cp -R "$LEGACY/$c.mlmodelc" "$OUT/"
done

echo ""
echo "[$(date)] === text sidecars ==="
SIDE_TEXT=(
    embed_tokens_q8.bin embed_tokens_scales.bin
    embed_tokens_per_layer_q8.bin embed_tokens_per_layer_scales.bin
    per_layer_projection.bin per_layer_norm_weight.bin
    cos_sliding.npy sin_sliding.npy cos_full.npy sin_full.npy
    hf_model model_config.json
)
for f in "${SIDE_TEXT[@]}"; do
    if [[ -e "$LEGACY/$f" ]]; then
        cp -R "$LEGACY/$f" "$OUT/"
    else
        echo "  [warn] missing $f"
    fi
done

echo ""
echo "[$(date)] === E4B encoders + audio sidecars ==="
# Vision (E4B-specific, output dim 2560 matches LM hidden)
if [[ -d "$VISION_ANE/vision.ane.mlmodelc" ]]; then
    cp -R "$VISION_ANE/vision.ane.mlmodelc" "$OUT/"
elif [[ -d "$VISION_ANE/vision.ane.mlpackage" ]]; then
    xcrun coremlcompiler compile "$VISION_ANE/vision.ane.mlpackage" "$OUT/" 2>&1 | tail -1
else
    echo "[error] vision.ane.{mlpackage,mlmodelc} missing in $VISION_ANE" >&2
    exit 1
fi
# Audio (E4B-specific, output [1, 50, 1024])
if [[ -d "$AUDIO/audio.mlmodelc" ]]; then
    cp -R "$AUDIO/audio.mlmodelc" "$OUT/"
elif [[ -d "$AUDIO/audio.mlpackage" ]]; then
    xcrun coremlcompiler compile "$AUDIO/audio.mlpackage" "$OUT/" 2>&1 | tail -1
else
    echo "[error] audio.{mlpackage,mlmodelc} missing in $AUDIO" >&2
    exit 1
fi
# Audio sidecars
SIDE_AUDIO=(
    audio_config.json
    output_proj_weight.npy output_proj_bias.npy embed_proj_weight.npy
)
for f in "${SIDE_AUDIO[@]}"; do
    if [[ -e "$AUDIO/$f" ]]; then
        cp "$AUDIO/$f" "$OUT/"
    else
        echo "  [warn] missing $f"
    fi
done
# mel_filterbank.bin (often shipped from a sibling audio build dir)
if [[ -e "$AUDIO/mel_filterbank.bin" ]]; then
    cp "$AUDIO/mel_filterbank.bin" "$OUT/"
elif [[ -e "$MEL_FALLBACK/mel_filterbank.bin" ]]; then
    cp "$MEL_FALLBACK/mel_filterbank.bin" "$OUT/"
else
    echo "  [warn] missing mel_filterbank.bin (audio path will fail at runtime)"
fi

echo ""
echo "=== assembled ==="
du -sh "$OUT"
ls "$OUT"
echo ""
echo "Push to iPhone (CLEAN sandbox — delete + reinstall app first; devicectl"
echo "doesn't remove orphan files from previous bundles):"
echo "  xcrun devicectl device copy to --device <ID> \\"
echo "      --domain-type appDataContainer \\"
echo "      --domain-identifier com.example.CoreMLLLMChat \\"
echo "      --source $OUT \\"
echo "      --destination Documents/Models/gemma4-e4b"
echo ""
echo "Scheme env vars: LLM_VISION_FORCE_ANE=1 (route vision.ane via ANE)."
