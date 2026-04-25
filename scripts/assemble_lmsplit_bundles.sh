#!/bin/bash
# Assemble three sibling bundles for the chunk3 LM-head split A/B/C.
#
#   build/lmsplit_bundles/
#     baseline/        (chunk1[from chunk1_3way] + chunk2_3way + chunk3_3way lm_splits=1)
#     lmsplit8/        (... + chunk3_3way lm_splits=8)
#     lmsplit16/       (... + chunk3_3way lm_splits=16)
#
# Each bundle is a fully self-contained 3-chunk-decode bundle:
#   * chunk1.mlmodelc            — renamed from our chunk1_3way (functionally
#                                  identical: both are L0-7 with own KV)
#   * chunk2_3way.mlmodelc       — from baseline build
#   * chunk3_3way.mlmodelc       — from variant build
#   * prefill_chunk{1-4}.mlmodelc — from staging (4-chunk prefill format)
#   * embed_tokens_q8.bin, embed_tokens_per_layer_q8.bin, scales — from staging
#   * cos/sin .npy, per_layer_*, hf_model/tokenizer.* — from staging
#   * model_config.json — from staging
#
# Skipped (not needed for text-only tok/s A/B):
#   * 4-chunk decode chunks (chunk{1-4}.mlmodelc) — replaced by 3-chunk
#   * vision.mlmodelc, vision_video.mlmodelc, audio.mlmodelc — text-only
#   * mtp_drafter.mlpackage, cross_vocab — drafter dead
#
# Each bundle ends up ~4 GB. Push with:
#
#   xcrun devicectl device copy to \
#     --device <ID> \
#     --domain-type appDataContainer \
#     --domain-identifier com.example.CoreMLLLMChat \
#     --source build/lmsplit_bundles/<variant> \
#     --destination Documents/Models/gemma4-e2b-<variant>
#
# Set LLM_3CHUNK=1 in the Xcode scheme so the runtime picks 3-way decode.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_BASELINE="$ROOT/output/gemma4-e2b/chunks_3way"
SRC_LM8="$ROOT/output/gemma4-e2b/chunks_3way_lmsplit8"
SRC_LM16="$ROOT/output/gemma4-e2b/chunks_3way_lmsplit16"
STAGING="/Users/majimadaisuke/Downloads/coreml-llm-artifacts/staging-2k-fast-prefill/gemma4-e2b"
OUT="$ROOT/build/lmsplit_bundles"

# Sanity
for d in "$SRC_BASELINE" "$SRC_LM8" "$SRC_LM16" "$STAGING"; do
    if [[ ! -d "$d" ]]; then
        echo "[error] missing $d" >&2
        exit 1
    fi
done
for chunk in chunk1_3way chunk2_3way chunk3_3way; do
    if [[ ! -d "$SRC_BASELINE/${chunk}.mlmodelc" ]]; then
        echo "[error] $SRC_BASELINE/${chunk}.mlmodelc missing — compile baseline first" >&2
        exit 1
    fi
done
for d in "$SRC_LM8" "$SRC_LM16"; do
    if [[ ! -d "$d/chunk3_3way.mlmodelc" ]]; then
        echo "[error] $d/chunk3_3way.mlmodelc missing" >&2
        exit 1
    fi
done

rm -rf "$OUT"
mkdir -p "$OUT/baseline" "$OUT/lmsplit8" "$OUT/lmsplit16"

# Files to copy from staging into every variant. Order: largest first
# (mostly cosmetic — `cp -R` doesn't parallelize).
STAGING_ITEMS=(
    "embed_tokens_per_layer_q8.bin"
    "prefill_chunk4.mlmodelc"
    "prefill_chunk3.mlmodelc"
    "embed_tokens_q8.bin"
    "prefill_chunk1.mlmodelc"
    "prefill_chunk2.mlmodelc"
    "per_layer_projection.bin"
    "hf_model"
    "sin_full.npy"
    "cos_full.npy"
    "sin_sliding.npy"
    "cos_sliding.npy"
    "embed_tokens_scales.bin"
    "embed_tokens_per_layer_scales.bin"
    "per_layer_norm_weight.bin"
    "model_config.json"
)

build_variant() {
    local name="$1"
    local chunk3_src="$2"
    local out="$OUT/$name"
    echo ""
    echo "[assemble] $name"

    # 1) chunk1.mlmodelc (rename from chunk1_3way — same L0-7 graph)
    cp -R "$SRC_BASELINE/chunk1_3way.mlmodelc" "$out/chunk1.mlmodelc"

    # 2) chunk2_3way.mlmodelc (same across variants)
    cp -R "$SRC_BASELINE/chunk2_3way.mlmodelc" "$out/chunk2_3way.mlmodelc"

    # 3) chunk3_3way.mlmodelc (variant-specific)
    cp -R "$chunk3_src" "$out/chunk3_3way.mlmodelc"

    # 4) Supporting files from staging
    for item in "${STAGING_ITEMS[@]}"; do
        if [[ -e "$STAGING/$item" ]]; then
            cp -R "$STAGING/$item" "$out/"
        else
            echo "  [warn] staging missing $item"
        fi
    done

    du -sh "$out" 2>/dev/null
}

build_variant "baseline"  "$SRC_BASELINE/chunk3_3way.mlmodelc"
build_variant "lmsplit8"  "$SRC_LM8/chunk3_3way.mlmodelc"
build_variant "lmsplit16" "$SRC_LM16/chunk3_3way.mlmodelc"

echo ""
echo "=== assembled ==="
du -sh "$OUT"/*/ 2>/dev/null
echo ""
echo "Push each variant:"
echo ""
echo "  DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF"
echo "  for v in baseline lmsplit8 lmsplit16; do"
echo "    case \$v in"
echo "      baseline) dst=gemma4-e2b-lmsplit-baseline ;;"
echo "      lmsplit8)  dst=gemma4-e2b-lmsplit8 ;;"
echo "      lmsplit16) dst=gemma4-e2b-lmsplit16 ;;"
echo "    esac"
echo "    xcrun devicectl device copy to --device \$DEVICE \\"
echo "      --domain-type appDataContainer \\"
echo "      --domain-identifier com.example.CoreMLLLMChat \\"
echo "      --source $OUT/\$v --destination Documents/Models/\$dst"
echo "  done"
echo ""
echo "Set LLM_3CHUNK=1 + LLM_SHOW_EXPERIMENTAL=1 + LLM_PROFILE_EVERY_STEP=1 in scheme."
