#!/usr/bin/env bash
# Mac-side drift measurement for PRIORITY_ROADMAP item 11c.
#
# Runs MtpMacCheck in two configurations against the SAME target chunks:
#   1. decode_q1 only (MTP modules absent → speculation disabled, pure decode)
#   2. verify_qK path  (MTP modules present → speculative decoding active)
#
# Both configurations share identical weights, prefill, and RoPE; the only
# difference is whether target positions are materialised via decode_q1
# (one position at a time) or verify_qK (K=3 batched). If the two produce
# different token streams, that divergence IS item 11c surfacing.
#
# This gives a reproducible, iPhone-free signal for any candidate 11c fix
# (reconverted chunks, fp32 logit cast, altered accumulation order, etc.):
# apply the fix, regenerate chunks, re-run this script, and compare
# "first flip position" + "flip rate" to the baseline run.
#
# Usage:
#   scripts/measure_verify_drift.sh <deploy-dir-with-mtp_modules> [prompt]
#
# The deploy-dir must have both mtp_module_0.mlmodelc and mtp_module_1.mlmodelc
# alongside the verify-capable chunks 1-4. We temporarily move the MTP modules
# aside to build the baseline config, then restore them.
#
# Outputs:
#   /tmp/verify_drift/base_tokens.txt     (one token per line, decode-only)
#   /tmp/verify_drift/spec_tokens.txt     (one token per line, speculation on)
#   /tmp/verify_drift/diff.txt            (unified diff between the two)
#   Console: first-flip position + flip rate summary.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <deploy-dir> [prompt]" >&2
    exit 1
fi

SPEC_DIR="$(cd "$1" && pwd)"
PROMPT="${2:-Hello}"
OUT_DIR=/tmp/verify_drift
BASE_DIR="$OUT_DIR/baseline_$(date +%s)"

mkdir -p "$OUT_DIR"

# Build a baseline copy without the MTP modules. CoreMLLLM auto-disables the
# speculative path when both mtp_module_*.mlmodelc are missing.
echo "=== Preparing baseline (decode-only) deploy at $BASE_DIR ==="
rm -rf "$BASE_DIR"
cp -R "$SPEC_DIR" "$BASE_DIR"
rm -rf "$BASE_DIR"/mtp_module_0.mlmodelc "$BASE_DIR"/mtp_module_1.mlmodelc

# Ensure the MtpMacCheck binary exists.
if [ ! -x .build/debug/MtpMacCheck ]; then
    echo "=== Building MtpMacCheck ==="
    swift build --product MtpMacCheck
fi

echo "=== Run 1: baseline (decode_q1 only) ==="
.build/debug/MtpMacCheck "$BASE_DIR" "$PROMPT" > "$OUT_DIR/base_raw.log" 2>&1 || true
grep -E '^  chunk:' "$OUT_DIR/base_raw.log" | sed 's/^  chunk: //' > "$OUT_DIR/base_tokens.txt" || true

echo "=== Run 2: speculation (verify_qK + MTP modules) ==="
.build/debug/MtpMacCheck "$SPEC_DIR" "$PROMPT" > "$OUT_DIR/spec_raw.log" 2>&1 || true
grep -E '^  chunk:' "$OUT_DIR/spec_raw.log" | sed 's/^  chunk: //' > "$OUT_DIR/spec_tokens.txt" || true

# Pad shorter stream so diff aligns cleanly.
BASE_N=$(wc -l < "$OUT_DIR/base_tokens.txt" | tr -d ' ')
SPEC_N=$(wc -l < "$OUT_DIR/spec_tokens.txt" | tr -d ' ')
N=$(( BASE_N < SPEC_N ? BASE_N : SPEC_N ))

if [ "$N" -eq 0 ]; then
    echo "ERROR: no tokens captured. Check the raw logs." >&2
    echo "  base_raw.log tail:"; tail -20 "$OUT_DIR/base_raw.log"
    echo "  spec_raw.log tail:"; tail -20 "$OUT_DIR/spec_raw.log"
    exit 2
fi

diff -U0 "$OUT_DIR/base_tokens.txt" "$OUT_DIR/spec_tokens.txt" > "$OUT_DIR/diff.txt" || true

echo ""
echo "=== Summary ==="
echo "  base_tokens: $BASE_N"
echo "  spec_tokens: $SPEC_N"
echo "  common length: $N"

# Find first position where they differ.
FIRST_FLIP=$(awk 'NR==FNR{a[NR]=$0;next} {if(a[FNR]!=$0){print FNR; exit}}' \
    "$OUT_DIR/base_tokens.txt" "$OUT_DIR/spec_tokens.txt" || true)
if [ -z "$FIRST_FLIP" ]; then
    FIRST_FLIP="none (sequences identical up to common length)"
fi
echo "  first flip position: $FIRST_FLIP"

# Flip rate over common prefix.
FLIPS=$(paste "$OUT_DIR/base_tokens.txt" "$OUT_DIR/spec_tokens.txt" \
    | head -n "$N" \
    | awk -F'\t' '{ if($1!=$2) c++ } END{ print c+0 }')
if [ "$N" -gt 0 ]; then
    RATE=$(awk -v a="$FLIPS" -v b="$N" 'BEGIN{ printf "%.1f", 100.0*a/b }')
    echo "  flip rate: $FLIPS / $N = ${RATE}%"
fi

echo ""
echo "Full artefacts under $OUT_DIR:"
ls -la "$OUT_DIR"

# Show baseline tok/s + speculation tok/s for context.
echo ""
echo "=== tok/s ==="
grep -E 'tok/s:' "$OUT_DIR/base_raw.log" | sed 's/^/  base: /' || true
grep -E 'tok/s:' "$OUT_DIR/spec_raw.log" | sed 's/^/  spec: /' || true
grep -E 'mtp acceptance' "$OUT_DIR/spec_raw.log" | sed 's/^/  spec: /' || true
