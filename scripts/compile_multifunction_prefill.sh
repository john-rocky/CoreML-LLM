#!/usr/bin/env bash
# Compile the 4 multifunction prefill mlpackages to mlmodelc for device push.
#
# Input : ./output/gemma4-e2b/prefill_multifunction/prefill_chunk{1..4}.mlpackage
# Output: ./output/gemma4-e2b/prefill_multifunction_compiled/prefill_chunk{1..4}.mlmodelc
#
# coremltools' MLModel.get_compiled_model_path() does the compile —
# requires macOS and a matching Xcode toolchain.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IN="${1:-$ROOT/output/gemma4-e2b/prefill_multifunction}"
OUT="${2:-$ROOT/output/gemma4-e2b/prefill_multifunction_compiled}"
PY="${PY:-python3.12}"

mkdir -p "$OUT"

for i in 1 2 3 4; do
    SRC="$IN/prefill_chunk${i}.mlpackage"
    DST="$OUT/prefill_chunk${i}.mlmodelc"
    if [ ! -d "$SRC" ]; then
        echo "missing $SRC — skipping"
        continue
    fi
    echo "compiling prefill_chunk${i}..."
    "$PY" - "$SRC" "$DST" <<'PY'
import sys, shutil, os
import coremltools as ct
src, dst = sys.argv[1], sys.argv[2]
m = ct.models.MLModel(src, compute_units=ct.ComputeUnit.CPU_AND_NE)
compiled = m.get_compiled_model_path()
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(compiled, dst)
print(f"  → {dst}")
PY
done

echo "done. outputs in $OUT"
ls -la "$OUT"
