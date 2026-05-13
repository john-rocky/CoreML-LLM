#!/bin/bash
# Compile /tmp/gemma4_chunks_K3_awq/chunk{1..4}.mlpackage and link them
# into /tmp/gemma4_e2b_awq_test/ for the Mac smoke bench.
set -e
SRC=/tmp/gemma4_chunks_K3_awq
DST=/tmp/gemma4_e2b_awq_test

if [ ! -d "$SRC/chunk4.mlpackage" ]; then
  echo "ERROR: chunk4 not found at $SRC. Build still running?"
  exit 1
fi

mkdir -p $DST
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT
for i in 1 2 3 4; do
  echo "[deploy] compiling chunk${i}.mlpackage -> chunk${i}.mlmodelc"
  rm -rf $DST/chunk${i}.mlmodelc
  xcrun coremlcompiler compile $SRC/chunk${i}.mlpackage $TMPDIR
  mv $TMPDIR/chunk${i}.mlmodelc $DST/chunk${i}.mlmodelc
done
echo "[deploy] done. Test bundle: $DST"
ls -la $DST/chunk*.mlmodelc | head
