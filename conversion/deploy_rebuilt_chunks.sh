#!/bin/zsh
# Compile + deploy the freshly rebuilt chunks into output/gemma4-e2b/bundle/
# Run after build_verify_chunks.py completes.
set -e
SRC=${1:-/tmp/gemma4_chunks_fp32norm}
DST=output/gemma4-e2b/bundle
echo "Source: $SRC"
echo "Destination: $DST"
ls -d "$SRC"/chunk*.mlpackage 2>&1
for i in 1 2 3 4; do
    PKG="$SRC/chunk${i}.mlpackage"
    if [[ ! -d "$PKG" ]]; then
        echo "ERROR: $PKG missing"
        exit 1
    fi
    echo "[$i] removing old chunk${i}.mlmodelc"
    rm -rf "$DST/chunk${i}.mlmodelc"
    echo "[$i] compiling $PKG"
    xcrun coremlcompiler compile "$PKG" "$DST/" 2>&1 | tail -1
done
ls -d "$DST"/chunk*.mlmodelc
