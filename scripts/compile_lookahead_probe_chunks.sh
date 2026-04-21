#!/usr/bin/env bash
# Compile the K=8 verify mlpackages produced by build_verify_chunks.py into
# .mlmodelc form that iOS/macOS can load directly. Skips any chunk already
# compiled so repeated runs are cheap.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${REPO_ROOT}/output/gemma4-e2b/chunks-k8"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: missing $SRC — run build_verify_chunks.py with --K 8 first." >&2
  exit 1
fi

for c in chunk1 chunk2 chunk3 chunk4; do
  in="${SRC}/${c}.mlpackage"
  out="${SRC}/${c}.mlmodelc"
  if [[ ! -d "$in" ]]; then
    echo "ERROR: $in not found" >&2
    exit 1
  fi
  if [[ -d "$out" ]]; then
    echo "[compile] $c.mlmodelc already exists, skipping"
    continue
  fi
  echo "[compile] $c.mlpackage → $c.mlmodelc"
  xcrun coremlcompiler compile "$in" "$SRC"
done

echo "[compile] done. K=8 chunks ready at: $SRC"
