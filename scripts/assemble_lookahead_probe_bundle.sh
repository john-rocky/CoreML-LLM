#!/usr/bin/env bash
# Assemble a LookAhead K=8 probe bundle by:
#   1. Copying the clean 2K iPhone baseline (no drafter artefacts)
#   2. Replacing chunk{1-4}.mlmodelc with the K=8 variants from chunks-k8/
#
# Existing bundles at BASELINE_SRC and on-device Documents/Models/gemma4-e2b/
# are NEVER touched by this script. The output goes to a NEW directory.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASELINE_SRC="/Users/majimadaisuke/Downloads/coreml-llm-artifacts/backup-iphone-2k"
K8_CHUNKS="${REPO_ROOT}/output/gemma4-e2b/chunks-k8"
DEST="/Users/majimadaisuke/Downloads/device_deploy_lookahead_probe"

if [[ ! -d "$BASELINE_SRC" ]]; then
  echo "ERROR: baseline missing: $BASELINE_SRC" >&2
  exit 1
fi

for c in chunk1 chunk2 chunk3 chunk4; do
  if [[ ! -d "$K8_CHUNKS/$c.mlmodelc" ]]; then
    echo "ERROR: K=8 chunk missing: $K8_CHUNKS/$c.mlmodelc" >&2
    echo "Run: cd conversion && PYENV_VERSION=lama-cml python build_verify_chunks.py" \
         "--K 8 --model gemma4-e2b --ctx 2048" \
         "--output ../output/gemma4-e2b/chunks-k8 --keep-tmp" >&2
    echo "Then compile each mlpackage to mlmodelc (see docs/LOOKAHEAD_PROBE_RESULTS.md)." >&2
    exit 1
  fi
done

if [[ -d "$DEST" ]]; then
  echo "Refusing to overwrite $DEST — remove it manually if you want to rebuild."
  exit 1
fi

echo "[bundle] copying baseline: $BASELINE_SRC → $DEST"
cp -R "$BASELINE_SRC" "$DEST"

for c in chunk1 chunk2 chunk3 chunk4; do
  echo "[bundle] swapping $c.mlmodelc → K=8"
  rm -rf "$DEST/$c.mlmodelc"
  cp -R "$K8_CHUNKS/$c.mlmodelc" "$DEST/$c.mlmodelc"
done

# Marker that LLMRunner.loadModel looks for to auto-enable
# SPECULATIVE_PROFILE at load time (so verify chunks actually load).
touch "$DEST/probe.marker"
echo "[bundle] wrote probe.marker (LLMRunner auto-enables verify loading)"

# Sanity-check token_ids shape on chunk4 to confirm K=8 went in. Requires
# plutil (ships with macOS). Look inside chunk4.mlmodelc/metadata.json.
META="$DEST/chunk4.mlmodelc/metadata.json"
if [[ -f "$META" ]]; then
  echo "[bundle] chunk4 metadata excerpt:"
  grep -o '"token_ids"[^}]*' "$META" | head -3 || true
fi

echo "[bundle] done. Probe bundle at: $DEST"
echo "[bundle] deploy with:"
cat <<EOF
  DEVICE=\$(xcrun devicectl list devices | awk '/connected/{print \$3}' | head -1)
  xcrun devicectl device copy from --device "\$DEVICE" \\
      --domain-type appDataContainer \\
      --domain-identifier com.example.CoreMLLLMChat \\
      --source Documents/Models/gemma4-e2b \\
      --destination ~/Downloads/coreml-llm-artifacts/backup-pre-k8-probe-\$(date +%Y%m%d-%H%M)
  xcrun devicectl device copy to --device "\$DEVICE" \\
      --domain-type appDataContainer \\
      --domain-identifier com.example.CoreMLLLMChat \\
      --source $DEST \\
      --destination Documents/Models/gemma4-e2b \\
      --remove-existing-content true
EOF
