#!/bin/bash
# Wipe a sideloaded model directory from a connected iPhone.
#
# `xcrun devicectl device copy to` lands files as UID 0 / 0755 inside
# the app container, which iOS then refuses to let the app remove
# from its own delete UI ("you don't have permission to access it").
#
# This script is the escape hatch: it copies an EMPTY directory to the
# sideloaded path with --remove-existing-content true, which the
# developer-disk-image transport will accept (it runs as the same
# privileged user that wrote the original tree).
#
# Usage:
#   scripts/uninstall_sideloaded_model.sh <model-folder>
#
# Examples:
#   scripts/uninstall_sideloaded_model.sh lfm2.5-350m
#   scripts/uninstall_sideloaded_model.sh gemma4-e2b
#
# Set BUNDLE_ID if you renamed the app from com.example.CoreMLLLMChat.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <model-folder>" >&2
    exit 1
fi

MODEL="$1"
BUNDLE_ID="${BUNDLE_ID:-com.example.CoreMLLLMChat}"

DEVICE=$(xcrun devicectl list devices 2>/dev/null \
    | awk '/connected/{print $3}' | head -1)
if [[ -z "$DEVICE" ]]; then
    echo "no connected device found via xcrun devicectl" >&2
    exit 1
fi

EMPTY=$(mktemp -d)
# devicectl's `--remove-existing-content true` rejects a literally-empty
# source directory (CoreDeviceError 7000); a single sentinel file is
# enough to satisfy the "is a directory" check, after which the original
# tree gets nuked.  The sentinel itself doesn't pass `localModelURL`'s
# `model.mlmodelc/...` probe, so the picker still treats the folder as
# "not downloaded" on next launch.
touch "$EMPTY/.placeholder"
trap 'rm -rf "$EMPTY"' EXIT

echo "Wiping Documents/Models/$MODEL on device $DEVICE..."
xcrun devicectl device copy to --device "$DEVICE" \
    --domain-type appDataContainer \
    --domain-identifier "$BUNDLE_ID" \
    --source "$EMPTY" \
    --destination "Documents/Models/$MODEL" \
    --remove-existing-content true

echo "Done. The model folder on the device now holds only a placeholder,"
echo "and the app's isDownloaded probe will return false on next launch."
