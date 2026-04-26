#!/bin/bash
# start_experiment.sh — thin wrapper that generates a single RUN_ID and exec's
# the ComfyUI start.sh. Every telemetry artifact (exec_log, sage trace,
# profiler outputs) and the workflow harness pick up RUN_ID from the env so
# all artifacts for one render share a directory key — making cross-system
# correlation trivial.
#
# Without this wrapper, the three loggers stamp their own filenames from
# time.time() at startup and drift apart by seconds, so files from the same
# conceptual render look unrelated. Single env var fixes it.
#
# RUN_ID format: ${ISO8601_UTC}_${rand4}   e.g. 20260426T134522Z_a3f1
#   - lexicographically sortable (sort by mtime ≡ sort by RUN_ID)
#   - collision-resistant (rand4 is two bytes from openssl)
#   - readable in `ls`
#
# Usage:
#   ./start_experiment.sh [mode]              # same modes as start.sh
#   RUN_ID=my_test ./start_experiment.sh      # override RUN_ID for a known run
#
# Artifacts land at (relative to this plugin):
#   data/runs/${RUN_ID}/exec.jsonl
#   data/runs/${RUN_ID}/sage.jsonl
#   data/runs/${RUN_ID}/profiler/
# and at ComfyUI's configured output directory:
#   <comfyui_output>/LTX-2_${RUN_ID}_*.{mp4,png}    (when harness mutates filename_prefix)
#
# Set RUN_ID empty to opt out of correlation:
#   RUN_ID= ./start_experiment.sh    # falls back to legacy timestamped paths

set -e

export RUN_ID=${RUN_ID-$(date -u +%Y%m%dT%H%M%SZ)_$(openssl rand -hex 2)}
echo "[start_experiment.sh] RUN_ID=$RUN_ID"

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ComfyUI plugin layout: <comfyui>/custom_nodes/<this-plugin>/
COMFYUI_DIR="$(cd "$PLUGIN_DIR/../.." && pwd)"
START_SH="$COMFYUI_DIR/start.sh"

if [[ ! -f "$START_SH" ]]; then
    echo "[start_experiment.sh] ERROR: $START_SH not found" >&2
    echo "  Expected ComfyUI start script two levels up from this plugin." >&2
    exit 1
fi

# Use bash explicitly — start.sh may not have +x.
exec bash "$START_SH" "$@"
