#!/bin/bash
# Verify the sage override is firing on every loop iteration (Item 7:
# offload-asymmetry detection). Run this after a 3+ iteration render with
# AUDIOLOOPHELPER_SAGE_TRACE enabled -- reads the most recent JSONL under
# internal/analysis/runs/sage/ and prints per-iteration call counts.
#
# Interpretation:
#   - Counts roughly equal across iters   -> override survives model offload/reload. GOOD.
#   - iter 0 has N, iter 1+ have 0        -> override dropped by model_patches_to.
#                                            This is the NAG-asymmetry sibling bug.
#   - No rows with iter != null           -> LoopIterationStamp not firing. Check
#                                            workflow wiring (apply_iteration_stamp.py).
#
# Usage:
#   scripts/verify_sage_iteration_trace.sh           # use latest trace
#   scripts/verify_sage_iteration_trace.sh <path>    # specific JSONL

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="$REPO_ROOT/internal/analysis/runs/sage"

if [[ $# -ge 1 ]]; then
    TRACE="$1"
else
    TRACE=$(ls -t "$RUNS_DIR"/sage_*.jsonl 2>/dev/null | head -1 || true)
    if [[ -z "$TRACE" ]]; then
        echo "No trace files found in $RUNS_DIR/"
        echo "Run ComfyUI with AUDIOLOOPHELPER_SAGE_TRACE=auto and a 3+ iteration workflow first."
        exit 1
    fi
fi

if [[ ! -f "$TRACE" ]]; then
    echo "Trace file not found: $TRACE"
    exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required but not installed. Install with: sudo apt install jq"
    exit 1
fi

echo "trace: $TRACE"
echo

# Overall stats from the summary row (emitted on ON_CLEANUP).
echo "=== summary ==="
jq 'select(.event == "summary")' "$TRACE"
echo

echo "=== per-iteration call counts (with effective_mode breakdown) ==="
jq 'select(.iter != null) | {iter, effective_mode, fell_back}' "$TRACE" \
    | jq -s 'group_by(.iter) | map({
        iter: .[0].iter,
        total_calls: length,
        by_kernel: (group_by(.effective_mode) | map({kernel: .[0].effective_mode, calls: length})),
        fallbacks: (map(select(.fell_back == true)) | length)
    })'
echo

# Quick null-iter check: rows without iteration stamp mean LoopIterationStamp didn't stamp.
NULL_COUNT=$(jq -s '[.[] | select(.event != "summary" and .iter == null)] | length' "$TRACE")
if [[ "$NULL_COUNT" -gt 0 ]]; then
    echo "warning: $NULL_COUNT rows have iter=null (LoopIterationStamp may not be wired,"
    echo "         or these rows are from the initial render which runs outside the loop)."
fi
