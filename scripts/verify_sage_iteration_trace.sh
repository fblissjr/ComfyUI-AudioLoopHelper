#!/bin/bash
# Verify the sage override is firing on every loop iteration (Item 7:
# offload-asymmetry detection). Run after a 3+ iteration render with
# AUDIOLOOPHELPER_SAGE_TRACE enabled -- reads the most recent JSONL under
# internal/analysis/runs/sage/ and prints per-iteration call counts.
#
# Interpretation:
#   - Counts roughly equal across iters   -> override survives model offload. GOOD.
#   - iter 0 has N, iter 1+ have 0        -> override dropped by model_patches_to.
#                                            NAG-asymmetry sibling bug.
#   - Zero rows with iter != null         -> LoopIterationStamp not firing.
#
# Usage:
#   scripts/verify_sage_iteration_trace.sh           # latest trace
#   scripts/verify_sage_iteration_trace.sh <path>    # specific JSONL

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEGACY_DIR="$REPO_ROOT/internal/analysis/runs/sage"
DATA_DIR="$REPO_ROOT/data/runs"

if [[ $# -ge 1 ]]; then
    TRACE="$1"
else
    # Search both layouts and pick the most recent by mtime:
    #   Legacy: internal/analysis/runs/sage/sage_<TS>.jsonl   (pre-2026-04-26)
    #   RUN_ID: data/runs/<RUN_ID>/sage.jsonl                  (post-RUN_ID propagation)
    # Use mtime since RUN_ID format and legacy timestamp format don't sort
    # lexicographically against each other.
    TRACE=$( {
        ls -1 "$LEGACY_DIR"/sage_*.jsonl 2>/dev/null || true
        ls -1 "$DATA_DIR"/*/sage.jsonl 2>/dev/null || true
    } | while read -r f; do
        [[ -n "$f" ]] && echo "$(stat -c '%Y' "$f") $f"
    done | sort -n | tail -1 | cut -d' ' -f2- || true)
    if [[ -z "$TRACE" ]]; then
        echo "No trace files found in:"
        echo "  $LEGACY_DIR/"
        echo "  $DATA_DIR/*/"
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

# Single pass over the file. Emits three top-level fields:
#   summary       -- the ON_CLEANUP summary row (may be absent if the run
#                    crashed before cleanup).
#   per_iteration -- grouped by .iter, with per-kernel and fallback counts.
#                    Only rows with .iter != null are counted (initial-render
#                    rows before the loop starts have no stamp, which is
#                    expected and benign).
#   stamp_missing -- count of non-summary rows lacking an iter stamp.
#                    Nonzero is normal (initial render); very large relative
#                    to total_calls suggests LoopIterationStamp isn't wired.
jq -s '{
    summary:        map(select(.event == "summary")) | first,
    stamp_missing:  map(select(.event != "summary" and .iter == null)) | length,
    per_iteration:  map(select(.iter != null))
                    | group_by(.iter)
                    | map({
                        iter:        .[0].iter,
                        total_calls: length,
                        by_kernel:   (group_by(.effective_mode)
                                      | map({kernel: .[0].effective_mode, calls: length})),
                        fallbacks:   (map(select(.fell_back)) | length)
                    })
}' "$TRACE"
