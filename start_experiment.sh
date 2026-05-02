#!/bin/bash
# start_experiment.sh — telemetry-enabled wrapper around ComfyUI's start.sh.
#
# Owns four env-var exports that make this plugin's instrumentation work:
#   - RUN_ID                       (single correlation key for one render)
#   - AUDIOLOOPHELPER_SAGE_TRACE   (per-attention-call sage tracer)
#   - COMFYUI_EXEC_LOG             (per-node ComfyUI execution logger)
#   - AUDIOLOOPHELPER_PER_PROMPT   (route artifacts under data/runs/${RUN_ID}/${prompt_id}/; off by default — only relevant for multi-prompt-per-session bench tools like sage-fork's bench_e2e_ltx)
#
# All three default to "auto" (or auto-generated, for RUN_ID) when unset,
# so just running `./start_experiment.sh [mode]` gets you a fully-traced
# render. Plain `<comfyui>/start.sh [mode]` runs ComfyUI without any of
# this plugin's telemetry — that's the design split as of 2026-04-26.
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
#   data/runs/${RUN_ID}/exec.jsonl       (per-node ComfyUI events)
#   data/runs/${RUN_ID}/sage.jsonl       (per-attention-call sage telemetry)
#   data/runs/${RUN_ID}/profiler/        (torch.profiler outputs, if wired)
# and at ComfyUI's configured output directory:
#   <comfyui_output>/LTX-2_${RUN_ID}_*.{mp4,png}    (when harness mutates filename_prefix)
#
# Disable for one launch (no edits): set the var empty in the calling shell.
#   RUN_ID= AUDIOLOOPHELPER_SAGE_TRACE= COMFYUI_EXEC_LOG= ./start_experiment.sh
# All three treat empty/unset as "disabled". The `${VAR-auto}` form
# (no colon) substitutes "auto" only when the var is unset; if you set
# it to "" explicitly, it stays empty and the tracer skips.
#
# What each tracer captures:
#   AUDIOLOOPHELPER_SAGE_TRACE  (per-attention-call sage tracer)
#     One JSONL row per attention call: tensor shape, has_mask, mode,
#     effective_mode, fell_back, elapsed_us, iter, prompt_id,
#     dispatched_kernel. NO prompt text, NO tensor values, NO model
#     weights. Forensic-grade -- ~22k syscalls per 5-iter LTX render.
#     Unset before a perf-sensitive production gen if the syscall
#     overhead shows up.
#
#   COMFYUI_EXEC_LOG  (per-node ComfyUI execution logger)
#     One JSONL row per node start + per node end: prompt_id, node_id,
#     class_type, duration_s, input/output shape snapshots. The input
#     snapshot CAN capture short string node-inputs up to 120 chars,
#     which includes prompt text. If your prompts are sensitive, either
#     keep this off or redact traces before sharing them.
#
# Retention: NO auto-cleanup. Files accumulate until you `rm` them.
# Manual cleanup: rm -rf data/runs/<glob>/  (or
#   rm -rf internal/analysis/runs/{sage,exec_log}/  for legacy runs).
#
# Full reference: docs/reference/environment.md and
# docs/reference/telemetry_and_tracing.md.

set -e

export RUN_ID=${RUN_ID-$(date -u +%Y%m%dT%H%M%SZ)_$(openssl rand -hex 2)}
export AUDIOLOOPHELPER_SAGE_TRACE=${AUDIOLOOPHELPER_SAGE_TRACE-auto}
export COMFYUI_EXEC_LOG=${COMFYUI_EXEC_LOG-auto}
# Per-prompt artifact routing: data/runs/${RUN_ID}/${prompt_id}/<category>.<ext>
# instead of flat data/runs/${RUN_ID}/<category>.<ext>. Lets multiple prompts
# in the same ComfyUI session land in their own subdirs (warm-cache benches,
# parameter sweeps). Reader scripts (bench_compare_runs, exec_log_summary,
# sage_telemetry_summary) auto-detect both layouts.
export AUDIOLOOPHELPER_PER_PROMPT=${AUDIOLOOPHELPER_PER_PROMPT-1}

echo "[start_experiment.sh] RUN_ID=$RUN_ID  SAGE_TRACE=$AUDIOLOOPHELPER_SAGE_TRACE  EXEC_LOG=$COMFYUI_EXEC_LOG  PER_PROMPT=$AUDIOLOOPHELPER_PER_PROMPT"

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clean prior chrome-trace.json files (~1.8GB each) before launching.
# Keeps summary.txt + memory_timeline.html (small, structured). Without
# this, trace.json files accumulate at ~1.8GB per profiled bench render
# and reach 10GB+ after a session of experiments.
if [[ -d "$PLUGIN_DIR/data/runs" ]]; then
    # Single-pass: list-and-delete via -print -delete; count from the printed lines.
    deleted_count=$(find "$PLUGIN_DIR/data/runs" -name "trace.json" -type f -print -delete 2>/dev/null | wc -l)
    if [[ "$deleted_count" -gt 0 ]]; then
        echo "[start_experiment.sh] cleaned $deleted_count prior chrome trace.json file(s) (kept summary.txt)"
    fi
fi

# ComfyUI plugin layout: <comfyui>/custom_nodes/<this-plugin>/
COMFYUI_DIR="$(cd "$PLUGIN_DIR/../.." && pwd)"
START_SH="$COMFYUI_DIR/start.sh"

if [[ ! -f "$START_SH" ]]; then
    echo "[start_experiment.sh] ERROR: $START_SH not found" >&2
    echo "  Expected ComfyUI start script two levels up from this plugin." >&2
    exit 1
fi

# `cd` so start.sh's relative paths (`./temp`, `python main.py`) resolve.
# Lets users invoke this wrapper from any working directory.
cd "$COMFYUI_DIR"

# Use bash explicitly — start.sh may not have +x.
exec bash "$START_SH" "$@"
