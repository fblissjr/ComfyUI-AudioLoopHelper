#!/bin/bash
# ComfyUI startup script — canonical deploy template.
# Last updated: 2026-06-04
#
# Tuned for 24GB-class consumer cards (RTX 4090 etc.) running LTX 2.3 /
# WAN2.1 / similar fp8-scaled video diffusion models. Adjust the mode
# definitions below if your hardware budget differs.
#
# Deploy by copying this to <comfyui_root>/start.sh and editing the
# environment-variable defaults at the top (COMFYUI_OUTPUT_DIR /
# COMFYUI_INPUT_DIR / COMFYUI_TEMP_DIR / COMFYUI_PORT). Defaults work
# out of the box if you use ComfyUI's standard relative layout (./output,
# ./input, ./temp).
#
# Usage:
#   ./start.sh                       # default mode
#   ./start.sh [mode]                # pick a mode
#   ./start.sh [mode] [flags...]     # mode + extra flags forwarded to main.py
#   ./start.sh [flags...]            # default mode + extra flags
#   ./start.sh -h | --help           # show this help
#
# Modes:
#   default     LTX 2.3 / WAN2.1: perf flags + 0.5GB reserve, dynamic VRAM ON (node cache ON)
#   safe        Fallback when default OOMs (lowvram, fp16-unet, fp32-vae, 4GB reserve)
#   extreme     Maximum speed, may OOM (fp8_e4m3fn-unet, fp16-vae)
#   minimal     Last resort, very slow (novram, cpu-vae, async-offload)
#   nodynvram   BENCH ONLY (alias: bench) — perf flags + the no-dynvram kill
#               switch + --cache-none + reserve 0 (see comments in that case)
#   highvram    Keep models resident after use (may OOM with large models
#               + heavy text encoder on a 24GB card)
#
# Extra ComfyUI flags after the mode are forwarded verbatim to main.py.
# Examples:
#   ./start.sh default --verbose DEBUG
#   ./start.sh nodynvram --reserve-vram 0
#   ./start.sh --disable-dynamic-vram          # implicit default mode + flag
#
# For experiment-mode launches that enable AudioLoopHelper's per-render
# telemetry (RUN_ID + sage tracer + exec logger), use the wrapper at
#   <comfyui_root>/custom_nodes/ComfyUI-AudioLoopHelper/start_experiment.sh
# which sets the relevant env vars then exec's back here.

set -e

# ---- Customize these for your deployment if non-default --------------------
# Override at invocation time (e.g. COMFYUI_OUTPUT_DIR=/mnt/data/output ./start.sh)
# or edit the defaults here. All four resolve to ComfyUI's standard relative
# layout if unset.
: "${COMFYUI_OUTPUT_DIR:=./output}"
: "${COMFYUI_INPUT_DIR:=./input}"
: "${COMFYUI_TEMP_DIR:=./temp}"
: "${COMFYUI_PORT:=8188}"
# ----------------------------------------------------------------------------

# Help short-circuit before any other parsing.
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

# Pick the mode. If $1 starts with `-`, treat it as a flag (not a mode) and
# fall back to the default mode without shifting. Lets callers pass flags
# inline without thinking about the positional structure.
if [[ -z "$1" || "$1" == -* ]]; then
    MODE="default"
else
    MODE="$1"
    shift
fi

# Base configuration — always used, regardless of mode.
BASE_ARGS=(
    --output-directory "$COMFYUI_OUTPUT_DIR"
    --temp-directory "$COMFYUI_TEMP_DIR"
    --input-directory "$COMFYUI_INPUT_DIR"
    --preview-method none
    --preview-size 512
    --disable-api-nodes
    --port "$COMFYUI_PORT"
)

# Common perf flags reused by every "go fast" mode (default, extreme,
# nodynvram, highvram). Safe on Ada/Hopper with fp8-scaled models.
# mmap loading reduces RAM pressure when checkpoints are large.
#
# Deliberately NOT including --fast: that's a blanket switch for four
# experimental optimizations (fp16_accumulation, fp8_matrix_mult,
# cublas_ops, autotune). Newer fp8-quantized weights manage their own
# per-layer matmul dispatch and don't need it. If you want a specific
# sub-optimization, pass it explicitly, e.g.: `--fast fp16_accumulation`.
PERF_ARGS=(
    --cuda-malloc
    --supports-fp8-compute
    --mmap-torch-files
)

# Dynamic-VRAM kill switch (bench mode only). OOM-instead-of-offload:
#   --disable-dynamic-vram  : kills aimdo page-level offload during inference
#   --disable-async-offload : kills async weight streams (the lower-level mechanism)
# DO NOT promote into default. Tried 2026-06-04, reverted same day: full-song
# loop renders kernel-OOM at the FINAL full-video VAE decode — with paging
# disabled the resident diffusion model is never evicted, the VideoVAE loads
# with 0 MB usable, and the decode balloons system RAM until the kernel kills
# the process (silently — see the exit-status line at the bottom). Dynamic
# VRAM is required for the repo's primary workload (full-song loop renders).
NODYNVRAM_ARGS=(
    --disable-dynamic-vram
    --disable-async-offload
)

CMD_ARGS=("${BASE_ARGS[@]}")

case "$MODE" in
    default)
        echo "[start.sh] mode=default — LTX 2.3 / WAN2.1 (perf flags + 0.5GB reserve, dynamic VRAM + node cache ON, 48GB RAM headroom)"
        # Dynamic VRAM stays ON here — see the NODYNVRAM_ARGS comment above for
        # why promoting the kill switch into default crashes full-song loop
        # renders. The node cache stays ON too — NEVER add --cache-none here
        # (fatal for loop renders; see docs/reference/debug_tools.md).
        #
        # --cache-ram 48 raises the RAM-pressure cache's per-node free-RAM
        # floor (ComfyUI default: ~10GB). The full-song final VAE decode
        # allocates tens of GB inside ONE node where no eviction can run —
        # with only the default floor, the kernel OOM-kills the process at
        # the last step. 48GB guarantees the spike fits; the executor frees
        # pinned staging before evicting cache entries. If loop iterations
        # start re-encoding upstream (text encoder / audio VAE per iter),
        # the floor is too aggressive for that workload — lower toward 32.
        CMD_ARGS+=(
            "${PERF_ARGS[@]}"
            --reserve-vram 0.5
            --cache-ram 48
        )
        ;;

    safe)
        echo "[start.sh] mode=safe — conservative (lowvram + fp16-unet + fp32-vae + 4GB reserve)"
        CMD_ARGS+=(
            --lowvram
            --fp16-unet
            --fp32-vae
            --cache-none
            --reserve-vram 4
        )
        ;;

    extreme)
        echo "[start.sh] mode=extreme — max perf, may OOM (fp8_e4m3fn-unet, fp16-vae + perf flags)"
        CMD_ARGS+=(
            "${PERF_ARGS[@]}"
            --fp8_e4m3fn-unet
            --fp16-vae
        )
        ;;

    minimal)
        echo "[start.sh] mode=minimal — max memory savings (novram, fp16-unet, cpu-vae)"
        CMD_ARGS+=(
            --novram
            --fp16-unet
            --cpu-vae
            --cache-none
            --async-offload
        )
        ;;

    nodynvram|bench)
        echo "[start.sh] mode=bench (nodynvram) — BENCH ONLY: default's no-dynvram flags + --cache-none + reserve 0 (NEVER for loop renders)"
        # Targets the "ComfyUI memory management is masking my kernel's actual
        # memory profile" scenario. Model load/unload between stages stays
        # normal (text encoder offloads after use, etc.), but during a forward
        # pass nothing shuffles weights (NODYNVRAM_ARGS, defined above) — so if
        # a kernel's per-call working set exceeds budget, you OOM cleanly
        # instead of seeing offload slowdown. Notes:
        #   --cache-none            : no node-output cache between renders (cleaner repro).
        #                             WARNING: catastrophic for TensorLoop/looping
        #                             workflows — see the cache-none guard below esac.
        #   --reserve-vram 0        : maximize the budget (user budget == actual budget)
        # NOT included: --gpu-only / --highvram (would OOM at load on 24GB
        # with LTX 2.3 + large text encoders), --disable-smart-memory
        # (inverted name — forces MORE offload).
        CMD_ARGS+=(
            "${PERF_ARGS[@]}"
            --reserve-vram 0
            "${NODYNVRAM_ARGS[@]}"
            --cache-none
        )
        ;;

    highvram)
        echo "[start.sh] mode=highvram — keep models resident after use (may OOM with large models)"
        CMD_ARGS+=(
            "${PERF_ARGS[@]}"
            --highvram
            --reserve-vram 0
        )
        ;;

    *)
        echo "Unknown mode: $MODE" >&2
        echo "Run \`$0 --help\` for available modes." >&2
        exit 1
        ;;
esac

# --cache-none guard. --cache-none maps to ComfyUI's NullCache (caches nothing).
# That is intended ONLY for single-render benchmarking/tracing — a clean repro,
# and it stops the node cache from short-circuiting identical-input tracer
# queues. It is CATASTROPHIC for looping workflows: TensorLoop /
# execution-inversion loops (ComfyUI-NativeLooping_testing TensorLoopOpen/Close)
# re-emit an expanded subgraph every iteration and rely on the node-output cache
# to reuse non-contained UPSTREAM nodes. With no cache, every iteration
# re-executes all of them — prompt batch-encode (text-encoder reload), full-audio
# VAE encode, keyframe video extract + VAE encode, and model re-patching — for an
# N x slowdown (N = iterations). Use the 'default' mode for loop / full-song
# renders. This is launch config, not a workflow-wiring bug.
if [[ " ${CMD_ARGS[*]} $* " == *" --cache-none "* ]]; then
    echo "[start.sh] WARNING: --cache-none active -> node-output cache OFF."
    echo "[start.sh]   Looping (TensorLoop) workflows re-execute ALL upstream nodes each"
    echo "[start.sh]   iteration (text-encode, audio/keyframe VAE, model re-patch) => N x slow."
    echo "[start.sh]   Use 'default' mode for loop/full-song renders; --cache-none is for"
    echo "[start.sh]   single-render benchmarking/tracing only."
fi

# Surface exactly what main.py receives. Useful for verifying flag passthrough
# from wrappers and for reproducibility (anyone re-running a bench can copy
# the resolved arg list from this line).
echo "[start.sh] forwarding to main.py: ${CMD_ARGS[*]} $*"

# Execute ComfyUI. The grep filter strips a noisy SSL warning that fires
# during normal operation. pipefail + the status echo make abnormal exits
# visible: without them, the pipe reports grep's exit code and a SIGKILL
# (e.g. the kernel OOM killer) drops you back to the prompt with no message.
set +e
set -o pipefail
uv run --active python main.py "${CMD_ARGS[@]}" "$@" 2>&1 | grep -v "SSL connection is closed"
status=$?
echo "[start.sh] ComfyUI exited with status ${status}" \
     "(137 = SIGKILL, usually the kernel OOM killer: journalctl -k | grep -i oom)"
exit "$status"
