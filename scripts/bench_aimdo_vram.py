"""Poll ComfyUI's /aimdo/vram endpoint and log per-model VRAM residency over time.

The endpoint is provided by the ComfyUI-MemoryVisualization custom node. It
returns per-model `loaded_size` vs `total_size` (i.e. how much of each model
is currently resident in VRAM vs offloaded to CPU under ComfyUI's dynamic
VRAM loader), driver-level free/total VRAM, pytorch active/reserved bytes,
pinned RAM, and (when aimdo is active) per-page residency heatmaps.

Sampling this endpoint during a render captures the *offload pressure*
curve over time — the load-bearing measurement for any attention-kernel
or mask-path change on memory-constrained workloads. "Didn't crash" is
not equivalent to "fit in budget"; the real cost manifests as offload-
driven slowdown, not failure.

Usage
-----
Start ComfyUI with the MemoryVisualization custom node loaded (verify with
`curl http://localhost:8188/aimdo/vram` returning JSON). Then in a separate
terminal:

    uv run python scripts/bench_aimdo_vram.py --output bench_run_A.ndjson

Run the workflow in ComfyUI. When the render completes, Ctrl-C the
polling script. Repeat for the comparison run with a different output
file. Compare the two NDJSON traces with `scripts/compare_aimdo_traces.py`
(if it exists) or pandas / jq.

Output format
-------------
One JSON object per line (NDJSON). Each line:

    {
        "ts": <unix seconds, float>,
        "elapsed_s": <seconds since start, float>,
        "data": <full /aimdo/vram response>
    }

Fields inside `data` of interest:
- `total_vram` / `free_vram`: driver-level VRAM budget snapshot
- `models[i].loaded_size`: bytes of model i currently resident in VRAM
- `models[i].total_size`: bytes of model i total (loaded + offloaded)
- `models[i].vbar_loaded`: aimdo VBAR resident bytes (subset of loaded_size)
- `models[i].pinned_ram`: bytes of model i in pinned host memory

`loaded_size / total_size` per model = residency fraction. Trace the
fraction over time to see offload pressure.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import httpx
import orjson


DEFAULT_ENDPOINT = "http://localhost:8188/aimdo/vram"
DEFAULT_INTERVAL = 1.0  # seconds
REQUEST_TIMEOUT = 2.0  # seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"ComfyUI /aimdo/vram endpoint URL (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL,
        help=f"Polling interval in seconds (default: {DEFAULT_INTERVAL})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NDJSON file path",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional max polling duration in seconds (stops on Ctrl-C otherwise)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # First request to verify the endpoint is reachable
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.get(args.endpoint)
            resp.raise_for_status()
            initial = resp.json()
    except Exception as e:
        print(f"ERROR: cannot reach endpoint {args.endpoint}: {e}", file=sys.stderr)
        print(
            "Verify ComfyUI is running with the MemoryVisualization custom node loaded.",
            file=sys.stderr,
        )
        return 1

    if not initial.get("enabled"):
        print(
            "WARNING: endpoint reports enabled=False (no CUDA device or non-CUDA torch)",
            file=sys.stderr,
        )

    print(f"Polling {args.endpoint} every {args.interval}s -> {args.output}")
    print(f"Initial response: {len(initial.get('models', []))} models loaded, "
          f"total_vram={initial.get('total_vram', 0) // (1024**3)} GiB")
    print("Ctrl-C to stop.")

    stop = False

    def _handle_signal(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    start = time.time()
    n_polls = 0
    errors = 0

    with open(args.output, "wb") as f, httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        while not stop:
            now = time.time()
            elapsed = now - start

            if args.max_duration and elapsed >= args.max_duration:
                break

            try:
                resp = client.get(args.endpoint)
                resp.raise_for_status()
                data = resp.json()
                line = orjson.dumps({"ts": now, "elapsed_s": elapsed, "data": data})
                f.write(line + b"\n")
                f.flush()
                n_polls += 1
            except Exception as e:
                errors += 1
                # Stay alive on transient errors (e.g. ComfyUI briefly busy)
                if errors == 1 or errors % 10 == 0:
                    print(f"WARN: poll {n_polls + errors} failed: {e}", file=sys.stderr)

            # Deadline-based sleep so per-poll work doesn't drift the cadence.
            # Signal delivery interrupts time.sleep on Linux, so Ctrl-C remains responsive.
            deadline = now + args.interval
            try:
                time.sleep(max(0.0, deadline - time.time()))
            except InterruptedError:
                pass

    duration = time.time() - start
    print(f"\nStopped after {duration:.1f}s: {n_polls} polls written, {errors} errors")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
