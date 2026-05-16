"""Per-module-path aten-op breakdown from torch.profiler chrome traces.

Consumes the `torch_profile.*.json` files produced by `tracers.torch_profile`
and reports, for each requested module path, the per-aten-op wall-time
breakdown: op_name x call_count x total_ms x avg_ms x primary_input_shape.

Usage:
    uv run --group dev python scripts/analyze_torch_profile.py \\
        data/runs/<RUN_ID>/<prompt_id>/torch_profile.0.json \\
        --modules audio_attn1 attn1

If `--modules` is omitted, the script reports the top-N module paths by
total CUDA wall-time. With `--modules` specified, the script filters to
just those paths and reports each in turn.

Designed for the LTX 2.3 cross-attention-path comparison: contrasting
`audio_attn1`'s per-op profile against video `attn1`'s. If audio's
Linears are 50-100x slower per call than video's (cold L2 after the
big video kernel), bandwidth-bound-at-small-T-after-big-kernel is
confirmed; concurrent dispatch still overlaps that wall-time.

## Interpreting `cpu_op.dur` vs `kernel.dur`

`cat=cpu_op` aten events carry a `dur` field measuring CPU-side
dispatch wall-clock, NOT the time the op's underlying GPU work
consumed. For async pytorch ops (most aten ops are async on CUDA),
the dispatcher fires, queues the kernel, returns. The `dur` is just
the dispatcher's CPU time — typically microseconds. Summing
`cpu_op.dur` across ~40k async ops gives a CPU-dispatch wall-clock
total in the tens of seconds, but on a compute-overlapping render
those ops fire concurrent with prior GPU work, so the sum DOES NOT
represent serial render time eaten.

For "X% of sampler wall-time" claims, anchor on `cat=kernel`
(actual CUDA kernel time) plus any synchronous CPU work that blocks
the dispatcher. NOT the sum-of-async-op-dispatch-times.

Practical example: an early read of an FML2V audit's chrome trace
showed `aten::copy_ + aten::to = 67s` on a 141s sampler. A
nodynvram A/B (designed to eliminate dynamic VRAM offload) showed
those numbers UNCHANGED. The 67s was CPU-dispatch wall-clock summed
across ~40k async ops, not 47% of sampler-serial time.

## Module attribution path

`torch.profiler.profile(with_modules=True)` is TorchScript-only per
the pytorch docs (silent no-op on eager-mode models). For eager-mode
LTX 2.3 we use the `record_function` annotations placed by
`tracers/ffn_attn.py`'s pre/post hooks instead. Each sub-module
forward emits a `cat=user_annotation` span named like
`audio_attn1/block_5` in the chrome trace; `_build_span_index` collects
them and `find_enclosing_span` bisects to attribute each aten op to
its parent. Verified end-to-end on an FML2V audit render: 384 distinct
annotation names (= 48 blocks × 8 sub-modules) emitted per render,
with the analyzer's `--modules audio_attn1 attn1` filter returning
properly-grouped op breakdowns.
"""

from __future__ import annotations

import argparse
import bisect
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from workflow_utils import DATA_RUNS_DIR

# Shared with `tracers/ffn_attn.py::BLOCK_ANNOTATION_MARKER` — keep in
# sync if the producer's format changes. Duplicated inline (not
# imported) because this script is a `scripts/`-side CLI tool and
# doesn't import from the package.
_BLOCK_ANNOTATION_MARKER = "/block_"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("path", type=Path, nargs="?",
                   help="torch_profile.*.json (default: most recent under data/runs)")
    p.add_argument("--modules", nargs="*", default=None,
                   help="Module paths to filter by (e.g. audio_attn1 attn1). "
                        "If omitted, reports the top-N modules by wall-time.")
    p.add_argument("--top", type=int, default=10,
                   help="Top-N to report when --modules is omitted (default: 10)")
    p.add_argument("--device", choices=("cuda", "cpu", "both"), default="cpu",
                   help="Which events to include: cpu=aten-op dispatch (default; "
                        "right for per-aten breakdown), cuda=raw kernel time only, "
                        "both=union.")
    return p.parse_args()


def find_latest_trace() -> Path | None:
    """Most-recent `torch_profile.*.json` under data/runs/."""
    candidates = sorted(
        DATA_RUNS_DIR.rglob("torch_profile.*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_trace(path: Path) -> list[dict]:
    """Load a chrome trace JSON and return its event list."""
    import orjson
    data = orjson.loads(path.read_bytes())
    # Chrome trace format: top-level is either a list of events or a
    # dict with a `traceEvents` key. torch.profiler emits the dict shape.
    if isinstance(data, dict):
        return data.get("traceEvents", [])
    return data


def keep_event(ev: dict, device: str) -> bool:
    """Filter to compute events on the requested device.

    Note: aten:: ops in torch.profiler chrome traces are categorized as
    `cpu_op` (the CPU-side dispatch), even when the actual work happens
    on GPU via async kernel launches. The aten op's `dur` reflects the
    full wall-time from CPU perspective. For per-aten-op aggregation,
    `device='cpu'` is the right filter.

    `device='cuda'` filters to the templated kernel events (cat=kernel)
    — useful when you want raw GPU kernel time without dispatch overhead.
    """
    if ev.get("ph") != "X":  # only complete (duration) events
        return False
    cat = ev.get("cat", "")
    if device == "cuda":
        return cat in ("kernel", "gpu_memcpy", "gpu_memset")
    if device == "cpu":
        return cat in ("cpu_op", "operator")
    return cat in ("cpu_op", "kernel", "gpu_memcpy", "gpu_memset", "operator")


def is_aten_op(ev: dict) -> bool:
    """Is this an aten-namespace op (not a kernel-name-mangled event)?"""
    name = ev.get("name", "")
    return name.startswith("aten::")


def module_path_of(ev: dict) -> str | None:
    """Extract the module path from event args.

    `with_modules=True` is TorchScript-only; for eager-mode LTX 2.3 we
    derive module identity from the `record_function` annotations
    placed by `tracers/ffn_attn.py`'s pre/post hooks (named like
    `audio_attn1/block_5`). Walk the event's parent chain via the
    chrome trace `External id` to find the nearest enclosing
    record_function span.

    Returns None if the event isn't nested under a known annotation —
    e.g. ops outside `BasicAVTransformerBlock` sub-modules.
    """
    args = ev.get("args") or {}
    # TorchScript path (kept for completeness, no-op on eager)
    for key in ("Module Hierarchy", "module_hierarchy", "module"):
        val = args.get(key)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, list) and val:
            return "/".join(str(x) for x in val)
    return None


def primary_input_shape(ev: dict) -> str:
    """Format the first input shape from event args, or '?' if absent."""
    args = ev.get("args") or {}
    shapes = args.get("Input Dims") or args.get("input_shapes")
    if not shapes:
        return "?"
    # Format: list of lists like [[1, 100, 2048], [2048, 2048]]
    if isinstance(shapes, list) and shapes:
        first = shapes[0]
        if isinstance(first, list):
            return "x".join(str(d) for d in first)
        return str(first)
    return "?"


def find_enclosing_span(
    starts: list[float],
    ends: list[float],
    names: list[str],
    ts: float,
) -> str | None:
    """Binary-search the span list for one containing `ts`.

    Parallel-array layout (starts/ends/names instead of list of tuples)
    saves a tuple-unpack per probe. `bisect_right(starts, ts) - 1`
    gives the highest span that started at or before `ts`; check its
    end. For LTX 2.3 ~5000 spans × ~1M aten ops, linear scan blows the
    analyzer's 30s budget; bisect is 5-7 probes per lookup.
    """
    if not starts:
        return None
    idx = bisect.bisect_right(starts, ts) - 1
    if idx < 0:
        return None
    if ts <= ends[idx]:
        return names[idx]
    return None


def _build_span_index(events: list[dict]) -> dict[tuple, tuple[list[float], list[float], list[str]]]:
    """Collect `record_function` annotations into bisect-ready arrays.

    Per (pid, tid) returns parallel arrays (starts, ends, names) sorted
    by start. Spans come from `tracers/ffn_attn.py`'s pre/post hooks
    (names match `BLOCK_ANNOTATION_MARKER`). Sort is defensive against
    chrome traces that aren't strictly time-ordered.
    """
    raw: dict[tuple, list[tuple[float, float, str]]] = defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X" or ev.get("cat") != "user_annotation":
            continue
        name = ev.get("name", "")
        if _BLOCK_ANNOTATION_MARKER not in name:
            continue
        ts = float(ev.get("ts", 0.0))
        dur = float(ev.get("dur", 0.0))
        raw[(ev.get("pid"), ev.get("tid"))].append((ts, ts + dur, name))
    index: dict[tuple, tuple[list[float], list[float], list[str]]] = {}
    for key, spans in raw.items():
        spans.sort(key=lambda s: s[0])
        index[key] = (
            [s[0] for s in spans],
            [s[1] for s in spans],
            [s[2] for s in spans],
        )
    return index


def aggregate_by_module(events: list[dict], device: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Build {module_path: {op_name: {count, total_us, shape}}}.

    Attribution sources, in order: (a) `Module Hierarchy` from
    `with_modules=True` (TorchScript path), or (b) the
    `record_function` annotation an aten op falls inside (eager-mode
    path, via `tracers/ffn_attn.py`'s pre/post hooks). Falls back to
    `<unattributed>` for ops outside any annotated span.
    """
    index = _build_span_index(events)
    empty: tuple[list[float], list[float], list[str]] = ([], [], [])
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {
        "count": 0,
        "total_us": 0.0,
        "shapes": set(),
    }))
    for ev in events:
        if not keep_event(ev, device):
            continue
        if not is_aten_op(ev):
            continue
        path = module_path_of(ev)
        if path is None:
            starts, ends, names = index.get((ev.get("pid"), ev.get("tid")), empty)
            path = find_enclosing_span(starts, ends, names, float(ev.get("ts", 0.0)))
        if path is None:
            path = "<unattributed>"
        dur_us = float(ev.get("dur", 0.0))
        entry = out[path][ev["name"]]
        entry["count"] += 1
        entry["total_us"] += dur_us
        entry["shapes"].add(primary_input_shape(ev))
    return out


def report_module(path: str, ops: dict[str, dict[str, Any]]) -> None:
    """Print a per-op breakdown for one module path."""
    total_us = sum(e["total_us"] for e in ops.values())
    total_count = sum(e["count"] for e in ops.values())
    print(f"\n=== {path}  ({total_count} ops, total {total_us/1000:.2f} ms) ===")
    print(f"{'op_name':40s} {'count':>6s} {'total_ms':>10s} {'avg_us':>10s}  shape(s)")
    print("-" * 100)
    rows = sorted(ops.items(), key=lambda kv: -kv[1]["total_us"])
    for op_name, e in rows:
        avg_us = e["total_us"] / e["count"] if e["count"] else 0
        shapes = ", ".join(sorted(e["shapes"]))[:30]
        print(f"{op_name:40s} {e['count']:6d} {e['total_us']/1000:10.3f} {avg_us:10.2f}  {shapes}")


def select_modules(by_module: dict, requested: list[str] | None, top: int) -> list[str]:
    """Pick which module paths to report."""
    if requested:
        # Substring match on each requested module name. Lets users say
        # `audio_attn1` and match `model.transformer_blocks.5.audio_attn1`.
        result = []
        for needle in requested:
            for path in by_module:
                if needle in path and path not in result:
                    result.append(path)
        if not result:
            print(f"[analyze_torch_profile] no module paths matched {requested}", file=sys.stderr)
            print(f"  available paths (top {top}):", file=sys.stderr)
            for p in sorted(by_module, key=lambda k: -sum(e["total_us"] for e in by_module[k].values()))[:top]:
                print(f"    {p}", file=sys.stderr)
        return result
    # No filter — top-N by total wall-time
    ranked = sorted(by_module, key=lambda k: -sum(e["total_us"] for e in by_module[k].values()))
    return ranked[:top]


def main() -> int:
    args = parse_args()
    path = args.path or find_latest_trace()
    if path is None:
        print("[analyze_torch_profile] no trace found under data/runs/", file=sys.stderr)
        return 1
    if not path.exists():
        print(f"[analyze_torch_profile] {path} not found", file=sys.stderr)
        return 1

    print(f"[analyze_torch_profile] loading {path} ({path.stat().st_size / 1e6:.1f} MB)")
    events = load_trace(path)
    print(f"[analyze_torch_profile] {len(events)} events")

    by_module = aggregate_by_module(events, args.device)
    if not by_module:
        print("[analyze_torch_profile] no aten ops found on requested device", file=sys.stderr)
        return 1

    selected = select_modules(by_module, args.modules, args.top)
    if not selected:
        return 1

    for p in selected:
        report_module(p, by_module[p])

    return 0


if __name__ == "__main__":
    sys.exit(main())
