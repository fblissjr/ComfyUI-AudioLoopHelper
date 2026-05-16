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
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from workflow_utils import DATA_RUNS_DIR


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

    torch.profiler stamps `Module Hierarchy` or `python frame` info into
    each event's args. The exact key varies by torch version; we check
    the most common ones in order.
    """
    args = ev.get("args") or {}
    # Most direct: when record_shapes=True + module annotation is on
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


def aggregate_by_module(events: list[dict], device: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Build {module_path: {op_name: {count, total_us, shape}}}."""
    # Two-level dict, lazy-initialized
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
            path = "<unattributed>"
        op_name = ev["name"]
        dur_us = float(ev.get("dur", 0.0))
        entry = out[path][op_name]
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
