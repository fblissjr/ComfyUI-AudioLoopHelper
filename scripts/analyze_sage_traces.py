"""Aggregate per-shape kernel timing across one or more sage.jsonl traces.

Use after ComfyUI renders captured with start_experiment.sh to characterize
where attention time went. Aggregates calls by (shape, has_mask) and reports
median + p95 elapsed_us, plus dispatched-kernel and fallback counts.

Multi-file input lets you check reproducibility across runs (identical
config → identical stats is a strong signal) or compare across configs
(different sage mode → different kernel dispatch + timing).

Usage
-----
    uv run python scripts/analyze_sage_traces.py path/to/sage.jsonl [path/to/another.jsonl ...]

By default scans `data/runs/<RUN_ID>/<prompt_id>/sage.jsonl` if no args
given. Reports a per-run summary table + an aggregate per-shape timing
table across all files.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import orjson


def load_trace(path: Path) -> list[dict[str, Any]]:
    """Read an NDJSON sage trace; skip the header event and malformed lines."""
    entries: list[dict[str, Any]] = []
    with open(path, "rb") as f:
        for line in f:
            try:
                e = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if e.get("event") == "header":
                continue
            entries.append(e)
    return entries


def per_run_summary(label: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    ran = [e for e in entries if not e.get("skipped", False)]
    masked = [e for e in ran if e.get("has_mask")]
    fallbacks = [e for e in ran if e.get("fell_back")]
    kernels = Counter(e.get("dispatched_kernel") for e in masked if e.get("dispatched_kernel"))
    masked_shapes = Counter(tuple(e.get("shape", [])) for e in masked)
    return {
        "label": label,
        "total": len(entries),
        "ran": len(ran),
        "masked": len(masked),
        "fallbacks": len(fallbacks),
        "kernels": dict(kernels),
        "top_masked_shapes": dict(masked_shapes.most_common(3)),
    }


def aggregate(entries: list[dict[str, Any]]) -> None:
    # Group elapsed_us per (has_mask, shape) in one pass — avoids O(shapes × entries)
    # rescans when shapes is large.
    masked_by_shape: dict[tuple, list[float]] = defaultdict(list)
    unmasked_by_shape: dict[tuple, list[float]] = defaultdict(list)
    for e in entries:
        if e.get("skipped", False):
            continue
        elapsed = e.get("elapsed_us")
        if elapsed is None:
            continue
        sh = tuple(e.get("shape", []))
        (masked_by_shape if e.get("has_mask") else unmasked_by_shape)[sh].append(elapsed)

    print("\n=== Per-shape masked timing (aggregated across all input traces) ===")
    print(f"{'shape':<25}{'n':>6}  {'p50_us':>10}  {'p95_us':>10}  {'unmasked p50':>14}")
    for sh, _ in Counter({k: len(v) for k, v in masked_by_shape.items()}).most_common():
        masked_times = masked_by_shape[sh]
        if not masked_times:
            continue
        p50 = statistics.median(masked_times)
        # quantiles requires n>=2; for small samples fall back to max as p95.
        p95 = statistics.quantiles(masked_times, n=20)[18] if len(masked_times) >= 20 else max(masked_times)
        unmasked_times = unmasked_by_shape.get(sh, [])
        u50_s = f"{statistics.median(unmasked_times):.0f}" if unmasked_times else "—"
        print(f"{str(sh):<25}{len(masked_times):>6}  {p50:>10.0f}  {p95:>10.0f}  {u50_s:>14}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "traces",
        nargs="*",
        type=Path,
        help="Path(s) to sage.jsonl trace files. If omitted, scans data/runs/*/*/sage.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.traces:
        paths = args.traces
    else:
        paths = sorted(Path("data/runs").glob("*/*/sage.jsonl"))
        if not paths:
            print("No traces found under data/runs/*/*/sage.jsonl", file=sys.stderr)
            return 1

    print(f"=== Per-run summary ({len(paths)} files) ===")
    print(f"{'label':<40}  {'total':>6} {'ran':>6} {'masked':>6} {'fall':>4}  kernel(s)")
    print("-" * 120)

    all_entries: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        entries = load_trace(path)
        summary = per_run_summary(str(path), entries)
        label = path.parent.name[:38]  # prompt_id is the dir name
        kernels_str = ",".join(f"{k}={v}" for k, v in summary["kernels"].items())
        print(f"{label:<40}  {summary['total']:>6} {summary['ran']:>6} {summary['masked']:>6} {summary['fallbacks']:>4}  {kernels_str}")
        all_entries.extend(entries)

    aggregate(all_entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
