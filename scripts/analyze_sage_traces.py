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
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def load_trace(path: Path) -> list[dict[str, Any]]:
    """Read an NDJSON sage trace; skip the header event and malformed lines."""
    entries: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
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
    ran = [e for e in entries if not e.get("skipped", False)]
    masked = [e for e in ran if e.get("has_mask")]
    unmasked = [e for e in ran if not e.get("has_mask")]

    masked_shapes = Counter(tuple(e.get("shape", [])) for e in masked)
    print(f"\n=== Per-shape masked timing (aggregated across all input traces) ===")
    print(f"{'shape':<25}{'n':>6}  {'p50_us':>10}  {'p95_us':>10}  {'unmasked p50':>14}")
    for sh, n in masked_shapes.most_common():
        masked_times = sorted(e["elapsed_us"] for e in masked if tuple(e.get("shape", [])) == sh)
        unmasked_times = sorted(e["elapsed_us"] for e in unmasked if tuple(e.get("shape", [])) == sh)
        if not masked_times:
            continue
        p50 = masked_times[len(masked_times) // 2]
        p95 = masked_times[int(len(masked_times) * 0.95)] if len(masked_times) >= 20 else max(masked_times)
        u50 = unmasked_times[len(unmasked_times) // 2] if unmasked_times else None
        u50_s = f"{u50:.0f}" if u50 is not None else "—"
        print(f"{str(sh):<25}{n:>6}  {p50:>10.0f}  {p95:>10.0f}  {u50_s:>14}")


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
