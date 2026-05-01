"""Aggregate ComfyUI exec.jsonl into a per-node-class bottleneck report.

Reads the JSONL emitted by `exec_logger.py` when `COMFYUI_EXEC_LOG` is set
(see `start_experiment.sh`). Groups `event=="end"` rows by `class_type`
and reports total wall time, share of total, count, mean, median, p90,
and per-iter median (for nodes that fire >once).

Companion to `scripts/sage_telemetry_summary.py`:
- sage_telemetry_summary covers ATTENTION (per-call routing + masked vs
  unmasked split + gate verdict)
- exec_log_summary covers EVERYTHING ELSE (per-node wall time ranked,
  identifies non-attention bottlenecks: VAEEncode, sampler step,
  text encoding, etc.)

Run both for a complete bottleneck picture. They share the same RUN_ID
artifact root so paths align.

Usage:
    # Auto-find latest exec.jsonl under data/runs/<RUN_ID>/
    uv run --group dev python scripts/exec_log_summary.py --latest

    # Explicit
    uv run --group dev python scripts/exec_log_summary.py \\
        --exec-log data/runs/20260501T101530Z_a3f1/exec.jsonl

    # Per-prompt grouping (when multiple prompts ran in one session)
    uv run --group dev python scripts/exec_log_summary.py --latest --per-prompt

    # Top-N override (default 15)
    uv run --group dev python scripts/exec_log_summary.py --latest --top 25

Companion to telemetry doc: docs/reference/telemetry_and_tracing.md.
Bench procedure: docs/guides/bench_workflow_guide.md.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import orjson


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "data" / "runs"
LEGACY_RUNS_ROOT = REPO_ROOT / "internal" / "analysis" / "runs" / "exec_log"


@dataclass
class NodeStats:
    class_type: str
    count: int
    total_s: float
    mean_s: float
    median_s: float
    p90_s: float
    durations: list[float]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(orjson.loads(line))
            except orjson.JSONDecodeError:
                continue
    return rows


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    k = (len(s) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def aggregate(rows: list[dict], filter_prompt_id: str | None = None) -> tuple[list[NodeStats], float]:
    by_class: dict[str, list[float]] = defaultdict(list)
    total_s = 0.0
    for row in rows:
        if row.get("event") != "end":
            continue
        if filter_prompt_id is not None and row.get("prompt_id") != filter_prompt_id:
            continue
        ct = row.get("class_type") or "?"
        d = row.get("duration_s")
        if not isinstance(d, (int, float)):
            continue
        by_class[ct].append(float(d))
        total_s += float(d)

    stats: list[NodeStats] = []
    for ct, durs in by_class.items():
        stats.append(NodeStats(
            class_type=ct,
            count=len(durs),
            total_s=sum(durs),
            mean_s=sum(durs) / len(durs),
            median_s=statistics.median(durs),
            p90_s=_percentile(durs, 0.9),
            durations=durs,
        ))
    stats.sort(key=lambda s: s.total_s, reverse=True)
    return stats, total_s


def _list_prompt_ids(rows: list[dict]) -> list[str]:
    seen = []
    for row in rows:
        pid = row.get("prompt_id")
        if pid and pid not in seen:
            seen.append(pid)
    return seen


def _format_table(stats: list[NodeStats], total_s: float, top: int = 15) -> str:
    lines = []
    lines.append(f"{'CLASS_TYPE':<40} {'CALLS':>6} {'TOTAL_S':>10} {'PCT':>7} {'MEDIAN_S':>10} {'P90_S':>10}")
    lines.append("-" * 90)
    for s in stats[:top]:
        pct = (s.total_s / total_s * 100.0) if total_s > 0 else 0.0
        lines.append(
            f"{s.class_type[:40]:<40} {s.count:>6} {s.total_s:>10.3f} {pct:>6.1f}% "
            f"{s.median_s:>10.4f} {s.p90_s:>10.4f}"
        )
    if len(stats) > top:
        rest_total = sum(s.total_s for s in stats[top:])
        rest_pct = (rest_total / total_s * 100.0) if total_s > 0 else 0.0
        lines.append("-" * 90)
        lines.append(
            f"{'(other ' + str(len(stats) - top) + ' classes)':<40} "
            f"{'':>6} {rest_total:>10.3f} {rest_pct:>6.1f}% {'':>10} {'':>10}"
        )
    lines.append("-" * 90)
    lines.append(f"{'TOTAL':<40} {sum(s.count for s in stats):>6} {total_s:>10.3f} {'100.0%':>7}")
    return "\n".join(lines)


def _resolve_exec_log(args: argparse.Namespace) -> Path:
    if args.exec_log:
        return Path(args.exec_log)
    if not args.latest:
        raise SystemExit("--exec-log <path> or --latest required")
    candidates: list[Path] = []
    if DEFAULT_RUNS_ROOT.is_dir():
        for run_dir in sorted(DEFAULT_RUNS_ROOT.iterdir(), reverse=True):
            f = run_dir / "exec.jsonl"
            if f.exists() and f.stat().st_size > 0:
                candidates.append(f)
                break
    if not candidates and LEGACY_RUNS_ROOT.is_dir():
        legacy = sorted(LEGACY_RUNS_ROOT.glob("exec_*.jsonl"), reverse=True)
        if legacy:
            candidates.append(legacy[0])
    if not candidates:
        raise SystemExit(
            f"--latest: no exec.jsonl found under {DEFAULT_RUNS_ROOT}/<run>/exec.jsonl "
            f"or {LEGACY_RUNS_ROOT}/exec_*.jsonl. Did you launch via ./start_experiment.sh?"
        )
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--exec-log", help="Path to exec.jsonl (or use --latest).")
    ap.add_argument("--latest", action="store_true",
                    help="Auto-pick latest exec.jsonl under data/runs/<run>/.")
    ap.add_argument("--per-prompt", action="store_true",
                    help="Emit one table per prompt_id (default: aggregated).")
    ap.add_argument("--top", type=int, default=15,
                    help="Show top N classes by total_s (default 15).")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of the formatted table.")
    args = ap.parse_args()

    path = _resolve_exec_log(args)
    rows = _load_jsonl(path)
    print(f"# exec_log_summary  log={path}  rows={len(rows)}", file=sys.stderr)

    if args.json:
        out: dict = {}
        prompts = _list_prompt_ids(rows) if args.per_prompt else [None]
        for pid in prompts:
            stats, total = aggregate(rows, filter_prompt_id=pid)
            key = pid or "all"
            out[key] = {
                "total_s": total,
                "classes": [
                    {"class_type": s.class_type, "count": s.count,
                     "total_s": s.total_s, "mean_s": s.mean_s,
                     "median_s": s.median_s, "p90_s": s.p90_s,
                     "pct_of_total": (s.total_s / total * 100.0) if total > 0 else 0.0}
                    for s in stats
                ],
            }
        sys.stdout.buffer.write(orjson.dumps(out, option=orjson.OPT_INDENT_2))
        sys.stdout.buffer.write(b"\n")
        return 0

    if args.per_prompt:
        for pid in _list_prompt_ids(rows):
            stats, total = aggregate(rows, filter_prompt_id=pid)
            print(f"\n=== prompt_id={pid}  total_wall={total:.2f}s  ({sum(s.count for s in stats)} node-runs) ===\n")
            print(_format_table(stats, total, top=args.top))
    else:
        stats, total = aggregate(rows)
        print(f"\n=== exec.jsonl  total_wall={total:.2f}s  ({sum(s.count for s in stats)} node-runs) ===\n")
        print(_format_table(stats, total, top=args.top))

    return 0


if __name__ == "__main__":
    sys.exit(main())
