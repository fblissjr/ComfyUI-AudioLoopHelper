"""bench_compare_runs.

Last updated: 2026-05-01

Side-by-side comparator across multiple bench runs. Reads each run's
exec.jsonl + sage.jsonl from `data/runs/<RUN_ID>/`, aggregates per-class
wall time and per-attention-call cost, and emits a comparison table
optimized for sage A/B analysis (or any controlled-variable bench).

Composes with:
  - apply_iclora_bench_sage_arm.py (produces the arm variants you'll
    render)
  - scripts/exec_log_summary.py (single-run per-node breakdown)
  - scripts/sage_telemetry_summary.py (single-run attention split)

This script is the multi-run roll-up — answers questions like:
  "is arm_kj faster than arm_ours, and where does the delta come from?"
  "did arm_off (sage disabled) regress more than the attention share
   would suggest?"
  "is the per-iter VAE encode cost stable across arms, or is sage
   bleeding into other timings?"

Usage:
    # Compare 2+ named runs
    uv run --group dev python scripts/bench_compare_runs.py \\
        --runs arm_ours arm_off arm_kj arm_stacked

    # Auto-pick the N most recent runs
    uv run --group dev python scripts/bench_compare_runs.py --latest 4

    # Set a baseline arm; deltas are computed against it
    uv run --group dev python scripts/bench_compare_runs.py \\
        --runs arm_ours arm_off arm_kj --baseline arm_ours

    # JSON output for downstream tooling
    uv run --group dev python scripts/bench_compare_runs.py \\
        --runs arm_ours arm_kj --json

Requires telemetry from `start_experiment.sh` (sets RUN_ID +
AUDIOLOOPHELPER_SAGE_TRACE + COMFYUI_EXEC_LOG). If a run is missing
sage.jsonl, attention rows show '-'; if missing exec.jsonl, that run is
skipped with a warning.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import orjson


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO_ROOT / "data" / "runs"

KSAMPLER_CLASSES = (
    "KSampler", "KSamplerAdvanced",
    "SamplerCustom", "SamplerCustomAdvanced",
)


@dataclass
class RunStats:
    run_id: str
    exec_path: Path
    sage_path: Path | None
    total_wall_s: float = 0.0
    sampler_wall_s: float = 0.0
    by_class: dict[str, float] = field(default_factory=dict)
    by_class_count: dict[str, int] = field(default_factory=dict)
    attention_calls: int = 0
    attention_total_us: float = 0.0
    attention_by_mode: dict[str, dict] = field(default_factory=dict)


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


def _resolve_run_path(run_id: str) -> Path | None:
    """Resolve a run id to its root directory.

    Supports BOTH layouts:
    - flat:       data/runs/<id>/exec.jsonl
    - per-prompt: data/runs/<id>/<prompt_id>/exec.jsonl  (when launched
                  with AUDIOLOOPHELPER_PER_PROMPT=1)

    Returns the run root (data/runs/<id>) in both cases. Caller uses
    `_run_exec_files(run_dir)` to get the actual exec.jsonl path(s)."""
    candidates = [RUNS_ROOT / run_id]
    if not candidates[0].is_dir():
        for p in RUNS_ROOT.iterdir() if RUNS_ROOT.is_dir() else []:
            if p.is_dir() and run_id.lower() in p.name.lower():
                candidates.append(p)
    for c in candidates:
        if c.is_dir() and _run_jsonl_files(c, "exec"):
            return c
    return None


def _run_jsonl_files(run_dir: Path, name: str) -> list[Path]:
    """Return all `<name>.jsonl` files under a run dir, supporting both
    flat (`<run>/<name>.jsonl`) and per-prompt subdir
    (`<run>/<prompt_id>/<name>.jsonl`) layouts. Returns flat-only if
    present (single-prompt run), else lists per-prompt subdirs sorted
    by mtime so warmup → measurement ordering is preserved. Empty list
    if no matching files."""
    flat = run_dir / f"{name}.jsonl"
    if flat.exists():
        return [flat]
    return sorted(run_dir.glob(f"*/{name}.jsonl"), key=lambda p: p.stat().st_mtime)


def _aggregate_run(run_id: str) -> RunStats | None:
    run_dir = _resolve_run_path(run_id)
    if run_dir is None:
        print(f"warn: run '{run_id}' not found under {RUNS_ROOT}", file=sys.stderr)
        return None
    exec_files = _run_jsonl_files(run_dir, "exec")
    sage_files = _run_jsonl_files(run_dir, "sage")
    # Aggregated mode: concat all per-prompt files into one set of stats.
    if not exec_files:
        return None
    primary_exec = exec_files[0]
    primary_sage = sage_files[0] if sage_files else None

    stats = RunStats(run_id=run_dir.name, exec_path=primary_exec, sage_path=primary_sage)

    by_class: dict[str, float] = defaultdict(float)
    by_class_count: dict[str, int] = defaultdict(int)
    for exec_path in exec_files:
        for row in _load_jsonl(exec_path):
            if row.get("event") != "end":
                continue
            d = row.get("duration_s")
            if not isinstance(d, (int, float)):
                continue
            ct = row.get("class_type") or "?"
            by_class[ct] += float(d)
            by_class_count[ct] += 1
            stats.total_wall_s += float(d)
            if ct in KSAMPLER_CLASSES:
                stats.sampler_wall_s += float(d)
    stats.by_class = dict(by_class)
    stats.by_class_count = dict(by_class_count)

    if sage_files:
        atten_by_mode: dict[tuple[str, bool], list[float]] = defaultdict(list)
        for sage_path in sage_files:
            for row in _load_jsonl(sage_path):
                elapsed = row.get("elapsed_us")
                if not isinstance(elapsed, (int, float)):
                    continue
                mode = row.get("effective_mode") or "auto"
                has_mask = bool(row.get("has_mask"))
                atten_by_mode[(mode, has_mask)].append(float(elapsed))
                stats.attention_calls += 1
                stats.attention_total_us += float(elapsed)

        for (mode, has_mask), durs in atten_by_mode.items():
            stats.attention_by_mode[f"{mode}{'@masked' if has_mask else '@unmasked'}"] = {
                "count": len(durs),
                "total_us": sum(durs),
                "median_us": statistics.median(durs),
            }

    return stats


def _format_delta(value: float, baseline: float | None) -> str:
    if baseline is None or baseline == 0.0:
        return ""
    delta = (value - baseline) / baseline * 100.0
    sign = "+" if delta > 0 else ""
    return f" ({sign}{delta:.1f}%)"


def _format_table(runs: list[RunStats], baseline: RunStats | None, top: int) -> str:
    if not runs:
        return "(no runs to compare)"

    lines = []
    headers = ["metric"] + [r.run_id for r in runs]
    lines.append(" | ".join(f"{h:<28}" if i == 0 else f"{h:<22}"
                            for i, h in enumerate(headers)))
    lines.append("-+-".join("-" * (28 if i == 0 else 22)
                            for i in range(len(headers))))

    def row(label: str, cells: list[str]) -> str:
        parts = [f"{label:<28}"] + [f"{v:<22}" for v in cells]
        return " | ".join(parts)

    # Top-line summary
    base_total = baseline.total_wall_s if baseline else None
    base_sampler = baseline.sampler_wall_s if baseline else None
    base_atten = baseline.attention_total_us / 1_000_000.0 if baseline else None

    lines.append(row("total_wall_s",
        [f"{r.total_wall_s:>8.2f}{_format_delta(r.total_wall_s, base_total)}" for r in runs]))
    lines.append(row("sampler_wall_s",
        [f"{r.sampler_wall_s:>8.2f}{_format_delta(r.sampler_wall_s, base_sampler)}" for r in runs]))
    lines.append(row("sampler_pct",
        [f"{(r.sampler_wall_s/r.total_wall_s*100.0 if r.total_wall_s else 0):>5.1f}%" for r in runs]))
    lines.append(row("attention_calls",
        [f"{r.attention_calls:>8d}" if r.sage_path else "       -" for r in runs]))
    lines.append(row("attention_wall_s",
        [f"{r.attention_total_us/1_000_000.0:>8.2f}{_format_delta(r.attention_total_us/1_000_000.0, base_atten)}"
         if r.sage_path else "       -" for r in runs]))
    lines.append(row("attention_pct_of_total",
        [f"{(r.attention_total_us/1_000_000.0/r.total_wall_s*100.0 if r.total_wall_s and r.sage_path else 0):>5.1f}%"
         if r.sage_path else "    -" for r in runs]))

    lines.append("")

    # Per-class wall, ranked by max share across runs
    all_classes = set()
    for r in runs:
        all_classes.update(r.by_class.keys())

    ranked = sorted(
        all_classes,
        key=lambda c: max(r.by_class.get(c, 0.0) for r in runs),
        reverse=True,
    )

    lines.append("=== per-node-class wall time (top {}; sorted by max-across-runs) ===".format(top))
    lines.append(row("CLASS_TYPE", [r.run_id for r in runs]))
    lines.append("-+-".join("-" * (28 if i == 0 else 22) for i in range(len(headers))))
    for ct in ranked[:top]:
        cells = []
        for r in runs:
            v = r.by_class.get(ct, 0.0)
            n = r.by_class_count.get(ct, 0)
            base_v = baseline.by_class.get(ct, 0.0) if baseline else None
            delta = _format_delta(v, base_v) if baseline and ct in baseline.by_class else ""
            cells.append(f"{v:>7.2f}s ×{n}{delta}")
        lines.append(row(ct[:28], cells))

    lines.append("")

    # Sage mode breakdown
    if any(r.sage_path for r in runs):
        lines.append("=== sage attention by (effective_mode, has_mask) ===")
        all_modes = set()
        for r in runs:
            all_modes.update(r.attention_by_mode.keys())
        for mode in sorted(all_modes):
            cells = []
            for r in runs:
                m = r.attention_by_mode.get(mode)
                if not r.sage_path:
                    cells.append("       -")
                elif m is None:
                    cells.append("       0")
                else:
                    cells.append(f"{m['count']:>5d}@{m['total_us']/1000:.0f}ms")
            lines.append(row(mode[:28], cells))

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--runs", nargs="+", help="Run IDs (matches data/runs/<id>/).")
    ap.add_argument("--latest", type=int,
                    help="Auto-pick N most recent runs under data/runs/.")
    ap.add_argument("--baseline", help="Run ID to treat as baseline; deltas relative.")
    ap.add_argument("--top", type=int, default=10,
                    help="Top N node classes to show (default 10).")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of formatted table.")
    args = ap.parse_args()

    if args.runs:
        run_ids = args.runs
    elif args.latest:
        if not RUNS_ROOT.is_dir():
            raise SystemExit(f"--latest: {RUNS_ROOT} doesn't exist")
        # Match runs in either flat or per-prompt layout.
        candidates = sorted(
            (p for p in RUNS_ROOT.iterdir()
             if p.is_dir() and _run_jsonl_files(p, "exec")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        run_ids = [p.name for p in candidates[:args.latest]]
    else:
        raise SystemExit("--runs <id...> or --latest <N> required")

    runs: list[RunStats] = []
    for rid in run_ids:
        s = _aggregate_run(rid)
        if s:
            runs.append(s)
    if not runs:
        raise SystemExit("no runs loaded")

    baseline = None
    if args.baseline:
        baseline = next((r for r in runs if r.run_id == args.baseline or args.baseline in r.run_id), None)
        if baseline is None:
            print(f"warn: baseline '{args.baseline}' not in loaded runs", file=sys.stderr)

    if args.json:
        out = {
            "baseline": baseline.run_id if baseline else None,
            "runs": [
                {
                    "run_id": r.run_id,
                    "total_wall_s": r.total_wall_s,
                    "sampler_wall_s": r.sampler_wall_s,
                    "attention_calls": r.attention_calls,
                    "attention_total_us": r.attention_total_us,
                    "by_class": r.by_class,
                    "by_class_count": r.by_class_count,
                    "attention_by_mode": r.attention_by_mode,
                }
                for r in runs
            ],
        }
        sys.stdout.buffer.write(orjson.dumps(out, option=orjson.OPT_INDENT_2))
        sys.stdout.buffer.write(b"\n")
        return 0

    print(_format_table(runs, baseline, args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
