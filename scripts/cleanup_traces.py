"""Trim accumulated tracer output under `data/runs/` to the N most-recent runs.

Usage:
    uv run --group dev python scripts/cleanup_traces.py             # dry-run, keeps 20
    uv run --group dev python scripts/cleanup_traces.py --apply     # actually delete
    uv run --group dev python scripts/cleanup_traces.py --keep 10 --apply
    uv run --group dev python scripts/cleanup_traces.py --pattern 'adaln_audit_*' --apply

Each top-level dir under `data/runs/` is one RUN_ID. The script sorts
by mtime (most recent first), keeps the top N, and deletes the rest.

Safety:
- Defaults to **dry-run**. Must pass `--apply` to actually delete.
- Default `--keep` is 20.
- `--pattern` (glob) lets you scope the operation to a subset of
  RUN_IDs without touching anything else (e.g. only clean up
  `bench_profile_*` runs, leaving `adaln_audit_*` intact).
- Refuses to touch anything outside `data/runs/`.

Companion: produced by `tracers/` (see `tracers/__init__.py`). Run after
a session of experiments to keep disk under control. Doesn't auto-run.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

from workflow_utils import DATA_RUNS_DIR


def human_bytes(n: int) -> str:
    """Format a byte count as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n //= 1024
    return f"{n}PB"


def dir_size(path: Path) -> int:
    """Sum file sizes under a directory tree. Doesn't follow symlinks."""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--keep", type=int, default=20,
                   help="Keep the N most-recent RUN_ID dirs (default: 20)")
    p.add_argument("--pattern", default="*",
                   help="Glob pattern to scope which RUN_IDs are considered (default: '*')")
    p.add_argument("--apply", action="store_true",
                   help="Actually delete (default is dry-run)")
    p.add_argument("--runs-dir", type=Path, default=None,
                   help="Override the data/runs path (default: <repo>/data/runs)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir or DATA_RUNS_DIR
    if not runs_dir.exists():
        print(f"[cleanup_traces] no runs dir at {runs_dir}", file=sys.stderr)
        return 0

    # Enumerate top-level RUN_ID dirs matching the pattern.
    candidates = [
        d for d in runs_dir.iterdir()
        if d.is_dir() and fnmatch.fnmatch(d.name, args.pattern)
    ]
    if not candidates:
        print(f"[cleanup_traces] no run dirs match pattern '{args.pattern}' under {runs_dir}")
        return 0

    # Sort by mtime descending. Most recent first.
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    keep = candidates[: args.keep]
    drop = candidates[args.keep :]

    print(f"[cleanup_traces] runs_dir={runs_dir}")
    print(f"[cleanup_traces] pattern='{args.pattern}'  matched={len(candidates)}  keeping={len(keep)}  dropping={len(drop)}")

    if not drop:
        print("[cleanup_traces] nothing to drop. Done.")
        return 0

    total_to_drop = 0
    for d in drop:
        sz = dir_size(d)
        total_to_drop += sz
        action = "DELETE" if args.apply else "would delete"
        print(f"  {action:14s} {d.name}  ({human_bytes(sz)})")

    print(f"[cleanup_traces] total reclaimed: {human_bytes(total_to_drop)}  (dry_run={not args.apply})")

    if args.apply:
        for d in drop:
            # Sanity: refuse to delete anything outside runs_dir.
            if not d.resolve().is_relative_to(runs_dir.resolve()):
                print(f"  REFUSED: {d} is outside {runs_dir}", file=sys.stderr)
                continue
            shutil.rmtree(d, ignore_errors=True)
        print(f"[cleanup_traces] deleted {len(drop)} run dir(s)")
    else:
        print("[cleanup_traces] dry-run only. Re-run with --apply to actually delete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
