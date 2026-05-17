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
import time
from pathlib import Path
from typing import Any

from workflow_utils import DATA_RUNS_DIR

# Suffix marker used to identify sidecars produced by `extract_module_summary`.
# Any chrome trace `torch_profile.N.json` that already has a sibling
# `torch_profile.N.modules_summary.json` is treated as already-extracted.
_MODULES_SUMMARY_SUFFIX = ".modules_summary.json"


def human_bytes(n: int) -> str:
    """Format a byte count as a short human-readable string."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def extract_module_summary(chrome_trace: Path) -> Path | None:
    """Write `<stem>.modules_summary.json` alongside a chrome trace.

    Idempotent: skips if the sidecar already exists. Defensive: catches
    every exception and returns None so a single bad trace doesn't block
    the rest of a cleanup pass.

    The sidecar carries per-module aten-op aggregations derived from the
    `record_function` annotations placed by `tracers/ffn_attn.py`. Shape
    sets are converted to sorted lists for JSON serializability. Source
    trace provenance (path, size, event count) is preserved so a future
    reader can verify which raw trace produced this summary.
    """
    sidecar = chrome_trace.parent / (chrome_trace.stem + _MODULES_SUMMARY_SUFFIX)
    if sidecar.exists():
        return sidecar

    try:
        import orjson
        # Lazy import — keeps the analyzer dependency out of the cleanup
        # script's startup cost for the common "no chrome traces present"
        # case. `scripts/` is on sys.path via conftest / direct invocation.
        from analyze_torch_profile import aggregate_by_module, load_trace
    except Exception as e:
        print(f"  [extract] skipped {chrome_trace.name}: import failed ({type(e).__name__}: {e})", file=sys.stderr)
        return None

    t0 = time.time()
    try:
        events = load_trace(chrome_trace)
        by_module = aggregate_by_module(events, device="cpu")
    except Exception as e:
        print(f"  [extract] skipped {chrome_trace.name}: load/aggregate failed ({type(e).__name__}: {e})", file=sys.stderr)
        return None

    # Convert shape sets to sorted lists; keep the rest of the structure.
    payload: dict[str, Any] = {
        "source_trace": str(chrome_trace),
        "source_size_bytes": chrome_trace.stat().st_size,
        "total_events": len(events),
        "modules": {
            module_path: {
                op_name: {
                    "count": entry["count"],
                    "total_us": round(entry["total_us"], 1),
                    "shapes": sorted(entry["shapes"]),
                }
                for op_name, entry in ops.items()
            }
            for module_path, ops in by_module.items()
        },
    }
    try:
        sidecar.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    except Exception as e:
        print(f"  [extract] write failed for {sidecar.name}: {type(e).__name__}: {e}", file=sys.stderr)
        return None

    elapsed = time.time() - t0
    print(f"  [extract] {chrome_trace.name} -> {sidecar.name} ({sidecar.stat().st_size//1024} KB, {elapsed:.1f}s)", file=sys.stderr)
    return sidecar


def find_chrome_traces(run_dir: Path) -> list[Path]:
    """Walk a RUN_ID dir for `torch_profile.*.json` raw chrome traces.

    Excludes existing `*.modules_summary.json` sidecars so we don't try
    to recursively aggregate our own output.
    """
    return [
        p for p in run_dir.rglob("torch_profile.*.json")
        if not p.name.endswith(_MODULES_SUMMARY_SUFFIX)
    ]


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
    p.add_argument("--no-extract", action="store_true",
                   help="Skip per-module sidecar extraction. Default behaviour "
                        "runs `analyze_torch_profile.aggregate_by_module` on "
                        "each chrome trace and writes a `*.modules_summary.json` "
                        "alongside before deletion. This flag is the escape hatch "
                        "for cases where the user has already extracted manually.")
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
        # Extract per-module sidecars before rm-tree, then HOIST them
        # out of the to-be-deleted tree to a sibling archive so they
        # survive retention. Without the hoist, sidecars inside the
        # doomed RUN_ID dir get deleted along with the raw traces —
        # destroying the very data the sidecar was meant to preserve.
        if not args.no_extract:
            chrome_traces = [p for d in drop for p in find_chrome_traces(d)]
            if chrome_traces:
                archive_root = runs_dir / "_archived_sidecars"
                print(f"[cleanup_traces] extracting per-module sidecars for {len(chrome_traces)} chrome trace(s) before delete...")
                for ct in chrome_traces:
                    sidecar = extract_module_summary(ct)
                    if sidecar is None:
                        continue
                    # Mirror the relative path under the archive root so
                    # the sidecar's RUN_ID + prompt_id provenance is
                    # recoverable from the archive layout alone.
                    try:
                        rel = sidecar.relative_to(runs_dir)
                    except ValueError:
                        continue
                    target = archive_root / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sidecar, target)
                    print(f"  [archive] {sidecar.name} -> {target.relative_to(runs_dir)}", file=sys.stderr)

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
