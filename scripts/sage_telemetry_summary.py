"""Aggregate sage tracer JSONL into a per-mode summary.

Reads the JSONL emitted by `nodes_sage.py` when `AUDIOLOOPHELPER_SAGE_TRACE`
is set, groups by effective_mode + has_mask, and reports median / p90 /
count / total_us / pct_of_total per group.

The two cross-sections that gate further kernel-side work:

- `(effective_mode='fp16_triton', has_mask=True)` -- the masked cross-attn
  path. If this is <5% of total gen wall time, mask kernel work in
  sage-fork closes permanently. 5-15%, defer 6 months. >15%, justified.
- `(effective_mode='fp8_cuda++', has_mask=False)` -- the unmasked self-attn
  path. Gives the "where time actually goes" denominator.

Total gen wall time can be provided two ways:
- Explicit: `--total-wall-ms <N>` (most accurate, from a separate timer).
- Inferred: from a companion exec-log JSONL (`--exec-log <path>`), summing
  the duration_s of nodes whose class_type matches one of
  `--ksampler-class` (default: KSampler*, SamplerCustomAdvanced).

If neither is provided, the report omits `pct_of_total`.

Usage:
    uv run --group dev python scripts/sage_telemetry_summary.py \\
        --sage-log internal/analysis/runs/sage/sage_2026-04-25_*.jsonl \\
        --exec-log internal/analysis/runs/exec_log/exec_2026-04-25_*.jsonl
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Iterable, Iterator

import orjson


_DEFAULT_KSAMPLER_CLASSES = (
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustomAdvanced",
    "SamplerCustom",
)


def load_jsonl(path: Path) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSONL file. Skips blank lines and
    malformed lines silently -- the tracer's line-buffered output can be
    truncated by a crash mid-line, and a single bad row shouldn't kill a
    whole forensic run."""
    with open(path, "rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield orjson.loads(raw)
            except orjson.JSONDecodeError:
                continue


def _percentile(samples: list[float], pct: float) -> float:
    """Inclusive nearest-rank percentile. For p90 of 10 samples, returns
    sample[9] (the largest). Avoids depending on numpy."""
    if not samples:
        return 0.0
    sorted_samples = sorted(samples)
    idx = max(0, min(len(sorted_samples) - 1, int(round(pct * (len(sorted_samples) - 1)))))
    return sorted_samples[idx]


def aggregate(rows: Iterable[dict], total_wall_us: float | None = None) -> dict:
    """Group per-call samples by (effective_mode, has_mask) and compute
    median/p90/count/total. If `total_wall_us` is supplied, also compute
    pct_of_total per group."""
    groups: dict[tuple[str, bool], list[float]] = {}
    fallbacks = 0
    total_calls = 0

    for row in rows:
        # Skip the SageTracer summary line (event=summary). Per-call rows
        # don't have an "event" field.
        if row.get("event") == "summary":
            continue
        elapsed = row.get("elapsed_us")
        if elapsed is None:
            continue
        total_calls += 1
        if row.get("fell_back"):
            fallbacks += 1
        key = (row.get("effective_mode") or row.get("mode") or "?", bool(row.get("has_mask")))
        groups.setdefault(key, []).append(float(elapsed))

    out_groups: dict[tuple[str, bool], dict] = {}
    for key, samples in groups.items():
        entry = {
            "count": len(samples),
            "median_us": float(statistics.median(samples)),
            "p90_us": _percentile(samples, 0.90),
            "total_us": float(sum(samples)),
        }
        if total_wall_us is not None and total_wall_us > 0:
            entry["pct_of_total"] = round(100.0 * entry["total_us"] / total_wall_us, 2)
        out_groups[key] = entry

    return {
        "total_calls": total_calls,
        "fallback_count": fallbacks,
        "groups": out_groups,
        "total_wall_us": total_wall_us,
    }


def load_jsonl_with_count(path: Path) -> tuple[list[dict], int]:
    """Like `load_jsonl` but materializes the list and returns the count
    of malformed lines that were skipped. Use when you want the operator
    to know about silent corruption rather than letting it slide."""
    rows: list[dict] = []
    malformed = 0
    with open(path, "rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(orjson.loads(raw))
            except orjson.JSONDecodeError:
                malformed += 1
    return rows, malformed


def gate_section(summary: dict, *, effective_mode: str, has_mask: bool) -> dict | None:
    """Return the canonical gate dict for one (mode, mask) pair, or None
    if the pair has no samples in this trace. The gate cross-section is
    `(fp16_triton, True)` and `(fp8_cuda++, False)` -- the masked-triton
    fraction of total wall time decides whether further mask-kernel work
    in sage-fork is justified."""
    return summary["groups"].get((effective_mode, has_mask))


# Backwards-compatible alias for prior callers / tests.
phase0_section = gate_section


def total_wall_us_from_exec_log(exec_log_path: Path, ksampler_classes: tuple[str, ...]) -> float | None:
    """Sum duration_s of nodes whose class_type is in ksampler_classes.
    Returns total in microseconds, or None if no matching events found."""
    total_s = 0.0
    matched = 0
    for row in load_jsonl(exec_log_path):
        if row.get("event") != "end":
            continue
        if row.get("class_type") not in ksampler_classes:
            continue
        d = row.get("duration_s")
        if d is None:
            continue
        total_s += float(d)
        matched += 1
    if matched == 0:
        return None
    return total_s * 1_000_000.0


def _format_table(summary: dict) -> str:
    rows = [
        f"{'effective_mode':<16} {'mask':<6} {'count':>6} {'median_ms':>10} {'p90_ms':>10} "
        f"{'total_ms':>10} {'pct':>6}",
    ]
    if not summary["groups"]:
        return rows[0] + "\n  (no per-call samples)"
    for (mode, has_mask), entry in sorted(summary["groups"].items()):
        median_ms = entry["median_us"] / 1000.0
        p90_ms = entry["p90_us"] / 1000.0
        total_ms = entry["total_us"] / 1000.0
        pct = f"{entry.get('pct_of_total', '-')}" if "pct_of_total" in entry else "  -"
        rows.append(
            f"{mode:<16} {str(has_mask):<6} {entry['count']:>6} {median_ms:>10.3f} "
            f"{p90_ms:>10.3f} {total_ms:>10.3f} {pct:>6}"
        )
    return "\n".join(rows)


def _format_gate_cross_section(summary: dict) -> list[str]:
    """The gate-relevant cross-sections, formatted in the canonical shape."""
    lines = []
    masked_triton = gate_section(summary, effective_mode="fp16_triton", has_mask=True)
    unmasked_fp8pp = gate_section(summary, effective_mode="fp8_cuda++", has_mask=False)

    def _line(label: str, entry: dict | None) -> str:
        if entry is None:
            return f"{label}: no samples"
        median_ms = entry["median_us"] / 1000.0
        p90_ms = entry["p90_us"] / 1000.0
        pct = entry.get("pct_of_total")
        pct_str = f"{pct}%" if pct is not None else "-"
        return (
            f"{label}: median={median_ms:.2f} ms, p90={p90_ms:.2f} ms, "
            f"count={entry['count']}, %_of_total={pct_str}"
        )

    lines.append(_line("masked_triton", masked_triton))
    lines.append(_line("unmasked_fp8++", unmasked_fp8pp))
    return lines


def _gate_verdict(summary: dict) -> str:
    """Apply the gate criteria to masked_triton's pct of total wall time.
    Decision text only -- the user makes the actual call from the data."""
    masked = gate_section(summary, effective_mode="fp16_triton", has_mask=True)
    if masked is None or "pct_of_total" not in masked:
        return "gate verdict: skipped (no pct_of_total available)"
    pct = masked["pct_of_total"]
    if pct < 5.0:
        return f"gate verdict: <5% ({pct}%) -- mask kernel work closed permanently"
    if pct <= 15.0:
        return f"gate verdict: 5-15% ({pct}%) -- revisit in 6 months, don't act now"
    return f"gate verdict: >15% ({pct}%) -- mask kernel work justified"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sage-log", type=Path, required=True, help="Sage tracer JSONL path.")
    parser.add_argument("--exec-log", type=Path, default=None, help="Companion exec-log JSONL path.")
    parser.add_argument("--total-wall-ms", type=float, default=None, help="Explicit total gen wall time in ms.")
    parser.add_argument("--ksampler-class", action="append", default=None,
                        help="class_type to treat as a sampler when reading --exec-log. Can repeat.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table.")
    args = parser.parse_args(argv)

    if not args.sage_log.exists():
        print(f"sage log not found: {args.sage_log}", file=sys.stderr)
        return 2

    total_wall_us: float | None = None
    if args.total_wall_ms is not None:
        total_wall_us = args.total_wall_ms * 1000.0
    elif args.exec_log is not None and args.exec_log.exists():
        ks_classes = tuple(args.ksampler_class) if args.ksampler_class else _DEFAULT_KSAMPLER_CLASSES
        total_wall_us = total_wall_us_from_exec_log(args.exec_log, ks_classes)

    sage_rows, malformed = load_jsonl_with_count(args.sage_log)
    if malformed > 0:
        print(f"warning: skipped {malformed} malformed line(s) in {args.sage_log}", file=sys.stderr)
    summary = aggregate(sage_rows, total_wall_us=total_wall_us)

    if args.json:
        # Convert tuple keys to strings for JSON encodability.
        out = {
            "total_calls": summary["total_calls"],
            "fallback_count": summary["fallback_count"],
            "total_wall_us": summary["total_wall_us"],
            "groups": {
                f"{mode}|{int(has_mask)}": entry
                for (mode, has_mask), entry in summary["groups"].items()
            },
        }
        print(orjson.dumps(out, option=orjson.OPT_INDENT_2).decode())
        return 0

    print(_format_table(summary))
    print()
    print("gate cross-section:")
    for line in _format_gate_cross_section(summary):
        print(f"  {line}")
    print()
    print(_gate_verdict(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
