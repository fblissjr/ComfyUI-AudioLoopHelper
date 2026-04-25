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

Total gen wall time can be provided three ways:
- Explicit: `--total-wall-ms <N>` (most accurate, from a separate timer).
- Inferred from exec log: `--exec-log <path>`. If the exec log spans
  multiple prompt_ids, the script switches to per-prompt grouping
  automatically -- single-bucket pct across multiple prompts is the
  central denominator failure mode (sums ksampler durations regardless
  of prompt, double-counts loops).
- Sage-span fallback: `--use-sage-span` uses (max_ts - min_ts) of the
  sage rows. Self-contained, no exec log needed.

If none provided, the report omits `pct_of_total`.

Routing-table mirror: sage's `effective_mode` field records the
consumer-visible routing decision -- masked cross-attn calls correctly
land on `fp16_triton`, but unmasked calls record `auto` because
`sageattn()` dispatches inside sage-fork where the consumer can't see.
`--arch sm89_cuda12_8` (or arch field stamped in the trace itself by
the tracer) enables post-hoc kernel inference: `(auto, has_mask=False)`
on sm89+CUDA12.8 maps to `fp8_cuda++` (`sageattn_qk_int8_pv_fp8_cuda`
with `pv_accum_dtype="fp32+fp16"`). This mirrors the subset of
`sageattention/core.py::sageattn` that the consumer's call pattern
reaches (no `smooth_k`, no LSE, head_dim in {64,120,128}); broaden if
the consumer's call pattern changes. Replace with sage-fork's
`get_last_dispatched_kernel()` once that ships.

Usage:
    uv run --group dev python scripts/sage_telemetry_summary.py \\
        --sage-log internal/analysis/runs/sage/sage_2026-04-25_*.jsonl \\
        --exec-log internal/analysis/runs/exec_log/exec_2026-04-25_*.jsonl
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import orjson


_DEFAULT_KSAMPLER_CLASSES = (
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustomAdvanced",
    "SamplerCustom",
)


# ---------------------------------------------------------------------------
# Per-prompt windowing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptWindow:
    """A prompt's lifetime in the exec log: earliest start to latest end.

    `ts_min` and `ts_max` are inclusive boundaries -- a sage row whose ts
    equals either belongs to this prompt. Half-open intervals silently
    misattribute the first/last attention call of every prompt; inclusive
    is the right call for forensic reconstruction.
    """
    prompt_id: str
    ts_min: float
    ts_max: float

    def contains(self, ts: float) -> bool:
        return self.ts_min <= ts <= self.ts_max


def parse_exec_log_windows(exec_log_path: Path) -> list[PromptWindow]:
    """Build per-prompt (ts_min, ts_max) windows from an exec log.

    `start` events stamp ts_min, `end` and `error` events stamp ts_max.
    Multiple events per prompt collapse into the outer envelope.
    """
    bounds: dict[str, list[float]] = {}
    for row in load_jsonl(exec_log_path):
        prompt_id = row.get("prompt_id")
        ts = row.get("ts")
        event = row.get("event")
        if prompt_id is None or ts is None or event not in ("start", "end", "error"):
            continue
        b = bounds.setdefault(str(prompt_id), [float(ts), float(ts)])
        if ts < b[0]:
            b[0] = float(ts)
        if ts > b[1]:
            b[1] = float(ts)
    return [PromptWindow(prompt_id=pid, ts_min=lo, ts_max=hi)
            for pid, (lo, hi) in bounds.items()]


def assign_prompt_id(rows: Iterable[dict], windows: list[PromptWindow]) -> Iterator[dict]:
    """Annotate each row with the prompt_id of the window containing its ts.

    Rows whose ts falls outside every window get `prompt_id="unknown"` --
    the bucket exists so the operator sees data quality issues (sage
    trace extending past the exec log's reach, exec log started late,
    etc.) instead of silently dropping rows. First-match wins on
    overlapping windows; documented behavior, not undefined.
    """
    for row in rows:
        ts = row.get("ts")
        if ts is None:
            yield {**row, "prompt_id": "unknown"}
            continue
        annotated = dict(row)
        annotated["prompt_id"] = "unknown"
        for w in windows:
            if w.contains(float(ts)):
                annotated["prompt_id"] = w.prompt_id
                break
        yield annotated


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


def infer_kernel(effective_mode: str, *, has_mask: bool, arch: str) -> str:
    """Routing-table mirror for the consumer's call pattern (no smooth_k,
    no LSE, head_dim in {64,120,128}): masked -> fp16_triton, unmasked
    on sm89_cuda12_8 -> fp8_cuda++. Fallback for traces lacking
    `dispatched_kernel`; aggregate() prefers that field when present.
    """
    if effective_mode in ("fp16_cuda", "fp16_triton", "fp8_cuda", "fp8_cuda++"):
        return effective_mode
    if effective_mode != "auto":
        return effective_mode
    if has_mask:
        return "fp16_triton"
    if arch == "sm89_cuda12_8":
        return "fp8_cuda++"
    return effective_mode


def total_wall_us_from_sage_span(rows: Iterable[dict]) -> float | None:
    """Use `(max_ts - min_ts) * 1e6` of the sage rows themselves as a
    self-contained denominator. Returns None for fewer than 2 rows
    (a single sample doesn't bound a span -- caller decides whether to
    omit pct or fall back).
    """
    timestamps: list[float] = []
    for row in rows:
        if row.get("event") == "summary":
            continue
        ts = row.get("ts")
        if ts is None:
            continue
        timestamps.append(float(ts))
    if len(timestamps) < 2:
        return None
    return (max(timestamps) - min(timestamps)) * 1_000_000.0


def aggregate(
    rows: Iterable[dict],
    total_wall_us: float | None = None,
    *,
    arch: str | None = None,
) -> dict:
    """Group per-call samples by (effective_mode, has_mask).
    Arch precedence for kernel inference: explicit arg > per-row 'arch' field.
    """
    groups: dict[tuple[str, bool], list[float]] = {}
    fallbacks = 0
    total_calls = 0
    kernel_source_counts = {"sage_telemetry": 0, "mirror_inferred": 0, "unknown": 0}

    for row in rows:
        if row.get("event") == "summary":
            continue
        elapsed = row.get("elapsed_us")
        if elapsed is None:
            continue
        total_calls += 1
        if row.get("fell_back"):
            fallbacks += 1
        has_mask = bool(row.get("has_mask"))

        # Precedence: row['dispatched_kernel'] (real sage-fork telemetry)
        # > row['effective_mode'] + routing-table mirror via arch.
        # Empty / None in dispatched_kernel falls through.
        dispatched = row.get("dispatched_kernel")
        if dispatched:
            effective = dispatched
            kernel_source_counts["sage_telemetry"] += 1
        else:
            base_effective = row.get("effective_mode") or row.get("mode") or "?"
            row_arch = arch if arch is not None else row.get("arch")
            if row_arch is not None:
                inferred = infer_kernel(base_effective, has_mask=has_mask, arch=str(row_arch))
                if inferred != base_effective:
                    kernel_source_counts["mirror_inferred"] += 1
                else:
                    kernel_source_counts["unknown"] += 1
                effective = inferred
            else:
                effective = base_effective
                kernel_source_counts["unknown"] += 1
        key = (effective, has_mask)
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
        "kernel_source_counts": kernel_source_counts,
    }


def aggregate_per_prompt(
    rows: Iterable[dict],
    windows: list[PromptWindow],
    *,
    arch: str | None = None,
) -> dict[str, dict]:
    """Bucket sage rows by prompt_id (via ts), then aggregate each
    bucket. Per-prompt wall time = window duration in microseconds, so
    pct_of_total is honest per-prompt and not contaminated by sibling
    prompts in the same exec log.

    Output: dict mapping prompt_id -> summary (same shape as
    `aggregate()`). Includes a 'unknown' bucket for rows whose ts fell
    outside every window, with `total_wall_us=None` (no pct available
    for orphan rows).
    """
    annotated = list(assign_prompt_id(rows, windows))
    by_window = {w.prompt_id: w for w in windows}

    by_prompt: dict[str, list[dict]] = {}
    for row in annotated:
        by_prompt.setdefault(row["prompt_id"], []).append(row)

    out: dict[str, dict] = {}
    for prompt_id, prompt_rows in by_prompt.items():
        if prompt_id == "unknown":
            wall_us = None
        else:
            w = by_window[prompt_id]
            wall_us = (w.ts_max - w.ts_min) * 1_000_000.0
        out[prompt_id] = aggregate(prompt_rows, total_wall_us=wall_us, arch=arch)
    return out


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


def _format_attribution_line(summary: dict) -> str | None:
    """One-line trace-freshness signal: how many rows came from real
    sage telemetry vs mirror inference vs neither. Returns None when
    `kernel_source_counts` isn't present (legacy summaries / external
    callers that constructed a summary by hand)."""
    counts = summary.get("kernel_source_counts")
    if not counts:
        return None
    total = sum(counts.values())
    if total == 0:
        return None
    return (
        f"attribution: {counts['sage_telemetry']} sage_telemetry / "
        f"{counts['mirror_inferred']} mirror_inferred / "
        f"{counts['unknown']} unknown"
    )


def _format_table(summary: dict) -> str:
    lines: list[str] = []
    attribution = _format_attribution_line(summary)
    if attribution is not None:
        lines.append(attribution)
    lines.append(
        f"{'effective_mode':<16} {'mask':<6} {'count':>6} {'median_ms':>10} {'p90_ms':>10} "
        f"{'total_ms':>10} {'pct':>6}"
    )
    if not summary["groups"]:
        lines.append("  (no per-call samples)")
        return "\n".join(lines)
    for (mode, has_mask), entry in sorted(summary["groups"].items()):
        median_ms = entry["median_us"] / 1000.0
        p90_ms = entry["p90_us"] / 1000.0
        total_ms = entry["total_us"] / 1000.0
        pct = f"{entry.get('pct_of_total', '-')}" if "pct_of_total" in entry else "  -"
        lines.append(
            f"{mode:<16} {str(has_mask):<6} {entry['count']:>6} {median_ms:>10.3f} "
            f"{p90_ms:>10.3f} {total_ms:>10.3f} {pct:>6}"
        )
    return "\n".join(lines)


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


_DEFAULT_RUNS_DIR = Path("internal/analysis/runs")


def _latest_jsonl(subdir: str, prefix: str, runs_dir: Path = _DEFAULT_RUNS_DIR) -> Path | None:
    """Return the most recent <runs_dir>/<subdir>/<prefix>_*.jsonl, or None."""
    target = runs_dir / subdir
    if not target.is_dir():
        return None
    candidates = sorted(target.glob(f"{prefix}_*.jsonl"))
    return candidates[-1] if candidates else None


def _summary_to_jsonable(summary: dict) -> dict:
    """Tuple keys aren't JSON-encodable; flatten ('mode', has_mask) to 'mode|0|1'."""
    out = {
        "total_calls": summary["total_calls"],
        "fallback_count": summary["fallback_count"],
        "total_wall_us": summary["total_wall_us"],
        "groups": {
            f"{mode}|{int(has_mask)}": entry
            for (mode, has_mask), entry in summary["groups"].items()
        },
    }
    if "kernel_source_counts" in summary:
        out["kernel_source_counts"] = summary["kernel_source_counts"]
    return out


def _autodetect_arch() -> str | None:
    """Last-resort arch detection from the local GPU. Only meaningful
    when the summary script runs on the same host that produced the
    trace. Returns sm89_cuda12_8 if the local box matches; else None.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        cuda_version = torch.version.cuda or ""
        if (major, minor) == (8, 9):
            major_cuda = int(cuda_version.split(".")[0]) if cuda_version else 0
            minor_cuda = int(cuda_version.split(".")[1]) if "." in cuda_version else 0
            if (major_cuda, minor_cuda) >= (12, 8):
                return "sm89_cuda12_8"
    except Exception:
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sage-log", type=Path, default=None, help="Sage tracer JSONL path.")
    parser.add_argument("--exec-log", type=Path, default=None, help="Companion exec-log JSONL path.")
    parser.add_argument("--latest", action="store_true",
                        help="Auto-find the newest sage + exec JSONL under internal/analysis/runs/ "
                             "(equivalent to --sage-log internal/analysis/runs/sage/sage_*.jsonl "
                             "--exec-log internal/analysis/runs/exec_log/exec_*.jsonl, picking the "
                             "lexically-greatest match in each).")
    parser.add_argument("--total-wall-ms", type=float, default=None, help="Explicit total gen wall time in ms.")
    parser.add_argument("--use-sage-span", action="store_true",
                        help="Use (max_ts - min_ts) of sage rows as the denominator. "
                             "Self-contained; no exec log needed.")
    parser.add_argument("--arch", type=str, default=None,
                        help="GPU/CUDA arch for post-hoc kernel inference (e.g. sm89_cuda12_8). "
                             "Maps tracer 'auto' (unmasked) to the kernel sage actually dispatches. "
                             "Precedence: this flag > sage row 'arch' field > local autodetect.")
    parser.add_argument("--ksampler-class", action="append", default=None,
                        help="class_type to treat as a sampler when reading --exec-log. Can repeat.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table.")
    args = parser.parse_args(argv)

    if args.latest:
        if args.sage_log is None:
            args.sage_log = _latest_jsonl("sage", "sage")
        if args.exec_log is None:
            args.exec_log = _latest_jsonl("exec_log", "exec")

    if args.sage_log is None:
        print("error: provide --sage-log <path> or --latest", file=sys.stderr)
        return 2
    if not args.sage_log.exists():
        print(f"sage log not found: {args.sage_log}", file=sys.stderr)
        return 2

    sage_rows, malformed = load_jsonl_with_count(args.sage_log)
    if malformed > 0:
        print(f"warning: skipped {malformed} malformed line(s) in {args.sage_log}", file=sys.stderr)

    # Arch precedence: --arch > per-row 'arch' field (handled inside
    # aggregate()) > autodetect (last resort, only when --arch unset and
    # rows have no arch field).
    arch = args.arch
    if arch is None:
        rows_have_arch = any(r.get("arch") for r in sage_rows if r.get("event") != "summary")
        if not rows_have_arch:
            detected = _autodetect_arch()
            if detected is not None:
                arch = detected
                print(f"note: --arch not given; trace has no 'arch' field; "
                      f"using local autodetect: {detected}", file=sys.stderr)

    # Multi-prompt exec log -> per-prompt path. Single-prompt -> use that
    # prompt's wall window. No exec log -> --total-wall-ms or
    # --use-sage-span. Sums-of-ksampler-durations (legacy) is the last
    # resort; it's the broken denominator from before this fix.
    windows: list[PromptWindow] = []
    if args.exec_log is not None and args.exec_log.exists():
        windows = parse_exec_log_windows(args.exec_log)

    if len(windows) > 1:
        per_prompt = aggregate_per_prompt(sage_rows, windows, arch=arch)
        if args.json:
            print(orjson.dumps(
                {pid: _summary_to_jsonable(s) for pid, s in per_prompt.items()},
                option=orjson.OPT_INDENT_2,
            ).decode())
            return 0
        print(f"exec log has {len(windows)} prompt_ids; reporting per-prompt.")
        print()
        for pid in sorted(per_prompt):
            summary = per_prompt[pid]
            wall = summary["total_wall_us"]
            wall_str = f"{wall/1e6:.1f}s wall" if wall else "wall=unknown"
            print(f"prompt_id={pid}  ({summary['total_calls']} calls, {wall_str})")
            print(_format_table(summary))
            print("  gate cross-section:")
            for line in _format_gate_cross_section(summary):
                print(f"    {line}")
            print(f"  {_gate_verdict(summary)}")
            print()
        return 0

    total_wall_us: float | None = None
    if args.total_wall_ms is not None:
        total_wall_us = args.total_wall_ms * 1000.0
    elif args.use_sage_span:
        total_wall_us = total_wall_us_from_sage_span(sage_rows)
    elif len(windows) == 1:
        w = windows[0]
        total_wall_us = (w.ts_max - w.ts_min) * 1_000_000.0
    elif args.exec_log is not None and args.exec_log.exists():
        ks_classes = tuple(args.ksampler_class) if args.ksampler_class else _DEFAULT_KSAMPLER_CLASSES
        total_wall_us = total_wall_us_from_exec_log(args.exec_log, ks_classes)

    summary = aggregate(sage_rows, total_wall_us=total_wall_us, arch=arch)

    if args.json:
        print(orjson.dumps(_summary_to_jsonable(summary), option=orjson.OPT_INDENT_2).decode())
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
