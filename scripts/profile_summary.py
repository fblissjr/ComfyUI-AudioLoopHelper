"""Produce a categorized text summary from a torch.profiler chrome trace JSON.

Can be re-run on existing traces without re-executing the workflow. Useful
for deeper analysis after the initial ProfileEnd output.

Usage:
    uv run python scripts/profile_summary.py ./profile_output/20260417_120000/trace.json
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import orjson

# Category rules: substring match against kernel/op name. First match wins.
CATEGORY_RULES = [
    ("attention", ("flash", "sage", "scaled_dot_product", "attention", "attn")),
    ("matmul", ("gemm", "_mm", "linear", "cutlass", "fp8", "int8_mm")),
    ("norm", ("layer_norm", "rms_norm", "layernorm", "rmsnorm")),
    ("rope", ("rotary", "apply_rotary", "rope")),
    ("softmax", ("softmax",)),
    ("elementwise", ("elementwise", "mul_", "add_", "div_", "sub_")),
    ("memory", ("memcpy", "memset", "copy_")),
    ("reduction", ("reduce", "sum_", "mean_")),
]


def categorize(name: str) -> str:
    name_lower = name.lower()
    for category, patterns in CATEGORY_RULES:
        for pat in patterns:
            if pat in name_lower:
                return category
    return "other"


def summarize(trace_path: Path) -> str:
    data = orjson.loads(trace_path.read_bytes())
    events = data.get("traceEvents", [])

    # Collect kernel/op events with duration
    op_totals: dict[str, float] = defaultdict(float)
    op_counts: dict[str, int] = defaultdict(int)
    op_category: dict[str, str] = {}  # cache per-name categorization
    category_totals: dict[str, float] = defaultdict(float)
    total_gpu_time = 0.0
    total_cpu_time = 0.0

    for ev in events:
        if ev.get("ph") != "X":  # only complete events
            continue
        name = ev.get("name", "")
        dur = ev.get("dur", 0)  # microseconds
        cat = ev.get("cat", "")
        cat_lower = cat.lower()

        if "kernel" in cat or "cuda" in cat_lower:
            op_totals[name] += dur
            op_counts[name] += 1
            kcat = op_category.get(name)
            if kcat is None:
                kcat = categorize(name)
                op_category[name] = kcat
            category_totals[kcat] += dur
            total_gpu_time += dur
        elif "cpu" in cat_lower or cat == "":
            total_cpu_time += dur

    lines = []
    lines.append("=" * 78)
    lines.append(f"Profile Summary: {trace_path.name}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Total GPU time (sum of kernel durations): {total_gpu_time / 1e3:.1f} ms")
    lines.append(f"Total CPU time (sum of CPU op durations): {total_cpu_time / 1e3:.1f} ms")
    lines.append("")

    lines.append("GPU time by category (percent of total GPU time)")
    lines.append("-" * 78)
    for cat, total in sorted(category_totals.items(), key=lambda x: -x[1]):
        pct = 100.0 * total / total_gpu_time if total_gpu_time else 0
        lines.append(f"  {cat:15s}  {total / 1e3:10.2f} ms  {pct:6.2f}%")
    lines.append("")

    lines.append("Top 30 kernels by cumulative time")
    lines.append("-" * 78)
    lines.append(f"  {'cum_ms':>10s}  {'calls':>6s}  {'per_call_us':>12s}  {'cat':<12s}  name")
    for name, total in sorted(op_totals.items(), key=lambda x: -x[1])[:30]:
        count = op_counts[name]
        per_call_us = total / count if count else 0
        short_name = name if len(name) <= 60 else name[:57] + "..."
        lines.append(
            f"  {total / 1e3:10.2f}  {count:6d}  {per_call_us:12.1f}  "
            f"{op_category[name]:<12s}  {short_name}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Path to trace.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output file path (default: summary.txt in the same dir as trace)",
    )
    args = parser.parse_args()

    if not args.trace.exists():
        print(f"Trace file not found: {args.trace}", file=sys.stderr)
        sys.exit(1)

    summary = summarize(args.trace)
    out_path = args.output or args.trace.with_name("summary.txt")
    out_path.write_text(summary)
    print(summary)
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
