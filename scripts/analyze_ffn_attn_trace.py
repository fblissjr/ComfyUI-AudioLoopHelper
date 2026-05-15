"""Aggregate per-stage FFN-vs-attention GPU-time breakdown from
ffn_attn_breakdown.jsonl traces. Stages are distinguished by T:

  - Stage-1: T <= STAGE_T_CUTOFF (smaller post-chunking video shape + audio)
  - Stage-2: T >  STAGE_T_CUTOFF (multi-guide-expanded refine shape)

Usage:
    uv run --group dev python scripts/analyze_ffn_attn_trace.py [<jsonl>]

If no path is given, picks the most recent ffn_attn_breakdown.jsonl
under data/runs/ or the internal/analysis/runs/ffn_attn/ fallback.

Output: per-stage breakdown table + per-sub-module breakdown + verdict
bucket per the v0.6 FFN-fusion decision gate.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import orjson


# Stage cutoff. T=10780 stage-1, T=44880 stage-2 per workflow audit.
STAGE_T_CUTOFF = 16384

# Categorize sub-modules.
ATTN_LABELS = frozenset({"attn1", "attn2", "audio_attn1", "audio_attn2", "video_to_audio_attn"})
FFN_LABELS = frozenset({"ff", "audio_ff"})

# LTX 2.3 distilled has 48 transformer blocks. Each sampler step pairs a
# positive + negative CFG branch -> 2 forwards. So `ff` (video FFN) fires
# 2 * 48 = 96 times per sampler step. Use `ff` alone (not summed with
# audio_ff) to infer step count: audio_ff cadence differs slightly across
# stages (audio path stays at smaller T while video expands).
NUM_TRANSFORMER_BLOCKS = 48
CFG_FORWARDS_PER_STEP = 2
FF_CALLS_PER_STEP = NUM_TRANSFORMER_BLOCKS * CFG_FORWARDS_PER_STEP  # 96


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to ffn_attn_breakdown.jsonl. Defaults to most recent.",
    )
    return p.parse_args()


def find_default_trace() -> Path | None:
    """Most recent ffn_attn_breakdown.jsonl under data/runs/."""
    candidates = sorted(Path("data/runs").rglob("ffn_attn_breakdown.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def categorize_stage(T: int | None) -> str:
    if T is None:
        return "unknown"
    return "stage-1" if T <= STAGE_T_CUTOFF else "stage-2"


def categorize_kind(label: str) -> str:
    if label in ATTN_LABELS:
        return "attn"
    if label in FFN_LABELS:
        return "ffn"
    return "other"


def main() -> int:
    args = parse_args()
    path = args.path or find_default_trace()
    if path is None or not path.exists():
        print(f"ERROR: no trace file found ({path})", file=sys.stderr)
        return 1

    print(f"Trace: {path}")

    # (stage, kind) -> sum of elapsed_ms
    totals: dict[tuple[str, str], float] = defaultdict(float)
    # (stage, kind) -> count
    counts: dict[tuple[str, str], int] = defaultdict(int)
    # (stage, label) -> elapsed values (for per-sub-module breakdown).
    per_label: dict[tuple[str, str], list[float]] = defaultdict(list)
    # (stage) -> count of canonical video `ff` calls, used to infer step count.
    ff_calls_by_stage: dict[str, int] = defaultdict(int)
    # Track distinct prompt_ids + T values for sanity.
    prompt_ids: set[str] = set()
    T_values: set[int] = set()

    with open(path, "rb") as f:
        for line in f:
            try:
                e = orjson.loads(line)
            except Exception:
                continue
            T = e.get("T")
            stage = categorize_stage(T)
            label = e.get("label", "")
            kind = categorize_kind(label)
            elapsed = e.get("elapsed_ms", 0.0)
            totals[(stage, kind)] += elapsed
            counts[(stage, kind)] += 1
            per_label[(stage, label)].append(elapsed)
            if label == "ff":
                ff_calls_by_stage[stage] += 1
            if e.get("prompt_id"):
                prompt_ids.add(e["prompt_id"])
            if T is not None:
                T_values.add(T)

    print(f"Prompt IDs: {len(prompt_ids)}")
    print(f"Distinct T values: {sorted(T_values)}")
    print()

    print("=" * 72)
    print(f"{'Stage':<10} {'FFN ms/step':>14} {'Attn ms/step':>14} {'FFN share':>12}")
    print("=" * 72)
    for stage in ("stage-1", "stage-2"):
        ffn_total = totals.get((stage, "ffn"), 0.0)
        attn_total = totals.get((stage, "attn"), 0.0)
        ffn_count = counts.get((stage, "ffn"), 0)
        attn_count = counts.get((stage, "attn"), 0)
        if ffn_count == 0 and attn_count == 0:
            continue

        # Infer sampler-step count from video `ff` calls only. `audio_ff`
        # cadence varies across stages (audio path stays at smaller T while
        # video expands) so summing both before dividing inflates the count.
        ff_calls = ff_calls_by_stage.get(stage, 0)
        if ff_calls >= FF_CALLS_PER_STEP:
            num_steps = round(ff_calls / FF_CALLS_PER_STEP)
        else:
            num_steps = 1

        ffn_per_step = ffn_total / max(num_steps, 1)
        attn_per_step = attn_total / max(num_steps, 1)
        share = ffn_total / (ffn_total + attn_total) * 100 if (ffn_total + attn_total) > 0 else 0
        print(f"{stage:<10} {ffn_per_step:>13.2f}  {attn_per_step:>13.2f}  {share:>10.1f} %    (n_steps={num_steps}, ff_calls={ff_calls}, ffn_calls={ffn_count}, attn_calls={attn_count})")
    print("=" * 72)
    print()

    print("Per-sub-module breakdown:")
    for stage in ("stage-1", "stage-2"):
        labels_this_stage = sorted({label for (st, label) in per_label.keys() if st == stage})
        if not labels_this_stage:
            continue
        stage_total = sum(sum(per_label[(stage, l)]) for l in labels_this_stage)
        print(f"  --- {stage} (total {stage_total:.1f} ms) ---")
        for label in labels_this_stage:
            vals = per_label[(stage, label)]
            if not vals:
                continue
            tot = sum(vals)
            share_of_stage = tot / stage_total * 100 if stage_total > 0 else 0
            print(f"    {label:>22s}  n={len(vals):4d}  total={tot:8.1f} ms  per-call={tot/len(vals):.3f} ms  share-of-stage={share_of_stage:5.1f}%")
    print()

    # Verdict bucket per the v0.6 gate.
    print("=" * 72)
    print("Verdict bucket (v0.6 FFN-fusion decision gate)")
    print("=" * 72)
    for stage in ("stage-1", "stage-2"):
        ffn_total = totals.get((stage, "ffn"), 0.0)
        attn_total = totals.get((stage, "attn"), 0.0)
        if ffn_total + attn_total == 0:
            continue
        share = ffn_total / (ffn_total + attn_total) * 100
        if share < 10:
            verdict = "< 10% -- branch (3) do-nothing wins. v0.6 work is dead at this stage."
        elif share < 25:
            verdict = "10-25% -- contingent on gate #2 (sage FA bench)."
        else:
            verdict = "> 25% -- real lever; branch (1) build justified."
        print(f"  {stage}: FFN share = {share:.1f}%  -> {verdict}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
