"""Render an HTML visualization of a bench aggregate JSON.

Reads the per-shape masked/unmasked elapsed_us aggregate produced by the
bench pipeline (e.g. data/runs/bench_profile/fml2v_aggregate_*.json) and
writes a self-contained HTML next to it. No external libs — inline SVG.

Usage:
    uv run python scripts/viz_bench_aggregate.py <aggregate.json> [-o out.html]
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import orjson


def _load(path: Path) -> dict:
    return orjson.loads(path.read_bytes())


def _fmt_us(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 1000:
        return f"{v / 1000:.2f} ms"
    return f"{v:.0f} us"


def _bar(label: str, value: float, vmax: float, color: str, width: int = 420) -> str:
    w = max(2, int((value / vmax) * width)) if vmax > 0 else 0
    return (
        f'<div class="bar-row">'
        f'<span class="bar-label">{html.escape(label)}</span>'
        f'<span class="bar-track"><span class="bar-fill" style="width:{w}px;background:{color}"></span></span>'
        f'<span class="bar-value">{_fmt_us(value)}</span>'
        f"</div>"
    )


def render(agg: dict) -> str:
    shapes = agg["per_shape_aggregate"]
    totals = agg["totals_across_runs"]
    per_run = agg["per_run_summary"]
    n_runs = agg["n_runs"]

    all_vals = []
    for s in shapes.values():
        for k in ("masked_p50_us", "masked_p95_us", "unmasked_p50_us"):
            v = s.get(k)
            if v is not None:
                all_vals.append(v)
    vmax = max(all_vals) if all_vals else 1.0

    shape_blocks = []
    for shape, s in shapes.items():
        rows = []
        if s.get("masked_p50_us") is not None:
            rows.append(_bar(f"masked p50 (n={s['n_masked']})", s["masked_p50_us"], vmax, "#c0392b"))
        if s.get("masked_p95_us") is not None:
            rows.append(_bar("masked p95", s["masked_p95_us"], vmax, "#e74c3c"))
        if s.get("unmasked_p50_us") is not None:
            rows.append(_bar(f"unmasked p50 (n={s['n_unmasked']})", s["unmasked_p50_us"], vmax, "#27ae60"))
        else:
            rows.append('<div class="bar-row na">unmasked p50 — no dispatches</div>')

        ratio = ""
        if s.get("masked_p50_us") and s.get("unmasked_p50_us"):
            r = s["masked_p50_us"] / s["unmasked_p50_us"]
            ratio = f'<div class="ratio">masked / unmasked p50 = <b>{r:.2f}×</b></div>'

        shape_blocks.append(
            f'<section class="shape">'
            f"<h3>shape {html.escape(shape)}</h3>"
            + "".join(rows)
            + ratio
            + "</section>"
        )

    run_rows = []
    for r in per_run:
        notes = f' <span class="notes">{html.escape(r.get("notes", ""))}</span>' if r.get("notes") else ""
        run_rows.append(
            f"<tr>"
            f'<td class="mono">{html.escape(r["prompt_id"][:8])}</td>'
            f"<td>{r['ran']}</td>"
            f"<td>{r['masked']}</td>"
            f"<td>{r['fallbacks']}</td>"
            f"<td>{', '.join(f'{k}={v}' for k, v in r['kernels'].items())}{notes}</td>"
            f"</tr>"
        )

    desc = html.escape(agg.get("description", ""))
    workflow = html.escape(agg.get("workflow", ""))
    fork = html.escape(agg.get("fork_version", ""))
    hw = html.escape(agg.get("hardware", ""))
    res = html.escape(agg.get("resolution", ""))
    notes = html.escape(agg.get("notes", ""))

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>FML2V bench aggregate — {html.escape(agg['last_updated'])}</title>
<style>
  body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif; max-width: 880px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ margin-bottom: 0.2em; }}
  h1 .date {{ font-weight: normal; color: #888; font-size: 0.7em; }}
  .meta {{ color: #555; font-size: 0.92em; margin-bottom: 1.6em; }}
  .meta b {{ color: #222; }}
  section.shape {{ margin: 1.4em 0; padding: 0.8em 1em; background: #f7f7f7; border-radius: 6px; }}
  section.shape h3 {{ margin: 0 0 0.6em 0; font-size: 1em; font-family: monospace; color: #333; }}
  .bar-row {{ display: flex; align-items: center; margin: 4px 0; }}
  .bar-label {{ width: 220px; font-size: 0.88em; color: #444; }}
  .bar-track {{ flex: 0 0 420px; height: 18px; background: #e0e0e0; border-radius: 2px; overflow: hidden; }}
  .bar-fill {{ display: block; height: 100%; }}
  .bar-value {{ margin-left: 10px; font-family: monospace; font-size: 0.88em; color: #333; min-width: 70px; }}
  .bar-row.na {{ color: #999; font-style: italic; padding-left: 220px; }}
  .ratio {{ margin-top: 0.6em; font-size: 0.9em; color: #555; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.88em; margin-top: 0.6em; }}
  th, td {{ text-align: left; padding: 4px 8px; border-bottom: 1px solid #eee; }}
  th {{ background: #fafafa; }}
  td.mono {{ font-family: monospace; }}
  .notes {{ color: #b85; font-size: 0.9em; }}
  .totals {{ display: flex; gap: 1.5em; margin: 1em 0; }}
  .totals .stat {{ background: #eef; padding: 0.6em 1em; border-radius: 4px; }}
  .totals .stat b {{ display: block; font-size: 1.4em; }}
  .footer {{ margin-top: 2em; font-size: 0.85em; color: #999; border-top: 1px solid #eee; padding-top: 1em; }}
</style></head>
<body>
<h1>FML2V bench aggregate <span class="date">{html.escape(agg['last_updated'])}</span></h1>
<div class="meta">
  {desc}<br>
  <b>workflow</b> {workflow} &nbsp; <b>hw</b> {hw} &nbsp; <b>res</b> {res}<br>
  <b>fork</b> {fork} &nbsp; <b>runs</b> {n_runs}
</div>

<h2>Per-shape latency (elapsed_us)</h2>
{''.join(shape_blocks)}

<h2>Totals across runs</h2>
<div class="totals">
  <div class="stat"><b>{totals['masked_dispatches']:,}</b>masked dispatches</div>
  <div class="stat"><b>{totals['unmasked_dispatches']:,}</b>unmasked dispatches</div>
  <div class="stat"><b>{totals['fallback_events']}</b>fallback events</div>
</div>

<h2>Per-run summary</h2>
<table>
  <thead><tr><th>prompt</th><th>ran</th><th>masked</th><th>fallbacks</th><th>kernels</th></tr></thead>
  <tbody>{''.join(run_rows)}</tbody>
</table>

<div class="footer">{notes}</div>
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("aggregate", type=Path, help="Path to aggregate JSON.")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output HTML path (default: alongside input).")
    args = ap.parse_args()

    agg = _load(args.aggregate)
    out = args.output or args.aggregate.with_suffix(".html")
    out.write_text(render(agg), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
