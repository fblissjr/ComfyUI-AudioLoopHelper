"""Tests for `scripts/sage_telemetry_summary.py`: Phase 0 of the
optimization plan. The script reads a sage JSONL trace and outputs a
per-mode-and-mask summary (median, p90, count, % of total time)."""

from __future__ import annotations

import importlib

import orjson
import pytest


def _mod():
    import sage_telemetry_summary  # noqa: F401
    return importlib.reload(importlib.import_module("sage_telemetry_summary"))


# ---------------------------------------------------------------------------
# Sample row factories
# ---------------------------------------------------------------------------

def _row(*, mode: str, has_mask: bool, elapsed_us: float, effective_mode: str | None = None,
         fell_back: bool = False) -> dict:
    return {
        "ts": 1.0,
        "iter": 0,
        "shape": [1, 32, 8192, 64],
        "has_mask": has_mask,
        "mode": mode,
        "effective_mode": effective_mode if effective_mode is not None else mode,
        "fell_back": fell_back,
        "elapsed_us": elapsed_us,
    }


# ---------------------------------------------------------------------------
# 1. Aggregation core
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_summary():
    m = _mod()
    summary = m.aggregate([])
    assert summary["groups"] == {}
    assert summary["total_calls"] == 0


def test_skips_summary_event_lines():
    """The tracer emits a summary line at flush time. Aggregator must
    ignore it -- it's metadata, not a per-call sample."""
    m = _mod()
    rows = [
        _row(mode="auto", has_mask=False, elapsed_us=1000.0),
        {"ts": 2.0, "event": "summary", "total_calls": 1, "fallback_count": 0, "distinct_shapes": 1},
    ]
    summary = m.aggregate(rows)
    assert summary["total_calls"] == 1


def test_groups_by_effective_mode_and_mask():
    """Grouping uses effective_mode (the kernel that actually ran) plus
    has_mask. This is the cross-section the optimization plan needs:
    'masked-triton' is `(effective_mode='fp16_triton', has_mask=True)`,
    'unmasked-fp8++' is `(effective_mode='fp8_cuda++', has_mask=False)`."""
    m = _mod()
    rows = [
        _row(mode="auto", effective_mode="fp16_triton", has_mask=True, elapsed_us=800.0),
        _row(mode="auto", effective_mode="fp16_triton", has_mask=True, elapsed_us=900.0),
        _row(mode="auto", effective_mode="fp8_cuda++", has_mask=False, elapsed_us=20000.0),
    ]
    summary = m.aggregate(rows)
    groups = summary["groups"]
    assert ("fp16_triton", True) in groups
    assert ("fp8_cuda++", False) in groups
    assert groups[("fp16_triton", True)]["count"] == 2
    assert groups[("fp8_cuda++", False)]["count"] == 1


def test_median_and_p90_computed_per_group():
    m = _mod()
    rows = [_row(mode="auto", has_mask=False, elapsed_us=float(x)) for x in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]]
    summary = m.aggregate(rows)
    g = summary["groups"][("auto", False)]
    # statistics.median averages the two middle values for even-length
    # samples (550 here). The aggregator's contract is "use python
    # statistics.median," so just check it's the middle.
    assert g["median_us"] == 550.0
    # p90 by inclusive nearest-rank: idx = round(0.9 * (N-1)) = 8 for
    # N=10. sorted[8] = 900. (1000 would be p100.)
    assert g["p90_us"] == 900.0
    assert g["count"] == 10


def test_total_us_is_sum_of_elapsed_us_per_group():
    m = _mod()
    rows = [
        _row(mode="auto", has_mask=False, elapsed_us=100.0),
        _row(mode="auto", has_mask=False, elapsed_us=200.0),
    ]
    summary = m.aggregate(rows)
    g = summary["groups"][("auto", False)]
    assert g["total_us"] == 300.0


# ---------------------------------------------------------------------------
# 2. Total-gen-time references
# ---------------------------------------------------------------------------

def test_percent_of_total_when_total_provided():
    m = _mod()
    rows = [
        _row(mode="auto", has_mask=True, elapsed_us=500_000.0),  # 500 ms
        _row(mode="auto", has_mask=False, elapsed_us=1_500_000.0),  # 1.5 s
    ]
    # Total wall time = 10 s = 10_000_000 us.
    summary = m.aggregate(rows, total_wall_us=10_000_000.0)
    g_masked = summary["groups"][("auto", True)]
    g_unmasked = summary["groups"][("auto", False)]
    assert g_masked["pct_of_total"] == pytest.approx(5.0)
    assert g_unmasked["pct_of_total"] == pytest.approx(15.0)


def test_pct_omitted_when_total_not_provided():
    m = _mod()
    rows = [_row(mode="auto", has_mask=False, elapsed_us=100.0)]
    summary = m.aggregate(rows)
    assert "pct_of_total" not in summary["groups"][("auto", False)]


# ---------------------------------------------------------------------------
# 3. JSONL ingestion
# ---------------------------------------------------------------------------

def test_load_jsonl_skips_blank_and_malformed_lines(tmp_path):
    m = _mod()
    p = tmp_path / "trace.jsonl"
    good = orjson.dumps(_row(mode="auto", has_mask=False, elapsed_us=1.0)).decode()
    p.write_text(good + "\n\n" + "{not json}\n" + good + "\n")
    rows = list(m.load_jsonl(p))
    assert len(rows) == 2  # blank skipped, malformed skipped


# ---------------------------------------------------------------------------
# 4. Gate cross-section
# ---------------------------------------------------------------------------

def test_masked_triton_section_matches_canonical_format():
    """The aggregator surfaces 'masked_triton: median=X ms, p90=Y ms,
    count=N, %_of_total=Z%' under the canonical (effective_mode, has_mask)
    key. This row is the gate input for further mask-kernel work decisions."""
    m = _mod()
    rows = [
        _row(mode="auto", effective_mode="fp16_triton", has_mask=True, elapsed_us=800.0),
        _row(mode="auto", effective_mode="fp16_triton", has_mask=True, elapsed_us=900.0),
        _row(mode="auto", effective_mode="fp16_triton", has_mask=True, elapsed_us=1100.0),
    ]
    summary = m.aggregate(rows, total_wall_us=100_000.0)
    section = m.gate_section(summary, effective_mode="fp16_triton", has_mask=True)
    # Section is a dict ready for JSON output or pretty-printing.
    assert section["count"] == 3
    assert section["median_us"] == 900.0
    assert section["pct_of_total"] == pytest.approx(2.8)
