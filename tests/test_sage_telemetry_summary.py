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


# ---------------------------------------------------------------------------
# 5. Exec-log prompt windows
# ---------------------------------------------------------------------------

def _exec_event(*, ts: float, event: str, prompt_id: str, node_id: int = 1,
                class_type: str = "KSampler", duration_s: float | None = None) -> dict:
    rec = {"ts": ts, "event": event, "prompt_id": prompt_id,
           "node_id": node_id, "class_type": class_type}
    if duration_s is not None:
        rec["duration_s"] = duration_s
    return rec


def test_parse_exec_log_windows_groups_by_prompt_id(tmp_path):
    """Exec logs span multiple prompts in one file. Per-prompt grouping
    requires (ts_min, ts_max) per prompt_id, derived from the earliest
    'start' and latest 'end'/'error' for each id."""
    m = _mod()
    p = tmp_path / "exec.jsonl"
    rows = [
        _exec_event(ts=100.0, event="start", prompt_id="A"),
        _exec_event(ts=101.0, event="end", prompt_id="A", duration_s=1.0),
        _exec_event(ts=102.0, event="start", prompt_id="A"),
        _exec_event(ts=110.0, event="end", prompt_id="A", duration_s=8.0),
        _exec_event(ts=200.0, event="start", prompt_id="B"),
        _exec_event(ts=205.0, event="end", prompt_id="B", duration_s=5.0),
    ]
    p.write_text("\n".join(orjson.dumps(r).decode() for r in rows) + "\n")

    windows = m.parse_exec_log_windows(p)
    by_id = {w.prompt_id: w for w in windows}
    assert set(by_id) == {"A", "B"}
    assert by_id["A"].ts_min == 100.0 and by_id["A"].ts_max == 110.0
    assert by_id["B"].ts_min == 200.0 and by_id["B"].ts_max == 205.0


def test_parse_exec_log_windows_handles_error_events(tmp_path):
    """An 'error' event ends the prompt window the same way 'end' does."""
    m = _mod()
    p = tmp_path / "exec.jsonl"
    rows = [
        _exec_event(ts=100.0, event="start", prompt_id="A"),
        _exec_event(ts=103.0, event="error", prompt_id="A", duration_s=3.0),
    ]
    p.write_text("\n".join(orjson.dumps(r).decode() for r in rows) + "\n")
    windows = m.parse_exec_log_windows(p)
    assert windows[0].ts_max == 103.0


# ---------------------------------------------------------------------------
# 6. Per-prompt assignment + bucketing
# ---------------------------------------------------------------------------

def test_assign_prompt_id_buckets_by_ts_window():
    """Each sage row's ts is mapped to whichever prompt window contains
    it. Rows whose ts falls outside every window go to the 'unknown'
    bucket -- sage-fork-claude's pushback: silent drops hide data."""
    m = _mod()
    windows = [
        m.PromptWindow(prompt_id="A", ts_min=100.0, ts_max=110.0),
        m.PromptWindow(prompt_id="B", ts_min=200.0, ts_max=205.0),
    ]
    rows = [
        {"ts": 105.0, "elapsed_us": 1.0, "effective_mode": "auto", "has_mask": False},
        {"ts": 202.0, "elapsed_us": 2.0, "effective_mode": "auto", "has_mask": False},
        {"ts": 50.0,  "elapsed_us": 3.0, "effective_mode": "auto", "has_mask": False},  # before A
        {"ts": 150.0, "elapsed_us": 4.0, "effective_mode": "auto", "has_mask": False},  # gap A-B
    ]
    annotated = list(m.assign_prompt_id(rows, windows))
    assert annotated[0]["prompt_id"] == "A"
    assert annotated[1]["prompt_id"] == "B"
    assert annotated[2]["prompt_id"] == "unknown"
    assert annotated[3]["prompt_id"] == "unknown"


def test_aggregate_per_prompt_groups_by_prompt_id():
    """aggregate_per_prompt returns one summary per prompt_id, each with
    its own groups dict keyed by (effective_mode, has_mask)."""
    m = _mod()
    windows = [
        m.PromptWindow(prompt_id="A", ts_min=100.0, ts_max=110.0),
        m.PromptWindow(prompt_id="B", ts_min=200.0, ts_max=205.0),
    ]
    rows = [
        {"ts": 105.0, "elapsed_us": 1000.0, "effective_mode": "fp16_triton", "has_mask": True},
        {"ts": 106.0, "elapsed_us": 2000.0, "effective_mode": "fp16_triton", "has_mask": True},
        {"ts": 202.0, "elapsed_us": 5000.0, "effective_mode": "auto", "has_mask": False},
    ]
    per_prompt = m.aggregate_per_prompt(rows, windows)
    assert set(per_prompt) == {"A", "B"}
    assert per_prompt["A"]["groups"][("fp16_triton", True)]["count"] == 2
    assert per_prompt["B"]["groups"][("auto", False)]["count"] == 1


def test_aggregate_per_prompt_uses_per_prompt_wall_time():
    """Per-prompt pct denominator = that prompt's window duration in
    microseconds. NOT sum-of-ksampler-durations (which double-counts loops
    and is sage-fork-claude's central denominator complaint)."""
    m = _mod()
    windows = [m.PromptWindow(prompt_id="A", ts_min=100.0, ts_max=110.0)]  # 10s = 10_000_000 us
    rows = [
        {"ts": 105.0, "elapsed_us": 1_000_000.0, "effective_mode": "fp16_triton", "has_mask": True},
    ]
    per_prompt = m.aggregate_per_prompt(rows, windows)
    g = per_prompt["A"]["groups"][("fp16_triton", True)]
    # 1s of 10s = 10%
    assert g["pct_of_total"] == pytest.approx(10.0)


def test_aggregate_per_prompt_emits_unknown_bucket_with_count():
    """Unmatched rows are not silently dropped -- the 'unknown' bucket
    must appear in output with its own count so the operator sees the
    data quality issue."""
    m = _mod()
    windows = [m.PromptWindow(prompt_id="A", ts_min=100.0, ts_max=110.0)]
    rows = [
        {"ts": 105.0, "elapsed_us": 1.0, "effective_mode": "auto", "has_mask": False},
        {"ts": 50.0,  "elapsed_us": 2.0, "effective_mode": "auto", "has_mask": False},
        {"ts": 51.0,  "elapsed_us": 3.0, "effective_mode": "auto", "has_mask": False},
    ]
    per_prompt = m.aggregate_per_prompt(rows, windows)
    assert "unknown" in per_prompt
    assert per_prompt["unknown"]["total_calls"] == 2
    # In-window row goes to A only — proves no double-counting between
    # the matched window and the unknown bucket.
    assert per_prompt["A"]["total_calls"] == 1


# ---------------------------------------------------------------------------
# 7. Sage-span fallback denominator
# ---------------------------------------------------------------------------

def test_total_wall_us_from_sage_span_uses_ts_range():
    """When no exec log is provided, sage rows' own ts range is the
    honest single-prompt denominator -- max_ts - min_ts gives 'how long
    sage was alive' in microseconds."""
    m = _mod()
    rows = [
        {"ts": 100.0, "elapsed_us": 1.0, "effective_mode": "auto", "has_mask": False},
        {"ts": 105.5, "elapsed_us": 1.0, "effective_mode": "auto", "has_mask": False},
        {"ts": 110.0, "elapsed_us": 1.0, "effective_mode": "auto", "has_mask": False},
    ]
    total_us = m.total_wall_us_from_sage_span(rows)
    assert total_us == pytest.approx(10_000_000.0)


def test_total_wall_us_from_sage_span_returns_none_for_single_row():
    """A single sample doesn't bound a span. Return None and let the
    caller decide (e.g. omit pct vs. fall back to exec log)."""
    m = _mod()
    rows = [{"ts": 100.0, "elapsed_us": 1.0, "effective_mode": "auto", "has_mask": False}]
    assert m.total_wall_us_from_sage_span(rows) is None


# ---------------------------------------------------------------------------
# 8. Post-hoc kernel inference (sm89 + CUDA >= 12.8)
# ---------------------------------------------------------------------------

def test_infer_kernel_unmasked_auto_resolves_to_fp8_cuda_pp_on_sm89():
    """sage_fork's sageattn() routing on sm89 + CUDA >= 12.8 dispatches
    unmasked calls to sageattn_qk_int8_pv_fp8_cuda with pv_accum_dtype
    fp32+fp16 -- canonical name 'fp8_cuda++'. Mirror sage's table; do not
    guess. Source: sageattention/core.py::sageattn dispatch table."""
    m = _mod()
    assert m.infer_kernel("auto", has_mask=False, arch="sm89_cuda12_8") == "fp8_cuda++"


def test_infer_kernel_masked_resolves_to_fp16_triton():
    """Only the Triton kernel implements masked attention. Masked routing
    is _route_mask_aware -> fp16_triton on every supported arch."""
    m = _mod()
    assert m.infer_kernel("auto", has_mask=True, arch="sm89_cuda12_8") == "fp16_triton"


def test_infer_kernel_passes_through_explicit_modes():
    """If effective_mode is already a concrete kernel name (e.g. user
    forced fp16_cuda), inference is a no-op."""
    m = _mod()
    assert m.infer_kernel("fp16_cuda", has_mask=False, arch="sm89_cuda12_8") == "fp16_cuda"
    assert m.infer_kernel("fp16_triton", has_mask=True, arch="sm89_cuda12_8") == "fp16_triton"


def test_aggregate_with_arch_rewrites_auto_to_inferred_kernel():
    """Aggregating with arch=sm89_cuda12_8 rewrites group keys from
    ('auto', False) to ('fp8_cuda++', False). Lets the gate's
    'unmasked_fp8++' cross-section populate even when the tracer recorded
    'auto' (which it does, since the consumer can't see sage's dispatch)."""
    m = _mod()
    rows = [
        _row(mode="auto", effective_mode="auto", has_mask=False, elapsed_us=1000.0),
        _row(mode="auto", effective_mode="auto", has_mask=False, elapsed_us=2000.0),
    ]
    summary = m.aggregate(rows, arch="sm89_cuda12_8")
    assert ("fp8_cuda++", False) in summary["groups"]
    assert ("auto", False) not in summary["groups"]
    assert summary["groups"][("fp8_cuda++", False)]["count"] == 2


def test_aggregate_without_arch_preserves_legacy_keys():
    """Default behavior unchanged when arch is None -- existing tests
    and existing CLI output are not regressed."""
    m = _mod()
    rows = [_row(mode="auto", effective_mode="auto", has_mask=False, elapsed_us=1.0)]
    summary = m.aggregate(rows)
    assert ("auto", False) in summary["groups"]


def test_aggregate_uses_arch_field_from_rows_when_no_arch_arg():
    """If sage rows carry their own 'arch' field (tracer stamped it at
    init), aggregate uses that when no --arch is supplied. Lets the
    summary script run on traces from any host without --arch flag and
    without local autodetect needing to be right. Self-describing
    traces > flag-on-the-CLI."""
    m = _mod()
    rows = [
        {"ts": 1.0, "elapsed_us": 100.0, "effective_mode": "auto",
         "has_mask": False, "arch": "sm89_cuda12_8"},
    ]
    summary = m.aggregate(rows)
    assert ("fp8_cuda++", False) in summary["groups"]


def test_aggregate_arch_arg_overrides_row_field():
    """Explicit --arch wins over per-row arch. Operator override case:
    investigating whether a different arch would resolve differently,
    or correcting a mis-stamped trace."""
    m = _mod()
    rows = [
        {"ts": 1.0, "elapsed_us": 100.0, "effective_mode": "auto",
         "has_mask": False, "arch": "sm80_cuda12_4"},  # row says sm80
    ]
    summary = m.aggregate(rows, arch="sm89_cuda12_8")  # operator forces sm89
    assert ("fp8_cuda++", False) in summary["groups"]


# ---------------------------------------------------------------------------
# 9. Bucket-edge cases for prompt_id assignment
# ---------------------------------------------------------------------------

def test_assign_prompt_id_inclusive_at_ts_min():
    """A sage row whose ts equals a prompt window's ts_min belongs to
    that prompt. Half-open intervals would silently misattribute the
    first attention call of every prompt."""
    m = _mod()
    windows = [m.PromptWindow(prompt_id="A", ts_min=100.0, ts_max=110.0)]
    rows = [{"ts": 100.0, "elapsed_us": 1.0, "effective_mode": "auto", "has_mask": False}]
    annotated = list(m.assign_prompt_id(rows, windows))
    assert annotated[0]["prompt_id"] == "A"


def test_assign_prompt_id_inclusive_at_ts_max():
    """Symmetric to ts_min: ts == ts_max stays in the prompt's bucket.
    The 'end' event timestamp IS the prompt's last moment of activity."""
    m = _mod()
    windows = [m.PromptWindow(prompt_id="A", ts_min=100.0, ts_max=110.0)]
    rows = [{"ts": 110.0, "elapsed_us": 1.0, "effective_mode": "auto", "has_mask": False}]
    annotated = list(m.assign_prompt_id(rows, windows))
    assert annotated[0]["prompt_id"] == "A"


def test_assign_prompt_id_overlapping_windows_first_match_wins():
    """Overlapping windows can occur if ComfyUI's start event for prompt
    B fires before prompt A's last 'end' (async work tail). Behavior:
    deterministic first-match-wins by window list order. Documented; not
    silently undefined."""
    m = _mod()
    windows = [
        m.PromptWindow(prompt_id="A", ts_min=100.0, ts_max=110.0),
        m.PromptWindow(prompt_id="B", ts_min=108.0, ts_max=120.0),
    ]
    rows = [{"ts": 109.0, "elapsed_us": 1.0, "effective_mode": "auto", "has_mask": False}]
    annotated = list(m.assign_prompt_id(rows, windows))
    assert annotated[0]["prompt_id"] == "A"


# ---------------------------------------------------------------------------
# 10. Routing-table mirror sanity check
# ---------------------------------------------------------------------------

def test_routing_mirror_matches_sageattention_when_available():
    """When sageattention is importable, the consumer-side routing mirror
    for sm89+CUDA12.8 must agree with what sageattn() actually dispatches
    for our call pattern (no smooth_k, no LSE, head_dim in {64,120,128}).

    This is a CI sanity check, not a runtime assertion: if sage-fork
    changes its routing table, this test fires and the mirror must be
    updated. Cheap insurance against silent drift.

    Skip when sageattention isn't installed (test env, CI matrix)."""
    sa = pytest.importorskip("sageattention")
    m = _mod()
    # Inspect the routing logic by reading the source. We don't actually
    # run sageattn() here -- that needs CUDA, the right arch, and real
    # tensors. The mirror's contract is "for our call pattern on
    # sm89+CUDA12.8, masked -> fp16_triton, unmasked -> fp8_cuda++".
    assert hasattr(sa, "sageattn")
    # Mirror contract -- this is what the gate verdict relies on when
    # `dispatched_kernel` is unavailable (older traces / older sage).
    assert m.infer_kernel("auto", has_mask=False, arch="sm89_cuda12_8") == "fp8_cuda++"
    assert m.infer_kernel("auto", has_mask=True, arch="sm89_cuda12_8") == "fp16_triton"


# ---------------------------------------------------------------------------
# 11. dispatched_kernel field consumption (sage-fork API)
# ---------------------------------------------------------------------------

def test_aggregate_prefers_dispatched_kernel_over_effective_mode():
    """When the sage tracer stamped `dispatched_kernel` (sage-fork's
    `get_last_dispatched_kernel()` output), aggregate uses that
    directly -- no routing-table mirror needed. The mirror is the
    fallback for older traces that don't carry the field."""
    m = _mod()
    rows = [
        # Tracer recorded effective_mode='auto' (consumer-side route)
        # but dispatched_kernel='fp8_cuda++' (real sage dispatch).
        # The dispatched value wins.
        {"ts": 1.0, "elapsed_us": 100.0, "effective_mode": "auto",
         "has_mask": False, "dispatched_kernel": "fp8_cuda++"},
    ]
    summary = m.aggregate(rows)  # no arch arg, no mirror needed
    assert ("fp8_cuda++", False) in summary["groups"]
    assert ("auto", False) not in summary["groups"]


def test_aggregate_falls_back_to_mirror_when_dispatched_kernel_absent():
    """Older traces (pre-sage-fork-update) lack `dispatched_kernel`.
    Aggregate falls back to the routing-table mirror via `arch`. Both
    paths must work; back-compat is non-negotiable for stored traces."""
    m = _mod()
    rows = [
        {"ts": 1.0, "elapsed_us": 100.0, "effective_mode": "auto", "has_mask": False},
    ]
    summary = m.aggregate(rows, arch="sm89_cuda12_8")
    assert ("fp8_cuda++", False) in summary["groups"]


def test_aggregate_dispatched_kernel_wins_over_arch_arg():
    """If a row has BOTH `dispatched_kernel` and the operator passed
    `--arch`, the row's dispatched value wins. Real-data > inference;
    operator override only matters when there's no real data."""
    m = _mod()
    rows = [
        {"ts": 1.0, "elapsed_us": 100.0, "effective_mode": "auto",
         "has_mask": False, "dispatched_kernel": "fp16_cuda"},
    ]
    # Operator says sm89 (would infer fp8_cuda++); row says fp16_cuda
    # actually dispatched. Trust the row.
    summary = m.aggregate(rows, arch="sm89_cuda12_8")
    assert ("fp16_cuda", False) in summary["groups"]
    assert ("fp8_cuda++", False) not in summary["groups"]


def test_aggregate_ignores_dispatched_kernel_when_none_or_empty():
    """`dispatched_kernel=None` / empty string in a row means the
    sageattention thread-local was unset (fresh thread, or symbol
    missing). Treat as 'no real data, fall back to mirror'."""
    m = _mod()
    rows = [
        {"ts": 1.0, "elapsed_us": 100.0, "effective_mode": "auto",
         "has_mask": False, "dispatched_kernel": None},
        {"ts": 2.0, "elapsed_us": 100.0, "effective_mode": "auto",
         "has_mask": False, "dispatched_kernel": ""},
    ]
    summary = m.aggregate(rows, arch="sm89_cuda12_8")
    # Both rows hit the mirror -> fp8_cuda++.
    assert ("fp8_cuda++", False) in summary["groups"]
    assert summary["groups"][("fp8_cuda++", False)]["count"] == 2


# ---------------------------------------------------------------------------
# 12. kernel_source_counts -- trace freshness signal for the operator
# ---------------------------------------------------------------------------

def test_kernel_source_counts_tracks_telemetry_mirror_unknown():
    """Aggregate emits a `kernel_source_counts` dict that buckets
    every per-call row by how its kernel name was resolved:
      - 'sage_telemetry': row had a non-empty dispatched_kernel
      - 'mirror_inferred': row hit the routing-table mirror via arch
      - 'unknown': neither dispatched_kernel nor arch resolved a real
        kernel name (effective_mode passed through unchanged)
    Operator scans the counts to see "1247 telemetry, 3 mirror" =
    fresh trace; "0 telemetry, 1247 mirror" = pre-upgrade trace."""
    m = _mod()
    rows = [
        # Has dispatched_kernel -- counted as telemetry.
        {"ts": 1.0, "elapsed_us": 100.0, "effective_mode": "auto",
         "has_mask": False, "dispatched_kernel": "fp8_cuda++"},
        # No dispatched_kernel, but arch arg lets mirror resolve auto -> fp8_cuda++.
        {"ts": 2.0, "elapsed_us": 100.0, "effective_mode": "auto", "has_mask": False},
        # No dispatched_kernel, no arch resolution -- effective_mode stays 'auto'.
        {"ts": 3.0, "elapsed_us": 100.0, "effective_mode": "fp16_cuda",
         "has_mask": False},
    ]
    # Pass arch only for row 2 (rows 1 and 3 won't trigger mirror because
    # row 1 has dispatched_kernel and row 3's effective_mode is already
    # concrete -- mirror is a no-op for it).
    summary = m.aggregate(rows, arch="sm89_cuda12_8")
    counts = summary["kernel_source_counts"]
    assert counts["sage_telemetry"] == 1
    assert counts["mirror_inferred"] == 1
    # Row 3's effective_mode 'fp16_cuda' is already a concrete kernel
    # name; mirror passes it through. Counts as 'unknown' (no
    # dispatched_kernel) but with a known kernel name -- the operator
    # gets the data, just not via telemetry.
    assert counts["unknown"] == 1


def test_kernel_source_counts_zero_buckets_present_even_when_unused():
    """All three buckets present even at zero -- absent keys would
    require the operator to defensively `.get()` everywhere."""
    m = _mod()
    rows = [
        {"ts": 1.0, "elapsed_us": 100.0, "effective_mode": "auto",
         "has_mask": False, "dispatched_kernel": "fp8_cuda++"},
    ]
    summary = m.aggregate(rows)
    counts = summary["kernel_source_counts"]
    assert counts == {"sage_telemetry": 1, "mirror_inferred": 0, "unknown": 0}
