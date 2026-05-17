"""Tests for `scripts/cleanup_traces.py` retention + per-module sidecar extraction.

`cleanup_traces.py --apply` walks each to-be-deleted RUN_ID dir for raw
`torch_profile.*.json` chrome traces and writes a
`<stem>.modules_summary.json` sidecar (via
`analyze_torch_profile.aggregate_by_module`) BEFORE `shutil.rmtree`. The
sidecar carries per-module aten-op data so retention doesn't destroy the
breakdown that the analyzer's `--modules` filter would produce on the
raw trace. `--no-extract` is the escape hatch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import orjson
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


# `tests/conftest.py` puts both the repo root and `scripts/` on sys.path
# at collection time. Import directly.
import cleanup_traces  # noqa: E402


MODULE_NAME = "audio_attn1/block_0"


def _build_synthetic_chrome_trace(path: Path) -> None:
    """Tiny chrome-trace JSON with one `record_function` span + two
    nested aten ops. Mirrors the shape `tracers/ffn_attn.py` produces
    in real audit renders.
    """
    events = [
        {
            "ph": "X", "cat": "user_annotation", "name": MODULE_NAME,
            "ts": 100.0, "dur": 50.0, "pid": 1, "tid": 2,
        },
        {
            "ph": "X", "cat": "cpu_op", "name": "aten::linear",
            "ts": 110.0, "dur": 20.0, "pid": 1, "tid": 2,
            "args": {"Input Dims": [[1, 100, 2048], [2048, 2048]]},
        },
        {
            "ph": "X", "cat": "cpu_op", "name": "aten::matmul",
            "ts": 135.0, "dur": 10.0, "pid": 1, "tid": 2,
            "args": {"Input Dims": [[1, 100, 2048], [2048, 2048]]},
        },
    ]
    path.write_bytes(orjson.dumps({"traceEvents": events}))


@pytest.fixture
def fake_runs_dir(tmp_path: Path) -> Path:
    """A tmp_path-backed mock `data/runs/` with two RUN_IDs.

    `keep_me/`: newest mtime → survives `--keep 1`.
    `drop_me/<prompt>/`: oldest mtime, holds a synthetic chrome trace → dropped.
    """
    runs = tmp_path / "runs"
    runs.mkdir()

    drop_me = runs / "drop_me"
    prompt_dir = drop_me / "prompt-uuid"
    prompt_dir.mkdir(parents=True)
    _build_synthetic_chrome_trace(prompt_dir / "torch_profile.0.json")

    old = 1_000_000.0
    os.utime(prompt_dir / "torch_profile.0.json", (old, old))
    os.utime(prompt_dir, (old, old))
    os.utime(drop_me, (old, old))

    (runs / "keep_me").mkdir()
    (runs / "keep_me" / ".marker").write_text("keep")
    return runs


def _run_cleanup(runs_dir: Path, extra: list[str], monkeypatch, capsys) -> str:
    """Invoke `cleanup_traces.main()` in-process via argparse override.

    Avoids subprocess startup overhead and gives capsys-native stderr
    capture. Returns the captured stderr for assertion.
    """
    argv = [
        "cleanup_traces.py",
        "--runs-dir", str(runs_dir),
        "--keep", "1",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cleanup_traces.main()
    return capsys.readouterr().err + capsys.readouterr().out


def test_apply_extracts_sidecar_before_delete(fake_runs_dir, monkeypatch, capsys):
    """`--apply` must produce `<stem>.modules_summary.json` before rm-treeing
    the drop directory. Verified via the `[extract]` log line in stderr."""
    out = _run_cleanup(fake_runs_dir, ["--apply"], monkeypatch, capsys)
    assert not (fake_runs_dir / "drop_me").exists()
    assert (fake_runs_dir / "keep_me").exists()
    assert "[extract]" in out
    assert "torch_profile.0.json -> torch_profile.0.modules_summary.json" in out


def test_no_extract_flag_skips_sidecar_creation(fake_runs_dir, monkeypatch, capsys):
    """`--no-extract` is the escape hatch; deletion still proceeds."""
    out = _run_cleanup(fake_runs_dir, ["--apply", "--no-extract"], monkeypatch, capsys)
    assert not (fake_runs_dir / "drop_me").exists()
    assert "[extract]" not in out


def test_dry_run_does_not_extract_or_delete(fake_runs_dir, monkeypatch, capsys):
    """Dry-run leaves everything in place — extraction is gated behind
    `--apply` so a preview-mode invocation doesn't parse multi-GB traces."""
    out = _run_cleanup(fake_runs_dir, [], monkeypatch, capsys)
    assert (fake_runs_dir / "drop_me").exists()
    assert not (fake_runs_dir / "drop_me" / "prompt-uuid" / "torch_profile.0.modules_summary.json").exists()
    assert "[extract]" not in out


def test_extract_module_summary_is_idempotent(tmp_path):
    """Running extraction twice produces no second write."""
    trace_path = tmp_path / "torch_profile.0.json"
    _build_synthetic_chrome_trace(trace_path)

    out1 = cleanup_traces.extract_module_summary(trace_path)
    assert out1 is not None
    assert out1.exists()
    mtime1 = out1.stat().st_mtime

    out2 = cleanup_traces.extract_module_summary(trace_path)
    assert out2 == out1
    assert out2.stat().st_mtime == mtime1


def test_extracted_sidecar_attributes_aten_ops_to_module(tmp_path):
    """The sidecar's `modules` dict must contain the annotation name with
    the two nested aten ops grouped under it. Catches regressions in the
    bisect span-lookup attribution.
    """
    trace_path = tmp_path / "torch_profile.0.json"
    _build_synthetic_chrome_trace(trace_path)

    sidecar = cleanup_traces.extract_module_summary(trace_path)
    assert sidecar is not None
    payload = orjson.loads(sidecar.read_bytes())

    assert payload["source_trace"] == str(trace_path)
    assert payload["total_events"] == 3

    block_ops = payload["modules"][MODULE_NAME]
    assert "aten::linear" in block_ops
    assert "aten::matmul" in block_ops
    assert block_ops["aten::linear"]["count"] == 1
    assert block_ops["aten::linear"]["total_us"] == 20.0
    assert isinstance(block_ops["aten::linear"]["shapes"], list)
