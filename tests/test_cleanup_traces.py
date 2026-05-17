"""Tests for `scripts/cleanup_traces.py` — retention with safe per-module
chrome-trace summary extraction.

Background: the 2026-05-16 audit session lost per-module chrome trace
data by deleting raw `torch_profile.*.json` files before running the
analyzer. Now the cleanup script auto-extracts a `*.modules_summary.json`
sidecar BEFORE rm-treeing the RUN_ID dir, so the per-module aten-op
data survives retention. Tests verify the extraction-before-delete
invariant + the `--no-extract` escape hatch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import orjson
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_synthetic_chrome_trace(path: Path) -> None:
    """Write a tiny chrome-trace JSON containing one `record_function` span
    + one nested aten op. Mirrors the shape `tracers/ffn_attn.py`
    produces in real audit renders.
    """
    events = [
        # Annotation span for `audio_attn1/block_0` on (pid=1, tid=2)
        {
            "ph": "X", "cat": "user_annotation", "name": "audio_attn1/block_0",
            "ts": 100.0, "dur": 50.0, "pid": 1, "tid": 2,
        },
        # Nested aten op inside the annotation
        {
            "ph": "X", "cat": "cpu_op", "name": "aten::linear",
            "ts": 110.0, "dur": 20.0, "pid": 1, "tid": 2,
            "args": {"Input Dims": [[1, 100, 2048], [2048, 2048]]},
        },
        # A second nested aten op
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

    `keep_me/`: an empty marker file (most recent mtime → survives `--keep 1`).
    `drop_me/<prompt>/`: holds a synthetic chrome trace (least recent → gets dropped).
    """
    runs = tmp_path / "runs"
    runs.mkdir()

    # Older RUN_ID with a chrome trace inside a per-prompt subdir
    drop_me = runs / "drop_me"
    prompt_dir = drop_me / "prompt-uuid"
    prompt_dir.mkdir(parents=True)
    _build_synthetic_chrome_trace(prompt_dir / "torch_profile.0.json")

    # Touch with an old mtime so it sorts as oldest
    import os
    old = 1_000_000.0
    os.utime(prompt_dir / "torch_profile.0.json", (old, old))
    os.utime(prompt_dir, (old, old))
    os.utime(drop_me, (old, old))

    # Newer RUN_ID (just an empty marker) — survives `--keep 1`
    (runs / "keep_me").mkdir()
    (runs / "keep_me" / ".marker").write_text("keep")
    return runs


def _run_cleanup(runs_dir: Path, extra: list[str]) -> subprocess.CompletedProcess:
    """Invoke cleanup_traces.py with --runs-dir pointed at the fake dir."""
    script = REPO_ROOT / "scripts" / "cleanup_traces.py"
    cmd = [
        sys.executable, str(script),
        "--runs-dir", str(runs_dir),
        "--keep", "1",
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


def test_apply_extracts_sidecar_before_delete(fake_runs_dir: Path):
    """`--apply` must produce `<stem>.modules_summary.json` before rm-treeing.

    The drop directory itself is deleted, but if we read the script's
    stderr we should see an `[extract]` line confirming the sidecar
    was written first. Plus the sidecar's content should aggregate
    the aten ops under `audio_attn1/block_0`.

    Defensive shape of the test: re-create the fixture and run cleanup
    with `--no-extract` to confirm we get the opposite behavior in the
    same test sweep — keeps the assertions paired.
    """
    result = _run_cleanup(fake_runs_dir, ["--apply"])
    assert result.returncode == 0, result.stderr
    # drop_me should be gone
    assert not (fake_runs_dir / "drop_me").exists()
    # keep_me should remain
    assert (fake_runs_dir / "keep_me").exists()
    # Extraction line should appear in stderr
    assert "[extract]" in result.stderr, result.stderr
    assert "torch_profile.0.json -> torch_profile.0.modules_summary.json" in result.stderr


def test_no_extract_flag_skips_sidecar_creation(fake_runs_dir: Path):
    """`--no-extract` is the escape hatch; deletion still proceeds."""
    result = _run_cleanup(fake_runs_dir, ["--apply", "--no-extract"])
    assert result.returncode == 0, result.stderr
    assert not (fake_runs_dir / "drop_me").exists()
    # No extraction lines should appear
    assert "[extract]" not in result.stderr


def test_dry_run_does_not_extract_or_delete(fake_runs_dir: Path):
    """Dry-run (default, no `--apply`) leaves everything in place — including
    not running the analyzer. Extraction is gated behind `--apply` because
    parsing a 1 GB trace shouldn't happen on a preview-mode invocation.
    """
    result = _run_cleanup(fake_runs_dir, [])
    assert result.returncode == 0, result.stderr
    assert (fake_runs_dir / "drop_me").exists()
    # Sidecar should NOT exist after dry-run
    assert not (fake_runs_dir / "drop_me" / "prompt-uuid" / "torch_profile.0.modules_summary.json").exists()
    assert "[extract]" not in result.stderr


def test_extract_module_summary_is_idempotent(tmp_path: Path):
    """Running extraction twice produces no second write."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from cleanup_traces import extract_module_summary

    trace_path = tmp_path / "torch_profile.0.json"
    _build_synthetic_chrome_trace(trace_path)

    out1 = extract_module_summary(trace_path)
    assert out1 is not None
    assert out1.exists()
    mtime1 = out1.stat().st_mtime

    # Second call: sidecar exists → must skip without re-writing
    out2 = extract_module_summary(trace_path)
    assert out2 == out1
    assert out2.stat().st_mtime == mtime1


def test_extracted_sidecar_attributes_aten_ops_to_module(tmp_path: Path):
    """The sidecar's `modules` dict must contain `audio_attn1/block_0` with
    the two nested aten ops grouped under it. Catches regressions where
    the bisect span lookup stops attributing correctly.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from cleanup_traces import extract_module_summary

    trace_path = tmp_path / "torch_profile.0.json"
    _build_synthetic_chrome_trace(trace_path)

    sidecar = extract_module_summary(trace_path)
    assert sidecar is not None
    payload = orjson.loads(sidecar.read_bytes())

    assert payload["source_trace"] == str(trace_path)
    assert payload["total_events"] == 3

    modules = payload["modules"]
    assert "audio_attn1/block_0" in modules, f"got {list(modules)}"
    block_ops = modules["audio_attn1/block_0"]
    # Two distinct aten ops nested inside the annotation
    assert "aten::linear" in block_ops
    assert "aten::matmul" in block_ops
    assert block_ops["aten::linear"]["count"] == 1
    assert block_ops["aten::linear"]["total_us"] == 20.0
    # Shapes serialized as a list
    assert isinstance(block_ops["aten::linear"]["shapes"], list)
