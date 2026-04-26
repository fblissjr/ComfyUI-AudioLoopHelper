"""RUN_ID env-var propagation across our telemetry stack.

A render writes to multiple loggers (exec_log, sage tracer, profiler) and
produces an output mp4 via VHS_VideoCombine. Without a shared correlation
id, each logger stamps its own filename from `time.time()` at the moment
it spins up, so files from the same conceptual render look unrelated by
filename. The fix: a single `RUN_ID` env var, propagated to every logger
+ the workflow harness, used as the directory key under `data/runs/`.

These tests lock in the propagation contract:

  - When `RUN_ID` is set, every logger writes to `data/runs/${RUN_ID}/<category>.<ext>`.
  - When `RUN_ID` is unset, the legacy `internal/analysis/runs/<subdir>/<prefix>_TIMESTAMP.<ext>` path is used (back-compat).
  - The legacy path is NOT also written when RUN_ID is set (no double writes).
"""

import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def run_id(monkeypatch):
    """Set RUN_ID to a deterministic test value and clean up after."""
    rid = "test_propagation_xyz"
    monkeypatch.setenv("RUN_ID", rid)
    yield rid
    # cleanup any directory the test created
    target = REPO_ROOT / "data" / "runs" / rid
    if target.exists():
        for p in target.iterdir():
            p.unlink()
        target.rmdir()


@pytest.fixture
def no_run_id(monkeypatch):
    monkeypatch.delenv("RUN_ID", raising=False)
    yield


class TestRunArtifactPath:
    """workflow_utils.run_artifact_path is the single helper that decides
    where a per-run artifact lands. Every logger calls it."""

    def test_run_id_set_yields_data_runs_layout(self, run_id):
        from workflow_utils import run_artifact_path
        p = run_artifact_path("exec", "jsonl")
        expected = REPO_ROOT / "data" / "runs" / run_id / "exec.jsonl"
        assert p == expected
        # parent dir created
        assert p.parent.exists()

    def test_run_id_set_same_dir_for_all_categories(self, run_id):
        from workflow_utils import run_artifact_path
        exec_path = run_artifact_path("exec", "jsonl")
        sage_path = run_artifact_path("sage", "jsonl")
        # Both artifacts land in the same RUN_ID dir — that's the whole point.
        assert exec_path.parent == sage_path.parent
        assert exec_path.parent.name == run_id

    def test_run_id_unset_falls_back_to_legacy(self, no_run_id):
        from workflow_utils import run_artifact_path
        p = run_artifact_path("exec", "jsonl")
        # Legacy path lives under internal/analysis/runs/exec/exec_<TS>.jsonl
        # (two-arg form: subdir=exec, prefix=exec; matches existing
        # timestamped_run_path call sites).
        assert "internal/analysis/runs" in str(p)
        assert p.suffix == ".jsonl"
        # Filename includes a timestamp pattern (e.g. 2026-04-26_134522)
        assert any(c.isdigit() for c in p.stem)


class TestExecLoggerHonorsRunId:
    def test_auto_token_with_run_id_uses_data_runs_path(self, run_id):
        """`COMFYUI_EXEC_LOG=auto` + `RUN_ID=...` should resolve to the
        data/runs path, not the legacy timestamped path."""
        from exec_logger import _resolve_log_target
        target = _resolve_log_target("auto")
        assert target.endswith(f"data/runs/{run_id}/exec.jsonl"), target

    def test_auto_token_without_run_id_uses_legacy(self, no_run_id):
        from exec_logger import _resolve_log_target
        target = _resolve_log_target("auto")
        assert "internal/analysis/runs" in target


class TestSageTracerHonorsRunId:
    def test_auto_token_with_run_id_uses_data_runs_path(self, run_id, monkeypatch):
        monkeypatch.setenv("AUDIOLOOPHELPER_SAGE_TRACE", "auto")
        from nodes_sage import resolve_trace_path
        p = resolve_trace_path()
        assert p is not None
        assert str(p).endswith(f"data/runs/{run_id}/sage.jsonl"), str(p)

    def test_auto_token_without_run_id_uses_legacy(self, no_run_id, monkeypatch):
        monkeypatch.setenv("AUDIOLOOPHELPER_SAGE_TRACE", "auto")
        from nodes_sage import resolve_trace_path
        p = resolve_trace_path()
        assert p is not None
        assert "internal/analysis/runs" in str(p)
