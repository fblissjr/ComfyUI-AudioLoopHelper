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
    """Set RUN_ID to a deterministic test value and clean up after.

    Recursively removes the test's run dir on teardown — per-prompt
    routing tests legitimately create nested `${RUN_ID}/${prompt_id}/`
    subdirs, so a flat unlink-loop won't suffice."""
    import shutil

    rid = "test_propagation_xyz"
    monkeypatch.setenv("RUN_ID", rid)
    target = REPO_ROOT / "data" / "runs" / rid
    # Pre-clean: prior failed runs may have left a partial tree behind.
    if target.exists():
        shutil.rmtree(target)
    yield rid
    if target.exists():
        shutil.rmtree(target)


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
    def test_auto_token_with_run_id_uses_data_runs_path(self, run_id, monkeypatch):
        """`COMFYUI_EXEC_LOG=auto` + `RUN_ID=...` should resolve to the
        data/runs path, not the legacy timestamped path."""
        monkeypatch.setenv("COMFYUI_EXEC_LOG", "auto")
        from tracers._base import resolve_path_from_env
        target = str(resolve_path_from_env("COMFYUI_EXEC_LOG", "exec", "jsonl"))
        assert target.endswith(f"data/runs/{run_id}/exec.jsonl"), target

    def test_auto_token_without_run_id_uses_legacy(self, no_run_id, monkeypatch):
        monkeypatch.setenv("COMFYUI_EXEC_LOG", "auto")
        from tracers._base import resolve_path_from_env
        target = str(resolve_path_from_env("COMFYUI_EXEC_LOG", "exec", "jsonl"))
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


# -----------------------------------------------------------------------------
# Per-prompt routing (AUDIOLOOPHELPER_PER_PROMPT)
#
# When this env var is set AND ComfyUI's executing-context contextvar has a
# prompt_id (i.e. we're inside /prompt execution, not graph-build), the
# auto-path moves from `data/runs/${RUN_ID}/<cat>.<ext>` to
# `data/runs/${RUN_ID}/${prompt_id}/<cat>.<ext>`. Every telemetry consumer
# that goes through `run_artifact_path` / `run_artifact_dir` (exec_logger,
# nodes_sage, ProfileBegin) inherits this for free.
#
# Default behavior (env var unset) is unchanged — that's the back-compat
# guard for the existing 502+ tests + shipping bench tools.
# -----------------------------------------------------------------------------


@pytest.fixture
def per_prompt_on(monkeypatch):
    """Toggle AUDIOLOOPHELPER_PER_PROMPT=1 for the test."""
    monkeypatch.setenv("AUDIOLOOPHELPER_PER_PROMPT", "1")
    yield


@pytest.fixture
def fake_executing_context(monkeypatch):
    """Stub `comfy_execution.utils.get_executing_context` with a
    settable-prompt-id factory. Returns a setter so each test can drive
    the contextvar value without actually starting ComfyUI.

    Mirrors how nodes_sage.py:567-573 reads the context: imports lazily
    inside the helper, tolerates ImportError. We register a fake module
    in sys.modules so the lazy import inside `_current_prompt_id` finds
    our stub instead of the real ComfyUI module (which may or may not be
    importable in the test env).
    """
    import sys
    import types

    state = {"prompt_id": None}

    fake_module = types.ModuleType("comfy_execution.utils")

    class _FakeCtx:
        def __init__(self, pid):
            self.prompt_id = pid

    def get_executing_context():
        pid = state["prompt_id"]
        return _FakeCtx(pid) if pid is not None else None

    fake_module.get_executing_context = get_executing_context
    # Parent package stub so `from comfy_execution.utils import ...` resolves.
    parent = types.ModuleType("comfy_execution")
    parent.utils = fake_module
    monkeypatch.setitem(sys.modules, "comfy_execution", parent)
    monkeypatch.setitem(sys.modules, "comfy_execution.utils", fake_module)

    def setter(pid):
        state["prompt_id"] = pid

    yield setter


class TestPerPromptRouting:
    """`AUDIOLOOPHELPER_PER_PROMPT=1` opts every `run_artifact_path` call
    into per-prompt subdirectory routing. With it unset, behavior is
    identical to the pre-existing `data/runs/${RUN_ID}/<cat>.<ext>` shape
    (back-compat for shipping tools and the 502+ existing tests)."""

    def test_per_prompt_unset_yields_flat_path(self, run_id, fake_executing_context):
        """Default (env var unset) — current behavior must not change.
        Even with an active executing-context, no per-prompt subdir."""
        fake_executing_context("prompt_abc123")
        from workflow_utils import run_artifact_path
        p = run_artifact_path("sage", "jsonl")
        expected = REPO_ROOT / "data" / "runs" / run_id / "sage.jsonl"
        assert p == expected

    def test_per_prompt_set_with_context_yields_nested_path(
        self, run_id, per_prompt_on, fake_executing_context,
    ):
        """env var on + executing-context has prompt_id ->
        data/runs/${RUN_ID}/${prompt_id}/<cat>.<ext>."""
        fake_executing_context("prompt_abc123")
        from workflow_utils import run_artifact_path
        p = run_artifact_path("sage", "jsonl")
        expected = REPO_ROOT / "data" / "runs" / run_id / "prompt_abc123" / "sage.jsonl"
        assert p == expected
        assert p.parent.exists()

    def test_per_prompt_set_no_context_falls_back_to_flat(
        self, run_id, per_prompt_on, fake_executing_context,
    ):
        """env var on but NO executing-context (e.g. graph-build phase,
        before /prompt fires) -> flat data/runs/${RUN_ID}/ path. We can't
        route what we don't have a key for."""
        fake_executing_context(None)  # context returns None
        from workflow_utils import run_artifact_path
        p = run_artifact_path("sage", "jsonl")
        expected = REPO_ROOT / "data" / "runs" / run_id / "sage.jsonl"
        assert p == expected

    def test_different_prompt_ids_different_paths(
        self, run_id, per_prompt_on, fake_executing_context,
    ):
        """Two prompts in the same ComfyUI session route to distinct
        subdirs — that's the whole point of the fix."""
        from workflow_utils import run_artifact_path

        fake_executing_context("prompt_AAA")
        p1 = run_artifact_path("sage", "jsonl")

        fake_executing_context("prompt_BBB")
        p2 = run_artifact_path("sage", "jsonl")

        assert p1.parent != p2.parent
        assert p1.parent.name == "prompt_AAA"
        assert p2.parent.name == "prompt_BBB"

    def test_same_prompt_id_same_dir_across_categories(
        self, run_id, per_prompt_on, fake_executing_context,
    ):
        """All telemetry categories (exec, sage, profiler) for ONE prompt
        land in the same subdir — proves stickiness within a prompt and
        is the cross-system correlation guarantee."""
        from workflow_utils import run_artifact_dir, run_artifact_path

        fake_executing_context("prompt_sticky")
        exec_p = run_artifact_path("exec", "jsonl")
        sage_p = run_artifact_path("sage", "jsonl")
        prof_dir = run_artifact_dir("profiler")

        assert exec_p.parent == sage_p.parent
        assert prof_dir.parent == sage_p.parent
        assert exec_p.parent.name == "prompt_sticky"
