"""Tests for `tracers.exec_log.ExecLogTracer`: ComfyUI executor monkey-patch.

Focus: the sentinel-on-execute pattern that prevents chained-wrapping
across ComfyUI module reloads. Background: if a module-level _INSTALLED
guard were the only thing protecting against double-wrap, ComfyUI's
HotReloadHack would reset it to False on reload, install would run
again, capture the previously-wrapped `_exec_mod.execute` as `original`,
and add a new sink in front of it. After N reloads you'd have N sinks
all writing the same data to N different files (the 7-near-duplicate-
files mystery from 2026-04-25). The sentinel on `_exec_mod.execute`
itself survives the reload; sentinel only goes away when ComfyUI replaces
`_exec_mod.execute` wholesale (a future ComfyUI change), which is the
right behaviour.

After the 2026-05-16 tracer refactor this test file targets the new
`tracers.exec_log.ExecLogTracer` class but the contract under test
hasn't changed — same sentinel, same idempotence guarantees.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _stub_exec_module() -> types.ModuleType:
    """Create a minimal `execution` module suitable for `import execution`.

    Real ComfyUI's `execute()` is a coroutine with a long signature; we
    only need attribute presence to test the install() logic, not real
    behaviour.
    """
    mod = types.ModuleType("execution")

    async def _execute(*args, **kwargs):
        return None

    mod.execute = _execute
    return mod


@pytest.fixture
def fresh_tracer(monkeypatch, tmp_path: Path):
    """Provide a fresh `ExecLogTracer` instance + stub `execution` module.

    Each test gets its own tracer instance (no shared install state)
    and its own tmp log path.
    """
    stub = _stub_exec_module()
    monkeypatch.setitem(sys.modules, "execution", stub)
    monkeypatch.setenv("COMFYUI_EXEC_LOG", str(tmp_path / "exec.jsonl"))

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tracers.exec_log import ExecLogTracer
    tracer = ExecLogTracer()
    yield tracer, stub


def test_install_marks_execute_with_sentinel(fresh_tracer):
    """After install_at_import(), `_exec_mod.execute` must carry the
    sentinel `_audioloophelper_wrapped = True`. The sentinel lives on
    the function object so it survives module reloads of the tracer
    itself."""
    tracer, stub = fresh_tracer
    assert tracer.install_at_import() is True
    assert getattr(stub.execute, "_audioloophelper_wrapped", False) is True


def test_second_install_does_not_chain_wrap(fresh_tracer):
    """Reload simulation: build a fresh tracer instance and call
    install_at_import() on it. The sentinel on the (already-wrapped)
    `stub.execute` must short-circuit re-wrapping — otherwise we get a
    chain of wrappers each writing to its own sink.
    """
    tracer, stub = fresh_tracer
    assert tracer.install_at_import() is True
    wrapped_first = stub.execute

    from tracers.exec_log import ExecLogTracer
    second_tracer = ExecLogTracer()
    second_install = second_tracer.install_at_import()

    assert second_install is True
    assert stub.execute is wrapped_first
    assert getattr(stub.execute, "_audioloophelper_wrapped", False) is True


def test_install_proceeds_after_execute_is_replaced_wholesale(fresh_tracer):
    """If ComfyUI replaces `_exec_mod.execute` wholesale (e.g. a future
    upstream change), the sentinel is gone and re-install correctly
    re-wraps the new function. This guards against accidentally getting
    permanently stuck on the first wrap.
    """
    tracer, stub = fresh_tracer
    assert tracer.install_at_import() is True

    async def new_execute(*args, **kwargs):
        return "replaced"
    stub.execute = new_execute
    assert not getattr(stub.execute, "_audioloophelper_wrapped", False)

    from tracers.exec_log import ExecLogTracer
    second_tracer = ExecLogTracer()
    assert second_tracer.install_at_import() is True
    assert getattr(stub.execute, "_audioloophelper_wrapped", False) is True


def test_install_no_op_when_env_var_unset(monkeypatch):
    """install_at_import() must return False when the env var is unset.

    Zero-overhead-when-disabled is a load-bearing property of the tracer
    framework; a missing env var has to short-circuit before any
    monkey-patch logic runs.
    """
    monkeypatch.delenv("COMFYUI_EXEC_LOG", raising=False)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tracers.exec_log import ExecLogTracer
    tracer = ExecLogTracer()
    assert tracer.install_at_import() is False
    assert tracer._installed is False
