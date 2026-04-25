"""Tests for `exec_logger.py`: opt-in execution logger.

Focus: the sentinel-on-execute pattern that prevents chained-wrapping
across ComfyUI module reloads. Background: if `_INSTALLED` is the only
guard and ComfyUI reloads `audioloophelper`, `_INSTALLED` resets to
False, `install()` runs again, captures the previously-wrapped
`_exec_mod.execute` as `original`, and adds a new sink in front of it.
After N reloads you have N sinks all writing the same data to N
different files (the 7-near-duplicate-files mystery from
2026-04-25). Sentinel on `_exec_mod.execute` itself survives the
reload; sentinel only goes away when `_exec_mod.execute` is replaced
wholesale (a future ComfyUI change), which is the right behavior.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import orjson
import pytest


def _stub_exec_module() -> types.ModuleType:
    """Create a minimal `execution` module suitable for `import execution`.

    Real ComfyUI's `execute()` is a coroutine with a long signature; we
    only need attribute presence to test the install() logic, not real
    behavior. Tests that exercise the wrapper itself would need a richer
    stub; the sentinel test only inspects `_exec_mod.execute`.
    """
    mod = types.ModuleType("execution")

    async def _execute(*args, **kwargs):
        return None

    mod.execute = _execute
    return mod


@pytest.fixture
def fresh_exec_logger(monkeypatch, tmp_path: Path):
    """Provide a freshly-imported `exec_logger` with a stub `execution`
    module and a tmp log path. Each test gets its own module instance --
    the module's _INSTALLED state is module-global, so reusing across
    tests would leak."""
    stub = _stub_exec_module()
    monkeypatch.setitem(sys.modules, "execution", stub)
    monkeypatch.setenv("COMFYUI_EXEC_LOG", str(tmp_path / "exec.jsonl"))

    # Force a fresh import so _INSTALLED starts as False.
    if "exec_logger" in sys.modules:
        del sys.modules["exec_logger"]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    el = importlib.import_module("exec_logger")
    yield el, stub


def test_install_marks_execute_with_sentinel(fresh_exec_logger):
    """After install(), _exec_mod.execute must carry the sentinel
    `_audioloophelper_wrapped = True`. The sentinel lives on the
    function object so it survives module reloads of exec_logger
    itself."""
    el, stub = fresh_exec_logger
    assert el.install() is True
    assert getattr(stub.execute, "_audioloophelper_wrapped", False) is True


def test_second_install_does_not_chain_wrap(fresh_exec_logger):
    """Module-reload simulation: reset _INSTALLED to False (as a fresh
    import would) and call install() again. The sentinel on
    stub.execute must short-circuit the wrap -- otherwise we get a
    chain of wrappers each writing to its own sink."""
    el, stub = fresh_exec_logger
    assert el.install() is True
    wrapped_first = stub.execute  # the wrapped function

    # Simulate module reload: _INSTALLED back to False, but sentinel
    # is still on stub.execute because it lives on the function object,
    # not in this module's globals.
    el._INSTALLED = False
    second_install = el.install()

    # Contract: install() returns True (idempotent — already wired) and
    # stub.execute is the SAME function object — no new wrapping layer.
    assert second_install is True
    assert stub.execute is wrapped_first
    # And the sentinel is still set (would be on the new wrapper too,
    # but identity check above already proves no new wrapper).
    assert getattr(stub.execute, "_audioloophelper_wrapped", False) is True


def test_install_proceeds_after_execute_is_replaced_wholesale(fresh_exec_logger):
    """If a future ComfyUI version replaces _exec_mod.execute (not just
    wraps it), the sentinel is gone -- and re-install should proceed.
    The pattern survives the right reloads and yields to the right
    replacements."""
    el, stub = fresh_exec_logger
    assert el.install() is True

    # ComfyUI replaced execute with a fresh function (no sentinel).
    async def fresh_execute(*args, **kwargs):
        return None
    stub.execute = fresh_execute
    el._INSTALLED = False

    # Should proceed and wrap the new function.
    assert el.install() is True
    assert stub.execute is not fresh_execute  # got wrapped
    assert getattr(stub.execute, "_audioloophelper_wrapped", False) is True


def test_install_no_op_when_env_var_unset(monkeypatch, tmp_path: Path):
    """No env var -> no install. Sentinel must NOT be set in this case
    -- the absence of the sentinel is what allows install() to proceed
    later if the env var gets set after process start."""
    stub = _stub_exec_module()
    monkeypatch.setitem(sys.modules, "execution", stub)
    monkeypatch.delenv("COMFYUI_EXEC_LOG", raising=False)

    if "exec_logger" in sys.modules:
        del sys.modules["exec_logger"]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    el = importlib.import_module("exec_logger")

    assert el.install() is False
    assert not getattr(stub.execute, "_audioloophelper_wrapped", False)
