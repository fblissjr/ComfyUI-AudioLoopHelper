"""Behavioral tests for PurgeVRAM and its helper _purge_stale_loaded_models.

Last updated: 2026-05-10

Targets the ComfyUI weakref-finalize race where
``current_loaded_models`` holds wrappers whose ``.model`` was GC'd to
None. ``free_memory()`` and ``cleanup_models()`` both crash walking
those entries. The helper prunes stale entries before calling cleanup,
and the node wires that as a pass-through so users can splice it
between sampler output and the next model-using node.

ComfyUI isn't loaded in pytest; tests inject a fake
``comfy.model_management`` module to exercise the prune logic and
verify graceful fallback when the import fails.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from nodes import PurgeVRAM, _purge_stale_loaded_models


@pytest.fixture
def fake_mm(monkeypatch):
    """Inject a fake comfy.model_management with mutable
    current_loaded_models + a recorded cleanup_models()."""
    parent = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = []  # type: ignore[attr-defined]
    mm.cleanup_calls = 0  # type: ignore[attr-defined]

    def _cleanup_models():
        mm.cleanup_calls += 1  # type: ignore[attr-defined]
    mm.cleanup_models = _cleanup_models  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "comfy", parent)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return mm


class _Stale:
    """Wrapper whose ``.model`` attribute has been finalized to None —
    the failure mode crashes ComfyUI's free_memory walk."""
    model = None


class _Live:
    def __init__(self):
        self.model = object()


def test_helper_prunes_stale_entries(fake_mm):
    live1 = _Live()
    live2 = _Live()
    fake_mm.current_loaded_models = [_Stale(), live1, _Stale(), live2]
    _purge_stale_loaded_models()
    assert fake_mm.current_loaded_models == [live1, live2]


def test_helper_calls_cleanup_after_prune(fake_mm):
    _purge_stale_loaded_models()
    assert fake_mm.cleanup_calls == 1


def test_helper_survives_cleanup_models_raising(fake_mm):
    def _boom():
        raise RuntimeError("simulated ComfyUI internals crash")
    fake_mm.cleanup_models = _boom
    fake_mm.current_loaded_models = [_Live()]
    # must not raise
    _purge_stale_loaded_models()
    # state survived
    assert len(fake_mm.current_loaded_models) == 1


def test_helper_no_op_when_comfy_unavailable(monkeypatch):
    """When comfy.model_management isn't importable (tests, headless
    harness), the helper silently no-ops."""
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    # must not raise
    _purge_stale_loaded_models()


def test_execute_is_latent_passthrough(fake_mm):
    latent = {"samples": torch.zeros(1, 4, 8, 8, 8)}
    out = PurgeVRAM.execute(latent=latent)
    assert out[0] is latent  # same dict, not a copy


def test_execute_runs_purge_as_side_effect(fake_mm):
    fake_mm.current_loaded_models = [_Stale(), _Live(), _Stale()]
    latent = {"samples": torch.zeros(1, 4, 8, 8, 8)}
    PurgeVRAM.execute(latent=latent)
    # one Live survived
    assert len(fake_mm.current_loaded_models) == 1
    assert fake_mm.cleanup_calls == 1


def test_node_is_registered_in_extension():
    from _node_registry import assert_node_registered
    assert_node_registered("PurgeVRAM")
