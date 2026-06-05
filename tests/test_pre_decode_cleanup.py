"""Tests for PreDecodeCleanup — LATENT passthrough that unloads models and
frees pinned staging right before the full-song final decode.

Why this node exists: the decode stage of a full-song loop render is a
single-node RAM spike (~37GB fp32 output at 960x544/200s) on top of ~24GB
page-locked staging + offloaded model + text encoder — the sum kernel-OOMs a
125GB box at the LAST step, after all sampling succeeded. By decode time the
diffusion model is no longer needed; dropping it + the pins removes ~40-50GB
from the profile. Launch-flag tuning cannot fix this (page-locked memory is
unswappable; the spike happens inside one node where cache eviction can't
run).

Call-order contract: free_pins MUST run before unload_all_models —
free_pins iterates comfy's current_loaded_models, which unload_all_models
empties (pins-after-unload is a silent no-op).

comfy.model_management is faked via sys.modules (same pattern as
test_keyframe_guides_time_spaced's nodes_lt fake).
"""

from __future__ import annotations

import sys
import types

import pytest
import torch


@pytest.fixture()
def fake_mm(monkeypatch):
    """Fake comfy.model_management recording calls in order."""
    calls: list[tuple] = []
    mm = types.ModuleType("comfy.model_management")
    mm.calls = calls
    mm.unload_all_models = lambda: calls.append(("unload_all_models",))
    mm.free_pins = lambda size, evict_active=False: (
        calls.append(("free_pins", evict_active)) or 0
    )
    mm.soft_empty_cache = lambda force=False: calls.append(("soft_empty_cache",))
    comfy_pkg = sys.modules.get("comfy") or types.ModuleType("comfy")
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setattr(comfy_pkg, "model_management", mm, raising=False)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return mm


def _latent() -> dict:
    return {"samples": torch.zeros(1, 4, 2, 2, 2)}


class TestPreDecodeCleanup:
    def test_latent_passes_through_identity(self, fake_mm):
        from nodes import PreDecodeCleanup

        lat = _latent()
        out = PreDecodeCleanup.execute(latent=lat, mode="always")
        assert out[0] is lat  # passthrough, not a copy

    def test_always_unloads_and_frees_pins(self, fake_mm):
        from nodes import PreDecodeCleanup

        PreDecodeCleanup.execute(latent=_latent(), mode="always")
        names = [c[0] for c in fake_mm.calls]
        assert "unload_all_models" in names
        assert "free_pins" in names
        assert "soft_empty_cache" in names

    def test_free_pins_runs_before_unload(self, fake_mm):
        """free_pins iterates current_loaded_models; unload_all_models empties
        it — the reverse order silently frees nothing."""
        from nodes import PreDecodeCleanup

        PreDecodeCleanup.execute(latent=_latent(), mode="always")
        names = [c[0] for c in fake_mm.calls]
        assert names.index("free_pins") < names.index("unload_all_models")

    def test_free_pins_evicts_active(self, fake_mm):
        """Staging pins for the just-used diffusion model count as active;
        without evict_active=True they survive the cleanup."""
        from nodes import PreDecodeCleanup

        PreDecodeCleanup.execute(latent=_latent(), mode="always")
        pin_calls = [c for c in fake_mm.calls if c[0] == "free_pins"]
        assert pin_calls and pin_calls[0][1] is True

    def test_never_mode_is_pure_passthrough(self, fake_mm):
        from nodes import PreDecodeCleanup

        lat = _latent()
        out = PreDecodeCleanup.execute(latent=lat, mode="never")
        assert out[0] is lat
        assert fake_mm.calls == []

    def test_mm_errors_warn_but_do_not_kill_the_render(self, fake_mm, recwarn):
        """This node runs at the LAST step of a long render — a comfy-internals
        error in the cleanup must warn and pass the latent through, never raise
        (same defensive contract as _purge_stale_loaded_models)."""
        from nodes import PreDecodeCleanup

        def boom():
            raise RuntimeError("comfy internals")

        fake_mm.unload_all_models = boom
        lat = _latent()
        out = PreDecodeCleanup.execute(latent=lat, mode="always")
        assert out[0] is lat
        assert any("unload_all_models" in str(w.message) for w in recwarn.list)


def test_pre_decode_cleanup_registered():
    from _node_registry import assert_node_registered

    assert_node_registered("PreDecodeCleanup")
