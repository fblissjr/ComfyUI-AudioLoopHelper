"""Tests for PreDecodeCleanup — LATENT passthrough that unloads models and
frees pinned staging right before the full-song final decode.

Why this node exists: by decode time the diffusion model + pinned staging
are no longer needed; dropping them removes ~40-50GB of RAM/VRAM pressure
before the full-song final decode. This is HYGIENE, not the decode-OOM fix —
the decode buffer-stack OOM reproduced with this node proven in-graph; the
bound is a temporal-chunked decode. (Launch-flag/cache tuning can't help
either: the big tensors are node-live, invisible to cache eviction.)

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

    def test_cleanup_logs_what_it_freed(self, fake_mm, caplog):
        """The node must be log-visible: three kills in a row were
        undiagnosable because nothing recorded whether the cleanup ran or
        what it actually freed."""
        import logging

        with caplog.at_level(logging.INFO):
            from nodes import PreDecodeCleanup

            PreDecodeCleanup.execute(latent=_latent(), mode="always")
        msgs = [r.message for r in caplog.records if "PreDecodeCleanup" in r.message]
        assert any("freed" in m for m in msgs)

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


@pytest.fixture()
def fake_folder_paths(monkeypatch, tmp_path):
    """Fake comfy's folder_paths with a faithful get_save_image_path mini.

    Mirrors the real semantics _save_latent_checkpoint depends on: prefix
    splits into subfolder/filename, counter = 1 + max existing counter for
    `<filename>_NNNNN_` files in the target folder.
    """
    import os
    import re

    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(tmp_path)

    def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
        subfolder, filename = os.path.split(filename_prefix)
        full = os.path.join(output_dir, subfolder)
        os.makedirs(full, exist_ok=True)
        pat = re.compile(re.escape(filename) + r"_(\d+)_")
        counters = [
            int(m.group(1)) for f in os.listdir(full) if (m := pat.match(f))
        ]
        return full, filename, max(counters, default=0) + 1, subfolder, filename_prefix

    fp.get_save_image_path = get_save_image_path
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    return tmp_path


class TestLatentCheckpoint:
    """checkpoint_keep/checkpoint_prefix on PreDecodeCleanup — saves the
    assembled latent (core-SaveLatent-compatible format) then rotates,
    keeping the newest N checkpoint files. Replaces the standalone
    always-on SaveLatent whose per-render timestamped folders accumulate
    GB-scale .latent files with no cleanup."""

    def test_checkpoint_saves_loadable_latent(self, fake_mm, fake_folder_paths):
        import safetensors.torch

        from nodes import PreDecodeCleanup

        lat = _latent()
        out = PreDecodeCleanup.execute(
            latent=lat, mode="never",
            checkpoint_keep=1, checkpoint_prefix="latents/ckpt/test",
        )
        assert out[0] is lat
        path = fake_folder_paths / "latents" / "ckpt" / "test_00001_.latent"
        assert path.exists()
        # Core-LoadLatent contract: latent_format_version_0 must be present
        # or the loader applies the legacy SD 1/0.18215 multiplier.
        data = safetensors.torch.load_file(str(path))
        assert "latent_format_version_0" in data
        assert torch.equal(data["latent_tensor"], lat["samples"])

    def test_rotation_keeps_newest_n(self, fake_mm, fake_folder_paths):
        from nodes import PreDecodeCleanup

        for _ in range(3):
            PreDecodeCleanup.execute(
                latent=_latent(), mode="never",
                checkpoint_keep=2, checkpoint_prefix="latents/ckpt/test",
            )
        folder = fake_folder_paths / "latents" / "ckpt"
        names = sorted(p.name for p in folder.glob("*.latent"))
        assert names == ["test_00002_.latent", "test_00003_.latent"]

    def test_rotation_ignores_unrelated_files(self, fake_mm, fake_folder_paths):
        """Rotation must only touch this prefix's own checkpoint files —
        a sibling workflow checkpointing into the same folder is not ours
        to delete."""
        from nodes import PreDecodeCleanup

        folder = fake_folder_paths / "latents" / "ckpt"
        folder.mkdir(parents=True)
        other = folder / "other_00001_.latent"
        other.write_bytes(b"not ours")
        PreDecodeCleanup.execute(
            latent=_latent(), mode="never",
            checkpoint_keep=1, checkpoint_prefix="latents/ckpt/test",
        )
        assert other.exists()

    def test_keep_zero_is_noop_without_folder_paths(self, fake_mm):
        """Default checkpoint_keep=0 must not touch disk and must not need
        folder_paths at all (no-op schema default rule: existing saved
        workflows gain the widgets on reload with zero behavior change)."""
        from nodes import PreDecodeCleanup

        lat = _latent()
        out = PreDecodeCleanup.execute(
            latent=lat, mode="never",
            checkpoint_keep=0, checkpoint_prefix="latents/ckpt/test",
        )
        assert out[0] is lat

    def test_save_failure_never_kills_the_render(
        self, fake_mm, fake_folder_paths, recwarn
    ):
        """Same defensive contract as the unload path: this runs at the LAST
        step of a long render — checkpoint failure warns and passes through."""
        import folder_paths as fp

        def boom(*a, **k):
            raise OSError("disk full")

        fp.get_save_image_path = boom
        from nodes import PreDecodeCleanup

        lat = _latent()
        out = PreDecodeCleanup.execute(
            latent=lat, mode="never",
            checkpoint_keep=1, checkpoint_prefix="latents/ckpt/test",
        )
        assert out[0] is lat
        assert any("checkpoint" in str(w.message).lower() for w in recwarn.list)

    def test_metadata_embeds_prompt_json(self, fake_folder_paths):
        """The embedded prompt metadata is load-bearing forensics — the OOM
        postmortem recovered executed graphs from exactly this field. Parity
        with core SaveLatent: stdlib-json prompt under metadata['prompt']."""
        import json

        import safetensors

        from nodes import _save_latent_checkpoint

        prompt = {"42": {"class_type": "LatentConcat"}}
        saved = _save_latent_checkpoint(
            _latent(), "latents/ckpt/test", keep=1, prompt=prompt,
        )
        assert saved is not None
        with safetensors.safe_open(saved, framework="pt") as f:
            meta = f.metadata()
        assert json.loads(meta["prompt"]) == prompt


def test_pre_decode_cleanup_registered():
    from _node_registry import assert_node_registered

    assert_node_registered("PreDecodeCleanup")
