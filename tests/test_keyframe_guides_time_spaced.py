"""Tests for KeyframeGuidesTimeSpaced — single-pass keyframe fill-in-the-middle.

Collapses the manual N-guide chain (Index -> Get Image from Batch ->
Math Expression -> LTXVAddGuideMulti) into one node: feed a DENSE keyframe
IMAGE batch + (output_fps, seconds_per_keyframe), get ready-to-sample
(positive, negative, latent) with every keyframe placed as an LTX guide.
No loop; video-only latent (no frozen audio).

Distinct from KJNodes `LTXVAddGuidesFromBatch`, which places keyframe i at
frame i (consecutive) and needs a black-padded full-length batch for spacing;
this node time-spaces a dense batch via fps math and reuses the same core
`LTXVAddGuide` machinery (resize + encode + get_latent_index + append_keyframe).

Surfaces (added after the first cut):
- `placement_info` STRING output + a WARN when keyframes are dropped past the
  latent end (no silent data loss).
- a WARN when `output_fps` disagrees with the conditioning's stamped frame_rate
  (otherwise keyframes land at the wrong times with no signal).

Test surfaces:
- `_keyframe_guide_placements` is the pure placement seam (no ComfyUI).
- `execute()` is tested with a faked `comfy_extras.nodes_lt` so it runs on CI.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest
import torch


# --------------------------------------------------------------------------
# Pure placement-math helper
# --------------------------------------------------------------------------
class TestPlacementMath:
    def _place(
        self,
        *,
        batch_size: int,
        n_latent_frames: int = 64,
        output_fps: float = 25.0,
        seconds_per_keyframe: float = 1.0,
        temporal_scale: int = 8,
    ):
        from nodes import _keyframe_guide_placements
        return _keyframe_guide_placements(
            batch_size=batch_size,
            n_latent_frames=n_latent_frames,
            output_fps=output_fps,
            seconds_per_keyframe=seconds_per_keyframe,
            temporal_scale=temporal_scale,
        )

    def test_first_keyframe_anchors_frame_zero(self):
        assert self._place(batch_size=3)[0] == (0, 0)

    def test_one_per_second_at_25fps_hits_exact_frame(self):
        placements = self._place(batch_size=6, output_fps=25.0, seconds_per_keyframe=1.0)
        assert placements == [(0, 0), (1, 25), (2, 50), (3, 75), (4, 100), (5, 125)]

    def test_fractional_fps_rounds(self):
        placements = self._place(batch_size=4, output_fps=24.0, seconds_per_keyframe=0.5)
        assert [f for _i, f in placements] == [0, 12, 24, 36]

    def test_frame_idx_strictly_increasing(self):
        placements = self._place(batch_size=12, seconds_per_keyframe=0.01)
        frames = [f for _i, f in placements]
        assert frames == sorted(frames)
        assert len(set(frames)) == len(frames)

    def test_keyframes_past_latent_length_dropped(self):
        placements = self._place(batch_size=50, n_latent_frames=10)
        for _i, frame_idx in placements:
            assert (frame_idx + 8 - 1) // 8 < 10
        assert len(placements) < 50

    def test_empty_batch_returns_empty(self):
        assert self._place(batch_size=0) == []

    def test_zero_latent_frames_returns_empty(self):
        assert self._place(batch_size=5, n_latent_frames=0) == []


# --------------------------------------------------------------------------
# execute() orchestration with a faked comfy_extras.nodes_lt
# --------------------------------------------------------------------------
class _FakeVAE:
    downscale_index_formula = (8, 32, 32)


@pytest.fixture
def fake_nodes_lt(monkeypatch):
    call_log: list[dict] = []
    mod = types.ModuleType("comfy_extras.nodes_lt")

    def get_noise_mask(latent):
        nm = latent.get("noise_mask")
        if nm is None:
            b, _, f, _, _ = latent["samples"].shape
            return torch.ones((b, 1, f, 1, 1), dtype=torch.float32)
        return nm.clone()

    class LTXVAddGuide:
        @classmethod
        def encode(cls, vae, latent_width, latent_height, images, scale_factors, latent_downscale_factor=1):
            call_log.append({"encode_to": (latent_height, latent_width)})
            t = torch.zeros(1, 128, 1, latent_height, latent_width, dtype=torch.float32)
            return images, t

        @classmethod
        def get_latent_index(cls, cond, latent_length, guide_length, frame_idx, scale_factors, latent_shape=None):
            latent_idx = (frame_idx + scale_factors[0] - 1) // scale_factors[0]
            return frame_idx, latent_idx

        @classmethod
        def append_keyframe(cls, positive, negative, frame_idx, latent_image, noise_mask, guiding_latent, strength, scale_factors, **kw):
            call_log.append({"append_frame_idx": frame_idx, "strength": strength})
            new_latent = torch.cat([latent_image, guiding_latent], dim=2)
            mask = torch.full(
                (noise_mask.shape[0], 1, guiding_latent.shape[2], noise_mask.shape[3], noise_mask.shape[4]),
                max(0.0, 1.0 - strength), dtype=noise_mask.dtype,
            )
            new_mask = torch.cat([noise_mask, mask], dim=2)
            return (positive + [frame_idx], negative + [frame_idx], new_latent, new_mask)

    mod.get_noise_mask = get_noise_mask
    mod.LTXVAddGuide = LTXVAddGuide
    parent = sys.modules.get("comfy_extras") or types.ModuleType("comfy_extras")
    monkeypatch.setitem(sys.modules, "comfy_extras", parent)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_lt", mod)
    return call_log


def _video_latent(n_latent_frames: int = 64, hw: tuple[int, int] = (8, 8)) -> dict:
    return {"samples": torch.zeros(1, 128, n_latent_frames, hw[0], hw[1], dtype=torch.float32)}


def _images(n: int, hw: tuple[int, int] = (256, 256)) -> torch.Tensor:
    return torch.zeros(n, hw[0], hw[1], 3, dtype=torch.float32)


def _cond_with_frame_rate(fr: float) -> list:
    """A conditioning shaped like LTXVConditioning output: [[tensor, {dict}]]."""
    return [[None, {"frame_rate": fr}]]


def _appends(call_log: list[dict]) -> list[dict]:
    return [c for c in call_log if "append_frame_idx" in c]


class TestExecute:
    def _run(self, *, vae, images, latent, positive=None, negative=None,
             output_fps=25.0, seconds_per_keyframe=1.0, strength=1.0):
        from nodes import KeyframeGuidesTimeSpaced
        return KeyframeGuidesTimeSpaced.execute(
            vae=vae, positive=[] if positive is None else positive,
            negative=[] if negative is None else negative,
            latent=latent, images=images,
            output_fps=output_fps, seconds_per_keyframe=seconds_per_keyframe, strength=strength,
        )

    def test_one_append_per_placed_keyframe(self, fake_nodes_lt):
        self._run(vae=_FakeVAE(), images=_images(5), latent=_video_latent(64))
        assert len(_appends(fake_nodes_lt)) == 5

    def test_guides_placed_at_exact_time_frames(self, fake_nodes_lt):
        self._run(vae=_FakeVAE(), images=_images(4), latent=_video_latent(64))
        frames = [c["append_frame_idx"] for c in _appends(fake_nodes_lt)]
        assert frames == [0, 25, 50, 75]

    def test_conditioning_threaded_through_chain(self, fake_nodes_lt):
        out = self._run(vae=_FakeVAE(), images=_images(3), latent=_video_latent(64))
        frames = [c["append_frame_idx"] for c in _appends(fake_nodes_lt)]
        assert out[0] == frames
        assert out[1] == frames

    def test_output_latent_carries_noise_mask(self, fake_nodes_lt):
        out = self._run(vae=_FakeVAE(), images=_images(3), latent=_video_latent(64))
        assert "samples" in out[2] and "noise_mask" in out[2]

    def test_strength_forwarded(self, fake_nodes_lt):
        self._run(vae=_FakeVAE(), images=_images(2), latent=_video_latent(64), strength=0.4)
        assert all(c["strength"] == pytest.approx(0.4) for c in _appends(fake_nodes_lt))

    def test_keyframes_resized_to_latent_resolution(self, fake_nodes_lt):
        self._run(vae=_FakeVAE(), images=_images(3, hw=(123, 77)), latent=_video_latent(64, hw=(8, 8)))
        encodes = [c for c in fake_nodes_lt if "encode_to" in c]
        assert encodes and all(c["encode_to"] == (8, 8) for c in encodes)

    def test_keyframes_past_latent_truncated(self, fake_nodes_lt):
        self._run(vae=_FakeVAE(), images=_images(50), latent=_video_latent(10))
        assert len(_appends(fake_nodes_lt)) < 50

    # --- #1: placement_info output + WARN on drop ---
    def test_placement_info_reports_all_placed(self, fake_nodes_lt):
        out = self._run(vae=_FakeVAE(), images=_images(5), latent=_video_latent(64))
        info = out[3]
        assert isinstance(info, str)
        assert "5/5" in info

    def test_dropped_keyframes_reported_in_info(self, fake_nodes_lt):
        out = self._run(vae=_FakeVAE(), images=_images(50), latent=_video_latent(10))
        info = out[3]
        placed = len(_appends(fake_nodes_lt))
        assert f"{placed}/50" in info
        assert "dropped" in info.lower()

    def test_dropped_keyframes_emit_warning(self, fake_nodes_lt, caplog):
        with caplog.at_level(logging.WARNING):
            self._run(vae=_FakeVAE(), images=_images(50), latent=_video_latent(10))
        assert any("drop" in r.message.lower() for r in caplog.records)

    def test_no_warning_when_all_placed(self, fake_nodes_lt, caplog):
        with caplog.at_level(logging.WARNING):
            self._run(vae=_FakeVAE(), images=_images(5), latent=_video_latent(64))
        assert not [r for r in caplog.records if "drop" in r.message.lower()]

    # --- #2: fps mismatch warning ---
    def test_fps_mismatch_warns(self, fake_nodes_lt, caplog):
        with caplog.at_level(logging.WARNING):
            self._run(vae=_FakeVAE(), images=_images(2), latent=_video_latent(64),
                      positive=_cond_with_frame_rate(30.0), output_fps=25.0)
        assert any("frame_rate" in r.message for r in caplog.records)

    def test_fps_match_no_warn(self, fake_nodes_lt, caplog):
        with caplog.at_level(logging.WARNING):
            self._run(vae=_FakeVAE(), images=_images(2), latent=_video_latent(64),
                      positive=_cond_with_frame_rate(25.0), output_fps=25.0)
        assert not [r for r in caplog.records if "frame_rate" in r.message]

    def test_fps_absent_no_warn(self, fake_nodes_lt, caplog):
        with caplog.at_level(logging.WARNING):
            self._run(vae=_FakeVAE(), images=_images(2), latent=_video_latent(64), output_fps=25.0)
        assert not [r for r in caplog.records if "frame_rate" in r.message]


class TestRegistration:
    def test_node_registered(self):
        from _node_registry import assert_node_registered
        assert_node_registered("KeyframeGuidesTimeSpaced")
