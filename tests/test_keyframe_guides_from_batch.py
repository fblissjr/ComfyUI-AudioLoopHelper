"""Tests for KeyframeGuidesFromBatch — single-pass keyframe fill-in-the-middle.

Collapses the manual N-guide chain (Index -> Get Image from Batch ->
Math Expression -> LTXVAddGuideMulti image_N/frame_idx_N) into one node:
feed a DENSE keyframe IMAGE batch + (output_fps, seconds_per_keyframe), get
ready-to-sample (positive, negative, latent) with every keyframe placed
as an LTX guide. No loop; video-only latent (no frozen audio).

Distinct from KJNodes `LTXVAddGuidesFromBatch`, which places keyframe i at
frame i (consecutive) and needs a black-padded full-length batch for spacing;
this node time-spaces a dense batch via fps math and reuses the same core
`LTXVAddGuide` machinery (resize + encode + get_latent_index + append_keyframe).

Two test surfaces:
- `_keyframe_guide_placements` is the logic-bearing seam (fps -> exact pixel
  frame, monotonic, drop-past-end). Tested pure, no ComfyUI.
- `execute()` orchestration is tested with a faked `comfy_extras.nodes_lt`
  so it runs on CI where ComfyUI is absent.
"""

from __future__ import annotations

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
        """Keyframe 0 lands at pixel frame 0 (hard first frame)."""
        placements = self._place(batch_size=3)
        assert placements[0] == (0, 0)

    def test_one_per_second_at_25fps_hits_exact_frame(self):
        """Single-frame guides are not 8-snapped, so keyframe i lands at the
        EXACT pixel frame i*25 (= i seconds @ 25fps)."""
        placements = self._place(batch_size=6, output_fps=25.0, seconds_per_keyframe=1.0)
        assert placements == [(0, 0), (1, 25), (2, 50), (3, 75), (4, 100), (5, 125)]

    def test_fractional_fps_rounds(self):
        """seconds*fps that isn't integer rounds to nearest pixel frame."""
        # 0.5s @ 24fps = 12px spacing.
        placements = self._place(batch_size=4, output_fps=24.0, seconds_per_keyframe=0.5)
        assert [f for _i, f in placements] == [0, 12, 24, 36]

    def test_frame_idx_strictly_increasing(self):
        """Placements never collide or go backwards, even for dense keyframes."""
        # 0.01s @ 25fps = 0.25px/keyframe < 1 -> would collide without the bump.
        placements = self._place(batch_size=12, seconds_per_keyframe=0.01)
        frames = [f for _i, f in placements]
        assert frames == sorted(frames)
        assert len(set(frames)) == len(frames)

    def test_keyframes_past_latent_length_dropped(self):
        """Batch larger than the latent can hold -> truncated. Drop boundary is
        ceil(frame_idx/temporal) >= n_latent_frames (core's 1-frame-guide bound)."""
        placements = self._place(batch_size=50, n_latent_frames=10)
        for _i, frame_idx in placements:
            latent_idx = (frame_idx + 8 - 1) // 8
            assert latent_idx < 10
        assert len(placements) < 50

    def test_empty_batch_returns_empty(self):
        assert self._place(batch_size=0) == []

    def test_zero_latent_frames_returns_empty(self):
        assert self._place(batch_size=5, n_latent_frames=0) == []


# --------------------------------------------------------------------------
# execute() orchestration with a faked comfy_extras.nodes_lt
# --------------------------------------------------------------------------
class _FakeVAE:
    """Video VAE stub: temporal x8, spatial x32 (downscale_index_formula)."""

    downscale_index_formula = (8, 32, 32)


@pytest.fixture
def fake_nodes_lt(monkeypatch):
    """Install a minimal comfy_extras.nodes_lt so execute() runs on CI.

    Mimics the core `LTXVAddGuide` surface the node composes:
      - encode(): resizes (records the requested latent W/H) and returns a
        single-frame guide latent already at the latent's spatial dims — i.e.
        the resize guarantees a spatial match (the whole point of (A)).
      - get_latent_index(): core's ceil mapping, no 8-snap for 1-frame guides.
      - append_keyframe(): cat guide onto latent + mask, thread conditioning.
    """
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
            # Resize-then-encode always yields a guide at the latent's spatial dims.
            call_log.append({"encode_to": (latent_height, latent_width)})
            t = torch.zeros(1, 128, 1, latent_height, latent_width, dtype=torch.float32)
            return images, t

        @classmethod
        def get_latent_index(cls, cond, latent_length, guide_length, frame_idx, scale_factors, latent_shape=None):
            time_scale = scale_factors[0]
            latent_idx = (frame_idx + time_scale - 1) // time_scale
            return frame_idx, latent_idx

        @classmethod
        def append_keyframe(cls, positive, negative, frame_idx, latent_image, noise_mask, guiding_latent, strength, scale_factors, **kw):
            call_log.append({"append_frame_idx": frame_idx, "strength": strength})
            new_latent = torch.cat([latent_image, guiding_latent], dim=2)
            mask = torch.full(
                (noise_mask.shape[0], 1, guiding_latent.shape[2],
                 noise_mask.shape[3], noise_mask.shape[4]),
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


def _appends(call_log: list[dict]) -> list[dict]:
    return [c for c in call_log if "append_frame_idx" in c]


class TestExecute:
    def _run(self, *, vae, images, latent, output_fps=25.0, seconds_per_keyframe=1.0, strength=1.0):
        from nodes import KeyframeGuidesFromBatch
        return KeyframeGuidesFromBatch.execute(
            vae=vae, positive=[], negative=[], latent=latent, images=images,
            output_fps=output_fps, seconds_per_keyframe=seconds_per_keyframe,
            strength=strength,
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
        positive, negative = out[0], out[1]
        frames = [c["append_frame_idx"] for c in _appends(fake_nodes_lt)]
        assert positive == frames
        assert negative == frames

    def test_output_latent_carries_noise_mask(self, fake_nodes_lt):
        out = self._run(vae=_FakeVAE(), images=_images(3), latent=_video_latent(64))
        latent = out[2]
        assert "samples" in latent and "noise_mask" in latent

    def test_strength_forwarded(self, fake_nodes_lt):
        self._run(vae=_FakeVAE(), images=_images(2), latent=_video_latent(64), strength=0.4)
        assert all(c["strength"] == pytest.approx(0.4) for c in _appends(fake_nodes_lt))

    def test_keyframes_resized_to_latent_resolution(self, fake_nodes_lt):
        """Keyframes of arbitrary size are accepted — core encode resizes them
        to the latent's pixel dims (latent 8x8 latent -> 256x256 px @ 32x)."""
        # Deliberately odd input resolution; must NOT raise.
        self._run(vae=_FakeVAE(), images=_images(3, hw=(123, 77)), latent=_video_latent(64, hw=(8, 8)))
        encodes = [c for c in fake_nodes_lt if "encode_to" in c]
        assert encodes and all(c["encode_to"] == (8, 8) for c in encodes)

    def test_keyframes_past_latent_truncated(self, fake_nodes_lt):
        self._run(vae=_FakeVAE(), images=_images(50), latent=_video_latent(10))
        # Only the keyframes that fit the 10-frame latent get applied.
        assert len(_appends(fake_nodes_lt)) < 50


class TestRegistration:
    def test_node_registered(self):
        from _node_registry import assert_node_registered
        assert_node_registered("KeyframeGuidesFromBatch")
