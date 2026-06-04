"""Tests for KeyframeFillLength — sizes the empty latent so a time-spaced
keyframe batch never under-fits.

Companion to `KeyframeGuidesTimeSpaced`: that node only WARNS (and drops the
keyframe) once a keyframe's frame index falls past the latent end. This node
computes the PIXEL length up front so the drop can't exist — wire its `length`
output into `EmptyLTXVLatentVideo.length`.

Test surfaces:
- `_required_pixel_length` is the pure sizing seam (no torch / no ComfyUI). Its
  last-keyframe placement math MUST match `_keyframe_guide_placements`
  (keyframe i at `round(i * spk * fps)`).
- `execute()` reads batch_size from `images.shape[0]`; tested with a tiny tensor.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------
# Pure sizing helper
# --------------------------------------------------------------------------
class TestRequiredPixelLength:
    def _len(
        self,
        *,
        batch_size: int,
        output_fps: float = 25.0,
        seconds_per_keyframe: float = 1.0,
        tail_seconds: float = 0.0,
    ) -> int:
        from nodes import _required_pixel_length
        return _required_pixel_length(
            batch_size=batch_size,
            output_fps=output_fps,
            seconds_per_keyframe=seconds_per_keyframe,
            tail_seconds=tail_seconds,
        )

    def test_returns_valid_8n_plus_1(self):
        # Every output must satisfy the temporal grid constraint.
        for bs in range(0, 20):
            length = self._len(batch_size=bs)
            assert (length - 1) % 8 == 0, (bs, length)

    def test_worked_example_ten_keyframes_one_per_second_25fps(self):
        # last keyframe i=9 -> round(9*1*25)=225 px -> +1 = 226 -> snap UP to 233.
        assert self._len(batch_size=10, output_fps=25.0, seconds_per_keyframe=1.0) == 233

    def test_last_target_matches_placement_math(self):
        # The sizing helper's last-keyframe pixel frame must equal what
        # _keyframe_guide_placements would compute for the same params, so the
        # latent is sized to hold exactly that keyframe (no drop).
        from nodes import _keyframe_guide_placements
        bs, fps, spk = 7, 24.0, 0.7
        length = self._len(batch_size=bs, output_fps=fps, seconds_per_keyframe=spk, tail_seconds=0.0)
        placements = _keyframe_guide_placements(
            batch_size=bs, n_latent_frames=length, output_fps=fps,
            seconds_per_keyframe=spk, temporal_scale=8,
        )
        # All keyframes fit (none dropped) once the latent is this long.
        assert len(placements) == bs
        last_px = placements[-1][1]
        # length is the smallest valid 8n+1 >= last_px + 1.
        assert length >= last_px + 1
        assert length - 8 < last_px + 1  # not over-sized by a whole grid step

    def test_batch_size_one_is_first_frame_plus_tail(self):
        # last px = round(0*...) = 0 -> +1 = 1 -> snap UP to 9 (minimal valid).
        assert self._len(batch_size=1) == 9

    def test_batch_size_zero_returns_minimal_valid_length(self):
        # Degenerate: no keyframes. Must not crash; returns the minimal valid 9.
        assert self._len(batch_size=0) == 9

    def test_tail_default_is_no_op(self):
        # tail_seconds=0.0 must not change the result vs omitting it.
        assert self._len(batch_size=10, tail_seconds=0.0) == self._len(batch_size=10)

    def test_tail_adds_room(self):
        # 10 kf -> base 233. tail 1s @ 25fps = +25 px -> 226+25=251 -> snap UP.
        # 251-1=250, 250%8=2 -> next 8n+1 is 257.
        assert self._len(batch_size=10, tail_seconds=1.0) == 257

    def test_already_valid_length_unchanged(self):
        # Construct params landing exactly on an 8n+1 boundary: want last_px+1=249.
        # last_px=248 -> round(i*spk*fps)=248. i=8, spk=1.24, fps=25 -> 8*1.24*25=248.
        assert self._len(batch_size=9, output_fps=25.0, seconds_per_keyframe=1.24) == 249

    def test_snap_is_up_never_down(self):
        # Required raw length always <= returned length (never truncates, which
        # would re-introduce the drop KeyframeGuidesTimeSpaced warns about).
        for bs in range(1, 30):
            length = self._len(batch_size=bs)
            last_px = round((bs - 1) * 1.0 * 25.0)
            assert length >= last_px + 1


# --------------------------------------------------------------------------
# execute()
# --------------------------------------------------------------------------
def _images(n: int, hw: tuple[int, int] = (64, 64)) -> torch.Tensor:
    return torch.zeros(n, hw[0], hw[1], 3, dtype=torch.float32)


class TestExecute:
    def _run(self, *, images, output_fps=25.0, seconds_per_keyframe=1.0, tail_seconds=0.0):
        from nodes import KeyframeFillLength
        return KeyframeFillLength.execute(
            images=images, output_fps=output_fps,
            seconds_per_keyframe=seconds_per_keyframe, tail_seconds=tail_seconds,
        )

    def test_reads_batch_size_from_images(self):
        out = self._run(images=_images(10))
        assert out[0] == 233

    def test_output_is_int(self):
        out = self._run(images=_images(5))
        assert isinstance(out[0], int)

    def test_empty_batch_does_not_crash(self):
        out = self._run(images=_images(0))
        assert out[0] == 9

    def test_matches_pure_helper(self):
        from nodes import _required_pixel_length
        out = self._run(images=_images(13), output_fps=24.0, seconds_per_keyframe=0.5, tail_seconds=0.4)
        expected = _required_pixel_length(
            batch_size=13, output_fps=24.0, seconds_per_keyframe=0.5, tail_seconds=0.4,
        )
        assert out[0] == expected


class TestRegistration:
    def test_node_registered(self):
        from _node_registry import assert_node_registered
        assert_node_registered("KeyframeFillLength")
