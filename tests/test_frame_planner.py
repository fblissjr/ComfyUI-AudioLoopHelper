"""Tests for LTXFramePlanner and AudioLoopPlanner.max_iterations."""

import pytest
import torch

from nodes import (
    AudioLoopPlanner,
    LTXFramePlanner,
    _snap_dimensions,
    _snap_frames,
)


def _audio(duration_s: float = 600.0, sr: int = 100):
    return {
        "waveform": torch.zeros(1, 1, int(duration_s * sr)),
        "sample_rate": sr,
    }


class TestSnapDimensions:
    """target_width/target_height -> div-32-snapped (w, h). Snap DOWN to bias
    toward latent-volume safety (matches existing _compute_ltx_resolution
    convention)."""

    def test_already_aligned(self):
        assert _snap_dimensions(832, 448) == (832, 448)

    def test_round_down_when_above_grid(self):
        # 833 -> 832 (snap down), 449 -> 448
        assert _snap_dimensions(833, 449) == (832, 448)

    def test_portrait_dimensions_preserved(self):
        # 9:16 portrait shape
        assert _snap_dimensions(448, 832) == (448, 832)

    def test_floor_at_minimum(self):
        # tiny values still produce a valid div-32 result
        w, h = _snap_dimensions(50, 50)
        assert w == 32 and h == 32

    def test_high_res(self):
        assert _snap_dimensions(960, 544) == (960, 544)


class TestSnapFrames:
    """target_seconds + fps -> (frames, actual_seconds). frames satisfies
    (frames - 1) % 8 == 0 (LTX video VAE temporal constraint). Snap DOWN
    to bias toward smaller chunks (less VRAM, more re-anchoring)."""

    @pytest.mark.parametrize(
        "target_seconds,fps,expected_frames,expected_actual",
        [
            (20.0, 25, 497, 19.88),     # canonical default
            (10.0, 25, 249, 9.96),       # half-window
            (5.0, 25, 121, 4.84),        # short chunk
            (8.0, 24, 185, 7.708333),    # 24fps variant
            (1.0, 25, 25, 1.0),          # very short
        ],
    )
    def test_canonical_values(self, target_seconds, fps, expected_frames, expected_actual):
        frames, actual = _snap_frames(target_seconds, fps)
        assert frames == expected_frames
        assert actual == pytest.approx(expected_actual, rel=1e-3)
        # Invariant
        assert (frames - 1) % 8 == 0

    def test_actual_seconds_never_exceeds_target(self):
        # Snap DOWN means actual_seconds <= target_seconds
        for target in (5.0, 10.0, 19.88, 20.0, 25.0, 30.0):
            _, actual = _snap_frames(target, 25)
            assert actual <= target + 1e-6

    def test_minimum_one_frame(self):
        # At very small target, frames=1 (the minimum valid)
        frames, _ = _snap_frames(0.01, 25)
        assert frames == 1


class TestLTXFramePlanner:
    """End-to-end node behavior."""

    def test_default_canonical_setup(self):
        out = LTXFramePlanner.execute(
            target_width=832,
            target_height=448,
            target_seconds=20.0,
            fps=25,
        )
        # NodeOutput is tuple-like in test harness
        width, height, frames, actual_seconds, fps_int, fps_float, latent_volume, status, summary = (
            out[i] if hasattr(out, "__getitem__") else out.result[i] for i in range(9)
        )
        assert width == 832
        assert height == 448
        assert frames == 497
        assert actual_seconds == pytest.approx(19.88, rel=1e-3)
        assert fps_int == 25
        assert fps_float == pytest.approx(25.0)
        assert latent_volume == 22932
        assert "OK" in status or "NEAR_EDGE" in status
        # Summary is human-readable: must mention key values
        assert "832" in summary and "448" in summary
        assert "19.88" in summary or "19.9" in summary
        assert "497" in summary
        assert "25" in summary

    def test_user_typed_off_grid_dimensions(self):
        # User typed 833 x 449 — should snap silently to 832 x 448 in summary
        out = LTXFramePlanner.execute(
            target_width=833,
            target_height=449,
            target_seconds=20.0,
            fps=25,
        )
        width, height = out[0], out[1]
        assert width == 832 and height == 448

    def test_high_res_over_ceiling(self):
        # 960 x 544 x 497 = 32130 — over the 24570 ceiling
        out = LTXFramePlanner.execute(
            target_width=960,
            target_height=544,
            target_seconds=20.0,
            fps=25,
        )
        latent_volume = out[6]
        status = out[7]
        assert latent_volume > 24570
        assert "OVER" in status or "ERR" in status

    def test_high_res_with_short_window_under_ceiling(self):
        # 960 x 544 x 121 = 8160 — under the 20K OK threshold
        out = LTXFramePlanner.execute(
            target_width=960,
            target_height=544,
            target_seconds=5.0,
            fps=25,
        )
        latent_volume = out[6]
        status = out[7]
        assert latent_volume == 8160
        assert "OK" in status


class TestAudioLoopPlannerMaxIterations:
    """max_iterations widget — debug-only override of the auto-computed
    iteration count. Default 0 = auto (current behavior); >0 caps."""

    def test_default_zero_means_auto(self):
        # Default should match unset-max behavior
        out = AudioLoopPlanner.execute(
            audio=_audio(60.0),
            window_seconds=19.88,
            overlap_seconds=2.0,
            fps=25,
            max_iterations=0,
        )
        iters = out[1] if hasattr(out, "__getitem__") else out.result[1]
        assert iters >= 1  # auto-computed

    def test_cap_at_three(self):
        # Long audio (600s) would normally yield many iters; cap at 3
        out = AudioLoopPlanner.execute(
            audio=_audio(600.0),
            window_seconds=19.88,
            overlap_seconds=2.0,
            fps=25,
            max_iterations=3,
        )
        iters = out[1] if hasattr(out, "__getitem__") else out.result[1]
        assert iters == 3

    def test_cap_higher_than_auto_does_not_inflate(self):
        # Short audio (10s) auto-yields 1 iter; setting max=99 shouldn't
        # inflate it — we cap, never extend
        out = AudioLoopPlanner.execute(
            audio=_audio(10.0),
            window_seconds=19.88,
            overlap_seconds=2.0,
            fps=25,
            max_iterations=99,
        )
        iters = out[1] if hasattr(out, "__getitem__") else out.result[1]
        # auto would be 1 (10s audio < 19.88s window)
        assert iters <= 99
        assert iters == 1  # cap doesn't extend

    def test_summary_mentions_cap(self):
        # When capped, summary should make it visible to the user
        out = AudioLoopPlanner.execute(
            audio=_audio(600.0),
            window_seconds=19.88,
            overlap_seconds=2.0,
            fps=25,
            max_iterations=3,
        )
        summary = out[0] if hasattr(out, "__getitem__") else out.result[0]
        assert "max_iterations" in summary or "capped" in summary.lower() or "cap" in summary.lower()
