"""Tests for nodes_audio_latent_slice.

Covers the slicing math (rate inference, index computation, edge cases)
and the node's execute() seam (noise_mask passthrough, dict preservation).

GPU not required — the node is pure-tensor temporal slicing on a fixed-
shape audio latent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from nodes_audio_latent_slice import (  # noqa: E402
    AudioLatentSlice,
    _compute_slice_indices,
    _infer_latent_rate,
    _slice_latent,
)


# -----------------------------------------------------------------------------
# _infer_latent_rate
# -----------------------------------------------------------------------------

class TestInferLatentRate:
    def test_basic(self):
        # 100 frames over 10 seconds = 10 fps
        assert _infer_latent_rate(100, 10.0) == 10.0

    def test_fractional(self):
        # 50 frames over 17.92 seconds (canonical iclora window)
        assert _infer_latent_rate(50, 17.92) == pytest.approx(2.79, abs=0.01)

    def test_zero_seconds_raises(self):
        with pytest.raises(ValueError, match="source_seconds must be > 0"):
            _infer_latent_rate(100, 0.0)

    def test_negative_seconds_raises(self):
        with pytest.raises(ValueError, match="source_seconds must be > 0"):
            _infer_latent_rate(100, -1.0)

    def test_zero_temporal_dim_raises(self):
        with pytest.raises(ValueError, match="latent_temporal_dim must be > 0"):
            _infer_latent_rate(0, 10.0)


# -----------------------------------------------------------------------------
# _compute_slice_indices
# -----------------------------------------------------------------------------

class TestComputeSliceIndices:
    def test_full_range(self):
        # 100 frames / 10s = 10 fps; window [0, 10] = full range
        s, e = _compute_slice_indices(100, 10.0, 0.0, 10.0)
        assert (s, e) == (0, 100)

    def test_first_quarter(self):
        # 100 / 10s = 10 fps; window [0, 2.5] = [0, 25]
        s, e = _compute_slice_indices(100, 10.0, 0.0, 2.5)
        assert (s, e) == (0, 25)

    def test_middle_window(self):
        # 100 / 10s = 10 fps; window [3, 6] = [30, 60]
        s, e = _compute_slice_indices(100, 10.0, 3.0, 3.0)
        assert (s, e) == (30, 60)

    def test_iclora_canonical_window(self):
        # Realistic case: ~3 latents/sec audio rate, 17.92s window starting at 0
        # 300 latents / 100s = 3 fps; window [0, 17.92] = [0, 54]
        s, e = _compute_slice_indices(300, 100.0, 0.0, 17.92)
        assert s == 0
        assert e == 54  # round(17.92 * 3) = 54

    def test_iclora_canonical_window_offset(self):
        # Per-iter advance: start=15.84, duration=17.92 (overlap of 2s on a stride)
        s, e = _compute_slice_indices(300, 100.0, 15.84, 17.92)
        assert s == round(15.84 * 3)  # = 48
        assert e == s + 54

    def test_clamp_start_beyond_end(self):
        # Asking for window past the end → clamped to last frame, never empty
        s, e = _compute_slice_indices(100, 10.0, 100.0, 5.0)
        assert s == 99  # clamped to len - 1
        assert e == 100  # at least 1 frame
        assert e > s

    def test_clamp_negative_start(self):
        # _compute_slice_indices doesn't get negative start_seconds in the
        # supported topology (video_start_time is >= 0), but defensive math:
        s, e = _compute_slice_indices(100, 10.0, -5.0, 3.0)
        # round(-5 * 10) = -50 → clamped to 0
        assert s == 0
        assert e > s

    def test_zero_duration_yields_one_frame(self):
        # If caller asks for 0 duration, return at least 1 frame so
        # downstream nodes don't get an empty tensor.
        s, e = _compute_slice_indices(100, 10.0, 5.0, 0.0)
        assert e - s == 1


# -----------------------------------------------------------------------------
# _slice_latent
# -----------------------------------------------------------------------------

class TestSliceLatent:
    def _make_audio_latent(self, t: int, c: int = 8, b: int = 1) -> torch.Tensor:
        """Synthetic [B, C, T] audio-latent shape. Distinct values per
        temporal position so we can verify slicing returns the right
        sub-range."""
        return torch.arange(t, dtype=torch.float32).view(1, 1, t).expand(b, c, t).contiguous()

    def test_basic_slice_returns_correct_range(self):
        latent = self._make_audio_latent(t=100)
        sliced = _slice_latent(latent, source_seconds=10.0, start_seconds=2.0, duration_seconds=3.0)
        # 100 / 10s = 10 fps; window [2, 5] = [20, 50]; expect 30 frames
        assert sliced.shape == (1, 8, 30)
        # First value should be 20, last should be 49
        assert sliced[0, 0, 0].item() == 20.0
        assert sliced[0, 0, -1].item() == 49.0

    def test_preserves_non_temporal_dims(self):
        latent = self._make_audio_latent(t=100, c=4, b=2)
        sliced = _slice_latent(latent, source_seconds=10.0, start_seconds=0.0, duration_seconds=10.0)
        assert sliced.shape == (2, 4, 100)

    def test_higher_dim_audio_latent(self):
        # Some VAEs encode to [B, C, T, H, W] (e.g. mel-spec encoders).
        # Slicing must only touch dim 2.
        latent = torch.randn(1, 8, 100, 16, 8)
        sliced = _slice_latent(latent, 10.0, 0.0, 5.0)
        assert sliced.shape == (1, 8, 50, 16, 8)

    def test_too_few_dims_raises(self):
        # Audio latent must be at least 3D
        latent = torch.randn(100)
        with pytest.raises(ValueError, match=r"audio latent must have shape"):
            _slice_latent(latent, 10.0, 0.0, 5.0)

    def test_slice_at_end_clamps_safely(self):
        latent = self._make_audio_latent(t=100)
        sliced = _slice_latent(latent, source_seconds=10.0, start_seconds=15.0, duration_seconds=3.0)
        # Out-of-range start → clamped, returns at least 1 frame
        assert sliced.shape[2] >= 1


# -----------------------------------------------------------------------------
# AudioLatentSlice.execute
# -----------------------------------------------------------------------------

class TestExecuteSeam:
    def test_returns_dict_with_sliced_samples(self):
        latent_in = {"samples": torch.arange(100, dtype=torch.float32).view(1, 1, 100)}
        out = AudioLatentSlice.execute(
            latent_in, source_seconds=10.0, start_seconds=2.0, duration_seconds=3.0,
        )
        # io.NodeOutput stub returns the args tuple
        out_dict = out[0]
        assert "samples" in out_dict
        assert out_dict["samples"].shape == (1, 1, 30)

    def test_preserves_noise_mask_passthrough(self):
        latent_in = {
            "samples": torch.randn(1, 8, 100),
            "noise_mask": torch.randn(1, 1, 100),
        }
        out = AudioLatentSlice.execute(latent_in, 10.0, 0.0, 10.0)
        out_dict = out[0]
        # noise_mask is passed through (canonical workflow strips it before
        # audio chains; this is documented in the node's execute()).
        assert "noise_mask" in out_dict
        # Samples are sliced
        assert out_dict["samples"].shape == (1, 8, 100)

    def test_preserves_arbitrary_non_samples_keys(self):
        latent_in = {
            "samples": torch.randn(1, 8, 100),
            "batch_index": [0],
            "extra_metadata": {"foo": "bar"},
        }
        out = AudioLatentSlice.execute(latent_in, 10.0, 0.0, 5.0)
        out_dict = out[0]
        assert out_dict["batch_index"] == [0]
        assert out_dict["extra_metadata"] == {"foo": "bar"}
