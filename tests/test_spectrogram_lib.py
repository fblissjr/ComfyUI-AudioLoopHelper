"""Tests for spectrogram-to-reference core functions.

Exercises the pure-function core of `scripts/spectrogram_to_reference.py`
(`compute_mel_log`, `prepare_mel_for_render`, `render_frame`,
`render_spectrogram_frame`, `frame_count_for`, `time_bin_for_frame`).
CLI behavior is exercised manually.
"""

import numpy as np
import pytest

from spectrogram_to_reference import (
    compute_mel_log,
    frame_count_for,
    prepare_mel_for_render,
    render_frame,
    render_spectrogram_frame,
    time_bin_for_frame,
)


@pytest.fixture
def synthetic_audio():
    """5 seconds of 440 Hz sine at 22050 Hz -- stable, predictable
    spectrogram with one dominant Mel bin."""
    sr = 22050
    duration = 5.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    return audio, sr


class TestComputeMelLog:
    def test_returns_2d_array(self, synthetic_audio):
        audio, sr = synthetic_audio
        mel = compute_mel_log(audio, sr=sr, n_mels=64, hop_length=512)
        assert mel.ndim == 2
        assert mel.shape[0] == 64

    def test_time_bins_scale_with_hop_length(self, synthetic_audio):
        audio, sr = synthetic_audio
        mel_512 = compute_mel_log(audio, sr=sr, n_mels=64, hop_length=512)
        mel_1024 = compute_mel_log(audio, sr=sr, n_mels=64, hop_length=1024)
        # Larger hop -> fewer time bins (roughly 2x)
        assert mel_512.shape[1] > mel_1024.shape[1]
        assert abs(mel_512.shape[1] / mel_1024.shape[1] - 2.0) < 0.1

    def test_log_scale_compresses_range(self, synthetic_audio):
        audio, sr = synthetic_audio
        mel_linear = compute_mel_log(audio, sr=sr, n_mels=64, log_scale=False)
        mel_log = compute_mel_log(audio, sr=sr, n_mels=64, log_scale=True)
        assert mel_linear.std() > mel_log.std()


class TestRenderSpectrogramFrame:
    def test_returns_uint8_rgb(self, synthetic_audio):
        """Each frame renders as H x W x 3 uint8 -- the format
        LTXAddVideoICLoRAGuide.image expects after torch conversion."""
        audio, sr = synthetic_audio
        mel = compute_mel_log(audio, sr=sr, n_mels=64)
        frame = render_spectrogram_frame(
            mel, time_idx=10, window_bins=40, resolution=(448, 832), mode="normalized",
        )
        assert frame.dtype == np.uint8
        assert frame.shape == (448, 832, 3)

    def test_contrast_normalized_mode_in_range(self, synthetic_audio):
        audio, sr = synthetic_audio
        mel = compute_mel_log(audio, sr=sr, n_mels=64)
        frame = render_spectrogram_frame(
            mel, time_idx=10, window_bins=40, resolution=(256, 256), mode="normalized",
        )
        assert 0 <= frame.min()
        assert frame.max() <= 255

    def test_sliding_window_shows_past_context(self, synthetic_audio):
        """At time_idx T with window_bins W, the visible mel slice covers
        [T-W, T]; frames late in the clip differ from frames at time_idx=0."""
        audio, sr = synthetic_audio
        mel = compute_mel_log(audio, sr=sr, n_mels=64)
        frame_early = render_spectrogram_frame(
            mel, time_idx=5, window_bins=40, resolution=(128, 128), mode="normalized",
        )
        frame_late = render_spectrogram_frame(
            mel, time_idx=mel.shape[1] - 5, window_bins=40, resolution=(128, 128), mode="normalized",
        )
        assert not np.array_equal(frame_early, frame_late)

    def test_blur_mode_smooths(self, synthetic_audio):
        """`blurred` mode reduces total variation vs `normalized`."""
        audio, sr = synthetic_audio
        mel = compute_mel_log(audio, sr=sr, n_mels=64)
        sharp = render_spectrogram_frame(
            mel, time_idx=20, window_bins=40, resolution=(128, 128), mode="normalized",
        )
        blurred = render_spectrogram_frame(
            mel, time_idx=20, window_bins=40, resolution=(128, 128), mode="blurred",
            blur_sigma=2.0,
        )
        def total_variation(img):
            return float(np.abs(np.diff(img.astype(np.int32), axis=0)).sum() +
                         np.abs(np.diff(img.astype(np.int32), axis=1)).sum())
        assert total_variation(blurred) < total_variation(sharp)

    def test_unknown_mode_raises(self, synthetic_audio):
        audio, sr = synthetic_audio
        mel = compute_mel_log(audio, sr=sr, n_mels=64)
        with pytest.raises(ValueError, match="mode"):
            render_spectrogram_frame(
                mel, time_idx=10, window_bins=40, resolution=(64, 64), mode="banana",  # type: ignore[arg-type]
            )


class TestGlobalNormalizationPipeline:
    def test_prepared_mel_is_uint8(self, synthetic_audio):
        """`prepare_mel_for_render` returns the flipped uint8 array that
        the CLI slices per-frame."""
        audio, sr = synthetic_audio
        mel = compute_mel_log(audio, sr=sr, n_mels=64)
        prepared = prepare_mel_for_render(mel, mode="blurred", blur_sigma=1.5)
        assert prepared.dtype == np.uint8
        assert prepared.shape == mel.shape  # same H, W; just preprocessed + flipped

    def test_render_frame_on_prepared_matches_wrapper(self, synthetic_audio):
        """Prepared-once + render_frame-many-times produces the SAME
        bytes as the convenience wrapper (which re-prepares per call)."""
        audio, sr = synthetic_audio
        mel = compute_mel_log(audio, sr=sr, n_mels=64)
        prepared = prepare_mel_for_render(mel, mode="normalized")

        via_wrapper = render_spectrogram_frame(
            mel, time_idx=30, window_bins=40, resolution=(64, 64), mode="normalized",
        )
        via_primitives = render_frame(prepared, time_idx=30, window_bins=40, resolution=(64, 64))
        assert np.array_equal(via_wrapper, via_primitives)


class TestFrameCountFromDuration:
    def test_frame_count_matches_fps_times_duration(self):
        assert frame_count_for(duration_seconds=5.0, fps=25.0) == 125
        assert frame_count_for(duration_seconds=19.88, fps=25.0) == 497

    def test_latent_alignment_frame_count(self):
        """With align=True, frame count satisfies (n-1) % 8 == 0."""
        n = frame_count_for(duration_seconds=10.0, fps=25.0, align_ltx_latent=True)
        assert (n - 1) % 8 == 0
        assert n >= 250
        n_naive = frame_count_for(duration_seconds=10.0, fps=25.0, align_ltx_latent=False)
        assert n_naive == 250


class TestTimeIndexForFrame:
    def test_linear_map(self):
        sr = 22050
        hop = 512
        # Frame 25 at 25 fps = 1.0s = 22050/512 ≈ 43 bins
        assert time_bin_for_frame(frame_idx=25, fps=25.0, sr=sr, hop_length=hop) == pytest.approx(43, abs=1)
        assert time_bin_for_frame(frame_idx=0, fps=25.0, sr=sr, hop_length=hop) == 0
