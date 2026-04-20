"""Tests for AudioLoopController stride quantization invariant.

The invariant: audio stride per iteration MUST equal the number of
pixel frames the video decoder emits per iteration, divided by fps.
Otherwise lip-sync drifts as iterations accumulate.

The fix (2026-04-20): compute stride from integer-latent counts rather
than (window_seconds - overlap_seconds). See AudioLoopController.execute
for the full derivation. These tests lock in the invariant.
"""

import pytest
import torch

from nodes import AudioLoopController, LTX_TEMPORAL_SCALE


def _run(
    *,
    window_seconds: float,
    overlap_seconds: float,
    audio_duration: float = 600.0,
    fps: int = 25,
    current_iteration: int = 1,
    seed: int = 0,
):
    """Invoke the node with a minimal synthetic audio tensor.

    Only the duration matters — `AudioLoopController.execute` reads
    `shape[-1] / sample_rate`. A low sample_rate keeps the tensor tiny
    (100 Hz = 60k floats for 600s vs 26M at 44.1kHz).
    """
    sample_rate = 100
    waveform = torch.zeros(1, 1, int(audio_duration * sample_rate))
    audio = {"waveform": waveform, "sample_rate": sample_rate}
    result = AudioLoopController.execute(
        current_iteration=current_iteration,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        audio=audio,
        seed=seed,
        fps=fps,
    )
    # io.NodeOutput packs positional values; unpack via the result attribute.
    out = result.result if hasattr(result, "result") else result
    # NodeOutput returns a tuple-like of the 8 outputs in schema order:
    # (start_index, should_stop, audio_duration, iteration_seed,
    #  stride, overlap_frames, overlap_latent_frames, overlap_seconds)
    return {
        "start_index": out[0],
        "should_stop": out[1],
        "audio_duration": out[2],
        "iteration_seed": out[3],
        "stride_seconds": out[4],
        "overlap_frames": out[5],
        "overlap_latent_frames": out[6],
        "overlap_seconds": out[7],
    }


class TestStrideMatchesDecodedPixelFrames:
    """Audio stride must equal the per-iteration video pixel advance / fps.

    If this fails, lip-sync drifts in accumulating iterations.
    """

    @pytest.mark.parametrize("overlap_seconds", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    def test_stride_aligned_to_8_pixel_boundary(self, overlap_seconds):
        """`stride_seconds * fps` must be an integer multiple of 8.

        In a concatenated decoded sequence, every latent after the first
        of the whole tensor contributes 8 pixels. Per-iter video advance
        = new_latent_frames * 8 pixels. Stride in seconds must match
        that exactly.
        """
        out = _run(window_seconds=19.88, overlap_seconds=overlap_seconds)
        stride_pixel_frames = out["stride_seconds"] * 25
        pixel_frames_int = int(round(stride_pixel_frames))
        # Must round cleanly back to an integer (binary-float tolerance).
        assert abs(stride_pixel_frames - pixel_frames_int) < 1e-6
        assert pixel_frames_int % LTX_TEMPORAL_SCALE == 0

    def test_stride_matches_new_latents_times_eight(self):
        """stride * fps == (window_latents - overlap_latents) * 8."""
        window_seconds = 19.88
        overlap_seconds = 2.0
        fps = 25
        window_pixel_frames = round(window_seconds * fps)  # 497
        overlap_pixel_frames = round(overlap_seconds * fps)  # 50
        window_latents = (window_pixel_frames - 1) // LTX_TEMPORAL_SCALE + 1  # 63
        overlap_latents = (overlap_pixel_frames - 1) // LTX_TEMPORAL_SCALE + 1  # 7
        expected_new_latents = window_latents - overlap_latents  # 56
        expected_stride_pixels = expected_new_latents * LTX_TEMPORAL_SCALE  # 448

        out = _run(window_seconds=window_seconds, overlap_seconds=overlap_seconds)
        assert out["stride_seconds"] * fps == pytest.approx(expected_stride_pixels)


class TestNoDriftOverManyIterations:
    """Cumulative audio position must equal cumulative video pixel position.

    Simulates N iterations and confirms audio start_index tracks video
    decoder output exactly. Before the fix, overlap=4 accumulated ~1.3s
    drift over 10 iterations.
    """

    @pytest.mark.parametrize(
        "overlap_seconds,n_iters",
        [(0.0, 10), (1.0, 10), (2.0, 10), (3.0, 11), (4.0, 11), (5.0, 12)],
    )
    def test_zero_cumulative_drift(self, overlap_seconds, n_iters):
        fps = 25
        window_seconds = 19.88
        window_pixel_frames = round(window_seconds * fps)  # 497
        window_latents = (window_pixel_frames - 1) // LTX_TEMPORAL_SCALE + 1  # 63

        # Pull stride and overlap_latents from the node.
        out = _run(window_seconds=window_seconds, overlap_seconds=overlap_seconds)
        stride = out["stride_seconds"]
        overlap_latents = out["overlap_latent_frames"]
        new_latents = window_latents - overlap_latents

        # Simulate the decoder assembly for N iterations.
        # Iter 0 contributes the initial render: full window_latents latents,
        # decoded as the sequence start → (window_latents - 1) * 8 + 1 pixels.
        # Iter 1..N each append (new_latents) latents into the concatenation,
        # contributing exactly new_latents * 8 pixels to the decoded output.
        video_pixels_after_init = (window_latents - 1) * LTX_TEMPORAL_SCALE + 1
        video_pixels_after_iter_n = (
            video_pixels_after_init + n_iters * new_latents * LTX_TEMPORAL_SCALE
        )
        video_seconds_after_iter_n = video_pixels_after_iter_n / fps

        # Audio position after N iterations: the (N+1)-th iter's window would
        # start at (N+1) * stride. Correspondingly, iter N consumes audio
        # from N*stride to N*stride + window_seconds.
        # End of iter N's audio window = N * stride + window_seconds
        audio_seconds_after_iter_n = n_iters * stride + window_seconds

        # Video should cover exactly the same duration as the audio consumed
        # up through iter N's end.
        assert video_seconds_after_iter_n == pytest.approx(
            audio_seconds_after_iter_n, abs=1e-9
        )


class TestStartIndexUsesQuantizedStride:
    """start_index for iter N must be N * (quantized stride), not N * (widget stride)."""

    def test_iter_1_start_index_at_overlap_2(self):
        out = _run(window_seconds=19.88, overlap_seconds=2.0, current_iteration=1)
        # Quantized stride = 56 latents * 8 / 25 = 17.92s
        assert out["stride_seconds"] == pytest.approx(17.92)
        assert out["start_index"] == pytest.approx(17.92)

    def test_iter_5_start_index_at_overlap_4(self):
        out = _run(window_seconds=19.88, overlap_seconds=4.0, current_iteration=5)
        # Quantized stride = 50 latents * 8 / 25 = 16.0s (cleanly aligned)
        assert out["stride_seconds"] == pytest.approx(16.0)
        assert out["start_index"] == pytest.approx(80.0)


class TestEffectiveOverlapMatchesTarget:
    """The overlap_seconds OUTPUT reflects the quantized value, not the widget.

    Widget overlap_seconds is the user's target; output overlap_seconds
    is what actually happens after latent quantization.
    """

    def test_overlap_2_widget_produces_effective_1_96s(self):
        out = _run(window_seconds=19.88, overlap_seconds=2.0)
        # window=497 pixels, stride=448 pixels, effective overlap=49 pixels = 1.96s
        assert out["overlap_frames"] == 49
        assert out["overlap_seconds"] == pytest.approx(1.96)

    def test_overlap_4_widget_produces_effective_3_88s(self):
        out = _run(window_seconds=19.88, overlap_seconds=4.0)
        # window=497, stride=400, effective overlap=97 pixels = 3.88s
        assert out["overlap_frames"] == 97
        assert out["overlap_seconds"] == pytest.approx(3.88)

    def test_aligned_combo_stays_clean(self):
        """window=20.48, overlap=2.56 is already 8-pixel aligned.

        Outputs should match widget inputs exactly (no rounding shift).
        """
        out = _run(window_seconds=20.48, overlap_seconds=2.56)
        # window=512 pixels=64 latents, overlap=64 pixels=8 latents,
        # new=56 latents → stride=448 pixels = 17.92s
        assert out["overlap_frames"] == 64
        assert out["overlap_seconds"] == pytest.approx(2.56)
        assert out["stride_seconds"] == pytest.approx(17.92)


class TestEdgeCases:
    def test_zero_overlap_works(self):
        out = _run(window_seconds=19.88, overlap_seconds=0.0)
        # No overlap: all window latents are new.
        # window_latents=63, overlap_latents=0, new=63, stride=63*8/25 = 20.16s
        assert out["overlap_latent_frames"] == 0
        assert out["stride_seconds"] == pytest.approx(20.16)

    def test_overlap_equal_to_window_clamped(self):
        """Overlap >= window should clamp to leave >=1 new latent frame."""
        out = _run(window_seconds=19.88, overlap_seconds=19.88)
        # window_latents=63. Must leave at least 1 new latent.
        assert out["overlap_latent_frames"] == 62
        # Stride = 1 * 8 / 25 = 0.32s
        assert out["stride_seconds"] == pytest.approx(0.32)
