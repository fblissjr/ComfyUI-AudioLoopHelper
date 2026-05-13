"""Behavioral tests for LTXHeadTrim.

Last updated: 2026-05-12

Composite IMAGE + AUDIO head-trim. Drops the first
``trim_latent_frames * LTX_TEMPORAL_SCALE`` (= 8) pixel frames from
the image batch and the matching audio span from the waveform,
keeping the saved mp4 in lockstep.

Use case: LTX 2.3 i2v "filler" frames at clip start. The model
spends 0.5-2s easing out of the init image; head-trim discards
that window after sampling. Opt-in via the ``trim_latent_frames``
widget; default 0 = no-op pass-through.
"""

from __future__ import annotations

import torch

from nodes import LTX_TEMPORAL_SCALE, LTXHeadTrim
from _node_registry import assert_node_registered


def _make_audio(seconds: float, sample_rate: int = 44100) -> dict:
    samples = int(round(seconds * sample_rate))
    waveform = torch.zeros(1, 1, samples)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _make_images(n_frames: int, h: int = 4, w: int = 4) -> torch.Tensor:
    # Use unique per-frame values so we can verify the right slice survived.
    return torch.arange(n_frames).float().reshape(n_frames, 1, 1, 1).expand(n_frames, h, w, 3).clone()


def test_trim_zero_is_passthrough():
    images = _make_images(100)
    audio = _make_audio(4.0)
    out_img, out_aud = LTXHeadTrim.execute(
        images=images, audio=audio, trim_latent_frames=0, fps=25,
    )
    assert torch.equal(out_img, images)
    assert torch.equal(out_aud["waveform"], audio["waveform"])
    assert out_aud["sample_rate"] == audio["sample_rate"]


def test_trim_one_latent_frame_drops_8_pixel_frames():
    images = _make_images(100)
    audio = _make_audio(4.0, sample_rate=25000)  # 100k samples
    out_img, out_aud = LTXHeadTrim.execute(
        images=images, audio=audio, trim_latent_frames=1, fps=25,
    )
    assert out_img.shape[0] == 100 - LTX_TEMPORAL_SCALE
    # The image at index 0 of the output should be the image that was
    # at index 8 of the input (LTX_TEMPORAL_SCALE).
    assert out_img[0, 0, 0, 0].item() == 8.0
    # 8 pixel frames / 25 fps = 0.32 s. At 25000 Hz that's 8000 samples.
    expected_samples = 100000 - 8000
    assert out_aud["waveform"].shape[-1] == expected_samples


def test_trim_n_latent_frames_drops_n_times_temporal_scale():
    images = _make_images(200)
    audio = _make_audio(8.0, sample_rate=16000)
    out_img, out_aud = LTXHeadTrim.execute(
        images=images, audio=audio, trim_latent_frames=4, fps=25,
    )
    assert out_img.shape[0] == 200 - 4 * LTX_TEMPORAL_SCALE
    # 32 pixel frames / 25 fps = 1.28 s. At 16000 Hz = 20480 samples.
    expected_samples = 8 * 16000 - int(round(32 / 25 * 16000))
    assert out_aud["waveform"].shape[-1] == expected_samples


def test_trim_at_or_beyond_batch_clamps_to_last_frame():
    images = _make_images(10)
    audio = _make_audio(1.0)
    # trim_latent_frames * 8 = 80 pixel frames, way larger than 10.
    out_img, out_aud = LTXHeadTrim.execute(
        images=images, audio=audio, trim_latent_frames=10, fps=25,
    )
    # Should clamp -- keep last frame, drop nearly all audio.
    assert out_img.shape[0] == 1
    assert out_img[0, 0, 0, 0].item() == 9.0  # last input frame
    # Audio should also be at-most-near-empty (clamped to keep something).
    assert out_aud["waveform"].shape[-1] >= 0


def test_preserves_dtype_and_other_dims():
    images = torch.rand(50, 64, 96, 3, dtype=torch.float16)
    audio = _make_audio(2.0)
    out_img, _ = LTXHeadTrim.execute(
        images=images, audio=audio, trim_latent_frames=2, fps=25,
    )
    assert out_img.dtype == torch.float16
    assert out_img.shape == (50 - 16, 64, 96, 3)


def test_audio_sample_rate_preserved():
    images = _make_images(40)
    for sr in (16000, 22050, 44100, 48000):
        audio = _make_audio(2.0, sample_rate=sr)
        _, out_aud = LTXHeadTrim.execute(
            images=images, audio=audio, trim_latent_frames=2, fps=25,
        )
        assert out_aud["sample_rate"] == sr


def test_fps_affects_audio_trim_only():
    """Pixel-trim is fixed at trim_latent_frames * 8; only the audio-trim
    duration scales with fps. Faster fps -> less audio trimmed."""
    images = _make_images(50)
    audio_25 = _make_audio(2.0, sample_rate=10000)
    audio_50 = _make_audio(2.0, sample_rate=10000)

    out_img_25, out_aud_25 = LTXHeadTrim.execute(
        images=images, audio=audio_25, trim_latent_frames=2, fps=25,
    )
    out_img_50, out_aud_50 = LTXHeadTrim.execute(
        images=images, audio=audio_50, trim_latent_frames=2, fps=50,
    )
    # Image trim is identical regardless of fps (it's in pixel-frame units).
    assert out_img_25.shape[0] == out_img_50.shape[0]
    # Audio trim halves when fps doubles.
    aud_trim_25 = audio_25["waveform"].shape[-1] - out_aud_25["waveform"].shape[-1]
    aud_trim_50 = audio_50["waveform"].shape[-1] - out_aud_50["waveform"].shape[-1]
    assert aud_trim_25 == 2 * aud_trim_50


def test_single_frame_input_passes_through():
    """Edge case: a 1-frame image batch with trim>0 must still emit at
    least one frame (VHS_VideoCombine errors on empty batches).
    Exercises the `max(0, shape[0]-1)` floor in the clamp."""
    images = _make_images(1)
    audio = _make_audio(0.04)
    out_img, _ = LTXHeadTrim.execute(
        images=images, audio=audio, trim_latent_frames=4, fps=25,
    )
    assert out_img.shape[0] == 1


def test_fps_zero_is_passthrough():
    """fps=0 from a wired INTConstant (schema min=1 doesn't constrain
    link-fed values) must not ZeroDivisionError."""
    images = _make_images(40)
    audio = _make_audio(2.0)
    out_img, out_aud = LTXHeadTrim.execute(
        images=images, audio=audio, trim_latent_frames=4, fps=0,
    )
    assert torch.equal(out_img, images)
    assert torch.equal(out_aud["waveform"], audio["waveform"])


def test_node_is_registered_in_extension():
    assert_node_registered("LTXHeadTrim")
