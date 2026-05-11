"""Behavioral tests for TrimVideoLatentToAudio.

Last updated: 2026-05-10

The latent-level companion to TrimImageBatchToAudio (F14). Clips the
video LATENT's temporal dim to the SMALLEST count whose decoded pixel
count >= ``int(audio_duration * fps)``. Saves VAE decode work on the
overshoot frames; F14's image-batch trim then clips the residual 0-7
pixel-frame overshoot to exact audio length.

Snap UP, not down: snap-DOWN clips up to 7 pixel-frames (0.28s @ 25fps)
of audio at the END because ffmpeg ``-shortest`` clips audio when
video < audio. User-reported audio loss on 2026-05-10 forced this
change after the A/B couldn't distinguish "audio naturally this length"
from "audio clipped by short video."

LTX video VAE temporal convention: ``pixel_frames = (latent_frames - 1) * 8 + 1``.
"""

from __future__ import annotations

import torch

from nodes import TrimVideoLatentToAudio


def _make_audio(seconds: float, sample_rate: int = 44100) -> dict:
    return {
        "waveform": torch.zeros(1, 1, int(round(seconds * sample_rate))),
        "sample_rate": sample_rate,
    }


def _make_latent(latent_frames: int, c: int = 128, h: int = 60, w: int = 104) -> dict:
    return {"samples": torch.zeros(1, c, latent_frames, h, w)}


def test_clips_when_latent_longer_than_audio():
    """Canonical bug case: audio=166.733s, fps=25 → target_pixel=4168.
    Snap UP: ceil((4168-1)/8) + 1 = 521 + 1 = 522 latent frames.
    Decoded = (522-1)*8+1 = 4169 pixels, 1 pixel over target. F14
    handles the 1-pixel residue.

    Loop emits ~567 latent frames. Trim brings it down to 522.
    """
    latent = _make_latent(567)
    audio = _make_audio(166.733)
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    trimmed_samples = out[0]["samples"]
    assert trimmed_samples.shape[2] == 522, (
        f"expected 522 latent frames, got {trimmed_samples.shape[2]}"
    )


def test_passthrough_when_latent_shorter_than_audio():
    latent = _make_latent(50)
    audio = _make_audio(166.733)  # would target 521 latent frames
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    assert out[0]["samples"].shape[2] == 50


def test_passthrough_when_latent_exactly_matches_target():
    """At the snap-UP boundary (522 latents → 4169 pixels), trim is no-op."""
    latent = _make_latent(522)
    audio = _make_audio(166.733)
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    assert out[0]["samples"].shape[2] == 522


def test_zero_duration_audio_keeps_at_least_one_latent_frame():
    """Defensive: empty latent would crash downstream VAE decode."""
    latent = _make_latent(100)
    audio = _make_audio(0.0)
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    assert out[0]["samples"].shape[2] >= 1


def test_snap_up_to_valid_ltx_boundary():
    """Snap UP — decoded pixel count must be >= target so audio survives
    ffmpeg -shortest (which would otherwise clip audio to match a
    too-short video). F14 image-trim downstream handles the small
    residue (0-7 pixel frames)."""
    latent = _make_latent(1000)
    audio = _make_audio(166.733)
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    decoded_pixel_count = (out[0]["samples"].shape[2] - 1) * 8 + 1
    target = int(166.733 * 25)
    assert decoded_pixel_count >= target, (
        f"decoded {decoded_pixel_count} short of target {target}; "
        "audio would clip under -shortest"
    )
    # Residue must stay within one latent boundary (max overshoot < 8 pixels)
    assert decoded_pixel_count - target < 8


def test_preserves_dtype_and_non_temporal_dims():
    latent = {"samples": torch.zeros(1, 128, 567, 60, 104, dtype=torch.bfloat16)}
    audio = _make_audio(166.733)
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    s = out[0]["samples"]
    assert s.shape == (1, 128, 522, 60, 104)
    assert s.dtype == torch.bfloat16


def test_preserves_extra_keys_in_latent_dict():
    """`samples` is what we slice; other keys (e.g. noise_mask in some
    pipelines) should ride through unchanged."""
    latent = {
        "samples": torch.zeros(1, 128, 567, 60, 104),
        "extra_marker": "audit",
    }
    audio = _make_audio(166.733)
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    assert out[0].get("extra_marker") == "audit"


def test_node_is_registered_in_extension():
    from _node_registry import assert_node_registered
    assert_node_registered("TrimVideoLatentToAudio")
