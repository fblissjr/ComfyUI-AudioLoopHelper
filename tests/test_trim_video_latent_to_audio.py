"""Behavioral tests for TrimVideoLatentToAudio.

Last updated: 2026-05-10

The latent-level companion to TrimImageBatchToAudio (F14). Clips the
video LATENT's temporal dim to a count that — when LTX VAE decoded —
produces at most ``int(audio_duration * fps)`` pixel frames. Saves
VAE decode work on overshoot frames; F14's image-batch trim remains
the safety net for any off-by-one.

LTX video VAE temporal convention: ``pixel_frames = (latent_frames - 1) * 8 + 1``.
So for a target pixel count ``P`` we snap down to the largest valid
``P' = ((P - 1) // 8) * 8 + 1`` and emit ``L = (P' - 1) // 8 + 1``
latent frames.
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
    """Canonical bug case: audio=166.733s, fps=25 → target_pixel=4168 →
    snap_down to (4167//8)*8+1 = 4161 → 521 latent frames.

    Loop emits ~567 latent frames (overshoot of 46 latents = 368 pixels
    = 14.7s). Trim should bring it down to 521.
    """
    latent = _make_latent(567)
    audio = _make_audio(166.733)
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    trimmed_samples = out[0]["samples"]
    assert trimmed_samples.shape[2] == 521, (
        f"expected 521 latent frames, got {trimmed_samples.shape[2]}"
    )


def test_passthrough_when_latent_shorter_than_audio():
    latent = _make_latent(50)
    audio = _make_audio(166.733)  # would target 521 latent frames
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    assert out[0]["samples"].shape[2] == 50


def test_passthrough_when_latent_exactly_matches_target():
    latent = _make_latent(521)
    audio = _make_audio(166.733)
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    assert out[0]["samples"].shape[2] == 521


def test_zero_duration_audio_keeps_at_least_one_latent_frame():
    """Defensive: empty latent would crash downstream VAE decode."""
    latent = _make_latent(100)
    audio = _make_audio(0.0)
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    assert out[0]["samples"].shape[2] >= 1


def test_snap_down_to_valid_ltx_boundary():
    """For target_pixel=4168 the LTX (L-1)%8==0 constraint snaps DOWN
    to 4161 pixels (521 latents). Going UP to 4169 (522 latents) would
    overshoot audio by 1 pixel — F14 would catch it but we don't want
    to depend on that."""
    latent = _make_latent(1000)
    audio = _make_audio(166.733)
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    decoded_pixel_count = (out[0]["samples"].shape[2] - 1) * 8 + 1
    target = int(166.733 * 25)
    assert decoded_pixel_count <= target, (
        f"decoded {decoded_pixel_count} exceeds target {target}"
    )


def test_preserves_dtype_and_non_temporal_dims():
    latent = {"samples": torch.zeros(1, 128, 567, 60, 104, dtype=torch.bfloat16)}
    audio = _make_audio(166.733)
    out = TrimVideoLatentToAudio.execute(latent=latent, audio=audio, fps=25)
    s = out[0]["samples"]
    assert s.shape == (1, 128, 521, 60, 104)
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
