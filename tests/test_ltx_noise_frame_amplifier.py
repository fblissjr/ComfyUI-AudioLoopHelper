"""Behavioral tests for LTXNoiseFrameAmplifier.

Last updated: 2026-05-11

The node wraps a NOISE generator and multiplies the first N temporal
frames of the generated noise tensor by a scalar amplifier. Equivalent
to giving those frames a higher initial sigma without touching the
sigma schedule -- a non-uniform-sampling effect achieved by perturbing
the input noise, not the sampler.

Use case: LTX 2.3 i2v init-image filler problem. Higher noise on
early latent frames pushes the model to do more denoising work there
-> more divergence from the init-image temporal prior -> motion
starts earlier.
"""

from __future__ import annotations

import torch

from nodes import LTXNoiseFrameAmplifier


class FakeFixedNoise:
    """Test fake: generate_noise returns a constant-filled tensor."""

    def __init__(self, value: float = 1.0, seed: int = 42):
        self.seed = seed
        self._value = value

    def generate_noise(self, input_latent: dict) -> torch.Tensor:
        latent = input_latent["samples"]
        return torch.full(latent.shape, self._value, dtype=latent.dtype)


def _video_latent(t: int = 10, c: int = 4, h: int = 8, w: int = 8) -> dict:
    return {"samples": torch.zeros(1, c, t, h, w)}


def test_amplifies_first_n_frames():
    wrapped = LTXNoiseFrameAmplifier.execute(
        noise=FakeFixedNoise(value=1.0),
        n_frames=3,
        amplifier=2.0,
    )[0]
    out = wrapped.generate_noise(_video_latent(t=10))
    # First 3 frames doubled.
    assert torch.allclose(out[:, :, :3], torch.full_like(out[:, :, :3], 2.0))
    # Frames 3..9 unchanged (still 1.0).
    assert torch.allclose(out[:, :, 3:], torch.full_like(out[:, :, 3:], 1.0))


def test_amplifier_one_is_identity():
    wrapped = LTXNoiseFrameAmplifier.execute(
        noise=FakeFixedNoise(value=0.7),
        n_frames=5,
        amplifier=1.0,
    )[0]
    out = wrapped.generate_noise(_video_latent(t=10))
    assert torch.allclose(out, torch.full_like(out, 0.7))


def test_n_frames_zero_is_identity():
    wrapped = LTXNoiseFrameAmplifier.execute(
        noise=FakeFixedNoise(value=0.5),
        n_frames=0,
        amplifier=3.0,
    )[0]
    out = wrapped.generate_noise(_video_latent(t=10))
    assert torch.allclose(out, torch.full_like(out, 0.5))


def test_n_frames_exceeds_tensor_amplifies_all_safely():
    """Asking for n_frames=99 on a T=10 tensor amplifies all of T, no crash."""
    wrapped = LTXNoiseFrameAmplifier.execute(
        noise=FakeFixedNoise(value=1.0),
        n_frames=99,
        amplifier=2.0,
    )[0]
    out = wrapped.generate_noise(_video_latent(t=10))
    assert torch.allclose(out, torch.full_like(out, 2.0))


def test_seed_pass_through():
    wrapped = LTXNoiseFrameAmplifier.execute(
        noise=FakeFixedNoise(seed=12345),
        n_frames=2,
        amplifier=1.5,
    )[0]
    assert wrapped.seed == 12345


def test_amplifier_less_than_one_attenuates():
    """amplifier < 1.0 reduces noise on early frames -- inverse use case
    (anchor MORE strongly to init image rather than less). Not the typical
    flow but the math should still work."""
    wrapped = LTXNoiseFrameAmplifier.execute(
        noise=FakeFixedNoise(value=1.0),
        n_frames=3,
        amplifier=0.25,
    )[0]
    out = wrapped.generate_noise(_video_latent(t=10))
    assert torch.allclose(out[:, :, :3], torch.full_like(out[:, :, :3], 0.25))
    assert torch.allclose(out[:, :, 3:], torch.full_like(out[:, :, 3:], 1.0))


def test_node_is_registered_in_extension():
    from _node_registry import assert_node_registered
    assert_node_registered("LTXNoiseFrameAmplifier")
