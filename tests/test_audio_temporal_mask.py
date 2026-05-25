"""Tests for AudioTemporalMask — partial-freeze noise_mask for AUDIO latents.

Companion to LatentTemporalMask (video). Where the video node maps seconds to
latent frames via a fixed `fps / 8` temporal scale, the audio VAE's latent rate
is not a clean constant (mel hop_length / autoencoder downscale), so this node
derives the rate empirically from the latent's own temporal dim and the known
source duration:

    rate (audio-latent frames / sec) = T / audio_duration_seconds
    start_latent = int(start_time * rate)
    end_latent   = int(end_time   * rate) + 1   # exclusive, generous

Audio latent shape is [B, C, T, F] (rank 4: batch, channels=8, time, mel_bins=16),
distinct from the video latent's [B, C, F, H, W] (rank 5). The node masks along
dim 2 (time) and broadcasts over the rest, so it is rank-agnostic.

Primary use case: AV temporal-extension probe — keep the first N seconds of audio
frozen (context), regenerate the tail. `start_time=N, end_time=audio_duration`.
"""

import torch


def _make_audio_latent(
    time_frames: int = 50,
    batch: int = 1,
    channels: int = 8,
    mel_bins: int = 16,
) -> dict:
    """Audio latent in non-patchified [B, C, T, F] form (LTX audio VAE convention)."""
    return {"samples": torch.zeros(batch, channels, time_frames, mel_bins)}


class TestAudioTemporalMask:
    def test_basic_mask_construction(self):
        """Mask is 1.0 inside [start, end], 0.0 outside, mapped by empirical rate.

        T=50, audio_duration=10s → rate = 5 latent-frames/sec.
        start=2.0 → int(2*5) = 10
        end=10.0  → int(10*5)+1 = 51 → clamped to 50
        → frames [0..9] = 0.0 (frozen prefix), [10..49] = 1.0 (regen tail).
        """
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=50)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=2.0, end_time=10.0, audio_duration_seconds=10.0,
        )
        frames = result[0]["noise_mask"][0, 0, :, 0]
        assert frames[9].item() == 0.0
        assert frames[10].item() == 1.0
        assert frames[49].item() == 1.0

    def test_matches_samples_shape_exactly(self):
        """Mask shape matches the 4D audio samples shape (no broadcast shortcut)."""
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=40, batch=2, channels=8, mel_bins=16)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=0.0, end_time=8.0, audio_duration_seconds=8.0,
        )
        assert result[0]["noise_mask"].shape == (2, 8, 40, 16)

    def test_rate_independent_of_mel_bins(self):
        """The temporal boundary depends only on T and duration, not mel_bins (F)."""
        from nodes import AudioTemporalMask

        a = AudioTemporalMask.execute(
            latent=_make_audio_latent(time_frames=50, mel_bins=16),
            start_time=2.0, end_time=10.0, audio_duration_seconds=10.0,
        )
        b = AudioTemporalMask.execute(
            latent=_make_audio_latent(time_frames=50, mel_bins=32),
            start_time=2.0, end_time=10.0, audio_duration_seconds=10.0,
        )
        # First mel-bin column of the temporal profile is identical.
        assert torch.equal(a[0]["noise_mask"][0, 0, :, 0], b[0]["noise_mask"][0, 0, :, 0])

    def test_preserves_samples_unmodified(self):
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=20)
        latent["samples"] = torch.randn_like(latent["samples"])
        samples_before = latent["samples"].clone()
        result = AudioTemporalMask.execute(
            latent=latent, start_time=1.0, end_time=4.0, audio_duration_seconds=4.0,
        )
        assert torch.equal(result[0]["samples"], samples_before)

    def test_preserves_other_dict_keys(self):
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=20)
        latent["batch_index"] = 7
        result = AudioTemporalMask.execute(
            latent=latent, start_time=0.0, end_time=2.0, audio_duration_seconds=4.0,
        )
        assert result[0]["batch_index"] == 7

    def test_zero_width_range_all_zeros(self):
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=20)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=2.0, end_time=2.0, audio_duration_seconds=4.0,
        )
        assert result[0]["noise_mask"].sum().item() == 0.0

    def test_reversed_range_all_zeros(self):
        """end < start is forgiving (UI widget drift): all-zero mask, no raise."""
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=20)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=3.0, end_time=1.0, audio_duration_seconds=4.0,
        )
        assert result[0]["noise_mask"].sum().item() == 0.0

    def test_negative_start_clamped_to_zero(self):
        from nodes import AudioTemporalMask

        neg = AudioTemporalMask.execute(
            latent=_make_audio_latent(time_frames=20),
            start_time=-1.0, end_time=2.0, audio_duration_seconds=4.0,
        )
        zero = AudioTemporalMask.execute(
            latent=_make_audio_latent(time_frames=20),
            start_time=0.0, end_time=2.0, audio_duration_seconds=4.0,
        )
        assert torch.equal(neg[0]["noise_mask"], zero[0]["noise_mask"])

    def test_end_beyond_duration_clamped(self):
        """end > duration regenerates to the final latent frame (extend-to-end)."""
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=20)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=0.0, end_time=999.0, audio_duration_seconds=4.0,
        )
        assert torch.all(result[0]["noise_mask"][0, 0, :, 0] == 1.0)

    def test_start_beyond_duration_all_zeros(self):
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=20)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=99.0, end_time=200.0, audio_duration_seconds=4.0,
        )
        assert result[0]["noise_mask"].sum().item() == 0.0

    def test_zero_duration_all_zeros(self):
        """audio_duration_seconds <= 0 can't define a rate; degrade to no-op, no raise."""
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=20)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=0.0, end_time=2.0, audio_duration_seconds=0.0,
        )
        assert result[0]["noise_mask"].sum().item() == 0.0

    def test_overwrites_existing_noise_mask(self):
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=50)
        latent["noise_mask"] = torch.full_like(latent["samples"], 0.5)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=2.0, end_time=10.0, audio_duration_seconds=10.0,
        )
        unique = set(torch.unique(result[0]["noise_mask"]).tolist())
        assert unique.issubset({0.0, 1.0})

    def test_default_yields_hard_mask(self):
        """Without edge_taper_seconds the mask is exactly {0.0, 1.0}."""
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=50)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=2.0, end_time=10.0, audio_duration_seconds=10.0,
        )
        unique = set(torch.unique(result[0]["noise_mask"]).tolist())
        assert unique.issubset({0.0, 1.0})

    def test_taper_produces_smooth_ramps(self):
        """edge_taper_seconds > 0 cosine-ramps the mask edges on the 4D audio latent.

        T=100, duration=20s → rate=5/s. start=0, end=20 → [0, 100).
        edge_taper_seconds=1.0 → taper = int(1.0*5) = 5 latent frames.
        Leading ramp [0:5) strictly increasing in (0,1); plateau 1.0; trailing
        ramp strictly decreasing in (0,1).
        """
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=100)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=0.0, end_time=20.0, audio_duration_seconds=20.0,
            edge_taper_seconds=1.0,
        )
        frames = result[0]["noise_mask"][0, 0, :, 0]
        # Leading ramp strictly increasing, strictly partial
        assert 0.0 < frames[0].item() < frames[1].item() < frames[4].item() < 1.0
        # Plateau
        assert frames[50].item() == 1.0
        # Trailing ramp strictly decreasing, strictly partial
        assert 1.0 > frames[97].item() > frames[98].item() > frames[99].item() > 0.0

    def test_taper_preserves_valid_range(self):
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=50)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=2.0, end_time=10.0, audio_duration_seconds=10.0,
            edge_taper_seconds=0.5,
        )
        mask = result[0]["noise_mask"]
        assert (mask >= 0.0).all()
        assert (mask <= 1.0).all()


class TestAudioTemporalMaskInvert:
    """invert=True flips the semantics: [start_time, end_time] becomes the KEPT
    seed window (mask 0), everything else regenerates (mask 1). Lets you pick an
    arbitrary slice of audio as the voice-clone seed, not just the prefix."""

    def test_invert_keeps_window_regenerates_outside(self):
        """T=50, dur=10 -> rate=5. seed window [4,6]s -> latent [20,31).
        invert keeps [20,31)=0, regenerates [0,20) and [31,50)=1."""
        from nodes import AudioTemporalMask

        latent = _make_audio_latent(time_frames=50)
        result = AudioTemporalMask.execute(
            latent=latent, start_time=4.0, end_time=6.0, audio_duration_seconds=10.0,
            invert=True,
        )
        frames = result[0]["noise_mask"][0, 0, :, 0]
        assert frames[0].item() == 1.0    # before seed -> regenerate
        assert frames[19].item() == 1.0
        assert frames[20].item() == 0.0   # seed window -> kept
        assert frames[30].item() == 0.0
        assert frames[31].item() == 1.0   # after seed -> regenerate
        assert frames[49].item() == 1.0

    def test_invert_is_complement_of_default(self):
        """invert=True mask == 1 - (invert=False mask) for the same window."""
        from nodes import AudioTemporalMask

        base = AudioTemporalMask.execute(
            latent=_make_audio_latent(time_frames=50),
            start_time=4.0, end_time=6.0, audio_duration_seconds=10.0,
        )[0]["noise_mask"]
        inv = AudioTemporalMask.execute(
            latent=_make_audio_latent(time_frames=50),
            start_time=4.0, end_time=6.0, audio_duration_seconds=10.0, invert=True,
        )[0]["noise_mask"]
        assert torch.equal(inv, 1.0 - base)

    def test_default_invert_false_unchanged(self):
        """Omitting invert == invert=False (regression guard for existing probes)."""
        from nodes import AudioTemporalMask

        default = AudioTemporalMask.execute(
            latent=_make_audio_latent(time_frames=50),
            start_time=2.0, end_time=10.0, audio_duration_seconds=10.0,
        )[0]["noise_mask"]
        explicit = AudioTemporalMask.execute(
            latent=_make_audio_latent(time_frames=50),
            start_time=2.0, end_time=10.0, audio_duration_seconds=10.0, invert=False,
        )[0]["noise_mask"]
        assert torch.equal(default, explicit)

    def test_invert_degenerate_is_noop(self):
        """A degenerate window (reversed/zero-width/OOB) is an all-zero no-op REGARDLESS
        of invert — a fat-fingered seed window must not silently wipe all audio to
        regenerate-from-scratch. Invert only flips a VALID window."""
        from nodes import AudioTemporalMask

        for start, end in ((5.0, 5.0), (6.0, 3.0), (99.0, 200.0)):
            result = AudioTemporalMask.execute(
                latent=_make_audio_latent(time_frames=20),
                start_time=start, end_time=end, audio_duration_seconds=10.0, invert=True,
            )
            assert result[0]["noise_mask"].sum().item() == 0.0, (start, end)


def test_audio_temporal_mask_registered():
    from _node_registry import assert_node_registered

    assert_node_registered("AudioTemporalMask")
