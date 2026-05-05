"""Tests for LatentTemporalMask — retake support.

Latent-frame math (per CLAUDE.md "Key patterns"):
  start_latent = int(start_time * fps / 8)
  end_latent   = int(end_time   * fps / 8) + 1   # exclusive, generous
"""

import torch


def _make_latent(
    frames: int = 16,
    batch: int = 1,
    channels: int = 128,
    height: int = 8,
    width: int = 8,
) -> dict:
    return {"samples": torch.zeros(batch, channels, frames, height, width)}


class TestLatentTemporalMask:
    def test_basic_mask_construction(self):
        """Mask is 1.0 inside [start, end] latent frames, 0.0 outside.

        fps=25, 8-to-1 temporal compression.
        start=0.32 → int(0.32*25/8) = int(1.0) = 1
        end=0.96   → int(0.96*25/8)+1 = int(3.0)+1 = 4
        → mask frames [1,2,3] = 1.0
        """
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=16)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.32, end_time=0.96, fps=25.0,
        )
        out_latent = result[0]
        mask = out_latent["noise_mask"]

        assert mask.shape == latent["samples"].shape
        frames = mask[0, 0, :, 0, 0]
        assert frames[0].item() == 0.0
        assert frames[1].item() == 1.0
        assert frames[2].item() == 1.0
        assert frames[3].item() == 1.0
        assert frames[4].item() == 0.0
        assert frames[15].item() == 0.0

    def test_preserves_samples_unmodified(self):
        """The samples tensor must pass through untouched."""
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=8)
        latent["samples"] = torch.randn_like(latent["samples"])
        samples_before = latent["samples"].clone()

        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.0, end_time=1.0, fps=25.0,
        )
        assert torch.equal(result[0]["samples"], samples_before)

    def test_preserves_other_dict_keys(self):
        """Any other keys in the latent dict must survive the mask write."""
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=8)
        latent["batch_index"] = 7
        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.0, end_time=0.32, fps=25.0,
        )
        assert result[0]["batch_index"] == 7

    def test_zero_width_range_all_zeros(self):
        """start == end → nothing regenerates; mask is all zeros."""
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=8)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.5, end_time=0.5, fps=25.0,
        )
        assert result[0]["noise_mask"].sum().item() == 0.0

    def test_negative_start_clamped_to_zero(self):
        """start < 0 is equivalent to start=0 (retake from the very start).

        Tests behavioral equivalence rather than specific frame indices —
        the point is that negative input doesn't break the node or skip
        frames; frame-index math is covered by `test_basic_mask_construction`.
        """
        from nodes import LatentTemporalMask

        result_neg = LatentTemporalMask.execute(
            latent=_make_latent(frames=8), start_time=-1.0, end_time=0.32, fps=25.0,
        )
        result_zero = LatentTemporalMask.execute(
            latent=_make_latent(frames=8), start_time=0.0, end_time=0.32, fps=25.0,
        )
        assert torch.equal(result_neg[0]["noise_mask"], result_zero[0]["noise_mask"])

    def test_end_beyond_duration_clamped(self):
        """end > total duration clamps to last latent frame (retake to-end)."""
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=8)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.0, end_time=100.0, fps=25.0,
        )
        # All 8 frames should be 1.0
        mask = result[0]["noise_mask"]
        assert torch.all(mask[0, 0, :, 0, 0] == 1.0)

    def test_start_beyond_duration_all_zeros(self):
        """start >= total duration → mask is all zeros."""
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=8)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=100.0, end_time=200.0, fps=25.0,
        )
        assert result[0]["noise_mask"].sum().item() == 0.0

    def test_end_less_than_start_all_zeros(self):
        """Reversed range (end < start) is forgiving: all-zero mask.

        Rationale: this is user input from UI widgets. A raise would make
        the workflow crash mid-render; silent zero is safer (nothing
        regenerates, user sees no change and notices the widget bug).
        """
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=8)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=1.0, end_time=0.5, fps=25.0,
        )
        assert result[0]["noise_mask"].sum().item() == 0.0

    def test_overwrites_existing_noise_mask(self):
        """A pre-existing noise_mask on the input must be replaced, not merged.

        Uses end=0.24s so the retake window only covers latent frame 0:
        int(0.24*25/8)+1 = int(0.75)+1 = 1 → mask[0:1] on, mask[1:] off.
        """
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=8)
        latent["noise_mask"] = torch.full_like(latent["samples"], 0.5)

        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.0, end_time=0.24, fps=25.0,
        )
        mask = result[0]["noise_mask"]
        unique_values = torch.unique(mask).tolist()
        assert set(unique_values).issubset({0.0, 1.0})
        assert mask[0, 0, 0, 0, 0].item() == 1.0
        assert mask[0, 0, 1, 0, 0].item() == 0.0

    def test_custom_fps_scales_correctly(self):
        """fps=30 moves the latent-frame boundaries accordingly.

        start=0.27 @ fps=30 → int(0.27*30/8) = int(1.0125) = 1
        end=1.07 @ fps=30  → int(1.07*30/8)+1 = int(4.01)+1 = 5
        → mask frames [1,2,3,4] = 1.0
        """
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=16)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.27, end_time=1.07, fps=30.0,
        )
        frames = result[0]["noise_mask"][0, 0, :, 0, 0]
        assert frames[0].item() == 0.0
        assert frames[1].item() == 1.0
        assert frames[4].item() == 1.0
        assert frames[5].item() == 0.0

    def test_matches_samples_shape_exactly(self):
        """Mask shape matches samples shape [B, C, F, H, W] (no broadcast shortcut).

        Upstream `LTXVSetAudioVideoMaskByTime` uses full-shape masks;
        following suit avoids surprises when downstream code indexes by
        channel or spatial dim.
        """
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=12, batch=2, channels=64, height=16, width=24)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.0, end_time=1.0, fps=25.0,
        )
        assert result[0]["noise_mask"].shape == (2, 64, 12, 16, 24)

    def test_default_yields_hard_mask(self):
        """Without `edge_taper_seconds` the mask is exactly {0.0, 1.0}.

        Regression guard: the soft-taper feature must not change the
        default-call output. Existing retake workflows that don't wire
        the new input must produce bit-identical masks.
        """
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=16)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.32, end_time=0.96, fps=25.0,
        )
        unique = set(torch.unique(result[0]["noise_mask"]).tolist())
        assert unique.issubset({0.0, 1.0})

    def test_explicit_zero_taper_yields_hard_mask(self):
        """edge_taper_seconds=0.0 explicitly is also a hard mask."""
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=16)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.32, end_time=0.96, fps=25.0,
            edge_taper_seconds=0.0,
        )
        unique = set(torch.unique(result[0]["noise_mask"]).tolist())
        assert unique.issubset({0.0, 1.0})

    def test_taper_produces_smooth_ramps(self):
        """edge_taper_seconds > 0 ramps the mask 0->1 at the leading edge,
        1->0 at the trailing edge, with a 1.0 plateau in between.

        fps=25, scale=8 → 1 latent ≈ 0.32s.
        start=0.0, end=8.32 → start_latent=0, end_latent=int(8.32*25/8)+1=27.
        edge_taper_seconds=0.96 → taper_latents = int(0.96*25/8) = 3.

        Expected mask shape (per-frame, broadcast over [B,C,H,W]):
          frames [0, 1, 2]   = strictly increasing values in (0, 1)  (leading ramp)
          frames [3 .. 23]   = 1.0                                   (plateau)
          frames [24, 25, 26]= strictly decreasing values in (0, 1)  (trailing ramp)
          frames [27 .. 31]  = 0.0
        """
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=32)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.0, end_time=8.32, fps=25.0,
            edge_taper_seconds=0.96,
        )
        frames = result[0]["noise_mask"][0, 0, :, 0, 0]

        # Outside retake: zero
        assert frames[27].item() == 0.0
        assert frames[31].item() == 0.0
        # Plateau: full
        assert frames[10].item() == 1.0
        assert frames[20].item() == 1.0
        # Leading ramp: strictly monotone increasing in open (0, 1)
        assert 0.0 < frames[0].item() < frames[1].item() < frames[2].item() < 1.0
        # Trailing ramp: strictly monotone decreasing in open (0, 1)
        assert 1.0 > frames[24].item() > frames[25].item() > frames[26].item() > 0.0

    def test_taper_clamped_to_half_range(self):
        """Excessive taper is clamped so leading and trailing ramps don't overlap.

        retake range = end_latent - start_latent = 6 latents (start=0.32, end=1.92,
        fps=25 → start_latent=1, end_latent=7).
        edge_taper_seconds=10.0 → would compute 31 latents but must clamp to 3
        (= range // 2). With taper=3 the ramps cover [1:4] (leading) and [4:7]
        (trailing) — no overlap, no plateau.
        """
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=16)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.32, end_time=1.92, fps=25.0,
            edge_taper_seconds=10.0,
        )
        mask = result[0]["noise_mask"]
        # Valid range
        assert (mask >= 0.0).all()
        assert (mask <= 1.0).all()
        # Frames outside the retake range stay zero
        assert mask[0, 0, 0, 0, 0].item() == 0.0
        assert mask[0, 0, 7, 0, 0].item() == 0.0
        # Entire retake range [1, 7) is taper (3 leading + 3 trailing, no plateau).
        # Strictly partial values: every frame is in the open interval (0, 1).
        retake_slice = mask[0, 0, 1:7, 0, 0]
        assert (retake_slice > 0.0).all()
        assert (retake_slice < 1.0).all()

    def test_taper_preserves_samples_and_other_keys(self):
        """Soft taper does not modify samples and preserves other dict keys."""
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=16)
        latent["samples"] = torch.randn_like(latent["samples"])
        latent["batch_index"] = 11
        samples_before = latent["samples"].clone()

        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.32, end_time=2.56, fps=25.0,
            edge_taper_seconds=0.32,
        )
        assert torch.equal(result[0]["samples"], samples_before)
        assert result[0]["batch_index"] == 11

    def test_taper_zero_width_range_still_zero(self):
        """Zero-width range stays all-zero even when taper requested."""
        from nodes import LatentTemporalMask

        latent = _make_latent(frames=8)
        result = LatentTemporalMask.execute(
            latent=latent, start_time=0.5, end_time=0.5, fps=25.0,
            edge_taper_seconds=0.32,
        )
        assert result[0]["noise_mask"].sum().item() == 0.0
