"""Tests for LatentTemporalMask — retake support.

`LatentTemporalMask` writes a noise_mask to a latent so that only a
time range regenerates on a re-sample pass. Canonical use: the user
finishes a run, sees one bad section, loads the accumulated latent,
sets `[start_time, end_time]` on the bad range, and re-samples —
only the masked region changes.

Mask semantics (per CLAUDE.md "Critical constraints"):
  - `noise_mask == 1.0` → frame regenerates from noise
  - `noise_mask == 0.0` → frame stays fixed (context)

Latent-frame math (per CLAUDE.md "Key patterns"):
  - `latent_frame = int(time_seconds * fps / 8)` (start, inclusive)
  - `end_latent_frame = int(end_time * fps / 8) + 1` (exclusive)
  - `LTX_TEMPORAL_SCALE = 8` is the LTX 2.3 VAE's temporal compression.
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
