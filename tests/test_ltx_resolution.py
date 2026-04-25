"""Tests for LTX 2.3 resolution helper.

Reference for the latent-volume math + thresholds:
docs/reference/ltx23_model_reference.md §"Resolution and latent volume".

The helper enforces div-by-32 width/height (LTX single-stage requirement)
and (frames-1)%8==0, then computes the latent volume
(W/32)*(H/32)*((F-1)/8 + 1) and classifies it against the doc-authoritative
ceiling — OK <= 20000 < NEAR_EDGE <= 24570 < OVER_EDGE.
"""

import pytest

from nodes import _compute_ltx_resolution


class TestComputeLtxResolution:
    def test_16_9_landscape_832_long_edge(self):
        """16:9 at target_long_edge=832, 497 frames -> 832x448 (cinema-style
        snap-down). 22,932 is in NEAR_EDGE territory per the doc thresholds —
        users render this routinely without artifacts but it's not the safest
        operating point."""
        w, h, vol, status = _compute_ltx_resolution(16 / 9, 832, 497, "landscape")
        assert (w, h) == (832, 448)
        assert vol == 26 * 14 * 63  # 22,932
        assert status.startswith("NEAR_EDGE")

    def test_704x704_committed_canonical_is_over_edge(self):
        """Sanity: the committed-canonical 704x704 at 497 frames is over the
        artifact ceiling per the reference doc."""
        w, h, vol, status = _compute_ltx_resolution(1.0, 704, 497, "square")
        assert (w, h) == (704, 704)
        assert vol == 22 * 22 * 63  # 30,492
        assert status.startswith("OVER_EDGE")

    def test_832x480_at_edge_classified_near_edge(self):
        """The reference doc names 832x480 as 'already at the edge' (24,570).
        That sits in NEAR_EDGE territory (20001-24570)."""
        w, h, vol, status = _compute_ltx_resolution(832 / 480, 832, 497, "landscape")
        assert (w, h) == (832, 480)
        assert vol == 26 * 15 * 63  # 24,570
        assert status.startswith("NEAR_EDGE")

    def test_1216_long_edge_landscape_over_edge(self):
        """Higher-res 16:9 stretch at long_edge=1216 is well over the ceiling
        regardless of snap direction."""
        w, h, vol, status = _compute_ltx_resolution(16 / 9, 1216, 497, "landscape")
        assert (w, h) == (1216, 672)  # snap-down 1216/1.778=684 -> 672
        assert vol == 38 * 21 * 63  # 50,274
        assert status.startswith("OVER_EDGE")

    def test_portrait_swaps_long_edge_to_height(self):
        w, h, vol, status = _compute_ltx_resolution(16 / 9, 832, 497, "portrait")
        assert (w, h) == (448, 832)
        assert vol == 14 * 26 * 63  # 22,932 — symmetric to landscape
        assert status.startswith("NEAR_EDGE")

    def test_square_orientation_ignores_aspect(self):
        w, h, _, _ = _compute_ltx_resolution(16 / 9, 512, 497, "square")
        assert w == h == 512

    def test_target_long_edge_not_div_32_snaps_up(self):
        """Caller passes 850; helper snaps to next div-32 boundary (864)."""
        w, h, _, _ = _compute_ltx_resolution(16 / 9, 850, 497, "landscape")
        assert w == 864  # 832 < 850 < 864
        assert h % 32 == 0

    def test_invalid_frames_raises(self):
        """(frames - 1) % 8 must equal 0 (LTX video VAE temporal formula)."""
        with pytest.raises(AssertionError, match="frames"):
            _compute_ltx_resolution(16 / 9, 832, 496, "landscape")  # 495 % 8 = 7

    def test_short_edge_clamps_to_32_minimum(self):
        """Very wide aspect ratios shouldn't produce 0-height output."""
        w, h, _, _ = _compute_ltx_resolution(20.0, 640, 497, "landscape")
        assert h >= 32
        assert h % 32 == 0

    def test_status_includes_volume(self):
        """Status string is structured so callers can parse the volume."""
        _, _, vol, status = _compute_ltx_resolution(16 / 9, 832, 497, "landscape")
        assert str(vol) in status
