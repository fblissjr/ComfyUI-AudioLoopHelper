"""Tests for LTX 2.3 resolution helper.

Reference for the latent-volume math + thresholds:
docs/reference/ltx23_model_reference.md §"Resolution and latent volume" and
docs/reference/frame_planner_reference.md §"Latent-volume classification".

The helper enforces div-by-32 width/height (LTX single-stage requirement) and
(frames-1)%8==0 (video VAE temporal rule) — the ONLY hard model-side
constraints. It then computes latent volume (W/32)*(H/32)*((F-1)/8 + 1) and
classifies it as a **VRAM/render-cost advisory** (NOT a quality cliff): there is
no hard latent-volume ceiling. The anchor is LTX-2's own HQ production default
(960x544 @ 497 = 32,130 latent tokens, coderef/LTX-2/.../utils/constants.py
LTX_2_3_HQ_PARAMS). <= that = OK; above = HIGH_VRAM advisory.
"""

import pytest

from nodes import (
    _compute_ltx_resolution,
    _classify_latent_volume,
    _LTX_HQ_PRODUCTION_VOLUME,
)


class TestComputeLtxResolution:
    def test_16_9_landscape_832_long_edge(self):
        """16:9 at target_long_edge=832, 497 frames -> 832x448. 22,932 tokens is
        well within the HQ production budget -> OK."""
        w, h, vol, status = _compute_ltx_resolution(16 / 9, 832, 497, "landscape")
        assert (w, h) == (832, 448)
        assert vol == 26 * 14 * 63  # 22,932
        assert status.startswith("OK")

    def test_960x544_hq_default_is_ok(self):
        """The shipped resolution 960x544 @ 497 = 32,130 is exactly LTX-2's HQ
        production default and MUST classify OK (regression guard against the
        old 24,570 'ceiling' that wrongly flagged the shipped config)."""
        w, h, vol, status = _compute_ltx_resolution(960 / 544, 960, 497, "landscape")
        assert (w, h) == (960, 544)
        assert vol == 30 * 17 * 63  # 32,130
        assert vol == _LTX_HQ_PRODUCTION_VOLUME
        assert status.startswith("OK")

    def test_704x704_under_hq_default_is_ok(self):
        """704x704 @ 497 = 30,492 is under the HQ production volume -> OK (was
        wrongly OVER_EDGE under the old artifact-ceiling thresholds)."""
        w, h, vol, status = _compute_ltx_resolution(1.0, 704, 497, "square")
        assert (w, h) == (704, 704)
        assert vol == 22 * 22 * 63  # 30,492
        assert status.startswith("OK")

    def test_832x480_is_ok(self):
        """832x480 @ 497 = 24,570 — the old docs called this 'at the edge', but
        it is under the HQ production volume and renders fine -> OK."""
        w, h, vol, status = _compute_ltx_resolution(832 / 480, 832, 497, "landscape")
        assert (w, h) == (832, 480)
        assert vol == 26 * 15 * 63  # 24,570
        assert status.startswith("OK")

    def test_1216_long_edge_landscape_high_vram(self):
        """Higher-res 16:9 at long_edge=1216 -> 50,274 tokens, above the HQ
        production default -> HIGH_VRAM advisory (more VRAM, not a quality cliff)."""
        w, h, vol, status = _compute_ltx_resolution(16 / 9, 1216, 497, "landscape")
        assert (w, h) == (1216, 672)  # snap-down 1216/1.778=684 -> 672
        assert vol == 38 * 21 * 63  # 50,274
        assert status.startswith("HIGH_VRAM")

    def test_portrait_swaps_long_edge_to_height(self):
        w, h, vol, status = _compute_ltx_resolution(16 / 9, 832, 497, "portrait")
        assert (w, h) == (448, 832)
        assert vol == 14 * 26 * 63  # 22,932 — symmetric to landscape
        assert status.startswith("OK")

    def test_square_orientation_ignores_aspect(self):
        w, h, _, _ = _compute_ltx_resolution(16 / 9, 512, 497, "square")
        assert w == h == 512

    def test_target_long_edge_not_div_32_snaps_up(self):
        """Caller passes 850; helper snaps to next div-32 boundary (864)."""
        w, h, _, _ = _compute_ltx_resolution(16 / 9, 850, 497, "landscape")
        assert w == 864  # 832 < 850 < 864
        assert h % 32 == 0

    def test_invalid_frames_raises(self):
        """(frames - 1) % 8 must equal 0 (LTX video VAE temporal formula) — this
        IS a hard constraint, unlike latent volume."""
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


class TestClassifyLatentVolume:
    """Direct boundary tests on the classifier (the VRAM advisory, not a cliff)."""

    def test_exactly_hq_production_volume_is_ok(self):
        # 960x544x497 = 32,130 = the anchor; boundary is inclusive.
        vol, status = _classify_latent_volume(960, 544, 497)
        assert vol == _LTX_HQ_PRODUCTION_VOLUME
        assert status.startswith("OK")

    def test_one_step_above_anchor_is_high_vram(self):
        # 992x544x497 = 31*17*63 = 33,201 > 32,130 -> HIGH_VRAM.
        vol, status = _classify_latent_volume(992, 544, 497)
        assert vol > _LTX_HQ_PRODUCTION_VOLUME
        assert status.startswith("HIGH_VRAM")

    def test_status_never_claims_artifact_ceiling(self):
        """Regression: the status wording is a VRAM advisory, not an artifact
        cliff. No 'edge'/'ceiling'/'artifact' language."""
        for status in (
            _classify_latent_volume(832, 448, 497)[1],
            _classify_latent_volume(1216, 672, 497)[1],
        ):
            low = status.lower()
            assert "edge" not in low
            assert "ceiling" not in low
            assert "artifact" not in low
