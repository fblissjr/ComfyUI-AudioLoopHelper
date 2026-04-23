"""Tests for LoopConfigValidator rules.

Each rule gets a pass and fail case. The pure `_build_report` function is
the test surface; execute() is a thin wrapper that reads AUDIO duration.
"""

import pytest

from nodes_validation import (
    _bracket,
    _build_report,
    _reachable_overlaps,
    _widget_range_for_overlap_px,
)

from nodes import LTX_TEMPORAL_SCALE


def _run(
    *,
    audio_duration: float = 154.28,
    window_seconds: float = 9.96,
    overlap_seconds: float = 2.0,
    fps: int = 25,
    length: int = 0,
    width: int = 0,
    height: int = 0,
    schedule: str = "",
    resolution_rule: str = "div_by_32",
    seam_tolerance_seconds: float = 0.2,
    keyframe_schedule: str = "",
    keyframe_batch_size: int = 0,
):
    return _build_report(
        audio_duration=audio_duration,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
        fps=fps,
        length=length,
        width=width,
        height=height,
        schedule=schedule,
        resolution_rule=resolution_rule,
        seam_tolerance_seconds=seam_tolerance_seconds,
        keyframe_schedule=keyframe_schedule,
        keyframe_batch_size=keyframe_batch_size,
    )


class TestBracket:
    def test_length_lattice_valid_passes_through(self):
        # step=8 offset=1 -> LTX length lattice (1, 9, 17, ...).
        assert _bracket(249, LTX_TEMPORAL_SCALE, offset=1) == (249, 249)
        assert _bracket(1, LTX_TEMPORAL_SCALE, offset=1) == (1, 1)

    def test_length_lattice_invalid_returns_bracket(self):
        assert _bracket(247, LTX_TEMPORAL_SCALE, offset=1) == (241, 249)
        assert _bracket(250, LTX_TEMPORAL_SCALE, offset=1) == (249, 257)

    def test_divisibility_exact(self):
        assert _bracket(832, 32) == (832, 832)
        assert _bracket(832, 64) == (832, 832)

    def test_divisibility_non_multiple(self):
        assert _bracket(830, 32) == (800, 832)
        assert _bracket(900, 64) == (896, 960)


class TestReachableOverlaps:
    def test_short_window(self):
        # window=9.96s, fps=25 -> window_px=249, window_latents=32
        # reachable = [(249 - n*8)/25 for n in 1..31 if >=0]
        values = _reachable_overlaps(249, 25)
        assert len(values) == 31
        assert values[0] == pytest.approx(0.04)  # n=31: 1px / 25
        # Next value up is 9/25 = 0.36
        assert values[1] == pytest.approx(0.36)
        # Values around 2.0s:
        assert any(abs(v - 1.96) < 0.001 for v in values)
        assert any(abs(v - 2.28) < 0.001 for v in values)


class TestWidgetRangeForOverlap:
    def test_band_contains_user_widget_value(self):
        # overlap_px=49 corresponds to overlap_latents=(49-1)//8+1 = 7.
        # Any overlap_px in [41..48] is latents=6; [49..56] is latents=7.
        # widget range for latents=7 band: overlap_px must round to 49..56.
        lo, hi = _widget_range_for_overlap_px(49, 25)
        # Widget 2.0 -> round(2.0*25)=50 -> latents=7 -> should be in band.
        assert lo <= 2.0 <= hi
        # Widget 1.96 -> round(1.96*25)=49 -> latents=7 -> in band.
        assert lo <= 1.96 <= hi


class TestMathBlock:
    def test_shows_stride_and_effective_overlap(self):
        report, ok, _, errs, stride = _run(window_seconds=9.96, overlap_seconds=2.0)
        assert "stride_seconds        = 200/25 = 8.0000" in report
        assert "effective_overlap_sec = 49/25 = 1.9600" in report
        assert stride == pytest.approx(8.0)
        # No length/res/schedule given, so only the target-vs-effective warning
        # (and context-ratio warning for 9.96s short window) should fire.
        assert errs == 0

    def test_default_window_19_88(self):
        report, ok, _, errs, stride = _run(
            window_seconds=19.88, overlap_seconds=2.0, audio_duration=154.28
        )
        assert "stride_seconds        = 448/25 = 17.9200" in report
        assert stride == pytest.approx(17.92)
        assert errs == 0


class TestLengthValidity:
    def test_valid_length_passes(self):
        report, _, _, errs, _ = _run(length=249)
        assert "[OK]   length=249 valid" in report
        assert errs == 0

    def test_invalid_length_errors(self):
        report, ok, _, errs, _ = _run(length=247, window_seconds=9.88)
        assert "[ERROR]" in report
        assert "length=247 invalid" in report
        assert "Fix: set length=241 or 249" in report
        assert errs >= 1
        assert ok is False


class TestLengthVsWindowAgreement:
    def test_matching_passes(self):
        report, _, _, _, _ = _run(length=249, window_seconds=9.96)
        assert "length == window*fps: 249 == round(9.96*25)" in report

    def test_mismatch_warns(self):
        # length=497 but window=9.96s -> window*fps=249, mismatch.
        report, _, warns, _, _ = _run(length=497, window_seconds=9.96)
        assert "length (497) != window*fps (249)" in report
        assert warns >= 1


class TestResolutionDivisibility:
    def test_832x448_div_by_32(self):
        report, _, _, errs, _ = _run(width=832, height=448, resolution_rule="div_by_32")
        assert "resolution 832x448 divisible by 32" in report
        assert errs == 0

    def test_832x448_div_by_64(self):
        report, _, _, errs, _ = _run(width=832, height=448, resolution_rule="div_by_64")
        assert "resolution 832x448 divisible by 64" in report
        assert errs == 0

    def test_830x448_fails_32(self):
        report, ok, _, errs, _ = _run(width=830, height=448, resolution_rule="div_by_32")
        assert "not divisible by 32" in report
        assert errs >= 1
        assert ok is False


class TestEffectiveOverlapDelta:
    def test_off_target_warns_with_reachable_values(self):
        # window=9.96, target=2.0 -> effective 1.96s, delta -0.04.
        report, _, warns, _, _ = _run(window_seconds=9.96, overlap_seconds=2.0)
        assert "effective overlap 1.960s != target 2.000s" in report
        assert "Reachable values" in report
        assert warns >= 1

    def test_on_target_passes(self):
        # overlap=1.96 exactly -> no delta.
        report, _, _, _, _ = _run(window_seconds=9.96, overlap_seconds=1.96)
        assert "effective overlap matches target" in report


class TestContextRatio:
    def test_thin_ratio_short_window_warns(self):
        # 7/32 = 22% > 15% threshold, so shouldn't fire at overlap=2.0.
        # Push to overlap=1.0 -> 4 latents / 32 = 12.5% -> should fire.
        report, _, warns, _, _ = _run(
            window_seconds=9.96, overlap_seconds=1.0, audio_duration=100.0
        )
        assert "thin context on short window" in report
        assert warns >= 1

    def test_default_window_no_warn(self):
        # window=19.88, overlap=2.0 -> 7/63 = 11% but window > 12s threshold.
        report, _, _, _, _ = _run(
            window_seconds=19.88, overlap_seconds=2.0, audio_duration=154.28
        )
        assert "thin context on short window" not in report


class TestAudioDuration:
    def test_audio_too_short_errors(self):
        report, ok, _, errs, _ = _run(audio_duration=5.0, window_seconds=10.0)
        assert "audio (5.0s) shorter than window" in report
        assert errs >= 1
        assert ok is False

    def test_audio_barely_fits_warns(self):
        report, _, warns, _, _ = _run(audio_duration=15.0, window_seconds=10.0)
        assert "barely longer than window" in report
        assert warns >= 1


class TestScheduleSeamAlignment:
    def test_seam_on_boundary_warns(self):
        # window=9.96, stride=8.0 -> iter 1 seam at 0:08.
        # schedule boundary at 0:08 -> should hit.
        schedule = (
            "0:00-0:08: wide shot\n"
            "0:08-0:16: close-up\n"
            "0:16+: medium shot\n"
        )
        report, _, warns, _, _ = _run(
            window_seconds=9.96, overlap_seconds=2.0,
            audio_duration=60.0, schedule=schedule,
        )
        assert "seam(s) within" in report
        assert "schedule boundaries" in report
        assert warns >= 1

    def test_no_alignment_passes(self):
        # Boundaries off the seams.
        schedule = (
            "0:00-0:10: wide shot\n"
            "0:10-0:20: close-up\n"
        )
        report, _, _, _, _ = _run(
            window_seconds=19.88, overlap_seconds=2.0,
            audio_duration=60.0, schedule=schedule,
        )
        assert "no iter seams within" in report


class TestFullReport:
    def test_v6_broken_config_shows_expected_warnings(self):
        """The exact failing setup: window=9.96, overlap=2.0, schedule at 0:08."""
        schedule = (
            "0:00-0:08: wide shot, stormy sky\n"
            "0:08-0:16: Cut to close-up, dolly in\n"
        )
        report, ok, warns, errs, stride = _run(
            window_seconds=9.96,
            overlap_seconds=2.0,
            length=249,
            width=832,
            height=448,
            schedule=schedule,
            audio_duration=154.28,
            resolution_rule="div_by_32",
        )
        assert stride == pytest.approx(8.0)
        assert ok is True  # nothing is an ERROR yet, just warnings
        assert warns >= 2  # at least effective-overlap + seam-alignment
        assert "seam(s) within" in report
        assert "effective overlap 1.960s" in report


class TestKeyframeChecks:
    """Validation for KeyframeImageSchedule wiring — catches the
    'wired single image + multi-entry schedule' + 'multi-entry batch
    + no schedule' + 'schedule-collapses-to-index-0' footguns before
    a 45-minute run.

    All checks gated on keyframe_batch_size > 0; when 0 (default),
    no keyframe-related lines appear in the report.
    """

    def test_keyframe_checks_skipped_when_batch_size_zero(self):
        # With batch=0 the block is fully gated off; no "keyframe" text
        # should appear in the report.
        report, _, _, _, _ = _run(
            keyframe_schedule="", keyframe_batch_size=0,
        )
        assert "keyframe" not in report.lower()

    def test_warn_on_batch_with_no_schedule(self):
        # User wired a 3-image batch but never authored a schedule.
        # Every iteration will silently use index 0 — unused keyframes.
        report, _, warns, errs, _ = _run(
            keyframe_schedule="", keyframe_batch_size=3,
        )
        assert "keyframe batch has 3 image(s) but schedule is empty" in report
        assert "Fix:" in report
        assert warns >= 1
        assert errs == 0

    def test_warn_on_schedule_that_parses_to_empty_entries(self):
        # Whitespace-only schedule passes the `.strip()` early-return
        # but parses to no entries. Separate WARN branch.
        # NOTE: a line of just whitespace is considered "non-empty" by
        # strip() because the whole string has content, so use a string
        # with visible content that doesn't match the timestamp format.
        report, _, warns, errs, _ = _run(
            keyframe_schedule="garbage line with no timestamp\n",
            keyframe_batch_size=3,
        )
        assert "did not parse any entries" in report
        assert "Fix:" in report
        assert warns >= 1
        assert errs == 0

    def test_error_on_index_out_of_bounds(self):
        # Schedule references index 5 but batch only has 3 images.
        # Runtime: KeyframeImageSchedule clamps silently, so the user's
        # intended keyframe is never used. Pre-run: ERROR.
        schedule = "0:00-0:42: 0\n0:42-1:28: 5\n1:28+: 2\n"
        report, ok, _, errs, _ = _run(
            keyframe_schedule=schedule, keyframe_batch_size=3,
        )
        assert errs >= 1
        assert ok is False
        assert "index 5" in report or "out of bounds" in report.lower()

    def test_warn_on_schedule_collapse_to_single_index(self):
        # User authored "0:00+: 0" with a 3-image batch — other keyframes
        # unused. This is the exact bug the shipped _latent_keyframe.json
        # had before today's wiring fix.
        report, _, warns, _, _ = _run(
            keyframe_schedule="0:00+: 0\n", keyframe_batch_size=3,
        )
        assert warns >= 1
        # Production message says "always selects index 0" + "unused".
        assert "always selects index 0" in report
        assert "unused" in report.lower()

    def test_error_on_negative_index(self):
        # _safe_int tolerates negative integers — schedule "0:00+: -1"
        # parses successfully to index=-1 which is out-of-bounds in
        # either direction. ERROR branch covers this.
        report, ok, _, errs, _ = _run(
            keyframe_schedule="0:00+: -1\n", keyframe_batch_size=3,
        )
        assert errs >= 1
        assert ok is False
        assert "index -1" in report

    def test_no_collapse_warn_when_schedule_uses_all_indices(self):
        # 3 distinct indices across schedule + 3-image batch = nothing unused.
        schedule = "0:00-0:42: 0\n0:42-1:28: 1\n1:28+: 2\n"
        report, ok, _, errs, _ = _run(
            keyframe_schedule=schedule, keyframe_batch_size=3,
        )
        assert errs == 0
        # Per-index unused check should pass
        assert "unused" not in report.lower() or "not unused" in report.lower()
        # A positive OK line should appear for the keyframe block
        assert "keyframe" in report.lower()

    def test_single_index_with_batch_size_one_is_ok(self):
        # batch_size=1 with schedule "0:00+: 0" is legitimate (one keyframe,
        # one entry). No warning should fire.
        report, _, warns, errs, _ = _run(
            keyframe_schedule="0:00+: 0\n", keyframe_batch_size=1,
        )
        # No "unused" warning when batch_size == 1
        assert "unused" not in report.lower()
        assert errs == 0

    def test_warn_count_is_additive_with_existing_checks(self):
        # Keyframe warn must ADD to existing warn counts, not replace them.
        # Measure baseline without keyframes, then add keyframes and
        # assert the delta.
        baseline_kwargs = dict(
            window_seconds=9.96, overlap_seconds=2.0, fps=25,
            schedule="0:00-0:38: alpha\n0:38-1:15: beta\n1:15+: gamma\n",
        )
        _, _, baseline_warns, _, _ = _run(**baseline_kwargs)
        report, _, warns, _, _ = _run(
            **baseline_kwargs,
            keyframe_schedule="0:00+: 0\n",
            keyframe_batch_size=5,  # 5-image batch, schedule uses only 0 → WARN
        )
        # Delta must be exactly the keyframe warn — proves additivity, not replacement.
        assert warns == baseline_warns + 1, (
            f"Expected exactly 1 new keyframe warn on top of {baseline_warns} "
            f"baseline warns, got {warns} total"
        )
        # And the new warn line is actually present.
        assert "always selects index 0" in report
