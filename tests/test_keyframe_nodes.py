"""Tests for multi-frame video conditioning nodes.

Tests KeyframeImageSchedule, VideoFrameExtract, ImageBlend, and their
helper functions using synthetic torch tensors.
"""

import torch
import pytest


# --- Helper function tests ---


class TestParseImageSchedule:
    def test_basic_ranges(self):
        from nodes import _parse_image_schedule

        schedule = "0:00-0:38: 0\n0:38-1:15: 1\n1:15+: 2"
        entries = _parse_image_schedule(schedule)
        assert len(entries) == 3
        assert entries[0] == (0.0, 38.0, 0)
        assert entries[1] == (38.0, 75.0, 1)
        assert entries[2] == (75.0, None, 2)

    def test_bare_seconds(self):
        from nodes import _parse_image_schedule

        schedule = "0-38: 0\n38-75: 1\n75+: 2"
        entries = _parse_image_schedule(schedule)
        assert len(entries) == 3
        assert entries[0] == (0.0, 38.0, 0)
        assert entries[1] == (38.0, 75.0, 1)

    def test_whitespace_and_empty_lines(self):
        from nodes import _parse_image_schedule

        schedule = "\n  0:00+: 0  \n\n  0:30+: 1\n\n"
        entries = _parse_image_schedule(schedule)
        assert len(entries) == 2

    def test_non_integer_values_skipped(self):
        from nodes import _parse_image_schedule

        schedule = "0:00+: zero\n0:30+: 1"
        entries = _parse_image_schedule(schedule)
        assert len(entries) == 1
        assert entries[0] == (30.0, None, 1)

    def test_empty_schedule(self):
        from nodes import _parse_image_schedule

        entries = _parse_image_schedule("")
        assert entries == []

    def test_point_match(self):
        from nodes import _parse_image_schedule

        schedule = "0:30: 2"
        entries = _parse_image_schedule(schedule)
        assert len(entries) == 1
        assert entries[0] == (30.0, 30.0, 2)


class TestMatchImageSchedule:
    def test_basic_match(self):
        from nodes import _parse_image_schedule, _match_image_schedule

        entries = _parse_image_schedule("0:00-0:38: 0\n0:38-1:15: 1\n1:15+: 2")
        assert _match_image_schedule(entries, 10.0) == 0
        assert _match_image_schedule(entries, 40.0) == 1
        assert _match_image_schedule(entries, 80.0) == 2

    def test_last_match_wins(self):
        from nodes import _parse_image_schedule, _match_image_schedule

        # Overlapping ranges -- second entry covers same time
        entries = _parse_image_schedule("0:00+: 0\n0:00+: 3")
        assert _match_image_schedule(entries, 5.0) == 3

    def test_fallback_to_last_entry(self):
        from nodes import _parse_image_schedule, _match_image_schedule

        entries = _parse_image_schedule("0:10-0:20: 1")
        # Time 5.0 doesn't match any range, fallback to last entry
        assert _match_image_schedule(entries, 5.0) == 1

    def test_boundary_values(self):
        from nodes import _parse_image_schedule, _match_image_schedule

        entries = _parse_image_schedule("0:00-0:38: 0\n0:38-1:15: 1")
        # Exact boundary -- both ranges match at 38.0, last match wins
        assert _match_image_schedule(entries, 38.0) == 1


class TestMatchImageScheduleWithNext:
    def test_no_blend(self):
        from nodes import _parse_image_schedule, _match_image_schedule_with_next

        entries = _parse_image_schedule("0:00-0:38: 0\n0:38+: 1")
        idx, next_idx, factor = _match_image_schedule_with_next(entries, 10.0, 0.0)
        assert idx == 0
        assert next_idx == 0
        assert factor == 0.0

    def test_blend_near_boundary(self):
        from nodes import _parse_image_schedule, _match_image_schedule_with_next

        entries = _parse_image_schedule("0:00-0:38: 0\n0:38+: 1")
        # 3 seconds before boundary, with 5 second blend window
        idx, next_idx, factor = _match_image_schedule_with_next(entries, 35.0, 5.0)
        assert idx == 0
        assert next_idx == 1
        assert factor == pytest.approx(0.4)  # 1.0 - (3.0 / 5.0)

    def test_far_from_boundary(self):
        from nodes import _parse_image_schedule, _match_image_schedule_with_next

        entries = _parse_image_schedule("0:00-0:38: 0\n0:38+: 1")
        idx, next_idx, factor = _match_image_schedule_with_next(entries, 10.0, 5.0)
        assert idx == 0
        assert next_idx == 0
        assert factor == 0.0

    def test_last_range_no_blend(self):
        from nodes import _parse_image_schedule, _match_image_schedule_with_next

        entries = _parse_image_schedule("0:00-0:38: 0\n0:38+: 1")
        # In the last range, no upcoming boundary
        idx, next_idx, factor = _match_image_schedule_with_next(entries, 50.0, 5.0)
        assert idx == 1
        assert next_idx == 1
        assert factor == 0.0


# --- Node tests ---


def _make_image_batch(n: int, h: int = 8, w: int = 8, c: int = 3) -> torch.Tensor:
    """Create a synthetic IMAGE batch [B,H,W,C] with distinct values per frame."""
    batch = torch.zeros(n, h, w, c)
    for i in range(n):
        batch[i] = float(i) / max(n - 1, 1)
    return batch


class TestKeyframeImageSchedule:
    def test_single_image_default(self):
        from nodes import KeyframeImageSchedule

        images = _make_image_batch(1)
        result = KeyframeImageSchedule.execute(
            images=images,
            current_iteration=1,
            stride_seconds=18.88,
            schedule="0:00+: 0",
            blend_seconds=0.0,
        )
        assert result[0].shape == (1, 8, 8, 3)  # image
        assert result[4] == 0  # image_index

    def test_selects_correct_keyframe(self):
        from nodes import KeyframeImageSchedule

        images = _make_image_batch(3)
        # iteration=1, stride=18.88 => current_time=18.88
        result = KeyframeImageSchedule.execute(
            images=images,
            current_iteration=1,
            stride_seconds=18.88,
            schedule="0:00-0:30: 0\n0:30-1:00: 1\n1:00+: 2",
            blend_seconds=0.0,
        )
        assert result[4] == 0  # image_index (18.88s is in 0:00-0:30 range)

        # iteration=3 => current_time=56.64, in 0:30-1:00 range
        result = KeyframeImageSchedule.execute(
            images=images,
            current_iteration=3,
            stride_seconds=18.88,
            schedule="0:00-0:30: 0\n0:30-1:00: 1\n1:00+: 2",
            blend_seconds=0.0,
        )
        assert result[4] == 1  # image_index

    def test_index_clamping(self):
        from nodes import KeyframeImageSchedule

        images = _make_image_batch(2)
        # Schedule references index 5 but batch only has 2 images
        result = KeyframeImageSchedule.execute(
            images=images,
            current_iteration=1,
            stride_seconds=18.88,
            schedule="0:00+: 5",
            blend_seconds=0.0,
        )
        assert result[4] == 1  # clamped to batch_size - 1
        assert result[0].shape == (1, 8, 8, 3)

    def test_blend_outputs(self):
        from nodes import KeyframeImageSchedule

        images = _make_image_batch(2)
        # current_time=35.0, boundary at 38.0, blend_seconds=5.0
        # time_to_boundary=3.0, factor=1.0-(3.0/5.0)=0.4
        result = KeyframeImageSchedule.execute(
            images=images,
            current_iteration=1,
            stride_seconds=35.0,
            schedule="0:00-0:38: 0\n0:38+: 1",
            blend_seconds=5.0,
        )
        assert result[2] == pytest.approx(0.4)  # blend_factor
        assert result[4] == 0  # current image_index
        # next_image should be different from image
        assert not torch.equal(result[0], result[1])

    def test_current_time_output(self):
        from nodes import KeyframeImageSchedule

        images = _make_image_batch(1)
        result = KeyframeImageSchedule.execute(
            images=images,
            current_iteration=3,
            stride_seconds=18.88,
            schedule="0:00+: 0",
            blend_seconds=0.0,
        )
        assert result[3] == pytest.approx(56.64)  # current_time


class TestVideoFrameExtract:
    def test_basic_extraction(self):
        from nodes import VideoFrameExtract

        images = _make_image_batch(100)
        # iteration=0 => time=0, frame_index=0
        result = VideoFrameExtract.execute(
            images=images,
            current_iteration=0,
            stride_seconds=18.88,
            source_fps=25.0,
        )
        assert result[0].shape == (1, 8, 8, 3)
        assert result[1] == 0  # frame_index

    def test_correct_timestamp(self):
        from nodes import VideoFrameExtract

        images = _make_image_batch(1000)
        # iteration=2, stride=18.88 => time=37.76s, frame=round(37.76*25)=944
        result = VideoFrameExtract.execute(
            images=images,
            current_iteration=2,
            stride_seconds=18.88,
            source_fps=25.0,
        )
        assert result[1] == 944

    def test_clamp_beyond_length(self):
        from nodes import VideoFrameExtract

        images = _make_image_batch(10)
        # iteration=100 => time=1888s => frame=47200, way past batch size 10
        result = VideoFrameExtract.execute(
            images=images,
            current_iteration=100,
            stride_seconds=18.88,
            source_fps=25.0,
        )
        assert result[1] == 9  # clamped to batch_size - 1
        assert result[0].shape == (1, 8, 8, 3)

    def test_different_fps(self):
        from nodes import VideoFrameExtract

        images = _make_image_batch(500)
        # iteration=1, stride=18.88, fps=10 => frame=round(18.88*10)=189
        result = VideoFrameExtract.execute(
            images=images,
            current_iteration=1,
            stride_seconds=18.88,
            source_fps=10.0,
        )
        assert result[1] == 189


class TestImageBlend:
    def test_factor_zero_passthrough(self):
        from nodes import ImageBlend

        a = torch.ones(1, 4, 4, 3)
        b = torch.zeros(1, 4, 4, 3)
        result = ImageBlend.execute(image_a=a, image_b=b, blend_factor=0.0)
        assert torch.equal(result[0], a)

    def test_factor_one_passthrough(self):
        from nodes import ImageBlend

        a = torch.ones(1, 4, 4, 3)
        b = torch.zeros(1, 4, 4, 3)
        result = ImageBlend.execute(image_a=a, image_b=b, blend_factor=1.0)
        assert torch.equal(result[0], b)

    def test_midpoint_blend(self):
        from nodes import ImageBlend

        a = torch.ones(1, 4, 4, 3)
        b = torch.zeros(1, 4, 4, 3)
        result = ImageBlend.execute(image_a=a, image_b=b, blend_factor=0.5)
        expected = torch.full((1, 4, 4, 3), 0.5)
        assert torch.allclose(result[0], expected)

    def test_quarter_blend(self):
        from nodes import ImageBlend

        a = torch.ones(1, 4, 4, 3)
        b = torch.zeros(1, 4, 4, 3)
        result = ImageBlend.execute(image_a=a, image_b=b, blend_factor=0.25)
        expected = torch.full((1, 4, 4, 3), 0.75)
        assert torch.allclose(result[0], expected)

    def test_preserves_shape(self):
        from nodes import ImageBlend

        a = torch.rand(1, 16, 32, 3)
        b = torch.rand(1, 16, 32, 3)
        result = ImageBlend.execute(image_a=a, image_b=b, blend_factor=0.3)
        assert result[0].shape == (1, 16, 32, 3)
