"""Tests for scripts/validate_workflow_decoder.py helpers.

The `_expected_stride_widgets` helper emits the advisory widget values
users paste into `VAEDecodeTiled` when falling back from
`LTXVTiledVAEDecode`. Round-trip: its output must produce a tile stride
that matches the requested iteration stride within one frame.
"""

from validate_workflow_decoder import _FPS, _expected_stride_widgets


def _tile_stride_s(ts: int, to: int) -> float:
    return (ts - to) / _FPS


class TestExpectedStrideWidgets:
    def test_round_trip_at_overlap_2(self):
        # window=19.88, overlap=2 -> iter_stride=17.88s
        ts, to = _expected_stride_widgets(17.88)
        assert abs(_tile_stride_s(ts, to) - 17.88) <= 1 / _FPS

    def test_round_trip_at_overlap_3(self):
        # window=19.88, overlap=3 -> iter_stride=16.88s
        ts, to = _expected_stride_widgets(16.88)
        assert abs(_tile_stride_s(ts, to) - 16.88) <= 1 / _FPS

    def test_round_trip_at_overlap_1(self):
        ts, to = _expected_stride_widgets(18.88)
        assert abs(_tile_stride_s(ts, to) - 18.88) <= 1 / _FPS

    def test_overlap_respects_quarter_size_constraint(self):
        # ComfyUI's VAEDecodeTiled rejects temporal_overlap > temporal_size/4.
        for stride in (10.0, 16.88, 17.88, 18.88, 25.0):
            ts, to = _expected_stride_widgets(stride)
            assert to <= ts // 4, f"stride={stride}: to={to} > ts/4={ts // 4}"

    def test_values_are_multiples_of_four(self):
        # temporal_overlap is masked to multiples of 4 to stay on
        # convolution-friendly boundaries.
        for stride in (10.0, 16.88, 17.88, 18.88, 25.0):
            _, to = _expected_stride_widgets(stride)
            assert to % 4 == 0, f"stride={stride}: temporal_overlap={to} not a multiple of 4"

    def test_minimum_overlap_floor(self):
        _, to = _expected_stride_widgets(0.5)
        assert to >= 8
