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


class TestSpatiotemporalWidgets:
    """apply_ltx_decoder._spatiotemporal_widgets — per-workflow widget
    derivation for the LTXVSpatioTemporalTiledVAEDecode swap. Units are
    LATENT frames (8 px frames each at LTX temporal scale). Derivation goes
    through nodes._compute_loop_geometry so chunk stride is BIT-EXACT with
    the controller's runtime stride (no re-rounding of pre-rounded seconds)."""

    def test_canonical_geometry(self):
        # window=19.88, overlap=2 @ 25fps -> 63/7 latents, stride 56.
        from apply_ltx_decoder import _spatiotemporal_widgets
        w = _spatiotemporal_widgets(19.88, 2.0)
        spatial, s_overlap, t_len, t_overlap, lff, dev, dt = w
        assert (spatial, s_overlap) == (1, 1)          # single spatial tile
        assert (t_len, t_overlap) == (63, 7)           # window/overlap latents
        assert t_len - t_overlap == 56                  # 56 latents = 17.92s stride
        assert lff is True
        assert (dev, dt) == ("cpu", "float16")          # bounded accumulator

    def test_window15s_variant_geometry(self):
        # Planner snaps target 15s -> 14.76s (369 frames); window_latents=47,
        # overlap_latents=7, stride 40 latents = 12.8s.
        from apply_ltx_decoder import _spatiotemporal_widgets
        w = _spatiotemporal_widgets(14.76, 2.0)
        assert (w[2], w[3]) == (47, 7)

    def test_overlap_capped_at_node_max(self):
        # overlap=3s -> overlap_latents=10 > node max 8; cap the chunk
        # overlap but preserve the stride (t_len - t_overlap == new_latents).
        from apply_ltx_decoder import _spatiotemporal_widgets
        from nodes import _compute_loop_geometry
        w = _spatiotemporal_widgets(19.88, 3.0)
        g = _compute_loop_geometry(19.88, 3.0, 25)
        assert w[3] <= 8
        assert w[2] - w[3] == g.new_latent_frames

    def test_chunk_length_exceeds_overlap(self):
        # Node raises when temporal_tile_length < temporal_overlap + 1.
        from apply_ltx_decoder import _spatiotemporal_widgets
        for window, overlap in ((0.5, 0.0), (5.0, 2.0), (14.76, 2.0),
                                (19.88, 2.0), (30.0, 5.0)):
            w = _spatiotemporal_widgets(window, overlap)
            assert w[2] >= w[3] + 2, f"{window=} {overlap=}: {w[2]=} {w[3]=}"


class TestDecoderTypeAllowlists:
    """The shared graph-walk allowlist (workflow_utils.DECODER_TYPES) must
    cover every decoder type the validator handles. The 2026-06 decoder
    swap added LTXVSpatioTemporalTiledVAEDecode to the validator but not
    to DECODER_TYPES — which silently no-opped the ERR-level F14 audit
    (trim_video_latent_to_audio_present walks VHS.images back to "a
    decoder" and found none) and broke decoder discovery in
    apply_trim_video_latent_to_audio / apply_pre_decode_cleanup."""

    def test_validator_types_subset_of_shared_allowlist(self):
        from validate_workflow_decoder import _DECODER_TYPES
        from workflow_utils import DECODER_TYPES

        missing = set(_DECODER_TYPES) - DECODER_TYPES
        assert not missing, (
            f"{missing} handled by the validator but absent from "
            f"workflow_utils.DECODER_TYPES — graph walks (F14 audit, "
            f"apply_trim_video_latent_to_audio, apply_pre_decode_cleanup) "
            f"will not recognize these nodes as decoders."
        )


class TestGetWindowAndOverlap:
    """_get_window_and_overlap must trace LINKS, not trust local widgets.

    AudioLoopController's window_seconds/overlap_seconds are widget-inputs
    LINKED from LTXFramePlanner.actual_seconds and a FloatConstant in every
    shipped loop workflow; the controller's own widgets_values are stale
    placeholders (all variants carry the canonical 19.88/2 there). Reading
    widgets mis-derives the stride for every non-default variant."""

    @staticmethod
    def _load(rel):
        from pathlib import Path
        from validate_workflow_decoder import REPO_ROOT
        from workflow_utils import WorkflowEditor
        return WorkflowEditor(REPO_ROOT / rel)

    def test_canonical_traces_planner(self):
        from validate_workflow_decoder import _get_window_and_overlap
        ed = self._load("example_workflows/audio-loop-music-video_latent.json")
        w, o = _get_window_and_overlap(ed)
        assert abs(w - 19.88) < 1e-6   # planner target 19.88 -> snap-stable
        assert abs(o - 2.0) < 1e-6     # FloatConstant #2013

    def test_window15s_traces_planner_snapped(self):
        # Planner widget says target 15s; _snap_frames gives 369 frames
        # = 14.76s actual. The controller's stale widgets say 19.88 — the
        # link-traced value MUST win.
        from validate_workflow_decoder import _get_window_and_overlap
        ed = self._load(
            "example_workflows/experimental/audio-loop-music-video_latent_window15s.json"
        )
        w, o = _get_window_and_overlap(ed)
        assert abs(w - 14.76) < 1e-6
        assert abs(o - 2.0) < 1e-6

    def test_unlinked_falls_back_to_widget(self):
        # fml2v variant: window_seconds is unlinked (widget-driven).
        from validate_workflow_decoder import _get_window_and_overlap
        ed = self._load(
            "example_workflows/experimental/fml2v_var_d_audio_loop.json"
        )
        result = _get_window_and_overlap(ed)
        assert result is not None
        w, o = result
        assert 5.0 <= w <= 60.0 and 0.0 <= o < w
