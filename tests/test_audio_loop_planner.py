"""Tests for AudioLoopPlanner — must compute stride internally.

Pre-2026-04-27 the planner accepted `stride_seconds` as an input wired from
AudioLoopController. Combined with the iterations auto-wire
(`AudioLoopPlanner.total_iterations -> TensorLoopOpen.iterations_in`,
2026-04-26) and the existing `TensorLoopOpen.current_iteration ->
AudioLoopController.current_iteration` edge, this closed a dependency cycle:

    AudioLoopController -> AudioLoopPlanner -> TensorLoopOpen -> AudioLoopController

ComfyUI's prompt validator rejects the workflow with "Dependency cycle
detected" before any node executes.

The fix: AudioLoopPlanner now takes `window_seconds`, `overlap_seconds`,
`fps` directly (matching AudioLoopController) and computes stride internally
via `_compute_loop_geometry`. Both nodes apply the same integer-latent
quantization formula, so iteration counts stay consistent. Cycle gone.
"""

import inspect

import pytest
import torch

from nodes import AudioLoopController, AudioLoopPlanner, _compute_loop_geometry


def _audio(duration_s: float = 600.0, sr: int = 100):
    return {
        "waveform": torch.zeros(1, 1, int(duration_s * sr)),
        "sample_rate": sr,
    }


class TestPlannerSchemaNoStrideInput:
    def test_no_stride_seconds_input_breaks_cycle(self):
        """Planner must NOT accept `stride_seconds` as an input — that's
        what closed the controller→planner→tensorloop→controller cycle.
        Planner derives stride from window + overlap + fps locally."""
        sig = inspect.signature(AudioLoopPlanner.execute)
        assert "stride_seconds" not in sig.parameters, (
            "AudioLoopPlanner.execute must not declare a stride_seconds "
            "parameter — wiring it from AudioLoopController.stride_seconds "
            "creates a dependency cycle with TensorLoopOpen.iterations_in. "
            "Planner must compute stride internally."
        )

    def test_planner_takes_window_overlap_fps(self):
        """Planner takes the same primitives the controller takes, so both
        independently apply `_compute_loop_geometry` and produce a matching
        stride without depending on each other."""
        sig = inspect.signature(AudioLoopPlanner.execute)
        params = sig.parameters
        for required in ("audio", "window_seconds", "overlap_seconds", "fps"):
            assert required in params, (
                f"AudioLoopPlanner.execute is missing required parameter "
                f"`{required}`."
            )


class TestPlannerStrideMatchesController:
    """Planner stride must match controller stride EXACTLY when given the
    same primitives. If they drifted, total_iterations would be off-by-one
    relative to the actual loop, and the experiment harness's auto-wired
    `iterations_in` would terminate either too early or too late."""

    @pytest.mark.parametrize(
        "window_seconds,overlap_seconds,fps",
        [
            (19.88, 2.0, 25),
            (19.88, 1.0, 25),
            (10.0, 2.5, 24),
            (15.0, 0.0, 30),
        ],
    )
    def test_strides_agree(self, window_seconds, overlap_seconds, fps):
        geometry = _compute_loop_geometry(window_seconds, overlap_seconds, fps)
        ctrl_out = AudioLoopController.execute(
            current_iteration=1,
            window_seconds=window_seconds,
            overlap_seconds=overlap_seconds,
            audio=_audio(600.0),
            base_seed=0,
            fps=fps,
        )
        # ctrl_out is io.NodeOutput — .result attribute or [.] access (mirror
        # how other tests in tests/test_audio_loop_controller.py read outputs)
        ctrl_stride = ctrl_out[4] if hasattr(ctrl_out, "__getitem__") else ctrl_out.result[4]
        assert ctrl_stride == pytest.approx(geometry.stride_seconds)


class TestPlannerExecuteBasic:
    def test_returns_summary_and_iterations(self):
        out = AudioLoopPlanner.execute(
            audio=_audio(60.0),
            window_seconds=19.88,
            overlap_seconds=2.0,
            fps=25,
        )
        # io.NodeOutput supports tuple-style read in unit-test harness mode
        summary, iters = (out[0], out[1]) if hasattr(out, "__getitem__") else (out.result[0], out.result[1])
        assert isinstance(summary, str)
        assert isinstance(iters, int)
        assert iters >= 1
        assert "Audio:" in summary
