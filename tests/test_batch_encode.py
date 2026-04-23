"""Tests for pre-loop batch encoding of timestamp prompt schedules.

Covers `TimestampPromptScheduleBatchEncode` (runs ONCE outside the loop,
encodes every per-iteration prompt up front) and
`ConditioningSelectByIteration` (runs inside the loop, picks pre-encoded
CONDITIONING by index -- no CLIP dependency).

This pair replaces the per-iteration CachedTextEncode + ConditioningBlend
chain that forced CLIP/DiT offload thrash and silenced NAG on iteration 2+.
"""

from __future__ import annotations

import pytest


class FakeCLIP:
    """Minimal CLIP stand-in recording tokenize+encode calls.

    Returns a distinct object per text so the selector's output identity
    can be asserted per iteration. Records every encode call so dedup
    behavior can be verified.
    """

    def __init__(self) -> None:
        self.tokenize_calls: list[str] = []
        self.encode_calls: list[str] = []

    def tokenize(self, text: str) -> dict:
        self.tokenize_calls.append(text)
        return {"_text": text}

    def encode_from_tokens_scheduled(self, tokens: dict) -> list:
        text = tokens["_text"]
        self.encode_calls.append(text)
        # Shape a minimal ComfyUI CONDITIONING: [[tensor_placeholder, {}]].
        # Use the text as the placeholder so identity-by-content holds.
        return [[f"COND::{text}", {"pooled_output": None}]]


def _schedule_text() -> str:
    # Two boundaries; distinct prompts per section. Matches the shape of
    # a typical music video schedule.
    return (
        "0:00-0:20: verse\n"
        "0:20-0:40: chorus\n"
        "0:40+: outro\n"
    )


class TestTimestampPromptScheduleBatchEncode:
    """Batch encoder invariants -- runs ONCE per generation."""

    def _execute(
        self,
        *,
        clip: FakeCLIP,
        stride_seconds: float = 20.0,
        audio_duration: float = 60.0,
        schedule: str | None = None,
        snap_boundaries: bool = True,
    ) -> tuple[list, int]:
        from nodes import TimestampPromptScheduleBatchEncode
        return TimestampPromptScheduleBatchEncode.execute(
            clip=clip,
            schedule=schedule if schedule is not None else _schedule_text(),
            stride_seconds=stride_seconds,
            audio_duration=audio_duration,
            snap_boundaries=snap_boundaries,
        )

    def test_emits_list_with_one_entry_per_iteration(self):
        """Output length matches ceil(audio_duration/stride)+1 so overshoot is safe."""
        clip = FakeCLIP()
        conditioning_list, iteration_count = self._execute(
            clip=clip, stride_seconds=20.0, audio_duration=60.0,
        )
        # 60/20 = 3 iterations plus +1 headroom = 4. Selector clamps
        # if the loop runs exactly 3.
        assert iteration_count == 4
        assert len(conditioning_list) == 4

    def test_deduplicates_identical_prompts(self):
        """Same text across iterations encodes exactly once per unique prompt."""
        clip = FakeCLIP()
        # A schedule where the same prompt spans multiple iterations:
        #   "verse" covers iter 0 and iter 1 at stride=10, audio=40s.
        schedule = "0:00-0:20: verse\n0:20+: chorus\n"
        self._execute(
            clip=clip, schedule=schedule, stride_seconds=10.0, audio_duration=40.0,
        )
        # Unique prompts should be 2 ("verse", "chorus"). Encode called
        # exactly twice even though we have 5 iterations (40/10 + 1).
        assert sorted(set(clip.encode_calls)) == ["chorus", "verse"]
        assert len(clip.encode_calls) == 2

    def test_each_iteration_gets_the_snapped_prompt(self):
        """With snap_boundaries=True, the Nth entry matches what
        TimestampPromptSchedule would emit at current_iteration=N."""
        from nodes import TimestampPromptSchedule
        clip = FakeCLIP()
        schedule = _schedule_text()
        stride = 20.0
        conditioning_list, _ = self._execute(
            clip=clip, schedule=schedule, stride_seconds=stride,
            audio_duration=60.0, snap_boundaries=True,
        )
        # For each iteration, compute what the per-iteration schedule node
        # would emit, then confirm the batch encoder emitted the same
        # CONDITIONING (by our FakeCLIP content tag).
        for i in range(len(conditioning_list)):
            result = TimestampPromptSchedule.execute(
                current_iteration=i,
                stride_seconds=stride,
                schedule=schedule,
                blend_seconds=0.0,
                snap_boundaries=True,
            )
            expected_text = result[0]  # prompt output
            got = conditioning_list[i]
            assert got[0][0] == f"COND::{expected_text}", (
                f"iteration {i}: batch emitted {got} but schedule says {expected_text!r}"
            )

    def test_no_snap_uses_raw_schedule_times(self):
        """With snap off, boundaries are not rounded to the stride grid."""
        from nodes import TimestampPromptSchedule
        clip = FakeCLIP()
        schedule = "0:00-0:15: a\n0:15-0:35: b\n0:35+: c\n"
        stride = 17.88
        conditioning_list, _ = self._execute(
            clip=clip, schedule=schedule, stride_seconds=stride,
            audio_duration=60.0, snap_boundaries=False,
        )
        for i in range(len(conditioning_list)):
            result = TimestampPromptSchedule.execute(
                current_iteration=i,
                stride_seconds=stride,
                schedule=schedule,
                blend_seconds=0.0,
                snap_boundaries=False,
            )
            expected_text = result[0]
            got = conditioning_list[i]
            assert got[0][0] == f"COND::{expected_text}", (
                f"iteration {i}: batch={got} schedule={expected_text!r}"
            )

    def test_open_end_entry_covers_final_iterations(self):
        """An open-ended `0:40+: outro` entry fills every iteration past 0:40."""
        clip = FakeCLIP()
        schedule = "0:00-0:40: intro\n0:40+: outro\n"
        conditioning_list, _ = self._execute(
            clip=clip, schedule=schedule, stride_seconds=10.0, audio_duration=100.0,
        )
        # iter >=4 should be outro
        for i in range(4, len(conditioning_list)):
            got = conditioning_list[i]
            assert got[0][0] == "COND::outro"

    def test_minimum_one_iteration(self):
        """Edge case: audio shorter than stride still yields at least one entry."""
        clip = FakeCLIP()
        conditioning_list, iteration_count = self._execute(
            clip=clip, stride_seconds=20.0, audio_duration=5.0,
        )
        assert iteration_count >= 1
        assert len(conditioning_list) >= 1

    def test_empty_schedule_defaults_to_empty_string(self):
        """Missing schedule text still encodes (empty string prompt)."""
        clip = FakeCLIP()
        conditioning_list, _ = self._execute(
            clip=clip, schedule="", stride_seconds=20.0, audio_duration=40.0,
        )
        # Still emits valid CONDITIONING per iteration (encoded empty prompt).
        assert len(conditioning_list) >= 1
        # Encode called exactly once since all iterations share the default.
        assert len(clip.encode_calls) == 1


class TestConditioningSelectByIteration:
    """Per-iteration selector -- runs inside the loop, no CLIP ref."""

    def _execute(
        self, *, conditioning_list: list, current_iteration: int,
    ) -> list:
        from nodes import ConditioningSelectByIteration
        return ConditioningSelectByIteration.execute(
            conditioning_list=conditioning_list,
            current_iteration=current_iteration,
        )[0]

    def test_selects_by_index(self):
        conditioning_list = [
            [["cond_0", {}]],
            [["cond_1", {}]],
            [["cond_2", {}]],
        ]
        assert self._execute(conditioning_list=conditioning_list, current_iteration=0)[0][0] == "cond_0"
        assert self._execute(conditioning_list=conditioning_list, current_iteration=1)[0][0] == "cond_1"
        assert self._execute(conditioning_list=conditioning_list, current_iteration=2)[0][0] == "cond_2"

    def test_clamps_beyond_list_to_last(self):
        """Overshoot (loop runs one more iter than batch encoded) returns last entry."""
        conditioning_list = [
            [["cond_0", {}]],
            [["cond_1", {}]],
        ]
        got = self._execute(conditioning_list=conditioning_list, current_iteration=99)
        assert got[0][0] == "cond_1"

    def test_clamps_negative_index_to_zero(self):
        """Defensive: negative indices clamp to 0 rather than wrap."""
        conditioning_list = [[["cond_0", {}]], [["cond_1", {}]]]
        got = self._execute(conditioning_list=conditioning_list, current_iteration=-5)
        assert got[0][0] == "cond_0"

    def test_empty_list_raises(self):
        """Empty batch is a wiring bug; fail loudly, not silently."""
        from nodes import ConditioningSelectByIteration
        with pytest.raises((ValueError, IndexError)):
            ConditioningSelectByIteration.execute(
                conditioning_list=[], current_iteration=0,
            )


class TestBatchAndSelectIntegration:
    """End-to-end: batch encode then select, compare vs TimestampPromptSchedule."""

    def test_integration_matches_timestamp_prompt_schedule_output(self):
        """For every iteration, batch->select must return CONDITIONING whose
        content tag matches the prompt TimestampPromptSchedule would have
        produced. This is the behavioral-equivalence contract."""
        from nodes import (
            ConditioningSelectByIteration,
            TimestampPromptSchedule,
            TimestampPromptScheduleBatchEncode,
        )
        clip = FakeCLIP()
        schedule = _schedule_text()
        stride = 20.0
        audio_duration = 80.0

        conditioning_list, iteration_count = TimestampPromptScheduleBatchEncode.execute(
            clip=clip,
            schedule=schedule,
            stride_seconds=stride,
            audio_duration=audio_duration,
            snap_boundaries=True,
        )
        # CRITICAL invariant: CLIP encode is called exactly once per unique
        # prompt across the full run -- never re-entered during the loop.
        assert sorted(set(clip.encode_calls)) == ["chorus", "outro", "verse"]
        total_encode_calls_after_setup = len(clip.encode_calls)

        for i in range(iteration_count):
            selected = ConditioningSelectByIteration.execute(
                conditioning_list=conditioning_list, current_iteration=i,
            )[0]
            expected = TimestampPromptSchedule.execute(
                current_iteration=i, stride_seconds=stride,
                schedule=schedule, blend_seconds=0.0, snap_boundaries=True,
            )
            assert selected[0][0] == f"COND::{expected[0]}"

        # Selector must not have triggered any more CLIP calls.
        assert len(clip.encode_calls) == total_encode_calls_after_setup
