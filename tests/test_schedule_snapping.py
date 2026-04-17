"""Tests for schedule boundary snapping and raised-cosine blend ramp.

Covers the Phase 1 fix for the `blend_seconds` jitter bug — see
`/home/fbliss/.claude/plans/partitioned-yawning-pond.md`.
"""

import pytest

from nodes import (
    BlendShape,
    _match_schedule_with_next_generic,
    _snap_schedule_to_iterations,
)


# --- _snap_schedule_to_iterations ---


class TestSnapScheduleToIterations:
    def test_snap_preserves_first_entry_at_zero(self):
        """Boundary at 0 stays at 0 regardless of stride."""
        entries = [
            (0.0, 15.0, "a"),
            (15.0, None, "b"),
        ]
        snapped = _snap_schedule_to_iterations(entries, stride_seconds=17.88)
        assert snapped[0][0] == 0.0

    def test_snap_to_nearest_iteration_multiple(self):
        """Each boundary rounds to round(t/stride)*stride."""
        entries = [
            (0.0, 15.0, "a"),
            (15.0, 35.0, "b"),
            (35.0, 55.0, "c"),
            (55.0, None, "d"),
        ]
        stride = 17.88
        snapped = _snap_schedule_to_iterations(entries, stride_seconds=stride)
        # 15 / 17.88 = 0.839 -> round to 1 -> 17.88
        # 35 / 17.88 = 1.958 -> round to 2 -> 35.76
        # 55 / 17.88 = 3.076 -> round to 3 -> 53.64
        starts = [round(e[0], 4) for e in snapped]
        assert starts == [0.0, 17.88, 35.76, 53.64]

    def test_snap_preserves_open_last_entry(self):
        """Last entry with end=None keeps end=None after snap."""
        entries = [(0.0, 20.0, "a"), (20.0, None, "b")]
        snapped = _snap_schedule_to_iterations(entries, stride_seconds=10.0)
        assert snapped[-1][1] is None

    def test_snap_merges_collapsed_entries(self):
        """Two entries whose snapped starts collide: later prompt wins."""
        # With stride=20, 15 and 18 both snap to 20. Later wins.
        entries = [
            (0.0, 15.0, "a"),
            (15.0, 18.0, "b_early"),
            (18.0, None, "c_later_wins"),
        ]
        snapped = _snap_schedule_to_iterations(entries, stride_seconds=20.0)
        # After snap: (0.0, 20.0, a), (20.0, 20.0 -> dropped zero-length),
        #             (20.0, None, c_later_wins)
        # Merge duplicates at start=20.0 -> keep c_later_wins
        values_at_20 = [e[2] for e in snapped if e[0] == 20.0]
        assert values_at_20 == ["c_later_wins"]

    def test_snap_drops_zero_length_entries(self):
        """If start and end snap to the same value, entry is dropped."""
        # With stride=20: entry (14, 18) has start->20 and end->20, zero-length, drop.
        entries = [
            (0.0, 14.0, "a"),
            (14.0, 18.0, "should_drop"),
            (18.0, None, "c"),
        ]
        snapped = _snap_schedule_to_iterations(entries, stride_seconds=20.0)
        values = [e[2] for e in snapped]
        assert "should_drop" not in values

    def test_snap_empty_returns_empty(self):
        assert _snap_schedule_to_iterations([], stride_seconds=10.0) == []

    def test_snap_zero_stride_is_noop(self):
        """Edge case: stride <= 0 returns input unchanged to avoid divide-by-zero."""
        entries = [(0.0, 15.0, "a"), (15.0, None, "b")]
        assert _snap_schedule_to_iterations(entries, stride_seconds=0.0) == entries


# --- _match_schedule_with_next_generic raised-cosine blend ---


class TestRaisedCosineBlend:
    def _schedule(self) -> list[tuple[float, float | None, str]]:
        # Two entries, boundary at t=20
        return [(0.0, 20.0, "a"), (20.0, None, "b")]

    def test_outside_blend_window_is_pure_current(self):
        """Far from any boundary, blend_factor=0 and values=current."""
        entries = self._schedule()
        current, nxt, bf = _match_schedule_with_next_generic(
            entries, current_time=5.0, blend_seconds=4.0, default="",
        )
        assert current == "a"
        assert nxt == "a"
        assert bf == 0.0

    def test_blend_at_exact_boundary_is_half(self):
        """At the boundary itself, raised-cosine is 0.5."""
        entries = self._schedule()
        current, nxt, bf = _match_schedule_with_next_generic(
            entries, current_time=20.0, blend_seconds=4.0, default="",
        )
        assert current == "a"   # pre-boundary
        assert nxt == "b"       # post-boundary
        assert bf == pytest.approx(0.5, abs=1e-9)

    def test_blend_at_window_start_is_zero(self):
        """blend_factor=0 at current_time = boundary - blend_seconds/2."""
        entries = self._schedule()
        _, _, bf = _match_schedule_with_next_generic(
            entries, current_time=18.0, blend_seconds=4.0, default="",
        )
        assert bf == pytest.approx(0.0, abs=1e-9)

    def test_blend_at_window_end_is_one(self):
        """blend_factor=1 at current_time = boundary + blend_seconds/2."""
        entries = self._schedule()
        current, nxt, bf = _match_schedule_with_next_generic(
            entries, current_time=22.0, blend_seconds=4.0, default="",
        )
        # Even past the boundary, current stays as the PRE-boundary prompt
        # so ConditioningBlend can lerp smoothly.
        assert current == "a"
        assert nxt == "b"
        assert bf == pytest.approx(1.0, abs=1e-9)

    def test_blend_is_monotonic_across_boundary(self):
        """As current_time advances through the window, blend_factor increases."""
        entries = self._schedule()
        times = [18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5]
        factors = [
            _match_schedule_with_next_generic(entries, t, 4.0, "")[2]
            for t in times
        ]
        for a, b in zip(factors, factors[1:]):
            assert b >= a - 1e-9, f"blend_factor must be monotonic: {factors}"

    def test_effective_conditioning_smooth_across_iterations(self):
        """The effective (lerped) conditioning pre*(1-bf) + post*bf must
        change smoothly across adjacent iterations — no perceptible jitter.
        This is the anti-jitter property (the raw blend_factor legitimately
        drops from 1.0 to 0.0 when exiting the blend window, since the
        tuple's meaning changes from (pre, post, ramp) to (post, post, 0);
        effective conditioning stays continuous even then).
        """
        # stride = 20, blend = 80 (4x stride — smooth ramp across 4 iters)
        stride = 20.0
        blend = 80.0
        # Numeric values so we can compute effective conditioning
        entries: list[tuple[float, float | None, float]] = [
            (0.0, 100.0, 0.0),
            (100.0, None, 1.0),
        ]
        iters = [i * stride for i in range(10)]
        effectives = []
        for t in iters:
            pre, post, bf = _match_schedule_with_next_generic(
                entries, t, blend, 0.0,
            )
            effectives.append(pre * (1 - bf) + post * bf)
        max_jump = max(abs(b - a) for a, b in zip(effectives, effectives[1:]))
        # At blend=4*stride with raised-cosine, the steepest per-step jump is
        # ~0.36. The broken spike implementation would show jumps of ~0.72.
        assert max_jump < 0.5, (
            f"Effective conditioning jumps too much: {effectives}, "
            f"max jump {max_jump}"
        )

    def test_spike_produces_larger_jumps_than_raised_cosine(self):
        """Regression anchor: the raised-cosine curve must produce materially
        smoother transitions than the legacy spike for the same inputs."""
        stride = 20.0
        blend = 20.0  # EXACTLY one stride — maximum difference between shapes
        entries: list[tuple[float, float | None, float]] = [
            (0.0, 100.0, 0.0),
            (100.0, None, 1.0),
        ]
        iters = [i * stride for i in range(8)]

        def effectives(shape: BlendShape) -> list[float]:
            out = []
            for t in iters:
                pre, post, bf = _match_schedule_with_next_generic(
                    entries, t, blend, 0.0, blend_shape=shape,
                )
                out.append(pre * (1 - bf) + post * bf)
            return out

        rc = effectives("raised_cosine")
        sp = effectives("spike")
        rc_jump = max(abs(b - a) for a, b in zip(rc, rc[1:]))
        sp_jump = max(abs(b - a) for a, b in zip(sp, sp[1:]))
        # Spike should have at least a 1.0 jump somewhere (snap from mixed
        # conditioning to pure post). Raised-cosine should be smoother.
        assert sp_jump >= 0.9, f"Spike should produce near-discontinuous jumps: {sp}"
        assert rc_jump < sp_jump, (
            f"Raised-cosine should be smoother than spike. "
            f"rc={rc} sp={sp}"
        )

    def test_hard_switch_when_blend_seconds_zero(self):
        """blend_seconds=0 → pure hard switch, matches current=next always."""
        entries = self._schedule()
        for t in [10.0, 19.0, 20.0, 21.0, 30.0]:
            current, nxt, bf = _match_schedule_with_next_generic(
                entries, current_time=t, blend_seconds=0.0, default="",
            )
            assert bf == 0.0
            assert current == nxt


# --- Legacy spike blend (backcompat path) ---


class TestLegacySpikeBlend:
    def test_snap_boundaries_false_preserves_legacy_spike(self):
        """When blend_shape='spike', behavior matches the pre-fix implementation."""
        entries = [(0.0, 20.0, "a"), (20.0, None, "b")]
        # At t=18 with blend_seconds=5, time_to_boundary=2, so spike gives
        # blend_factor = 1 - 2/5 = 0.6; current=a, next=b.
        current, nxt, bf = _match_schedule_with_next_generic(
            entries, current_time=18.0, blend_seconds=5.0, default="",
            blend_shape="spike",
        )
        assert current == "a"
        assert nxt == "b"
        assert bf == pytest.approx(0.6, abs=1e-9)

    def test_spike_is_zero_outside_window(self):
        """At t=10 (far from boundary) with spike + blend_seconds=5, blend=0."""
        entries = [(0.0, 20.0, "a"), (20.0, None, "b")]
        _, _, bf = _match_schedule_with_next_generic(
            entries, current_time=10.0, blend_seconds=5.0, default="",
            blend_shape="spike",
        )
        assert bf == 0.0


# --- Integration test for TimestampPromptSchedule execute path ---


class TestTimestampPromptScheduleIntegration:
    """End-to-end test via the node class, covering auto-clamp + snap."""

    def _make_schedule_text(self) -> str:
        # Boundaries at 0:15, 0:35, 0:55 — the user's standup schedule shape.
        # At stride=17.88 these snap to 17.88, 35.76, 53.64.
        return (
            "0:00-0:15: A\n"
            "0:15-0:35: B\n"
            "0:35-0:55: C\n"
            "0:55+: D\n"
        )

    def _execute(
        self,
        current_iteration: int,
        stride_seconds: float = 17.88,
        blend_seconds: float = 0.0,
        snap_boundaries: bool = True,
    ):
        from nodes import TimestampPromptSchedule
        return TimestampPromptSchedule.execute(
            current_iteration=current_iteration,
            stride_seconds=stride_seconds,
            schedule=self._make_schedule_text(),
            blend_seconds=blend_seconds,
            snap_boundaries=snap_boundaries,
        )

    def test_snap_boundaries_true_by_default_prevents_mid_iter_mix(self):
        """With snap on, every iteration lands on a pure prompt."""
        # stride=17.88; iter 3 => current_time=53.64 which IS the snapped
        # boundary of entry C. Snap puts boundary AT iteration. Hard switch at
        # this point means current=C (first time at this boundary).
        result = self._execute(current_iteration=3, blend_seconds=0.0)
        prompt, next_prompt, blend_factor, current_time = result
        assert blend_factor == 0.0
        assert prompt in ("C", "D")  # at exact boundary, either side is valid

    def test_autoclamp_sub_stride_blend_seconds(self):
        """blend_seconds < stride triggers clamp to stride, emits one warning."""
        # Import and reset warned-keys so we can observe the warning in test
        import nodes
        nodes._get_warned_keys().discard("blend_seconds_clamped")

        # User's original buggy value: 5s blend on 17.88s stride.
        # Should clamp to 17.88 and warn once.
        result_clamped = self._execute(
            current_iteration=1, blend_seconds=5.0, snap_boundaries=True
        )
        # Compare to explicit stride_seconds value — outputs should match.
        result_explicit = self._execute(
            current_iteration=1, blend_seconds=17.88, snap_boundaries=True
        )
        assert result_clamped[2] == pytest.approx(result_explicit[2], abs=1e-9)

    def test_no_clamp_when_snap_disabled(self):
        """snap_boundaries=False → blend_seconds is passed through unchanged.

        Verifies the clamp ONLY fires on the snap path. With snap off, a
        sub-stride blend_seconds value should produce the legacy spike
        result with its ORIGINAL (unclamped) value — we confirm by matching
        the output against a direct spike computation at that same value.
        """
        import nodes
        from nodes import _match_schedule_with_next_generic, _parse_schedule

        nodes._get_warned_keys().discard("blend_seconds_clamped")

        # stride=17.88, blend_seconds=5 (sub-stride). With snap=False the
        # clamp must NOT fire — the node should call spike with blend=5.
        res_no_snap = self._execute(
            current_iteration=1, blend_seconds=5.0, snap_boundaries=False,
        )
        # Independently compute what spike with blend=5 produces on the
        # UN-snapped entries, and verify the node's output matches exactly.
        entries = _parse_schedule(self._make_schedule_text())
        expected = _match_schedule_with_next_generic(
            entries, current_time=17.88, blend_seconds=5.0, default="",
            blend_shape="spike",
        )
        assert res_no_snap[0] == expected[0]  # prompt
        assert res_no_snap[1] == expected[1]  # next_prompt
        assert res_no_snap[2] == pytest.approx(expected[2], abs=1e-9)

        # Regression anchor: with snap=True at the same blend=5, behavior
        # MUST differ (either clamped or shape-different).
        nodes._get_warned_keys().discard("blend_seconds_clamped")
        res_with_snap = self._execute(
            current_iteration=1, blend_seconds=5.0, snap_boundaries=True,
        )
        assert res_with_snap != res_no_snap
