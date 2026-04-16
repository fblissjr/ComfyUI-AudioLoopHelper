"""Tests for ProfileBegin, ProfileIterStep, ProfileEnd nodes.

Focus on the disabled-path behavior (passthroughs, zero overhead) since
the enabled path requires CUDA and a running torch.profiler which we
don't exercise in unit tests. The profiler integration itself is
covered by manual end-to-end testing via the profiled workflow variant.
"""

import torch


class TestProfileBeginDisabled:
    def setup_method(self):
        from nodes import _PROFILER_STATE
        _PROFILER_STATE.clear()

    def test_disabled_returns_trigger_unchanged(self):
        from nodes import ProfileBegin

        trigger = {"samples": torch.zeros(1)}
        result = ProfileBegin.execute(
            trigger=trigger,
            enabled=False,
            output_dir="./profile_output/",
            warmup_iterations=1,
            active_iterations=3,
            include_cpu=True,
            include_memory=True,
            include_shapes=True,
            include_flops=False,
        )
        assert result[0] is trigger

    def test_disabled_leaves_state_empty(self):
        from nodes import ProfileBegin, _PROFILER_STATE

        ProfileBegin.execute(
            trigger="anything",
            enabled=False,
            output_dir="./profile_output/",
            warmup_iterations=1,
            active_iterations=3,
            include_cpu=True,
            include_memory=True,
            include_shapes=True,
            include_flops=False,
        )
        assert _PROFILER_STATE.get("profiler") is None


class TestProfileIterStepDisabled:
    def setup_method(self):
        from nodes import _PROFILER_STATE
        _PROFILER_STATE.clear()

    def test_passthrough_when_no_profiler(self):
        from nodes import ProfileIterStep

        latent = {"samples": torch.zeros(1)}
        result = ProfileIterStep.execute(latent=latent)
        assert result[0] is latent

    def test_warning_logged_once_when_uninitialized(self, monkeypatch):
        from nodes import ProfileIterStep, _PROFILER_STATE

        warnings = []
        monkeypatch.setattr(
            "builtins.print",
            lambda *args, **kwargs: warnings.append(" ".join(str(a) for a in args)),
        )

        latent = {"samples": torch.zeros(1)}
        ProfileIterStep.execute(latent=latent)
        ProfileIterStep.execute(latent=latent)
        ProfileIterStep.execute(latent=latent)

        # Only one warning about uninitialized profiler
        warn_lines = [w for w in warnings if "ProfileIterStep" in w and "without" in w]
        assert len(warn_lines) == 1, f"Expected 1 warning, got {len(warn_lines)}: {warn_lines}"


class TestProfileEndDisabled:
    def setup_method(self):
        from nodes import _PROFILER_STATE
        _PROFILER_STATE.clear()

    def test_passthrough_when_no_profiler(self):
        from nodes import ProfileEnd

        trigger = "anything"
        result = ProfileEnd.execute(trigger=trigger)
        assert result[0] is trigger

    def test_warning_logged_once_when_uninitialized(self, monkeypatch):
        from nodes import ProfileEnd

        warnings = []
        monkeypatch.setattr(
            "builtins.print",
            lambda *args, **kwargs: warnings.append(" ".join(str(a) for a in args)),
        )

        ProfileEnd.execute(trigger=1)
        ProfileEnd.execute(trigger=2)

        warn_lines = [w for w in warnings if "ProfileEnd" in w and "without" in w]
        assert len(warn_lines) == 1


class TestProfilerStateCoordination:
    """Verify the three nodes coordinate correctly via module state
    (in the disabled / uninitialized case).
    """

    def setup_method(self):
        from nodes import _PROFILER_STATE
        _PROFILER_STATE.clear()

    def test_full_disabled_chain_is_all_passthrough(self):
        from nodes import ProfileBegin, ProfileIterStep, ProfileEnd

        trigger = "start"
        latent = {"samples": torch.zeros(1)}

        begin_out = ProfileBegin.execute(
            trigger=trigger,
            enabled=False,
            output_dir="./profile_output/",
            warmup_iterations=1,
            active_iterations=3,
            include_cpu=True,
            include_memory=True,
            include_shapes=True,
            include_flops=False,
        )
        step_out = ProfileIterStep.execute(latent=latent)
        end_out = ProfileEnd.execute(trigger="end")

        assert begin_out[0] is trigger
        assert step_out[0] is latent
        assert end_out[0] == "end"
