"""Tests for `tracers/ffn_attn.py::FfnAttnTracer`.

Locks in the fingerprint-coexistence warning: when both
`AUDIOLOOPHELPER_FFN_ATTN_TRACE` and `AUDIOLOOPHELPER_SAGE_OUTPUT_FINGERPRINT`
are set, the sage tracer's `.item()` CUDA sync inflates ffn_attn's
forward-hook `elapsed_ms` measurements 1.5-2× because the surrounding
sub-module forward waits on the sync. The warning fires at install
so the distortion is visible at every render start instead of
silently propagating into "production-comparable" claims.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr

import pytest


class _FakeDiffusionModel:
    transformer_blocks: list = []


class _FakeModelClone:
    """Minimal stand-in exposing `get_model_object("diffusion_model")`.

    Different surface than `tests/_fakes.py::FakeModelPatcher` (that one
    targets `model_options`/`transformer_options`; this one targets the
    `diffusion_model.transformer_blocks` walk that `FfnAttnTracer`'s
    install path expects). Two consumers' worth — left local to this file
    until a third test needs the same shape.
    """
    def get_model_object(self, _name: str) -> _FakeDiffusionModel:
        return _FakeDiffusionModel()


@pytest.fixture
def fake_model_clone() -> _FakeModelClone:
    return _FakeModelClone()


@pytest.fixture
def base_env(monkeypatch):
    """Set the ffn_attn + per-prompt env state shared across tests.

    Each test overrides the fingerprint / torch_profile env vars on top.
    """
    monkeypatch.setenv("AUDIOLOOPHELPER_FFN_ATTN_TRACE", "auto")
    monkeypatch.setenv("RUN_ID", "ffn_attn_test")
    monkeypatch.setenv("AUDIOLOOPHELPER_PER_PROMPT", "0")


def _install_and_capture_stderr(tracer, model_clone) -> str:
    buf = io.StringIO()
    with redirect_stderr(buf):
        tracer.install_at_render(model_clone)
    return buf.getvalue()


def test_warn_when_fingerprint_also_active(base_env, monkeypatch, fake_model_clone):
    """With both env vars set, `install_at_render` must emit a stderr line
    naming the fingerprint-mode timing inflation. Catches future refactors
    that drop the cross-env-var check.
    """
    monkeypatch.setenv("AUDIOLOOPHELPER_TORCH_PROFILE", "auto")
    monkeypatch.setenv("AUDIOLOOPHELPER_SAGE_OUTPUT_FINGERPRINT", "auto")

    from tracers.ffn_attn import FfnAttnTracer
    err = _install_and_capture_stderr(FfnAttnTracer(), fake_model_clone)

    assert "fingerprint mode also active" in err, err
    assert "inflated" in err


def test_no_warn_when_fingerprint_inactive(base_env, monkeypatch, fake_model_clone):
    """Without the fingerprint env var, the warning must not appear.
    Verifies the gate isn't bug-on-by-default.
    """
    monkeypatch.setenv("AUDIOLOOPHELPER_TORCH_PROFILE", "auto")
    monkeypatch.delenv("AUDIOLOOPHELPER_SAGE_OUTPUT_FINGERPRINT", raising=False)

    from tracers.ffn_attn import FfnAttnTracer
    err = _install_and_capture_stderr(FfnAttnTracer(), fake_model_clone)

    assert "fingerprint mode" not in err


def test_emit_annotations_gated_on_torch_profile_env(base_env, monkeypatch, fake_model_clone):
    """`_emit_annotations` must mirror `AUDIOLOOPHELPER_TORCH_PROFILE`
    at install time. Locks in the hot-path optimisation: hooks skip the
    C++-cost `record_function` bookkeeping when torch.profiler isn't
    capturing.
    """
    from tracers.ffn_attn import FfnAttnTracer

    monkeypatch.delenv("AUDIOLOOPHELPER_TORCH_PROFILE", raising=False)
    tracer_off = FfnAttnTracer()
    tracer_off.install_at_render(fake_model_clone)
    assert tracer_off._emit_annotations is False

    monkeypatch.setenv("AUDIOLOOPHELPER_TORCH_PROFILE", "auto")
    tracer_on = FfnAttnTracer()
    tracer_on.install_at_render(fake_model_clone)
    assert tracer_on._emit_annotations is True


def test_maybe_rotate_for_new_prompt_updates_state(base_env, monkeypatch, fake_model_clone, tmp_path):
    """When the executing prompt_id changes underneath an already-installed
    tracer, `_maybe_rotate_for_new_prompt` must flush pending events to
    the old path and rebind to the new one. Otherwise events from later
    renders get attributed to the first render's prompt_id.
    """
    from tracers.ffn_attn import FfnAttnTracer
    import tracers._base as base_mod

    tracer = FfnAttnTracer()
    tracer.install_at_render(fake_model_clone)

    tracer._cached_prompt_id = "prompt-A"
    path_a = tmp_path / "prompt-A" / "ffn_attn_breakdown.jsonl"
    tracer._output_path = path_a

    monkeypatch.setattr(base_mod, "get_executing_prompt_id", lambda: "prompt-B")
    path_b = tmp_path / "prompt-B" / "ffn_attn_breakdown.jsonl"
    monkeypatch.setattr(tracer, "resolve_output_path", lambda: path_b)

    tracer._maybe_rotate_for_new_prompt()

    assert tracer._cached_prompt_id == "prompt-B"
    assert tracer._output_path == path_b


def test_maybe_rotate_is_noop_when_prompt_unchanged(base_env, monkeypatch, fake_model_clone):
    """Rotation must be cheap: when prompt_id hasn't changed, no state
    mutation, no flush. Verifies the hot-path optimisation (rotation
    check fires per hook call, must be skip-fast in the common case)."""
    from tracers.ffn_attn import FfnAttnTracer
    import tracers._base as base_mod

    tracer = FfnAttnTracer()
    tracer.install_at_render(fake_model_clone)
    tracer._cached_prompt_id = "prompt-A"
    original_path = tracer._output_path

    monkeypatch.setattr(base_mod, "get_executing_prompt_id", lambda: "prompt-A")
    tracer._maybe_rotate_for_new_prompt()

    assert tracer._cached_prompt_id == "prompt-A"
    assert tracer._output_path == original_path


def test_orchestrator_refreshes_prompt_state_on_cleanup(monkeypatch):
    """Orchestrator's `_refresh_prompt_state_if_needed` must update
    `_PROMPT_ID` and `_PROMPT_START_TS` when the contextvar reports a
    new prompt. Closes the bug where the manifest written at
    on_cleanup carried a stale prompt_id from a prior render.
    """
    import tracers._orchestrator as orch
    monkeypatch.setattr(orch, "_PROMPT_ID", "prompt-A")
    monkeypatch.setattr(orch, "_PROMPT_START_TS", 100.0)
    monkeypatch.setattr(orch, "get_executing_prompt_id", lambda: "prompt-B")

    changed = orch._refresh_prompt_state_if_needed()

    assert changed is True
    assert orch._PROMPT_ID == "prompt-B"
    assert orch._PROMPT_START_TS != 100.0


def test_orchestrator_refresh_is_noop_when_unchanged(monkeypatch):
    import tracers._orchestrator as orch
    monkeypatch.setattr(orch, "_PROMPT_ID", "prompt-A")
    monkeypatch.setattr(orch, "_PROMPT_START_TS", 100.0)
    monkeypatch.setattr(orch, "get_executing_prompt_id", lambda: "prompt-A")

    changed = orch._refresh_prompt_state_if_needed()

    assert changed is False
    assert orch._PROMPT_ID == "prompt-A"
    assert orch._PROMPT_START_TS == 100.0


def test_sage_tracer_rotates_file_handle_on_prompt_change(tmp_path):
    """SageTracer must rotate its file handle when the emit's prompt_id
    differs from the cached one. ComfyUI caches the sage node's
    `_patch_impl` output across renders, so the same SageTracer instance
    is reused — without rotation, all renders' events accumulate in the
    first prompt's file.
    """
    from nodes_sage import SageTracer

    path_a = tmp_path / "prompt-A" / "sage.jsonl"
    path_b = tmp_path / "prompt-B" / "sage.jsonl"
    next_path = [path_a]

    tracer = SageTracer(path_a, resolve_log_path=lambda: next_path[0])
    tracer.emit(shape=(1, 100, 2048), has_mask=False, mode="auto",
                fell_back=False, elapsed_us=10.0, prompt_id="A")
    tracer.emit(shape=(1, 100, 2048), has_mask=False, mode="auto",
                fell_back=False, elapsed_us=10.0, prompt_id="A")
    same_prompt_lines = len(path_a.read_text().splitlines())

    # Prompt changes; resolver returns a new path
    next_path[0] = path_b
    tracer.emit(shape=(1, 100, 2048), has_mask=False, mode="auto",
                fell_back=False, elapsed_us=10.0, prompt_id="B")

    assert path_a.read_text().splitlines() == path_a.read_text().splitlines()[:same_prompt_lines]
    assert path_b.exists()
    assert "prompt_id" in path_b.read_text()
    assert '"B"' in path_b.read_text()
    # path_a should not have the prompt-B event
    assert '"B"' not in path_a.read_text()


def test_sage_tracer_no_rotation_without_resolver(tmp_path):
    """If no resolver is provided (legacy call sites), SageTracer must
    not crash on prompt change — it just keeps writing to the initial
    path. Preserves backwards compatibility for any caller that hasn't
    been updated.
    """
    from nodes_sage import SageTracer

    path = tmp_path / "sage.jsonl"
    tracer = SageTracer(path, resolve_log_path=None)
    tracer.emit(shape=(1, 100, 2048), has_mask=False, mode="auto",
                fell_back=False, elapsed_us=10.0, prompt_id="A")
    tracer.emit(shape=(1, 100, 2048), has_mask=False, mode="auto",
                fell_back=False, elapsed_us=10.0, prompt_id="B")

    # Both events land in the same file
    assert '"A"' in path.read_text()
    assert '"B"' in path.read_text()
