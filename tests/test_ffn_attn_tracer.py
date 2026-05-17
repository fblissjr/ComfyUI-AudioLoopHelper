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
