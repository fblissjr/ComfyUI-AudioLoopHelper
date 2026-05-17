"""Tests for `tracers/ffn_attn.py::FfnAttnTracer`.

Background: the 2026-05-16 audit session ran with
`AUDIOLOOPHELPER_SAGE_OUTPUT_FINGERPRINT=auto` simultaneously enabled
and quoted the forward-hook `elapsed_ms` measurements as production-
comparable timings. They were actually inflated 1.5-2× because each
sage `.item()` call in fingerprint mode forces a CUDA sync the
surrounding sub-module forward waits on. The tracer now warns when
both env vars are active so future renders make the distortion visible.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _install_path() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def test_warn_when_fingerprint_also_active(monkeypatch, tmp_path: Path):
    """With both env vars set, `install_at_render` must emit a stderr line
    naming the fingerprint-mode timing inflation. Catches future
    refactors that drop the cross-env-var check.
    """
    _install_path()
    monkeypatch.setenv("AUDIOLOOPHELPER_FFN_ATTN_TRACE", "auto")
    monkeypatch.setenv("AUDIOLOOPHELPER_TORCH_PROFILE", "auto")
    monkeypatch.setenv("AUDIOLOOPHELPER_SAGE_OUTPUT_FINGERPRINT", "auto")
    monkeypatch.setenv("RUN_ID", "ffn_attn_warn_test")
    monkeypatch.setenv("AUDIOLOOPHELPER_PER_PROMPT", "0")

    from tracers.ffn_attn import FfnAttnTracer
    tracer = FfnAttnTracer()

    # Fake model_clone: install path walks `transformer_blocks`. Empty list
    # means "nothing to hook" — install returns False but the log line
    # we're testing fires BEFORE the hook walk. We capture stderr around
    # the entire call.
    class _FakeModelClone:
        def get_model_object(self, _name):
            class _M:
                transformer_blocks = []
            return _M()

    buf = io.StringIO()
    with redirect_stderr(buf):
        tracer.install_at_render(_FakeModelClone())
    err = buf.getvalue()

    assert "fingerprint mode also active" in err, err
    assert "inflated" in err


def test_no_warn_when_fingerprint_inactive(monkeypatch, tmp_path: Path):
    """Without `AUDIOLOOPHELPER_SAGE_OUTPUT_FINGERPRINT` the warning must
    not appear. Verifies the gate isn't bug-on-by-default.
    """
    _install_path()
    monkeypatch.setenv("AUDIOLOOPHELPER_FFN_ATTN_TRACE", "auto")
    monkeypatch.setenv("AUDIOLOOPHELPER_TORCH_PROFILE", "auto")
    monkeypatch.delenv("AUDIOLOOPHELPER_SAGE_OUTPUT_FINGERPRINT", raising=False)
    monkeypatch.setenv("RUN_ID", "ffn_attn_clean_test")
    monkeypatch.setenv("AUDIOLOOPHELPER_PER_PROMPT", "0")

    from tracers.ffn_attn import FfnAttnTracer
    tracer = FfnAttnTracer()

    class _FakeModelClone:
        def get_model_object(self, _name):
            class _M:
                transformer_blocks = []
            return _M()

    buf = io.StringIO()
    with redirect_stderr(buf):
        tracer.install_at_render(_FakeModelClone())
    err = buf.getvalue()

    assert "fingerprint mode" not in err


def test_emit_annotations_gated_on_torch_profile_env(monkeypatch):
    """Independent gate test — `_emit_annotations` must mirror the
    `AUDIOLOOPHELPER_TORCH_PROFILE` env state at install time. Locks in
    the hot-path optimisation: hooks skip the C++-cost `record_function`
    bookkeeping entirely when torch.profiler isn't capturing.
    """
    _install_path()
    monkeypatch.setenv("AUDIOLOOPHELPER_FFN_ATTN_TRACE", "auto")
    monkeypatch.setenv("RUN_ID", "ffn_attn_gate_test")
    monkeypatch.setenv("AUDIOLOOPHELPER_PER_PROMPT", "0")

    from tracers.ffn_attn import FfnAttnTracer

    class _FakeModelClone:
        def get_model_object(self, _name):
            class _M:
                transformer_blocks = []
            return _M()

    # Case 1: torch_profile OFF → _emit_annotations False
    monkeypatch.delenv("AUDIOLOOPHELPER_TORCH_PROFILE", raising=False)
    tracer = FfnAttnTracer()
    tracer.install_at_render(_FakeModelClone())
    assert tracer._emit_annotations is False

    # Case 2: torch_profile ON → _emit_annotations True
    monkeypatch.setenv("AUDIOLOOPHELPER_TORCH_PROFILE", "auto")
    tracer2 = FfnAttnTracer()
    tracer2.install_at_render(_FakeModelClone())
    assert tracer2._emit_annotations is True
