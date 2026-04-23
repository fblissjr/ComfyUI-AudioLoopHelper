"""Tests for `nodes_sage.py`: the AudioLoopHelper-native sage attention node.

TDD red-first. Written before nodes_sage.py exists -- each test below fails
initially with ImportError, drives the implementation.

No GPU required: sage and pytorch fallback functions are injected as fakes
that return shape-correct tensors or raise on demand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import orjson
import pytest
import torch


# ---------------------------------------------------------------------------
# Import the module under test lazily so the file can be collected even
# before nodes_sage.py exists. Each test calls _mod() at its start.
# ---------------------------------------------------------------------------

def _mod():
    import importlib

    import nodes_sage  # noqa: F401 -- re-imported to make reload safe
    return importlib.reload(importlib.import_module("nodes_sage"))


# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------

class FakeModel:
    """Minimal stand-in for comfy.model_patcher.ModelPatcher.

    Only the surface that AudioLoopHelperSageAttention touches:
      - clone() returns a deep-ish copy (new dict for model_options)
      - model_options["transformer_options"] dict
      - add_callback(call_type, fn) stores callbacks for later invocation
    """

    def __init__(self):
        self.model_options: dict = {"transformer_options": {}}
        self.callbacks: list[tuple[str, Callable]] = []

    def clone(self) -> "FakeModel":
        clone = FakeModel()
        # Deep-enough copy: a fresh transformer_options dict so override
        # writes don't leak between clone and original.
        clone.model_options = {
            "transformer_options": dict(self.model_options.get("transformer_options", {}))
        }
        return clone

    def add_callback(self, call_type: str, fn):
        self.callbacks.append((call_type, fn))


def _fake_q(batch=1, seq=64, heads=4, dim_head=16, dtype=torch.float16):
    # Layout matches LTX (skip_reshape=False): (B, S, heads*dim_head).
    return torch.randn(batch, seq, heads * dim_head, dtype=dtype)


def _fake_kv(q):
    return torch.randn_like(q), torch.randn_like(q)


# ---------------------------------------------------------------------------
# 1. Arch-filtered mode list
# ---------------------------------------------------------------------------

def test_arch_filter_sm89_excludes_blackwell_and_hopper():
    m = _mod()
    modes = m.build_mode_list("sm89")
    assert "disabled" in modes
    assert "auto" in modes
    assert "sageattn_qk_int8_pv_fp8_cuda++" in modes
    assert "sageattn_qk_int8_pv_fp16_cuda" in modes
    # sm89 must NOT see sm90/Blackwell footguns.
    assert not any("sm90" in x for x in modes)
    assert not any("sageattn3" in x for x in modes)


def test_arch_filter_sm90_includes_sm90_mode():
    m = _mod()
    modes = m.build_mode_list("sm90")
    assert any("sm90" in x for x in modes), modes


def test_arch_filter_sm100_includes_sageattn3():
    m = _mod()
    modes = m.build_mode_list("sm100")
    assert any("sageattn3" in x for x in modes), modes


def test_arch_filter_unknown_falls_back_to_minimal():
    m = _mod()
    modes = m.build_mode_list(None)
    # Even with an unknown arch the Triton path is always listed as an
    # option, along with disabled/auto.
    assert "disabled" in modes
    assert "auto" in modes
    assert "sageattn_qk_int8_pv_fp16_triton" in modes


# ---------------------------------------------------------------------------
# 2. Dedup'd fallback logger
# ---------------------------------------------------------------------------

def test_fallback_logger_dedups_by_shape_mode_errtype():
    m = _mod()
    logged = []
    logger = m.SageFallbackLogger(emit=logged.append)
    err = RuntimeError("boom")
    shape = (1, 64, 64)
    for _ in range(100):
        logger.log_once(shape, "auto", err)
    assert len(logged) == 1, logged
    # Distinct shape -> another line.
    logger.log_once((1, 128, 64), "auto", err)
    assert len(logged) == 2
    # Distinct error type -> another line.
    logger.log_once(shape, "auto", ValueError("x"))
    assert len(logged) == 3


# ---------------------------------------------------------------------------
# 3. Override fn: success path, fallback path, fallback disabled
# ---------------------------------------------------------------------------

def test_override_success_calls_sage_fn_and_returns_result():
    m = _mod()
    q = _fake_q()
    k, v = _fake_kv(q)

    calls = {"sage": 0, "pt": 0}

    def sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape):
        calls["sage"] += 1
        return torch.zeros_like(q)

    def pytorch_fn(*args, **kwargs):
        calls["pt"] += 1
        return torch.ones_like(q)

    override = m.make_sage_override(
        sage_fn=sage_fn,
        pytorch_fn=pytorch_fn,
        mode="auto",
        fallback_on_error=True,
        tracer=m.SageTracer(log_path=None),  # disabled tracer
        logger=m.SageFallbackLogger(emit=lambda _: None),
    )
    out = override(None, q, k, v, heads=4, mask=None)
    assert calls == {"sage": 1, "pt": 0}
    assert torch.equal(out, torch.zeros_like(q))


def test_override_falls_back_to_pytorch_on_exception():
    m = _mod()
    q = _fake_q()
    k, v = _fake_kv(q)
    calls = {"sage": 0, "pt": 0}
    logged: list = []

    def sage_fn(*args, **kwargs):
        calls["sage"] += 1
        raise RuntimeError("sage blew up")

    def pytorch_fn(*args, **kwargs):
        calls["pt"] += 1
        return torch.ones_like(q)

    override = m.make_sage_override(
        sage_fn=sage_fn,
        pytorch_fn=pytorch_fn,
        mode="auto",
        fallback_on_error=True,
        tracer=m.SageTracer(log_path=None),
        logger=m.SageFallbackLogger(emit=logged.append),
    )
    out = override(None, q, k, v, heads=4, mask=None)
    assert calls["sage"] == 1
    assert calls["pt"] == 1
    assert torch.equal(out, torch.ones_like(q))
    assert len(logged) == 1, logged


def test_override_reraises_when_fallback_disabled():
    m = _mod()
    q = _fake_q()
    k, v = _fake_kv(q)

    def sage_fn(*args, **kwargs):
        raise RuntimeError("nope")

    def pytorch_fn(*args, **kwargs):
        pytest.fail("pytorch_fn must not be called when fallback_on_error=False")

    override = m.make_sage_override(
        sage_fn=sage_fn,
        pytorch_fn=pytorch_fn,
        mode="auto",
        fallback_on_error=False,
        tracer=m.SageTracer(log_path=None),
        logger=m.SageFallbackLogger(emit=lambda _: None),
    )
    with pytest.raises(RuntimeError, match="nope"):
        override(None, q, k, v, heads=4, mask=None)


# ---------------------------------------------------------------------------
# 4. Telemetry: disabled (no file), enabled (jsonl output + summary)
# ---------------------------------------------------------------------------

def test_tracer_disabled_writes_nothing(tmp_path: Path):
    m = _mod()
    tracer = m.SageTracer(log_path=None)
    tracer.emit(shape=(1, 64, 64), has_mask=False, mode="auto", fell_back=False, elapsed_us=12.0)
    tracer.flush_summary()
    # Disabled tracer cannot produce a file because it has no path.
    # Just verify no crash and no side-effects.
    assert list(tmp_path.iterdir()) == []


def test_tracer_enabled_writes_per_call_rows_and_summary(tmp_path: Path):
    m = _mod()
    log = tmp_path / "sage_test.jsonl"
    tracer = m.SageTracer(log_path=log)
    tracer.emit(shape=(1, 64, 64), has_mask=False, mode="auto", fell_back=False, elapsed_us=12.0)
    tracer.emit(shape=(1, 64, 64), has_mask=True, mode="auto", fell_back=True, elapsed_us=80.0)
    tracer.flush_summary()

    lines = log.read_text().splitlines()
    # 2 per-call rows + 1 summary row.
    assert len(lines) == 3
    rows = [orjson.loads(l) for l in lines]
    assert rows[0]["shape"] == [1, 64, 64]
    assert rows[0]["has_mask"] is False
    assert rows[0]["fell_back"] is False
    assert rows[1]["fell_back"] is True
    summary = rows[-1]
    assert summary["event"] == "summary"
    assert summary["total_calls"] == 2
    assert summary["fallback_count"] == 1


# ---------------------------------------------------------------------------
# 5. Node.execute: disabled returns input model; active sets override + cleanup
# ---------------------------------------------------------------------------

def test_execute_disabled_returns_same_model():
    m = _mod()
    model = FakeModel()
    (out,) = m.AudioLoopHelperSageAttention._patch_impl(
        model, mode="disabled", fallback_on_error=True
    )
    assert out is model
    assert "optimized_attention_override" not in out.model_options["transformer_options"]


def test_execute_sets_override_and_registers_cleanup():
    m = _mod()
    model = FakeModel()
    (clone,) = m.AudioLoopHelperSageAttention._patch_impl(
        model, mode="auto", fallback_on_error=True
    )
    # Cloned (not the same instance) so the original model stays clean.
    assert clone is not model
    assert "optimized_attention_override" not in model.model_options["transformer_options"]

    override = clone.model_options["transformer_options"].get("optimized_attention_override")
    assert callable(override)

    cleanup_events = [ct for ct, _ in clone.callbacks]
    # Use the CallbacksMP.ON_CLEANUP constant value from comfy.patcher_extension.
    # Its literal is "on_cleanup".
    assert "on_cleanup" in cleanup_events


def test_cleanup_callback_removes_override():
    m = _mod()
    model = FakeModel()
    (clone,) = m.AudioLoopHelperSageAttention._patch_impl(
        model, mode="auto", fallback_on_error=True
    )
    override = clone.model_options["transformer_options"]["optimized_attention_override"]
    # Find and invoke the cleanup callback.
    cleanup_fn = next(fn for ct, fn in clone.callbacks if ct == "on_cleanup")
    cleanup_fn()
    assert "optimized_attention_override" not in clone.model_options["transformer_options"]
    # Idempotent: running again on a clean state doesn't raise.
    cleanup_fn()


# ---------------------------------------------------------------------------
# 6. Env-gated tracer integration (does not require running the full node)
# ---------------------------------------------------------------------------

def test_resolve_trace_path_unset_returns_none(monkeypatch: pytest.MonkeyPatch):
    m = _mod()
    monkeypatch.delenv("AUDIOLOOPHELPER_SAGE_TRACE", raising=False)
    assert m.resolve_trace_path() is None


def test_resolve_trace_path_truthy_returns_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    m = _mod()
    monkeypatch.setenv("AUDIOLOOPHELPER_SAGE_TRACE", str(tmp_path / "sage.jsonl"))
    path = m.resolve_trace_path()
    assert path is not None
    assert str(path).endswith("sage.jsonl")


# Autouse: nodes_sage caches its arch at import. For tests that depend on a
# specific arch list, we reload the module after monkeypatching.
@pytest.fixture(autouse=True)
def _reset_trace_env(monkeypatch: pytest.MonkeyPatch):
    # Tests should not leak an env var that enables telemetry.
    monkeypatch.delenv("AUDIOLOOPHELPER_SAGE_TRACE", raising=False)
    yield
