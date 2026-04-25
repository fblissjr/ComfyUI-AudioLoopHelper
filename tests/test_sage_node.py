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

from _fakes import FakeModelWithCallbacks as FakeModel


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


def test_tracer_enabled_writes_per_call_rows_and_summary(tmp_path: Path, monkeypatch):
    m = _mod()
    # Force no-arch path so the row count is deterministic across CUDA
    # and CPU-only test hosts. The arch-stamping behavior has its own
    # test below.
    monkeypatch.setattr(m, "_detect_arch_tag", lambda: None)
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


def test_tracer_writes_header_row_with_arch_when_detected(tmp_path: Path, monkeypatch):
    """When `_detect_arch_tag()` returns a string, the tracer writes a
    one-time header row at init AND stamps the arch field into every
    per-call row. This is what makes traces self-describing for the
    summary script's kernel inference -- no --arch flag needed."""
    m = _mod()
    monkeypatch.setattr(m, "_detect_arch_tag", lambda: "sm89_cuda12_8")
    log = tmp_path / "sage_test_with_arch.jsonl"
    tracer = m.SageTracer(log_path=log)
    tracer.emit(shape=(1, 64, 64), has_mask=False, mode="auto", fell_back=False, elapsed_us=10.0)
    tracer.flush_summary()

    rows = [orjson.loads(line) for line in log.read_text().splitlines()]
    # 1 header + 1 per-call + 1 summary
    assert len(rows) == 3
    assert rows[0]["event"] == "header"
    assert rows[0]["arch"] == "sm89_cuda12_8"
    assert rows[1]["arch"] == "sm89_cuda12_8"  # stamped per-call too
    assert rows[1]["effective_mode"] == "auto"  # unchanged behavior


def test_tracer_omits_arch_when_not_detected(tmp_path: Path, monkeypatch):
    """No GPU / unsupported arch -> no header, no arch field. Don't
    stamp 'unknown' or empty strings -- the absence is itself signal
    that the summary script should fall back to --arch / autodetect."""
    m = _mod()
    monkeypatch.setattr(m, "_detect_arch_tag", lambda: None)
    log = tmp_path / "sage_test_no_arch.jsonl"
    tracer = m.SageTracer(log_path=log)
    tracer.emit(shape=(1, 64, 64), has_mask=False, mode="auto", fell_back=False, elapsed_us=10.0)
    tracer.flush_summary()

    rows = [orjson.loads(line) for line in log.read_text().splitlines()]
    # 1 per-call + 1 summary (no header)
    assert len(rows) == 2
    assert "arch" not in rows[0]
    assert rows[0]["effective_mode"] == "auto"


def test_tracer_emit_stamps_dispatched_kernel_when_given(tmp_path: Path, monkeypatch):
    """When `dispatched_kernel` is passed (non-None), tracer stamps it
    into the per-call row. This is the field consumed by sage-fork's
    `get_last_dispatched_kernel()` -- exact string from sage's
    KNOWN_KERNEL_NAMES vocabulary, no consumer-side inference needed."""
    m = _mod()
    monkeypatch.setattr(m, "_detect_arch_tag", lambda: None)
    log = tmp_path / "sage_dispatched.jsonl"
    tracer = m.SageTracer(log_path=log)
    tracer.emit(shape=(1, 64, 64), has_mask=False, mode="auto",
                fell_back=False, elapsed_us=10.0, dispatched_kernel="fp8_cuda++")
    tracer.flush_summary()

    rows = [orjson.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]["dispatched_kernel"] == "fp8_cuda++"


def test_tracer_emit_omits_dispatched_kernel_when_none(tmp_path: Path, monkeypatch):
    """No symbol available / not yet measured -> field absent, not
    stamped as null. The summary script's contract is 'field present
    means trustworthy'; stamping null would conflate 'not measured'
    with 'measured to be unknown'."""
    m = _mod()
    monkeypatch.setattr(m, "_detect_arch_tag", lambda: None)
    log = tmp_path / "sage_no_dispatched.jsonl"
    tracer = m.SageTracer(log_path=log)
    tracer.emit(shape=(1, 64, 64), has_mask=False, mode="auto",
                fell_back=False, elapsed_us=10.0, dispatched_kernel=None)
    tracer.flush_summary()

    rows = [orjson.loads(line) for line in log.read_text().splitlines()]
    assert "dispatched_kernel" not in rows[0]


def test_override_reads_get_last_dispatched_kernel_after_sage_call(tmp_path: Path, monkeypatch):
    """The override reads sage-fork's `get_last_dispatched_kernel()`
    immediately after sage_fn returns and forwards the resolved kernel
    name to tracer.emit. Thread-local API, must be read before any
    await/yield -- our override is synchronous, so this is safe."""
    m = _mod()
    monkeypatch.setattr(m, "_detect_arch_tag", lambda: None)
    # Stub sage-fork's symbol: a value that the next override call
    # should observe and stamp.
    monkeypatch.setattr(m, "_GET_DISPATCHED_KERNEL", lambda: "fp8_cuda++")

    q = _fake_q()
    k, v = _fake_kv(q)

    def sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape):
        return torch.zeros_like(q)

    log = tmp_path / "sage_override.jsonl"
    tracer = m.SageTracer(log_path=log)
    override = m.make_sage_override(
        sage_fn=sage_fn,
        pytorch_fn=lambda *a, **kw: torch.ones_like(q),
        mode="auto_mask_aware",
        fallback_on_error=True,
        tracer=tracer,
        logger=m.SageFallbackLogger(emit=lambda _: None),
    )
    override(None, q, k, v, heads=4, mask=None)
    tracer.flush_summary()

    rows = [orjson.loads(line) for line in log.read_text().splitlines()]
    per_call = next(r for r in rows if r.get("event") != "summary" and r.get("event") != "header")
    assert per_call["dispatched_kernel"] == "fp8_cuda++"


def test_override_handles_missing_get_last_dispatched_kernel(tmp_path: Path, monkeypatch):
    """Older sageattention installs lack the symbol. Override must
    not crash and must omit the field. Defensive: this is the back-
    compat case."""
    m = _mod()
    monkeypatch.setattr(m, "_detect_arch_tag", lambda: None)
    monkeypatch.setattr(m, "_GET_DISPATCHED_KERNEL", None)

    q = _fake_q()
    k, v = _fake_kv(q)

    def sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape):
        return torch.zeros_like(q)

    log = tmp_path / "sage_no_symbol.jsonl"
    tracer = m.SageTracer(log_path=log)
    override = m.make_sage_override(
        sage_fn=sage_fn,
        pytorch_fn=lambda *a, **kw: torch.ones_like(q),
        mode="auto_mask_aware",
        fallback_on_error=True,
        tracer=tracer,
        logger=m.SageFallbackLogger(emit=lambda _: None),
    )
    override(None, q, k, v, heads=4, mask=None)
    tracer.flush_summary()

    rows = [orjson.loads(line) for line in log.read_text().splitlines()]
    per_call = next(r for r in rows if r.get("event") != "summary" and r.get("event") != "header")
    assert "dispatched_kernel" not in per_call


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

    assert "on_cleanup" in clone.callbacks


def test_cleanup_callback_removes_override():
    m = _mod()
    model = FakeModel()
    (clone,) = m.AudioLoopHelperSageAttention._patch_impl(
        model, mode="auto", fallback_on_error=True
    )
    clone.callbacks["on_cleanup"]()
    assert "optimized_attention_override" not in clone.model_options["transformer_options"]
    clone.callbacks["on_cleanup"]()


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


# ---------------------------------------------------------------------------
# 7. Mask-aware mode. Sage's CUDA kernels don't implement mask support
# (MaskMode enum is {kNone, kCausal}; attn_mask is silently dropped via
# kwargs); only fp16_triton has a masked path. auto_mask_aware routes
# masked calls to triton and unmasked calls to the primary kernel.
# ---------------------------------------------------------------------------


def test_all_arches_include_auto_mask_aware():
    m = _mod()
    for arch in ("sm80", "sm86", "sm87", "sm89", "sm90", "sm100", "sm120", "sm121", None):
        assert "auto_mask_aware" in m.build_mode_list(arch), arch


def test_auto_mask_aware_routes_masked_call_to_triton_kernel(monkeypatch: pytest.MonkeyPatch):
    m = _mod()
    dispatches: list[str] = []

    def fake_triton(q, k, v, **kw):
        dispatches.append("triton")
        return v  # any shape-compatible tensor

    def fake_auto(q, k, v, **kw):
        dispatches.append("auto")
        return v

    monkeypatch.setitem(m._SAGE_KERNELS, "sageattn_qk_int8_pv_fp16_triton", fake_triton)
    monkeypatch.setitem(m._SAGE_KERNELS, "auto", fake_auto)

    sage_fn = m._build_sage_fn("auto_mask_aware")
    q = _fake_q(); k, v = _fake_kv(q)
    mask = torch.zeros(1, q.shape[1], q.shape[1])

    sage_fn(q, k, v, heads=4, mask=mask, skip_reshape=False, skip_output_reshape=False)
    assert dispatches == ["triton"]


def test_auto_mask_aware_routes_unmasked_call_to_primary_kernel(monkeypatch: pytest.MonkeyPatch):
    m = _mod()
    dispatches: list[str] = []

    def fake_triton(q, k, v, **kw):
        dispatches.append("triton")
        return v

    def fake_auto(q, k, v, **kw):
        dispatches.append("auto")
        return v

    monkeypatch.setitem(m._SAGE_KERNELS, "sageattn_qk_int8_pv_fp16_triton", fake_triton)
    monkeypatch.setitem(m._SAGE_KERNELS, "auto", fake_auto)

    sage_fn = m._build_sage_fn("auto_mask_aware")
    q = _fake_q(); k, v = _fake_kv(q)

    sage_fn(q, k, v, heads=4, mask=None, skip_reshape=False, skip_output_reshape=False)
    assert dispatches == ["auto"]


def test_override_tracer_records_effective_mode_for_mask_aware(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    m = _mod()

    def fake_sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape):
        return v

    log = tmp_path / "sage_test.jsonl"
    tracer = m.SageTracer(log_path=log)
    override = m.make_sage_override(
        sage_fn=fake_sage_fn,
        pytorch_fn=lambda *a, **kw: None,
        mode="auto_mask_aware",
        fallback_on_error=True,
        tracer=tracer,
        logger=m.SageFallbackLogger(emit=lambda _: None),
    )

    q = _fake_q(); k, v = _fake_kv(q)
    mask = torch.zeros(1, 1, q.shape[1])
    override(None, q, k, v, heads=4, mask=mask)       # masked call
    override(None, q, k, v, heads=4, mask=None)       # unmasked call
    tracer.flush_summary()

    rows = [orjson.loads(l) for l in log.read_text().splitlines() if l]
    per_call = [r for r in rows if "event" not in r]
    assert len(per_call) == 2
    # Masked call effective_mode == triton; unmasked == configured primary.
    assert per_call[0]["mode"] == "auto_mask_aware"
    assert per_call[0]["effective_mode"] == "sageattn_qk_int8_pv_fp16_triton"
    assert per_call[1]["mode"] == "auto_mask_aware"
    assert per_call[1]["effective_mode"] != "sageattn_qk_int8_pv_fp16_triton"


def test_explicit_mode_tracer_effective_mode_matches_configured_mode(tmp_path: Path):
    # Regression: non-mask-aware modes must report effective_mode == mode
    # for both masked and unmasked calls.
    m = _mod()

    def fake_sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape):
        return v

    log = tmp_path / "sage_explicit.jsonl"
    tracer = m.SageTracer(log_path=log)
    override = m.make_sage_override(
        sage_fn=fake_sage_fn,
        pytorch_fn=lambda *a, **kw: None,
        mode="sageattn_qk_int8_pv_fp8_cuda++",
        fallback_on_error=True,
        tracer=tracer,
        logger=m.SageFallbackLogger(emit=lambda _: None),
    )
    q = _fake_q(); k, v = _fake_kv(q)
    override(None, q, k, v, heads=4, mask=torch.zeros(1, 1, q.shape[1]))
    override(None, q, k, v, heads=4, mask=None)
    tracer.flush_summary()

    rows = [orjson.loads(l) for l in log.read_text().splitlines() if l]
    per_call = [r for r in rows if "event" not in r]
    assert all(r["effective_mode"] == "sageattn_qk_int8_pv_fp8_cuda++" for r in per_call)


def test_default_mode_is_mask_aware():
    """auto_mask_aware is the right default: fp8++ on the masked cross-attn
    path produces rtol=0.44 vs SDPA on LTX shapes. auto_mask_aware routes
    around this while keeping fp8++ speed on self-attn."""
    m = _mod()
    assert m._DEFAULT_MODE == "auto_mask_aware"
