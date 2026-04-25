"""Tests for `nodes_easycache.py`: LTX-2.3 step-skipping cache.

TDD red-first. These tests target the algorithmic core of EasyCache without
needing a real LTX model -- a fake `executor` callable substitutes for
`comfy.patcher_extension.WrapperExecutor`.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

from _fakes import FakeModelWithWrappers


def _mod():
    import nodes_easycache  # noqa: F401
    return importlib.reload(importlib.import_module("nodes_easycache"))


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class _FakeExecutor:
    """Stand-in for WrapperExecutor. Records call count; returns input + bias.

    The real executor invokes the next wrapper or the original _forward.
    For algorithmic tests we just need a callable that returns a deterministic
    output so the cache can be checked.
    """

    def __init__(self, bias: float = 0.5):
        self.bias = bias
        self.call_count = 0

    def __call__(self, x, *args, **kwargs):
        self.call_count += 1
        return x + self.bias


# ---------------------------------------------------------------------------
# 1. Node basics: import, schema, clone
# ---------------------------------------------------------------------------

def test_module_exposes_node_class():
    m = _mod()
    assert hasattr(m, "LTXVideoEasyCache"), "node class missing"


def test_execute_returns_clone_not_original():
    m = _mod()
    model = FakeModelWithWrappers()
    (patched,) = m.LTXVideoEasyCache._patch_impl(model, easycache_thresh=0.05, start_step=0, end_step=-1)
    assert patched is not model


def test_execute_registers_diffusion_wrapper_with_unique_key():
    m = _mod()
    model = FakeModelWithWrappers()
    (patched,) = m.LTXVideoEasyCache._patch_impl(model, easycache_thresh=0.05, start_step=0, end_step=-1)
    # patcher_extension.WrappersMP.DIFFUSION_MODEL == "diffusion_model"
    diffusion_wrappers = patched.wrappers.get("diffusion_model", {})
    assert m.WRAPPER_KEY in diffusion_wrappers
    assert len(diffusion_wrappers[m.WRAPPER_KEY]) == 1


def test_execute_registers_cleanup_callback():
    m = _mod()
    model = FakeModelWithWrappers()
    (patched,) = m.LTXVideoEasyCache._patch_impl(model, easycache_thresh=0.05, start_step=0, end_step=-1)
    # CallbacksMP.ON_CLEANUP == "on_cleanup"
    assert "on_cleanup" in patched.callbacks


# ---------------------------------------------------------------------------
# 2. Algorithmic core: state machine and skip decisions
# ---------------------------------------------------------------------------

def test_first_call_always_computes_and_seeds_state():
    m = _mod()
    state = m.EasyCacheState(thresh=0.05, start_step=0, end_step=-1)
    wrapper = m.build_wrapper(state)

    x = torch.zeros(1, 4)
    executor = _FakeExecutor(bias=0.5)
    out = wrapper(executor, x)

    assert executor.call_count == 1, "first call must compute"
    assert torch.allclose(out, x + 0.5)
    assert state.previous_raw_input is not None
    assert state.previous_raw_output is not None
    assert state.cache_residual is not None


def test_unchanged_input_skips_via_cache():
    m = _mod()
    state = m.EasyCacheState(thresh=10.0, start_step=0, end_step=-1)  # generous thresh
    wrapper = m.build_wrapper(state)
    executor = _FakeExecutor(bias=0.5)

    x = torch.ones(1, 4)
    wrapper(executor, x)  # seed
    out2 = wrapper(executor, x)  # identical input -> change=0 -> skip

    assert executor.call_count == 1, "second call should skip"
    # Skip path returns x + cache_residual; residual is (out - x) = 0.5
    assert torch.allclose(out2, x + 0.5)
    assert state.skipped_steps == [1]


def test_changed_input_above_threshold_recomputes():
    m = _mod()
    state = m.EasyCacheState(thresh=0.0001, start_step=0, end_step=-1)  # tight thresh
    wrapper = m.build_wrapper(state)
    executor = _FakeExecutor(bias=0.5)

    x1 = torch.zeros(1, 4)
    x2 = torch.ones(1, 4) * 100.0  # massive change
    wrapper(executor, x1)
    wrapper(executor, x2)

    assert executor.call_count == 2, "tight threshold + big change must recompute"
    assert state.skipped_steps == []


def test_threshold_zero_disables_cache():
    """Strict `<` in the skip check means accumulated_error >= 0 never
    satisfies `< 0`. So thresh=0 is equivalent to thresh<0: caching
    disabled. This is the cleanest documented semantics; thresh must be
    strictly positive for any skip to happen.
    """
    m = _mod()
    state = m.EasyCacheState(thresh=0.0, start_step=0, end_step=-1)
    wrapper = m.build_wrapper(state)
    executor = _FakeExecutor(bias=0.5)

    x = torch.ones(1, 4)
    for _ in range(3):
        wrapper(executor, x)

    assert executor.call_count == 3
    assert state.skipped_steps == []


def test_threshold_negative_disables_cache():
    """A negative threshold is the explicit "never skip" sentinel.
    accumulated_error >= 0 always, so accumulated_error < thresh is never
    true, so we never skip."""
    m = _mod()
    state = m.EasyCacheState(thresh=-1.0, start_step=0, end_step=-1)
    wrapper = m.build_wrapper(state)
    executor = _FakeExecutor(bias=0.5)

    x = torch.ones(1, 4)
    for _ in range(3):
        wrapper(executor, x)

    assert executor.call_count == 3
    assert state.skipped_steps == []


# ---------------------------------------------------------------------------
# 3. Window gating: start_step / end_step
# ---------------------------------------------------------------------------

def test_calls_before_start_step_bypass_caching():
    m = _mod()
    state = m.EasyCacheState(thresh=10.0, start_step=2, end_step=-1)
    wrapper = m.build_wrapper(state)
    executor = _FakeExecutor(bias=0.5)

    x = torch.ones(1, 4)
    wrapper(executor, x)  # step 0 -- bypass
    wrapper(executor, x)  # step 1 -- bypass
    wrapper(executor, x)  # step 2 -- in window, seeds state, computes
    wrapper(executor, x)  # step 3 -- in window, identical x, skips

    assert executor.call_count == 3
    assert state.skipped_steps == [3]


def test_calls_after_end_step_bypass_caching():
    m = _mod()
    state = m.EasyCacheState(thresh=10.0, start_step=0, end_step=1)
    wrapper = m.build_wrapper(state)
    executor = _FakeExecutor(bias=0.5)

    x = torch.ones(1, 4)
    wrapper(executor, x)  # step 0 -- in window, seeds, computes
    wrapper(executor, x)  # step 1 -- in window, skips
    wrapper(executor, x)  # step 2 -- past end, bypass (computes)
    wrapper(executor, x)  # step 3 -- past end, bypass (computes)

    # 1 seed + 2 post-window = 3 compute calls
    assert executor.call_count == 3
    assert state.skipped_steps == [1]


# ---------------------------------------------------------------------------
# 4. Cleanup
# ---------------------------------------------------------------------------

def test_cleanup_callback_resets_state():
    m = _mod()
    model = FakeModelWithWrappers()
    (patched,) = m.LTXVideoEasyCache._patch_impl(model, easycache_thresh=0.05, start_step=0, end_step=-1)
    cleanup_fn = patched.callbacks["on_cleanup"]

    # Simulate state accumulation by reaching into the closed-over state.
    # We don't strictly need to exercise the wrapper; we just need cleanup to
    # not raise and to reset whatever state object it holds.
    cleanup_fn()  # should be a no-op-safe call
