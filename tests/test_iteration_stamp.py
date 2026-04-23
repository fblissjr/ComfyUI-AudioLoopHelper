"""Tests for LoopIterationStamp: a MODEL passthrough that writes
`transformer_options["iteration"]` from TensorLoopOpen's current_iteration.

Red-first. Drives the implementation in nodes.py. Unblocks backlog item 7
(sage offload-asymmetry verification) by giving the sage tracer a real
per-iteration stamp to group by.
"""

from __future__ import annotations

from typing import Callable

from _fakes import FakeModelPatcher as FakeModel


def _stamp(model, current_iteration):
    """Invoke the node under test through its testable seam."""
    from nodes import LoopIterationStamp

    return LoopIterationStamp._stamp_impl(model, current_iteration=current_iteration)


# ---------------------------------------------------------------------------
# Core semantics
# ---------------------------------------------------------------------------


def test_stamp_writes_iteration_into_transformer_options():
    model = FakeModel()
    (out,) = _stamp(model, 5)
    assert out.model_options["transformer_options"]["iteration"] == 5


def test_stamp_does_not_mutate_input_model():
    model = FakeModel()
    _stamp(model, 7)
    assert "iteration" not in model.model_options["transformer_options"]


def test_stamp_preserves_existing_transformer_options_keys():
    sentinel: Callable = lambda *_a, **_kw: None  # noqa: E731 -- stand-in for sage override
    model = FakeModel(transformer_options={"optimized_attention_override": sentinel})
    (out,) = _stamp(model, 3)
    opts = out.model_options["transformer_options"]
    assert opts["optimized_attention_override"] is sentinel
    assert opts["iteration"] == 3


def test_stamp_coerces_iteration_to_int():
    model = FakeModel()
    (out,) = _stamp(model, "4")  # type: ignore[arg-type]
    assert out.model_options["transformer_options"]["iteration"] == 4
    assert isinstance(out.model_options["transformer_options"]["iteration"], int)


def test_stamp_successive_calls_yield_independent_clones():
    model = FakeModel()
    (out_a,) = _stamp(model, 1)
    (out_b,) = _stamp(model, 2)
    assert out_a.model_options["transformer_options"]["iteration"] == 1
    assert out_b.model_options["transformer_options"]["iteration"] == 2
    # Mutating one clone must not leak into the other.
    out_a.model_options["transformer_options"]["iteration"] = 999
    assert out_b.model_options["transformer_options"]["iteration"] == 2


def test_stamp_creates_transformer_options_when_missing():
    model = FakeModel()
    # Simulate a bare ModelPatcher with no transformer_options dict at all.
    model.model_options = {}
    (out,) = _stamp(model, 0)
    assert out.model_options["transformer_options"]["iteration"] == 0


def test_stamp_overwrites_stale_iteration_from_prior_pass():
    # Loop reuses a patched model across iterations; a prior pass's stamp
    # lives in model_options. New stamp must win.
    model = FakeModel(transformer_options={"iteration": 99})
    (out,) = _stamp(model, 5)
    assert out.model_options["transformer_options"]["iteration"] == 5


def test_stamp_preserves_nested_transformer_options_by_clone():
    # Production uses deepcopy_list_dict for clone(); a regression that
    # shallow-copies transformer_options could let the stamp mutate a
    # nested patches dict shared across clones.
    nested_patches = {"some_block": [lambda x: x]}
    model = FakeModel(transformer_options={"patches": nested_patches})
    (out,) = _stamp(model, 1)
    # Mutating the clone's patches dict must not affect the original.
    out.model_options["transformer_options"]["patches"]["new_block"] = "x"
    assert "new_block" not in model.model_options["transformer_options"]["patches"]
