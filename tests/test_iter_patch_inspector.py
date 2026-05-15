"""Tests for IterPatchInspector: a MODEL passthrough that logs per-call
patch state so we can verify whether NAG / sage / AttentionTuner /
ChunkedFFN patches survive comfy-aimdo's dynamic VRAM reload between
loop iterations.

Red-first. Drives the implementation in nodes.py.
"""

from __future__ import annotations

import logging

from _fakes import FakeModelPatcher as FakeModel
from _node_registry import assert_node_registered


def _inspect(model, label: str = "patch_inspect", verbose: bool = False):
    """Invoke the node under test through its testable seam."""
    from nodes import IterPatchInspector

    return IterPatchInspector._inspect_impl(
        model, label=label, verbose=verbose
    )


# ---------------------------------------------------------------------------
# Identity / passthrough
# ---------------------------------------------------------------------------


def test_inspector_returns_same_model_object_unchanged():
    model = FakeModel()
    (out,) = _inspect(model)
    assert out is model


def test_inspector_does_not_mutate_model_options():
    model = FakeModel(transformer_options={"existing_key": "v"})
    snapshot = dict(model.model_options["transformer_options"])
    _inspect(model)
    assert model.model_options["transformer_options"] == snapshot


def test_inspector_does_not_mutate_patches_dict():
    model = FakeModel()
    model.patches = {"blk.0.weight": [("foo", 1.0)]}
    before = dict(model.patches)
    _inspect(model)
    assert model.patches == before


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_inspector_logs_patch_count(caplog):
    model = FakeModel()
    model.patches = {"a": [], "b": []}
    with caplog.at_level(logging.INFO):
        _inspect(model, label="lbl1")
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "lbl1" in joined
    assert "patches=2" in joined


def test_inspector_logs_zero_patches_when_empty(caplog):
    model = FakeModel()
    model.patches = {}
    with caplog.at_level(logging.INFO):
        _inspect(model, label="lblZ")
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "patches=0" in joined


def test_inspector_logs_object_patches_count(caplog):
    model = FakeModel()
    model.patches = {}
    model.object_patches = {"diffusion_model.x": object()}
    with caplog.at_level(logging.INFO):
        _inspect(model)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "object_patches=1" in joined


def test_inspector_logs_transformer_options_keys(caplog):
    model = FakeModel(transformer_options={"iteration": 3, "foo": "bar"})
    with caplog.at_level(logging.INFO):
        _inspect(model)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    # Both keys must appear
    assert "iteration" in joined
    assert "foo" in joined


def test_inspector_logs_attention_override_flag_when_set(caplog):
    sentinel = lambda *a, **kw: None  # noqa: E731
    model = FakeModel(
        transformer_options={"optimized_attention_override": sentinel}
    )
    with caplog.at_level(logging.INFO):
        _inspect(model)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "attention_override=True" in joined


def test_inspector_logs_attention_override_flag_false_when_absent(caplog):
    model = FakeModel()
    with caplog.at_level(logging.INFO):
        _inspect(model)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "attention_override=False" in joined


def test_inspector_verbose_emits_patch_keys(caplog):
    model = FakeModel()
    model.patches = {"diffusion_model.blocks.0.foo": [], "lora_unet.bar": []}
    with caplog.at_level(logging.INFO):
        _inspect(model, verbose=True)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "diffusion_model.blocks.0.foo" in joined
    assert "lora_unet.bar" in joined


def test_inspector_non_verbose_omits_full_patch_keys(caplog):
    model = FakeModel()
    model.patches = {"diffusion_model.blocks.0.specific_key": []}
    with caplog.at_level(logging.INFO):
        _inspect(model, verbose=False)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "specific_key" not in joined


# ---------------------------------------------------------------------------
# Call counter
# ---------------------------------------------------------------------------


def test_inspector_call_counter_increments_across_calls(caplog):
    model = FakeModel()
    import nodes as _nodes

    # Reset the module-level counter so this test doesn't depend on prior
    # test ordering. Counter moved off the class because ComfyUI's v3 _io
    # API locks class attributes on the executor's clone — see nodes.py
    # comment above `_INSPECTOR_CALL_COUNTERS`.
    _nodes._INSPECTOR_CALL_COUNTERS.pop("patch_inspect", None)
    with caplog.at_level(logging.INFO):
        _inspect(model)
        _inspect(model)
        _inspect(model)
    counters = [
        r.getMessage() for r in caplog.records if "patch_inspect" in r.getMessage()
    ]
    assert any("call=1" in m for m in counters)
    assert any("call=2" in m for m in counters)
    assert any("call=3" in m for m in counters)


# ---------------------------------------------------------------------------
# Tolerance for missing surfaces
# ---------------------------------------------------------------------------


def test_inspector_handles_model_with_no_patches_attr(caplog):
    """Real ModelPatcher always has .patches, but FakeModel might not.
    Verifies inspector doesn't AttributeError on a minimal model."""
    model = FakeModel()
    # FakeModel has no `.patches` by default.
    with caplog.at_level(logging.INFO):
        (out,) = _inspect(model)
    assert out is model
    joined = "\n".join(r.getMessage() for r in caplog.records)
    # Missing patches => patches=0 (or "patches=?"), accept either.
    assert "patches=" in joined


def test_inspector_handles_model_with_no_object_patches_attr(caplog):
    model = FakeModel()
    with caplog.at_level(logging.INFO):
        _inspect(model)
    joined = "\n".join(r.getMessage() for r in caplog.records)
    # Either logged as 0 or omitted; either is fine — but must not crash.
    # (Smoke check: at least one log line emitted.)
    assert any("patch_inspect" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_iter_patch_inspector_registered():
    assert_node_registered("IterPatchInspector")
