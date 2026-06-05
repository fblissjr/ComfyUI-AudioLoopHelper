"""Tests for LTXIterKeyframeSchedule node (iter-gated LATENT selector).

Last updated: 2026-05-20

The node picks which pre-encoded keyframe latent anchors the current
iteration (by per-row target_iters lists), or passes a fallback latent
through when no row matches. Runs outside the loop body; output feeds
the loop's existing guide_latent input. No VAE / no tensor mutation —
pure selection, so tests pass plain latent dicts and assert identity.

Covers:
- Registration via comfy_entrypoint's node list (AST helper).
- AST default guard: target_iters_1 defaults to "" (never-match default
  → drop-in safe; behaves as the no-keyframe canonical).
- Passthrough to fallback when no row matches current_iteration.
- Selection of the matching row's keyframe latent.
- Lowest-index row wins when an iteration appears in multiple rows.
- Spatial-dims guard: a keyframe latent whose H/W don't match the
  fallback's fails FAST with the slot named (instead of core
  LTXVAddGuide asserting mid-render at the first iteration that selects
  it — the lost-resize-wire footgun on hand-copied keyframe branches).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest
import torch

from _node_registry import assert_node_registered


REPO_ROOT = Path(__file__).resolve().parent.parent


def _input_name_starts_with(arg: ast.expr, prefix: str) -> bool:
    """True if `arg` is a string literal or f-string whose first segment
    starts with `prefix`. Handles both `"target_iters_1"` and
    `f"target_iters_{i}"` source shapes."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value.startswith(prefix)
    if isinstance(arg, ast.JoinedStr) and arg.values:
        first = arg.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value.startswith(prefix)
    return False


def _scan_input_defaults_by_prefix(io_type: str, prefix: str) -> list:
    """Default values of every `io.<io_type>.Input(<name>, default=...)`
    whose name starts with `prefix`."""
    src = (REPO_ROOT / "nodes.py").read_text()
    tree = ast.parse(src, filename="nodes.py")
    defaults: list = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "Input"):
            continue
        if not (isinstance(f.value, ast.Attribute) and f.value.attr == io_type):
            continue
        if not node.args or not _input_name_starts_with(node.args[0], prefix):
            continue
        default_kw = next((kw for kw in node.keywords if kw.arg == "default"), None)
        if default_kw is not None and isinstance(default_kw.value, ast.Constant):
            defaults.append(default_kw.value.value)
    return defaults


def test_iter_keyframe_target_iters_default_is_empty():
    """`target_iters_N` default must be `""` (never matches).

    A drop-in of this node with no widget values must behave like the
    no-keyframe canonical: every iteration falls through to the fallback
    latent. A non-empty default would silently anchor on parsed iters.
    """
    defaults = _scan_input_defaults_by_prefix("String", "target_iters_")
    assert defaults, "Expected io.String.Input('target_iters_N', ...) calls; found none."
    bad = [d for d in defaults if d != ""]
    assert not bad, f"target_iters_N defaults must be '' (never-match); found: {bad}"


def test_iter_keyframe_schedule_registered():
    assert_node_registered("LTXIterKeyframeSchedule")


# --- Behavioral tests (pure selection; latents are opaque dicts) ---


def _kf(tag: str) -> dict:
    """A distinguishable stand-in latent dict (identity-checked in tests)."""
    return {"samples": tag}


def test_passthrough_to_fallback_when_target_iters_empty():
    from nodes import LTXIterKeyframeSchedule
    fallback = _kf("fallback")
    num_keyframes = {"keyframe_latent_1": _kf("kf1"), "target_iters_1": ""}
    out = LTXIterKeyframeSchedule.execute(fallback, 0, num_keyframes)
    assert out[0] is fallback


def test_passthrough_when_iter_not_targeted():
    from nodes import LTXIterKeyframeSchedule
    fallback = _kf("fallback")
    num_keyframes = {"keyframe_latent_1": _kf("kf1"), "target_iters_1": "10, 20"}
    out = LTXIterKeyframeSchedule.execute(fallback, 5, num_keyframes)
    assert out[0] is fallback


def test_selects_matching_keyframe():
    from nodes import LTXIterKeyframeSchedule
    fallback = _kf("fallback")
    kf1 = _kf("kf1")
    num_keyframes = {"keyframe_latent_1": kf1, "target_iters_1": "10, 25"}
    out = LTXIterKeyframeSchedule.execute(fallback, 25, num_keyframes)
    assert out[0] is kf1


def test_multi_row_selection():
    # 1-based iters: TensorLoopOpen.current_iteration emits 1,2,3,… — schedules
    # use 1-based indices (target_iters='0' would be dead, iter 0 = out-of-loop
    # init render). Use realistic 1-based values here so the test doubles as
    # documentation of the loop's actual iteration base.
    from nodes import LTXIterKeyframeSchedule
    fallback = _kf("fallback")
    kf1, kf2, kf3 = _kf("kf1"), _kf("kf2"), _kf("kf3")
    num_keyframes = {
        "keyframe_latent_1": kf1, "target_iters_1": "1",
        "keyframe_latent_2": kf2, "target_iters_2": "3",
        "keyframe_latent_3": kf3, "target_iters_3": "5",
    }
    assert LTXIterKeyframeSchedule.execute(fallback, 1, num_keyframes)[0] is kf1
    assert LTXIterKeyframeSchedule.execute(fallback, 3, num_keyframes)[0] is kf2
    assert LTXIterKeyframeSchedule.execute(fallback, 5, num_keyframes)[0] is kf3
    assert LTXIterKeyframeSchedule.execute(fallback, 2, num_keyframes)[0] is fallback


def test_lowest_index_row_wins_on_overlap():
    from nodes import LTXIterKeyframeSchedule
    fallback = _kf("fallback")
    kf1, kf2 = _kf("kf1"), _kf("kf2")
    # Both rows target iter 7; lowest-index row (1) wins.
    num_keyframes = {
        "keyframe_latent_1": kf1, "target_iters_1": "7",
        "keyframe_latent_2": kf2, "target_iters_2": "7",
    }
    out = LTXIterKeyframeSchedule.execute(fallback, 7, num_keyframes)
    assert out[0] is kf1


def test_key_ordering_is_numeric_not_lexical():
    """keyframe_latent_10 must sort after _2, not before (numeric order)."""
    from nodes import LTXIterKeyframeSchedule
    fallback = _kf("fallback")
    kf2, kf10 = _kf("kf2"), _kf("kf10")
    # iter 7 only in row 10; row 2 doesn't match. Must still find row 10.
    num_keyframes = {
        "keyframe_latent_2": kf2, "target_iters_2": "3",
        "keyframe_latent_10": kf10, "target_iters_10": "7",
    }
    out = LTXIterKeyframeSchedule.execute(fallback, 7, num_keyframes)
    assert out[0] is kf10


# --- Spatial-dims guard (fail fast on mis-sized keyframe latents) ---


def _lat(h: int, w: int) -> dict:
    """A real-tensor latent stand-in with LTX video latent rank [B,C,T,H,W]."""
    return {"samples": torch.zeros(1, 4, 3, h, w)}


def test_mismatched_keyframe_dims_raise_with_slot_named():
    """448x832 keyframe (14x26 latent) into a 544x960 render (17x30 latent):
    core would assert at iter 5; the selector must raise at iter 1, naming
    the slot and the likely cause (lost resize wires on a pasted copy)."""
    from nodes import LTXIterKeyframeSchedule
    fallback = _lat(17, 30)
    num_keyframes = {
        "keyframe_latent_1": _lat(17, 30), "target_iters_1": "1",
        "keyframe_latent_2": _lat(14, 26), "target_iters_2": "5",
    }
    with pytest.raises(ValueError, match="keyframe_latent_2"):
        LTXIterKeyframeSchedule.execute(fallback, 1, num_keyframes)


def test_matching_dims_do_not_raise():
    from nodes import LTXIterKeyframeSchedule
    fallback = _lat(17, 30)
    kf1 = _lat(17, 30)
    num_keyframes = {"keyframe_latent_1": kf1, "target_iters_1": "1"}
    out = LTXIterKeyframeSchedule.execute(fallback, 1, num_keyframes)
    assert out[0] is kf1


def test_mismatched_dims_with_empty_targets_warns_not_raises(caplog):
    """A mis-sized row that can never fire (empty target_iters) must not
    block the render — but it IS a landmine for the next target re-spread,
    so it warns."""
    from nodes import LTXIterKeyframeSchedule
    fallback = _lat(17, 30)
    num_keyframes = {"keyframe_latent_1": _lat(14, 26), "target_iters_1": ""}
    with caplog.at_level(logging.WARNING):
        out = LTXIterKeyframeSchedule.execute(fallback, 1, num_keyframes)
    assert out[0] is fallback
    assert any("keyframe_latent_1" in r.message for r in caplog.records)


def test_integer_ratio_mismatch_warns_not_raises(caplog):
    """Half-res guide (core accepts integer ratios) — legal, so warn only."""
    from nodes import LTXIterKeyframeSchedule
    fallback = _lat(16, 32)
    kf1 = _lat(8, 16)
    num_keyframes = {"keyframe_latent_1": kf1, "target_iters_1": "1"}
    with caplog.at_level(logging.WARNING):
        out = LTXIterKeyframeSchedule.execute(fallback, 1, num_keyframes)
    assert out[0] is kf1
    assert any("keyframe_latent_1" in r.message for r in caplog.records)


def test_non_tensor_latents_skip_the_guard():
    """Opaque stand-ins (and exotic latent types) have no usable shape —
    the guard must skip them, not crash."""
    from nodes import LTXIterKeyframeSchedule
    fallback = _kf("fallback")
    num_keyframes = {"keyframe_latent_1": _kf("kf1"), "target_iters_1": "5"}
    out = LTXIterKeyframeSchedule.execute(fallback, 1, num_keyframes)
    assert out[0] is fallback


# --- Decision message (what the node reports it actually used) ---


def test_select_message_on_match_names_the_keyframe():
    from nodes import _kf_select
    kf2 = _kf("kf2")
    rows = [("1", {10}, _kf("kf1")), ("2", {25}, kf2)]
    chosen, msg, matched = _kf_select(rows, _kf("fallback"), 25)
    assert chosen is kf2
    assert matched == "2"
    assert "keyframe" in msg.lower() and "2" in msg and "matched" in msg.lower()


def test_select_message_on_no_match_says_fallback():
    from nodes import _kf_select
    fallback = _kf("fallback")
    rows = [("1", {10}, _kf("kf1")), ("2", {25}, _kf("kf2"))]
    chosen, msg, matched = _kf_select(rows, fallback, 5)
    assert chosen is fallback and matched is None
    assert "fallback" in msg.lower()
    assert "disabled" not in msg.lower()  # rows DO have targets; just none this iter


def test_select_message_on_all_empty_flags_disabled():
    """The footgun: every row empty → fallback every iter → say keyframes disabled."""
    from nodes import _kf_select
    fallback = _kf("fallback")
    rows = [("1", set(), _kf("kf1")), ("2", set(), _kf("kf2"))]
    chosen, msg, matched = _kf_select(rows, fallback, 3)
    assert chosen is fallback and matched is None
    assert "disabled" in msg.lower() or "empty" in msg.lower()


def test_execute_prints_decision_by_default(capsys):
    """execute() reports what it used to the console with no env flag set."""
    from nodes import LTXIterKeyframeSchedule
    num_keyframes = {"keyframe_latent_1": _kf("kf1"), "target_iters_1": "3"}
    LTXIterKeyframeSchedule.execute(_kf("fallback"), 3, num_keyframes)
    out = capsys.readouterr()
    combined = out.out + out.err
    assert "keyframe selector" in combined.lower()
