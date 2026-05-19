"""Tests for LTXIterKeyframeSchedule node.

Last updated: 2026-05-19

Covers:
- Registration via `comfy_entrypoint`'s node list (AST helper).
- AST default guard: `target_iters_1` defaults to "" (no-op when dropped
  into existing workflow — required for safe iteration on canonical).
- Passthrough behavior when no row's `target_iters` matches the
  current iteration.
- Mutation behavior when a row matches: latent samples written at
  `target_idx`, noise_mask set to 0 at that frame.
- Encode cache hit on repeated calls with the same image tensor.

A minimal `FakeVae` is used (no comfy import). The `_IOStub` fallback in
nodes.py makes `io.NodeOutput` a plain tuple in this pytest context.
"""

from __future__ import annotations

import ast
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
    """Find every `io.<io_type>.Input(<name>, default=...)` whose name
    starts with `prefix`. Returns the list of default values."""
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
    """`LTXIterKeyframeSchedule.target_iters_N` default must be `""`.

    Empty = no-op for that row. Non-empty default would silently anchor on
    every iter when dropped into an existing workflow with no widget
    values — invisible behavior change. Same no-op-default discipline as
    LatentTemporalMask.edge_taper_seconds.
    """
    defaults = _scan_input_defaults_by_prefix("String", "target_iters_")
    assert defaults, "Expected io.String.Input('target_iters_N', ...) calls; found none."
    bad = [d for d in defaults if d != ""]
    assert not bad, f"target_iters_N defaults must be '' (no-op); found: {bad}"


def test_iter_keyframe_target_idx_default_is_zero():
    """`target_idx_N` defaults to 0 — first latent frame."""
    defaults = _scan_input_defaults_by_prefix("Int", "target_idx_")
    assert defaults, "Expected io.Int.Input('target_idx_N', ...) calls; found none."
    bad = [d for d in defaults if d != 0]
    assert not bad, f"target_idx_N defaults must be 0; found: {bad}"


# --- Registration ---


def test_iter_keyframe_schedule_registered():
    assert_node_registered("LTXIterKeyframeSchedule")


# --- Behavioral tests ---


class _FakeVae:
    """Minimal VAE for tests. encode() returns deterministic tensor.

    Real LTX video VAE: 32:1 spatial, 8:1 temporal. encode() input shape
    (B, H, W, C); output shape (B, 128, T_lat, H/32, W/32) where
    T_lat = (T_pixel - 1) // 8 + 1.
    """
    def __init__(self, channels: int = 128):
        self.channels = channels
        self.encode_call_count = 0

    def encode(self, image):
        self.encode_call_count += 1
        B, H, W, _ = image.shape
        H_lat = H // 32
        W_lat = W // 32
        return torch.full(
            (B, self.channels, 1, H_lat, W_lat),
            fill_value=float(self.encode_call_count),  # distinguishable per encode
        )


@pytest.fixture(autouse=True)
def _clear_encode_cache():
    """Prevent id()-recycling ghost hits between tests."""
    from nodes import _ITER_KEYFRAME_ENCODE_CACHE
    _ITER_KEYFRAME_ENCODE_CACHE.clear()
    yield
    _ITER_KEYFRAME_ENCODE_CACHE.clear()


def _make_latent(B=1, C=128, T=8, H_lat=14, W_lat=20):
    """Latent shape (B, C, T, H_lat, W_lat). Pixel dims = (H_lat*32, W_lat*32)."""
    return {
        "samples": torch.zeros((B, C, T, H_lat, W_lat)),
        "noise_mask": torch.ones((B, 1, T, 1, 1)),
    }


def _make_image(B=1, H=448, W=640, C=3):
    return torch.zeros((B, H, W, C))


def test_passthrough_when_target_iters_empty():
    """Empty target_iters → no match → latent returned unchanged (same dict ref)."""
    from nodes import LTXIterKeyframeSchedule
    latent = _make_latent()
    vae = _FakeVae()
    num_images = {
        "image_1": _make_image(),
        "target_iters_1": "",
        "target_idx_1": 0,
    }
    out = LTXIterKeyframeSchedule.execute(latent, 0, vae, num_images)
    # _IOStub.NodeOutput returns tuple; result at index 0.
    assert out[0] is latent, "Expected identical dict passthrough when no rows match"
    assert vae.encode_call_count == 0, "VAE should not be called on passthrough"


def test_passthrough_when_iter_not_in_targets():
    """target_iters='10, 20' + current_iteration=5 → no match → passthrough."""
    from nodes import LTXIterKeyframeSchedule
    latent = _make_latent()
    vae = _FakeVae()
    num_images = {
        "image_1": _make_image(),
        "target_iters_1": "10, 20",
        "target_idx_1": 0,
    }
    out = LTXIterKeyframeSchedule.execute(latent, 5, vae, num_images)
    assert out[0] is latent
    assert vae.encode_call_count == 0


def test_mutation_on_match_writes_samples_and_zeros_mask():
    """current_iteration=10 matches '10, 25' → write to target_idx, mask[idx]=0."""
    from nodes import LTXIterKeyframeSchedule
    latent = _make_latent(T=8)
    vae = _FakeVae()
    num_images = {
        "image_1": _make_image(),
        "target_iters_1": "10, 25",
        "target_idx_1": 3,
    }
    out = LTXIterKeyframeSchedule.execute(latent, 10, vae, num_images)
    new_latent = out[0]
    assert new_latent is not latent, "Should return new dict (shallow copy)"
    # samples[3] should be encoded (=1.0 from fake), others should be 0.
    assert (new_latent["samples"][:, :, 3] == 1.0).all()
    assert (new_latent["samples"][:, :, 0] == 0.0).all()
    assert (new_latent["samples"][:, :, 4] == 0.0).all()
    # noise_mask[3] should be 0; others should be 1.
    assert (new_latent["noise_mask"][:, :, 3] == 0).all()
    assert (new_latent["noise_mask"][:, :, 0] == 1).all()
    # Upstream latent must NOT be mutated.
    assert (latent["samples"] == 0.0).all()
    assert (latent["noise_mask"] == 1).all()


def test_negative_target_idx_counts_from_end():
    """target_idx=-1 should write to last latent frame."""
    from nodes import LTXIterKeyframeSchedule
    latent = _make_latent(T=8)
    vae = _FakeVae()
    num_images = {
        "image_1": _make_image(),
        "target_iters_1": "0",
        "target_idx_1": -1,
    }
    out = LTXIterKeyframeSchedule.execute(latent, 0, vae, num_images)
    new_latent = out[0]
    assert (new_latent["samples"][:, :, 7] == 1.0).all()  # last frame
    assert (new_latent["noise_mask"][:, :, 7] == 0).all()
    assert (new_latent["samples"][:, :, 0] == 0.0).all()  # untouched


def test_multirow_stacks_writes_at_different_indices():
    """Two rows both targeting iter 5, different target_idx → both write."""
    from nodes import LTXIterKeyframeSchedule
    latent = _make_latent(T=8)
    vae = _FakeVae()
    img1 = _make_image()
    img2 = _make_image()
    num_images = {
        "image_1": img1, "target_iters_1": "5", "target_idx_1": 0,
        "image_2": img2, "target_iters_2": "5", "target_idx_2": 4,
    }
    out = LTXIterKeyframeSchedule.execute(latent, 5, vae, num_images)
    new_latent = out[0]
    # Both frames written; encode count = 2 (one per image).
    assert (new_latent["noise_mask"][:, :, 0] == 0).all()
    assert (new_latent["noise_mask"][:, :, 4] == 0).all()
    assert vae.encode_call_count == 2


def test_encode_cache_hit_across_repeated_calls():
    """Same image, same vae, repeated execute calls → VAE encode runs once."""
    from nodes import LTXIterKeyframeSchedule
    latent = _make_latent()
    vae = _FakeVae()
    img = _make_image()
    num_images = {
        "image_1": img,
        "target_iters_1": "0, 1, 2",
        "target_idx_1": 0,
    }
    for it in (0, 1, 2):
        LTXIterKeyframeSchedule.execute(_make_latent(), it, vae, num_images)
    assert vae.encode_call_count == 1, (
        f"Expected single encode across 3 iters via cache; got {vae.encode_call_count}"
    )
