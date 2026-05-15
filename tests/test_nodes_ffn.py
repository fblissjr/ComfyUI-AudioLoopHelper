"""Tests for `AudioLoopHelperSageFFN` (sage v0.6 consumer-side patch).

Mock phase: sage v0.6 hasn't shipped yet, so the patched forward
falls through to the stock `self_module.net(x)` path. Tests verify:

1. Dispatch: 44 blocks patched, 4 bookend blocks (`{0, 1, 46, 47}`)
   skipped.
2. Patched forward invokes `self_module.net(x)` when `sage_ffn_fn`
   is None (mock-phase fallback).
3. Patched forward invokes `sage_ffn_fn(...)` when provided.
4. Defensive fallback: sage_ffn raises -> stock path.
5. Compatibility guard: model without `transformer_blocks` -> unpatched
   return (no crash).
6. `enabled=False` -> pass-through.
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest

from _fakes import FakeModelPatcher, _walk_callables

import nodes_ffn
from nodes_ffn import BF16_FFN_BLOCKS, AudioLoopHelperSageFFN, _FFNPatch


# -- Test fakes ---------------------------------------------------------------


class FakeFF:
    """Stand-in for `BasicAVTransformerBlock.ff` (a `FeedForward` module)."""

    def __init__(self, has_weight_scale: bool = True):
        self.net = MagicMock(name="ff.net")
        # Wire .net[0].proj and .net[2] to look like the real fp8-quantized
        # FeedForward layout: net is nn.Sequential of (GELU_approx, Dropout,
        # Linear). GELU_approx wraps a `proj` Linear at .proj.
        proj_in = MagicMock(name="net[0].proj")
        proj_in.weight = MagicMock(name="W1_fp8")
        proj_in.weight_scale = MagicMock(name="s1") if has_weight_scale else None
        proj_in.bias = MagicMock(name="b1")
        proj_out = MagicMock(name="net[2]")
        proj_out.weight = MagicMock(name="W2_fp8")
        proj_out.weight_scale = MagicMock(name="s2") if has_weight_scale else None
        proj_out.bias = MagicMock(name="b2")
        # Wire net[0].proj and net[2] via __getitem__.
        ff_net_0 = MagicMock(name="net[0]")
        ff_net_0.proj = proj_in
        self.net.__getitem__ = lambda _self, idx: {0: ff_net_0, 2: proj_out}[idx]


class FakeAVBlock:
    def __init__(self, idx: int):
        self.idx = idx
        self.ff = FakeFF()


class FakeDiffusionModel:
    def __init__(self, num_blocks: int = 48):
        self.transformer_blocks = [FakeAVBlock(i) for i in range(num_blocks)]


class FakeModelForFFN(FakeModelPatcher):
    """FakeModelPatcher + the `get_model_object` and `add_object_patch`
    surface that `AudioLoopHelperSageFFN._patch_impl` touches."""

    def __init__(self, transformer_options: dict | None = None, num_blocks: int = 48):
        super().__init__(transformer_options)
        self._diffusion_model = FakeDiffusionModel(num_blocks)
        self.object_patches: dict[str, object] = {}

    def get_model_object(self, name: str):
        if name == "diffusion_model":
            return self._diffusion_model
        raise KeyError(name)

    def add_object_patch(self, path: str, fn) -> None:
        self.object_patches[path] = fn

    def clone(self):
        c = type(self)(num_blocks=len(self._diffusion_model.transformer_blocks))
        c.model_options = copy.deepcopy(
            self.model_options,
            memo={id(v): v for v in _walk_callables(self.model_options)},
        )
        # Share diffusion_model + start fresh patches dict per clone (matches
        # ComfyUI ModelPatcher semantics).
        c._diffusion_model = self._diffusion_model
        c.object_patches = {}
        return c


# -- Dispatch correctness -----------------------------------------------------


def test_dispatch_skips_bookend_blocks_and_patches_the_rest():
    model = FakeModelForFFN()
    (patched_model,) = AudioLoopHelperSageFFN._patch_impl(model, sage_ffn_fn=None)

    patched_paths = set(patched_model.object_patches.keys())
    assert len(patched_paths) == 48 - len(BF16_FFN_BLOCKS) == 44

    for idx in BF16_FFN_BLOCKS:
        assert f"diffusion_model.transformer_blocks.{idx}.ff.forward" not in patched_paths

    for idx in range(48):
        if idx in BF16_FFN_BLOCKS:
            continue
        assert f"diffusion_model.transformer_blocks.{idx}.ff.forward" in patched_paths


def test_bookend_set_is_first_two_and_last_two():
    # Canary against accidental change of the bookend pattern. If the audit
    # ever revises which blocks are bf16, this test must be updated AND the
    # node's compatibility guard re-checked against the new audit.
    assert BF16_FFN_BLOCKS == frozenset({0, 1, 46, 47})


# -- Patched-forward behavior -------------------------------------------------


def test_patched_forward_mock_mode_calls_self_net():
    """When sage_ffn_fn is None (mock phase), patched forward must call
    `self_module.net(x)` -- numerically identical to no-patch."""
    ff = FakeFF()
    ff.net.return_value = "stock_path_result"

    patch = _FFNPatch(sage_ffn_fn=None)
    bound = patch.__get__(ff, type(ff))

    result = bound("input_tensor")

    assert result == "stock_path_result"
    ff.net.assert_called_once_with("input_tensor")


def test_patched_forward_with_sage_ffn_routes_through_kernel():
    ff = FakeFF()
    fake_sage_ffn = MagicMock(name="sage_ffn", return_value="sage_path_result")

    patch = _FFNPatch(sage_ffn_fn=fake_sage_ffn)
    bound = patch.__get__(ff, type(ff))

    result = bound("input_tensor")

    assert result == "sage_path_result"
    fake_sage_ffn.assert_called_once()
    call_args = fake_sage_ffn.call_args[0]
    assert call_args[0] == "input_tensor"
    assert call_args[1] is ff.net[0].proj.weight   # W1
    assert call_args[2] is ff.net[0].proj.weight_scale  # s1
    assert call_args[3] is ff.net[2].weight        # W2
    assert call_args[4] is ff.net[2].weight_scale  # s2


def test_patched_forward_falls_through_when_weight_scale_missing():
    """If a block doesn't have weight_scale attrs (e.g. user accidentally
    enabled the node on a bf16 checkpoint), patched forward must fall
    through to the stock path rather than crash."""
    ff = FakeFF(has_weight_scale=False)
    ff.net.return_value = "stock_fallback"
    fake_sage_ffn = MagicMock(name="sage_ffn", return_value="should_not_run")

    patch = _FFNPatch(sage_ffn_fn=fake_sage_ffn)
    bound = patch.__get__(ff, type(ff))

    result = bound("input_tensor")

    assert result == "stock_fallback"
    fake_sage_ffn.assert_not_called()
    ff.net.assert_called_once_with("input_tensor")


def test_patched_forward_falls_through_when_sage_ffn_raises():
    """Defensive: sage_ffn raising must not crash the render."""
    ff = FakeFF()
    ff.net.return_value = "stock_fallback"
    fake_sage_ffn = MagicMock(name="sage_ffn", side_effect=RuntimeError("kernel boom"))

    patch = _FFNPatch(sage_ffn_fn=fake_sage_ffn)
    bound = patch.__get__(ff, type(ff))

    result = bound("input_tensor")

    assert result == "stock_fallback"
    ff.net.assert_called_once_with("input_tensor")


# -- Compatibility guards -----------------------------------------------------


def test_model_without_diffusion_model_returns_unpatched():
    class FakeModelNoDM(FakeModelPatcher):
        def get_model_object(self, _name):
            raise KeyError("no diffusion_model")

        def clone(self):
            return FakeModelNoDM()

    model = FakeModelNoDM()
    (out,) = AudioLoopHelperSageFFN._patch_impl(model, sage_ffn_fn=None)
    # No crash, returned an unpatched clone.
    assert out is not model
    assert not hasattr(out, "object_patches") or not getattr(out, "object_patches", {})


def test_model_without_transformer_blocks_returns_unpatched():
    class EmptyDM:
        pass

    class FakeModelEmptyDM(FakeModelPatcher):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self._diffusion_model = EmptyDM()
            self.object_patches: dict = {}

        def get_model_object(self, name):
            if name == "diffusion_model":
                return self._diffusion_model
            raise KeyError(name)

        def add_object_patch(self, path, fn):
            self.object_patches[path] = fn

        def clone(self):
            return type(self)()

    model = FakeModelEmptyDM()
    (out,) = AudioLoopHelperSageFFN._patch_impl(model, sage_ffn_fn=None)
    assert out.object_patches == {}


def test_enabled_false_returns_model_unchanged():
    """The boolean toggle gives users an A/B switch without rewiring.
    `execute(model, enabled=False)` must return the model verbatim
    (no clone, no patches)."""
    model = FakeModelForFFN()
    out = AudioLoopHelperSageFFN.execute(model, enabled=False)
    # Under the io stub, NodeOutput(*args) -> args. Real ComfyUI wraps;
    # both paths yield the same indexed access.
    returned = out[0] if isinstance(out, tuple) else getattr(out, "result", out)[0]
    assert returned is model


def test_enabled_true_patches_44_blocks():
    """Sanity: the dispatch loop runs when enabled=True. Real execute()
    path resolves sage_ffn fresh (likely None in CI), so patches install
    in mock-phase mode."""
    model = FakeModelForFFN()
    out = AudioLoopHelperSageFFN.execute(model, enabled=True)
    returned = out[0] if isinstance(out, tuple) else getattr(out, "result", out)[0]
    assert returned is not model
    assert len(returned.object_patches) == 44


# -- Schema invariants (AST) --------------------------------------------------


def test_audioloophelpersageffn_registered_in_node_list():
    from _node_registry import assert_node_registered
    assert_node_registered("AudioLoopHelperSageFFN")


def test_bf16_ffn_blocks_is_frozenset_not_mutable():
    """Defensive: BF16_FFN_BLOCKS must be immutable to prevent accidental
    mutation that would change dispatch behavior at runtime."""
    assert isinstance(BF16_FFN_BLOCKS, frozenset)
    with pytest.raises(AttributeError):
        BF16_FFN_BLOCKS.add(2)  # type: ignore[attr-defined]


# -- Module-import-time guard -------------------------------------------------


def test_resolve_sage_ffn_handles_both_present_and_absent():
    """Resolver returns either a callable (sage v0.6+ installed) or None
    (sage absent / older version). Both branches are valid; node behavior
    switches on it correctly per other tests."""
    fn = nodes_ffn._resolve_sage_ffn()
    assert fn is None or callable(fn)
