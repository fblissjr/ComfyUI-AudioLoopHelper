"""Tests for nodes_regional_compile.LTXVideoRegionalCompile.

Spike-shape unit tests covering:
- _patch_blocks replaces block.ff with a torch.compile wrapper
- _restore_blocks puts originals back, clears state
- Re-applying detects existing OptimizedModule and unwraps cleanly (no double-wrap)
- _patch_impl wires cleanup callback that restores on invocation

These tests do NOT touch real torch.compile (it would invoke Inductor,
needing a full torch+CUDA setup beyond the unit-test ceiling). Instead
we monkeypatch torch.compile to a no-op identity-marker AND the
OptimizedModule import to recognize that marker as the wrapper class.
"""

from __future__ import annotations

import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest  # noqa: F401  (kept for future parametrize)
import torch  # noqa: F401  (used by helper modules)
import torch.nn as nn

# Add scripts/ + tests/ to sys.path (matches conftest.py)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from _fakes import FakeModelWithCallbacks  # noqa: E402

from nodes_regional_compile import (  # noqa: E402
    PATCH_KEY,
    LTXVideoRegionalCompile,
    _patch_blocks,
    _restore_blocks,
)


# -----------------------------------------------------------------------------
# Fakes specific to this node — mirror the LTX-2.3 transformer-block shape.
# -----------------------------------------------------------------------------

class _FakeFFN(nn.Module):
    """Stands in for `BasicTransformerBlock.ff` (FeedForward)."""
    def __init__(self, dim: int = 32, name: str = "ffn"):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.name = name  # for identity asserts

    def forward(self, x):
        return self.linear(x)


class _FakeTransformerBlock(nn.Module):
    def __init__(self, dim: int = 32, idx: int = 0):
        super().__init__()
        self.attn1 = nn.Linear(dim, dim)  # untouched by regional compile
        self.attn2 = nn.Linear(dim, dim)  # untouched
        self.ff = _FakeFFN(dim=dim, name=f"block{idx}_ffn")


class _FakeDiffusionModel(nn.Module):
    def __init__(self, n_blocks: int = 4, dim: int = 32):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_FakeTransformerBlock(dim=dim, idx=i) for i in range(n_blocks)]
        )


class _FakeModelInner:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


class FakeLTXModel(FakeModelWithCallbacks):
    """FakeModelWithCallbacks + .model.diffusion_model.transformer_blocks shape.

    Mirrors the production access pattern: clone.model.diffusion_model.transformer_blocks[i].ff.
    """
    def __init__(self, n_blocks: int = 4):
        super().__init__()
        self._diffusion = _FakeDiffusionModel(n_blocks=n_blocks)
        self.model = _FakeModelInner(self._diffusion)

    def add_callback_with_key(self, call_type: str, key: str, fn):
        # Production ModelPatcher uses keyed callbacks; mirror the signature.
        self.callbacks[(call_type, key)] = fn

    def clone(self):
        # Shallow-clone the patcher state but SHARE the underlying model
        # (mirrors production ModelPatcher.clone() behavior — the model is
        # shared, only model_options + callbacks are per-clone).
        c = FakeLTXModel.__new__(FakeLTXModel)
        c.model_options = {"transformer_options": dict(self.model_options.get("transformer_options", {}))}
        c.callbacks = {}
        c._diffusion = self._diffusion  # SHARED
        c.model = self.model            # SHARED
        return c


# -----------------------------------------------------------------------------
# Helpers — fake OptimizedModule + identity-compile that respects the
# production unwrap path (`isinstance(ff, OptimizedModule)` + `_orig_mod`).
# -----------------------------------------------------------------------------

class _FakeOptimizedModule(nn.Module):
    """Stands in for `torch._dynamo.eval_frame.OptimizedModule`. Production
    code unwraps via `isinstance(ff, OptimizedModule)` + `ff._orig_mod`;
    these tests patch the import so this class IS OptimizedModule."""
    def __init__(self, mod, mode):
        super().__init__()
        self._orig_mod = mod
        self._compile_mode = mode  # test-only assertion handle

    def forward(self, x):
        return self._orig_mod(x)


def _identity_compile(module, mode="default", **_kwargs):
    """torch.compile stand-in that returns a fake OptimizedModule wrapping
    the input. Pairs with the patched _unwrap_compiled below so the
    unwrap path is exercised in tests."""
    return _FakeOptimizedModule(module, mode)


def _fake_unwrap(ff):
    """Recognize our `_FakeOptimizedModule` as the OptimizedModule the
    production `_unwrap_compiled` handles. Used by the patch context."""
    if isinstance(ff, _FakeOptimizedModule):
        return ff._orig_mod
    return ff


def _patch_compile_and_unwrap(_target_path=None):
    """Context manager that patches both `torch.compile` in
    `nodes_regional_compile` AND `_unwrap_compiled` so the fake-
    OptimizedModule unwrap path is exercised. Uses `ExitStack` so
    both patches lift cleanly even on exception."""
    stack = ExitStack()
    stack.enter_context(patch("nodes_regional_compile.torch.compile", _identity_compile))
    stack.enter_context(patch("nodes_regional_compile._unwrap_compiled", _fake_unwrap))
    return stack


# -----------------------------------------------------------------------------
# _patch_blocks / _restore_blocks
# -----------------------------------------------------------------------------

class TestPatchBlocks:

    def test_patches_every_block(self):
        model = FakeLTXModel(n_blocks=4)
        diffusion = model.model.diffusion_model
        original_ffs = [b.ff for b in diffusion.transformer_blocks]

        with _patch_compile_and_unwrap("nodes_regional_compile"):
            originals = _patch_blocks(diffusion, mode="default")

        assert len(originals) == 4
        for i, block in enumerate(diffusion.transformer_blocks):
            assert block.ff is not original_ffs[i], (
                f"block {i}.ff was not replaced by the compile wrapper"
            )
            # OptimizedModule API: original is at _orig_mod
            assert block.ff._orig_mod is original_ffs[i]
            assert originals[i] is original_ffs[i]

    def test_no_setattr_cycle_in_compiled_wrapper(self):
        """Regression for the 2026-05-01 RecursionError: the prior version
        did setattr(compiled, '_compile_orig', ff) which auto-registered
        ff as a submodule of compiled (because nn.Module.__setattr__
        registers Module values), creating a state_dict cycle that broke
        downstream IC-LoRA loader. This test confirms compiled wrappers
        only have the canonical _orig_mod child, no extra Module-typed
        attrs."""
        model = FakeLTXModel(n_blocks=2)
        diffusion = model.model.diffusion_model

        with _patch_compile_and_unwrap("nodes_regional_compile"):
            _patch_blocks(diffusion, mode="default")

        for block in diffusion.transformer_blocks:
            # The fake OptimizedModule should only have _orig_mod as its
            # registered submodule. No `_compile_orig` or other Module-
            # typed sentinel attribute that would duplicate registration.
            module_children = dict(block.ff.named_children())
            assert set(module_children.keys()) == {"_orig_mod"}, (
                f"compiled wrapper has unexpected child modules: "
                f"{list(module_children.keys())} — would create state_dict cycle"
            )

    def test_respects_compile_mode(self):
        model = FakeLTXModel(n_blocks=2)
        diffusion = model.model.diffusion_model

        with _patch_compile_and_unwrap("nodes_regional_compile"):
            _patch_blocks(diffusion, mode="reduce-overhead")

        assert diffusion.transformer_blocks[0].ff._compile_mode == "reduce-overhead"

    def test_restore_puts_originals_back(self):
        model = FakeLTXModel(n_blocks=3)
        diffusion = model.model.diffusion_model
        original_ffs = [b.ff for b in diffusion.transformer_blocks]

        with _patch_compile_and_unwrap("nodes_regional_compile"):
            originals = _patch_blocks(diffusion, mode="default")
            _restore_blocks(diffusion, originals)

        for i, block in enumerate(diffusion.transformer_blocks):
            assert block.ff is original_ffs[i], f"block {i}.ff was not restored"
        assert originals == {}, "originals dict should be cleared after restore"

    def test_reapply_unwraps_optimized_module_and_refreshes(self):
        """Re-running _patch_blocks on already-compiled blocks must unwrap
        via OptimizedModule._orig_mod and recompile cleanly, not double-
        wrap the prior compiled module."""
        model = FakeLTXModel(n_blocks=2)
        diffusion = model.model.diffusion_model
        original_ffs = [b.ff for b in diffusion.transformer_blocks]

        with _patch_compile_and_unwrap("nodes_regional_compile"):
            _patch_blocks(diffusion, mode="default")
            first_compiled = [b.ff for b in diffusion.transformer_blocks]
            originals2 = _patch_blocks(diffusion, mode="reduce-overhead")

        for i in range(2):
            # Originals tracked by reapply must be the TRUE original FF,
            # not the first-pass wrapper.
            assert originals2[i] is original_ffs[i], (
                f"reapply must track ORIGINAL ff (idx {i}), not the prior wrapper"
            )
            # New compiled wrapper is a fresh object
            assert diffusion.transformer_blocks[i].ff is not first_compiled[i]
            # New wrapper unwraps to the true original
            assert diffusion.transformer_blocks[i].ff._orig_mod is original_ffs[i]

    def test_missing_transformer_blocks_raises(self):
        """Hard fail if diffusion_model lacks transformer_blocks — better
        than silent no-op."""
        broken = nn.Module()
        with pytest.raises(RuntimeError, match="transformer_blocks"):
            _patch_blocks(broken, mode="default")


# -----------------------------------------------------------------------------
# _patch_impl: full node-execution surface
# -----------------------------------------------------------------------------

class TestPatchImpl:

    def test_returns_clone_with_compiled_blocks(self):
        model = FakeLTXModel(n_blocks=2)
        original_ffs = [b.ff for b in model.model.diffusion_model.transformer_blocks]

        with _patch_compile_and_unwrap("nodes_regional_compile"):
            patched = LTXVideoRegionalCompile._patch_impl(model, mode="default")

        # Returned clone shares the model, so its blocks are the patched ones
        for i, block in enumerate(patched.model.diffusion_model.transformer_blocks):
            assert block.ff is not original_ffs[i]

    def test_no_cleanup_callback_registered(self):
        """Regression guard: ON_CLEANUP fires after EVERY model invocation
        (= every sampler), not at model-unload. Registering cleanup to
        restore-and-recompile would trash the compile cache between
        samplers, paying cold-compile per sampler. Bench 2026-05-01:
        with cleanup registered, regional_compile was +6% slower than
        baseline. The fix is to skip cleanup entirely; let compile state
        persist for the lifetime of the loaded diffusion model.
        """
        model = FakeLTXModel(n_blocks=2)
        with _patch_compile_and_unwrap("nodes_regional_compile"):
            patched = LTXVideoRegionalCompile._patch_impl(model, mode="default")

        assert patched.callbacks == {}, (
            f"no callbacks should be registered (cleanup would trash compile "
            f"cache between samplers); have {list(patched.callbacks.keys())}"
        )
