"""Tests for `AudioLoopHelperSageFFN` (sage v0.6 consumer-side patch).

Tests verify:

1. Dispatch: 44 blocks patched, 4 bookend blocks (`{0, 1, 46, 47}`)
   skipped when sage_ffn is available.
2. No-op behavior: when sage_ffn is None (v0.6 not installed), node
   applies NO patches so prior FFN patches in the chain (KJNodes
   LTXVChunkFeedForward) survive.
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
import torch

from _fakes import FakeModelPatcher, _walk_callables

import nodes_ffn
from nodes_ffn import (
    BF16_FFN_BLOCKS,
    SAGE_FFN_CHUNK_SEQ,
    AudioLoopHelperSageFFN,
    _FFNFallbackLogger,
    _FFNPatch,
)


@pytest.fixture
def fake_sage_ffn():
    """Stand-in for the real `sageattention.sage_ffn` callable. Tests that
    assert dispatch behavior wire `return_value` / `side_effect` per-test."""
    return MagicMock(name="fake_sage_ffn")


def _make_input(seq_len: int, dim: int = 8) -> torch.Tensor:
    """LTX FFN input shape `[batch=1, seq, dim]`. Tests only read
    `.shape[1]` to drive the chunked-vs-single-call branch; `sage_ffn` is
    mocked so the channel dim is never touched. Defaults to dim=8 to keep
    the boundary test's 4096-seq allocation cheap (~128 KB instead of
    ~67 MB at production dim=4096)."""
    return torch.empty(1, seq_len, dim)


# -- Test fakes ---------------------------------------------------------------


class FakeFF:
    """Stand-in for `BasicAVTransformerBlock.ff` (a `FeedForward` module).

    `has_weight_scale=True` mirrors the legacy fp8_ops convention where the
    scale lives as `Linear.scale_weight`. Tests covering the modern
    `QuantizedTensor`-wrapped path set `scale_path="weight._params"` to
    place the scale under `weight._params.scale` instead."""

    def __init__(self, has_weight_scale: bool = True, scale_path: str = "scale_weight"):
        self.net = MagicMock(name="ff.net")
        proj_in = MagicMock(name="net[0].proj")
        proj_out = MagicMock(name="net[2]")
        # net[0].proj is up-projection (GELU_approx.proj); net[2] is the
        # down-projection Linear. Production weights are
        # torch.float8_e4m3fn; for these tests any Tensor works since
        # sage_ffn_fn is mocked.
        proj_in.bias = MagicMock(name="b1")
        proj_out.bias = MagicMock(name="b2")
        if has_weight_scale:
            s1 = torch.tensor(1.0, dtype=torch.float32)
            s2 = torch.tensor(1.0, dtype=torch.float32)
        else:
            s1 = s2 = None
        if scale_path == "scale_weight":
            # Legacy fp8_ops convention: weight is the raw fp8 tensor,
            # scale lives directly on Linear. Stamp `.device` to a real
            # torch.device so the wrapper's device-mismatch guard sees a
            # real comparand (MagicMock auto-attrs would break equality).
            proj_in.weight = MagicMock(name="W1_fp8")
            proj_out.weight = MagicMock(name="W2_fp8")
            proj_in.weight.device = torch.device("cpu")
            proj_out.weight.device = torch.device("cpu")
            proj_in.scale_weight = s1
            proj_out.scale_weight = s2
            proj_in.weight_scale = None
            proj_out.weight_scale = None
        elif scale_path == "weight._params":
            # Modern QuantizedTensor wrapping: ._qdata holds raw fp8 tensor,
            # ._params.scale holds the per-tensor scale. `_extract_fp8_weight_and_scale`
            # uses `isinstance(qdata, torch.Tensor)` to guard — so _qdata
            # must be a real Tensor, not a MagicMock.
            w1 = MagicMock(name="W1_QuantizedTensor")
            w1._qdata = torch.empty(1, dtype=torch.uint8)  # fp8 surrogate
            w2 = MagicMock(name="W2_QuantizedTensor")
            w2._qdata = torch.empty(1, dtype=torch.uint8)
            w1._params = MagicMock(scale=s1) if s1 is not None else None
            w2._params = MagicMock(scale=s2) if s2 is not None else None
            proj_in.weight = w1
            proj_out.weight = w2
            proj_in.scale_weight = None
            proj_out.scale_weight = None
            proj_in.weight_scale = None
            proj_out.weight_scale = None
        else:
            raise ValueError(f"unknown scale_path: {scale_path}")
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


def test_dispatch_skips_bookend_blocks_and_patches_the_rest(fake_sage_ffn):
    """When sage_ffn is available, _patch_impl patches 44 of 48 blocks
    (skipping the 4 bookend bf16 blocks)."""
    model = FakeModelForFFN()
    (patched_model,) = AudioLoopHelperSageFFN._patch_impl(model, sage_ffn_fn=fake_sage_ffn)

    patched_paths = set(patched_model.object_patches.keys())
    assert len(patched_paths) == 48 - len(BF16_FFN_BLOCKS) == 44

    for idx in BF16_FFN_BLOCKS:
        assert f"diffusion_model.transformer_blocks.{idx}.ff.forward" not in patched_paths

    for idx in range(48):
        if idx in BF16_FFN_BLOCKS:
            continue
        assert f"diffusion_model.transformer_blocks.{idx}.ff.forward" in patched_paths


def test_no_patch_when_sage_ffn_unavailable():
    """When sage v0.6 is not installed (`sage_ffn_fn=None`), _patch_impl
    must return the model UNCHANGED (no clone, no patches). This preserves
    prior FFN patches in the chain (e.g. KJNodes LTXVChunkFeedForward).
    Without this, a stock-fallback wrapper would overwrite ChunkFFN's
    patch at the same object_patches key and silently disable upstream
    chunking. Mirrors `nodes_sage.py:856-857` "disabled" precedent."""
    model = FakeModelForFFN()
    (out,) = AudioLoopHelperSageFFN._patch_impl(model, sage_ffn_fn=None)
    assert out is model, (
        "Must return the input model directly (no clone) when sage_ffn "
        "is unavailable. Mirrors nodes_sage.py disabled-mode precedent."
    )


def test_bookend_set_is_first_two_and_last_two():
    # Canary against accidental change of the bookend pattern. If the audit
    # ever revises which blocks are bf16, this test must be updated AND the
    # node's compatibility guard re-checked against the new audit.
    assert BF16_FFN_BLOCKS == frozenset({0, 1, 46, 47})


# -- Patched-forward behavior -------------------------------------------------


def test_patched_forward_short_seq_single_call():
    """Sequence ≤ SAGE_FFN_CHUNK_SEQ: sage_ffn called exactly once,
    no chunking overhead."""
    ff = FakeFF()
    short = _make_input(seq_len=100)  # well under 4096
    expected = torch.ones(1, 100, 8)
    sage_fn = MagicMock(name="sage_ffn", return_value=expected)

    patch = _FFNPatch(sage_ffn_fn=sage_fn, logger=_FFNFallbackLogger())
    bound = patch.__get__(ff, type(ff))

    result = bound(short)

    assert torch.equal(result, expected)
    sage_fn.assert_called_once()
    call_args = sage_fn.call_args[0]
    assert call_args[0] is short
    # Legacy fp8_ops path: weight is the raw fp8 tensor (no QuantizedTensor
    # unwrap needed). sage_ffn gets Linear.weight directly + Linear.scale_weight.
    assert call_args[1] is ff.net[0].proj.weight
    assert call_args[2] is ff.net[0].proj.scale_weight
    assert call_args[3] is ff.net[2].weight
    assert call_args[4] is ff.net[2].scale_weight


def test_patched_forward_long_seq_chunks_along_dim1():
    """Sequence > SAGE_FFN_CHUNK_SEQ: sage_ffn called once per chunk,
    output is the chunks concatenated along seq dim."""
    ff = FakeFF()
    long = _make_input(seq_len=10000)  # 10000 / 4096 → 3 chunks
    # sage_ffn returns the same shape as its input, so per-chunk
    # return needs to match the chunked input dims.
    sage_fn = MagicMock(name="sage_ffn", side_effect=lambda x, *_a, **_kw: torch.full_like(x, 0.5))

    patch = _FFNPatch(sage_ffn_fn=sage_fn, logger=_FFNFallbackLogger())
    bound = patch.__get__(ff, type(ff))

    result = bound(long)

    # 10000 split by 4096 → 3 chunks (4096 + 4096 + 1808)
    assert sage_fn.call_count == 3
    assert result.shape == (1, 10000, 8)
    assert torch.equal(result, torch.full((1, 10000, 8), 0.5))
    # Per-call inputs are the seq-split chunks.
    chunk_seq_lens = [args[0].shape[1] for args, _ in sage_fn.call_args_list]
    assert chunk_seq_lens == [4096, 4096, 1808]


def test_patched_forward_chunk_threshold_boundary():
    """Sequence == SAGE_FFN_CHUNK_SEQ: single call (≤, not <). Off-by-one
    canary on the chunk-vs-single branch condition."""
    ff = FakeFF()
    boundary = _make_input(seq_len=SAGE_FFN_CHUNK_SEQ)
    sage_fn = MagicMock(name="sage_ffn", side_effect=lambda x, *_a, **_kw: x)

    patch = _FFNPatch(sage_ffn_fn=sage_fn, logger=_FFNFallbackLogger())
    bound = patch.__get__(ff, type(ff))

    bound(boundary)

    assert sage_fn.call_count == 1


def test_patched_forward_unwraps_quantized_tensor_for_sage_ffn():
    """ComfyUI's modern fp8 path wraps weights in `QuantizedTensor`. sage_ffn
    asserts `w.dtype == float8_e4m3fn`; the raw fp8 storage lives on
    `weight._qdata` and the per-tensor scale on `weight._params.scale`. The
    wrapper must pass the unwrapped tensors, not the QuantizedTensor."""
    ff = FakeFF(scale_path="weight._params")
    short = _make_input(seq_len=100)
    sage_fn = MagicMock(name="sage_ffn", side_effect=lambda x, *_a, **_kw: x)

    patch = _FFNPatch(sage_ffn_fn=sage_fn, logger=_FFNFallbackLogger())
    bound = patch.__get__(ff, type(ff))

    bound(short)
    sage_fn.assert_called_once()
    call_args = sage_fn.call_args[0]
    # Critical: sage_ffn gets the raw _qdata tensor, NOT the QuantizedTensor
    # wrapper. Passing the wrapper would fail sage_ffn's dtype assert.
    assert call_args[1] is ff.net[0].proj.weight._qdata
    assert call_args[3] is ff.net[2].weight._qdata
    assert call_args[2] is ff.net[0].proj.weight._params.scale
    assert call_args[4] is ff.net[2].weight._params.scale


def test_patched_forward_falls_through_when_weight_scale_missing():
    """If a block doesn't have weight_scale attrs (e.g. user accidentally
    enabled the node on a bf16 checkpoint), patched forward must fall
    through to the stock path rather than crash."""
    ff = FakeFF(has_weight_scale=False)
    ff.net.return_value = "stock_fallback"
    sage_fn = MagicMock(name="sage_ffn", return_value="should_not_run")

    patch = _FFNPatch(sage_ffn_fn=sage_fn, logger=_FFNFallbackLogger())
    bound = patch.__get__(ff, type(ff))

    short = _make_input(seq_len=100)
    result = bound(short)

    assert result == "stock_fallback"
    sage_fn.assert_not_called()
    ff.net.assert_called_once_with(short)


def test_patched_forward_fallback_chains_to_prior_forward_when_provided():
    """When `prior_forward` is set (a previous patch at the same
    add_object_patch key, e.g. KJNodes ChunkFFN), the scale-missing
    fallback must call that wrapper instead of the unwrapped
    `self_module.net(x)`. Otherwise installing this node disables
    upstream chunking even when it can't activate sage_ffn."""
    ff = FakeFF(has_weight_scale=False)
    ff.net.return_value = "should_not_run"
    prior = MagicMock(name="prior_forward", return_value="prior_chained")
    sage_fn = MagicMock(name="sage_ffn")

    patch = _FFNPatch(sage_ffn_fn=sage_fn, logger=_FFNFallbackLogger(),
                     prior_forward=prior)
    bound = patch.__get__(ff, type(ff))

    short = _make_input(seq_len=100)
    result = bound(short)

    assert result == "prior_chained"
    prior.assert_called_once_with(short)
    ff.net.assert_not_called()
    sage_fn.assert_not_called()


def test_patched_forward_falls_through_when_weight_on_different_device_than_activation():
    """ComfyUI's model loader moves fp8 weights to cuda AFTER our patch
    installs; partial-load offload (the 2.6 GB "offloaded" portion seen
    in the v06_2 audit log) can also leave a specific block's weights on
    CPU at call time. Resolving weights at install time captures stale
    storage and triggers `sage_ffn: all tensors must be on CUDA, got
    x=cuda:0 w1=cpu w2=cpu` at every FFN call.

    The wrapper must (a) resolve weights at call time, not install time,
    so the live device is read; (b) compare against `x.device` and fall
    through to prior_forward when they don't match. ChunkFFN's wrapper
    (prior_forward) routes through ComfyUI's `cast_bias_weight` which
    handles the cpu→cuda move correctly via `fp8_linear`.
    """
    ff = FakeFF(scale_path="weight._params")
    # Activation on `meta` device, weights stay on cpu via the FakeFF
    # default `torch.empty(1, dtype=torch.uint8)`. Device mismatch →
    # wrapper must fall through to prior_forward.
    short = torch.empty(1, 100, 8, device="meta")
    prior = MagicMock(name="prior_forward", return_value="prior_chained")
    sage_fn = MagicMock(name="sage_ffn", return_value="should_not_run")

    patch = _FFNPatch(sage_ffn_fn=sage_fn, logger=_FFNFallbackLogger(),
                     prior_forward=prior)
    bound = patch.__get__(ff, type(ff))

    result = bound(short)

    assert result == "prior_chained"
    sage_fn.assert_not_called()
    prior.assert_called_once_with(short)


def test_patched_forward_falls_through_when_sage_ffn_raises():
    """Defensive: sage_ffn raising must not crash the render."""
    ff = FakeFF()
    ff.net.return_value = "stock_fallback"
    sage_fn = MagicMock(name="sage_ffn", side_effect=RuntimeError("kernel boom"))

    patch = _FFNPatch(sage_ffn_fn=sage_fn, logger=_FFNFallbackLogger())
    bound = patch.__get__(ff, type(ff))

    short = _make_input(seq_len=100)
    result = bound(short)

    assert result == "stock_fallback"
    ff.net.assert_called_once_with(short)


# -- Compatibility guards -----------------------------------------------------


def test_model_without_diffusion_model_returns_unpatched(fake_sage_ffn):
    class FakeModelNoDM(FakeModelPatcher):
        def get_model_object(self, _name):
            raise KeyError("no diffusion_model")

        def clone(self):
            return FakeModelNoDM()

    model = FakeModelNoDM()
    # Use a fake sage_ffn so we actually exercise the diffusion_model guard
    # (the None path short-circuits earlier with the no-op behavior).
    (out,) = AudioLoopHelperSageFFN._patch_impl(model, sage_ffn_fn=fake_sage_ffn)
    # No crash, returned an unpatched clone.
    assert out is not model
    assert not hasattr(out, "object_patches") or not getattr(out, "object_patches", {})


def test_model_without_transformer_blocks_returns_unpatched(fake_sage_ffn):
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
    (out,) = AudioLoopHelperSageFFN._patch_impl(model, sage_ffn_fn=fake_sage_ffn)
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


def test_enabled_true_no_patch_when_sage_unavailable(monkeypatch):
    """Sanity: with enabled=True but sage v0.6 unavailable, _resolve_sage_ffn
    returns None and the node is a no-op. Model passes through unchanged
    so prior FFN patches in the chain (LTXVChunkFeedForward) survive."""
    monkeypatch.setattr(nodes_ffn, "_resolve_sage_ffn", lambda: None)
    model = FakeModelForFFN()
    out = AudioLoopHelperSageFFN.execute(model, enabled=True)
    returned = out[0] if isinstance(out, tuple) else getattr(out, "result", out)[0]
    assert returned is model


def test_enabled_true_patches_44_blocks_when_sage_available(monkeypatch):
    """When sage v0.6 IS available, _resolve_sage_ffn returns a callable
    and the dispatch loop installs 44 patches (48 minus bookends)."""
    monkeypatch.setattr(nodes_ffn, "_resolve_sage_ffn",
                        lambda: MagicMock(name="fake_sage_ffn"))
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
