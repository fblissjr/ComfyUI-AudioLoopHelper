"""AudioLoopHelperSageFFN -- consumer-side patch for sage v0.6 fused MLP.

Patches `block.ff.forward` on the 44 fp8-quantized transformer blocks of
LTX 2.3 distilled checkpoints, routing them through
`sageattention.sage_ffn`. The 4 bookend blocks `[0, 1, 46, 47]` stay
on stock `block.ff.forward` (their weights are bf16, not fp8).

When `sageattention.sage_ffn` is not available, the node is a no-op:
returns the model unchanged with no patches applied. Prior FFN
patches in the chain (notably KJNodes `LTXVChunkFeedForward`) remain
active. Adding a stock-fallback wrapper would overwrite their patch
at the same `add_object_patch` key.

When sage v0.6 IS available, the patched forward splits inputs along
the sequence dim at `SAGE_FFN_CHUNK_SEQ` (default 4096) and calls
sage_ffn on each chunk. The in-wrapper chunking preserves L2 locality
against neighboring attention sub-modules' working set since our
patch necessarily overwrites ChunkFFN's at the same key.

Bookend audit + perf history: `internal/reference/sage_optimization_landscape.md`
"""

from __future__ import annotations

import logging
import types

import torch

# Mirror the io stub pattern from nodes_sage.py + nodes_easycache.py so
# this module imports cleanly under pytest (no ComfyUI).
try:
    from comfy_api.latest import io
    from typing_extensions import override
except ImportError:  # pragma: no cover -- only hit outside ComfyUI runtime
    class _Passthrough:  # noqa: D401 -- minimal stub
        def __init__(self, *_args, **_kwargs):
            pass
        def __call__(self, *_args, **_kwargs):
            return self
        def __getattr__(self, _name):
            return self

    class _IOStub:
        class ComfyNode:
            pass
        class NodeOutput:
            def __new__(cls, *args, **_kwargs):
                # Matches comfy_api's tuple-return contract under the stub.
                return args
        class Schema(_Passthrough):
            pass
        class Model(_Passthrough):
            class Input(_Passthrough):
                pass
            class Output(_Passthrough):
                pass
        class Boolean(_Passthrough):
            class Input(_Passthrough):
                pass

    io = _IOStub()  # type: ignore[assignment]
    def override(f):  # type: ignore[misc]
        return f


_LOGGER = logging.getLogger(__name__)


# Bookend pattern from the fp8 audit: these 4 transformer blocks keep their
# FFN weights as bf16 (likely because Lightricks found empirically that fp8
# quantization degraded accuracy at the entry/exit of the network). The
# other 44 blocks have fp8 FFN weights and route through sage_ffn.
BF16_FFN_BLOCKS: frozenset[int] = frozenset({0, 1, 46, 47})


# Seq-dim chunk size for the in-wrapper chunked sage_ffn path. Matches the
# KJNodes LTXVChunkFeedForward widget default (chunk_size=4096) so the
# L2-locality discipline carries over even though our add_object_patch
# replaces theirs at the same key. Sequences ≤ this length skip the chunk
# loop and hit sage_ffn in a single call.
SAGE_FFN_CHUNK_SEQ: int = 4096


def _resolve_sage_ffn():
    """Return `sageattention.sage_ffn` if available, else None.

    Mirrors the sage-symbol resolution shape used at `nodes_sage.py:125-129`;
    a third consumer should factor these into a shared helper.

    Resolved fresh on every node execute so users who `pip install` v0.6
    while ComfyUI is running can pick it up on the next render. Import
    miss is cached by `sys.modules` after first call.
    """
    try:
        import sageattention as _sa
    except ImportError:
        return None
    return getattr(_sa, "sage_ffn", None)


def _extract_fp8_scale(linear_module):
    """Return `(scale, path_label)` for an fp8 Linear's per-tensor weight scale.

    Two conventions, both probed:
    - `Linear.scale_weight` (legacy fp8_ops, `comfy/ops.py:836`)
    - `Linear.weight.{_params,layout_params}.scale` — modern
      MixedPrecisionOps where weight is a `comfy_kitchen.QuantizedTensor`.
      `_params` is the raw storage; `.layout_params` is the public property
      that aliases it (`comfy_kitchen.tensor.base`). Probing both means we
      survive future renames in either direction without a code change.

    Returns `(None, None)` when no convention matches — the caller treats
    this as "not fp8-quantized in any recognised format" and installs a
    stock-passthrough forward instead of `sage_ffn`.
    """
    s = getattr(linear_module, "scale_weight", None)
    if isinstance(s, torch.Tensor):
        return s, "Linear.scale_weight"
    s = getattr(linear_module, "weight_scale", None)
    if isinstance(s, torch.Tensor):
        return s, "Linear.weight_scale"
    w = getattr(linear_module, "weight", None)
    if w is not None:
        for attr in ("_params", "layout_params"):
            params = getattr(w, attr, None)
            if params is not None:
                scale = getattr(params, "scale", None)
                if isinstance(scale, torch.Tensor):
                    return scale, f"weight.{attr}.scale"
    return None, None


class _FFNFallbackLogger:
    """Per-instance dedup'd warnings + info for the sage FFN patch.

    Three keyspaces, three lifetimes — kept together so a single logger
    instance threaded through every block's `_FFNPatch` (one per
    `_patch_impl` call) carries all the dedup state for a render.
    """

    def __init__(self):
        # Per-(exc_type, shape) for sage_ffn runtime failures.
        self._seen: set[tuple[str, tuple]] = set()
        # Per-resolution-path for first-time scale-found messages.
        self._seen_paths: set[str] = set()
        # Binary: no scale path resolved for any block this render.
        self._scale_missing_logged: bool = False

    def log_once(self, exc: BaseException, shape: tuple) -> None:
        key = (type(exc).__name__, shape)
        if key in self._seen:
            return
        self._seen.add(key)
        _LOGGER.warning(
            "AudioLoopHelperSageFFN: sage_ffn raised %s at shape %r; falling "
            "through to stock FFN path. Further identical failures suppressed.",
            type(exc).__name__, shape,
        )

    def log_scale_path_once(self, path: str) -> None:
        if path in self._seen_paths:
            return
        self._seen_paths.add(path)
        _LOGGER.info("AudioLoopHelperSageFFN: fp8 scale found via %s", path)

    def log_scale_missing_once(self) -> None:
        if self._scale_missing_logged:
            return
        self._scale_missing_logged = True
        _LOGGER.warning(
            "AudioLoopHelperSageFFN: could not extract fp8 weight_scale on a "
            "patched block. Falling back to stock path for that block. Check "
            "ComfyUI fp8 ops version or the model's weight wrapper."
        )


class _FFNPatch:
    """Descriptor-style binder mirroring KJNodes' `LTXVffnChunkPatch`
    pattern (`ComfyUI-KJNodes/nodes/ltxv_nodes.py:544-554`).

    Replaces `block.ff.forward` via `add_object_patch`. The patched
    callable receives `self_module` (= `block.ff`, a `FeedForward`
    instance) plus the activation tensor. Mirrors KJNodes' shape
    deliberately so composition with `LTXVChunkFeedForward` works
    correctly when both nodes appear in the same model patch chain.
    """

    def __init__(self, sage_ffn_fn, logger: _FFNFallbackLogger, prior_forward=None):
        self._sage_ffn = sage_ffn_fn
        self._logger = logger
        # Prior add_object_patch value at the same key, if any (e.g.
        # KJNodes LTXVChunkFeedForward's chunked-stock wrapper). Used as
        # the fallback target so falling out of sage_ffn preserves prior
        # patches' L2-locality discipline instead of dropping to the
        # unwrapped LTX forward.
        self._prior_forward = prior_forward

    def __get__(self, obj, objtype=None):
        sage_ffn_fn = self._sage_ffn
        logger = self._logger
        prior_forward = self._prior_forward

        # Resolve scale + weights + biases ONCE at bind time. ComfyUI
        # invokes `__get__` per block when `add_object_patch` registers
        # the override, so this fires ~44 times per node-execute. Moving
        # the resolution out of `wrapped_forward` saves ~6 getattrs ×
        # ~1000 FFN calls per render.
        proj_in = obj.net[0].proj
        proj_out = obj.net[2]
        s1, path1 = _extract_fp8_scale(proj_in)
        s2, _ = _extract_fp8_scale(proj_out)
        if s1 is None or s2 is None or path1 is None:
            # Scale not resolvable on this block. Install a stock
            # passthrough that respects prior patches (e.g. ChunkFFN).
            # Logged once per logger instance — a wrong-attribute-name
            # regression surfaces loudly at install, not as silent
            # per-call fallback.
            logger.log_scale_missing_once()
            def stock_passthrough(self_module, x, *args, **kwargs):
                if prior_forward is not None:
                    return prior_forward(x)
                return self_module.net(x)
            return types.MethodType(stock_passthrough, obj)

        logger.log_scale_path_once(path1)
        w1 = proj_in.weight
        w2 = proj_out.weight
        b1 = getattr(proj_in, "bias", None)
        b2 = getattr(proj_out, "bias", None)

        def wrapped_forward(self_module, x, *args, **kwargs):
            # sage_ffn signature: (x, w1, s1, w2, s2, b1=None, b2=None) -> y
            try:
                if x.shape[1] <= SAGE_FFN_CHUNK_SEQ:
                    return sage_ffn_fn(x, w1, s1, w2, s2, b1, b2)
                outs = [
                    sage_ffn_fn(c, w1, s1, w2, s2, b1, b2)
                    for c in x.split(SAGE_FFN_CHUNK_SEQ, dim=1)
                ]
                return torch.cat(outs, dim=1)
            except Exception as exc:
                # Don't crash a render if sage_ffn raises. Log once per
                # (error_type, shape) so a real kernel bug surfaces
                # without spamming logs per FFN call.
                shape = tuple(getattr(x, "shape", ()))
                logger.log_once(exc, shape)
                if prior_forward is not None:
                    return prior_forward(x)
                return self_module.net(x)

        return types.MethodType(wrapped_forward, obj)


class AudioLoopHelperSageFFN(io.ComfyNode):
    """Patch `block.ff.forward` on LTX 2.3's fp8-quantized FFN blocks
    to route through sage v0.6's fused fp8 MLP kernel.

    Skips the 4 bookend blocks `[0, 1, 46, 47]` whose FFN weights are
    bf16 in the distilled checkpoint. Composes with `LTXVChunkFeedForward`
    (KJNodes) — does not replace it.

    Default `enabled=False` pending a clean in-pipeline A/B against
    chunked stock fp8-dequant.

    When sage v0.6 is unavailable, the node is a no-op: returns the
    model unchanged with no patches applied. Prior FFN patches in the
    chain (e.g. KJNodes LTXVChunkFeedForward) remain active.

    Perf history + prior-measurement postmortems:
    `internal/reference/sage_optimization_landscape.md`.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioLoopHelperSageFFN",
            display_name="AudioLoopHelper Sage FFN (v0.6)",
            category="AudioLoopHelper/experimental",
            description=(
                "Routes LTX 2.3 distilled FFN matmuls through "
                "sage v0.6's fused fp8 MLP kernel (sage_ffn). Skips "
                "the 4 bookend blocks [0, 1, 46, 47] whose FFN weights "
                "are bf16 in the distilled checkpoint.\n\n"
                "REQUIRES SageAttention-ada >= v0.6 "
                "(github.com/fblissjr/SageAttention-ada). When sage_ffn "
                "is unavailable, this node is a no-op (model passes "
                "through unchanged; prior FFN patches in the chain like "
                "LTXVChunkFeedForward remain active). Restart ComfyUI "
                "after installing v0.6.\n\n"
                "Chunks along seq dim at SAGE_FFN_CHUNK_SEQ "
                "(default 4096) to preserve L2 locality against "
                "neighboring attention sub-modules. Our patch overwrites "
                "KJNodes LTXVChunkFeedForward at the same "
                "add_object_patch key; the chunking lives in this wrapper "
                "instead.\n\n"
                "Default off pending a clean in-pipeline A/B against "
                "chunked stock fp8-dequant. Perf history in "
                "internal/reference/sage_optimization_landscape.md."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "enabled",
                    default=False,
                    tooltip=(
                        "When False (default), the model passes through "
                        "unchanged — recommended until a v0.6.1+ closes "
                        "the production gap with torch reference. When "
                        "True, routes FFN through sage_ffn. Useful for "
                        "A/B comparisons and forward-compat testing."
                    ),
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    @override
    def execute(cls, model, enabled: bool = True) -> io.NodeOutput:  # type: ignore[override]
        if not enabled:
            return io.NodeOutput(model)
        # Resolve sage_ffn fresh on every execute so install-then-rerun works
        # without a ComfyUI restart. One ImportError per execute when sage is
        # absent; cached by sys.modules after the first miss.
        (patched,) = cls._patch_impl(model, sage_ffn_fn=_resolve_sage_ffn())
        return io.NodeOutput(patched)

    @classmethod
    def _patch_impl(cls, model, *, sage_ffn_fn):
        """Testable seam. Returns `(patched_model,)`.

        Separated from `execute()` so tests can pass a fake model +
        a fake `sage_ffn_fn` without needing the v3 io.NodeOutput
        wrapper or the real `sageattention` import.
        """
        if sage_ffn_fn is None:
            # No-op: return the model unchanged so prior FFN patches in
            # the chain (e.g. KJNodes LTXVChunkFeedForward) survive.
            # Patching a stock-path fallback here would overwrite their
            # patch at the same add_object_patch key. Mirrors the
            # `mode == "disabled"` precedent at `nodes_sage.py:856-857`.
            _LOGGER.warning(
                "AudioLoopHelperSageFFN: sageattention.sage_ffn not available; "
                "node is a no-op. Install SageAttention-ada >= v0.6 + restart "
                "ComfyUI to activate."
            )
            return (model,)

        model_clone = model.clone()
        try:
            diffusion_model = model_clone.get_model_object("diffusion_model")
        except Exception:
            _LOGGER.warning(
                "AudioLoopHelperSageFFN: model has no 'diffusion_model' attribute; "
                "returning unpatched. Probably not an LTX 2.3 model."
            )
            return (model_clone,)

        blocks = getattr(diffusion_model, "transformer_blocks", None)
        if blocks is None:
            _LOGGER.warning(
                "AudioLoopHelperSageFFN: diffusion_model has no 'transformer_blocks'; "
                "returning unpatched. Probably not an LTX 2.3 model."
            )
            return (model_clone,)

        # Read existing patches so we can chain our fallback to them
        # instead of dropping to the unwrapped LTX forward. ComfyUI
        # ModelPatcher stores patches in `object_patches`; tolerate the
        # attribute being absent on older ComfyUI versions or fake test
        # models — chaining then degrades to plain self_module.net(x).
        existing_patches = getattr(model_clone, "object_patches", None) or {}

        logger = _FFNFallbackLogger()
        n_patched = 0
        n_chained = 0
        for idx, block in enumerate(blocks):
            if idx in BF16_FFN_BLOCKS:
                continue
            ff_module = getattr(block, "ff", None)
            if ff_module is None:
                continue
            key = f"diffusion_model.transformer_blocks.{idx}.ff.forward"
            prior = existing_patches.get(key)
            if prior is not None:
                n_chained += 1
            patched = _FFNPatch(sage_ffn_fn, logger, prior_forward=prior).__get__(ff_module, block.__class__)
            model_clone.add_object_patch(key, patched)
            n_patched += 1

        _LOGGER.info(
            "AudioLoopHelperSageFFN: patched %d FFN forwards (skipped %d bookend blocks: %s; "
            "chained-to-prior-patch on %d blocks)",
            n_patched, len(BF16_FFN_BLOCKS), sorted(BF16_FFN_BLOCKS), n_chained,
        )
        return (model_clone,)
