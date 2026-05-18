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
    """Find the per-tensor fp8 weight scale on a Linear, trying multiple
    storage conventions used across ComfyUI's fp8 stack versions.

    Returns the scale Tensor on success, None when the layer isn't
    fp8-quantized in any recognised format. The path that succeeded
    is logged once per session via `_log_scale_path_once` so the next
    render confirms which storage convention is active.
    """
    # Legacy fp8_ops convention (comfy/ops.py:836): direct attribute on
    # the Linear instance.
    s = getattr(linear_module, "scale_weight", None)
    if isinstance(s, torch.Tensor):
        _log_scale_path_once("Linear.scale_weight")
        return s
    s = getattr(linear_module, "weight_scale", None)
    if isinstance(s, torch.Tensor):
        _log_scale_path_once("Linear.weight_scale")
        return s
    # Modern MixedPrecisionOps path: weight is wrapped in a
    # comfy_kitchen.QuantizedTensor whose .Params carries the scale.
    w = getattr(linear_module, "weight", None)
    if w is not None:
        # comfy_kitchen QuantizedTensor exposes its layout params under
        # a few possible names depending on version. Try in order.
        for attr in ("_params", "params", "layout_params", "_layout_params"):
            params = getattr(w, attr, None)
            if params is not None:
                scale = getattr(params, "scale", None)
                if isinstance(scale, torch.Tensor):
                    _log_scale_path_once(f"weight.{attr}.scale")
                    return scale
    return None


_LOGGED_SCALE_PATHS: set[str] = set()


def _log_scale_path_once(path: str) -> None:
    """Logs the first time each scale-resolution path succeeds so logs
    confirm whether `_extract_fp8_scale` is finding the scale on the
    current ComfyUI version. One line per session per path."""
    if path in _LOGGED_SCALE_PATHS:
        return
    _LOGGED_SCALE_PATHS.add(path)
    _LOGGER.info("AudioLoopHelperSageFFN: fp8 scale found via %s", path)


class _FFNFallbackLogger:
    """Dedup'd warning when `sage_ffn` raises and we fall through to the
    stock path. Mirrors `nodes_sage.SageFallbackLogger`. Prevents a v0.6
    kernel bug from silently regressing every block to stock-path while
    the user sees "no perf gain" with no error in the log.

    Key shape `(error_type, shape)` keeps the log small — one line per
    distinct failure mode per shape, not per call.
    """

    def __init__(self):
        self._seen: set[tuple[str, tuple]] = set()
        # Track per-block fallback-because-no-scale events. Logged once
        # per session so a wrong-attribute-name regression surfaces loudly.
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

    def log_scale_missing_once(self) -> None:
        """First time per session that no scale could be extracted from any
        recognised attribute path. Logs once and stays quiet thereafter."""
        if self._scale_missing_logged:
            return
        self._scale_missing_logged = True
        _LOGGER.warning(
            "AudioLoopHelperSageFFN: could not extract fp8 weight_scale via "
            "any known attribute path (legacy scale_weight / weight_scale / "
            "modern QuantizedTensor.{_params,params,layout_params}.scale). "
            "Every FFN call will fall back to stock path. Check ComfyUI fp8 "
            "ops version or the model's weight wrapper."
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

        def _fallback(self_module, x):
            # When prior_forward is set, it's an already-bound method
            # (captured from model_clone.object_patches). Call positionally.
            if prior_forward is not None:
                return prior_forward(x)
            return self_module.net(x)

        def wrapped_forward(self_module, x, *args, **kwargs):
            # sage_ffn signature: (x, w1, s1, w2, s2, b1=None, b2=None) -> y
            # net[0].proj is up-projection (hidden -> inner);
            # net[2] is down-projection (inner -> hidden).
            try:
                proj_in = self_module.net[0].proj
                proj_out = self_module.net[2]
                s1 = _extract_fp8_scale(proj_in)
                s2 = _extract_fp8_scale(proj_out)
                if s1 is None or s2 is None:
                    logger.log_scale_missing_once()
                    return _fallback(self_module, x)
                b1 = getattr(proj_in, "bias", None)
                b2 = getattr(proj_out, "bias", None)
                if x.shape[1] <= SAGE_FFN_CHUNK_SEQ:
                    return sage_ffn_fn(x, proj_in.weight, s1, proj_out.weight, s2, b1, b2)
                outs = [
                    sage_ffn_fn(c, proj_in.weight, s1, proj_out.weight, s2, b1, b2)
                    for c in x.split(SAGE_FFN_CHUNK_SEQ, dim=1)
                ]
                return torch.cat(outs, dim=1)
            except Exception as exc:
                # Don't crash a render if sage_ffn raises. Log once per
                # (error_type, shape) so a real kernel bug surfaces
                # without spamming logs per FFN call.
                shape = tuple(getattr(x, "shape", ()))
                logger.log_once(exc, shape)
                return _fallback(self_module, x)

        return types.MethodType(wrapped_forward, obj)


class AudioLoopHelperSageFFN(io.ComfyNode):
    """Patch `block.ff.forward` on LTX 2.3's fp8-quantized FFN blocks
    to route through sage v0.6's fused fp8 MLP kernel.

    Skips the 4 bookend blocks `[0, 1, 46, 47]` whose FFN weights are
    bf16 in the distilled checkpoint. Composes with `LTXVChunkFeedForward`
    (KJNodes) — does not replace it.

    Default `enabled=False` pending a clean in-pipeline A/B. The
    2026-05-18 measurement that suggested ~12% slowdown was invalidated
    by a wrong-attribute-name bug in `_extract_fp8_scale`: the wrapper
    only checked `Linear.weight_scale`, but modern ComfyUI fp8 ops store
    the scale on a `QuantizedTensor` wrapping `Linear.weight`. sage_ffn
    never fired; the wrapper fell back to `self_module.net(x)` on every
    call (un-chunked stock FFN) since our `add_object_patch` overwrote
    KJNodes ChunkFFN at the same key. The +12% delta was the lost
    chunking, not anything to do with sage. Fix: multi-path scale lookup
    + prior-patch chaining on fallback. Re-baseline required.

    When sage v0.6 is unavailable, the node is a no-op: returns the
    model unchanged with no patches applied. Prior FFN patches in the
    chain (e.g. KJNodes LTXVChunkFeedForward) remain active.

    Detail: `internal/reference/sage_optimization_landscape.md`.
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
                "Default off pending a clean A/B — earlier 2026-05-18 "
                "result was invalidated by a wrong-attribute-name bug "
                "that prevented sage_ffn from firing at all. Fix landed "
                "2026-05-18; re-baseline before declaring perf direction. "
                "Perf history in "
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
