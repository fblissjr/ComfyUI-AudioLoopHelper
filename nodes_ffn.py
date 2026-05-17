"""AudioLoopHelperSageFFN -- consumer-side patch for sage v0.6 fused MLP.

Patches `block.ff.forward` on the 44 fp8-quantized transformer blocks of
LTX 2.3 distilled checkpoints, routing them through
`sageattention.sage_ffn` when v0.6+ of the fork is installed. The 4
bookend blocks `[0, 1, 46, 47]` stay on stock `block.ff.forward` (their
weights are bf16, not fp8 — confirmed by the 2026-05-15 safetensors
audit).

When `sageattention.sage_ffn` is not available, this node is a
**complete no-op**: returns the model unchanged with no patches
applied. This preserves any prior FFN-touching patches in the chain
(notably KJNodes `LTXVChunkFeedForward`). Adding a stock-fallback
wrapper would overwrite the prior patch at the same
`add_object_patch` key and silently disable upstream chunking.

When sage v0.6 IS available and the node fires, the sage_ffn path
replaces any prior FFN patch in the chain. This is intentional —
sage_ffn's two-kernel split has the same memory footprint as the
un-chunked baseline, so stacking chunking on top is redundant.

Bookend pattern reference: `internal/reference/sage_optimization_landscape.md`
"""

from __future__ import annotations

import logging
import types

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


class _FFNPatch:
    """Descriptor-style binder mirroring KJNodes' `LTXVffnChunkPatch`
    pattern (`ComfyUI-KJNodes/nodes/ltxv_nodes.py:544-554`).

    Replaces `block.ff.forward` via `add_object_patch`. The patched
    callable receives `self_module` (= `block.ff`, a `FeedForward`
    instance) plus the activation tensor. Mirrors KJNodes' shape
    deliberately so composition with `LTXVChunkFeedForward` works
    correctly when both nodes appear in the same model patch chain.
    """

    def __init__(self, sage_ffn_fn, logger: _FFNFallbackLogger):
        self._sage_ffn = sage_ffn_fn
        self._logger = logger

    def __get__(self, obj, objtype=None):
        sage_ffn_fn = self._sage_ffn
        logger = self._logger

        def wrapped_forward(self_module, x, *args, **kwargs):
            # Wrapper signature per sage v0.6 scoping doc:
            #   sage_ffn(x, w1, s1, w2, s2, b1=None, b2=None) -> y
            # net[0].proj is up-projection (hidden -> inner);
            # net[2] is down-projection (inner -> hidden).
            # `.weight_scale` is the per-tensor f32 scalar from the
            # comfy fp8 convention.
            try:
                proj_in = self_module.net[0].proj
                proj_out = self_module.net[2]
                s1 = getattr(proj_in, "weight_scale", None)
                s2 = getattr(proj_out, "weight_scale", None)
                if s1 is None or s2 is None:
                    # Block isn't fp8-quantized in the expected format
                    # (e.g. user loaded a bf16 checkpoint into an fp8
                    # block index). Stock path is correct.
                    return self_module.net(x)
                b1 = getattr(proj_in, "bias", None)
                b2 = getattr(proj_out, "bias", None)
                return sage_ffn_fn(x, proj_in.weight, s1, proj_out.weight, s2, b1, b2)
            except Exception as exc:
                # Don't crash a render if sage_ffn raises. Log once per
                # (error_type, shape) so a real kernel bug surfaces
                # without spamming logs per FFN call.
                shape = tuple(getattr(x, "shape", ()))
                logger.log_once(exc, shape)
                return self_module.net(x)

        return types.MethodType(wrapped_forward, obj)


class AudioLoopHelperSageFFN(io.ComfyNode):
    """Patch `block.ff.forward` on LTX 2.3's fp8-quantized FFN blocks
    to route through sage v0.6's fused fp8 MLP kernel.

    Skips the 4 bookend blocks `[0, 1, 46, 47]` whose FFN weights are
    bf16 in the distilled checkpoint. Composes with `LTXVChunkFeedForward`
    (KJNodes) — does not replace it.

    Default `enabled=False`: production A/B (2026-05-15) showed
    sage_ffn is slower than torch's bf16-dequant reference at LTX FFN
    shapes (3% at stage-1, 20% at stage-2 T=44880; ~1.8% e2e). The
    synthetic-bench speedup didn't translate due to L2 cache
    contention with attention's working set + kernel-launch overhead
    at LTX call counts. Ships as a completeness primitive (the only
    fp8-native fused MLP for ComfyUI consumer-app on sm89), not a
    perf win on the current Triton stack.

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
                "COMPOSE WITH LTXVChunkFeedForward (KJNodes), don't "
                "replace it. The v0.6 design is a two-kernel split: "
                "intermediate hits HBM, same memory footprint as the "
                "un-chunked baseline.\n\n"
                "STATUS (post in-pipeline validation 2026-05-15): "
                "sage_ffn is the only fp8-native fused MLP kernel for "
                "ComfyUI consumer-app on sm89, but production A/B on "
                "FML2V multi-guide showed it is ~1-20% SLOWER than "
                "torch's bf16-dequant reference path (3% at stage-1, "
                "20% at stage-2 T=44880; ~1.8% e2e wall-time slower). "
                "Synthetic-bench predicted a speedup; the gap closes "
                "in production due to L2 cache contention with the "
                "neighboring attention sub-modules + kernel-launch "
                "overhead across ~1000 FFN calls per render. Ship "
                "STATUS: completeness primitive — opt-in if you want "
                "the fp8 path for forward-compat reasons. Default off."
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

        logger = _FFNFallbackLogger()
        n_patched = 0
        for idx, block in enumerate(blocks):
            if idx in BF16_FFN_BLOCKS:
                continue
            ff_module = getattr(block, "ff", None)
            if ff_module is None:
                continue
            patched = _FFNPatch(sage_ffn_fn, logger).__get__(ff_module, block.__class__)
            model_clone.add_object_patch(
                f"diffusion_model.transformer_blocks.{idx}.ff.forward",
                patched,
            )
            n_patched += 1

        _LOGGER.debug(
            "AudioLoopHelperSageFFN: patched %d FFN forwards (skipped %d bookend blocks: %s)",
            n_patched, len(BF16_FFN_BLOCKS), sorted(BF16_FFN_BLOCKS),
        )
        return (model_clone,)
