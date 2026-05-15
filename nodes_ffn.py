"""AudioLoopHelperSageFFN -- consumer-side patch for sage v0.6 fused MLP.

Patches `block.ff.forward` on the 44 fp8-quantized transformer blocks of
LTX 2.3 distilled checkpoints, routing them through
`sageattention.sage_ffn` when v0.6+ of the fork is installed. The 4
bookend blocks `[0, 1, 46, 47]` stay on stock `block.ff.forward` (their
weights are bf16, not fp8 — confirmed by the 2026-05-15 safetensors
audit).

Mocking phase (pre-v0.6): when `sageattention.sage_ffn` is not available
in the installed sage package, the patched forward falls through to the
stock path (`self_module.net(x)`). Numerically identical to no-patch;
exercises the patching machinery so the wiring is testable now and the
swap to the real sage_ffn at v0.6 ship is a one-line change.

Compose with `LTXVChunkFeedForward`: the v0.6 design is two-kernel
split (intermediate hits HBM, same memory footprint as un-chunked
baseline). On 24 GiB cards you still need chunking for stage-2 memory
management; sage_ffn provides the fp8-native matmul speedup on top.

Bookend pattern reference: `internal/reference/sage_optimization_landscape.md`
"Day 9: our work when it fires."
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

    Resolved fresh on every node execute so users who `pip install` v0.6
    while ComfyUI is running can pick it up on the next render without a
    full restart. Cost: one ImportError per node execute when sage is
    absent (cached by `sys.modules` after first miss).
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

    def __init__(self, sage_ffn_fn, logger: _FFNFallbackLogger | None = None):
        self._sage_ffn = sage_ffn_fn
        self._logger = logger

    def __get__(self, obj, objtype=None):
        sage_ffn_fn = self._sage_ffn
        logger = self._logger

        def wrapped_forward(self_module, x, *args, **kwargs):
            # Mock-phase fallback: sage v0.6 not installed -> stock path.
            # Numerically identical to no-patch; lets us land the wiring
            # and tests now, swap the implementation when v0.6 ships.
            if sage_ffn_fn is None:
                return self_module.net(x)

            # When sage v0.6 lands, this branch hits. Expected wrapper
            # signature (per sage's scoping doc):
            #   sage_ffn(x, w1, s1, w2, s2, b1=None, b2=None) -> y
            # Weight + scale access: net[0].proj is the up-projection
            # (hidden -> inner); net[2] is the down-projection
            # (inner -> hidden). `.weight_scale` is the per-tensor f32
            # scalar from the comfy fp8 convention.
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
                # Defensive: don't crash a render if sage_ffn raises.
                # Log once per (error_type, shape) so a real kernel bug
                # surfaces without spamming logs per FFN call.
                shape = tuple(getattr(x, "shape", ()))
                if logger is not None:
                    logger.log_once(exc, shape)
                return self_module.net(x)

        return types.MethodType(wrapped_forward, obj)


class AudioLoopHelperSageFFN(io.ComfyNode):
    """Patch `block.ff.forward` on LTX 2.3's fp8-quantized FFN blocks
    to route through sage v0.6's fused fp8 MLP kernel.

    Skips the 4 bookend blocks `[0, 1, 46, 47]` whose FFN weights are
    bf16 in the distilled checkpoint. Composes with `LTXVChunkFeedForward`
    (KJNodes) — does not replace it. On 24 GiB cards keep chunking
    enabled for stage-2 memory management; sage_ffn adds fp8-native
    matmul speedup on top.

    Mocking phase: when `sageattention.sage_ffn` is not available
    (sage < v0.6 installed), the node still installs patches but they
    fall through to the stock path. Numerically identical to no-patch;
    lets users wire the node into workflows before v0.6 ships.

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
                "REQUIRES the SageAttention-ada fork v0.6+ "
                "(github.com/fblissjr/SageAttention-ada). Older sage "
                "versions: patches are still installed but fall through "
                "to the stock FFN path (numerically identical to "
                "no-patch). Restart ComfyUI after installing v0.6 to "
                "pick up the real kernel.\n\n"
                "COMPOSE WITH LTXVChunkFeedForward (KJNodes), don't "
                "replace it. The v0.6 design is two-kernel split: "
                "intermediate hits HBM, same memory footprint as "
                "un-chunked baseline. On 24 GiB cards keep chunking "
                "enabled for stage-2 memory management; sage_ffn adds "
                "fp8-native matmul speedup on top.\n\n"
                "Expected delivered speedup: ~10-25% FFN matmul vs "
                "torch's bf16-dequant reference path, depending on "
                "Triton autotune. e2e impact at our measured 24-27% "
                "FFN share: ~2-5% wall-time reduction. Modest but real; "
                "the uncontested-availability framing (only fp8-native "
                "fused MLP for ComfyUI consumer-app on sm89) is the "
                "primary wedge, not the absolute speedup magnitude."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip=(
                        "When False, the model passes through unchanged. "
                        "Useful for A/B testing sage_ffn against baseline "
                        "without rewiring the workflow."
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

        if sage_ffn_fn is None:
            _LOGGER.info(
                "AudioLoopHelperSageFFN: sageattention.sage_ffn not available "
                "(install SageAttention-ada >= v0.6 to activate). Patching "
                "with stock-path fallback for now."
            )

        logger = _FFNFallbackLogger() if sage_ffn_fn is not None else None
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
