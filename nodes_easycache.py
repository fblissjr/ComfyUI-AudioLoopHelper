"""LTX-2.3 EasyCache: training-free step-skipping cache for video DiTs.

Algorithm sourced from "Less is Enough: Training-Free Video Diffusion
Acceleration via Runtime-Adaptive Caching" (arxiv 2507.02860). Hooks via
`comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL` -- the supported
wrapper API in ComfyUI core, not a monkey patch.

State machine: window-gated by [start_step, end_step]. First in-window
call seeds (compute, stash input/output_norm/residual). Subsequent calls
accumulate `mean|x - prev_in| / prev_out_norm` per step. Below threshold,
skip and return `x + cache_residual`. Above, compute, refresh state, reset
accumulator. Negative threshold disables caching (strict-never-skip
sentinel since accumulated_error is always >= 0).
"""

from __future__ import annotations

from typing import Callable

import torch

try:
    from comfy_api.latest import io
    from typing_extensions import override
except ImportError:
    # Absolute (not relative) so this works whether the plugin is loaded
    # as a package by ComfyUI or as a top-level module by pytest. The
    # pytest rootdir puts the plugin root on sys.path either way.
    from _comfy_stubs import io_stub as io, override_stub as override  # type: ignore[assignment,no-redef]

try:
    from comfy.patcher_extension import CallbacksMP, WrappersMP  # type: ignore
    _DIFFUSION_MODEL = WrappersMP.DIFFUSION_MODEL
    _ON_CLEANUP = CallbacksMP.ON_CLEANUP
except ImportError:
    from _comfy_stubs import _stub_constants  # type: ignore[no-redef]
    _DIFFUSION_MODEL, _ON_CLEANUP = _stub_constants()


# Single key for the wrapper registration. add_wrapper_with_key uses this as
# the dict key inside ModelPatcher.wrappers; reusing the same key on a
# re-patch overwrites cleanly.
WRAPPER_KEY = "ltxv_easycache"


class EasyCacheState:
    """Per-patched-model cache state. Lives in a closure over the wrapper.

    Holds two tensors (`previous_raw_input` for the drift check, and
    `cache_residual` for skip-path reconstruction) plus a precomputed
    scalar `prev_out_norm`. The previous output tensor itself is not
    retained -- only its `.abs().mean()` value, which is the only thing
    the comparison needs.

    `cache_device` controls where the retained tensors live. None means
    "wherever the input arrived" (typically GPU). Setting it to "cpu"
    offloads the cache so it doesn't compete with the model for VRAM --
    pays a one-time HtoD copy on each cache hit (read path) but frees
    several latent-shape tensors on the GPU.
    """

    __slots__ = (
        "thresh", "start_step", "end_step", "cache_device",
        "step_idx",
        "previous_raw_input", "cache_residual", "prev_out_norm",
        "accumulated_error", "skipped_steps",
    )

    def __init__(
        self,
        thresh: float,
        start_step: int,
        end_step: int,
        cache_device: str | torch.device | None = None,
    ):
        self.thresh = thresh
        self.start_step = start_step
        self.end_step = end_step  # -1 sentinel for "no upper bound"
        self.cache_device = (
            torch.device(cache_device) if isinstance(cache_device, str) else cache_device
        )
        self.step_idx = 0
        self.previous_raw_input: torch.Tensor | None = None
        self.cache_residual: torch.Tensor | None = None
        self.prev_out_norm: torch.Tensor | None = None
        self.accumulated_error: float = 0.0
        self.skipped_steps: list[int] = []

    def reset(self) -> None:
        self.step_idx = 0
        self.previous_raw_input = None
        self.cache_residual = None
        self.prev_out_norm = None
        self.accumulated_error = 0.0
        self.skipped_steps = []


def _input_tensor(args: tuple, kwargs: dict) -> torch.Tensor:
    """Pull the diffusion-model input tensor from a wrapper invocation.

    The wrapped function is `LTXBaseModel._forward(x, timestep, context, ...)`.
    `x` is positional in real calls; in tests we may pass it as the first
    positional or keyword. Cover both paths defensively without changing
    the wrapper's interface contract.
    """
    if args:
        return args[0]
    return kwargs["x"]


def build_wrapper(state: EasyCacheState) -> Callable:
    """Construct the WrappersMP.DIFFUSION_MODEL-shaped wrapper.

    Signature follows comfy.patcher_extension.WrapperExecutor: the first
    positional is the executor, the rest are forwarded to the next
    wrapper or to the original `_forward`.
    """

    def _seed(out: torch.Tensor, x: torch.Tensor) -> None:
        # Clamp the precomputed norm to avoid divide-by-zero on degenerate
        # outputs; 1e-8 is below any plausible bf16/fp16 nonzero scale so
        # it only kicks in on identically-zero outputs.
        prev_in = x.detach()
        residual = (out - x).detach()
        norm = out.detach().abs().mean().clamp(min=1e-8)
        if state.cache_device is not None:
            prev_in = prev_in.to(state.cache_device)
            residual = residual.to(state.cache_device)
            norm = norm.to(state.cache_device)
        state.previous_raw_input = prev_in
        state.cache_residual = residual
        state.prev_out_norm = norm
        state.accumulated_error = 0.0

    def _wrapper(executor, *args, **kwargs):
        x = _input_tensor(args, kwargs)
        step = state.step_idx
        state.step_idx += 1

        in_window = (
            step >= state.start_step
            and (state.end_step < 0 or step <= state.end_step)
        )
        if not in_window:
            return executor(*args, **kwargs)

        if state.previous_raw_input is None or state.prev_out_norm is None:
            out = executor(*args, **kwargs)
            _seed(out, x)
            return out

        prev_in = state.previous_raw_input.to(x.device)
        prev_norm = state.prev_out_norm.to(x.device)
        change_metric = ((x - prev_in).abs().mean() / prev_norm).item()
        state.accumulated_error += change_metric

        if state.accumulated_error < state.thresh:
            state.skipped_steps.append(step)
            return x + state.cache_residual.to(x.device)  # type: ignore[union-attr]

        out = executor(*args, **kwargs)
        _seed(out, x)
        return out

    return _wrapper


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------


class LTXVideoEasyCache(io.ComfyNode):
    """Patches an LTX MODEL with EasyCache step-skipping.

    Connect between the LTX checkpoint loader and KSampler. Single threshold
    knob (start with 0.015, sweep up for more aggressive caching). Negative
    threshold disables caching (the strict-never-skip sentinel). Output is
    the patched model -- pass it forward as you would any other patched
    MODEL.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXVideoEasyCache",
            display_name="LTX Video EasyCache",
            category="AudioLoopHelper/experimental",
            description=(
                "Training-free step-skipping cache for LTX-2.3. Reuses prior "
                "step output's residual when input drift stays below the "
                "threshold. Single tunable knob. Use sweep "
                "{0.015, 0.02, 0.03, 0.05} as a starting range; higher = "
                "faster but lower fidelity. Set threshold negative to "
                "disable without removing the node."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input(
                    "easycache_thresh",
                    default=0.015,
                    min=-1.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "Cumulative input-drift tolerance before forcing a "
                        "fresh compute. Wan ports use 0.015 default. "
                        "Negative = caching disabled."
                    ),
                ),
                io.Int.Input(
                    "start_step",
                    default=10,
                    min=0,
                    max=9999,
                    step=1,
                    tooltip=(
                        "Steps before this index always compute (no caching, "
                        "no state update). Lets the model establish structure "
                        "before caching kicks in."
                    ),
                ),
                io.Int.Input(
                    "end_step",
                    default=-1,
                    min=-1,
                    max=9999,
                    step=1,
                    tooltip="-1 = no upper bound. Otherwise: caching disabled past this step.",
                ),
                io.Combo.Input(
                    "cache_device",
                    options=["main", "cpu"],
                    default="main",
                    tooltip=(
                        "Where to keep the retained cache tensors. 'main' "
                        "leaves them on whichever device the model runs on "
                        "(typically GPU). 'cpu' offloads them to host RAM "
                        "to free VRAM, paying a one-time HtoD copy on each "
                        "cache hit. Pick 'cpu' if VRAM is tight."
                    ),
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    @override
    def execute(cls, model, easycache_thresh, start_step, end_step, cache_device) -> io.NodeOutput:  # type: ignore[override]
        (patched,) = cls._patch_impl(
            model,
            easycache_thresh=easycache_thresh,
            start_step=start_step,
            end_step=end_step,
            cache_device=cache_device,
        )
        return io.NodeOutput(patched)

    @classmethod
    def _patch_impl(
        cls,
        model,
        *,
        easycache_thresh: float,
        start_step: int,
        end_step: int,
        cache_device: str | torch.device | None = None,
    ):
        """Testable seam. Wraps the model with the EasyCache state machine
        and returns the patched clone. Same shape as the sage node's
        _patch_impl so tests can use FakeModelWithWrappers.

        cache_device accepts the widget strings "main" / "cpu" or a real
        torch.device / device-string. "main" -> None (stay on input device)."""
        if cache_device == "main":
            cache_device = None
        clone = model.clone()
        state = EasyCacheState(
            thresh=easycache_thresh,
            start_step=start_step,
            end_step=end_step,
            cache_device=cache_device,
        )
        wrapper = build_wrapper(state)
        clone.add_wrapper_with_key(_DIFFUSION_MODEL, WRAPPER_KEY, wrapper)

        def _cleanup(*_args, **_kwargs):
            state.reset()

        # Use add_callback (with a string key it matches the sage node
        # pattern). The fake model in tests stores by call_type only, which
        # is sufficient for the cleanup-exists test.
        if hasattr(clone, "add_callback_with_key"):
            clone.add_callback_with_key(_ON_CLEANUP, WRAPPER_KEY, _cleanup)
        else:
            clone.add_callback(_ON_CLEANUP, _cleanup)

        return (clone,)
