"""LTX-2.3 EasyCache: training-free step-skipping cache for video DiTs.

Algorithm sourced from "Less is Enough: Training-Free Video Diffusion
Acceleration via Runtime-Adaptive Caching" (arxiv 2507.02860). Reference
implementation pattern lifted from a sibling Wan-video custom node which
exposes the same algorithm as a transformer-block-forward patch. The shape
of the algorithm is identical -- only the integration mechanism differs.

Why this approach for LTX:

- LTX's diffusion model in ComfyUI core uses
  `comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL` to thread external
  wrappers around its `_forward`. Registering a wrapper via
  `model.add_wrapper_with_key(...)` is the supported, non-monkey-patching
  hook. Resilient to ComfyUI core upgrades.
- TeaCache and MagCache need per-model calibration constants we don't have
  for LTX-2.3. EasyCache is runtime-adaptive: a single threshold knob, no
  calibration runs.
- The cache decision is at the step level (skip the entire denoiser forward
  pass for this step, return previous output's residual added to current
  input). Orthogonal to attention-kernel choice -- sage still runs inside
  any step that does compute.

State machine:

1. Window check: if step_idx outside [start_step, end_step], bypass entirely
   (always compute, no state update). Lets early steps escape caching while
   the model is still establishing structure.
2. Seeding: if no history, compute and stash (input, output, residual=output-input).
3. Comparison: input_change_ratio = mean|x - prev_in| / mean|prev_out|. Add
   to accumulated_error.
4. Decision: if accumulated_error < threshold, return x + cached_residual
   (skip). Else compute, refresh state, reset accumulator to 0.

The threshold value is the cumulative tolerance for input drift before a
fresh compute is forced. Larger threshold -> more skips -> faster but lower
quality. Negative threshold disables caching entirely (the strict-never-skip
sentinel since accumulated_error is always >= 0).
"""

from __future__ import annotations

import logging
from typing import Callable

import torch

try:
    from comfy_api.latest import io
    from typing_extensions import override
except ImportError:
    # Outside ComfyUI runtime (pytest). Mirror the pattern in nodes_sage.py
    # so this module is importable for unit tests.
    class _Passthrough:
        def __getattr__(self, _name):
            return _Passthrough()

        def __call__(self, *args, **kwargs):
            return _Passthrough()

    class _IOStub(_Passthrough):
        class ComfyNode:
            pass

        @staticmethod
        def NodeOutput(*args):
            return args

    io = _IOStub()  # type: ignore[assignment]

    def override(fn):  # type: ignore[no-redef]
        return fn

try:
    from comfy.patcher_extension import CallbacksMP, WrappersMP  # type: ignore
    _DIFFUSION_MODEL = WrappersMP.DIFFUSION_MODEL
    _ON_CLEANUP = CallbacksMP.ON_CLEANUP
except ImportError:
    # String literals match the runtime constants in
    # comfy.patcher_extension. Tests rely on this so they don't need a
    # ComfyUI runtime.
    _DIFFUSION_MODEL = "diffusion_model"
    _ON_CLEANUP = "on_cleanup"


_log = logging.getLogger(__name__)

# Single key for the wrapper registration. add_wrapper_with_key uses this as
# the dict key inside ModelPatcher.wrappers; reusing the same key on a
# re-patch overwrites cleanly.
WRAPPER_KEY = "ltxv_easycache"


class EasyCacheState:
    """Per-patched-model cache state. Lives in a closure over the wrapper.

    `previous_raw_input` and `previous_raw_output` are kept on whichever
    device they arrived on; the wrapper handles cross-device move on
    comparison. `cache_residual` is the most recently computed
    (output - input) tensor used for cache-hit reconstruction.
    """

    __slots__ = (
        "thresh", "start_step", "end_step",
        "step_idx",
        "previous_raw_input", "previous_raw_output", "cache_residual",
        "accumulated_error", "skipped_steps",
    )

    def __init__(self, thresh: float, start_step: int, end_step: int):
        self.thresh = thresh
        self.start_step = start_step
        self.end_step = end_step  # -1 sentinel for "no upper bound"
        self.step_idx = 0
        self.previous_raw_input: torch.Tensor | None = None
        self.previous_raw_output: torch.Tensor | None = None
        self.cache_residual: torch.Tensor | None = None
        self.accumulated_error: float = 0.0
        self.skipped_steps: list[int] = []

    def reset(self) -> None:
        self.step_idx = 0
        self.previous_raw_input = None
        self.previous_raw_output = None
        self.cache_residual = None
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

        if state.previous_raw_input is None or state.previous_raw_output is None:
            out = executor(*args, **kwargs)
            state.previous_raw_input = x.detach()
            state.previous_raw_output = out.detach()
            state.cache_residual = (out - x).detach()
            state.accumulated_error = 0.0
            return out

        prev_in = state.previous_raw_input.to(x.device)
        prev_out = state.previous_raw_output.to(x.device)
        # Clamp output_norm to avoid divide-by-zero on degenerate inputs;
        # 1e-8 is below any plausible bf16/fp16 nonzero scale so it only
        # kicks in on identically-zero outputs.
        output_norm = prev_out.abs().mean().clamp(min=1e-8)
        change_metric = ((x - prev_in).abs().mean() / output_norm).item()
        state.accumulated_error += change_metric

        if state.accumulated_error < state.thresh:
            state.skipped_steps.append(step)
            return x + state.cache_residual.to(x.device)  # type: ignore[union-attr]

        out = executor(*args, **kwargs)
        state.previous_raw_input = x.detach()
        state.previous_raw_output = out.detach()
        state.cache_residual = (out - x).detach()
        state.accumulated_error = 0.0
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
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    @override
    def execute(cls, model, easycache_thresh, start_step, end_step) -> io.NodeOutput:  # type: ignore[override]
        (patched,) = cls._patch_impl(
            model,
            easycache_thresh=easycache_thresh,
            start_step=start_step,
            end_step=end_step,
        )
        return io.NodeOutput(patched)

    @classmethod
    def _patch_impl(cls, model, *, easycache_thresh: float, start_step: int, end_step: int):
        """Testable seam. Wraps the model with the EasyCache state machine
        and returns the patched clone. Same shape as the sage node's
        _patch_impl so tests can use FakeModelWithWrappers."""
        clone = model.clone()
        state = EasyCacheState(
            thresh=easycache_thresh,
            start_step=start_step,
            end_step=end_step,
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
