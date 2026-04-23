"""AudioLoopHelper-native sage attention node.

An MVP alternative to KJNodes' `PathchSageAttentionKJ` with three properties
the KJ node lacks:

1. Try/except fallback to pytorch attention on any sage exception, with
   deduplicated logging so long iteration loops don't spam. Matches the
   pattern in `comfy.ldm.modules.attention.attention_sage`.
2. `CallbacksMP.ON_CLEANUP` cleanup -- override is removed when the model
   is unloaded instead of persisting on the clone forever.
3. Opt-in per-call telemetry to `internal/analysis/runs/sage/sage_*.jsonl`
   gated by the `AUDIOLOOPHELPER_SAGE_TRACE` env var. Zero overhead when
   unset. Designed to answer backlog item 7 (is the override surviving the
   model offload/reload cycle inside the iteration loop?).

See `internal/analysis/sage_attention_analysis.md` for the patch-chain
analysis that motivates this node, and `internal/design/sage_backlog.md`
for the deferred mask-aware / per-block / baselining work.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import orjson
import torch
from typing_extensions import override

try:
    from comfy_api.latest import io
except ImportError:
    # Outside ComfyUI runtime (pytest). See nodes.py for the stub pattern.
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

    io = _IOStub()

# Best-effort imports. When this file is imported outside ComfyUI (pytest),
# these may be unavailable -- we degrade to string constants / no-ops.
try:
    from comfy.patcher_extension import CallbacksMP  # type: ignore
    _ON_CLEANUP = CallbacksMP.ON_CLEANUP
except ImportError:
    _ON_CLEANUP = "on_cleanup"

try:
    from comfy.ldm.modules.attention import attention_pytorch as _PYTORCH_ATTN  # type: ignore
except ImportError:
    _PYTORCH_ATTN = None  # Callers must inject a pytorch_fn in this case.

# scripts/ has timestamped_run_path. conftest.py already adds scripts/ to
# sys.path for tests; at runtime, __init__.py imports exec_logger which
# also handles the path insertion. Re-do it here defensively so this
# module can be imported standalone.
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    from workflow_utils import timestamped_run_path  # type: ignore
except ImportError:
    def timestamped_run_path(subdir: str, prefix: str, ext: str) -> Path:
        # Unreachable in a normal ComfyUI install; defensive fallback.
        return Path.cwd() / f"{prefix}.{ext}"


_TRACE_ENV = "AUDIOLOOPHELPER_SAGE_TRACE"

try:
    from exec_logger import _AUTO_TOKENS  # type: ignore
except ImportError:
    _AUTO_TOKENS = {"auto", "1", "true", "yes"}


# ---------------------------------------------------------------------------
# Arch detection + mode-list construction
# ---------------------------------------------------------------------------

_TRITON_MODE = "sageattn_qk_int8_pv_fp16_triton"
_MASK_AWARE_MODE = "auto_mask_aware"
_DEFAULT_MODE = _MASK_AWARE_MODE

# Mask-aware is listed first after disabled/auto because it is the safe
# default per `internal/design/sage_backlog.md` item 2: sage's CUDA mask
# kernels (fp16_cuda, fp8++) fail at rtol=0.44 on LTX cross-attn, while
# fp16_triton is clean. auto_mask_aware routes around the bug without
# giving up self-attn speed. Always available: fp16_triton is JIT so it
# runs on any arch.
_MODES_DEFAULT = ["disabled", _MASK_AWARE_MODE, "auto", _TRITON_MODE]

_MODES_BY_ARCH: dict[str, list[str]] = {
    "sm80": ["disabled", _MASK_AWARE_MODE, "auto", "sageattn_qk_int8_pv_fp16_cuda", _TRITON_MODE],
    "sm86": ["disabled", _MASK_AWARE_MODE, "auto", "sageattn_qk_int8_pv_fp16_cuda", _TRITON_MODE],
    "sm87": ["disabled", _MASK_AWARE_MODE, "auto", "sageattn_qk_int8_pv_fp16_cuda", _TRITON_MODE],
    "sm89": ["disabled", _MASK_AWARE_MODE, "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp8_cuda++", _TRITON_MODE],
    "sm90": ["disabled", _MASK_AWARE_MODE, "auto", "sageattn_qk_int8_pv_fp8_cuda_sm90", _TRITON_MODE],
    "sm100": ["disabled", _MASK_AWARE_MODE, "auto", "sageattn3", "sageattn3_per_block_mean", _TRITON_MODE],
    "sm120": ["disabled", _MASK_AWARE_MODE, "auto", "sageattn3", "sageattn3_per_block_mean", _TRITON_MODE],
    "sm121": ["disabled", _MASK_AWARE_MODE, "auto", "sageattn3", "sageattn3_per_block_mean", _TRITON_MODE],
}


def _detect_arch() -> str | None:
    """Return `"sm89"`/`"sm90"`/... for the first CUDA device, or None.

    Uses `sageattention.core.get_cuda_arch_versions()` so the reported arch
    matches exactly what sage's dispatch will see.
    """
    try:
        from sageattention.core import get_cuda_arch_versions
    except ImportError:
        return None
    try:
        archs = get_cuda_arch_versions()
    except Exception:
        return None
    if not archs:
        return None
    return archs[0]


def build_mode_list(arch: str | None) -> list[str]:
    """Return the sage modes that can actually run on `arch`. Unknown arch
    still gets disabled/auto/triton because `auto` delegates to sage's own
    dispatch and triton has no arch requirement."""
    return _MODES_BY_ARCH.get(arch or "", _MODES_DEFAULT)


_CACHED_ARCH = _detect_arch()
_MODE_LIST = build_mode_list(_CACHED_ARCH)


# ---------------------------------------------------------------------------
# Fallback logger (dedup'd)
# ---------------------------------------------------------------------------

class SageFallbackLogger:
    """Emits one line per distinct (shape, mode, error-class) tuple.

    Long iteration loops can trip the fallback hundreds of thousands of
    times if a particular shape is unsupported. We want the signal, not
    the spam. Dedup on (shape, mode, type(err)) so every novel failure is
    recorded exactly once.
    """

    def __init__(self, emit: Callable[[str], None] | None = None):
        self._seen: set[tuple] = set()
        self._emit = emit if emit is not None else self._default_emit

    @staticmethod
    def _default_emit(line: str) -> None:
        logging.error(line)

    def log_once(self, shape: tuple, mode: str, err: BaseException) -> None:
        key = (shape, mode, type(err).__name__)
        if key in self._seen:
            return
        self._seen.add(key)
        self._emit(
            f"[AudioLoopHelperSage] sage call failed (shape={shape}, mode={mode}): "
            f"{type(err).__name__}: {err}. Falling back to pytorch attention."
        )


# ---------------------------------------------------------------------------
# Telemetry (JSONL, env-gated)
# ---------------------------------------------------------------------------

def resolve_trace_path() -> Path | None:
    """Return the JSONL path if AUDIOLOOPHELPER_SAGE_TRACE is set, else None.

    Follows the same semantics as `exec_logger._resolve_log_target`:
    - unset / empty -> None (disabled)
    - "auto"/"1"/"true"/"yes" -> timestamped path under internal/analysis/runs/sage/
    - any other value -> treated as an explicit file path
    """
    raw = os.environ.get(_TRACE_ENV, "").strip()
    if not raw:
        return None
    if raw.lower() in _AUTO_TOKENS:
        return timestamped_run_path("sage", "sage", "jsonl")
    return Path(raw)


class SageTracer:
    """Eagerly-opened JSONL writer. No-op when log_path is None.

    Emits one line per attention call plus a summary line on flush.
    Counters and file writes are both short-circuited when disabled so the
    hot-path cost is one attribute check per call.
    """

    def __init__(self, log_path: Path | None):
        self._log_path = log_path
        self._fh = None
        self._total = 0
        self._fallbacks = 0
        self._shapes: set[tuple] = set()
        self._summary_flushed = False
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # buffering=1 is line-buffered. Flush per line so a crash
            # mid-run still leaves a useful trace -- cost is ~1 syscall
            # per attention call, acceptable for the forensic-only path.
            self._fh = open(log_path, "a", buffering=1)

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def emit(
        self,
        *,
        shape: tuple,
        has_mask: bool,
        mode: str,
        fell_back: bool,
        elapsed_us: float,
        iter_idx: int | None = None,
        effective_mode: str | None = None,
    ) -> None:
        if self._fh is None:
            return
        self._total += 1
        if fell_back:
            self._fallbacks += 1
        self._shapes.add(tuple(shape))
        self._fh.write(orjson.dumps({
            "ts": time.time(),
            "iter": iter_idx,
            "shape": list(shape),
            "has_mask": has_mask,
            "mode": mode,
            # effective_mode reflects the kernel that actually dispatched.
            # Equal to mode for non-routing modes; differs on mask-aware
            # calls. Reading the trace without this field hides the
            # "mask-aware stopped being mask-aware" failure mode.
            "effective_mode": effective_mode if effective_mode is not None else mode,
            "fell_back": fell_back,
            "elapsed_us": round(elapsed_us, 2),
        }).decode() + "\n")

    def flush_summary(self) -> None:
        if self._summary_flushed or self._fh is None:
            return
        self._summary_flushed = True
        self._fh.write(orjson.dumps({
            "ts": time.time(),
            "event": "summary",
            "total_calls": self._total,
            "fallback_count": self._fallbacks,
            "distinct_shapes": len(self._shapes),
        }).decode() + "\n")
        self._fh.flush()


# ---------------------------------------------------------------------------
# The sage function factory (mode -> callable)
# ---------------------------------------------------------------------------

def _kernel_auto(q, k, v, *, is_causal, attn_mask, tensor_layout):
    import sageattention as _sa
    return _sa.sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)


def _kernel_fp16_cuda(q, k, v, *, is_causal, attn_mask, tensor_layout):
    import sageattention as _sa
    return _sa.sageattn_qk_int8_pv_fp16_cuda(
        q, k, v, is_causal=is_causal, attn_mask=attn_mask,
        pv_accum_dtype="fp32", tensor_layout=tensor_layout,
    )


def _kernel_fp16_triton(q, k, v, *, is_causal, attn_mask, tensor_layout):
    import sageattention as _sa
    return _sa.sageattn_qk_int8_pv_fp16_triton(
        q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout,
    )


def _kernel_fp8_cuda(q, k, v, *, is_causal, attn_mask, tensor_layout):
    import sageattention as _sa
    return _sa.sageattn_qk_int8_pv_fp8_cuda(
        q, k, v, is_causal=is_causal, attn_mask=attn_mask,
        pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout,
    )


def _kernel_fp8_cuda_pp(q, k, v, *, is_causal, attn_mask, tensor_layout):
    import sageattention as _sa
    return _sa.sageattn_qk_int8_pv_fp8_cuda(
        q, k, v, is_causal=is_causal, attn_mask=attn_mask,
        pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout,
    )


def _kernel_sage3(q, k, v, *, is_causal, attn_mask, tensor_layout, per_block_mean: bool):
    from sageattn3 import sageattn3_blackwell  # type: ignore
    q_, k_, v_ = [x.transpose(1, 2) if tensor_layout == "NHD" else x for x in (q, k, v)]
    out = sageattn3_blackwell(
        q_, k_, v_, is_causal=is_causal, attn_mask=attn_mask,
        per_block_mean=per_block_mean,
    )
    return out.transpose(1, 2) if tensor_layout == "NHD" else out


_SAGE_KERNELS: dict[str, Callable] = {
    "auto": _kernel_auto,
    "sageattn_qk_int8_pv_fp16_cuda": _kernel_fp16_cuda,
    "sageattn_qk_int8_pv_fp16_triton": _kernel_fp16_triton,
    "sageattn_qk_int8_pv_fp8_cuda": _kernel_fp8_cuda,
    "sageattn_qk_int8_pv_fp8_cuda++": _kernel_fp8_cuda_pp,
    "sageattn3": lambda q, k, v, **kw: _kernel_sage3(q, k, v, per_block_mean=False, **kw),
    "sageattn3_per_block_mean": lambda q, k, v, **kw: _kernel_sage3(q, k, v, per_block_mean=True, **kw),
}


def _run_sage_kernel(kernel, q, k, v, heads, mask, skip_reshape, skip_output_reshape):
    """Apply the reshape/mask prep that mirrors
    `comfy.ldm.modules.attention.attention_sage:532-575`, then dispatch to
    `kernel(q, k, v, is_causal=False, attn_mask=mask, tensor_layout=...)`.

    Split out of `_build_sage_fn` so the mask-aware path can pick the
    kernel per-call without duplicating the reshape logic.
    """
    in_dtype = v.dtype
    if q.dtype == torch.float32 or k.dtype == torch.float32 or v.dtype == torch.float32:
        q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)
    if skip_reshape:
        b, _, _, dim_head = q.shape
        tensor_layout = "HND"
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
        tensor_layout = "NHD"
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
    out = kernel(q, k, v, is_causal=False, attn_mask=mask, tensor_layout=tensor_layout).to(in_dtype)
    if tensor_layout == "HND":
        if not skip_output_reshape:
            out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    else:
        if skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(b, -1, heads * dim_head)
    return out


def _build_sage_fn(mode: str) -> Callable:
    """Return a callable matching the signature:

        sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape) -> Tensor

    sageattention is imported lazily inside the kernel functions, so this
    factory is safe to call in environments where sage isn't installed --
    ImportError surfaces only when an actual attention call is made.

    `auto_mask_aware` dispatches per call: masked paths -> fp16_triton
    (sage's CUDA masked kernels are broken at LTX cross-attn shapes per
    `internal/design/sage_backlog.md` item 2), unmasked -> sage auto.
    """
    if mode == _MASK_AWARE_MODE:
        def sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape):
            # Per-call, stateless decision: no closure state to go stale
            # across model offload/reload. The iteration-stamp tracer
            # catches silent disengagement empirically.
            kernel_name = _TRITON_MODE if mask is not None else "auto"
            return _run_sage_kernel(
                _SAGE_KERNELS[kernel_name], q, k, v, heads, mask,
                skip_reshape, skip_output_reshape,
            )
        return torch.compiler.disable()(sage_fn)

    kernel = _SAGE_KERNELS.get(mode)
    if kernel is None:
        raise ValueError(f"unknown sage mode: {mode!r}")

    def sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape):
        return _run_sage_kernel(kernel, q, k, v, heads, mask, skip_reshape, skip_output_reshape)

    # Default to disabling torch.compile around the sage call. The sage
    # fork's torch.compile support is recent and thin; wrap to avoid
    # graph-break surprises. Users who explicitly want compile can
    # override this via a future widget.
    return torch.compiler.disable()(sage_fn)


# ---------------------------------------------------------------------------
# The override factory
# ---------------------------------------------------------------------------

def _effective_mode(mode: str, mask) -> str:
    """Return the kernel name that actually dispatched for this call.

    For non-routing modes this equals `mode`. For `auto_mask_aware`, it
    depends on mask presence: masked -> fp16_triton, unmasked -> auto.
    Mirror `_build_sage_fn`'s routing policy so the tracer's
    `effective_mode` field matches what the kernel function did.
    """
    if mode == _MASK_AWARE_MODE:
        return _TRITON_MODE if mask is not None else "auto"
    return mode


def make_sage_override(
    *,
    sage_fn: Callable,
    pytorch_fn: Callable | None,
    mode: str,
    fallback_on_error: bool,
    tracer: SageTracer,
    logger: SageFallbackLogger,
) -> Callable:
    """Return the callable that ComfyUI invokes via
    `transformer_options["optimized_attention_override"](func, *args, **kwargs)`.

    `func` is the original `optimized_attention` passed by `wrap_attn`; we
    discard it and call `sage_fn` directly. On exception, optionally fall
    back to `pytorch_fn`. `pytorch_fn` should be the unwrapped version
    (e.g. `attention_pytorch.__wrapped__`) to avoid re-entering `wrap_attn`.
    """
    if pytorch_fn is None:
        # Fall back to the real one if available. Tests inject their own.
        if _PYTORCH_ATTN is not None:
            pytorch_fn = getattr(_PYTORCH_ATTN, "__wrapped__", _PYTORCH_ATTN)

    def _iter_from_kwargs(kwargs: dict) -> int | None:
        # Best-effort: honor an explicit iteration stamp if someone put
        # one on transformer_options (backlog item 7 will wire this up);
        # otherwise expose the sampler step instead so traces can still
        # be grouped by sampler progression.
        opts = kwargs.get("transformer_options") or {}
        it = opts.get("iteration")
        if it is not None:
            return int(it)
        step = opts.get("step")
        return int(step) if step is not None else None

    def override(
        func,
        q, k, v, heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        t0 = time.perf_counter() if tracer.enabled else 0.0
        fell_back = False
        try:
            out = sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape)
        except Exception as err:
            if not fallback_on_error or pytorch_fn is None:
                raise
            logger.log_once(tuple(q.shape), mode, err)
            fell_back = True
            out = pytorch_fn(
                q, k, v, heads,
                mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        if tracer.enabled:
            tracer.emit(
                shape=tuple(q.shape), has_mask=mask is not None, mode=mode,
                fell_back=fell_back,
                elapsed_us=(time.perf_counter() - t0) * 1e6,
                iter_idx=_iter_from_kwargs(kwargs),
                effective_mode=_effective_mode(mode, mask),
            )
        return out

    return override


# ---------------------------------------------------------------------------
# The ComfyUI node
# ---------------------------------------------------------------------------

class AudioLoopHelperSageAttention(io.ComfyNode):
    """Arch-aware sage attention patch with pytorch fallback and opt-in telemetry.

    A drop-in replacement for KJNodes' `PathchSageAttentionKJ` with three
    differences: (1) try/except fallback, (2) ON_CLEANUP handler, (3) JSONL
    telemetry when `AUDIOLOOPHELPER_SAGE_TRACE` is set. See
    `internal/analysis/sage_attention_analysis.md`.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioLoopHelperSageAttention",
            display_name="AudioLoopHelper Sage Attention",
            category="AudioLoopHelper/experimental",
            description=(
                "Patch ComfyUI's optimized_attention with sage, with "
                "pytorch fallback and optional per-call telemetry (gated "
                "by AUDIOLOOPHELPER_SAGE_TRACE env). Mode combo is "
                "filtered to what your GPU can actually run."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "mode",
                    options=_MODE_LIST,
                    default=_DEFAULT_MODE,
                    tooltip=(
                        "Sage kernel to use. 'auto_mask_aware' (default) "
                        "routes masked cross-attn to fp16_triton (the one "
                        "kernel that handles LTX cross-attn cleanly) and "
                        "unmasked self-attn to sage auto. 'auto' delegates "
                        "fully to sage's dispatch. 'disabled' is a no-op. "
                        "Other options are explicit kernel choices filtered "
                        "to those your GPU supports."
                    ),
                ),
                io.Boolean.Input(
                    "fallback_on_error",
                    default=True,
                    tooltip=(
                        "If sage raises, fall back to pytorch attention "
                        "for that call instead of crashing. One log line "
                        "is emitted per distinct (shape, mode, error)."
                    ),
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    @override
    def execute(cls, model, mode, fallback_on_error) -> io.NodeOutput:  # type: ignore[override]
        (patched,) = cls._patch_impl(model, mode=mode, fallback_on_error=fallback_on_error)
        return io.NodeOutput(patched)

    @classmethod
    def _patch_impl(cls, model, *, mode: str, fallback_on_error: bool):
        """Testable seam. Returns (patched_model,) regardless of mode.

        Kept separate from execute() so tests can pass a FakeModel without
        needing the v3 io.NodeOutput wrapper in scope.
        """
        if mode == "disabled":
            return (model,)

        model_clone = model.clone()

        tracer = SageTracer(resolve_trace_path())
        logger = SageFallbackLogger()
        sage_fn = _build_sage_fn(mode)
        pytorch_fn = None
        if _PYTORCH_ATTN is not None:
            pytorch_fn = getattr(_PYTORCH_ATTN, "__wrapped__", _PYTORCH_ATTN)

        override_fn = make_sage_override(
            sage_fn=sage_fn,
            pytorch_fn=pytorch_fn,
            mode=mode,
            fallback_on_error=fallback_on_error,
            tracer=tracer,
            logger=logger,
        )

        transformer_options = model_clone.model_options.setdefault("transformer_options", {})
        transformer_options["optimized_attention_override"] = override_fn

        def _cleanup(*_args, **_kwargs):
            opts = model_clone.model_options.get("transformer_options", {})
            if opts.get("optimized_attention_override") is override_fn:
                opts.pop("optimized_attention_override", None)
            tracer.flush_summary()

        model_clone.add_callback(_ON_CLEANUP, _cleanup)
        return (model_clone,)
