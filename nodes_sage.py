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
    from workflow_utils import run_artifact_path  # type: ignore
except ImportError:
    def run_artifact_path(category: str, ext: str) -> Path:
        # Unreachable in a normal ComfyUI install; defensive fallback.
        return Path.cwd() / f"{category}.{ext}"


_TRACE_ENV = "AUDIOLOOPHELPER_SAGE_TRACE"

# Audit-mode toggle: when set, every sage trace emit also captures
# `out_sum` + `out_absmax` + `out_dtype` for the attention output. Forces
# a CUDA sync per call — distorts timing, so only enable for correctness
# audits (cross-stream bit-stability comparison), NOT perf measurements.
_FINGERPRINT_ENV = "AUDIOLOOPHELPER_SAGE_OUTPUT_FINGERPRINT"

try:
    from .tracers._base import _AUTO_TOKENS  # type: ignore
except Exception:
    _AUTO_TOKENS = frozenset({"auto", "1", "true", "yes"})


def _fingerprint_tensor(output: Any) -> dict[str, Any]:
    """Capture sum + absmax + dtype of an output tensor in one CUDA sync.

    Reductions stay in the tensor's native dtype (bf16 sum is fine);
    only the final two scalars get fp32-cast for stable JSON serialization.
    `torch.stack` + single `.tolist()` collapses the two scalar transfers
    into one host sync.
    """
    s = output.sum()
    m = output.abs().max()
    sum_v, max_v = torch.stack([s.float(), m.float()]).tolist()
    return {
        "out_sum": float(sum_v),
        "out_absmax": float(max_v),
        "out_dtype": str(output.dtype),
    }


# Sage-fork's `get_last_dispatched_kernel()` returns the resolved kernel
# name (one of `sageattention.KNOWN_KERNEL_NAMES`) for the most recent
# `sageattn*` call on this thread. Cached at import for hot-path
# minimalism. Defensive: older sageattention installs lack the symbol;
# fall back silently and let the summary script's routing-table mirror
# resolve the trace post-hoc.
#
# Thread-local: read immediately after `sage_fn` returns. The override
# is synchronous (no awaits between sage_fn and the read), so this is
# safe. Reference: sageattention commit 246425d -- shipped 2026-04-25.
try:
    import sageattention as _sa_mod
    _GET_DISPATCHED_KERNEL = getattr(_sa_mod, "get_last_dispatched_kernel", None)
except ImportError:
    _GET_DISPATCHED_KERNEL = None


# ---------------------------------------------------------------------------
# Arch detection + mode-list construction
# ---------------------------------------------------------------------------

_TRITON_MODE = "sageattn_qk_int8_pv_fp16_triton"
_MASK_AWARE_MODE = "auto_mask_aware"
_DEFAULT_MODE = _MASK_AWARE_MODE

# Mask-aware is listed first after disabled/auto because it is the safe
# default per `internal/design/sage_backlog.md` item 2: sage's CUDA
# kernels don't implement masked attention (MaskMode enum is
# {kNone, kCausal}; attn_mask is silently dropped via kwargs). Only
# fp16_triton has a masked path. auto_mask_aware routes masked calls
# there without giving up self-attn speed on the unmasked path. Always
# available: fp16_triton is JIT so it runs on any arch.
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
    - "auto"/"1"/"true"/"yes" -> RUN_ID-keyed path under data/runs/${RUN_ID}/sage.jsonl
      if RUN_ID is set, else legacy timestamped path under
      internal/analysis/runs/sage/. See workflow_utils.run_artifact_path.
    - any other value -> treated as an explicit file path
    """
    raw = os.environ.get(_TRACE_ENV, "").strip()
    if not raw:
        return None
    if raw.lower() in _AUTO_TOKENS:
        return run_artifact_path("sage", "jsonl")
    return Path(raw)


def _detect_arch_tag() -> str | None:
    """Return 'sm<MM>_cuda<MAJ>_<MIN>' (e.g. 'sm89_cuda12_8') for the
    local GPU + CUDA toolkit, or None when torch.cuda is unavailable.
    """
    try:
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        cuda_v = (torch.version.cuda or "").split(".")
        if len(cuda_v) < 2:
            return None
        return f"sm{major}{minor}_cuda{int(cuda_v[0])}_{int(cuda_v[1])}"
    except Exception:
        return None


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
        self._fingerprint = bool(
            log_path is not None and os.environ.get(_FINGERPRINT_ENV, "").strip()
        )
        # Stamped into every emit() so summaries can resolve 'auto' ->
        # kernel without a --arch flag.
        self._arch_tag = _detect_arch_tag() if log_path is not None else None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # buffering=1 is line-buffered. Flush per line so a crash
            # mid-run still leaves a useful trace -- cost is ~1 syscall
            # per attention call, acceptable for the forensic-only path.
            self._fh = open(log_path, "a", buffering=1)
            # Header row: arch + ts so per-prompt joins know the trace's
            # provenance even when per-call rows are filtered out.
            if self._arch_tag is not None:
                self._fh.write(orjson.dumps({
                    "ts": time.time(),
                    "event": "header",
                    "arch": self._arch_tag,
                }).decode() + "\n")

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
        dispatched_kernel: str | None = None,
        prompt_id: str | None = None,
        skipped: bool = False,
        skip_reason: str | None = None,
        output: Any = None,
    ) -> None:
        if self._fh is None:
            return
        self._total += 1
        if fell_back:
            self._fallbacks += 1
        self._shapes.add(tuple(shape))
        record: dict[str, Any] = {
            "ts": time.time(),
            "iter": iter_idx,
            "shape": list(shape),
            "has_mask": has_mask,
            "mode": mode,
            # effective_mode reflects the consumer-side routing decision.
            # Equal to mode for non-routing modes; differs on mask-aware
            # calls. Reading the trace without this field hides the
            # "mask-aware stopped being mask-aware" failure mode.
            "effective_mode": effective_mode if effective_mode is not None else mode,
            "fell_back": fell_back,
            "elapsed_us": round(elapsed_us, 2),
        }
        # Optional fields use the "absent means N/A" contract — keeps
        # trace files compact and lets summary tooling distinguish
        # "unknown" from "no". `dispatched_kernel`-None covers two
        # observationally-identical cases (old sage symbol missing OR
        # no dispatch on this thread yet); the routing-table mirror
        # handles both equivalently.
        if self._arch_tag is not None:
            record["arch"] = self._arch_tag
        if dispatched_kernel is not None:
            record["dispatched_kernel"] = dispatched_kernel
        if prompt_id is not None:
            record["prompt_id"] = prompt_id
        if skipped:
            record["skipped"] = True
            if skip_reason is not None:
                record["skip_reason"] = skip_reason
        if self._fingerprint and output is not None:
            try:
                record.update(_fingerprint_tensor(output))
            except Exception:
                pass
        self._fh.write(orjson.dumps(record).decode() + "\n")

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


def _route_mask_aware(mask) -> str:
    """Single source of truth for `auto_mask_aware`'s routing policy.
    Masked paths -> fp16_triton; unmasked -> sage auto.

    Sage's INT8-QK-FP8/FP16-PV CUDA kernels don't implement mask support
    (MaskMode enum is `{kNone, kCausal}`; `attn_mask` passed to `sageattn()`
    is accepted via **kwargs and silently dropped). Feeding a masked call
    into those kernels yields attention contaminated by padded positions
    -- rtol scales ~1/seq_kv, which is what our LTX-shape sweep measured
    (rtol 0.26-0.94 across seq_kv 32-1024). Only the Triton kernel
    implements masked attention. Full characterization:
    `internal/design/sage_backlog.md` item 2.

    Keeping this function as the one place the policy lives means
    `_build_sage_fn` and `_effective_mode` won't drift if a third routing
    rule is added later.
    """
    return _TRITON_MODE if mask is not None else "auto"


def _build_sage_fn(mode: str) -> Callable:
    """Return a callable matching the signature:

        sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape) -> Tensor

    sageattention is imported lazily inside the kernel functions, so this
    factory is safe to call in environments where sage isn't installed --
    ImportError surfaces only when an actual attention call is made.
    """
    if mode == _MASK_AWARE_MODE:
        def sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape):
            # Per-call, stateless decision: no closure state to go stale
            # across model offload/reload. The iteration-stamp tracer
            # catches silent disengagement empirically.
            return _run_sage_kernel(
                _SAGE_KERNELS[_route_mask_aware(mask)], q, k, v, heads, mask,
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
    Non-routing modes return `mode` unchanged; `auto_mask_aware` delegates
    to `_route_mask_aware` so the tracer's `effective_mode` field matches
    what `_build_sage_fn` actually ran.
    """
    if mode == _MASK_AWARE_MODE:
        return _route_mask_aware(mask)
    return mode


def make_sage_override(
    *,
    sage_fn: Callable,
    pytorch_fn: Callable | None,
    mode: str,
    fallback_on_error: bool,
    tracer: SageTracer,
    logger: SageFallbackLogger,
    skip_under_seq_len: int = 0,
) -> Callable:
    """Return the callable that ComfyUI invokes via
    `transformer_options["optimized_attention_override"](func, *args, **kwargs)`.

    `func` is the original `optimized_attention` passed by `wrap_attn`; we
    discard it and call `sage_fn` directly. On exception, optionally fall
    back to `pytorch_fn`. `pytorch_fn` should be the unwrapped version
    (e.g. `attention_pytorch.__wrapped__`) to avoid re-entering `wrap_attn`.

    `skip_under_seq_len`: when > 0, route calls with `q.shape[1] < threshold`
    directly to `pytorch_fn` instead of `sage_fn`. Sage's int8 quant +
    kernel-launch overhead dominates on short sequences (sage-fork v0.4.1
    bench: ~0.45× torch_flash at seq=497/498). Trace rows on the skip
    path carry `skipped: true` + `skip_reason: "under_seq_len"`.
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

    def _prompt_id_from_kwargs(kwargs: dict) -> str | None:
        # ComfyUI exposes the currently-executing prompt's id via a
        # contextvar in `comfy_execution.utils.get_executing_context()`.
        # Earlier code looked for it on transformer_options — that's not
        # where ComfyUI plants it. Forward the contextvar value so
        # sage-fork's bench can filter rows by run identity rather than
        # a timestamp window. Fallback to transformer_options for any
        # callers that explicitly thread it (none in the current path,
        # but cheap insurance).
        try:
            from comfy_execution.utils import get_executing_context
            ctx = get_executing_context()
            if ctx is not None and ctx.prompt_id is not None:
                return str(ctx.prompt_id)
        except ImportError:
            pass
        opts = kwargs.get("transformer_options") or {}
        pid = opts.get("prompt_id")
        return str(pid) if pid is not None else None

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
        skipped = False

        def _call_pytorch():
            return pytorch_fn(
                q, k, v, heads,
                mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                **kwargs,
            )

        # Consumer-side policy short-circuit: route short-Q calls to
        # pytorch directly. Distinct from `fell_back` (sage's error path) —
        # `skipped` is "we never tried sage at this shape."
        if (
            skip_under_seq_len > 0
            and pytorch_fn is not None
            and q.shape[1] < skip_under_seq_len
        ):
            skipped = True
            out = _call_pytorch()
        else:
            try:
                out = sage_fn(q, k, v, heads, mask, skip_reshape, skip_output_reshape)
            except Exception as err:
                if not fallback_on_error or pytorch_fn is None:
                    raise
                logger.log_once(tuple(q.shape), mode, err)
                fell_back = True
                out = _call_pytorch()
        if tracer.enabled:
            # Thread-local read; safe because ComfyUI runs attention
            # sequentially on the worker thread. Skipped on fallback --
            # the thread-local may hold a stale value from a prior
            # layer's sage call, not this failed one. Also skipped on
            # the consumer-side shortcut (no sage call → no dispatch).
            dispatched = (
                _GET_DISPATCHED_KERNEL()
                if not fell_back and not skipped and _GET_DISPATCHED_KERNEL is not None
                else None
            )
            tracer.emit(
                shape=tuple(q.shape), has_mask=mask is not None, mode=mode,
                fell_back=fell_back,
                elapsed_us=(time.perf_counter() - t0) * 1e6,
                iter_idx=_iter_from_kwargs(kwargs),
                effective_mode=_effective_mode(mode, mask),
                dispatched_kernel=dispatched,
                prompt_id=_prompt_id_from_kwargs(kwargs),
                skipped=skipped,
                skip_reason="under_seq_len" if skipped else None,
                output=out,
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
                "Patches ComfyUI's optimized_attention to route through "
                "SageAttention. Drop-in replacement for KJNodes' "
                "PatchSageAttentionKJ, with three additions: try/except "
                "fallback to pytorch on per-call kernel error, an "
                "ON_CLEANUP handler that resets state, and opt-in JSONL "
                "telemetry (set AUDIOLOOPHELPER_SAGE_TRACE=auto).\n\n"
                "WHICH KERNEL ACTUALLY RUNS depends on which sage package "
                "is installed in your venv. This node calls sage's "
                "dispatcher; the dispatcher picks the kernel. On sm89 + "
                "CUDA>=12.8, the SageAttention-ada fork (github.com/"
                "fblissjr/SageAttention-ada >= v0.5.5) adds a CUDA mask "
                "kernel that the upstream package doesn't have -- masked "
                "LTX cross-attn lands on fp8_cuda++ instead of falling "
                "back to Triton. Upstream sage on the same card uses "
                "Triton for masked calls (slower; per-call working set "
                "larger). The fork is a pip install; you don't need this "
                "specific node to benefit from it -- any sage node that "
                "calls sage's dispatcher does. This node adds the "
                "fallback, ON_CLEANUP, and trace-data ergonomics on top.\n\n"
                "Mode combo is filtered to kernels your GPU can actually "
                "run (queried via sageattention.core.get_cuda_arch_versions)."
            ),
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "mode",
                    options=_MODE_LIST,
                    default=_DEFAULT_MODE,
                    tooltip=(
                        "Which sage kernel to use.\n\n"
                        "'disabled' -- no-op; ComfyUI's default attention "
                        "runs. Useful for A/B comparisons.\n\n"
                        "'auto_mask_aware' (default) -- splits routing: "
                        "masked calls go to fp16_triton (the historically "
                        "safe path for masked attention across all archs); "
                        "unmasked calls go to sage's dispatcher. Picked "
                        "as default before the SageAttention-ada fork "
                        "added a masked CUDA kernel. NOTE: if you have "
                        "the fork installed on sm89 + CUDA>=12.8, prefer "
                        "'auto' -- this mode forces Triton for masked "
                        "calls and skips the faster fp8_cuda++ path.\n\n"
                        "'auto' -- delegates everything to sage's "
                        "dispatcher. On sm89 + CUDA>=12.8 with the fork "
                        "installed, masked calls route to fp8_cuda++ "
                        "(faster, smaller per-call working set than "
                        "Triton). On upstream sage or other archs, the "
                        "dispatcher's masked routing may fall back to "
                        "SDPA or drop the mask depending on version -- "
                        "verify with telemetry.\n\n"
                        "Other entries are explicit kernel selections "
                        "(filtered to those your GPU supports). Use "
                        "these when measuring; not recommended for "
                        "everyday rendering."
                    ),
                ),
                io.Boolean.Input(
                    "fallback_on_error",
                    default=True,
                    tooltip=(
                        "If a sage kernel call raises, catch the error "
                        "and run pytorch's attention for that one call "
                        "instead of crashing the render. One log line is "
                        "emitted per distinct (shape, mode, error) so "
                        "you find out -- it won't spam if the same call "
                        "site keeps failing.\n\n"
                        "Recommended: leave on. The cost when sage "
                        "succeeds (every call) is one try/except wrap. "
                        "Turn off only when you want crashes to surface "
                        "loudly for debugging."
                    ),
                ),
                io.Int.Input(
                    "skip_under_seq_len",
                    default=0,
                    min=0,
                    max=8192,
                    tooltip=(
                        "Route short attention calls to pytorch instead "
                        "of sage. If q.shape[1] is below this threshold, "
                        "the call skips sage and uses pytorch's "
                        "attention. 0 disables the skip (everything goes "
                        "to sage).\n\n"
                        "Why: sage's int8 quantization + kernel-launch "
                        "overhead is a flat per-call cost. On short "
                        "sequences (e.g. text-encoder shapes at "
                        "seq~377-500) that overhead exceeds the matmul "
                        "speedup, and sage runs slower than pytorch. On "
                        "long sequences (LTX video self-attn at "
                        "seq>10000) sage wins decisively.\n\n"
                        "Recommended: 1024. Trace rows for skipped "
                        "calls carry skipped=true and "
                        "skip_reason='under_seq_len' so you can verify "
                        "the threshold is hitting what you expect."
                    ),
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    @override
    def execute(cls, model, mode, fallback_on_error, skip_under_seq_len=0) -> io.NodeOutput:  # type: ignore[override]
        (patched,) = cls._patch_impl(
            model,
            mode=mode,
            fallback_on_error=fallback_on_error,
            skip_under_seq_len=skip_under_seq_len,
        )
        return io.NodeOutput(patched)

    @classmethod
    def _patch_impl(
        cls, model, *, mode: str, fallback_on_error: bool,
        skip_under_seq_len: int = 0,
    ):
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
            skip_under_seq_len=skip_under_seq_len,
        )

        transformer_options = model_clone.model_options.setdefault("transformer_options", {})
        transformer_options["optimized_attention_override"] = override_fn

        # Install all render-lifecycle tracers via the unified
        # orchestrator (ffn_attn forward hooks, torch.profiler aten-op
        # trace, plus any future tracers). Each is env-gated; orchestrator
        # logs lifecycle events to stderr for observability. See
        # `tracers/__init__.py` for the public API + per-tracer details.
        from . import tracers as _tracers
        _tracers.install_render_tracers(model_clone)

        def _cleanup(*_args, **_kwargs):
            opts = model_clone.model_options.get("transformer_options", {})
            if opts.get("optimized_attention_override") is override_fn:
                opts.pop("optimized_attention_override", None)
            tracer.flush_summary()
            # Single call drains every render-lifecycle tracer + writes
            # the per-prompt manifest. Tracers handle their own errors
            # (orchestrator catches + logs); failures don't block render.
            try:
                _tracers.on_cleanup()
            except Exception:
                pass

        model_clone.add_callback(_ON_CLEANUP, _cleanup)
        return (model_clone,)
