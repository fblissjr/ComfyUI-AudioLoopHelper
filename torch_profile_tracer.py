"""Opt-in torch.profiler tracer for aten-op-level timing.

Activates when `AUDIOLOOPHELPER_TORCH_PROFILE` env var is set to a
non-empty value. Wraps the sampler invocation in a
`torch.profiler.profile` context and dumps a Chrome-trace JSON that
`scripts/profile_summary.py` can post-aggregate. Output goes to:

  - `auto`/`1`/`true`/`yes` -> RUN_ID-keyed path under
    `data/runs/${RUN_ID}/${prompt_id}/torch_profile.json`
    (mirrors `ffn_attn_tracer.py` + sage tracer layout)
  - any other value -> treated as an explicit file path

Captures CPU + CUDA activity with `record_shapes=True` (needed to
key per-tensor-shape aggregations like AdaLN broadcast multiplies)
and `with_stack=False` (Python stack capture roughly triples trace
size for the scale-shift-table dominated paths we care about most).

Use case: measuring AdaLN/RoPE/norm/non-Module ops that the
forward-hook-based `ffn_attn_tracer.py` cannot reach. Specifically
the broadcast multiplies driven by `nn.Parameter` scale-shift tables
on `BasicAVTransformerBlock`.

Zero overhead when env var unset (import check + return). When
active, profiler overhead is ~5-15% wall-time depending on op
count + trace size; budget accordingly for one-shot bench renders,
NOT for production loops.

Install pattern: `start_profile()` is called from
`AudioLoopHelperSageAttention._patch_impl` when env var set. On a
new render, the prior profiler is stopped + exported to its file
BEFORE the new one starts (per-render rotation, mirrors
`ffn_attn_tracer._OUTPUT_PATH` rotation).
"""

from __future__ import annotations

import atexit
import os
import sys
import time
from pathlib import Path
from typing import Any

# Lazy imports inside functions so module-import is free when env unset.

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
try:
    from workflow_utils import run_artifact_path  # type: ignore
except Exception:
    def run_artifact_path(category: str, ext: str) -> Path:
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        return Path(__file__).resolve().parent / "internal" / "analysis" / "runs" / "torch_profile" / f"torch_profile_{ts}.{ext}"

try:
    from exec_logger import _AUTO_TOKENS  # type: ignore
except Exception:
    _AUTO_TOKENS = frozenset({"auto", "1", "true", "yes"})

_TRACE_ENV = "AUDIOLOOPHELPER_TORCH_PROFILE"

# Module-level state. ComfyUI runs renders single-threaded per process.
_ACTIVE_PROFILER: Any = None
_ACTIVE_PATH: Path | None = None
_ATEXIT_REGISTERED = False


def is_enabled() -> bool:
    """Cheap check: should we start the profiler at all?"""
    return bool(os.environ.get(_TRACE_ENV, "").strip())


def _resolve_path() -> Path | None:
    """Mirror sage tracer's path resolution semantics."""
    raw = os.environ.get(_TRACE_ENV, "").strip()
    if not raw:
        return None
    if raw.lower() in _AUTO_TOKENS:
        return run_artifact_path("torch_profile", "json")
    return Path(raw)


def _export_active() -> None:
    """Stop the active profiler and dump its chrome trace."""
    global _ACTIVE_PROFILER, _ACTIVE_PATH
    if _ACTIVE_PROFILER is None or _ACTIVE_PATH is None:
        return
    try:
        _ACTIVE_PROFILER.stop()
        _ACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_PROFILER.export_chrome_trace(str(_ACTIVE_PATH))
    except Exception:
        # Defensive: profile export must not block rendering or unwind.
        pass
    finally:
        _ACTIVE_PROFILER = None
        _ACTIVE_PATH = None


def start_profile() -> bool:
    """Start a new profile, rotating any prior active profile to disk.

    Idempotent within a single render; ComfyUI re-patches the model
    when needed, but the per-prompt path resolution gives us a stable
    target file across re-patches within the same render.

    Returns True if the profiler started.
    """
    global _ACTIVE_PROFILER, _ACTIVE_PATH, _ATEXIT_REGISTERED
    if not is_enabled():
        return False

    new_path = _resolve_path()
    if new_path is None:
        return False

    # Re-patches within the same render hit the same target path; skip
    # restart so we don't truncate an in-flight capture.
    if _ACTIVE_PROFILER is not None and _ACTIVE_PATH == new_path:
        return True

    # New render: drain prior profile to its file before starting new.
    if _ACTIVE_PROFILER is not None:
        _export_active()

    try:
        import torch.profiler
    except ImportError:
        return False

    try:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
        )
        prof.start()
    except Exception:
        return False

    _ACTIVE_PROFILER = prof
    _ACTIVE_PATH = new_path

    if not _ATEXIT_REGISTERED:
        atexit.register(_export_active)
        _ATEXIT_REGISTERED = True

    return True


def maybe_export() -> None:
    """Flush the active profile to disk if env enabled.

    Called from ON_CLEANUP. Conservative: we DON'T export here by
    default because ON_CLEANUP fires after every model invocation,
    not at render end (see CLAUDE.md gotcha). For two-stage workflows
    that would truncate the stage-2 capture mid-render. The atexit
    handler does the final export; this hook exists for parity with
    `ffn_attn_tracer.maybe_flush()` so future callers can switch to
    explicit per-stage export if/when needed.
    """
    # Intentional no-op for now; atexit handles export.
    return
