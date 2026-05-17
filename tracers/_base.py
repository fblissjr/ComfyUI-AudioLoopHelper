"""Tracer base class + shared utilities.

Every tracer is a `Tracer` subclass with metadata-as-class-attrs and a
small set of lifecycle methods. The orchestrator (`_orchestrator.py`)
drives the lifecycle; tracers don't manage their own atexit / env gates
/ path resolution.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

_AUTO_TOKENS: frozenset[str] = frozenset({"auto", "1", "true", "yes"})

# Sentinel returned by `resolve_path_from_env` when the env value is
# literally "stderr". Consumers (e.g. `ExecLogTracer`) treat this as
# "write to sys.stderr instead of a file."
STDERR_SENTINEL: Path = Path("/dev/stderr")


# --- shared utility: stderr observability --------------------------


def log_event(tracer_name: str, message: str) -> None:
    """Emit one diagnostic line for a tracer lifecycle event.

    Routed to stderr so it interleaves with ComfyUI's own startup logs.
    Cheap (single write); fires at most a few times per render.
    """
    sys.stderr.write(f"[tracers.{tracer_name}] {message}\n")
    sys.stderr.flush()


def _resolve_run_artifact_path(category: str, ext: str) -> Path:
    """Compute the canonical output path for a tracer artifact.

    `scripts/workflow_utils.run_artifact_path` honors RUN_ID +
    AUDIOLOOPHELPER_PER_PROMPT + executing prompt_id. Lazy sys.path
    injection so this module stays cheap on import.
    """
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from workflow_utils import run_artifact_path  # type: ignore
        return run_artifact_path(category, ext)
    except Exception:
        # `workflow_utils` unreachable (package layout change). Keep
        # tracers running with a legacy-style timestamped fallback.
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        return (Path(__file__).resolve().parent.parent
                / "internal" / "analysis" / "runs" / category
                / f"{category}_{ts}.{ext}")


def resolve_path_from_env(env_var: str, category: str, ext: str) -> Path | None:
    """Resolve the output path implied by an env var's value.

    - empty / unset -> None (tracer disabled)
    - value in `_AUTO_TOKENS` -> canonical `data/runs/.../<category>.<ext>`
    - "stderr" -> `STDERR_SENTINEL` (see module top)
    - any other value -> treated as an explicit file path
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return None
    if raw.lower() == "stderr":
        return STDERR_SENTINEL
    if raw.lower() in _AUTO_TOKENS:
        return _resolve_run_artifact_path(category, ext)
    return Path(raw)


# --- shared utility: current prompt id ----------------------------


def get_executing_prompt_id() -> str | None:
    """Read the active ComfyUI prompt_id from the executing contextvar.

    Returns None outside of a render (e.g. during install-at-import or
    at atexit). Lazy import keeps the package import-time cost zero on
    older ComfyUI builds that don't expose the contextvar.
    """
    try:
        from comfy_execution.utils import get_executing_context  # type: ignore
        ctx = get_executing_context()
        if ctx is not None and ctx.prompt_id is not None:
            return str(ctx.prompt_id)
    except Exception:
        pass
    return None


# --- Tracer base class --------------------------------------------


Lifecycle = Literal["process", "render"]


class Tracer:
    """Base class for all tracers.

    Subclasses set the four class-level metadata attrs and override
    whichever lifecycle methods they need. Default implementations are
    no-ops so subclasses only override what's relevant.
    """

    # --- subclass-set metadata ---

    name: ClassVar[str] = "<unnamed>"
    """Short identifier used in stderr logs + manifest keys."""

    env_var: ClassVar[str] = "<unset>"
    """Env var that gates this tracer."""

    lifecycle: ClassVar[Lifecycle] = "render"
    """When to install: `process` (once at import) or `render` (per sage patch)."""

    artifact_category: ClassVar[str] = "<unset>"
    """Used by path resolver: `data/runs/.../<category>.<ext>`."""

    artifact_ext: ClassVar[str] = "jsonl"
    """File extension for the output artifact."""

    # --- shared helpers (don't override) ---

    def is_enabled(self) -> bool:
        return bool(os.environ.get(self.env_var, "").strip())

    def resolve_output_path(self) -> Path | None:
        return resolve_path_from_env(self.env_var, self.artifact_category, self.artifact_ext)

    def log(self, message: str) -> None:
        log_event(self.name, message)

    def _prompt_id_changed(self) -> bool:
        """Detect a prompt boundary crossing since the last check.

        Updates `self._cached_prompt_id` on change. Returns True only
        when the contextvar reports a non-None prompt_id different from
        the cached value. Subclasses call this in their hot paths (per
        hook or per cleanup) and run their rotation side effects on True.

        Needed because ComfyUI caches the sage node's `_patch_impl`
        output when inputs are unchanged across renders —
        `install_at_render` therefore doesn't re-fire per prompt, and
        tracers that bound output_path at install time would attribute
        later renders' events to the first render's prompt_id.
        """
        current = get_executing_prompt_id()
        if current is None or current == getattr(self, "_cached_prompt_id", None):
            return False
        self._cached_prompt_id = current
        return True

    # --- lifecycle methods (override as needed) ---

    def install_at_import(self) -> bool:
        """Called once at package import for lifecycle='process' tracers.

        Return True if the tracer is now active. Return False if it
        declined to install (e.g. env var unset, or upstream API
        signature changed and the monkey-patch failed).
        """
        return False

    def install_at_render(self, model_clone: Any) -> bool:
        """Called from sage._patch_impl for lifecycle='render' tracers.

        `model_clone` is the ComfyUI ModelPatcher about to run the sampler.
        Tracer can extract `diffusion_model` via
        `model_clone.get_model_object("diffusion_model")` if needed.

        Return True if the tracer is now active for this render.
        """
        return False

    def on_cleanup(self) -> None:
        """Called from ON_CLEANUP callback after each model invocation.

        Render-lifecycle tracers should flush buffered data here. This
        is the primary reliability mechanism (atexit is the safety net).
        """
        return

    def on_atexit(self) -> None:
        """Final safety-net flush on process exit. Registered automatically."""
        return

    # --- manifest reporting ---

    def manifest_entry(self) -> dict[str, Any]:
        """Per-tracer block in the per-prompt manifest.

        Default reports enabled+output. Subclasses can add fields like
        event count, file size, dispatched-kernel breakdown, etc.
        """
        return {
            "enabled": self.is_enabled(),
            "output": str(self.resolve_output_path()) if self.is_enabled() else None,
        }
