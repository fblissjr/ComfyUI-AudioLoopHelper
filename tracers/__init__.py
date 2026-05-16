"""Unified tracer framework for ComfyUI-AudioLoopHelper.

Replaces the standalone modules `ffn_attn_tracer.py`, `torch_profile_tracer.py`,
and `exec_logger.py` (each grew its own copy of env-var gating + path
resolution + flush lifecycle). The framework gives every tracer:

- shared env-var grammar (`auto` / `1` / `true` / `yes` -> default path)
- shared path resolution (per-run + per-prompt under `data/runs/${RUN_ID}/`)
- explicit lifecycle (install / on_cleanup / on_atexit)
- stderr observability (one log line per lifecycle event)
- per-prompt manifest (records which tracers fired + where the output landed)

## Lifecycle classes

Tracers declare one of two lifecycles:

- **process**: install once at package import. Used for tracers that
  monkey-patch ComfyUI internals (e.g. `exec_log` hooks ComfyUI's executor
  on import; the same hook then fires for every render in this process).

- **render**: install on every `AudioLoopHelperSageAttention._patch_impl`
  invocation. Used for tracers that need to attach to per-render model
  state (e.g. `ffn_attn` hooks `BasicAVTransformerBlock` sub-modules on a
  model_clone; `torch_profile` opens a fresh chrome trace context).

## Public API

- `install_process_tracers()` — call once from package `__init__.py`
- `install_render_tracers(model_clone)` — call from `nodes_sage._patch_impl`
- `on_cleanup()` — call from the sage node's ON_CLEANUP callback
- (atexit is registered automatically on first install)

## Adding a new tracer

1. Subclass `tracers._base.Tracer`. Set `name`, `env_var`, `lifecycle`,
   `artifact_category`, `artifact_ext` on the class.
2. Implement the lifecycle methods you need (`install_at_import` for
   process-lifecycle, `install_at_render` + optional `on_cleanup` for
   render-lifecycle).
3. Register the tracer by importing it from `tracers/__init__.py` below
   and calling `_orchestrator.register(MyTracer())` in `_REGISTRY_INIT`.

That's it. The orchestrator handles env-var gating, path resolution,
manifest updates, and atexit safety.
"""

from __future__ import annotations

from . import _orchestrator
from .exec_log import ExecLogTracer
from .ffn_attn import FfnAttnTracer
from .torch_profile import TorchProfileTracer


def _REGISTRY_INIT() -> None:
    """Register all built-in tracers. Called once on package import."""
    _orchestrator.register(ExecLogTracer())
    _orchestrator.register(FfnAttnTracer())
    _orchestrator.register(TorchProfileTracer())


_REGISTRY_INIT()


def install_process_tracers() -> None:
    """Install all `lifecycle='process'` tracers. Call from package __init__."""
    _orchestrator.install_process_tracers()


def install_render_tracers(model_clone) -> None:
    """Install all `lifecycle='render'` tracers. Call from sage node patch impl."""
    _orchestrator.install_render_tracers(model_clone)


def on_cleanup() -> None:
    """Notify render-lifecycle tracers that a model invocation just ended.

    Each tracer flushes whatever it has buffered. Per-cleanup flush is the
    primary reliability mechanism; atexit is the safety net.
    """
    _orchestrator.on_cleanup()


__all__ = [
    "install_process_tracers",
    "install_render_tracers",
    "on_cleanup",
]
