"""Orchestrator: registry + lifecycle dispatch + atexit safety net.

The orchestrator is the single entry point that consumers
(`__init__.py`, `nodes_sage.py`) call. It maintains a registry of
`Tracer` instances, dispatches lifecycle events to each, and ensures
atexit fires once per process.
"""

from __future__ import annotations

import atexit
from typing import Any

from . import _manifest
from ._base import Tracer, log_event


_REGISTRY: list[Tracer] = []
_ATEXIT_REGISTERED: bool = False


def register(tracer: Tracer) -> None:
    """Register a tracer instance. Idempotent by `tracer.name`.

    Called from `tracers/__init__.py`. Idempotent so module reloads
    (HotReloadHack) don't double-register and double-fire lifecycle
    events.
    """
    for existing in _REGISTRY:
        if existing.name == tracer.name:
            return
    _REGISTRY.append(tracer)


# --- lifecycle dispatchers ---


def install_process_tracers() -> None:
    """Install all `lifecycle='process'` tracers. Called from package import."""
    _register_atexit_once()
    for t in _REGISTRY:
        if t.lifecycle != "process":
            continue
        if not t.is_enabled():
            continue
        try:
            ok = t.install_at_import()
            if ok:
                t.log(f"installed -> {t.resolve_output_path()}")
            else:
                t.log("install_at_import declined (returned False)")
        except Exception as e:
            t.log(f"install_at_import raised {type(e).__name__}: {e}")


def install_render_tracers(model_clone: Any) -> None:
    """Install all `lifecycle='render'` tracers. Called per sage._patch_impl."""
    _register_atexit_once()
    _manifest.begin_prompt()
    for t in _REGISTRY:
        if t.lifecycle != "render":
            continue
        if not t.is_enabled():
            continue
        try:
            ok = t.install_at_render(model_clone)
            if ok:
                t.log(f"render install -> {t.resolve_output_path()}")
                _manifest.record_install(t)
            else:
                t.log("install_at_render declined (returned False)")
        except Exception as e:
            t.log(f"install_at_render raised {type(e).__name__}: {e}")


def on_cleanup() -> None:
    """Fire ON_CLEANUP for every render-lifecycle tracer that's enabled."""
    for t in _REGISTRY:
        if t.lifecycle != "render":
            continue
        if not t.is_enabled():
            continue
        try:
            t.on_cleanup()
        except Exception as e:
            t.log(f"on_cleanup raised {type(e).__name__}: {e}")
    _manifest.finalize_prompt(_REGISTRY)


def _on_atexit() -> None:
    """Final flush for every enabled tracer. Idempotent."""
    for t in _REGISTRY:
        if not t.is_enabled():
            continue
        try:
            t.on_atexit()
        except Exception as e:
            t.log(f"on_atexit raised {type(e).__name__}: {e}")
    _manifest.finalize_prompt(_REGISTRY)


def _register_atexit_once() -> None:
    """Register the global atexit handler exactly once per process."""
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(_on_atexit)
    _ATEXIT_REGISTERED = True
    log_event("orchestrator", "atexit handler registered")


# --- introspection (used by tests + ad-hoc debug) ---


def registered_tracers() -> list[Tracer]:
    return list(_REGISTRY)
