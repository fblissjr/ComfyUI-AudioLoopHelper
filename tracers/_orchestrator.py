"""Orchestrator: registry + lifecycle dispatch + atexit safety net.

Single entry point that consumers (`__init__.py`, `nodes_sage.py`)
call. Maintains a registry of `Tracer` instances, dispatches lifecycle
events, and ensures atexit fires once per process.
"""

from __future__ import annotations

import atexit
import time
from typing import Any

from . import _manifest
from ._base import Tracer, get_executing_prompt_id, log_event


_REGISTRY: list[Tracer] = []
_ATEXIT_REGISTERED: bool = False

# Bridge state for the current prompt. Set on `install_render_tracers`;
# consumed on `on_cleanup` / atexit. Cleared after each finalize so the
# next render starts fresh.
_PROMPT_ID: str | None = None
_PROMPT_START_TS: float | None = None
_ANY_RENDER_TRACER_ACTIVE: bool = False


def register(tracer: Tracer) -> None:
    """Register a tracer. Idempotent by `tracer.name` so module reloads
    (HotReloadHack) don't double-register and double-fire lifecycle events.
    """
    for existing in _REGISTRY:
        if existing.name == tracer.name:
            return
    _REGISTRY.append(tracer)


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
    global _PROMPT_ID, _PROMPT_START_TS, _ANY_RENDER_TRACER_ACTIVE
    _register_atexit_once()
    _PROMPT_ID = get_executing_prompt_id()
    _PROMPT_START_TS = time.time()
    _ANY_RENDER_TRACER_ACTIVE = False
    for t in _REGISTRY:
        if t.lifecycle != "render":
            continue
        if not t.is_enabled():
            continue
        try:
            ok = t.install_at_render(model_clone)
            if ok:
                t.log(f"render install -> {t.resolve_output_path()}")
                _ANY_RENDER_TRACER_ACTIVE = True
            else:
                t.log("install_at_render declined (returned False)")
        except Exception as e:
            t.log(f"install_at_render raised {type(e).__name__}: {e}")


def _refresh_prompt_state_if_needed() -> bool:
    """Re-read the executing prompt_id and refresh `_PROMPT_ID` if changed.

    Necessary because ComfyUI caches the sage node's `_patch_impl` output
    when inputs are unchanged across renders, so `install_render_tracers`
    may not re-fire per prompt. Without this, the manifest written at
    on_cleanup carries the prior render's prompt_id even though the
    cleanup itself fired for a new render. Returns True if the state was
    rotated.
    """
    global _PROMPT_ID, _PROMPT_START_TS
    current = get_executing_prompt_id()
    if current is None or current == _PROMPT_ID:
        return False
    _PROMPT_ID = current
    _PROMPT_START_TS = time.time()
    return True


def on_cleanup() -> None:
    """Fire ON_CLEANUP for every render-lifecycle tracer that's enabled.

    Manifest gets re-written here. State is NOT reset because ON_CLEANUP
    fires once per sampler invocation, not per render — a multi-sampler
    workflow (FML2V two-stage, audio-loop N-iteration) needs each cleanup
    to refresh the manifest with newly-flushed artifacts. State is reset
    by `install_render_tracers` at the start of the next render OR by
    `_refresh_prompt_state_if_needed` when the prompt_id changes
    underneath us (the ComfyUI-caches-sage-node case).
    """
    _refresh_prompt_state_if_needed()
    for t in _REGISTRY:
        if t.lifecycle != "render":
            continue
        if not t.is_enabled():
            continue
        try:
            t.on_cleanup()
        except Exception as e:
            t.log(f"on_cleanup raised {type(e).__name__}: {e}")
    if _ANY_RENDER_TRACER_ACTIVE:
        _manifest.finalize_prompt(_REGISTRY, _PROMPT_ID, _PROMPT_START_TS)


def _on_atexit() -> None:
    """Final flush for every enabled tracer. Idempotent."""
    for t in _REGISTRY:
        if not t.is_enabled():
            continue
        try:
            t.on_atexit()
        except Exception as e:
            t.log(f"on_atexit raised {type(e).__name__}: {e}")
    if _ANY_RENDER_TRACER_ACTIVE:
        _manifest.finalize_prompt(_REGISTRY, _PROMPT_ID, _PROMPT_START_TS)


def _register_atexit_once() -> None:
    """Register the global atexit handler exactly once per process."""
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(_on_atexit)
    _ATEXIT_REGISTERED = True
    log_event("orchestrator", "atexit handler registered")


def registered_tracers() -> list[Tracer]:
    """Introspection used by tests + ad-hoc debug."""
    return list(_REGISTRY)
