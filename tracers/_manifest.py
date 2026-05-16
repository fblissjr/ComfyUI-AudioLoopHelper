"""Per-prompt manifest writer.

After every render (on_cleanup), writes a JSON file under the same
per-prompt directory as the tracer outputs:

    data/runs/${RUN_ID}/${prompt_id}/manifest.json

Lists every tracer that was registered, whether it was enabled, where
its output landed, and any tracer-specific metadata reported via
`Tracer.manifest_entry()`.

Goal: a downstream reader can find every artifact for a given render
from one canonical file, without having to glob the per-prompt dir
and guess which tracers ran on which renders.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ._base import Tracer, get_executing_prompt_id, log_event


# Module-level state: which prompt are we currently tracking? Set by
# `begin_prompt`, cleared by `finalize_prompt`. Single-threaded
# (ComfyUI runs renders single-threaded per process).
_CURRENT_PROMPT_ID: str | None = None
_PROMPT_START_TS: float | None = None


def begin_prompt() -> None:
    """Mark the start of a render. Called from `install_render_tracers`."""
    global _CURRENT_PROMPT_ID, _PROMPT_START_TS
    _CURRENT_PROMPT_ID = get_executing_prompt_id()
    _PROMPT_START_TS = time.time()


def record_install(tracer: Tracer) -> None:
    """Called when a render-lifecycle tracer successfully installs.

    Currently a no-op (manifest is written at finalize time, not
    per-install) but keeps the API symmetrical for future expansion.
    """
    return


def finalize_prompt(tracers: list[Tracer]) -> None:
    """Write the manifest JSON for the current prompt.

    Idempotent: safe to call from both on_cleanup AND on_atexit. If the
    prompt context is gone (cleared between renders) we write a manifest
    keyed on RUN_ID alone (no per-prompt subdir).
    """
    global _CURRENT_PROMPT_ID, _PROMPT_START_TS

    # Resolve manifest output path: same directory as the tracer
    # artifacts. Use whichever enabled tracer's path implies a directory.
    output_dir = _resolve_manifest_dir()
    if output_dir is None:
        return

    payload: dict[str, Any] = {
        "run_id": os.environ.get("RUN_ID", ""),
        "prompt_id": _CURRENT_PROMPT_ID,
        "start_ts": _PROMPT_START_TS,
        "end_ts": time.time(),
        "tracers": {},
    }

    for t in tracers:
        try:
            payload["tracers"][t.name] = t.manifest_entry()
        except Exception as e:
            payload["tracers"][t.name] = {
                "enabled": True,
                "error": f"{type(e).__name__}: {e}",
            }

    manifest_path = output_dir / "manifest.json"
    try:
        import orjson
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        log_event("manifest", f"wrote {manifest_path}")
    except Exception as e:
        log_event("manifest", f"write failed: {type(e).__name__}: {e}")

    # Reset for the next render.
    _CURRENT_PROMPT_ID = None
    _PROMPT_START_TS = None


def _resolve_manifest_dir() -> Path | None:
    """Find the per-prompt output directory by asking any enabled tracer.

    Uses the same path resolver every tracer uses, so the manifest always
    lands next to the artifacts. Returns None if no tracer is enabled
    (nothing to manifest).
    """
    # Import lazily to avoid a circular dependency at module import.
    from ._base import resolve_path_from_env

    # We need ANY enabled tracer's path-shape, so try the canonical
    # render-lifecycle tracers first. We don't read from the orchestrator
    # registry here (would be circular) — instead derive directly from
    # env using a known render-tracer's category.
    for env_var, category, ext in (
        ("AUDIOLOOPHELPER_FFN_ATTN_TRACE", "ffn_attn_breakdown", "jsonl"),
        ("AUDIOLOOPHELPER_TORCH_PROFILE", "torch_profile", "json"),
        ("AUDIOLOOPHELPER_SAGE_TRACE", "sage", "jsonl"),
    ):
        path = resolve_path_from_env(env_var, category, ext)
        if path is not None and str(path) != "/dev/stderr":
            return path.parent
    return None
