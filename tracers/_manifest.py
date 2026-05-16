"""Per-prompt manifest writer.

After every render (`on_cleanup`), writes
`data/runs/${RUN_ID}/${prompt_id}/manifest.json` listing every tracer
that was registered, whether it was enabled, where its output landed,
and any tracer-specific metadata reported via `Tracer.manifest_entry()`.

Goal: a downstream reader can find every artifact for a given render
from one canonical file, without globbing the per-prompt dir and
guessing which tracers ran.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ._base import STDERR_SENTINEL, Tracer, log_event


def finalize_prompt(
    tracers: list[Tracer],
    prompt_id: str | None,
    start_ts: float | None,
) -> None:
    """Write the manifest JSON for the prompt that just finished.

    Idempotent — safe to call from both `on_cleanup` and the atexit
    safety net. Returns silently if no enabled tracer's output path
    implies a directory (i.e. nothing was captured).
    """
    output_dir = _resolve_manifest_dir(tracers)
    if output_dir is None:
        return

    payload: dict[str, Any] = {
        "run_id": os.environ.get("RUN_ID", ""),
        "prompt_id": prompt_id,
        "start_ts": start_ts,
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


def _resolve_manifest_dir(tracers: list[Tracer]) -> Path | None:
    """Find the per-prompt output dir by asking each enabled tracer.

    The first enabled tracer with a file-shaped output path wins; its
    parent directory is where the manifest lands. This keeps the
    manifest co-located with the artifacts without hardcoding env vars.
    """
    for t in tracers:
        if not t.is_enabled():
            continue
        path = t.resolve_output_path()
        if path is None or path == STDERR_SENTINEL:
            continue
        return path.parent
    return None
