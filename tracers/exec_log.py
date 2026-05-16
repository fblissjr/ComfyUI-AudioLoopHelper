"""ComfyUI per-node execution logger.

Monkey-patches ComfyUI's per-node execution function to emit one JSONL
line per node start + end. Captures: prompt_id, node_id, class_type,
wall-clock duration, input/output shape snapshots.

Process-lifecycle: installs once at package import. The monkey-patch
then fires for every render in this process. Idempotent across reloads
via a sentinel attribute on the wrapped function.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from typing import Any

import orjson

from ._base import STDERR_SENTINEL, Tracer, log_event


_SHAPE_LIMIT_ENV = "COMFYUI_EXEC_LOG_SHAPE_LIMIT"
_SHAPE_RECURSION_DEPTH_LIMIT = 2

# Stamped on the wrapped `_exec_mod.execute` to detect re-wrapping
# across module reloads. Module-level state resets on reload; the
# sentinel survives until ComfyUI replaces `_exec_mod.execute`
# wholesale.
_SENTINEL_ATTR = "_audioloophelper_wrapped"


class ExecLogTracer(Tracer):
    name = "exec_log"
    env_var = "COMFYUI_EXEC_LOG"
    lifecycle = "process"
    artifact_category = "exec"
    artifact_ext = "jsonl"

    def __init__(self) -> None:
        self._installed: bool = False
        self._sink: Any = None
        self._sink_path: str | None = None

    # --- lifecycle ---

    def install_at_import(self) -> bool:
        if self._installed:
            return True

        # Resolve the sink. stderr is a special token; auto / 1 / etc.
        # resolve to a file path; everything else is treated as a path.
        raw = os.environ.get(self.env_var, "").strip()
        if not raw:
            return False

        try:
            import execution as _exec_mod  # ComfyUI's execution.py
        except ImportError:
            log_event(self.name, "ComfyUI execution.py not importable; skipped")
            return False

        # If a sibling import already wrapped execute, don't chain.
        if getattr(_exec_mod.execute, _SENTINEL_ATTR, False):
            self._installed = True
            return True

        path = self.resolve_output_path()
        if path is None:
            return False

        if path == STDERR_SENTINEL:
            sink = sys.stderr
            self._sink_path = "stderr"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            sink = open(path, "a", buffering=1)
            self._sink_path = str(path)
        self._sink = sink

        original = _exec_mod.execute
        wrapped = _make_wrapped_execute(original, sink)
        setattr(wrapped, _SENTINEL_ATTR, True)
        _exec_mod.execute = wrapped
        self._installed = True
        return True

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "enabled": self.is_enabled(),
            "output": self._sink_path,
            "installed": self._installed,
        }


# --- shape extraction (cheap, JSON-safe, depth-bounded) ---


def _shape_of(value: Any, depth: int, limit: int) -> Any:
    """JSON-safe compact summary of a value. Tensor.shape, dict of
    names → shapes, list of lengths. Refuses to descend past
    `_SHAPE_RECURSION_DEPTH_LIMIT`.
    """
    if depth > _SHAPE_RECURSION_DEPTH_LIMIT:
        return "<...>"
    try:
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        if shape is not None and dtype is not None:
            return {
                "shape": list(shape),
                "dtype": str(dtype),
                "device": str(device) if device is not None else None,
            }
    except Exception:
        pass
    if isinstance(value, (int, float, bool, str)) or value is None:
        if isinstance(value, str) and len(value) > 120:
            return value[:117] + "..."
        return value
    if isinstance(value, (list, tuple)):
        head = [_shape_of(v, depth + 1, limit) for v in value[:limit]]
        if len(value) > limit:
            head.append(f"<+{len(value) - limit} more>")
        return head
    if isinstance(value, dict):
        out: dict = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= limit:
                out["<truncated>"] = len(value) - limit
                break
            out[str(k)] = _shape_of(v, depth + 1, limit)
        return out
    return f"<{type(value).__name__}>"


def _emit(sink: Any, record: dict) -> None:
    try:
        line = orjson.dumps(record).decode()
    except Exception:
        # Last-resort fallback for non-JSON-safe values.
        line = orjson.dumps({
            k: (str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v)
            for k, v in record.items()
        }).decode()
    if sink is sys.stderr:
        print(line, file=sink, flush=True)
    else:
        sink.write(line + "\n")
        sink.flush()


def _make_wrapped_execute(original, sink):
    """Build the wrapped async execute that emits start+end events.

    If ComfyUI changes the `execute` signature, the wrapper raises
    TypeError on first workflow run — safe-fail (workflow won't run
    silently-wrong).
    """
    # Read the shape-limit once per install, not per node-execute.
    shape_limit = int(os.environ.get(_SHAPE_LIMIT_ENV, "8"))

    async def wrapped_execute(
        server, dynprompt, caches, current_item, extra_data,
        executed, prompt_id, execution_list, pending_subgraph_results,
        pending_async_nodes, ui_outputs,
    ):
        node_id = current_item
        try:
            node_info = dynprompt.get_node(node_id)
        except Exception:
            node_info = {}
        class_type = node_info.get("class_type", "?") if node_info else "?"
        inputs_snapshot = _shape_of(node_info.get("inputs"), 0, shape_limit)

        t0 = time.time()
        _emit(sink, {
            "ts": t0,
            "event": "start",
            "prompt_id": prompt_id,
            "node_id": node_id,
            "class_type": class_type,
            "inputs": inputs_snapshot,
        })

        try:
            result = await original(
                server, dynprompt, caches, current_item, extra_data,
                executed, prompt_id, execution_list, pending_subgraph_results,
                pending_async_nodes, ui_outputs,
            )
        except Exception as exc:
            duration = time.time() - t0
            _emit(sink, {
                "ts": time.time(),
                "event": "error",
                "prompt_id": prompt_id,
                "node_id": node_id,
                "class_type": class_type,
                "duration_s": round(duration, 4),
                "error": str(exc)[:500],
            })
            raise

        duration = time.time() - t0
        output_shapes: Any = None
        try:
            cache = caches.outputs if hasattr(caches, "outputs") else caches
            if hasattr(cache, "get"):
                cached = cache.get(node_id)
                if inspect.iscoroutine(cached):
                    cached = await cached
                if cached is not None:
                    output_shapes = _shape_of(cached, 0, shape_limit)
        except Exception:
            pass

        _emit(sink, {
            "ts": time.time(),
            "event": "end",
            "prompt_id": prompt_id,
            "node_id": node_id,
            "class_type": class_type,
            "duration_s": round(duration, 4),
            "outputs": output_shapes,
        })
        return result

    return wrapped_execute
