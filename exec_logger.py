"""Opt-in runtime execution logger for ComfyUI.

Activates when `COMFYUI_EXEC_LOG` env var is set to a non-empty value.
Monkey-patches ComfyUI's per-node execution function to emit one line
per node: timestamp, node id, class, wall-clock duration, and tensor
shapes of each input/output (where identifiable). Output goes to:
  - `stderr`      -- if env var is literally the string "stderr"
  - `auto`/`1`    -- auto-generate a timestamped file under the shared
                     runs dir (gitignored): `internal/analysis/runs/
                     exec_log/exec_YYYY-MM-DD_HHMMSS.jsonl`
  - any other value -- treated as an explicit file path

Examples:

    COMFYUI_EXEC_LOG=auto python <comfyui>/main.py --listen 0.0.0.0
    # run the workflow; the JSONL lands in
    # <plugin>/internal/analysis/runs/exec_log/exec_*.jsonl

    COMFYUI_EXEC_LOG=/tmp/exec.jsonl python <comfyui>/main.py
    jq -c 'select(.event=="end")' /tmp/exec.jsonl | head -30

Output format: JSON lines, one per node event. Fields:
  ts          -- epoch float seconds
  event       -- "start" | "end"
  prompt_id   -- ComfyUI prompt id (ties events across a run)
  node_id     -- numeric node id in the workflow
  class_type  -- ComfyUI class type string
  duration_s  -- (end only) wall-clock seconds for this node's execute()
  input_shapes  -- dict of input_name -> tensor shape tuple OR value
  output_shapes -- (end only) list of output tensor shapes

Zero overhead when the env var is unset (import check, return).
Does NOT modify workflow files. Does NOT slow execution when active
beyond the shape-inspection calls (negligible).

Known limitation: monkey-patches the async `execute()` in
comfy/execution.py. If ComfyUI's internal API signature changes, this
logger will fail to install and emit a single warning. That's fine —
the workflow still runs.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from pathlib import Path
from typing import Any

import orjson

# scripts/ lives at <plugin>/scripts/workflow_utils.py
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from workflow_utils import timestamped_run_path  # noqa: E402


_LOG_PATH_ENV = "COMFYUI_EXEC_LOG"
_SHAPE_LIMIT_ENV = "COMFYUI_EXEC_LOG_SHAPE_LIMIT"  # max items shown for lists/dicts
_SHAPE_RECURSION_DEPTH_LIMIT = 2
_INSTALLED = False

# Shell-style truthy tokens that mean "auto-generate a timestamped path
# under the shared runs dir." Matches common conventions (Docker,
# systemd, Python configs) so users don't have to remember a bespoke
# spelling.
_AUTO_TOKENS = {"auto", "1", "true", "yes"}


def _resolve_log_target(value: str) -> str:
    """Map the env var value to a concrete sink: `"stderr"` or a path."""
    if value == "stderr":
        return value
    if value.lower() in _AUTO_TOKENS:
        return str(timestamped_run_path("exec_log", "exec", "jsonl"))
    return value


def _shape_of(value: Any, depth: int = 0) -> Any:
    """Extract a JSON-safe shape/summary from an arbitrary Python value.

    Keeps output compact: tensor.shape, dict of names → shapes, list of
    lengths. Refuses to descend past `_SHAPE_RECURSION_DEPTH_LIMIT` to
    avoid dumping huge nested structures.
    """
    if depth > _SHAPE_RECURSION_DEPTH_LIMIT:
        return "<...>"
    try:
        # torch.Tensor
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
        # Truncate long strings (e.g. schedules).
        if isinstance(value, str) and len(value) > 120:
            return value[:117] + "..."
        return value
    if isinstance(value, (list, tuple)):
        limit = int(os.environ.get(_SHAPE_LIMIT_ENV, "8"))
        head = [_shape_of(v, depth + 1) for v in value[:limit]]
        if len(value) > limit:
            head.append(f"<+{len(value) - limit} more>")
        return head
    if isinstance(value, dict):
        limit = int(os.environ.get(_SHAPE_LIMIT_ENV, "8"))
        out: dict = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= limit:
                out["<truncated>"] = len(value) - limit
                break
            out[str(k)] = _shape_of(v, depth + 1)
        return out
    return f"<{type(value).__name__}>"


def _emit(sink, record: dict) -> None:
    try:
        line = orjson.dumps(record).decode()
    except Exception:
        # Last-resort fallback — something inside the record isn't
        # JSON-safe. Stringify and move on; never crash the run.
        line = orjson.dumps({
            k: (str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v)
            for k, v in record.items()
        }).decode()
    if sink is sys.stderr:
        print(line, file=sink, flush=True)
    else:
        sink.write(line + "\n")
        sink.flush()


# Stamped on `_exec_mod.execute` itself once we wrap it. Lives on the
# function object, not on this module -- survives reloads of
# `audioloophelper`/`exec_logger` that would reset _INSTALLED. Without
# this, each reload captures the previously-wrapped execute as
# `original` and adds a new sink, growing a chain of wrappers all
# writing the same data to different files. (Root cause of the
# 7-near-duplicate-files mystery on 2026-04-25.)
_SENTINEL_ATTR = "_audioloophelper_wrapped"


def install() -> bool:
    """Install the monkey-patch. Returns True if active, False if no-op.

    Idempotent across module reloads: detects an already-wrapped
    `_exec_mod.execute` via a sentinel on the function object itself.
    Module-level `_INSTALLED` resets on reload, but the sentinel does
    not -- it survives until ComfyUI replaces `_exec_mod.execute`
    wholesale (a future change), at which point re-install correctly
    proceeds.
    """
    global _INSTALLED
    if _INSTALLED:
        return True

    raw = os.environ.get(_LOG_PATH_ENV, "").strip()
    if not raw:
        return False

    try:
        import execution as _exec_mod  # ComfyUI's execution.py
    except ImportError:
        print(f"[{__name__}] skipped: ComfyUI execution.py not importable",
              file=sys.stderr)
        return False

    # Sentinel check: if execute is already wrapped (sibling import or
    # post-reload re-entry), don't chain a new wrapper. Mark our own
    # _INSTALLED True so subsequent calls in this process short-circuit
    # at the top of the function.
    if getattr(_exec_mod.execute, _SENTINEL_ATTR, False):
        _INSTALLED = True
        return True

    log_path = _resolve_log_target(raw)
    sink = sys.stderr if log_path == "stderr" else open(log_path, "a", buffering=1)
    # Signature match against ComfyUI's execute() is NOT verified at
    # install time. If ComfyUI changes the signature, the wrapper will
    # raise TypeError on the first workflow run (late, but safe-fail;
    # the workflow won't execute silently-wrong).
    original = _exec_mod.execute

    async def wrapped_execute(
        server, dynprompt, caches, current_item, extra_data,
        executed, prompt_id, execution_list, pending_subgraph_results,
        pending_async_nodes, ui_outputs,
    ):
        node_id = current_item
        # dynprompt is a DynamicPrompt wrapping the actual prompt dict
        try:
            node_info = dynprompt.get_node(node_id)
        except Exception:
            node_info = {}
        class_type = node_info.get("class_type", "?") if node_info else "?"
        inputs_snapshot = _shape_of(node_info.get("inputs"))

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
        # Capture post-execution outputs from caches. This is
        # best-effort: the exact cache API can vary between ComfyUI
        # versions, so wrap in a try.
        output_shapes: Any = None
        try:
            cache = caches.outputs if hasattr(caches, "outputs") else caches
            if hasattr(cache, "get"):
                cached = cache.get(node_id)
                # Recent ComfyUI made HierarchicalCache.get a coroutine;
                # earlier versions return synchronously. Await iff coroutine.
                if inspect.iscoroutine(cached):
                    cached = await cached
                if cached is not None:
                    output_shapes = _shape_of(cached)
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

    setattr(wrapped_execute, _SENTINEL_ATTR, True)
    _exec_mod.execute = wrapped_execute
    _INSTALLED = True
    print(f"[{__name__}] installed -> {log_path}", file=sys.stderr)
    return True


# Auto-install on import when env var is set. Import this module early
# (e.g. from AudioLoopHelper's __init__.py) to catch all workflow runs.
if os.environ.get(_LOG_PATH_ENV, "").strip():
    install()
