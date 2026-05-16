"""Per-sub-block GPU-time tracer for LTX 2.3 AV transformer blocks.

Activates when `AUDIOLOOPHELPER_FFN_ATTN_TRACE` env var is set to a
non-empty value. Installs PyTorch forward hooks on every
`BasicAVTransformerBlock`'s sub-modules (`attn1`, `audio_attn1`,
`attn2`, `audio_attn2`, `video_to_audio_attn`, `ff`, `audio_ff`) and
records per-call wall-time via `torch.cuda.Event`. Output goes to:

  - `auto`/`1`/`true`/`yes` -> RUN_ID-keyed path under
    `data/runs/${RUN_ID}/${prompt_id}/ffn_attn_breakdown.jsonl`
    (mirrors the sage tracer layout)
  - any other value -> treated as an explicit file path

Output format: JSON lines, one per sub-block call. Fields:
  ts          -- epoch float seconds
  label       -- sub-module name (`ff` / `attn1` / etc.)
  block_idx   -- transformer block index 0..47
  T           -- sequence length (10780 = stage-1, 44880 = stage-2)
  elapsed_ms  -- GPU wall-time for this sub-module call
  prompt_id   -- ComfyUI prompt id (ties events across a run)

Aggregation: use `scripts/analyze_ffn_attn_trace.py` to bucket by
(stage, sub-module) and compute share-of-step.

Zero overhead when env var unset (import check + return). When active,
hook overhead is ~10us per call -- ~30ms per full render at LTX 2.3
multi-guide scale, ~5% of forward time. Tolerable for one-shot bench.

Install pattern: `install_hooks(diffusion_model)` is called from
`AudioLoopHelperSageAttention._patch_impl` when env var set. The
sage node is on every workflow's model path so it's a natural
install point.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# Lazy imports inside functions so module-import is free when env unset.

# Reuse run_artifact_path from scripts/workflow_utils.py via the same
# sys.path trick nodes_sage.py uses (workflow_utils lives at scripts/,
# not package root).
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
try:
    from workflow_utils import run_artifact_path  # type: ignore
except Exception:
    def run_artifact_path(category: str, ext: str) -> Path:
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        return Path(__file__).resolve().parent / "internal" / "analysis" / "runs" / "ffn_attn" / f"ffn_attn_{ts}.{ext}"

# Reuse _AUTO_TOKENS from exec_logger to keep the env-var grammar
# consistent across all three opt-in tracers in this package.
try:
    from exec_logger import _AUTO_TOKENS  # type: ignore
except Exception:
    _AUTO_TOKENS = frozenset({"auto", "1", "true", "yes"})

_TRACE_ENV = "AUDIOLOOPHELPER_FFN_ATTN_TRACE"

# Sub-module attributes on `BasicAVTransformerBlock` to instrument.
# Audited from comfy/ldm/lightricks/av_model.py:107-184.
SUB_MODULE_NAMES = (
    "attn1",
    "audio_attn1",
    "attn2",
    "audio_attn2",
    "video_to_audio_attn",
    "audio_to_video_attn",
    "ff",
    "audio_ff",
)

_PendingEvent = tuple[str, int, int | None, Any, Any, str | None, float]

# Module-level state. ComfyUI runs renders single-threaded per process.
_PENDING: list[_PendingEvent] = []
_FLUSH_THRESHOLD = 256
_OUTPUT_PATH: Path | None = None
_ATEXIT_REGISTERED = False
_CACHED_PROMPT_ID: str | None = None


def is_enabled() -> bool:
    """Cheap check: should we install hooks at all?"""
    return bool(os.environ.get(_TRACE_ENV, "").strip())


def _resolve_path() -> Path | None:
    """Mirror sage tracer's path resolution semantics."""
    raw = os.environ.get(_TRACE_ENV, "").strip()
    if not raw:
        return None
    if raw.lower() in _AUTO_TOKENS:
        return run_artifact_path("ffn_attn_breakdown", "jsonl")
    return Path(raw)


def _get_prompt_id() -> str | None:
    """Read the active ComfyUI prompt id via contextvar (best-effort).

    Cached per-render so hot-path hooks don't re-resolve via contextvar.
    Refresh boundary is `install_hooks` (new render -> new install).
    """
    global _CACHED_PROMPT_ID
    if _CACHED_PROMPT_ID is not None:
        return _CACHED_PROMPT_ID
    try:
        from comfy_execution.utils import get_executing_context
        ctx = get_executing_context()
        if ctx is not None and ctx.prompt_id is not None:
            _CACHED_PROMPT_ID = str(ctx.prompt_id)
    except Exception:
        pass
    return _CACHED_PROMPT_ID


def _flush_pending() -> None:
    """Drain queued events to JSONL. Triggers cuda sync via elapsed_time."""
    global _PENDING
    if not _PENDING or _OUTPUT_PATH is None:
        return
    try:
        import orjson
        import torch
    except ImportError:
        return

    # cuda.synchronize implicit in elapsed_time; just one sync for the batch.
    torch.cuda.synchronize()

    lines = []
    for label, block_idx, T, e_start, e_end, prompt_id, ts in _PENDING:
        try:
            elapsed_ms = e_start.elapsed_time(e_end)
        except Exception:
            continue
        lines.append(orjson.dumps({
            "ts": ts,
            "label": label,
            "block_idx": block_idx,
            "T": T,
            "elapsed_ms": round(elapsed_ms, 4),
            "prompt_id": prompt_id,
        }))
    _PENDING = []

    if not lines:
        return
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "ab") as fh:
        fh.write(b"\n".join(lines) + b"\n")


def _make_pre_hook():
    def pre_hook(module, inputs):
        import torch
        x = inputs[0] if inputs else None
        T = None
        if x is not None and hasattr(x, "shape") and x.shape and len(x.shape) >= 2:
            try:
                T = int(x.shape[1])
            except Exception:
                T = None
        e_start = torch.cuda.Event(enable_timing=True)
        e_start.record()
        # Stash on the module instance for the post-hook to pair.
        # Safe because the value is a tuple — if you ever extend this to
        # carry a Tensor or another nn.Module, you'll trip the auto-
        # register-as-submodule footgun (see root CLAUDE.md "ComfyUI
        # gotchas" entry on `nn.Module.__setattr__`). Use a non-Module
        # container instead (dict-on-the-module is fine, since the dict
        # itself isn't Module-typed).
        module._ffn_attn_trace_state = (T, e_start, time.time())
    return pre_hook


def _make_post_hook(label: str, block_idx: int):
    def post_hook(module, inputs, output):
        import torch
        state = getattr(module, "_ffn_attn_trace_state", None)
        if state is None:
            return
        T, e_start, ts = state
        e_end = torch.cuda.Event(enable_timing=True)
        e_end.record()
        _PENDING.append((label, block_idx, T, e_start, e_end, _get_prompt_id(), ts))
        del module._ffn_attn_trace_state
        if len(_PENDING) >= _FLUSH_THRESHOLD:
            _flush_pending()
    return post_hook


def install_hooks(diffusion_model: Any) -> bool:
    """Install pre+post hooks on every transformer block's sub-modules.

    Idempotent: if a sub-module already has `._ffn_attn_traced`, skip it.

    Returns True if at least one hook was installed.
    """
    global _OUTPUT_PATH, _ATEXIT_REGISTERED, _CACHED_PROMPT_ID
    if not is_enabled():
        return False

    # New render → flush any tail events from a prior render to its file
    # BEFORE we swap _OUTPUT_PATH, so they don't land in the new render's
    # file. Then re-resolve the path freshly so this render writes to its
    # own prompt_id-keyed location (previously _OUTPUT_PATH was cached on
    # first call and every subsequent render appended to the first file).
    if _OUTPUT_PATH is not None and _PENDING:
        _flush_pending()
    _OUTPUT_PATH = _resolve_path()
    if _OUTPUT_PATH is None:
        return False

    blocks = getattr(diffusion_model, "transformer_blocks", None)
    if blocks is None:
        return False

    # New render → refresh the cached prompt_id at the next event.
    _CACHED_PROMPT_ID = None

    installed_any = False
    for idx, block in enumerate(blocks):
        for label in SUB_MODULE_NAMES:
            sub = getattr(block, label, None)
            if sub is None:
                continue
            if getattr(sub, "_ffn_attn_traced", False):
                continue
            try:
                sub.register_forward_pre_hook(_make_pre_hook())
                sub.register_forward_hook(_make_post_hook(label, idx))
                sub._ffn_attn_traced = True
                installed_any = True
            except Exception:
                # Defensive: don't crash the render if hook registration fails.
                continue

    if installed_any and not _ATEXIT_REGISTERED:
        import atexit
        atexit.register(_flush_pending)
        _ATEXIT_REGISTERED = True

    return installed_any


def maybe_flush() -> None:
    """Public flush trigger -- callable from outside (e.g. after a render)."""
    _flush_pending()
