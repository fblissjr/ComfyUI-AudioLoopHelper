"""Forward-hook tracer for `BasicAVTransformerBlock` sub-modules.

Installs PyTorch forward pre+post hooks on every transformer block's
sub-modules and records per-call CUDA wall-time via `torch.cuda.Event`.
Output is JSONL, consumable by `scripts/analyze_ffn_attn_trace.py`.
"""

from __future__ import annotations

import time
from typing import Any

import orjson
import torch

from ._base import Tracer, get_executing_prompt_id


# Sub-modules of `BasicAVTransformerBlock` we instrument. Verified
# against `comfy/ldm/lightricks/av_model.py::BasicAVTransformerBlock`.
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

_FLUSH_THRESHOLD = 256
_PendingEvent = tuple[str, int, Any, Any, Any, str | None, float]


class FfnAttnTracer(Tracer):
    name = "ffn_attn"
    env_var = "AUDIOLOOPHELPER_FFN_ATTN_TRACE"
    lifecycle = "render"
    artifact_category = "ffn_attn_breakdown"
    artifact_ext = "jsonl"

    def __init__(self) -> None:
        self._pending: list[_PendingEvent] = []
        self._output_path = None
        self._cached_prompt_id: str | None = None
        self._installed_on: set[int] = set()
        self._total_events_written: int = 0

    # --- lifecycle ---

    def install_at_render(self, model_clone: Any) -> bool:
        # If a prior render left buffered events, flush them to their
        # original target file BEFORE we swap output paths. Otherwise
        # the tail events of render N would land in render N+1's file.
        if self._output_path is not None and self._pending:
            self._flush_pending()

        self._output_path = self.resolve_output_path()
        if self._output_path is None:
            return False

        # Refresh the prompt-id cache each render. `_get_prompt_id` is
        # hot-path: reads a contextvar once per render, caches.
        self._cached_prompt_id = get_executing_prompt_id()

        try:
            diffusion_model = model_clone.get_model_object("diffusion_model")
        except Exception:
            return False

        blocks = getattr(diffusion_model, "transformer_blocks", None)
        if blocks is None:
            return False

        installed_any = False
        for block_idx, block in enumerate(blocks):
            for label in SUB_MODULE_NAMES:
                sub = getattr(block, label, None)
                if sub is None:
                    continue
                # Idempotent: skip if we already hooked this submodule.
                if id(sub) in self._installed_on:
                    continue
                sub.register_forward_pre_hook(self._make_pre_hook())
                sub.register_forward_hook(self._make_post_hook(label, block_idx))
                self._installed_on.add(id(sub))
                installed_any = True

        return installed_any

    def on_cleanup(self) -> None:
        self._flush_pending()

    def on_atexit(self) -> None:
        self._flush_pending()

    # --- hooks ---

    def _make_pre_hook(self):
        def pre_hook(module, inputs):
            x = inputs[0] if inputs else None
            T = None
            if x is not None and getattr(x, "ndim", 0) >= 2:
                try:
                    T = int(x.shape[1])
                except Exception:
                    T = None
            e_start = torch.cuda.Event(enable_timing=True)
            e_start.record()
            # Tuple of non-Module values — safe to stash on the module.
            # Storing a Tensor or nn.Module here would trip
            # `nn.Module.__setattr__`'s auto-register-as-submodule footgun
            # and double-count in `state_dict()`.
            module._ffn_attn_trace_state = (T, e_start, time.time())
        return pre_hook

    def _make_post_hook(self, label: str, block_idx: int):
        def post_hook(module, inputs, output):
            state = getattr(module, "_ffn_attn_trace_state", None)
            if state is None:
                return
            T, e_start, ts = state
            e_end = torch.cuda.Event(enable_timing=True)
            e_end.record()
            self._pending.append(
                (label, block_idx, T, e_start, e_end, self._cached_prompt_id, ts)
            )
            del module._ffn_attn_trace_state
            if len(self._pending) >= _FLUSH_THRESHOLD:
                self._flush_pending()
        return post_hook

    # --- flush ---

    def _flush_pending(self) -> None:
        if not self._pending or self._output_path is None:
            return

        torch.cuda.synchronize()

        lines = []
        for label, block_idx, T, e_start, e_end, prompt_id, ts in self._pending:
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
        n_written = len(lines)
        self._pending = []

        if not lines:
            return
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_path, "ab") as fh:
            fh.write(b"\n".join(lines) + b"\n")
        self._total_events_written += n_written

    # --- manifest reporting ---

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "enabled": self.is_enabled(),
            "output": str(self._output_path) if self._output_path else None,
            "events_written": self._total_events_written,
            "blocks_hooked": len(self._installed_on) // len(SUB_MODULE_NAMES),
        }
