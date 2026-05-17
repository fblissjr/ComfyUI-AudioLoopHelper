"""Forward-hook tracer for `BasicAVTransformerBlock` sub-modules.

Installs PyTorch forward pre+post hooks on every transformer block's
sub-modules and records per-call CUDA wall-time via `torch.cuda.Event`.
Output is JSONL, consumable by `scripts/analyze_ffn_attn_trace.py`.

When the torch.profiler tracer is also enabled, each sub-module
forward is additionally wrapped in a `torch.profiler.record_function`
span so `scripts/analyze_torch_profile.py` can attribute aten ops back
to their parent sub-module. This is the eager-mode equivalent of
`torch.profiler.profile(with_modules=True)` — which per the PyTorch
docs is TorchScript-only and a silent no-op on eager-mode models like
LTX 2.3.

## Verified annotation emission

End-to-end verified on an FML2V two-stage audit render: the chrome
trace's `cat=user_annotation` event stream contains 2304 annotations
in stage-2 alone (48 blocks × 8 sub-modules × 6 sampler forwards),
named like `attn1/block_0`, `audio_to_video_attn/block_5`, etc. The
analyzer's `_build_span_index` bisect-attribution lookup finds them
via the `BLOCK_ANNOTATION_MARKER` substring and attributes every
nested aten op to its parent sub-module.
"""

from __future__ import annotations

import os
import time
from typing import Any

import orjson
import torch
import torch.profiler

from ._base import Tracer, get_executing_prompt_id


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

# Shared contract: `tracers/ffn_attn.py` produces annotations of this
# shape; `scripts/analyze_torch_profile.py` consumes by matching the
# marker. Drift-resistance via single source of truth.
BLOCK_ANNOTATION_FMT = "{label}/block_{idx}"
BLOCK_ANNOTATION_MARKER = "/block_"


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
        # Whether to emit `record_function` annotations. Only True when
        # the torch.profiler tracer is also enabled — annotations cost
        # ~1-3 us each via C++ profiler bookkeeping even when no profile
        # is capturing, so we skip them on attribute-check fast path
        # otherwise.
        self._emit_annotations: bool = False

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
        # Cache the torch_profile gate once per install; per-call env
        # lookups would dominate hook overhead.
        self._emit_annotations = bool(
            os.environ.get("AUDIOLOOPHELPER_TORCH_PROFILE", "").strip()
        )
        # Fingerprint mode forces a CUDA sync per sage call (via `.item()`
        # in `_fingerprint_tensor`); the surrounding sub-module forward
        # waits on it, so `elapsed_ms` here is inflated 1.5-2× vs the
        # non-fingerprint baseline. Surface this loudly so a future
        # reader doesn't quote these timings as production-comparable.
        if os.environ.get("AUDIOLOOPHELPER_SAGE_OUTPUT_FINGERPRINT", "").strip():
            self.log(
                "fingerprint mode also active — forward-hook elapsed_ms "
                "inflated 1.5-2x by sage's per-call CUDA sync; audit-only."
            )

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
                sub.register_forward_pre_hook(self._make_pre_hook(label, block_idx))
                sub.register_forward_hook(self._make_post_hook(label, block_idx))
                self._installed_on.add(id(sub))
                installed_any = True

        return installed_any

    def on_cleanup(self) -> None:
        self._flush_pending()

    def on_atexit(self) -> None:
        self._flush_pending()

    # --- per-prompt state rotation ---

    def _maybe_rotate_for_new_prompt(self) -> None:
        """If the current executing prompt_id differs from the cached one,
        flush pending events to the prior path and rebind state to the new
        prompt's path. Necessary because ComfyUI caches `_patch_impl`'s
        output when inputs are unchanged across renders, so
        `install_at_render` may not re-fire per prompt — but forward hooks
        installed during the first render keep firing for subsequent ones.
        """
        current = get_executing_prompt_id()
        if current is None or current == self._cached_prompt_id:
            return
        if self._pending:
            self._flush_pending()
        self._cached_prompt_id = current
        new_path = self.resolve_output_path()
        if new_path is not None:
            self._output_path = new_path

    # --- hooks ---

    def _make_pre_hook(self, label: str, block_idx: int):
        annotation_name = BLOCK_ANNOTATION_FMT.format(label=label, idx=block_idx)
        def pre_hook(module, inputs):
            self._maybe_rotate_for_new_prompt()
            x = inputs[0] if inputs else None
            T = None
            if x is not None and getattr(x, "ndim", 0) >= 2:
                try:
                    T = int(x.shape[1])
                except Exception:
                    T = None
            rec_fn = None
            if self._emit_annotations:
                rec_fn = torch.profiler.record_function(annotation_name)
                rec_fn.__enter__()
            e_start = torch.cuda.Event(enable_timing=True)
            e_start.record()
            # Tuple of non-Module values — safe to stash on the module.
            # Storing a Tensor or nn.Module here would trip
            # `nn.Module.__setattr__`'s auto-register-as-submodule footgun
            # and double-count in `state_dict()`.
            module._ffn_attn_trace_state = (T, e_start, time.time(), rec_fn)
        return pre_hook

    def _make_post_hook(self, label: str, block_idx: int):
        def post_hook(module, inputs, output):
            state = getattr(module, "_ffn_attn_trace_state", None)
            if state is None:
                return
            T, e_start, ts, rec_fn = state
            e_end = torch.cuda.Event(enable_timing=True)
            e_end.record()
            if rec_fn is not None:
                rec_fn.__exit__(None, None, None)
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
