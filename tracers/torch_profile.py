"""torch.profiler aten-op tracer.

Wraps each model invocation in a `torch.profiler.profile` context and
exports a Chrome trace JSON. Captures op-level timing for things the
forward-hook approach (`FfnAttnTracer`) cannot reach: AdaLN broadcast
multiplies, RoPE rotation kernels, LayerNorm, in-sampler NAG, etc.

Flushes per cleanup: each `SamplerCustomAdvanced` invocation produces
its own numbered file (`torch_profile.0.json`, `torch_profile.1.json`,
...). Multi-stage workflows like FML2V produce multiple files; the
analyzer combines them. atexit is the safety-net flush for the last
invocation in a session.
"""

from __future__ import annotations

from typing import Any

from ._base import Tracer, get_executing_prompt_id


class TorchProfileTracer(Tracer):
    name = "torch_profile"
    env_var = "AUDIOLOOPHELPER_TORCH_PROFILE"
    lifecycle = "render"
    artifact_category = "torch_profile"
    artifact_ext = "json"

    def __init__(self) -> None:
        self._active_profiler: Any = None
        self._base_path = None
        self._cached_prompt_id: str | None = None
        self._cleanup_count: int = 0
        self._files_written: list[str] = []

    # --- lifecycle ---

    def install_at_render(self, model_clone: Any) -> bool:
        # If a prior render left a profiler open (cleanup didn't fire),
        # export it before starting a new one.
        if self._active_profiler is not None:
            self._export_active("install-rotation")

        self._base_path = self.resolve_output_path()
        if self._base_path is None:
            return False
        self._cached_prompt_id = get_executing_prompt_id()
        self._cleanup_count = 0

        return self._start_profiler()

    def on_cleanup(self) -> None:
        """Stop+export the current profile, optionally rotate output path
        for a new prompt, start a fresh profile.

        Per-cleanup export is the reliability mechanism — without it,
        long-running ComfyUI sessions never see their data written.

        Prompt-boundary handling: if the executing prompt_id has changed
        since the last cleanup, the export still lands at the prior
        `_base_path` (the trace it captured belongs to the prior prompt),
        then we rebind for the next capture. ComfyUI caches the sage
        node's output when inputs are unchanged, so `install_at_render`
        doesn't refresh `_base_path` per prompt.
        """
        if self._active_profiler is None:
            return
        self._export_active("on_cleanup")
        if self._prompt_id_changed():
            new_path = self.resolve_output_path()
            if new_path is not None:
                self._base_path = new_path
            self._cleanup_count = 0
        self._start_profiler()

    def on_atexit(self) -> None:
        if self._active_profiler is not None:
            self._export_active("on_atexit")

    # --- internal ---

    def _start_profiler(self) -> bool:
        try:
            import torch.profiler
        except ImportError:
            self.log("torch.profiler import failed")
            return False
        try:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=False,
                profile_memory=False,
                # `with_modules=True` is TorchScript-only per pytorch
                # docs. For eager-mode models, per-sub-module attribution
                # comes from `ffn_attn_tracer`'s `record_function`
                # annotations.
            )
            prof.start()
        except Exception as e:
            self.log(f"profiler start raised {type(e).__name__}: {e}")
            return False
        self._active_profiler = prof
        return True

    def _export_active(self, trigger: str) -> None:
        if self._active_profiler is None or self._base_path is None:
            return

        # Numbered filename per cleanup. `torch_profile.json` becomes
        # `torch_profile.0.json` / `torch_profile.1.json` / ...
        suffix = self._base_path.suffix
        stem = self._base_path.stem
        numbered = self._base_path.with_name(f"{stem}.{self._cleanup_count}{suffix}")

        try:
            self._active_profiler.stop()
        except Exception as e:
            self.log(f"profiler stop raised {type(e).__name__}: {e}")
            self._active_profiler = None
            return
        try:
            numbered.parent.mkdir(parents=True, exist_ok=True)
            self._active_profiler.export_chrome_trace(str(numbered))
            self._files_written.append(str(numbered))
            self.log(f"exported [{trigger}] -> {numbered}")
        except Exception as e:
            self.log(f"export raised {type(e).__name__}: {e}")
        finally:
            self._active_profiler = None
            self._cleanup_count += 1

    # --- manifest reporting ---

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "enabled": self.is_enabled(),
            "output_pattern": str(self._base_path) if self._base_path else None,
            "files_written": list(self._files_written),
            "cleanup_count": self._cleanup_count,
        }
