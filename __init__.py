try:
    from comfy_api.latest import ComfyExtension  # noqa: F401 -- probe only
except ImportError:
    pass  # Outside ComfyUI runtime (e.g., pytest)
else:
    from .nodes import comfy_entrypoint  # noqa: F401

    # Unified tracer framework. Installs process-lifecycle tracers
    # (currently exec_log; monkey-patches ComfyUI's executor) at import.
    # Render-lifecycle tracers (ffn_attn, torch_profile) are installed
    # later from `nodes_sage.py::AudioLoopHelperSageAttention._patch_impl`.
    # All are env-gated; zero overhead when env vars are unset. See
    # `tracers/__init__.py` for the public API + lifecycle overview.
    from . import tracers  # noqa: F401
    tracers.install_process_tracers()

    # Clear profiler run artifacts on ComfyUI startup. Guarded on torch so
    # ComfyUI-HotReloadHack reimports don't re-wipe mid-run (guard survives
    # module reloads since `torch` isn't hot-reloaded).
    def _clear_profiler_run_artifacts() -> None:
        import shutil
        from pathlib import Path

        try:
            import torch
        except ImportError:
            return
        flag = "_audioloophelper_startup_cleaned"
        if getattr(torch, flag, False):
            return
        setattr(torch, flag, True)

        profile_dir = Path(__file__).resolve().parent / "internal" / "analysis" / "runs" / "profiler"
        shutil.rmtree(profile_dir, ignore_errors=True)
        profile_dir.mkdir(parents=True, exist_ok=True)

    _clear_profiler_run_artifacts()


# Frontend assets (the Compose Reference Audio waveform/slice editor widget). Module-level so
# ComfyUI serves web/ regardless of the ComfyExtension probe above.
WEB_DIRECTORY = "./web"
