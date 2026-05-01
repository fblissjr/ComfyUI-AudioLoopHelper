try:
    from comfy_api.latest import ComfyExtension  # noqa: F401 -- probe only
except ImportError:
    pass  # Outside ComfyUI runtime (e.g., pytest)
else:
    from .nodes import comfy_entrypoint  # noqa: F401

    # Opt-in runtime execution logger. Activates only when
    # COMFYUI_EXEC_LOG env var is set. Zero overhead otherwise.
    # See exec_logger.py for output format.
    from . import exec_logger  # noqa: F401

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
