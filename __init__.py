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

    # Clear stale profile output on ComfyUI startup. Guarded on torch so
    # ComfyUI-HotReloadHack reimports don't re-wipe mid-run (guard survives
    # module reloads since `torch` isn't hot-reloaded).
    def _clear_stale_profile_output() -> None:
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

        # Clean both the legacy location and the new canonical one so
        # stale runs don't accumulate after the move.
        plugin_root = Path(__file__).resolve().parent
        for profile_dir in (
            plugin_root / "profile_output",
            plugin_root / "internal" / "analysis" / "runs" / "profiler",
        ):
            if not profile_dir.exists():
                continue
            for child in profile_dir.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                except OSError as e:  # permission / concurrent access
                    print(f"[AudioLoopHelper] skipped {child.name}: {e}")

    _clear_stale_profile_output()
