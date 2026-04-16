try:
    from comfy_api.latest import ComfyExtension  # noqa: F401 -- probe only
except ImportError:
    pass  # Outside ComfyUI runtime (e.g., pytest)
else:
    from .nodes import comfy_entrypoint  # noqa: F401

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

        profile_dir = Path(__file__).resolve().parent / "profile_output"
        if not profile_dir.exists():
            return
        for child in profile_dir.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except OSError as e:  # permission / concurrent access
                print(f"[AudioLoopHelper] skipped {child.name}: {e}")

    _clear_stale_profile_output()
