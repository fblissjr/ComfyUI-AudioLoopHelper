try:
    from comfy_api.latest import ComfyExtension  # noqa: F401 -- probe only
except ImportError:
    pass  # Outside ComfyUI runtime (e.g., pytest)
else:
    from .nodes import comfy_entrypoint  # noqa: F401
