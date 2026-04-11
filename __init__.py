try:
    from .nodes import comfy_entrypoint
except ImportError:
    pass  # Outside ComfyUI runtime (e.g., pytest)
