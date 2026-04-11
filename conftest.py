"""Root conftest -- prevents pytest from importing __init__.py as a package.

The root __init__.py uses relative imports that only work inside ComfyUI.
"""

collect_ignore = ["__init__.py", "nodes.py"]
