"""Shared helper for node-registration tests.

`tests/` is on sys.path via `tests/conftest.py`, so tests import as
`from _node_registry import assert_node_registered`.

The registration test exists because `nodes.py::AudioLoopHelperExtension.get_node_list`
uses ``from .X import Y`` relative imports that fail when nodes.py is
loaded as a top-level module (pytest default — no package context). We
can't call the method, so we AST-walk the source for class names
referenced inside the extension class. See `tests/CLAUDE.md`
"Testing ComfyExtension.get_node_list() requires AST" for the
underlying constraint.
"""

from __future__ import annotations

import ast
import pathlib


def assert_node_registered(node_class_name: str) -> None:
    """Assert ``node_class_name`` is referenced inside
    ``AudioLoopHelperExtension`` in ``nodes.py``.

    Use in a node's behavioral test as the smoke check that ComfyUI
    will discover it. Failure means the new node was implemented but
    not added to ``get_node_list()``."""
    src = pathlib.Path("nodes.py").read_text()
    tree = ast.parse(src)
    found_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "AudioLoopHelperExtension":
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    found_names.add(child.id)
    assert node_class_name in found_names, (
        f"{node_class_name} not referenced inside AudioLoopHelperExtension. "
        "Add it to the get_node_list() return value."
    )
