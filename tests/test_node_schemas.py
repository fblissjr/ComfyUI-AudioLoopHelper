"""Schema-level invariants for our ComfyUI nodes.

Source-level (AST) checks rather than runtime introspection — the test must
work both with and without ComfyUI loaded, and `define_schema()` returns
opaque `_Passthrough` stubs in the stub-import path used by pytest.
"""

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent

_NODE_FILES = (
    "nodes.py",
    "nodes_analysis.py",
    "nodes_sage.py",
    "nodes_easycache.py",
    "nodes_validation.py",
)


def _scan_io_input_names(path: Path):
    """Find every `io.<Type>.Input(...)` call and yield (lineno, name).

    Recognizes both positional name (`io.Int.Input("seed", ...)`) and the
    explicit kwarg form (`io.Int.Input(name="seed", ...)`).
    """
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        # Match io.<SomeType>.Input(...)
        if not (
            isinstance(f, ast.Attribute)
            and f.attr == "Input"
            and isinstance(f.value, ast.Attribute)
            and isinstance(f.value.value, ast.Name)
            and f.value.value.id == "io"
        ):
            continue
        name_value = None
        if node.args and isinstance(node.args[0], ast.Constant):
            name_value = node.args[0].value
        else:
            for kw in node.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    name_value = kw.value.value
                    break
        if isinstance(name_value, str):
            yield (node.lineno, name_value)


def test_no_seed_or_noise_seed_named_inputs():
    """ComfyUI's frontend auto-attaches a `control_after_generate` dropdown
    to any INT widget literally named `"seed"` or `"noise_seed"`. After every
    successful run the dropdown mutates the saved widget value, polluting
    workflow JSONs — even when the input is wired (the link supersedes the
    widget at execute time, but the mutated widget still gets serialized,
    which makes saved workflows look like they're randomizing seeds when
    they aren't).

    Diagnosed 2026-04-26 during the ID-LoRA ablation; full writeup at
    `internal/analysis/id_lora_ablation_and_seed_widget_audit.md`.

    Use `base_seed`, `seed_in`, `random_seed`, or any other name to suppress
    the auto-attach.
    """
    forbidden = {"seed", "noise_seed"}
    leaks = []
    for module in _NODE_FILES:
        path = REPO_ROOT / module
        if not path.exists():
            pytest.skip(f"missing module file: {module}")
        for lineno, name in _scan_io_input_names(path):
            if name in forbidden:
                leaks.append(f"{module}:{lineno} -> io.*.Input(\"{name}\", ...)")
    assert not leaks, (
        "Inputs named exactly 'seed'/'noise_seed' detected:\n  "
        + "\n  ".join(leaks)
        + "\n\nComfyUI auto-attaches `control_after_generate` to these "
        "widget names, causing saved-widget-value drift across runs. "
        "Rename to e.g. 'base_seed'. See "
        "internal/analysis/id_lora_ablation_and_seed_widget_audit.md."
    )
