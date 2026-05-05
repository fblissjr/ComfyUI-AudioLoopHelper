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


def _scan_conditioning_set_values(path: Path):
    """Yield (lineno, key, value_node) for every literal key in any
    `node_helpers.conditioning_set_values(cond, {KEY: VAL, ...})` or
    `conditioning_set_values(cond, {KEY: VAL, ...})` call.

    The frame's second arg must be a Dict literal — that's the only shape
    we care about. Computed dicts (`**kwargs`-style) are rare here and
    would warrant a manual review anyway.
    """
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        is_csv = (
            (isinstance(f, ast.Attribute) and f.attr == "conditioning_set_values")
            or (isinstance(f, ast.Name) and f.id == "conditioning_set_values")
        )
        if not is_csv or len(node.args) < 2:
            continue
        values = node.args[1]
        if not isinstance(values, ast.Dict):
            continue
        for k, v in zip(values.keys, values.values):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                yield (node.lineno, k.value, v)


def test_keyframe_idxs_cleared_to_none_not_empty_list():
    """LTXVCropGuides-equivalents must clear `keyframe_idxs` to `None`,
    never to `[]`. KJNodes' `OuterSampleCallbackWrapper` (`ltxv_nodes.py:867`)
    gates `if keyframe_idxs is not None:` then indexes as a 4D tensor;
    `[]` slips through and crashes the loop-body sampler with
    `TypeError: list indices must be integers or slices, not tuple`.

    Diagnosed 2026-04-27 — initial render OK because it used upstream
    `LTXVCropGuides` (sets None); loop body crashed because it used our
    `LTXVCropGuidesNoLatent` (was setting `[]`). Same shape applies to
    `guide_attention_entries`, which upstream also clears to None.
    """
    leaks: list[str] = []
    forbidden_keys = {"keyframe_idxs", "guide_attention_entries"}
    for module in _NODE_FILES:
        path = REPO_ROOT / module
        if not path.exists():
            continue
        for lineno, key, value in _scan_conditioning_set_values(path):
            if key not in forbidden_keys:
                continue
            # Variable references and function calls are fine — those are
            # computed values that may legitimately be tensors at runtime.
            # We only flag literal list expressions like `[]` or `[a, b]`.
            if isinstance(value, ast.List):
                leaks.append(
                    f"{module}:{lineno} -> conditioning_set_values(..., "
                    f"{{'{key}': {ast.unparse(value)}}})"
                )
    assert not leaks, (
        "Literal list assignment to keyframe_idxs / guide_attention_entries "
        "detected:\n  "
        + "\n  ".join(leaks)
        + "\n\nKJNodes' OuterSampleCallbackWrapper (ltxv_nodes.py:867) gates "
        "`if keyframe_idxs is not None:` then indexes as a 4D tensor. An "
        "empty list slips through the gate and TypeErrors on tuple-indexing. "
        "Use `None` (matches upstream LTXVCropGuides at "
        "comfy_extras/nodes_lt.py:404,408)."
    )


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


def _scan_io_input_records(path: Path):
    """Yield (lineno, name, defaults_dict) for every `io.<Type>.Input(...)` call.

    `defaults_dict` carries any literal kwargs (default, min, max, step) that
    are simple constants. Used for invariants that need to inspect the schema
    default, not just the name.
    """
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
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
        if not isinstance(name_value, str):
            continue
        kwargs: dict = {}
        for kw in node.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
        yield (node.lineno, name_value, kwargs)


def test_latent_temporal_mask_edge_taper_default_is_zero():
    """`LatentTemporalMask.edge_taper_seconds` must default to 0.0.

    Saved retake workflows that predate this input carry no widget value
    for it; ComfyUI fills the slot from the schema default at load time.
    A non-zero default would silently change the noise_mask written by
    every existing retake render — invisible regression. The default
    must remain 0.0; opt-in soft-mask is the contract.

    The test also catches:
      - rename (e.g. `edge_taper_seconds` → `edge_taper`): the named slot
        disappears entirely; ComfyUI's positional widget pop would shift
        saved widget values into adjacent slots, corrupting fps / start /
        end widget values.
      - removal: zero occurrences left; the test fails so the deletion
        is at least deliberate (update the test to reflect the change).
      - accidental duplication across modules: > 1 occurrence; flags a
        copy-paste of the schema into another node without renaming.
    """
    matches: list[tuple[str, int, dict]] = []
    for module in _NODE_FILES:
        path = REPO_ROOT / module
        if not path.exists():
            continue
        for lineno, name, kwargs in _scan_io_input_records(path):
            if name == "edge_taper_seconds":
                matches.append((module, lineno, kwargs))
    assert len(matches) == 1, (
        f"Expected exactly one input named 'edge_taper_seconds'; found "
        f"{len(matches)}: {matches}"
    )
    module, lineno, kwargs = matches[0]
    default = kwargs.get("default")
    assert default == 0.0, (
        f"{module}:{lineno} -> io.Float.Input('edge_taper_seconds', "
        f"default={default!r}, ...) — default must be 0.0 (saved workflows "
        f"that lack the widget value would silently change behavior)."
    )
