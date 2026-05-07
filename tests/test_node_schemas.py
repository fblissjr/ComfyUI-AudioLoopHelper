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

    Thin projection over `_scan_io_input_records` — kept as a separate
    helper because most schema invariants only need the name, and the
    name-only signature reads more clearly at the call site.
    """
    for lineno, name, _ in _scan_io_input_records(path):
        yield (lineno, name)


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


def _scan_io_input_records_in_class(path: Path, class_name: str):
    """Yield (lineno, name, defaults_dict) for every `io.<Type>.Input(...)`
    call defined INSIDE the body of the named class.

    Useful for class-scoped schema invariants like "LatentTemporalMask's
    edge_taper_seconds default must be 0.0" — without this, the same input
    name on another node would false-positive against the invariant.

    Implementation: locate the class definition by source-text search,
    then bound its body by the next top-level `\\nclass ` line; filter
    `_scan_io_input_records` results by lineno against that range.
    """
    src = path.read_text()
    needle = f"class {class_name}"
    if needle not in src:
        return
    cls_start = src.index(needle)
    cls_start_line = src.count("\n", 0, cls_start) + 1
    next_cls_offset = src.find("\nclass ", cls_start + len(needle))
    cls_end_line = (
        src.count("\n", 0, next_cls_offset) + 1
        if next_cls_offset != -1 else float("inf")
    )
    for lineno, name, kwargs in _scan_io_input_records(path):
        if cls_start_line <= lineno < cls_end_line:
            yield (lineno, name, kwargs)


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
        for lineno, name, kwargs in _scan_io_input_records_in_class(path, "LatentTemporalMask"):
            if name == "edge_taper_seconds":
                matches.append((module, lineno, kwargs))
    assert len(matches) == 1, (
        f"Expected exactly one io.Float.Input('edge_taper_seconds', ...) inside "
        f"LatentTemporalMask; found {len(matches)}: {matches}"
    )
    module, lineno, kwargs = matches[0]
    default = kwargs.get("default")
    assert default == 0.0, (
        f"{module}:{lineno} -> io.Float.Input('edge_taper_seconds', "
        f"default={default!r}, ...) — default must be 0.0 (saved workflows "
        f"that lack the widget value would silently change behavior)."
    )


def _scan_io_outputs_in_class(path: Path, class_name: str):
    """Yield (lineno, name) for every `io.<Type>.Output(...)` call inside `class class_name`.

    Class-bounded analogue of `_scan_io_input_records_in_class` for output
    slots. Used to assert a node exposes a specific named output without
    coupling the test to the surrounding sibling outputs.
    """
    src = path.read_text()
    needle = f"class {class_name}"
    if needle not in src:
        return
    cls_start = src.index(needle)
    cls_start_line = src.count("\n", 0, cls_start) + 1
    next_cls_offset = src.find("\nclass ", cls_start + len(needle))
    cls_end_line = (
        src.count("\n", 0, next_cls_offset) + 1
        if next_cls_offset != -1 else float("inf")
    )
    tree = ast.parse(src, filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (
            isinstance(f, ast.Attribute)
            and f.attr == "Output"
            and isinstance(f.value, ast.Attribute)
            and isinstance(f.value.value, ast.Name)
            and f.value.value.id == "io"
        ):
            continue
        if not (cls_start_line <= node.lineno < cls_end_line):
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


def test_audio_loop_planner_exposes_stride_and_duration_outputs():
    """`AudioLoopPlanner` must output `stride_seconds` + `audio_duration`.

    These are the load-bearing outputs the batch encoder reads to break
    the cycle introduced when initial-render conditioning is sourced
    from `conditioning_list[0]` (Phase 1 of the workflow-organization
    rework). Sourcing them from `AudioLoopController` instead would
    transitively pull `current_iteration` into the encoder's input
    closure — turning the initial-render conditioning chain into a
    cycle through the loop. The planner is the cycle-free source.

    Removing or renaming either output without a paired migration
    would re-open the cycle on every workflow that picked up the new
    wiring; this AST guard fails fast on either change.
    """
    path = REPO_ROOT / "nodes.py"
    outputs = {name for _ln, name in _scan_io_outputs_in_class(path, "AudioLoopPlanner")}
    assert "stride_seconds" in outputs, (
        "AudioLoopPlanner is missing io.Float.Output('stride_seconds'). "
        "Restore it to keep the batch-encoder rewire from re-opening the cycle."
    )
    assert "audio_duration" in outputs, (
        "AudioLoopPlanner is missing io.Float.Output('audio_duration'). "
        "Restore it to keep the batch-encoder rewire from re-opening the cycle."
    )


def test_latent_seam_zone_mask_iteration_count_default_is_one():
    """`LatentSeamZoneMask.iteration_count` must default to 1 (no seams).

    A default of 2+ would write a non-zero mask the first time the node
    is dropped onto a workflow, surprising users. Default 1 = single
    iteration = no internal seams = all-zero mask (no-op). The user
    sets iteration_count to match their actual loop run.

    Companion to `test_latent_temporal_mask_edge_taper_default_is_zero`:
    same default-must-be-zero-effect contract for the seam-zone family.
    """
    matches: list[tuple[str, int, dict]] = []
    for module in _NODE_FILES:
        path = REPO_ROOT / module
        if not path.exists():
            continue
        for lineno, name, kwargs in _scan_io_input_records_in_class(path, "LatentSeamZoneMask"):
            if name == "iteration_count":
                matches.append((module, lineno, kwargs))
    assert len(matches) == 1, (
        f"Expected exactly one io.Int.Input('iteration_count', ...) inside "
        f"LatentSeamZoneMask; found {len(matches)}: {matches}"
    )
    module, lineno, kwargs = matches[0]
    default = kwargs.get("default")
    assert default == 1, (
        f"{module}:{lineno} -> io.Int.Input('iteration_count', default={default!r}, ...) "
        f"— default must be 1 (single iteration = no seams = all-zero mask = no-op)."
    )
