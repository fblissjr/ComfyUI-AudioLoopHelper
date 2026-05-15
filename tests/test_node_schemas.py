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


# Names every io.X.Input("...") call is matched against to flag an
# fps-bearing widget. Keep this synced with the runtime widget-name
# conventions across our nodes + the upstream allowlist (LTXVAudioVideoMask
# uses video_fps; VHS_VideoCombine uses frame_rate; ours uses fps or
# frame_rate). Add to this set if a new fps-alias surfaces.
_FPS_INPUT_NAMES = frozenset({"fps", "frame_rate", "video_fps"})

# Upstream node classes whose schemas we don't control but whose widgets
# the apply script MUST sweep. Hand-maintained because we can't AST-scan
# packages outside the tree without import-time side-effects.
# Cross-reference with scripts/apply_fps_24_default.py LIST_WIDGET_NODES.
_UPSTREAM_FPS_NODES = frozenset({
    "LTXVAudioVideoMask",  # KJNodes: widget[0] = video_fps (default 25 upstream)
    "LTXVConditioning",    # comfy-core (nodes_lt.py): widget[0] = frame_rate
    "LTXVEmptyLatentAudio",  # comfy-core (nodes_lt_audio.py): widget[1] = frame_rate
})

# Carve-out: VHS_VideoCombine.frame_rate IS fps-bearing but uses dict-shape
# widgets_values, handled via a separate code path in apply_fps_24_default.py
# (not LIST_WIDGET_NODES). The list-shape coverage test below doesn't apply.
_DICT_SHAPE_FPS_NODES = frozenset({"VHS_VideoCombine"})


def _classes_with_fps_widgets(path: Path):
    """Yield class names in `path` whose define_schema declares any
    io.<Type>.Input(...) with a name in _FPS_INPUT_NAMES.

    Two-pass: walk the AST once to enumerate top-level class names, then
    reuse `_scan_io_input_records_in_class` for each. Avoids reinventing
    the class-bounded line-range logic.
    """
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    class_names = [
        n.name for n in ast.iter_child_nodes(tree)
        if isinstance(n, ast.ClassDef)
    ]
    for cls in class_names:
        for _lineno, name, _kwargs in _scan_io_input_records_in_class(path, cls):
            if name in _FPS_INPUT_NAMES:
                yield cls
                break  # one fps input is enough to flag the class


def _parse_apply_script_list_widget_keys(apply_script_path: Path) -> set[str]:
    """Extract the keys of the LIST_WIDGET_NODES dict literal in
    apply_fps_24_default.py via AST. Text-only — does not import the
    module (which would drag scripts/ into sys.path side-effects)."""
    src = apply_script_path.read_text()
    tree = ast.parse(src, filename=str(apply_script_path))
    for node in ast.iter_child_nodes(tree):
        # Match either `LIST_WIDGET_NODES = {...}` or `LIST_WIDGET_NODES: dict = {...}`
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
            value = node.value
        if not targets or value is None:
            continue
        if not any(t.id == "LIST_WIDGET_NODES" for t in targets):
            continue
        if not isinstance(value, ast.Dict):
            return set()  # unexpected shape — surface as empty so test fails loudly
        keys: set[str] = set()
        for k in value.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.add(k.value)
        return keys
    return set()


def test_apply_fps_24_default_covers_all_fps_bearing_widgets():
    """Property: every fps-bearing widget in our nodes_*.py plus the
    upstream allowlist must be covered by LIST_WIDGET_NODES in
    scripts/apply_fps_24_default.py.

    Reproduces the bug class that caused 35-40s audio drift after the
    25→24 migration: LTXVAudioVideoMask (KJNodes upstream) shipped with
    video_fps=25 by default and was missed by the original sweep because
    the apply script's LIST_WIDGET_NODES was a hand-curated allowlist.
    Adding a new fps-bearing input to our source — or surfacing an upstream
    one — must now trip this test until the apply script is extended.
    """
    discovered: set[str] = set()
    for module in _NODE_FILES:
        path = REPO_ROOT / module
        if not path.exists():
            continue
        discovered.update(_classes_with_fps_widgets(path))

    expected_coverage = discovered | _UPSTREAM_FPS_NODES
    apply_script = REPO_ROOT / "scripts" / "apply_fps_24_default.py"
    assert apply_script.exists(), f"missing: {apply_script}"
    covered = _parse_apply_script_list_widget_keys(apply_script)
    assert covered, (
        f"Failed to parse LIST_WIDGET_NODES from {apply_script.name}; "
        "the AST shape may have changed (was it converted to a function "
        "call or moved to another module?)."
    )

    missing = expected_coverage - covered - _DICT_SHAPE_FPS_NODES
    assert not missing, (
        "fps-bearing widget node(s) not covered by "
        "scripts/apply_fps_24_default.py LIST_WIDGET_NODES:\n  "
        + "\n  ".join(sorted(missing))
        + "\n\nWhen adding a new fps/frame_rate input to a node — or "
        "surfacing a new upstream node that has one — add the class name "
        "+ widget index to LIST_WIDGET_NODES in the apply script. If the "
        "node uses dict-shape widgets_values (like VHS_VideoCombine), add "
        "it to _DICT_SHAPE_FPS_NODES in this test instead and extend the "
        "apply script's VHS_VideoCombine code path."
    )

    # Catch stale entries — node types listed in LIST_WIDGET_NODES that
    # no longer correspond to anything we discovered. Forces cleanup
    # when a node is renamed or removed.
    stale = covered - expected_coverage
    # Allow upstream nodes that aren't in our source files but ARE in
    # the upstream allowlist (they'd otherwise look stale by AST-scan
    # alone). Already excluded by construction via expected_coverage.
    assert not stale, (
        "Stale entries in LIST_WIDGET_NODES (no matching fps-bearing "
        "node found in source or upstream allowlist):\n  "
        + "\n  ".join(sorted(stale))
        + "\n\nIf the node was renamed/removed, drop the entry from "
        "LIST_WIDGET_NODES. If it's an upstream node we still need to "
        "sweep, add it to _UPSTREAM_FPS_NODES in this test."
    )
