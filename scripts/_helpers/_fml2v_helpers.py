"""Shared scaffolding for fml2v audio-loop staged-variant apply scripts.

Promoted from `apply_fml2v_smoke_toggle.py`, `apply_fml2v_no_pass2_blend.py`,
and `apply_fml2v_pass1_recovery.py` per CLAUDE.md "Promote helpers at the
3rd call site, not the 2nd." The three sibling scripts share:

  * Phase 5 stash lookup (each script needs `tlo` and other IDs the
    builder wrote into `wf["properties"]["build_fml2v_phase5"]`).
  * Smoke loop config (unwire `TensorLoopOpen.iterations_in` + set the
    widget to a short iteration count for fast diagnostic renders).
  * The migrate/revert/argparse CLI scaffold (copy input → output, apply
    mutations idempotently, support `--dry-run` and `--revert`).

These helpers are fml2v-specific (they assume the `build_fml2v_phase5`
stash shape from `scripts/build_fml2v_audio_loop.py`). Generic apply
helpers live in `_apply_helpers.py`; the canonical edit API lives in
`workflow_utils.py`.
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflow_utils import WorkflowEditor

DEFAULT_INPUT = "example_workflows/experimental/fml2v_var_d_audio_loop.json"
SMOKE_ITERATIONS = 2
TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "_node_templates_fml2v.json"


@functools.lru_cache(maxsize=1)
def _templates() -> dict[str, dict]:
    return json.loads(TEMPLATES_PATH.read_text())


def _add_from_template(
    ed: WorkflowEditor,
    type_name: str,
    pos: tuple[int, int],
    *,
    widget_values: list | None = None,
    title: str | None = None,
    size: tuple[int, int] = (270, 100),
    mode: int = 0,
) -> int:
    """Insert a new top-level node from the saved template JSON.

    Slot dicts copy from template, but every input ``link`` is reset to None
    and every output ``links`` to an empty list — caller wires after.
    """
    tmpl = _templates()[type_name]
    node_id = ed.next_node_id()
    inputs = []
    for inp in tmpl.get("inputs", []):
        inp_copy = {k: v for k, v in inp.items() if k != "link"}
        inp_copy["link"] = None
        inputs.append(inp_copy)
    outputs = []
    for out in tmpl.get("outputs", []):
        out_copy = {k: v for k, v in out.items() if k != "links"}
        out_copy["links"] = []
        outputs.append(out_copy)
    node = {
        "id": node_id,
        "type": type_name,
        "pos": list(pos),
        "size": list(size),
        "flags": {},
        "order": 0,
        "mode": mode,
        "inputs": inputs,
        "outputs": outputs,
        "properties": dict(tmpl.get("properties", {})),
        "widgets_values": list(widget_values) if widget_values is not None else (
            list(tmpl.get("widgets_values") or [])
        ),
    }
    if title:
        node["title"] = title
    ed.add_node(node)
    return node_id


def phase5_stash(ed: WorkflowEditor) -> dict:
    """Return `wf["properties"]["build_fml2v_phase5"]` or SystemExit."""
    stash = ed.wf.get("properties", {}).get("build_fml2v_phase5", {})
    if not stash:
        raise SystemExit(
            "build_fml2v_phase5 stash missing — workflow wasn't built by "
            "scripts/build_fml2v_audio_loop.py. Run the builder first."
        )
    return stash


def apply_smoke_iters_config(
    ed: WorkflowEditor, tlo_id: int, *, iterations: int = SMOKE_ITERATIONS,
) -> None:
    """Unwire TensorLoopOpen.iterations_in and set widget[1] = iterations."""
    tlo = ed.find_node(tlo_id)
    iter_slot = WorkflowEditor.find_input_slot(tlo, "iterations_in")
    existing = ed.find_link_to_slot(tlo_id, iter_slot)
    if existing is not None:
        ed.remove_link(existing[0])
        print(f"  TensorLoopOpen #{tlo_id}.iterations_in: unwired (was link {existing[0]})")
    wv = tlo.get("widgets_values") or []
    while len(wv) < 2:
        wv.append(0)
    old_iters = wv[1]
    wv[1] = iterations
    tlo["widgets_values"] = wv
    print(f"  TensorLoopOpen #{tlo_id}.widgets[1] (iterations): {old_iters} -> {iterations}")


def smoke_iters_applied(tlo: dict, *, iterations: int = SMOKE_ITERATIONS) -> bool:
    """Idempotence check for `apply_smoke_iters_config`."""
    iter_input = next((i for i in tlo.get("inputs", [])
                       if i.get("name") == "iterations_in"), None)
    iter_unwired = iter_input is not None and iter_input.get("link") is None
    widget_short = (tlo.get("widgets_values") or [None, None])[1] == iterations
    return iter_unwired and widget_short


def find_by_type_and_title(ed: WorkflowEditor, type_name: str, title: str) -> dict | None:
    """Standard idempotence-marker lookup for variant-staged adds."""
    for n in ed.wf.get("nodes", []):
        if n.get("type") == type_name and n.get("title") == title:
            return n
    return None


def stage_variant(
    input_path: Path,
    output_path: Path,
    *,
    apply_fn: Callable[[WorkflowEditor], None],
    already_toggled_fn: Callable[[WorkflowEditor], bool],
    dry_run: bool,
    variant_label: str,
    next_steps: list[str] | None = None,
) -> None:
    """Canonical migrate-or-skip scaffold for an apply script's output path."""
    if not input_path.exists():
        raise SystemExit(f"Input workflow missing: {input_path}")

    if output_path.exists() and input_path != output_path:
        ed_existing = WorkflowEditor(output_path)
        if already_toggled_fn(ed_existing):
            print(f"{output_path.name}: already toggled, skipping. Run --revert to reset.")
            return

    if dry_run:
        ed = WorkflowEditor(input_path)
        print(f"would copy {input_path} -> {output_path}")
        print(f"would apply {variant_label} ops:")
        apply_fn(ed)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    print(f"  copied {input_path} -> {output_path}")

    ed = WorkflowEditor(output_path)
    if already_toggled_fn(ed):
        print(f"{output_path.name}: already toggled (from input), skipping mutations.")
        return

    apply_fn(ed)
    ed.save()
    print(f"  wrote {output_path}")
    if next_steps:
        print()
        print("Next steps:")
        for line in next_steps:
            print(f"  {line}")


def revert_variant(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def make_argparser(description: str | None, default_output: str, default_input: str = DEFAULT_INPUT) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=default_input)
    ap.add_argument("--output", default=default_output)
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be copied/changed without writing.")
    return ap
