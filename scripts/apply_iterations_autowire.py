"""apply_iterations_autowire.

Last updated: 2026-04-26

Symptom it fixes: TensorLoopOpen.iterations is buried inside the loop
subgraph as a `DynamicCombo.Option` widget — manual short tests require
hunting for the node, and any experiment runner has to dig into subgraph
internals to set the iteration count. The widget is also unwireable in
its original form, so the iteration count cannot auto-track audio
length even though `AudioLoopPlanner.total_iterations` already computes
exactly that value.

Root cause: TensorLoopOpen's schema declared `iterations` as a nested
DynamicCombo option, not a top-level wireable input. ComfyUI does not
expose nested DynamicCombo options as input slots; they only appear as
widget controls.

Fix (two parts):

  1. Upstream NativeLooping schema patch — adds an OPTIONAL top-level
     `iterations_in` Int input. When wired and >0, it supersedes the
     mode-widget value. Backward compatible — workflows that don't wire
     it keep using the existing widget. Patched at
     ComfyUI-NativeLooping_testing/nodes.py same date.

  2. This script — for every shipped workflow that has both an
     AudioLoopPlanner and a TensorLoopOpen, wires
     `AudioLoopPlanner.total_iterations -> TensorLoopOpen.iterations_in`
     and appends the new input slot to TensorLoopOpen's inputs list.
     The link is added with `shape: 7` (the optional-input shape) so the
     UI renders it as an optional slot consistent with other
     optional-typed inputs in our graphs.

After running this, default behavior is "iterations auto-matches the
input audio length." For short experiment runs the harness can rewire
the link to an INTConstant; manual short tests can drag in an
INTConstant + edit one widget value (recipe in
docs/guides/debugging_guide.md).

Compatibility:
  - audio-loop-music-video_retake.json has no TensorLoopOpen (retake
    strips loop machinery) → script reports skip.
  - All other 6 shipped workflows already have AudioLoopPlanner present;
    no scaffolding needed.
  - Orthogonal to apply_alc_seed_rename, F2/F3, sage_mode.

Usage:
    uv run --group dev python scripts/apply_iterations_autowire.py
    uv run --group dev python scripts/apply_iterations_autowire.py --revert
    uv run --group dev python scripts/apply_iterations_autowire.py --dry-run

Idempotent. Run repeatedly; already-wired workflows report "no change".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

ITERATIONS_IN_NAME = "iterations_in"
PLANNER_TOTAL_ITERATIONS_SLOT = 1  # AudioLoopPlanner outputs: [summary, total_iterations]
PLANNER_TOTAL_ITERATIONS_NAME = "total_iterations"

# LiteGraph slot shape encoding: 7 marks an input as optional (renders cyan,
# acceptable for unwired graphs). Other optional-typed inputs in our workflow
# JSONs use the same value; consolidate to a WorkflowEditor helper if a
# third apply script needs to append an optional input slot.
_OPTIONAL_INPUT_SHAPE = 7


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    tlo_nodes = ed.find_nodes_by_type("TensorLoopOpen")
    if not tlo_nodes:
        return "skip (no TensorLoopOpen)"
    if len(tlo_nodes) > 1:
        return f"skip (multiple TensorLoopOpen nodes: {[n['id'] for n in tlo_nodes]})"
    tlo = tlo_nodes[0]

    planner_nodes = ed.find_nodes_by_type("AudioLoopPlanner")
    if not planner_nodes:
        return "skip (no AudioLoopPlanner; scaffold one or wire iterations manually)"
    if len(planner_nodes) > 1:
        return f"skip (multiple AudioLoopPlanner nodes: {[n['id'] for n in planner_nodes]})"
    planner = planner_nodes[0]

    # Confirm the planner's output slot we're targeting is still
    # `total_iterations`. Schema drift in AudioLoopPlanner (e.g. a new output
    # inserted at slot 0) would silently misroute the wire otherwise.
    planner_outputs = planner.get("outputs") or []
    if (PLANNER_TOTAL_ITERATIONS_SLOT >= len(planner_outputs)
            or planner_outputs[PLANNER_TOTAL_ITERATIONS_SLOT].get("name") != PLANNER_TOTAL_ITERATIONS_NAME):
        return (f"skip (AudioLoopPlanner({planner['id']}).outputs[{PLANNER_TOTAL_ITERATIONS_SLOT}] "
                f"is not '{PLANNER_TOTAL_ITERATIONS_NAME}'; schema drift?)")

    try:
        slot_idx = WorkflowEditor.find_input_slot(tlo, ITERATIONS_IN_NAME)
        existing = (slot_idx, tlo["inputs"][slot_idx])
    except ValueError:
        existing = None

    if revert:
        if existing is None:
            return "already reverted"
        slot_idx, inp = existing
        link_id = inp.get("link")
        if dry_run:
            return f"would revert (remove TensorLoopOpen({tlo['id']}).inputs[{slot_idx}] iterations_in, drop link {link_id})"
        if link_id is not None:
            try:
                ed.remove_link(link_id)
            except ValueError:
                pass  # link already gone; just drop the input slot
        # Remove the input slot itself.
        tlo["inputs"].pop(slot_idx)
        ed.save(wf_path)
        return f"reverted (removed slot {slot_idx} + link {link_id})"

    # Forward direction.
    if existing is not None:
        slot_idx, inp = existing
        if inp.get("link") is not None:
            return f"no change (already wired: TensorLoopOpen({tlo['id']}).iterations_in <- link {inp['link']})"
        # slot exists but is unwired — wire it up below using the existing slot
        new_slot = slot_idx
        existing_slot_dict = inp
    else:
        new_slot = len(tlo.get("inputs") or [])
        existing_slot_dict = None

    if dry_run:
        return (f"would wire (AudioLoopPlanner({planner['id']}).total_iterations -> "
                f"TensorLoopOpen({tlo['id']}).iterations_in slot {new_slot})")

    if existing_slot_dict is None:
        new_input = {
            "name": ITERATIONS_IN_NAME,
            "type": "INT",
            "shape": _OPTIONAL_INPUT_SHAPE,
            "link": None,
        }
        tlo.setdefault("inputs", []).append(new_input)

    link_id = ed.add_link(
        planner["id"], PLANNER_TOTAL_ITERATIONS_SLOT,
        tlo["id"], new_slot, "INT",
    )
    ed.save(wf_path)
    return (f"wired (AudioLoopPlanner({planner['id']}).total_iterations -> "
            f"TensorLoopOpen({tlo['id']}).iterations_in slot {new_slot}, link {link_id})")


def _iter_workflow_paths():
    yield from sorted(WORKFLOWS_DIR.glob("*.json"))
    yield from sorted((WORKFLOWS_DIR / "experimental").glob("*.json"))


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} apply_iterations_autowire across example_workflows/...")
    fail = 0
    for wf_path in _iter_workflow_paths():
        rel = wf_path.relative_to(REPO_ROOT)
        status = _apply_one(wf_path, revert, dry_run)
        print(f"  {rel}: {status}")
        if status.startswith("load error"):
            fail += 1
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--revert", action="store_true",
                    help="Remove the iterations_in slot + link from each workflow.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without touching files.")
    args = ap.parse_args()
    return apply(args.revert, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
