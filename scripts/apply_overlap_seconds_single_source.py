"""apply_overlap_seconds_single_source.

Last updated: 2026-05-04

Eliminates the AudioLoopController ↔ AudioLoopPlanner overlap_seconds
divergence footgun by routing both through a shared FloatConstant.

Symptom: changing `overlap_seconds` on the controller updates loop
geometry but the planner's iteration-count summary still shows the
old value. Worse — if the user updates the planner widget thinking
it'll change the loop, nothing happens (controller drives the loop,
planner is informational). The two widgets have no enforced parity.

Why we can't just wire controller.overlap_seconds → planner directly:
F7 audit invariant. The planner's `total_iterations` already feeds
`TensorLoopOpen.iterations_in`, which feeds back to the controller's
`current_iteration` via `TensorLoopOpen.current_iteration`. A direct
controller→planner edge on overlap_seconds closes that cycle and
ComfyUI's prompt validator rejects the workflow.

Fix: introduce a shared `FloatConstant` titled `overlap_seconds` with
no upstream inputs. Wire it to both the controller and the planner.
No cycle since the FloatConstant has no incoming edges.

Topology added (per workflow):
  FloatConstant("overlap_seconds", default=2.0)
    ├─→ AudioLoopController.overlap_seconds (slot appended to inputs)
    └─→ AudioLoopPlanner.overlap_seconds    (slot appended to inputs)

The widgets_values positional slot for overlap_seconds remains in each
node's widgets_values array (link supersedes widget at runtime, but
ComfyUI reads widgets_values positionally so we leave it intact).

Targets: every shipped workflow that has both AudioLoopController +
AudioLoopPlanner (i.e. every non-retake variant).

Usage:
    uv run --group dev python scripts/apply_overlap_seconds_single_source.py
    uv run --group dev python scripts/apply_overlap_seconds_single_source.py --revert
    uv run --group dev python scripts/apply_overlap_seconds_single_source.py --dry-run

Idempotent. `--revert` removes the FloatConstant + the two wires +
the new input slots, restoring widget-only state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

OVERLAP_NODE_TITLE = "overlap_seconds"
OVERLAP_DEFAULT = 2.0

DEFAULT_TARGETS = [
    "example_workflows/audio-loop-music-video_latent.json",
    "example_workflows/audio-loop-music-video_latent_keyframe.json",
    "example_workflows/experimental/init_guide_amplification_poc.json",
    # retake.json has no AudioLoopController/Planner; skipped
    # spectrogram_iclora_minimal.json + iclora_amplification_poc.json
    # don't have a loop controller/planner — skipped
    # _stg / _validator / _image_adain_perstep archived 2026-05-25 (migration baked in)
]


def _find_first_by_type(ed: WorkflowEditor, node_type: str) -> dict | None:
    """Thin wrapper over `WorkflowEditor.find_nodes_by_type` for the
    'expect at most one' shape."""
    return next(iter(ed.find_nodes_by_type(node_type)), None)


def _find_overlap_constant(ed: WorkflowEditor) -> dict | None:
    """The shared FloatConstant we add. Identified by title."""
    for n in ed.wf.get("nodes") or []:
        if isinstance(n, dict) and n.get("type") == "FloatConstant" \
                and n.get("title") == OVERLAP_NODE_TITLE:
            return n
    return None


def _ensure_overlap_input(node: dict, *, name: str = "overlap_seconds") -> int:
    """Make sure the node's `overlap_seconds` widget is exposed as an
    input. Returns the input slot index. Appends if missing; preserves
    if already present."""
    inputs = node.setdefault("inputs", [])
    for i, inp in enumerate(inputs):
        if inp.get("name") == name:
            return i
    inputs.append({
        "name": name,
        "type": "FLOAT",
        "widget": {"name": name},
        "link": None,
    })
    return len(inputs) - 1


def _is_applied(ed: WorkflowEditor) -> bool:
    """Applied iff both AudioLoopController and AudioLoopPlanner have a
    wired `overlap_seconds` input. Doesn't matter which FloatConstant
    they come from — what matters is they share a source. Tightened
    after a workflow with a pre-existing overlap-seconds FC wired to
    a different second consumer (LoopConfigValidator) was incorrectly
    marked applied. Source-parity is enforced separately by the audit
    check `overlap_seconds_single_source`."""
    controller = _find_first_by_type(ed, "AudioLoopController")
    planner = _find_first_by_type(ed, "AudioLoopPlanner")
    if controller is None or planner is None:
        return False
    for n in (controller, planner):
        ovr = next(
            (i for i in n.get("inputs", []) if i.get("name") == "overlap_seconds"),
            None,
        )
        if ovr is None or ovr.get("link") is None:
            return False
    return True


def _apply(ed: WorkflowEditor) -> tuple[bool, list[str]]:
    if _is_applied(ed):
        return False, ["already applied"]

    controller = _find_first_by_type(ed, "AudioLoopController")
    planner = _find_first_by_type(ed, "AudioLoopPlanner")
    if controller is None:
        return False, ["AudioLoopController missing — skipping"]
    if planner is None:
        return False, ["AudioLoopPlanner missing — skipping"]

    actions: list[str] = []

    fc = _find_overlap_constant(ed)
    if fc is None:
        # Place near the existing FloatConstant (#1269 in canonical) at
        # roughly the same x, slightly offset y. Cosmetic only — the
        # layout pass in apply_intro_workflow.py doesn't classify this
        # node, so it'll sit at the source position when intro rebuilds.
        # Pick coords near the controller for visual proximity.
        cx, cy = controller["pos"][0], controller["pos"][1]
        fc_id = ed.add_top_level_node(
            node_type="FloatConstant",
            pos=[cx + 320, cy - 110],
            size=[220, 58],
            inputs=[],
            outputs=[{"name": "value", "type": "FLOAT", "links": []}],
            widgets_values=[OVERLAP_DEFAULT],
            properties={
                "cnr_id": "comfyui-kjnodes",
                "Node name for S&R": "FloatConstant",
            },
            title=OVERLAP_NODE_TITLE,
        )
        actions.append(f"added FloatConstant #{fc_id} (overlap_seconds = {OVERLAP_DEFAULT})")
        fc = ed.find_node(fc_id)

    fc_id = fc["id"]

    # Make sure both consumers have an overlap_seconds input slot, then
    # wire to the FloatConstant. We check `input.link` (the per-node
    # field) rather than `find_link_to_slot` (the global links array) —
    # workflows with prior partial-state edits can have stale link
    # records in `wf["links"]` that reference targets the inputs no
    # longer claim. The input's own `.link` is the source of truth.
    for label, consumer in (("AudioLoopController", controller),
                            ("AudioLoopPlanner", planner)):
        slot = _ensure_overlap_input(consumer)
        if consumer["inputs"][slot].get("link") is None:
            ed.add_link(fc_id, 0, consumer["id"], slot, "FLOAT")
            actions.append(
                f"wired #{fc_id} → {label}(#{consumer['id']})."
                f"overlap_seconds (slot {slot})"
            )

    return bool(actions), actions or ["no-op"]


def _revert(ed: WorkflowEditor) -> tuple[bool, list[str]]:
    fc = _find_overlap_constant(ed)
    if fc is None:
        return False, ["not applied (no overlap_seconds FloatConstant)"]

    actions: list[str] = []
    fc_id = fc["id"]

    # Remove links and the corresponding input slots on the consumers.
    for consumer_type in ("AudioLoopController", "AudioLoopPlanner"):
        consumer = _find_first_by_type(ed, consumer_type)
        if consumer is None:
            continue
        inputs = consumer.get("inputs", [])
        slot = next(
            (i for i, inp in enumerate(inputs) if inp.get("name") == "overlap_seconds"),
            None,
        )
        if slot is None:
            continue
        link = ed.find_link_to_slot(consumer["id"], slot)
        if link is not None:
            ed.remove_link(link[0])
        inputs.pop(slot)
        actions.append(
            f"removed overlap_seconds input on {consumer_type}(#{consumer['id']})"
        )

    # Remove the FloatConstant itself.
    ed.remove_node_and_links(fc_id)
    actions.append(f"removed FloatConstant #{fc_id}")
    return True, actions


def _process(target: Path, *, revert: bool, dry_run: bool) -> None:
    if not target.exists():
        print(f"  skip (missing): {target}")
        return

    ed = WorkflowEditor(target)

    if dry_run:
        applied = _is_applied(ed)
        verb = "would revert" if revert else "would apply"
        marker = "applied" if applied else "not applied"
        print(f"  {target.name}: {verb} (currently {marker})")
        return

    op = _revert if revert else _apply
    mutated, actions = op(ed)
    if mutated:
        ed.save()
        for a in actions:
            print(f"  {target.name}: {a}")
    else:
        print(f"  {target.name}: {actions[0]} (no-op)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--target", action="append", default=None,
                    help=f"Workflow JSON (repeatable). Default: {DEFAULT_TARGETS}")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = [Path(t) if Path(t).is_absolute() else REPO_ROOT / t
               for t in (args.target or DEFAULT_TARGETS)]

    for t in targets:
        _process(t, revert=args.revert, dry_run=args.dry_run)

    if not args.dry_run and not args.revert:
        print()
        print("Next steps:")
        print("  1. Audit: uv run --group dev python scripts/audit_workflows.py")
        print("  2. Rebuild intro: uv run --group dev python scripts/apply_intro_workflow.py "
              "--revert && uv run --group dev python scripts/apply_intro_workflow.py")


if __name__ == "__main__":
    main()
