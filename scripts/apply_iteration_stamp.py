"""Insert LoopIterationStamp between the patch-chain MODEL and the loop
body's MODEL input. Idempotent: safe to re-run.

Wires:
    (existing model source)         -> LoopIterationStamp.model
    TensorLoopOpen.current_iteration -> LoopIterationStamp.current_iteration
    LoopIterationStamp.model         -> subgraph-invoker[model slot]

Leaves the initial-render MODEL path untouched (the initial sampler
does not need an iteration stamp; the loop-body sampler does).

Usage:
    uv run python scripts/apply_iteration_stamp.py [workflow_path]

Defaults to example_workflows/audio-loop-music-video_latent.json. Pass a
path to stage on a scratch workflow first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from workflow_utils import WorkflowEditor  # noqa: E402

DEFAULT_WF = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"
STAMP_NODE_TYPE = "LoopIterationStamp"
TENSORLOOP_TYPE = "TensorLoopOpen"
CURRENT_ITER_OUTPUT_SLOT = 3  # TensorLoopOpen.current_iteration (0 flow, 1 prev, 2 acc, 3 iter)


def _already_applied(ed: WorkflowEditor, invoker: dict, model_slot: int) -> bool:
    """True iff a fully-wired LoopIterationStamp sits between model source
    and invoker. Verifies all three links: source->stamp[0], TensorLoopOpen
    ->stamp[1], stamp[0]->invoker[model_slot]. A partial apply (crashed
    mid-surgery) reports as not-applied so re-running will complete it."""
    link = ed.find_link_to_slot(invoker["id"], model_slot)
    if link is None:
        return False
    src_node = ed.find_node(link[1])
    if src_node.get("type") != STAMP_NODE_TYPE:
        return False
    stamp_id = src_node["id"]
    iter_link = ed.find_link_to_slot(stamp_id, 1)
    if iter_link is None:
        return False
    iter_src = ed.find_node(iter_link[1])
    return iter_src.get("type") == TENSORLOOP_TYPE


def apply(wf_path: Path) -> bool:
    ed = WorkflowEditor(wf_path)

    invoker = ed.find_subgraph_invoker()
    if invoker is None:
        raise RuntimeError(f"{wf_path.name}: no subgraph invoker found.")
    model_slot = ed.find_input_slot(invoker, "model")

    if _already_applied(ed, invoker, model_slot):
        print(f"  already applied: {wf_path.name}")
        return False

    # Locate the existing MODEL link feeding the invoker.
    model_link = ed.find_link_to_slot(invoker["id"], model_slot)
    if model_link is None:
        raise RuntimeError(f"{wf_path.name}: no link feeding invoker[{model_slot}].")
    model_link_id, src_node_id, src_slot, _, _, dtype = model_link
    if dtype != "MODEL":
        raise RuntimeError(f"{wf_path.name}: expected MODEL dtype on invoker[{model_slot}], got {dtype}.")

    # Locate TensorLoopOpen.
    tensorloop = next(iter(ed.find_nodes_by_type(TENSORLOOP_TYPE)), None)
    if tensorloop is None:
        raise RuntimeError(f"{wf_path.name}: no {TENSORLOOP_TYPE} node found.")

    # Place the stamp next to the invoker for readability.
    inv_pos = invoker.get("pos", [0, 0])
    stamp_pos = [inv_pos[0] - 300, inv_pos[1] - 40]

    stamp_id = ed.add_top_level_node(
        node_type=STAMP_NODE_TYPE,
        pos=stamp_pos,
        size=[260, 78],
        inputs=[
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "current_iteration", "type": "INT", "link": None, "widget": {"name": "current_iteration"}},
        ],
        outputs=[
            {"name": "model", "type": "MODEL", "links": []},
        ],
        widgets_values=[0],
        title="Loop Iteration Stamp",
    )

    # Rewire: remove the direct source->invoker link, then wire source->stamp and stamp->invoker.
    ed.remove_link(model_link_id)
    ed.add_link(src_node_id, src_slot, stamp_id, 0, "MODEL")
    ed.add_link(tensorloop["id"], CURRENT_ITER_OUTPUT_SLOT, stamp_id, 1, "INT")
    ed.add_link(stamp_id, 0, invoker["id"], model_slot, "MODEL")

    ed.save()
    print(f"  inserted LoopIterationStamp (id={stamp_id}) in {wf_path.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow",
        nargs="?",
        default=str(DEFAULT_WF),
        help="Workflow JSON to modify (default: audio-loop-music-video_latent.json).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Apply to every example_workflows/audio-loop-music-video_*.json that has a subgraph + TensorLoopOpen.",
    )
    args = parser.parse_args()

    if args.all:
        targets = sorted((REPO_ROOT / "example_workflows").glob("audio-loop-music-video_*.json"))
    else:
        targets = [Path(args.workflow)]

    for target in targets:
        if not target.exists():
            print(f"  missing: {target}")
            continue
        apply(target)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
