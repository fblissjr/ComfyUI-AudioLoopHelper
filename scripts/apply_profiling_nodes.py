"""Insert ProfileBegin / ProfileIterStep / ProfileEnd into the latent workflow.

Idempotent. Does three insertions:

  1. ProfileBegin: intercepts the LATENT link into TensorLoopOpen.initial_value
     (Reroute 1606 -> TensorLoopOpen 1539). Runs at workflow start.
  2. ProfileIterStep: inside subgraph #843, intercepts the link from
     IterationCleanup (2007) -> subgraph output. Marks iteration boundary.
  3. ProfileEnd: intercepts TensorLoopClose (1540) -> LatentConcat (1605).
     Runs at workflow end, writes trace/summary/memory_timeline to
     `internal/analysis/runs/profiler/<timestamp>/` by default (gitignored).

Default: enabled=True so the workflow demonstrates profiling when run. Toggle
off via the `enabled` widget on ProfileBegin, or right-click bypass any of
the three profile nodes.

Usage:
    uv run python scripts/apply_profiling_nodes.py [workflow.json]
"""

import sys
from pathlib import Path

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOW = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"

TENSOR_LOOP_OPEN = 1539
TENSOR_LOOP_CLOSE = 1540
SG_NODE_SUBGRAPH = 843  # not used directly -- for context


def _insert_top_level_passthrough(
    ed: WorkflowEditor,
    node_type: str,
    node_id_source_link: int,
    pos: list,
    widgets: list,
    title: str,
) -> bool:
    """Intercept a top-level link with an AnyType passthrough node.

    Finds the link with id == node_id_source_link, removes it, creates
    the new node, and wires source -> new_node -> original_target.
    Returns True if inserted, False if already patched (link gone).
    """
    orig_link = next(
        (l for l in ed.wf["links"] if isinstance(l, list) and l[0] == node_id_source_link),
        None,
    )
    if orig_link is None:
        return False

    _, src_node, src_slot, tgt_node, tgt_slot, dtype = orig_link

    new_node_id = ed.next_node_id()
    new_node = ed.make_node(
        node_id=new_node_id,
        node_type=node_type,
        pos=pos,
        widgets=widgets,
        title=title,
        inputs=[
            {"name": "trigger", "type": "*", "link": None},
        ],
        outputs=[
            {"name": "trigger", "type": "*", "links": []},
        ],
    )
    new_node["size"] = [320, 220 if widgets else 80]
    new_node["properties"] = {
        "cnr_id": "comfyui-audioloophelper",
        "Node name for S&R": node_type,
    }
    ed.add_node(new_node)

    # Remove old direct link, add two new ones
    ed.remove_link(node_id_source_link)
    ed.add_link(src_node, src_slot, new_node_id, 0, dtype)
    ed.add_link(new_node_id, 0, tgt_node, tgt_slot, dtype)
    return True


def _insert_profile_iter_step(ed: WorkflowEditor) -> bool:
    """Insert ProfileIterStep inside subgraph #843, intercepting the link
    from IterationCleanup (2007) to the subgraph output (-20).
    Returns True if inserted, False if already present or not applicable.
    """
    sg = ed.get_subgraph(0)
    if sg is None:
        return False

    for n in sg["nodes"]:
        if n["type"] == "ProfileIterStep_AudioLoop":
            return False

    cleanup_node = next((n for n in sg["nodes"] if n["type"] == "IterationCleanup"), None)
    if cleanup_node is None:
        return False

    # Link: IterationCleanup.slot0 -> output(-20)
    orig = next(
        (l for l in sg["links"]
         if l.get("origin_id") == cleanup_node["id"]
         and l.get("origin_slot") == 0
         and l.get("target_id") == -20),
        None,
    )
    if orig is None:
        return False

    old_link_id = orig["id"]
    output_target_slot = orig["target_slot"]

    cleanup_pos = cleanup_node.get("pos", [0, 0])
    new_node_id = ed.add_subgraph_node(
        node_type="ProfileIterStep_AudioLoop",
        pos=[cleanup_pos[0] + 280, cleanup_pos[1]],
        size=[220, 58],
        inputs=[
            {"name": "latent", "type": "LATENT", "link": None},
        ],
        outputs=[
            {"name": "latent", "type": "LATENT", "links": []},
        ],
        properties={
            "cnr_id": "comfyui-audioloophelper",
            "Node name for S&R": "ProfileIterStep_AudioLoop",
        },
        widgets_values=[],
        title="Profile Iter Step",
    )

    ed.remove_subgraph_link(old_link_id)
    ed.add_subgraph_link(cleanup_node["id"], 0, new_node_id, 0, "LATENT")
    ed.add_subgraph_link(new_node_id, 0, -20, output_target_slot, "LATENT")

    return True


def patch_workflow(path: Path) -> None:
    print(f"\n=== {path.name} ===")
    ed = WorkflowEditor(path)

    # Idempotency check for ProfileBegin
    if ed.find_nodes_by_type("ProfileBegin_AudioLoop"):
        print("  ProfileBegin: already present, skipping top-level insertions")
    else:
        # Find the link feeding TensorLoopOpen.initial_value
        tlo_initial_link = None
        for l in ed.wf["links"]:
            if isinstance(l, list) and l[3] == TENSOR_LOOP_OPEN and l[4] == 0:
                tlo_initial_link = l[0]
                break
        if tlo_initial_link is None:
            print("  ProfileBegin: could not find TensorLoopOpen.initial_value link, skipping")
        else:
            inserted = _insert_top_level_passthrough(
                ed,
                node_type="ProfileBegin_AudioLoop",
                node_id_source_link=tlo_initial_link,
                pos=[2600, 4400],
                widgets=[
                    True,                                   # enabled
                    "internal/analysis/runs/profiler",      # output_dir (plugin-relative, gitignored)
                    1,                                      # warmup_iterations
                    3,                                      # active_iterations
                    True,                                   # include_cpu
                    True,                                   # include_memory
                    True,                                   # include_shapes
                    False,                                  # include_flops
                ],
                title="Profile Begin",
            )
            print(f"  ProfileBegin: {'inserted' if inserted else 'skipped'}")

        # ProfileEnd: intercept TensorLoopClose output
        tlc_out_link = None
        for l in ed.wf["links"]:
            if isinstance(l, list) and l[1] == TENSOR_LOOP_CLOSE and l[2] == 0:
                tlc_out_link = l[0]
                break
        if tlc_out_link is None:
            print("  ProfileEnd: could not find TensorLoopClose.output link, skipping")
        else:
            inserted = _insert_top_level_passthrough(
                ed,
                node_type="ProfileEnd_AudioLoop",
                node_id_source_link=tlc_out_link,
                pos=[3900, 5000],
                widgets=[],
                title="Profile End",
            )
            print(f"  ProfileEnd: {'inserted' if inserted else 'skipped'}")

    # ProfileIterStep inside subgraph
    inserted = _insert_profile_iter_step(ed)
    print(f"  ProfileIterStep: {'inserted' if inserted else 'skipped (already present or no IterationCleanup)'}")

    ed.save()


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKFLOW
    if not path.exists():
        print(f"Workflow not found: {path}")
        return
    patch_workflow(path)


if __name__ == "__main__":
    main()
