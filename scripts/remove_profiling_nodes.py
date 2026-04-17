"""Remove ProfileBegin / ProfileIterStep / ProfileEnd from the latent workflow.

Inverse of `scripts/apply_profiling_nodes.py`. Idempotent: running against a
workflow without the profile nodes is a no-op.

Restores the original link topology:
  - ProfileBegin (top-level): original source is reconnected directly to
    the consumer that ProfileBegin intercepted.
  - ProfileEnd (top-level): same treatment.
  - ProfileIterStep (inside subgraph #843): IterationCleanup's output goes
    directly to the subgraph output slot again.

Usage:
    uv run python scripts/remove_profiling_nodes.py
"""

from pathlib import Path

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"


def _remove_top_level_profile_node(ed: WorkflowEditor, node_type: str) -> bool:
    """Remove a single top-level profile node, bridging src -> consumer.

    Finds the one node of `node_type`, locates its inbound trigger link and
    its outbound trigger link, removes the node + both links, and restitches
    a direct link from the original source to the original consumer.
    Returns True if a node was removed.
    """
    matches = ed.find_nodes_by_type(node_type)
    if not matches:
        return False

    for node in matches:
        nid = node["id"]

        in_link = next((l for l in ed.find_links_to(nid) if l[4] == 0), None)
        out_link = next((l for l in ed.find_links_from(nid) if l[2] == 0), None)

        # Capture original source/target before mutating.
        src_node = in_link[1] if in_link else None
        src_slot = in_link[2] if in_link else None
        tgt_node = out_link[3] if out_link else None
        tgt_slot = out_link[4] if out_link else None
        dtype = (in_link[5] if in_link else None) or (out_link[5] if out_link else "*")

        if in_link:
            ed.remove_link(in_link[0])
        if out_link:
            ed.remove_link(out_link[0])

        # Remove the node from the top-level nodes list.
        ed.wf["nodes"] = [n for n in ed.wf["nodes"] if n["id"] != nid]

        # Restitch direct link src -> tgt. src_node/tgt_node and their
        # slots are always paired (both come from the same link entry).
        if src_node is not None and tgt_node is not None:
            assert src_slot is not None and tgt_slot is not None
            ed.add_link(src_node, src_slot, tgt_node, tgt_slot, dtype)

    return True


def _remove_profile_iter_step(ed: WorkflowEditor) -> bool:
    """Remove ProfileIterStep_AudioLoop from subgraph #843, restoring the
    direct IterationCleanup -> subgraph-output link.
    Returns True if a node was removed.
    """
    sg = ed.get_subgraph(0)
    if sg is None:
        return False

    step_node = next(
        (n for n in sg["nodes"] if n["type"] == "ProfileIterStep_AudioLoop"),
        None,
    )
    if step_node is None:
        return False

    step_id = step_node["id"]

    # Inbound (IterationCleanup -> step.slot0) and outbound (step.slot0 -> output).
    in_link = next(
        (l for l in sg["links"]
         if l["target_id"] == step_id and l["target_slot"] == 0),
        None,
    )
    out_link = next(
        (l for l in sg["links"]
         if l["origin_id"] == step_id and l["origin_slot"] == 0),
        None,
    )
    if in_link is None or out_link is None:
        # Mangled state; bail gracefully.
        sg["nodes"] = [n for n in sg["nodes"] if n["id"] != step_id]
        return True

    src_id = in_link["origin_id"]
    src_slot = in_link["origin_slot"]
    tgt_id = out_link["target_id"]
    tgt_slot = out_link["target_slot"]
    dtype = in_link.get("type", "LATENT")

    old_in_id = in_link["id"]
    old_out_id = out_link["id"]

    # Remove both links + the node.
    sg["links"] = [l for l in sg["links"] if l["id"] not in (old_in_id, old_out_id)]
    sg["nodes"] = [n for n in sg["nodes"] if n["id"] != step_id]

    # Clean up the source node's output slot entry (strip the dead link id).
    for n in sg["nodes"]:
        if n["id"] == src_id:
            outs = n.get("outputs", [])
            if src_slot < len(outs):
                outs[src_slot]["links"] = [
                    lid for lid in outs[src_slot].get("links", [])
                    if lid not in (old_in_id, old_out_id)
                ]
            break

    # Clean up the output-boundary entry (subgraph sg["outputs"]).
    if tgt_id == -20:
        sg_outputs = sg.get("outputs", [])
        if tgt_slot < len(sg_outputs):
            sg_outputs[tgt_slot]["linkIds"] = [
                lid for lid in sg_outputs[tgt_slot].get("linkIds", [])
                if lid not in (old_in_id, old_out_id)
            ]

    # New direct link from source to subgraph output.
    new_link_id = ed.next_link_id()
    sg["links"].append({
        "id": new_link_id,
        "origin_id": src_id,
        "origin_slot": src_slot,
        "target_id": tgt_id,
        "target_slot": tgt_slot,
        "type": dtype,
    })

    # Register the new link on source output.
    for n in sg["nodes"]:
        if n["id"] == src_id:
            outs = n.get("outputs", [])
            if src_slot < len(outs):
                links_list = outs[src_slot].get("links") or []
                links_list.append(new_link_id)
                outs[src_slot]["links"] = links_list
            break

    # Register on subgraph output boundary.
    if tgt_id == -20:
        sg_outputs = sg.get("outputs", [])
        if tgt_slot < len(sg_outputs):
            link_ids = sg_outputs[tgt_slot].get("linkIds") or []
            link_ids.append(new_link_id)
            sg_outputs[tgt_slot]["linkIds"] = link_ids

    return True


def unpatch_workflow(path: Path) -> None:
    print(f"\n=== {path.name} ===")
    ed = WorkflowEditor(path)

    begin_removed = _remove_top_level_profile_node(ed, "ProfileBegin_AudioLoop")
    print(f"  ProfileBegin: {'removed' if begin_removed else 'not present'}")

    end_removed = _remove_top_level_profile_node(ed, "ProfileEnd_AudioLoop")
    print(f"  ProfileEnd: {'removed' if end_removed else 'not present'}")

    iter_removed = _remove_profile_iter_step(ed)
    print(f"  ProfileIterStep: {'removed' if iter_removed else 'not present'}")

    ed.save()


def main() -> None:
    if not WORKFLOW.exists():
        print(f"Workflow not found: {WORKFLOW}")
        return
    unpatch_workflow(WORKFLOW)


if __name__ == "__main__":
    main()
