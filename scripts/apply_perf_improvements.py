"""Apply Step 0 performance improvements to audio-loop workflows.

Two in-place modifications per workflow:
  1. Swap the two in-loop CLIPTextEncode nodes (titled "Loop Prompt Encode"
     and "Next Prompt Encode") to CachedTextEncode_AudioLoop. Outputs and
     wiring are unchanged -- same (CLIP, STRING) -> CONDITIONING signature.
  2. (Latent workflows only) Insert an IterationCleanup node inside subgraph
     #843 between the final LatentOverlapTrim and the subgraph output.
     Image workflow gets no IterationCleanup because its subgraph output is
     IMAGE-typed; it's tested and stable so we don't want a typed swap there.

The script is idempotent: running twice on the same workflow leaves it
unchanged after the first run (detects already-patched nodes).

Usage:
    uv run python scripts/apply_perf_improvements.py
"""

from pathlib import Path

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = [
    REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json",
    REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent_keyframe.json",
    REPO_ROOT / "example_workflows" / "audio-loop-music-video_image.json",
]

IN_LOOP_ENCODE_TITLES = {"Loop Prompt Encode", "Next Prompt Encode"}


def swap_clip_text_encodes(ed: WorkflowEditor) -> tuple[int, int]:
    """Swap in-loop CLIPTextEncode nodes to CachedTextEncode_AudioLoop.

    Returns (swapped_count, candidate_count) where candidate_count is the
    total number of CLIPTextEncode nodes found. When candidate_count > 0 but
    swapped_count == 0, titles don't match IN_LOOP_ENCODE_TITLES -- caller
    should warn.
    """
    swapped = 0
    candidates = ed.find_nodes_by_type("CLIPTextEncode")
    for n in candidates:
        if n.get("title") in IN_LOOP_ENCODE_TITLES:
            n["type"] = "CachedTextEncode_AudioLoop"
            props = n.setdefault("properties", {})
            props["Node name for S&R"] = "CachedTextEncode_AudioLoop"
            props["cnr_id"] = "comfyui-audioloophelper"
            # Custom nodes don't carry an upstream "ver" like comfy-core does.
            props.pop("ver", None)
            swapped += 1
    return swapped, len(candidates)


def insert_iteration_cleanup(ed: WorkflowEditor) -> bool:
    """Insert IterationCleanup after LatentOverlapTrim in the extension
    subgraph. Returns True if inserted, False if already present or not
    applicable (e.g., image workflow has no LatentOverlapTrim).
    """
    sg = ed.get_subgraph(0)
    if sg is None:
        return False

    # Idempotency: skip if IterationCleanup already in subgraph
    for n in sg["nodes"]:
        if n["type"] == "IterationCleanup":
            return False

    # Find LatentOverlapTrim
    trim_node = None
    for n in sg["nodes"]:
        if n["type"] == "LatentOverlapTrim":
            trim_node = n
            break
    if trim_node is None:
        return False  # image workflow has no LatentOverlapTrim

    # Find the single outgoing LATENT link from trim_node slot 0 to output (-20)
    trim_out_link = None
    for l in sg["links"]:
        if (l.get("origin_id") == trim_node["id"]
                and l.get("origin_slot") == 0
                and l.get("target_id") == -20):
            trim_out_link = l
            break
    if trim_out_link is None:
        raise RuntimeError(
            f"Expected exactly one link from LatentOverlapTrim (id={trim_node['id']}) "
            f"slot 0 to output collector (-20); found none."
        )

    output_target_slot = trim_out_link["target_slot"]
    old_link_id = trim_out_link["id"]

    new_node_id = max((n["id"] for n in sg["nodes"]), default=0) + 1
    link_id_trim_to_cleanup = ed.next_link_id()
    link_id_cleanup_to_output = ed.next_link_id()

    trim_pos = trim_node.get("pos", [0, 0])
    pos = [trim_pos[0] + 280, trim_pos[1]]

    cleanup_node = ed.make_node(
        node_id=new_node_id,
        node_type="IterationCleanup",
        pos=pos,
        widgets=["always"],
        inputs=[
            {"name": "latent", "type": "LATENT", "link": link_id_trim_to_cleanup},
            {"name": "mode", "type": "COMBO", "link": None,
             "widget": {"name": "mode"}},
        ],
        outputs=[
            {"name": "latent", "type": "LATENT", "links": [link_id_cleanup_to_output]},
        ],
    )
    cleanup_node["size"] = [240, 80]
    cleanup_node["order"] = trim_node.get("order", 0) + 1
    cleanup_node["properties"] = {
        "cnr_id": "comfyui-audioloophelper",
        "Node name for S&R": "IterationCleanup",
    }
    sg["nodes"].append(cleanup_node)

    # Remove the old direct link from trim to output.
    sg["links"] = [l for l in sg["links"] if l["id"] != old_link_id]

    # Node-output slots use "links" (not "linkIds") -- "linkIds" is only for
    # subgraph boundary entries (sg.inputs / sg.outputs). Update only "links"
    # on the trim node's output slot.
    trim_out = trim_node["outputs"][0]
    trim_out["links"] = [
        lid for lid in trim_out.get("links", []) if lid != old_link_id
    ] + [link_id_trim_to_cleanup]

    # Subgraph output entries (the virtual target_id == -20 sink) use "linkIds".
    sg_output = sg["outputs"][output_target_slot]
    sg_output["linkIds"] = [
        lid if lid != old_link_id else link_id_cleanup_to_output
        for lid in sg_output.get("linkIds", [])
    ]

    # Add the two new internal links (dict format for subgraph links)
    sg["links"].append({
        "id": link_id_trim_to_cleanup,
        "origin_id": trim_node["id"],
        "origin_slot": 0,
        "target_id": new_node_id,
        "target_slot": 0,
        "type": "LATENT",
    })
    sg["links"].append({
        "id": link_id_cleanup_to_output,
        "origin_id": new_node_id,
        "origin_slot": 0,
        "target_id": -20,
        "target_slot": output_target_slot,
        "type": "LATENT",
    })

    return True


def patch_workflow(path: Path) -> None:
    print(f"\n=== {path.name} ===")
    ed = WorkflowEditor(path)

    already_patched = ed.find_nodes_by_type("CachedTextEncode_AudioLoop")
    swapped, candidates = swap_clip_text_encodes(ed)
    if swapped:
        print(f"  Swapped {swapped} CLIPTextEncode -> CachedTextEncode_AudioLoop")
    elif already_patched:
        print(f"  Already patched ({len(already_patched)} CachedTextEncode_AudioLoop nodes)")
    elif candidates:
        # CLIPTextEncode nodes exist, none have an IN_LOOP_ENCODE_TITLES title,
        # and no CachedTextEncode_AudioLoop has taken their place. Likely the
        # user renamed the in-loop encoders.
        print(
            f"  WARN: {candidates} CLIPTextEncode node(s) found but none match "
            f"IN_LOOP_ENCODE_TITLES={sorted(IN_LOOP_ENCODE_TITLES)}. "
            f"Rename the in-loop prompt encoders to enable caching, or edit "
            f"IN_LOOP_ENCODE_TITLES in this script."
        )
    else:
        print("  No CLIPTextEncode nodes present")

    inserted = insert_iteration_cleanup(ed)
    if inserted:
        print("  Inserted IterationCleanup after LatentOverlapTrim")
    else:
        print("  IterationCleanup: skipped (no LatentOverlapTrim or already present)")

    ed.save()


def main() -> None:
    for p in WORKFLOWS:
        if not p.exists():
            print(f"  SKIP {p.name}: does not exist")
            continue
        patch_workflow(p)


if __name__ == "__main__":
    main()
