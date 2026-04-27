"""apply_frame_planner_consolidation.

Last updated: 2026-04-27

Migrate the latent workflow to use LTXFramePlanner as the single source of
truth for dimension config. Removes scattered widget values + 4 helper
nodes; wires 6 consumer-node inputs to flow from the planner.

What this migration does:

  1. Add LTXFramePlanner node with widget defaults that reproduce current
     behavior:
        target_width=832, target_height=448
        target_seconds=20.0  (snaps to 19.88s @ 25fps -> 497 frames)
        fps=25

  2. Wire LTXFramePlanner outputs to 6 consumer-node inputs:
        EmptyLTXVLatentVideo(344).width      <- width
        EmptyLTXVLatentVideo(344).height     <- height
        EmptyLTXVLatentVideo(344).length     <- frames
        ImageResizeKJv2(445).width           <- width
        ImageResizeKJv2(445).height          <- height
        LTXVConditioning(164).frame_rate     <- fps_float
        AudioLoopController(1582).window_seconds  <- actual_seconds
        AudioLoopController(1582).fps        <- fps_int
        AudioLoopPlanner(1560).window_seconds     <- actual_seconds
        AudioLoopPlanner(1560).fps           <- fps_int
        subgraph(843).video_end_time         <- actual_seconds

  3. Remove now-redundant nodes:
        FloatConstant(688)  "window_size_seconds"
        SetNode(689)        "Set_window_size_seconds"
        GetNode(691)        "Get_window_size_seconds"
        PrimitiveNode(526)  "length"

Net node count: 102 -> 99 (-3 nodes; +1 planner -4 helpers).

The benefit isn't node count — it's:
  - User edits ONE node for dimension config (was 5 places)
  - LTXFramePlanner snaps to LTX-valid neighborhood automatically
  - summary STRING output reports what was used (visible to user)
  - no more (L-1)%8 footgun, no more length-vs-window_seconds drift

Source-level audit (background subagent run 2026-04-27) verified all
6 consumer inputs are wireable and the wire supersedes widget value at
runtime. EmptyLTXVLatentVideo silently floors invalid lengths internally
(((L-1)//8)+1) so our snap-on-input is required upstream — confirmed
by reading comfy_extras/nodes_lt.py:36.

Compatibility:
  - Independent of all other apply scripts (LoRA chain, ID-LoRA runtime,
    canonical sigmas, F2/F3/F4/F5/F6/F7 invariants).
  - Only touches the latent workflow by default. Other workflows can be
    migrated by passing their path as the workflow argument.

Usage:
    uv run --group dev python scripts/apply_frame_planner_consolidation.py
    uv run --group dev python scripts/apply_frame_planner_consolidation.py --revert
    uv run --group dev python scripts/apply_frame_planner_consolidation.py --dry-run

Idempotent. Re-run reports "no change". To revert: `git checkout` the
workflow JSON from a pre-migration commit. The migration restructures
6 wires + removes 4 helper nodes; restoring the exact pre-state via
JSON edit is too brittle to ship as an apply-script revert. (Most other
apply scripts in this repo DO ship --revert; this one is the exception
because its rewire shape is irreversible without slot-index gymnastics.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOW = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"

# Existing node IDs (verified for the latent workflow; other workflows
# may differ — script aborts cleanly if any expected node is missing).
EMPTY_LATENT_VIDEO_ID = 344
IMAGE_RESIZE_ID = 445
LTXV_CONDITIONING_INITIAL_ID = 164
AUDIO_LOOP_CONTROLLER_ID = 1582
AUDIO_LOOP_PLANNER_ID = 1560
SUBGRAPH_ID = 843
SUBGRAPH_VIDEO_END_TIME_SLOT = 5

# Nodes to remove (after rewiring their downstream consumers through
# LTXFramePlanner instead).
FLOATCONSTANT_WINDOW_SECONDS_ID = 688
SETNODE_WINDOW_SECONDS_ID = 689
GETNODE_WINDOW_SECONDS_ID = 691
PRIMITIVE_LENGTH_ID = 526

# Title marker for the new node (idempotence signal).
PLANNER_TITLE = "LTX Frame Planner (single source for dimension config)"

# Default widget values reproduce current effective behavior (832x448,
# 19.88s snapped from target 20.0, 25fps).
DEFAULT_TARGET_WIDTH = 832
DEFAULT_TARGET_HEIGHT = 448
DEFAULT_TARGET_SECONDS = 20.0  # snaps DOWN to 19.88s @ 25fps -> 497 frames
DEFAULT_FPS = 25

# Layout: place the planner near the top of the canvas where the user
# will look first. Positioned to the left of the resolution + audio nodes
# so its outputs sweep right toward consumers.
_PLANNER_X = -2300
_PLANNER_Y = 4500


def _node_exists(ed: WorkflowEditor, nid: int) -> bool:
    return any(n.get("id") == nid for n in ed.wf.get("nodes", []))


def _find_planner(ed: WorkflowEditor) -> dict | None:
    for n in ed.wf.get("nodes", []):
        if n.get("type") == "LTXFramePlanner" and n.get("title") == PLANNER_TITLE:
            return n
    return None


def _add_planner(ed: WorkflowEditor) -> int:
    return ed.add_top_level_node(
        node_type="LTXFramePlanner",
        pos=[_PLANNER_X, _PLANNER_Y],
        size=[420, 220],
        inputs=[
            {"name": "target_width", "type": "INT",
             "widget": {"name": "target_width"}, "link": None},
            {"name": "target_height", "type": "INT",
             "widget": {"name": "target_height"}, "link": None},
            {"name": "target_seconds", "type": "FLOAT",
             "widget": {"name": "target_seconds"}, "link": None},
            {"name": "fps", "type": "INT",
             "widget": {"name": "fps"}, "link": None},
        ],
        outputs=[
            {"name": "width", "type": "INT", "links": []},
            {"name": "height", "type": "INT", "links": []},
            {"name": "frames", "type": "INT", "links": []},
            {"name": "actual_seconds", "type": "FLOAT", "links": []},
            {"name": "fps_int", "type": "INT", "links": []},
            {"name": "fps_float", "type": "FLOAT", "links": []},
            {"name": "latent_volume", "type": "INT", "links": []},
            {"name": "status", "type": "STRING", "links": []},
            {"name": "summary", "type": "STRING", "links": []},
        ],
        widgets_values=[
            DEFAULT_TARGET_WIDTH,
            DEFAULT_TARGET_HEIGHT,
            DEFAULT_TARGET_SECONDS,
            DEFAULT_FPS,
        ],
        properties={"Node name for S&R": "LTXFramePlanner"},
        title=PLANNER_TITLE,
    )


def _wire_by_name(ed: WorkflowEditor, *, src_node: int, src_slot: int,
                  tgt_node: int, tgt_input_name: str, dtype: str) -> None:
    """Wire src_node.out[src_slot] -> tgt_node.in[<tgt_input_name>], handling
    the widget-vs-wired-input asymmetry in saved JSON.

    Saved workflow JSONs only include input entries that were either wired
    OR rendered visibly in the UI at save time; pure widget-only inputs are
    sometimes absent from inputs[]. Wiring such an input requires APPENDING
    a new entry with proper widget metadata, then adding the link.
    """
    node = ed.find_node(tgt_node)
    inputs = node.setdefault("inputs", [])
    inp = next((i for i in inputs if i.get("name") == tgt_input_name), None)
    if inp is None:
        inp = {
            "name": tgt_input_name,
            "type": dtype,
            "widget": {"name": tgt_input_name},
            "link": None,
        }
        inputs.append(inp)
    tgt_slot = inputs.index(inp)

    if inp.get("link") is not None:
        ed.remove_link(inp["link"])

    ed.add_link(src_node, src_slot, tgt_node, tgt_slot, dtype)


def _apply(ed: WorkflowEditor) -> tuple[bool, str]:
    if _find_planner(ed) is not None:
        return False, "no change (LTXFramePlanner already present)"

    # Verify all expected source nodes exist
    missing = ed.require_nodes((
        EMPTY_LATENT_VIDEO_ID, IMAGE_RESIZE_ID,
        LTXV_CONDITIONING_INITIAL_ID,
        AUDIO_LOOP_CONTROLLER_ID, AUDIO_LOOP_PLANNER_ID,
        SUBGRAPH_ID,
    ))
    if missing:
        return False, f"skip (missing required nodes: {missing})"

    actions: list[str] = []
    planner_id = _add_planner(ed)
    actions.append(f"added #{planner_id} LTXFramePlanner")

    # Map planner outputs -> consumers, wiring by INPUT NAME (not slot
    # index, which varies between saved-JSON state and live schema for
    # widget-vs-wired inputs).
    # Planner output slot order matches define_schema:
    #   0:width  1:height  2:frames  3:actual_seconds  4:fps_int  5:fps_float
    #   6:latent_volume  7:status  8:summary

    wires = [
        # (tgt_node, tgt_input_name, planner_out_slot, dtype, label)
        (EMPTY_LATENT_VIDEO_ID, "width",  0, "INT",   "EmptyLTXVLatentVideo.width"),
        (EMPTY_LATENT_VIDEO_ID, "height", 1, "INT",   "EmptyLTXVLatentVideo.height"),
        (EMPTY_LATENT_VIDEO_ID, "length", 2, "INT",   "EmptyLTXVLatentVideo.length"),
        (IMAGE_RESIZE_ID,       "width",  0, "INT",   "ImageResizeKJv2.width"),
        (IMAGE_RESIZE_ID,       "height", 1, "INT",   "ImageResizeKJv2.height"),
        (LTXV_CONDITIONING_INITIAL_ID, "frame_rate", 5, "FLOAT", "LTXVConditioning.frame_rate"),
        (AUDIO_LOOP_CONTROLLER_ID, "window_seconds", 3, "FLOAT", "AudioLoopController.window_seconds"),
        (AUDIO_LOOP_CONTROLLER_ID, "fps",            4, "INT",   "AudioLoopController.fps"),
        (AUDIO_LOOP_PLANNER_ID,    "window_seconds", 3, "FLOAT", "AudioLoopPlanner.window_seconds"),
        (AUDIO_LOOP_PLANNER_ID,    "fps",            4, "INT",   "AudioLoopPlanner.fps"),
    ]
    for tgt, name, src_slot, dtype, label in wires:
        _wire_by_name(ed, src_node=planner_id, src_slot=src_slot,
                      tgt_node=tgt, tgt_input_name=name, dtype=dtype)
        actions.append(label)

    # subgraph slot 5 (video_end_time) is by-index; subgraphs use slot
    # numbers directly (the input list IS the schema).
    existing = ed.find_link_to_slot(SUBGRAPH_ID, SUBGRAPH_VIDEO_END_TIME_SLOT)
    if existing is not None:
        ed.remove_link(existing[0])
    ed.add_link(planner_id, 3, SUBGRAPH_ID, SUBGRAPH_VIDEO_END_TIME_SLOT, "FLOAT")
    actions.append("subgraph.video_end_time")

    # Remove the now-redundant helper nodes
    for nid in (
        FLOATCONSTANT_WINDOW_SECONDS_ID,
        SETNODE_WINDOW_SECONDS_ID,
        GETNODE_WINDOW_SECONDS_ID,
        PRIMITIVE_LENGTH_ID,
    ):
        if _node_exists(ed, nid):
            ed.remove_node_and_links(nid)
            actions.append(f"removed #{nid}")

    return True, "; ".join(actions)


def apply(dry_run: bool, wf_path: Path) -> int:
    wf_path = wf_path.resolve()
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        print(f"load error: {e}")
        return 1

    changed, message = _apply(ed)
    prefix = "would " if dry_run and changed else ""
    try:
        rel = wf_path.relative_to(REPO_ROOT)
    except ValueError:
        rel = wf_path
    print(f"  {rel}:")
    for line in message.split("; "):
        print(f"    {prefix}{line}")
    if changed and not dry_run:
        ed.save(wf_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("workflow", nargs="?", default=str(DEFAULT_WORKFLOW))
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing.")
    args = ap.parse_args()
    return apply(args.dry_run, Path(args.workflow))


if __name__ == "__main__":
    sys.exit(main())
