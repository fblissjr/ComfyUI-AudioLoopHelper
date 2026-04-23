"""Rewire a latent-loop music-video workflow to pre-encode prompts.

Replaces the per-iteration CachedTextEncode + ConditioningBlend chain
(which forced CLIP/DiT offload thrash and silenced NAG on iter 2+) with
a single outside-loop batch encoder plus an in-loop index selector.

Also cleans up dead wiring discovered during the offload-bug
investigation:
  - Set_guider (stored but never Get'd)
  - Set_base_cond_pos + Get_base_cond_pos (stored/retrieved but unused)

Usage:
    uv run --group dev python scripts/apply_batch_encode_fix.py <workflow.json>

The edit is idempotent: re-running on an already-migrated workflow
is a no-op.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor


NODES_TO_REMOVE = {
    # (id, reason) -- order doesn't matter because remove_node_and_links
    # detaches links as it goes.
    1608: "ConditioningBlend 'Prompt Blend' -- replaced by selector",
    1607: "CachedTextEncode_AudioLoop 'Next Prompt Encode' -- removed",
    1559: "CachedTextEncode_AudioLoop 'Loop Prompt Encode' -- removed",
    1558: "TimestampPromptSchedule -- replaced by batch encoder",
    575: "Set_guider -- dead (no Get_guider consumers)",
    645: "Set_base_cond_pos -- dead",
    647: "Get_base_cond_pos -- dead (no downstream consumers)",
    1588: "Get_base_cond_pos (Static Mode) -- dead",
}

# Source slots we will wire into the new batch encoder.
CLIP_LOADER_ID = 416           # DualCLIPLoader; slot 0 = CLIP
AUDIO_CONTROLLER_ID = 1582     # AudioLoopController; slot 2=audio_duration, 4=stride_seconds
TENSOR_LOOP_OPEN_ID = 1539     # TensorLoopOpen; slot 3 = current_iteration
SUBGRAPH_INSTANCE_ID = 843     # subgraph node in the main graph
SUBGRAPH_POSITIVE_SLOT = 6     # "positive" CONDITIONING input on the subgraph


def _schedule_widget_value(ed: WorkflowEditor) -> tuple[str, bool]:
    """Pull schedule text + snap_boundaries from the legacy node 1558.

    Widget order on TimestampPromptSchedule is:
      [current_iteration, stride_seconds, schedule, blend_seconds, snap_boundaries]
    Returns a safe default if the node was already migrated.
    """
    try:
        node = ed.find_node(1558)
    except ValueError:
        return "0:00+: default prompt", True
    widgets = node.get("widgets_values", [])
    schedule = widgets[2] if len(widgets) > 2 else "0:00+: default prompt"
    snap = bool(widgets[4]) if len(widgets) > 4 else True
    return schedule, snap


def _already_migrated(ed: WorkflowEditor) -> bool:
    return bool(ed.find_nodes_by_type("TimestampPromptScheduleBatchEncode"))


def _remove_legacy_nodes(ed: WorkflowEditor) -> list[int]:
    removed: list[int] = []
    for nid, reason in NODES_TO_REMOVE.items():
        try:
            ed.find_node(nid)
        except ValueError:
            continue
        print(f"  remove node {nid}: {reason}")
        ed.remove_node_and_links(nid)
        removed.append(nid)
    return removed


def _add_batch_encoder(
    ed: WorkflowEditor,
    *,
    schedule_text: str,
    snap_boundaries: bool,
) -> int:
    """Add TimestampPromptScheduleBatchEncode at top level.

    Inputs all wired:
      clip           <- DualCLIPLoader(416)  slot 0
      schedule       -> widget (from legacy 1558)
      stride_seconds <- AudioLoopController(1582)  slot 4
      audio_duration <- AudioLoopController(1582)  slot 2
      snap_boundaries -> widget (from legacy 1558)
    """
    nid = ed.next_node_id()
    node = {
        "id": nid,
        "type": "TimestampPromptScheduleBatchEncode",
        "pos": [-960, 4780],
        "size": [400, 190],
        "flags": {},
        "order": 30,
        "mode": 0,
        "inputs": [
            {"name": "clip", "type": "CLIP", "link": None},
            {
                "name": "schedule", "type": "STRING",
                "widget": {"name": "schedule"}, "link": None,
            },
            {
                "name": "stride_seconds", "type": "FLOAT",
                "widget": {"name": "stride_seconds"}, "link": None,
            },
            {
                "name": "audio_duration", "type": "FLOAT",
                "widget": {"name": "audio_duration"}, "link": None,
            },
            {
                "name": "snap_boundaries", "type": "BOOLEAN",
                "widget": {"name": "snap_boundaries"}, "link": None,
            },
        ],
        "outputs": [
            {"name": "conditioning_list", "type": "*", "links": []},
            {"name": "iteration_count", "type": "INT", "links": []},
        ],
        "title": "Prompt Schedule (Batch Encode)",
        "properties": {
            "cnr_id": "comfyui-audioloophelper",
            "Node name for S&R": "TimestampPromptScheduleBatchEncode",
        },
        "widgets_values": [schedule_text, 17.92, 180.0, snap_boundaries],
        "color": "#232",
        "bgcolor": "#353",
    }
    ed.add_node(node)

    ed.add_link(CLIP_LOADER_ID, 0, nid, 0, "CLIP")
    ed.add_link(AUDIO_CONTROLLER_ID, 4, nid, 2, "FLOAT")
    ed.add_link(AUDIO_CONTROLLER_ID, 2, nid, 3, "FLOAT")
    return nid


def _add_selector(ed: WorkflowEditor, batch_encoder_id: int) -> int:
    """Add ConditioningSelectByIteration + its outgoing wire into the
    subgraph's 'positive' input slot (replacing the legacy blend output).
    """
    nid = ed.next_node_id()
    node = {
        "id": nid,
        "type": "ConditioningSelectByIteration",
        "pos": [-520, 4780],
        "size": [290, 78],
        "flags": {},
        "order": 80,
        "mode": 0,
        "inputs": [
            {"name": "conditioning_list", "type": "*", "link": None},
            {
                "name": "current_iteration", "type": "INT",
                "widget": {"name": "current_iteration"}, "link": None,
            },
        ],
        "outputs": [
            {"name": "conditioning", "type": "CONDITIONING", "links": []},
        ],
        "title": "Conditioning Select (by Iteration)",
        "properties": {
            "cnr_id": "comfyui-audioloophelper",
            "Node name for S&R": "ConditioningSelectByIteration",
        },
        "widgets_values": [0],
    }
    ed.add_node(node)

    ed.add_link(batch_encoder_id, 0, nid, 0, "*")
    ed.add_link(TENSOR_LOOP_OPEN_ID, 3, nid, 1, "INT")
    ed.add_link(nid, 0, SUBGRAPH_INSTANCE_ID, SUBGRAPH_POSITIVE_SLOT, "CONDITIONING")
    return nid


def rewire(path: Path) -> None:
    ed = WorkflowEditor(path)

    if _already_migrated(ed):
        print(f"{path.name}: already migrated (batch encoder present), skipping.")
        return

    schedule_text, snap_boundaries = _schedule_widget_value(ed)
    print(f"{path.name}: migrating...")
    print(f"  carrying schedule ({len(schedule_text)} chars) and "
          f"snap_boundaries={snap_boundaries}")

    removed = _remove_legacy_nodes(ed)
    print(f"  removed {len(removed)} legacy node(s)")

    batch_encoder_id = _add_batch_encoder(
        ed, schedule_text=schedule_text, snap_boundaries=snap_boundaries,
    )
    print(f"  added TimestampPromptScheduleBatchEncode as node {batch_encoder_id}")

    selector_id = _add_selector(ed, batch_encoder_id)
    print(f"  added ConditioningSelectByIteration as node {selector_id}")

    ed.save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "workflow", nargs="?",
        default="example_workflows/audio-loop-music-video_latent.json",
        help="Workflow JSON path (default: %(default)s)",
    )
    args = ap.parse_args()
    rewire(Path(args.workflow))


if __name__ == "__main__":
    main()
