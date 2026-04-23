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
is a no-op. Required source nodes are validated UP FRONT so a partial
migration cannot leave the workflow in an inconsistent state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor


NODES_TO_REMOVE = {
    1608: "ConditioningBlend 'Prompt Blend' -- replaced by selector",
    1607: "CachedTextEncode_AudioLoop 'Next Prompt Encode' -- removed",
    1559: "CachedTextEncode_AudioLoop 'Loop Prompt Encode' -- removed",
    1558: "TimestampPromptSchedule -- replaced by batch encoder",
    575: "Set_guider -- dead (no Get_guider consumers)",
    645: "Set_base_cond_pos -- dead",
    647: "Get_base_cond_pos -- dead (no downstream consumers)",
    1588: "Get_base_cond_pos (Static Mode) -- dead",
}

# Types left over from the legacy schedule/encode chain. Variants gave
# their instances different IDs (e.g. 'Next Prompt Encode' was 1604 in
# _image.json vs 1607 elsewhere), so an ID-based pass alone misses
# stragglers — sweep by type after the main removal.
_LEGACY_ORPHAN_TYPES = ("CachedTextEncode_AudioLoop", "TimestampPromptSchedule")

# Source nodes the new encoder + selector will wire into. All must
# exist before we mutate anything, else partial migration is possible.
CLIP_LOADER_ID = 416           # DualCLIPLoader; slot 0 = CLIP
AUDIO_CONTROLLER_ID = 1582     # AudioLoopController; slot 2=audio_duration, 4=stride_seconds
TENSOR_LOOP_OPEN_ID = 1539     # TensorLoopOpen; slot 3 = current_iteration
SUBGRAPH_INSTANCE_ID = 843     # subgraph node in the main graph
SUBGRAPH_POSITIVE_SLOT = 6     # "positive" CONDITIONING input on the subgraph

REQUIRED_SOURCE_NODES = (
    CLIP_LOADER_ID,
    AUDIO_CONTROLLER_ID,
    TENSOR_LOOP_OPEN_ID,
    SUBGRAPH_INSTANCE_ID,
)


def _schedule_widget_value(ed: WorkflowEditor) -> tuple[str, bool]:
    """Pull schedule text + snap_boundaries from the legacy node 1558.

    Widget order on TimestampPromptSchedule is:
      [current_iteration, stride_seconds, schedule, blend_seconds, snap_boundaries]
    Returns a safe default if the node is absent.
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


def _assert_required_nodes_present(ed: WorkflowEditor) -> None:
    missing = []
    for nid in REQUIRED_SOURCE_NODES:
        try:
            ed.find_node(nid)
        except ValueError:
            missing.append(nid)
    if missing:
        raise SystemExit(
            f"Refusing to migrate: required source node(s) missing: {missing}. "
            "This script assumes the canonical latent workflow layout "
            "(DualCLIPLoader=416, AudioLoopController=1582, "
            "TensorLoopOpen=1539, subgraph instance=843). "
            "If your workflow was edited, update the constants at the top "
            "of this script."
        )


def _remove_nodes_by_type(
    ed: WorkflowEditor, types: tuple[str, ...], reason: str,
) -> list[int]:
    removed: list[int] = []
    for node_type in types:
        for node in list(ed.find_nodes_by_type(node_type)):
            nid = node["id"]
            print(f"  remove {node_type}(id={nid}): {reason}")
            ed.remove_node_and_links(nid)
            removed.append(nid)
    return removed


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
    removed.extend(_remove_nodes_by_type(
        ed, _LEGACY_ORPHAN_TYPES, "orphan after migration",
    ))
    return removed


def _widget_input(name: str, dtype: str) -> dict:
    return {"name": name, "type": dtype, "widget": {"name": name}, "link": None}


def _add_batch_encoder(
    ed: WorkflowEditor,
    *,
    schedule_text: str,
    snap_boundaries: bool,
) -> int:
    """Add TimestampPromptScheduleBatchEncode at top level and wire its
    three non-widget inputs."""
    nid = ed.add_top_level_node(
        node_type="TimestampPromptScheduleBatchEncode",
        pos=[-960, 4780],
        size=[400, 190],
        inputs=[
            {"name": "clip", "type": "CLIP", "link": None},
            _widget_input("schedule", "STRING"),
            _widget_input("stride_seconds", "FLOAT"),
            _widget_input("audio_duration", "FLOAT"),
            _widget_input("snap_boundaries", "BOOLEAN"),
        ],
        outputs=[
            {"name": "conditioning_list", "type": "*", "links": []},
            {"name": "iteration_count", "type": "INT", "links": []},
        ],
        widgets_values=[schedule_text, 17.92, 180.0, snap_boundaries],
        properties={
            "cnr_id": "comfyui-audioloophelper",
            "Node name for S&R": "TimestampPromptScheduleBatchEncode",
        },
        title="Prompt Schedule (Batch Encode)",
    )
    ed.add_link(CLIP_LOADER_ID, 0, nid, 0, "CLIP")
    ed.add_link(AUDIO_CONTROLLER_ID, 4, nid, 2, "FLOAT")
    ed.add_link(AUDIO_CONTROLLER_ID, 2, nid, 3, "FLOAT")
    return nid


def _add_selector(ed: WorkflowEditor, batch_encoder_id: int) -> int:
    """Add ConditioningSelectByIteration and wire it into the subgraph's
    'positive' input slot (replacing the legacy blend output)."""
    nid = ed.add_top_level_node(
        node_type="ConditioningSelectByIteration",
        pos=[-520, 4780],
        size=[290, 78],
        inputs=[
            {"name": "conditioning_list", "type": "*", "link": None},
            _widget_input("current_iteration", "INT"),
        ],
        outputs=[
            {"name": "conditioning", "type": "CONDITIONING", "links": []},
        ],
        widgets_values=[0],
        properties={
            "cnr_id": "comfyui-audioloophelper",
            "Node name for S&R": "ConditioningSelectByIteration",
        },
        title="Conditioning Select (by Iteration)",
    )
    ed.add_link(batch_encoder_id, 0, nid, 0, "*")
    ed.add_link(TENSOR_LOOP_OPEN_ID, 3, nid, 1, "INT")
    ed.add_link(nid, 0, SUBGRAPH_INSTANCE_ID, SUBGRAPH_POSITIVE_SLOT, "CONDITIONING")
    return nid


def rewire(path: Path) -> None:
    ed = WorkflowEditor(path)

    if _already_migrated(ed):
        swept = _remove_nodes_by_type(
            ed, _LEGACY_ORPHAN_TYPES,
            "orphan (missed by ID-based pass in an earlier migration)",
        )
        if swept:
            print(f"{path.name}: cleaned {len(swept)} orphan(s), skipping full migration.")
            ed.save()
        else:
            print(f"{path.name}: already migrated, no orphans, skipping.")
        return

    _assert_required_nodes_present(ed)

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
