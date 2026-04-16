"""Build the keyframe-conditioned workflow from the latent base workflow.

Copies audio-loop-music-video_latent.json and rewires:
  - Removes constant Get_input_image -> subgraph init_image link
  - Adds KeyframeImageSchedule node (receives current_iteration + stride_seconds)
  - Adds ImageBlend node (receives image + next_image + blend_factor)
  - Wires ImageBlend output -> subgraph init_image (slot 8)
  - Keeps Get_input_image -> LTXVImgToVideoInplaceKJ (initial render) intact

Usage:
    uv run python scripts/build_keyframe_workflow.py
"""

import shutil
from pathlib import Path

from workflow_utils import WorkflowEditor

SRC = Path(__file__).resolve().parent.parent / "example_workflows" / "audio-loop-music-video_latent.json"
DST = Path(__file__).resolve().parent.parent / "example_workflows" / "audio-loop-music-video_latent_keyframe.json"

# Copy the base workflow
shutil.copy2(SRC, DST)
ed = WorkflowEditor(DST)

# --- Key existing node IDs (from workflow analysis) ---
SUBGRAPH = 843          # Extension subgraph component
GET_INPUT_IMAGE = 651   # GetNode "Get_input_image" -> constant IMAGE
TENSOR_LOOP_OPEN = 1539 # TensorLoopOpen (output slot 3 = current_iteration)
AUDIO_LOOP_CTRL = 1582  # AudioLoopController (output slot 4 = stride_seconds)

# --- Remove the constant Get_input_image -> subgraph link ---
link_id = ed.find_link(GET_INPUT_IMAGE, SUBGRAPH)
assert link_id is not None, f"No link found from node {GET_INPUT_IMAGE} to node {SUBGRAPH}"
ed.remove_link(link_id)
print(f"Removed link {link_id} (Get_input_image -> subgraph init_image)")

# Get_input_image (node 651) still exists and feeds the initial render via
# LTXVImgToVideoInplaceKJ (node 531 input slot 2). That wiring stays.

# --- Position new nodes near the loop control area ---
# TensorLoopOpen is at ~[3280, 4400], AudioLoopController at ~[3650, 4700]
# Place KeyframeImageSchedule and ImageBlend in that region

# --- Add KeyframeImageSchedule node ---
kf_id = ed.next_node_id()
kf_node = ed.make_node(
    node_id=kf_id,
    node_type="KeyframeImageSchedule",
    pos=[2700, 4100],
    title="Keyframe Image Schedule",
    widgets=[
        "0:00+: 0",
        0.0,
    ],
    inputs=[
        {"name": "images", "type": "IMAGE", "link": None},
        {"name": "current_iteration", "type": "INT", "link": None,
         "widget": {"name": "current_iteration"}},
        {"name": "stride_seconds", "type": "FLOAT", "link": None,
         "widget": {"name": "stride_seconds"}},
        {"name": "schedule", "type": "STRING", "link": None,
         "widget": {"name": "schedule"}},
        {"name": "blend_seconds", "type": "FLOAT", "link": None,
         "widget": {"name": "blend_seconds"}},
    ],
    outputs=[
        {"name": "image", "type": "IMAGE", "links": []},
        {"name": "next_image", "type": "IMAGE", "links": []},
        {"name": "blend_factor", "type": "FLOAT", "links": []},
        {"name": "current_time", "type": "FLOAT", "links": []},
        {"name": "image_index", "type": "INT", "links": []},
    ],
)
kf_node["size"] = [340, 200]
kf_node["properties"] = {
    "cnr_id": "comfyui-audioloophelper",
    "Node name for S&R": "KeyframeImageSchedule",
}
ed.add_node(kf_node)
print(f"Added KeyframeImageSchedule node {kf_id}")

# --- Add ImageBlend node ---
blend_id = ed.next_node_id()
blend_node = ed.make_node(
    node_id=blend_id,
    node_type="ImageBlend_AudioLoop",
    pos=[3100, 4100],
    title="Image Blend",
    widgets=[
        0.0,  # blend_factor default
    ],
    inputs=[
        {"name": "image_a", "type": "IMAGE", "link": None},
        {"name": "image_b", "type": "IMAGE", "link": None},
        {"name": "blend_factor", "type": "FLOAT", "link": None,
         "widget": {"name": "blend_factor"}},
    ],
    outputs=[
        {"name": "image", "type": "IMAGE", "links": []},
    ],
)
blend_node["size"] = [270, 120]
blend_node["properties"] = {
    "cnr_id": "comfyui-audioloophelper",
    "Node name for S&R": "ImageBlend_AudioLoop",
}
ed.add_node(blend_node)
print(f"Added ImageBlend node {blend_id}")

# --- Wire: Get_input_image -> KeyframeImageSchedule.images ---
# Get_input_image provides the preprocessed image batch (for now, single image;
# user adds more LoadImage -> ImageBatch nodes to feed multiple keyframes)
ed.add_link(GET_INPUT_IMAGE, 0, kf_id, 0, "IMAGE")
print(f"Wired Get_input_image -> KeyframeImageSchedule.images")

# --- Wire: TensorLoopOpen.current_iteration -> KeyframeImageSchedule ---
ed.add_link(TENSOR_LOOP_OPEN, 3, kf_id, 1, "INT")
print(f"Wired TensorLoopOpen.current_iteration -> KeyframeImageSchedule")

# --- Wire: AudioLoopController.stride_seconds -> KeyframeImageSchedule ---
ed.add_link(AUDIO_LOOP_CTRL, 4, kf_id, 2, "FLOAT")
print(f"Wired AudioLoopController.stride_seconds -> KeyframeImageSchedule")

# --- Wire: KeyframeImageSchedule outputs -> ImageBlend inputs ---
ed.add_link(kf_id, 0, blend_id, 0, "IMAGE")  # image -> image_a
print(f"Wired KeyframeImageSchedule.image -> ImageBlend.image_a")

ed.add_link(kf_id, 1, blend_id, 1, "IMAGE")  # next_image -> image_b
print(f"Wired KeyframeImageSchedule.next_image -> ImageBlend.image_b")

ed.add_link(kf_id, 2, blend_id, 2, "FLOAT")  # blend_factor -> blend_factor
print(f"Wired KeyframeImageSchedule.blend_factor -> ImageBlend.blend_factor")

# --- Wire: ImageBlend.image -> subgraph #843 slot 8 (init_image) ---
ed.add_link(blend_id, 0, SUBGRAPH, 8, "IMAGE")
print(f"Wired ImageBlend.image -> subgraph #843 init_image (slot 8)")

# --- Save ---
ed.save()
print(f"\nDone! Workflow saved to {DST}")
print(f"  Nodes: {len(ed.wf['nodes'])}")
print(f"  Links: {len(ed.wf['links'])}")
print("\nTo use:")
print("  1. Load multiple keyframe images (LoadImage nodes)")
print("  2. Batch them (ImageBatch node) -> wire to KeyframeImageSchedule.images")
print("  3. Edit the schedule widget to map timestamps to image indices")
print("  4. Set blend_seconds > 0 for smooth transitions (or 0.0 for hard cuts)")
