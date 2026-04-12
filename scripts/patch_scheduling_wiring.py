"""Wire TimestampPromptSchedule + ConditioningBlend into all example workflows.

Adds:
1. A second CLIPTextEncode for next_prompt (from TimestampPromptSchedule output 1)
2. A ConditioningBlend node that blends current + next prompt conditionings
3. Wires blend_factor from TimestampPromptSchedule output 2

Rewires:
- Extension subgraph input 6 (positive conditioning) from static GetNode 1588
  to ConditioningBlend output

Run: python scripts/patch_scheduling_wiring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from workflow_utils import WorkflowEditor

WORKFLOW_DIR = Path(__file__).parent.parent / "example_workflows"
WORKFLOWS = [
    "audio-loop-music-video_image.json",
    "audio-loop-music-video_latent.json",
    "audio-loop-music-video_image_adain_perstep.json",
]

# Known node IDs (identical across all 3 workflows)
NODE_TPS = 1558        # TimestampPromptSchedule
NODE_ENCODE_A = 1559   # CLIPTextEncode (current prompt - already exists)
NODE_CLIP = 416        # DualCLIPLoader (shared CLIP model)
NODE_STATIC_GET = 1588 # GetNode "base_cond_pos" (to be disconnected from 843)
NODE_SUBGRAPH = 843    # Extension subgraph
SUBGRAPH_POS_SLOT = 6  # Extension input slot for positive conditioning


def patch_workflow(path: Path) -> bool:
    """Add ConditioningBlend wiring to a single workflow. Returns True if modified."""
    ed = WorkflowEditor(path)
    name = path.name

    # Check if already patched (ConditioningBlend exists)
    existing = ed.find_nodes_by_type("ConditioningBlend")
    if existing:
        print(f"  {name}: already has ConditioningBlend (node {existing[0]['id']}), skipping")
        return False

    # Verify expected nodes exist
    try:
        ed.find_node(NODE_TPS)
        ed.find_node(NODE_ENCODE_A)
        ed.find_node(NODE_CLIP)
        ed.find_node(NODE_SUBGRAPH)
    except ValueError as e:
        print(f"  {name}: missing expected node: {e}")
        return False

    # Get node 1559 position for placement reference
    n1559 = ed.find_node(NODE_ENCODE_A)
    x_base = n1559["pos"][0]
    y_base = n1559["pos"][1]

    # --- Step 1: Create second CLIPTextEncode for next_prompt ---
    encode_b_id = ed.next_node_id()
    encode_b = {
        "id": encode_b_id,
        "type": "CLIPTextEncode",
        "pos": [x_base, y_base + 120],  # Below node 1559
        "size": [300, 88],
        "flags": {},
        "order": 91,
        "mode": 0,
        "inputs": [
            {"name": "clip", "type": "CLIP", "link": None},
            {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None},
        ],
        "outputs": [
            {"name": "CONDITIONING", "type": "CONDITIONING", "links": []},
        ],
        "title": "Next Prompt Encode",
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.18.5",
            "Node name for S&R": "CLIPTextEncode",
        },
        "widgets_values": ["(next prompt - auto-filled by TimestampPromptSchedule)"],
    }
    ed.add_node(encode_b)

    # Wire CLIP input from DualCLIPLoader 416
    ed.add_link(NODE_CLIP, 0, encode_b_id, 0, "CLIP")

    # Wire text input from TimestampPromptSchedule output 1 (next_prompt)
    lid_next_prompt = ed.add_link(NODE_TPS, 1, encode_b_id, 1, "STRING")
    # Fix TPS output 1 links field (was None)
    tps = ed.find_node(NODE_TPS)
    tps["outputs"][1]["links"] = [lid_next_prompt]

    # --- Step 2: Create ConditioningBlend node ---
    blend_id = ed.next_node_id()
    blend_node = {
        "id": blend_id,
        "type": "ConditioningBlend",
        "pos": [x_base + 340, y_base + 60],  # Right of the encode nodes
        "size": [270, 100],
        "flags": {},
        "order": 92,
        "mode": 0,
        "inputs": [
            {"name": "conditioning_a", "type": "CONDITIONING", "link": None},
            {"name": "conditioning_b", "type": "CONDITIONING", "link": None},
            {"name": "blend_factor", "type": "FLOAT", "widget": {"name": "blend_factor"}, "link": None},
        ],
        "outputs": [
            {"name": "CONDITIONING", "type": "CONDITIONING", "links": []},
        ],
        "title": "Prompt Blend",
        "properties": {
            "cnr_id": "comfyui-audioloophelper",
            "Node name for S&R": "ConditioningBlend",
        },
        "widgets_values": [0.0],
    }
    ed.add_node(blend_node)

    # Wire conditioning_a from CLIPTextEncode 1559 (current prompt)
    ed.add_link(NODE_ENCODE_A, 0, blend_id, 0, "CONDITIONING")

    # Wire conditioning_b from new CLIPTextEncode (next prompt)
    ed.add_link(encode_b_id, 0, blend_id, 1, "CONDITIONING")

    # Wire blend_factor from TimestampPromptSchedule output 2
    lid_blend = ed.add_link(NODE_TPS, 2, blend_id, 2, "FLOAT")
    # Fix TPS output 2 links field (was None)
    tps["outputs"][2]["links"] = [lid_blend]

    # --- Step 3: Rewire Extension subgraph input 6 ---

    # Find and remove the old static link (1588 -> 843 slot 6)
    old_link = ed.find_link(NODE_STATIC_GET, NODE_SUBGRAPH)
    if old_link:
        ed.remove_link(old_link)
        print(f"  Removed static link {old_link} (GetNode 1588 -> 843 slot 6)")
    else:
        print(f"  Warning: no link found from 1588 to 843")

    # Add new link from ConditioningBlend output to Extension input 6
    ed.add_link(blend_id, 0, NODE_SUBGRAPH, SUBGRAPH_POS_SLOT, "CONDITIONING")

    # --- Save ---
    ed.save()
    print(f"  {name}: patched (encode_b={encode_b_id}, blend={blend_id})")
    return True


def main():
    print("Patching scheduling wiring in all workflows...\n")

    patched = 0
    for wf_name in WORKFLOWS:
        path = WORKFLOW_DIR / wf_name
        if not path.exists():
            print(f"  {wf_name}: NOT FOUND, skipping")
            continue
        print(f"Processing {wf_name}:")
        if patch_workflow(path):
            patched += 1

    print(f"\nDone. Patched {patched} workflows.")

    # Validate
    print("\nValidating...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "internal" / "scripts"))
    from test_workflow_integrity import validate

    errors_found = False
    for wf_name in WORKFLOWS:
        path = WORKFLOW_DIR / wf_name
        if path.exists():
            errors = validate(str(path))
            if errors:
                print(f"  FAIL {wf_name}:")
                for e in errors:
                    print(f"    {e}")
                errors_found = True
            else:
                print(f"  OK {wf_name}")

    if errors_found:
        print("\nValidation FAILED")
        sys.exit(1)
    else:
        print("\nAll validations passed")


if __name__ == "__main__":
    main()
