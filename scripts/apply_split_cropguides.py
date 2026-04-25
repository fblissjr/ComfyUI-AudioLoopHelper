"""Split LTXVCropGuides into two instances to break the LTX 2.3 loop cycle.

Last updated: 2026-04-25

The canonical loop subgraph wired a single `LTXVCropGuides(655)` such that
its CONDITIONING outputs feed `CFGGuider(644)` (F3) AND its LATENT input
read post-sampling output (via `SeparateAVLatent(596)` / `AdainLatent(2006)`).
Older ComfyUI tolerated the node-internal CONDITIONING/LATENT independence;
recent versions enforce strict cycle detection and reject the dependency
graph at prompt validation, blocking `VHS_VideoCombine` output.

Fix: two instances of the same upstream `LTXVCropGuides` node, each
handling one half of the original responsibility:

  CropGuides(655) — CONDITIONING-only role
    All three inputs from `LTXVAddLatentGuide(1519)` (pre-sampling).
    Outputs feed `CFGGuider(644)` (F3 honored).
    LATENT output is a dead end here.

  CropGuides(2008) — LATENT-only role (new)
    `positive` / `negative` from `LTXVAddLatentGuide(1519)` (read-only,
      satisfies required inputs and provides `num_keyframes`).
    `latent` from `SeparateAVLatent(596).video_latent` (post-sampling,
      video-only — no `NestedTensor` problem).
    LATENT output feeds `AdainLatent(2006)` -> `LatentOverlapTrim` -> output.
    CONDITIONING outputs are dead ends here.

No new node code needed. Each instance only depends on upstream of itself,
so no cycle. F3 honored. Post-sampling keyframe-padding cropping preserved
(the LATENT-only instance does the same crop the original wiring did).
Color correction intact (AdainLatent active).

Audit pairing: `audit_workflows.py::cropguides_split_topology` ERRs if
the split topology is damaged (per CLAUDE.md "bake topology constraints
into audit").

Usage:
    uv run --group dev python scripts/apply_split_cropguides.py
    uv run --group dev python scripts/apply_split_cropguides.py --revert
    uv run --group dev python scripts/apply_split_cropguides.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent

ADD_LATENT_GUIDE_ID = 1519
SEPARATE_AV_ID = 596
ADAIN_ID = 2006
CROP_COND_ID = 655
SUBGRAPH_INDEX = 0

SPLIT_NODE_TITLE = "CropGuides (LATENT-only — split)"

# Workflows that don't have a loop subgraph (retake) or intentionally
# exercise edge-case wiring (validator) are skipped.
SKIP_FILES = {
    "audio-loop-music-video_retake.json",
    "audio-loop-music-video_latent_validator.json",
}


def _find_workflow_files() -> list[Path]:
    paths = list((REPO_ROOT / "example_workflows").glob("*.json"))
    paths += list((REPO_ROOT / "example_workflows" / "experimental").glob("*.json"))
    return sorted(p for p in paths if p.name not in SKIP_FILES)


def _find_split_node(sg: dict) -> dict | None:
    for n in sg.get("nodes", []):
        if n.get("type") == "LTXVCropGuides" and n.get("title") == SPLIT_NODE_TITLE:
            return n
    return None


def _apply(ed: WorkflowEditor) -> str:
    sg = ed.get_subgraph(SUBGRAPH_INDEX)
    if sg is None:
        return "skip (no subgraph)"

    sg_node_ids = {n["id"] for n in sg.get("nodes", [])}
    required = {ADD_LATENT_GUIDE_ID, SEPARATE_AV_ID, ADAIN_ID, CROP_COND_ID}
    missing = required - sg_node_ids
    if missing:
        return f"skip (missing subgraph nodes {sorted(missing)})"

    if _find_split_node(sg) is not None:
        return "no change (already split)"

    crop_cond = ed.find_subgraph_node(CROP_COND_ID, SUBGRAPH_INDEX)
    addguide = ed.find_subgraph_node(ADD_LATENT_GUIDE_ID, SUBGRAPH_INDEX)
    sepav = ed.find_subgraph_node(SEPARATE_AV_ID, SUBGRAPH_INDEX)
    adain = ed.find_subgraph_node(ADAIN_ID, SUBGRAPH_INDEX)
    assert crop_cond and addguide and sepav and adain  # by `missing` check above

    # Repoint CropGuides(655).latent input to AddLatentGuide(1519).slot 2
    # (pre-sampling, video-only). The CONDITIONING-only role no longer needs
    # the post-sampling latent.
    ed.rewire_subgraph_input(
        CROP_COND_ID, 2, ADD_LATENT_GUIDE_ID, 2, "LATENT", SUBGRAPH_INDEX,
    )

    new_node_id = ed.next_node_id()
    sg["nodes"].append({
        "id": new_node_id,
        "type": "LTXVCropGuides",
        "pos": [crop_cond["pos"][0] + 400, crop_cond["pos"][1] + 200],
        "size": [240, 86],
        "flags": {},
        "order": crop_cond.get("order", 70) + 1,
        "mode": 0,
        "inputs": [
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "latent", "type": "LATENT", "link": None},
        ],
        "outputs": [
            {"name": "positive", "type": "CONDITIONING", "linkIds": []},
            {"name": "negative", "type": "CONDITIONING", "linkIds": []},
            {"name": "latent", "type": "LATENT", "linkIds": []},
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "Node name for S&R": "LTXVCropGuides",
        },
        "title": SPLIT_NODE_TITLE,
    })
    ed.add_subgraph_link(ADD_LATENT_GUIDE_ID, 0, new_node_id, 0, "CONDITIONING", SUBGRAPH_INDEX)
    ed.add_subgraph_link(ADD_LATENT_GUIDE_ID, 1, new_node_id, 1, "CONDITIONING", SUBGRAPH_INDEX)
    ed.add_subgraph_link(SEPARATE_AV_ID, 0, new_node_id, 2, "LATENT", SUBGRAPH_INDEX)

    # Hand off AdainLatent's samples input from CropGuides(655) to the new node.
    adain_samples_link = ed.find_subgraph_link_to_slot(ADAIN_ID, 0, SUBGRAPH_INDEX)
    if adain_samples_link is None:
        return f"WARN: AdainLatent({ADAIN_ID}) has no samples input link to rewire"
    ed.rewire_subgraph_input(ADAIN_ID, 0, new_node_id, 2, "LATENT", SUBGRAPH_INDEX)

    return f"split applied — added LTXVCropGuides #{new_node_id} (LATENT-only)"


def _revert(ed: WorkflowEditor) -> str:
    sg = ed.get_subgraph(SUBGRAPH_INDEX)
    if sg is None:
        return "skip (no subgraph)"
    split_node = _find_split_node(sg)
    if split_node is None:
        return "no change (not split)"

    new_node_id = split_node["id"]

    # Restore AdainLatent.samples to read from CropGuides(655).slot 2
    ed.rewire_subgraph_input(ADAIN_ID, 0, CROP_COND_ID, 2, "LATENT", SUBGRAPH_INDEX)

    # Restore CropGuides(655).latent input to read from SeparateAV(596).video_latent
    ed.rewire_subgraph_input(CROP_COND_ID, 2, SEPARATE_AV_ID, 0, "LATENT", SUBGRAPH_INDEX)

    # Drop the split node and any links touching it.
    sg["nodes"] = [n for n in sg["nodes"] if n["id"] != new_node_id]
    sg["links"] = [
        l for l in sg["links"]
        if l.get("origin_id") != new_node_id and l.get("target_id") != new_node_id
    ]

    return f"reverted — dropped node {new_node_id}"


def _process(path: Path, revert: bool, dry_run: bool) -> None:
    ed = WorkflowEditor(path)
    msg = _revert(ed) if revert else _apply(ed)
    if msg.startswith(("split applied", "reverted")) and not dry_run:
        ed.save()
    suffix = " (dry-run)" if dry_run else ""
    print(f"  {path.name}: {msg}{suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = _find_workflow_files()
    mode = "revert" if args.revert else "apply"
    suffix = " (dry-run)" if args.dry_run else ""
    print(f"{mode} split-CropGuides across {len(files)} workflow(s){suffix}")
    for p in files:
        _process(p, args.revert, args.dry_run)


if __name__ == "__main__":
    main()
