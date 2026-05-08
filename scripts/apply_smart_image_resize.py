"""apply_smart_image_resize — swap KJv2 single-pass init resize for LTXSmartImageResize.

Last updated: 2026-05-07

Replaces `ImageResizeKJv2` (Node 445 in canonical loop variants) with
`LTXSmartImageResize` across all canonical workflows. The new node
adapts the number of stages based on the source/target ratio at
runtime: large source images (4K+, AI-generated 2K+) get multi-stage
downscaling that keeps each pass within lanczos's clean anti-alias
range; small sources still single-pass.

Why: LTX 2.3's i2v cross-attention reads aliasing artifacts on faces
/ text / fine textures as "high-frequency content to explore" and
tends to push the camera in the first window — manifesting as
spurious zoom/dolly motion in i2v renders even when the prompt asks
for static framing. Single-pass lanczos at >2x linear reduction
leaves enough aliasing to trigger this; staged downscaling suppresses
it.

Note on multi-stage kernel choice: `LTXSmartImageResize` uses
`F.interpolate(bicubic, antialias=True)` for intermediate stages
(float32 throughout) and PIL lanczos only for the final stage. Naive
multi-stage PIL lanczos would stack 8-bit quantization rounds and
re-introduce the same motion-cue noise we're trying to suppress.
Postmortem: `internal/analysis/smart_resize_quantization_postmortem.md`
(private clone only).

Mechanics (in-place type swap; preserves Node 445 ID):
  - type:    ImageResizeKJv2 -> LTXSmartImageResize
  - inputs:  [image, mask?, width, height, ...] -> [image, width,
             height, keep_proportion, crop_position]; the optional MASK
             wire is dropped (LTXSmartImageResize has no mask input —
             canonical workflows don't feed a meaningful mask, just
             the LoadImage MASK placeholder).
  - outputs: [IMAGE, width, height, mask] -> [image, width, height];
             slots 0/1/2 keep their consumer link arrays so the
             validator workflow's width/height taps survive.
  - widgets_values: [w, h, ...KJv2 extras] -> [w, h, True, 'top']
  - top-level links targeting Node 445 input slots get re-mapped to
    new schema slot indices.

`--revert` rebuilds the KJv2 shape from canonical defaults (the
legacy widgets/inputs are stable across all five loop variants). It
does NOT preserve the dropped MASK wire — if you reverted to recover
a mask wire, re-add it manually or restore from git. This avoids
leaking stash metadata into committed JSON.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor, resolve_repo_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOWS_DIR = "example_workflows"

INIT_RESIZE_NODE_ID = 445
LEGACY_TYPE = "ImageResizeKJv2"
NEW_TYPE = "LTXSmartImageResize"

# Canonical KJv2 defaults — used to rebuild the legacy shape on revert.
# These values match all five shipped loop variants where Node 445
# exists. If a user has hand-tuned widgets, --revert will normalize.
LEGACY_KJV2_WIDGETS = [832, 448, "lanczos", "crop", "0, 0, 0", "top", 2, "cpu"]
LEGACY_KJV2_PROPERTIES = {"Node name for S&R": LEGACY_TYPE}


def _migrate_node(ed: WorkflowEditor) -> bool:
    """Apply the swap on a single workflow. Returns True if mutated."""
    if not ed.has_node(INIT_RESIZE_NODE_ID):
        return False
    n = ed.find_node(INIT_RESIZE_NODE_ID)
    if n.get("type") == NEW_TYPE:
        return False
    if n.get("type") != LEGACY_TYPE:
        return False  # foreign node at this id; refuse to touch

    old_inputs = n.get("inputs", [])
    name_to_old_slot = {inp.get("name"): i for i, inp in enumerate(old_inputs)}

    # Capture wire IDs by name BEFORE rewriting the inputs list.
    incoming = {
        nm: old_inputs[name_to_old_slot[nm]].get("link")
        for nm in ("image", "width", "height")
        if nm in name_to_old_slot
    }

    # Drop the mask wire (no equivalent on the new schema).
    mask_link_id = (
        old_inputs[name_to_old_slot["mask"]].get("link")
        if "mask" in name_to_old_slot else None
    )
    if mask_link_id is not None:
        try:
            ed.remove_link(mask_link_id)
        except ValueError:
            pass  # already absent

    # Rewrite top-level links so width/height links point to new slot
    # indices on Node 445. New slot order: image=0, width=1, height=2.
    new_slot_for_name = {"image": 0, "width": 1, "height": 2}
    for link in ed.find_links_to(INIT_RESIZE_NODE_ID):
        old_slot = link[4]
        if old_slot is None or old_slot >= len(old_inputs):
            continue
        old_name = old_inputs[old_slot].get("name")
        if old_name in new_slot_for_name:
            link[4] = new_slot_for_name[old_name]

    # Preserve outgoing consumer-link arrays on slots 0/1/2 so the
    # validator workflow's width/height taps keep working. Slot 3
    # (mask output) is dropped; canonical workflows don't consume it.
    old_outputs = n.get("outputs", [])
    def _preserved_output_links(idx: int) -> list:
        return list(old_outputs[idx].get("links") or []) if idx < len(old_outputs) else []

    n["type"] = NEW_TYPE
    n["inputs"] = [
        WorkflowEditor.io_in("image", "IMAGE", incoming.get("image")),
        WorkflowEditor.widget_in("width", "INT", incoming.get("width")),
        WorkflowEditor.widget_in("height", "INT", incoming.get("height")),
        WorkflowEditor.widget_in("keep_proportion", "BOOLEAN"),
        WorkflowEditor.widget_in("crop_position", "COMBO"),
    ]
    n["outputs"] = [
        {"name": "image", "type": "IMAGE", "links": _preserved_output_links(0)},
        {"name": "width", "type": "INT", "links": _preserved_output_links(1)},
        {"name": "height", "type": "INT", "links": _preserved_output_links(2)},
    ]

    old_widgets = n.get("widgets_values", [])
    width_val = old_widgets[0] if len(old_widgets) > 0 else 832
    height_val = old_widgets[1] if len(old_widgets) > 1 else 448
    crop_pos = old_widgets[5] if len(old_widgets) > 5 else "top"
    if crop_pos not in ("center", "top", "bottom", "left", "right"):
        crop_pos = "top"
    n["widgets_values"] = [width_val, height_val, True, crop_pos]
    n["properties"] = {
        "aux_id": "fblissjr/ComfyUI-AudioLoopHelper",
        "cnr_id": "comfyui-audioloophelper",
        "Node name for S&R": NEW_TYPE,
    }
    return True


def _revert_node(ed: WorkflowEditor) -> bool:
    """Restore the KJv2 shape from canonical defaults.

    Width/height wires on Node 445 are remapped from new-schema slots
    (1, 2) back to legacy KJv2 slots (2, 3). The old MASK wire is not
    restored; --revert is for "undo this script's swap," not a full
    history replay.
    """
    if not ed.has_node(INIT_RESIZE_NODE_ID):
        return False
    n = ed.find_node(INIT_RESIZE_NODE_ID)
    if n.get("type") != NEW_TYPE:
        return False  # not migrated by this script

    cur_inputs = n.get("inputs", [])
    cur_name_to_slot = {inp.get("name"): i for i, inp in enumerate(cur_inputs)}
    incoming = {
        nm: cur_inputs[cur_name_to_slot[nm]].get("link")
        for nm in ("image", "width", "height")
        if nm in cur_name_to_slot
    }

    # Remap top-level links from new slots (image=0, width=1, height=2)
    # to legacy KJv2 slots (image=0, mask=1, width=2, height=3).
    new_slot_to_legacy = {0: 0, 1: 2, 2: 3}
    for link in ed.find_links_to(INIT_RESIZE_NODE_ID):
        new_slot = link[4]
        if new_slot in new_slot_to_legacy:
            link[4] = new_slot_to_legacy[new_slot]

    cur_outputs = n.get("outputs", [])
    def _preserved_output_links(idx: int) -> list:
        return list(cur_outputs[idx].get("links") or []) if idx < len(cur_outputs) else []

    n["type"] = LEGACY_TYPE
    n["inputs"] = [
        WorkflowEditor.io_in("image", "IMAGE", incoming.get("image")),
        # Mask slot exists for shape-compat with the legacy schema; no wire
        # is restored (we don't track the dropped wire).
        {"name": "mask", "shape": 7, "type": "MASK", "link": None},
        WorkflowEditor.widget_in("width", "INT", incoming.get("width")),
        WorkflowEditor.widget_in("height", "INT", incoming.get("height")),
    ]
    n["outputs"] = [
        {"name": "IMAGE", "type": "IMAGE", "links": _preserved_output_links(0)},
        {"name": "width", "type": "INT", "links": _preserved_output_links(1)},
        {"name": "height", "type": "INT", "links": _preserved_output_links(2)},
        {"name": "mask", "type": "MASK", "links": []},
    ]
    n["widgets_values"] = list(LEGACY_KJV2_WIDGETS)
    n["properties"] = dict(LEGACY_KJV2_PROPERTIES)
    return True


def _walk(workflows_dir: Path, dry_run: bool, revert: bool) -> int:
    if not workflows_dir.is_dir():
        raise SystemExit(f"Not a directory: {workflows_dir}")
    n_changed = 0
    for jp in sorted(workflows_dir.glob("*.json")):
        ed = WorkflowEditor(jp)
        changed = _revert_node(ed) if revert else _migrate_node(ed)
        verb = "reverted" if revert else "migrated"
        if changed:
            n_changed += 1
            print(f"  {jp.name}: {verb}")
            if not dry_run:
                ed.save()
        else:
            print(f"  {jp.name}: skipped")
    return n_changed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--workflows-dir", default=DEFAULT_WORKFLOWS_DIR,
                    help=f"Directory of workflow JSON files (default: {DEFAULT_WORKFLOWS_DIR}).")
    ap.add_argument("--revert", action="store_true",
                    help="Restore the KJv2 shape from canonical defaults.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    n_changed = _walk(resolve_repo_path(args.workflows_dir),
                      dry_run=args.dry_run, revert=args.revert)
    print(f"Changed: {n_changed} workflow(s).")


if __name__ == "__main__":
    main()
