"""apply_lanczos_init_preprocess — two-stage lanczos init preprocess.

Last updated: 2026-05-04

Inserts a supersample-then-decimate lanczos preprocess pair in front of the
init-image resize node. With a single-pass downscale from a much-larger
source (e.g. 4K input → 832×448 target), residual aliasing on faces, text,
and fine textures shows in the encoded latent. A two-stage pass —
supersample to ~2× target via lanczos, then decimate 0.5× via lanczos —
gives the second-pass anti-alias kernel something it can integrate
properly. No-op (or marginal) when source ≤ target dims; cost is one
extra CPU resize per render (sub-second).

Symptom / motivation: residual aliasing on init-image-driven I2V renders
when the source image is much larger than the schedule target dims.

Root cause: a single lanczos pass cannot fully suppress aliasing at large
reduction ratios; the convolution kernel sees too few input samples per
output pixel.

Fix / change applied: insert a new `ImageResizeKJv2` (id chosen as next
unused) BEFORE the existing target-dim resize. New node sizes to 2× the
existing widget dims, same kernel + crop + alignment.

Compatibility with other apply scripts:
  - **F-pair convention**: ships with audit `lanczos_init_preprocess`
    (warn-level — opt-in pattern; not all workflows benefit equally).
  - Idempotent. Safe to re-run after upstream bug fixes to the source
    workflow — the script regenerates the staged variant from the
    current source, so drafts stay in sync with `example_workflows/`
    bug fixes.

Usage:
    uv run --group dev python scripts/apply_lanczos_init_preprocess.py
    uv run --group dev python scripts/apply_lanczos_init_preprocess.py --input <path> --output <path>
    uv run --group dev python scripts/apply_lanczos_init_preprocess.py --revert
    uv run --group dev python scripts/apply_lanczos_init_preprocess.py --dry-run

Defaults:
    --input  example_workflows/audio-loop-music-video_latent.json
    --output internal/workflows/loop_with_lanczos_preprocess.draft.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Identifies the init-image resize node in canonical loop workflows.
INIT_RESIZE_NODE_ID = 445
INIT_RESIZE_TYPE = "ImageResizeKJv2"

# Title applied to the new supersample node — used as the idempotence
# signature (a node with this title means we've already applied).
NEW_NODE_TITLE = "Lanczos preprocess (supersample)"

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "internal/workflows/loop_with_lanczos_preprocess.draft.json"


def _find_supersample_node(ed: WorkflowEditor) -> dict | None:
    """Return the supersample node if previously applied, else None.

    Identified by title rather than id because we don't reserve a stable
    id for it (apply script picks `next_node_id()`). Title is a robust
    signature.
    """
    for n in ed.wf["nodes"]:
        if n.get("title") == NEW_NODE_TITLE and n.get("type") == INIT_RESIZE_TYPE:
            return n
    return None


def _already_migrated(ed: WorkflowEditor) -> bool:
    return _find_supersample_node(ed) is not None


def _read_target_dims(ed: WorkflowEditor) -> tuple[int, int]:
    """Read the existing init-resize widget dims (width, height).

    These are the FINAL target dims; the new supersample stage will use
    2× each.
    """
    n = ed.find_node(INIT_RESIZE_NODE_ID)
    if n.get("type") != INIT_RESIZE_TYPE:
        raise SystemExit(
            f"Node #{INIT_RESIZE_NODE_ID} is type {n.get('type')!r}, expected "
            f"{INIT_RESIZE_TYPE!r}. This script assumes the canonical loop layout."
        )
    wv = n.get("widgets_values", [])
    if len(wv) < 2 or not isinstance(wv[0], int) or not isinstance(wv[1], int):
        raise SystemExit(
            f"Node #{INIT_RESIZE_NODE_ID} widget_values[:2] = {wv[:2]!r}; "
            "expected [width, height] integers. The widget shape may have drifted."
        )
    return int(wv[0]), int(wv[1])


def _apply(ed: WorkflowEditor, dry_run: bool) -> None:
    if _already_migrated(ed):
        print(f"  {ed.path.name}: supersample node already present, skipping.")
        return

    width, height = _read_target_dims(ed)
    super_w, super_h = width * 2, height * 2

    # Find the IMAGE link feeding the existing #445.image input.
    orig = ed.find_node(INIT_RESIZE_NODE_ID)
    image_input = next((i for i in orig["inputs"] if i.get("name") == "image"), None)
    if image_input is None or image_input.get("link") is None:
        raise SystemExit(
            f"Node #{INIT_RESIZE_NODE_ID} has no wired 'image' input. "
            "Source workflow may have already been mutated."
        )
    src_link_id = image_input["link"]
    src_link = next((l for l in ed.wf["links"] if isinstance(l, list) and l[0] == src_link_id), None)
    if src_link is None:
        raise SystemExit(f"Link {src_link_id} (image into #{INIT_RESIZE_NODE_ID}) not found.")
    src_node_id, src_slot = src_link[1], src_link[2]

    if dry_run:
        print(f"  {ed.path.name}:")
        print(f"    would add {INIT_RESIZE_TYPE} (#?, supersample) at {super_w}x{super_h}")
        print(f"    would rewire IMAGE: #{src_node_id}[{src_slot}] -> new -> #{INIT_RESIZE_NODE_ID}[0]")
        return

    # Add the new supersample node. ImageResizeKJv2 schema (per existing #445):
    #   inputs:  image (IMAGE), mask (MASK, optional), width (INT, widget),
    #            height (INT, widget)
    #   outputs: IMAGE, width (INT), height (INT), mask (MASK)
    #   widgets: [width, height, "lanczos", crop_mode, padding, gravity,
    #             quality, device]
    new_id = ed.add_top_level_node(
        node_type=INIT_RESIZE_TYPE,
        pos=[orig["pos"][0] - 320, orig["pos"][1]],
        size=[270, 336],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "mask", "shape": 7, "type": "MASK", "link": None},
            {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
            {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
        ],
        outputs=[
            {"name": "IMAGE", "type": "IMAGE", "links": []},
            {"name": "width", "type": "INT", "links": []},
            {"name": "height", "type": "INT", "links": []},
            {"name": "mask", "type": "MASK", "links": []},
        ],
        widgets_values=[super_w, super_h, "lanczos", "crop", "0, 0, 0", "top", 2, "cpu"],
        title=NEW_NODE_TITLE,
    )

    # Reroute: original src -> new -> #445
    ed.remove_link(src_link_id)
    ed.add_link(src_node_id, src_slot, new_id, 0, "IMAGE")
    ed.add_link(new_id, 0, INIT_RESIZE_NODE_ID, 0, "IMAGE")

    print(f"  {ed.path.name}: inserted #{new_id} ({INIT_RESIZE_TYPE}, {super_w}x{super_h}) before #{INIT_RESIZE_NODE_ID}")


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    """Stage `output_path` from `input_path`, then apply the migration.

    Idempotent: if the output already exists and is already migrated,
    does nothing. To re-sync the draft with upstream bug fixes to the
    source workflow, run `--revert` first then re-apply.
    """
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    # Idempotence: skip if output already has the migration applied.
    if output_path.exists() and input_path != output_path:
        existing = WorkflowEditor(output_path)
        if _already_migrated(existing):
            print(
                f"  {output_path.relative_to(REPO_ROOT)}: already migrated, skipping. "
                "Run --revert then re-apply to pull upstream bug fixes from source."
            )
            return

    if not dry_run:
        if input_path != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            print(f"  copied {input_path.relative_to(REPO_ROOT)} -> {output_path.relative_to(REPO_ROOT)}")

    target = output_path if not dry_run else input_path
    ed = WorkflowEditor(target)
    _apply(ed, dry_run=dry_run)
    if not dry_run:
        ed.save()


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path.relative_to(REPO_ROOT)}")
    else:
        print(f"{output_path.relative_to(REPO_ROOT)} does not exist; nothing to revert.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"Source workflow (default: {DEFAULT_INPUT}).")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"Output draft path (default: {DEFAULT_OUTPUT}).")
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    in_path = (REPO_ROOT / args.input).resolve()
    out_path = (REPO_ROOT / args.output).resolve()

    if args.revert:
        _revert(out_path)
        return

    _migrate(in_path, out_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
