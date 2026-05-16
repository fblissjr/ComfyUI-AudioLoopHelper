"""apply_promote_schedule_to_inputs — surface the prompt schedule in the inputs panel.

Last updated: 2026-05-07

Layout-only mutation: moves Node 1615 (TimestampPromptScheduleBatchEncode)
into the canonical "1. Inputs" group at a featured top position, and
expands the group's bounding box horizontally to encompass it.

Why: the prompt schedule is a primary user input, but in the canonical
loop workflow it currently sits at x=2080 inside group 4
("Conditioning + Frame Planner"). New users miss it; experienced users
have to scroll. The single most-edited piece of text in a render
should sit in the inputs panel where the eye lands first.

This is a pure layout edit. No links change, no node count changes,
no other node's pos changes. Behavioral no-op.

Usage:
    uv run --group dev python scripts/apply_promote_schedule_to_inputs.py
    uv run --group dev python scripts/apply_promote_schedule_to_inputs.py --dry-run
    uv run --group dev python scripts/apply_promote_schedule_to_inputs.py --revert
    uv run --group dev python scripts/apply_promote_schedule_to_inputs.py --workflow <path>

Default target: example_workflows/audio-loop-music-video_latent.json
(in-place edit; canonical is the source of truth for shipped layout).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor, resolve_repo_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOW = "example_workflows/audio-loop-music-video_latent.json"

NODE_1615_SCHEDULE = 1615
INPUTS_GROUP_TITLE = "1. Inputs"

# Target layout. Coordinates chosen to:
#   - sit at the top of group 1 (y ~= 260, alongside the existing GetNode at y=260)
#   - clear the existing left column at x=30 (which uses x[30..330])
#   - fit a 420w x 360h panel that's tall enough to read the schedule
#     textarea comfortably without scrolling
TARGET_NODE_POS = [440, 260]
TARGET_NODE_SIZE = [420, 360]

# Group expansion: width must encompass node right edge + breathing room.
# 880 minimum; final width is max(current_w, 880).
TARGET_GROUP_MIN_WIDTH = 880

# Pre-apply layout (canonical, before this script ran). Used by --revert
# to restore without persisting metadata into shipped workflow JSON.
LEGACY_NODE_POS = [2080, 1021.8244144977641]
LEGACY_NODE_SIZE = [400, 204]
LEGACY_GROUP_BOUNDING = [0, 170, 438.1861328125, 1920.7051202626735]


def _find_inputs_group(ed: WorkflowEditor) -> dict | None:
    for g in ed.wf.get("groups", []):
        if g.get("title") == INPUTS_GROUP_TITLE:
            return g
    return None


def _already_migrated(ed: WorkflowEditor) -> bool:
    n = ed.find_node(NODE_1615_SCHEDULE)
    return list(n.get("pos", [])) == list(TARGET_NODE_POS)


def _apply(ed: WorkflowEditor, dry_run: bool) -> None:
    n = ed.find_node(NODE_1615_SCHEDULE)
    if n.get("type") != "TimestampPromptScheduleBatchEncode":
        raise SystemExit(
            f"Node #{NODE_1615_SCHEDULE} is type {n.get('type')!r}, expected "
            "'TimestampPromptScheduleBatchEncode'. Layout drift; refusing."
        )

    g = _find_inputs_group(ed)
    if g is None:
        raise SystemExit(
            f"Group {INPUTS_GROUP_TITLE!r} not found. This script assumes the "
            "canonical group set."
        )

    if _already_migrated(ed):
        print(f"  {ed.path.name}: schedule already in inputs panel, skipping.")
        return

    if dry_run:
        print(f"  {ed.path.name}:")
        print(f"    would move #{NODE_1615_SCHEDULE} {n['pos']} -> {TARGET_NODE_POS}")
        bx, by, bw, bh = g["bounding"]
        new_w = max(bw, TARGET_GROUP_MIN_WIDTH)
        print(f"    would expand group {INPUTS_GROUP_TITLE!r} width "
              f"{bw:.0f} -> {new_w:.0f}")
        return

    n["pos"] = list(TARGET_NODE_POS)
    n["size"] = list(TARGET_NODE_SIZE)
    bx, by, bw, bh = g["bounding"]
    g["bounding"] = [bx, by, max(bw, TARGET_GROUP_MIN_WIDTH), bh]

    print(
        f"  {ed.path.name}: moved #{NODE_1615_SCHEDULE} -> {TARGET_NODE_POS}; "
        f"expanded {INPUTS_GROUP_TITLE!r} width to {g['bounding'][2]:.0f}."
    )


def _revert(ed: WorkflowEditor) -> bool:
    """Restore the canonical pre-apply layout from hardcoded constants.

    Avoids leaking stash metadata into committed workflow JSON. The
    constants below match the canonical state before this script first
    ran; if a user has hand-tuned pos/bounding, --revert will normalize.
    """
    n = ed.find_node(NODE_1615_SCHEDULE)
    g = _find_inputs_group(ed)
    if g is None:
        raise SystemExit(f"Group {INPUTS_GROUP_TITLE!r} not found; can't revert.")

    if list(n.get("pos", [])) == list(LEGACY_NODE_POS):
        print(f"  {ed.path.name}: already at legacy pos; nothing to revert.")
        return False

    # Drop legacy stash keys if present (cleans up older apply runs).
    n.pop("_promote_schedule_pre", None)
    g.pop("_promote_schedule_pre", None)

    n["pos"] = list(LEGACY_NODE_POS)
    n["size"] = list(LEGACY_NODE_SIZE)
    g["bounding"] = list(LEGACY_GROUP_BOUNDING)

    print(f"  {ed.path.name}: reverted #{NODE_1615_SCHEDULE} pos and "
          f"{INPUTS_GROUP_TITLE!r} bounding.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--workflow", default=DEFAULT_WORKFLOW,
                    help=f"Target workflow JSON (default: {DEFAULT_WORKFLOW}).")
    ap.add_argument("--revert", action="store_true",
                    help="Restore the canonical pre-apply pos/bounding.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    target = resolve_repo_path(args.workflow)
    if not target.exists():
        raise SystemExit(f"Workflow not found: {target}")

    ed = WorkflowEditor(target)
    if args.revert:
        changed = _revert(ed)
        if changed:
            ed.save()
        return

    _apply(ed, dry_run=args.dry_run)
    if not args.dry_run:
        ed.save()


if __name__ == "__main__":
    main()
