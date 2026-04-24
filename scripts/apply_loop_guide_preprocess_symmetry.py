"""Fix initial-render vs loop-guide asymmetry on `LTXVPreprocess`.

Last updated: 2026-04-24

Symptom it fixes: in later loop iterations an unrelated photoreal woman
(often holding a microphone) replaces the reference-image subject. ~20%
regression rate on otherwise-working generations.

Root cause: the six shipped workflows branch the init image at
`ImageResizeKJv2` (#445) BEFORE `LTXVPreprocess` (#446). The initial
render consumes the preprocessed image (`img_compression=18`) via
`LTXVImgToVideoInplaceKJ` (#531). The loop guide branch picks up the
RAW resized image via `Set_input_image` (#650) -> `Get_input_image`
(#651) -> `VAEEncode` (#1617) -> subgraph slot 8 -> `LTXVAddLatentGuide`
(#1519). Iter 0 locks in preprocessed stats; iters 1+ anchor to raw
stats. Cross-attention (photoreal-trained) drifts across that delta
iteration-over-iteration and reasserts its "singing woman" prior --
microphone + replacement subject is the textbook shape.

CLAUDE.md flags `img_compression=0` vs `18` as a frozen-first-frame /
drift footgun explicitly. The loop branch was effectively running `=0`
(no preprocess at all) while initial ran `=18`.

Fix: reroute `#445 -> #650` to `#446 -> #650`. Initial render and loop
guide now share the same preprocessed image. No new nodes; single link
swap per workflow.

Compatibility with other apply scripts:
  - `apply_sage_mode.py`: touches node 268 only. Orthogonal.
  - `apply_iclora_initial_render.py`: adds a NEW outbound link from
    #446 to the IC-LoRA guide without removing existing ones. After
    this fix, #446 has outbound links to #531, #650, and (post-IC-LoRA)
    the guide. All independent.

Usage:
    uv run --group dev python scripts/apply_loop_guide_preprocess_symmetry.py
    uv run --group dev python scripts/apply_loop_guide_preprocess_symmetry.py --revert
    uv run --group dev python scripts/apply_loop_guide_preprocess_symmetry.py --dry-run

Idempotent. Run repeatedly; already-fixed workflows report "no change".
`--dry-run` reports the planned change without writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

IMAGE_RESIZE_ID = 445       # ImageResizeKJv2 (pre-preprocess)
LTXV_PREPROCESS_ID = 446    # LTXVPreprocess (post-preprocess, img_compression=18)
SET_INPUT_IMAGE_ID = 650    # SetNode "input_image" -- feeds loop subgraph guide


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    missing = ed.require_nodes((IMAGE_RESIZE_ID, LTXV_PREPROCESS_ID, SET_INPUT_IMAGE_ID))
    if missing:
        return f"skip (missing nodes {missing})"

    row = ed.find_link_to_slot(SET_INPUT_IMAGE_ID, 0)
    if row is None:
        return f"skip (node {SET_INPUT_IMAGE_ID}.in[0] has no link)"
    _, src_node, src_slot, *_ = row

    expected_src = LTXV_PREPROCESS_ID if revert else IMAGE_RESIZE_ID
    target_src = IMAGE_RESIZE_ID if revert else LTXV_PREPROCESS_ID

    if src_node == target_src and src_slot == 0:
        return "already reverted" if revert else "no change (already symmetric)"
    if src_node != expected_src or src_slot != 0:
        return f"skip (unexpected inbound source {src_node}/{src_slot})"

    verb = (
        "would revert" if dry_run and revert else
        "would update" if dry_run else
        "reverted" if revert else "updated"
    )
    if not dry_run:
        ed.rewire_input(SET_INPUT_IMAGE_ID, 0, target_src, 0, "IMAGE")
        ed.save(wf_path)

    if revert:
        return f"{verb} (446 -> 445 as input_image source)"
    return f"{verb} (loop guide now shares LTXVPreprocess output)"


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} loop-guide preprocess symmetry fix across example_workflows/...")
    fail = 0
    for wf_path in sorted(WORKFLOWS_DIR.glob("*.json")):
        status = _apply_one(wf_path, revert, dry_run)
        print(f"  {wf_path.name}: {status}")
        if status.startswith("load error"):
            fail += 1
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--revert", action="store_true",
        help="Restore the original 445 -> 650 wiring (loop guide skips LTXVPreprocess).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report what WOULD change without writing files.",
    )
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
