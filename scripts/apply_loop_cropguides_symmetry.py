"""Wire loop-body CFGGuider through LTXVCropGuides on the CONDITIONING path.

Last updated: 2026-04-24

Symptom it fixes: subtle iter-over-iter identity drift that the F2 preprocess
symmetry fix (`apply_loop_guide_preprocess_symmetry.py`) doesn't fully address.
Where F2's fingerprint is "microphone + unrelated woman replaces subject",
F3's fingerprint is gradual feature drift (hair, clothing, facial structure)
without the microphone, concentrated in later windows.

Root cause: the initial-render CONDITIONING path runs through `#381
LTXVCropGuides` before `#153 CFGGuider` — guide-keyframe metadata is stripped
from CONDITIONING before the sampler sees it. The loop body's subgraph
contains the mirror node `#655 LTXVCropGuides` but its positive/negative
CONDITIONING OUTPUTS are unconsumed: `#644 CFGGuider` reads directly from
`#1519 LTXVAddLatentGuide[0,1]`, bypassing #655. Crop is computed then
discarded. Over N iterations, guide metadata accumulates in CONDITIONING
differently than the initial render ever saw.

The existing #655 is already wired correctly on inputs:
  - #1519 out[0] (pos) -> #655 in[0]   (subgraph link 2832, kept)
  - #1519 out[1] (neg) -> #655 in[1]   (subgraph link 2833, kept)
  - #2006 out[0]       -> #655 in[2] LATENT (kept)

Fix: rewire #644's positive/negative inputs from #1519 outputs to #655 outputs.
Topologically symmetric to the initial path's `#164 -> #381 -> #153`.

  Before: #1519[0,1] -> #644[1,2]   (uncropped CONDITIONING)
  After:  #655 [0,1] -> #644[1,2]   (cropped CONDITIONING, matches initial)

Compatibility with other apply scripts:
  - `apply_loop_guide_preprocess_symmetry.py`: orthogonal (touches #650 on
    top-level, not subgraph).
  - `apply_iclora_initial_render.py`: touches `#164 LTXVConditioning` outputs
    on the top-level initial-render path; does NOT touch subgraph. Orthogonal.
  - `apply_sage_mode.py`: touches node 268 only. Orthogonal.

Usage:
    uv run --group dev python scripts/apply_loop_cropguides_symmetry.py
    uv run --group dev python scripts/apply_loop_cropguides_symmetry.py --revert
    uv run --group dev python scripts/apply_loop_cropguides_symmetry.py --dry-run

Idempotent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

LTXV_ADD_LATENT_GUIDE_ID = 1519   # Subgraph node: positive/negative CONDITIONING outputs
CFG_GUIDER_ID = 644               # Subgraph node: positive/negative CONDITIONING inputs
LTXV_CROP_GUIDES_ID = 655         # Subgraph node: mirrors initial-path #381

SUBGRAPH_INDEX = 0


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    sg = ed.get_subgraph(SUBGRAPH_INDEX)
    if sg is None:
        return "skip (no subgraph)"

    sg_node_ids = {n["id"] for n in sg.get("nodes", [])}
    missing = [nid for nid in (LTXV_ADD_LATENT_GUIDE_ID, CFG_GUIDER_ID, LTXV_CROP_GUIDES_ID)
               if nid not in sg_node_ids]
    if missing:
        return f"skip (missing subgraph nodes {missing})"

    pos = ed.find_subgraph_link_to_slot(CFG_GUIDER_ID, 1, SUBGRAPH_INDEX)
    neg = ed.find_subgraph_link_to_slot(CFG_GUIDER_ID, 2, SUBGRAPH_INDEX)
    if pos is None or neg is None:
        return f"skip (CFGGuider {CFG_GUIDER_ID} missing inbound pos/neg links)"

    expected_src = LTXV_CROP_GUIDES_ID if revert else LTXV_ADD_LATENT_GUIDE_ID
    target_src = LTXV_ADD_LATENT_GUIDE_ID if revert else LTXV_CROP_GUIDES_ID

    pos_at_target = pos["origin_id"] == target_src and pos["origin_slot"] == 0
    neg_at_target = neg["origin_id"] == target_src and neg["origin_slot"] == 1
    if pos_at_target and neg_at_target:
        return "already reverted" if revert else "no change (already symmetric)"

    pos_at_expected = pos["origin_id"] == expected_src and pos["origin_slot"] == 0
    neg_at_expected = neg["origin_id"] == expected_src and neg["origin_slot"] == 1
    if not (pos_at_expected and neg_at_expected):
        return (
            f"skip (unexpected inbound sources pos={pos['origin_id']}/{pos['origin_slot']} "
            f"neg={neg['origin_id']}/{neg['origin_slot']})"
        )

    verb = (
        "would revert" if dry_run and revert else
        "would update" if dry_run else
        "reverted" if revert else "updated"
    )
    if not dry_run:
        ed.rewire_subgraph_input(CFG_GUIDER_ID, 1, target_src, 0, "CONDITIONING", SUBGRAPH_INDEX)
        ed.rewire_subgraph_input(CFG_GUIDER_ID, 2, target_src, 1, "CONDITIONING", SUBGRAPH_INDEX)
        ed.save(wf_path)

    if revert:
        return f"{verb} (CFGGuider CONDITIONING now from LTXVAddLatentGuide direct)"
    return f"{verb} (CFGGuider CONDITIONING now cropped via LTXVCropGuides)"


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} loop-body CFGGuider CONDITIONING through LTXVCropGuides across example_workflows/...")
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
    ap.add_argument("--revert", action="store_true",
                    help="Restore CFGGuider CONDITIONING to feed directly from LTXVAddLatentGuide.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
