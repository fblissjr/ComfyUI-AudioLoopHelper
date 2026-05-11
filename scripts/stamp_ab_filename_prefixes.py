"""stamp_ab_filename_prefixes.

Last updated: 2026-05-11

Stamps per-arm `VHS_VideoCombine.filename_prefix` widgets on every
LTX 2.3 I2V tiled-sampler A/B draft under `internal/workflows/`, so
each arm's renders land in its own subdirectory under ComfyUI's
`output/` (or `temp/` for the first-pass preview).

Why a separate step: the variants script produces drafts by copying
the optimized baseline and editing a few topology / widget knobs.
The baseline carries the source workflow's original filename_prefix
strings, so without this stamper every arm would write into the
same folder, overwriting each other.

Mapping (per draft, both `VHS_VideoCombine` nodes):

    ltx_i2v_tiled_optimized.draft.json -> ltx_i2v_tiled/arm0/...
    ltx_i2v_tiled_arm1.draft.json      -> ltx_i2v_tiled/arm1/...
    ltx_i2v_tiled_arm2.draft.json      -> ltx_i2v_tiled/arm2/...
    ltx_i2v_tiled_arm3.draft.json      -> ltx_i2v_tiled/arm3/...
    ltx_i2v_tiled_arm4.draft.json      -> ltx_i2v_tiled/arm4/...
    ltx_i2v_tiled_arm5.draft.json      -> ltx_i2v_tiled/arm5/...
    ltx_i2v_tiled_no_rtx.draft.json    -> ltx_i2v_tiled/no_rtx/...

Per draft:
  - First-pass preview `VHS_VideoCombine` (`save_output=False`,
    writes to `temp/`):     <ROOT>/<arm>/firstpass_preview
  - Final `VHS_VideoCombine` (`save_output=True`, writes to
    `output/`):             <ROOT>/<arm>/output

Idempotent. `--dry-run` reports the planned mutations. `--revert`
restores the original prefix strings from the optimized baseline if
it's still present, or no-ops if the baseline was reverted (the
optimize script's `--revert` already removes the baseline file).

Usage:
    uv run --group dev python scripts/stamp_ab_filename_prefixes.py
    uv run --group dev python scripts/stamp_ab_filename_prefixes.py --dry-run
    uv run --group dev python scripts/stamp_ab_filename_prefixes.py --revert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
DRAFTS_DIR = REPO_ROOT / "internal" / "workflows"

# A/B drafts produced by apply_ltx_i2v_tiled_{optimizations,ab_variants}.py.
# Filename stem -> arm label used in the output path tree.
ARM_FROM_STEM: dict[str, str] = {
    "ltx_i2v_tiled_optimized.draft": "arm0",
    "ltx_i2v_tiled_arm1.draft": "arm1",
    "ltx_i2v_tiled_arm2.draft": "arm2",
    "ltx_i2v_tiled_arm3.draft": "arm3",
    "ltx_i2v_tiled_arm4.draft": "arm4",
    "ltx_i2v_tiled_arm5.draft": "arm5",
    "ltx_i2v_tiled_no_rtx.draft": "no_rtx",
}

ROOT_FOLDER = "ltx_i2v_tiled"

# Original prefix strings on the unstamped baseline. Used by --revert to
# restore the source workflow's filenames. Identified by their save_output
# value (preview is False, final is True) rather than by node id.
ORIGINAL_PREVIEW_PREFIX = "10E_firstpass"
ORIGINAL_FINAL_PREFIX = "10/10E_9-16_I2V"


def _prefix_for(arm: str, save_output: bool) -> str:
    leaf = "output" if save_output else "firstpass_preview"
    return f"{ROOT_FOLDER}/{arm}/{leaf}"


def _stamp_one(path: Path, arm: str, dry_run: bool, revert: bool) -> bool:
    ed = WorkflowEditor(path)
    changed = False
    for n in ed.wf["nodes"]:
        if n.get("type") != "VHS_VideoCombine":
            continue
        wv = n.get("widgets_values")
        if not isinstance(wv, dict):
            continue
        save_output = bool(wv.get("save_output", False))
        if revert:
            target = ORIGINAL_FINAL_PREFIX if save_output else ORIGINAL_PREVIEW_PREFIX
        else:
            target = _prefix_for(arm, save_output)
        current = wv.get("filename_prefix")
        if current == target:
            continue
        if dry_run:
            print(f"  [{path.name}] #{n['id']} filename_prefix: {current!r} -> {target!r}  (dry-run)")
        else:
            wv["filename_prefix"] = target
            print(f"  [{path.name}] #{n['id']} filename_prefix: {current!r} -> {target!r}")
        changed = True
    if changed and not dry_run:
        ed.save()
    return changed


def _iter_drafts() -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for stem, arm in ARM_FROM_STEM.items():
        path = DRAFTS_DIR / f"{stem}.json"
        if path.exists():
            pairs.append((path, arm))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="Report planned changes without writing.")
    ap.add_argument("--revert", action="store_true",
                    help="Restore original source-workflow filename prefixes.")
    args = ap.parse_args()

    pairs = _iter_drafts()
    if not pairs:
        print(f"No A/B drafts found under {DRAFTS_DIR}. Produce them first with:")
        print("  scripts/apply_ltx_i2v_tiled_optimizations.py --input <scratch source>")
        print("  scripts/apply_ltx_i2v_tiled_ab_variants.py --arm {arm1,arm2,arm3,arm4,arm5,no_rtx}")
        return

    action = "revert" if args.revert else "stamp"
    print(f"{action.title()}ing filename prefixes across {len(pairs)} draft(s){' (dry-run)' if args.dry_run else ''}:")
    touched = 0
    for path, arm in pairs:
        if _stamp_one(path, arm, dry_run=args.dry_run, revert=args.revert):
            touched += 1
    if touched == 0:
        print("All drafts already at the target state.")
    elif args.dry_run:
        print(f"\nWould modify {touched} draft(s). Re-run without --dry-run to apply.")
    else:
        print(f"\nStamped {touched} draft(s).")
        print(f"After rendering, find outputs under `<comfy>/output/{ROOT_FOLDER}/<arm>/` and previews under `<comfy>/temp/{ROOT_FOLDER}/<arm>/`.")


if __name__ == "__main__":
    main()
