"""stamp_ab_filename_prefixes.

Last updated: 2026-05-11

Stamps per-arm filename prefixes on every LTX 2.3 I2V tiled-sampler
A/B draft under `internal/workflows/`, so each arm's renders land
in its own subdirectory under ComfyUI's `output/` (or `temp/` for
the first-pass preview).

This has to handle TWO mechanisms because the source workflow uses
both:

1. `VHS_VideoCombine.filename_prefix` widget. The default path
   when there's no upstream wiring.

2. `RunIdPrefix` (F15 from this repo). When present, its
   `video_prefix` STRING output is wired INTO each
   `VHS_VideoCombine.filename_prefix` slot, overriding the widget.
   ComfyUI honors the input wire when present and ignores the
   widget. So if `RunIdPrefix` is in the workflow, its
   `workflow_name` widget is what actually determines the output
   directory -- not the VHS widget.

The optimize script's source workflow has `RunIdPrefix` (added by
`scripts/apply_run_id_layout.py` upstream), so all seven A/B drafts
inherit it. The stamper handles both shapes: if a `RunIdPrefix`
node exists and feeds `VHS_VideoCombine.filename_prefix`, the
stamper writes the arm into `RunIdPrefix.workflow_name`; otherwise
it falls back to stamping the VHS widget directly.

Why a separate step: the variants script produces drafts by copying
the optimized baseline and editing a few topology / widget knobs.
The baseline carries the source workflow's original prefix strings
and a single `RunIdPrefix` value, so without this stamper every
arm would write into the same folder, overwriting each other.

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

Idempotent. `--dry-run` reports the planned mutations.

Usage:
    uv run --group dev python scripts/stamp_ab_filename_prefixes.py
    uv run --group dev python scripts/stamp_ab_filename_prefixes.py --dry-run

To restore source-workflow defaults, revert + reapply the optimize
script instead -- this stamper has no reverse path since the source
prefixes are workflow-specific.
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
    "ltx_i2v_tiled_arm3.draft": "arm3",
    "ltx_i2v_tiled_arm4.draft": "arm4",
    "ltx_i2v_tiled_arm5.draft": "arm5",
    "ltx_i2v_tiled_no_rtx.draft": "no_rtx",
}

ROOT_FOLDER = "ltx_i2v_tiled"


def _vhs_prefix_for(arm: str, save_output: bool) -> str:
    leaf = "output" if save_output else "firstpass_preview"
    return f"{ROOT_FOLDER}/{arm}/{leaf}"


def _run_id_workflow_name_for(arm: str) -> str:
    return f"{ROOT_FOLDER}_{arm}"


def _vhs_filename_prefix_is_wired(node: dict) -> bool:
    for inp in node.get("inputs", []) or []:
        if inp.get("name") == "filename_prefix" and inp.get("link") is not None:
            return True
    return False


def _stamp_one(path: Path, arm: str, dry_run: bool) -> bool:
    ed = WorkflowEditor(path)
    changed = False

    # Pass 1: RunIdPrefix. If present, its widget[0] (workflow_name)
    # is the path key that ComfyUI uses for the output folder.
    for n in ed.wf["nodes"]:
        if n.get("type") != "RunIdPrefix":
            continue
        wv = n.get("widgets_values")
        if not isinstance(wv, list) or not wv:
            continue
        target = _run_id_workflow_name_for(arm)
        if wv[0] == target:
            continue
        if dry_run:
            print(f"  [{path.name}] #{n['id']} RunIdPrefix.workflow_name: {wv[0]!r} -> {target!r}  (dry-run)")
        else:
            wv[0] = target
            print(f"  [{path.name}] #{n['id']} RunIdPrefix.workflow_name: -> {target!r}")
        changed = True

    # Pass 2: VHS_VideoCombine.filename_prefix. Only the widget when the
    # `filename_prefix` input is unwired -- otherwise the upstream
    # RunIdPrefix output takes effect and the widget is dead. We still
    # write the widget for visual consistency.
    for n in ed.wf["nodes"]:
        if n.get("type") != "VHS_VideoCombine":
            continue
        wv = n.get("widgets_values")
        if not isinstance(wv, dict):
            continue
        save_output = bool(wv.get("save_output", False))
        target = _vhs_prefix_for(arm, save_output)
        current = wv.get("filename_prefix")
        if current == target:
            continue
        wired_note = " (wired -- widget is dead path)" if _vhs_filename_prefix_is_wired(n) else ""
        if dry_run:
            print(f"  [{path.name}] #{n['id']} filename_prefix: {current!r} -> {target!r}{wired_note}  (dry-run)")
        else:
            wv["filename_prefix"] = target
            print(f"  [{path.name}] #{n['id']} filename_prefix: -> {target!r}{wired_note}")
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
    args = ap.parse_args()

    pairs = _iter_drafts()
    if not pairs:
        print(f"No A/B drafts found under {DRAFTS_DIR}. Produce them first with:")
        print("  scripts/apply_ltx_i2v_tiled_optimizations.py --input <scratch source>")
        print("  scripts/apply_ltx_i2v_tiled_ab_variants.py --arm {arm3,arm4,no_rtx}")
        return

    print(f"Stamping filename prefixes across {len(pairs)} draft(s){' (dry-run)' if args.dry_run else ''}:")
    touched = 0
    for path, arm in pairs:
        if _stamp_one(path, arm, dry_run=args.dry_run):
            touched += 1
    if touched == 0:
        print("All drafts already at the target state.")
    elif args.dry_run:
        print(f"\nWould modify {touched} draft(s). Re-run without --dry-run to apply.")
    else:
        print(f"\nStamped {touched} draft(s).")
        print(f"After rendering, find outputs at `<comfy_output>/{ROOT_FOLDER}_<arm>/<TIMESTAMP>_<NNNNN>...`.")
        print(f"Mapping: arm0 -> {ROOT_FOLDER}_arm0, arm1 -> {ROOT_FOLDER}_arm1, ..., no_rtx -> {ROOT_FOLDER}_no_rtx.")


if __name__ == "__main__":
    main()
