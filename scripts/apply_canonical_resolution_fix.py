"""apply_canonical_resolution_fix.

Last updated: 2026-04-25

Brings every shipped production workflow's `EmptyLTXVLatentVideo`
widget into spec with `docs/reference/ltx23_model_reference.md`
§"Resolution and latent volume" AND with `ImageResizeKJv2`'s actual
target. Default fix: 704x704 (volume 30,492 — over the 24,570 artifact
ceiling) -> 832x448 (22,932 — NEAR_EDGE, users' actual operating
point). Idempotent + `--revert` + `--dry-run`.

Audit pairing: `scripts/audit_workflows.py` ERRs on volumes >24,570
with a remediation pointer back to this script — re-introducing
704x704 anywhere will trip the audit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from workflow_utils import WorkflowEditor
from nodes import _compute_ltx_resolution

EMPTY_LATENT_NODE_ID = 344

DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 448
PRIOR_WIDTH = 704
PRIOR_HEIGHT = 704

# Validator workflow intentionally exercises edge resolutions; skip it.
SKIP_FILES = {"audio-loop-music-video_latent_validator.json"}

WORKFLOWS_DIR = REPO_ROOT / "example_workflows"


def _find_workflow_files() -> list[Path]:
    return sorted(p for p in WORKFLOWS_DIR.glob("*.json") if p.name not in SKIP_FILES)


def _patch_one(path: Path, target_w: int, target_h: int, dry_run: bool) -> bool:
    """Returns True if a change was applied (or would be applied in --dry-run)."""
    ed = WorkflowEditor(path)
    try:
        node = ed.find_node(EMPTY_LATENT_NODE_ID)
    except ValueError:
        print(f"  {path.name}: no EmptyLTXVLatentVideo({EMPTY_LATENT_NODE_ID}); skipping")
        return False

    if node["type"] != "EmptyLTXVLatentVideo":
        print(f"  {path.name}: node #{EMPTY_LATENT_NODE_ID} is {node['type']!r}; skipping")
        return False

    wv = node.get("widgets_values", [])
    if len(wv) < 2:
        print(f"  {path.name}: widgets_values too short; skipping")
        return False

    cur_w, cur_h = wv[0], wv[1]
    if (cur_w, cur_h) == (target_w, target_h):
        print(f"  {path.name}: already {target_w}x{target_h} — no change")
        return False

    verb = "would patch" if dry_run else "patched"
    print(f"  {path.name}: {verb} {cur_w}x{cur_h} -> {target_w}x{target_h}")
    if dry_run:
        return True
    wv[0], wv[1] = target_w, target_h
    ed.save()
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument(
        "--revert", action="store_true",
        help=f"Restore prior {PRIOR_WIDTH}x{PRIOR_HEIGHT} (A/B testing only).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print intended changes; don't write.")
    args = ap.parse_args()

    if args.width % 32 != 0 or args.height % 32 != 0:
        raise SystemExit(f"--width and --height must both be div by 32; got {args.width}x{args.height}")

    target_w, target_h = (PRIOR_WIDTH, PRIOR_HEIGHT) if args.revert else (args.width, args.height)
    _, _, _, status = _compute_ltx_resolution(target_w / target_h, target_w, 497, "landscape")
    print(f"target dimensions: {target_w}x{target_h} at length=497 -> {status}")
    print()

    files = _find_workflow_files()
    suffix = " (dry-run)" if args.dry_run else (" (revert mode)" if args.revert else "")
    print(f"Patching {len(files)} workflow(s){suffix}")
    for p in files:
        _patch_one(p, target_w, target_h, args.dry_run)


if __name__ == "__main__":
    main()
