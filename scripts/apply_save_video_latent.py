"""apply_save_video_latent — stage a loop variant that captures video_latent.

Last updated: 2026-05-07

Inserts a `SaveLatent` node wired to the top-level `LTXVSeparateAVLatent`
output (Node 245 in canonical loop workflows) so a render emits a `.latent`
file consumable by `scripts/diagnose_overlap_seams.py`. Phase A enabler
for the seam-zone refinement design (`internal/design/polish_passes_design.md §P5`).

Why a draft variant instead of editing the canonical: the SaveLatent
node is a sink with side effects (writes to ComfyUI's output dir on
every render); we don't want it firing on every shipped render. A
gitignored draft under `internal/workflows/` is opt-in.

Idempotent. `--revert` deletes the staged draft. `--dry-run` reports
without writing. No paired audit invariant (per scripts/CLAUDE.md
"Carve-out for staged-variant scripts" — F-pair applies at promotion
time).

Usage:
    uv run --group dev python scripts/apply_save_video_latent.py
    uv run --group dev python scripts/apply_save_video_latent.py --dry-run
    uv run --group dev python scripts/apply_save_video_latent.py --revert
    uv run --group dev python scripts/apply_save_video_latent.py --input <path> --output <path>

Defaults:
    --input  example_workflows/audio-loop-music-video_latent.json
    --output internal/workflows/loop_with_save_latent.draft.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor, resolve_repo_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

SEPARATE_AV_NODE_ID = 245
SEPARATE_AV_TYPE = "LTXVSeparateAVLatent"
VIDEO_LATENT_OUTPUT_SLOT = 0

SAVE_LATENT_TYPE = "SaveLatent"
NEW_NODE_TITLE = "Save video_latent (seam diagnostic)"
DEFAULT_FILENAME_PREFIX = "seam_diag/loop_video_latent"

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "internal/workflows/loop_with_save_latent.draft.json"


def _find_save_latent_node(ed: WorkflowEditor) -> dict | None:
    for n in ed.wf["nodes"]:
        if n.get("type") == SAVE_LATENT_TYPE and n.get("title") == NEW_NODE_TITLE:
            return n
    return None


def _already_migrated(ed: WorkflowEditor) -> bool:
    return _find_save_latent_node(ed) is not None


def _apply(ed: WorkflowEditor, dry_run: bool) -> None:
    if _already_migrated(ed):
        print(f"  {ed.path.name}: SaveLatent node already present, skipping.")
        return

    src = ed.find_node(SEPARATE_AV_NODE_ID)
    if src.get("type") != SEPARATE_AV_TYPE:
        raise SystemExit(
            f"Node #{SEPARATE_AV_NODE_ID} is type {src.get('type')!r}, expected "
            f"{SEPARATE_AV_TYPE!r}. This script assumes the canonical loop layout."
        )

    if dry_run:
        print(f"  {ed.path.name}:")
        print(
            f"    would add {SAVE_LATENT_TYPE} (#?, '{NEW_NODE_TITLE}') "
            f"wired from #{SEPARATE_AV_NODE_ID} output {VIDEO_LATENT_OUTPUT_SLOT}"
        )
        print(f"    filename_prefix = {DEFAULT_FILENAME_PREFIX!r}")
        return

    new_id = ed.add_top_level_node(
        node_type=SAVE_LATENT_TYPE,
        pos=[src["pos"][0] + 240, src["pos"][1] + 80],
        size=[270, 60],
        inputs=[
            WorkflowEditor.io_in("samples", "LATENT"),
            WorkflowEditor.widget_in("filename_prefix", "STRING"),
        ],
        outputs=[],
        widgets_values=[DEFAULT_FILENAME_PREFIX],
        properties={"cnr_id": "comfy-core", "Node name for S&R": SAVE_LATENT_TYPE},
        title=NEW_NODE_TITLE,
    )

    ed.add_link(
        SEPARATE_AV_NODE_ID,
        VIDEO_LATENT_OUTPUT_SLOT,
        new_id,
        0,
        "LATENT",
    )

    print(
        f"  {ed.path.name}: inserted #{new_id} ({SAVE_LATENT_TYPE}) "
        f"capturing #{SEPARATE_AV_NODE_ID}.video_latent -> "
        f"output/{DEFAULT_FILENAME_PREFIX}_NNNNN_.latent"
    )


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    """Stage `output_path` from `input_path`, then apply the migration.

    Idempotent. To re-sync the draft with upstream bug fixes to the
    source workflow, run `--revert` first then re-apply.
    """
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    if output_path.exists() and input_path != output_path:
        existing = WorkflowEditor(output_path)
        if _already_migrated(existing):
            print(
                f"  {output_path}: already migrated, skipping. "
                "Run --revert then re-apply to pull upstream bug fixes from source."
            )
            return

    if not dry_run:
        if input_path != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            print(f"  copied {input_path} -> {output_path}")

    target = output_path if not dry_run else input_path
    ed = WorkflowEditor(target)
    _apply(ed, dry_run=dry_run)
    if not dry_run:
        ed.save()


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


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

    in_path = resolve_repo_path(args.input)
    out_path = resolve_repo_path(args.output)

    if args.revert:
        _revert(out_path)
        return

    _migrate(in_path, out_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
