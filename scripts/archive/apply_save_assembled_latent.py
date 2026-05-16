"""apply_save_assembled_latent — capture LatentConcat output for length-mismatch diagnostic.

Last updated: 2026-05-10

> **For production use (assembled-latent capture feeding the LoadLatent
> upscale path), use ``scripts/apply_run_id_layout.py`` instead.** It
> wires a *bypassed* SaveLatent onto the canonical workflow with a
> per-render filename prefix; the user toggles ``mode=0`` in the UI
> to enable, then back to ``mode=4`` when done. No separate draft file
> to maintain.
>
> THIS script remains for the diagnostic carve-out: staging a separate
> draft file with the SaveLatent always-on, useful when isolating
> length-mismatch / shape-drift bugs without touching the canonical.

Stages a workflow variant with a `SaveLatent` node wired to the
`LatentConcat #1605` output (the assembled `initial_render + loop_body`
latent that feeds final VAE decode). Used to diagnose the open
audio/video length mismatch: predicted internal video length is
73.6s for 3 iters at canonical config but observed output is 64s
(~31 latent frames missing). Capturing the assembled latent localizes
whether the gap is in the loop assembly or downstream in VHS/ffmpeg.

Drafts into `internal/workflows/loop_with_save_assembled_latent.draft.json`.
Per scripts/CLAUDE.md staged-variant carve-out, no F-pair audit needed.

After rendering, the saved `.latent` file lives at
`<comfy_output>/seam_diag/assembled_latent_NNNNN_.latent`. Read its
shape via `safetensors`/`torch` to count latent frames.

Usage:
    uv run --group dev python scripts/apply_save_assembled_latent.py
    uv run --group dev python scripts/apply_save_assembled_latent.py --revert

Defaults:
    --input  example_workflows/audio-loop-music-video_latent.json
    --output internal/workflows/loop_with_save_assembled_latent.draft.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor, resolve_repo_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

LATENT_CONCAT_NODE_ID = 1605
LATENT_CONCAT_OUTPUT_SLOT = 0

SAVE_LATENT_TYPE = "SaveLatent"
NEW_NODE_TITLE = "Save assembled latent (length diagnostic)"
DEFAULT_FILENAME_PREFIX = "seam_diag/assembled_latent"

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "internal/workflows/loop_with_save_assembled_latent.draft.json"


def _find_save_latent_node(ed: WorkflowEditor) -> dict | None:
    for n in ed.wf["nodes"]:
        if n.get("type") == SAVE_LATENT_TYPE and n.get("title") == NEW_NODE_TITLE:
            return n
    return None


def _already_migrated(ed: WorkflowEditor) -> bool:
    return _find_save_latent_node(ed) is not None


def _apply(ed: WorkflowEditor, dry_run: bool) -> None:
    if _already_migrated(ed):
        print(f"  {ed.path.name}: SaveLatent already present, skipping.")
        return

    if not ed.has_node(LATENT_CONCAT_NODE_ID):
        raise SystemExit(
            f"Node #{LATENT_CONCAT_NODE_ID} (LatentConcat 'Prepend Initial Render') "
            "not found. This script assumes the canonical loop layout."
        )
    src = ed.find_node(LATENT_CONCAT_NODE_ID)
    if src.get("type") != "LatentConcat":
        raise SystemExit(
            f"Node #{LATENT_CONCAT_NODE_ID} is type {src.get('type')!r}, "
            "expected 'LatentConcat'."
        )

    if dry_run:
        print(f"  {ed.path.name}: would add {SAVE_LATENT_TYPE} off "
              f"#{LATENT_CONCAT_NODE_ID} output {LATENT_CONCAT_OUTPUT_SLOT}")
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
    ed.add_link(LATENT_CONCAT_NODE_ID, LATENT_CONCAT_OUTPUT_SLOT, new_id, 0, "LATENT")
    print(f"  {ed.path.name}: inserted #{new_id} -> output/{DEFAULT_FILENAME_PREFIX}_NNNNN_.latent")


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    if output_path.exists() and input_path != output_path:
        existing = WorkflowEditor(output_path)
        if _already_migrated(existing):
            print(f"  {output_path}: already migrated, skipping.")
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
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_path = resolve_repo_path(args.input)
    out_path = resolve_repo_path(args.output)

    if args.revert:
        _revert(out_path)
        return

    _migrate(in_path, out_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
