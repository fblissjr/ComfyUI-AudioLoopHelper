"""promote_latent_for_upscale.

Last updated: 2026-05-10

Find the most recent assembled video latent saved by a loop workflow's
bypassed-SaveLatent toggle (added by ``scripts/apply_run_id_layout.py``)
and copy it to ComfyUI's input directory under a deterministic name
so the upscale workflow's ``LoadLatent`` widget can pick it up.

Saves the user from manually walking
``<output>/<workflow_name>/<timestamp>/latents/segment_NNNNN_.latent``
on every render iteration.

Usage:
    # Most common — uses env vars or interactive prompts for paths
    uv run --group dev python scripts/promote_latent_for_upscale.py audio-loop-music-video_latent

    # Explicit dirs
    uv run --group dev python scripts/promote_latent_for_upscale.py audio-loop-music-video_latent \\
        --output-dir /path/to/comfy/output --input-dir /path/to/comfy/input

    # Different destination filename (default: assembled_latent.latent)
    uv run --group dev python scripts/promote_latent_for_upscale.py audio-loop-music-video_latent \\
        --dest-name myrun.latent

    # Just print what would happen
    uv run --group dev python scripts/promote_latent_for_upscale.py audio-loop-music-video_latent --dry-run

Env vars consulted (only if --output-dir / --input-dir not given):
  COMFYUI_OUTPUT_DIR  — base output dir (typically from extra_model_paths.yaml)
  COMFYUI_INPUT_DIR   — base input dir

No defaults contain absolute paths (path-privacy rule). If neither flag
nor env var is set the script errors out with a clear message.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

LATENT_GLOB = "segment_*.latent"


def find_latest_assembled_latent(output_dir: Path, workflow_name: str) -> Path:
    """Return the path of the newest ``segment_*.latent`` under
    ``<output_dir>/<workflow_name>/*/latents/``.

    Raises ``FileNotFoundError`` with a useful message when the
    workflow folder is missing or when no matching latent files exist
    under any of its per-render sub-folders.
    """
    workflow_root = output_dir / workflow_name
    if not workflow_root.is_dir():
        raise FileNotFoundError(
            f"no output folder for workflow {workflow_name!r} under {output_dir} "
            f"(expected {workflow_root}). Did you run "
            "scripts/apply_run_id_layout.py and toggle the SaveLatent node?"
        )
    candidates = list(workflow_root.glob(f"*/latents/{LATENT_GLOB}"))
    if not candidates:
        raise FileNotFoundError(
            f"no {LATENT_GLOB} files under {workflow_root}/*/latents/. "
            "Toggle the 'Save assembled latent (toggle)' node to mode=0 in "
            "the workflow UI and re-render."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_dir(arg_value: str | None, env_var: str, label: str) -> Path:
    if arg_value:
        return Path(arg_value)
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    raise SystemExit(
        f"error: {label} not set. Pass --{label.replace('_', '-')} or export {env_var}."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "workflow_name",
        help="Workflow folder name under <output-dir>, typically the .json basename "
             "(e.g. 'audio-loop-music-video_latent').",
    )
    ap.add_argument("--output-dir", help="ComfyUI output dir (env: COMFYUI_OUTPUT_DIR).")
    ap.add_argument("--input-dir", help="ComfyUI input dir (env: COMFYUI_INPUT_DIR).")
    ap.add_argument(
        "--dest-name", default="assembled_latent.latent",
        help="Filename to write under input-dir (default: %(default)s).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print what would happen.")
    args = ap.parse_args()

    output_dir = _resolve_dir(args.output_dir, "COMFYUI_OUTPUT_DIR", "output_dir")
    input_dir = _resolve_dir(args.input_dir, "COMFYUI_INPUT_DIR", "input_dir")

    src = find_latest_assembled_latent(output_dir, args.workflow_name)
    dest = input_dir / args.dest_name
    print(f"src:  {src}")
    print(f"dest: {dest}")

    if args.dry_run:
        print("(dry-run — no copy performed)")
        return 0

    if not input_dir.is_dir():
        raise SystemExit(f"error: input dir does not exist: {input_dir}")

    shutil.copy2(src, dest)
    print(f"copied {src.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
