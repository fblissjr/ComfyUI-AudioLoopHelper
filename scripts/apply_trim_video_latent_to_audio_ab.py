"""apply_trim_video_latent_to_audio_ab.

Last updated: 2026-05-10

A/B staged variant for the latent-level audio-overshoot trim.

Forks ``example_workflows/audio-loop-music-video_latent.json`` to
``internal/scratch/audio-loop-music-video_latent_LATENT_TRIM_AB.json``
and inserts a ``TrimVideoLatentToAudio`` node between
``LatentConcat #1605`` (assembled video latent) and
``LTXVTiledVAEDecode #1604.latents``. The existing F14
``TrimImageBatchToAudio`` (post-decode, image-space) remains in place
as a safety net.

A/B procedure:
  1. Render the canonical (Arm A): the post-decode F14 trim runs.
  2. Render the staged variant (Arm B): the latent trim clips
     overshoot frames BEFORE VAE decode; F14 catches any off-by-one.
  3. ffprobe both saved mp4s — durations should match exactly.
  4. Watch VAE-decode VRAM + wall-clock on each. Arm B should be
     slightly lower (saves decode work on the trimmed-off frames).

Strict scope: canonical is untouched; only writes to
``internal/scratch/`` (gitignored).

Usage:
    uv run --group dev python scripts/apply_trim_video_latent_to_audio_ab.py
    uv run --group dev python scripts/apply_trim_video_latent_to_audio_ab.py --revert
    uv run --group dev python scripts/apply_trim_video_latent_to_audio_ab.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"
STAGED = REPO_ROOT / "internal" / "scratch" / "audio-loop-music-video_latent_LATENT_TRIM_AB.json"

LATENT_CONCAT_ID = 1605
VAE_DECODE_ID = 1604
ORIG_AUDIO_GETNODE_ID = 604  # GetNode "orig_audio"
FRAME_PLANNER_ID = 1634
FRAME_PLANNER_FPS_OUT_SLOT = 4


def _apply(revert: bool, dry_run: bool) -> int:
    if revert:
        if STAGED.exists():
            if dry_run:
                print(f"would delete {STAGED.relative_to(REPO_ROOT)}")
                return 0
            STAGED.unlink()
            print(f"deleted {STAGED.relative_to(REPO_ROOT)}")
        else:
            print(f"already absent: {STAGED.relative_to(REPO_ROOT)}")
        return 0

    if dry_run:
        print(f"would fork {CANONICAL.relative_to(REPO_ROOT)} -> {STAGED.relative_to(REPO_ROOT)}")
        print(f"would splice TrimVideoLatentToAudio between #{LATENT_CONCAT_ID} and #{VAE_DECODE_ID}")
        return 0

    # Idempotence: if STAGED already has the trim wired, no-op (don't
    # re-fork from canonical and re-apply — that would discard any
    # user edits to the staged variant).
    if STAGED.exists():
        existing = WorkflowEditor(STAGED)
        for n in existing.wf["nodes"]:
            if n.get("type") == "TrimVideoLatentToAudio":
                print(f"no change (TrimVideoLatentToAudio #{n['id']} already wired in staged)")
                return 0

    STAGED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CANONICAL, STAGED)
    ed = WorkflowEditor(STAGED)

    missing = ed.require_nodes((
        LATENT_CONCAT_ID, VAE_DECODE_ID,
        ORIG_AUDIO_GETNODE_ID, FRAME_PLANNER_ID,
    ))
    if missing:
        print(f"skip (missing expected canonical node ids: {missing})")
        return 1

    decode_node = ed.find_node(VAE_DECODE_ID)
    latents_slot = WorkflowEditor.find_input_slot(decode_node, "latents")
    existing = ed.find_link_to_slot(VAE_DECODE_ID, latents_slot)
    if existing is None or existing[1] != LATENT_CONCAT_ID:
        print(f"skip (unexpected source on #{VAE_DECODE_ID}.latents: {existing})")
        return 1

    # Place the trim node visually between LatentConcat and the decoder.
    concat = ed.find_node(LATENT_CONCAT_ID)
    cx, cy = concat.get("pos", [0, 0])

    trim_id = ed.add_top_level_node(
        node_type="TrimVideoLatentToAudio",
        pos=[cx + 250, cy], size=[300, 100],
        inputs=[
            WorkflowEditor.io_in("latent", "LATENT"),
            WorkflowEditor.io_in("audio", "AUDIO"),
            WorkflowEditor.widget_in("fps", "INT"),
        ],
        outputs=[WorkflowEditor.out("latent", "LATENT")],
        widgets_values=[25],
        title="Trim latent to audio (A/B)",
    )

    # Rewire: combine.latents <- trim.output
    ed.rewire_input(VAE_DECODE_ID, latents_slot, trim_id, 0, "LATENT")
    # trim.latent <- LatentConcat.output
    ed.add_link(LATENT_CONCAT_ID, 0, trim_id, 0, "LATENT")
    # trim.audio <- orig_audio GetNode
    ed.add_link(ORIG_AUDIO_GETNODE_ID, 0, trim_id, 1, "AUDIO")
    # trim.fps <- LTXFramePlanner.fps_int
    ed.add_link(FRAME_PLANNER_ID, FRAME_PLANNER_FPS_OUT_SLOT, trim_id, 2, "INT")

    ed.save()
    print(f"wrote {STAGED.relative_to(REPO_ROOT)} (added TrimVideoLatentToAudio #{trim_id})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--revert", action="store_true", help="Delete the staged variant.")
    ap.add_argument("--dry-run", action="store_true", help="Print without writing.")
    args = ap.parse_args()
    return _apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
