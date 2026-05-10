"""apply_trim_image_batch_to_audio.

Last updated: 2026-05-10

Symptom it fixes: every saved mp4 ends with a few seconds of silence.
Container/video duration exceeds audio stream duration by 4-10 seconds
across every recent render (verified via ffprobe sweep across 20
random renders / 3 distinct audio sources).

Root cause: fixed-stride iteration math. Per-iter video generation
contributes a uniform ``stride_pixel_frames`` of new content, plus a
fixed ``initial_render`` contribution; total assembled video matches
``245 + N * 448`` exactly (canonical defaults), where
``N = floor(audio_duration / stride_seconds)``. The total can exceed
audio by up to ``window_seconds - stride_seconds`` per run. The
``-shortest`` flag in VHS_VideoCombine's ffmpeg invocation does not
truncate ``-c:v copy`` streams, so the saved container ends up the
longer of audio/video.

Fix: insert ``TrimImageBatchToAudio`` between the IMAGE source feeding
``VHS_VideoCombine.images`` and the combine itself. The trim node
clips the batch to ``floor(audio_duration * fps)`` frames using the
same audio source already wired to ``VHS_VideoCombine.audio``. Mp4
output ends up exactly audio-length; the few overshoot frames at the
tail are dropped before muxing. ``fps`` autowires from
``LTXFramePlanner.fps_int`` when present so changing the planner's
fps cascades through.

Skips workflows that already have the trim wired (idempotent), have
no active VHS_VideoCombine, or are post-loop processors (upscale +
seam refinement) where the input video already has correct length.

Compatibility:
  - Independent of all sigma / sampler / conditioning apply scripts.
  - Independent of F-pair audit invariants. Adds its own
    ``trim_image_batch_to_audio_present`` warn-level audit.

Usage:
    uv run --group dev python scripts/apply_trim_image_batch_to_audio.py
    uv run --group dev python scripts/apply_trim_image_batch_to_audio.py --revert
    uv run --group dev python scripts/apply_trim_image_batch_to_audio.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent

# Workflows the trim DOES NOT apply to: third-party reference files
# we don't own. Post-loop processors (upscale, seam refinement) DO get
# the trim — they take a VHS_LoadVideo whose audio passes through to
# VHS_VideoCombine.audio, so the trim correctly clips any inherited
# video > audio mismatch from older loop outputs (no-op when input is
# already correct).
SKIP_FILES = {
    "edit_anything_v2v_reference.json",
    "upscale_3pass_reference.json",
}


def _find_trim_for_combine(ed: WorkflowEditor, combine_id: int) -> int | None:
    """Return the TrimImageBatchToAudio node id feeding combine.images,
    or None if combine.images is not fed by a trim node."""
    link = ed.find_link_to_slot(combine_id, 0)
    if link is None:
        return None
    src = ed.find_node(link[1])
    return src["id"] if src.get("type") == "TrimImageBatchToAudio" else None


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    if wf_path.name in SKIP_FILES:
        return "skip (excluded by SKIP_FILES)"
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    combines = [
        n for n in ed.wf["nodes"]
        if n.get("type") == "VHS_VideoCombine" and n.get("mode", 0) == 0
    ]
    if not combines:
        return "skip (no active VHS_VideoCombine)"
    if len(combines) > 1:
        return f"skip (multiple active VHS_VideoCombine: {[c['id'] for c in combines]})"

    combine = combines[0]
    combine_id = combine["id"]
    existing_trim_id = _find_trim_for_combine(ed, combine_id)

    if revert:
        if existing_trim_id is None:
            return "already reverted"
        # Splice out: rewire combine.images directly to the trim's IMAGE source.
        trim_in_link = ed.find_link_to_slot(existing_trim_id, 0)
        if trim_in_link is None:
            return f"skip (TrimImageBatchToAudio #{existing_trim_id}.images has no link)"
        _, src_id, src_slot, *_ = trim_in_link
        if dry_run:
            return f"would revert (remove #{existing_trim_id}, restore #{src_id}.{src_slot} -> combine.images)"
        ed.rewire_input(combine_id, 0, src_id, src_slot, "IMAGE")
        ed.remove_node_and_links(existing_trim_id)
        ed.save()
        return f"reverted (removed TrimImageBatchToAudio #{existing_trim_id})"

    if existing_trim_id is not None:
        return f"no change (TrimImageBatchToAudio #{existing_trim_id} already wired)"

    images_link = ed.find_link_to_slot(combine_id, 0)
    audio_link = ed.find_link_to_slot(combine_id, 1)
    if images_link is None:
        return "skip (VHS_VideoCombine.images has no link)"
    if audio_link is None:
        return "skip (VHS_VideoCombine.audio has no link — nothing to size against)"

    _, img_src, img_src_slot, *_ = images_link
    _, aud_src, aud_src_slot, *_ = audio_link

    fp = next(
        (n for n in ed.wf["nodes"] if n.get("type") == "LTXFramePlanner"),
        None,
    )

    cx, cy = combine.get("pos", [0, 0])
    if dry_run:
        fps_note = "fps from LTXFramePlanner" if fp else "fps widget=25"
        return f"would update (insert TrimImageBatchToAudio between #{img_src} and #{combine_id}, {fps_note})"

    new_id = ed.add_top_level_node(
        node_type="TrimImageBatchToAudio",
        pos=[cx - 320, cy],
        size=[280, 90],
        inputs=[
            WorkflowEditor.io_in("images", "IMAGE"),
            WorkflowEditor.io_in("audio", "AUDIO"),
            WorkflowEditor.widget_in("fps", "INT"),
        ],
        outputs=[WorkflowEditor.out("images", "IMAGE")],
        widgets_values=[25],
        title="Trim to audio",
    )

    ed.rewire_input(combine_id, 0, new_id, 0, "IMAGE")
    ed.add_link(img_src, img_src_slot, new_id, 0, "IMAGE")
    ed.add_link(aud_src, aud_src_slot, new_id, 1, "AUDIO")
    if fp is not None:
        # LTXFramePlanner.fps_int is output slot 4 (per nodes.py LTXFramePlanner.execute).
        ed.add_link(fp["id"], 4, new_id, 2, "INT")

    ed.save()
    fps_note = "fps from LTXFramePlanner" if fp else "fps widget=25"
    return f"updated (added TrimImageBatchToAudio #{new_id}, {fps_note})"


def _iter_workflows() -> list[Path]:
    paths: list[Path] = []
    for d in (REPO_ROOT / "example_workflows", REPO_ROOT / "internal" / "workflows"):
        if not d.exists():
            continue
        paths.extend(sorted(d.rglob("*.json")))
    return paths


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} TrimImageBatchToAudio across loop workflows...")
    fail = 0
    for wf_path in _iter_workflows():
        rel = wf_path.relative_to(REPO_ROOT)
        status = _apply_one(wf_path, revert, dry_run)
        print(f"  {rel}: {status}")
        if status.startswith("load error"):
            fail += 1
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--revert", action="store_true", help="Undo the change.")
    ap.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
