#!/usr/bin/env python
"""Build the keyframe auto-extract variant — keyframes sampled from the clip, no hand-loading.

Forks audio-loop-music-video_latent_keyframe.json. That workflow drives
LTXIterKeyframeSchedule#2042 from 3 keyframe latents, each its own
LoadImage -> LTXSmartImageResize -> ... -> VAEEncode chain (you hand-load 3 shots).

This replaces the 3 LoadImage sources with frames pulled straight from a loaded video:

    VHS_LoadVideo (clip) -> EvenlySpacedKeyframes(count=3) -> 3x GetImageRangeFromBatch
      (start_index 0/1/2, num_frames 1) -> the 3 existing resize/encode chains -> schedule

EvenlySpacedKeyframes(3) picks first/middle/last frames of the clip (endpoints always
included); the 3 GetImageRangeFromBatch nodes split that 3-frame batch into singles for the
schedule's 3 separate keyframe_latent inputs. count is fixed at 3 to match the schedule's 3
slots. The old LoadImage chains are left in place but orphaned (override path: rewire a
resize back to a LoadImage, or swap GetImageRangeFromBatch start_index to hand-pick frames).

CLOBBER WARNING: re-forks from keyframe.json, so re-running DROPS any hand edits made
directly to the autoextract output (and inherits whatever state keyframe.json is in —
regenerate that first via apply_keyframe_iter_anchor.py if it's stale). Shipped JSON is the
source of truth; `git diff` the output before re-running.

Usage:
    uv run --group dev python scripts/build_keyframe_autoextract_workflow.py            # build
    uv run --group dev python scripts/build_keyframe_autoextract_workflow.py --dry-run  # preview
    uv run --group dev python scripts/build_keyframe_autoextract_workflow.py --revert   # delete output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from workflow_utils import WorkflowEditor  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "example_workflows" / "audio-loop-music-video_latent_keyframe.json"
OUT = REPO / "example_workflows" / "experimental" / "audio-loop-music-video_latent_keyframe_autoextract.json"

# The 3 keyframe-chain resize nodes (image input slot 0) fed by the 3 LoadImage shots.
RESIZE_NODES = [2031, 2035, 2039]
# The 3 keyframe LoadImage nodes feeding those resizes — orphaned once we rewire to the
# auto-extracted frames, so remove them. (Init/fallback LoadImage#444 stays.)
KEYFRAME_LOADIMAGES = [2030, 2034, 2038]
KEYFRAME_COUNT = 3  # matches LTXIterKeyframeSchedule#2042's 3 keyframe_latent slots

VHS_WIDGETS = {
    "video": "your_clip.mp4", "force_rate": 25, "custom_width": 0, "custom_height": 0,
    "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1, "format": "LTXV",
}


def build(dry_run: bool = False) -> int:
    if not SRC.exists():
        print(f"ERROR: base workflow missing: {SRC.relative_to(REPO)}")
        return 1
    ed = WorkflowEditor(SRC)
    for nid in RESIZE_NODES:
        if not ed.has_node(nid):
            print(f"ERROR: keyframe resize node #{nid} missing — base drifted.")
            return 1

    vhs = ed.add_top_level_node(
        "VHS_LoadVideo", pos=[-360, 2720], size=[300, 300],
        inputs=[WorkflowEditor.io_in("meta_batch", "VHS_BatchManager"), WorkflowEditor.io_in("vae", "VAE")],
        outputs=[WorkflowEditor.out("IMAGE", "IMAGE"), WorkflowEditor.out("frame_count", "INT"),
                 WorkflowEditor.out("audio", "AUDIO"), WorkflowEditor.out("video_info", "VHS_VIDEOINFO")],
        widgets_values=dict(VHS_WIDGETS),
        properties={"Node name for S&R": "VHS_LoadVideo", "cnr_id": "comfyui-videohelpersuite"},
        title="Keyframe source clip (auto-extract)",
    )
    esk = ed.add_top_level_node(
        "EvenlySpacedKeyframes", pos=[-360, 3060], size=[260, 82],
        inputs=[WorkflowEditor.io_in("images", "IMAGE")],
        outputs=[WorkflowEditor.out("IMAGE", "IMAGE")],
        widgets_values=[KEYFRAME_COUNT],
        properties={"Node name for S&R": "EvenlySpacedKeyframes", "aux_id": "fblissjr/ComfyUI-AudioLoopHelper"},
        title=f"Auto keyframes ({KEYFRAME_COUNT} evenly-spaced)",
    )
    ed.add_link(vhs, 0, esk, 0, "IMAGE")  # VHS frames -> EvenlySpacedKeyframes

    # Split the count-frame batch into singles and feed the 3 existing resize chains.
    for i, resize_id in enumerate(RESIZE_NODES):
        sel = ed.add_top_level_node(
            "GetImageRangeFromBatch", pos=[-40, 2720 + i * 200], size=[260, 100],
            inputs=[WorkflowEditor.io_in("images", "IMAGE")],
            outputs=[WorkflowEditor.out("IMAGE", "IMAGE"), WorkflowEditor.out("MASK", "MASK")],
            widgets_values=[i, 1],  # start_index=i, num_frames=1
            properties={"Node name for S&R": "GetImageRangeFromBatch", "cnr_id": "comfyui-kjnodes"},
            title=f"keyframe {i + 1} (frame {i})",
        )
        ed.add_link(esk, 0, sel, 0, "IMAGE")          # batch -> selector
        ed.rewire_input(resize_id, 0, sel, 0, "IMAGE")  # resize.image <- this single frame

    # Remove the now-orphaned keyframe LoadImage nodes (resizes no longer read them).
    for nid in KEYFRAME_LOADIMAGES:
        if ed.has_node(nid) and not ed.find_links_from(nid):
            ed.remove_node_and_links(nid)

    if dry_run:
        print("--dry-run: would write", OUT.relative_to(REPO))
        print(f"  + VHS_LoadVideo#{vhs} -> EvenlySpacedKeyframes#{esk}(count={KEYFRAME_COUNT})")
        print(f"  + 3x GetImageRangeFromBatch (start 0/1/2) -> resize {RESIZE_NODES}")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ed.save(OUT)
    print(f"Wrote {OUT.relative_to(REPO)}.")
    print("Drop your clip into the 'Keyframe source clip' VHS_LoadVideo; keyframes auto-extract")
    print("(first/middle/last). Hand-pick by changing a GetImageRangeFromBatch start_index. Render-gate pending.")
    return 0


def revert() -> int:
    if OUT.exists():
        OUT.unlink(); print(f"Removed {OUT.relative_to(REPO)}.")
    else:
        print("Nothing to revert.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    return revert() if args.revert else build(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
