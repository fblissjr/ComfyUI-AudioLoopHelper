#!/usr/bin/env python
"""Build the AV inversion probe — full real video frozen + 2 s audio seed -> generate audio.

The exact opposite of the shipped pipeline. Today: PARTIAL video (1 init frame) +
FULL audio (frozen) -> model generates the video. The inversion: FULL video (frozen
context) + PARTIAL audio (2 s seed, rest generated) -> model generates the AUDIO. You
watch the real footage and hear what the model inferred for it.

Forks the audio-extension variant (which already carries `AudioTemporalMask` on the
audio AND the generated-audio decode+mux) and makes it the inversion with bounded edits:

  1. add a VHS_LoadVideo to ingest the full clip (video frames + its own audio, in sync)
  2. video frames -> the existing resize/preprocess/InplaceKJ chain. InplaceKJ#531 keeps
     the canonical widgets ['1', 1, 0] (= num_images=1, strength=1.0, index=0), which the
     shipped i2v already uses to FREEZE the inserted frames (mask = 1 - strength = 0).
     Feeding the full-clip batch there freezes the ENTIRE video as context.
  3. audio seed sourced from the SAME clip (VHS audio) -> existing trim/encode/AudioTemporalMask

The generated-audio output handling (decode the sampled audio latent + mux it instead of
the input passthrough) lives in the av_extension base and is inherited — both probes need
to HEAR the generated audio. The video is kept (frozen video decodes back to the clip).

Caveats (render-gate pending — structural validation only):
  * Window length is planner-driven: LTXFramePlanner#1634 = 19.88 s @ 25 fps -> 497 px
    frames. Use a clip >= ~20 s so it fills the window and is fully frozen; a shorter clip
    leaves the video tail (and a misaligned audio seed) to generate. WINDOW_FRAME_CAP must
    track the planner duration, not the (stale) EmptyLTXVLatentVideo widget.
  * Run with iterations=1 (single window; the loop body re-freezes audio via the subgraph
    LTXVAudioVideoMask). Do NOT touch first_frame_guide_strength here — it feeds only the
    loop-body invoker, not the initial render.
  * Keep the neutral prompt — describe only the scene, never the audio (see
    example_workflows/working_docs/av_inversion_test_examples.md).

Pre-req: the audio-extension variant must exist (run apply_av_extension_probe.py first).

Usage:
    uv run --group dev python scripts/build_av_inversion_workflow.py            # build
    uv run --group dev python scripts/build_av_inversion_workflow.py --dry-run  # preview
    uv run --group dev python scripts/build_av_inversion_workflow.py --revert   # delete output
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from workflow_utils import WorkflowEditor  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "example_workflows" / "experimental"
SRC = EXP / "audio-loop-music-video_latent_av_extension.json"
OUT = EXP / "audio-loop-music-video_latent_av_inversion.json"

# Node IDs (shared with the canonical / av_extension variant).
RESIZE = 445          # LTXSmartImageResize — image input slot 0
TRIM_ACTUAL = 567     # TrimAudioDuration feeding actual_audio — audio input slot 0
INPLACE = 531         # LTXVImgToVideoInplaceKJ — left at canonical ['1', 1, 0] (freeze)
LOAD_IMAGE = 444      # LoadImage — i2v init source, orphaned once resize feeds from VHS
LOAD_AUDIO = 565      # LoadAudio — audio source, orphaned once trim feeds from VHS

# Planner-driven window: LTXFramePlanner#1634 = 19.88 s @ 25 fps -> 497 px frames.
# (The EmptyLTXVLatentVideo length widget reads 353 but is WIRED from the planner, so the
# real window is 497; cap must match the planner, not the stale widget.)
WINDOW_FRAME_CAP = 497

VHS_WIDGETS = {
    "video": "your_clip.mp4",
    "force_rate": 25,
    "custom_width": 0,
    "custom_height": 0,
    "frame_load_cap": WINDOW_FRAME_CAP,
    "skip_first_frames": 0,
    "select_every_nth": 1,
    "format": "LTXV",
}


def build(dry_run: bool = False) -> int:
    if not SRC.exists():
        print(f"ERROR: base variant missing: {SRC.relative_to(REPO)}")
        print("Run: uv run --group dev python scripts/apply_av_extension_probe.py")
        return 1

    ed = WorkflowEditor(SRC)
    for nid in (RESIZE, TRIM_ACTUAL, INPLACE):
        if not ed.has_node(nid):
            print(f"ERROR: expected node #{nid} missing — base variant drifted.")
            return 1
    # The generated-audio decode+mux is provided by the av_extension base; refuse if absent
    # (means the base predates that fix — re-run apply_av_extension_probe.py).
    if not ed.find_nodes_by_type("LTXVAudioVAEDecode"):
        print("ERROR: base variant lacks LTXVAudioVAEDecode (generated-audio mux).")
        print("Re-run: uv run --group dev python scripts/apply_av_extension_probe.py")
        return 1

    # 1. VHS_LoadVideo: full clip ingress (frames + synced audio).
    vhs_id = ed.add_top_level_node(
        "VHS_LoadVideo",
        pos=[30, 2120],
        size=[300, 300],
        inputs=[
            WorkflowEditor.io_in("meta_batch", "VHS_BatchManager"),
            WorkflowEditor.io_in("vae", "VAE"),
        ],
        outputs=[
            WorkflowEditor.out("IMAGE", "IMAGE"),
            WorkflowEditor.out("frame_count", "INT"),
            WorkflowEditor.out("audio", "AUDIO"),
            WorkflowEditor.out("video_info", "VHS_VIDEOINFO"),
        ],
        widgets_values=copy.deepcopy(VHS_WIDGETS),
        properties={"Node name for S&R": "VHS_LoadVideo", "cnr_id": "comfyui-videohelpersuite"},
        title="AV Inversion: full clip (video frozen + audio seed)",
    )

    # 2. Full video into the existing resize -> preprocess -> InplaceKJ chain. InplaceKJ#531
    # keeps its canonical widgets ['1', 1, 0] (strength=1.0 -> mask=0 = frozen, index=0);
    # feeding the full-clip batch there freezes the ENTIRE video as context. (Do NOT rewrite
    # the widgets — the order is [num_images, strength, index], so ['1', 0, 1.0] would set
    # strength=0 and NOT freeze.)
    ed.rewire_input(RESIZE, 0, vhs_id, 0, "IMAGE")          # resize.image <- VHS frames

    # 3. Audio seed from the SAME clip (guaranteed sync).
    ed.rewire_input(TRIM_ACTUAL, 0, vhs_id, 2, "AUDIO")     # actual-audio trim <- VHS audio

    # 4. Remove the now-orphaned i2v init sources (no consumers after the rewires above).
    for nid in (LOAD_IMAGE, LOAD_AUDIO):
        if ed.has_node(nid) and not ed.find_links_from(nid):
            ed.remove_node_and_links(nid)

    if dry_run:
        print("--dry-run: would write", OUT.relative_to(REPO))
        print(f"  + VHS_LoadVideo#{vhs_id} (frame_load_cap={WINDOW_FRAME_CAP})")
        print(f"  resize#{RESIZE}.image <- VHS frames (InplaceKJ#{INPLACE} unchanged = freeze)")
        print(f"  trim#{TRIM_ACTUAL}.audio <- VHS audio")
        print(f"  remove orphaned LoadImage#{LOAD_IMAGE}, LoadAudio#{LOAD_AUDIO}")
        print("  generated-audio decode+mux inherited from av_extension base")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ed.save(OUT)
    print(f"Wrote {OUT.relative_to(REPO)} (VHS_LoadVideo#{vhs_id}).")
    print(f"Load a clip >= ~20s (fills the {WINDOW_FRAME_CAP}-frame window), neutral prompt, "
          "iterations=1. Render-gate pending.")
    return 0


def revert() -> int:
    if OUT.exists():
        OUT.unlink()
        print(f"Removed {OUT.relative_to(REPO)}.")
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
