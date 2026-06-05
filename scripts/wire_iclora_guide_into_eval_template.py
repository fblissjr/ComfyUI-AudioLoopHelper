#!/usr/bin/env python3
"""One-shot: add LTXAddVideoICLoRAGuideAdvanced + VHS_LoadVideo to the
single-shot audio-swap eval template so the IC-LoRA reference-video
pathway is wired correctly for our trained LoRA.

Our LoRA was trained with video_to_video / condition mode + reference =
the same clip (multi-frame video, not a static frame). At inference the
LoRA's cross-modal-attention adapters need that reference pathway active
or the output degrades vs baseline.

Inserts the guide BETWEEN the existing LTXVReferenceAudio output and
CFGGuider, so the conditioning chain becomes:

    LTXVConditioning (#164)
        → LTXVReferenceAudio (#1632, bypassed)
        → LTXAddVideoICLoRAGuideAdvanced (NEW)        ← reference video tokens added here
        → CFGGuider (#153)

VHS_LoadVideo loads audio_iclora_eval_ref.mp4 from ComfyUI's input dir (the
100-BPM neutral reference video generated for E1.1 eval). The guide's
latent comes from EmptyLTXVLatentVideo (#344); vae from the video VAE
loader (#1537).

Idempotent: if the guide is already present, the script no-ops.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/ on path for workflow_utils
sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflow_utils import WorkflowEditor


# Existing node IDs in the template (verified via inspection)
CFG_GUIDER_ID = 153
LTXV_REFERENCE_AUDIO_ID = 1632      # bypassed pass-through node sitting between conditioning + CFGGuider
VIDEO_VAE_LOADER_ID = 1537          # VAELoaderKJ for LTX23_video_vae
EMPTY_LATENT_ID = 344               # EmptyLTXVLatentVideo

# What slot indices on each node (from schema inspection)
# CFGGuider.inputs: [model, positive, negative]                — positive=1, negative=2
# LTXVReferenceAudio.outputs: [MODEL, positive, negative]      — positive=1, negative=2
# LTXAddVideoICLoRAGuideAdvanced.inputs:
#   [positive=0, negative=1, vae=2, latent=3, image=4, ...widget inputs after]
# LTXAddVideoICLoRAGuideAdvanced.outputs: [positive, negative] — positive=0, negative=1
# VHS_LoadVideo.outputs: [IMAGE=0, frame_count=1, audio=2, video_info=3]
# VAELoaderKJ.outputs: [VAE=0]
# EmptyLTXVLatentVideo.outputs: [LATENT=0]


def already_wired(ed: WorkflowEditor) -> bool:
    return bool(ed.find_nodes_by_type("LTXAddVideoICLoRAGuideAdvanced"))


def add_vhs_load_video(ed: WorkflowEditor, video_filename: str, frame_load_cap: int) -> int:
    nid = ed.next_node_id()
    node = {
        "id": nid,
        "type": "VHS_LoadVideo",
        "title": "IC-LoRA Reference Video (audio-swap eval)",
        "pos": [-800, -1200],   # rough placement, ComfyUI repositions as needed
        "size": [320, 400],
        "mode": 0,
        "flags": {},
        "order": 0,
        "inputs": [],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "frame_count", "type": "INT", "links": [], "slot_index": 1},
            {"name": "audio", "type": "AUDIO", "links": [], "slot_index": 2},
            {"name": "video_info", "type": "VHS_VIDEOINFO", "links": [], "slot_index": 3},
        ],
        "properties": {"Node name for S&R": "VHS_LoadVideo"},
        "widgets_values": {
            "video": video_filename,
            "force_rate": 25,
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": frame_load_cap,
            "skip_first_frames": 0,
            "select_every_nth": 1,
            "format": "LTXV",
            "videopreview": {
                "hidden": False, "paused": False,
                "params": {"filename": video_filename, "type": "input",
                           "format": "video/mp4", "force_rate": 25,
                           "frame_load_cap": frame_load_cap},
            },
        },
    }
    ed.add_node(node)
    return nid


def add_guide_advanced(ed: WorkflowEditor) -> int:
    nid = ed.next_node_id()
    node = {
        "id": nid,
        "type": "LTXAddVideoICLoRAGuideAdvanced",
        "title": "IC-LoRA Guide (audio-swap eval)",
        "pos": [-400, -1200],
        "size": [400, 350],
        "mode": 0,
        "flags": {},
        "order": 0,
        "inputs": [
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
            {"name": "latent", "type": "LATENT", "link": None},
            {"name": "image", "type": "IMAGE", "link": None},
        ],
        "outputs": [
            {"name": "positive", "type": "CONDITIONING", "links": [], "slot_index": 0},
            {"name": "negative", "type": "CONDITIONING", "links": [], "slot_index": 1},
        ],
        "properties": {"Node name for S&R": "LTXAddVideoICLoRAGuideAdvanced"},
        # Widgets in schema order: frame_idx, strength, latent_downscale_factor,
        # crop, use_tiled_encode, tile_size, tile_overlap, attention_strength
        "widgets_values": [
            0,            # frame_idx — start at frame 0
            1.0,          # strength
            1.0,          # latent_downscale_factor
            "disabled",   # crop
            False,        # use_tiled_encode
            256,          # tile_size
            64,           # tile_overlap
            1.0,          # attention_strength — knob for amplifying the IC-LoRA effect
        ],
    }
    ed.add_node(node)
    return nid


def rewire_cfg_guider_through_guide(ed: WorkflowEditor, guide_id: int) -> None:
    """Sever LTXVReferenceAudio → CFGGuider.{positive,negative} and route
    them through the new guide first.

    Before:
        LTXVReferenceAudio.positive (link L1) → CFGGuider.positive
        LTXVReferenceAudio.negative (link L2) → CFGGuider.negative

    After:
        LTXVReferenceAudio.positive (link L1) → guide.positive
        LTXVReferenceAudio.negative (link L2) → guide.negative
        guide.positive (NEW) → CFGGuider.positive
        guide.negative (NEW) → CFGGuider.negative
    """
    # Find existing links into CFGGuider's positive (slot 1) and negative (slot 2).
    # find_link_to_slot returns the link tuple directly: [link_id, src, src_slot, tgt, tgt_slot, dtype].
    pos_link = ed.find_link_to_slot(CFG_GUIDER_ID, 1)
    neg_link = ed.find_link_to_slot(CFG_GUIDER_ID, 2)
    if pos_link is None or neg_link is None:
        raise RuntimeError(
            f"Expected CFGGuider({CFG_GUIDER_ID}) to have positive+negative links; "
            f"got positive={pos_link} negative={neg_link}"
        )

    pos_link_id, pos_src_node, pos_src_slot = pos_link[0], pos_link[1], pos_link[2]
    neg_link_id, neg_src_node, neg_src_slot = neg_link[0], neg_link[1], neg_link[2]

    # Sever the existing links into CFGGuider
    ed.remove_link(pos_link_id)
    ed.remove_link(neg_link_id)

    # Route source → guide
    ed.add_link(pos_src_node, pos_src_slot, guide_id, 0, "CONDITIONING")  # → guide.positive (slot 0)
    ed.add_link(neg_src_node, neg_src_slot, guide_id, 1, "CONDITIONING")  # → guide.negative (slot 1)
    # Route guide → CFGGuider
    ed.add_link(guide_id, 0, CFG_GUIDER_ID, 1, "CONDITIONING")  # guide.positive → CFGGuider.positive
    ed.add_link(guide_id, 1, CFG_GUIDER_ID, 2, "CONDITIONING")  # guide.negative → CFGGuider.negative


def wire_supporting_inputs(ed: WorkflowEditor, vhs_id: int, guide_id: int) -> None:
    """Wire VHS_LoadVideo.IMAGE → guide.image, video VAE → guide.vae,
    EmptyLTXVLatentVideo → guide.latent. The VAE + latent are BRANCHED off
    their existing single consumers (multiple consumers OK)."""
    # IMAGE: new edge from VHS
    ed.add_link(vhs_id, 0, guide_id, 4, "IMAGE")               # VHS.IMAGE → guide.image (slot 4)
    # VAE: branch off video VAE loader (slot 0 is the VAE output)
    ed.add_link(VIDEO_VAE_LOADER_ID, 0, guide_id, 2, "VAE")    # video VAE → guide.vae (slot 2)
    # LATENT: branch off EmptyLTXVLatentVideo
    ed.add_link(EMPTY_LATENT_ID, 0, guide_id, 3, "LATENT")     # EmptyLatent → guide.latent (slot 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path,
                    default=Path("internal/workflows/audio_swap_eval/_template_e1_run1.json"))
    ap.add_argument("--ref-video", default="audio_iclora_eval_ref.mp4",
                    help="filename in ComfyUI input dir (relative)")
    ap.add_argument("--frame-cap", type=int, default=73,
                    help="frames to load from ref video (73 = full 3s @ 25fps)")
    args = ap.parse_args()

    ed = WorkflowEditor(args.template)

    if already_wired(ed):
        print(f"already wired: LTXAddVideoICLoRAGuideAdvanced present in {args.template} — no-op")
        return

    vhs_id = add_vhs_load_video(ed, args.ref_video, args.frame_cap)
    guide_id = add_guide_advanced(ed)
    rewire_cfg_guider_through_guide(ed, guide_id)
    wire_supporting_inputs(ed, vhs_id, guide_id)
    ed.save(verbose=True)

    print(f"wrote {args.template}")
    print(f"  added VHS_LoadVideo #{vhs_id} → {args.ref_video}")
    print(f"  added LTXAddVideoICLoRAGuideAdvanced #{guide_id}")
    print(f"  rewired CFGGuider conditioning chain through the guide")
    print()
    print("Now re-run scripts/generate_audio_swap_workflows.py to refresh the 14 variants.")


if __name__ == "__main__":
    main()
