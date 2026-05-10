"""Build the post-loop seam-zone refinement workflow from scratch.

Last updated: 2026-05-04

Produces `internal/workflows/seam_zone_refinement.draft.json` per the
topology described in `internal/design/polish_passes_design.md §P5`:

    VHS_LoadVideo → VAEEncode
                       ↓
            LatentSeamZoneMask (multi-band mask centered on each
                                internal iteration boundary)
                       ↓
            LTXVConcatAVLatent (re-attach empty audio latent;
                                audio FROZEN via mask=0)
                       ↓
            SamplerCustomAdvanced (3-step σ-tail [0.85, 0.7250,
                                   0.4219, 0.0], euler, CFG=1)
                       ↓
            LTXVSeparateAVLatent → LTXVCropGuides → LTXVTiledVAEDecode
                       ↓
            VHS_VideoCombine (with original audio passthrough)

Gating: this workflow is the corrective half of the seam-zone story.
Run `scripts/diagnose_overlap_seams.py` against an assembled latent
first; if boundary-zone scores exceed the noise floor by ~1.5x or
more, this corrective pass is justified. If not, skip it.

Idempotent: re-running overwrites the output file with deterministic
content. `--dry-run` prints the node table without writing. `--revert`
deletes the output file if present.

Usage:
    uv run --group dev python scripts/build_seam_refinement_workflow.py
    uv run --group dev python scripts/build_seam_refinement_workflow.py --dry-run
    uv run --group dev python scripts/build_seam_refinement_workflow.py --revert

After re-building, chain `apply_trim_image_batch_to_audio.py` to splice
the loop-audio-overshoot fix back in (skeleton vs. apply convention per
`scripts/CLAUDE.md`):

    uv run --group dev python scripts/build_seam_refinement_workflow.py
    uv run --group dev python scripts/apply_trim_image_batch_to_audio.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "internal" / "workflows" / "seam_zone_refinement.draft.json"

# Loaders — same names as the canonical loop / upscale workflows.
UNET_NAME = "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"
CLIP_NAMES = ("gemma_3_12B_it_fpmixed.safetensors", "ltx-2.3_text_projection_bf16.safetensors")
VIDEO_VAE_NAME = "vae/LTX23_video_vae_bf16.safetensors"
AUDIO_VAE_NAME = "vae/LTX23_audio_vae_bf16.safetensors"

REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
FRAME_RATE = 25


# Slot-dict helpers live on WorkflowEditor as static methods.
_basic_io_in = WorkflowEditor.io_in
_widget_in = WorkflowEditor.widget_in
_out = WorkflowEditor.out


def build(ed: WorkflowEditor) -> dict[str, int]:
    """Add all nodes and links. Returns a map of role-name -> node id."""

    ids: dict[str, int] = {}

    ids["unet"] = ed.add_top_level_node(
        node_type="UNETLoader",
        pos=[-1900, 200], size=[480, 82],
        inputs=[],
        outputs=[_out("MODEL", "MODEL")],
        widgets_values=[UNET_NAME, "default"],
    )
    ids["clip"] = ed.add_top_level_node(
        node_type="DualCLIPLoader",
        pos=[-1900, 320], size=[480, 130],
        inputs=[],
        outputs=[_out("CLIP", "CLIP")],
        widgets_values=[CLIP_NAMES[0], CLIP_NAMES[1], "ltxv", "default"],
    )
    ids["video_vae"] = ed.add_top_level_node(
        node_type="VAELoaderKJ",
        pos=[-1900, 480], size=[460, 130],
        inputs=[],
        outputs=[_out("VAE", "VAE")],
        widgets_values=[VIDEO_VAE_NAME, "main_device", "bf16"],
    )
    ids["audio_vae"] = ed.add_top_level_node(
        node_type="VAELoaderKJ",
        pos=[-1900, 640], size=[460, 130],
        inputs=[],
        outputs=[_out("VAE", "VAE")],
        widgets_values=[AUDIO_VAE_NAME, "main_device", "bf16"],
    )

    ids["load_video"] = ed.add_top_level_node(
        node_type="VHS_LoadVideo",
        pos=[-1900, 800], size=[460, 280],
        inputs=[
            _basic_io_in("meta_batch", "VHS_BatchManager"),
            _basic_io_in("vae", "VAE"),
            _widget_in("frame_load_cap", "INT"),
        ],
        outputs=[
            _out("IMAGE", "IMAGE"),
            _out("frame_count", "INT"),
            _out("audio", "AUDIO"),
            _out("video_info", "VHS_VIDEOINFO"),
        ],
        widgets_values={
            "video": "loop_output.mp4",
            "force_rate": FRAME_RATE,
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": 0,
            "skip_first_frames": 0,
            "select_every_nth": 1,
            "format": "Wildcard",
        },
        title="Load loop output",
    )

    ids["pos_text"] = ed.add_top_level_node(
        node_type="CLIPTextEncode",
        pos=[-1400, 320], size=[400, 200],
        inputs=[_basic_io_in("clip", "CLIP")],
        outputs=[_out("CONDITIONING", "CONDITIONING")],
        widgets_values=[""],
        title="Positive prompt (match loop)",
    )
    ids["neg_text"] = ed.add_top_level_node(
        node_type="CLIPTextEncode",
        pos=[-1400, 540], size=[400, 100],
        inputs=[_basic_io_in("clip", "CLIP")],
        outputs=[_out("CONDITIONING", "CONDITIONING")],
        widgets_values=[""],
        title="Negative (empty)",
    )
    ids["zero_neg"] = ed.add_top_level_node(
        node_type="ConditioningZeroOut",
        pos=[-940, 540], size=[210, 26],
        inputs=[_basic_io_in("conditioning", "CONDITIONING")],
        outputs=[_out("CONDITIONING", "CONDITIONING")],
        widgets_values=[],
    )
    ids["ltx_cond"] = ed.add_top_level_node(
        node_type="LTXVConditioning",
        pos=[-700, 360], size=[270, 90],
        inputs=[
            _basic_io_in("positive", "CONDITIONING"),
            _basic_io_in("negative", "CONDITIONING"),
            _widget_in("frame_rate", "FLOAT"),
        ],
        outputs=[
            _out("positive", "CONDITIONING"),
            _out("negative", "CONDITIONING"),
        ],
        widgets_values=[FRAME_RATE],
    )

    ids["vae_encode"] = ed.add_top_level_node(
        node_type="VAEEncode",
        pos=[-1400, 800], size=[210, 50],
        inputs=[
            _basic_io_in("pixels", "IMAGE"),
            _basic_io_in("vae", "VAE"),
        ],
        outputs=[_out("LATENT", "LATENT")],
        widgets_values=[],
    )
    ids["seam_mask"] = ed.add_top_level_node(
        node_type="LatentSeamZoneMask",
        pos=[-1100, 800], size=[330, 200],
        inputs=[
            _basic_io_in("latent", "LATENT"),
            _widget_in("iteration_count", "INT"),
            _widget_in("window_latents", "INT"),
            _widget_in("overlap_latents", "INT"),
            _widget_in("seam_band_seconds", "FLOAT"),
            _widget_in("edge_taper_seconds", "FLOAT"),
            _widget_in("fps", "FLOAT"),
        ],
        outputs=[_out("LATENT", "LATENT")],
        # Defaults match the schema; user dials iteration_count etc. to match the loop.
        widgets_values=[1, 8, 2, 0.96, 0.0, FRAME_RATE],
        title="Seam-zone mask (multi-band)",
    )
    ids["empty_audio"] = ed.add_top_level_node(
        node_type="LTXVEmptyLatentAudio",
        pos=[-700, 1000], size=[260, 110],
        inputs=[
            _basic_io_in("audio_vae", "VAE"),
            _widget_in("frames_number", "INT"),
            _widget_in("frame_rate", "INT"),
        ],
        outputs=[_out("Latent", "LATENT")],
        widgets_values=[497, FRAME_RATE, 1],
    )
    ids["av_concat"] = ed.add_top_level_node(
        node_type="LTXVConcatAVLatent",
        pos=[-450, 900], size=[200, 50],
        inputs=[
            _basic_io_in("video_latent", "LATENT"),
            _basic_io_in("audio_latent", "LATENT"),
        ],
        outputs=[_out("latent", "LATENT")],
        widgets_values=[],
    )

    ids["sigmas"] = ed.add_top_level_node(
        node_type="ManualSigmas",
        pos=[-200, 700], size=[270, 100],
        inputs=[],
        outputs=[_out("SIGMAS", "SIGMAS")],
        widgets_values=[REFINE_SIGMAS],
    )
    ids["sampler_select"] = ed.add_top_level_node(
        node_type="KSamplerSelect",
        pos=[-200, 820], size=[270, 60],
        inputs=[],
        outputs=[_out("SAMPLER", "SAMPLER")],
        widgets_values=["euler"],
    )
    ids["cfg_guider"] = ed.add_top_level_node(
        node_type="CFGGuider",
        pos=[-200, 900], size=[270, 110],
        inputs=[
            _basic_io_in("model", "MODEL"),
            _basic_io_in("positive", "CONDITIONING"),
            _basic_io_in("negative", "CONDITIONING"),
        ],
        outputs=[_out("GUIDER", "GUIDER")],
        widgets_values=[1.0],
    )
    ids["random_noise"] = ed.add_top_level_node(
        node_type="RandomNoise",
        pos=[-200, 1030], size=[270, 80],
        inputs=[],
        outputs=[_out("NOISE", "NOISE")],
        widgets_values=[42, "fixed"],
    )
    ids["sampler"] = ed.add_top_level_node(
        node_type="SamplerCustomAdvanced",
        pos=[100, 820], size=[280, 130],
        inputs=[
            _basic_io_in("noise", "NOISE"),
            _basic_io_in("guider", "GUIDER"),
            _basic_io_in("sampler", "SAMPLER"),
            _basic_io_in("sigmas", "SIGMAS"),
            _basic_io_in("latent_image", "LATENT"),
        ],
        outputs=[
            _out("output", "LATENT"),
            _out("denoised_output", "LATENT"),
        ],
        widgets_values=[],
    )

    ids["sep_av"] = ed.add_top_level_node(
        node_type="LTXVSeparateAVLatent",
        pos=[450, 820], size=[260, 80],
        inputs=[_basic_io_in("av_latent", "LATENT")],
        outputs=[
            _out("video_latent", "LATENT"),
            _out("audio_latent", "LATENT"),
        ],
        widgets_values=[],
    )
    ids["crop_guides"] = ed.add_top_level_node(
        node_type="LTXVCropGuides",
        pos=[750, 820], size=[260, 110],
        inputs=[
            _basic_io_in("positive", "CONDITIONING"),
            _basic_io_in("negative", "CONDITIONING"),
            _basic_io_in("latent", "LATENT"),
        ],
        outputs=[
            _out("positive", "CONDITIONING"),
            _out("negative", "CONDITIONING"),
            _out("latent", "LATENT"),
        ],
        widgets_values=[],
    )
    ids["decode"] = ed.add_top_level_node(
        node_type="LTXVTiledVAEDecode",
        pos=[1050, 820], size=[280, 180],
        inputs=[
            _basic_io_in("samples", "LATENT"),
            _basic_io_in("vae", "VAE"),
        ],
        outputs=[_out("IMAGE", "IMAGE")],
        widgets_values=[1, 1, 1, True, "auto", "auto"],
    )
    ids["combine"] = ed.add_top_level_node(
        node_type="VHS_VideoCombine",
        pos=[1400, 820], size=[480, 320],
        inputs=[
            _basic_io_in("images", "IMAGE"),
            _basic_io_in("audio", "AUDIO"),
            _basic_io_in("meta_batch", "VHS_BatchManager"),
            _basic_io_in("vae", "VAE"),
        ],
        outputs=[_out("Filenames", "VHS_FILENAMES")],
        widgets_values={
            "frame_rate": FRAME_RATE,
            "loop_count": 0,
            "filename_prefix": "LTX-seam-refined",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": True,
            "pingpong": False,
            "save_output": True,
        },
    )

    # Wire it up.
    ed.add_link(ids["clip"], 0, ids["pos_text"], 0, "CLIP")
    ed.add_link(ids["clip"], 0, ids["neg_text"], 0, "CLIP")
    ed.add_link(ids["neg_text"], 0, ids["zero_neg"], 0, "CONDITIONING")
    ed.add_link(ids["pos_text"], 0, ids["ltx_cond"], 0, "CONDITIONING")
    ed.add_link(ids["zero_neg"], 0, ids["ltx_cond"], 1, "CONDITIONING")

    ed.add_link(ids["load_video"], 0, ids["vae_encode"], 0, "IMAGE")
    ed.add_link(ids["video_vae"], 0, ids["vae_encode"], 1, "VAE")
    ed.add_link(ids["vae_encode"], 0, ids["seam_mask"], 0, "LATENT")

    ed.add_link(ids["audio_vae"], 0, ids["empty_audio"], 0, "VAE")
    # Track loaded video's frame count so AV-concat shapes always match.
    ed.add_link(ids["load_video"], 1, ids["empty_audio"], 1, "INT")

    ed.add_link(ids["seam_mask"], 0, ids["av_concat"], 0, "LATENT")
    ed.add_link(ids["empty_audio"], 0, ids["av_concat"], 1, "LATENT")

    ed.add_link(ids["unet"], 0, ids["cfg_guider"], 0, "MODEL")
    ed.add_link(ids["ltx_cond"], 0, ids["cfg_guider"], 1, "CONDITIONING")
    ed.add_link(ids["ltx_cond"], 1, ids["cfg_guider"], 2, "CONDITIONING")
    ed.add_link(ids["random_noise"], 0, ids["sampler"], 0, "NOISE")
    ed.add_link(ids["cfg_guider"], 0, ids["sampler"], 1, "GUIDER")
    ed.add_link(ids["sampler_select"], 0, ids["sampler"], 2, "SAMPLER")
    ed.add_link(ids["sigmas"], 0, ids["sampler"], 3, "SIGMAS")
    ed.add_link(ids["av_concat"], 0, ids["sampler"], 4, "LATENT")

    ed.add_link(ids["sampler"], 0, ids["sep_av"], 0, "LATENT")
    ed.add_link(ids["ltx_cond"], 0, ids["crop_guides"], 0, "CONDITIONING")
    ed.add_link(ids["ltx_cond"], 1, ids["crop_guides"], 1, "CONDITIONING")
    ed.add_link(ids["sep_av"], 0, ids["crop_guides"], 2, "LATENT")
    ed.add_link(ids["crop_guides"], 2, ids["decode"], 0, "LATENT")
    ed.add_link(ids["video_vae"], 0, ids["decode"], 1, "VAE")

    ed.add_link(ids["decode"], 0, ids["combine"], 0, "IMAGE")
    ed.add_link(ids["load_video"], 2, ids["combine"], 1, "AUDIO")

    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="Print node table without writing.")
    parser.add_argument("--revert", action="store_true", help="Delete the output file if present.")
    args = parser.parse_args()

    if args.revert:
        if OUTPUT.exists():
            OUTPUT.unlink()
            print(f"Deleted {OUTPUT.relative_to(REPO_ROOT)}")
        else:
            print(f"Already absent: {OUTPUT.relative_to(REPO_ROOT)}")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ed = WorkflowEditor.from_scratch(OUTPUT)
    ids = build(ed)

    if args.dry_run:
        print(f"DRY RUN — would write {OUTPUT.relative_to(REPO_ROOT)}")
        print(f"Nodes: {len(ed.wf['nodes'])}, Links: {len(ed.wf['links'])}")
        for role, nid in ids.items():
            n = ed.find_node(nid)
            print(f"  #{nid:3} [{role:14}] {n['type']}")
        return 0

    ed.save()
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  {len(ed.wf['nodes'])} nodes, {len(ed.wf['links'])} links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
