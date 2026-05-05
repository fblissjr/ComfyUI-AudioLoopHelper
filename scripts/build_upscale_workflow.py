"""Build the post-loop spatial-upscale workflow from scratch.

Produces `internal/workflows/upscale_loop_output.draft.json` per the
topology described in `internal/design/upscale_workflow_design.md`:

    VHS_LoadVideo → VAEEncode → LTXVLatentUpsampler (2x)
                                       ↓
                LTXVImgToVideoConditionOnly (re-condition with first frame)
                                       ↓
                LTXVConcatAVLatent (re-attach empty audio latent)
                                       ↓
                SamplerCustomAdvanced (3-step low-σ tail, euler, cfg=1)
                                       ↓
                LTXVSeparateAVLatent → LTXVCropGuides
                                       ↓
                VAEDecodeTiled → VHS_VideoCombine (with original audio)

Sigmas: `0.85, 0.7250, 0.4219, 0.0` (3-step partial refine starting at
σ=0.85 — respect the upsample, polish detail without hallucinating).
Sampler: `euler` (canonical distilled), CFG=1.0.

Idempotent: re-running overwrites the output file with deterministic
content (constant node ids, deterministic link order). `--dry-run`
prints the node table without writing. `--revert` deletes the output
file if present.

Usage:
    uv run --group dev python scripts/build_upscale_workflow.py
    uv run --group dev python scripts/build_upscale_workflow.py --dry-run
    uv run --group dev python scripts/build_upscale_workflow.py --revert
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add scripts/ to path so we can import workflow_utils when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "internal" / "workflows" / "upscale_loop_output.draft.json"

# ---------------------------------------------------------------------------
# Model + checkpoint references — the user must have these on disk.
# Match the names used by the shipped `audio-loop-music-video_latent_*.json`
# workflows so dependencies overlap.
# ---------------------------------------------------------------------------
UNET_NAME = "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"
CLIP_NAMES = ("gemma_3_12B_it_fpmixed.safetensors", "ltx-2.3_text_projection_bf16.safetensors")
VIDEO_VAE_NAME = "vae/LTX23_video_vae_bf16.safetensors"
AUDIO_VAE_NAME = "vae/LTX23_audio_vae_bf16.safetensors"
SPATIAL_UPSCALER_NAME = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# ---------------------------------------------------------------------------
# Distilled refine sigma profile — 4 values = 3 integration steps.
# Cross-validated from independent LTX 2.3 workflow authors.
# ---------------------------------------------------------------------------
REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"

# Frame rate of the loop output (LTX 2.3 default in this repo)
FRAME_RATE = 25


# Slot-dict helpers live on WorkflowEditor as static methods. Locally aliased
# here for call-site readability (the original `_basic_io_in` etc. names made
# the build() function easier to scan).
_basic_io_in = WorkflowEditor.io_in
_widget_in = WorkflowEditor.widget_in
_out = WorkflowEditor.out


def build(ed: WorkflowEditor) -> dict[str, int]:
    """Add all nodes and links. Returns a map of role-name -> node id for
    callers that want to inspect or further mutate the result."""

    ids: dict[str, int] = {}

    # -----------------------------------------------------------------------
    # Loaders
    # -----------------------------------------------------------------------
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
    ids["upscale_model"] = ed.add_top_level_node(
        node_type="LatentUpscaleModelLoader",
        pos=[-1900, 800], size=[460, 60],
        inputs=[],
        outputs=[_out("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL")],
        widgets_values=[SPATIAL_UPSCALER_NAME],
    )

    # -----------------------------------------------------------------------
    # Source loop output
    # -----------------------------------------------------------------------
    ids["load_video"] = ed.add_top_level_node(
        node_type="VHS_LoadVideo",
        pos=[-1900, 920], size=[460, 280],
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
            "video": "loop_output.mp4",  # placeholder — user sets in UI
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

    # -----------------------------------------------------------------------
    # Conditioning — same prompt as the loop; refinement should not drift
    # -----------------------------------------------------------------------
    ids["pos_text"] = ed.add_top_level_node(
        node_type="CLIPTextEncode",
        pos=[-1400, 320], size=[400, 200],
        inputs=[_basic_io_in("clip", "CLIP")],
        outputs=[_out("CONDITIONING", "CONDITIONING")],
        widgets_values=[""],  # user fills with the loop prompt
        title="Positive prompt (match loop)",
    )
    ids["neg_text"] = ed.add_top_level_node(
        node_type="CLIPTextEncode",
        pos=[-1400, 540], size=[400, 100],
        inputs=[_basic_io_in("clip", "CLIP")],
        outputs=[_out("CONDITIONING", "CONDITIONING")],
        widgets_values=[""],  # empty negative for distilled CFG=1
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

    # -----------------------------------------------------------------------
    # Latent path: encode → upscale → re-condition → AV concat
    # -----------------------------------------------------------------------
    ids["vae_encode"] = ed.add_top_level_node(
        node_type="VAEEncode",
        pos=[-1400, 920], size=[210, 50],
        inputs=[
            _basic_io_in("pixels", "IMAGE"),
            _basic_io_in("vae", "VAE"),
        ],
        outputs=[_out("LATENT", "LATENT")],
        widgets_values=[],
    )
    ids["upsampler"] = ed.add_top_level_node(
        node_type="LTXVLatentUpsampler",
        pos=[-1100, 920], size=[260, 90],
        inputs=[
            _basic_io_in("samples", "LATENT"),
            _basic_io_in("upscale_model", "LATENT_UPSCALE_MODEL"),
            _basic_io_in("vae", "VAE"),
        ],
        outputs=[_out("LATENT", "LATENT")],
        widgets_values=[],
    )
    ids["i2v_cond"] = ed.add_top_level_node(
        node_type="LTXVImgToVideoConditionOnly",
        pos=[-800, 920], size=[300, 130],
        inputs=[
            _basic_io_in("vae", "VAE"),
            _basic_io_in("image", "IMAGE"),
            _basic_io_in("latent", "LATENT"),
            _widget_in("bypass", "BOOLEAN"),
        ],
        outputs=[_out("latent", "LATENT")],
        widgets_values=[1.0, False],  # full strength, not bypassed
    )
    ids["empty_audio"] = ed.add_top_level_node(
        node_type="LTXVEmptyLatentAudio",
        pos=[-800, 1100], size=[260, 110],
        inputs=[
            _basic_io_in("audio_vae", "VAE"),
            _widget_in("frames_number", "INT"),
            _widget_in("frame_rate", "INT"),
        ],
        outputs=[_out("Latent", "LATENT")],
        widgets_values=[497, FRAME_RATE, 1],  # frames matches loop default; user adjusts to clip length
    )
    ids["av_concat"] = ed.add_top_level_node(
        node_type="LTXVConcatAVLatent",
        pos=[-450, 1000], size=[200, 50],
        inputs=[
            _basic_io_in("video_latent", "LATENT"),
            _basic_io_in("audio_latent", "LATENT"),
        ],
        outputs=[_out("latent", "LATENT")],
        widgets_values=[],
    )

    # -----------------------------------------------------------------------
    # Sampler stack
    # -----------------------------------------------------------------------
    ids["sigmas"] = ed.add_top_level_node(
        node_type="ManualSigmas",
        pos=[-200, 800], size=[270, 100],
        inputs=[],
        outputs=[_out("SIGMAS", "SIGMAS")],
        widgets_values=[REFINE_SIGMAS],
    )
    ids["sampler_select"] = ed.add_top_level_node(
        node_type="KSamplerSelect",
        pos=[-200, 920], size=[270, 60],
        inputs=[],
        outputs=[_out("SAMPLER", "SAMPLER")],
        widgets_values=["euler"],
    )
    ids["cfg_guider"] = ed.add_top_level_node(
        node_type="CFGGuider",
        pos=[-200, 1000], size=[270, 110],
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
        pos=[-200, 1130], size=[270, 80],
        inputs=[],
        outputs=[_out("NOISE", "NOISE")],
        widgets_values=[42, "fixed"],
    )
    ids["sampler"] = ed.add_top_level_node(
        node_type="SamplerCustomAdvanced",
        pos=[100, 920], size=[280, 130],
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

    # -----------------------------------------------------------------------
    # Post-sample: separate, crop guides, decode
    # -----------------------------------------------------------------------
    ids["sep_av"] = ed.add_top_level_node(
        node_type="LTXVSeparateAVLatent",
        pos=[450, 920], size=[260, 80],
        inputs=[_basic_io_in("av_latent", "LATENT")],
        outputs=[
            _out("video_latent", "LATENT"),
            _out("audio_latent", "LATENT"),
        ],
        widgets_values=[],
    )
    ids["crop_guides"] = ed.add_top_level_node(
        node_type="LTXVCropGuides",
        pos=[750, 920], size=[260, 110],
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
        pos=[1050, 920], size=[280, 180],
        inputs=[
            _basic_io_in("samples", "LATENT"),
            _basic_io_in("vae", "VAE"),
        ],
        outputs=[_out("IMAGE", "IMAGE")],
        widgets_values=[1, 1, 1, True, "auto", "auto"],  # single-tile, 24GB+ default
    )
    ids["combine"] = ed.add_top_level_node(
        node_type="VHS_VideoCombine",
        pos=[1400, 920], size=[480, 320],
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
            "filename_prefix": "LTX-upscaled",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": True,
            "pingpong": False,
            "save_output": True,
        },
    )

    # -----------------------------------------------------------------------
    # Wire it up
    # -----------------------------------------------------------------------
    # Conditioning chain
    ed.add_link(ids["clip"], 0, ids["pos_text"], 0, "CLIP")
    ed.add_link(ids["clip"], 0, ids["neg_text"], 0, "CLIP")
    ed.add_link(ids["neg_text"], 0, ids["zero_neg"], 0, "CONDITIONING")
    ed.add_link(ids["pos_text"], 0, ids["ltx_cond"], 0, "CONDITIONING")
    ed.add_link(ids["zero_neg"], 0, ids["ltx_cond"], 1, "CONDITIONING")

    # Source video → encode → upscale
    ed.add_link(ids["load_video"], 0, ids["vae_encode"], 0, "IMAGE")
    ed.add_link(ids["video_vae"], 0, ids["vae_encode"], 1, "VAE")
    ed.add_link(ids["vae_encode"], 0, ids["upsampler"], 0, "LATENT")
    ed.add_link(ids["upscale_model"], 0, ids["upsampler"], 1, "LATENT_UPSCALE_MODEL")
    ed.add_link(ids["video_vae"], 0, ids["upsampler"], 2, "VAE")

    # I2V re-condition with original first-frame image stack
    ed.add_link(ids["video_vae"], 0, ids["i2v_cond"], 0, "VAE")
    ed.add_link(ids["load_video"], 0, ids["i2v_cond"], 1, "IMAGE")
    ed.add_link(ids["upsampler"], 0, ids["i2v_cond"], 2, "LATENT")

    # Empty audio latent at upscaled length
    ed.add_link(ids["audio_vae"], 0, ids["empty_audio"], 0, "VAE")

    # AV concat: re-conditioned video + empty audio
    ed.add_link(ids["i2v_cond"], 0, ids["av_concat"], 0, "LATENT")
    ed.add_link(ids["empty_audio"], 0, ids["av_concat"], 1, "LATENT")

    # Sampler stack
    ed.add_link(ids["unet"], 0, ids["cfg_guider"], 0, "MODEL")
    ed.add_link(ids["ltx_cond"], 0, ids["cfg_guider"], 1, "CONDITIONING")
    ed.add_link(ids["ltx_cond"], 1, ids["cfg_guider"], 2, "CONDITIONING")
    ed.add_link(ids["random_noise"], 0, ids["sampler"], 0, "NOISE")
    ed.add_link(ids["cfg_guider"], 0, ids["sampler"], 1, "GUIDER")
    ed.add_link(ids["sampler_select"], 0, ids["sampler"], 2, "SAMPLER")
    ed.add_link(ids["sigmas"], 0, ids["sampler"], 3, "SIGMAS")
    ed.add_link(ids["av_concat"], 0, ids["sampler"], 4, "LATENT")

    # Post-sample: separate AV, crop guides, decode
    ed.add_link(ids["sampler"], 0, ids["sep_av"], 0, "LATENT")
    ed.add_link(ids["ltx_cond"], 0, ids["crop_guides"], 0, "CONDITIONING")
    ed.add_link(ids["ltx_cond"], 1, ids["crop_guides"], 1, "CONDITIONING")
    ed.add_link(ids["sep_av"], 0, ids["crop_guides"], 2, "LATENT")
    ed.add_link(ids["crop_guides"], 2, ids["decode"], 0, "LATENT")
    ed.add_link(ids["video_vae"], 0, ids["decode"], 1, "VAE")

    # Combine: decoded images + original audio passthrough
    ed.add_link(ids["decode"], 0, ids["combine"], 0, "IMAGE")
    ed.add_link(ids["load_video"], 2, ids["combine"], 1, "AUDIO")

    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
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
