"""Build the minimal IC-LoRA spectrogram-reference test workflow from scratch.

Outputs `example_workflows/experimental/spectrogram_iclora_minimal.json`.

Uses our production loader stack (UNETLoader + DualCLIPLoader +
VAELoaderKJ × 2 + AudioLoopHelperSageAttention) — no API nodes. Produces
a single-pass T2V rendering with IC-LoRA reference = spectrogram mp4,
no audio loop, no init image. This is the Phase 2.0 minimal test rig
per `docs/experimental/spectrogram_iclora_tutorial.md`.

Usage:
    uv run --group dev python scripts/apply_spectrogram_iclora_minimal.py
    uv run --group dev python scripts/apply_spectrogram_iclora_minimal.py --revert

Idempotent on the output path; `--revert` deletes the output file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflow_utils import WorkflowEditor


DEFAULT_OUTPUT = Path("example_workflows/experimental/spectrogram_iclora_minimal.json")

UNET_FILE = "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"
CLIP_GEMMA = "gemma_3_12B_it_fpmixed.safetensors"
CLIP_PROJECTION = "ltx-2.3_text_projection_bf16.safetensors"
VAE_VIDEO = "vae/LTX23_video_vae_bf16.safetensors"
VAE_AUDIO = "vae/LTX23_audio_vae_bf16.safetensors"
ICLORA_FILE = "MergeGreen_IC-lora_ltx2.3.safetensors"
ICLORA_STRENGTH = 0.9

VIDEO_WIDTH = 832
VIDEO_HEIGHT = 448
VIDEO_LENGTH = 121  # (121-1) % 8 == 0; ~5s at 24fps / 4.84s at 25fps
VIDEO_FPS = 25

PLACEHOLDER_VIDEO = "REPLACE_WITH_SPECTROGRAM.mp4"
DEFAULT_POSITIVE = (
    "A drummer performing energetically on a dimly lit stage, warm stage lighting, "
    "shallow depth of field, cinematic. The performer's motion pulses with the music, "
    "confident and rhythmic."
)
DEFAULT_NEGATIVE = "still image, frozen, deformed, duplicate, blurry, low quality"


def _in(name: str, dtype: str) -> dict:
    return {"name": name, "type": dtype, "link": None}


def _out(name: str, dtype: str) -> dict:
    return {"name": name, "type": dtype, "links": []}


def build_workflow(output_path: Path) -> WorkflowEditor:
    ed = WorkflowEditor.from_scratch(output_path)

    unet = ed.add_top_level_node(
        "UNETLoader", pos=[-2400, -400], size=[480, 82],
        inputs=[], outputs=[_out("MODEL", "MODEL")],
        widgets_values=[UNET_FILE, "default"],
        title="UNET (merged distilled 22B)",
    )
    clip = ed.add_top_level_node(
        "DualCLIPLoader", pos=[-2400, -260], size=[480, 130],
        inputs=[], outputs=[_out("CLIP", "CLIP")],
        widgets_values=[CLIP_GEMMA, CLIP_PROJECTION, "ltxv", "default"],
        title="Gemma text encoder",
    )
    vae_v = ed.add_top_level_node(
        "VAELoaderKJ", pos=[-2400, -80], size=[480, 106],
        inputs=[], outputs=[_out("VAE", "VAE")],
        widgets_values=[VAE_VIDEO, "main_device", "bf16"],
        title="Video VAE",
    )
    vae_a = ed.add_top_level_node(
        "VAELoaderKJ", pos=[-2400, 80], size=[480, 106],
        inputs=[], outputs=[_out("VAE", "VAE")],
        widgets_values=[VAE_AUDIO, "main_device", "bf16"],
        title="Audio VAE (for dummy silent latent)",
    )

    sage = ed.add_top_level_node(
        "AudioLoopHelperSageAttention", pos=[-1850, -400], size=[380, 82],
        inputs=[_in("model", "MODEL")], outputs=[_out("MODEL", "MODEL")],
        widgets_values=["auto_mask_aware", True],
        title="Sage attention (mask-aware)",
    )
    mss = ed.add_top_level_node(
        "ModelSamplingSD3", pos=[-1850, -280], size=[380, 58],
        inputs=[_in("model", "MODEL")], outputs=[_out("MODEL", "MODEL")],
        widgets_values=[13],
        title="shift=13 (distilled)",
    )
    iclora_loader = ed.add_top_level_node(
        "LTXICLoRALoaderModelOnly", pos=[-1850, -160], size=[380, 110],
        inputs=[_in("model", "MODEL")],
        outputs=[_out("model", "MODEL"), _out("latent_downscale_factor", "FLOAT")],
        widgets_values=[ICLORA_FILE, ICLORA_STRENGTH],
        title="IC-LoRA loader",
    )
    ed.add_link(unet, 0, sage, 0, "MODEL")
    ed.add_link(sage, 0, mss, 0, "MODEL")
    ed.add_link(mss, 0, iclora_loader, 0, "MODEL")

    pos_enc = ed.add_top_level_node(
        "CLIPTextEncode", pos=[-1400, -520], size=[420, 150],
        inputs=[_in("clip", "CLIP")], outputs=[_out("CONDITIONING", "CONDITIONING")],
        widgets_values=[DEFAULT_POSITIVE],
        title="Positive prompt",
    )
    neg_enc = ed.add_top_level_node(
        "CLIPTextEncode", pos=[-1400, -340], size=[420, 110],
        inputs=[_in("clip", "CLIP")], outputs=[_out("CONDITIONING", "CONDITIONING")],
        widgets_values=[DEFAULT_NEGATIVE],
        title="Negative prompt",
    )
    ltxv_cond = ed.add_top_level_node(
        "LTXVConditioning", pos=[-1400, -190], size=[380, 78],
        inputs=[_in("positive", "CONDITIONING"), _in("negative", "CONDITIONING")],
        outputs=[_out("positive", "CONDITIONING"), _out("negative", "CONDITIONING")],
        widgets_values=[VIDEO_FPS],
        title="LTXV conditioning (frame_rate stamp)",
    )
    ed.add_link(clip, 0, pos_enc, 0, "CLIP")
    ed.add_link(clip, 0, neg_enc, 0, "CLIP")
    ed.add_link(pos_enc, 0, ltxv_cond, 0, "CONDITIONING")
    ed.add_link(neg_enc, 0, ltxv_cond, 1, "CONDITIONING")

    empty_vid = ed.add_top_level_node(
        "EmptyLTXVLatentVideo", pos=[-1400, 40], size=[380, 106],
        inputs=[], outputs=[_out("LATENT", "LATENT")],
        widgets_values=[VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_LENGTH, 1],
        title="Empty video latent",
    )
    empty_aud = ed.add_top_level_node(
        "LTXVEmptyLatentAudio", pos=[-1400, 170], size=[380, 106],
        inputs=[_in("audio_vae", "VAE")], outputs=[_out("Latent", "LATENT")],
        widgets_values=[VIDEO_LENGTH, VIDEO_FPS, 1],
        title="Dummy silent audio latent",
    )
    ed.add_link(vae_a, 0, empty_aud, 0, "VAE")

    load_vid = ed.add_top_level_node(
        "LoadVideo", pos=[-1400, 320], size=[380, 300],
        inputs=[], outputs=[_out("VIDEO", "VIDEO")],
        widgets_values=[PLACEHOLDER_VIDEO, "image"],
        title="Spectrogram mp4 (REPLACE widget)",
    )
    get_comp = ed.add_top_level_node(
        "GetVideoComponents", pos=[-980, 320], size=[340, 106],
        inputs=[_in("video", "VIDEO")],
        outputs=[_out("image", "IMAGE"), _out("audio", "AUDIO"), _out("fps", "FLOAT")],
        widgets_values=[],
        title="Video → frames",
    )
    ed.add_link(load_vid, 0, get_comp, 0, "VIDEO")

    iclora_guide = ed.add_top_level_node(
        "LTXAddVideoICLoRAGuide", pos=[-900, -400], size=[380, 280],
        inputs=[
            _in("positive", "CONDITIONING"),
            _in("negative", "CONDITIONING"),
            _in("vae", "VAE"),
            _in("latent", "LATENT"),
            _in("image", "IMAGE"),
            _in("latent_downscale_factor", "FLOAT"),
        ],
        outputs=[
            _out("positive", "CONDITIONING"),
            _out("negative", "CONDITIONING"),
            _out("latent", "LATENT"),
        ],
        widgets_values=[0, 1.0, 1.0, "disabled", False, 256, 64],
        title="IC-LoRA guide (spectrogram reference)",
    )
    ed.add_link(ltxv_cond, 0, iclora_guide, 0, "CONDITIONING")
    ed.add_link(ltxv_cond, 1, iclora_guide, 1, "CONDITIONING")
    ed.add_link(vae_v, 0, iclora_guide, 2, "VAE")
    ed.add_link(empty_vid, 0, iclora_guide, 3, "LATENT")
    ed.add_link(get_comp, 0, iclora_guide, 4, "IMAGE")
    ed.add_link(iclora_loader, 1, iclora_guide, 5, "FLOAT")

    av_concat = ed.add_top_level_node(
        "LTXVConcatAVLatent", pos=[-480, 80], size=[340, 78],
        inputs=[_in("video_latent", "LATENT"), _in("audio_latent", "LATENT")],
        outputs=[_out("latent", "LATENT")],
        widgets_values=[],
        title="AV concat",
    )
    ed.add_link(iclora_guide, 2, av_concat, 0, "LATENT")
    ed.add_link(empty_aud, 0, av_concat, 1, "LATENT")

    basic_sched = ed.add_top_level_node(
        "BasicScheduler", pos=[-480, -400], size=[340, 106],
        inputs=[_in("model", "MODEL")], outputs=[_out("SIGMAS", "SIGMAS")],
        widgets_values=["linear_quadratic", 8, 1.0],
        title="Distilled 8-step sigmas",
    )
    ed.add_link(iclora_loader, 0, basic_sched, 0, "MODEL")

    ksel = ed.add_top_level_node(
        "KSamplerSelect", pos=[-480, -260], size=[340, 58],
        inputs=[], outputs=[_out("SAMPLER", "SAMPLER")],
        widgets_values=["euler"],
        title="Euler (NOT ancestral)",
    )
    cfg_guider = ed.add_top_level_node(
        "CFGGuider", pos=[-480, -180], size=[340, 98],
        inputs=[
            _in("model", "MODEL"),
            _in("positive", "CONDITIONING"),
            _in("negative", "CONDITIONING"),
        ],
        outputs=[_out("GUIDER", "GUIDER")],
        widgets_values=[1.0],
        title="CFG=1 (distilled)",
    )
    ed.add_link(iclora_loader, 0, cfg_guider, 0, "MODEL")
    ed.add_link(iclora_guide, 0, cfg_guider, 1, "CONDITIONING")
    ed.add_link(iclora_guide, 1, cfg_guider, 2, "CONDITIONING")

    noise = ed.add_top_level_node(
        "RandomNoise", pos=[-480, -60], size=[340, 82],
        inputs=[], outputs=[_out("NOISE", "NOISE")],
        widgets_values=[42, "fixed"],
        title="Seed=42",
    )

    sampler = ed.add_top_level_node(
        "SamplerCustomAdvanced", pos=[-100, -200], size=[320, 106],
        inputs=[
            _in("noise", "NOISE"),
            _in("guider", "GUIDER"),
            _in("sampler", "SAMPLER"),
            _in("sigmas", "SIGMAS"),
            _in("latent_image", "LATENT"),
        ],
        outputs=[_out("output", "LATENT"), _out("denoised_output", "LATENT")],
        widgets_values=[],
        title="Sampler",
    )
    ed.add_link(noise, 0, sampler, 0, "NOISE")
    ed.add_link(cfg_guider, 0, sampler, 1, "GUIDER")
    ed.add_link(ksel, 0, sampler, 2, "SAMPLER")
    ed.add_link(basic_sched, 0, sampler, 3, "SIGMAS")
    ed.add_link(av_concat, 0, sampler, 4, "LATENT")

    crop = ed.add_top_level_node(
        "LTXVCropGuides", pos=[260, -200], size=[320, 78],
        inputs=[
            _in("positive", "CONDITIONING"),
            _in("negative", "CONDITIONING"),
            _in("latent", "LATENT"),
        ],
        outputs=[
            _out("positive", "CONDITIONING"),
            _out("negative", "CONDITIONING"),
            _out("latent", "LATENT"),
        ],
        widgets_values=[],
        title="Strip IC-LoRA guide frames",
    )
    ed.add_link(iclora_guide, 0, crop, 0, "CONDITIONING")
    ed.add_link(iclora_guide, 1, crop, 1, "CONDITIONING")
    ed.add_link(sampler, 0, crop, 2, "LATENT")

    separate = ed.add_top_level_node(
        "LTXVSeparateAVLatent", pos=[620, -200], size=[320, 78],
        inputs=[_in("av_latent", "LATENT")],
        outputs=[_out("video_latent", "LATENT"), _out("audio_latent", "LATENT")],
        widgets_values=[],
        title="Separate video from audio",
    )
    ed.add_link(crop, 2, separate, 0, "LATENT")

    decode = ed.add_top_level_node(
        "LTXVTiledVAEDecode", pos=[620, -60], size=[320, 150],
        inputs=[_in("vae", "VAE"), _in("latents", "LATENT")],
        outputs=[_out("IMAGE", "IMAGE")],
        widgets_values=[2, 2, 1, True, "auto", "auto"],
        title="Tiled VAE decode",
    )
    ed.add_link(vae_v, 0, decode, 0, "VAE")
    ed.add_link(separate, 0, decode, 1, "LATENT")

    # VHS_VideoCombine widgets are a dict (not a list) -- matches the
    # production schema. Other nodes use list-shaped widgets_values.
    combine = ed.add_top_level_node(
        "VHS_VideoCombine", pos=[980, -200], size=[340, 560],
        inputs=[
            _in("images", "IMAGE"),
            _in("audio", "AUDIO"),
            _in("meta_batch", "VHS_BatchManager"),
            _in("vae", "VAE"),
        ],
        outputs=[_out("Filenames", "VHS_FILENAMES")],
        widgets_values={
            "frame_rate": VIDEO_FPS,
            "loop_count": 0,
            "filename_prefix": "spectrogram_iclora_test",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        },
        title="Save video (silent; dub audio via ffmpeg)",
    )
    ed.add_link(decode, 0, combine, 0, "IMAGE")

    return ed


def _build(output_path: Path) -> None:
    if output_path.exists():
        print(f"{output_path}: already exists; use --revert first or delete manually.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ed = build_workflow(output_path)
    ed.save()
    node_count = len(ed.wf["nodes"])
    link_count = len(ed.wf["links"])
    print(f"wrote {output_path}")
    print(f"  {node_count} nodes, {link_count} links")
    print()
    print("Files this workflow references (download any that aren't in your ComfyUI models tree):")
    print(f"  models/diffusion_models/{UNET_FILE}")
    print(f"  models/text_encoders/{CLIP_GEMMA}")
    print(f"  models/text_encoders/{CLIP_PROJECTION}")
    print(f"  models/{VAE_VIDEO}")
    print(f"  models/{VAE_AUDIO}")
    print(f"  models/loras/{ICLORA_FILE}")
    print()
    print("Next: follow docs/experimental/spectrogram_iclora_tutorial.md")


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="Output path (default: %(default)s)")
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output workflow file.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return
    _build(output_path)


if __name__ == "__main__":
    main()
