"""Build the minimal IC-LoRA spectrogram-reference test workflow from scratch.

Outputs `example_workflows/experimental/spectrogram_iclora_minimal.json`.

Uses our production loader stack (UNETLoader + DualCLIPLoader +
VAELoaderKJ × 2 + AudioLoopHelperSageAttention) — no API nodes, no
files the user doesn't already have. Produces a single-pass T2V
rendering with IC-LoRA reference = spectrogram mp4, no audio
loop, no init image. This is the Phase 2.0 minimal test rig per
`docs/experimental/spectrogram_iclora_tutorial.md`.

Usage:
    uv run --group dev python scripts/apply_spectrogram_iclora_minimal.py
    uv run --group dev python scripts/apply_spectrogram_iclora_minimal.py --revert

Idempotent on the output path; `--revert` deletes the output file.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))


DEFAULT_OUTPUT = Path("example_workflows/experimental/spectrogram_iclora_minimal.json")

# Files the user must have. Verified present in our production workflow.
UNET_FILE = "ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"
CLIP_GEMMA = "gemma_3_12B_it_fpmixed.safetensors"
CLIP_PROJECTION = "ltx-2.3_text_projection_bf16.safetensors"
VAE_VIDEO = "vae/LTX23_video_vae_bf16.safetensors"
VAE_AUDIO = "vae/LTX23_audio_vae_bf16.safetensors"
ICLORA_FILE = "MergeGreen_IC-lora_ltx2.3.safetensors"
ICLORA_STRENGTH = 0.9

# Target render shape
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


class Builder:
    def __init__(self):
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self.next_id = 1
        self.next_link = 1

    def add_node(
        self,
        node_type: str,
        pos: tuple[float, float],
        *,
        size: tuple[float, float] = (300.0, 100.0),
        widgets: list | dict | None = None,
        inputs: list[tuple[str, str]] | None = None,
        outputs: list[tuple[str, str]] | None = None,
        properties: dict | None = None,
        title: str | None = None,
    ) -> int:
        nid = self.next_id
        self.next_id += 1
        node = {
            "id": nid,
            "type": node_type,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in (inputs or [])],
            "outputs": [{"name": n, "type": t, "links": []} for n, t in (outputs or [])],
            "properties": properties or {"Node name for S&R": node_type},
            "widgets_values": widgets if widgets is not None else [],
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return nid

    def connect(self, src_id: int, src_slot: int, tgt_id: int, tgt_slot: int, dtype: str) -> int:
        lid = self.next_link
        self.next_link += 1
        self.links.append([lid, src_id, src_slot, tgt_id, tgt_slot, dtype])
        src = next(n for n in self.nodes if n["id"] == src_id)
        tgt = next(n for n in self.nodes if n["id"] == tgt_id)
        src["outputs"][src_slot]["links"].append(lid)
        tgt["inputs"][tgt_slot]["link"] = lid
        return lid

    def serialize(self) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "revision": 0,
            "last_node_id": self.next_id - 1,
            "last_link_id": self.next_link - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "definitions": {"subgraphs": []},
            "config": {},
            "extra": {"ds": {"scale": 0.5, "offset": [0, 0]}},
            "version": 0.4,
        }


def build_workflow() -> dict:
    b = Builder()

    # --- Column 1: Loaders (x = -2400) ------------------------------------
    unet = b.add_node(
        "UNETLoader", pos=(-2400, -400),
        size=(480, 82),
        widgets=[UNET_FILE, "default"],
        outputs=[("MODEL", "MODEL")],
        title="UNET (merged distilled 22B)",
    )
    clip = b.add_node(
        "DualCLIPLoader", pos=(-2400, -260),
        size=(480, 130),
        widgets=[CLIP_GEMMA, CLIP_PROJECTION, "ltxv", "default"],
        outputs=[("CLIP", "CLIP")],
        title="Gemma text encoder",
    )
    vae_v = b.add_node(
        "VAELoaderKJ", pos=(-2400, -80),
        size=(480, 106),
        widgets=[VAE_VIDEO, "main_device", "bf16"],
        outputs=[("VAE", "VAE")],
        title="Video VAE",
    )
    vae_a = b.add_node(
        "VAELoaderKJ", pos=(-2400, 80),
        size=(480, 106),
        widgets=[VAE_AUDIO, "main_device", "bf16"],
        outputs=[("VAE", "VAE")],
        title="Audio VAE (for dummy silent latent)",
    )

    # --- Column 2: MODEL patch chain (x = -1850) --------------------------
    sage = b.add_node(
        "AudioLoopHelperSageAttention", pos=(-1850, -400),
        size=(380, 82),
        widgets=["auto_mask_aware", True],
        inputs=[("model", "MODEL")],
        outputs=[("MODEL", "MODEL")],
        title="Sage attention (mask-aware)",
    )
    mss = b.add_node(
        "ModelSamplingSD3", pos=(-1850, -280),
        size=(380, 58),
        widgets=[13],
        inputs=[("model", "MODEL")],
        outputs=[("MODEL", "MODEL")],
        title="shift=13 (distilled)",
    )
    iclora_loader = b.add_node(
        "LTXICLoRALoaderModelOnly", pos=(-1850, -160),
        size=(380, 110),
        widgets=[ICLORA_FILE, ICLORA_STRENGTH],
        inputs=[("model", "MODEL")],
        outputs=[("model", "MODEL"), ("latent_downscale_factor", "FLOAT")],
        title="IC-LoRA loader",
    )

    b.connect(unet, 0, sage, 0, "MODEL")
    b.connect(sage, 0, mss, 0, "MODEL")
    b.connect(mss, 0, iclora_loader, 0, "MODEL")

    # --- Column 3: Text encoding (x = -1400) ------------------------------
    pos_enc = b.add_node(
        "CLIPTextEncode", pos=(-1400, -520),
        size=(420, 150),
        widgets=[DEFAULT_POSITIVE],
        inputs=[("clip", "CLIP")],
        outputs=[("CONDITIONING", "CONDITIONING")],
        title="Positive prompt",
    )
    neg_enc = b.add_node(
        "CLIPTextEncode", pos=(-1400, -340),
        size=(420, 110),
        widgets=[DEFAULT_NEGATIVE],
        inputs=[("clip", "CLIP")],
        outputs=[("CONDITIONING", "CONDITIONING")],
        title="Negative prompt",
    )
    ltxv_cond = b.add_node(
        "LTXVConditioning", pos=(-1400, -190),
        size=(380, 78),
        widgets=[VIDEO_FPS],
        inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
        outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
        title="LTXV conditioning (frame_rate stamp)",
    )

    b.connect(clip, 0, pos_enc, 0, "CLIP")
    b.connect(clip, 0, neg_enc, 0, "CLIP")
    b.connect(pos_enc, 0, ltxv_cond, 0, "CONDITIONING")
    b.connect(neg_enc, 0, ltxv_cond, 1, "CONDITIONING")

    # --- Column 3/4: Latents + video input (x = -1400, lower) -------------
    empty_vid = b.add_node(
        "EmptyLTXVLatentVideo", pos=(-1400, 40),
        size=(380, 106),
        widgets=[VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_LENGTH, 1],
        outputs=[("LATENT", "LATENT")],
        title="Empty video latent",
    )
    empty_aud = b.add_node(
        "LTXVEmptyLatentAudio", pos=(-1400, 170),
        size=(380, 106),
        widgets=[VIDEO_LENGTH, VIDEO_FPS, 1],
        inputs=[("audio_vae", "VAE")],
        outputs=[("Latent", "LATENT")],
        title="Dummy silent audio latent",
    )
    b.connect(vae_a, 0, empty_aud, 0, "VAE")

    load_vid = b.add_node(
        "LoadVideo", pos=(-1400, 320),
        size=(380, 300),
        widgets=[PLACEHOLDER_VIDEO, "image"],
        outputs=[("VIDEO", "VIDEO")],
        title="Spectrogram mp4 (REPLACE widget)",
    )
    get_comp = b.add_node(
        "GetVideoComponents", pos=(-980, 320),
        size=(340, 106),
        inputs=[("video", "VIDEO")],
        outputs=[("image", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT")],
        title="Video → frames",
    )
    b.connect(load_vid, 0, get_comp, 0, "VIDEO")

    # --- Column 4: IC-LoRA guide (x = -900) -------------------------------
    iclora_guide = b.add_node(
        "LTXAddVideoICLoRAGuide", pos=(-900, -400),
        size=(380, 280),
        widgets=[0, 1.0, 1.0, "disabled", False, 256, 64],
        inputs=[
            ("positive", "CONDITIONING"),
            ("negative", "CONDITIONING"),
            ("vae", "VAE"),
            ("latent", "LATENT"),
            ("image", "IMAGE"),
            ("latent_downscale_factor", "FLOAT"),
        ],
        outputs=[
            ("positive", "CONDITIONING"),
            ("negative", "CONDITIONING"),
            ("latent", "LATENT"),
        ],
        title="IC-LoRA guide (spectrogram reference)",
    )
    b.connect(ltxv_cond, 0, iclora_guide, 0, "CONDITIONING")
    b.connect(ltxv_cond, 1, iclora_guide, 1, "CONDITIONING")
    b.connect(vae_v, 0, iclora_guide, 2, "VAE")
    b.connect(empty_vid, 0, iclora_guide, 3, "LATENT")
    b.connect(get_comp, 0, iclora_guide, 4, "IMAGE")
    b.connect(iclora_loader, 1, iclora_guide, 5, "FLOAT")

    # --- Column 5: AV concat + sampler (x = -480) -------------------------
    av_concat = b.add_node(
        "LTXVConcatAVLatent", pos=(-480, 80),
        size=(340, 78),
        inputs=[("video_latent", "LATENT"), ("audio_latent", "LATENT")],
        outputs=[("latent", "LATENT")],
        title="AV concat",
    )
    b.connect(iclora_guide, 2, av_concat, 0, "LATENT")
    b.connect(empty_aud, 0, av_concat, 1, "LATENT")

    basic_sched = b.add_node(
        "BasicScheduler", pos=(-480, -400),
        size=(340, 106),
        widgets=["linear_quadratic", 8, 1.0],
        inputs=[("model", "MODEL")],
        outputs=[("SIGMAS", "SIGMAS")],
        title="Distilled 8-step sigmas",
    )
    b.connect(iclora_loader, 0, basic_sched, 0, "MODEL")

    ksel = b.add_node(
        "KSamplerSelect", pos=(-480, -260),
        size=(340, 58),
        widgets=["euler"],
        outputs=[("SAMPLER", "SAMPLER")],
        title="Euler (NOT ancestral)",
    )
    cfg_guider = b.add_node(
        "CFGGuider", pos=(-480, -180),
        size=(340, 98),
        widgets=[1.0],
        inputs=[("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
        outputs=[("GUIDER", "GUIDER")],
        title="CFG=1 (distilled)",
    )
    b.connect(iclora_loader, 0, cfg_guider, 0, "MODEL")
    b.connect(iclora_guide, 0, cfg_guider, 1, "CONDITIONING")
    b.connect(iclora_guide, 1, cfg_guider, 2, "CONDITIONING")

    noise = b.add_node(
        "RandomNoise", pos=(-480, -60),
        size=(340, 82),
        widgets=[42, "fixed"],
        outputs=[("NOISE", "NOISE")],
        title="Seed=42",
    )

    # --- Column 6: Sampler + crop (x = -100) ------------------------------
    sampler = b.add_node(
        "SamplerCustomAdvanced", pos=(-100, -200),
        size=(320, 106),
        inputs=[
            ("noise", "NOISE"),
            ("guider", "GUIDER"),
            ("sampler", "SAMPLER"),
            ("sigmas", "SIGMAS"),
            ("latent_image", "LATENT"),
        ],
        outputs=[("output", "LATENT"), ("denoised_output", "LATENT")],
        title="Sampler",
    )
    b.connect(noise, 0, sampler, 0, "NOISE")
    b.connect(cfg_guider, 0, sampler, 1, "GUIDER")
    b.connect(ksel, 0, sampler, 2, "SAMPLER")
    b.connect(basic_sched, 0, sampler, 3, "SIGMAS")
    b.connect(av_concat, 0, sampler, 4, "LATENT")

    crop = b.add_node(
        "LTXVCropGuides", pos=(260, -200),
        size=(320, 78),
        inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")],
        outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent", "LATENT")],
        title="Strip IC-LoRA guide frames",
    )
    b.connect(iclora_guide, 0, crop, 0, "CONDITIONING")
    b.connect(iclora_guide, 1, crop, 1, "CONDITIONING")
    b.connect(sampler, 0, crop, 2, "LATENT")

    # --- Column 7: Separate + decode + output (x = 620) -------------------
    separate = b.add_node(
        "LTXVSeparateAVLatent", pos=(620, -200),
        size=(320, 78),
        inputs=[("av_latent", "LATENT")],
        outputs=[("video_latent", "LATENT"), ("audio_latent", "LATENT")],
        title="Separate video from audio",
    )
    b.connect(crop, 2, separate, 0, "LATENT")

    decode = b.add_node(
        "LTXVTiledVAEDecode", pos=(620, -60),
        size=(320, 150),
        widgets=[2, 2, 1, True, "auto", "auto"],
        inputs=[("vae", "VAE"), ("latents", "LATENT")],
        outputs=[("IMAGE", "IMAGE")],
        title="Tiled VAE decode",
    )
    b.connect(vae_v, 0, decode, 0, "VAE")
    b.connect(separate, 0, decode, 1, "LATENT")

    combine = b.add_node(
        "VHS_VideoCombine", pos=(980, -200),
        size=(340, 560),
        widgets={
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
        inputs=[("images", "IMAGE"), ("audio", "AUDIO"), ("meta_batch", "VHS_BatchManager"), ("vae", "VAE")],
        outputs=[("Filenames", "VHS_FILENAMES")],
        title="Save video (silent; dub audio via ffmpeg)",
    )
    b.connect(decode, 0, combine, 0, "IMAGE")

    return b.serialize()


def _build(output_path: Path) -> None:
    if output_path.exists():
        print(f"{output_path}: already exists; use --revert first or delete manually.")
        return
    wf = build_workflow()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(wf, option=orjson.OPT_INDENT_2))
    node_count = len(wf["nodes"])
    link_count = len(wf["links"])
    print(f"wrote {output_path}")
    print(f"  {node_count} nodes, {link_count} links")
    print()
    print("Files this workflow requires (all should already be on disk):")
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
