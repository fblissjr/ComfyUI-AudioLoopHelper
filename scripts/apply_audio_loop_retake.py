"""Build the retake workflow by forking production.

Outputs `example_workflows/audio-loop-music-video_retake.json`.

The retake workflow regenerates a `[start_time, end_time]` window of a
previously-generated video. Forks `audio-loop-music-video_latent.json`,
strips the loop machinery + audio path + initial-image init chain, and
inserts: LoadVideo + GetVideoComponents + VAEEncode + LatentTemporalMask.

Audio is muxed back from the source mp4 via VHS_VideoCombine.audio
(Option A — video-only retake). Lip-sync preservation in the retake
range is a known limitation; if it surfaces in user A/B, motivates
Phase 3.5 (AV-aware retake) — see `internal/design/retake_workflow_design.md`.

The output preserves Set/Get/Reroute virtual nodes; the ComfyUI frontend
resolves them natively on queue. No API-format conversion needed.

Usage:
    uv run --group dev python scripts/apply_audio_loop_retake.py
    uv run --group dev python scripts/apply_audio_loop_retake.py --dry-run
    uv run --group dev python scripts/apply_audio_loop_retake.py --revert

Idempotent on the output path; re-running produces a byte-identical file
modulo node-coordinate noise. `--revert` deletes the file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers._apply_helpers import (
    add_link as _add_link,
    find_input_slot as _find_input_slot,
    find_link_to_slot as _find_link_to_slot,
    find_node as _find_node,
    in_ as _in,
    next_id as _next_id,
    out as _out,
    remove_link_by_id as _remove_link_by_id,
    remove_node_and_links as _remove_node_and_links,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = Path("example_workflows/audio-loop-music-video_retake.json")
RETAKE_VIDEO_PLACEHOLDER = "REPLACE_WITH_PRIOR_GENERATION.mp4"

# Loop + audio + init-image nodes to strip from production.
STRIP_IDS = {
    # TensorLoop machinery
    1539, 1540,        # TensorLoopOpen, TensorLoopClose
    843,               # subgraph invoker (loop body)
    # Loop control
    1582, 1560, 1618,  # AudioLoopController, AudioLoopPlanner, LoopIterationStamp
    # Per-iter conditioning
    1615, 1616,        # TimestampPromptScheduleBatchEncode, ConditioningSelectByIteration
    # Audio source path (frozen-song is irrelevant for retake; we passthrough source mp4 audio)
    565, 567, 601,     # LoadAudio, TrimAudioDuration x2
    568, 569,          # MelBandRoFormerModelLoader, MelBandRoFormerSampler
    566, 570,          # LTXVAudioVAEEncode, SetLatentNoiseMask
    # AV concat/separate (no audio in sampler for video-only retake)
    350, 245,          # LTXVConcatAVLatent, LTXVSeparateAVLatent
    # Loop-side latent prep
    1605,              # LatentConcat "Prepend Initial Render"
    1617,              # VAEEncode (init-image -> guide-latent for loop)
    1606,              # Reroute fed subgraph reference_latent
    1318,              # LTXVTiledVAEDecode standalone "initial render preview"
    # Initial-image init chain (we encode the loaded video, not an init image)
    444,               # LoadImage (reference_image.png)
    445,               # ImageResizeKJv2
    446,               # LTXVPreprocess
    531,               # LTXVImgToVideoInplaceKJ
    344,               # EmptyLTXVLatentVideo (replaced by VAEEncode of loaded video)
    1587,              # Loop-body LTXVConditioning (top-level if any; else no-op)
    # Stale Set/Get pairs after strips above
    581, 582, 604,     # Set_orig_audio, Get_orig_audio x2
    640, 641,          # Set_actual_audio, Get_actual_audio
    650, 651,          # Set_input_image, Get_input_image
    # Orphan Get nodes left behind by upstream Set strips. These are
    # virtual GetNode references whose Set source got stripped above;
    # ComfyUI tolerates them at runtime, but they clutter the workflow
    # and the dead-wire audit (_is_retake) WARNs on them.
    254, 599,          # Get_audio_vae x2 (audio path stripped)
    578, 580,          # Get_sampler, Get_sigmas (initial-render Set side stripped)
    654,               # Get_model
    648,               # Get_base_cond_neg (loop neg-cond branch)
    1273,              # Get_first_frame_guide_strength
    236, 619,          # Get_video_vae x2 (extras; 413 keeps the live one)
    1529,              # Get_start_seed (Set_start_seed retained)
    691,               # Get_window_size_seconds
}

# Production node IDs we wire to / read from.
SAMPLER = 161                 # SamplerCustomAdvanced
LTXV_CROP_GUIDES = 381        # LTXVCropGuides
LTXV_TILED_DECODE_FINAL = 1604
VHS_COMBINE = 617             # VHS_VideoCombine
GET_VIDEO_VAE = 413           # GetNode "video_vae"


def build(output_path: Path, dry_run: bool = False) -> None:
    if not SRC.exists():
        raise SystemExit(f"Production workflow not found: {SRC}")

    wf = orjson.loads(SRC.read_bytes())
    initial_count = (len(wf["nodes"]), len(wf["links"]))
    print(f"Loaded production {SRC.name}: {initial_count[0]} nodes, {initial_count[1]} links")

    # Strip loop + audio + init-image nodes
    stripped = []
    for nid in list(STRIP_IDS):
        if _find_node(wf, nid):
            _remove_node_and_links(wf, nid)
            stripped.append(nid)
    print(f"  stripped {len(stripped)} nodes (IDs: {sorted(stripped)})")
    print(f"  -> {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    # Add LoadVideo
    load_video = _next_id(wf)
    wf["nodes"].append({
        "id": load_video, "type": "LoadVideo",
        "pos": [-2200, 1650], "size": [340, 300], "flags": {}, "order": 0, "mode": 0,
        "inputs": [], "outputs": [_out("VIDEO", "VIDEO")],
        "properties": {"Node name for S&R": "LoadVideo"},
        "widgets_values": [RETAKE_VIDEO_PLACEHOLDER, "image"],
        "title": "Prior generation (mp4)",
    })

    # Add GetVideoComponents (IMAGE + AUDIO + fps from VIDEO)
    get_comp = _next_id(wf)
    wf["nodes"].append({
        "id": get_comp, "type": "GetVideoComponents",
        "pos": [-1800, 1650], "size": [300, 106], "flags": {}, "order": 0, "mode": 0,
        "inputs": [_in("video", "VIDEO")],
        "outputs": [_out("image", "IMAGE"), _out("audio", "AUDIO"), _out("fps", "FLOAT")],
        "properties": {"Node name for S&R": "GetVideoComponents"},
        "widgets_values": [],
    })
    _add_link(wf, load_video, 0, get_comp, 0, "VIDEO")

    # LTX ships no image VAE encode node, so use core VAEEncode for IMAGE -> LATENT.
    vae_encode = _next_id(wf)
    wf["nodes"].append({
        "id": vae_encode, "type": "VAEEncode",
        "pos": [-1400, 1650], "size": [220, 46], "flags": {}, "order": 0, "mode": 0,
        "inputs": [_in("pixels", "IMAGE"), _in("vae", "VAE")],
        "outputs": [_out("LATENT", "LATENT")],
        "properties": {"Node name for S&R": "VAEEncode"},
        "widgets_values": [],
    })
    _add_link(wf, get_comp, 0, vae_encode, 0, "IMAGE")
    _add_link(wf, GET_VIDEO_VAE, 0, vae_encode, 1, "VAE")

    # Add LatentTemporalMask. Defaults match the node's own defaults.
    mask = _next_id(wf)
    wf["nodes"].append({
        "id": mask, "type": "LatentTemporalMask",
        "pos": [-1100, 1650], "size": [320, 130], "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            _in("latent", "LATENT"),
            {"name": "start_time", "type": "FLOAT",
             "widget": {"name": "start_time"}, "link": None},
            {"name": "end_time", "type": "FLOAT",
             "widget": {"name": "end_time"}, "link": None},
            {"name": "fps", "type": "FLOAT",
             "widget": {"name": "fps"}, "link": None},
        ],
        "outputs": [_out("LATENT", "LATENT")],
        "properties": {"Node name for S&R": "LatentTemporalMask"},
        "widgets_values": [4.0, 8.0, 25.0],
        "title": "Retake range (start, end, fps)",
    })
    _add_link(wf, vae_encode, 0, mask, 0, "LATENT")

    # Rewire SamplerCustomAdvanced.latent_image: was 350(ConcatAV).slot0, now mask.slot0
    sampler = _find_node(wf, SAMPLER)
    if sampler is None:
        raise SystemExit(f"Production missing SamplerCustomAdvanced({SAMPLER})")
    latent_img_slot = _find_input_slot(sampler, "latent_image")
    existing = _find_link_to_slot(wf, SAMPLER, latent_img_slot)
    if existing:
        _remove_link_by_id(wf, existing[0])
    _add_link(wf, mask, 0, SAMPLER, latent_img_slot, "LATENT")

    # Rewire LTXVCropGuides.latent: was 245(SeparateAV).video_latent, now sampler.output
    crop = _find_node(wf, LTXV_CROP_GUIDES)
    if crop is None:
        raise SystemExit(f"Production missing LTXVCropGuides({LTXV_CROP_GUIDES})")
    crop_latent_slot = _find_input_slot(crop, "latent")
    existing = _find_link_to_slot(wf, LTXV_CROP_GUIDES, crop_latent_slot)
    if existing:
        _remove_link_by_id(wf, existing[0])
    _add_link(wf, SAMPLER, 0, LTXV_CROP_GUIDES, crop_latent_slot, "LATENT")

    # Rewire LTXVTiledVAEDecode.latents: was 1605(LatentConcat).slot0, now CropGuides.latent_output (slot 2)
    decode = _find_node(wf, LTXV_TILED_DECODE_FINAL)
    if decode is None:
        raise SystemExit(f"Production missing LTXVTiledVAEDecode({LTXV_TILED_DECODE_FINAL})")
    decode_latents_slot = _find_input_slot(decode, "latents")
    existing = _find_link_to_slot(wf, LTXV_TILED_DECODE_FINAL, decode_latents_slot)
    if existing:
        _remove_link_by_id(wf, existing[0])
    _add_link(wf, LTXV_CROP_GUIDES, 2, LTXV_TILED_DECODE_FINAL, decode_latents_slot, "LATENT")

    # Rewire VHS_VideoCombine.audio: was Get_orig_audio (now stripped) -> GetVideoComponents.audio
    vhs = _find_node(wf, VHS_COMBINE)
    if vhs is None:
        raise SystemExit(f"Production missing VHS_VideoCombine({VHS_COMBINE})")
    vhs_audio_slot = _find_input_slot(vhs, "audio")
    existing = _find_link_to_slot(wf, VHS_COMBINE, vhs_audio_slot)
    if existing:
        _remove_link_by_id(wf, existing[0])
    _add_link(wf, get_comp, 1, VHS_COMBINE, vhs_audio_slot, "AUDIO")

    print(f"  added LoadVideo({load_video}) + GetVideoComponents({get_comp}) + VAEEncode({vae_encode}) + LatentTemporalMask({mask})")
    print(f"  rewired: VAEEncode -> LatentTemporalMask -> SamplerCustomAdvanced({SAMPLER}).latent_image")
    print(f"  rewired: SamplerCustomAdvanced({SAMPLER}) -> LTXVCropGuides({LTXV_CROP_GUIDES}).latent -> LTXVTiledVAEDecode({LTXV_TILED_DECODE_FINAL}).latents")
    print(f"  rewired: GetVideoComponents.audio -> VHS_VideoCombine({VHS_COMBINE}).audio (Option A passthrough)")
    print(f"  final: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    if dry_run:
        print(f"\n[DRY-RUN] would write {output_path}")
        return

    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(wf, option=orjson.OPT_INDENT_2))
    print(f"\nWrote {output_path}")
    print("\nVerify with:")
    print(f"  uv run --group dev python scripts/audit_workflows.py")
    print(f"  uv run --group dev python scripts/analyze_workflow_dag.py {output_path} --format ascii | tail -50")
    print(f"\nOpen in ComfyUI; LoadVideo placeholder is '{RETAKE_VIDEO_PLACEHOLDER}'.")


def revert(output_path: Path) -> None:
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
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
                    help="Output workflow path (default: %(default)s)")
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output file.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the build diff without writing.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        revert(output_path)
        return
    build(output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
