"""Build the experimental spectrogram-IC-LoRA workflow by forking production.

Outputs `example_workflows/experimental/spectrogram_iclora_minimal.json`.

Rather than scratch-build the workflow (earlier approach produced chroma-
static output -- LTX 2.3 distilled needs the full production patch chain),
this script forks `example_workflows/audio-loop-music-video_latent.json`,
strips the loop infrastructure, and inserts the IC-LoRA loader + guide.
Audio is switched from frozen-song to generated: `LTXVAudioVAEEncode` +
`SetLatentNoiseMask` removed, `LTXVEmptyLatentAudio` feeds the AV concat,
`LTXVAudioVAEDecode` pipes generated audio into `VHS_VideoCombine.audio`.
This enables the V2A round-trip test -- does LTX reconstruct audio from
the spectrogram visual encoding?

The output preserves Set/Get/Reroute virtual nodes; the ComfyUI frontend
resolves them natively on queue. No API-format conversion needed.

Verified by `scripts/analyze_workflow_dag.py`: every link from loaders
to `VHS_VideoCombine` connects cleanly (the key blocker caught during
diagnostics: `LatentConcat(1605)` "Prepend Initial Render" had a dangling
second input once the loop was stripped; now bypassed so
`LTXVCropGuides(381) -> LTXVTiledVAEDecode(1604)` wires directly).

Usage:
    uv run --group dev python scripts/apply_spectrogram_iclora_minimal.py
    uv run --group dev python scripts/apply_spectrogram_iclora_minimal.py --revert

Idempotent on the output path; `--revert` deletes the file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _apply_helpers import (
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
DEFAULT_OUTPUT = Path("example_workflows/experimental/spectrogram_iclora_minimal.json")
SPECTROGRAM_MP4 = "rocks_spectrogram_edge.mp4"  # LoadVideo widget placeholder

# Loop + audio-input nodes to strip (present in production main graph)
STRIP_IDS = {
    1539, 1540,       # TensorLoopOpen/Close
    843,              # subgraph invoker (the loop body)
    1582, 1560, 1618, # AudioLoopController, AudioLoopPlanner, LoopIterationStamp
    1615, 1616,       # TimestampPromptScheduleBatchEncode + ConditioningSelectByIteration
    565, 567, 601,    # LoadAudio + TrimAudioDuration pair
    568, 569,         # MelBand
    566, 570,         # LTXVAudioVAEEncode + SetLatentNoiseMask (audio freeze gate)
    560,              # Loop VHS_VideoCombine (keep only initial-render VHS 617)
    1605,             # LatentConcat "Prepend Initial Render" (2nd input dangling post-loop-strip)
    1617,             # VAEEncode init-image -> guide-latent (fed the loop's guide input; dead)
    1318,             # LTXVTiledVAEDecode standalone "initial render" decode (now unused)
    1606,             # Reroute that fed the subgraph's reference_latent input
}

# Key production node IDs we need to reference
LTX2_PREVIEW_OVERRIDE = 503
SETNODE_MODEL = 572
LTXV_CONDITIONING = 164
CFG_GUIDER = 153
LTXV_CROP_GUIDES = 381
LTXV_IMG_TO_VID_INPLACE = 531
LTXV_CONCAT_AV = 350
LTXV_SEPARATE_AV = 245
VHS_COMBINE_INITIAL = 617
GET_VIDEO_VAE = 413
GET_AUDIO_VAE = 599  # Get_audio_vae (not 619 -- that's video_vae despite the prior assumption)
LTXV_TILED_DECODE_FINAL = 1604
LENGTH_PRIMITIVE = 526  # PrimitiveNode "length" -- single-source-of-truth for video length

ICLORA_FILE = "MergeGreen_IC-lora_ltx2.3.safetensors"
ICLORA_STRENGTH = 0.9


def build(output_path: Path) -> None:
    if not SRC.exists():
        raise SystemExit(f"Production workflow not found: {SRC}")

    wf = orjson.loads(SRC.read_bytes())
    print(f"Loaded production {SRC.name}: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    for nid in list(STRIP_IDS):
        _remove_node_and_links(wf, nid)
    print(f"After strip: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    # IC-LoRA loader between LTX2SamplingPreviewOverride -> SetNode(model)
    existing = _find_link_to_slot(wf, SETNODE_MODEL, 0)
    if existing is None:
        raise SystemExit(f"Can't find inbound link on SetNode({SETNODE_MODEL}).slot0")
    _remove_link_by_id(wf, existing[0])

    iclora_loader = _next_id(wf)
    wf["nodes"].append({
        "id": iclora_loader, "type": "LTXICLoRALoaderModelOnly",
        "pos": [-2200, 1050], "size": [400, 110], "flags": {}, "order": 0, "mode": 0,
        "inputs": [_in("model", "MODEL")],
        "outputs": [_out("model", "MODEL"), _out("latent_downscale_factor", "FLOAT")],
        "properties": {"Node name for S&R": "LTXICLoRALoaderModelOnly"},
        "widgets_values": [ICLORA_FILE, ICLORA_STRENGTH],
        "title": "IC-LoRA loader",
    })
    _add_link(wf, LTX2_PREVIEW_OVERRIDE, 0, iclora_loader, 0, "MODEL")
    _add_link(wf, iclora_loader, 0, SETNODE_MODEL, 0, "MODEL")
    print(f"  inserted IC-LoRA loader as node {iclora_loader}")

    # LoadVideo + GetVideoComponents for the spectrogram mp4
    load_video = _next_id(wf)
    wf["nodes"].append({
        "id": load_video, "type": "LoadVideo",
        "pos": [-2200, 1650], "size": [340, 300], "flags": {}, "order": 0, "mode": 0,
        "inputs": [], "outputs": [_out("VIDEO", "VIDEO")],
        "properties": {"Node name for S&R": "LoadVideo"},
        "widgets_values": [SPECTROGRAM_MP4, "image"],
        "title": "Spectrogram mp4 (IC-LoRA reference)",
    })
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

    # IC-LoRA guide: inserted between LTXVConditioning outputs and the initial
    # CFGGuider / LTXVCropGuides / LTXVConcatAVLatent.video_latent. Re-routes
    # all three so the guide-stamped conditioning + latent reach the sampler.
    guide = _next_id(wf)
    wf["nodes"].append({
        "id": guide, "type": "LTXAddVideoICLoRAGuide",
        "pos": [-1000, 1100], "size": [360, 280], "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            _in("positive", "CONDITIONING"),
            _in("negative", "CONDITIONING"),
            _in("vae", "VAE"),
            _in("latent", "LATENT"),
            _in("image", "IMAGE"),
            {"name": "latent_downscale_factor", "type": "FLOAT",
             "widget": {"name": "latent_downscale_factor"}, "link": None},
        ],
        "outputs": [
            _out("positive", "CONDITIONING"),
            _out("negative", "CONDITIONING"),
            _out("latent", "LATENT"),
        ],
        "properties": {
            "Node name for S&R": "LTXAddVideoICLoRAGuide",
            "cnr_id": "ComfyUI-LTXVideo",
        },
        "widgets_values": [0, 1.0, 1.0, "disabled", False, 256, 64],
        "title": "IC-LoRA guide (spectrogram)",
    })
    _add_link(wf, LTXV_CONDITIONING, 0, guide, 0, "CONDITIONING")
    _add_link(wf, LTXV_CONDITIONING, 1, guide, 1, "CONDITIONING")
    _add_link(wf, GET_VIDEO_VAE, 0, guide, 2, "VAE")
    _add_link(wf, LTXV_IMG_TO_VID_INPLACE, 0, guide, 3, "LATENT")
    _add_link(wf, get_comp, 0, guide, 4, "IMAGE")
    _add_link(wf, iclora_loader, 1, guide, 5, "FLOAT")

    reroutes = [
        (CFG_GUIDER, "positive", 0, "CONDITIONING"),
        (CFG_GUIDER, "negative", 1, "CONDITIONING"),
        (LTXV_CROP_GUIDES, "positive", 0, "CONDITIONING"),
        (LTXV_CROP_GUIDES, "negative", 1, "CONDITIONING"),
        (LTXV_CONCAT_AV, "video_latent", 2, "LATENT"),
    ]
    for tgt, name, guide_out, dtype in reroutes:
        node = _find_node(wf, tgt)
        if node is None:
            continue
        slot = _find_input_slot(node, name)
        existing = _find_link_to_slot(wf, tgt, slot)
        if existing:
            _remove_link_by_id(wf, existing[0])
        _add_link(wf, guide, guide_out, tgt, slot, dtype)
    print(f"  inserted IC-LoRA guide as node {guide}")

    # Audio: empty latent -> concat (replaces the stripped SetLatentNoiseMask
    # chain). Audio will be generated during sampling.
    empty_audio = _next_id(wf)
    wf["nodes"].append({
        "id": empty_audio, "type": "LTXVEmptyLatentAudio",
        "pos": [-1800, 1050], "size": [300, 106], "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            _in("audio_vae", "VAE"),
            {"name": "frames_number", "type": "INT", "widget": {"name": "frames_number"}, "link": None},
        ],
        "outputs": [_out("Latent", "LATENT")],
        "properties": {"Node name for S&R": "LTXVEmptyLatentAudio"},
        "widgets_values": [25, 1],  # fps, batch -- frames_number wired from PrimitiveNode 'length'
        "title": "Empty audio latent (generate)",
    })
    _add_link(wf, GET_AUDIO_VAE, 0, empty_audio, 0, "VAE")
    # Wire frames_number from the same PrimitiveNode that drives EmptyLTXVLatentVideo.length
    # so audio + video latent length stay in sync (single source of truth).
    _add_link(wf, LENGTH_PRIMITIVE, 0, empty_audio, 1, "INT")
    concat = _find_node(wf, LTXV_CONCAT_AV)
    if concat is None:
        raise SystemExit(f"Production workflow missing expected LTXVConcatAVLatent({LTXV_CONCAT_AV})")
    audio_slot = _find_input_slot(concat, "audio_latent")
    existing = _find_link_to_slot(wf, LTXV_CONCAT_AV, audio_slot)
    if existing:
        _remove_link_by_id(wf, existing[0])
    _add_link(wf, empty_audio, 0, LTXV_CONCAT_AV, audio_slot, "LATENT")
    print(f"  added LTXVEmptyLatentAudio({empty_audio}) -- audio will be generated")

    # Video decode path: LTXVCropGuides -> final LTXVTiledVAEDecode (bypass
    # the stripped LatentConcat "Prepend Initial Render").
    decode_final = _find_node(wf, LTXV_TILED_DECODE_FINAL)
    if decode_final:
        decode_latents_slot = _find_input_slot(decode_final, "latents")
        existing = _find_link_to_slot(wf, LTXV_TILED_DECODE_FINAL, decode_latents_slot)
        if existing:
            _remove_link_by_id(wf, existing[0])
        _add_link(wf, LTXV_CROP_GUIDES, 2, LTXV_TILED_DECODE_FINAL, decode_latents_slot, "LATENT")
        print(f"  rewired LTXVCropGuides -> LTXVTiledVAEDecode({LTXV_TILED_DECODE_FINAL}) (bypass LatentConcat)")

    # Audio decode: LTXVSeparateAVLatent.audio_latent -> LTXVAudioVAEDecode -> VHS.audio
    audio_decode = _next_id(wf)
    wf["nodes"].append({
        "id": audio_decode, "type": "LTXVAudioVAEDecode",
        "pos": [800, 1600], "size": [300, 78], "flags": {}, "order": 0, "mode": 0,
        "inputs": [_in("samples", "LATENT"), _in("audio_vae", "VAE")],
        "outputs": [_out("AUDIO", "AUDIO")],
        "properties": {"Node name for S&R": "LTXVAudioVAEDecode"},
        "widgets_values": [],
    })
    _add_link(wf, LTXV_SEPARATE_AV, 1, audio_decode, 0, "LATENT")
    _add_link(wf, GET_AUDIO_VAE, 0, audio_decode, 1, "VAE")

    vhs = _find_node(wf, VHS_COMBINE_INITIAL)
    if vhs:
        vhs_audio_slot = _find_input_slot(vhs, "audio")
        existing = _find_link_to_slot(wf, VHS_COMBINE_INITIAL, vhs_audio_slot)
        if existing:
            _remove_link_by_id(wf, existing[0])
        _add_link(wf, audio_decode, 0, VHS_COMBINE_INITIAL, vhs_audio_slot, "AUDIO")
        print(f"  VHS_VideoCombine({VHS_COMBINE_INITIAL}).audio <- generated audio")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(wf, option=orjson.OPT_INDENT_2))
    print(f"\nWrote {output_path}")
    print(f"  {len(wf['nodes'])} nodes, {len(wf['links'])} links")
    print("\nVerify with: uv run --group dev python scripts/analyze_workflow_dag.py "
          f"{output_path} --format ascii | tail -50")
    print("Open in ComfyUI frontend and queue. LoadVideo widget placeholder is "
          f"'{SPECTROGRAM_MP4}' — edit if your mp4 is named differently.")


def revert(output_path: Path) -> None:
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
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        revert(output_path)
        return
    build(output_path)


if __name__ == "__main__":
    main()
