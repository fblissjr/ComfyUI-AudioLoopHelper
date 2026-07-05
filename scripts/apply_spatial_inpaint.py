"""Build the spatial-inpaint retake workflow by forking the retake workflow.

Outputs `example_workflows/experimental/audio-loop-music-video_spatial_inpaint.json`.

Post-loop spatial edit: paint a B/W mask video over a region of a finished
music-video render; the official Lightricks in-outpainting IC-LoRA
regenerates only the masked region; the song is kept bit-identical
(passthrough, no VAE round-trip). Complements the temporal-only
`LatentTemporalMask` retake (regenerates a whole [start,end] time span).

Faithful single-stage port of `ComfyUI-LTXVideo`'s
`LTX-2.3_ICLoRA_Inpaint_Two_Stage_Distilled.json` stage 1, adapted to our
conventions: `euler` (not the official `euler_ancestral_cfg_pp`), fp8
distilled UNETLoader (no distilled-lora-384), fps 25, bit-identical audio
passthrough. Preservation of unmasked regions is the official mechanism:
a final pixel-space `LTXVLaplacianPyramidBlend` composites generated
frames only inside the mask over the clean source. Full design +
render-gate: `example_workflows/working_docs/spatial_inpaint_design.md`.

Forks `audio-loop-music-video_retake.json` (already loop-stripped, video
ingress, audio passthrough, fp8 model + sage/nag chain, canonical decode +
trim chain). Deltas: insert `LTXICLoRALoaderModelOnly` in the model chain;
add the mask branch + inpaint preprocess + empty-latent base + IC-LoRA
guide; add the Laplacian blend before the trim; drop the temporal-mask
path.

Usage:
    uv run --group dev python scripts/apply_spatial_inpaint.py
    uv run --group dev python scripts/apply_spatial_inpaint.py --dry-run
    uv run --group dev python scripts/apply_spatial_inpaint.py --revert

Staged experimental variant (like apply_audio_loop_retake.py): the output
file IS the artifact, rebuilt deterministically each run (byte-identical
re-run). `--revert` deletes it. Audit coverage is WARN-level presence
(experimental surface); F-pair regression invariants earn their keep only
at promotion to a shipped workflow.
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
    next_id as _next_id,
    out as _out,
    remove_link_by_id as _remove_link_by_id,
    remove_node_and_links as _remove_node_and_links,
    widget_in as _widget_socket,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "example_workflows/audio-loop-music-video_retake.json"
DEFAULT_OUTPUT = Path(
    "example_workflows/experimental/audio-loop-music-video_spatial_inpaint.json"
)
SOURCE_VIDEO_PLACEHOLDER = "REPLACE_WITH_PRIOR_GENERATION.mp4"
MASK_VIDEO_PLACEHOLDER = "REPLACE_WITH_MASK_VIDEO.mp4"
IN_OUTPAINTING_LORA = "ltxv/ltx2/ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors"

# Production anchors — hand-authored IDs copied unchanged from `_latent.json`
# into `_retake.json`; stable across a retake regeneration.
UNET_LOADER = 414             # UNETLoader (fp8 distilled)
SAGE_ATTN = 268               # AudioLoopHelperSageAttention (head of model-patch chain)
LTXV_CONDITIONING = 164       # LTXVConditioning (edit prompt)
CFG_GUIDER = 153              # CFGGuider
LTXV_CROP_GUIDES = 381        # LTXVCropGuides
SAMPLER = 161                 # SamplerCustomAdvanced
GET_VIDEO_VAE = 413           # GetNode "video_vae"
LTXV_TILED_DECODE = 1604      # LTXVTiledVAEDecode (final)

# Retake-SYNTHESIZED nodes (LoadVideo/GetVideoComponents/VAEEncode/
# LatentTemporalMask/TrimImageBatchToAudio) carry `_next_id()` counter IDs
# from the historical retake build — NOT stable if `_retake.json` is
# regenerated. Each is unique-by-type in the freshly-loaded retake fork, so
# resolve by type instead of hardcoding the counter output.
SRC_GET_COMP_TYPE = "GetVideoComponents"   # source: image[0], audio[1], fps[2]
STRIP_TYPES = (
    "VAEEncode",         # encoded-source base — replaced by empty-latent base
    "LatentTemporalMask",  # temporal masking — replaced by spatial mask + IC-LoRA
)
TRIM_IMAGE_TYPE = "TrimImageBatchToAudio"


def _only_of_type(wf: dict, node_type: str) -> int:
    """Return the id of the single node of `node_type` (fail loud otherwise)."""
    ids = [n["id"] for n in wf["nodes"] if n["type"] == node_type]
    if len(ids) != 1:
        raise SystemExit(
            f"Expected exactly one {node_type} in {SRC.name}, found {len(ids)}. "
            "The retake fork's shape changed; update apply_spatial_inpaint.py."
        )
    return ids[0]


def build(output_path: Path, dry_run: bool = False) -> None:
    if not SRC.exists():
        raise SystemExit(
            f"Retake workflow not found: {SRC}\n"
            "Run scripts/apply_audio_loop_retake.py first."
        )

    wf = orjson.loads(SRC.read_bytes())
    print(f"Loaded retake {SRC.name}: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    for nid in (UNET_LOADER, SAGE_ATTN, LTXV_CONDITIONING, CFG_GUIDER,
                LTXV_CROP_GUIDES, SAMPLER, GET_VIDEO_VAE, LTXV_TILED_DECODE):
        if _find_node(wf, nid) is None:
            raise SystemExit(f"Retake workflow missing expected node #{nid}")

    # Resolve retake-synthesized nodes by type BEFORE any additions (the mask
    # branch adds a second GetVideoComponents; resolve the source one first).
    src_get_comp = _only_of_type(wf, SRC_GET_COMP_TYPE)
    trim_image = _only_of_type(wf, TRIM_IMAGE_TYPE)
    strip_ids = [_only_of_type(wf, t) for t in STRIP_TYPES]

    def _rewire(tgt_id: int, slot_name: str, src_id: int, src_slot: int, dtype: str) -> None:
        """Repoint tgt_id.slot_name to src_id[src_slot], dropping the old link."""
        slot = _find_input_slot(_find_node(wf, tgt_id), slot_name)
        existing = _find_link_to_slot(wf, tgt_id, slot)
        if existing:
            _remove_link_by_id(wf, existing[0])
        _add_link(wf, src_id, src_slot, tgt_id, slot, dtype)

    # Strip the temporal-retake latent path.
    for nid in strip_ids:
        _remove_node_and_links(wf, nid)
    print(f"  stripped temporal-mask path {sorted(strip_ids)} -> "
          f"{len(wf['nodes'])} nodes")

    # --- Model chain: insert LTXICLoRALoaderModelOnly between UNETLoader and Sage.
    iclora = _next_id(wf)
    wf["nodes"].append({
        "id": iclora, "type": "LTXICLoRALoaderModelOnly",
        "pos": [-2600, 200], "size": [510, 102], "flags": {}, "order": 0, "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": None}],
        "outputs": [_out("model", "MODEL"),
                    _out("latent_downscale_factor", "FLOAT")],
        "properties": {"Node name for S&R": "LTXICLoRALoaderModelOnly"},
        "widgets_values": [IN_OUTPAINTING_LORA, 1],
        "title": "In/Outpainting IC-LoRA",
    })
    _add_link(wf, UNET_LOADER, 0, iclora, 0, "MODEL")
    _rewire(SAGE_ATTN, "model", iclora, 0, "MODEL")

    # --- Mask branch: LoadVideo -> GetVideoComponents -> ImageToMask -> DilateVideoMask.
    mask_load = _next_id(wf)
    wf["nodes"].append({
        "id": mask_load, "type": "LoadVideo",
        "pos": [-2600, 1650], "size": [340, 300], "flags": {}, "order": 0, "mode": 0,
        "inputs": [], "outputs": [_out("VIDEO", "VIDEO")],
        "properties": {"Node name for S&R": "LoadVideo"},
        "widgets_values": [MASK_VIDEO_PLACEHOLDER, "image"],
        "title": "Mask video (white = regenerate)",
    })
    mask_comp = _next_id(wf)
    wf["nodes"].append({
        "id": mask_comp, "type": "GetVideoComponents",
        "pos": [-2200, 1650], "size": [300, 106], "flags": {}, "order": 0, "mode": 0,
        "inputs": [{"name": "video", "type": "VIDEO", "link": None}],
        "outputs": [_out("image", "IMAGE"), _out("audio", "AUDIO"), _out("fps", "FLOAT")],
        "properties": {"Node name for S&R": "GetVideoComponents"},
        "widgets_values": [],
    })
    _add_link(wf, mask_load, 0, mask_comp, 0, "VIDEO")
    img2mask = _next_id(wf)
    wf["nodes"].append({
        "id": img2mask, "type": "ImageToMask",
        "pos": [-1850, 1650], "size": [270, 58], "flags": {}, "order": 0, "mode": 0,
        "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
        "outputs": [_out("MASK", "MASK")],
        "properties": {"Node name for S&R": "ImageToMask"},
        "widgets_values": ["red"],
    })
    _add_link(wf, mask_comp, 0, img2mask, 0, "IMAGE")
    dilate = _next_id(wf)
    wf["nodes"].append({
        "id": dilate, "type": "LTXVDilateVideoMask",
        "pos": [-1550, 1650], "size": [266, 105], "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            {"name": "mask", "type": "MASK", "shape": 7, "link": None},
            {"name": "image_as_mask", "type": "IMAGE", "shape": 7, "link": None},
        ],
        "outputs": [_out("mask", "MASK")],
        "properties": {"Node name for S&R": "LTXVDilateVideoMask"},
        "widgets_values": [15, 0],
        "title": "Dilate mask (spatial_radius, temporal_radius)",
    })
    _add_link(wf, img2mask, 0, dilate, 0, "MASK")

    # --- Inpaint preprocess: green-composite the source frames where masked.
    inpaint = _next_id(wf)
    wf["nodes"].append({
        "id": inpaint, "type": "LTXVInpaintPreprocess",
        "pos": [-1250, 1650], "size": [272, 54], "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            {"name": "images", "type": "IMAGE", "link": None},
            {"name": "mask", "type": "MASK", "link": None},
        ],
        "outputs": [_out("image", "IMAGE")],
        "properties": {"Node name for S&R": "LTXVInpaintPreprocess"},
        "widgets_values": [],
    })
    _add_link(wf, src_get_comp, 0, inpaint, 0, "IMAGE")   # source frames
    _add_link(wf, dilate, 0, inpaint, 1, "MASK")

    # --- Empty base latent sized to the source (GetImageSize -> EmptyLTXVLatentVideo).
    getsize = _next_id(wf)
    wf["nodes"].append({
        "id": getsize, "type": "GetImageSize",
        "pos": [-1250, 1450], "size": [163, 124], "flags": {}, "order": 0, "mode": 0,
        "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
        "outputs": [_out("width", "INT"), _out("height", "INT"), _out("batch_size", "INT")],
        "properties": {"Node name for S&R": "GetImageSize"},
        "widgets_values": [],
    })
    _add_link(wf, src_get_comp, 0, getsize, 0, "IMAGE")
    empty_lat = _next_id(wf)
    wf["nodes"].append({
        "id": empty_lat, "type": "EmptyLTXVLatentVideo",
        "pos": [-1000, 1450], "size": [210, 130], "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            _widget_socket("width", "INT"),
            _widget_socket("height", "INT"),
            _widget_socket("length", "INT"),
        ],
        "outputs": [_out("LATENT", "LATENT")],
        "properties": {"Node name for S&R": "EmptyLTXVLatentVideo"},
        "widgets_values": [960, 544, 121, 1],
    })
    _add_link(wf, getsize, 0, empty_lat, 0, "INT")  # width
    _add_link(wf, getsize, 1, empty_lat, 1, "INT")  # height
    _add_link(wf, getsize, 2, empty_lat, 2, "INT")  # length (source frame count)

    # --- IC-LoRA guide: interpose between conditioning and the sampler.
    guide = _next_id(wf)
    wf["nodes"].append({
        "id": guide, "type": "LTXAddVideoICLoRAGuideAdvanced",
        "pos": [-700, 1600], "size": [388, 326], "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
            {"name": "latent", "type": "LATENT", "link": None},
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "attention_mask", "type": "MASK", "shape": 7, "link": None},
        ],
        "outputs": [_out("positive", "CONDITIONING"), _out("negative", "CONDITIONING"),
                    _out("latent", "LATENT")],
        "properties": {"Node name for S&R": "LTXAddVideoICLoRAGuideAdvanced"},
        # [frame_idx, strength, latent_downscale, crop, use_tiled, tile_size, tile_overlap, attention_strength]
        "widgets_values": [0, 1, 1, "disabled", False, 256, 64, 1],
        "title": "In/Outpainting IC-LoRA guide",
    })
    _add_link(wf, LTXV_CONDITIONING, 0, guide, 0, "CONDITIONING")
    _add_link(wf, LTXV_CONDITIONING, 1, guide, 1, "CONDITIONING")
    _add_link(wf, GET_VIDEO_VAE, 0, guide, 2, "VAE")
    _add_link(wf, empty_lat, 0, guide, 3, "LATENT")
    _add_link(wf, inpaint, 0, guide, 4, "IMAGE")

    # Rewire CFGGuider + CropGuides conditioning from LTXVConditioning -> guide.
    _rewire(CFG_GUIDER, "positive", guide, 0, "CONDITIONING")
    _rewire(CFG_GUIDER, "negative", guide, 1, "CONDITIONING")
    _rewire(LTXV_CROP_GUIDES, "positive", guide, 0, "CONDITIONING")
    _rewire(LTXV_CROP_GUIDES, "negative", guide, 1, "CONDITIONING")
    # Rewire sampler.latent_image: LatentTemporalMask (stripped) -> guide.latent.
    _rewire(SAMPLER, "latent_image", guide, 2, "LATENT")

    # --- Laplacian blend: composite generated frames only inside the mask over clean source.
    blend = _next_id(wf)
    wf["nodes"].append({
        "id": blend, "type": "LTXVLaplacianPyramidBlend",
        "pos": [200, 1650], "size": [338, 122], "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            {"name": "image_a", "type": "IMAGE", "link": None},
            {"name": "image_b", "type": "IMAGE", "link": None},
            {"name": "mask", "type": "MASK", "link": None},
        ],
        "outputs": [_out("image", "IMAGE")],
        "properties": {"Node name for S&R": "LTXVLaplacianPyramidBlend"},
        "widgets_values": [True, 5],
        "title": "Blend edit into clean source",
    })
    _add_link(wf, LTXV_TILED_DECODE, 0, blend, 0, "IMAGE")   # generated frames
    _add_link(wf, src_get_comp, 0, blend, 1, "IMAGE")        # clean source frames
    _add_link(wf, dilate, 0, blend, 2, "MASK")
    # Splice the blend before the image trim (was decode -> TrimImageBatchToAudio).
    _rewire(trim_image, "images", blend, 0, "IMAGE")

    # --- Usage MarkdownNote (part of the build so re-runs preserve it).
    note = _next_id(wf)
    wf["nodes"].append({
        "id": note, "type": "MarkdownNote",
        "pos": [-2600, 1050], "size": [560, 360], "flags": {}, "order": 0, "mode": 0,
        "inputs": [], "outputs": [],
        "properties": {},
        "widgets_values": [
            "## Spatial-inpaint retake (EXPERIMENTAL — render-unvalidated)\n\n"
            "Regenerate a masked spatial region of a finished render via the "
            "official in-outpainting IC-LoRA; the song stays bit-identical.\n\n"
            "**Inputs**\n"
            "- Source video = a prior full render (mp4).\n"
            "- Mask video = B/W, SAME resolution + frame count as the source. "
            "WHITE = regenerate, BLACK = keep. (`ImageToMask` reads the red "
            "channel.)\n\n"
            "**Dials**\n"
            "- Dilate `spatial_radius` (15): grows the mask; raise to feather "
            "edges, lower for tighter edits. `temporal_radius` 0.\n"
            "- Guide `attention_strength` (widget 8, =1): lower to soften the "
            "IC-LoRA's influence.\n"
            "- Edit prompt = CLIPTextEncode #169 (describe the desired content "
            "of the masked region).\n\n"
            "**Preservation**: the Laplacian blend composites generated pixels "
            "ONLY inside the mask over the clean source — unmasked regions are "
            "untouched.\n\n"
            "**Deviations from the official workflow**: `euler` (not "
            "`euler_ancestral_cfg_pp`); fp8 distilled model (no distilled-lora-"
            "384); audio passthrough (not VAE round-trip); single-stage (no 2x "
            "refine). See `example_workflows/working_docs/spatial_inpaint_design.md`.\n\n"
            "**Render gate**: unproven at distilled CFG=1. If the masked region "
            "stalls, try ancestral or raise `attention_strength`; if unmasked "
            "regions drift, the blend mask is too small."
        ],
        "title": "How to use — spatial inpaint",
    })

    print(f"  added: IC-LoRA loader({iclora}) + mask branch({mask_load},{mask_comp},"
          f"{img2mask},{dilate}) + inpaint({inpaint}) + base({getsize},{empty_lat}) + "
          f"guide({guide}) + blend({blend})")
    print(f"  final: {len(wf['nodes'])} nodes, {len(wf['links'])} links")

    if dry_run:
        print(f"\n[DRY-RUN] would write {output_path}")
        return

    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(wf, option=orjson.OPT_INDENT_2))
    try:
        shown = output_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = output_path
    print(f"\nWrote {output_path}")
    print("\nVerify:")
    print(f"  uv run --group dev python scripts/audit_workflows.py {shown}")
    print(f"\nLoadVideo placeholders: source '{SOURCE_VIDEO_PLACEHOLDER}', "
          f"mask '{MASK_VIDEO_PLACEHOLDER}'. Design + render gate: "
          "example_workflows/working_docs/spatial_inpaint_design.md")


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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        revert(output_path)
        return
    build(output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
