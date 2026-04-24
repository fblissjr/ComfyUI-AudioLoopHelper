"""Phase 0a: Wire IC-LoRA on the initial-render path of the latent loop workflow.

Inserts two ComfyUI-LTXVideo nodes into a copy of the canonical
audio-loop-music-video_latent.json:

  - LTXICLoRALoaderModelOnly: between LTX2SamplingPreviewOverride(503)
    and SetNode(572). The Set/Get pair stores the MODEL under the
    "model" key, which is read by BOTH the initial-render sampler
    AND the loop subgraph (via GetNode 654 -> LoopIterationStamp 1618).
    Consequence: the LoRA-patched MODEL is active for every pass.
    This is the open MODEL-fork question the assessment doc raises;
    Phase 0a deliberately accepts it so the test exposes whether
    LoRA-patched MODEL without an IC-LoRA guide hurts loop iterations.

  - LTXAddVideoICLoRAGuide: between the initial-render conditioning +
    latent and the SamplerCustomAdvanced(161). Re-routes:
      LTXVConditioning(164).positive  -> CFGGuider(153).positive,
                                         LTXVCropGuides(381).positive
      LTXVConditioning(164).negative  -> CFGGuider(153).negative,
                                         LTXVCropGuides(381).negative
      LTXVImgToVideoInplaceKJ(531).latent -> LTXVConcatAVLatent(350).video_latent
    All re-routes pass through the IC-LoRA guide so the modified
    conditioning + latent reach the sampler. The link from
    LTXVConditioning(164).negative -> SetNode(646) "base_cond_neg"
    (loop-body negative source) is INTENTIONALLY left untouched so
    loop iterations see the unmodified conditioning baseline.

The reference image is the preprocessed init image (LTXVPreprocess 446
output), which is already what feeds LTXVImgToVideoInplaceKJ(531) for
frame 0. Same image both places means IC-LoRA reinforces the init.

Default IC-LoRA model: MergeGreen_IC-lora_ltx2.3.safetensors (single-frame
variant; matches the reference workflow shipped with the LoRA). The
loop-specific MergeGreen_Loop_IC-lora_ltx2.3.safetensors becomes
relevant in Phase 0b/Phase 1 when the IC-LoRA guide is per-iteration.

Usage:
    # Default: read example_workflows/_latent.json, write staging copy
    uv run --group dev python scripts/apply_iclora_initial_render.py

    # Custom output location
    uv run --group dev python scripts/apply_iclora_initial_render.py \\
        --output internal/scratch/my_iclora_test.json

    # In-place on a workflow you've already copied
    uv run --group dev python scripts/apply_iclora_initial_render.py \\
        --input internal/scratch/my_test.json --output internal/scratch/my_test.json

    # Revert (deletes the staging output if it exists)
    uv run --group dev python scripts/apply_iclora_initial_render.py --revert

The edit is idempotent on the OUTPUT path: re-running on an
already-migrated workflow is a no-op.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor


# Source workflow nodes that must exist for the migration to apply.
LTX2_PREVIEW_OVERRIDE_ID = 503        # MODEL output -> SetNode(572) "model"
SETNODE_MODEL_ID = 572                # Set "model" -> Get 654 -> initial CFGGuider AND loop subgraph
LTXVCONDITIONING_ID = 164             # initial-render LTXVConditioning (positive/negative)
CFGGUIDER_ID = 153                    # initial-render CFGGuider
LTXVCROPGUIDES_ID = 381               # initial-render LTXVCropGuides
LTXV_IMG_TO_VID_INPLACE_KJ_ID = 531   # writes init image into frame 0 of latent
LTXV_CONCAT_AV_LATENT_ID = 350        # joins video latent + audio latent for sampler
LTXV_PREPROCESS_ID = 446              # produces the preprocessed init image
GET_VIDEO_VAE_ID = 413                # GetNode 'video_vae' (already wired to KJ 531)

REQUIRED_SOURCE_NODES = (
    LTX2_PREVIEW_OVERRIDE_ID,
    SETNODE_MODEL_ID,
    LTXVCONDITIONING_ID,
    CFGGUIDER_ID,
    LTXVCROPGUIDES_ID,
    LTXV_IMG_TO_VID_INPLACE_KJ_ID,
    LTXV_CONCAT_AV_LATENT_ID,
    LTXV_PREPROCESS_ID,
    GET_VIDEO_VAE_ID,
)

# IC-LoRA defaults
IC_LORA_FILE = "MergeGreen_IC-lora_ltx2.3.safetensors"
IC_LORA_STRENGTH = 0.9
IC_LORA_GUIDE_FRAME_IDX = 0           # condition on frame 0 (init-image position)
IC_LORA_GUIDE_STRENGTH = 1.0          # full strength (matches reference workflow)
IC_LORA_LATENT_DOWNSCALE = 1.0        # default; overridden by loader's metadata via wired input

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "internal/scratch/audio-loop-music-video_latent_iclora_phase0a.json"


def _already_migrated(ed: WorkflowEditor) -> bool:
    return bool(ed.find_nodes_by_type("LTXICLoRALoaderModelOnly"))


def _assert_required_nodes_present(ed: WorkflowEditor) -> None:
    missing = []
    for nid in REQUIRED_SOURCE_NODES:
        try:
            ed.find_node(nid)
        except ValueError:
            missing.append(nid)
    if missing:
        raise SystemExit(
            f"Refusing to migrate: required source node(s) missing: {missing}. "
            "This script assumes the canonical latent workflow layout. "
            "If your workflow was edited, update the constants at the top "
            "of this script."
        )


def _add_iclora_loader(ed: WorkflowEditor) -> int:
    """Insert LTXICLoRALoaderModelOnly between LTX2SamplingPreviewOverride
    and the existing SetNode 'model'. The loader patches MODEL with the
    IC-LoRA at strength_model. Output: model + latent_downscale_factor."""
    nid = ed.add_top_level_node(
        node_type="LTXICLoRALoaderModelOnly",
        pos=[-2200, 1100],
        size=[400, 110],
        inputs=[
            {"name": "model", "type": "MODEL", "link": None},
        ],
        outputs=[
            {"name": "model", "type": "MODEL", "links": []},
            {"name": "latent_downscale_factor", "type": "FLOAT", "links": []},
        ],
        widgets_values=[IC_LORA_FILE, IC_LORA_STRENGTH],
        properties={
            "cnr_id": "ComfyUI-LTXVideo",
            "Node name for S&R": "LTXICLoRALoaderModelOnly",
        },
        title="IC-LoRA Loader (initial-render)",
    )

    # Splice into the model chain: 503 -> SetNode(572) becomes 503 -> loader -> SetNode(572)
    existing = ed.find_link_to_slot(SETNODE_MODEL_ID, 0)
    if existing is None:
        raise SystemExit(
            f"SetNode({SETNODE_MODEL_ID}).MODEL has no inbound link; "
            "workflow shape is unexpected."
        )
    old_link_id = existing[0]
    ed.remove_link(old_link_id)
    ed.add_link(LTX2_PREVIEW_OVERRIDE_ID, 0, nid, 0, "MODEL")
    ed.add_link(nid, 0, SETNODE_MODEL_ID, 0, "MODEL")
    return nid


def _reroute_through_iclora_guide(
    ed: WorkflowEditor, *, loader_id: int,
) -> int:
    """Insert LTXAddVideoICLoRAGuide between the initial-render
    conditioning + latent sources and their consumers. Re-routes positive,
    negative, and latent through the guide; leaves the loop-body
    'base_cond_neg' Set link untouched."""
    nid = ed.add_top_level_node(
        node_type="LTXAddVideoICLoRAGuide",
        pos=[-1000, 1100],
        size=[360, 280],
        inputs=[
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
            {"name": "latent", "type": "LATENT", "link": None},
            {"name": "image", "type": "IMAGE", "link": None},
            {
                "name": "latent_downscale_factor", "type": "FLOAT",
                "widget": {"name": "latent_downscale_factor"}, "link": None,
            },
        ],
        outputs=[
            {"name": "positive", "type": "CONDITIONING", "links": []},
            {"name": "negative", "type": "CONDITIONING", "links": []},
            {"name": "latent", "type": "LATENT", "links": []},
        ],
        widgets_values=[
            IC_LORA_GUIDE_FRAME_IDX,
            IC_LORA_GUIDE_STRENGTH,
            IC_LORA_LATENT_DOWNSCALE,
            "disabled",   # crop
            False,        # use_tiled_encode
            256,          # tile_size
            64,           # tile_overlap
        ],
        properties={
            "cnr_id": "ComfyUI-LTXVideo",
            "Node name for S&R": "LTXAddVideoICLoRAGuide",
            "ue_properties": {
                "widget_ue_connectable": {"latent_downscale_factor": True},
                "input_ue_unconnectable": {},
            },
        },
        title="IC-LoRA Guide (initial-render)",
    )

    # --- Source feeds into the IC-LoRA guide ---
    # positive: from LTXVConditioning(164).positive (slot 0)
    ed.add_link(LTXVCONDITIONING_ID, 0, nid, 0, "CONDITIONING")
    # negative: from LTXVConditioning(164).negative (slot 1)
    ed.add_link(LTXVCONDITIONING_ID, 1, nid, 1, "CONDITIONING")
    # vae: from GetNode 'video_vae' (already exists)
    ed.add_link(GET_VIDEO_VAE_ID, 0, nid, 2, "VAE")
    # latent: from LTXVImgToVideoInplaceKJ(531).latent (slot 0)
    ed.add_link(LTXV_IMG_TO_VID_INPLACE_KJ_ID, 0, nid, 3, "LATENT")
    # image: from LTXVPreprocess(446).output_image (slot 0) — same image
    # that ImgToVideoInplaceKJ already writes to frame 0
    ed.add_link(LTXV_PREPROCESS_ID, 0, nid, 4, "IMAGE")
    # latent_downscale_factor: from IC-LoRA loader output slot 1 (FLOAT
    # extracted from safetensors metadata; defaults to 1.0 if missing)
    ed.add_link(loader_id, 1, nid, 5, "FLOAT")

    # --- Re-route consumers from original sources to the guide outputs ---
    # Existing links to remove (after capturing what they fed):
    reroutes = [
        # (existing_target_node, existing_target_slot_name, guide_output_slot, dtype)
        (CFGGUIDER_ID, "positive", 0, "CONDITIONING"),
        (CFGGUIDER_ID, "negative", 1, "CONDITIONING"),
        (LTXVCROPGUIDES_ID, "positive", 0, "CONDITIONING"),
        (LTXVCROPGUIDES_ID, "negative", 1, "CONDITIONING"),
        (LTXV_CONCAT_AV_LATENT_ID, "video_latent", 2, "LATENT"),
    ]
    for tgt_id, tgt_slot_name, guide_out_slot, dtype in reroutes:
        tgt_node = ed.find_node(tgt_id)
        tgt_slot = WorkflowEditor.find_input_slot(tgt_node, tgt_slot_name)
        existing = ed.find_link_to_slot(tgt_id, tgt_slot)
        if existing is None:
            raise SystemExit(
                f"Expected inbound link on {tgt_node['type']}({tgt_id}).{tgt_slot_name} "
                "but none found."
            )
        ed.remove_link(existing[0])
        ed.add_link(nid, guide_out_slot, tgt_id, tgt_slot, dtype)

    return nid


def _migrate(input_path: Path, output_path: Path) -> None:
    if output_path.exists() and input_path != output_path:
        existing = WorkflowEditor(output_path)
        if _already_migrated(existing):
            print(
                f"{output_path.name}: already migrated, skipping. "
                "Run with --revert to delete and start over."
            )
            return

    if input_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"  copied {input_path} -> {output_path}")

    ed = WorkflowEditor(output_path)

    if _already_migrated(ed):
        print(f"{output_path.name}: already migrated (LTXICLoRALoaderModelOnly present), skipping.")
        return

    _assert_required_nodes_present(ed)

    print(f"{output_path.name}: applying Phase 0a IC-LoRA wiring...")
    loader_id = _add_iclora_loader(ed)
    print(f"  added LTXICLoRALoaderModelOnly as node {loader_id} "
          f"(model: {IC_LORA_FILE}, strength: {IC_LORA_STRENGTH})")

    guide_id = _reroute_through_iclora_guide(ed, loader_id=loader_id)
    print(f"  added LTXAddVideoICLoRAGuide as node {guide_id} "
          f"(frame_idx={IC_LORA_GUIDE_FRAME_IDX}, strength={IC_LORA_GUIDE_STRENGTH})")

    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate JSON: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Open in ComfyUI: load {output_path}")
    print( "  3. A/B render against the canonical baseline (same seed, same prompts).")
    print( "  4. Look for: changed initial-render style/structure (expected),")
    print( "     drift behavior on loop iterations (open MODEL-fork question).")


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
    ap.add_argument(
        "--input", default=DEFAULT_INPUT,
        help="Source workflow to fork from (default: %(default)s)",
    )
    ap.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output workflow path (default: %(default)s)",
    )
    ap.add_argument(
        "--revert", action="store_true",
        help="Delete the output staging file (does not touch --input).",
    )
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return

    _migrate(Path(args.input), output_path)


if __name__ == "__main__":
    main()
