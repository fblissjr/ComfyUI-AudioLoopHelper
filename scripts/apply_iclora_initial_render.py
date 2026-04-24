"""Phase 0a: Wire IC-LoRA on the initial-render path of the latent loop workflow.

Forks `example_workflows/audio-loop-music-video_latent.json` into a
staging copy under `internal/scratch/`, splicing in two
ComfyUI-LTXVideo nodes:

  - `LTXICLoRALoaderModelOnly` between `LTX2SamplingPreviewOverride(503)`
    and `SetNode(572)`. The Set/Get(654) pair also feeds the loop
    subgraph via `LoopIterationStamp(1618)`, so the LoRA-patched MODEL
    is active for BOTH initial render AND loop iterations. This is the
    open MODEL-fork question from `internal/ic_lora_assessment.md`:
    Phase 0a accepts the coupling so the A/B exposes whether a
    LoRA-patched MODEL without an attached guide hurts loop iterations.

  - `LTXAddVideoICLoRAGuide` on the initial-render conditioning + latent
    path. The loop-body negative source (`SetNode 646 base_cond_neg`)
    is intentionally untouched so loop iterations see the unmodified
    conditioning baseline.

Usage:
    uv run --group dev python scripts/apply_iclora_initial_render.py
    uv run --group dev python scripts/apply_iclora_initial_render.py --revert

Idempotent on the OUTPUT path; `--revert` deletes the staging file.
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


# (target_node, target_slot_name, guide_output_slot, dtype)
# The sources feeding these slots get re-routed through the IC-LoRA guide
# so the modified conditioning + latent reach the sampler.
GUIDE_REROUTES = (
    (CFGGUIDER_ID, "positive", 0, "CONDITIONING"),
    (CFGGUIDER_ID, "negative", 1, "CONDITIONING"),
    (LTXVCROPGUIDES_ID, "positive", 0, "CONDITIONING"),
    (LTXVCROPGUIDES_ID, "negative", 1, "CONDITIONING"),
    (LTXV_CONCAT_AV_LATENT_ID, "video_latent", 2, "LATENT"),
)


def _add_guide_node(ed: WorkflowEditor) -> int:
    return ed.add_top_level_node(
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


def _wire_guide_inputs(ed: WorkflowEditor, guide_id: int, loader_id: int) -> None:
    ed.add_link(LTXVCONDITIONING_ID, 0, guide_id, 0, "CONDITIONING")
    ed.add_link(LTXVCONDITIONING_ID, 1, guide_id, 1, "CONDITIONING")
    ed.add_link(GET_VIDEO_VAE_ID, 0, guide_id, 2, "VAE")
    ed.add_link(LTXV_IMG_TO_VID_INPLACE_KJ_ID, 0, guide_id, 3, "LATENT")
    # Same preprocessed init image that already writes to frame 0 via
    # ImgToVideoInplaceKJ -- means IC-LoRA reinforces the init commitment.
    ed.add_link(LTXV_PREPROCESS_ID, 0, guide_id, 4, "IMAGE")
    # latent_downscale_factor auto-extracts from safetensors metadata
    # (defaults to 1.0 if missing); wire as input so the loader drives it.
    ed.add_link(loader_id, 1, guide_id, 5, "FLOAT")


def _reroute_consumers_through_guide(ed: WorkflowEditor, guide_id: int) -> None:
    for tgt_id, tgt_slot_name, guide_out_slot, dtype in GUIDE_REROUTES:
        tgt_node = ed.find_node(tgt_id)
        tgt_slot = WorkflowEditor.find_input_slot(tgt_node, tgt_slot_name)
        if ed.find_link_to_slot(tgt_id, tgt_slot) is None:
            raise SystemExit(
                f"Expected inbound link on {tgt_node['type']}({tgt_id}).{tgt_slot_name} "
                "but none found."
            )
        ed.rewire_input(tgt_id, tgt_slot, guide_id, guide_out_slot, dtype)


def _reroute_through_iclora_guide(ed: WorkflowEditor, *, loader_id: int) -> int:
    """Insert LTXAddVideoICLoRAGuide between the initial-render conditioning
    + latent sources and their consumers. Loop-body `base_cond_neg` Set
    link is intentionally untouched so loop iterations see the unmodified
    baseline."""
    guide_id = _add_guide_node(ed)
    _wire_guide_inputs(ed, guide_id, loader_id)
    _reroute_consumers_through_guide(ed, guide_id)
    return guide_id


def _migrate(input_path: Path, output_path: Path) -> None:
    # Preserve user edits: if the staging file is already migrated, bail
    # before overwriting. --revert deletes it for a fresh start.
    if output_path.exists() and input_path != output_path and _already_migrated(WorkflowEditor(output_path)):
        print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
        return

    if input_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"  copied {input_path} -> {output_path}")

    ed = WorkflowEditor(output_path)

    if _already_migrated(ed):
        print(f"{output_path.name}: already migrated, skipping.")
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
