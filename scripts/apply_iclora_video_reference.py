"""apply_iclora_video_reference.

Last updated: 2026-04-30

Wires video-reference IC-LoRA into the audio-loop pipeline. Forks
`example_workflows/audio-loop-music-video_latent.json` (post-Step-0,
i.e. with dead LoRA scaffolding stripped) into a staging copy under
`internal/scratch/`, splicing in:

  Top-level (outside loop, runs once):
    - LTXICLoRALoaderModelOnly between #503 (LTX2SamplingPreviewOverride)
      and #572 (SetNode "model"). Patches the LTX UNet with the chosen
      IC-LoRA. The patched MODEL feeds BOTH initial render AND every
      loop iteration via the existing SetNode broadcast.
    - VHS_LoadVideo for the reference clip MP4.
    - ImageResizeKJv2 reading width/height from LTXFramePlanner
      (matches init-image preprocessing).
    - LTXVPreprocess(val=18) — F2 symmetry: identical preprocessing
      across the two image paths (init image and ref video).

  Subgraph (per-iter, inside the loop body):
    - New IMAGE input slot `reference_video`.
    - GetImageRangeFromBatch (KJNodes) — slices a window of
      `num_frames` reference frames starting at `start_index`. Default
      widget values (start_index=0, num_frames=25) yield STATIC reuse
      (same slice every iter); changing widgets enables sliding mode.
    - LTXAddVideoICLoRAGuide — inserted between #1519 (LTXVAddLatentGuide
      that handles the init-image guide) and the existing F3 cropguides
      consumers (#655 LTXVCropGuidesNoLatent feeds CFGGuider, #2008
      LTXVCropGuides feeds the LATENT path). The new guide receives the
      init-image-conditioned outputs of #1519 and adds the IC-LoRA
      conditioning on top, preserving F3 symmetry.

Pre-flight checks:
  - --reference-video file exists
  - --ic-lora-file file exists
  - Input workflow is post-Step-0 (no dead scaffolding nodes
    #1625/#1626/#1627). Refuses if Step 0 hasn't been applied.

Reference-window evolution is controlled by --ref-mode:
  - static (default): same window every iter (start_index widget).
  - sliding: window advances with video_start_time per iter via a
    SimpleCalculatorKJ in the loop subgraph computing
    `round(video_start_time * ref_fps)`. --ref-fps (default 25) is
    baked into BOTH VHS_LoadVideo.force_rate and the calculator's
    expression as a single source of truth.

Compatibility:
  - Requires `apply_strip_dead_lora_loaders.py` to have been applied
    first. Pre-flight blocks if not.
  - Orthogonal to F2/F3/F4/F5/F6/F7/F8/F9/F10. Touches only the post-
    strip MODEL chain (#503 -> #572) and the subgraph CONDITIONING +
    LATENT chain through #1519's outputs.
  - Phase 0a's `apply_iclora_initial_render.py` is independent: it
    targets the INITIAL-RENDER conditioning path with a different
    `LTXAddVideoICLoRAGuide` instance. The two scripts can coexist
    (initial-only IC-LoRA + in-loop IC-LoRA on different reference
    images), but typical use is one OR the other.

Usage:
    uv run --group dev python scripts/apply_iclora_video_reference.py \\
        --reference-video /path/to/ref.mp4 \\
        --ic-lora-file <comfyui_models>/loras/<...>/lora_weights.safetensors

    uv run --group dev python scripts/apply_iclora_video_reference.py --revert
    uv run --group dev python scripts/apply_iclora_video_reference.py --dry-run \\
        --reference-video ... --ic-lora-file ...

Idempotent on the OUTPUT path. `--revert` deletes the staging file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level anchor node IDs in the canonical (post-strip).
LTX2_PREVIEW_OVERRIDE_ID = 503        # MODEL output -> SetNode(572) via direct link
SETNODE_MODEL_ID = 572                # Set "model"
LTX_FRAME_PLANNER_ID = 1634           # SSoT: width / height / fps / frames

# Subgraph anchor node IDs.
SG_LATENT_GUIDE_ID = 1519             # LTXVAddLatentGuide (init-image guide)
SG_CROPGUIDES_NOLATENT_ID = 655       # LTXVCropGuidesNoLatent (CONDITIONING path to CFGGuider)
SG_CROPGUIDES_ID = 2008               # LTXVCropGuides (LATENT path)
SG_CONCAT_AV_ID = 583                 # LTXVConcatAVLatent (consumes #1519.latent)

# Dead scaffolding ids that MUST be absent (Step 0 must have run).
_STEP0_REQUIRED_ABSENT_IDS = (1625, 1626, 1627)

# Widget defaults — match the cameraman reference workflow's shape.
DEFAULT_IC_LORA_STRENGTH = 1.0
DEFAULT_GUIDE_FRAME_IDX = 1            # 1 mod 8 == 1, satisfies iclora.py constraint
DEFAULT_GUIDE_STRENGTH = 1.0
DEFAULT_LATENT_DOWNSCALE = 1.0
DEFAULT_GUIDE_CROP = "disabled"
DEFAULT_GUIDE_TILED = False
DEFAULT_GUIDE_TILE_SIZE = 256
DEFAULT_GUIDE_TILE_OVERLAP = 64
DEFAULT_REF_NUM_FRAMES = 25            # matches our typical window length
DEFAULT_REF_START_INDEX = 0            # static reuse (slot 0 each iter)
DEFAULT_PREPROCESS_VAL = 18            # F2 symmetry with init-image LTXVPreprocess
DEFAULT_REF_FPS = 25                   # VHS_LoadVideo.force_rate; also baked
                                       # into sliding-mode calculator expression
                                       # (single source of truth)
DEFAULT_REF_MODE = "static"            # "static" = same ref-window every iter;
                                       # "sliding" = window advances with
                                       # video_start_time per iter
SG_VIDEO_START_TIME_INPUT = "video_start_time"  # subgraph FLOAT input slot;
                                       # name-resolved at apply time

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "internal/scratch/audio-loop-music-video_latent_iclora_video.json"


# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------

def _preflight(
    ed: WorkflowEditor,
    reference_video: Path,
    ic_lora_file: Path,
) -> str | None:
    """Return error message string if any check fails; None if all good."""
    if not reference_video.exists():
        return (
            f"--reference-video file does not exist: {reference_video}. "
            "Pass an absolute path to an mp4/mov of the driving reference clip."
        )
    if not ic_lora_file.exists():
        return (
            f"--ic-lora-file does not exist: {ic_lora_file}. "
            "Pass an absolute path to the IC-LoRA safetensors file."
        )
    # Step 0 must have run: no dead scaffolding.
    present = [nid for nid in _STEP0_REQUIRED_ABSENT_IDS if ed.has_node(nid)]
    if present:
        return (
            f"input workflow still contains dead LoRA scaffolding "
            f"(nodes {present}). Run scripts/apply_strip_dead_lora_loaders.py "
            "first, then re-run this script."
        )
    # Required source nodes for the splice.
    missing = ed.require_nodes((
        LTX2_PREVIEW_OVERRIDE_ID, SETNODE_MODEL_ID, LTX_FRAME_PLANNER_ID,
    ))
    if missing:
        return f"missing required top-level nodes: {missing}"
    sg = ed.get_subgraph(0)
    if sg is None:
        return "input workflow has no subgraph (loop body)"
    sg_node_ids = {n["id"] for n in sg.get("nodes", [])}
    sg_missing = [
        nid for nid in (SG_LATENT_GUIDE_ID, SG_CROPGUIDES_NOLATENT_ID,
                        SG_CROPGUIDES_ID, SG_CONCAT_AV_ID)
        if nid not in sg_node_ids
    ]
    if sg_missing:
        return f"missing required subgraph nodes: {sg_missing}"
    return None


def _already_applied(ed: WorkflowEditor) -> bool:
    """Idempotency check: the staging file already has the splice."""
    if not ed.find_nodes_by_type("LTXICLoRALoaderModelOnly"):
        return False
    sg = ed.get_subgraph(0)
    if sg is None:
        return False
    if not any(n.get("type") == "LTXAddVideoICLoRAGuide" for n in sg.get("nodes", [])):
        return False
    return True


# --------------------------------------------------------------------------
# Top-level splice: IC-LoRA loader
# --------------------------------------------------------------------------

def _add_iclora_loader(ed: WorkflowEditor, ic_lora_file: str, strength: float) -> int:
    """Insert LTXICLoRALoaderModelOnly between #503 and #572.

    Pre-condition: a direct link #503.0 -> #572.0 exists (the post-Step-0
    rebridge). Replaces it with: #503.0 -> loader.0; loader.0 -> #572.0.
    """
    nid = ed.add_top_level_node(
        node_type="LTXICLoRALoaderModelOnly",
        pos=[-2200, 1100],
        size=[480, 90],
        inputs=[
            {"name": "model", "type": "MODEL", "link": None},
        ],
        outputs=[
            {"name": "model", "type": "MODEL", "links": []},
            {"name": "latent_downscale_factor", "type": "FLOAT", "links": []},
        ],
        widgets_values=[ic_lora_file, strength],
        properties={
            "cnr_id": "ComfyUI-LTXVideo",
            "Node name for S&R": "LTXICLoRALoaderModelOnly",
        },
        title="IC-LoRA Loader (video reference)",
    )
    if ed.find_link_to_slot(SETNODE_MODEL_ID, 0) is None:
        raise SystemExit(
            f"SetNode({SETNODE_MODEL_ID}).MODEL has no inbound link; "
            "expected post-Step-0 direct rebridge link #503.0 -> #572.0."
        )
    ed.rewire_input(SETNODE_MODEL_ID, 0, nid, 0, "MODEL")
    ed.add_link(LTX2_PREVIEW_OVERRIDE_ID, 0, nid, 0, "MODEL")
    return nid


# --------------------------------------------------------------------------
# Top-level splice: ref-video preprocessing chain
# --------------------------------------------------------------------------

def _add_ref_video_chain(
    ed: WorkflowEditor, reference_video: Path, ref_fps: int,
) -> tuple[int, int, int]:
    """Add VHS_LoadVideo -> ImageResizeKJv2 -> LTXVPreprocess(val=18). Returns
    (loader_id, resizer_id, preproc_id). The preproc.output_image is what
    flows into the subgraph invoker's new IMAGE slot. ref_fps is baked into
    VHS_LoadVideo.force_rate (single source of truth — also baked into the
    sliding-mode calculator expression when --ref-mode sliding)."""
    loader_id = ed.add_top_level_node(
        node_type="VHS_LoadVideo",
        pos=[-2200, 1300],
        size=[270, 460],
        inputs=[
            {"name": "meta_batch", "shape": 7, "type": "VHS_BatchManager", "link": None},
            {"name": "vae", "shape": 7, "type": "VAE", "link": None},
        ],
        outputs=[
            {"name": "IMAGE", "type": "IMAGE", "links": []},
            {"name": "frame_count", "type": "INT", "links": []},
            {"name": "audio", "type": "AUDIO", "links": []},
            {"name": "video_info", "type": "VHS_VIDEOINFO", "links": []},
        ],
        widgets_values={
            "video": str(reference_video),
            "force_rate": ref_fps,
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": 0,
            "skip_first_frames": 0,
            "select_every_nth": 1,
            "format": "LTXV",
        },
        properties={
            "cnr_id": "ComfyUI-VideoHelperSuite",
            "Node name for S&R": "VHS_LoadVideo",
        },
        title="Reference Video (IC-LoRA)",
    )

    resizer_id = ed.add_top_level_node(
        node_type="ImageResizeKJv2",
        pos=[-1850, 1300],
        size=[270, 290],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "mask", "shape": 7, "type": "MASK", "link": None},
            {"name": "width", "type": "INT",
             "widget": {"name": "width"}, "link": None},
            {"name": "height", "type": "INT",
             "widget": {"name": "height"}, "link": None},
        ],
        outputs=[
            {"name": "IMAGE", "type": "IMAGE", "links": []},
            {"name": "width", "type": "INT", "links": []},
            {"name": "height", "type": "INT", "links": []},
            {"name": "mask", "type": "MASK", "links": []},
        ],
        widgets_values=[
            512, 512, "lanczos", "stretch",
            "0, 0, 0", "center", 32, "cpu",
        ],
        properties={"Node name for S&R": "ImageResizeKJv2"},
        title="Resize ref-video frames",
    )

    preproc_id = ed.add_top_level_node(
        node_type="LTXVPreprocess",
        pos=[-1500, 1300],
        size=[270, 80],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
        ],
        outputs=[
            {"name": "output_image", "type": "IMAGE", "links": []},
        ],
        widgets_values=[DEFAULT_PREPROCESS_VAL],
        properties={
            "cnr_id": "ComfyUI-LTXVideo",
            "Node name for S&R": "LTXVPreprocess",
        },
        title="Preprocess ref-video (F2 symmetric)",
    )

    ed.add_link(loader_id, 0, resizer_id, 0, "IMAGE")
    ed.add_link(LTX_FRAME_PLANNER_ID, 0, resizer_id, 2, "INT")
    ed.add_link(LTX_FRAME_PLANNER_ID, 1, resizer_id, 3, "INT")
    ed.add_link(resizer_id, 0, preproc_id, 0, "IMAGE")

    return loader_id, resizer_id, preproc_id


# --------------------------------------------------------------------------
# Subgraph schema: add new IMAGE input slot
# --------------------------------------------------------------------------

def _add_ref_video_subgraph_input(ed: WorkflowEditor) -> int:
    """Append a new IMAGE input named `reference_video` to the subgraph
    schema. Returns its slot index (the position in sg["inputs"])."""
    sg = ed.get_subgraph(0)
    assert sg is not None
    inputs = sg.setdefault("inputs", [])
    new_slot = len(inputs)
    new_input = {
        "id": str(uuid.uuid4()),
        "name": "reference_video",
        "type": "IMAGE",
        "linkIds": [],
        "localized_name": "reference_video",
        "label": "reference video frames",
        "pos": [-3015, 3590],  # near the existing input column
    }
    inputs.append(new_input)
    return new_slot


def _add_invoker_input(ed: WorkflowEditor) -> int:
    """Append a new IMAGE input slot to the top-level subgraph invoker
    node so the ref-video preprocessed output can flow in. Returns the
    slot index."""
    invoker = ed.find_subgraph_invoker(0)
    if invoker is None:
        raise SystemExit("subgraph invoker not found; cannot add IMAGE input")
    inputs = invoker.setdefault("inputs", [])
    new_slot = len(inputs)
    inputs.append({
        "name": "reference_video",
        "type": "IMAGE",
        "label": "reference video frames",
        "link": None,
    })
    return new_slot


# --------------------------------------------------------------------------
# Subgraph splice: GetImageRangeFromBatch + LTXAddVideoICLoRAGuide
# --------------------------------------------------------------------------

def _splice_subgraph(
    ed: WorkflowEditor,
    new_input_slot: int,
    *,
    frame_idx: int,
    guide_strength: float,
    latent_downscale: float,
    ref_start_index: int,
    ref_num_frames: int,
) -> tuple[int, int]:
    """Insert GetImageRangeFromBatch + LTXAddVideoICLoRAGuide inside the
    subgraph and rewire #1519's downstream consumers through the new
    guide. Returns (slicer_id, guide_id)."""
    sg = ed.get_subgraph(0)
    assert sg is not None

    # The subgraph schema now has the new "reference_video" input. The
    # virtual input distributor (-10) feeds nodes inside the subgraph
    # via slot indices that mirror sg["inputs"] order.

    # Add GetImageRangeFromBatch
    slicer_id = ed.add_subgraph_node(
        node_type="GetImageRangeFromBatch",
        pos=[1200, 3500],
        size=[270, 78],
        inputs=[
            {"name": "start_index", "type": "INT",
             "widget": {"name": "start_index"}, "link": None},
            {"name": "num_frames", "type": "INT",
             "widget": {"name": "num_frames"}, "link": None},
            {"name": "images", "shape": 7, "type": "IMAGE", "link": None},
            {"name": "masks", "shape": 7, "type": "MASK", "link": None},
        ],
        outputs=[
            {"name": "IMAGE", "type": "IMAGE", "links": []},
            {"name": "MASK", "type": "MASK", "links": []},
        ],
        widgets_values=[ref_start_index, ref_num_frames],
        properties={
            "cnr_id": "kjnodes",
            "Node name for S&R": "GetImageRangeFromBatch",
        },
        title="Slice ref-video for this iter",
    )
    # Wire slicer.images from the new subgraph input distributor
    ed.add_subgraph_link(-10, new_input_slot, slicer_id, 2, "IMAGE")

    # Find existing wiring on #1519 to splice through
    n_1519 = ed.find_subgraph_node(SG_LATENT_GUIDE_ID, 0)
    assert n_1519 is not None

    # Add LTXAddVideoICLoRAGuide. We'll wire its inputs from #1519's
    # outputs and from the slicer (image), and rewire #1519's existing
    # output consumers to come from this guide instead.
    guide_id = ed.add_subgraph_node(
        node_type="LTXAddVideoICLoRAGuide",
        pos=[1500, 3300],
        size=[280, 280],
        inputs=[
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
            {"name": "latent", "type": "LATENT", "link": None},
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "frame_idx", "type": "INT",
             "widget": {"name": "frame_idx"}, "link": None},
            {"name": "strength", "type": "FLOAT",
             "widget": {"name": "strength"}, "link": None},
            {"name": "latent_downscale_factor", "type": "FLOAT",
             "widget": {"name": "latent_downscale_factor"}, "link": None},
            {"name": "crop", "type": "COMBO",
             "widget": {"name": "crop"}, "link": None},
            {"name": "use_tiled_encode", "type": "BOOLEAN",
             "widget": {"name": "use_tiled_encode"}, "link": None},
            {"name": "tile_size", "type": "INT",
             "widget": {"name": "tile_size"}, "link": None},
            {"name": "tile_overlap", "type": "INT",
             "widget": {"name": "tile_overlap"}, "link": None},
        ],
        outputs=[
            {"name": "positive", "type": "CONDITIONING", "links": []},
            {"name": "negative", "type": "CONDITIONING", "links": []},
            {"name": "latent", "type": "LATENT", "links": []},
        ],
        widgets_values=[
            frame_idx, guide_strength, latent_downscale,
            DEFAULT_GUIDE_CROP, DEFAULT_GUIDE_TILED,
            DEFAULT_GUIDE_TILE_SIZE, DEFAULT_GUIDE_TILE_OVERLAP,
        ],
        properties={
            "cnr_id": "ComfyUI-LTXVideo",
            "Node name for S&R": "LTXAddVideoICLoRAGuide",
        },
        title="IC-LoRA Guide (video reference)",
    )

    ed.add_subgraph_link(slicer_id, 0, guide_id, 4, "IMAGE")

    vae_slot = WorkflowEditor.find_input_slot(sg, "vae")
    ed.add_subgraph_link(-10, vae_slot, guide_id, 2, "VAE")

    # Single-pass collect of #1519's existing consumers, grouped by output slot.
    # remove_subgraph_link rebinds the list, so snapshot first.
    consumers_by_slot: dict[int, list[tuple[int, int]]] = {0: [], 1: [], 2: []}
    consumer_link_ids: list[int] = []
    for link in sg.get("links", []):
        if link.get("origin_id") != SG_LATENT_GUIDE_ID:
            continue
        slot = link.get("origin_slot")
        if slot in consumers_by_slot:
            consumers_by_slot[slot].append((link["target_id"], link["target_slot"]))
            consumer_link_ids.append(link["id"])

    for lid in consumer_link_ids:
        ed.remove_subgraph_link(lid, 0)

    ed.add_subgraph_link(SG_LATENT_GUIDE_ID, 0, guide_id, 0, "CONDITIONING")
    ed.add_subgraph_link(SG_LATENT_GUIDE_ID, 1, guide_id, 1, "CONDITIONING")
    ed.add_subgraph_link(SG_LATENT_GUIDE_ID, 2, guide_id, 3, "LATENT")

    for src_slot, dtype in ((0, "CONDITIONING"), (1, "CONDITIONING"), (2, "LATENT")):
        guide_out_slot = src_slot if src_slot < 2 else 2
        for tgt_id, tgt_slot in consumers_by_slot[src_slot]:
            ed.add_subgraph_link(guide_id, guide_out_slot, tgt_id, tgt_slot, dtype)

    return slicer_id, guide_id


# --------------------------------------------------------------------------
# Subgraph splice: sliding-mode calculator (Phase 2)
# --------------------------------------------------------------------------

def _splice_sliding_calculator(
    ed: WorkflowEditor, slicer_id: int, ref_fps: int,
) -> int:
    """Add SimpleCalculatorKJ inside the subgraph + rewire
    GetImageRangeFromBatch.start_index from widget to a wired INT input
    fed by the calculator's Int output (slot 1).

    The calculator evaluates `round(a * <ref_fps>)` where `a` is the
    subgraph's video_start_time (FLOAT, varies per-iter from
    AudioLoopController). Result is the frame index into the ref-video
    batch, advancing the consumed window per iter.

    ref_fps is baked into the expression string rather than carried as
    a separate variable widget — keeps the workflow self-documenting
    and avoids the SimpleCalculatorKJ Autogrow widget complexity.

    Returns the calculator's node id.
    """
    sg = ed.get_subgraph(0)
    assert sg is not None

    # Locate the subgraph's video_start_time input slot (FLOAT).
    vst_slot = WorkflowEditor.find_input_slot(sg, SG_VIDEO_START_TIME_INPUT)
    if vst_slot is None:
        raise SystemExit(
            f"--ref-mode sliding requires subgraph input "
            f"'{SG_VIDEO_START_TIME_INPUT}' (FLOAT). Not found in this workflow."
        )

    # Add SimpleCalculatorKJ. The shape mirrors the upstream KJNodes
    # workflow convention: top-level `a`/`b` inputs (backwards-compat
    # path), `expression` widget input, plus `variables.a`/`variables.b`
    # autogrow children. Only `a` is wired; `b` is unused (force_rate
    # baked into expression).
    calc_id = ed.add_subgraph_node(
        node_type="SimpleCalculatorKJ",
        pos=[900, 3700],
        size=[270, 130],
        inputs=[
            {"localized_name": "a", "name": "a", "shape": 7,
             "type": "*", "link": None},
            {"localized_name": "b", "name": "b", "shape": 7,
             "type": "*", "link": None},
            {"localized_name": "expression", "name": "expression",
             "type": "STRING", "widget": {"name": "expression"}, "link": None},
            {"label": "a", "localized_name": "variables.a",
             "name": "variables.a", "shape": 7,
             "type": "INT,FLOAT,BOOLEAN", "link": None},
            {"label": "b", "localized_name": "variables.b",
             "name": "variables.b", "shape": 7,
             "type": "INT,FLOAT,BOOLEAN", "link": None},
        ],
        outputs=[
            {"name": "FLOAT", "type": "FLOAT", "links": []},
            {"name": "INT", "type": "INT", "links": []},
            {"name": "BOOLEAN", "type": "BOOLEAN", "links": []},
        ],
        widgets_values=[f"round(a * {ref_fps})"],
        properties={
            "cnr_id": "kjnodes",
            "Node name for S&R": "SimpleCalculatorKJ",
        },
        title=f"Sliding-mode start_index = round(video_start_time * {ref_fps})",
    )

    # Wire calculator.a from subgraph input video_start_time (FLOAT, via -10).
    ed.add_subgraph_link(-10, vst_slot, calc_id, 0, "*")

    # Rewire slicer.start_index: widget -> wired INT from calculator slot 1.
    slicer = ed.find_subgraph_node(slicer_id, 0)
    assert slicer is not None
    start_idx_input = next(
        (i for i in slicer["inputs"] if i.get("name") == "start_index"),
        None,
    )
    if start_idx_input is None:
        raise SystemExit("slicer GetImageRangeFromBatch missing 'start_index' input")
    # Drop the widget field; the link will supply the value at runtime.
    start_idx_input.pop("widget", None)
    # add_subgraph_link sets the target's input.link
    ed.add_subgraph_link(calc_id, 1, slicer_id, 0, "INT")

    return calc_id


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _migrate(
    input_path: Path, output_path: Path,
    reference_video: Path, ic_lora_file: Path,
    ic_lora_strength: float, guide_frame_idx: int, guide_strength: float,
    latent_downscale: float, ref_start_index: int, ref_num_frames: int,
    ref_mode: str, ref_fps: int,
    dry_run: bool,
) -> None:
    if input_path != output_path and output_path.exists():
        if _already_applied(WorkflowEditor(output_path)):
            print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
            return

    if not dry_run and input_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"  copied {input_path} -> {output_path}")

    ed_target = output_path if (output_path.exists() and not dry_run) else input_path
    ed = WorkflowEditor(ed_target)

    if _already_applied(ed):
        print(f"{output_path.name}: already migrated, skipping.")
        return

    err = _preflight(ed, reference_video, ic_lora_file)
    if err is not None:
        raise SystemExit(f"Refusing to migrate: {err}")

    if dry_run:
        print(f"would migrate {output_path.name} (loader + ref-video chain + subgraph splice)")
        return

    print(f"{output_path.name}: applying video-reference IC-LoRA wiring...")
    loader_id = _add_iclora_loader(ed, str(ic_lora_file), ic_lora_strength)
    print(f"  added LTXICLoRALoaderModelOnly as node {loader_id}")

    vhs_id, resize_id, preproc_id = _add_ref_video_chain(ed, reference_video, ref_fps)
    print(f"  added VHS_LoadVideo({vhs_id}) [force_rate={ref_fps}] "
          f"-> ImageResizeKJv2({resize_id}) -> LTXVPreprocess({preproc_id})")

    sg_input_slot = _add_ref_video_subgraph_input(ed)
    invoker_slot = _add_invoker_input(ed)
    invoker = ed.find_subgraph_invoker(0)
    assert invoker is not None
    ed.add_link(preproc_id, 0, invoker["id"], invoker_slot, "IMAGE")
    print(f"  added subgraph IMAGE input 'reference_video' "
          f"(sg slot {sg_input_slot}, invoker slot {invoker_slot})")

    slicer_id, guide_id = _splice_subgraph(
        ed, sg_input_slot,
        frame_idx=guide_frame_idx,
        guide_strength=guide_strength,
        latent_downscale=latent_downscale,
        ref_start_index=ref_start_index,
        ref_num_frames=ref_num_frames,
    )
    print(f"  added subgraph GetImageRangeFromBatch({slicer_id}) "
          f"+ LTXAddVideoICLoRAGuide({guide_id}); rewired #1519 consumers")

    if ref_mode == "sliding":
        calc_id = _splice_sliding_calculator(ed, slicer_id, ref_fps)
        print(f"  added SimpleCalculatorKJ({calc_id}) "
              f"[expr='round(a * {ref_fps})']; rewired "
              f"GetImageRangeFromBatch.start_index widget -> wired INT")

    ed.save(output_path)
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Audit:    uv run --group dev python scripts/audit_workflows.py {output_path}")
    print( "  3. Open in ComfyUI; subgraph schema changed — delete-and-re-add the loop subgraph node")
    print( "  4. Render against the canonical baseline. A/B for video-ref influence on every iteration")


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--reference-video", required=False,
                    help="Path to ref-video MP4/MOV. Required unless --revert.")
    ap.add_argument("--ic-lora-file", required=False,
                    help="Path to IC-LoRA safetensors file. Required unless --revert.")
    ap.add_argument("--ic-lora-strength", type=float, default=DEFAULT_IC_LORA_STRENGTH)
    ap.add_argument("--guide-frame-idx", type=int, default=DEFAULT_GUIDE_FRAME_IDX,
                    help=f"frame_idx widget on the guide (default {DEFAULT_GUIDE_FRAME_IDX}; "
                         "must be 0 or 1 mod 8 per upstream constraint)")
    ap.add_argument("--guide-strength", type=float, default=DEFAULT_GUIDE_STRENGTH)
    ap.add_argument("--latent-downscale", type=float, default=DEFAULT_LATENT_DOWNSCALE)
    ap.add_argument("--ref-start-index", type=int, default=DEFAULT_REF_START_INDEX,
                    help=f"slicer.start_index widget (default {DEFAULT_REF_START_INDEX} = static)")
    ap.add_argument("--ref-num-frames", type=int, default=DEFAULT_REF_NUM_FRAMES,
                    help=f"slicer.num_frames widget (default {DEFAULT_REF_NUM_FRAMES})")
    ap.add_argument("--ref-mode", choices=("static", "sliding"),
                    default=DEFAULT_REF_MODE,
                    help=f"Reference-window evolution per iter. "
                         f"'static' (default): same window every iter "
                         f"(start_index widget). 'sliding': window advances "
                         f"with video_start_time (SimpleCalculatorKJ wired in).")
    ap.add_argument("--ref-fps", type=int, default=DEFAULT_REF_FPS,
                    help=f"VHS_LoadVideo.force_rate (default {DEFAULT_REF_FPS}); "
                         "single source of truth — also baked into the "
                         "sliding-mode calculator expression.")
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return 0

    if not args.reference_video or not args.ic_lora_file:
        raise SystemExit("--reference-video and --ic-lora-file are required (unless --revert)")

    if args.ref_fps <= 0:
        raise SystemExit(
            f"--ref-fps must be a positive integer (got {args.ref_fps}). "
            "If your ref-video should use its native fps, this apply script "
            "doesn't support that today — file a follow-up if needed."
        )

    _migrate(
        Path(args.input), output_path,
        Path(args.reference_video), Path(args.ic_lora_file),
        args.ic_lora_strength, args.guide_frame_idx, args.guide_strength,
        args.latent_downscale, args.ref_start_index, args.ref_num_frames,
        args.ref_mode, args.ref_fps,
        args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
