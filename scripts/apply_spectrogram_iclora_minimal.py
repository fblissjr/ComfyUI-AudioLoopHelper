"""Fork the upstream IC-LoRA Union Control workflow into a minimal
spectrogram-reference test rig.

Strips the canny/pose/depth preprocessor branches (our PNG/MP4 sequence
is already the structural reference — no extraction needed) and swaps
to the MergeGreen IC-LoRA file that's already downloaded locally.

Output: `internal/scratch/spectrogram_iclora_minimal.json`.

This is the Phase 2.0 minimal test rig per
`internal/design/spectrogram_reference_design.md` §Phase 2.0 execution.

Usage:
    uv run --group dev python scripts/apply_spectrogram_iclora_minimal.py
    uv run --group dev python scripts/apply_spectrogram_iclora_minimal.py --revert

If the upstream workflow is not at the default location, set the
COMFYUI_HOME environment variable or pass `--input <path>`.

Idempotent on the output path; `--revert` deletes it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflow_utils import WorkflowEditor


UPSTREAM_REL = "custom_nodes/ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_ICLoRA_Union_Control_Distilled.json"
DEFAULT_OUTPUT = Path("internal/scratch/spectrogram_iclora_minimal.json")


def _find_upstream() -> Path:
    """Resolve the upstream ComfyUI-LTXVideo IC-LoRA workflow without
    baking a private path. COMFYUI_HOME env wins; else probe a few
    canonical install locations."""
    env = os.environ.get("COMFYUI_HOME")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env) / UPSTREAM_REL)
    # Probe: sibling of this repo (most common consumer install layout).
    repo = Path(__file__).resolve().parent.parent
    candidates.append(repo.parent.parent / UPSTREAM_REL)
    # Probe: the repo's own custom_nodes parent (when AudioLoopHelper
    # IS installed under a ComfyUI tree, which is the typical case).
    candidates.append(repo.parent / "ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_ICLoRA_Union_Control_Distilled.json")
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        "Upstream workflow not found. Set COMFYUI_HOME env to your ComfyUI "
        "install root, or pass --input <full-path>."
    )


# Preprocessor + scaffolding nodes to strip. Our PNG/MP4 spectrogram sequence
# IS the reference — we don't need canny/pose/depth extraction.
STRIP_NODE_IDS = (
    4991,  # CannyEdgePreprocessor
    4986,  # DWPreprocessor
    5061,  # VideoDepthAnythingProcess
    5060,  # LoadVideoDepthAnythingModel
    5062,  # VideoDepthAnythingOutput
    5026,  # ResizeImageMaskNode (scale shorter to 544)
    5028,  # ResizeImageMaskNode (scale to multiple of 32)
    5035,  # ResizeImageMaskNode (third one)
    5029,  # GetImageSize
    5034,  # SimpleMath+
    5066,  # LTXFloatToInt
    2004,  # LoadImage (spare unused reference slot)
    5022,  # PrimitiveString
    5019,  # PrimitiveBoolean
    5063,  # Note (preprocessor-choice documentation)
)

LOAD_VIDEO_ID = 5001
GET_VIDEO_COMPONENTS_ID = 5000
ICLORA_GUIDE_ID = 5012
ICLORA_LOADER_ID = 5011
EMPTY_LATENT_ID = 3059

LOCAL_ICLORA_FILE = "MergeGreen_IC-lora_ltx2.3.safetensors"
LOCAL_ICLORA_STRENGTH = 0.9
PLACEHOLDER_VIDEO = "REPLACE_WITH_SPECTROGRAM.mp4"
TARGET_RESOLUTION = (832, 448)
TARGET_LENGTH = 121  # (121 - 1) % 8 == 0; ~5s at 24fps


def _already_migrated(ed: WorkflowEditor) -> bool:
    try:
        loader = ed.find_node(ICLORA_LOADER_ID)
    except ValueError:
        return False
    widgets = loader.get("widgets_values", [])
    return bool(widgets) and widgets[0] == LOCAL_ICLORA_FILE


def _strip_preprocessor_branches(ed: WorkflowEditor) -> int:
    removed = 0
    for nid in STRIP_NODE_IDS:
        try:
            ed.find_node(nid)
        except ValueError:
            continue
        ed.remove_node_and_links(nid)
        removed += 1
    return removed


def _rewire_image_source(ed: WorkflowEditor) -> None:
    """After stripping preprocessors, wire GetVideoComponents.image
    directly into LTXAddVideoICLoRAGuide.image. The IC-LoRA guide
    consumes an IMAGE batch natively (multi-frame reference)."""
    guide = ed.find_node(ICLORA_GUIDE_ID)
    image_slot = WorkflowEditor.find_input_slot(guide, "image")

    existing = ed.find_link_to_slot(ICLORA_GUIDE_ID, image_slot)
    if existing is not None:
        ed.remove_link(existing[0])

    # GetVideoComponents outputs: [0]=image (batch), [1]=audio, [2]=fps.
    ed.add_link(GET_VIDEO_COMPONENTS_ID, 0, ICLORA_GUIDE_ID, image_slot, "IMAGE")


def _update_widgets(ed: WorkflowEditor) -> None:
    loader = ed.find_node(ICLORA_LOADER_ID)
    loader["widgets_values"] = [LOCAL_ICLORA_FILE, LOCAL_ICLORA_STRENGTH]

    load_video = ed.find_node(LOAD_VIDEO_ID)
    existing = load_video.get("widgets_values", [])
    if existing:
        load_video["widgets_values"] = [PLACEHOLDER_VIDEO] + list(existing[1:])
    else:
        load_video["widgets_values"] = [PLACEHOLDER_VIDEO, "image"]

    try:
        empty = ed.find_node(EMPTY_LATENT_ID)
    except ValueError:
        return
    w, h = TARGET_RESOLUTION
    wv = empty.get("widgets_values", [])
    empty["widgets_values"] = [w, h, TARGET_LENGTH] + list(wv[3:4] if len(wv) > 3 else [1])


def _build(input_path: Path, output_path: Path) -> None:
    if output_path.exists():
        existing = WorkflowEditor(output_path)
        if _already_migrated(existing):
            print(f"{output_path}: already migrated, skipping. Use --revert to reset.")
            return

    if not input_path.exists():
        raise SystemExit(f"Upstream workflow not found at {input_path}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    print(f"  copied {input_path.name} -> {output_path}")

    ed = WorkflowEditor(output_path)
    removed = _strip_preprocessor_branches(ed)
    print(f"  stripped {removed} preprocessor/scaffolding nodes")
    _rewire_image_source(ed)
    print(f"  rewired GetVideoComponents({GET_VIDEO_COMPONENTS_ID}).image -> LTXAddVideoICLoRAGuide({ICLORA_GUIDE_ID}).image")
    _update_widgets(ed)
    print(f"  LTXICLoRALoaderModelOnly -> {LOCAL_ICLORA_FILE} @ strength {LOCAL_ICLORA_STRENGTH}")
    print(f"  LoadVideo widget -> {PLACEHOLDER_VIDEO} (user edits)")
    print(f"  EmptyLTXVLatentVideo -> {TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]}x{TARGET_LENGTH}")

    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next:")
    print("  1. Run spectrogram_to_reference.py on your song with --emit-video:")
    print("     uv run --group analysis python scripts/spectrogram_to_reference.py \\")
    print("         --audio /path/to/song.wav --duration 5.0 --emit-video")
    print("     -> emits frames + spectrogram.mp4 under internal/scratch/spectrogram_runs/<ts>/")
    print()
    print(f"  2. Open {output_path} in ComfyUI.")
    print("  3. Update LoadVideo widget to point at the emitted spectrogram.mp4.")
    print("  4. (Optional) Adjust positive prompt for the visual target.")
    print("  5. Queue. Render duration ~= LoadVideo clip duration.")
    print("  6. Dub original song audio onto output via VHS_VideoCombine for A/B.")


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
    ap.add_argument("--input", default=None,
                    help="Upstream IC-LoRA workflow path (default: autodetect via COMFYUI_HOME or probe).")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="Output staging path (default: %(default)s)")
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return
    input_path = Path(args.input).expanduser() if args.input else _find_upstream()
    _build(input_path, output_path)


if __name__ == "__main__":
    main()
