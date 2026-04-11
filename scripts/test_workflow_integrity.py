"""Validate workflow JSON structural integrity and widget consistency.

Catches bugs like:
- Widget values count mismatch (e.g., VAEDecodeTiled got 3 values, needs 4)
- Dangling links (referenced in node but missing from links array, or vice versa)
- Subgraph link/linkIds desync
- Missing nodes referenced by links

Run: python scripts/test_workflow_integrity.py example_workflows/audio-loop-music-video_image.json
"""

import ast
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Widget-count extraction from Python source (static, no imports needed)
# ---------------------------------------------------------------------------

def _extract_input_types_widget_count(source: str, class_name: str) -> int | None:
    """Count widget inputs (non-connected) from INPUT_TYPES in Python source.

    Widgets are inputs whose type is a plain tuple like ("INT", {...}) or
    ("FLOAT", {...}) -- as opposed to connected inputs which are just type
    strings like ("LATENT",) with no default/min/max dict.

    Returns None if we can't parse it.
    """
    # This is a best-effort heuristic -- not a full Python parser
    # Look for the class and count widget-like entries in required/optional
    return None  # see _build_known_widgets below


def _build_known_widgets() -> dict[str, int]:
    """Hardcoded widget counts for nodes we use in our workflows.

    Extracted from reading the actual INPUT_TYPES / define_schema of each node.
    This is the source of truth for validation -- add entries as needed.
    """
    return {
        # Core ComfyUI nodes
        "VAEDecodeTiled": 4,        # tile_size, overlap, temporal_size, temporal_overlap
        "VAEEncodeTiled": 4,        # tile_size, overlap, temporal_size, temporal_overlap
        "VAEEncode": 0,
        "VAEDecode": 0,
        "CLIPTextEncode": 1,        # text
        "KSamplerSelect": 1,        # sampler_name
        "RandomNoise": 2,           # noise_seed, mode (when widget, not linked)
        "SamplerCustomAdvanced": 0,
        "CFGGuider": 1,             # cfg
        "ManualSigmas": 1,          # sigmas (string)
        "BasicScheduler": 3,        # scheduler, steps, denoise
        "EmptyLTXVLatentVideo": 4,  # width, height, length, batch_size
        "ConditioningZeroOut": 0,
        "ImageBatch": 0,

        # LTX nodes (comfy_extras)
        "LTXVConditioning": 1,      # frame_rate
        "LTXVAddGuide": 2,          # frame_idx, strength
        "LTXVConcatAVLatent": 0,
        "LTXVSeparateAVLatent": 0,
        "LTXVCropGuides": 0,
        "LTXVPreprocess": 1,        # img_compression
        "LatentUpscaleModelLoader": 1,  # model_name
        "LTXVLatentUpsampler": 0,

        # KJNodes
        "SetNode": 1,               # variable name
        "GetNode": 1,               # variable name
        "FloatConstant": 1,         # value
        "LTX2_NAG": 4,              # nag_scale, alpha, tau, inplace
        "ImageResizeKJv2": 8,       # width, height, method, crop, offset, interp, oversample, device

        # ComfyUI-LTXVideo
        "LTXVAddLatentGuide": 2,    # latent_idx, strength
        "LTXVAudioVideoMask": 7,    # video_fps, video_start, video_end, audio_start, audio_end, max_length, existing_mask_mode

        # Note: DynamicCombo nodes (LTXVImgToVideoInplaceKJ, LTXVAddGuideMulti)
        # have variable widget counts depending on num_images selection.
        # These are excluded from strict validation.
    }


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def validate_links(wf: dict) -> list[str]:
    """Check top-level link consistency."""
    errors = []
    node_ids = {n["id"] for n in wf["nodes"]}
    link_ids_in_array = set()

    for l in wf.get("links", []):
        if not isinstance(l, list) or len(l) < 6:
            errors.append(f"Malformed link: {l}")
            continue

        lid, src_id, src_slot, tgt_id, tgt_slot, dtype = l[0], l[1], l[2], l[3], l[4], l[5]
        link_ids_in_array.add(lid)

        if src_id not in node_ids:
            errors.append(f"Link {lid}: source node {src_id} not found")
        if tgt_id not in node_ids:
            errors.append(f"Link {lid}: target node {tgt_id} not found")

    # Check node inputs reference valid links
    for n in wf["nodes"]:
        for i, inp in enumerate(n.get("inputs", [])):
            link_id = inp.get("link")
            if link_id is not None and link_id not in link_ids_in_array:
                errors.append(f"Node {n['id']} input[{i}] '{inp.get('name','')}' references link {link_id} not in links array")

        for i, out in enumerate(n.get("outputs", [])):
            for link_id in (out.get("links") or []):
                if link_id not in link_ids_in_array:
                    errors.append(f"Node {n['id']} output[{i}] '{out.get('name','')}' references link {link_id} not in links array")

    return errors


def validate_subgraph_links(wf: dict) -> list[str]:
    """Check subgraph internal link consistency."""
    errors = []
    defs = wf.get("definitions", {})
    if not isinstance(defs, dict):
        return errors

    for sg in defs.get("subgraphs", []):
        sg_name = sg.get("name", sg.get("id", "?"))
        sg_node_ids = {n["id"] for n in sg.get("nodes", [])}
        sg_node_ids.add(-10)  # input distributor
        sg_node_ids.add(-20)  # output collector
        sg_link_ids = set()

        for l in sg.get("links", []):
            lid = l.get("id")
            sg_link_ids.add(lid)
            if l.get("origin_id") not in sg_node_ids:
                errors.append(f"Subgraph '{sg_name}' link {lid}: origin {l.get('origin_id')} not found")
            if l.get("target_id") not in sg_node_ids:
                errors.append(f"Subgraph '{sg_name}' link {lid}: target {l.get('target_id')} not found")

        # Check linkIds on inputs
        for inp in sg.get("inputs", []):
            for lid in inp.get("linkIds", []):
                if lid not in sg_link_ids:
                    errors.append(f"Subgraph '{sg_name}' input '{inp.get('name','')}' linkId {lid} not in internal links")

        # Check node input link references
        for n in sg.get("nodes", []):
            for i, inp in enumerate(n.get("inputs", [])):
                link_id = inp.get("link")
                if link_id is not None and link_id not in sg_link_ids:
                    errors.append(f"Subgraph '{sg_name}' node {n['id']} input[{i}] references link {link_id} not in internal links")

    return errors


def validate_widgets(wf: dict) -> list[str]:
    """Check widget_values count against known node definitions."""
    errors = []
    known = _build_known_widgets()

    for n in wf.get("nodes", []):
        ntype = n.get("type", "")
        if ntype not in known:
            continue

        expected = known[ntype]
        widgets = n.get("widgets_values", [])
        actual = len(widgets)

        # Widget values can include linked-widget overrides, so actual >= expected is OK.
        # But actual < expected means missing values.
        if actual < expected:
            errors.append(
                f"Node {n['id']} ({ntype}): expected >= {expected} widget values, got {actual}. "
                f"Values: {widgets}"
            )

    # Also check subgraph internal nodes
    defs = wf.get("definitions", {})
    if isinstance(defs, dict):
        for sg in defs.get("subgraphs", []):
            for n in sg.get("nodes", []):
                ntype = n.get("type", "")
                if ntype not in known:
                    continue
                expected = known[ntype]
                widgets = n.get("widgets_values", [])
                if len(widgets) < expected:
                    errors.append(
                        f"Subgraph node {n['id']} ({ntype}): expected >= {expected} widget values, got {len(widgets)}. "
                        f"Values: {widgets}"
                    )

    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(path: str) -> list[str]:
    wf = json.loads(Path(path).read_text())
    errors = []
    errors.extend(validate_links(wf))
    errors.extend(validate_subgraph_links(wf))
    errors.extend(validate_widgets(wf))
    return errors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_workflow_integrity.py <workflow.json> [workflow2.json ...]")
        sys.exit(1)

    total_errors = 0
    for path in sys.argv[1:]:
        print(f"\n=== {path} ===")
        errors = validate(path)
        if errors:
            for e in errors:
                print(f"  FAIL: {e}")
            total_errors += len(errors)
        else:
            print("  OK: all checks passed")

    if total_errors:
        print(f"\n{total_errors} error(s) found")
        sys.exit(1)
    else:
        print(f"\nAll workflows valid")
        sys.exit(0)
