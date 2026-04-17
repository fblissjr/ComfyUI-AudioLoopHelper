"""Swap generic VAEDecodeTiled → LTX-specific LTXVTiledVAEDecode in workflows.

Eliminates the stride-alignment fragility from the widget-tuning approach:
LTXVTiledVAEDecode (from ComfyUI-LTXVideo) does spatial-only tiling with
no temporal tiling at all, so there's no decoder tile stride that must be
kept aligned with `AudioLoopController.overlap_seconds`.

Idempotent in both directions. Run without args to apply; pass --revert
to go back to the generic decoder with the aligned widget values.

Usage:
    uv run python scripts/apply_ltx_decoder.py            # apply
    uv run python scripts/apply_ltx_decoder.py --revert   # restore

Context: Phase DR1 of the decoder-reliability track. See the plan file
for full spec and rationale.
"""

import argparse
from pathlib import Path

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((REPO_ROOT / "example_workflows").glob(
    "audio-loop-music-video_*.json"
))

# LTXVTiledVAEDecode defaults — 2×2 spatial tiles, 1 latent-frame overlap,
# last_frame_fix on (works around LTX's final-frame quirk), auto
# device/dtype. See ComfyUI-LTXVideo/tiled_vae_decode.py for the widget
# schema.
_LTX_WIDGETS = [2, 2, 1, True, "auto", "auto"]

# The VAEDecodeTiled widgets we're replacing — aligned for
# overlap_seconds=2 per the prior widget-fix commit.
_GENERIC_WIDGETS = [512, 64, 512, 64]

_GENERIC_TYPE = "VAEDecodeTiled"
_LTX_TYPE = "LTXVTiledVAEDecode"

_GENERIC_CNR = "comfy-core"
_LTX_CNR = "ComfyUI-LTXVideo"


def _swap_to_ltx(node: dict, links: list) -> bool:
    """Mutate `node` in-place from VAEDecodeTiled to LTXVTiledVAEDecode.

    Also mutates `links` (the workflow's top-level links array) to update
    target_slot values for the two inbound links, since the two node types
    list their inputs in different orders:

        VAEDecodeTiled:     [samples(LATENT), vae(VAE)]
        LTXVTiledVAEDecode: [vae(VAE), latents(LATENT)]

    Returns True if a swap occurred, False if node was already LTX.
    """
    if node.get("type") == _LTX_TYPE:
        return False
    if node.get("type") != _GENERIC_TYPE:
        raise ValueError(
            f"Expected node type {_GENERIC_TYPE} or {_LTX_TYPE}, "
            f"got {node.get('type')!r}"
        )

    # Capture current input links by name. VAEDecodeTiled's inputs are
    # [samples, vae] -> slots [0, 1].
    inputs = node.get("inputs") or []
    samples_link = next((i.get("link") for i in inputs if i.get("name") == "samples"), None)
    vae_link = next((i.get("link") for i in inputs if i.get("name") == "vae"), None)

    # Reorder + rename for LTX schema: [vae, latents] -> slots [0, 1].
    node["inputs"] = [
        {"name": "vae", "type": "VAE", "link": vae_link},
        {"name": "latents", "type": "LATENT", "link": samples_link},
    ]

    # Update the top-level links array: swap target_slot for this node's
    # two inbound links. The samples input (was slot 0) becomes latents
    # at slot 1; the vae input (was slot 1) becomes slot 0.
    node_id = node["id"]
    for link in links:
        if not (isinstance(link, list) and len(link) >= 5 and link[3] == node_id):
            continue
        old_slot = link[4]
        if old_slot == 0:       # was samples -> now latents at slot 1
            link[4] = 1
        elif old_slot == 1:     # was vae -> now vae at slot 0
            link[4] = 0

    # Output stays at slot 0, just rename for consistency with LTX node.
    outputs = node.get("outputs") or []
    if outputs:
        outputs[0]["name"] = "image"

    node["type"] = _LTX_TYPE
    node["widgets_values"] = list(_LTX_WIDGETS)

    props = node.setdefault("properties", {})
    props["cnr_id"] = _LTX_CNR
    props["Node name for S&R"] = _LTX_TYPE

    return True


def _swap_to_generic(node: dict, links: list) -> bool:
    """Inverse of `_swap_to_ltx` — restore `VAEDecodeTiled` with the
    stride-aligned widget values we'd had before DR1.

    Returns True if swapped, False if already generic.
    """
    if node.get("type") == _GENERIC_TYPE:
        return False
    if node.get("type") != _LTX_TYPE:
        raise ValueError(
            f"Expected node type {_LTX_TYPE} or {_GENERIC_TYPE}, "
            f"got {node.get('type')!r}"
        )

    inputs = node.get("inputs") or []
    vae_link = next((i.get("link") for i in inputs if i.get("name") == "vae"), None)
    latents_link = next((i.get("link") for i in inputs if i.get("name") == "latents"), None)

    node["inputs"] = [
        {"name": "samples", "type": "LATENT", "link": latents_link},
        {"name": "vae", "type": "VAE", "link": vae_link},
    ]

    node_id = node["id"]
    for link in links:
        if not (isinstance(link, list) and len(link) >= 5 and link[3] == node_id):
            continue
        old_slot = link[4]
        if old_slot == 0:       # was vae at slot 0 -> now slot 1
            link[4] = 1
        elif old_slot == 1:     # was latents at slot 1 -> now samples at slot 0
            link[4] = 0

    outputs = node.get("outputs") or []
    if outputs:
        outputs[0]["name"] = "IMAGE"

    node["type"] = _GENERIC_TYPE
    node["widgets_values"] = list(_GENERIC_WIDGETS)

    props = node.setdefault("properties", {})
    props["cnr_id"] = _GENERIC_CNR
    props["Node name for S&R"] = _GENERIC_TYPE

    return True


def patch_workflow(path: Path, revert: bool = False) -> int:
    """Run the swap across every decoder node in one workflow.

    Returns the number of nodes modified.
    """
    ed = WorkflowEditor(path)
    target_type = _GENERIC_TYPE if revert else _LTX_TYPE
    source_type = _LTX_TYPE if revert else _GENERIC_TYPE
    swap = _swap_to_generic if revert else _swap_to_ltx

    count = 0
    for node in ed.wf.get("nodes", []):
        if node.get("type") == source_type:
            if swap(node, ed.wf.get("links", [])):
                count += 1

    action = "reverted" if revert else "swapped"
    already = f"{target_type} nodes already present"
    if count:
        ed.save()
        print(f"  {action} {count} node(s) in {path.name}")
    else:
        print(f"  {path.name}: no {source_type} nodes to {action} ({already})")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument(
        "--revert", action="store_true",
        help=f"Swap {_LTX_TYPE} back to {_GENERIC_TYPE} with aligned widgets",
    )
    args = parser.parse_args()

    if not WORKFLOWS:
        print(f"No workflows found under {REPO_ROOT / 'example_workflows'}")
        return

    direction = (
        f"{_LTX_TYPE} → {_GENERIC_TYPE}" if args.revert
        else f"{_GENERIC_TYPE} → {_LTX_TYPE}"
    )
    print(f"Applying: {direction}")
    total = 0
    for path in WORKFLOWS:
        total += patch_workflow(path, revert=args.revert)
    print(f"Total nodes modified: {total}")


if __name__ == "__main__":
    main()
