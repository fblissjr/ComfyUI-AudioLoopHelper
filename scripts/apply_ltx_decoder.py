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
"""

import argparse
from pathlib import Path

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((REPO_ROOT / "example_workflows").glob(
    "audio-loop-music-video_*.json"
))

# Schema: [horizontal_tiles, vertical_tiles, overlap, last_frame_fix,
# working_device, working_dtype]. last_frame_fix works around an LTX
# quirk on the final frame. See ComfyUI-LTXVideo/tiled_vae_decode.py.
_LTX_WIDGETS = [2, 2, 1, True, "auto", "auto"]

# Fallback widgets for --revert: tile stride (512-64)/25 = 17.92s, aligned
# with iter stride at window=19.88, overlap=2.
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

    inputs = node.get("inputs") or []
    samples_link = next((i.get("link") for i in inputs if i.get("name") == "samples"), None)
    vae_link = next((i.get("link") for i in inputs if i.get("name") == "vae"), None)

    node["inputs"] = [
        {"name": "vae", "type": "VAE", "link": vae_link},
        {"name": "latents", "type": "LATENT", "link": samples_link},
    ]

    # Input-slot order differs between the two decoders, so every top-
    # level link targeting this node needs its target_slot swapped.
    node_id = node["id"]
    for link in links:
        if not (isinstance(link, list) and len(link) >= 5 and link[3] == node_id):
            continue
        if link[4] == 0:
            link[4] = 1
        elif link[4] == 1:
            link[4] = 0

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
    """Inverse of `_swap_to_ltx` — restore VAEDecodeTiled with stride-aligned widgets."""
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
        if link[4] == 0:
            link[4] = 1
        elif link[4] == 1:
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
    """Swap every decoder node in one workflow. Returns count modified."""
    ed = WorkflowEditor(path)
    source_type = _LTX_TYPE if revert else _GENERIC_TYPE
    swap = _swap_to_generic if revert else _swap_to_ltx

    links = ed.wf.get("links", [])
    count = sum(1 for node in ed.find_nodes_by_type(source_type) if swap(node, links))

    action = "reverted" if revert else "swapped"
    if count:
        ed.save()
        print(f"  {action} {count} node(s) in {path.name}")
    else:
        target_type = _GENERIC_TYPE if revert else _LTX_TYPE
        print(f"  {path.name}: no {source_type} nodes ({target_type} already present)")
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
