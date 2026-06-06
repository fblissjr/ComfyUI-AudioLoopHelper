"""Swap decoder nodes between the three acceptable configurations.

Three modes:
  (default)        VAEDecodeTiled → LTXVTiledVAEDecode (spatial-only; no
                   temporal tiling, no stride invariant — but MONOLITHIC:
                   decode RAM scales linearly with song length and >=4-min
                   songs kernel-OOM a 128GB box; see
                   docs/reference/benchmarking_memory_pressure.md).
  --revert         LTXVTiledVAEDecode → VAEDecodeTiled with the legacy
                   stride-aligned widgets.
  --spatiotemporal LTXVTiledVAEDecode → LTXVSpatioTemporalTiledVAEDecode
                   with PER-WORKFLOW stride-derived widgets: temporal chunk
                   stride (in latent frames) == the loop iteration stride,
                   so chunk seams land on iteration boundaries; single
                   spatial tile; cpu/float16 accumulator. Bounds decode RAM
                   to ~chunk size + one fp16 full-video accumulator
                   (~4.4GB per minute at 960x544) — removes the song-length
                   ceiling. Audit pair: scripts/validate_workflow_decoder.py.

Idempotent in every direction.

Usage:
    uv run python scripts/apply_ltx_decoder.py                    # → LTXV spatial-only
    uv run python scripts/apply_ltx_decoder.py --revert           # → core VAEDecodeTiled
    uv run python scripts/apply_ltx_decoder.py --spatiotemporal   # → LTX spatio-temporal
"""

import argparse
from pathlib import Path

from workflow_utils import WorkflowEditor
from validate_workflow_decoder import _FPS, _get_window_and_overlap, _loop_family
from nodes import LTX_TEMPORAL_SCALE, _compute_loop_geometry

REPO_ROOT = Path(__file__).resolve().parent.parent

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

_ST_TYPE = "LTXVSpatioTemporalTiledVAEDecode"


def _spatiotemporal_widgets(window_seconds: float, overlap_seconds: float,
                            fps: int = _FPS) -> list:
    """Widgets for LTXVSpatioTemporalTiledVAEDecode aligned to the loop.

    [spatial_tiles, spatial_overlap, temporal_tile_length, temporal_overlap,
     last_frame_fix, working_device, working_dtype] — temporal units are
    LATENT frames. Derives via nodes._compute_loop_geometry (the
    controller's own stride math) so chunk stride (tile_length - overlap)
    is BIT-EXACT with the iteration stride and decode-chunk seams coincide
    with iteration boundaries (where window transitions already exist).
    The chunk overlap mirrors the render overlap, capped at the node's
    max of 8 while preserving the stride. Single spatial tile (no spatial
    seams at <=1024px width); cpu/float16 bounds the full-video
    accumulator (the decode-buffer-stack OOM class —
    docs/reference/benchmarking_memory_pressure.md).
    """
    g = _compute_loop_geometry(window_seconds, overlap_seconds, fps)
    temporal_overlap = min(8, max(1, g.overlap_latent_frames))
    return [1, 1, g.new_latent_frames + temporal_overlap, temporal_overlap,
            True, "cpu", "float16"]


def _swap_ltx_to_spatiotemporal(node: dict, widgets: list) -> bool:
    """Mutate LTXVTiledVAEDecode → LTXVSpatioTemporalTiledVAEDecode.

    Both nodes share input order (vae, latents) and the `image` output, so
    no link surgery is needed — type, widgets, and properties only. An
    already-swapped node gets its widgets re-derived (heals stale stamps
    when the workflow's window/overlap changed after the first swap).
    """
    if node.get("type") == _ST_TYPE:
        if node.get("widgets_values") == widgets:
            return False
        node["widgets_values"] = list(widgets)
        return True
    if node.get("type") != _LTX_TYPE:
        raise ValueError(
            f"Expected node type {_LTX_TYPE} or {_ST_TYPE}, got {node.get('type')!r}"
        )
    node["type"] = _ST_TYPE
    node["widgets_values"] = list(widgets)
    props = node.setdefault("properties", {})
    props["cnr_id"] = _LTX_CNR
    props["Node name for S&R"] = _ST_TYPE
    return True


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
    """Inverse of `_swap_to_ltx` — restore VAEDecodeTiled with stride-aligned
    widgets. Accepts either LTXV decoder type (both share vae/latents input
    order)."""
    if node.get("type") == _GENERIC_TYPE:
        return False
    if node.get("type") not in (_LTX_TYPE, _ST_TYPE):
        raise ValueError(
            f"Expected node type {_LTX_TYPE}, {_ST_TYPE}, or {_GENERIC_TYPE}, "
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


def patch_workflow(path: Path, revert: bool = False,
                   spatiotemporal: bool = False) -> int:
    """Swap every decoder node in one workflow. Returns count modified."""
    ed = WorkflowEditor(path)
    links = ed.wf.get("links", [])

    if spatiotemporal:
        wo = _get_window_and_overlap(ed)
        if wo is None:
            print(f"  {path.name}: skip (no AudioLoopController stride to align to)")
            return 0
        widgets = _spatiotemporal_widgets(*wo)
        targets = ed.find_nodes_by_type(_LTX_TYPE) + ed.find_nodes_by_type(_ST_TYPE)
        count = sum(1 for node in targets
                    if _swap_ltx_to_spatiotemporal(node, widgets))
        if count:
            ed.save()
            stride_s = (widgets[2] - widgets[3]) * LTX_TEMPORAL_SCALE / _FPS
            print(f"  swapped/re-stamped {count} node(s) in {path.name} "
                  f"(stride {stride_s:.2f}s -> temporal [{widgets[2]}, {widgets[3]}] latents)")
        else:
            print(f"  {path.name}: already aligned" if targets
                  else f"  {path.name}: no LTXV decoder nodes")
        return count

    sources = (_LTX_TYPE, _ST_TYPE) if revert else (_GENERIC_TYPE,)
    swap = _swap_to_generic if revert else _swap_to_ltx
    count = sum(1 for t in sources for node in ed.find_nodes_by_type(t)
                if swap(node, links))

    action = "reverted" if revert else "swapped"
    if count:
        ed.save()
        print(f"  {action} {count} node(s) in {path.name}")
    else:
        target_type = _GENERIC_TYPE if revert else _LTX_TYPE
        print(f"  {path.name}: no {'/'.join(sources)} nodes ({target_type} already present)")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument(
        "--revert", action="store_true",
        help=f"Swap {_LTX_TYPE} back to {_GENERIC_TYPE} with aligned widgets",
    )
    parser.add_argument(
        "--spatiotemporal", action="store_true",
        help=(f"Swap {_LTX_TYPE} to {_ST_TYPE} with per-workflow stride-aligned "
              "temporal chunking (removes the song-length decode ceiling)"),
    )
    args = parser.parse_args()
    if args.revert and args.spatiotemporal:
        parser.error("--revert and --spatiotemporal are mutually exclusive")

    # Full loop family (anything with an active TensorLoop) — the historical
    # audio-loop-music-video_* glob missed audio_reactive_loop + experimental
    # variants, which share the same decode tail and OOM exposure.
    workflows = _loop_family()
    if not workflows:
        print(f"No workflows found under {REPO_ROOT / 'example_workflows'}")
        return

    if args.spatiotemporal:
        direction = f"{_LTX_TYPE} → {_ST_TYPE} (stride-aligned temporal chunks)"
    elif args.revert:
        direction = f"{_LTX_TYPE} → {_GENERIC_TYPE}"
    else:
        direction = f"{_GENERIC_TYPE} → {_LTX_TYPE}"
    print(f"Applying: {direction}")
    total = 0
    for path in workflows:
        total += patch_workflow(path, revert=args.revert,
                                spatiotemporal=args.spatiotemporal)
    print(f"Total nodes modified: {total}")


if __name__ == "__main__":
    main()
