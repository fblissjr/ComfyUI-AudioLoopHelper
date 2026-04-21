"""Swap KJNodes `VAELoaderKJ` → comfy core `VAELoader` for both VAEs.

ComfyUI core commit `ad94d472` ("Make the ltx audio vae more native")
refactored `comfy.ldm.lightricks.vae.audio_vae.AudioVAE` so that
`__init__` takes only `metadata` — weights load via the standard
`VAE` wrapper in `comfy/sd.py`. ComfyUI-KJNodes `VAELoaderKJ` still
calls `AudioVAE(sd, metadata)` (nodes/nodes.py:2455) and crashes with
`TypeError: AudioVAE.__init__() takes 2 positional arguments but 3
were given` when loading an LTX audio VAE.

This script swaps both VAE loaders (audio node 1538 and video node
1537) to core `VAELoader`. The audio swap is the crash fix; the
video swap is consistency + future-proofing (`VAELoaderKJ` adds zero
value over core `VAELoader` for LTX 2.3 — `device` override is
moot because core's audio branch sets `disable_offload=True`, and
`weight_dtype` override is moot because `working_dtypes=[float32]`
is hard-coded). One less KJNodes dependency in the VAE path.

Idempotent in both directions. Run without args to apply; pass
--revert to restore the KJNodes loaders.

Usage:
    uv run python scripts/apply_audio_vae_fix.py            # apply
    uv run python scripts/apply_audio_vae_fix.py --revert   # restore
"""

import argparse
from pathlib import Path

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = sorted((REPO_ROOT / "example_workflows").glob(
    "audio-loop-music-video_*.json"
))

# 1537 = video VAE, 1538 = audio VAE. Both loaded via VAELoaderKJ
# historically; both moving to core VAELoader post `ad94d472`.
_VAE_NODE_IDS: tuple[int, ...] = (1537, 1538)

_KJ_TYPE = "VAELoaderKJ"
_CORE_TYPE = "VAELoader"

_KJ_CNR = "comfyui-kjnodes"
_CORE_CNR = "comfy-core"

# KJNodes widgets: [vae_name, device, weight_dtype]. Core VAELoader
# widgets: [vae_name]. Device is always main_device and dtype is
# detected from the state_dict in core, so dropping the two extra
# widgets is a pure simplification.


def _swap_to_core(node: dict) -> bool:
    if node.get("type") == _CORE_TYPE:
        return False
    if node.get("type") != _KJ_TYPE:
        raise ValueError(
            f"Expected node type {_KJ_TYPE} or {_CORE_TYPE}, "
            f"got {node.get('type')!r}"
        )
    widgets = node.get("widgets_values") or []
    vae_name = widgets[0] if widgets else ""
    node["type"] = _CORE_TYPE
    node["widgets_values"] = [vae_name]
    props = node.setdefault("properties", {})
    props["cnr_id"] = _CORE_CNR
    props["Node name for S&R"] = _CORE_TYPE
    return True


def _swap_to_kj(node: dict) -> bool:
    if node.get("type") == _KJ_TYPE:
        return False
    if node.get("type") != _CORE_TYPE:
        raise ValueError(
            f"Expected node type {_CORE_TYPE} or {_KJ_TYPE}, "
            f"got {node.get('type')!r}"
        )
    widgets = node.get("widgets_values") or []
    vae_name = widgets[0] if widgets else ""
    node["type"] = _KJ_TYPE
    node["widgets_values"] = [vae_name, "main_device", "bf16"]
    props = node.setdefault("properties", {})
    props["cnr_id"] = _KJ_CNR
    props["Node name for S&R"] = _KJ_TYPE
    return True


def patch_workflow(path: Path, revert: bool = False) -> int:
    ed = WorkflowEditor(path)
    swap = _swap_to_kj if revert else _swap_to_core
    target_type = _KJ_TYPE if revert else _CORE_TYPE
    changed = 0
    for node_id in _VAE_NODE_IDS:
        node = next(
            (n for n in ed.wf["nodes"] if n.get("id") == node_id),
            None,
        )
        if node is None:
            print(f"  {path.name}: no node id={node_id}; skipping")
            continue
        if swap(node):
            action = "reverted" if revert else "swapped"
            print(f"  {action} node {node_id} in {path.name}")
            changed += 1
        else:
            print(f"  {path.name}: node {node_id} already {target_type}")
    if changed:
        ed.save()
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--revert", action="store_true",
        help=f"Swap {_CORE_TYPE} back to {_KJ_TYPE}",
    )
    args = parser.parse_args()

    if not WORKFLOWS:
        print(f"No workflows found under {REPO_ROOT / 'example_workflows'}")
        return

    direction = (
        f"{_CORE_TYPE} → {_KJ_TYPE}" if args.revert
        else f"{_KJ_TYPE} → {_CORE_TYPE}"
    )
    print(f"Applying: {direction}")
    total = sum(patch_workflow(p, revert=args.revert) for p in WORKFLOWS)
    print(f"Total nodes modified: {total}")


if __name__ == "__main__":
    main()
