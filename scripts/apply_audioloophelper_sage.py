"""Swap node 268 (PathchSageAttentionKJ) → AudioLoopHelperSageAttention
across the shipping workflows. Default mode: auto_mask_aware -- routes
masked cross-attn to fp16_triton (the one kernel that handles LTX cross-
attn cleanly per internal/design/sage_backlog.md item 2) and keeps fp8++
speed on self-attn.

Idempotent. Reversible via --revert (restores PathchSageAttentionKJ with
the sage_mode widget preserved under the original node id).

Why a node swap rather than additive wiring: KJ's node has no
mask-aware mode. Keeping both nodes active in parallel doesn't help --
only one can own the `transformer_options["optimized_attention_override"]`
slot at a time. Direct swap keeps the graph simple.

Usage:
    uv run python scripts/apply_audioloophelper_sage.py [workflow] [--all] [--revert]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from workflow_utils import WorkflowEditor  # noqa: E402

DEFAULT_WF = REPO_ROOT / "example_workflows" / "audio-loop-music-video_latent.json"
SAGE_NODE_ID = 268

KJ_TYPE = "PathchSageAttentionKJ"
OUR_TYPE = "AudioLoopHelperSageAttention"

OUR_DEFAULT_WIDGETS = ["auto_mask_aware", True]  # [mode, fallback_on_error]
KJ_DEFAULT_WIDGETS = ["sageattn_qk_int8_pv_fp8_cuda++", False]  # [sage_mode, allow_compile]


def _swap_to_ours(node: dict) -> bool:
    if node.get("type") == OUR_TYPE:
        return False
    node["type"] = OUR_TYPE
    node["widgets_values"] = list(OUR_DEFAULT_WIDGETS)
    node["properties"] = {"Node name for S&R": OUR_TYPE}
    node["mode"] = 0
    return True


def _swap_to_kj(node: dict) -> bool:
    if node.get("type") == KJ_TYPE:
        return False
    node["type"] = KJ_TYPE
    node["widgets_values"] = list(KJ_DEFAULT_WIDGETS)
    # Leave ver empty: ComfyUI treats missing/empty ver as "unknown" (no
    # version warning). Hardcoding a stale hash from the original swap
    # forward direction would trigger version-mismatch on user's actually-
    # installed KJNodes.
    node["properties"] = {
        "cnr_id": "comfyui-kjnodes",
        "Node name for S&R": KJ_TYPE,
    }
    node["mode"] = 0
    return True


def apply(wf_path: Path, revert: bool) -> bool:
    ed = WorkflowEditor(wf_path)
    node = ed.find_node(SAGE_NODE_ID)
    if node is None:
        print(f"  skip (no node {SAGE_NODE_ID}): {wf_path.name}")
        return False

    if revert:
        changed = _swap_to_kj(node)
    else:
        # Only swap if the node is currently KJ's -- don't touch other types.
        if node.get("type") not in (KJ_TYPE, OUR_TYPE):
            print(f"  skip (node {SAGE_NODE_ID} is {node.get('type')!r}, not KJ): {wf_path.name}")
            return False
        changed = _swap_to_ours(node)

    if not changed:
        print(f"  already applied: {wf_path.name}")
        return False

    ed.save()
    direction = "PathchSageAttentionKJ" if revert else "AudioLoopHelperSageAttention"
    print(f"  swapped node {SAGE_NODE_ID} -> {direction} in {wf_path.name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow",
        nargs="?",
        default=str(DEFAULT_WF),
        help="Workflow JSON to modify (default: audio-loop-music-video_latent.json).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Apply to every example_workflows/audio-loop-music-video_*.json that has node 268.",
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Swap back to PathchSageAttentionKJ.",
    )
    args = parser.parse_args()

    if args.all:
        targets = sorted((REPO_ROOT / "example_workflows").glob("audio-loop-music-video_*.json"))
    else:
        targets = [Path(args.workflow)]

    for target in targets:
        if not target.exists():
            print(f"  missing: {target}")
            continue
        apply(target, revert=args.revert)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
