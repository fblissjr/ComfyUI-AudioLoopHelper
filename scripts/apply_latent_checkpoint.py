"""apply_latent_checkpoint.

Last updated: 2026-06-06

Moves the loop family's decode-crash latent banking from the standalone
always-on ``SaveLatent`` node onto ``PreDecodeCleanup``'s checkpoint
widgets (``checkpoint_keep`` / ``checkpoint_prefix``).

Why: the standalone SaveLatent (active by default since the decode-OOM
cascade) writes a GB-scale ``.latent`` into a fresh per-render timestamped
folder on EVERY render and nothing ever deletes them. PreDecodeCleanup's
checkpointing writes the same core-SaveLatent-compatible file (prompt
metadata included — the recovery workflow and crash forensics both read
it) to a STABLE per-workflow prefix and rotates, keeping only the newest
``checkpoint_keep`` files.

Per eligible workflow (active TensorLoopOpen + active PreDecodeCleanup):
  1. Stamp PreDecodeCleanup widgets ``[mode, KEEP, latents/checkpoints/<stem>]``
     (mode preserved; KEEP=2 = current render + one prior).
  2. Remove every SaveLatent node (active or bypassed) and its links —
     superseded by the checkpoint widgets. RunIdPrefix keeps feeding
     VHS_VideoCombine; its latent_prefix output simply goes unconsumed.

``--revert`` restores the standalone SHAPE, not bytes: checkpoint widgets
back to the no-op defaults ``[mode, 0, latents/checkpoints/audio_loop]``
and an ACTIVE SaveLatent re-added, wired from PreDecodeCleanup's latent
source + RunIdPrefix.latent_prefix when present — the re-added node gets a
fresh id and position, so a revert->apply round-trip is functionally (not
byte-) identical to the pre-revert state.

Audit pair: ``latent_checkpoint`` (WARN) in ``scripts/audit_workflows.py``.

Usage:
    uv run --group dev python scripts/apply_latent_checkpoint.py
    uv run --group dev python scripts/apply_latent_checkpoint.py --dry-run
    uv run --group dev python scripts/apply_latent_checkpoint.py --revert

Idempotent in both directions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor, has_active_tensor_loop, is_active  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGET_GLOBS = (
    "example_workflows/*.json",
    "example_workflows/experimental/*.json",
)

CLEANUP_TYPE = "PreDecodeCleanup"
# Keep the current render's checkpoint plus one prior — enough history to
# recover from a decode death discovered a render late, without unbounded
# accumulation.
KEEP = 2
DEFAULT_PREFIX = "latents/checkpoints/audio_loop"


def _checkpoint_widgets(node: dict, keep: int, prefix: str) -> list:
    """Full widget array for PreDecodeCleanup, preserving the mode value."""
    widgets = node.get("widgets_values") or []
    mode = widgets[0] if widgets else "always"
    return [mode, keep, prefix]


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    if not has_active_tensor_loop(ed):
        return "skip (no active TensorLoop — single-pass/short decode, deliberately exempt)"
    cleanups = [n for n in ed.find_nodes_by_type(CLEANUP_TYPE) if is_active(n)]
    if not cleanups:
        return "skip (no active PreDecodeCleanup — run apply_pre_decode_cleanup.py first)"
    cleanup = cleanups[0]
    save_latents = ed.find_nodes_by_type("SaveLatent")

    if revert:
        widgets = cleanup.get("widgets_values") or []
        is_reverted_state = (
            len(widgets) >= 2 and widgets[1] == 0 and save_latents
        )
        if is_reverted_state:
            return "already reverted"
        if dry_run:
            return "would revert (checkpoint widgets -> no-op; re-add SaveLatent)"
        cleanup["widgets_values"] = _checkpoint_widgets(cleanup, 0, DEFAULT_PREFIX)
        if not save_latents:
            src_link = ed.find_link_to_slot(cleanup["id"], 0)
            if src_link is None:
                ed.save()
                return "reverted widgets only (cleanup latent input unlinked; no SaveLatent re-added)"
            _, src_id, src_slot, *_ = src_link
            cx, cy = cleanup.get("pos", [0, 0])
            sl_id = ed.add_top_level_node(
                node_type="SaveLatent",
                pos=[cx, cy - 160], size=[320, 80],
                inputs=[
                    WorkflowEditor.io_in("samples", "LATENT"),
                    {
                        "name": "filename_prefix",
                        "type": "STRING",
                        "widget": {"name": "filename_prefix"},
                    },
                ],
                outputs=[],
                widgets_values=["latents/segment"],
                title="Save assembled latent",
            )
            ed.add_link(src_id, src_slot, sl_id, 0, "LATENT")
            rips = [n for n in ed.find_nodes_by_type("RunIdPrefix") if is_active(n)]
            if rips:
                # RunIdPrefix outputs: [video_prefix(0), latent_prefix(1)]
                ed.add_link(rips[0]["id"], 1, sl_id, 1, "STRING")
        ed.save()
        return "reverted (checkpoint off; standalone SaveLatent restored)"

    prefix = f"latents/checkpoints/{wf_path.stem}"
    target = _checkpoint_widgets(cleanup, KEEP, prefix)
    no_widget_change = cleanup.get("widgets_values") == target
    if no_widget_change and not save_latents:
        return "no change (checkpoint already configured, no SaveLatent)"
    if dry_run:
        parts = []
        if not no_widget_change:
            parts.append(f"stamp {CLEANUP_TYPE} #{cleanup['id']} widgets {target}")
        if save_latents:
            parts.append(f"remove SaveLatent {[n['id'] for n in save_latents]}")
        return "would update (" + "; ".join(parts) + ")"
    cleanup["widgets_values"] = target
    for sl in save_latents:
        ed.remove_node_and_links(sl["id"])
    ed.prune_orphan_output_links()
    ed.save()
    removed = f", removed SaveLatent x{len(save_latents)}" if save_latents else ""
    return f"updated (checkpoint keep={KEEP} prefix={prefix}{removed})"


def apply(revert: bool, dry_run: bool) -> int:
    action = ("Would " if dry_run else "") + ("revert" if revert else "apply").capitalize()
    print(f"{action} latent checkpoint rotation across loop workflows...")
    fail = 0
    for pattern in TARGET_GLOBS:
        for wf_path in sorted(REPO_ROOT.glob(pattern)):
            status = _apply_one(wf_path, revert, dry_run)
            print(f"  {wf_path.relative_to(REPO_ROOT)}: {status}")
            if status.startswith("load error"):
                fail += 1
    # Family exit-code contract: 1 on any failure, 0 otherwise.
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
