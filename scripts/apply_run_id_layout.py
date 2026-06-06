"""apply_run_id_layout.

Last updated: 2026-05-10

Unifies a workflow's save-node filename prefixes through a single
``RunIdPrefix`` node so every artifact of one render lands under
``<output>/<workflow_name>/<timestamp>/`` — the run is now a folder,
not a counter on the global namespace.

Also (for loop workflows) inserts a **bypassed** ``SaveLatent`` wired
to ``LatentConcat #1605`` (the assembled video latent that feeds final
VAE decode). The user clicks the node in the UI to toggle ``mode=0``
and the next render saves the latent into
``<output>/<workflow_name>/<timestamp>/latents/segment_NNNNN_.latent``,
ready to feed into ``build_upscale_workflow.py``'s ``LoadLatent``
ingress without ever materializing a pixel mp4.

Effect on the output dir:

    <output>/
      <workflow_name>/
        <timestamp>_00001.mp4              ← VHS_VideoCombine video
        <timestamp>_00001.png              ← workflow snapshot (VHS auto)
        <timestamp>_00001-audio.mp4        ← VHS_VideoCombine + audio
        latents/
          segment_NNNNN_.latent            ← (only when SaveLatent toggled on)

What it does per workflow:
  1. Insert ``RunIdPrefix`` (workflow_name default = workflow filename base).
  2. Convert ``VHS_VideoCombine.filename_prefix`` from widget-only to
     a wired input; link from ``RunIdPrefix.video_prefix``.
  3. If a ``SaveLatent`` already exists, wire its ``filename_prefix``
     from ``RunIdPrefix.latent_prefix``. (Schema already has
     ``filename_prefix`` as an input slot — just adds the link.)
  4. Otherwise, if ``LatentConcat #1605`` exists, add a bypassed
     (``mode=4``) ``SaveLatent`` wired from that LatentConcat with
     ``filename_prefix`` from ``RunIdPrefix.latent_prefix``.

Skips files in ``SKIP_FILES`` (third-party references we don't own).
Post-loop processors (upscale, seam refinement) DO get the
``RunIdPrefix`` wiring on their ``VHS_VideoCombine`` — their outputs
should cluster into per-run folders too — but skip the SaveLatent
toggle since they have no LatentConcat to wire from.

Usage:
    uv run --group dev python scripts/apply_run_id_layout.py
    uv run --group dev python scripts/apply_run_id_layout.py --revert
    uv run --group dev python scripts/apply_run_id_layout.py --dry-run

Audit pair: ``run_id_layout_present`` in ``scripts/audit_workflows.py``
(WARN-level — not all forks need this if they predate the convention).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor, iter_all_workflows

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_FILES = {
    "edit_anything_v2v_reference.json",
    "upscale_3pass_reference.json",
}

LATENT_CONCAT_NODE_ID = 1605  # canonical assembled-latent producer


def _ensure_filename_prefix_input(node: dict) -> int:
    """Make ``filename_prefix`` an input slot on the node. Returns its
    slot index. Idempotent: if already converted, returns the existing
    index without mutating. Uses ``WorkflowEditor.find_input_slot`` per
    ``scripts/CLAUDE.md`` "Don't hand-roll link lookups or rewires."""
    try:
        return WorkflowEditor.find_input_slot(node, "filename_prefix")
    except ValueError:
        node.setdefault("inputs", []).append(
            {
                "name": "filename_prefix",
                "type": "STRING",
                "widget": {"name": "filename_prefix"},
            }
        )
        return len(node["inputs"]) - 1


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    if wf_path.name in SKIP_FILES:
        return "skip (excluded by SKIP_FILES)"
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    workflow_name = wf_path.stem
    nodes_by_type: dict[str, list[dict]] = {}
    for n in ed.wf["nodes"]:
        nodes_by_type.setdefault(n.get("type"), []).append(n)

    combines = [n for n in nodes_by_type.get("VHS_VideoCombine", []) if n.get("mode", 0) == 0]
    if not combines:
        return "skip (no active VHS_VideoCombine)"

    run_id_prefix_nodes = nodes_by_type.get("RunIdPrefix", [])

    if revert:
        if not run_id_prefix_nodes:
            return "already reverted"
        rip_id = run_id_prefix_nodes[0]["id"]

        if dry_run:
            return f"would revert (remove RunIdPrefix #{rip_id} + any bypassed-SaveLatent it fed)"

        # Identify bypassed SaveLatents fed by this RunIdPrefix's latent_prefix
        # output — those are the toggles we added on apply.
        bypassed_save_latents: list[int] = []
        for L in ed.wf["links"]:
            if not isinstance(L, list):
                continue
            _, src, src_slot, tgt, _, _ = L
            if src != rip_id or src_slot != 1:
                continue
            tgt_node = ed.find_node(tgt)
            if tgt_node.get("type") == "SaveLatent" and tgt_node.get("mode", 0) == 4:
                bypassed_save_latents.append(tgt_node["id"])

        # remove_node_and_links detaches everything for each id; order doesn't matter.
        for sl_id in bypassed_save_latents:
            ed.remove_node_and_links(sl_id)
        ed.remove_node_and_links(rip_id)
        ed.save()
        return f"reverted (removed RunIdPrefix #{rip_id})"

    if run_id_prefix_nodes:
        return f"no change (RunIdPrefix #{run_id_prefix_nodes[0]['id']} already wired)"

    # Find LatentConcat (only on loop workflows)
    latent_concat = next(
        (n for n in nodes_by_type.get("LatentConcat", []) if n.get("mode", 0) == 0),
        None,
    )
    # Latent banking moved to PreDecodeCleanup's checkpoint widgets
    # (apply_latent_checkpoint.py); don't re-add a SaveLatent toggle when
    # the checkpoint is configured (widgets [mode, checkpoint_keep, prefix]).
    checkpoint_active = any(
        n.get("mode", 0) == 0
        and len(n.get("widgets_values") or []) >= 2
        and isinstance((n.get("widgets_values") or [None, 0])[1], int)
        and (n.get("widgets_values") or [None, 0])[1] > 0
        for n in nodes_by_type.get("PreDecodeCleanup", [])
    )
    if checkpoint_active:
        latent_concat = None
    existing_save_latents = [
        n for n in nodes_by_type.get("SaveLatent", []) if n.get("mode", 0) == 0
    ]

    if dry_run:
        combine_ids = [c["id"] for c in combines]
        notes: list[str] = [f"add RunIdPrefix, wire VHS_VideoCombine{combine_ids}.filename_prefix"]
        if existing_save_latents:
            notes.append(f"wire SaveLatent{[s['id'] for s in existing_save_latents]}.filename_prefix from latent_prefix")
        elif latent_concat is not None:
            notes.append(f"add bypassed SaveLatent wired from LatentConcat #{latent_concat['id']}")
        return f"would update ({'; '.join(notes)})"

    # 1. Insert RunIdPrefix
    rip_id = ed.add_top_level_node(
        node_type="RunIdPrefix",
        pos=[-2500, -200], size=[300, 100],
        inputs=[],
        outputs=[
            WorkflowEditor.out("video_prefix", "STRING"),
            WorkflowEditor.out("latent_prefix", "STRING"),
        ],
        widgets_values=[workflow_name, "%Y%m%d_%H%M%S"],
        title="Run ID Prefix",
    )

    # 2. Wire VHS_VideoCombine(s).filename_prefix from RunIdPrefix.video_prefix
    for combine in combines:
        slot_idx = _ensure_filename_prefix_input(combine)
        ed.add_link(rip_id, 0, combine["id"], slot_idx, "STRING")

    # 3 / 4. SaveLatent handling
    if existing_save_latents:
        for sl in existing_save_latents:
            slot_idx = _ensure_filename_prefix_input(sl)
            ed.add_link(rip_id, 1, sl["id"], slot_idx, "STRING")
    elif latent_concat is not None:
        # Add a bypassed SaveLatent wired to LatentConcat + RunIdPrefix.latent_prefix.
        save_latent_id = ed.add_top_level_node(
            node_type="SaveLatent",
            pos=[1700, -200], size=[320, 80],
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
            title="Save assembled latent (toggle)",
        )
        # Bypass by default — user clicks to enable.
        ed.find_node(save_latent_id)["mode"] = 4
        # Wire samples ← LatentConcat output 0
        ed.add_link(latent_concat["id"], 0, save_latent_id, 0, "LATENT")
        # Wire filename_prefix ← RunIdPrefix.latent_prefix
        ed.add_link(rip_id, 1, save_latent_id, 1, "STRING")

    ed.save()
    if existing_save_latents:
        sl_note = f"wired existing SaveLatent{[s['id'] for s in existing_save_latents]} from latent_prefix"
    elif latent_concat is not None:
        sl_note = "added bypassed SaveLatent toggle"
    else:
        sl_note = "no SaveLatent (post-loop workflow)"
    return f"updated (RunIdPrefix #{rip_id}; {sl_note})"


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} RunIdPrefix layout across workflows...")
    fail = 0
    for wf_path in iter_all_workflows():
        rel = wf_path.relative_to(REPO_ROOT)
        status = _apply_one(wf_path, revert, dry_run)
        print(f"  {rel}: {status}")
        if status.startswith("load error"):
            fail += 1
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--revert", action="store_true", help="Undo the change.")
    ap.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
