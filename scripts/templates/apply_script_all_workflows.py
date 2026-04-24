"""<SCRIPT_NAME>.

Last updated: <YYYY-MM-DD>

Symptom it fixes: <SYMPTOM>

Root cause: <ROOT_CAUSE>

Fix: <FIX>

Compatibility with other apply scripts:
  - <COMPATIBILITY_NOTES — reference apply_sage_mode, apply_iclora_initial_render,
     apply_melband_default_off, apply_vae_and_cleanup, apply_loop_guide_preprocess_symmetry
     as relevant>

Usage:
    uv run --group dev python scripts/apply_<NAME>.py
    uv run --group dev python scripts/apply_<NAME>.py --revert
    uv run --group dev python scripts/apply_<NAME>.py --dry-run

Idempotent. Run repeatedly; already-fixed workflows report "no change".
`--dry-run` reports what WOULD change without touching files — pair with
`scripts/audit_workflows.py` to verify a hypothetical state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

# TODO: define the node IDs this script reads/writes. Keep a one-line
# comment per ID naming the node type and its role.
NODE_ID_A = 0   # <NodeTypeA> -- <role>
NODE_ID_B = 0   # <NodeTypeB> -- <role>


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    missing = ed.require_nodes((NODE_ID_A, NODE_ID_B))
    if missing:
        return f"skip (missing nodes {missing})"

    # TODO: detect current state. Return one of:
    #   - "no change (already ...)" if already in target state
    #   - "already reverted" if revert and already reverted
    #   - "skip (unexpected ...)" if layout differs from what this script expects
    # When a real change would happen: if dry_run, return "would update (...)"
    # WITHOUT calling ed.save(); otherwise save and return "updated (...)".
    # Example for a rewire:
    #
    #   row = ed.find_link_to_slot(NODE_ID_B, 0)
    #   if row is None:
    #       return f"skip (node {NODE_ID_B}.in[0] has no link)"
    #   _, src_node, src_slot, *_ = row
    #   expected_src = NODE_ID_B_OLD if revert else NODE_ID_A
    #   target_src   = NODE_ID_A     if revert else NODE_ID_B_OLD
    #   if src_node == target_src and src_slot == 0:
    #       return "already reverted" if revert else "no change (already symmetric)"
    #   if src_node != expected_src or src_slot != 0:
    #       return f"skip (unexpected inbound source {src_node}/{src_slot})"
    #
    #   verb = "would revert" if dry_run and revert else \
    #          "would update" if dry_run else \
    #          "reverted"      if revert else "updated"
    #   if not dry_run:
    #       ed.rewire_input(NODE_ID_B, 0, target_src, 0, "<DTYPE>")
    #       ed.save(wf_path)
    #   return f"{verb} (<describe the change>)"

    raise NotImplementedError("fill in _apply_one body")


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} <SCRIPT_NAME> across example_workflows/...")
    fail = 0
    for wf_path in sorted(WORKFLOWS_DIR.glob("*.json")):
        status = _apply_one(wf_path, revert, dry_run)
        print(f"  {wf_path.name}: {status}")
        if status.startswith("load error"):
            fail += 1
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--revert", action="store_true",
        help="Undo the change applied by this script.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report what WOULD change without writing files.",
    )
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
