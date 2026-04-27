"""Wire `skip_under_seq_len=1024` on AudioLoopHelperSageAttention nodes.

Last updated: 2026-04-27

Symptom it fixes: short-Q attention calls (q.shape[1] in [497, 498])
running on sage at ~0.45× torch_flash. Sage's int8 quant + kernel-
launch overhead dominates on short sequences; production renders
have ~1152 + 2304 = 3456 such calls per single-iteration loop pass
(observed in `data/runs/20260427T185316Z_82f0/sage.jsonl`).

Root cause: sage's int8 quant + `@triton.autotune` lookup + cuda
kernel launch ≈ 0.05ms baseline overhead. At seq=22932×22932 / d=128
the matmul work is ~1000× that; overhead is invisible. At
seq=497×497 / d=64 the matmul work is ~50µs; overhead doubles the
wall time. Sage-fork's v0.4.1 LTX bench confirmed empirically
(~0.45× torch_flash at the short-Q shapes). Documented in
`internal/SAGE_CLAUDE_TO_AUDIO_LOOP_CLAUDE_MEMO.md` (received
2026-04-27).

Fix: AudioLoopHelperSageAttention now exposes a `skip_under_seq_len`
INT widget (default 0 = disabled, current behavior). When > 0,
calls with `q.shape[1] < threshold` route directly to pytorch_fn
instead of sage_fn. Trace rows on the skip path carry
`skipped: true` + `skip_reason: "under_seq_len"` so workload-profile
tools can aggregate the policy at a glance.

This script sets the widget to 1024 across all shipped workflows.
1024 covers both [1, 497, 2048] and [1, 498, 2048] short-Q rows
plus any 1024-token cross-attn rows that show up in future traces.
Higher thresholds (e.g. 4096) are an option but skip more useful
sage calls; 1024 is the documented v0.4.1-evidence-driven default.

Compatibility:
  - Schema-additive on the AudioLoopHelperSageAttention node. Old
    saved JSONs that lack the widget value default to 0 = no
    behavior change. ComfyUI deserializes positionally; this
    script appends to widgets_values rather than inserting.
  - No interaction with apply_audioloophelper_sage.py (that script
    only handles the legacy node-type rename).
  - No interaction with apply_frame_planner_consolidation.py.

Usage:
    uv run --group dev python scripts/apply_skip_under_seq_len.py
    uv run --group dev python scripts/apply_skip_under_seq_len.py --revert
    uv run --group dev python scripts/apply_skip_under_seq_len.py --dry-run

Idempotent. `--revert` removes the appended widget value entirely
(reverts to default 0 = current behavior).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

# Schema layout (after this script applies):
#   widgets_values[0] = mode               (str, e.g. "auto_mask_aware")
#   widgets_values[1] = fallback_on_error  (bool, default true)
#   widgets_values[2] = skip_under_seq_len (int, default 0; this
#                       script sets to 1024)
EXPECTED_LEN_PRE  = 2
EXPECTED_LEN_POST = 3
SKIP_THRESHOLD    = 1024


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    sage_nodes = ed.find_nodes_by_type("AudioLoopHelperSageAttention")
    if not sage_nodes:
        return "skip (no AudioLoopHelperSageAttention node)"

    changes = []
    for n in sage_nodes:
        wv = n.get("widgets_values") or []
        if not isinstance(wv, list):
            return f"skip (#{n.get('id')} widgets_values not a list)"

        if revert:
            if len(wv) <= EXPECTED_LEN_PRE:
                continue  # already reverted
            n["widgets_values"] = wv[:EXPECTED_LEN_PRE]
            changes.append(f"#{n.get('id')} reverted (drop skip_under_seq_len={wv[EXPECTED_LEN_PRE]})")
        else:
            if len(wv) >= EXPECTED_LEN_POST and wv[EXPECTED_LEN_POST - 1] == SKIP_THRESHOLD:
                continue  # already at target
            if len(wv) < EXPECTED_LEN_PRE:
                return f"skip (#{n.get('id')} widgets_values len={len(wv)} < {EXPECTED_LEN_PRE})"
            new_wv = wv[:EXPECTED_LEN_PRE] + [SKIP_THRESHOLD]
            n["widgets_values"] = new_wv
            changes.append(f"#{n.get('id')} set skip_under_seq_len={SKIP_THRESHOLD}")

    if not changes:
        return "already reverted" if revert else f"no change (already at skip_under_seq_len={SKIP_THRESHOLD})"

    if dry_run:
        return "would " + "; ".join(changes)

    ed.save(wf_path)
    return "; ".join(changes)


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} skip_under_seq_len={SKIP_THRESHOLD} across example_workflows/...")
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
        help="Drop the skip_under_seq_len widget value (revert to schema default 0).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report what WOULD change without writing files.",
    )
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
