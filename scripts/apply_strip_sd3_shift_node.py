"""apply_strip_sd3_shift_node.

Last updated: 2026-04-30

Symptom it fixes: 8 of our shipped workflows include `ModelSamplingSD3
shift=13` between the model loader and the sampler, suggesting a
flow-matching shift is part of the canonical distilled chain. It is not.

Root cause: Lightricks's distilled inference applies NO shift between
sigma scheduling and denoising. Their `DISTILLED_SIGMA_VALUES` (per
`coderef/ID-LoRA/ID-LoRA-2.3/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py:13-15`)
are the FINAL sampling schedule, fed directly to `euler_denoising_loop`
in `coderef/ID-LoRA/ID-LoRA-2.3/.../distilled.py:106-112`. Their
official 2.3 distilled example workflows ship with no `ModelSampling*`
node either.

Patching the model with `ModelSamplingSD3 shift=13` applies
`t' = 13t / (1 + 12t)` to the sigma-to-timestep mapping (per
`comfy/model_sampling.py:278-281`), distorting the schedule the
distilled checkpoint was trained on.

Bonus finding: in all 8 of our workflows, `ModelSamplingSD3` is
already DEAD — its output links to nothing (`outputs[0].links == []`).
ComfyUI's executor either skips it or runs it and discards the result.
Stripping is pure cleanup; no behavior change at render time.

Fix: strip the dead/incorrect node from all shipped workflows.
Conservative match: id=1513 + type=ModelSamplingSD3 + mode=0 +
widgets_values=[13] (the canonical scaffolding shape). User-customized
shifts (different value, bypassed, different node id) are preserved.

Compatibility:
  - Independent of F2/F3/F4/F5/F6/F7/F8/F9/F10/F11/F12.
  - Does NOT affect the canonical (it never had the node).
  - Audit semantic flips: `model_sampling_shift` now WARNs when
    `ModelSamplingSD3` is PRESENT on production workflows (the inverse
    of the prior check).
  - Reference: `internal/analysis/ltx23_sigma_shift_audit.md`.

Usage:
    uv run --group dev python scripts/apply_strip_sd3_shift_node.py
    uv run --group dev python scripts/apply_strip_sd3_shift_node.py --revert
    uv run --group dev python scripts/apply_strip_sd3_shift_node.py --dry-run

Idempotent. Re-run reports "no change" without writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

# Canonical scaffolding signature for the dead/incorrect SD3 shift node.
SD3_NODE_ID = 1513
SD3_NODE_TYPE = "ModelSamplingSD3"
SD3_NODE_MODE = 0
SD3_NODE_WIDGETS = [13]
SD3_UPSTREAM_SOURCE = (503, 0)  # LTX2SamplingPreviewOverride.0 → SD3.0


def _matches_canonical_sd3(node: dict) -> bool:
    """Strict match. User customizations are preserved."""
    if node.get("id") != SD3_NODE_ID:
        return False
    if node.get("type") != SD3_NODE_TYPE:
        return False
    if node.get("mode") != SD3_NODE_MODE:
        return False
    if (node.get("widgets_values") or []) != SD3_NODE_WIDGETS:
        return False
    return True


def _classify(ed: WorkflowEditor) -> str:
    n = ed.find_nodes_by_type(SD3_NODE_TYPE)
    if not n:
        return "absent"
    canonical = [x for x in n if _matches_canonical_sd3(x)]
    if canonical and len(canonical) == len(n):
        return "canonical_present"
    if canonical and len(canonical) < len(n):
        return "mixed_canonical_and_custom"
    return "user_customized"


def _strip(ed: WorkflowEditor) -> str:
    """Remove the canonical SD3 node and its inbound link."""
    nodes = [n for n in ed.find_nodes_by_type(SD3_NODE_TYPE) if _matches_canonical_sd3(n)]
    for n in nodes:
        ed.remove_node_and_links(n["id"])
    return f"stripped {len(nodes)} ModelSamplingSD3 node(s)"


def _restore(ed: WorkflowEditor) -> str:
    """Re-create the canonical SD3 scaffolding."""
    if ed.find_nodes_by_type(SD3_NODE_TYPE):
        return "skip (ModelSamplingSD3 already present; nothing to restore)"
    src_id, src_slot = SD3_UPSTREAM_SOURCE
    if not ed.has_node(src_id):
        return f"skip (upstream source #{src_id} not present)"

    # Re-add the node with its canonical shape
    node = {
        "id": SD3_NODE_ID,
        "type": SD3_NODE_TYPE,
        "pos": [0, 0],
        "size": [315, 58],
        "flags": {},
        "order": 0,
        "mode": SD3_NODE_MODE,
        "inputs": [{"name": "model", "type": "MODEL", "link": None}],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
        "properties": {"Node name for S&R": SD3_NODE_TYPE},
        "widgets_values": list(SD3_NODE_WIDGETS),
    }
    ed.add_node(node)
    ed.add_link(src_id, src_slot, SD3_NODE_ID, 0, "MODEL")
    return "reverted (re-added ModelSamplingSD3 scaffolding)"


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    classification = _classify(ed)

    if revert:
        if classification == "canonical_present":
            return "already reverted (canonical SD3 present)"
        if classification != "absent":
            return f"skip ({classification})"
        if dry_run:
            return "would revert (re-add canonical ModelSamplingSD3)"
        status = _restore(ed)
        if not status.startswith("skip"):
            ed.save(wf_path)
        return status

    # Forward apply
    if classification == "absent":
        return "no change (no ModelSamplingSD3 present)"
    if classification == "user_customized":
        return "skip (user-customized — non-canonical id/widgets/mode)"
    # canonical_present (or mixed)
    if dry_run:
        return "would strip (canonical ModelSamplingSD3 + inbound link)"
    status = _strip(ed)
    ed.save(wf_path)
    return status


def _iter_workflow_paths(workflows_dir: Path):
    yield from sorted(workflows_dir.glob("*.json"))
    experimental = workflows_dir / "experimental"
    if experimental.is_dir():
        yield from sorted(experimental.glob("*.json"))


def apply(revert: bool, dry_run: bool, workflows_dir: Path) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} apply_strip_sd3_shift_node across {workflows_dir}/...")
    fail = 0
    for wf_path in _iter_workflow_paths(workflows_dir):
        try:
            rel = wf_path.relative_to(REPO_ROOT)
        except ValueError:
            rel = wf_path
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
    ap.add_argument("--revert", action="store_true",
                    help="Re-add the canonical ModelSamplingSD3 scaffolding.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    ap.add_argument("--workflows-dir", default=str(WORKFLOWS_DIR),
                    help="Directory of workflow JSONs to sweep (default: example_workflows/)")
    args = ap.parse_args()
    return apply(args.revert, args.dry_run, Path(args.workflows_dir))


if __name__ == "__main__":
    sys.exit(main())
