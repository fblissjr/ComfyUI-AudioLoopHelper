"""apply_canonical_sigmas.

Last updated: 2026-04-27

Replace `BasicScheduler` (linear_quadratic 8 1 approximation) with
`ManualSigmas` carrying Lightricks's canonical hand-tuned distilled sigma
values. Both nodes output SIGMAS; the swap is 1-for-1 on output.

The canonical values come from
`coderef/ID-LoRA/ID-LoRA-2.3/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py`:

    DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975,
                              0.909375, 0.725, 0.421875, 0.0]

These are the noise levels Lightricks's distilled checkpoint was trained
to denoise. `BasicScheduler linear_quadratic 8 1` produces an
APPROXIMATION of the same curve via a parametric formula; not identical,
likely close enough that prior generations worked, but the canonical
hand-tuned values are the spec.

What does NOT change:
  - `KSamplerSelect euler` — Lightricks's distilled inference uses
    plain Euler (`SimpleDenoiser` + `euler_denoising_loop` in
    coderef/LTX-Desktop/.../ltx_pipeline_common.py;
    `EulerDiffusionStep` in
    coderef/ID-LoRA-2.3/.../diffusion_steps.py). NOT
    `euler_ancestral_cfg_pp` — that's a community variant. The plateau
    at sigma~=0.99 for the first 4 steps would amplify ancestral
    re-noise enough to bleed across our TensorLoop iteration boundaries.
  - `ModelSamplingSD3 shift=13`
  - `CFGGuider cfg=1`

Both schedulers' outputs flow into the same downstream consumer
(`VisualizeSigmasKJ(1422)` -> `Set_sigmas(579)` -> the loop's sampler);
the SIGMAS type is identical, so no other re-wiring needed.

`BasicScheduler` takes `model` as an input (link). `ManualSigmas` takes
no model — its sole widget is a comma-separated string of sigma values.
The migration drops the model input wire (the source side, the
`ModelSamplingSD3(1513)` node, retains its output to other consumers if
any; otherwise its model output just has one fewer downstream).

Compatibility:
  - Independent of F2/F3/F4/F5/F6/F7. Independent of LoRA chain +
    ID-LoRA runtime apply scripts. Touches only the scheduler node.
  - Audit `EXPECTED_CHAIN["BasicScheduler"]` is replaced with a
    `ManualSigmas` entry checking the canonical sigma string.

Usage:
    uv run --group dev python scripts/apply_canonical_sigmas.py
    uv run --group dev python scripts/apply_canonical_sigmas.py --revert
    uv run --group dev python scripts/apply_canonical_sigmas.py --dry-run

Idempotent. Re-run is a no-op once converted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

SCHEDULER_NODE_ID = 1421  # BasicScheduler in every shipped workflow

CANONICAL_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LEGACY_WIDGETS = ["linear_quadratic", 8, 1]


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    node = ed.find_node(SCHEDULER_NODE_ID)
    if node is None:
        return f"skip (no node #{SCHEDULER_NODE_ID})"

    current_type = node.get("type")
    current_widgets = node.get("widgets_values")

    if revert:
        if current_type == "BasicScheduler" and current_widgets == LEGACY_WIDGETS:
            return "already reverted"
        if current_type != "ManualSigmas":
            return f"skip (unexpected current type {current_type!r})"
        # Revert: restore BasicScheduler with the legacy widgets and reattach
        # the model input. We need to find what was feeding ManualSigmas;
        # it has no model input, so we have to look at the SetNode("model")
        # broadcast to find the original source. Easier: assume the
        # ModelSamplingSD3(1513) -> BasicScheduler edge is the canonical
        # source (link 2795 in all shipped workflows).
        if not dry_run:
            node["type"] = "BasicScheduler"
            node["widgets_values"] = list(LEGACY_WIDGETS)
            # Re-add the model input (link is None — user must rewire if revert is real)
            node["inputs"] = [{"name": "model", "type": "MODEL", "link": None}]
            # Properties
            props = node.get("properties") or {}
            props["Node name for S&R"] = "BasicScheduler"
            props["cnr_id"] = "comfy-core"
            node["properties"] = props
            ed.save(wf_path)
        return "reverted (BasicScheduler restored; model input UNWIRED — rewire manually)"

    if current_type == "ManualSigmas" and current_widgets == [CANONICAL_SIGMAS]:
        return "no change (already canonical ManualSigmas)"

    if current_type != "BasicScheduler":
        return f"skip (unexpected current type {current_type!r}; expected BasicScheduler)"

    # Forward apply: drop the model input link, swap node type, replace widgets.
    model_inp = next(
        (i for i in (node.get("inputs") or []) if i.get("name") == "model"),
        None,
    )
    incoming_link_id = model_inp.get("link") if model_inp else None

    if not dry_run:
        if incoming_link_id is not None:
            ed.remove_link(incoming_link_id)
        node["type"] = "ManualSigmas"
        node["widgets_values"] = [CANONICAL_SIGMAS]
        # ManualSigmas has no inputs
        node["inputs"] = []
        # Update properties
        props = node.get("properties") or {}
        props["Node name for S&R"] = "ManualSigmas"
        props["cnr_id"] = "comfy-core"
        node["properties"] = props
        ed.save(wf_path)

    verb = "would convert" if dry_run else "converted"
    return f"{verb} (BasicScheduler -> ManualSigmas with canonical 9 values)"


def _iter_workflow_paths():
    yield from sorted(WORKFLOWS_DIR.glob("*.json"))
    yield from sorted((WORKFLOWS_DIR / "experimental").glob("*.json"))


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} apply_canonical_sigmas across example_workflows/...")
    fail = 0
    for wf_path in _iter_workflow_paths():
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
    ap.add_argument("--revert", action="store_true",
                    help="Restore BasicScheduler linear_quadratic 8 1 (model input left UNWIRED).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    args = ap.parse_args()
    return apply(args.revert, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
