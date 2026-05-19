"""apply_fml2v_smoke_toggle.

Last updated: 2026-05-19

Stages a smoke-test variant: short iteration count for fast diagnostic
renders + optional `IterPatchInspector` activation. Does NOT mutate
`example_workflows/`; output lands in `internal/scratch/` (gitignored)
so the canonical stays audit-clean.

Tests whether ComfyUI-NativeLooping's `_WhileLoopClose._explore_dependencies`
re-clones the model patch chain per iter, or freezes at iter 0. Two
iterations is enough to answer the question.

Mutations (idempotent):
  1. Strip the wire from `TensorLoopOpen.iterations_in` + set the
     iteration widget to 2.
  2. *(opt-in via `--with-inspector`)* Toggle `IterPatchInspector`
     mode 4 → 0 so per-call patch state lands in the console log.
     Off by default — once per-iter patch survival is verified, the
     inspector's per-call CUDA sync just slows down subsequent runs.

Usage:
    uv run --group dev python scripts/apply_fml2v_smoke_toggle.py
    uv run --group dev python scripts/apply_fml2v_smoke_toggle.py --revert
    uv run --group dev python scripts/apply_fml2v_smoke_toggle.py --dry-run
    uv run --group dev python scripts/apply_fml2v_smoke_toggle.py --with-inspector
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers._fml2v_helpers import (
    SMOKE_ITERATIONS,
    apply_smoke_iters_config,
    make_argparser,
    phase5_stash,
    revert_variant,
    smoke_iters_applied,
    stage_variant,
)
from workflow_utils import WorkflowEditor

DEFAULT_OUTPUT = "internal/scratch/fml2v_var_d_audio_loop_smoke.json"


def _already_toggled(ed: WorkflowEditor, *, with_inspector: bool) -> bool:
    stash = ed.wf.get("properties", {}).get("build_fml2v_phase5") or {}
    inspector_id = stash.get("iter_patch_inspector")
    tlo_id = stash.get("tlo")
    if inspector_id is None or tlo_id is None:
        return False
    try:
        inspector = ed.find_node(inspector_id)
        tlo = ed.find_node(tlo_id)
    except ValueError:
        return False
    expected_inspector_mode = 0 if with_inspector else 4
    inspector_matches = inspector.get("mode", 4) == expected_inspector_mode
    return smoke_iters_applied(tlo) and inspector_matches


def _apply(ed: WorkflowEditor, *, with_inspector: bool) -> None:
    stash = phase5_stash(ed)
    inspector = ed.find_node(stash["iter_patch_inspector"])
    inspector["mode"] = 0 if with_inspector else 4
    label = "mode 4 -> 0 (active; opt-in diagnostic)" if with_inspector else "mode=4 (bypassed; performance default)"
    print(f"  IterPatchInspector #{inspector['id']}: {label}")
    apply_smoke_iters_config(ed, stash["tlo"])


def main() -> None:
    ap = make_argparser(__doc__, DEFAULT_OUTPUT)
    ap.add_argument("--with-inspector", action="store_true",
                    help="Activate IterPatchInspector (opt-in diagnostic). "
                         "Default is bypassed — once per-iter patch survival "
                         "has been verified, the per-call CUDA sync overhead "
                         "just slows down subsequent smoke runs.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        revert_variant(output_path)
        return

    stage_variant(
        Path(args.input), output_path,
        apply_fn=lambda ed: _apply(ed, with_inspector=args.with_inspector),
        already_toggled_fn=lambda ed: _already_toggled(ed, with_inspector=args.with_inspector),
        dry_run=args.dry_run,
        variant_label="smoke-toggle",
        next_steps=[
            "1. bash start_experiment.sh default   # sets RUN_ID + SAGE_TRACE + EXEC_LOG",
            f"2. Load {output_path.name} in ComfyUI",
            "3. Wire LoadAudio #2307 to a ~20s audio clip + LoadImage #45 to init image",
            "4. Queue prompt — IterPatchInspector logs per-call patch state to console",
            "5. Post-render: uv run --group dev python scripts/exec_log_summary.py \\",
            "       data/runs/$RUN_ID/<prompt_id>/exec.jsonl",
            f"   Look for LoopIterationStamp/ChunkFFN/NAG/Sage call counts == {SMOKE_ITERATIONS}",
            "   (patches re-fire per iter) or == 1 (patches frozen at iter 0).",
        ],
    )


if __name__ == "__main__":
    main()
