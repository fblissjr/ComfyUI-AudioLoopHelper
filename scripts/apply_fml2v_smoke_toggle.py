"""apply_fml2v_smoke_toggle.

Last updated: 2026-05-18

Stages a smoke-test variant of the fml2v audio-loop workflow that's
configured for the Test 1 (minimal-iter + IterPatchInspector ON)
described in the build session log. Does NOT mutate
``example_workflows/`` in place — output lands in ``internal/scratch/``
(gitignored) so the canonical workflow stays audit-clean (and the F5
``iterations_autowired`` invariant stays satisfied on the canonical).

Symptom / motivation: before the first live render of the fml2v build,
need to empirically answer whether the SetNode/GetNode bus carries
the per-iter dependency edge — i.e. does ComfyUI-NativeLooping's
``_WhileLoopClose._explore_dependencies`` re-clone the model patch
chain per iter, or does it freeze at iter 0?

Root cause / what this variant tests: the canonical build leaves the
``IterPatchInspector`` bypassed (mode=4) and wires
``TensorLoopOpen.iterations_in`` from ``AudioLoopPlanner.total_iterations``
(audit-required canonical wiring; full song length). For a smoke test
that question can be answered in 2 iterations.

Fix / change applied (idempotent):
  1. Toggle the ``IterPatchInspector`` from ``mode=4`` to ``mode=0``
     so per-call patch state lands in the console log on every iter.
  2. Strip the wire from ``TensorLoopOpen.iterations_in`` and set the
     ``iterations`` widget value to 2. With the input unwired, the
     widget value takes effect at runtime.

Compatibility with other apply scripts:
  - Operates only on the OUTPUT (staging) copy; does NOT mutate the
    canonical workflow, so F5 ``iterations_autowired`` audit stays
    green on the canonical.
  - The staging output is in ``internal/scratch/`` which the audit
    sweep does NOT cover by default, so the unwired TLO doesn't trip
    the F5 ERR on the staged variant either.

Usage:
    uv run --group dev python scripts/apply_fml2v_smoke_toggle.py
    uv run --group dev python scripts/apply_fml2v_smoke_toggle.py --revert
    uv run --group dev python scripts/apply_fml2v_smoke_toggle.py --dry-run

Idempotent on the OUTPUT path. ``--revert`` deletes the staging file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

DEFAULT_INPUT = "example_workflows/experimental/fml2v_var_d_audio_loop.json"
DEFAULT_OUTPUT = "internal/scratch/fml2v_var_d_audio_loop_smoke.json"
SMOKE_ITERATIONS = 2


def _phase5_ids(ed: WorkflowEditor) -> dict:
    """Pull the Phase 5 ID stash that build_fml2v_audio_loop.py wrote."""
    stash = ed.wf.get("properties", {}).get("build_fml2v_phase5", {})
    if not stash:
        raise SystemExit(
            "build_fml2v_phase5 stash missing — workflow wasn't built by "
            "scripts/build_fml2v_audio_loop.py. Run the builder first."
        )
    return stash


def _already_toggled(ed: WorkflowEditor) -> bool:
    """True iff the smoke-test mutations are already applied."""
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
    inspector_on = inspector.get("mode", 0) == 0
    iter_input = next((i for i in tlo.get("inputs", [])
                       if i.get("name") == "iterations_in"), None)
    iter_unwired = iter_input is not None and iter_input.get("link") is None
    widget_short = (tlo.get("widgets_values") or [None, None])[1] == SMOKE_ITERATIONS
    return inspector_on and iter_unwired and widget_short


def _apply(ed: WorkflowEditor) -> None:
    stash = _phase5_ids(ed)
    inspector_id = stash["iter_patch_inspector"]
    tlo_id = stash["tlo"]

    inspector = ed.find_node(inspector_id)
    inspector["mode"] = 0
    print(f"  IterPatchInspector #{inspector_id}: mode 4 -> 0 (active)")

    tlo = ed.find_node(tlo_id)
    iter_slot = WorkflowEditor.find_input_slot(tlo, "iterations_in")
    existing = ed.find_link_to_slot(tlo_id, iter_slot)
    if existing is not None:
        ed.remove_link(existing[0])
        print(f"  TensorLoopOpen #{tlo_id}.iterations_in: unwired (was link {existing[0]})")
    wv = tlo.get("widgets_values") or []
    while len(wv) < 2:
        wv.append(0)
    old_iters = wv[1]
    wv[1] = SMOKE_ITERATIONS
    tlo["widgets_values"] = wv
    print(f"  TensorLoopOpen #{tlo_id}.widgets[1] (iterations): {old_iters} -> {SMOKE_ITERATIONS}")


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input workflow missing: {input_path}")

    if output_path.exists() and input_path != output_path:
        ed_existing = WorkflowEditor(output_path)
        if _already_toggled(ed_existing):
            print(f"{output_path.name}: already toggled, skipping. Run --revert to reset.")
            return

    if dry_run:
        ed = WorkflowEditor(input_path)
        print(f"would copy {input_path} -> {output_path}")
        print("would apply smoke-toggle ops:")
        _apply(ed)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    print(f"  copied {input_path} -> {output_path}")

    ed = WorkflowEditor(output_path)
    if _already_toggled(ed):
        print(f"{output_path.name}: already toggled (from input), skipping mutations.")
        return

    _apply(ed)
    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. bash start_experiment.sh default   # sets RUN_ID + SAGE_TRACE + EXEC_LOG")
    print(f"  2. Load {output_path.name} in ComfyUI")
    print(f"  3. Wire LoadAudio #2307 to a ~20s audio clip + LoadImage #45 to init image")
    print(f"  4. Queue prompt — IterPatchInspector logs per-call patch state to console")
    print(f"  5. Post-render: uv run --group dev python scripts/exec_log_summary.py \\")
    print(f"         data/runs/$RUN_ID/<prompt_id>/exec.jsonl")
    print(f"     Look for LoopIterationStamp/ChunkFFN/NAG/Sage call counts == {SMOKE_ITERATIONS}")
    print(f"     (patches re-fire per iter) or == 1 (patches frozen at iter 0).")


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be copied/changed without writing.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return

    _migrate(Path(args.input), output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
