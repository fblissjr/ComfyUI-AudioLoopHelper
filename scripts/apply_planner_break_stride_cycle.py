"""apply_planner_break_stride_cycle.

Last updated: 2026-04-27

Symptom it fixes: ComfyUI rejects the workflow with:
    Dependency cycle detected:
      AudioLoopController -> TensorLoopOpen -> AudioLoopPlanner
        -> AudioLoopController

Root cause: `AudioLoopController.stride_seconds -> AudioLoopPlanner.stride_seconds`
combined with the 2026-04-26 auto-wire
(`AudioLoopPlanner.total_iterations -> TensorLoopOpen.iterations_in`) and
the existing `TensorLoopOpen.current_iteration ->
AudioLoopController.current_iteration` closes a back-edge.

Fix (paired with a schema change in nodes.py:1060): AudioLoopPlanner now
takes `(audio, window_seconds, overlap_seconds, fps)` and computes stride
internally via `_compute_loop_geometry` — the same formula the controller
uses, so total_iterations stays consistent with the loop.

This script migrates persisted workflows to match the new schema:

  Before:                              After:
    inputs:                              inputs:
      audio (link)                         audio (link)
      stride_seconds (link from ALC)       window_seconds (link)
      window_seconds (link)                overlap_seconds (widget)
                                           fps (widget)
    widgets_values:                      widgets_values:
      [stride_seconds, window_seconds]     [window_seconds, overlap_seconds, fps]

The migration:
  1. Drops the inbound stride_seconds link (severs the cycle).
  2. Replaces the planner's `inputs[]` schema with the new shape.
  3. Pulls overlap_seconds/fps default values from the companion
     AudioLoopController node's widgets (single source of truth) so the
     planner shows numbers consistent with the controller. Falls back to
     schema defaults (2.0, 25) if no controller is found.

Compatibility with other apply scripts:
  - Strict superset of `apply_iterations_autowire.py`: that script wired
    Planner.total_iterations into TensorLoopOpen.iterations_in, which is
    what created the cycle. This script does NOT undo that wire — it
    severs the OTHER edge (controller stride into planner). Iterations
    stay auto-tracked.
  - Independent of F2/F3, sage, melband, alc_seed_rename, and the strip
    script. Touches only the planner node.

Usage:
    uv run --group dev python scripts/apply_planner_break_stride_cycle.py
    uv run --group dev python scripts/apply_planner_break_stride_cycle.py --revert
    uv run --group dev python scripts/apply_planner_break_stride_cycle.py --dry-run

Idempotent. `--revert` restores the pre-fix shape (mainly for migration
round-trip testing — production should never need it because the new
shape is the only one nodes.py supports).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

NEW_INPUT_SHAPE = [
    {"name": "audio", "type": "AUDIO"},
    {"name": "window_seconds", "type": "FLOAT", "widget": {"name": "window_seconds"}},
    {"name": "overlap_seconds", "type": "FLOAT", "widget": {"name": "overlap_seconds"}},
    {"name": "fps", "type": "INT", "widget": {"name": "fps"}},
]
OLD_INPUT_SHAPE = [
    {"name": "audio", "type": "AUDIO"},
    {"name": "stride_seconds", "type": "FLOAT", "widget": {"name": "stride_seconds"}},
    {"name": "window_seconds", "type": "FLOAT", "widget": {"name": "window_seconds"}},
]


def _is_new_shape(node: dict) -> bool:
    names = [i.get("name") for i in (node.get("inputs") or [])]
    return names == [s["name"] for s in NEW_INPUT_SHAPE]


def _is_old_shape(node: dict) -> bool:
    names = [i.get("name") for i in (node.get("inputs") or [])]
    return names == [s["name"] for s in OLD_INPUT_SHAPE]


def _link_for(node: dict, input_name: str):
    """Return `inputs[].link` for the input slot named `input_name`, or
    None if absent. Avoids 4× repeated `next(...)` generator expressions."""
    return next(
        (i.get("link") for i in (node.get("inputs") or [])
         if i.get("name") == input_name),
        None,
    )


def _read_alc_overlap_fps(ed: WorkflowEditor) -> tuple[float, int]:
    """Pull (overlap_seconds, fps) from a companion AudioLoopController so
    the planner displays values consistent with the loop. Falls back to
    schema defaults if no ALC is present."""
    alcs = ed.find_nodes_by_type("AudioLoopController")
    if not alcs:
        return (2.0, 25)
    # Schema widgets order (post-strip): current_iteration, window_seconds,
    # overlap_seconds, base_seed, fps -> indices [2] and [4].
    wv = alcs[0].get("widgets_values") or []
    if len(wv) >= 5:
        try:
            return (float(wv[2]), int(wv[4]))
        except (TypeError, ValueError):
            pass
    return (2.0, 25)


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    planners = ed.find_nodes_by_type("AudioLoopPlanner")
    if not planners:
        return "skip (no AudioLoopPlanner)"

    actions: list[str] = []
    for planner in planners:
        nid = planner.get("id")
        if revert:
            if _is_old_shape(planner):
                actions.append(f"#{nid} already reverted")
                continue
            if not _is_new_shape(planner):
                actions.append(f"#{nid} skip (unexpected input shape)")
                continue
            wv = list(planner.get("widgets_values") or [])
            if len(wv) >= 1:
                window_seconds = wv[0]
            else:
                window_seconds = 19.88
            if not dry_run:
                planner["inputs"] = [
                    {"name": "audio", "type": "AUDIO",
                     "link": _link_for(planner, "audio")},
                    {"name": "stride_seconds", "type": "FLOAT",
                     "widget": {"name": "stride_seconds"}, "link": None},
                    {"name": "window_seconds", "type": "FLOAT",
                     "widget": {"name": "window_seconds"},
                     "link": _link_for(planner, "window_seconds")},
                ]
                planner["widgets_values"] = [18.88, window_seconds]
            actions.append(f"#{nid} reverted to (stride_seconds, window_seconds)")
            continue

        # Forward direction
        if _is_new_shape(planner):
            actions.append(f"#{nid} no change")
            continue
        if not _is_old_shape(planner):
            actions.append(f"#{nid} skip (unexpected input shape)")
            continue

        # Find and drop the inbound stride_seconds link
        stride_link_id = _link_for(planner, "stride_seconds")
        if not any(i.get("name") == "stride_seconds" for i in (planner.get("inputs") or [])):
            actions.append(f"#{nid} skip (no stride_seconds input found)")
            continue

        audio_link = _link_for(planner, "audio")
        window_link = _link_for(planner, "window_seconds")

        # Read prior widgets to preserve window_seconds (slot 1 in old shape)
        old_wv = planner.get("widgets_values") or []
        old_window = old_wv[1] if len(old_wv) >= 2 else 19.88

        overlap_default, fps_default = _read_alc_overlap_fps(ed)

        if not dry_run:
            if stride_link_id is not None:
                ed.remove_link(stride_link_id)

            planner["inputs"] = [
                {"name": "audio", "type": "AUDIO", "link": audio_link},
                {"name": "window_seconds", "type": "FLOAT",
                 "widget": {"name": "window_seconds"}, "link": window_link},
                {"name": "overlap_seconds", "type": "FLOAT",
                 "widget": {"name": "overlap_seconds"}, "link": None},
                {"name": "fps", "type": "INT",
                 "widget": {"name": "fps"}, "link": None},
            ]
            planner["widgets_values"] = [old_window, overlap_default, fps_default]

        verb = "would update" if dry_run else "updated"
        actions.append(
            f"#{nid} {verb} "
            f"(dropped stride_seconds link, set overlap={overlap_default}, fps={fps_default})"
        )

    if any("updated" in a or "reverted" in a for a in actions) and not dry_run:
        ed.save(wf_path)

    return ", ".join(actions)


def _iter_workflow_paths():
    yield from sorted(WORKFLOWS_DIR.glob("*.json"))
    yield from sorted((WORKFLOWS_DIR / "experimental").glob("*.json"))


def apply(revert: bool, dry_run: bool) -> int:
    if dry_run:
        action = f"Would {'revert' if revert else 'apply'}"
    else:
        action = "Reverting" if revert else "Applying"
    print(f"{action} apply_planner_break_stride_cycle across example_workflows/...")
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
                    help="Restore the pre-fix shape (stride_seconds input wired in).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
