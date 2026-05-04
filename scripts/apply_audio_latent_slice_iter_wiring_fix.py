"""apply_audio_latent_slice_iter_wiring_fix.

Last updated: 2026-05-04

Fixes two long-standing wiring bugs on AudioLatentSlice (subgraph node
2012). Companion to `apply_audio_latent_slice_source_seconds_autowire.py`
(which fixed `source_seconds`); this script fixes the remaining two
inputs that determined which audio window each iter receives.

Together, the three fixes mean per-iter audio finally tracks per-iter
video. Pre-encode rendering on any song other than coincidentally-300s
with default-everything was producing constant-window audio (same ~18s
of audio reused every iter) — manifested as lip-sync breaking the moment
the loop body took over from the initial render (~20s mark).

Bug 1 — `start_seconds` mis-sourced:
  - WAS: subgraph slot 16 (`video_start_time`), which is sourced from
    `AudioLoopController.overlap_seconds` (constant 1.0). Every iter's
    slice started at post-trim t=1.0 instead of advancing.
  - FIX: rewire from subgraph slot 11 (`start_index`), which IS the
    post-trim audio time where iter N's window starts. Same value
    feeds LTXVAudioVideoMask.

Bug 2 — `duration_seconds` widget-only:
  - WAS: hardcoded widget value 17.92, which doesn't match the 19.88s
    actual video window. Each iter served 1.96s less audio than video,
    plus the value didn't track FramePlanner config changes.
  - FIX: wire from subgraph slot 5 (`video_end_time`), which IS sourced
    from `LTXFramePlanner.actual_seconds`. Same value drives the video
    side and now drives the audio slice length too.

No subgraph schema change — both fixes use existing subgraph input
slots (11, 5). No UI re-add needed.

Targets:
  - example_workflows/audio-loop-music-video_latent_iclora_audio_pre_encode.json
  - example_workflows/audio-loop-music-video_latent_intro.json (rebuilt
    downstream by apply_intro_workflow.py from the fixed source).

Usage:
    uv run --group dev python scripts/apply_audio_latent_slice_iter_wiring_fix.py
    uv run --group dev python scripts/apply_audio_latent_slice_iter_wiring_fix.py --revert
    uv run --group dev python scripts/apply_audio_latent_slice_iter_wiring_fix.py --dry-run

Idempotent. `--revert` restores the buggy wiring (start_seconds←slot 16,
duration_seconds widget-only).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

AUDIO_LATENT_SLICE_ID = 2012

# AudioLatentSlice input slot indices (schema-stable)
SLICER_START_SECONDS_SLOT = 2
SLICER_DURATION_SECONDS_SLOT = 3

# Subgraph distributor (-10) output slots that we route from.
# These slot indices are workflow-specific (depend on the order inputs
# were added). On the canonical pre-encode workflow they are:
SG_SLOT_VIDEO_END_TIME = 5     # ← LTXFramePlanner.actual_seconds (= window_seconds)
SG_SLOT_START_INDEX = 11       # ← AudioLoopController.start_index (post-trim audio time)
SG_SLOT_VIDEO_START_TIME = 16  # ← AudioLoopController.overlap_seconds (BUGGY source for slicer)

DEFAULT_TARGETS = [
    "example_workflows/audio-loop-music-video_latent_iclora_audio_pre_encode.json",
]


def _preflight(ed: WorkflowEditor) -> str | None:
    sg = ed.get_subgraph(0)
    if sg is None:
        return "no subgraph"
    slicer = next(
        (n for n in sg.get("nodes", []) if n.get("id") == AUDIO_LATENT_SLICE_ID),
        None,
    )
    if slicer is None:
        return f"missing AudioLatentSlice #{AUDIO_LATENT_SLICE_ID}"

    # Verify the subgraph slots we're routing from have the expected names.
    inputs = sg.get("inputs", [])
    expected = {
        SG_SLOT_VIDEO_END_TIME: "video_end_time",
        SG_SLOT_START_INDEX: "start_index",
    }
    for slot, name in expected.items():
        if slot >= len(inputs):
            return f"subgraph has only {len(inputs)} inputs; expected slot {slot}"
        actual = inputs[slot].get("name")
        if actual != name:
            return f"subgraph slot {slot} is {actual!r}, expected {name!r}"
    return None


def _state(ed: WorkflowEditor) -> dict:
    """Snapshot the wiring state of the two slicer inputs."""
    start_link = ed.find_subgraph_link_to_slot(
        AUDIO_LATENT_SLICE_ID, SLICER_START_SECONDS_SLOT,
    )
    dur_link = ed.find_subgraph_link_to_slot(
        AUDIO_LATENT_SLICE_ID, SLICER_DURATION_SECONDS_SLOT,
    )
    return {
        "start_origin_slot": start_link.get("origin_slot") if start_link else None,
        "duration_link_id": dur_link.get("id") if dur_link else None,
    }


def _is_applied(ed: WorkflowEditor) -> bool:
    s = _state(ed)
    return (
        s["start_origin_slot"] == SG_SLOT_START_INDEX
        and s["duration_link_id"] is not None
    )


def _apply(ed: WorkflowEditor) -> tuple[bool, list[str]]:
    if _is_applied(ed):
        return False, ["already applied (start_seconds←start_index, duration_seconds wired)"]

    actions = []

    # Fix 1: rewire start_seconds source.
    start_link = ed.find_subgraph_link_to_slot(
        AUDIO_LATENT_SLICE_ID, SLICER_START_SECONDS_SLOT,
    )
    if start_link is not None and start_link.get("origin_slot") != SG_SLOT_START_INDEX:
        ed.remove_subgraph_link(start_link["id"])
        ed.add_subgraph_link(
            -10, SG_SLOT_START_INDEX,
            AUDIO_LATENT_SLICE_ID, SLICER_START_SECONDS_SLOT,
            "FLOAT",
        )
        actions.append(
            f"rewired start_seconds: -10 slot {start_link['origin_slot']} "
            f"(video_start_time/overlap_seconds=1.0) → -10 slot {SG_SLOT_START_INDEX} "
            f"(start_index)"
        )
    elif start_link is None:
        ed.add_subgraph_link(
            -10, SG_SLOT_START_INDEX,
            AUDIO_LATENT_SLICE_ID, SLICER_START_SECONDS_SLOT,
            "FLOAT",
        )
        actions.append(f"added start_seconds wire from -10 slot {SG_SLOT_START_INDEX}")

    # Fix 2: wire duration_seconds.
    if ed.find_subgraph_link_to_slot(
            AUDIO_LATENT_SLICE_ID, SLICER_DURATION_SECONDS_SLOT) is None:
        ed.add_subgraph_link(
            -10, SG_SLOT_VIDEO_END_TIME,
            AUDIO_LATENT_SLICE_ID, SLICER_DURATION_SECONDS_SLOT,
            "FLOAT",
        )
        actions.append(
            f"wired duration_seconds: -10 slot {SG_SLOT_VIDEO_END_TIME} "
            f"(video_end_time/actual_seconds)"
        )

    return bool(actions), actions or ["no-op"]


def _revert(ed: WorkflowEditor) -> tuple[bool, list[str]]:
    actions = []

    # Revert start_seconds back to the buggy source (slot 16).
    start_link = ed.find_subgraph_link_to_slot(
        AUDIO_LATENT_SLICE_ID, SLICER_START_SECONDS_SLOT,
    )
    if start_link is not None and start_link.get("origin_slot") == SG_SLOT_START_INDEX:
        ed.remove_subgraph_link(start_link["id"])
        ed.add_subgraph_link(
            -10, SG_SLOT_VIDEO_START_TIME,
            AUDIO_LATENT_SLICE_ID, SLICER_START_SECONDS_SLOT,
            "FLOAT",
        )
        actions.append(
            f"reverted start_seconds source: slot {SG_SLOT_START_INDEX} → "
            f"slot {SG_SLOT_VIDEO_START_TIME} (the original buggy wire)"
        )

    # Drop duration_seconds wire (back to widget-only).
    dur_link = ed.find_subgraph_link_to_slot(
        AUDIO_LATENT_SLICE_ID, SLICER_DURATION_SECONDS_SLOT,
    )
    if dur_link is not None:
        ed.remove_subgraph_link(dur_link["id"])
        actions.append("removed duration_seconds wire (back to widget=17.92)")

    return bool(actions), actions or ["nothing to revert"]


def _process(target: Path, *, revert: bool, dry_run: bool) -> None:
    if not target.exists():
        print(f"  skip (missing): {target}")
        return

    ed = WorkflowEditor(target)
    err = _preflight(ed)
    if err:
        print(f"  skip (preflight): {target.name}: {err}")
        return

    if dry_run:
        applied = _is_applied(ed)
        verb = "would revert" if revert else "would apply"
        marker = "applied" if applied else "not applied"
        print(f"  {target.name}: {verb} (currently {marker})")
        return

    op = _revert if revert else _apply
    mutated, actions = op(ed)
    if mutated:
        ed.save()
        for a in actions:
            print(f"  {target.name}: {a}")
    else:
        print(f"  {target.name}: {actions[0]} (no-op)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--target", action="append", default=None,
                    help=f"Workflow JSON (repeatable). Default: {DEFAULT_TARGETS}")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = [Path(t) if Path(t).is_absolute() else REPO_ROOT / t
               for t in (args.target or DEFAULT_TARGETS)]

    for t in targets:
        _process(t, revert=args.revert, dry_run=args.dry_run)

    if not args.dry_run and not args.revert:
        print()
        print("Next steps:")
        print("  1. Audit: uv run --group dev python scripts/audit_workflows.py")
        print("  2. Rebuild intro: uv run --group dev python scripts/apply_intro_workflow.py "
              "--revert && uv run --group dev python scripts/apply_intro_workflow.py")


if __name__ == "__main__":
    main()
