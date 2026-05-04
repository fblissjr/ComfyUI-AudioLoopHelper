"""apply_audio_latent_slice_source_seconds_autowire.

Last updated: 2026-05-04

Replaces the hardcoded `source_seconds=300` widget on AudioLatentSlice
with a wire from `AudioLoopController.audio_duration`. Eliminates the
silent lip-sync drift the pre-encode workflow exhibited on songs not
exactly 300s long.

Symptom (pre-fix): pre-encode workflow + any song shorter than ~300s
shows audio drift over the loop. Drift factor = `actual_seconds / 300`;
~30% drift on a 213s song by mid-render. The output mp4 plays audio at
the wrong rate relative to video, producing visible lip-sync mismatch.

Root cause: AudioLatentSlice computes the per-iter latent slice rate as
`latent_T / source_seconds`. With `source_seconds` hardcoded to 300 but
the encoded audio shorter (TrimAudioDuration[5, 300] silently clamps to
song length when shorter), the inferred rate undershoots — every per-
iter `start_idx` slices from too-early in the song. This is the kind of
"two-widgets-must-match" footgun that should have been auto-wired from
the start.

Fix:
  - Add a `source_seconds` (FLOAT) input slot to the loop subgraph
  - Add a matching slot on the top-level subgraph invoker
  - Wire `AudioLoopController.audio_duration` (output slot 2) →
    invoker.source_seconds
  - Wire subgraph distributor (-10) at the new slot → AudioLatentSlice
    (#2012) input slot 1 (`source_seconds`)

Behavior post-fix:
  - The widget value remains in the JSON (300.0) but the link supersedes
    at runtime — ComfyUI uses the wire when present.
  - AudioLoopController.audio_duration is computed against the actual
    post-trim audio (not the widget), so it's always correct regardless
    of the user's song length.

Subgraph schema change forces a UI re-add of the loop subgraph node
per CLAUDE.md "ComfyUI gotchas → Subgraph schema changes force a UI
re-add (slot indices baked at save time)". User must delete and re-add
the loop subgraph node in ComfyUI after applying.

Targets:
  - example_workflows/audio-loop-music-video_latent_iclora_audio_pre_encode.json
  - example_workflows/audio-loop-music-video_latent_intro.json
    (rebuilt downstream by `apply_intro_workflow.py` from the fixed
    source — re-run that script after this one)

Usage:
    uv run --group dev python scripts/apply_audio_latent_slice_source_seconds_autowire.py
    uv run --group dev python scripts/apply_audio_latent_slice_source_seconds_autowire.py --revert
    uv run --group dev python scripts/apply_audio_latent_slice_source_seconds_autowire.py --dry-run

Idempotent. `--revert` removes the new input slot + links, leaving the
widget value as the sole rate source (the pre-fix state).
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Anchor IDs in the source workflow ---
AUDIO_LOOP_CONTROLLER_ID = 1582       # top-level
AUDIO_LOOP_CONTROLLER_DURATION_SLOT = 2  # output slot for audio_duration
AUDIO_LATENT_SLICE_ID = 2012          # subgraph
AUDIO_LATENT_SLICE_SOURCE_SLOT = 1    # input slot for source_seconds

NEW_SUBGRAPH_INPUT_NAME = "source_seconds"
NEW_SUBGRAPH_INPUT_LABEL = "actual audio duration (auto from controller)"

DEFAULT_TARGETS = [
    "example_workflows/audio-loop-music-video_latent_iclora_audio_pre_encode.json",
]


# --------------------------------------------------------------------------
# Pre-flight + idempotency
# --------------------------------------------------------------------------

def _preflight(ed: WorkflowEditor) -> str | None:
    if ed.find_node(AUDIO_LOOP_CONTROLLER_ID) is None:
        return f"missing top-level AudioLoopController #{AUDIO_LOOP_CONTROLLER_ID}"
    sg = ed.get_subgraph(0)
    if sg is None:
        return "input has no subgraph"
    if not any(n.get("id") == AUDIO_LATENT_SLICE_ID for n in sg.get("nodes", [])):
        return (f"missing subgraph AudioLatentSlice #{AUDIO_LATENT_SLICE_ID} — "
                "this script targets workflows produced by "
                "apply_audio_latent_pre_encode.py")
    return None


def _find_existing_input_slot(ed: WorkflowEditor) -> int | None:
    """Return the subgraph slot index for our new input if already added."""
    sg = ed.get_subgraph(0)
    assert sg is not None
    for i, inp in enumerate(sg.get("inputs", [])):
        if inp.get("name") == NEW_SUBGRAPH_INPUT_NAME:
            return i
    return None


# --------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------

def _apply(ed: WorkflowEditor) -> tuple[bool, str]:
    """Returns (mutated, message)."""
    if _find_existing_input_slot(ed) is not None:
        return False, "already applied (subgraph already has 'source_seconds' input)"

    sg = ed.get_subgraph(0)
    assert sg is not None

    # 1. Append new input slot to subgraph schema.
    sg_inputs = sg.setdefault("inputs", [])
    new_sg_slot = len(sg_inputs)
    sg_inputs.append({
        "id": str(uuid.uuid4()),
        "name": NEW_SUBGRAPH_INPUT_NAME,
        "type": "FLOAT",
        "linkIds": [],
        "localized_name": NEW_SUBGRAPH_INPUT_NAME,
        "label": NEW_SUBGRAPH_INPUT_LABEL,
        "pos": [-3015, 3850],
    })

    # 2. Append matching invoker input slot (top-level).
    invoker = ed.find_subgraph_invoker(0)
    assert invoker is not None
    invoker_inputs = invoker.setdefault("inputs", [])
    new_invoker_slot = len(invoker_inputs)
    invoker_inputs.append({
        "name": NEW_SUBGRAPH_INPUT_NAME,
        "type": "FLOAT",
        "link": None,
    })

    # 3. Top-level link: AudioLoopController.audio_duration → invoker[new_slot]
    ed.add_link(
        AUDIO_LOOP_CONTROLLER_ID,
        AUDIO_LOOP_CONTROLLER_DURATION_SLOT,
        invoker["id"],
        new_invoker_slot,
        "FLOAT",
    )

    # 4. Internal subgraph link: distributor (-10, new_slot) → AudioLatentSlice
    ed.add_subgraph_link(
        -10, new_sg_slot,
        AUDIO_LATENT_SLICE_ID, AUDIO_LATENT_SLICE_SOURCE_SLOT,
        "FLOAT",
    )

    return True, (f"added subgraph input '{NEW_SUBGRAPH_INPUT_NAME}' (sg slot {new_sg_slot}, "
                  f"invoker slot {new_invoker_slot}); wired #{AUDIO_LOOP_CONTROLLER_ID}."
                  f"audio_duration → AudioLatentSlice")


# --------------------------------------------------------------------------
# Revert
# --------------------------------------------------------------------------

def _revert(ed: WorkflowEditor) -> tuple[bool, str]:
    sg_slot = _find_existing_input_slot(ed)
    if sg_slot is None:
        return False, "not applied (no 'source_seconds' input slot found)"

    sg = ed.get_subgraph(0)
    assert sg is not None

    # `remove_subgraph_link` also clears the target node's input.link, so
    # AudioLatentSlice's source_seconds becomes properly widget-driven again
    # (no dangling-ref hazard).
    link = ed.find_subgraph_link_to_slot(
        AUDIO_LATENT_SLICE_ID, AUDIO_LATENT_SLICE_SOURCE_SLOT,
    )
    if link is not None:
        ed.remove_subgraph_link(link["id"])

    # Remove subgraph input slot
    sg_inputs = sg.get("inputs", [])
    if 0 <= sg_slot < len(sg_inputs):
        sg_inputs.pop(sg_slot)

    # Remove invoker input + its incoming top-level link
    invoker = ed.find_subgraph_invoker(0)
    assert invoker is not None
    invoker_inputs = invoker.get("inputs", [])
    target_slot = None
    for i, inp in enumerate(invoker_inputs):
        if inp.get("name") == NEW_SUBGRAPH_INPUT_NAME and inp.get("type") == "FLOAT":
            target_slot = i
            break
    if target_slot is not None:
        # Remove the link feeding it
        link = ed.find_link_to_slot(invoker["id"], target_slot)
        if link is not None:
            ed.remove_link(link[0])
        invoker_inputs.pop(target_slot)

    return True, (f"removed subgraph input '{NEW_SUBGRAPH_INPUT_NAME}' "
                  "and the AudioLoopController.audio_duration wire")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _process(target: Path, *, revert: bool, dry_run: bool) -> None:
    if not target.exists():
        print(f"  skip (missing): {target}")
        return

    ed = WorkflowEditor(target)
    err = _preflight(ed)
    if err and not revert:
        print(f"  skip (preflight): {target.name}: {err}")
        return

    op = _revert if revert else _apply
    if dry_run:
        already = _find_existing_input_slot(ed)
        if revert:
            print(f"  {target.name}: would revert "
                  f"({'applied' if already is not None else 'not applied'})")
        else:
            print(f"  {target.name}: would apply "
                  f"({'already applied' if already is not None else 'pending'})")
        return

    mutated, msg = op(ed)
    if mutated:
        ed.save()
        print(f"  {target.name}: {msg}")
    else:
        print(f"  {target.name}: {msg} (no-op)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--target", action="append", default=None,
                    help="Workflow JSON to mutate (repeatable). "
                         f"Default: {DEFAULT_TARGETS}")
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
        print("  1. Validate: uv run --group dev python scripts/audit_workflows.py "
              "example_workflows/audio-loop-music-video_latent_iclora_audio_pre_encode.json")
        print("  2. Rebuild intro: uv run --group dev python scripts/apply_intro_workflow.py "
              "--revert && uv run --group dev python scripts/apply_intro_workflow.py")
        print("  3. In ComfyUI: delete + re-add the loop subgraph node "
              "(schema slot change forces re-add).")


if __name__ == "__main__":
    main()
