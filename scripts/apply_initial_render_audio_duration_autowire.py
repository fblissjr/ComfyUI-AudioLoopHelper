"""apply_initial_render_audio_duration_autowire.

Last updated: 2026-05-04

Wires `LTXFramePlanner.actual_seconds` → `#601 TrimAudioDuration.duration`,
replacing a hardcoded `duration=10` widget that under-shoots the initial
render's actual video length (19.88s default).

Symptom: lip sync is correct for the first ~10 seconds of generation,
then drifts. After 10s of generated video, the model has run out of
audio context and is essentially extrapolating mouth movement against
silence.

Root cause: the initial render encodes its audio context via
`#601 TrimAudioDuration → #566 LTXVAudioVAEEncode`. `#601`'s `duration`
widget defaults to 10. The initial render's video covers
`EmptyLTXVLatentVideo.length / fps` seconds (497 / 25 = 19.88s by
default), so the audio context is half the video length.

Historical context: pre-FramePlanner workflows had `#688 FloatConstant
(window_size_seconds)` wired to BOTH `AudioLoopController` AND
`#601.duration`. The FramePlanner consolidation
(`apply_frame_planner_consolidation.py`) migrated the AudioLoopController
side but left `#601.duration` orphaned at the static value 10. This
script re-establishes the wiring against the new SSoT.

Fix:
  - Top-level link: LTXFramePlanner.actual_seconds (output slot 3, FLOAT)
    → #601 TrimAudioDuration.duration

Behavior post-fix:
  - The widget value (10.0) remains in JSON but link supersedes at
    runtime — ComfyUI uses the wire when present.
  - LTXFramePlanner.actual_seconds is the snap-corrected target_seconds
    (matches EmptyLTXVLatentVideo.length / fps), so #601's audio context
    always matches the initial render's video length.

Targets:
  - example_workflows/audio-loop-music-video_latent_iclora_audio_pre_encode.json
  - example_workflows/audio-loop-music-video_latent_intro.json (rebuilt
    downstream by apply_intro_workflow.py from the fixed source).

Other workflows have a similar #601 with widget=10. They'll remain
broken until ported — likely via a generalized version of this script,
or until they're consolidated into the intro variant.

Usage:
    uv run --group dev python scripts/apply_initial_render_audio_duration_autowire.py
    uv run --group dev python scripts/apply_initial_render_audio_duration_autowire.py --revert
    uv run --group dev python scripts/apply_initial_render_audio_duration_autowire.py --dry-run

Idempotent. `--revert` removes the wire, leaving the widget value as
the sole duration source (the pre-fix state).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

FRAME_PLANNER_ACTUAL_SECONDS_SLOT = 3       # output slot for actual_seconds (FLOAT) — schema-stable
INITIAL_TRIM_ID = 601                       # TrimAudioDuration (initial render audio context)


def _find_frame_planner_id(ed: WorkflowEditor) -> int | None:
    """Look up LTXFramePlanner by type. Node ID varies across shipped
    workflows (1611, 1622, 1629, 1634 observed)."""
    for n in ed.wf.get("nodes") or []:
        if isinstance(n, dict) and n.get("type") == "LTXFramePlanner":
            return n["id"]
    return None

DEFAULT_TARGETS = [
    "example_workflows/audio-loop-music-video_latent.json",
    "example_workflows/audio-loop-music-video_latent_iclora.json",
    "example_workflows/audio-loop-music-video_latent_iclora_audio_pre_encode.json",
    "example_workflows/audio-loop-music-video_latent_intro.json",
    "example_workflows/audio-loop-music-video_latent_keyframe.json",
    "example_workflows/audio-loop-music-video_latent_stg.json",
    "example_workflows/audio-loop-music-video_latent_validator.json",
    "example_workflows/audio-loop-music-video_image_adain_perstep.json",
    # retake.json has no #601 (different topology); skipped
]


def _preflight(ed: WorkflowEditor) -> str | None:
    if _find_frame_planner_id(ed) is None:
        return "missing LTXFramePlanner"
    try:
        ed.find_node(INITIAL_TRIM_ID)
    except ValueError:
        return f"missing TrimAudioDuration #{INITIAL_TRIM_ID}"
    return None


def _existing_link(ed: WorkflowEditor) -> list | None:
    """Return the existing FramePlanner→#601.duration link if present."""
    fp_id = _find_frame_planner_id(ed)
    if fp_id is None:
        return None
    try:
        trim = ed.find_node(INITIAL_TRIM_ID)
    except ValueError:
        return None
    duration_slot = next(
        (i for i, inp in enumerate(trim.get("inputs", []))
         if inp.get("name") == "duration"),
        None,
    )
    if duration_slot is None:
        return None
    link = ed.find_link_to_slot(INITIAL_TRIM_ID, duration_slot)
    if link is not None and link[1] == fp_id \
            and link[2] == FRAME_PLANNER_ACTUAL_SECONDS_SLOT:
        return link
    return None


def _apply(ed: WorkflowEditor) -> tuple[bool, str]:
    if _existing_link(ed) is not None:
        return False, "already wired (FramePlanner.actual_seconds → #601.duration)"

    fp_id = _find_frame_planner_id(ed)
    assert fp_id is not None  # preflight guarantees this
    trim = ed.find_node(INITIAL_TRIM_ID)
    inputs = trim.setdefault("inputs", [])

    # Promote the `duration` widget to a wired input. ComfyUI represents
    # a wired-widget input as a regular input entry that carries a
    # `widget.name` ref alongside the `link` — see how AudioLoopController's
    # `window_seconds` is structured (widget + link). Without the inputs[]
    # entry, add_link silently records a link in the global array but
    # the target node won't actually receive it at runtime.
    if not any(i.get("name") == "duration" for i in inputs):
        inputs.append({
            "name": "duration",
            "type": "FLOAT",
            "widget": {"name": "duration"},
            "link": None,
        })
    duration_slot = next(
        i for i, inp in enumerate(inputs) if inp.get("name") == "duration"
    )

    # Drop any stale link record on this slot before re-adding (defensive
    # against re-runs that left a half-applied state).
    existing = ed.find_link_to_slot(INITIAL_TRIM_ID, duration_slot)
    if existing is not None:
        ed.remove_link(existing[0])

    ed.add_link(
        fp_id, FRAME_PLANNER_ACTUAL_SECONDS_SLOT,
        INITIAL_TRIM_ID, duration_slot,
        "FLOAT",
    )
    return True, (f"wired #{fp_id}.actual_seconds → "
                  f"#{INITIAL_TRIM_ID}.duration (was widget-only)")


def _revert(ed: WorkflowEditor) -> tuple[bool, str]:
    trim = ed.find_node(INITIAL_TRIM_ID)
    inputs = trim.get("inputs", [])

    # Find the duration link (regardless of source) and remove it.
    duration_slot = next(
        (i for i, inp in enumerate(inputs) if inp.get("name") == "duration"),
        None,
    )
    if duration_slot is None:
        return False, "not applied (no 'duration' input on #601)"

    link = ed.find_link_to_slot(INITIAL_TRIM_ID, duration_slot)
    if link is not None:
        ed.remove_link(link[0])

    # Drop the input entry so the widget reverts to its standalone form.
    inputs.pop(duration_slot)
    return True, (f"removed FramePlanner.actual_seconds → "
                  f"#{INITIAL_TRIM_ID}.duration wire")


def _process(target: Path, *, revert: bool, dry_run: bool) -> None:
    if not target.exists():
        print(f"  skip (missing): {target}")
        return

    ed = WorkflowEditor(target)
    err = _preflight(ed)
    if err and not revert:
        print(f"  skip (preflight): {target.name}: {err}")
        return

    if dry_run:
        already = _existing_link(ed) is not None
        if revert:
            print(f"  {target.name}: would revert "
                  f"({'applied' if already else 'not applied'})")
        else:
            print(f"  {target.name}: would apply "
                  f"({'already applied' if already else 'pending'})")
        return

    op = _revert if revert else _apply
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
        print("  1. Audit: uv run --group dev python scripts/audit_workflows.py "
              "example_workflows/audio-loop-music-video_latent_iclora_audio_pre_encode.json")
        print("  2. Rebuild intro: uv run --group dev python scripts/apply_intro_workflow.py "
              "--revert && uv run --group dev python scripts/apply_intro_workflow.py")


if __name__ == "__main__":
    main()
