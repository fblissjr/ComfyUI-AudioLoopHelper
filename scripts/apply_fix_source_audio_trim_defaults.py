"""apply_fix_source_audio_trim_defaults.

Last updated: 2026-05-10

Symptom this fixes: user renders a music video and the **first 5
seconds of their song are missing** from the saved mp4.

Root cause: ``TrimAudioDuration #567`` ships with widget defaults
``[start_index=5, duration=300]``. The historical reasoning was
"skip silent intros and cap at 5 minutes for music videos." Title
on the node read "Song Trim (skip intro, take N seconds)." But the
default-ON intro skip is a footgun — most users wanting their full
song don't notice the widget, and lose 5s of head every render.
Reported 2026-05-10 after a render where audio appeared cut off
post-F14-trim fix; F14 was working correctly, the audio was already
short of source by 5s when it entered the loop.

Fix: change ``[5, 300]`` -> ``[0, 600]``. Default is now "keep full
song, capped at 10 minutes." Users who genuinely want an intro skip
can set start_index themselves. The title also gets updated to
reflect the new default.

Applied across every workflow that has #567 with default-buggy
widgets. Idempotent (skips workflows where widgets already differ
from the bug shape; revert restores ``[5, 300]``).

Usage:
    uv run --group dev python scripts/apply_fix_source_audio_trim_defaults.py
    uv run --group dev python scripts/apply_fix_source_audio_trim_defaults.py --revert
    uv run --group dev python scripts/apply_fix_source_audio_trim_defaults.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCE_TRIM_NODE_ID = 567

BUGGY_WIDGETS = [5, 300]
FIXED_WIDGETS = [0, 600]
BUGGY_TITLE = "Song Trim (skip intro, take N seconds)"
FIXED_TITLE = "Song Trim (full song by default — set start_index > 0 to skip intro)"


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    if not ed.has_node(SOURCE_TRIM_NODE_ID):
        return "skip (no #567 TrimAudioDuration)"

    node = ed.find_node(SOURCE_TRIM_NODE_ID)
    if node.get("type") != "TrimAudioDuration":
        return f"skip (node #{SOURCE_TRIM_NODE_ID} is {node.get('type')!r}, not TrimAudioDuration)"

    current_widgets = list(node.get("widgets_values") or [])
    target_widgets = BUGGY_WIDGETS if revert else FIXED_WIDGETS
    expected_widgets = FIXED_WIDGETS if revert else BUGGY_WIDGETS

    if current_widgets[:2] == target_widgets:
        return "already at target" if revert else "no change (already fixed)"
    if current_widgets[:2] != expected_widgets:
        return f"skip (unexpected widgets {current_widgets[:2]}; manual edit?)"

    if dry_run:
        verb = "would revert" if revert else "would fix"
        return f"{verb} (#{SOURCE_TRIM_NODE_ID} widgets {expected_widgets} -> {target_widgets})"

    node["widgets_values"] = target_widgets + current_widgets[2:]
    # Only touch the title if it matches the known-buggy default —
    # respects user customization.
    cur_title = node.get("title")
    if revert and cur_title == FIXED_TITLE:
        node["title"] = BUGGY_TITLE
    elif (not revert) and cur_title == BUGGY_TITLE:
        node["title"] = FIXED_TITLE

    ed.save()
    verb = "reverted" if revert else "fixed"
    return f"{verb} (#{SOURCE_TRIM_NODE_ID} widgets -> {target_widgets})"


def _iter_workflows() -> list[Path]:
    paths: list[Path] = []
    for d in (REPO_ROOT / "example_workflows", REPO_ROOT / "internal" / "workflows"):
        if not d.exists():
            continue
        paths.extend(sorted(d.rglob("*.json")))
    return paths


def apply(revert: bool, dry_run: bool) -> int:
    action = ("Would " if dry_run else "") + ("revert" if revert else "apply").capitalize()
    print(f"{action} source-audio-trim defaults fix across workflows...")
    fail = 0
    for wf_path in _iter_workflows():
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
    ap.add_argument("--revert", action="store_true", help="Restore the buggy [5, 300] defaults.")
    ap.add_argument("--dry-run", action="store_true", help="Print without writing.")
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
