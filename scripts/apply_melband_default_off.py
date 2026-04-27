"""Disable MelBand vocal separation by default across all music-video workflows.

Two coordinated edits:
  1. Set `mode=4` (bypassed) on `MelBandRoFormerModelLoader` and
     `MelBandRoFormerSampler`.
  2. Rewire `Set_actual_audio` to take its AUDIO input directly from
     `TrimAudioDuration`, replacing the previous link through the
     sampler's `vocals` output.

Explicit rewiring (not relying on ComfyUI's bypass-passthrough) makes
the wiring obvious in the graph — anyone reading the workflow sees
`TrimAudioDuration -> Set_actual_audio` directly, no need to understand
bypass-slot type mapping.

Re-enabling separation is a manual two-step: flip either MelBand node
back to `mode=0` AND re-route `Set_actual_audio`'s input through the
sampler's `vocals` output. Users who want vocal separation are doing
it deliberately and can handle the rewire.

Idempotent. Runs against every `example_workflows/audio-loop-music-video_*.json`
by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor, is_active


BYPASS_TYPES = ("MelBandRoFormerModelLoader", "MelBandRoFormerSampler")

# Canonical node IDs in the family of music-video workflows. All six
# variants were forked from the same base so these are stable.
TRIM_AUDIO_ID = 567                 # TrimAudioDuration output 0 = AUDIO
SET_ACTUAL_AUDIO_ID = 640           # Set_actual_audio input 0 = AUDIO
MELBAND_SAMPLER_ID = 569            # MelBandRoFormerSampler output 0 = vocals


def _bypass_melband_nodes(ed: WorkflowEditor, path: Path) -> bool:
    changed = False
    for node_type in BYPASS_TYPES:
        for node in ed.find_nodes_by_type(node_type):
            if is_active(node):
                node["mode"] = 4
                print(f"  {path.name}: {node_type}(id={node['id']}) -> bypassed")
                changed = True
    return changed


def _rewire_actual_audio_direct(ed: WorkflowEditor, path: Path) -> bool:
    """Ensure Set_actual_audio pulls from TrimAudioDuration, not the sampler."""
    if ed.require_nodes((SET_ACTUAL_AUDIO_ID, TRIM_AUDIO_ID)):
        # Layout doesn't match this workflow family — skip silently.
        return False

    existing = ed.find_link_to_slot(SET_ACTUAL_AUDIO_ID, 0)
    if existing and existing[1] == TRIM_AUDIO_ID and existing[2] == 0:
        return False  # already direct

    if existing:
        print(f"  {path.name}: drop stale link {existing[0]} "
              f"(src={existing[1]} -> Set_actual_audio)")

    ed.rewire_input(SET_ACTUAL_AUDIO_ID, 0, TRIM_AUDIO_ID, 0, "AUDIO")
    print(f"  {path.name}: wire TrimAudioDuration({TRIM_AUDIO_ID}) "
          f"-> Set_actual_audio({SET_ACTUAL_AUDIO_ID}) directly")
    return True


def apply(path: Path) -> bool:
    ed = WorkflowEditor(path)
    changed = _bypass_melband_nodes(ed, path)
    changed = _rewire_actual_audio_direct(ed, path) or changed
    if changed:
        ed.save()
    else:
        print(f"  {path.name}: already bypassed and rewired, skipping.")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "workflows", nargs="*",
        help="Workflow JSON paths. Default: all audio-loop-music-video_*.json under example_workflows/",
    )
    args = ap.parse_args()
    paths = [Path(p) for p in args.workflows] if args.workflows else sorted(
        Path("example_workflows").glob("audio-loop-music-video_*.json")
    )
    for p in paths:
        apply(p)


if __name__ == "__main__":
    main()
