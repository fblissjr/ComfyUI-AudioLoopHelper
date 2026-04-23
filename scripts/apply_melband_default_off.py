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

from workflow_utils import WorkflowEditor


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
            if node.get("mode", 0) != 4:
                node["mode"] = 4
                print(f"  {path.name}: {node_type}(id={node['id']}) -> bypassed")
                changed = True
    return changed


def _rewire_actual_audio_direct(ed: WorkflowEditor, path: Path) -> bool:
    """Ensure Set_actual_audio pulls from TrimAudioDuration, not the sampler."""
    try:
        ed.find_node(SET_ACTUAL_AUDIO_ID)
        ed.find_node(TRIM_AUDIO_ID)
    except ValueError:
        # Layout doesn't match this workflow family — skip silently.
        return False

    # Drop any existing link into Set_actual_audio.
    for link in list(ed.wf["links"]):
        if isinstance(link, list) and link[3] == SET_ACTUAL_AUDIO_ID:
            if link[1] == TRIM_AUDIO_ID and link[2] == 0 and link[4] == 0:
                return False  # already direct
            print(f"  {path.name}: drop stale link {link[0]} "
                  f"(src={link[1]} -> Set_actual_audio)")
            ed.remove_link(link[0])

    ed.add_link(TRIM_AUDIO_ID, 0, SET_ACTUAL_AUDIO_ID, 0, "AUDIO")
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
