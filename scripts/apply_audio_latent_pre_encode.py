"""apply_audio_latent_pre_encode.

Last updated: 2026-05-04

RETIRED: the migration this script performs is baked into the canonical
`example_workflows/audio-loop-music-video_latent.json` as of the
2026-05-04 consolidation pass. The original input
(`audio-loop-music-video_latent_iclora.json`) and the staged output
(`internal/scratch/...`) are no longer in tree. Kept for reference + as
the design record for the pre-encode topology. Don't run.

Stages a workflow variant that encodes the full song's audio latent
ONCE outside the loop and slices it per-iter in latent space, replacing
the per-iter `LTXVAudioVAEEncode` + `TrimAudioDuration` subgraph chain.

Symptom / motivation: the canonical loop subgraph re-encodes the
windowed audio slice each iteration via `#598 LTXVAudioVAEEncode`
(~1.7s × 5 loop iters = ~8.5s/render). This also forces the AudioVAE
to be re-staged in VRAM between sampler runs (5-15s/render of console-
log "Model AudioVAE prepared for dynamic VRAM loading" overhead).

Root cause: the subgraph was designed assuming per-iter encode is the
only way to get a per-iter audio latent. AudioLatentSlice (shipped in
nodes_audio_latent_slice.py) lifts that constraint — slice in latent
space using empirically-inferred rate (latent.shape[T] / source_seconds).

Fix / change applied:
  TOP-LEVEL:
    + new LTXVAudioVAEEncode (full-song encode, runs ONCE)
    + new SetNode "full_audio_latent"
    Wired: existing TrimAudioDuration #567 → new encode → SetNode

  SUBGRAPH:
    + new LATENT input slot `full_audio_latent` (forces UI re-add)
    + new AudioLatentSlice node
    + new GetNode-equivalent: top-level SetNode broadcasts to subgraph
      invoker via the new LATENT input slot
    Subgraph #598 (LTXVAudioVAEEncode) + #600 (TrimAudioDuration)
    BYPASSED (mode=4) — preserves UI clarity + easy revert via this
    script's --revert. AudioLatentSlice wired into #606 LTXVAudioVideoMask's
    audio_latent input, replacing the link from #598.

CLI flags expose the two widgets that AudioLatentSlice needs:
  --source-seconds: total seconds of audio in the encoded latent. Should
    match the upstream TrimAudioDuration widget value (#567 default 300s).
  --window-seconds: per-iter window length (matches LTXFramePlanner /
    AudioLoopPlanner widget).

Compatibility with other apply scripts:
  - Independent of the IC-LoRA / sliding mode flags (no overlap on the
    audio chain). Works on canonical OR post-iclora workflows.
  - Subgraph schema changes force a UI delete-and-re-add of the loop
    subgraph node per CLAUDE.md.

Usage:
    uv run --group dev python scripts/apply_audio_latent_pre_encode.py
    uv run --group dev python scripts/apply_audio_latent_pre_encode.py --revert
    uv run --group dev python scripts/apply_audio_latent_pre_encode.py --dry-run
    uv run --group dev python scripts/apply_audio_latent_pre_encode.py \\
        --input example_workflows/audio-loop-music-video_latent_iclora.json \\
        --source-seconds 300 --window-seconds 17.92

Idempotent on the OUTPUT path. `--revert` deletes the staging file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Top-level node IDs the script keys off.
LOAD_AUDIO_ID = 565                # LoadAudio (orig_audio source)
SONG_TRIM_ID = 567                 # TrimAudioDuration (full-song trim, widget [5, 300])
INITIAL_TRIM_ID = 601              # TrimAudioDuration (initial-render 10s trim — UNTOUCHED)
INITIAL_ENCODE_ID = 566            # LTXVAudioVAEEncode (initial-render 10s encode — UNTOUCHED)
AUDIO_VAE_GETNODE_ID = 254         # GetNode "audio_vae" (top-level VAE source)

# Subgraph node IDs the script splices around.
SG_AUDIO_VAE_ENCODE_ID = 598       # LTXVAudioVAEEncode (per-iter encode — bypassed)
SG_TRIM_AUDIO_ID = 600             # TrimAudioDuration (per-iter slice — bypassed)
SG_AUDIO_VIDEO_MASK_ID = 606       # LTXVAudioVideoMask (consumes audio_latent)

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent_iclora.json"
DEFAULT_OUTPUT = "internal/scratch/audio-loop-music-video_latent_audio_pre_encode.json"
DEFAULT_SOURCE_SECONDS = 300.0     # matches #567 widget default
DEFAULT_WINDOW_SECONDS = 17.92     # matches LTXFramePlanner widget default

NEW_AUDIO_VAE_GETNODE_LABEL = "audio_vae"  # reuse existing label (it's a Get_audio_vae)
SETNODE_NAME = "full_audio_latent"


# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------

def _preflight(ed: WorkflowEditor) -> str | None:
    missing = ed.require_nodes((
        LOAD_AUDIO_ID, SONG_TRIM_ID, INITIAL_TRIM_ID, INITIAL_ENCODE_ID,
    ))
    if missing:
        return f"missing required top-level nodes: {missing}"
    sg = ed.get_subgraph(0)
    if sg is None:
        return "input workflow has no subgraph (loop body)"
    sg_node_ids = {n["id"] for n in sg.get("nodes", [])}
    sg_missing = [
        nid for nid in (SG_AUDIO_VAE_ENCODE_ID, SG_TRIM_AUDIO_ID, SG_AUDIO_VIDEO_MASK_ID)
        if nid not in sg_node_ids
    ]
    if sg_missing:
        return f"missing required subgraph nodes: {sg_missing}"
    return None


def _already_applied(ed: WorkflowEditor) -> bool:
    """Idempotency check: top-level has a SetNode named full_audio_latent
    AND subgraph has AudioLatentSlice."""
    has_setnode = any(
        n.get("type") == "SetNode" and (n.get("widgets_values") or [None])[0] == SETNODE_NAME
        for n in ed.wf["nodes"]
    )
    sg = ed.get_subgraph(0)
    if sg is None:
        return False
    has_slicer = any(n.get("type") == "AudioLatentSlice" for n in sg.get("nodes", []))
    return has_setnode and has_slicer


# --------------------------------------------------------------------------
# Top-level splice: full-song audio encode
# --------------------------------------------------------------------------

def _add_full_song_encode_chain(ed: WorkflowEditor) -> tuple[int, int]:
    """Add LTXVAudioVAEEncode + SetNode 'full_audio_latent' on the top
    level. Wires: existing #567 TrimAudioDuration → new encode →
    SetNode. Returns (encode_id, setnode_id).
    """
    encode_id = ed.add_top_level_node(
        node_type="LTXVAudioVAEEncode",
        pos=[-300, 6500],  # below the existing audio chain
        size=[202, 46],
        inputs=[
            {"name": "audio", "type": "AUDIO", "link": None},
            {"name": "audio_vae", "type": "VAE", "link": None},
        ],
        outputs=[
            {"name": "Audio Latent", "type": "LATENT", "links": []},
        ],
        widgets_values=[],
        properties={
            "cnr_id": "comfy-core",
            "Node name for S&R": "LTXVAudioVAEEncode",
        },
        title="Full-song Audio VAE Encode (pre-encode pattern)",
    )
    # Wire: #567 TrimAudioDuration (full-song output) → encode.audio
    ed.add_link(SONG_TRIM_ID, 0, encode_id, 0, "AUDIO")
    # Wire: GetNode audio_vae (#254) → encode.audio_vae
    ed.add_link(AUDIO_VAE_GETNODE_ID, 0, encode_id, 1, "VAE")

    setnode_id = ed.add_top_level_node(
        node_type="SetNode",
        pos=[0, 6500],
        size=[210, 60],
        inputs=[
            {"name": "LATENT", "type": "LATENT", "link": None},
        ],
        outputs=[
            {"name": "LATENT", "type": "LATENT", "links": None},
        ],
        widgets_values=[SETNODE_NAME],
        properties={
            "Node name for S&R": "SetNode",
            "aux_id": "kijai/ComfyUI-KJNodes",
            "previousName": SETNODE_NAME,
        },
        title=f"Set_{SETNODE_NAME}",
    )
    # Cosmetic styling on the SetNode after creation (matches existing
    # Set_* nodes in the canonical workflow).
    setnode = ed.find_node(setnode_id)
    setnode["flags"] = {"collapsed": True}
    setnode["color"] = "#322"
    setnode["bgcolor"] = "#533"

    ed.add_link(encode_id, 0, setnode_id, 0, "LATENT")
    return encode_id, setnode_id


# --------------------------------------------------------------------------
# Subgraph splice: AudioLatentSlice replaces #598 + #600 chain
# --------------------------------------------------------------------------

def _add_subgraph_input(ed: WorkflowEditor) -> int:
    """Append a new LATENT input named `full_audio_latent` to the
    subgraph schema. Returns its slot index."""
    sg = ed.get_subgraph(0)
    assert sg is not None
    inputs = sg.setdefault("inputs", [])
    new_slot = len(inputs)
    inputs.append({
        "id": str(uuid.uuid4()),
        "name": SETNODE_NAME,
        "type": "LATENT",
        "linkIds": [],
        "localized_name": SETNODE_NAME,
        "label": "full audio latent (pre-encoded)",
        "pos": [-3015, 3700],
    })
    return new_slot


def _add_invoker_input(ed: WorkflowEditor) -> int:
    """Append a new LATENT input slot on the top-level subgraph invoker
    so the SetNode-broadcast latent flows into the subgraph. Returns
    invoker's input slot index."""
    invoker = ed.find_subgraph_invoker(0)
    assert invoker is not None
    inputs = invoker.setdefault("inputs", [])
    new_slot_idx = len(inputs)
    inputs.append({
        "name": SETNODE_NAME,
        "type": "LATENT",
        "link": None,
    })
    return new_slot_idx


def _add_top_level_getnode(ed: WorkflowEditor) -> int:
    """Add top-level GetNode that reads the full_audio_latent SetNode
    and feeds it into the subgraph invoker."""
    nid = ed.add_top_level_node(
        node_type="GetNode",
        pos=[300, 6500],
        size=[210, 34],
        inputs=[],
        outputs=[
            {"name": "LATENT", "type": "LATENT", "links": []},
        ],
        widgets_values=[SETNODE_NAME],
        properties={
            "Node name for S&R": "GetNode",
            "aux_id": "kijai/ComfyUI-KJNodes",
        },
        title=f"Get_{SETNODE_NAME}",
    )
    node = ed.find_node(nid)
    node["flags"] = {"collapsed": True}
    node["color"] = "#322"
    node["bgcolor"] = "#533"
    return nid


def _splice_subgraph(
    ed: WorkflowEditor,
    new_input_slot: int,
    *,
    source_seconds: float,
    window_seconds: float,
) -> int:
    """Insert AudioLatentSlice in the subgraph; rewire #606's
    audio_latent input to read from the slicer instead of #598; bypass
    #598 + #600. Returns slicer's node id."""
    sg = ed.get_subgraph(0)
    assert sg is not None

    slicer_id = ed.add_subgraph_node(
        node_type="AudioLatentSlice",
        pos=[300, 4400],
        size=[270, 130],
        inputs=[
            {"name": "latent", "type": "LATENT", "link": None},
            {"name": "source_seconds", "type": "FLOAT",
             "widget": {"name": "source_seconds"}, "link": None},
            {"name": "start_seconds", "type": "FLOAT",
             "widget": {"name": "start_seconds"}, "link": None},
            {"name": "duration_seconds", "type": "FLOAT",
             "widget": {"name": "duration_seconds"}, "link": None},
        ],
        outputs=[
            {"name": "LATENT", "type": "LATENT", "links": []},
        ],
        widgets_values=[source_seconds, 0.0, window_seconds],
        properties={
            "Node name for S&R": "AudioLatentSlice",
            "aux_id": "fblissjr/ComfyUI-AudioLoopHelper",
        },
        title="Slice full-audio latent for this iter",
    )

    # Wire slicer.latent ← subgraph distributor (-10) at new input slot
    ed.add_subgraph_link(-10, new_input_slot, slicer_id, 0, "LATENT")
    # Wire slicer.start_seconds ← video_start_time (subgraph slot 16)
    vst_slot = WorkflowEditor.find_input_slot(sg, "video_start_time")
    if vst_slot is None:
        raise SystemExit("subgraph missing 'video_start_time' input slot")
    ed.add_subgraph_link(-10, vst_slot, slicer_id, 2, "FLOAT")

    # Find the existing link feeding #606.audio_latent (slot 1) — was from #598
    link_to_606 = ed.find_subgraph_link_to_slot(SG_AUDIO_VIDEO_MASK_ID, 1)
    if link_to_606 is None:
        raise SystemExit(f"subgraph #{SG_AUDIO_VIDEO_MASK_ID} has no inbound link on audio_latent slot")
    ed.remove_subgraph_link(link_to_606["id"], 0)
    # Rewire: AudioLatentSlice.LATENT (slot 0) → #606.audio_latent (slot 1)
    ed.add_subgraph_link(slicer_id, 0, SG_AUDIO_VIDEO_MASK_ID, 1, "LATENT")

    # Bypass #598 + #600 (preserves UI clarity, supports easy revert)
    for nid in (SG_AUDIO_VAE_ENCODE_ID, SG_TRIM_AUDIO_ID):
        node = ed.find_subgraph_node(nid, 0)
        if node is not None:
            node["mode"] = 4

    return slicer_id


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _migrate(
    input_path: Path, output_path: Path,
    source_seconds: float, window_seconds: float,
    dry_run: bool,
) -> None:
    if input_path != output_path and output_path.exists():
        if _already_applied(WorkflowEditor(output_path)):
            print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
            return

    if not dry_run and input_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"  copied {input_path} -> {output_path}")

    ed_target = output_path if (output_path.exists() and not dry_run) else input_path
    ed = WorkflowEditor(ed_target)

    if _already_applied(ed):
        print(f"{output_path.name}: already migrated, skipping.")
        return

    err = _preflight(ed)
    if err is not None:
        raise SystemExit(f"Refusing to migrate: {err}")

    if dry_run:
        print(f"would migrate {output_path.name}: top-level full-song encode + subgraph AudioLatentSlice splice")
        print(f"  source_seconds={source_seconds}, window_seconds={window_seconds}")
        return

    print(f"{output_path.name}: applying audio-latent pre-encode pattern...")
    encode_id, setnode_id = _add_full_song_encode_chain(ed)
    print(f"  added LTXVAudioVAEEncode({encode_id}) + SetNode({setnode_id}, '{SETNODE_NAME}')")

    sg_input_slot = _add_subgraph_input(ed)
    invoker_slot = _add_invoker_input(ed)
    invoker = ed.find_subgraph_invoker(0)
    assert invoker is not None
    getnode_id = _add_top_level_getnode(ed)
    ed.add_link(getnode_id, 0, invoker["id"], invoker_slot, "LATENT")
    print(f"  added subgraph LATENT input '{SETNODE_NAME}' "
          f"(sg slot {sg_input_slot}, invoker slot {invoker_slot}); "
          f"GetNode({getnode_id}) wired to invoker")

    slicer_id = _splice_subgraph(
        ed, sg_input_slot,
        source_seconds=source_seconds, window_seconds=window_seconds,
    )
    print(f"  added subgraph AudioLatentSlice({slicer_id}); "
          f"rewired #{SG_AUDIO_VIDEO_MASK_ID}.audio_latent; "
          f"bypassed #{SG_AUDIO_VAE_ENCODE_ID} + #{SG_TRIM_AUDIO_ID}")

    ed.save(output_path)
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate JSON: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Audit: uv run --group dev python scripts/audit_workflows.py {output_path}")
    print( "  3. Open in ComfyUI; subgraph schema changed — DELETE-AND-RE-ADD the loop subgraph node")
    print( "  4. A/B render against the canonical baseline; expected: -8 to -15s/render")


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--source-seconds", type=float, default=DEFAULT_SOURCE_SECONDS,
                    help=f"Total seconds of audio in the encoded latent "
                         f"(matches upstream TrimAudioDuration widget; default {DEFAULT_SOURCE_SECONDS}).")
    ap.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS,
                    help=f"Per-iter window length (matches LTXFramePlanner widget; default {DEFAULT_WINDOW_SECONDS}).")
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what WOULD change without writing files.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return 0

    _migrate(
        Path(args.input), output_path,
        args.source_seconds, args.window_seconds,
        args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
