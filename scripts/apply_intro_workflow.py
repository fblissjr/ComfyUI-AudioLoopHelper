"""apply_intro_workflow.

Last updated: 2026-05-04

Layout-maintenance script for the canonical
`example_workflows/audio-loop-music-video_latent.json`. The migrations
this script originally performed (LoRA chain splice, IC-LoRA bypass,
Notes, 9-group two-row layout) are baked into the canonical workflow
as of the 2026-05-04 consolidation pass that retired the
`_intro` / `_iclora` / `_iclora_audio_pre_encode` variants.

What's still useful here: the **layout pass** (`_layout_workflow`) and
the node-id → group classifier. If the canonical layout drifts (manual
edits, accidental coordinate changes), `--revert` then re-apply
restores the shipped layout.

For completely fresh builds (e.g. a new variant that wants to inherit
the intro shape), re-target via `--input` / `--output`.

Dependencies introduced: zero new node types. `LoraLoaderModelOnly` is
comfy-core.

Compatibility:
  - F11 (`dead_lora_loader_scaffolding_absent`) is keyed on specific
    legacy IDs (1625/1626/1627) + titles; the chain uses fresh IDs +
    different titles, so F11 won't fire on it.
  - F12 (IC-LoRA checks) doesn't filter by `mode`, so passes with the
    chain bypassed (loader still present in JSON; cropguides path intact).
  - Audio path untouched.

Usage:
    uv run --group dev python scripts/apply_intro_workflow.py
    uv run --group dev python scripts/apply_intro_workflow.py --revert
    uv run --group dev python scripts/apply_intro_workflow.py --dry-run

Idempotent. `--revert` is only valid in staging mode (input != output);
when self-targeting the canonical it refuses with a `git checkout`
hint rather than deleting the shipped workflow.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers._layout_classifications import compose  # noqa: E402
from workflow_utils import WorkflowEditor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Anchor node IDs (canonical latent workflow shape) ---
SPLICE_UPSTREAM_ID = 503        # LTX2SamplingPreviewOverride (MODEL upstream of LoRAs)
ICLORA_LOADER_ID = 1635         # LTXICLoRALoaderModelOnly
ICLORA_VHS_LOAD_ID = 1636       # VHS_LoadVideo (reference video)
ICLORA_GUIDE_SG_ID = 1640       # LTXAddVideoICLoRAGuide (inside subgraph)

REQUIRED_SOURCE_NODES = (SPLICE_UPSTREAM_ID, ICLORA_LOADER_ID, ICLORA_VHS_LOAD_ID)

# Self-targeting: the canonical latent IS the post-intro shape since the
# 2026-05-04 consolidation. Re-running rebuilds the layout idempotently.
DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "example_workflows/audio-loop-music-video_latent.json"

DISTILL_LORA_DEFAULT = "ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors"
DISTILL_LORA_STRENGTH = 0.5  # matches Lightricks's reference distilled workflow

# Marker title — used for idempotency checks and revert
MARKER_TITLE = "Distill LoRA (enable for base ltx-2.3 dev model)"

# --- Group keys (single source — referenced by classifier, GROUPS, note_placement)
G_INPUTS = "1_inputs"
G_MODELS = "2_models"
G_LORAS = "3_loras"
G_COND = "4_cond"
G_SAMPLER = "5_sampler"
G_LOOP = "6_loop"
G_OUTPUT = "7_output"
G_PREENCODE = "8_preencode"
G_ICLORA_REF = "9_iclora_ref"

# Tags written into a node's `properties` dict so the layout pass can
# classify nodes we created (instead of threading their IDs through
# every function). ComfyUI ignores unknown property keys.
ALH_GROUP_TAG = "_alh_group"
ALH_NOTE_KEY_TAG = "_alh_note_key"


# --------------------------------------------------------------------------
# Pre-flight + idempotency
# --------------------------------------------------------------------------

def _preflight(ed: WorkflowEditor) -> str | None:
    missing = ed.require_nodes(REQUIRED_SOURCE_NODES)
    if missing:
        return f"missing required source nodes: {missing}"
    sg = ed.get_subgraph(0)
    if sg is None:
        return "input has no subgraph (loop body)"
    sg_ids = {n["id"] for n in sg.get("nodes", [])}
    if ICLORA_GUIDE_SG_ID not in sg_ids:
        return f"subgraph missing #{ICLORA_GUIDE_SG_ID} LTXAddVideoICLoRAGuide"
    return None


def _already_applied(ed: WorkflowEditor) -> bool:
    """Marker: a top-level LoraLoaderModelOnly with our distinctive title."""
    for n in ed.wf.get("nodes") or []:
        if isinstance(n, dict) and n.get("type") == "LoraLoaderModelOnly" and n.get("title") == MARKER_TITLE:
            return True
    return False


# --------------------------------------------------------------------------
# Migration steps
# --------------------------------------------------------------------------

def _splice_lora_chain(ed: WorkflowEditor) -> tuple[int, int]:
    """Splice two LoraLoaderModelOnly nodes between #503 and #1635.

    Positions are placeholders — the layout pass relocates them.
    Returns (distill_id, style_id).
    """
    link = ed.find_link_to_slot(ICLORA_LOADER_ID, 0)
    if link is None:
        raise SystemExit(
            f"could not find MODEL link into #{ICLORA_LOADER_ID}.inputs[0]; "
            f"source workflow shape unexpected"
        )
    lid, src, src_slot, _, _, dtype = link
    if src != SPLICE_UPSTREAM_ID:
        raise SystemExit(
            f"expected MODEL input to #{ICLORA_LOADER_ID} to come from #{SPLICE_UPSTREAM_ID}, "
            f"got #{src}. Source workflow shape unexpected."
        )

    distill_id = _make_lora_loader(
        ed, pos=[0, 0], title=MARKER_TITLE,
        widgets=[DISTILL_LORA_DEFAULT, DISTILL_LORA_STRENGTH],
    )
    style_id = _make_lora_loader(
        ed, pos=[0, 0], title="Style or ID LoRA (optional)",
        widgets=["", 1.0],
    )

    ed.remove_link(lid)
    ed.add_link(SPLICE_UPSTREAM_ID, src_slot, distill_id, 0, dtype)
    ed.add_link(distill_id, 0, style_id, 0, dtype)
    ed.add_link(style_id, 0, ICLORA_LOADER_ID, 0, dtype)

    return distill_id, style_id


def _make_lora_loader(ed: WorkflowEditor, *, pos: list, title: str, widgets: list) -> int:
    """Add a bypassed LoraLoaderModelOnly. Returns its node ID."""
    inputs = [{"name": "model", "type": "MODEL", "link": None}]
    outputs = [{"name": "MODEL", "type": "MODEL", "links": []}]
    properties = {
        "cnr_id": "comfy-core",
        "ver": "0.8.2",
        "Node name for S&R": "LoraLoaderModelOnly",
        ALH_GROUP_TAG: G_LORAS,
    }
    nid = ed.add_top_level_node(
        node_type="LoraLoaderModelOnly",
        pos=pos,
        size=[440, 82],
        inputs=inputs,
        outputs=outputs,
        widgets_values=widgets,
        properties=properties,
        title=title,
    )
    ed.find_node(nid)["mode"] = 4
    return nid


def _bypass_iclora(ed: WorkflowEditor) -> list[str]:
    """Set mode=4 on the IC-LoRA loader, ref-video VHS_LoadVideo, and subgraph guide.

    Returns a list of human-readable bypass actions for logging.
    """
    actions = []
    for nid in (ICLORA_LOADER_ID, ICLORA_VHS_LOAD_ID):
        n = ed.find_node(nid)
        if n.get("mode") != 4:
            n["mode"] = 4
            actions.append(f"top-level #{nid} ({n.get('type')}) → mode=4")

    sg = ed.get_subgraph(0)
    for n in sg.get("nodes", []):
        if n.get("id") == ICLORA_GUIDE_SG_ID and n.get("mode") != 4:
            n["mode"] = 4
            actions.append(f"subgraph #{ICLORA_GUIDE_SG_ID} ({n.get('type')}) → mode=4")
    return actions


# TrimAudioDuration nodes ship without titles in the source workflow,
# making the canvas confusing — three TrimAudioDuration nodes with
# different roles all read as "TrimAudioDuration". Rename for clarity.
TRIM_RENAMES: dict[int, str] = {
    567: "Song Trim (skip intro, take N seconds)",
    601: "Initial-Render Audio Trim (10s context)",
    # #1631 already titled "ID-LoRA Reference Slice"
}


def _rename_trim_nodes(ed: WorkflowEditor) -> list[str]:
    """Apply human-readable titles to TrimAudioDuration nodes."""
    actions = []
    for nid, new_title in TRIM_RENAMES.items():
        n = ed.find_node(nid)
        if n is None or n.get("title") == new_title:
            continue
        old = n.get("title", "")
        n["title"] = new_title
        actions.append(f"#{nid} title {old!r} → {new_title!r}")
    return actions


# --------------------------------------------------------------------------
# Notes + groups
# --------------------------------------------------------------------------

NOTE_README = """AudioLoopHelper - Intro Workflow

1. LoadAudio (group 1):  drop your song.
2. LoadImage (group 1):  drop init image (matches first scene).
3. start_seed (group 1): any int.
4. Schedule + Node 169 (group 4): paste from
   analyze_audio_features.py --subject "..." output.

Run.

Optional:
  - LoRAs (group 3):    un-bypass + set lora_name.
  - IC-LoRA (group 9):  un-bypass loader + guide + ref video.

Sage attention (group 2): expects fblissjr/SageAttention-ada.
No build? Bypass the AudioLoopHelperSageAttention node and
fall back to default attention or KJNodes' Patch Sage Attention KJ.

Models (DiT, VAEs, CLIP, distill LoRA, etc.) are at:
  https://huggingface.co/Kijai/LTX2.3_comfy
"""

NOTE_NODE_169 = """Initial render prompt (Node 169).

Paste the "Node 169" block from analyze_audio_features.py here.

Verb choice is load-bearing. LTX 2.3 audio-video cross-attention binds
the visible action to the verb. Pick the verb that matches the action:
  - is singing / are singing together   (vocal performance)
  - is dancing                          (movement)
  - is playing <instrument>             (instrumental)
Generic verbs (performing, vocalizing) dilute the signal.

Concise > verbose. Tokens compete with audio + image alignment.
"""

NOTE_SCHEDULE = """Prompt schedule.

Paste the "TimestampPromptScheduleBatchEncode" block here.
Use 'In a [shot], [camera]' continuation framing for entries
after the first - NOT 'Cut to ...'. Lightricks's LTX 2.3 system
prompt explicitly trains the model to treat scene-cut language
as a discontinuation directive.
"""

NOTE_LORA = """LoRA chain.

Both bypassed by default. Model passes through unchanged.

Un-bypass "Distill LoRA" when running the BASE ltx-2.3 dev
checkpoint (NOT the merged distilled file the workflow ships
with by default).

Order: Distill -> Style. Either or both can be active.
"""

NOTE_ICLORA = """IC-LoRA chain.

Bypassed by default. To enable, un-bypass all THREE:
  - Reference Video (VHS_LoadVideo)
  - IC-LoRA Loader
  - IC-LoRA Guide (inside the loop subgraph)

Used for visual reference adapters: cameraman, outpaint,
union-control. Reference video is sliced per iteration via
GetImageRangeFromBatch inside the loop.

Cameraman IC-LoRA weights:
  https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v1
"""


# Note keys (referenced by note_placement table in the layout pass)
NOTE_KEY_README = "README"
NOTE_KEY_LORA = "LORA"
NOTE_KEY_NODE_169 = "NODE_169"
NOTE_KEY_SCHEDULE = "SCHEDULE"
NOTE_KEY_ICLORA = "ICLORA"


def _add_notes(ed: WorkflowEditor) -> int:
    """Add 5 Note nodes (placeholder positions; layout pass relocates).

    Each note is tagged with `ALH_NOTE_KEY_TAG` so the layout pass can
    find it without the caller threading IDs back. Returns the count
    of notes added.
    """
    note_specs = [
        (NOTE_KEY_README, NOTE_README, "README"),
        (NOTE_KEY_LORA, NOTE_LORA, "LoRA chain"),
        (NOTE_KEY_NODE_169, NOTE_NODE_169, "Node 169 prompt"),
        (NOTE_KEY_SCHEDULE, NOTE_SCHEDULE, "Schedule"),
        (NOTE_KEY_ICLORA, NOTE_ICLORA, "IC-LoRA chain"),
    ]
    for key, text, title in note_specs:
        nid = ed.add_top_level_node(
            node_type="Note", pos=[0, 0], size=[300, 240],
            inputs=[], outputs=[], widgets_values=[text],
            properties={ALH_NOTE_KEY_TAG: key}, title=title,
        )
        n = ed.find_node(nid)
        n["color"] = "#432"
        n["bgcolor"] = "#653"
    return len(note_specs)


# --------------------------------------------------------------------------
# Layout pass — assigns positions to every node + writes group bounds.
#
# Two-row layout, left-to-right data flow.
#
# Row 0 (main pipeline):  Inputs | Models | LoRAs | Conditioning | Sampler | Loop | Output
# Row 1 (preprocessing):  Audio pre-encode | Init render path | IC-LoRA ref
#
# Get/Set nodes are collapsed pills — placed at the top of their owning group.
# Notes are placed ABOVE their relevant group, never overlapping nodes.
# --------------------------------------------------------------------------

# Column origins (x positions). Row origins (y positions).
ROW0_Y = 200
ROW1_Y = 1900
ROW0_COL_X = [0, 700, 1400, 2050, 2900, 3500, 4500]      # 7 columns
ROW1_COL_X = [0, 1500, 3000]                              # 3 wide columns
GROUP_PAD = 40
NODE_X_OFFSET = 30   # node x within group
NODE_Y_OFFSET = 60   # node y within group (room for title)
INTRA_NODE_GAP = 50  # vertical gap between full-size nodes
COLLAPSED_GAP = 12   # tighter gap between collapsed pills

# Group palette (matching Lightricks/RuneXX patterns)
COLOR_INPUT = "#3f789e"     # blue — loaders / inputs
COLOR_MODEL = "#3f789e"
COLOR_LORA = "#1b4669"      # deep blue — bypassed-by-default scaffolding
COLOR_ICLORA = "#a18c25"    # gold — IC-LoRA reference (bypassed)
COLOR_COND = "#485248"      # green — conditioning
COLOR_SAMPLER = "#b58b2a"   # gold — sampling / generation
COLOR_LOOP = "#3f789e"
COLOR_OUTPUT = "#b58b2a"

# Functional column → group-key mapping for this script. Composed with
# `SHARED_NODE_FUNCTIONS` (in `scripts/_layout_classifications.py`) at
# import time to produce SOURCE_NODE_GROUPS. Nodes we add at runtime
# (LoRA loaders, Notes) carry their group via ALH_GROUP_TAG /
# ALH_NOTE_KEY_TAG in their `properties` dict instead of being listed
# in the shared table.
_FUNCTION_TO_GROUP: dict[str, str] = {
    "inputs":     G_INPUTS,
    "models":     G_MODELS,
    "loras":      G_LORAS,
    "cond":       G_COND,
    "sampler":    G_SAMPLER,
    "loop":       G_LOOP,
    "output":     G_OUTPUT,
    "preencode":  G_PREENCODE,
    "iclora_ref": G_ICLORA_REF,
}

SOURCE_NODE_GROUPS: dict[int, str] = compose(_FUNCTION_TO_GROUP)


def _is_pill(n: dict) -> bool:
    return n.get("flags", {}).get("collapsed", False) and n.get("type") in ("GetNode", "SetNode")


def _node_group(n: dict) -> str | None:
    """Resolve a node's group: tag wins over the source-id table.

    Tagged nodes are ones we created at runtime (LoRA loaders);
    everything else is mapped by id from SOURCE_NODE_GROUPS.
    """
    tag = (n.get("properties") or {}).get(ALH_GROUP_TAG)
    if tag:
        return tag
    return SOURCE_NODE_GROUPS.get(n["id"])


# Group origin + color + title. Order = render order. Each entry's
# (col_x, col_y) is independent of the others — change one, only one
# group moves.
GROUPS: dict[str, tuple[float, float, str, str]] = {
    G_INPUTS:      (ROW0_COL_X[0], ROW0_Y, COLOR_INPUT,   "1. Inputs"),
    G_MODELS:      (ROW0_COL_X[1], ROW0_Y, COLOR_MODEL,   "2. Models (DiT + VAEs + CLIP + Sage)"),
    G_LORAS:       (ROW0_COL_X[2], ROW0_Y, COLOR_LORA,    "3. LoRAs (bypassed)"),
    G_COND:        (ROW0_COL_X[3], ROW0_Y, COLOR_COND,    "4. Conditioning + Frame Planner"),
    G_SAMPLER:     (ROW0_COL_X[4], ROW0_Y, COLOR_SAMPLER, "5. Sampler"),
    G_LOOP:        (ROW0_COL_X[5], ROW0_Y, COLOR_LOOP,    "6. Loop"),
    G_OUTPUT:      (ROW0_COL_X[6], ROW0_Y, COLOR_OUTPUT,  "7. Output"),
    G_PREENCODE:   (ROW1_COL_X[0], ROW1_Y, COLOR_INPUT,   "8. Audio pre-encode + init render path"),
    G_ICLORA_REF:  (ROW1_COL_X[1], ROW1_Y, COLOR_ICLORA,  "9. IC-LoRA reference (bypassed)"),
}


# (anchor_group, dx, dy, w, h) per Note key. dx/dy are offsets from the
# anchor group's bounding box (top-left origin).
NOTE_PLACEMENT: dict[str, tuple[str, float, float, float, float]] = {
    NOTE_KEY_README:   (G_INPUTS,       0,  -680, 660, 600),
    NOTE_KEY_LORA:     (G_LORAS,        0,  -340, 600, 280),
    NOTE_KEY_NODE_169: (G_COND,       650,     0, 320, 320),
    NOTE_KEY_SCHEDULE: (G_COND,       650,   360, 320, 280),
    NOTE_KEY_ICLORA:   (G_ICLORA_REF,   0,  -300, 660, 240),
}


def _layout_workflow(ed: WorkflowEditor) -> None:
    """Reposition every classified node into its group column; compute
    per-group bounding boxes; position Note headers; replace the source
    workflow's groups with our 7+2 layout."""
    bins: dict[str, list[dict]] = {k: [] for k in GROUPS}
    notes_by_key: dict[str, dict] = {}
    for n in ed.wf["nodes"]:
        note_key = (n.get("properties") or {}).get(ALH_NOTE_KEY_TAG)
        if note_key is not None:
            notes_by_key[note_key] = n
            continue
        gkey = _node_group(n)
        if gkey is not None:
            bins[gkey].append(n)

    for gkey in bins:
        bins[gkey].sort(key=lambda n: (0 if _is_pill(n) else 1, n["id"]))

    group_bounds: dict[str, tuple[float, float, float, float]] = {}
    for gkey, (gx, gy, _color, _title) in GROUPS.items():
        nodes = bins[gkey]
        if not nodes:
            continue
        cur_y = gy + NODE_Y_OFFSET
        max_w = 0
        for n in nodes:
            sz = n.get("size", [280, 80])
            w, h = sz[0], sz[1]
            n["pos"] = [gx + NODE_X_OFFSET, cur_y]
            cur_y += h + (COLLAPSED_GAP if _is_pill(n) else INTRA_NODE_GAP)
            max_w = max(max_w, w)
        gw = max_w + 2 * GROUP_PAD + NODE_X_OFFSET
        gh = (cur_y - gy) + GROUP_PAD
        group_bounds[gkey] = (gx, gy - 30, gw, gh)  # 30px banner above content

    for note_key, (anchor_key, dx, dy, w, h) in NOTE_PLACEMENT.items():
        n = notes_by_key.get(note_key)
        anchor = group_bounds.get(anchor_key)
        if n is None or anchor is None:
            continue
        ax, ay, _aw, _ah = anchor
        n["pos"] = [ax + dx, ay + dy]
        n["size"] = [w, h]

    new_groups = []
    for i, (gkey, (_gx, _gy, color, title)) in enumerate(GROUPS.items(), start=1):
        b = group_bounds.get(gkey)
        if b is None:
            continue
        bx, by, bw, bh = b
        new_groups.append({
            "id": i,
            "title": title,
            "bounding": [bx, by, bw, bh],
            "color": color,
            "font_size": 24,
            "flags": {},
        })
    ed.wf["groups"] = new_groups


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _migrate(input_path: Path, output_path: Path, *, dry_run: bool) -> None:
    if not input_path.exists():
        raise SystemExit(f"input does not exist: {input_path}")

    if dry_run:
        ed = WorkflowEditor(input_path)
        err = _preflight(ed)
        if err:
            raise SystemExit(f"preflight failed: {err}")
        print(f"would copy   {input_path} -> {output_path}")
        print( "would splice 2x LoraLoaderModelOnly between #503 and #1635 (both mode=4)")
        print(f"             distill: {DISTILL_LORA_DEFAULT} @ {DISTILL_LORA_STRENGTH}")
        print( "             style:   '' @ 1.0")
        print(f"would bypass #{ICLORA_LOADER_ID} LTXICLoRALoaderModelOnly")
        print(f"would bypass #{ICLORA_VHS_LOAD_ID} VHS_LoadVideo")
        print(f"would bypass subgraph #{ICLORA_GUIDE_SG_ID} LTXAddVideoICLoRAGuide")
        print( "would add 5 Note nodes (README + per-group annotations)")
        print( "would add 2 groups (LoRAs, IC-LoRA)")
        return

    if output_path.exists() and _already_applied(WorkflowEditor(output_path)):
        print(f"{output_path.name}: already applied, skipping. Run --revert to reset.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path != output_path:
        shutil.copy2(input_path, output_path)
        print(f"  copied {input_path} -> {output_path}")

    ed = WorkflowEditor(output_path)
    err = _preflight(ed)
    if err:
        raise SystemExit(f"preflight failed: {err}")

    distill_id, style_id = _splice_lora_chain(ed)
    print(f"  spliced LoRA chain: distill=#{distill_id} style=#{style_id}")

    bypass_actions = _bypass_iclora(ed)
    for a in bypass_actions:
        print(f"  bypassed: {a}")

    rename_actions = _rename_trim_nodes(ed)
    for a in rename_actions:
        print(f"  renamed: {a}")

    note_count = _add_notes(ed)
    print(f"  added {note_count} Note nodes")

    _layout_workflow(ed)
    print(f"  laid out {len(ed.wf.get('groups', []))} groups (2-row layout)")

    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate JSON: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Audit:         uv run --group dev python scripts/audit_workflows.py {output_path}")
    print(f"  3. Load in ComfyUI: open {output_path}")


def _revert(input_path: Path, output_path: Path) -> None:
    """Revert is only meaningful when input != output (staging mode).
    With self-targeting defaults, --revert would delete the canonical;
    refuse instead and tell the user to git-restore."""
    if input_path == output_path:
        raise SystemExit(
            "--revert refused: input == output (self-targeting on the "
            "canonical). Deleting would lose the shipped workflow. "
            "Use `git checkout HEAD -- {0}` to restore from history "
            "instead.".format(output_path)
        )
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
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"source workflow (default: {DEFAULT_INPUT})")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"output path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--revert", action="store_true",
                    help="delete the output staged file")
    ap.add_argument("--dry-run", action="store_true",
                    help="report planned ops without writing")
    args = ap.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = REPO_ROOT / input_path

    if args.revert:
        _revert(input_path, output_path)
        return

    _migrate(input_path, output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
