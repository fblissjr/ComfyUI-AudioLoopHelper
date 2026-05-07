"""apply_dedupe_initial_render_prompt — single source of truth for the 0:00 prompt.

Last updated: 2026-05-07

Eliminates the duplicate-prompt footgun where Node 169 (CLIPTextEncode
for the initial render) repeats the schedule's 0:00 entry verbatim.
The shipped workflow asks the user to type the same prompt in two
places; runtime-level byte-equality is enforced by tests, but at
authoring time it's an easy way to drift.

Replaces Node 169 with a top-level `ConditioningSelectByIteration`
that reads `conditioning_list[0]` from the existing batch encoder
(Node 1615). The schedule string becomes the single source of truth
for both the initial render's prompt and every loop iteration's
prompt.

Mechanics:
  - Node 1615 already encodes one CONDITIONING per iteration. Index 0
    of `conditioning_list` corresponds to time `0 * stride` — i.e. the
    schedule's 0:00 entry, which is exactly what the initial render
    needs.
  - The new selector node fans its single CONDITIONING output to both
    consumers that previously read from Node 169:
      * Node 164 (LTXVConditioning) `positive`
      * Node 420 (ConditioningZeroOut) `conditioning` — kept intact
        per the CLAUDE.md "CFGGuider validates both slots" rule. The
        ZeroOut → negative chain stays runtime-inert at CFG=1.
  - Node 169 + its CLIP wire (link 1256) are removed. Other CLIP
    consumers from Node 416 (#507 CLIPTextEncode for negative,
    #1615 batch encoder) are unaffected.

Compatibility:
  - Idempotent. Title `"Initial render conditioning (from schedule[0])"`
    on the new selector is the signature.
  - Drafted into `internal/workflows/`. Per scripts/CLAUDE.md
    carve-out, staged-variant scripts skip the F-pair audit-invariant
    requirement until promotion.
  - Does NOT touch the loop subgraph. The subgraph's existing
    inside-loop selector (separate node) keeps reading
    `conditioning_list[current_iteration]` as before.

Coordination with apply_prompt_relay_initial_render.py:
  - That script splices PromptRelayEncode between Node 169 and
    Node 164. It and this script are mutually exclusive on the
    initial-render path. Detect-and-refuse if PromptRelayEncode is
    present (pre-flight guard).

Usage:
    uv run --group dev python scripts/apply_dedupe_initial_render_prompt.py
    uv run --group dev python scripts/apply_dedupe_initial_render_prompt.py --dry-run
    uv run --group dev python scripts/apply_dedupe_initial_render_prompt.py --revert

Defaults:
    --input  example_workflows/audio-loop-music-video_latent.json
    --output internal/workflows/loop_dedupe_initial_prompt.draft.json
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor, resolve_repo_path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

NODE_169_CLIP_TEXT_ENCODE = 169
NODE_164_LTXV_CONDITIONING = 164
NODE_420_ZERO_OUT = 420
NODE_1615_BATCH_ENCODER = 1615
NODE_1582_LOOP_CONTROLLER = 1582
NODE_1560_LOOP_PLANNER = 1560
BATCH_ENCODER_OUTPUT_SLOT = 0  # conditioning_list

# After the schema change adding stride_seconds + audio_duration to
# AudioLoopPlanner, these are the slot indices the new outputs occupy.
PLANNER_STRIDE_SECONDS_SLOT = 2
PLANNER_AUDIO_DURATION_SLOT = 3

NEW_NODE_TYPE = "ConditioningSelectByIteration"
NEW_NODE_TITLE = "Initial render conditioning (from schedule[0])"

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "internal/workflows/loop_dedupe_initial_prompt.draft.json"


def _find_new_selector(ed: WorkflowEditor) -> dict | None:
    for n in ed.wf["nodes"]:
        if n.get("type") == NEW_NODE_TYPE and n.get("title") == NEW_NODE_TITLE:
            return n
    return None


def _already_migrated(ed: WorkflowEditor) -> bool:
    return _find_new_selector(ed) is not None


def _check_no_prompt_relay(ed: WorkflowEditor) -> None:
    """Refuse if PromptRelayEncode is already wired into the initial-render path.

    The two scripts both rewire the conditioning chain at the same point.
    Re-running this script after PromptRelayEncode would silently disconnect
    the relay; refuse instead.
    """
    for n in ed.wf["nodes"]:
        if n.get("type") == "PromptRelayEncode":
            raise SystemExit(
                "PromptRelayEncode node detected — initial-render conditioning "
                "is already rewired by apply_prompt_relay_initial_render.py. "
                "These two scripts are mutually exclusive. Choose one path."
            )


def _apply(ed: WorkflowEditor, dry_run: bool) -> None:
    if _already_migrated(ed):
        print(f"  {ed.path.name}: already migrated, skipping.")
        return

    _check_no_prompt_relay(ed)

    for nid, label in (
        (NODE_169_CLIP_TEXT_ENCODE, "CLIPTextEncode for initial render"),
        (NODE_1615_BATCH_ENCODER, "TimestampPromptScheduleBatchEncode"),
        (NODE_1582_LOOP_CONTROLLER, "AudioLoopController"),
        (NODE_1560_LOOP_PLANNER, "AudioLoopPlanner"),
    ):
        if not ed.has_node(nid):
            raise SystemExit(
                f"Node #{nid} ({label}) not found. This script assumes the "
                "canonical loop layout."
            )

    n169 = ed.find_node(NODE_169_CLIP_TEXT_ENCODE)
    if n169.get("type") != "CLIPTextEncode":
        raise SystemExit(
            f"Node #{NODE_169_CLIP_TEXT_ENCODE} is type {n169.get('type')!r}, "
            "expected 'CLIPTextEncode'. Layout drift; refusing to mutate."
        )

    if dry_run:
        print(f"  {ed.path.name}:")
        print(f"    would add {NEW_NODE_TYPE} ('{NEW_NODE_TITLE}'), "
              "current_iteration=0")
        print(f"    would wire conditioning_list <- #{NODE_1615_BATCH_ENCODER} "
              f"output {BATCH_ENCODER_OUTPUT_SLOT}")
        print(f"    would rewire #{NODE_164_LTXV_CONDITIONING}.positive and "
              f"#{NODE_420_ZERO_OUT}.conditioning to the new selector")
        print(f"    would remove #{NODE_169_CLIP_TEXT_ENCODE} + "
              "all attached top-level links")
        print(f"    would extend #{NODE_1560_LOOP_PLANNER}.outputs[] with "
              "stride_seconds + audio_duration slots")
        print(f"    would rewire #{NODE_1615_BATCH_ENCODER}.stride_seconds + "
              f".audio_duration from #{NODE_1582_LOOP_CONTROLLER} to "
              f"#{NODE_1560_LOOP_PLANNER} (cycle break)")
        return

    # Add the new selector. Position it where Node 169 was so the visual
    # flow is preserved before Phase 3 layout pass.
    pos = list(n169["pos"])
    size = [320, 80]

    new_id = ed.add_top_level_node(
        node_type=NEW_NODE_TYPE,
        pos=pos,
        size=size,
        inputs=[
            WorkflowEditor.io_in("conditioning_list", "*"),
            WorkflowEditor.widget_in("current_iteration", "INT"),
        ],
        outputs=[WorkflowEditor.out("conditioning", "CONDITIONING")],
        widgets_values=[0],
        properties={
            "aux_id": "fblissjr/ComfyUI-AudioLoopHelper",
            "Node name for S&R": NEW_NODE_TYPE,
            "cnr_id": "comfyui-audioloophelper",
        },
        title=NEW_NODE_TITLE,
    )

    # Strip Node 169 and every top-level link touching it. After this,
    # Node 164 positive + Node 420 conditioning are dangling — we wire
    # them to the new selector below.
    ed.remove_node_and_links(NODE_169_CLIP_TEXT_ENCODE)

    # Wire: batch encoder.conditioning_list -> selector.conditioning_list
    ed.add_link(
        NODE_1615_BATCH_ENCODER, BATCH_ENCODER_OUTPUT_SLOT,
        new_id, 0,
        "*",
    )

    # Wire: selector.conditioning -> Node 164.positive
    n164 = ed.find_node(NODE_164_LTXV_CONDITIONING)
    pos_slot = WorkflowEditor.find_input_slot(n164, "positive")
    ed.add_link(new_id, 0, NODE_164_LTXV_CONDITIONING, pos_slot, "CONDITIONING")

    # Wire: selector.conditioning -> Node 420.conditioning
    n420 = ed.find_node(NODE_420_ZERO_OUT)
    zo_slot = WorkflowEditor.find_input_slot(n420, "conditioning")
    ed.add_link(new_id, 0, NODE_420_ZERO_OUT, zo_slot, "CONDITIONING")

    # Cycle break: rewire batch encoder's stride_seconds + audio_duration
    # from AudioLoopController (depends on current_iteration) to
    # AudioLoopPlanner (cycle-free; same _compute_loop_geometry formula).
    _rewire_batch_encoder_to_planner(ed)

    print(
        f"  {ed.path.name}: removed #{NODE_169_CLIP_TEXT_ENCODE}; "
        f"added #{new_id} ({NEW_NODE_TYPE}); rewired "
        f"#{NODE_164_LTXV_CONDITIONING}.positive + "
        f"#{NODE_420_ZERO_OUT}.conditioning to it; "
        f"rewired #{NODE_1615_BATCH_ENCODER}.stride/duration to "
        f"#{NODE_1560_LOOP_PLANNER}."
    )


def _ensure_planner_output_slot(planner: dict, slot_index: int, name: str) -> None:
    """Extend the planner's saved outputs[] to include the new slot.

    AudioLoopPlanner's schema gained `stride_seconds` (slot 2) and
    `audio_duration` (slot 3) outputs to provide a cycle-free source.
    Existing saved workflows have only outputs[0..1]; ComfyUI's loader
    will sync from the schema, but persisting the extension into the
    JSON keeps the file coherent with its declared shape.
    """
    outs = planner.setdefault("outputs", [])
    while len(outs) <= slot_index:
        outs.append({"name": "", "type": "FLOAT", "links": []})
    outs[slot_index]["name"] = name
    outs[slot_index]["type"] = "FLOAT"
    outs[slot_index].setdefault("links", [])


def _rewire_batch_encoder_to_planner(ed: WorkflowEditor) -> None:
    """Move batch encoder's stride/audio_duration sources from controller to planner."""
    planner = ed.find_node(NODE_1560_LOOP_PLANNER)
    _ensure_planner_output_slot(planner, PLANNER_STRIDE_SECONDS_SLOT, "stride_seconds")
    _ensure_planner_output_slot(planner, PLANNER_AUDIO_DURATION_SLOT, "audio_duration")

    encoder = ed.find_node(NODE_1615_BATCH_ENCODER)
    for input_name, planner_slot in (
        ("stride_seconds", PLANNER_STRIDE_SECONDS_SLOT),
        ("audio_duration", PLANNER_AUDIO_DURATION_SLOT),
    ):
        slot = WorkflowEditor.find_input_slot(encoder, input_name)
        existing_link = ed.find_link_to_slot(NODE_1615_BATCH_ENCODER, slot)
        if existing_link is not None:
            existing_src = existing_link[1]
            # Only rewire if the existing source is the controller; if the
            # workflow has already been migrated to read from the planner,
            # leave it alone (idempotence safety).
            if existing_src == NODE_1582_LOOP_CONTROLLER:
                ed.remove_link(existing_link[0])
            elif existing_src == NODE_1560_LOOP_PLANNER:
                continue
        ed.add_link(
            NODE_1560_LOOP_PLANNER, planner_slot,
            NODE_1615_BATCH_ENCODER, slot,
            "FLOAT",
        )


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    if output_path.exists() and input_path != output_path:
        existing = WorkflowEditor(output_path)
        if _already_migrated(existing):
            print(
                f"  {output_path}: already migrated, skipping. "
                "Run --revert then re-apply to pull upstream bug fixes from source."
            )
            return

    if not dry_run:
        if input_path != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            print(f"  copied {input_path} -> {output_path}")

    target = output_path if not dry_run else input_path
    ed = WorkflowEditor(target)
    _apply(ed, dry_run=dry_run)
    if not dry_run:
        ed.save()


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"Source workflow (default: {DEFAULT_INPUT}).")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"Output draft path (default: {DEFAULT_OUTPUT}).")
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    in_path = resolve_repo_path(args.input)
    out_path = resolve_repo_path(args.output)

    if args.revert:
        _revert(out_path)
        return

    _migrate(in_path, out_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
