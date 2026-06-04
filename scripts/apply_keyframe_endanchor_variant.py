"""apply_keyframe_endanchor_variant.

Last updated: 2026-06-04

Stages an experimental START/MID/END keyframe-anchoring variant of the
per-iter keyframe workflow to
`example_workflows/experimental/audio-loop-music-video_latent_keyframe_endanchor.json`.

Forks `example_workflows/audio-loop-music-video_latent_keyframe.json` (the
shipped per-iter keyframe anchor). That base already has:
  - 3 keyframe encode chains (LoadImage -> LTXSmartImageResize ->
    LTXVPreprocess -> VAEEncode): keyframe latents #2033 / #2037 / #2041.
  - the init/fallback latent #1617.
  - the START selector #2042 (LTXIterKeyframeSchedule) -> #843.guide_latent
    -> in-subgraph #1519 LTXVAddLatentGuide(latent_idx=-1) — the existing
    first-frame identity anchor (out-of-timeline reference, hard lock @1.0).

This variant adds POSITIONAL keyframe anchoring at the window END (active)
and MID (bypassed), on TOP of the existing START continuity + identity
anchor — without touching the audio path, sigma chain, or F2/F3 symmetry
chains.

------------------------------------------------------------------------
WHY each piece (the design):

START anchor — NO new work. Window k's start IS window k-1's frozen END
via the overlap context (LatentContextExtract pulls the prev tail -> this
head; LatentOverlapTrim only trims the head). The existing #1519 identity
guide stays as-is. We do not add a start guide.

END anchor (ACTIVE by default) — a SECOND LTXIterKeyframeSchedule selector
whose keyframe slots are wired ONE KEYFRAME AHEAD (window k receives
keyframe k+1), feeding a NEW chained in-subgraph LTXVAddLatentGuide at
latent_idx = <window's LAST latent index>, strength dial default 0.7 (a
hard 1.0 end anchor can visibly "snap" near seams). Effect: each window
generates from "where the song left off" TOWARD the next keyframe; because
the window tail becomes the next window's frozen context, anchoring the END
also sets the NEXT window's START — the chain the design wants.

MID anchor (PRESENT, BYPASSED by default, mode=4) — a THIRD selector +
chained in-subgraph guide at the window's MIDDLE latent index, own strength
dial. Un-bypass both the selector and the guide to use it.

------------------------------------------------------------------------
latent_idx DERIVATION (positional, latent_idx > 0):

LTXVAddLatentGuide (ComfyUI-LTXVideo latents.py:464):
    latent_idx <= 0 -> frame_idx = latent_idx * 8   (out-of-timeline / frame 0)
    latent_idx  > 0 -> frame_idx = 1 + (latent_idx - 1) * 8   (POSITIONAL)
The existing #1519 uses latent_idx=-1 (identity ref). Our END/MID anchors
are POSITIONAL (> 0) — a DIFFERENT use of the same node.

The per-window VIDEO latent that #1519 (and our new guides) operates on is
the output of #606 LTXVAudioVideoMask, whose video_latent comes from
LatentContextExtract (overlap tail). With max_length='pad' (the workflow's
widget), #606 PADS that context UP to `required_latent_frames`:
    required_latent_frames = (round(video_end_time * video_fps) - 1)//8 + 1
This equals `window_latent_frames` from _compute_loop_geometry. For the
shipped config (window_seconds=19.88, fps=25, video_end_time=actual_seconds
=19.88, 960x544):
    window_px = round(19.88 * 25) = 497
    window_latent_frames = (497 - 1)//8 + 1 = 63   => T = 63
    END  latent_idx = T - 1      = 62   (frame_idx 489)
    MID  latent_idx = T // 2     = 31   (frame_idx 241)

These are CONFIG-DEPENDENT. If you change window_seconds / fps / the window
duration, recompute T and re-set END_LATENT_IDX / MID_LATENT_IDX (and the
node widgets). The variant bakes the values for the shipped config AND
writes the derivation into a MarkdownNote in the workflow so a human can
re-check. (latent_idx is a node WIDGET on LTXVAddLatentGuide, not a wired
INT — there is no `window_latent_frames` controller output to autowire it
from, and adding one would be a second iteration-state schema change in the
same session. Documented-widget is the honest choice, matching how #1519
already hard-codes -1.)

------------------------------------------------------------------------
KEYFRAME -> WINDOW mapping (END selector is wired ONE AHEAD):

  base keyframe latents: kf1=#2033, kf2=#2037, kf3=#2041, init/fallback=#1617
  END selector keyframe_latent_1 (target_iters '1') <- kf2 (#2037)
  END selector keyframe_latent_2 (target_iters '2') <- kf3 (#2041)
  END selector keyframe_latent_3 (target_iters '3') <- kf3 (#2041)  (last
      window ends on the FINAL keyframe rather than regressing to init)
  unmatched iters -> END selector fallback_latent = init #1617 (no-op
      end anchor == the prior keyframe carried via overlap).

You need N+1 keyframes for N fully-distinct anchored windows; with 3
keyframes the last anchored window re-uses kf3. The selector's own
fallback/clamp semantics cover any iter no row claims (1-based; see
LTXIterKeyframeSchedule). The MID selector is wired the SAME (one-ahead) so
the mid anchor pulls toward the same next keyframe; both selectors share
the existing keyframe encode chains (no new encoders).

------------------------------------------------------------------------
GUIDE CHAIN ORDER (in-subgraph; guides accumulate via keyframe_idxs, F3
crops strip them):

  #606.video_latent[0] -> #1519(START/identity, idx=-1)
    -> #<END>(LTXVAddLatentGuide, idx=62, strength 0.7)
      -> #<MID>(LTXVAddLatentGuide, idx=31, strength 0.5, mode=4 bypassed)
        -> #1640(IC-LoRA video ref, mode=4 bypassed)
          -> #655 LTXVCropGuidesNoLatent  -> #644 CFGGuider   (F3)
          -> #2008 LTXVCropGuides (LATENT) -> #2006 AdaIN ...  (F3)
          -> #583 LTXVConcatAVLatent (latent) -> sampler

The new guides sit UPSTREAM of #1640 and BOTH crop nodes, so F3 still
strips every accumulated guide and conditioning flows per F3 unchanged.
When MID is bypassed (default) it passes pos/neg/latent straight through
(same-type bypass), so the chain is END -> #1640 effectively.

------------------------------------------------------------------------
FLEXIBLE KEYFRAME SOURCE (parked / bypassed by default):

A real boolean SWITCH node exists — KJNodes `LazySwitchKJ` (IO.ANY, lazy:
only the selected upstream branch executes). The variant adds an
alternative source branch:
    VHS_LoadVideo(force_rate=1) -> EvenlySpacedKeyframes(3)
      -> 3x GetImageRangeFromBatch(0/1/2, num=1)   [KJNodes]
parked alongside the existing 3 LoadImage shots, plus a MarkdownNote
explaining how to switch. The parked branch is left UNWIRED (its outputs
dead-end) so it doesn't run by default; to switch a keyframe to the
video-extracted frame, route both sources through a `LazySwitchKJ`
(boolean) into the resize chain, OR just rewire the resize.image input to
the GetImageRangeFromBatch output. The Note documents both. We do NOT wire
LazySwitchKJ in by default (it would require choosing one source as
default and re-routing all 3 resizes); the switch node is named in the Note
as the sanctioned toggle.

------------------------------------------------------------------------
Compatibility:
  - Composes with apply_keyframe_iter_anchor.py output (this script's
    INPUT). Re-run that first if the base keyframe.json is stale.
  - Does NOT mutate example_workflows/ in place (staged experimental).
  - Staged-variant carve-out: no paired audit invariant required.

Usage:
    uv run --group dev python scripts/apply_keyframe_endanchor_variant.py
    uv run --group dev python scripts/apply_keyframe_endanchor_variant.py --dry-run
    uv run --group dev python scripts/apply_keyframe_endanchor_variant.py --revert

Idempotent on the OUTPUT path. `--revert` deletes the staging file.
`--dry-run` reports the planned ops without writing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent_keyframe.json"
DEFAULT_OUTPUT = (
    "example_workflows/experimental/audio-loop-music-video_latent_keyframe_endanchor.json"
)

# --- Base-workflow node IDs (from the shipped keyframe variant). ---
SUBGRAPH_INVOKER = 843            # subgraph invoker (loop body)
START_GUIDE = 1519                # in-subgraph LTXVAddLatentGuide (identity, idx=-1)
ICLORA_GUIDE = 1640               # in-subgraph IC-LoRA guide (mode=4; downstream of #1519)
AUDIO_VIDEO_MASK = 606            # in-subgraph LTXVAudioVideoMask (emits per-window video latent)
DISTRIBUTOR = -10                 # virtual subgraph input distributor

# Existing keyframe latents + init in the base workflow.
INIT_FALLBACK_VAEENCODE = 1617    # init/fallback latent
KEYFRAME_VAEENCODES = [2033, 2037, 2041]  # kf1, kf2, kf3

# TensorLoopOpen current_iteration (1-based), out slot 3.
TLO = 1539
TLO_CURRENT_ITERATION_SLOT = 3

# Existing subgraph input slot count (so new inputs append at the END).
BASE_SUBGRAPH_INPUT_COUNT = 20    # validated against the base; new = 20 (end), 21 (mid)
END_GUIDE_INPUT_SLOT = 20
MID_GUIDE_INPUT_SLOT = 21

# --- Derived latent indices for the SHIPPED config (window 19.88s, 25fps). ---
# window_latent_frames T = 63  =>  END = T-1 = 62, MID = T//2 = 31.
WINDOW_LATENT_FRAMES = 63
END_LATENT_IDX = WINDOW_LATENT_FRAMES - 1     # 62
MID_LATENT_IDX = WINDOW_LATENT_FRAMES // 2    # 31

END_STRENGTH = 0.7   # hard 1.0 can "snap" at seams
MID_STRENGTH = 0.5

KEYFRAME_COUNT = 3
SELECTOR_END_TITLE = "LTX Iter Keyframe Schedule (END anchor — one keyframe ahead)"
SELECTOR_MID_TITLE = "LTX Iter Keyframe Schedule (MID anchor — one keyframe ahead, bypassed)"
END_GUIDE_TITLE = "END anchor — window tail toward next keyframe"
MID_GUIDE_TITLE = "MID anchor — window middle (bypassed by default)"

REQUIRED_SOURCE_NODES = (
    SUBGRAPH_INVOKER,
    INIT_FALLBACK_VAEENCODE,
    *KEYFRAME_VAEENCODES,
    TLO,
)


# --------------------------------------------------------------------------
# Idempotence / guards
# --------------------------------------------------------------------------
def _find_node_by_title(ed: WorkflowEditor, node_type: str, title: str) -> dict | None:
    for n in ed.wf.get("nodes", []):
        if n.get("type") == node_type and n.get("title") == title:
            return n
    return None


def _already_migrated(ed: WorkflowEditor) -> bool:
    return _find_node_by_title(ed, "LTXIterKeyframeSchedule", SELECTOR_END_TITLE) is not None


def _assert_required_nodes_present(ed: WorkflowEditor) -> None:
    missing = ed.require_nodes(REQUIRED_SOURCE_NODES)
    if missing:
        raise SystemExit(
            f"Refusing to migrate: required source node(s) missing: {missing}. "
            "This script assumes the per-iter keyframe workflow layout "
            "(run scripts/apply_keyframe_iter_anchor.py first)."
        )
    # Validate the subgraph schema we are about to extend.
    sg = ed.get_subgraph(0)
    if sg is None:
        raise SystemExit("Refusing to migrate: no subgraph found in base workflow.")
    n_in = len(sg.get("inputs", []))
    if n_in != BASE_SUBGRAPH_INPUT_COUNT:
        raise SystemExit(
            f"Refusing to migrate: subgraph has {n_in} inputs, expected "
            f"{BASE_SUBGRAPH_INPUT_COUNT}. Base drifted — re-derive the append slots."
        )
    if ed.find_subgraph_node(START_GUIDE, 0) is None:
        raise SystemExit(f"Refusing to migrate: in-subgraph #{START_GUIDE} missing.")


# --------------------------------------------------------------------------
# Node factories
# --------------------------------------------------------------------------
def _make_selector(ed: WorkflowEditor, title: str, pos: list, mode: int) -> int:
    """Hand-build an LTXIterKeyframeSchedule DynamicCombo node (num_keyframes=3)."""
    inputs = [
        {"name": "fallback_latent", "type": "LATENT", "link": None},
        {"name": "current_iteration", "type": "INT",
         "widget": {"name": "current_iteration"}, "link": None},
    ]
    for i in range(1, KEYFRAME_COUNT + 1):
        inputs.append({"label": f"keyframe_latent_{i}",
                       "name": f"num_keyframes.keyframe_latent_{i}",
                       "shape": 7, "type": "LATENT", "link": None})
        inputs.append({"name": f"num_keyframes.target_iters_{i}", "type": "STRING",
                       "widget": {"name": f"target_iters_{i}"}, "shape": 7, "link": None})
    nid = ed.next_node_id()
    node = {
        "id": nid, "type": "LTXIterKeyframeSchedule", "pos": pos,
        "size": [340, 260], "flags": {}, "order": 0, "mode": mode,
        "inputs": inputs,
        "outputs": [{"name": "latent", "type": "LATENT", "links": []}],
        "properties": {"aux_id": "fblissjr/ComfyUI-AudioLoopHelper",
                       "Node name for S&R": "LTXIterKeyframeSchedule"},
        # target_iters default 1,2,3 (1-based; TLO emits 1,2,3,...). Re-spread
        # per song's iter count (AudioLoopPlanner.summary).
        "widgets_values": [0, str(KEYFRAME_COUNT)] + [str(i) for i in range(1, KEYFRAME_COUNT + 1)],
        "title": title,
    }
    ed.add_node(node)
    return nid


def _new_sg_input(name: str, label: str, pos: list) -> dict:
    """Build a subgraph boundary input slot dict (appended to sg['inputs'])."""
    return {
        # uuid5 keyed on the input name: deterministic across regenerations,
        # so re-running the generator stays byte-stable (md5 regen discipline,
        # scripts/CLAUDE.md "Byte-identical refactor validation").
        "id": str(uuid.uuid5(uuid.NAMESPACE_OID, name)),
        "name": name,
        "type": "LATENT",
        "linkIds": [],
        "localized_name": name,
        "label": label,
        "pos": pos,
    }


def _add_subgraph_input(ed: WorkflowEditor, name: str, label: str, pos: list) -> int:
    """Append a new LATENT input to the subgraph boundary AND mirror it on the
    invoker node. Returns the new slot index (== distributor src_slot).

    APPEND-ONLY: never reorder/remove existing inputs (slot indices bake at
    save time; removal shifts higher slots). The new slot lands at the end.
    """
    sg = ed.get_subgraph(0)
    assert sg is not None, "subgraph[0] validated by _assert_required_nodes_present"
    slot = len(sg["inputs"])
    sg["inputs"].append(_new_sg_input(name, label, pos))
    # Mirror on the invoker node (same order; link filled by the top-level wire).
    inv = ed.find_node(SUBGRAPH_INVOKER)
    inv["inputs"].append({"label": label, "name": name, "type": "LATENT", "link": None})
    return slot


def _add_inner_guide(
    ed: WorkflowEditor, title: str, pos: list, mode: int, latent_idx: int, strength: float
) -> int:
    """Add an in-subgraph LTXVAddLatentGuide. `strength` is carried as a WIDGET
    (its own per-node dial); `guiding_latent` is wired from the distributor."""
    return ed.add_subgraph_node(
        "LTXVAddLatentGuide", pos=pos, size=[294.5, 162], mode=mode,
        inputs=[
            {"localized_name": "vae", "name": "vae", "type": "VAE", "link": None},
            {"localized_name": "positive", "name": "positive", "type": "CONDITIONING", "link": None},
            {"localized_name": "negative", "name": "negative", "type": "CONDITIONING", "link": None},
            {"localized_name": "latent", "name": "latent", "type": "LATENT", "link": None},
            {"localized_name": "guiding_latent", "name": "guiding_latent", "type": "LATENT", "link": None},
            {"localized_name": "strength", "name": "strength", "type": "FLOAT",
             "widget": {"name": "strength"}, "link": None},
        ],
        outputs=[
            {"localized_name": "positive", "name": "positive", "type": "CONDITIONING", "links": []},
            {"localized_name": "negative", "name": "negative", "type": "CONDITIONING", "links": []},
            {"localized_name": "latent", "name": "latent", "type": "LATENT", "links": []},
        ],
        properties={
            "cnr_id": "ComfyUI-LTXVideo",
            "Node name for S&R": "LTXVAddLatentGuide",
            "aux_id": "Lightricks/ComfyUI-LTXVideo",
        },
        widgets_values=[latent_idx, strength],
        title=title,
    )


def _make_note(ed: WorkflowEditor, pos: list, size: list, text: str, title: str) -> int:
    return ed.add_top_level_node(
        "MarkdownNote", pos=pos, size=size,
        inputs=[], outputs=[],
        widgets_values=[text],
        properties={"Node name for S&R": "MarkdownNote"},
        title=title,
    )


# --------------------------------------------------------------------------
# Core mutation
# --------------------------------------------------------------------------
def _apply(ed: WorkflowEditor) -> None:
    sg = ed.get_subgraph(0)
    assert sg is not None, "subgraph[0] validated by _assert_required_nodes_present"

    # 1. Two new subgraph inputs (APPENDED at the end -> slots 20, 21).
    end_slot = _add_subgraph_input(ed, "end_guide_latent",
                                   "end keyframe (one ahead)", [-3015, 3760])
    mid_slot = _add_subgraph_input(ed, "mid_guide_latent",
                                   "mid keyframe (one ahead)", [-3015, 3820])
    assert end_slot == END_GUIDE_INPUT_SLOT, (end_slot, END_GUIDE_INPUT_SLOT)
    assert mid_slot == MID_GUIDE_INPUT_SLOT, (mid_slot, MID_GUIDE_INPUT_SLOT)
    print(f"  + subgraph input slot {end_slot}: end_guide_latent (LATENT)")
    print(f"  + subgraph input slot {mid_slot}: mid_guide_latent (LATENT)")

    # 2. Two new in-subgraph guides, chained #1519 -> END -> MID -> #1640.
    #    Re-route #1519's three outputs (pos/neg/latent) into END's inputs,
    #    then END -> MID -> (whatever #1519 used to feed = #1640).
    # Capture the current consumers of #1519's three outputs (the #1640 chain).
    sg_links = sg["links"]
    downstream = {}  # out_slot -> list of (tgt_id, tgt_slot, dtype)
    for l in list(sg_links):
        if l["origin_id"] == START_GUIDE:
            downstream.setdefault(l["origin_slot"], []).append(
                (l["target_id"], l["target_slot"], l["type"])
            )

    end_guide = _add_inner_guide(
        ed, END_GUIDE_TITLE, [3320, 4720], mode=0,
        latent_idx=END_LATENT_IDX, strength=END_STRENGTH,
    )
    mid_guide = _add_inner_guide(
        ed, MID_GUIDE_TITLE, [3660, 4720], mode=4,  # bypassed by default
        latent_idx=MID_LATENT_IDX, strength=MID_STRENGTH,
    )
    print(f"  + in-subgraph END guide #{end_guide} (latent_idx={END_LATENT_IDX}, "
          f"strength={END_STRENGTH}, active)")
    print(f"  + in-subgraph MID guide #{mid_guide} (latent_idx={MID_LATENT_IDX}, "
          f"strength={MID_STRENGTH}, mode=4 bypassed)")

    # Detach #1519's old outbound links (to #1640 etc.), re-point START -> END.
    for l in list(sg_links):
        if l["origin_id"] == START_GUIDE and l["target_id"] in (ICLORA_GUIDE,):
            ed.remove_subgraph_link(l["id"], 0)
    # START -> END (pos/neg/latent on matching slots 0/1/2).
    ed.add_subgraph_link(START_GUIDE, 0, end_guide, 1, "CONDITIONING", 0)  # positive
    ed.add_subgraph_link(START_GUIDE, 1, end_guide, 2, "CONDITIONING", 0)  # negative
    ed.add_subgraph_link(START_GUIDE, 2, end_guide, 3, "LATENT", 0)        # latent
    # END.vae + END.guiding_latent.
    ed.add_subgraph_link(DISTRIBUTOR, 3, end_guide, 0, "VAE", 0)           # vae from distributor[3]
    ed.add_subgraph_link(DISTRIBUTOR, END_GUIDE_INPUT_SLOT, end_guide, 4, "LATENT", 0)
    # END -> MID.
    ed.add_subgraph_link(end_guide, 0, mid_guide, 1, "CONDITIONING", 0)
    ed.add_subgraph_link(end_guide, 1, mid_guide, 2, "CONDITIONING", 0)
    ed.add_subgraph_link(end_guide, 2, mid_guide, 3, "LATENT", 0)
    ed.add_subgraph_link(DISTRIBUTOR, 3, mid_guide, 0, "VAE", 0)
    ed.add_subgraph_link(DISTRIBUTOR, MID_GUIDE_INPUT_SLOT, mid_guide, 4, "LATENT", 0)
    # MID -> the original #1519 consumers (#1640 pos/neg/latent).
    for out_slot, consumers in sorted(downstream.items()):
        for tgt_id, tgt_slot, dtype in consumers:
            ed.add_subgraph_link(mid_guide, out_slot, tgt_id, tgt_slot, dtype, 0)
    print(f"  chain: #{START_GUIDE} -> #{end_guide} -> #{mid_guide} -> #{ICLORA_GUIDE} "
          "(F3 crops downstream, unchanged)")

    # 3. END + MID selectors at the top level (one keyframe ahead).
    #    keyframe_latent_n <- keyframe (n+1); last row re-uses final keyframe.
    ahead = [KEYFRAME_VAEENCODES[1], KEYFRAME_VAEENCODES[2], KEYFRAME_VAEENCODES[2]]
    end_sel = _make_selector(ed, SELECTOR_END_TITLE, [-700, 6400], mode=0)
    mid_sel = _make_selector(ed, SELECTOR_MID_TITLE, [-700, 6700], mode=4)
    for sel_id, sel_name in ((end_sel, "END"), (mid_sel, "MID")):
        ed.add_link(INIT_FALLBACK_VAEENCODE, 0, sel_id, 0, "LATENT")    # fallback <- init
        ed.add_link(TLO, TLO_CURRENT_ITERATION_SLOT, sel_id, 1, "INT")  # current_iteration
        for k, ve in enumerate(ahead):
            ed.add_link(ve, 0, sel_id, 2 + k * 2, "LATENT")
        print(f"  + {sel_name} selector #{sel_id}: kf-ahead {ahead} (fallback=init #{INIT_FALLBACK_VAEENCODE})")

    # 4. Wire selectors into the new subgraph inputs on the invoker.
    ed.add_link(end_sel, 0, SUBGRAPH_INVOKER, END_GUIDE_INPUT_SLOT, "LATENT")
    ed.add_link(mid_sel, 0, SUBGRAPH_INVOKER, MID_GUIDE_INPUT_SLOT, "LATENT")
    print(f"  wire END selector #{end_sel} -> #{SUBGRAPH_INVOKER}.end_guide_latent[{END_GUIDE_INPUT_SLOT}]")
    print(f"  wire MID selector #{mid_sel} -> #{SUBGRAPH_INVOKER}.mid_guide_latent[{MID_GUIDE_INPUT_SLOT}]")

    # 5. Parked alternative keyframe SOURCE branch (VHS_LoadVideo, bypassed).
    vhs = ed.add_top_level_node(
        "VHS_LoadVideo", pos=[-2400, 9200], size=[300, 300],
        inputs=[WorkflowEditor.io_in("meta_batch", "VHS_BatchManager"),
                WorkflowEditor.io_in("vae", "VAE")],
        outputs=[WorkflowEditor.out("IMAGE", "IMAGE"), WorkflowEditor.out("frame_count", "INT"),
                 WorkflowEditor.out("audio", "AUDIO"), WorkflowEditor.out("video_info", "VHS_VIDEOINFO")],
        widgets_values={"video": "your_keyframe_clip.mp4", "force_rate": 1, "custom_width": 0,
                        "custom_height": 0, "frame_load_cap": 0, "skip_first_frames": 0,
                        "select_every_nth": 1, "format": "LTXV"},
        properties={"Node name for S&R": "VHS_LoadVideo", "cnr_id": "comfyui-videohelpersuite"},
        title="Parked keyframe source clip (force_rate=1) — bypassed",
    )
    ed.find_node(vhs)["mode"] = 4
    esk = ed.add_top_level_node(
        "EvenlySpacedKeyframes", pos=[-2040, 9200], size=[260, 82],
        inputs=[WorkflowEditor.io_in("images", "IMAGE")],
        outputs=[WorkflowEditor.out("IMAGE", "IMAGE")],
        widgets_values=[KEYFRAME_COUNT],
        properties={"Node name for S&R": "EvenlySpacedKeyframes", "aux_id": "fblissjr/ComfyUI-AudioLoopHelper"},
        title=f"Parked: {KEYFRAME_COUNT} evenly-spaced — bypassed",
    )
    ed.find_node(esk)["mode"] = 4
    ed.add_link(vhs, 0, esk, 0, "IMAGE")
    for i in range(KEYFRAME_COUNT):
        sel = ed.add_top_level_node(
            "GetImageRangeFromBatch", pos=[-1720, 9200 + i * 160], size=[260, 100],
            inputs=[WorkflowEditor.io_in("images", "IMAGE")],
            outputs=[WorkflowEditor.out("IMAGE", "IMAGE"), WorkflowEditor.out("MASK", "MASK")],
            widgets_values=[i, 1],  # start_index=i, num_frames=1
            properties={"Node name for S&R": "GetImageRangeFromBatch", "cnr_id": "comfyui-kjnodes"},
            title=f"Parked: keyframe {i + 1} (frame {i}) — bypassed",
        )
        ed.find_node(sel)["mode"] = 4
        ed.add_link(esk, 0, sel, 0, "IMAGE")
    print(f"  + parked source branch (bypassed): VHS_LoadVideo #{vhs} -> "
          f"EvenlySpacedKeyframes #{esk} -> 3x GetImageRangeFromBatch")

    # 6. MarkdownNote documenting the design + derivation + how to switch source.
    note_text = (
        "## END/MID keyframe anchoring (experimental)\n\n"
        "**START anchor** = window k-1's frozen END, carried via the overlap "
        "context. No start guide — the existing identity guide (in-subgraph "
        f"#{START_GUIDE}, latent_idx=-1) stays.\n\n"
        "**END anchor (active)**: selector wired ONE KEYFRAME AHEAD (window k "
        "<- keyframe k+1) -> in-subgraph guide at the window's LAST latent "
        f"index ({END_LATENT_IDX}), strength {END_STRENGTH}. Anchoring the END "
        "also sets the NEXT window's START (tail becomes next frozen context). "
        "A hard 1.0 end anchor can visibly 'snap' near seams; 0.7 is softer.\n\n"
        "**MID anchor (bypassed, mode=4)**: third selector + guide at the "
        f"window's MIDDLE latent index ({MID_LATENT_IDX}), strength "
        f"{MID_STRENGTH}. Un-bypass BOTH the MID selector and the in-subgraph "
        "MID guide to use it.\n\n"
        "### latent_idx derivation (RE-CHECK if you change the window)\n"
        "Per-window video latent T = window_latent_frames =\n"
        "`(round(window_seconds*fps) - 1)//8 + 1`.\n"
        "Shipped config (window 19.88s, fps 25, 960x544): "
        f"T = {WINDOW_LATENT_FRAMES}, so END latent_idx = T-1 = {END_LATENT_IDX}, "
        f"MID latent_idx = T//2 = {MID_LATENT_IDX}.\n"
        "latent_idx > 0 -> frame_idx = 1+(idx-1)*8 (positional, in-window). "
        "These are NODE WIDGETS on the two LTXVAddLatentGuide nodes — if you "
        "change window_seconds/fps, recompute T and re-set both widgets.\n\n"
        "### keyframe -> window mapping\n"
        "Window k needs keyframe k+1, so you need N+1 keyframes for N "
        "anchored windows. With 3 keyframes the last anchored window re-uses "
        "the final keyframe; any iter no selector row claims falls back to "
        "the init image (no-op end anchor). target_iters is 1-BASED.\n\n"
        "### flexible keyframe source\n"
        "Default source = the 3 hand-loaded LoadImage shots. A parked "
        "VHS_LoadVideo(force_rate=1) -> EvenlySpacedKeyframes(3) -> 3x "
        "GetImageRangeFromBatch branch is BYPASSED below. To switch a "
        "keyframe to the video-extracted frame, either (a) rewire that "
        "keyframe's LTXSmartImageResize.image input to the matching "
        "GetImageRangeFromBatch.IMAGE output, or (b) route both sources "
        "through a KJNodes **LazySwitchKJ** (boolean switch, lazy — only the "
        "selected branch runs) into the resize. Un-bypass the parked branch "
        "first (mode 4 -> 0)."
    )
    note = _make_note(ed, [-700, 7100], [560, 620], note_text,
                      "END/MID anchor — design + latent_idx derivation")
    print(f"  + MarkdownNote #{note} (design + derivation)")

    # Keep the subgraph's denormalized state counters consistent.
    sg.setdefault("state", {})
    sg["state"]["lastNodeId"] = max(sg["state"].get("lastNodeId", 0),
                                    max(n["id"] for n in sg["nodes"]))
    sg["state"]["lastLinkId"] = max(sg["state"].get("lastLinkId", 0),
                                    max((l["id"] for l in sg["links"]), default=0))

    # Drop any orphan output-link caches the base may carry (fork hygiene).
    pruned = ed.prune_orphan_output_links()
    if pruned:
        print(f"  pruned {pruned} orphan output-link id(s)")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input workflow missing: {input_path}")

    if output_path.exists() and _already_migrated(WorkflowEditor(output_path)):
        print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
        return

    if dry_run:
        ed = WorkflowEditor(input_path)
        _assert_required_nodes_present(ed)
        print(f"would copy {input_path} -> {output_path}")
        _apply(ed)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    print(f"  copied {input_path} -> {output_path}")
    ed = WorkflowEditor(output_path)
    if _already_migrated(ed):
        print(f"{output_path.name}: already migrated, skipping.")
        return
    _assert_required_nodes_present(ed)
    _apply(ed)
    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print("  1. Validate: uv run --group dev python scripts/test_workflow_integrity.py "
          f"{output_path}")
    print("  2. Audit:    uv run --group dev python scripts/audit_workflows.py "
          f"{output_path}")
    print("  3. Load in ComfyUI; set the 3 keyframe LoadImage files + init LoadImage.")
    print("  4. END anchor is ACTIVE; MID is bypassed. target_iters pre-filled 1,2,3 "
          "(1-based) — re-spread per song.")
    print("  5. If you change window_seconds/fps, recompute T and re-set the END/MID "
          "guide latent_idx widgets (see the MarkdownNote).")


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--revert", action="store_true",
                    help="Delete the output staging file (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be copied/changed without writing.")
    args = ap.parse_args()

    output_path = Path(args.output)
    if args.revert:
        _revert(output_path)
        return
    _migrate(Path(args.input), output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
