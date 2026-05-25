"""apply_keyframe_iter_anchor.

Last updated: 2026-05-24

Generates the per-iter keyframe variant of the canonical audio-loop
workflow and writes it to
`example_workflows/audio-loop-music-video_latent_keyframe.json`
(replacing the stale prior file). Generator-style — its output IS the
shipped variant, so no F-pair audit (mirrors apply_audioreactive_loop.py).

Mechanism: insert the `LTXIterKeyframeSchedule` selector at the TOP LEVEL,
intercepting the feed into the subgraph's existing `guide_latent` input
(sg.input[8] → LTXVAddLatentGuide). No subgraph schema change — the
per-iter anchor machinery already exists; today it gets a static init
latent every iter, and we replace that with a per-iter selection.

Topology (all top-level), per keyframe replicating the init guide chain
so the keyframe latent is shape-compatible with guide_latent:
    LoadImage_kf → LTXSmartImageResize (FramePlanner dims) →
        LTXVPreprocess(18) → VAEEncode ─┐
                                        ├→ LTXIterKeyframeSchedule → #843.guide_latent
    #1617 VAEEncode (init) ─────────────┘ (fallback)
    TLO #1539.current_iteration ────────┘

Per iter the selector picks the keyframe whose `target_iters` contains
the current iteration (1-BASED — TensorLoopOpen emits 1,2,3,…); otherwise
passes the init latent through, so un-targeted iters are identical to the
no-keyframe canonical.

Anchor strength: sets `first_frame_guide_strength` (#1269) = 1.0. At 1.0
the LTXVAddLatentGuide noise_mask = max(0, 1-strength) = 0 → the guide
frame is FROZEN (hard lock), which (a) makes keyframes hold against drift
and (b) is the fast path (frozen frame is skipped in attention/FFN; <1.0
denoises it as an active token). The keyframe anchors at idx=-1 (window
tail); the overlap carries it into the next iter as frozen context, so a
keyframe change drives a smooth one-iter transition.

Mutations (idempotent):
  1. Per keyframe (N=3): LoadImage → LTXSmartImageResize → LTXVPreprocess
     → VAEEncode (chain mirrors the init guide path #444→#445→#446→#1617).
     LoadImage defaults to the init placeholder so unset keyframes don't
     crash the eager encode; un-targeted rows are never selected anyway.
  2. LTXIterKeyframeSchedule (num_keyframes=N): fallback ← #1617,
     current_iteration ← TLO #1539.out[3], keyframe_latent_K ← VAEEncode_K.
  3. Rewire invoker #843.guide_latent ← selector (was ← #1617 directly).
  4. FloatConstant #1269 first_frame_guide_strength: 0.7 → 1.0.

Usage:
    uv run --group dev python scripts/apply_keyframe_iter_anchor.py
    uv run --group dev python scripts/apply_keyframe_iter_anchor.py --revert
    uv run --group dev python scripts/apply_keyframe_iter_anchor.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import WorkflowEditor

DEFAULT_INPUT = "example_workflows/audio-loop-music-video_latent.json"
DEFAULT_OUTPUT = "example_workflows/audio-loop-music-video_latent_keyframe.json"
NUM_KEYFRAMES = 3

SUBGRAPH_INVOKER = 843
GUIDE_LATENT_SLOT = 8            # subgraph input slot for guide_latent
INIT_GUIDE_VAEENCODE = 1617      # VAEEncode "init image → guide latent" (becomes fallback)
TLO = 1539                       # TensorLoopOpen; current_iteration is out slot 3 (1-based)
TLO_CURRENT_ITERATION_SLOT = 3
FRAME_PLANNER = 1634             # LTXFramePlanner; width=out[0], height=out[1]
VIDEO_VAE_GET = 619              # GetNode "video_vae" feeding the init VAEEncode
FIRST_FRAME_GUIDE_STRENGTH = 1269  # FloatConstant feeding LTXVAddLatentGuide.strength
PREPROCESS_COMPRESSION = 18      # matches init LTXVPreprocess #446 (Lightricks default)
DEFAULT_KEYFRAME_IMAGE = "reference_image.png"  # exists in the shipped workflow → no eager crash

SELECTOR_TITLE = "LTX Iter Keyframe Schedule (per-iter keyframe anchor)"


def _find_selector(ed: WorkflowEditor) -> dict | None:
    for n in ed.wf.get("nodes", []):
        if n.get("type") == "LTXIterKeyframeSchedule" and n.get("title") == SELECTOR_TITLE:
            return n
    return None


def _already_applied(ed: WorkflowEditor) -> bool:
    sel = _find_selector(ed)
    if sel is None:
        return False
    inp = ed.find_node(SUBGRAPH_INVOKER)["inputs"][GUIDE_LATENT_SLOT]
    lk = inp.get("link")
    if lk is None:
        return False
    row = next((l for l in ed.wf.get("links", []) if l[0] == lk), None)
    return row is not None and row[1] == sel["id"]


def _make_loadimage(node_id: int, pos: list) -> dict:
    return {
        "id": node_id, "type": "LoadImage", "pos": pos, "size": [240, 310],
        "flags": {}, "order": 0, "mode": 0, "inputs": [],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": []},
            {"name": "MASK", "type": "MASK", "links": []},
        ],
        "properties": {"cnr_id": "comfy-core", "ver": "0.8.2", "Node name for S&R": "LoadImage"},
        "widgets_values": [DEFAULT_KEYFRAME_IMAGE, "image"],
    }


def _make_resize(node_id: int, pos: list) -> dict:
    return {
        "id": node_id, "type": "LTXSmartImageResize", "pos": pos, "size": [270, 336],
        "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
            {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
            {"name": "keep_proportion", "type": "BOOLEAN", "widget": {"name": "keep_proportion"}, "link": None},
            {"name": "crop_position", "type": "COMBO", "widget": {"name": "crop_position"}, "link": None},
        ],
        "outputs": [
            {"name": "image", "type": "IMAGE", "links": []},
            {"name": "width", "type": "INT", "links": []},
            {"name": "height", "type": "INT", "links": []},
        ],
        "properties": {"aux_id": "fblissjr/ComfyUI-AudioLoopHelper",
                       "cnr_id": "comfyui-audioloophelper",
                       "Node name for S&R": "LTXSmartImageResize"},
        "widgets_values": [832, 448, True, "top"],
    }


def _make_preprocess(node_id: int, pos: list) -> dict:
    return {
        "id": node_id, "type": "LTXVPreprocess", "pos": pos, "size": [270, 58],
        "flags": {}, "order": 0, "mode": 0,
        "inputs": [{"name": "image", "type": "IMAGE", "link": None}],
        "outputs": [{"name": "output_image", "type": "IMAGE", "links": []}],
        "properties": {"cnr_id": "comfy-core", "ver": "0.9.2", "Node name for S&R": "LTXVPreprocess"},
        "widgets_values": [PREPROCESS_COMPRESSION],
    }


def _make_vaeencode(node_id: int, pos: list) -> dict:
    return {
        "id": node_id, "type": "VAEEncode", "pos": pos, "size": [210, 46],
        "flags": {}, "order": 0, "mode": 0,
        "inputs": [
            {"name": "pixels", "type": "IMAGE", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
        ],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
        "properties": {"cnr_id": "comfy-core", "ver": "0.20.1", "Node name for S&R": "VAEEncode"},
        "widgets_values": [],
    }


def _make_selector_node(node_id: int) -> dict:
    """Hand-build the LTXIterKeyframeSchedule DynamicCombo node (num_keyframes=N).

    Per-row fields flatten into inputs as `num_keyframes.<field>_<i>` with
    shape 7; widget-bearing fields carry a `widget` sub-dict and a slot in
    widgets_values. Widget order = schema declaration order:
    [current_iteration, num_keyframes_combo, target_iters_1..N].
    """
    inputs = [
        {"name": "fallback_latent", "type": "LATENT", "link": None},
        {"name": "current_iteration", "type": "INT",
         "widget": {"name": "current_iteration"}, "link": None},
    ]
    for i in range(1, NUM_KEYFRAMES + 1):
        inputs.append({"label": f"keyframe_latent_{i}",
                       "name": f"num_keyframes.keyframe_latent_{i}",
                       "shape": 7, "type": "LATENT", "link": None})
        inputs.append({"name": f"num_keyframes.target_iters_{i}", "type": "STRING",
                       "widget": {"name": f"target_iters_{i}"}, "shape": 7, "link": None})
    return {
        "id": node_id, "type": "LTXIterKeyframeSchedule", "pos": [-700, 7200],
        "size": [320, 240], "flags": {}, "order": 0, "mode": 0,
        "inputs": inputs,
        "outputs": [{"name": "latent", "type": "LATENT", "links": []}],
        "properties": {"aux_id": "fblissjr/ComfyUI-AudioLoopHelper",
                       "Node name for S&R": "LTXIterKeyframeSchedule"},
        # target_iters default to consecutive 1-based iters (1,2,3,...) so the
        # keyframes FIRE out of the box. Empty defaults silently fell back to the
        # init image on every iter (looked like "only 1 keyframe in use"). User
        # re-spreads these across their song's iter count (AudioLoopPlanner.summary).
        "widgets_values": [0, str(NUM_KEYFRAMES)] + [str(i) for i in range(1, NUM_KEYFRAMES + 1)],
        "title": SELECTOR_TITLE,
    }


def _apply(ed: WorkflowEditor) -> None:
    base_x, base_y = -2000, 7000
    kf_latent_ids: list[int] = []
    for i in range(1, NUM_KEYFRAMES + 1):
        y = base_y + (i - 1) * 360
        li = ed.next_node_id(); ed.add_node(_make_loadimage(li, [base_x, y]))
        rs = ed.next_node_id(); ed.add_node(_make_resize(rs, [base_x + 280, y]))
        pp = ed.next_node_id(); ed.add_node(_make_preprocess(pp, [base_x + 580, y]))
        ve = ed.next_node_id(); ed.add_node(_make_vaeencode(ve, [base_x + 880, y]))
        # Chain: LoadImage → resize → preprocess → vaeencode (mirror init #444→#445→#446→#1617)
        ed.add_link(li, 0, rs, 0, "IMAGE")
        ed.add_link(FRAME_PLANNER, 0, rs, 1, "INT")   # width
        ed.add_link(FRAME_PLANNER, 1, rs, 2, "INT")   # height
        ed.add_link(rs, 0, pp, 0, "IMAGE")
        ed.add_link(pp, 0, ve, 0, "IMAGE")
        ed.add_link(VIDEO_VAE_GET, 0, ve, 1, "VAE")
        kf_latent_ids.append(ve)
        print(f"  + keyframe {i}: LoadImage #{li} → Resize #{rs} → Preprocess #{pp} → VAEEncode #{ve}")

    sel_id = ed.next_node_id()
    ed.add_node(_make_selector_node(sel_id))
    print(f"  + LTXIterKeyframeSchedule #{sel_id} (num_keyframes={NUM_KEYFRAMES})")

    # Wire selector: fallback(0), current_iteration(1), then [keyframe_latent_i, target_iters_i].
    ed.add_link(INIT_GUIDE_VAEENCODE, 0, sel_id, 0, "LATENT")
    ed.add_link(TLO, TLO_CURRENT_ITERATION_SLOT, sel_id, 1, "INT")
    for k, ve_id in enumerate(kf_latent_ids):
        ed.add_link(ve_id, 0, sel_id, 2 + k * 2, "LATENT")
    print(f"    fallback_latent ← #{INIT_GUIDE_VAEENCODE} (init guide)")
    print(f"    current_iteration ← TLO #{TLO}.out[{TLO_CURRENT_ITERATION_SLOT}] (1-based)")
    print(f"    keyframe_latent_1..{NUM_KEYFRAMES} ← keyframe VAEEncodes")

    ed.rewire_input(SUBGRAPH_INVOKER, GUIDE_LATENT_SLOT, sel_id, 0, "LATENT")
    print(f"  rewire invoker #{SUBGRAPH_INVOKER}.guide_latent ← #{sel_id} (was ← #{INIT_GUIDE_VAEENCODE})")

    # Hard-lock anchor + fast path: strength 1.0 (noise_mask=0 on the guide frame).
    fc = ed.find_node(FIRST_FRAME_GUIDE_STRENGTH)
    wv = list(fc.get("widgets_values") or [0.7])
    prev = wv[0]; wv[0] = 1.0; fc["widgets_values"] = wv
    print(f"  FloatConstant #{FIRST_FRAME_GUIDE_STRENGTH} first_frame_guide_strength: {prev} → 1.0 (hard lock + fast)")


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input workflow missing: {input_path}")
    if dry_run:
        ed = WorkflowEditor(input_path)
        print(f"would generate {output_path} from {input_path}")
        _apply(ed)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    print(f"  copied {input_path} -> {output_path}")
    ed = WorkflowEditor(output_path)
    if _already_applied(ed):
        print(f"{output_path.name}: already applied (from input), skipping.")
        return
    _apply(ed)
    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print("  1. bash start_experiment.sh nodynvram")
    print(f"  2. Reload {output_path.name} in ComfyUI")
    print("  3. Set the 3 keyframe LoadImage files (default = init placeholder) + init LoadImage #444")
    print("  4. target_iters pre-filled to 1,2,3 (keyframes fire on the first 3 iters).")
    print("     Re-spread per row across your song's iter count for long renders (1-BASED —")
    print("     TLO emits 1,2,3,…), e.g. target_iters_1='1' _2='3' _3='5'. Check")
    print("     AudioLoopPlanner.summary for your iter count.")
    print("  5. Queue. Un-targeted iters use the init image (identical to canonical).")
    print("  NOTE: if the DynamicCombo keyframe rows don't expand in the UI, delete +")
    print("  re-add LTXIterKeyframeSchedule from the node menu and rewire (slot indices")
    print("  bake at save time).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--revert", action="store_true",
                    help="Restore the output file from the canonical input (un-applied state).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    if args.revert:
        # Output is a tracked shipped variant; revert = regenerate-without-keyframes
        # is meaningless. Restore from git instead. Just report.
        print("--revert: output is a tracked shipped variant; restore via "
              "`git checkout -- <output>` or re-run without --revert to regenerate.")
        return
    _migrate(Path(args.input), out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
