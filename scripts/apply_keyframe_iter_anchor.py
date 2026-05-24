"""apply_keyframe_iter_anchor.

Last updated: 2026-05-20

Stages a per-iter keyframe variant of the canonical audio-loop workflow
(`audio-loop-music-video_latent.json`) by inserting the
`LTXIterKeyframeSchedule` selector at the TOP LEVEL — intercepting the
feed into the subgraph's existing `guide_latent` input. No subgraph
schema change: `guide_latent` already exists (sg.input[8] →
`LTXVAddLatentGuide`), today fed a static init-image latent for every
iter. We replace that static feed with a per-iter selection.

Topology (all top-level):
    LoadImage_kf_N → VAEEncode_kf_N ─┐
                                     ├→ LTXIterKeyframeSchedule ─→ invoker #843.guide_latent
    #1617 VAEEncode (init) ──────────┘ (fallback)
    TLO #1539.current_iteration ─────┘

Per iter, the selector picks the keyframe whose `target_iters` contains
the current iteration; otherwise passes the init latent through (so
un-targeted iters behave identically to the no-keyframe canonical). The
selected latent feeds the proven `LTXVAddLatentGuide` soft anchor inside
the subgraph — no new in-loop nodes, no VAE in the loop.

Mutations (idempotent):
  1. New `GetNode("video_vae")` for the keyframe encoders.
  2. N (=3) `LoadImage` + `VAEEncode` keyframe-encoder chains (placeholders;
     user picks image files in the UI).
  3. New `LTXIterKeyframeSchedule` (num_keyframes=N):
       fallback_latent ← #1617 VAEEncode (init guide)
       current_iteration ← TLO #1539.current_iteration (out slot 3)
       keyframe_latent_K ← VAEEncode_kf_K
       target_iters_K default "" (user sets, e.g. "0", "2", "4")
  4. Rewire invoker #843.guide_latent ← selector (was ← #1617 directly).

Output: internal/scratch/audio-loop-music-video_latent_iterkeyframe.json
(gitignored draft; promote to example_workflows/ after a render gate).

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
DEFAULT_OUTPUT = "internal/scratch/audio-loop-music-video_latent_iterkeyframe.json"
NUM_KEYFRAMES = 3

SUBGRAPH_INVOKER = 843
GUIDE_LATENT_SLOT = 8           # subgraph input slot for guide_latent
INIT_GUIDE_VAEENCODE = 1617     # VAEEncode "init image → guide latent" (becomes fallback)
TLO = 1539                      # TensorLoopOpen; current_iteration is out slot 3
TLO_CURRENT_ITERATION_SLOT = 3

SELECTOR_TITLE = "LTX Iter Keyframe Schedule (per-iter keyframe anchor)"


def _find_selector(ed: WorkflowEditor) -> dict | None:
    for n in ed.wf.get("nodes", []):
        if n.get("type") == "LTXIterKeyframeSchedule" and n.get("title") == SELECTOR_TITLE:
            return n
    return None


def _already_applied(ed: WorkflowEditor) -> bool:
    if _find_selector(ed) is None:
        return False
    # guide_latent must point at the selector, not the raw init VAEEncode.
    try:
        invoker = ed.find_node(SUBGRAPH_INVOKER)
    except ValueError:
        return False
    inp = invoker["inputs"][GUIDE_LATENT_SLOT]
    lk = inp.get("link")
    if lk is None:
        return False
    row = next((l for l in ed.wf.get("links", []) if l[0] == lk), None)
    return row is not None and row[1] != INIT_GUIDE_VAEENCODE


def _make_selector_node(node_id: int) -> dict:
    """Hand-build the LTXIterKeyframeSchedule DynamicCombo node (num_keyframes=N).

    Mirrors the KJNodes DynamicCombo serialization: per-row fields are
    flattened into inputs as `num_keyframes.<field>_<i>` with shape 7;
    widget-bearing fields also carry a `widget` sub-dict and a slot in
    widgets_values. Widget order follows schema declaration:
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
        "id": node_id,
        "type": "LTXIterKeyframeSchedule",
        "pos": [1430, 1000],
        "size": [320, 240],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": [{"name": "latent", "type": "LATENT", "links": []}],
        "properties": {
            "aux_id": "fblissjr/ComfyUI-AudioLoopHelper",
            "Node name for S&R": "LTXIterKeyframeSchedule",
        },
        "widgets_values": [0, str(NUM_KEYFRAMES)] + [""] * NUM_KEYFRAMES,
        "title": SELECTOR_TITLE,
    }


def _make_loadimage(node_id: int, pos: list, default_name: str) -> dict:
    return {
        "id": node_id, "type": "LoadImage", "pos": pos, "size": [240, 310],
        "flags": {}, "order": 0, "mode": 0, "inputs": [],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": []},
            {"name": "MASK", "type": "MASK", "links": []},
        ],
        "properties": {"cnr_id": "comfy-core", "ver": "0.8.2", "Node name for S&R": "LoadImage"},
        "widgets_values": [default_name, "image"],
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


def _apply(ed: WorkflowEditor) -> None:
    # 1. Get_video_vae for the keyframe encoders.
    get_vae_id = ed.next_node_id()
    ed.add_node(WorkflowEditor.make_get_node(
        get_vae_id, "video_vae", "VAE", [1430, 700], title="Get_video_vae (keyframes)",
    ))
    print(f"  + GetNode #{get_vae_id} video_vae (keyframe encoders)")

    # 2. N keyframe LoadImage + VAEEncode chains.
    kf_latent_ids: list[int] = []
    for i in range(1, NUM_KEYFRAMES + 1):
        li_id = ed.next_node_id()
        ed.add_node(_make_loadimage(li_id, [1080, 600 + (i - 1) * 360], f"keyframe_{i}.png"))
        ve_id = ed.next_node_id()
        ed.add_node(_make_vaeencode(ve_id, [1360, 600 + (i - 1) * 360]))
        ed.add_link(li_id, 0, ve_id, 0, "IMAGE")   # LoadImage.IMAGE → VAEEncode.pixels
        ed.add_link(get_vae_id, 0, ve_id, 1, "VAE")
        kf_latent_ids.append(ve_id)
        print(f"  + keyframe {i}: LoadImage #{li_id} → VAEEncode #{ve_id}")

    # 3. Selector node.
    sel_id = ed.next_node_id()
    ed.add_node(_make_selector_node(sel_id))
    print(f"  + LTXIterKeyframeSchedule #{sel_id} (num_keyframes={NUM_KEYFRAMES})")

    # 4. Wire selector inputs. Slot order: fallback_latent(0),
    #    current_iteration(1), then per-row [keyframe_latent_i, target_iters_i].
    ed.add_link(INIT_GUIDE_VAEENCODE, 0, sel_id, 0, "LATENT")        # fallback
    ed.add_link(TLO, TLO_CURRENT_ITERATION_SLOT, sel_id, 1, "INT")   # current_iteration
    for k, ve_id in enumerate(kf_latent_ids):
        kf_slot = 2 + k * 2  # keyframe_latent_i input slot (target_iters interleaved after)
        ed.add_link(ve_id, 0, sel_id, kf_slot, "LATENT")
    print(f"    fallback_latent ← #{INIT_GUIDE_VAEENCODE} (init guide)")
    print(f"    current_iteration ← TLO #{TLO}.out[{TLO_CURRENT_ITERATION_SLOT}]")
    print(f"    keyframe_latent_1..{NUM_KEYFRAMES} ← keyframe VAEEncodes")

    # 5. Rewire invoker.guide_latent ← selector (was ← init VAEEncode directly).
    ed.rewire_input(SUBGRAPH_INVOKER, GUIDE_LATENT_SLOT, sel_id, 0, "LATENT")
    print(f"  rewire invoker #{SUBGRAPH_INVOKER}.guide_latent ← #{sel_id} (was ← #{INIT_GUIDE_VAEENCODE})")


def _migrate(input_path: Path, output_path: Path, dry_run: bool) -> None:
    if not input_path.exists():
        raise SystemExit(f"Input workflow missing: {input_path}")
    if output_path.exists() and input_path != output_path:
        if _already_applied(WorkflowEditor(output_path)):
            print(f"{output_path.name}: already applied, skipping. Run --revert to reset.")
            return
    if dry_run:
        ed = WorkflowEditor(input_path)
        print(f"would copy {input_path} -> {output_path}")
        print("would apply keyframe-iter-anchor ops:")
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
    print(f"  1. bash start_experiment.sh nodynvram")
    print(f"  2. Reload {output_path.name} in ComfyUI")
    print(f"  3. Set the 3 keyframe LoadImage files + the init LoadImage #444")
    print(f"  4. On LTXIterKeyframeSchedule set target_iters per row, e.g.:")
    print(f"       target_iters_1 = '0'   target_iters_2 = '2'   target_iters_3 = '4'")
    print(f"     (check AudioLoopPlanner.summary for your song's iter count)")
    print(f"  5. Queue. Un-targeted iters use the init image (identical to canonical).")
    print(f"  NOTE: if the DynamicCombo node doesn't expand its keyframe rows in the")
    print(f"  UI, delete + re-add LTXIterKeyframeSchedule from the node menu and rewire")
    print(f"  (DynamicCombo slot indices bake at save time).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    if args.revert:
        if out.exists():
            out.unlink()
            print(f"removed {out}")
        else:
            print(f"{out} does not exist; nothing to revert.")
        return
    _migrate(Path(args.input), out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
