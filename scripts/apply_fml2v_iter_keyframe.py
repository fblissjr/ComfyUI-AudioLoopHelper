"""apply_fml2v_iter_keyframe.

Last updated: 2026-05-19

Stages the full option3 topology (ContextExtract + bypass upsampler +
un-bypass AdaIN) PLUS the new LTXIterKeyframeSchedule inserted between
AudioVideoMask and AddLatentGuide_frame0 PLUS init-render reduction
(AddGuideMulti #2221 collapsed from 3 keyframes to 1).

Why combined: the iter-keyframe schedule depends on option3's
ContextExtract chain (otherwise iters degrade regardless of
re-anchoring). Init-render reduction removes the "first 19.88s is a
slideshow through first/mid/last" effect — first/mid/last become
per-iter keyframes scheduled via the new node instead.

Usage:
    uv run --group dev python scripts/apply_fml2v_iter_keyframe.py
    uv run --group dev python scripts/apply_fml2v_iter_keyframe.py --revert
    uv run --group dev python scripts/apply_fml2v_iter_keyframe.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers._fml2v_helpers import (
    _add_from_template,
    find_by_type_and_title,
    make_argparser,
    phase5_stash,
    revert_variant,
    stage_variant,
)
from apply_fml2v_option3_context_extract import _apply as _apply_option3
from workflow_utils import WorkflowEditor

DEFAULT_OUTPUT = "internal/scratch/fml2v_var_d_audio_loop_iter_keyframe.json"

INIT_RENDER_ADD_GUIDE_MULTI = 2221  # init render's AddGuideMulti — collapsed from 3→1 keyframes
GET_FIRSTFRAME = 2220   # loop-body scope
GET_LASTFRAME = 2106    # loop-body scope
GET_MIDDLEFRAME = 2173  # top-level — only middleframe GetNode in the workflow

ITER_KEYFRAME_TITLE = "LTX Iter Keyframe Schedule (first/mid/last)"


def _already_toggled(ed: WorkflowEditor) -> bool:
    """Check iter-keyframe-specific markers only. Smoke vs full TLO wiring
    state is mode-dependent and isn't a stable toggle marker — revert + re-apply
    to switch modes."""
    stash = ed.wf.get("properties", {}).get("build_fml2v_phase5") or {}
    if stash.get("tlo") is None:
        return False
    try:
        guide_multi = ed.find_node(INIT_RENDER_ADD_GUIDE_MULTI)
    except ValueError:
        return False
    return (
        find_by_type_and_title(ed, "LTXIterKeyframeSchedule", ITER_KEYFRAME_TITLE) is not None
        and (guide_multi.get("widgets_values") or [None])[0] == "1"
    )


def _collapse_init_render_multi_to_single(ed: WorkflowEditor) -> None:
    """Reduce LTXVAddGuideMulti #2221 from 3 keyframes to 1.

    Promote-when-2nd-caller note: this hardcodes the LTXVAddGuideMulti
    widget shape `['N', frame_idx_1, strength_1, ...]`. A node-side
    schema reorder would silently corrupt this. If another fml2v variant
    needs to collapse a DynamicCombo, lift to `_fml2v_helpers` with
    generic widget-shape detection at that time.
    """
    n = ed.find_node(INIT_RENDER_ADD_GUIDE_MULTI)
    drop_names = {"num_guides.image_2", "num_guides.frame_idx_2", "num_guides.strength_2",
                  "num_guides.image_3", "num_guides.strength_3"}
    live_link_ids = {lk[0] for lk in ed.wf.get("links", [])}
    kept_inputs = []
    for inp in n.get("inputs", []):
        if inp.get("name") in drop_names:
            lk_id = inp.get("link")
            if lk_id is not None and lk_id in live_link_ids:
                ed.remove_link(lk_id)
            continue
        kept_inputs.append(inp)
    n["inputs"] = kept_inputs

    wv = list(n.get("widgets_values") or [])
    n["widgets_values"] = ["1", wv[1], wv[2]] if len(wv) >= 3 else ["1", 0, 1.0]
    print(f"  AddGuideMulti #{INIT_RENDER_ADD_GUIDE_MULTI} (init render): num_guides '3' -> '1' (drop image_2 + image_3)")


def _apply(ed: WorkflowEditor, *, smoke: bool = False) -> None:
    _apply_option3(ed, smoke=smoke)

    stash = phase5_stash(ed)
    tlo_id = stash["tlo"]
    av_mask_id = stash["av_mask"]
    add_guide_frame0_id = stash["add_latent_guide_frame0"]

    get_vae_id = ed.next_node_id()
    ed.add_node(WorkflowEditor.make_get_node(
        get_vae_id, "vae", "VAE", [1700, 2400], title="Get_vae (iter keyframe)",
    ))

    schedule_id = _add_from_template(
        ed, "LTXIterKeyframeSchedule", (1950, 2400),
        size=(340, 280),
        title=ITER_KEYFRAME_TITLE,
    )
    print(f"  + LTXIterKeyframeSchedule #{schedule_id} (num_images=3, target_iters defaults '')")

    ed.add_link(av_mask_id, 0, schedule_id, 0, "LATENT")
    ed.add_link(tlo_id, 3, schedule_id, 1, "INT")
    ed.add_link(get_vae_id, 0, schedule_id, 2, "VAE")
    ed.add_link(GET_FIRSTFRAME, 0, schedule_id, 3, "IMAGE")
    ed.add_link(GET_MIDDLEFRAME, 0, schedule_id, 6, "IMAGE")
    ed.add_link(GET_LASTFRAME, 0, schedule_id, 9, "IMAGE")
    print(f"    .latent ← #{av_mask_id}.video_latent")
    print(f"    .current_iteration ← TLO #{tlo_id}.current_iteration (slot 3)")
    print(f"    .vae ← #{get_vae_id} (new GetNode \"vae\")")
    print(f"    .image_1 ← #{GET_FIRSTFRAME} Get_firstframe")
    print(f"    .image_2 ← #{GET_MIDDLEFRAME} Get_middleframe")
    print(f"    .image_3 ← #{GET_LASTFRAME} Get_lastframe")

    # Rewire so AddLatentGuide_frame0's latent input flows through the new
    # schedule node — schedule's passthrough (empty target_iters) means
    # zero behavior change until the user configures it.
    frame0_guide = ed.find_node(add_guide_frame0_id)
    latent_slot = WorkflowEditor.find_input_slot(frame0_guide, "latent")
    ed.rewire_input(add_guide_frame0_id, latent_slot, schedule_id, 0, "LATENT")
    print(f"  rewire #{add_guide_frame0_id}.latent ← #{schedule_id} (was ← #{av_mask_id})")

    _collapse_init_render_multi_to_single(ed)


def main() -> None:
    ap = make_argparser(__doc__, DEFAULT_OUTPUT)
    ap.add_argument("--smoke", action="store_true",
                    help="Keep the 2-iter smoke config (TLO unwired + widget=2). "
                         "Default re-wires AudioLoopPlanner.total_iterations → TLO "
                         "for full-length renders (F5 canonical).")
    args = ap.parse_args()
    output_path = Path(args.output)
    if args.revert:
        revert_variant(output_path)
        return

    next_steps_smoke = [
        "1. bash start_experiment.sh nodynvram   # avoid offload-reload bug",
        f"2. Reload {output_path.name} in ComfyUI",
        "3. Wire LoadImage slots (first/middle/last) + LoadAudio",
        "4. Set target_iters_N on LTXIterKeyframeSchedule (target_iters_1='0' anchors iter 0).",
        "5. Queue (2 iters smoke).",
    ]
    next_steps_full = [
        "1. bash start_experiment.sh nodynvram   # avoid offload-reload bug",
        f"2. Reload {output_path.name} in ComfyUI",
        "3. Wire LoadImage slots (first/middle/last) + LoadAudio",
        "4. Check AudioLoopPlanner.summary in UI to see computed iter count",
        "   (= ceil((audio_duration − init_render_seconds) / stride_seconds)).",
        "5. Set target_iters_N on LTXIterKeyframeSchedule based on that count:",
        "     target_iters_1 = '0'              (first iter)",
        "     target_iters_2 = '<midpoint>'     (e.g. '2' for 5-iter song)",
        "     target_iters_3 = '<last>'         (e.g. '4' for 5-iter song)",
        "   Empty target_iters = no-op for that row.",
        "6. Queue (full-length render).",
    ]

    stage_variant(
        Path(args.input), output_path,
        apply_fn=lambda ed: _apply(ed, smoke=args.smoke),
        already_toggled_fn=_already_toggled,
        dry_run=args.dry_run,
        variant_label="iter-keyframe (smoke)" if args.smoke else "iter-keyframe (full)",
        next_steps=next_steps_smoke if args.smoke else next_steps_full,
    )


if __name__ == "__main__":
    main()
