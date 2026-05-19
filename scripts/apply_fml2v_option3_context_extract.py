"""apply_fml2v_option3_context_extract.

Last updated: 2026-05-19

Stages the "option 3" variant — full-res two-pass refine with
LatentContextExtract carrying prev-iter content, matching canonical's
loop-body pattern while preserving fml2v's two-pass refine quality
differentiator. Drops the spatial upsampler (incompatible with
ContextExtract's same-resolution-throughout requirement).

Wires TLO.previous_value (slot 1) → ContextExtract → AudioVideoMask
so each iter starts from real prev-iter content (last 4 latent-frames)
instead of EmptyLatent. Without this wire, pass1 had to invent the
entire middle of every iter from soft anchors, which 8 distilled steps
cannot do.

Mutations (idempotent):
  1. Add LatentContextExtract node.
  2. Wire 3 new links:
       ContextExtract.latent ← TLO.previous_value (slot 1)
       ContextExtract.overlap_latent_frames ← AudioLoopController.overlap_latent_frames (slot 6)
       AudioVideoMask.video_start_time ← AudioLoopController.overlap_seconds (slot 7)
  3. Rewire AudioVideoMask.video_latent ← ContextExtract.LATENT
     (was ← EmptyLatent_p1).
  4. Bypass nodes no longer in the latent path:
       LTXVLatentUpsampler #25 + LatentUpscaleModelLoader #182
         (spatial upsample dropped — incompatible with ContextExtract)
       LTXVImgToVideoInplaceKJ #2376 (our earlier diagnostic add,
         redundant once ContextExtract seeds content)
       LTXVSeparateAVLatent #2353 (post-pass1 separate, no longer needed)
       LTXVCropGuides #2222 (between-passes, no longer needed)
       LTXVAddGuideMulti #2182 (per-iter image injection)
       LTXVConcatAVLatent #34 (pre-pass2, no longer needed — audio
         stays in pass1's output through to pass2)
       EmptyLTXVLatentVideo #2335 (replaced by ContextExtract chain)
  5. Rewire pass2_sampler #21.latent_image ← pass1_sampler #2352.output
     (direct passthrough; pass2 re-noises to 0.85 and refines 3 steps).
  6. Un-bypass AdaIN_p1 #2347 and AdaIN_final #2365 (canonical pattern
     has both active — they normalize variance now that we have real
     seed content, not the zero-latent that NaN'd them before).
  7. Smoke loop config (2 iters via widget).

Pass2 CFGGuider's positive/negative still routes through the bypassed
CropGuides + AddGuideMulti chain via bypass passthrough (CONDITIONING
in → CONDITIONING out, same type). The re-arm patch chain stays —
comfy-aimdo offload boundary between passes still applies.

Usage:
    uv run --group dev python scripts/apply_fml2v_option3_context_extract.py
    uv run --group dev python scripts/apply_fml2v_option3_context_extract.py --revert
    uv run --group dev python scripts/apply_fml2v_option3_context_extract.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers._fml2v_helpers import (
    _add_from_template,
    apply_smoke_iters_config,
    find_by_type_and_title,
    make_argparser,
    phase5_stash,
    revert_variant,
    smoke_iters_applied,
    stage_variant,
)
from workflow_utils import WorkflowEditor, is_active

DEFAULT_OUTPUT = "internal/scratch/fml2v_var_d_audio_loop_option3.json"

AUDIO_VIDEO_MASK = 2340
PASS2_SAMPLER = 21
PASS2_UPSAMPLER = 25
UPSCALE_MODEL_LOADER = 182
PRE_PASS2_CONCAT_AV = 34
PRE_PASS2_GUIDE_MULTI = 2182
BETWEEN_CROPGUIDES = 2222
INPLACE_KJ_P1 = 2376  # added by apply_fml2v_inplace_kj_p1.py if present

BYPASS_TARGETS = [
    PASS2_UPSAMPLER, UPSCALE_MODEL_LOADER,
    PRE_PASS2_CONCAT_AV, PRE_PASS2_GUIDE_MULTI, BETWEEN_CROPGUIDES,
]

CONTEXT_EXTRACT_TITLE = "LatentContextExtract (pull last N frames of prev iter)"


def _already_toggled(ed: WorkflowEditor) -> bool:
    stash = ed.wf.get("properties", {}).get("build_fml2v_phase5") or {}
    tlo_id = stash.get("tlo")
    if tlo_id is None:
        return False
    try:
        tlo = ed.find_node(tlo_id)
        adain_p1 = ed.find_node(stash["adain_p1"])
    except (ValueError, KeyError):
        return False
    return (
        smoke_iters_applied(tlo)
        and is_active(adain_p1)
        and find_by_type_and_title(ed, "LatentContextExtract", CONTEXT_EXTRACT_TITLE) is not None
    )


def _apply(ed: WorkflowEditor) -> None:
    stash = phase5_stash(ed)
    tlo_id = stash["tlo"]
    sampler_p1_id = stash["sampler_p1"]
    empty_p1_id = stash["empty_latent_p1"]
    sep_p1_post_id = stash["separate_av_p1_post"]
    adain_p1_id = stash["adain_p1"]
    adain_final_id = stash["adain_final"]
    alc_id = ed.wf["properties"]["build_fml2v_phase2"]["audio_loop_controller"]

    extract_id = _add_from_template(
        ed, "LatentContextExtract", (1100, 2400),
        widget_values=[4],
        title=CONTEXT_EXTRACT_TITLE,
        size=(290, 100),
    )
    print(f"  + LatentContextExtract #{extract_id} (widget=[4] frames)")

    ed.add_link(tlo_id, 1, extract_id, 0, "LATENT")
    ed.add_link(alc_id, 6, extract_id, 1, "INT")
    print(f"    .latent ← TLO #{tlo_id} previous_value (slot 1)")
    print(f"    .overlap_latent_frames ← AudioLoopController #{alc_id}.overlap_latent_frames (slot 6)")

    vst_slot = WorkflowEditor.find_input_slot(ed.find_node(AUDIO_VIDEO_MASK), "video_start_time")
    ed.add_link(alc_id, 7, AUDIO_VIDEO_MASK, vst_slot, "FLOAT")
    print(f"  + AudioVideoMask #{AUDIO_VIDEO_MASK}.video_start_time ← AudioLoopController.overlap_seconds (slot 7)")

    avm = ed.find_node(AUDIO_VIDEO_MASK)
    vl_slot = WorkflowEditor.find_input_slot(avm, "video_latent")
    ed.rewire_input(AUDIO_VIDEO_MASK, vl_slot, extract_id, 0, "LATENT")
    print(f"  rewire AudioVideoMask #{AUDIO_VIDEO_MASK}.video_latent ← ContextExtract (was ← EmptyLatent_p1 #{empty_p1_id})")

    bypass_list = list(BYPASS_TARGETS) + [empty_p1_id, sep_p1_post_id]
    # InplaceKJ #2376 only exists if apply_fml2v_inplace_kj_p1.py ran first.
    if ed.has_node(INPLACE_KJ_P1):
        bypass_list.append(INPLACE_KJ_P1)
    for nid in bypass_list:
        n = ed.find_node(nid)
        prev = n.get("mode", 0)
        n["mode"] = 4
        print(f"  #{nid} {n['type']}: mode {prev} -> 4 (bypassed)")

    p2_sampler = ed.find_node(PASS2_SAMPLER)
    li_slot = WorkflowEditor.find_input_slot(p2_sampler, "latent_image")
    ed.rewire_input(PASS2_SAMPLER, li_slot, sampler_p1_id, 0, "LATENT")
    print(f"  rewire pass2_sampler #{PASS2_SAMPLER}.latent_image ← pass1_sampler #{sampler_p1_id} (direct, skip upsampler/concat chain)")

    # AdaIN un-bypass: with real seed content via ContextExtract, variance
    # is non-zero, so divide-by-near-zero NaN doesn't fire (which was why
    # AdaIN was bypassed in the first place).
    for nid in (adain_p1_id, adain_final_id):
        n = ed.find_node(nid)
        prev = n.get("mode", 0)
        n["mode"] = 0
        print(f"  #{nid} {n['type']}: mode {prev} -> 0 (un-bypassed)")

    apply_smoke_iters_config(ed, tlo_id)


def main() -> None:
    args = make_argparser(__doc__, DEFAULT_OUTPUT).parse_args()
    output_path = Path(args.output)
    if args.revert:
        revert_variant(output_path)
        return

    stage_variant(
        Path(args.input), output_path,
        apply_fn=_apply,
        already_toggled_fn=_already_toggled,
        dry_run=args.dry_run,
        variant_label="option3-context-extract",
        next_steps=[
            "1. bash start_experiment.sh default",
            f"2. Reload {output_path.name} in ComfyUI",
            "3. Wire LoadImage slots (first/middle/last) + LoadAudio",
            "4. Queue prompt — diagnostic outcomes for loop body frames (~20-37s):",
            "     coherent video w/ prev-iter continuity → confirms canonical pattern + two-pass refine works",
            "     coherent at iter 0 but degrades by iter 2-3 → ContextExtract working but trailing anchor too weak",
            "     still abstract → conditioning routing issue (CLIP path / re-arm patch chain)",
            "     NaN black                                → AdaIN still divides by near-zero; pass1 not converging",
        ],
    )


if __name__ == "__main__":
    main()
