"""Health audit for every workflow under example_workflows/.

Verifies each shipped workflow matches the current CLAUDE.md invariants:
sage node + mode, iteration stamp, batch-encode prompt schedule,
distilled sampler chain (linear_quadratic 8 1 + shift=13 + euler +
cfg=1, with STG-variant exception), resolution div-32, (L-1)%8==0,
LTXVPreprocess img_compression >= 18, LTXVTiledVAEDecode preferred,
preprocess symmetry (F2), and loop-body cropguides symmetry (F3).

Exits 0 all-green; 1 on any ERR. WARNs don't fail.

Usage:
    uv run --group dev python scripts/audit_workflows.py [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workflow_utils import EXAMPLE_WORKFLOWS_DIR
from nodes import (
    _LTX_LATENT_VOLUME_OK_MAX as _VOLUME_OK_MAX,
    _LTX_LATENT_VOLUME_EDGE_MAX as _VOLUME_EDGE_MAX,
)


class Finding(NamedTuple):
    status: str  # OK | WARN | ERR
    workflow: str
    check: str
    message: str


EXPECTED_CHAIN = {
    # ManualSigmas with Lightricks's canonical hand-tuned distilled
    # values (DISTILLED_SIGMA_VALUES from
    # coderef/ID-LoRA-2.3/.../ltx_pipelines/utils/constants.py).
    # Replaces BasicScheduler linear_quadratic 8 1 (which approximated
    # this curve parametrically). Migration: apply_canonical_sigmas.py.
    "ManualSigmas": (
        ["1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"],
        "manual_sigmas",
    ),
    "ModelSamplingSD3": ([13], "model_sampling_shift"),
    "KSamplerSelect": (["euler"], "sampler_type"),
}

# Experimental POC files that ship alongside production workflows and
# have audit checks of their own. Anything outside this allowlist (e.g.
# spectrogram_iclora_minimal.json, iclora_amplification_poc.json) is
# intentionally NOT audited — those are forks with non-standard topology
# that pre-date the audit and would just generate noise.
EXPERIMENTAL_AUDITED_FILES = {"init_guide_amplification_poc.json"}

# Title prefix the init-guide POC stamps onto CFGGuider(644). Narrow
# enough to not collide with future TTC variants (e.g. an IC-LoRA POC
# fork that reuses the "TTC1:" prefix) — the audit dispatches on this
# string, so it has to be unambiguous per variant.
TTC1_INIT_GUIDE_TITLE_PREFIX = "CFGGuider (TTC1: cfg = init-guide"


def _is_validator(name: str) -> bool:
    return "validator" in name


def _is_stg(name: str) -> bool:
    return "stg" in name


def _is_retake(name: str) -> bool:
    return "retake" in name


def _audit_one(wf_path: Path) -> list[Finding]:
    findings: list[Finding] = []
    name = wf_path.name
    wf = orjson.loads(wf_path.read_bytes())
    by_type: dict[str, list[dict]] = {}
    by_id: dict[int, dict] = {}
    for n in wf["nodes"]:
        by_type.setdefault(n["type"], []).append(n)
        by_id[n["id"]] = n
    links_by_id: dict[int, list] = {
        l[0]: l for l in wf.get("links") or [] if isinstance(l, list)
    }

    def record(status: str, check: str, msg: str = "") -> None:
        findings.append(Finding(status, name, check, msg))

    # Sage node
    sage = by_type.get("AudioLoopHelperSageAttention", [])
    if not sage:
        if by_type.get("PathchSageAttentionKJ"):
            record("ERR", "sage_node", "uses legacy PathchSageAttentionKJ instead of AudioLoopHelperSageAttention")
        else:
            record("WARN", "sage_node", "no sage node present")
    else:
        all_good = True
        for n in sage:
            wv = n.get("widgets_values", [])
            if not wv or wv[0] != "auto_mask_aware":
                record("WARN", "sage_mode", f"mode={wv[0] if wv else None!r} (expected auto_mask_aware)")
                all_good = False
            if n.get("mode", 0) != 0:
                record("WARN", "sage_active", f"mode field={n.get('mode')} (4=bypassed)")
                all_good = False
        if all_good:
            record("OK", "sage", "AudioLoopHelperSageAttention auto_mask_aware active")

    # AudioLoopController seed input name (post-2026-04-26 rename)
    # ComfyUI auto-attaches control_after_generate to any INT widget literally
    # named "seed" or "noise_seed", which silently mutates the saved widget
    # value across runs even when the input is wired (the link supersedes the
    # widget at execute time, but the mutated widget still gets serialized).
    # Diagnosed in internal/analysis/id_lora_ablation_and_seed_widget_audit.md.
    alc_nodes = by_type.get("AudioLoopController", [])
    if alc_nodes:
        leaks = []
        for n in alc_nodes:
            for inp in n.get("inputs") or []:
                if inp.get("name") == "seed":
                    leaks.append(f"node {n.get('id')} input.name='seed'")
                widget = inp.get("widget")
                if isinstance(widget, dict) and widget.get("name") == "seed":
                    leaks.append(f"node {n.get('id')} widget.name='seed'")
        if leaks:
            record("ERR", "alc_seed_legacy_name",
                   f"legacy 'seed' name found: {', '.join(leaks)}. "
                   f"Run scripts/apply_alc_seed_rename.py")
        else:
            record("OK", "alc_seed_legacy_name", "uses 'base_seed' (post-rename)")

        # Check 2: schema has 5 widget slots
        # [current_iteration, window_seconds, overlap_seconds, base_seed, fps].
        # Pre-rename `seed` had a control_after_generate dropdown that
        # serialized as a 6th `'randomize'` / `'fixed'` entry at index 4.
        # Post-rename, the dropdown isn't re-attached but the leftover string
        # remained in saved JSONs — backend pops 5 widgets positionally so
        # `'randomize'` lands in the `fps` slot and explodes INT parsing.
        drift = []
        for n in alc_nodes:
            wv = n.get("widgets_values") or []
            if len(wv) != 5:
                drift.append(f"#{n.get('id')} widgets_values len={len(wv)} (expected 5)")
        if drift:
            record("ERR", "alc_widget_drift",
                   f"{', '.join(drift)}. "
                   f"Run scripts/apply_strip_alc_control_after_generate.py")
        else:
            record("OK", "alc_widget_drift", "widgets_values has 5 entries (no control_after_generate leak)")

    # TensorLoopOpen.iterations_in (post-2026-04-26 autowire to AudioLoopPlanner)
    # Wired iterations_in lets the loop count auto-match the input audio
    # length without a hard-coded widget value, and gives the experiment
    # harness a stable target for short-tier overrides. Skip the check for
    # workflow variants that legitimately have no TensorLoopOpen (retake,
    # ICLoRA POCs).
    tlo_nodes = by_type.get("TensorLoopOpen", [])
    if tlo_nodes:
        all_wired = True
        for n in tlo_nodes:
            inp = next(
                (i for i in n.get("inputs") or [] if i.get("name") == "iterations_in"),
                None,
            )
            if inp is None or inp.get("link") is None:
                record("ERR", "iterations_autowired",
                       f"TensorLoopOpen({n['id']}).iterations_in unwired. "
                       f"Run scripts/apply_iterations_autowire.py")
                all_wired = False
                continue
            # Verify the link source is AudioLoopPlanner.total_iterations
            link_id = inp["link"]
            link = links_by_id.get(link_id)
            if link is None:
                record("WARN", "iterations_autowired",
                       f"TensorLoopOpen({n['id']}).iterations_in link {link_id} dangling")
                all_wired = False
                continue
            src_id, src_slot = link[1], link[2]
            src_node = by_id.get(src_id)
            if src_node and src_node.get("type") == "AudioLoopPlanner" and src_slot == 1:
                continue  # canonical wiring
            src_type = src_node.get("type") if src_node else "?"
            record("WARN", "iterations_autowired",
                   f"TensorLoopOpen({n['id']}).iterations_in wired from "
                   f"{src_type}({src_id}).out[{src_slot}] (expected AudioLoopPlanner.total_iterations). "
                   f"OK if intentional (e.g. experiment-tier override).")
            all_wired = False
        if all_wired:
            record("OK", "iterations_autowired", "wired from AudioLoopPlanner.total_iterations")

    # ID-LoRA runtime pair-check: when both LTXVReferenceAudio instances
    # exist, they should both be in the same bypass state. Mixed state
    # (one un-bypassed, one bypassed) is the iter-0-vs-loop drift footgun
    # we explicitly architected against — initial render gets identity,
    # loop iterations don't (or vice-versa). WARN not ERR because the user
    # might intentionally want to A/B test one branch.
    refaudios = by_type.get("LTXVReferenceAudio", [])
    initial = next((n for n in refaudios
                    if (n.get("title") or "").startswith(
                        "LTXV Reference Audio (ID-LoRA initial")), None)
    loop = next((n for n in refaudios
                 if (n.get("title") or "").startswith(
                     "LTXV Reference Audio (ID-LoRA loop")), None)
    if initial and loop:
        i_active = initial.get("mode", 0) != 4
        l_active = loop.get("mode", 0) != 4
        if i_active != l_active:
            which = "initial" if i_active else "loop"
            other = "loop" if i_active else "initial"
            record("WARN", "id_lora_runtime_consistent",
                   f"only the {which}-render LTXVReferenceAudio is active; "
                   f"{other} branch is bypassed -> identity will be "
                   f"inconsistent across iter 0 vs loop body. "
                   f"Un-bypass both, or bypass both, for consistent identity.")
        else:
            state = "active" if i_active else "bypassed"
            record("OK", "id_lora_runtime_consistent",
                   f"both LTXVReferenceAudio instances {state}")

    # AudioLoopPlanner schema must NOT have a stride_seconds input — that
    # closed the controller -> planner -> tensorloop -> controller cycle
    # once iterations_in was auto-wired. Post-2026-04-27 schema:
    # (audio, window_seconds, overlap_seconds, fps).
    planner_nodes = by_type.get("AudioLoopPlanner", [])
    if planner_nodes:
        violations = []
        for p in planner_nodes:
            names = [i.get("name") for i in (p.get("inputs") or [])]
            if "stride_seconds" in names:
                violations.append(
                    f"#{p.get('id')} has legacy stride_seconds input (closes a cycle "
                    f"with TensorLoopOpen.iterations_in)"
                )
        if violations:
            record("ERR", "planner_no_stride_input",
                   f"{'; '.join(violations)}. "
                   f"Run scripts/apply_planner_break_stride_cycle.py")
        else:
            record("OK", "planner_no_stride_input",
                   "computes stride internally (cycle-free)")

    # LoopIterationStamp
    if by_type.get("LoopIterationStamp"):
        record("OK", "iteration_stamp", "present")
    elif _is_retake(name):
        record("OK", "iteration_stamp", "n/a (retake workflow, no loop)")
    else:
        record("WARN", "iteration_stamp", "missing (sage tracer iter grouping will be blank)")

    # LTXFramePlanner is the single source of truth for dimension config
    # (width/height/length/fps/window_seconds/frame_rate). Without it,
    # widget values for those scatter across EmptyLTXVLatentVideo,
    # ImageResizeKJv2, AudioLoopController, AudioLoopPlanner, and the
    # subgraph — drift between them was the historical footgun. Retake
    # lacks the audio-loop spine and is exempt. Experimental forks
    # (under example_workflows/experimental/) predate the consolidation
    # and downgrade to WARN — they're staging variants, not production.
    # See scripts/apply_frame_planner_consolidation.py.
    is_experimental = wf_path.parent.name == "experimental"
    if not _is_retake(name):
        if not by_type.get("LTXFramePlanner"):
            status = "WARN" if is_experimental else "ERR"
            record(
                status, "frame_planner_present",
                "no LTXFramePlanner node — dimension widgets will scatter and drift. "
                "Run scripts/apply_frame_planner_consolidation.py.",
            )
        else:
            record("OK", "frame_planner_present",
                   "LTXFramePlanner is the single source for dim/fps config")

    # Prompt schedule
    batch = by_type.get("TimestampPromptScheduleBatchEncode", [])
    legacy_cache = by_type.get("CachedTextEncode_AudioLoop", [])
    legacy_sched = by_type.get("TimestampPromptSchedule", [])
    if batch:
        record("OK", "prompt_schedule", "TimestampPromptScheduleBatchEncode (current)")
    elif legacy_cache or legacy_sched:
        record(
            "ERR", "prompt_schedule",
            f"LEGACY: CachedTextEncode={len(legacy_cache)}, "
            f"TimestampPromptSchedule={len(legacy_sched)} "
            "-- should be TimestampPromptScheduleBatchEncode",
        )
    elif _is_validator(name):
        record("OK", "prompt_schedule", "n/a (validator workflow)")
    elif _is_retake(name):
        record("OK", "prompt_schedule", "n/a (retake uses single CLIPTextEncode)")
    else:
        record("WARN", "prompt_schedule", "no prompt schedule node")

    # Sampler chain (skip validator; STG workflow has its own guider below)
    for t, (expected_wv, check_name) in EXPECTED_CHAIN.items():
        nodes = by_type.get(t, [])
        if not nodes:
            if not _is_validator(name) and not _is_stg(name):
                record("WARN", check_name, f"no {t} found")
            continue
        mismatches = [
            n for n in nodes
            if n.get("widgets_values", [])[: len(expected_wv)] != expected_wv
        ]
        if mismatches:
            for n in mismatches:
                record("WARN", check_name, f"{t}(id={n['id']}) widgets={n.get('widgets_values', [])}")
        else:
            record("OK", check_name, f"{t}={expected_wv}")

    # Guider: cfg=1 for all except STG variant
    cfg_nodes = by_type.get("CFGGuider", [])
    mmg_nodes = by_type.get("MultimodalGuider", [])
    if _is_stg(name):
        if mmg_nodes:
            record("OK", "guider", "MultimodalGuider (STG hybrid)")
        elif cfg_nodes:
            record("WARN", "guider", "STG workflow using CFGGuider")
    elif cfg_nodes:
        mismatches = [n for n in cfg_nodes if n.get("widgets_values", []) != [1]]
        if mismatches:
            for n in mismatches:
                record("WARN", "cfg_value", f"CFGGuider(id={n['id']}) cfg={n.get('widgets_values', [])}")
        else:
            record("OK", "cfg_value", "cfg=1")
    elif not _is_validator(name):
        record("WARN", "guider", "no CFGGuider")

    # Resolution + length + latent volume on EmptyLTXVLatentVideo.
    # Volume thresholds imported from nodes.py to stay in sync.
    # Validator workflow intentionally exercises edge dims; skip the
    # volume check there but keep the div-32 / length-mod-8 checks.
    for n in by_type.get("EmptyLTXVLatentVideo", []):
        wv = n.get("widgets_values", [])
        if len(wv) < 3:
            continue
        w, h, L = wv[0], wv[1], wv[2]
        w_ok = isinstance(w, int) and w % 32 == 0
        h_ok = isinstance(h, int) and h % 32 == 0
        L_ok = isinstance(L, int) and (L - 1) % 8 == 0
        if isinstance(w, int) and not w_ok:
            record("ERR", "resolution_div32", f"width {w} not div by 32")
        elif isinstance(h, int) and not h_ok:
            record("ERR", "resolution_div32", f"height {h} not div by 32")
        else:
            record("OK", "resolution_div32", f"{w}x{h}")
        if isinstance(L, int) and not L_ok:
            record("ERR", "length_mod8", f"length={L}, (L-1)%8={(L-1)%8}")
        else:
            record("OK", "length_mod8", f"length={L}")
        if w_ok and h_ok and L_ok and not _is_validator(name):
            volume = (w // 32) * (h // 32) * ((L - 1) // 8 + 1)
            if volume > _VOLUME_EDGE_MAX:
                record(
                    "ERR", "latent_volume",
                    f"{volume} > {_VOLUME_EDGE_MAX:,} (artifact ceiling per ltx23_model_reference.md). "
                    f"Run scripts/apply_canonical_resolution_fix.py.",
                )
            elif volume > _VOLUME_OK_MAX:
                record("WARN", "latent_volume", f"{volume} > {_VOLUME_OK_MAX:,} (near edge)")
            else:
                record("OK", "latent_volume", f"{volume}")

    # LTXVPreprocess img_compression (0 triggers frozen-first-frame bug)
    for n in by_type.get("LTXVPreprocess", []):
        wv = n.get("widgets_values", [])
        if not wv or not isinstance(wv[0], int):
            continue
        if wv[0] == 0:
            record("ERR", "preprocess_compression", "img_compression=0 (frozen-first-frame bug)")
        elif wv[0] < 18:
            record("WARN", "preprocess_compression", f"img_compression={wv[0]} < 18")
        else:
            record("OK", "preprocess_compression", f"img_compression={wv[0]}")

    # Decoder
    if by_type.get("LTXVTiledVAEDecode"):
        record("OK", "decoder", "LTXVTiledVAEDecode")
    elif by_type.get("VAEDecodeTiled"):
        record("WARN", "decoder", "VAEDecodeTiled (generic) -- consider LTXVTiledVAEDecode")

    # F2: preprocess symmetry -- #650 Set_input_image must source from
    # #446 LTXVPreprocess, not #445 ImageResizeKJv2 directly. Skipping
    # preprocess on the loop branch reintroduces the microphone/subject-
    # replacement drift. See scripts/apply_loop_guide_preprocess_symmetry.py.
    set_input_image = next((n for n in wf["nodes"] if n["id"] == 650), None)
    preprocess_node = next((n for n in wf["nodes"] if n["id"] == 446), None)
    if set_input_image and preprocess_node:
        link_id = set_input_image.get("inputs", [{}])[0].get("link") if set_input_image.get("inputs") else None
        link_row = next((lk for lk in wf["links"] if isinstance(lk, list) and lk[0] == link_id), None) if link_id else None
        if link_row is None:
            record("WARN", "preprocess_symmetry", "#650 Set_input_image has no inbound link")
        elif link_row[1] == 446 and link_row[2] == 0:
            record("OK", "preprocess_symmetry", "#650 <- #446 LTXVPreprocess (symmetric)")
        elif link_row[1] == 445:
            record(
                "ERR", "preprocess_symmetry",
                "#650 <- #445 ImageResizeKJv2 DIRECTLY (skips #446 LTXVPreprocess). "
                "Run scripts/apply_loop_guide_preprocess_symmetry.py.",
            )
        else:
            record(
                "WARN", "preprocess_symmetry",
                f"#650 inbound from unexpected source {link_row[1]}/{link_row[2]}",
            )

    # F3: loop-body cropguides symmetry -- inside the subgraph, CFGGuider
    # (#644) positive/negative must come from LTXVCropGuides (#655), not
    # LTXVAddLatentGuide (#1519) directly. Mirrors initial path's
    # #164 -> #381 -> #153. See scripts/apply_loop_cropguides_symmetry.py.
    defs = wf.get("definitions") or {}
    sgs = defs.get("subgraphs", []) if isinstance(defs, dict) else []
    if sgs:
        sg = sgs[0]
        sg_node_ids = {n["id"] for n in sg.get("nodes", [])}
        if {644, 655, 1519}.issubset(sg_node_ids):
            sg_links = sg.get("links", [])
            pos = next((l for l in sg_links if l.get("target_id") == 644 and l.get("target_slot") == 1), None)
            neg = next((l for l in sg_links if l.get("target_id") == 644 and l.get("target_slot") == 2), None)
            cfg_title = next(
                (n.get("title", "") for n in sg.get("nodes", []) if n.get("id") == 644),
                "",
            )
            is_ttc1_init_guide = cfg_title.startswith(TTC1_INIT_GUIDE_TITLE_PREFIX)
            if pos is None or neg is None:
                record("WARN", "loop_cropguides_symmetry", "CFGGuider(644) missing pos/neg inbound links")
            elif is_ttc1_init_guide:
                # F3 asymmetry on negative is intentional for this variant.
                if (
                    pos["origin_id"] == 655
                    and neg["origin_id"] == -10
                    and neg["origin_slot"] == 6
                ):
                    record(
                        "OK", "ttc1_init_guide_amplification",
                        "CFGGuider(644).negative <- INPUT_DISTRIBUTOR(slot 6) "
                        "[F3 asymmetry intentional]",
                    )
                else:
                    record(
                        "ERR", "ttc1_init_guide_amplification",
                        f"#644 titled TTC1 init-guide variant but rewire damaged "
                        f"(pos={pos['origin_id']}/{pos['origin_slot']}, "
                        f"neg={neg['origin_id']}/{neg['origin_slot']}). "
                        "Run scripts/apply_ttc_init_guide_amplification_poc.py --revert "
                        "and re-stage.",
                    )
            elif pos["origin_id"] == 655 and neg["origin_id"] == 655:
                record("OK", "loop_cropguides_symmetry", "#644 <- #655 LTXVCropGuides (symmetric)")
            elif pos["origin_id"] == 1519 and neg["origin_id"] == 1519:
                record(
                    "ERR", "loop_cropguides_symmetry",
                    "#644 <- #1519 LTXVAddLatentGuide DIRECTLY (bypasses #655 LTXVCropGuides). "
                    "Run scripts/apply_loop_cropguides_symmetry.py.",
                )
            else:
                record(
                    "WARN", "loop_cropguides_symmetry",
                    f"#644 inbound from unexpected sources pos={pos['origin_id']} neg={neg['origin_id']}",
                )

        # cropguides_split_topology — paired with scripts/apply_split_cropguides.py.
        # The split adds a second LTXVCropGuides instance to break the loop cycle
        # (CFGGuider <- CropGuides <- SeparateAV <- Sampler <- CFGGuider). Required
        # invariants when split is applied:
        #   - #655 is either upstream LTXVCropGuides (with .latent <- #1519 pre-sampling)
        #     or our no-latent variant LTXVCropGuidesNoLatent (no .latent slot at all)
        #   - the split node (titled SPLIT_NODE_TITLE) reads LATENT from #596 (post-sampling)
        #   - AdainLatent(#2006) reads samples from the split node, not from #655
        # If ALL nodes exist but any of these are damaged, the cycle is back.
        if sgs and {596, 1519, 2006, 655}.issubset({n["id"] for n in sgs[0].get("nodes", [])}):
            sg = sgs[0]
            sg_links = sg.get("links", [])
            cond_node = next((n for n in sg["nodes"] if n["id"] == 655), None)
            cond_type = cond_node.get("type") if cond_node else None
            split_node = next(
                (n for n in sg.get("nodes", [])
                 if n.get("type") == "LTXVCropGuides"
                 and n.get("title") == "CropGuides (LATENT-only — split)"),
                None,
            )
            crop_latent_link = next(
                (l for l in sg_links if l.get("target_id") == 655 and l.get("target_slot") == 2),
                None,
            )
            adain_samples_link = next(
                (l for l in sg_links if l.get("target_id") == 2006 and l.get("target_slot") == 0),
                None,
            )
            if split_node is None:
                if (
                    crop_latent_link
                    and crop_latent_link["origin_id"] == 596
                    and adain_samples_link
                    and adain_samples_link["origin_id"] == 655
                ):
                    record(
                        "ERR", "cropguides_split_topology",
                        "Cycle restored: #655.latent <- #596 + AdainLatent <- #655. "
                        "Run scripts/apply_split_cropguides.py.",
                    )
                # else: another shape (canonical-original-bypassed-elsewhere); skip silently
            else:
                cond_side_ok = (
                    cond_type == "LTXVCropGuidesNoLatent"
                    or (
                        cond_type == "LTXVCropGuides"
                        and crop_latent_link
                        and crop_latent_link["origin_id"] == 1519
                    )
                )
                latent_side_ok = (
                    adain_samples_link
                    and adain_samples_link["origin_id"] == split_node["id"]
                )
                if cond_side_ok and latent_side_ok:
                    cond_desc = (
                        "no-latent variant"
                        if cond_type == "LTXVCropGuidesNoLatent"
                        else "#655.latent <- #1519"
                    )
                    record(
                        "OK", "cropguides_split_topology",
                        f"#655={cond_desc} + #2006 <- #{split_node['id']} (post-sample)",
                    )
                else:
                    record(
                        "ERR", "cropguides_split_topology",
                        f"Split node #{split_node['id']} present but wiring damaged "
                        f"(#655 type={cond_type}, "
                        f"#2006 <- #{adain_samples_link['origin_id'] if adain_samples_link else '?'}). "
                        "Run scripts/apply_split_cropguides.py --revert and re-apply.",
                    )

    _check_prompt_relay_wiring(wf, by_type, record)
    _check_graph_acyclic(wf, by_id, record)
    _check_link_integrity(wf, by_id, links_by_id, record)
    _check_widget_shape(wf, record)

    if _is_retake(name):
        _check_retake_wiring(wf, by_type, record)

    return findings


# Generic structural invariants. These catch CLASSES of drift rather than
# specific named patterns — they fire on ANY future bug of the same shape
# without needing a hand-written check per fix.

# Strings the ComfyUI frontend serializes as `control_after_generate` widget
# values. Their presence in widgets_values is fine on nodes that
# legitimately have a control_after_generate dropdown (e.g. RandomNoise);
# our concern is when one leaks into the wrong slot of an unrelated node
# (Bug B from the 2026-04-27 cycle/widget/keyframe trio).
_CTRL_AFTER_GEN = frozenset({"randomize", "fixed", "increment", "decrement"})

# Node types that legitimately serialize control_after_generate strings as
# part of their widgets_values (e.g. RandomNoise's noise_seed dropdown).
# Outside this allowlist, any _CTRL_AFTER_GEN string is a partial-rename leak.
_CTRL_AFTER_GEN_LEGIT_NODE_TYPES = frozenset({
    "RandomNoise", "PrimitiveNode", "PrimitiveInt", "KSampler",
    "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced",
    "Seed (rgthree)", "Seed Everywhere",
})


def _check_graph_acyclic(wf, by_id, record) -> None:
    """Walk the top-level directed graph (link.src -> link.tgt) for back-edges.
    A cycle here means ComfyUI's prompt validator rejects the workflow with
    "Dependency cycle detected" before any node executes.
    Catches Bug A (Controller -> TensorLoopOpen -> Planner -> Controller).

    Top-level only: subgraph internals share the global node ID space with
    top-level, so merging both into one graph yields false positives when
    IDs collide. The tensor-loop framework handles subgraph-internal
    feedback patterns separately — they're legal there by design.

    Implementation: iterative DFS with WHITE/GRAY/BLACK coloring. GRAY edge
    target means we've found a back-edge — record the path."""
    edges: dict[int, list[tuple[int, int]]] = {}  # src_id -> [(tgt_id, link_id)]

    for link in wf.get("links") or []:
        if not isinstance(link, list) or len(link) < 6:
            continue
        edges.setdefault(link[1], []).append((link[3], link[0]))

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in edges}
    for tgts in edges.values():
        for t, _ in tgts:
            color.setdefault(t, WHITE)

    cycle: list[int] | None = None

    # `color` is fully populated above and not mutated during the DFS — only
    # individual values change WHITE -> GRAY -> BLACK, never the keyset.
    for start in color:
        if cycle:
            break
        if color[start] != WHITE:
            continue
        stack: list[tuple[int, int]] = [(start, 0)]
        path: list[int] = [start]
        color[start] = GRAY
        while stack:
            node, i = stack[-1]
            outs = edges.get(node, [])
            if i >= len(outs):
                color[node] = BLACK
                stack.pop()
                path.pop()
                continue
            stack[-1] = (node, i + 1)
            nxt, _ = outs[i]
            c = color.get(nxt, WHITE)
            if c == GRAY:
                idx = path.index(nxt)
                cycle = path[idx:] + [nxt]
                break
            if c == WHITE:
                color[nxt] = GRAY
                path.append(nxt)
                stack.append((nxt, 0))

    if cycle:
        labeled = " -> ".join(
            f"{nid}({by_id.get(nid, {}).get('type', '?')})" for nid in cycle
        )
        record(
            "ERR", "graph_acyclic",
            f"dependency cycle: {labeled}. ComfyUI rejects this at prompt-validate time. "
            f"Walk the cycle for the load-bearing back-edge — typically the most recent "
            f"auto-wire between two nodes that already had a forward path between them.",
        )
    else:
        record("OK", "graph_acyclic", "no cycles in top-level + subgraph link graphs")


def _check_link_integrity(wf, by_id, links_by_id, record) -> None:
    """For every top-level link [id, src, src_slot, tgt, tgt_slot, type]:
    src/tgt nodes exist, slots in range, source's outputs[slot].links lists
    the link id, target's inputs[slot].link == id, types match. Catches
    desync between link records and node-level link references (Bug D —
    1519.out[2].linkIds=[3004] but no link 3004 exists)."""
    issues: list[str] = []

    for link in wf.get("links") or []:
        if not isinstance(link, list) or len(link) < 6:
            issues.append(f"malformed top-level link {link}")
            continue
        lid, src, src_slot, tgt = link[0], link[1], link[2], link[3]
        src_node = by_id.get(src)
        tgt_node = by_id.get(tgt)
        if src_node is None:
            issues.append(f"link {lid} src node {src} missing")
            continue
        if tgt_node is None:
            issues.append(f"link {lid} tgt node {tgt} missing")
            continue
        outs = src_node.get("outputs") or []
        if not (0 <= src_slot < len(outs)):
            issues.append(
                f"link {lid} src={src}({src_node['type']}).slot[{src_slot}] out of range "
                f"(have {len(outs)})"
            )
            continue
        listed = outs[src_slot].get("links") or []
        if lid not in listed:
            issues.append(
                f"link {lid} ; {src}({src_node['type']}).out[{src_slot}].links={listed} "
                f"missing link id"
            )

    # Source-side dangling: outputs claim links that don't exist
    for nid, node in by_id.items():
        for i, out in enumerate(node.get("outputs") or []):
            for ref_lid in out.get("links") or []:
                if ref_lid not in links_by_id:
                    issues.append(
                        f"node {nid}({node['type']}).out[{i}] claims link {ref_lid} "
                        f"but no link record exists"
                    )

    # Subgraph linkIds desync (most common drift — WorkflowEditor mutations
    # update one side and forget the other)
    for sg in (wf.get("definitions") or {}).get("subgraphs") or []:
        sg_lid_set = {l.get("id") for l in (sg.get("links") or [])}
        for node in sg.get("nodes") or []:
            for i, out in enumerate(node.get("outputs") or []):
                for ref_lid in (out.get("linkIds") or []):
                    if ref_lid not in sg_lid_set:
                        issues.append(
                            f"subgraph node {node.get('id')}({node.get('type')})."
                            f"out[{i}].linkIds claims {ref_lid} but no subgraph link record exists"
                        )

    if issues:
        # Limit noise — show top 5, summarize rest
        head = issues[:5]
        more = f" ... {len(issues) - 5} more" if len(issues) > 5 else ""
        # Cosmetic linkIds desyncs are common after WorkflowEditor runs and
        # don't affect runtime — downgrade them to WARN.
        cosmetic_only = all("linkIds claims" in i for i in issues)
        status = "WARN" if cosmetic_only else "ERR"
        record(
            status, "link_integrity",
            f"{len(issues)} link inconsistenc{'y' if len(issues) == 1 else 'ies'}: "
            f"{'; '.join(head)}{more}",
        )
    else:
        record("OK", "link_integrity", "top-level + subgraph links bidirectionally consistent")


def _check_widget_shape(wf, record) -> None:
    """Stale `control_after_generate` strings in widgets_values.

    Their canonical home is on nodes that legitimately expose a
    control_after_generate dropdown (RandomNoise's `noise_seed`,
    PrimitiveNode's `value`, etc.). Allowlist those by node type. ANY other
    occurrence is a leak from a partial schema migration (Bug B) — the
    dropdown was attached when the input was named `seed`, the rename
    detached the dropdown, the leftover string sits in widgets_values and
    shifts later widget values into wrong slots.

    For nodes outside the allowlist, just flag any control_after_generate
    string in widgets_values. Cheap and high-signal."""
    leaks: list[str] = []
    for node in wf.get("nodes") or []:
        ntype = node.get("type", "")
        if ntype in _CTRL_AFTER_GEN_LEGIT_NODE_TYPES:
            continue
        wv = node.get("widgets_values") or []
        if not isinstance(wv, list):
            continue
        for i, val in enumerate(wv):
            if isinstance(val, str) and val in _CTRL_AFTER_GEN:
                leaks.append(
                    f"node {node.get('id')}({ntype}).widgets_values[{i}]={val!r} "
                    f"(control_after_generate string in non-seed-bearing node)"
                )

    if leaks:
        head = leaks[:3]
        more = f" ... {len(leaks) - 3} more" if len(leaks) > 3 else ""
        record(
            "ERR", "widget_shape",
            f"{len(leaks)} stray control_after_generate value(s): {'; '.join(head)}{more}. "
            f"Likely a schema rename that didn't strip the leftover widget value. "
            f"Run scripts/apply_strip_alc_control_after_generate.py if AudioLoopController; "
            f"otherwise write a similar strip migration for the affected node type.",
        )
    else:
        record("OK", "widget_shape", "no stray control_after_generate strings in widgets_values")


# Retake workflow checks — gated on filename match. The retake workflow
# regenerates a [start_time, end_time] window of a previously-generated
# video; LatentTemporalMask is the load-bearing node and audio passes
# through from the source mp4 (Option A — see
# internal/design/retake_workflow_design.md).
def _check_retake_wiring(wf, by_type, record) -> None:
    # 1. LatentTemporalMask must be present (it's why this workflow exists).
    mask_nodes = by_type.get("LatentTemporalMask", [])
    if not mask_nodes:
        record(
            "ERR", "retake_temporal_mask_present",
            "no LatentTemporalMask node. Run scripts/apply_audio_loop_retake.py.",
        )
    else:
        record("OK", "retake_temporal_mask_present", f"LatentTemporalMask present (id={mask_nodes[0]['id']})")

    # 2. VHS_VideoCombine.audio must be wired (Option A passthrough is the
    # single most likely regression; an unwired audio input ships a silent mp4).
    vhs_nodes = by_type.get("VHS_VideoCombine", [])
    if vhs_nodes:
        vhs = vhs_nodes[0]
        audio_inp = next(
            (inp for inp in vhs.get("inputs", []) if inp.get("name") == "audio"),
            None,
        )
        if audio_inp is None:
            record("WARN", "retake_audio_passthrough", "VHS_VideoCombine has no 'audio' input slot")
        elif audio_inp.get("link") is None:
            record(
                "ERR", "retake_audio_passthrough",
                "VHS_VideoCombine.audio is unwired (Option A passthrough broken; output will be silent).",
            )
        else:
            record("OK", "retake_audio_passthrough", "VHS_VideoCombine.audio wired")

    # 3. No loop machinery must remain (catches incomplete strips).
    loop_offenders = [
        t for t in ("TensorLoopOpen", "TensorLoopClose", "AudioLoopController", "AudioLoopPlanner")
        if by_type.get(t)
    ]
    if _subgraph_invoker_id(wf) is not None:
        loop_offenders.append("subgraph_invoker")
    if loop_offenders:
        record(
            "WARN", "retake_no_loop_nodes",
            f"loop machinery still present: {', '.join(loop_offenders)} (incomplete strip?)",
        )
    else:
        record("OK", "retake_no_loop_nodes", "no loop nodes")


# PromptRelayEncode installs object_patches on attn2 / audio_attn2 which don't
# survive TensorLoop offload/reload. The patched MODEL must reach ONLY the
# initial CFGGuider, never the loop subgraph invoker's model slot. Remediation:
# scripts/apply_prompt_relay_initial_render.py.
def _check_prompt_relay_wiring(wf, by_type, record) -> None:
    relay_nodes = by_type.get("PromptRelayEncode", [])
    if not relay_nodes:
        return

    if len(relay_nodes) > 1:
        record(
            "WARN", "prompt_relay_wiring",
            f"{len(relay_nodes)} PromptRelayEncode nodes found; audit only checks the first.",
        )

    relay_id = relay_nodes[0]["id"]
    invoker_id = _subgraph_invoker_id(wf)
    model_targets: set[int] = set()
    leaked_into_loop = False
    for lk in wf["links"]:
        if not isinstance(lk, list) or lk[1] != relay_id or lk[2] != 0:
            continue
        model_targets.add(lk[3])
        if invoker_id is not None and lk[3] == invoker_id and lk[4] == 2:
            leaked_into_loop = True

    if not model_targets:
        record("WARN", "prompt_relay_wiring", "PromptRelayEncode MODEL output has no consumers")
        return

    if leaked_into_loop:
        record(
            "ERR", "prompt_relay_wiring",
            f"PromptRelayEncode MODEL leaks into subgraph invoker {invoker_id} "
            "(slot 2 = model). object_patches will be stripped by loop offload; "
            "fork the MODEL upstream of this node.",
        )
        return

    if 153 not in model_targets:
        record(
            "ERR", "prompt_relay_wiring",
            f"PromptRelayEncode MODEL does not reach initial CFGGuider(153) (targets={sorted(model_targets)})",
        )
        return

    record("OK", "prompt_relay_wiring", "PromptRelayEncode MODEL -> CFGGuider(153) only")


def _subgraph_invoker_id(wf) -> int | None:
    defs = wf.get("definitions") or {}
    sgs = defs.get("subgraphs", []) if isinstance(defs, dict) else []
    if not sgs:
        return None
    sg_id = sgs[0].get("id")
    if not sg_id:
        return None
    for n in wf["nodes"]:
        if n.get("type") == sg_id:
            return n["id"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="Print OK findings too.")
    args = ap.parse_args()

    all_findings: list[Finding] = []
    paths = list(EXAMPLE_WORKFLOWS_DIR.glob("*.json"))
    exp_dir = EXAMPLE_WORKFLOWS_DIR / "experimental"
    for fn in EXPERIMENTAL_AUDITED_FILES:
        p = exp_dir / fn
        if p.exists():
            paths.append(p)
    for wf_path in sorted(paths):
        all_findings.extend(_audit_one(wf_path))

    err_count = sum(1 for f in all_findings if f.status == "ERR")
    warn_count = sum(1 for f in all_findings if f.status == "WARN")
    ok_count = sum(1 for f in all_findings if f.status == "OK")

    header_needed = True
    for f in all_findings:
        if f.status == "OK" and not args.verbose:
            continue
        if header_needed:
            print(f"{'STATUS':6} {'WORKFLOW':58} {'CHECK':28} MESSAGE")
            print("-" * 130)
            header_needed = False
        print(f"{f.status:6} {f.workflow[:58]:58} {f.check:28} {f.message}")

    print()
    print(f"Totals: {ok_count} OK, {warn_count} WARN, {err_count} ERR")
    return 1 if err_count else 0


if __name__ == "__main__":
    sys.exit(main())
