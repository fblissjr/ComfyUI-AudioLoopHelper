"""Health audit for every workflow under example_workflows/.

Verifies each shipped workflow matches the current CLAUDE.md invariants:
sage node + mode, iteration stamp, batch-encode prompt schedule,
distilled sampler chain (linear_quadratic 8 1 + shift=13 + euler +
cfg=1, with STG-variant exception), resolution div-32, (L-1)%8==0,
LTXVPreprocess img_compression >= 18, LTXVTiledVAEDecode preferred,
preprocess symmetry (F2), and loop-body cropguides symmetry (F3).

Loop-only invariants (frame planner SSoT, iteration stamp, prompt
schedule, audio pre-encode wiring, F2/F3/F12 init-image symmetry,
sage tracer iter grouping) are gated on `_is_loop_workflow`: workflows
with no `TensorLoopOpen` / `TensorLoopClose` / `AudioLoopController`
nodes (e.g. post-loop upscale or polish passes) silently skip them.
Generic invariants (graph_acyclic, widget_shape, link_integrity,
no_sd3_shift_node, resolution/length/volume, preprocess_compression)
still run on every workflow.

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
from workflow_utils import EXAMPLE_WORKFLOWS_DIR, is_active
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
    "KSamplerSelect": (["euler"], "sampler_type"),
}

# Experimental POC files that ship alongside production workflows and
# have audit checks of their own. Anything outside this allowlist (e.g.
# spectrogram_iclora_minimal.json, iclora_amplification_poc.json) is
# intentionally NOT audited — those are forks with non-standard topology
# that pre-date the audit and would just generate noise.
EXPERIMENTAL_DIR_NAME = "experimental"
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


def _is_loop_workflow(by_type: dict[str, list[dict]]) -> bool:
    """Heuristic: a workflow is "loop-shaped" if it has any of the loop
    spine nodes. Drives gating of loop-only invariants so single-pass
    workflows (e.g. post-loop upscale/polish) don't get false-positive
    ERR / WARN hits from checks that have no semantic meaning outside
    a tensor-loop topology.

    Detection is structural — no user-facing flag, no JSON marker. Keep
    in sync with the loop-spine node list in CLAUDE.md (Architecture →
    Loop spine).
    """
    return any(
        by_type.get(t)
        for t in ("TensorLoopOpen", "TensorLoopClose", "AudioLoopController")
    )


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

    # Detect loop-shaped vs single-pass topology. Loop-only invariants
    # (frame planner SSoT, iteration stamp, prompt schedule, audio
    # pre-encode wiring, F2/F3/F12 init-image symmetry) are silently
    # skipped on single-pass workflows — they have no semantic role
    # there and would emit false-positive ERR / WARN otherwise. Generic
    # invariants (graph_acyclic, widget_shape, link_integrity,
    # no_sd3_shift_node) still run on all workflows.
    is_loop = _is_loop_workflow(by_type)

    # Sage node. Gated on loop topology: sage helps any sampler, but the
    # `AudioLoopHelperSageAttention` tracer-iter-grouping path is the
    # specific reason this WARN ships — single-pass workflows have no
    # iter axis to group on, so a missing sage node there is noise.
    # Legacy `PathchSageAttentionKJ` ERR also gates with loop because
    # the migration target is the loop-specific replacement.
    sage = by_type.get("AudioLoopHelperSageAttention", [])
    if not sage and is_loop:
        if by_type.get("PathchSageAttentionKJ"):
            record("ERR", "sage_node", "uses legacy PathchSageAttentionKJ instead of AudioLoopHelperSageAttention")
        else:
            record("WARN", "sage_node", "no sage node present")
    elif sage:
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
        i_active = is_active(initial)
        l_active = is_active(loop)
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

    # Dead LoRA-loader scaffolding (id 1625/1626/1627). Three bypassed
    # placeholder nodes were left in the canonical from earlier
    # exploration; they're never wired to a guide and one points at a
    # placeholder filename. Misleading UI clutter — strip via
    # scripts/apply_strip_dead_lora_loaders.py. See that script's
    # docstring for the full rationale.
    _dead_scaffolding_sigs = (
        (1625, "LoraLoaderModelOnly",
         "ID-LoRA File (audio-conditioned identity)",
         "LTX-2.3-ID-LoRA-CelebVHQ-3K/lora_weights.safetensors"),
        (1626, "LTXICLoRALoaderModelOnly",
         "IC-LoRA File (visual reference adapter)",
         "MergeGreen_IC-lora_ltx2.3.safetensors"),
        (1627, "LoraLoaderModelOnly",
         "Style/Generic LoRA",
         "your_style_lora.safetensors"),
    )
    _dead_matches = []
    for nid, ntype, ntitle, nfile in _dead_scaffolding_sigs:
        n = by_id.get(nid)
        if n is None:
            continue
        if n.get("type") != ntype:
            continue  # id collision with a different node type — ignore
        if n.get("title") != ntitle:
            continue  # user renamed; preserve their customization
        if n.get("mode") != 4:
            continue  # user un-bypassed; preserve their customization
        widgets = n.get("widgets_values") or []
        if not widgets or widgets[0] != nfile:
            continue
        _dead_matches.append(f"#{nid}")
    if _dead_matches:
        record("ERR", "dead_lora_loader_scaffolding_absent",
               f"dead bypassed scaffolding present: {', '.join(_dead_matches)}. "
               f"Run scripts/apply_strip_dead_lora_loaders.py")
    else:
        record("OK", "dead_lora_loader_scaffolding_absent",
               "no dead scaffolding (canonical post-strip shape)")

    # LoopIterationStamp (loop-only — only meaningful inside the
    # tensor-loop body for sage tracer iter grouping).
    if by_type.get("LoopIterationStamp"):
        record("OK", "iteration_stamp", "present")
    elif _is_retake(name):
        record("OK", "iteration_stamp", "n/a (retake workflow, no loop)")
    elif is_loop:
        record("WARN", "iteration_stamp", "missing (sage tracer iter grouping will be blank)")

    # LTXFramePlanner is the SSoT for dim/fps config; without it widget
    # values scatter across EmptyLTXVLatentVideo, ImageResizeKJv2,
    # AudioLoopController, AudioLoopPlanner, and the subgraph. Retake
    # and non-loop workflows are exempt (no loop spine to scatter
    # across); experimental forks downgrade to WARN.
    is_experimental = wf_path.parent.name == EXPERIMENTAL_DIR_NAME
    if not _is_retake(name) and is_loop:
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

    # Active LTXVTiledVAEDecode nodes should run [1,1,1] (single-tile)
    # on production workflows. Tiled decode pays per-tile prepare/stage
    # overhead that exceeds activation savings on 24GB+ cards. Empirical:
    # [2,2,1] cold-pass 143s vs [1,1,1] cold-pass 47s (~3x). WARN not
    # ERR because [2,2,1] is the safe fallback for ≤16GB cards. See
    # scripts/apply_no_tile_vae_decode.py.
    decoders = by_type.get("LTXVTiledVAEDecode", [])
    tile_violations = []
    for d in decoders:
        if not is_active(d):
            continue  # bypassed; user choice
        # Dead nodes (no downstream consumers) are ComfyUI-skipped.
        if not any(o.get("links") for o in d.get("outputs") or []):
            continue
        # widgets_values[0] = horizontal_tiles, [1] = vertical_tiles
        wv = d.get("widgets_values") or []
        if len(wv) >= 3 and (wv[0] != 1 or wv[1] != 1):
            tile_violations.append(f"#{d.get('id')} {wv[0]}x{wv[1]}")
    if tile_violations:
        record("WARN", "vae_decode_no_tile",
               f"{', '.join(tile_violations)} not at [1,1,1] (3x slower cold-pass on 24GB). "
               f"Run scripts/apply_no_tile_vae_decode.py if on 24GB+; "
               f"keep tiled if on ≤16GB.")
    elif decoders:
        record("OK", "vae_decode_no_tile",
               "active LTXVTiledVAEDecode at [1,1,1] (single-tile)")

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
    elif is_loop:
        record("WARN", "prompt_schedule", "no prompt schedule node")
    # else: non-loop workflow — per-iteration prompts are loop-only

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
                # Non-canonical schedule — informational. Some workflows
                # (e.g. post-loop upscale/polish) deliberately use a
                # shorter sigma profile; this WARN flags divergence from
                # the distilled 8-step canonical, not a defect.
                record(
                    "WARN", check_name,
                    f"{t}(id={n['id']}) non-canonical sigma profile: "
                    f"widgets={n.get('widgets_values', [])}",
                )
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
    # Loop-only: F2/F3/F12 are about the per-iter init-image path; on
    # single-pass workflows there's no loop branch to keep symmetric.
    set_input_image = next((n for n in wf["nodes"] if n["id"] == 650), None) if is_loop else None
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
    if sgs and is_loop:
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

    # Loop-only checks: audio pre-encode topology, init-image symmetry,
    # planner-driven wiring. Single-pass workflows (e.g. post-loop
    # upscale/polish) skip these silently — the invariants describe loop
    # body wiring that doesn't exist there.
    if is_loop:
        _check_prompt_relay_wiring(wf, by_type, record)
        _check_ltx2_nag_reaches_loop(wf, by_type, record)
        _check_iclora_video_reference_wiring(wf, by_type, record)
        _check_audio_latent_slice_source_seconds_wired(wf, by_type, record)
        _check_audio_latent_slice_iter_wiring(wf, by_type, record)
        _check_initial_render_audio_duration_wired(wf, by_type, record)
        _check_overlap_seconds_single_source(wf, by_type, record)
        _check_vhs_video_combine_frame_rate_parity(wf, by_type, record)

    # Generic invariants — apply to all workflows regardless of topology.
    _check_no_sd3_shift_node(wf, by_type, record)
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


# LTX2_NAG (KJNodes) installs `object_patches` capturing the negative
# conditioning tensor. Unlike PromptRelay (CLIP-driven, evicted on
# offload), NAG patches survive the loop offload/reload because DiT stays
# resident — but ONLY IF the patched MODEL reaches the loop subgraph at
# all. If a future edit forks the chain so NAG only feeds the initial
# CFGGuider, the loop body sees an unpatched MODEL and the NAG negative
# prompt silently disengages from iter 1+. Mirror failure mode of
# pre-2026-04-22 batch-encode bug, but rooted in topology rather than
# CLIP placement. Reference:
# docs/analysis/nag_object_patches_offload_asymmetry.md.
def _check_ltx2_nag_reaches_loop(wf, by_type, record) -> None:
    nag_nodes = by_type.get("LTX2_NAG", [])
    if not nag_nodes:
        return

    invoker_id = _subgraph_invoker_id(wf)
    if invoker_id is None:
        return  # no subgraph (e.g. retake)

    if len(nag_nodes) > 1:
        record(
            "WARN", "ltx2_nag_reaches_loop",
            f"{len(nag_nodes)} LTX2_NAG nodes found; audit only checks the first.",
        )

    nag_id = nag_nodes[0]["id"]

    # Forward MODEL-link BFS. The chain passes through chain-pure nodes
    # (LoraLoaderModelOnly, SetNode/GetNode, LTXVReferenceAudio,
    # LTX2SamplingPreviewOverride, LTXICLoRALoaderModelOnly), so we don't
    # need per-node-type knowledge — only the MODEL-typed link edges.
    edges_by_src: dict[int, list] = {}
    for lk in wf["links"]:
        if isinstance(lk, list) and len(lk) >= 6 and lk[5] == "MODEL":
            edges_by_src.setdefault(lk[1], []).append(lk)

    # SetNode/GetNode are virtual broadcasts (no MODEL-typed link from Set
    # to Get — they're matched by widgets_values[0] var name).
    setnode_var_by_id: dict[int, str] = {}
    getnodes_by_var: dict[str, list[dict]] = {}
    for n in wf["nodes"]:
        if n.get("type") not in ("SetNode", "GetNode"):
            continue
        wv = n.get("widgets_values")
        if not isinstance(wv, list) or not wv or not isinstance(wv[0], str):
            continue
        if n["type"] == "SetNode":
            setnode_var_by_id[n["id"]] = wv[0]
        else:
            getnodes_by_var.setdefault(wv[0], []).append(n)

    visited: set[int] = set()
    queue: list[int] = [nag_id]
    reached_invoker = False
    while queue:
        cur = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for lk in edges_by_src.get(cur, []):
            tgt = lk[3]
            if tgt == invoker_id:
                reached_invoker = True
                break
            queue.append(tgt)
        if reached_invoker:
            break
        var = setnode_var_by_id.get(cur)
        if var is not None:
            queue.extend(g["id"] for g in getnodes_by_var.get(var, []))

    if reached_invoker:
        record("OK", "ltx2_nag_reaches_loop",
               f"LTX2_NAG({nag_id}) MODEL reaches subgraph invoker({invoker_id})")
    else:
        record(
            "ERR", "ltx2_nag_reaches_loop",
            f"LTX2_NAG({nag_id}) MODEL does not reach loop subgraph invoker "
            f"({invoker_id}). NAG object_patches will silently disengage in "
            "loop iterations — every class NAG was suppressing returns iter 1+. "
            "Verify the SetNode/GetNode model-broadcast chain is intact.",
        )


# Video-reference IC-LoRA wiring (apply_iclora_video_reference.py).
# Class-of-drift checks that fire ONLY when the workflow uses the
# in-loop LTXAddVideoICLoRAGuide pattern. Fire-once: emit at most one
# of each check per workflow (no spam).
# ModelSamplingSD3 should NOT be present on LTX 2.3 distilled workflows.
# Lightricks's distilled inference applies no flow-matching shift; the
# DISTILLED_SIGMA_VALUES are the final schedule fed directly to the
# Euler denoising loop. Adding `ModelSamplingSD3 shift=13` distorts the
# sigma-to-timestep mapping in a way the distilled checkpoint was not
# trained for. See `internal/analysis/ltx23_sigma_shift_audit.md` for
# the full evidence trail (Lightricks' distilled.py + their official
# 2.3 distilled example workflows have no shift node).
#
# The 8 instances we shipped with `ModelSamplingSD3` had `outputs[0].links == []`
# (dead nodes) — strip is pure cleanup. Migration: apply_strip_sd3_shift_node.py.
def _check_no_sd3_shift_node(wf, by_type, record) -> None:
    sd3 = by_type.get("ModelSamplingSD3", [])
    if not sd3:
        record("OK", "model_sampling_shift",
               "no ModelSamplingSD3 (correct for LTX 2.3 distilled)")
        return
    active = [n for n in sd3 if n.get("mode") != 4]
    if not active:
        record("OK", "model_sampling_shift",
               f"{len(sd3)} ModelSamplingSD3 present but bypassed")
        return
    record("WARN", "model_sampling_shift",
           f"{len(active)} active ModelSamplingSD3 node(s) — distorts the "
           "distilled sigma schedule. Run scripts/apply_strip_sd3_shift_node.py.")


def _check_iclora_video_reference_wiring(wf, by_type, record) -> None:
    sgs = wf.get("definitions", {}).get("subgraphs", []) or []
    if not sgs:
        return
    sg = sgs[0]
    sg_nodes = sg.get("nodes", [])
    sg_links = sg.get("links", [])
    iclora_guides = [n for n in sg_nodes if n.get("type") == "LTXAddVideoICLoRAGuide"]
    if not iclora_guides:
        return  # video-ref wiring absent; checks don't apply

    # Check: F3 cropguides symmetry on guide CONDITIONING outputs. The
    # guide's positive/negative outputs (slots 0, 1) must reach CFGGuider
    # only via LTXVCropGuides or LTXVCropGuidesNoLatent — never directly.
    sg_node_by_id = {n.get("id"): n for n in sg_nodes}
    guide_ids = {g.get("id") for g in iclora_guides}
    direct_violation = next(
        (
            (link["origin_id"], link["origin_slot"], link["target_id"])
            for link in sg_links
            if link.get("origin_id") in guide_ids
            and link.get("origin_slot") in (0, 1)
            and (sg_node_by_id.get(link.get("target_id")) or {}).get("type") == "CFGGuider"
        ),
        None,
    )
    if direct_violation:
        gid, gslot, cfg_id = direct_violation
        record(
            "ERR", "iclora_video_reference_guide_in_loop_with_cropguides",
            f"LTXAddVideoICLoRAGuide({gid}).out[{gslot}] feeds CFGGuider({cfg_id}) "
            f"directly. Must pass through LTXVCropGuides or LTXVCropGuidesNoLatent. "
            f"Re-run scripts/archive/apply_iclora_video_reference.py.",
        )
    else:
        record(
            "OK", "iclora_video_reference_guide_in_loop_with_cropguides",
            f"{len(iclora_guides)} LTXAddVideoICLoRAGuide reaches CFGGuider only via cropguides",
        )

    # Check: guide implies loader on top-level. The patched MODEL must be
    # active for the guide to do anything; without the loader, only the
    # guide attention-entry append fires (no LoRA effect).
    if not by_type.get("LTXICLoRALoaderModelOnly"):
        record(
            "ERR", "iclora_loader_present_when_guide_present",
            "LTXAddVideoICLoRAGuide present in subgraph but no LTXICLoRALoaderModelOnly "
            "on the top-level MODEL chain. Re-run scripts/archive/apply_iclora_video_reference.py.",
        )
    else:
        record(
            "OK", "iclora_loader_present_when_guide_present",
            "LTXICLoRALoaderModelOnly on top-level MODEL chain",
        )

    # Check: F2 ref-video preprocess symmetry. The ref-video preprocessing
    # chain must include LTXVPreprocess(val=18) — same value the init-image
    # path uses. Without it, ref-video frames hit different edge-statistics
    # than the init image, causing iter-over-iter drift.
    vhs_loaders = by_type.get("VHS_LoadVideo", [])
    preprocs = by_type.get("LTXVPreprocess", [])
    if not vhs_loaders:
        record(
            "WARN", "iclora_ref_video_preprocess_symmetry",
            "video-ref IC-LoRA guide present but no VHS_LoadVideo on top-level. "
            "Cannot verify F2 preprocess symmetry.",
        )
    else:
        symmetric = any(
            (p.get("widgets_values") or [None])[0] == 18 for p in preprocs
        )
        if symmetric:
            record(
                "OK", "iclora_ref_video_preprocess_symmetry",
                "LTXVPreprocess(val=18) present (F2 symmetric with init-image path)",
            )
        else:
            record(
                "ERR", "iclora_ref_video_preprocess_symmetry",
                "no LTXVPreprocess(val=18) found — ref-video path may not match "
                "init-image preprocessing. Re-run scripts/archive/apply_iclora_video_reference.py.",
            )


def _check_audio_latent_slice_source_seconds_wired(wf, by_type, record) -> None:
    """ERR if AudioLatentSlice's `source_seconds` input is widget-driven.

    AudioLatentSlice infers the per-iter slice rate as
    `latent_T / source_seconds`. When `source_seconds` is a hardcoded
    widget value but the encoded audio is shorter (TrimAudioDuration
    silently clamps to song length), the inferred rate drifts and
    every per-iter slice misaligns with the corresponding video frames
    — visible as lip-sync drift. The widget MUST come from
    `AudioLoopController.audio_duration`. Migration:
    `scripts/apply_audio_latent_slice_source_seconds_autowire.py`.

    Note: AudioLatentSlice lives in the loop subgraph, so `by_type`
    (which only indexes top-level nodes) doesn't help here. We walk
    `sg["nodes"]` directly. Signature kept consistent with sibling
    checks per the audit-framework convention.
    """
    del by_type  # subgraph nodes; intentionally unused
    sgs = wf.get("definitions", {}).get("subgraphs", [])
    if not sgs:
        return
    sg = sgs[0]
    slicers = [n for n in sg.get("nodes", []) if n.get("type") == "AudioLatentSlice"]
    if not slicers:
        return  # node absent → no pre-encode chain → check N/A
    sg_link_ids = {l.get("id") for l in sg.get("links", [])}
    for slicer in slicers:
        # Input name kept in sync with NEW_SUBGRAPH_INPUT_NAME in
        # scripts/apply_audio_latent_slice_source_seconds_autowire.py
        source_input = next(
            (i for i in slicer.get("inputs", []) if i.get("name") == "source_seconds"),
            None,
        )
        if source_input is None:
            record(
                "ERR", "audio_latent_slice_source_seconds_wired",
                f"AudioLatentSlice(#{slicer.get('id')}) missing 'source_seconds' input. "
                "Re-run scripts/archive/apply_audio_latent_pre_encode.py.",
            )
            continue
        link_id = source_input.get("link")
        if link_id is None or link_id not in sg_link_ids:
            # Treat dangling refs (link id with no matching record) the same
            # as widget-only — neither carries a runtime value into the slicer.
            record(
                "ERR", "audio_latent_slice_source_seconds_wired",
                f"AudioLatentSlice(#{slicer.get('id')}).source_seconds is widget-driven; "
                "must wire from AudioLoopController.audio_duration to avoid drift on "
                "songs not exactly matching the widget value. Run "
                "scripts/apply_audio_latent_slice_source_seconds_autowire.py.",
            )
        else:
            record(
                "OK", "audio_latent_slice_source_seconds_wired",
                f"AudioLatentSlice(#{slicer.get('id')}).source_seconds is wired",
            )


def _check_audio_latent_slice_iter_wiring(wf, by_type, record) -> None:
    """ERR if AudioLatentSlice's `start_seconds` or `duration_seconds`
    are sourced incorrectly.

    `start_seconds` MUST come from the subgraph's `start_index` input
    (post-trim audio time of iter N's window). The `video_start_time`
    input, which is sourced from `AudioLoopController.overlap_seconds`
    (constant ~1.0), was the original buggy source — every iter sliced
    from t=1.0, never advancing through the song.

    `duration_seconds` MUST be wired (not widget-only). Canonical source
    is `video_end_time` (= `LTXFramePlanner.actual_seconds`), so the
    audio slice length tracks the video window length.

    Migration: scripts/apply_audio_latent_slice_iter_wiring_fix.py.
    """
    del by_type
    sgs = wf.get("definitions", {}).get("subgraphs", [])
    if not sgs:
        return
    sg = sgs[0]
    slicers = [n for n in sg.get("nodes", []) if n.get("type") == "AudioLatentSlice"]
    if not slicers:
        return

    # Resolve subgraph slot indices by name (not hardcoded — robust to
    # workflow-specific slot ordering).
    sg_inputs = sg.get("inputs", [])
    name_to_slot = {
        inp.get("name"): i for i, inp in enumerate(sg_inputs)
    }
    start_index_slot = name_to_slot.get("start_index")

    sg_links = sg.get("links", [])
    for slicer in slicers:
        sid = slicer.get("id")
        # AudioLatentSlice schema: input slot 2 = start_seconds, 3 = duration_seconds
        start_link = next(
            (l for l in sg_links if l.get("target_id") == sid and l.get("target_slot") == 2),
            None,
        )
        if start_link is None or start_link.get("origin_slot") != start_index_slot:
            origin = start_link.get("origin_slot") if start_link else None
            origin_name = (
                sg_inputs[origin].get("name") if origin is not None and origin < len(sg_inputs)
                else "unwired"
            )
            record(
                "ERR", "audio_latent_slice_iter_wiring",
                f"AudioLatentSlice(#{sid}).start_seconds sourced from {origin_name!r}; "
                "must come from 'start_index' (post-trim audio time per iter), not "
                "'video_start_time' (constant overlap_seconds). Run "
                "scripts/apply_audio_latent_slice_iter_wiring_fix.py.",
            )
        else:
            record(
                "OK", "audio_latent_slice_iter_wiring",
                f"AudioLatentSlice(#{sid}).start_seconds correctly sourced from start_index",
            )

        dur_link = next(
            (l for l in sg_links if l.get("target_id") == sid and l.get("target_slot") == 3),
            None,
        )
        if dur_link is None:
            record(
                "ERR", "audio_latent_slice_iter_wiring",
                f"AudioLatentSlice(#{sid}).duration_seconds is widget-only; "
                "must wire to 'video_end_time' (= FramePlanner.actual_seconds) so "
                "audio slice length matches video window length. Run "
                "scripts/apply_audio_latent_slice_iter_wiring_fix.py.",
            )
        else:
            record(
                "OK", "audio_latent_slice_iter_wiring",
                f"AudioLatentSlice(#{sid}).duration_seconds is wired",
            )


def _check_initial_render_audio_duration_wired(wf, by_type, record) -> None:
    """ERR if #601 TrimAudioDuration's `duration` is widget-only.

    The initial-render audio context comes from #601 TrimAudioDuration.
    Its `duration` widget must match `LTXFramePlanner.actual_seconds`
    (the initial render's video length). Static widget values silently
    truncate the audio context — the model has nothing to align lip
    movements to past the widget value, producing visible drift.
    Migration: scripts/apply_initial_render_audio_duration_autowire.py.

    Fires on any workflow that has #601 — the bug isn't specific to
    the pre-encode pipeline. Loop-body audio (per-iter encode or pre-
    encoded slice) is independent of the initial-render audio context,
    which #601 always controls.
    """
    del by_type
    trim = next(
        (n for n in wf.get("nodes", [])
         if n.get("type") == "TrimAudioDuration" and n.get("id") == 601),
        None,
    )
    if trim is None:
        return  # node renumbered or absent; not our concern here
    duration_input = next(
        (i for i in trim.get("inputs", []) if i.get("name") == "duration"),
        None,
    )
    link_id = duration_input.get("link") if duration_input else None
    top_level_link_ids = {l[0] for l in wf.get("links", []) if isinstance(l, list)}
    if link_id is None or link_id not in top_level_link_ids:
        record(
            "ERR", "initial_render_audio_duration_wired",
            f"TrimAudioDuration(#{trim.get('id')}).duration is widget-only; "
            "must wire from LTXFramePlanner.actual_seconds so the initial-"
            "render audio context matches the video length. Run "
            "scripts/apply_initial_render_audio_duration_autowire.py.",
        )
    else:
        record(
            "OK", "initial_render_audio_duration_wired",
            f"TrimAudioDuration(#{trim.get('id')}).duration is wired",
        )


def _check_overlap_seconds_single_source(wf, by_type, record) -> None:
    """ERR if AudioLoopController.overlap_seconds and AudioLoopPlanner.
    overlap_seconds aren't sourced from the same node.

    Both nodes shipped with widget-only `overlap_seconds` defaulting to
    2. If a user updates one but not the other, the controller drives
    the loop (correctly) while the planner's iteration-count summary
    silently shows the wrong value. F7 prevents wiring controller →
    planner directly (cycle), so the canonical fix is a shared
    FloatConstant feeding both. Migration:
    `scripts/apply_overlap_seconds_single_source.py`.
    """
    controllers = by_type.get("AudioLoopController", [])
    planners = by_type.get("AudioLoopPlanner", [])
    if not controllers or not planners:
        return  # not a loop workflow
    controller = controllers[0]
    planner = planners[0]

    links_by_id: dict[int, list] = {
        l[0]: l for l in wf.get("links") or [] if isinstance(l, list)
    }

    def _overlap_link_source(node):
        for inp in node.get("inputs", []):
            if inp.get("name") == "overlap_seconds":
                link = links_by_id.get(inp.get("link"))
                return link[1] if link else None
        return None

    c_src = _overlap_link_source(controller)
    p_src = _overlap_link_source(planner)

    if c_src is None or p_src is None:
        record(
            "ERR", "overlap_seconds_single_source",
            f"AudioLoopController.overlap_seconds wired={c_src is not None}, "
            f"AudioLoopPlanner.overlap_seconds wired={p_src is not None}; "
            "both must come from the same source node (typically a shared "
            "FloatConstant titled 'overlap_seconds'). Run "
            "scripts/apply_overlap_seconds_single_source.py.",
        )
    elif c_src != p_src:
        record(
            "ERR", "overlap_seconds_single_source",
            f"controller.overlap_seconds ← #{c_src}, planner.overlap_seconds ← #{p_src}. "
            "Different sources will silently diverge. Run "
            "scripts/apply_overlap_seconds_single_source.py.",
        )
    else:
        record(
            "OK", "overlap_seconds_single_source",
            f"both consumers sourced from #{c_src}",
        )


def _check_vhs_video_combine_frame_rate_parity(wf, by_type, record) -> None:
    """ERR if VHS_VideoCombine.frame_rate diverges from LTXFramePlanner.fps.

    VHS_VideoCombine's `frame_rate` lives in its dict-shaped
    `widgets_values` (not a converted input slot), so it can't be
    auto-wired the way other FramePlanner consumers are. Without a
    wire, a user who changes FramePlanner's fps but forgets to update
    VHS_VideoCombine produces an mp4 tagged at the wrong rate — audio
    drifts against video on playback. Audit-only enforcement: verify
    the two values match. Manual remediation: edit VHS_VideoCombine's
    `widgets_values.frame_rate` to match.
    """
    del wf
    planners = by_type.get("LTXFramePlanner", [])
    vhs_combines = by_type.get("VHS_VideoCombine", [])
    if not planners or not vhs_combines:
        return
    planner = planners[0]
    # FramePlanner widgets_values = [target_width, target_height, target_seconds, fps]
    fp_widgets = planner.get("widgets_values") or []
    if len(fp_widgets) < 4:
        return
    fp_fps = fp_widgets[3]
    for vhs in vhs_combines:
        wv = vhs.get("widgets_values")
        if not isinstance(wv, dict):
            continue
        vhs_fps = wv.get("frame_rate")
        if vhs_fps is None:
            continue
        if int(vhs_fps) != int(fp_fps):
            record(
                "ERR", "vhs_frame_rate_matches_planner",
                f"VHS_VideoCombine(#{vhs.get('id')}).frame_rate={vhs_fps} "
                f"diverges from LTXFramePlanner(#{planner.get('id')}).fps={fp_fps}. "
                "Saved mp4 will be tagged at the wrong rate. "
                f"Edit VHS_VideoCombine.widgets_values.frame_rate to {fp_fps}.",
            )
        else:
            record(
                "OK", "vhs_frame_rate_matches_planner",
                f"VHS_VideoCombine(#{vhs.get('id')}).frame_rate={vhs_fps} "
                f"matches LTXFramePlanner.fps",
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="Print OK findings too.")
    ap.add_argument(
        "paths", nargs="*", type=Path,
        help="Optional workflow JSON paths to audit. Default sweeps example_workflows/ "
             "(plus the audited subset of experimental/). Pass paths to audit a staged "
             "scratch file or any other JSON.",
    )
    args = ap.parse_args()

    all_findings: list[Finding] = []
    if args.paths:
        paths = list(args.paths)
        for p in paths:
            if not p.exists():
                print(f"error: path does not exist: {p}", file=sys.stderr)
                return 1
    else:
        paths = list(EXAMPLE_WORKFLOWS_DIR.glob("*.json"))
        exp_dir = EXAMPLE_WORKFLOWS_DIR / EXPERIMENTAL_DIR_NAME
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
