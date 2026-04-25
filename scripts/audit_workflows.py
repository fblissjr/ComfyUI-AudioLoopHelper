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
from workflow_utils import EXAMPLE_WORKFLOWS_DIR


class Finding(NamedTuple):
    status: str  # OK | WARN | ERR
    workflow: str
    check: str
    message: str


EXPECTED_CHAIN = {
    "BasicScheduler": (["linear_quadratic", 8, 1], "basic_scheduler"),
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
    for n in wf["nodes"]:
        by_type.setdefault(n["type"], []).append(n)

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

    # LoopIterationStamp
    if by_type.get("LoopIterationStamp"):
        record("OK", "iteration_stamp", "present")
    elif _is_retake(name):
        record("OK", "iteration_stamp", "n/a (retake workflow, no loop)")
    else:
        record("WARN", "iteration_stamp", "missing (sage tracer iter grouping will be blank)")

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

    # Resolution + length on EmptyLTXVLatentVideo
    for n in by_type.get("EmptyLTXVLatentVideo", []):
        wv = n.get("widgets_values", [])
        if len(wv) < 3:
            continue
        w, h, L = wv[0], wv[1], wv[2]
        if isinstance(w, int) and w % 32 != 0:
            record("ERR", "resolution_div32", f"width {w} not div by 32")
        elif isinstance(h, int) and h % 32 != 0:
            record("ERR", "resolution_div32", f"height {h} not div by 32")
        else:
            record("OK", "resolution_div32", f"{w}x{h}")
        if isinstance(L, int) and (L - 1) % 8 != 0:
            record("ERR", "length_mod8", f"length={L}, (L-1)%8={(L-1)%8}")
        else:
            record("OK", "length_mod8", f"length={L}")

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

    _check_prompt_relay_wiring(wf, by_type, record)

    if _is_retake(name):
        _check_retake_wiring(wf, by_type, record)

    return findings


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
