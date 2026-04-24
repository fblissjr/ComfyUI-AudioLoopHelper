"""Health audit for every workflow under example_workflows/.

Verifies each shipped workflow matches the current CLAUDE.md invariants:
sage node + mode, iteration stamp, batch-encode prompt schedule,
distilled sampler chain (linear_quadratic 8 1 + shift=13 + euler +
cfg=1, with STG-variant exception), resolution div-32, (L-1)%8==0,
LTXVPreprocess img_compression >= 18, LTXVTiledVAEDecode preferred.

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


def _is_validator(name: str) -> bool:
    return "validator" in name


def _is_stg(name: str) -> bool:
    return "stg" in name


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

    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="Print OK findings too.")
    args = ap.parse_args()

    all_findings: list[Finding] = []
    for wf_path in sorted(EXAMPLE_WORKFLOWS_DIR.glob("*.json")):
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
