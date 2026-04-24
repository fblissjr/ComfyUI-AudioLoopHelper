"""Health audit for every workflow under example_workflows/.

Verifies each shipped workflow matches the current CLAUDE.md invariants:

- Sage: `AudioLoopHelperSageAttention` with `auto_mask_aware`, mode=0
  (no legacy `PathchSageAttentionKJ`, no bypassed sage).
- Iteration stamp: `LoopIterationStamp` present (required for sage-tracer
  per-iter grouping).
- Prompt schedule: `TimestampPromptScheduleBatchEncode` (post-2026-04-22),
  not the legacy `CachedTextEncode_AudioLoop` + `TimestampPromptSchedule`
  in-loop pair.
- Sampler chain: `BasicScheduler linear_quadratic 8 1` + `ModelSamplingSD3
  shift=13` + `KSamplerSelect euler` + `CFGGuider cfg=1`. STG variants
  use `MultimodalGuider` instead of CFGGuider, checked accordingly.
- Resolution: width + height divisible by 32 (single-stage distilled).
- Length: `(length - 1) % 8 == 0` on `EmptyLTXVLatentVideo`.
- Preprocess: `LTXVPreprocess.img_compression >= 18` (img_compression=0
  triggers the frozen-first-frame bug; 18 is the Lightricks upstream
  value, 35 is comfy-core default).
- Decoder: `LTXVTiledVAEDecode` preferred over generic `VAEDecodeTiled`.

Exits 0 on all-green, 1 on any ERR. WARNs do not fail the check.

Usage:
    uv run --group dev python scripts/audit_workflows.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import orjson


class Finding(NamedTuple):
    status: str  # OK | WARN | ERR
    workflow: str
    check: str
    message: str


REPO_ROOT = Path(__file__).resolve().parent.parent
WF_DIR = REPO_ROOT / "example_workflows"

# Expected sampler chain widgets (first N elements)
EXPECTED_CHAIN = {
    "BasicScheduler": (["linear_quadratic", 8, 1], "basic_scheduler"),
    "ModelSamplingSD3": ([13], "model_sampling_shift"),
    "KSamplerSelect": (["euler"], "sampler_type"),
}


def _audit_one(wf_path: Path) -> list[Finding]:
    findings: list[Finding] = []
    name = wf_path.name
    wf = orjson.loads(wf_path.read_bytes())
    by_type: dict[str, list[dict]] = {}
    for n in wf["nodes"]:
        by_type.setdefault(n["type"], []).append(n)

    def ok(check: str, msg: str = ""):
        findings.append(Finding("OK", name, check, msg))

    def warn(check: str, msg: str):
        findings.append(Finding("WARN", name, check, msg))

    def err(check: str, msg: str):
        findings.append(Finding("ERR", name, check, msg))

    # Sage node
    sage = by_type.get("AudioLoopHelperSageAttention", [])
    if not sage:
        if by_type.get("PathchSageAttentionKJ"):
            err("sage_node", "uses legacy PathchSageAttentionKJ instead of AudioLoopHelperSageAttention")
        else:
            warn("sage_node", "no sage node present")
    else:
        all_good = True
        for n in sage:
            wv = n.get("widgets_values", [])
            if not wv or wv[0] != "auto_mask_aware":
                warn("sage_mode", f"mode={wv[0] if wv else None!r} (expected auto_mask_aware)")
                all_good = False
            if n.get("mode", 0) != 0:
                warn("sage_active", f"mode field={n.get('mode')} (4=bypassed)")
                all_good = False
        if all_good:
            ok("sage", "AudioLoopHelperSageAttention auto_mask_aware active")

    # LoopIterationStamp
    if by_type.get("LoopIterationStamp"):
        ok("iteration_stamp", "present")
    else:
        warn("iteration_stamp", "missing (sage tracer iter grouping will be blank)")

    # Prompt schedule
    batch = by_type.get("TimestampPromptScheduleBatchEncode", [])
    legacy_cache = by_type.get("CachedTextEncode_AudioLoop", [])
    legacy_sched = by_type.get("TimestampPromptSchedule", [])
    if batch:
        ok("prompt_schedule", "TimestampPromptScheduleBatchEncode (current)")
    elif legacy_cache or legacy_sched:
        err(
            "prompt_schedule",
            f"LEGACY: CachedTextEncode={len(legacy_cache)}, "
            f"TimestampPromptSchedule={len(legacy_sched)} "
            "-- should be TimestampPromptScheduleBatchEncode",
        )
    elif "validator" in name:
        ok("prompt_schedule", "n/a (validator workflow)")
    else:
        warn("prompt_schedule", "no prompt schedule node")

    # Sampler chain (skip validator; stg handles its own guider below)
    skip_chain = "validator" in name
    for t, (expected_wv, check_name) in EXPECTED_CHAIN.items():
        nodes = by_type.get(t, [])
        if not nodes:
            if not skip_chain and "stg" not in name:
                warn(check_name, f"no {t} found")
            continue
        mismatches = [
            n for n in nodes
            if n.get("widgets_values", [])[: len(expected_wv)] != expected_wv
        ]
        if mismatches:
            for n in mismatches:
                warn(check_name, f"{t}(id={n['id']}) widgets={n.get('widgets_values', [])}")
        else:
            ok(check_name, f"{t}={expected_wv}")

    # Guider: cfg=1 for all except STG variant
    cfg_nodes = by_type.get("CFGGuider", [])
    mmg_nodes = by_type.get("MultimodalGuider", [])
    if "stg" in name:
        if mmg_nodes:
            ok("guider", "MultimodalGuider (STG hybrid)")
        elif cfg_nodes:
            warn("guider", "STG workflow using CFGGuider")
    elif cfg_nodes:
        mismatches = [n for n in cfg_nodes if n.get("widgets_values", []) != [1]]
        if mismatches:
            for n in mismatches:
                warn("cfg_value", f"CFGGuider(id={n['id']}) cfg={n.get('widgets_values', [])}")
        else:
            ok("cfg_value", "cfg=1")
    elif "validator" not in name:
        warn("guider", "no CFGGuider")

    # Resolution + length
    for n in by_type.get("EmptyLTXVLatentVideo", []):
        wv = n.get("widgets_values", [])
        if len(wv) < 3:
            continue
        w, h, L = wv[0], wv[1], wv[2]
        if isinstance(w, int) and w % 32 != 0:
            err("resolution_div32", f"width {w} not div by 32")
        elif isinstance(h, int) and h % 32 != 0:
            err("resolution_div32", f"height {h} not div by 32")
        else:
            ok("resolution_div32", f"{w}x{h}")
        if isinstance(L, int) and (L - 1) % 8 != 0:
            err("length_mod8", f"length={L}, (L-1)%8={(L-1)%8}")
        else:
            ok("length_mod8", f"length={L}")

    # LTXVPreprocess compression
    for n in by_type.get("LTXVPreprocess", []):
        wv = n.get("widgets_values", [])
        if not wv or not isinstance(wv[0], int):
            continue
        if wv[0] == 0:
            err("preprocess_compression", "img_compression=0 (frozen-first-frame bug)")
        elif wv[0] < 18:
            warn("preprocess_compression", f"img_compression={wv[0]} < 18")
        else:
            ok("preprocess_compression", f"img_compression={wv[0]}")

    # Decoder
    if by_type.get("LTXVTiledVAEDecode"):
        ok("decoder", "LTXVTiledVAEDecode")
    elif by_type.get("VAEDecodeTiled"):
        warn("decoder", "VAEDecodeTiled (generic) -- consider LTXVTiledVAEDecode")

    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="Print OK findings too.")
    args = ap.parse_args()

    all_findings: list[Finding] = []
    for wf_path in sorted(WF_DIR.glob("*.json")):
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
