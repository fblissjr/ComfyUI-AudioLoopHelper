"""apply_ltx_i2v_tiled_ab_variants.

Last updated: 2026-05-11

Produces A/B variants of the LTX 2.3 I2V tiled-sampler optimized
baseline (Arm 0 = the output of `apply_ltx_i2v_tiled_optimizations.py`).
The A/B matrix is documented in
`internal/design/ltx_i2v_tiled_sampler_curve_ab.md`; this script
implements the per-arm edits and writes one variant JSON per arm.

Arms implemented:

- arm5 (cheapest single-knob): swap first-pass `KSamplerSelect` from
  `euler_ancestral` to `euler`. Keeps the 14-pt sigma curve, the
  RES4LYF easing, the anchor's `cache_at_step=6`, and the STG warmup
  intact. Isolates the question "is the ancestral noise injection
  doing anything?"

- arm1 (headline curve-length test): coordinated multi-widget edit
  that replaces the 14-pt sigma curve with the canonical distilled
  9-pt curve `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725,
  0.421875, 0.0`, swaps `KSamplerSelect` to `euler`, remaps
  `LTXLatentAnchorAware.cache_at_step` from 6 to 5 (matches the
  same-sigma slot on the new curve), and remaps `STGGuiderAdvanced`'s
  sigma-keyed cfg/stg/rescale tables onto the new curve. RES4LYF
  easing is kept in place and acts on the new (shorter) curve. The
  headline question is "8-step canonical euler vs 13-step
  euler_ancestral + easing?"

- no_rtx (upscaler-stack A/B, U-B): bypass `RTXVideoSuperResolution`
  via `mode=4`. The post-decode IMAGE flows straight from the tiled
  VAE decode into `VHS_VideoCombine`. Tests "is the NVIDIA pixel-
  space VSR adding value on top of the LTXV 2x latent upsample, or
  is it redundant?" Sampler / curve / anchor / STG unchanged from
  Arm 0.

- arm2 (curve A/B, easing ablation): Arm 1 with the RES4LYF
  `Sigmas Easing` bypassed via `mode=4`. Same canonical 9-pt curve
  reaches the sampler raw instead of being warped by the cubic
  in_out easing. SIGMAS input/output share type so ComfyUI's bypass
  passes the raw ManualSigmas straight through. Tests "is the
  easing doing anything once the curve is canonical?" Anchor and
  STG keep their Arm-1 remap (they already read the un-eased
  curve; nothing else moves).

Arms 2-4 of the curve matrix (drop easing / drop anchor / drop STG
warmup) plus arm `no_ltxv_upsample` (the deeper U-C topology
surgery) are designed in the matrix doc; add when the user wants
to run them.

Usage:
    uv run --group dev python scripts/apply_ltx_i2v_tiled_ab_variants.py --arm arm5
    uv run --group dev python scripts/apply_ltx_i2v_tiled_ab_variants.py --arm arm5 --dry-run
    uv run --group dev python scripts/apply_ltx_i2v_tiled_ab_variants.py --arm arm5 --revert

`--input` defaults to the canonical Arm 0 draft. `--output` defaults
to a per-arm path under `internal/workflows/`. Idempotent on the
OUTPUT path; the arm-specific signature is what `_already_migrated`
checks. `--revert` deletes the arm-specific output staging file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apply_canonical_sigmas import CANONICAL_SIGMAS
from workflow_utils import WorkflowEditor

# First-pass KSamplerSelect that drives `SamplerCustomAdvanced #510`.
# Survived the optimize-baseline pass (sage attention + tiled VAE
# decode etc. do not touch it).
ID_FIRSTPASS_KSAMPLER = 520
ID_FIRSTPASS_SIGMAS = 527        # ManualSigmas feeding the easing + anchor
ID_ANCHOR = 731                  # LTXLatentAnchorAware
ID_STG_GUIDER = 653              # STGGuiderAdvanced
ID_SIGMAS_EASING = 652           # RES4LYF Sigmas Easing (curve warp)
ID_RTX_VSR = 755                 # RTXVideoSuperResolution (post-decode pixel VSR)

# ComfyUI mode constants: 0 = active, 4 = bypass (passes input to
# output of same type only).
MODE_ACTIVE = 0
MODE_BYPASS = 4

# Arm-1 STG widget remap: 14-pt curve had 13-entry cfg/stg/rescale
# tables (one per sampler transition) and a 14-entry layers table
# (off-by-one but harmless because all entries are the same
# disabled-placeholder). New 9-pt curve gets 8-entry transition tables
# and a 9-entry layers table. Preserves the 2-step cfg/stg warmup at
# the top of the curve; flat 1s elsewhere; layer-skipping stays off.
ARM1_STG_SIGMAS = CANONICAL_SIGMAS
ARM1_STG_CFG = "2, 1.5, 1, 1, 1, 1, 1, 1"
ARM1_STG_STG_SCALE = "2, 1.5, 1, 1, 1, 1, 1, 1"
ARM1_STG_RESCALE = "1, 1, 1, 1, 1, 1, 1, 1"
ARM1_STG_LAYERS = "[9999], [9999], [9999], [9999], [9999], [9999], [9999], [9999], [9999]"

# Anchor cache step remap. Original `cache_at_step=6` against the
# 14-pt curve fires at sigma=0.812. Closest higher-sigma slot on the
# canonical 9-pt curve is idx 5 (sigma=0.909375); idx 6 is 0.725.
# Step 5 is the closest match.
ARM1_ANCHOR_CACHE_STEP = 5

DEFAULT_INPUT = "internal/workflows/ltx_i2v_tiled_optimized.draft.json"

# Output paths per arm. Anchored to the canonical staging dir.
_OUTPUTS: dict[str, str] = {
    "arm1": "internal/workflows/ltx_i2v_tiled_arm1.draft.json",
    "arm2": "internal/workflows/ltx_i2v_tiled_arm2.draft.json",
    "arm5": "internal/workflows/ltx_i2v_tiled_arm5.draft.json",
    "no_rtx": "internal/workflows/ltx_i2v_tiled_no_rtx.draft.json",
}

# Available arms. Add to both this set and the dispatch table below
# when implementing arms 1-4.
ARMS = tuple(_OUTPUTS.keys())


def _output_for(arm: str) -> Path:
    return Path(_OUTPUTS[arm])


def _apply_arm5(ed: WorkflowEditor) -> None:
    """Arm 5 -- swap first-pass KSamplerSelect to plain euler.

    Single widget edit. Keeps everything else identical to Arm 0
    (14-pt curve, RES4LYF easing, anchor cache_at_step=6, STG
    warmup). The point of arm5 is to isolate the ancestral-noise
    question from the curve-length question.
    """
    n = ed.find_node(ID_FIRSTPASS_KSAMPLER)
    if n.get("type") != "KSamplerSelect":
        raise SystemExit(
            f"Expected KSamplerSelect at #{ID_FIRSTPASS_KSAMPLER}, got {n.get('type')!r}. "
            "Did the canonical Arm 0 draft renumber nodes?"
        )
    wv = list(n.get("widgets_values") or [])
    if not wv:
        raise SystemExit(f"KSamplerSelect #{ID_FIRSTPASS_KSAMPLER} missing widget value.")
    old = wv[0]
    wv[0] = "euler"
    n["widgets_values"] = wv
    print(f"  KSamplerSelect #{ID_FIRSTPASS_KSAMPLER}: {old!r} -> 'euler'")


def _set_widget(node: dict, idx: int, value, label: str) -> None:
    wv = list(node.get("widgets_values") or [])
    if idx >= len(wv):
        raise SystemExit(f"Node #{node.get('id')} widget index {idx} out of range ({len(wv)} widgets).")
    old = wv[idx]
    wv[idx] = value
    node["widgets_values"] = wv
    print(f"  #{node.get('id')} [{node.get('type')}] {label}: {old!r} -> {value!r}")


def _apply_arm1(ed: WorkflowEditor) -> None:
    """Arm 1 -- canonical 9-pt curve + euler + matched downstream remap.

    Coordinated because the four widgets must move together: anchor's
    cache_at_step is indexed against the sigma list, STG's sigma-keyed
    cfg/stg tables key against the same list, and the sampler's noise
    behavior changes with both family and curve length. Partial
    application leaves an inconsistent curve.
    """
    _set_widget(ed.find_node(ID_FIRSTPASS_SIGMAS), 0, CANONICAL_SIGMAS, "sigmas")
    _set_widget(ed.find_node(ID_FIRSTPASS_KSAMPLER), 0, "euler", "sampler")
    _set_widget(ed.find_node(ID_ANCHOR), 1, ARM1_ANCHOR_CACHE_STEP, "cache_at_step")
    stg = ed.find_node(ID_STG_GUIDER)
    _set_widget(stg, 2, ARM1_STG_SIGMAS, "stg sigmas")
    _set_widget(stg, 3, ARM1_STG_CFG, "stg cfg_values")
    _set_widget(stg, 4, ARM1_STG_STG_SCALE, "stg stg_scale_values")
    _set_widget(stg, 5, ARM1_STG_RESCALE, "stg stg_rescale_values")
    _set_widget(stg, 6, ARM1_STG_LAYERS, "stg layers_indices")


def _apply_arm2(ed: WorkflowEditor) -> None:
    """Arm 2 = Arm 1 + bypass `Sigmas Easing`.

    Same `cubic_in_out(t**0.7)` warper that the original workflow
    inherited; bypass routes the canonical sigmas raw to the sampler.
    SIGMAS in/out share type so ComfyUI's mode=4 passthrough works
    without rewiring.
    """
    _apply_arm1(ed)
    n = ed.find_node(ID_SIGMAS_EASING)
    if n.get("type") != "Sigmas Easing":
        raise SystemExit(
            f"Expected `Sigmas Easing` at #{ID_SIGMAS_EASING}, got {n.get('type')!r}."
        )
    n["mode"] = MODE_BYPASS
    print(f"  #{ID_SIGMAS_EASING} [{n['type']}]: mode {MODE_ACTIVE} -> {MODE_BYPASS} (bypassed)")


def _apply_no_rtx(ed: WorkflowEditor) -> None:
    """Bypass `RTXVideoSuperResolution`. ComfyUI's mode=4 passes the IMAGE
    input through to the IMAGE output since they share type, so no
    rewiring is needed.
    """
    n = ed.find_node(ID_RTX_VSR)
    if n.get("type") != "RTXVideoSuperResolution":
        raise SystemExit(
            f"Expected RTXVideoSuperResolution at #{ID_RTX_VSR}, got {n.get('type')!r}."
        )
    n["mode"] = MODE_BYPASS
    print(f"  #{ID_RTX_VSR} [{n['type']}]: mode {MODE_ACTIVE} -> {MODE_BYPASS} (bypassed)")


_DISPATCH = {
    "arm1": _apply_arm1,
    "arm2": _apply_arm2,
    "arm5": _apply_arm5,
    "no_rtx": _apply_no_rtx,
}


def _already_migrated(ed: WorkflowEditor, arm: str) -> bool:
    if arm == "arm5":
        n = ed.find_node(ID_FIRSTPASS_KSAMPLER)
        wv = n.get("widgets_values") or []
        return bool(wv) and wv[0] == "euler"
    if arm == "arm1":
        # Check one widget per touched node so a crash mid-application
        # (between dispatch steps) doesn't short-circuit re-application.
        sigmas = ed.find_node(ID_FIRSTPASS_SIGMAS).get("widgets_values") or []
        sampler = ed.find_node(ID_FIRSTPASS_KSAMPLER).get("widgets_values") or []
        anchor = ed.find_node(ID_ANCHOR).get("widgets_values") or []
        stg = ed.find_node(ID_STG_GUIDER).get("widgets_values") or []
        return (
            bool(sigmas) and sigmas[0] == CANONICAL_SIGMAS
            and bool(sampler) and sampler[0] == "euler"
            and len(anchor) > 1 and anchor[1] == ARM1_ANCHOR_CACHE_STEP
            and len(stg) > 2 and stg[2] == ARM1_STG_SIGMAS
        )
    if arm == "no_rtx":
        return ed.find_node(ID_RTX_VSR).get("mode") == MODE_BYPASS
    if arm == "arm2":
        # Arm 2 builds on Arm 1's widget set + bypasses the easing.
        # Re-check the Arm-1 signature, plus the easing-bypass mode.
        sigmas = ed.find_node(ID_FIRSTPASS_SIGMAS).get("widgets_values") or []
        sampler = ed.find_node(ID_FIRSTPASS_KSAMPLER).get("widgets_values") or []
        return (
            bool(sigmas) and sigmas[0] == CANONICAL_SIGMAS
            and bool(sampler) and sampler[0] == "euler"
            and ed.find_node(ID_SIGMAS_EASING).get("mode") == MODE_BYPASS
        )
    raise SystemExit(f"Unknown arm: {arm!r}")


def _migrate(input_path: Path, output_path: Path, arm: str, dry_run: bool) -> None:
    if output_path.exists() and input_path != output_path:
        if _already_migrated(WorkflowEditor(output_path), arm):
            print(f"{output_path.name}: already migrated, skipping. Run --revert to reset.")
            return

    if dry_run:
        print(f"would copy {input_path} -> {output_path}")
        print(f"would apply {arm}.")
        return

    if input_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        print(f"  copied {input_path} -> {output_path}")

    ed = WorkflowEditor(output_path)
    if _already_migrated(ed, arm):
        print(f"{output_path.name}: already migrated, skipping.")
        return

    _DISPATCH[arm](ed)
    ed.save()
    print(f"  wrote {output_path}")
    print()
    print("Next steps:")
    print(f"  1. Validate JSON: python3 -c \"import json; json.load(open('{output_path}'))\"")
    print(f"  2. Audit:         uv run --group dev python scripts/audit_workflows.py {output_path}")
    print(f"  3. A/B render against the baseline at {DEFAULT_INPUT}.")


def _revert(output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
        print(f"removed {output_path}")
    else:
        print(f"{output_path} does not exist; nothing to revert.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--arm", required=True, choices=ARMS,
                    help="Which A/B arm to produce.")
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"Source workflow (default: {DEFAULT_INPUT}).")
    ap.add_argument("--output", default=None,
                    help="Output path. Defaults to per-arm path under internal/workflows/.")
    ap.add_argument("--revert", action="store_true",
                    help="Delete the arm-specific output (does not touch --input).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change without writing.")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _output_for(args.arm)

    if args.revert:
        _revert(output_path)
        return

    if not input_path.exists():
        raise SystemExit(
            f"Source not found: {input_path}. "
            f"Run `scripts/apply_ltx_i2v_tiled_optimizations.py --input <orig>` first."
        )

    _migrate(input_path, output_path, arm=args.arm, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
