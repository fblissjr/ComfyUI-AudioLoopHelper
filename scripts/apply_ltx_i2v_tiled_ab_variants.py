"""apply_ltx_i2v_tiled_ab_variants.

Last updated: 2026-05-11

Produces A/B variants of the LTX 2.3 I2V tiled-sampler optimized
baseline (Arm 0 = the output of `apply_ltx_i2v_tiled_optimizations.py`).
The A/B matrix is documented in
`internal/design/ltx_i2v_tiled_sampler_curve_ab.md`; this script
implements the per-arm edits and writes one variant JSON per arm.

Status note (2026-05-11): Arm 0 baseline now bakes in the keeper
config from the curve A/B (canonical 9-pt sigmas + euler + matched
anchor/STG remap + RES4LYF `Sigmas Easing` removed). That collapses
the matrix:

- arm1 (canonical curve + euler + matched downstream) is now
  identity-vs-baseline. RETIRED.
- arm2 (arm1 - easing) is now identity-vs-baseline. RETIRED.
- arm5 (euler-only on the source 14-pt curve) no longer maps onto
  the post-Phase-3 baseline cleanly. RETIRED.

Arms that remain meaningful against the keeper baseline:

- no_rtx (upscaler-stack A/B, U-B): bypass `RTXVideoSuperResolution`
  via `mode=4`. The post-decode IMAGE flows straight from the tiled
  VAE decode into `VHS_VideoCombine`. Tests "is the NVIDIA pixel-
  space VSR adding value on top of the LTXV 2x latent upsample, or
  is it redundant?"

- arm3 (curve A/B, anchor ablation): Arm 1 with
  `LTXLatentAnchorAware` internally bypassed (sets widget[5]
  bypass=True). The node stays in the graph (MODEL pass-through)
  but its patch returns the model unchanged. Tests "how much does
  the content-aware latent anchor contribute?" If the answer is
  "not much," the anchor + its reference-image energy map become
  optional for this pipeline shape.

- arm4 (curve A/B, STG warmup ablation): Arm 1 with
  `STGGuiderAdvanced` cfg + stg_scale value lists flattened to
  all-1s (kills the 2-step `2, 1.5, ...` warmup at the curve top).
  The STG layer-skipping mechanism is already disabled by the
  `[9999]` placeholder indices, so flat cfg/stg makes the guider
  behave as plain `CFGGuider(cfg=1)` without needing to swap the
  node out. Tests "is the 2-step CFG warmup at sigma=1.0 doing
  any real work?"

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

from workflow_utils import WorkflowEditor

ID_ANCHOR = 731                  # LTXLatentAnchorAware
ID_STG_GUIDER = 653              # STGGuiderAdvanced
ID_RTX_VSR = 755                 # RTXVideoSuperResolution (post-decode pixel VSR)

# ComfyUI mode constants: 0 = active, 4 = bypass (passes input to
# output of same type only).
MODE_ACTIVE = 0
MODE_BYPASS = 4

# Arm 4: flat-1 cfg + stg_scale tables. With STG layer-skipping
# already disabled by `[9999]` placeholders in the keeper baseline,
# this drops the only remaining warmup signal at the top of the
# curve. Guider then behaves like plain CFGGuider(cfg=1) without
# node replacement.
ARM4_FLAT_TABLE = "1, 1, 1, 1, 1, 1, 1, 1"

# LTXLatentAnchorAware widget index 5 is the `bypass` flag. True
# returns the input model unchanged (patch is a no-op) while keeping
# the node in the graph topologically -- no rewiring required.
ANCHOR_BYPASS_WIDGET_IDX = 5

DEFAULT_INPUT = "internal/workflows/ltx_i2v_tiled_optimized.draft.json"

# Output paths per arm. Anchored to the canonical staging dir.
_OUTPUTS: dict[str, str] = {
    "arm3": "internal/workflows/ltx_i2v_tiled_arm3.draft.json",
    "arm4": "internal/workflows/ltx_i2v_tiled_arm4.draft.json",
    "no_rtx": "internal/workflows/ltx_i2v_tiled_no_rtx.draft.json",
}

# Available arms. Add to both this set and the dispatch table below
# when implementing arms 1-4.
ARMS = tuple(_OUTPUTS.keys())


def _output_for(arm: str) -> Path:
    return Path(_OUTPUTS[arm])


def _set_widget(node: dict, idx: int, value, label: str) -> None:
    wv = list(node.get("widgets_values") or [])
    if idx >= len(wv):
        raise SystemExit(f"Node #{node.get('id')} widget index {idx} out of range ({len(wv)} widgets).")
    old = wv[idx]
    wv[idx] = value
    node["widgets_values"] = wv
    print(f"  #{node.get('id')} [{node.get('type')}] {label}: {old!r} -> {value!r}")


def _apply_arm3(ed: WorkflowEditor) -> None:
    """Arm 3 -- LTXLatentAnchorAware internally bypassed.

    The anchor's `bypass` widget (idx 5) flips to True; its `patch`
    method returns the model unchanged. Topology preserved -- the
    STGGuiderAdvanced still reads from the anchor's MODEL output,
    just gets the pass-through.
    """
    _set_widget(ed.find_node(ID_ANCHOR), ANCHOR_BYPASS_WIDGET_IDX, True, "anchor bypass")


def _apply_arm4(ed: WorkflowEditor) -> None:
    """Arm 4 -- STG warmup flattened.

    `STGGuiderAdvanced`'s `cfg_values` + `stg_scale_values` widgets
    drop from the baseline's `2, 1.5, 1, 1, 1, 1, 1, 1` (2-step
    warmup) to flat `1, 1, 1, 1, 1, 1, 1, 1`. Layer-skipping is
    already disabled by the keeper baseline's `[9999]` placeholder
    layer indices, so this flattening reduces the guider to plain
    `CFGGuider(cfg=1)` in behavior without swapping the node.
    """
    stg = ed.find_node(ID_STG_GUIDER)
    _set_widget(stg, 3, ARM4_FLAT_TABLE, "stg cfg_values (no warmup)")
    _set_widget(stg, 4, ARM4_FLAT_TABLE, "stg stg_scale_values (no warmup)")


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
    "arm3": _apply_arm3,
    "arm4": _apply_arm4,
    "no_rtx": _apply_no_rtx,
}


def _already_migrated(ed: WorkflowEditor, arm: str) -> bool:
    if arm == "no_rtx":
        return ed.find_node(ID_RTX_VSR).get("mode") == MODE_BYPASS
    if arm == "arm3":
        anchor = ed.find_node(ID_ANCHOR).get("widgets_values") or []
        return (
            len(anchor) > ANCHOR_BYPASS_WIDGET_IDX
            and anchor[ANCHOR_BYPASS_WIDGET_IDX] is True
        )
    if arm == "arm4":
        stg = ed.find_node(ID_STG_GUIDER).get("widgets_values") or []
        return (
            len(stg) > 4
            and stg[3] == ARM4_FLAT_TABLE
            and stg[4] == ARM4_FLAT_TABLE
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
