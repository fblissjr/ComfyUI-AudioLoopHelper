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

Arms 1-4 are designed in the matrix doc but not yet implemented here.
They require coordinated multi-widget edits (sigma curve + sampler +
STG sigma list + anchor cache step) -- added when the user wants to
run them.

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

# First-pass KSamplerSelect that drives `SamplerCustomAdvanced #510`.
# Survived the optimize-baseline pass (sage attention + tiled VAE
# decode etc. do not touch it).
ID_FIRSTPASS_KSAMPLER = 520

DEFAULT_INPUT = "internal/workflows/ltx_i2v_tiled_optimized.draft.json"

# Output paths per arm. Anchored to the canonical staging dir.
_OUTPUTS: dict[str, str] = {
    "arm5": "internal/workflows/ltx_i2v_tiled_arm5.draft.json",
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


_DISPATCH = {
    "arm5": _apply_arm5,
}


def _already_migrated(ed: WorkflowEditor, arm: str) -> bool:
    if arm == "arm5":
        n = ed.find_node(ID_FIRSTPASS_KSAMPLER)
        wv = n.get("widgets_values") or []
        return bool(wv) and wv[0] == "euler"
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
