"""apply_fml2v_pass1_recovery.

Last updated: 2026-05-19

Stages a diagnostic variant exercising the "pass1 variance starvation"
hypothesis: bumps the real first-frame strength source + un-bypasses
AdaIN_p1 + applies smoke iters. All in one pass so the user can reload,
wire LoadImages, and queue.

The `LTXVAddGuideMulti #2182` strength widgets are runtime-dead — its
strength inputs are wired from `PrimitiveFloat #2110` (first) and
`#2108` (last), and ComfyUI's runtime ignores a node's own widget value
when the input is wired. Strength changes in the UI on `#2182` have no
effect; `PrimitiveFloat #2110` is the real lever.

Pass2 sigmas are `[0.85, 0.7250, 0.4219, 0.0]` — only 3 sampling steps
starting at noise 0.85. This expects pass1 to deliver an already-mostly-
denoised latent. If pass1 is variance-starved (likely with AdaIN_p1
bypassed = no variance normalization), pass2 has no budget to recover
and decodes to LTX's prior (yellow stripes).

Mutations (idempotent):
  1. `PrimitiveFloat #2110` "FIRST FRAME STRENGTH": 0.7 → 1.0.
  2. `LTXVAdainLatent #2347` (AdaIN_p1): mode 4 → 0.
  3. Smoke loop config (2 iters via widget).

AdaIN_final `#2365` stays bypassed for this diagnostic. If AdaIN_p1
alone recovers pass1 variance, pass2 should produce variance-rich output
naturally — layering AdaIN_final on a still-broken pass1 confuses the
signal.

Diagnostic outcomes for loop body frames (~20-37s):
  - Coherent video → variance starvation was the cause; AdaIN_p1
    normalizes acceptable variance into a usable latent.
  - Yellow stripes persist → variance wasn't the bottleneck;
    issue is elsewhere (sigmas, conditioning routing, etc).
  - Black frames (NaN) → confirms variance starvation; divide-by-
    near-zero std produces NaN that propagates through pass2.

Usage:
    uv run --group dev python scripts/apply_fml2v_pass1_recovery.py
    uv run --group dev python scripts/apply_fml2v_pass1_recovery.py --revert
    uv run --group dev python scripts/apply_fml2v_pass1_recovery.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers._fml2v_helpers import (
    apply_smoke_iters_config,
    make_argparser,
    phase5_stash,
    revert_variant,
    smoke_iters_applied,
    stage_variant,
)
from workflow_utils import WorkflowEditor, is_active

DEFAULT_OUTPUT = "internal/scratch/fml2v_var_d_audio_loop_pass1_recovery.json"

FIRST_FRAME_STRENGTH_FLOAT = 2110   # PrimitiveFloat "FIRST FRAME STRENGTH" — real lever (widget on #2182 is wire-overridden)
LAST_FRAME_STRENGTH_FLOAT = 2108    # already 1.0; verified only
ADAIN_P1 = 2347                     # LTXVAdainLatent post-pass1 normalization
TARGET_FIRST_FRAME_STRENGTH = 1.0


def _already_toggled(ed: WorkflowEditor) -> bool:
    stash = ed.wf.get("properties", {}).get("build_fml2v_phase5") or {}
    tlo_id = stash.get("tlo")
    if tlo_id is None:
        return False
    try:
        tlo = ed.find_node(tlo_id)
        first_float = ed.find_node(FIRST_FRAME_STRENGTH_FLOAT)
        adain_p1 = ed.find_node(ADAIN_P1)
    except ValueError:
        return False
    first_bumped = (first_float.get("widgets_values") or [None])[0] == TARGET_FIRST_FRAME_STRENGTH
    return smoke_iters_applied(tlo) and first_bumped and is_active(adain_p1)


def _apply(ed: WorkflowEditor) -> None:
    stash = phase5_stash(ed)

    first_float = ed.find_node(FIRST_FRAME_STRENGTH_FLOAT)
    wv = first_float.get("widgets_values") or [0.0]
    prev = wv[0]
    wv[0] = TARGET_FIRST_FRAME_STRENGTH
    first_float["widgets_values"] = wv
    print(f"  PrimitiveFloat #{FIRST_FRAME_STRENGTH_FLOAT} \"FIRST FRAME STRENGTH\": {prev} -> {TARGET_FIRST_FRAME_STRENGTH}")

    # Sanity report: LAST FRAME STRENGTH is already 1.0 (verified, not mutated).
    last_float = ed.find_node(LAST_FRAME_STRENGTH_FLOAT)
    last_val = (last_float.get("widgets_values") or [None])[0]
    print(f"  PrimitiveFloat #{LAST_FRAME_STRENGTH_FLOAT} \"LAST FRAME STRENGTH\": {last_val} (verified, unchanged)")

    adain_p1 = ed.find_node(ADAIN_P1)
    prev_mode = adain_p1.get("mode", 0)
    adain_p1["mode"] = 0
    print(f"  LTXVAdainLatent #{ADAIN_P1} (AdaIN_p1): mode {prev_mode} -> 0 (un-bypassed)")

    apply_smoke_iters_config(ed, stash["tlo"])


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
        variant_label="pass1-recovery",
        next_steps=[
            "1. bash start_experiment.sh default",
            f"2. Reload {output_path.name} in ComfyUI",
            "3. Wire LoadImage slots (first/middle/last frames) + LoadAudio",
            "4. Queue prompt — diagnostic outcomes for loop body frames (~20-37s):",
            "     coherent video    → pass1 variance starvation was the bottleneck; AdaIN_p1 fixed it",
            "     yellow stripes    → variance wasn't the issue; look elsewhere (sigmas/conditioning)",
            "     black frames NaN  → confirms variance starvation; need to fix pass1 substrate",
            "                          (add InplaceKJ frame-0 lock, or LatentContextExtract from prev iter)",
        ],
    )


if __name__ == "__main__":
    main()
