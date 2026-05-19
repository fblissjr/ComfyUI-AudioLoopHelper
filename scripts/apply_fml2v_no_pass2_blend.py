"""apply_fml2v_no_pass2_blend.

Last updated: 2026-05-19

Stages a diagnostic variant that bypasses `LTXVAddGuideMulti #2182`
(the loop-body pass2 image-guide injection). Pairs with the smoke-test
loop config so the variant is queue-ready for a fast A/B against the
smoke baseline.

Hypothesis: the canonical audio-loop workflow uses zero image-based
guides per loop iter (only `LTXVAddLatentGuide` at `latent_idx=-1`,
fed by init `VAEEncode`'d once + `LatentContextExtract` from prev iter).
The fml2v build's pass2 `#2182` injects the same two reference images
(first + last frame) into every iter at static latent indices regardless
of song timeline position. Combined with pass1 variance starvation, the
resulting pass2 substrate is too noisy for those image guides to land
cleanly — they leak through as blended texture.

Mutations (idempotent):
  1. `#2182.mode = 4` (bypass). Pass2 then samples from
     `LatentUpsampler(pass1_output) + audio + LTXVConditioning` with no
     per-iter image-guide injection.
  2. Smoke loop config (2 iters via widget) so the diagnostic runs in
     ~2-3 minutes.

Bypass-vs-removal: bypass keeps the node in graph so revert is trivial
(no UI re-add needed); CONDITIONING + LATENT passthrough flow via
same-type input→output bypass routing.

Usage:
    uv run --group dev python scripts/apply_fml2v_no_pass2_blend.py
    uv run --group dev python scripts/apply_fml2v_no_pass2_blend.py --revert
    uv run --group dev python scripts/apply_fml2v_no_pass2_blend.py --dry-run
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

DEFAULT_OUTPUT = "internal/scratch/fml2v_var_d_audio_loop_no_pass2_blend.json"

PASS2_ADD_GUIDE_MULTI = 2182  # see _BENCH_PRE_PASS2_GUIDE_MULTI in build_fml2v_audio_loop.py


def _already_toggled(ed: WorkflowEditor) -> bool:
    stash = ed.wf.get("properties", {}).get("build_fml2v_phase5") or {}
    tlo_id = stash.get("tlo")
    if tlo_id is None:
        return False
    try:
        tlo = ed.find_node(tlo_id)
        guide_multi = ed.find_node(PASS2_ADD_GUIDE_MULTI)
    except ValueError:
        return False
    return smoke_iters_applied(tlo) and not is_active(guide_multi)


def _apply(ed: WorkflowEditor) -> None:
    stash = phase5_stash(ed)
    try:
        guide_multi = ed.find_node(PASS2_ADD_GUIDE_MULTI)
    except ValueError as e:
        raise SystemExit(
            f"LTXVAddGuideMulti #{PASS2_ADD_GUIDE_MULTI} (pass2) not found. "
            "Expected from fml2v build's pre-pass2 chain."
        ) from e
    prev_mode = guide_multi.get("mode", 0)
    guide_multi["mode"] = 4
    print(f"  LTXVAddGuideMulti #{PASS2_ADD_GUIDE_MULTI} (pass2): mode {prev_mode} -> 4 (bypassed)")
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
        variant_label="no-pass2-blend",
        next_steps=[
            "1. bash start_experiment.sh default",
            f"2. Reload {output_path.name} in ComfyUI",
            "3. Wire LoadAudio #2307 + LoadImage slots if not already set",
            "4. Queue prompt — pass2 sampler runs WITHOUT per-iter image-guide injection",
            "5. Compare loop-body frames (~20-37s) vs smoke baseline.",
            "   Expected: noise/blend reduces, content may look more 'lost' in the",
            "   pass1-starvation zone since the image guides aren't masking it anymore.",
        ],
    )


if __name__ == "__main__":
    main()
