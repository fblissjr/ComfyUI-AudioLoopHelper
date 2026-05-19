"""apply_fml2v_inplace_kj_p1.

Last updated: 2026-05-19

Stages a diagnostic variant that adds `LTXVImgToVideoInplaceKJ` to loop
body pass1 — the HARD frame-0 lock (init image written into latent[0]
with `noise_mask=0`). `LTXVAddLatentGuide` is a soft pull the sampler
can denoise away; InplaceKJ enforces the frame-0 lock because
`noise_mask=0` means the sampler literally cannot touch frame 0.

Mutations (idempotent):
  1. New `GetNode("vae")` near insertion point (loop body scope).
  2. New `LTXVImgToVideoInplaceKJ` (loop body pass1):
       vae   ← new GetNode "vae"
       latent ← `AudioVideoMask #2340.video_latent` (was feeding #2342)
       image_1 ← existing `GetNode #2220 "firstframe"` (same init image
                  every iter, matching init render's pattern)
       widgets: ["1", 1, 0]   (num_images="1", strength=1, noise=0)
  3. Rewire `AddLatentGuide_frame0 #2342.latent` to read from the new
     InplaceKJ instead of `AudioVideoMask #2340.video_latent`. Keeps
     the downstream chain intact (trailing anchor + cropguides + AdaIN).
  4. Bypass `AddLatentGuide_frame0 #2342` (mode 0 → 4) — redundant now
     that InplaceKJ enforces frame-0 lock with `noise_mask=0`.
  5. Smoke loop config (2 iters via widget).

AdaIN_p1 #2347 stays bypassed (pre-existing canonical state). The
pass1_recovery diagnostic already showed AdaIN_p1 active doesn't help
when the substrate is starved — and with InplaceKJ providing a real
frame-0 anchor, pass1 should produce variance-rich output naturally.

Usage:
    uv run --group dev python scripts/apply_fml2v_inplace_kj_p1.py
    uv run --group dev python scripts/apply_fml2v_inplace_kj_p1.py --revert
    uv run --group dev python scripts/apply_fml2v_inplace_kj_p1.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers._fml2v_helpers import (
    _add_from_template,
    apply_smoke_iters_config,
    find_by_type_and_title,
    make_argparser,
    phase5_stash,
    revert_variant,
    smoke_iters_applied,
    stage_variant,
)
from workflow_utils import WorkflowEditor, is_active

DEFAULT_OUTPUT = "internal/scratch/fml2v_var_d_audio_loop_inplace_kj_p1.json"

AUDIO_VIDEO_MASK = 2340
ADD_LATENT_GUIDE_FRAME0 = 2342
FIRSTFRAME_GETNODE = 2220
INPLACE_KJ_SENTINEL_TITLE = "LTXVImgToVideoInplaceKJ (pass1 hard frame-0 lock)"


def _already_toggled(ed: WorkflowEditor) -> bool:
    stash = ed.wf.get("properties", {}).get("build_fml2v_phase5") or {}
    tlo_id = stash.get("tlo")
    if tlo_id is None:
        return False
    try:
        tlo = ed.find_node(tlo_id)
        frame0_guide = ed.find_node(ADD_LATENT_GUIDE_FRAME0)
    except ValueError:
        return False
    return (
        smoke_iters_applied(tlo)
        and not is_active(frame0_guide)
        and find_by_type_and_title(ed, "LTXVImgToVideoInplaceKJ", INPLACE_KJ_SENTINEL_TITLE) is not None
    )


def _apply(ed: WorkflowEditor) -> None:
    stash = phase5_stash(ed)

    get_vae_id = ed.next_node_id()
    ed.add_node(WorkflowEditor.make_get_node(
        get_vae_id, "vae", "VAE", [1400, 2700], title="Get_vae (p1 InplaceKJ)",
    ))
    print(f"  + GetNode #{get_vae_id} \"vae\" (loop body)")

    inplace_id = _add_from_template(
        ed, "LTXVImgToVideoInplaceKJ", (1650, 2700),
        size=(290, 130),
        title=INPLACE_KJ_SENTINEL_TITLE,
    )
    print(f"  + LTXVImgToVideoInplaceKJ #{inplace_id} (loop body pass1)")

    ed.add_link(get_vae_id, 0, inplace_id, 0, "VAE")
    ed.add_link(AUDIO_VIDEO_MASK, 0, inplace_id, 1, "LATENT")
    ed.add_link(FIRSTFRAME_GETNODE, 0, inplace_id, 2, "IMAGE")
    print(f"    .vae ← #{get_vae_id} (new GetNode \"vae\")")
    print(f"    .latent ← #{AUDIO_VIDEO_MASK}.video_latent")
    print(f"    .image_1 ← #{FIRSTFRAME_GETNODE} (GetNode \"firstframe\")")

    # Rewire so AddLatentGuide_frame0's input comes from the new InplaceKJ,
    # not directly from AudioVideoMask — keeps the trailing-anchor + cropguides
    # chain intact downstream.
    frame0_guide = ed.find_node(ADD_LATENT_GUIDE_FRAME0)
    latent_slot = WorkflowEditor.find_input_slot(frame0_guide, "latent")
    ed.rewire_input(ADD_LATENT_GUIDE_FRAME0, latent_slot, inplace_id, 0, "LATENT")
    print(f"  rewire #{ADD_LATENT_GUIDE_FRAME0}.latent ← #{inplace_id} (was ← #{AUDIO_VIDEO_MASK})")

    # Bypass: InplaceKJ enforces frame-0 lock with noise_mask=0 (hard);
    # AddLatentGuide is a soft pull the sampler can denoise away. Redundant.
    frame0_guide["mode"] = 4
    print(f"  #{ADD_LATENT_GUIDE_FRAME0} AddLatentGuide_frame0: mode 0 -> 4 (redundant w/ InplaceKJ)")

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
        variant_label="inplace-kj-p1",
        next_steps=[
            "1. bash start_experiment.sh default",
            f"2. Reload {output_path.name} in ComfyUI",
            "3. Wire LoadImage slots (first/middle/last) + LoadAudio",
            "4. Queue prompt — diagnostic outcomes for loop body frames (~20-37s):",
            "     coherent + frame 0 matches init    → fix confirmed; promote to canonical",
            "     frame 0 locked but middle abstract → need more than frame-0 anchor",
            "                                          (try LatentContextExtract or single-pass full-res)",
            "     NaN black frames                   → InplaceKJ size/shape mismatch; fork our own",
        ],
    )


if __name__ == "__main__":
    main()
