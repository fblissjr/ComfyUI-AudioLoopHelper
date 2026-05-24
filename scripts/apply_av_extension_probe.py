#!/usr/bin/env python
"""Stage an AV audio-generation / extension probe variant of the canonical latent workflow.

Forks ``example_workflows/audio-loop-music-video_latent.json`` and replaces the
all-frozen audio path (``SolidMask#571 -> SetLatentNoiseMask#570``, which holds the
ENTIRE audio latent fixed) with a single ``AudioTemporalMask`` node that can freeze
only a prefix and regenerate the tail.

This tests the open question "can LTX 2.3 generate audio at all?" — every shipped
workflow freezes audio (noise_mask=0), so the audio branch has never been asked to
generate. The new node's ``start_time`` widget selects the probe mode:

    start_time = 0.0  -> regenerate the ENTIRE audio latent (Probe 1: "does it
                         generate coherent audio from scratch?"). Decisive precursor.
    start_time = 2.0  -> keep the first 2 s of audio as context, regenerate the rest
                         (Probe 2: Defu's audio temporal-extension idea).

``audio_duration_seconds`` is WIRED from ``LTXFramePlanner#1634`` slot 3
(``actual_seconds``) — the same source that drives ``TrimAudioDuration#601.duration``
— so the seconds->audio-latent-frame mapping is correct-by-construction regardless of
the rendered window length.

Scope / caveats:
  * VIDEO stays i2v (init image -> generated video). This variant only un-freezes
    AUDIO. Adding a real video PREFIX (VHS_LoadVideo -> VAEEncode -> LatentTemporalMask)
    is the Phase-2 follow-up, worth wiring only if this probe shows the audio branch
    generates anything coherent.
  * Single-window by construction: the loop is bypassed (TensorLoopOpen count=0 via an
    INTConstant override), so only the initial render samples — one pass. The loop body
    re-freezes audio, so it must not run; this removes the need for a manual iterations=1.

Staged variant (outputs to ``example_workflows/experimental/``); skips the F-pair
audit-invariant requirement per scripts/CLAUDE.md.

Usage:
    uv run --group dev python scripts/apply_av_extension_probe.py            # apply
    uv run --group dev python scripts/apply_av_extension_probe.py --dry-run  # preview
    uv run --group dev python scripts/apply_av_extension_probe.py --revert   # delete staged file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from workflow_utils import WorkflowEditor  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "example_workflows" / "audio-loop-music-video_latent.json"
OUT = REPO / "example_workflows" / "experimental" / "audio-loop-music-video_latent_av_extension.json"

# Node IDs in the canonical latent workflow.
AUDIO_ENCODE = 566    # LTXVAudioVAEEncode  -> "Audio Latent" (LATENT) out slot 0
CONCAT = 350          # LTXVConcatAVLatent  -> audio_latent input slot 1
SET_MASK = 570        # SetLatentNoiseMask  (removed)
SOLID_MASK = 571      # SolidMask           (removed)
FRAME_PLANNER = 1634  # LTXFramePlanner     -> actual_seconds (FLOAT) out slot 3
CONCAT_AUDIO_SLOT = 1  # audio_latent input slot on LTXVConcatAVLatent
SEPARATE = 245        # LTXVSeparateAVLatent -> audio_latent output slot 1
AUDIO_VAE_GET = 254   # Get_audio_vae GetNode -> VAE output slot 0
VHS_COMBINE = 617     # VHS_VideoCombine     -> audio input slot 1
TENSOR_LOOP_OPEN = 1539  # TensorLoopOpen      -> iterations_in input slot 1

# Default probe widgets: start_time=2.0 — keep the first 2 s of real audio as
# context, regenerate the tail. This is the meaningful test: audio CONTINUATION
# from a partial prefix (audio inpainting), distinct from whole-clip text->AV
# generation, which LTX already does and needs no probe.
# end_time huge so it always clamps to the real duration (regen to end).
# audio_duration_seconds (10.0) is a display fallback; the wired link overrides it.
ATM_WIDGETS = [2.0, 10000.0, 10.0, 0.0]  # [start_time, end_time, audio_duration_seconds, edge_taper_seconds]


def apply(dry_run: bool = False) -> int:
    # From-scratch generator (forks the canonical -> staged variant). Re-running
    # regenerates the variant deterministically (fixed node/link IDs), so there is
    # no in-place idempotence guard — it always rebuilds from the current canonical.
    if not SRC.exists():
        print(f"ERROR: source workflow missing: {SRC}")
        return 1

    ed = WorkflowEditor(SRC)

    missing = [
        n for n in (AUDIO_ENCODE, CONCAT, SET_MASK, SOLID_MASK, FRAME_PLANNER,
                    SEPARATE, AUDIO_VAE_GET, VHS_COMBINE, TENSOR_LOOP_OPEN)
        if not ed.has_node(n)
    ]
    if missing:
        print(f"ERROR: expected nodes missing from canonical: {missing}. Canonical may have drifted.")
        return 1

    # Drop the all-frozen audio pair (detaches links 2323 / 1568 / 1569).
    ed.remove_node_and_links(SET_MASK)
    ed.remove_node_and_links(SOLID_MASK)

    # Add the partial-freeze node. inputs[0]=latent (socket), inputs[1]=audio_duration_seconds
    # (widget-backed, wired). add_link fills the link ids by slot index.
    inputs = [
        WorkflowEditor.io_in("latent", "LATENT"),
        WorkflowEditor.widget_in("audio_duration_seconds", "FLOAT"),
    ]
    outputs = [WorkflowEditor.out("LATENT", "LATENT")]
    atm_id = ed.add_top_level_node(
        "AudioTemporalMask",
        pos=[30, 3540],
        size=[300, 150],
        inputs=inputs,
        outputs=outputs,
        widgets_values=list(ATM_WIDGETS),
        properties={"Node name for S&R": "AudioTemporalMask", "aux_id": "fblissjr/ComfyUI-AudioLoopHelper"},
        title="AV Probe: audio noise_mask (start_time 0=gen all / 2.0=keep 2s prefix)",
    )

    ed.add_link(AUDIO_ENCODE, 0, atm_id, 0, "LATENT")          # #566 -> ATM.latent
    ed.add_link(FRAME_PLANNER, 3, atm_id, 1, "FLOAT")          # actual_seconds -> ATM.audio_duration_seconds
    ed.add_link(atm_id, 0, CONCAT, CONCAT_AUDIO_SLOT, "LATENT")  # ATM -> #350.audio_latent

    # Output: decode the GENERATED audio and mux THAT (not the input passthrough).
    # Without this the regenerated audio tail is unobservable — VHS would play the
    # original input track and the separated audio latent would have no consumer.
    dec_id = ed.add_top_level_node(
        "LTXVAudioVAEDecode",
        pos=[4530, 640],
        size=[260, 90],
        inputs=[
            WorkflowEditor.io_in("samples", "LATENT"),
            WorkflowEditor.io_in("audio_vae", "VAE"),
        ],
        outputs=[WorkflowEditor.out("Audio", "AUDIO")],
        widgets_values=[],
        properties={"Node name for S&R": "LTXVAudioVAEDecode"},
        title="Generated audio (decode sampled audio latent)",
    )
    ed.add_link(SEPARATE, 1, dec_id, 0, "LATENT")              # separated audio latent -> decode
    ed.add_link(AUDIO_VAE_GET, 0, dec_id, 1, "VAE")            # audio VAE -> decode
    ed.rewire_input(VHS_COMBINE, 1, dec_id, 0, "AUDIO")        # VHS audio <- GENERATED audio

    # Single-window probe: bypass the loop so ONLY the initial render samples. The loop
    # body re-freezes audio (so it isn't the probe), and a single window is all the test
    # needs. ComfyUI-NativeLooping bypasses (passes the initial render straight through,
    # one sampler pass) when count==0, which needs iterations_in<=0 AND the mode widget at
    # 0. Replace the AudioLoopPlanner -> iterations_in wire with an INTConstant=0: keeps
    # the F5 audit at WARN ("experiment-tier override") instead of ERR (unwired).
    iter_const = ed.add_top_level_node(
        "INTConstant",
        pos=[30, 2010],
        size=[210, 58],
        inputs=[],
        outputs=[WorkflowEditor.out("value", "INT")],
        widgets_values=[0],
        properties={"Node name for S&R": "INTConstant", "cnr_id": "comfyui-kjnodes"},
        title="probe: single window (loop bypassed, iterations=0)",
    )
    planner_link = ed.find_link_to_slot(TENSOR_LOOP_OPEN, 1)  # iterations_in
    if planner_link:
        ed.remove_link(planner_link[0])
    ed.add_link(iter_const, 0, TENSOR_LOOP_OPEN, 1, "INT")
    # mode widget -> count 0: ['iterations', 0, 0] (iterations=0, total_frames=0 -> bypass)
    ed.find_node(TENSOR_LOOP_OPEN)["widgets_values"] = ["iterations", 0, 0]

    if dry_run:
        print("--dry-run: would write", OUT.relative_to(REPO))
        print(f"  - remove SetLatentNoiseMask#{SET_MASK}, SolidMask#{SOLID_MASK}")
        print(f"  - add AudioTemporalMask#{atm_id} between #{AUDIO_ENCODE} and #{CONCAT}.audio_latent")
        print(f"  - wire audio_duration_seconds <- LTXFramePlanner#{FRAME_PLANNER}.actual_seconds")
        print(f"  - add LTXVAudioVAEDecode + mux GENERATED audio into VHS#{VHS_COMBINE}")
        print(f"  - bypass loop (INTConstant=0 -> TensorLoopOpen#{TENSOR_LOOP_OPEN}) so only the initial render samples")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ed.save(OUT)
    print(f"Wrote {OUT.relative_to(REPO)} (AudioTemporalMask#{atm_id}).")
    print("Single-window by default (loop bypassed) — one sampler pass, no manual iterations override.")
    print("Set start_time=0.0 for 'generate all audio' (Probe 1), 2.0 for 'keep 2s prefix' (extension).")
    return 0


def revert() -> int:
    if OUT.exists():
        OUT.unlink()
        print(f"Removed staged variant {OUT.relative_to(REPO)}.")
    else:
        print("Nothing to revert (staged variant not present).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Show changes without writing.")
    ap.add_argument("--revert", action="store_true", help="Delete the staged variant file.")
    args = ap.parse_args()
    if args.revert:
        return revert()
    return apply(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
