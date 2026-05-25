#!/usr/bin/env python
"""Build the AV voice-reference variant — clone the voice via CONDITIONING, no kept words.

Forks the av_inversion workflow and switches the voice-cloning mechanism from the
AudioTemporalMask "kept seed" (which leaks the original words in the kept window) to
LTXVReferenceAudio, which encodes a short reference clip into the conditioning + runs an
identity-guidance pass. The voice is referenced, NOT output — so the audio fully
regenerates as new dialogue with ZERO original words.

The av_inversion base already ships LTXVReferenceAudio#1632 fully wired
(model + positive + negative -> CFGGuider, reference_audio <- TrimAudioDuration#1631,
audio_vae) but bypassed (ID-LoRA scaffolding default). This variant just enables it:

  1. un-bypass LTXVReferenceAudio#1632 (mode 0) — reference conditioning + identity guidance
     (identity_guidance_scale=3; dial to 1-2 if it over-cooks on the distilled model)
  2. un-bypass + fix the reference trim TrimAudioDuration#1631: its long-song default
     [start_index=30, duration=5] is past the end of a ~20s clip; reset to [0, 5] (first
     5s — the ~5s the node recommends). Tune start_index to the cleanest voice slice.
  3. AudioTemporalMask#2030 start_time -> 0.0: fully regenerate the audio (no kept seed),
     since the voice now comes from the reference, not a held window.

A/B vs the av_inversion mask-seed: this should produce voice WITHOUT any original words.
Zero-shot on base LTX 2.3 is untested — the node was built for the ID-LoRA flow, so if the
clone is weak this is where a trained ID-LoRA on the speaker would strengthen it.

Video stays the frozen real clip (same as av_inversion) so the A/B isolates the audio
mechanism. Lips won't match new words (dub effect) — regenerate video via the keyframe
variant for lip-sync, exactly as with av_inversion.

Usage:
    uv run --group dev python scripts/build_av_voiceref_workflow.py            # build
    uv run --group dev python scripts/build_av_voiceref_workflow.py --dry-run  # preview
    uv run --group dev python scripts/build_av_voiceref_workflow.py --revert   # delete output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from workflow_utils import WorkflowEditor  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "example_workflows" / "audio-loop-music-video_latent_av_inversion.json"
OUT = REPO / "example_workflows" / "experimental" / "audio-loop-music-video_latent_av_voiceref.json"

REF_AUDIO = 1632   # LTXVReferenceAudio (initial render) — un-bypass
REF_TRIM = 1631    # TrimAudioDuration feeding reference_audio — un-bypass + fix window
ATM = 2030         # AudioTemporalMask — start_time=0 => full audio regen (no kept seed)
REF_WINDOW = [0, 5]  # [start_index, duration] s — valid for short clips; tune to cleanest voice


def build(dry_run: bool = False) -> int:
    if not SRC.exists():
        print(f"ERROR: base workflow missing: {SRC.relative_to(REPO)}")
        return 1

    ed = WorkflowEditor(SRC)
    for nid in (REF_AUDIO, REF_TRIM, ATM):
        if not ed.has_node(nid):
            print(f"ERROR: expected node #{nid} missing — av_inversion base drifted.")
            return 1

    ref = ed.find_node(REF_AUDIO)
    if ref["type"] != "LTXVReferenceAudio":
        print(f"ERROR: #{REF_AUDIO} is {ref['type']}, expected LTXVReferenceAudio.")
        return 1

    ref["mode"] = 0                                  # un-bypass voice reference
    trim = ed.find_node(REF_TRIM)
    trim["mode"] = 0                                 # un-bypass reference trim
    trim["widgets_values"] = list(REF_WINDOW)        # valid short-clip reference window
    ed.find_node(ATM)["widgets_values"][0] = 0.0     # start_time=0 -> regenerate ALL audio

    if dry_run:
        print("--dry-run: would write", OUT.relative_to(REPO))
        print(f"  un-bypass LTXVReferenceAudio#{REF_AUDIO} (identity_guidance_scale={ref['widgets_values'][0]})")
        print(f"  un-bypass TrimAudioDuration#{REF_TRIM}, reference window -> {REF_WINDOW} (start_index, duration s)")
        print(f"  AudioTemporalMask#{ATM} start_time -> 0.0 (full audio regen, no kept seed)")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ed.save(OUT)
    print(f"Wrote {OUT.relative_to(REPO)}.")
    print("Voice via reference conditioning (no original words in output). Set the reference")
    print(f"window on TrimAudioDuration#{REF_TRIM} to the cleanest ~5s of voice. Render-gate pending.")
    return 0


def revert() -> int:
    if OUT.exists():
        OUT.unlink()
        print(f"Removed {OUT.relative_to(REPO)}.")
    else:
        print("Nothing to revert.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()
    return revert() if args.revert else build(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
