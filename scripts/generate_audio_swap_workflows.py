#!/usr/bin/env python3
"""Generate the A/B audio-swap eval workflow variants for E1.1 from the
single-shot eval template.

For each (BPM x arm) combo, write a pre-wired workflow with:
- #565 LoadAudio:                  audio_iclora_eval/<BPM>bpm.wav   (path in ComfyUI input/)
- #2015 LoraLoaderModelOnly mode:  0 (active) for lora arm, 4 (bypassed) for baseline
- #617 VHS_VideoCombine prefix:    audio_iclora_eval/<arm>/<BPM>bpm

ComfyUI auto-saves to output/<filename_prefix>_<NNNNN>_.mp4, so the renders
land at predictable paths the manifest-builder can find later.

Output dir: internal/workflows/audio_swap_eval/<arm>/<BPM>bpm.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


BPMS = [50, 70, 90, 110, 130, 150, 170]
ARMS = ("lora", "baseline")
LORA_LOADER_ID = 2015      # the empty Style-LoRA slot we point at our trained LoRA
LOAD_AUDIO_ID = 565
VHS_COMBINE_ID = 617


def render(template: dict, bpm: int, arm: str) -> dict:
    w = copy.deepcopy(template)
    for n in w["nodes"]:
        nid = n.get("id")
        if nid == LOAD_AUDIO_ID:
            # LoadAudio widget shape: [filename_in_input_dir, ...]
            n["widgets_values"] = [f"audio_iclora_eval/{bpm}bpm.wav", None, None]
        elif nid == LORA_LOADER_ID:
            # mode 0 = active, mode 4 = bypassed
            n["mode"] = 0 if arm == "lora" else 4
        elif nid == VHS_COMBINE_ID:
            # filename_prefix is the routing knob for where outputs land in ComfyUI/output/
            wv = n["widgets_values"]
            wv["filename_prefix"] = f"audio_iclora_eval/{arm}/{bpm}bpm"
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path,
                    default=Path("internal/workflows/audio_swap_eval/_template_e1_run1.json"),
                    help="base eval workflow (with our LoRA already wired into #2015) — "
                         "gitignored since it's an internal artifact, not a public example")
    ap.add_argument("--out-dir", type=Path, default=Path("internal/workflows/audio_swap_eval"))
    args = ap.parse_args()

    template = json.loads(args.template.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for arm in ARMS:
        arm_dir = args.out_dir / arm
        arm_dir.mkdir(exist_ok=True)
        for bpm in BPMS:
            w = render(template, bpm, arm)
            out = arm_dir / f"{bpm}bpm.json"
            out.write_text(json.dumps(w, indent=2))
            n_written += 1

    print(f"wrote {n_written} workflows to {args.out_dir}/")
    print()
    print("Drag each into ComfyUI + Queue Prompt. Output paths (relative to ComfyUI/output/):")
    print("  audio_iclora_eval/lora/<BPM>bpm_*.mp4")
    print("  audio_iclora_eval/baseline/<BPM>bpm_*.mp4")


if __name__ == "__main__":
    main()
