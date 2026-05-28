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
# Our trained LoRA is IC-LoRA shape (condition-mode v2v strategy, reference
# conditioning + audio context). It must load into the IC-LoRA loader, NOT
# the generic Style/ID LoRA slot. Loading into #2015 (regular LoRA) attaches
# the cross-modal-attention adapters without the IC-LoRA reference pathway
# they were trained to use → output is visibly bad (worse than baseline).
# Empirically confirmed 2026-05-28: first eval attempt with #2015 active
# produced clearly degraded output vs the no-LoRA baseline.
IC_LORA_LOADER_ID = 1635   # LTXICLoRALoaderModelOnly — the right slot
STYLE_LORA_ID = 2015       # generic LoraLoaderModelOnly — must stay bypassed
LOAD_AUDIO_ID = 565
RUN_ID_PREFIX_ID = 2026    # drives output path → output/<prefix>/<timestamp>/<file>.mp4


def render(template: dict, bpm: int, arm: str) -> dict:
    w = copy.deepcopy(template)
    for n in w["nodes"]:
        nid = n.get("id")
        if nid == LOAD_AUDIO_ID:
            # LoadAudio widget shape: [filename_in_input_dir, ...]. LoadAudio
            # doesn't surface subfolders in its dropdown, so the .wav files
            # live in the ROOT input dir (not a subdir).
            n["widgets_values"] = [f"{bpm}bpm.wav", None, None]
        elif nid == IC_LORA_LOADER_ID:
            # IC-LoRA loader: widgets = [lora_name, strength_model]. Point
            # at our trained LoRA + active for the lora arm, bypassed for baseline.
            n["widgets_values"] = ["audio_iclora/e1_run1_step300_rank16_broad.safetensors", 1.0]
            n["mode"] = 0 if arm == "lora" else 4
        elif nid == STYLE_LORA_ID:
            # Always bypassed — wrong slot for an IC-LoRA, attaching here
            # would silently degrade output (the cross-modal adapters wouldn't
            # get the reference-conditioning pathway they need).
            n["mode"] = 4
        elif nid == RUN_ID_PREFIX_ID:
            # RunIdPrefix #2026 is the ACTUAL driver of output paths — VHS's
            # filename_prefix input is wired to RunIdPrefix.video_prefix
            # (link 3201). The node produces output paths in the shape
            # `<output>/<prefix>/<timestamp>/<file>.mp4` (per its docstring).
            # Setting VHS's filename_prefix widget directly is a no-op
            # because the link overrides it.
            #
            # widget shape: [prefix_string, timestamp_format]. Per-workflow
            # prefix so each render lands in its own labelled folder tree:
            #   output/lora_eval/lora/110bpm/<timestamp>/<file>.mp4
            #   output/lora_eval/baseline/110bpm/<timestamp>/<file>.mp4
            n["widgets_values"][0] = f"lora_eval/{arm}/{bpm}bpm"
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
    print("  lora_eval/lora/<BPM>bpm_*.mp4")
    print("  lora_eval/baseline/<BPM>bpm_*.mp4")


if __name__ == "__main__":
    main()
