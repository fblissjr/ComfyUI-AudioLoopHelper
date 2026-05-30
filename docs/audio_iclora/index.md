Last updated: 2026-05-30

# Audio-only IC-LoRA ("helium")

The current effort: an audio-only IC-LoRA on LTX-2.3-22B (distilled) where an in-context reference
**tone** steers the pitch of the jointly-generated speech. No image or video reference. Experimental,
very much a proof of concept (whether it cleanly steers pitch at inference is still under evaluation).

## Start here

- **[`audio_only_ic_lora.md`](./audio_only_ic_lora.md)**: how the custom nodes work, how to run it in
  ComfyUI, how to evaluate it (the F0-tracking gate), and the automated trust gates that keep a broken
  setup from producing a meaningless result. Read this first.

## Where the pieces live

| Half | Where | What |
|---|---|---|
| Nodes + eval + ComfyUI | this repo | `nodes_audio_iclora.py` (the loader + reference-token nodes), the eval workflow builder, the F0 gate |
| Trainer + model card | the LTX-2 fork ([fblissjr/LTX-2 @ audio-guidance-iclora-vtv](https://github.com/fblissjr/LTX-2/tree/audio-guidance-iclora-vtv)) | the `audio_reference` training strategy, the data recipe, the published model card + checkpoints |

## Prior experiments

[`prior_experiments/`](./prior_experiments/) holds the earlier audio→**video** coupling direction
(beat→pulse, turn-left / turn-right). It is superseded by the audio→audio pitch work above, kept for the
process notes and the eval confounds documented along the way.

Distinct from the inference-only audio-reactive work (no training, works today):
[`../experimental/audio_reactive_workflows.md`](../experimental/audio_reactive_workflows.md).
