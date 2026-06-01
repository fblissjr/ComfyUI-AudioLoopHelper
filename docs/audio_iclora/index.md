Last updated: 2026-05-31

# Audio-only IC-LoRA

An audio-only IC-LoRA on LTX-2.3-22B (distilled): an in-context reference **audio** clip steers the
jointly-generated audio+video. No image or video reference. Two models so far, both proof-of-concept
(qualitatively steered, no published quantitative eval): a **pitch** probe ("helium") and the broader
**voice -> identity / mannerism** model (Audio-Only-Context).

## Start here

- **[`audio_only_context.md`](./audio_only_context.md)**: the current **Audio-Only-Context** model
  (voice -> identity / mannerism), the two checkpoints (audio-only vs cross-modal), the granular-strength
  nodes (per-stream loader + advanced guide), and the honest measurement note (why the reference-attribution
  gap reads ~0 for an identity task and why generation-from-noise is the real test). Released on HF.
- **[`audio_only_ic_lora.md`](./audio_only_ic_lora.md)**: the pitch ("helium") predecessor and the shared
  node mechanics — how the custom nodes work, the observed **inference behavior** (no reference-length clamp,
  the strength band, audio→identity coupling, why pure tones produce no speech), how to run it in ComfyUI,
  the F0-tracking gate, and the automated trust gates.
- **[`../guides/build_multimodal_dataset.md`](../guides/build_multimodal_dataset.md)**: turn a folder of
  sweep renders (prompt × reference × strength → output) into a schema'd JSONL dataset.

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
