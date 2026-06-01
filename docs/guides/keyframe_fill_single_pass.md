Last updated: 2026-06-01

# Single-pass keyframe fill ("sparse keyframes -> LTX fills the gaps")

Pin a handful of keyframe images at their real times and let LTX 2.3
interpolate the frames between them, in **one** generation — no loop, no
per-keyframe wiring, no frozen audio. The classic case: a video loaded at
1 fps becomes one keyframe per second, and the model fills 25 fps in
between.

This is **not** the audio-loop pipeline. There's no `AudioLoopController`,
no stride math, no `LTXVConcatAVLatent`. It's a plain single-stage distilled
render whose only twist is the guide source.

## When this works (and when you need the loop instead)

Single pass holds while the whole clip fits one generation:

- Total frames under the latent-volume ceiling (see
  [`reference/frame_planner_reference.md`](../reference/frame_planner_reference.md)).
  At ~960x544 / 25 fps that's roughly 10-15 s comfortably.
- The clip length is the only real limit on keyframe count here (each guide is
  one latent frame); past the single-pass frame budget you must chunk across a
  loop — that's what the rest of this repo is for.

Past either limit, move to the keyframe loop workflow
(`example_workflows/audio-loop-music-video_latent_keyframe.json`) and strip its
audio path.

## The node

`KeyframeGuidesFromBatch` (category `looping/keyframes`) collapses the entire
`Index -> Get Image from Batch -> Math Expression -> LTXVAddGuideMulti` fan-out
into one node. Feed it a **dense** keyframe batch (just the keyframes, no black
padding) plus the playback fps and the real-time gap between keyframes; it
resizes, encodes, and places each keyframe at its exact frame index.

Inputs: `vae, positive, negative, latent, images, output_fps,
seconds_per_keyframe, strength`. Outputs `(positive, negative, latent)` straight
into the sampler.

- `output_fps` — fps of the video you're generating (match
  `LTXVConditioning.frame_rate`; canonical **25**).
- `seconds_per_keyframe` — real-time gap between consecutive keyframes. `1.0`
  for a 1 fps source. Keyframe `i` lands at the exact pixel frame
  `round(i * seconds_per_keyframe * output_fps)` (1 s @ 25 fps -> frame 25).
- `strength` — `1.0` = hard anchor per keyframe; lower frees interpolation.

Size the empty latent for the clip duration; keyframes whose frame index falls
past the latent length are dropped (logged as `placed N/M keyframes`).

### vs KJNodes `LTXVAddGuidesFromBatch`

KJNodes ships a batch-guide node too, but it places keyframe `i` at frame `i`
(consecutive) — to space keyframes you must feed a **full-length** batch with
black frames in the gaps (it skips black images). Use it when your batch already
*is* the full-length sparse track. Use `KeyframeGuidesFromBatch` when you have a
dense batch and want it time-spaced without building (and VAE-iterating) a big
mostly-black tensor. Both reuse the same core `LTXVAddGuide` machinery.

## Minimal graph

```
VHS_LoadVideo (force_rate=1)──┐
                              │ images
   CheckpointLoader ── vae ───┼──────────────┐
                              │               │
   (Gemma) positive ─────────┐│               │
   (Gemma) negative ────────┐││               │
                            │││               ▼
   LTXFramePlanner ─ w/h/len ─→ EmptyLTXVLatentVideo ─ latent ─┐
                            │││                                 │
                            ▼▼▼                                 ▼
                       KeyframeGuidesFromBatch(output_fps=25, seconds_per_keyframe=1.0)
                            │ positive │ negative │ latent
                            ▼          ▼          ▼
                       CFGGuider(cfg=1) ── ManualSigmas(distilled 8-step) ── KSamplerSelect(euler)
                            │
                            ▼
                       SamplerCustomAdvanced ── LTXVCropGuides ── LTXVTiledVAEDecode ── VHS_VideoCombine
```

Sampler chain is the standard distilled 8-step path — same sigmas / `cfg=1` /
no SD3 shift node as every other workflow here (see
[`reference/sampler_reference.md`](../reference/sampler_reference.md)). The guide
node is the only non-standard piece.

## Notes

- **Video-only latent.** `latent` must be a plain `EmptyLTXVLatentVideo`, not a
  combined AV latent. Audio is left free for the model to generate; wiring a
  frozen-audio AV latent in raises `"Adding guide to a combined AV latent is not
  supported"` from core — an intentional guard, not a bug.
- **Resolution is handled.** Keyframes are resized to the generation resolution
  internally; they don't need pre-matching.
- **Prompt verb still matters.** Cross-attention binds the action to the verb;
  keep prompts concise and action-led (see
  [`guides/prompt_creation_guide.md`](prompt_creation_guide.md)).
