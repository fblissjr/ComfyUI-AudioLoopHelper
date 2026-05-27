# Combined single-graph ADR (audio→video) workflow — build spec

Last updated: 2026-05-27

> **STATUS: design / build spec, NOT yet built.** The two passes work today as
> two separate workflows (`_av_inversion` → `_keyframe_autoextract`, with a manual
> audio hand-off). This doc specifies welding them into ONE ComfyUI graph so a
> single queue does both. It is a ~20-node two-sampler weld that needs render
> iteration to validate — written as the spec to build+test from, not a built
> artifact. Background on the working two-pass:
> [`av_dialogue_replacement_design.md`](av_dialogue_replacement_design.md).

## Goal

One graph, one queue: feed a source clip + a dialogue/scene prompt, get back a
video whose **audio is generated** (voice-cloned, new words / new sound) and
whose **video is regenerated to match that audio** (lip-sync / audio-reactive
motion). This is the in-context "audio guides video" inference path; the IC-LoRA
sharpens the same relationship in the model itself (see `internal/trainer_audio_iclora_plan.md` — private clone only).

## Why it must be two sampler passes (not one)

`noise_mask` on the two streams is **opposite** between the passes, and a single
sampler invocation can't hold both:

| pass | video stream | audio stream | sampler generates |
|---|---|---|---|
| 1 (audio) | frozen (mask 0) — source clip | seed kept + rest mask 1 | the **audio** |
| 2 (video) | keyframe init + mask 1 | frozen (mask 0) — pass-1 audio | the **video** |

So the graph has **two `SamplerCustomAdvanced` nodes**; pass 2 depends on pass
1's output, so ComfyUI runs them in order. Both passes are **single-window**
(loop bypassed) — a source clip is ~one window (~20 s / 497 frames @ 25 fps), so
neither pass needs the TensorLoop. (A full-length looped variant is a later
extension; start single-window.)

## The shared base

Both `_av_inversion` and `_keyframe_autoextract` are forks of the SAME canonical
graph and reuse the same node ids: `#161 SamplerCustomAdvanced`, `#350
LTXVConcatAVLatent`, `#245 LTXVSeparateAVLatent`, `#566`/`#2009
LTXVAudioVAEEncode`, `#531 LTXVImgToVideoInplaceKJ`, `#344 EmptyLTXVLatentVideo`.
The combined graph keeps ONE of these chains as pass 2 and adds a **parallel
copy** for pass 1.

## Graph (node-level)

### Pass 1 — generate the audio (clone from a seed, new words from the prompt)

Mirror `_av_inversion` exactly, as a self-contained sub-chain:

1. `VHS_LoadVideo` (source clip, `frame_load_cap=497`) → `VAEEncode` → source
   **video latent**, held frozen.
2. `LTXVAudioVAEEncode` (source clip audio) → `AudioTemporalMask`
   (`start_time=2.0, invert=False` → keep first 2 s as the voice seed, regen the
   rest) → **audio latent w/ noise_mask**.
3. `LTXVConcatAVLatent`(video-frozen, audio-masked) → **`SamplerCustomAdvanced`
   #A** → `LTXVSeparateAVLatent` → **`A_audio`** (the generated audio latent) +
   discard the video side.
4. Conditioning: the dialogue prompt via `TimestampPromptScheduleBatchEncode`
   (`0:00+:` entry) → the pass-1 sampler. Negative via `CLIPTextEncode`. NAG
   **bypassed** (video frozen in pass 1 — nothing to steer).

### The bridge — re-freeze the generated audio

`A_audio` is a clean denoised latent. To hand it to pass 2 as frozen context:

- `A_audio` → `AudioTemporalMask` (`start_time=0, end_time=0, invert=False` →
  empty regen window → **noise_mask=0 on every audio token** = fully frozen) →
  **`A_audio_frozen`**.

(`AudioTemporalMask` with an empty/zero-width window is a no-op-then-freeze: it
sets the whole track to "keep". This is the same node already used for the seed
in pass 1, with a different window.)

### Pass 2 — generate the video to match the frozen audio

Mirror `_keyframe_autoextract`, single-window:

1. Keyframes from the source clip: `VHS_LoadVideo` (source clip, all frames,
   `frame_load_cap=0`) → `EvenlySpacedKeyframes(count=3)` → 3×
   `GetImageRangeFromBatch(0/1/2)` → 3× resize → 3× `VAEEncode`. Single-window:
   feed **keyframe 1** as the init image into `LTXVImgToVideoInplaceKJ`
   (`['1', 1, 0]` = frozen frame-0 anchor); keyframes 2/3 are only meaningful in
   the looped variant, so omit `LTXIterKeyframeSchedule` here.
2. `EmptyLTXVLatentVideo` (clip-sized) + keyframe-1 init via the inplace node →
   **new video latent (mask 1)**.
3. `LTXVConcatAVLatent`(new-video mask-1, `A_audio_frozen` mask-0) →
   **`SamplerCustomAdvanced` #B** → `LTXVSeparateAVLatent` → **`B_video`**.
4. Conditioning: the **same** dialogue prompt → pass-2 sampler. NAG **active**
   (video is generated here); `nag_scale` 3–7 (distilled freeze-risk). Crank
   `audio_to_video_scale` (the `#1523`-class knob, ~2.5+) to tighten how hard the
   frozen audio drives the video — this is the lever the audio-reactive design
   doc calls out as the native audio→video coupling control.

### Output

- `B_video` → `LTXVTiledVAEDecode` → frames → `TrimImageBatchToAudio` →
  `VHS_VideoCombine.images`.
- `A_audio` (the **generated** audio, pre-freeze copy) → `LTXVAudioVAEDecode` →
  `VHS_VideoCombine.audio`. Mux the generated audio, not the source.

## Sigma / sampler discipline (both passes)

Both samplers use the canonical distilled chain: `ManualSigmas` 8-step + euler +
`CFGGuider cfg=1`, fps=25, no SD3 shift node. (Same as every shipped workflow —
see `docs/reference/sampler_reference.md`.) Two passes = ~2× the single-window
sampler cost; budget VRAM for both latent sets if not offloading between passes.

## Build plan (next session, with render iteration)

1. `scripts/apply_combined_adr_workflow.py` — fork `_keyframe_autoextract` as the
   pass-2 base; add the pass-1 sub-chain (clone `#161/#350/#245` + the
   video-freeze + audio-seed) via `WorkflowEditor`; route `A_audio_frozen` into
   pass 2's concat where `#2009`'s frozen audio currently enters; route `A_audio`
   to the audio decode/mux. Single-window (loop bypassed both passes).
2. Audit: `graph_acyclic` / `link_integrity` / `widget_shape` must pass; add a
   structural check that traces `Sampler#A → separate → AudioTemporalMask(freeze)
   → concat → Sampler#B` (the bridge) so a future edit can't sever it.
3. Render-gate: confirm pass-1 audio is voice-cloned + on-words, and pass-2 lips
   track it. Tune `audio_to_video_scale` and `first_frame_guide_strength`.

## Why not built here

A ~20-node two-sampler weld can't be validated without a render, and the
structural audit catches dangling links / cycles but not a semantically-wrong
latent route (e.g. feeding the un-frozen `A_audio` into pass 2). Shipping an
unrendered weld risks burning render time on a subtle mis-wire. The two existing
passes already produce the result manually; this spec turns the combine into a
fast, render-validated build rather than a blind guess.

## Cross-links

- Working two-pass (the validated manual pipeline): [`av_dialogue_replacement_design.md`](av_dialogue_replacement_design.md)
- Native audio→video coupling + the `audio_to_video_scale` lever + coupling ceiling: [`audio_reactive_loop_design.md`](audio_reactive_loop_design.md)
- Keyframe selector mechanics: [`keyframe_iter_anchor_design.md`](keyframe_iter_anchor_design.md)
- User-facing two-pass guide: [`../../docs/guides/dialogue_replacement_guide.md`](../../docs/guides/dialogue_replacement_guide.md)
- The IC-LoRA training side (sharpens the same audio→video relationship): `internal/trainer_audio_iclora_plan.md` (private clone)
