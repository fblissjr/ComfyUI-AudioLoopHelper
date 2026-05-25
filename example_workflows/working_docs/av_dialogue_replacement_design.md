# AV dialogue replacement — design

Last updated: 2026-05-25

> **STATUS:** Shipped variant + experimental variants. The
> dialogue-replacement workflow (`audio-loop-music-video_latent_av_inversion.json`)
> ships at the top level; the `_av_voiceref` / `_av_extension` /
> `_keyframe_autoextract` variants are staged under
> `example_workflows/experimental/`. The new nodes they rely on
> (`AudioTemporalMask`, `EvenlySpacedKeyframes`) ship in `nodes.py` (tested).
> Render-gate pending on the dialogue-replacement pipeline end to end.

## What this is

Take a source video clip and **replace its spoken dialogue** with new words —
voice-cloned (so the new audio sounds like the original speaker / scene) and,
optionally, lip-synced to newly generated video. This is the production
application of the video→audio inversion probe: that probe asks *"can the
model infer audio from video + a seed?"* (the research question — see
[`av_inversion_test_examples.md`](av_inversion_test_examples.md)); this doc is
the *"now put specific words in their mouth"* product built on the same spine.

Two orthogonal choices, mix and match:

| axis | options |
|---|---|
| **voice** (how the new audio gets the speaker's identity) | inversion (mask-seed) **or** voiceref (`LTXVReferenceAudio` conditioning) |
| **video** (does the picture change for the new dialogue) | frozen source video **or** generated/keyframed video for lip-sync |

The **prompt** carries the new words (via the `/ltx-dialogue-prompt` skill) in
every variant — that's what makes the model speak the chosen line.

## Voice axis

### Inversion (mask-seed) — `audio-loop-music-video_latent_av_inversion.json` (shipped)

The source video is held as frozen context; a window of the source audio is
kept as the voice-clone seed; the rest of the audio is regenerated to the new
dialogue prompt. The seed window is set on `AudioTemporalMask`:

- **Prefix seed (shipped default).** `start_time=N, end_time=audio_duration,
  invert=False` → `[N, end]` REGENERATES, the first `N` seconds are kept as the
  seed (the workflow ships `start_time=2.0` = keep 2 s prefix, generate the
  tail). The clean prefix carries the speaker's timbre into the generated tail.
- **Arbitrary window seed.** `invert=True` flips it: `[start_time, end_time]`
  becomes the KEPT seed and everything else regenerates. Use this to pick the
  *cleanest* 2 s of voice anywhere in the clip (not just the head) as the clone
  seed — e.g. a quiet, single-speaker moment mid-clip.

**Tradeoff — the seed window keeps the ORIGINAL words.** Whatever audio span is
kept as the seed plays back the source dialogue verbatim (it's real audio held
as context, not generated). So a prefix/window seed buys voice identity at the
cost of those seconds carrying the original line. If the output must contain
*none* of the original words, you can't keep any real-audio seed — use voiceref
instead.

### Voiceref (`LTXVReferenceAudio` conditioning) — `_av_voiceref` (experimental)

Staged under `example_workflows/experimental/audio-loop-music-video_latent_av_voiceref.json`.
Voice identity comes from `LTXVReferenceAudio` conditioning (the initial-render
`LTXVReferenceAudio #1632` is un-bypassed / active) instead of a kept audio
window. `AudioTemporalMask` is set to `start_time=0.0` → **all** audio
regenerates, no real-audio prefix kept. Result: the speaker's voice is cloned
from the reference, but **no original words survive** in the output — the entire
audio track is generated to the new dialogue prompt.

Choose voiceref when the output must not contain any of the source dialogue;
choose inversion (mask-seed) when a few seconds of original audio are acceptable
and you want the strongest timbre anchor.

## Video axis

- **Frozen source video (default for dialogue replacement).** The source video
  latent is held as frozen context and decodes back to the input clip (minus VAE
  round-trip loss). The picture doesn't change; only the audio is new. Mux the
  **generated** audio over the frozen video (see the output-handling section in
  [`av_inversion_test_examples.md`](av_inversion_test_examples.md)).
- **Generated / keyframed video for lip-sync.** When the new dialogue needs the
  mouth to match (true dubbing), let the video regenerate and pin the look with
  the keyframe schedule — see
  [`keyframe_iter_anchor_design.md`](keyframe_iter_anchor_design.md). The
  `_keyframe_autoextract` variant auto-samples the keyframes straight from the
  source clip (`EvenlySpacedKeyframes` + `GetImageRangeFromBatch`), so you don't
  hand-load reference images.

## The dialogue prompt — `/ltx-dialogue-prompt`

The new line is delivered through the prompt, formatted per the official LTX 2.3
prompt guide. The `/ltx-dialogue-prompt` skill turns *"make them say X, in this
scene"* into a paste-ready `0:00+:` schedule entry for
`TimestampPromptScheduleBatchEncode`: dialogue in double quotes, broken into
short segments with acting directions between each (the load-bearing rule — one
long quoted block degrades), plus accent and acoustic-environment description.

Note this is the deliberate inverse of the inversion *research probe's* neutral
prompt: the probe forbids describing the audio so a good result proves *context*
inference; dialogue replacement *wants* the prompt to carry the words, because
the goal is to control them.

## Why NAG is bypassed when the video is frozen

In the frozen-video variants (`_av_inversion`, `_av_voiceref`), `LTX2_NAG #508`
is set to bypass (`mode=4`). NAG steers the *video* generation away from a
negative prompt; when the video latent is frozen context (not being generated),
there's nothing for NAG to steer, so it's inert work at best. Bypassing keeps the
graph honest about what's actually being generated (the audio) and avoids the
distilled freeze-risk that aggressive NAG scale carries. Un-bypass NAG only in
variants where the video is actually regenerated (e.g. the keyframe-lip-sync
path).

## Variant: `_av_extension` (audio continuation probe)

Staged under `example_workflows/experimental/audio-loop-music-video_latent_av_extension.json`.
A related probe: rather than replacing dialogue, it tests **audio continuation**
— freeze the first N seconds of audio as context and regenerate the tail,
pairing the video stream's `LatentTemporalMask` with `AudioTemporalMask` on the
SAME seconds so both clean prefixes align in time. Distinct source layout
(separate `LoadAudio`); same `AudioTemporalMask` mechanism.

## Cross-links

- Research probe + output-handling (keep video, mux generated audio) +
  failure-mode taxonomy: [`av_inversion_test_examples.md`](av_inversion_test_examples.md).
- Keyframe lip-sync video + the empty-`target_iters` footgun:
  [`keyframe_iter_anchor_design.md`](keyframe_iter_anchor_design.md).
- Dialogue prompt format: `/ltx-dialogue-prompt` skill.
