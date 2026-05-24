# AV inversion test — example scenarios (video → audio)

Last updated: 2026-05-24

> **STATUS:** Test-design doc for the audio-side inversion probe. The probe
> workflow itself (full video as clean context + 2 s audio seed → generate
> audio) is a stripped standalone V2A graph, staged under
> `example_workflows/experimental/`. The `AudioTemporalMask` node it relies on
> ships in `nodes.py` (tested). Render-gate pending.

## What this tests

Every shipped workflow here does **partial video + full audio**: a single init
frame seeds the video, the whole audio track is handed over frozen, and the
model fills in the video. This probe runs the **inversion**:

| | video | audio | model fills in |
|---|---|---|---|
| shipped pipeline | partial (1 init frame) | full (frozen) | the video |
| **inversion (this doc)** | **full (frozen context)** | **partial (2 s seed)** | **the audio** |

The open question is **context-driven audio inference**: given the complete
video and only 2 s of audio to anchor voice/timbre, does the model produce the
*expected* matching audio for the rest? This is distinct from prompted audio
generation, which LTX-2 already does natively (dialogue in quotes, music, SFX).

## The one control that makes it a real test: a NEUTRAL prompt

LTX-2 generates prompted audio by design. So the prompt must **not** describe
the audio — no quoted dialogue, no "upbeat music", no "angry tone". Describe
only the visible scene. If the prompt carries the audio, a good result proves
nothing (that's the documented capability). The continuation must come from the
**video + the 2 s seed**, or not at all. Same logic as decorrelating identity
from audio in the cross-attention probe.

Judge the **tail** (after the 2 s seed): does it continue the seed's
voice/timbre and match the on-screen action, with the prompt staying mum about
the audio?

## Example scenarios (use your own / public-domain footage — no specific titles)

Clip length must FILL the sampling window (planner-driven: ~20 s / 497 frames at
25 fps in the shipped config) so the whole video freezes — a shorter clip leaves
the video tail regenerated and shifts the audio seed boundary. ~20 s at the render
resolution is still well clear of the long-length video-encode OOM. For each: full
video frozen, first 2 s of its real audio as the seed, neutral scene-only prompt.

1. **Talking head** — one person speaking to camera, clear lip movement.
   - prompt: `a person seated indoors, speaking to the camera, soft lighting`
   - judge: does the generated tail keep the same voice and stay lip-synced to
     the visible mouth movements? (hardest sub-claim — voice identity over a
     generated span)

2. **Single instrument** — someone playing guitar / piano, hands visible.
   - prompt: `a musician playing an instrument, close shot of the hands`
   - judge: does the audio track the visible playing (note onsets at strums/key
     presses), continuing the seed's timbre?

3. **Footsteps / foley** — a person walking on a hard floor, feet in frame.
   - prompt: `a person walking through a room, wide shot`
   - judge: do footstep sounds land on the visible steps (timing), matching the
     surface implied by the seed?

4. **Ambient scene** — a busy street or cafe, no single sound source.
   - prompt: `a busy street corner during the day`
   - judge: does the generated ambience stay consistent with the seed's texture
     rather than drifting or going silent?

5. **Two-shot dialogue** — two people, alternating speakers.
   - prompt: `two people sitting across a table, indoor scene`
   - judge: does the audio attribute speech to whichever mouth is moving (the
     hard test of video→audio attention), holding voices from the seed?

## Failure modes to name (so a null result is interpretable)

- **Prompt leak**: the tail is generic ambience matching the neutral prompt, not
  a continuation of the seed. → not context inference; prompt-gen leaking.
- **Hard seam at 2 s**: clean prefix, then unrelated audio. → no continuation;
  the branch treats the seed as conditioning it ignores.
- **Voice drift**: right content, wrong/shifting voice. → partial — content
  inference works, identity persistence doesn't (the thing a LoRA would target).
- **Silence / noise in the tail**: video→audio inference is weak out of the box.

## Output handling — keep the video, mux the GENERATED audio

The output is the real (frozen) video with the model's generated audio over it —
you watch the footage and hear what the model inferred. Two requirements:

- **Keep the video.** The video latent is frozen context, so it decodes back to
  the input clip (minus minor VAE round-trip loss). Decode it and feed the frames
  to `VHS_VideoCombine.images`. Seeing the footage is the point.
- **Mux the GENERATED audio, not the input.** The shipped pipeline muxes
  `orig_audio` because its audio is frozen (generated == input). Here they differ
  — the 2 s seed is real, the tail is generated — so muxing the input would hide
  the result. Decode the **sampled** audio latent and mux that:
  `SamplerCustomAdvanced → LTXVSeparateAVLatent → audio latent → LTXVAudioVAEDecode
  → VHS_VideoCombine.audio`. You'll hear the seed flow into the generated tail.

## Per-scenario data prep (generic)

```bash
# clip from your own source — length >= the window (~20s); replace in/out + timestamps
ffmpeg -ss <start> -i <your_clip>.mp4 -t 20 -c copy clip.mp4
# the workflow loads clip.mp4 directly (video frames + its own audio, in sync)
```

Match the clip length to the planner window (~20 s) so the whole video freezes;
at the workflow's render resolution this stays clear of the long-length
video-encode OOM.
