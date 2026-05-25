# Dialogue replacement guide — put new words in someone's mouth (voice-clone + lip-sync)

Last updated: 2026-05-25

> **EXPERIMENTAL — and the two passes are not equally proven.**
> **Pass 1 (generate the new audio)** is the part that's actually been run end to
> end: a real clip came back with the speaker's voice saying new lines, matched to
> the scene. **Pass 2 (regenerate the video for lip-sync)** is conceptually sound
> and wired up, but it's render-gate-pending — not yet validated on a real clip,
> so its parameters are best guesses. Treat every value here as a starting point,
> not a tuned optimum, and expect to re-roll seeds and nudge knobs per clip.

## What you're going to do

Take a short video of someone talking and replace what they say. The new audio
sounds like the same person (voice-cloned from a slice of the original), and —
if you want the mouth to match — you regenerate the video so the lips move to
the new words.

It's two passes:

1. **Pass 1 — make the new audio.** Freeze the original video, keep a couple
   seconds of the original voice as a clone seed, and prompt the new line. LTX
   generates new dialogue in that voice. The picture doesn't change yet, so the
   lips still match the *old* words (a dub).
2. **Pass 2 — make the lips match (optional).** Take the new audio from pass 1,
   freeze it, pull a few still frames from the original clip as look-anchors, and
   regenerate the video. Now the mouth moves to the new words.

If you only care about the audio (e.g. you're fine with a dub), pass 1 is the
whole job. Pass 2 is what turns a dub into a real lip-sync.

```
original clip ─┬─► [PASS 1: av_inversion] ─► new audio (cloned voice, new words)
               │            video frozen, audio seed kept, rest generated
               │
               └─► still frames ─┐
                                 ▼
        new audio (frozen) ─► [PASS 2: keyframe_autoextract] ─► lip-synced video
                                     video generated to match the frozen audio
```

## Why it works (the one-paragraph version)

LTX 2.3 is a **joint audio-video model** — it generates sound and picture
together through shared cross-attention, so the voice, the face, and the words
are all bound to each other during sampling. We exploit two levers:

- **`noise_mask`** on a latent decides what's frozen vs regenerated. `0` = keep
  this region exactly as-is (context the model reads but never changes). `1` =
  regenerate it. We set the mask on the **audio** latent in pass 1 (keep the
  voice-seed window, regenerate the rest) and the model keeps the audio frozen
  while regenerating **video** in pass 2.
- **The prompt carries the words.** LTX speaks the dialogue you put in quotes.
  The voice *identity* comes from the kept seed (real audio held as context); the
  *words* come from the prompt; the *scene* comes from the frozen video or the
  keyframes. The model fills in audio (or video) consistent with all three.

So: seed = whose voice, prompt = what they say, frozen modality = what stays put.

## Before you start

- A working ltx-2.3 distilled setup (this is the same model the default
  music-video workflow uses — see the repo [README](../../README.md) quick start
  for models + dependencies).
- `ffmpeg` on your PATH.
- A source clip. One clear speaker, visible mouth, clean-ish audio works best.

### Prep the clip

The workflow freezes a fixed-length window (~20 s at 25 fps = 497 frames in the
shipped config). Cut your clip to roughly that length so the **whole** video
freezes — a shorter clip leaves the tail regenerating and shifts the audio seed
boundary.

```bash
# cut ~20s starting at <start> from your source; placeholders — fill in your own
ffmpeg -ss <start> -i source.mp4 -t 20 -c copy your_clip.mp4
# drop your_clip.mp4 into ComfyUI's input dir
cp your_clip.mp4 <comfyui>/input/
```

## Pass 1 — generate the new audio

Open `example_workflows/audio-loop-music-video_latent_av_inversion.json`.

### What to set

| Node | What it is | Set it to |
|---|---|---|
| `#2033 VHS_LoadVideo` ("AV Inversion: full clip") | the source video, frozen | `video` = `your_clip.mp4`; leave `frame_load_cap=497`, `force_rate=25` |
| `#2030 AudioTemporalMask` | which audio is kept (voice seed) vs regenerated | start with `start_time=2.0`, `invert=False` (keep the first 2 s as the voice seed, regenerate the rest) |
| `#1615 TimestampPromptScheduleBatchEncode` | the new line | paste your dialogue prompt (see below) into the `0:00+:` entry |
| `#507 CLIPTextEncode` | negative prompt | leave the shipped default unless you see artifacts |
| `#1527 INTConstant` ("start_seed") | the roll | any int; change it to re-roll |
| `#508 LTX2_NAG` | guidance on the *video* | leave **bypassed** — the video is frozen here, NAG has nothing to steer |

Everything else is wired for you. Queue it. The output (`#617 VHS_VideoCombine`)
is the original footage with the **new** generated audio muxed over it.

### The dialogue prompt

Author it with the `/ltx-dialogue-prompt` skill, or follow the shape by hand.
The load-bearing rule from the official LTX 2.3 prompt guide: **break the speech
into short quoted segments with an acting direction between each** — one long
quoted block degrades badly. Lead with a shot + scene, end with accent + delivery
+ acoustic environment.

Example (paste into `#1615`'s `0:00+:` entry):

```
0:00+: In a medium shot at a dim restaurant table, an animated man leans in, grinning, and says "what's really fuckin' funny is — you don't even know who the fuck I am." He laughs, gesturing wide, "you think you can just steal from my fuckin' boss." He jabs a thumb at his own face, "you're lookin' right at the executioner." He shrugs, "and you're sitting there eating your burger, this motherfucker." His voice starts climbing, "so I ask you — do you know what the fuck the Bible says?" He cocks his head, "Ezekiel?" His voice cracks, incredulous, "EZEKIEL?! You know the fuckin' passage?!" He leans back, spreading his arms, "I will strike down upon thee with great fuckin' vengeance." He waves a hand, "now you're lookin' at me like I'm the asshole." His tone hardens, low, "fuckin' guy. eat your fuckin' burger." American accent, tense and animated delivery, close intimate restaurant ambience.
```

Another (same shape, different content):

```
0:00+: In a medium shot at a dim restaurant table, an animated man leans in, grinning, and says "what's really fuckin' funny is — I told Abe Lincoln not to go to that fuckin' theater." He laughs, gesturing wide, "there ain't fuckin' anything good playing anyway." He jabs a thumb at his own face, "I'm lookin' right at his tall fuckin' face, tellin' him to stay home." He shrugs, "and he goes to the fuckin' play anyway, this motherfucker." His voice starts climbing, "so I say it again — you're fuckin' going to the theater?" He cocks his head, "the fuckin' theater?" His voice cracks, incredulous, "THE THEATER?! To see a fuckin' PLAY?!" He leans back, spreading his arms, "ain't nobody even fuckin' acting in the show." He waves a hand, "now you're lookin' at me like I'm the fuckin' asshole." His tone hardens, low, "fuckin' guy. wouldn't fuckin' listen." American accent, tense and animated delivery, close intimate restaurant ambience.
```

Notes on the prompt:

- **Quoted words = what gets spoken.** For the eventual lip-sync (pass 2), the
  mouth syncs to these.
- **Profanity and exact long lines aren't guaranteed.** LTX isn't a TTS — it
  biases toward your words and nails short phrases, but may soften or drop
  explicit words and won't reproduce 15 s+ of scripted lines verbatim. Segment
  aggressively; that's why the examples are chopped into beats.
- **Describe the audio on purpose here.** (The research probe that this is built
  on deliberately *forbids* describing the audio — but that's a different goal.
  For dialogue replacement you *want* the prompt to carry the line.)

### `AudioTemporalMask` — the voice-seed control, in detail

This node sets the `noise_mask` on the audio latent: which seconds are kept
(the real voice seed) and which are regenerated to your prompt.

| Param | What it does | Start | Turn it up / down |
|---|---|---|---|
| `start_time` | with `invert=False`, the regenerate window is `[start_time, end]`; the **first `start_time` seconds are kept** as the voice seed | `2.0` | Longer seed (`3–4`) = stronger voice clone, but more of the **original words** survive and less room for the new line. Shorter (`1`) = more new dialogue, weaker timbre anchor. |
| `end_time` | end of the regenerate window (clamped to clip length) | leave high (`10000` = "to the end") | Lower it to keep a *tail* of original audio too. |
| `audio_duration_seconds` | the clip's real audio length; maps seconds → latent frames | **auto-wired** — leave it | — |
| `edge_taper_seconds` | cosine ramp at the seed↔generated boundary | `0.0` (hard cut) | Raise to `0.2–0.5` if you hear a click/seam where the seed hands off to the generated audio. |
| `invert` | flips which range is kept | `False` | `True` keeps `[start_time, end_time]` as the seed and regenerates everything else — use it to grab the **cleanest** 2 s of voice from the *middle* of the clip (a quiet single-speaker moment) instead of the opening. |

**The tradeoff to understand:** whatever window you keep as the seed plays back
the **original words** verbatim — it's real audio, not generated. So a 2 s prefix
seed buys you voice identity at the cost of 2 s of the old line at the start. If
the output must contain *none* of the original words, you can't keep any real
seed — use the voiceref variant instead (below).

### Other pass-1 knobs

| Node | Param | Start | Effect |
|---|---|---|---|
| `#1527 INTConstant` | `start_seed` | `42` | Audio identity varies seed-to-seed (genre/voice character can swing); if a roll sounds off, just change this and re-queue. Try 3–5 rolls before tuning anything else. |
| `#1269 FloatConstant` | `first_frame_guide_strength` | `0.7` | How hard the init frame is pinned. The video's frozen in pass 1 so this matters little; leave it. |
| `#508 LTX2_NAG` | (bypassed) | bypass | Only un-bypass when the video is actually being generated (pass 2). |

### What you get

An mp4 of the original footage with new spoken audio in the cloned voice. The
lips still match the *old* words — that's expected, the picture didn't change.
If a dub is all you need, you're done. For lip-sync, go to pass 2.

## Hand off the audio to pass 2

Extract the generated audio from pass 1's mp4 and stage it for pass 2:

```bash
# pull the audio track out of the pass-1 render
ffmpeg -i <comfyui>/output/<pass1_render>.mp4 -vn -acodec pcm_s16le new_dialogue.wav
cp new_dialogue.wav <comfyui>/input/
```

## Pass 2 — regenerate the video to lip-sync

> **Least-tested step.** Pass 2 hasn't been validated on a real clip yet. The
> wiring is sound and the mechanism is the same frozen-audio / generated-video
> path the music-video workflow uses every day, but the specific tension here —
> keyframes pinning the *original* face hard (`first_frame_guide_strength=1.0`)
> while the model tries to invent *new* mouth motion — is exactly where it might
> need tuning. If the lips stay locked to the old words, that's the first knob to
> back off. Report what you find.

Open `example_workflows/experimental/audio-loop-music-video_latent_keyframe_autoextract.json`.

Now the roles flip: the **audio is frozen** (your new dialogue) and the **video
is generated** to match it. Still frames from the original clip pin the look so
the person/scene stays recognizable; the model fills in mouth motion that tracks
the frozen audio.

### What to set

| Node | What it is | Set it to |
|---|---|---|
| `#565 LoadAudio` | the audio to lip-sync to | `new_dialogue.wav` (from pass 1) |
| `#2043 VHS_LoadVideo` ("Keyframe source clip") | clip the look-anchors come from | `your_clip.mp4`; leave `frame_load_cap=0` (load all frames so it can sample across the whole clip) |
| `#2044 EvenlySpacedKeyframes` | how many anchor frames to pull | `count=3` (first / middle / last) |
| `#2042 LTXIterKeyframeSchedule` | which iterations each anchor pins | `target_iters` ships pre-filled `1,2,3`; re-spread for longer renders (below) |
| `#1615 TimestampPromptScheduleBatchEncode` | the line again | paste the **same** dialogue prompt as pass 1 |
| `#508 LTX2_NAG` | guidance on the video | **active** here (video is generated) — `nag_scale=11` is the shipped default but **aggressive for distilled**; dial to `3–7` if the video freezes or stops moving |
| `#1269 FloatConstant` | `first_frame_guide_strength` | `1.0` = hard look-lock on the keyframes; lower toward `0.5` if the anchors fight the lip motion |

### `LTXIterKeyframeSchedule.target_iters`, in detail

The schedule picks which pre-extracted keyframe anchors the current loop
iteration. Each row's `target_iters` is a comma-separated, **1-based** list of
iterations (the loop emits 1, 2, 3, …; iteration 0 is the out-of-loop init
render, so `0` is dead).

- Ships pre-filled `1,2,3` so the three keyframes fire on the first three iters
  out of the box.
- For a longer render, spread them across your song's iteration count (check
  `AudioLoopPlanner.summary`) — e.g. `target_iters_1='1'`, `_2='5'`, `_3='10'`.
- **Clearing a row to empty makes that keyframe silently do nothing** — the
  iteration falls back to the init image and you'll see "only one image in use."
  That's the #1 gotcha; a firing default is shipped specifically to avoid it.

### What you get

A regenerated video where the mouth tracks the new dialogue, the look held to
the original clip via the keyframes, with the new audio muxed over it.

## Variant: clone the voice with NO original words (voiceref)

If keeping any of the original line is unacceptable, use
`example_workflows/experimental/audio-loop-music-video_latent_av_voiceref.json`
instead of the inversion workflow for pass 1. It takes voice identity from
`LTXVReferenceAudio` conditioning (a reference clip of the target voice) rather
than from a kept audio window, so `AudioTemporalMask` regenerates **all** the
audio — no original words survive. Tradeoff: the timbre anchor is a little looser
than a real kept seed. Full design notes:
[`../../example_workflows/working_docs/av_dialogue_replacement_design.md`](../../example_workflows/working_docs/av_dialogue_replacement_design.md).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| New audio is generic / ignores the seed voice | seed too short, or prompt over-describing audio character that overrides the seed | raise `AudioTemporalMask.start_time` to 3–4; trim the delivery description |
| Old words bleed into the start | that's the kept seed playing real audio | shorten the seed, or switch to the voiceref variant |
| Click/seam where seed meets generated audio | hard mask boundary | set `edge_taper_seconds` to 0.2–0.5 |
| Lines garbled / wrong words | one long quoted block, or too much scripted text | segment into shorter quoted beats with directions between |
| Pass-2 video freezes / stops moving | NAG too aggressive on distilled | drop `nag_scale` to 3–7 |
| Pass-2 "only one image in use" | a `target_iters` row was cleared to empty | re-fill it (1-based) |
| Lips don't match in pass 2 | keyframe look-lock fighting motion | lower `first_frame_guide_strength` toward 0.5 |
| Tail of the video isn't frozen in pass 1 | clip longer than the window | re-cut to ~20 s |

## How the whole thing fits together (under the hood)

- **One model, both modalities.** LTX 2.3's DiT attends across audio and video
  latents jointly. That's why a voice seed + a face + a written line cohere into
  one believable output instead of three disconnected streams — the cross-
  attention ties them during sampling.
- **`noise_mask` is the freeze switch.** `AudioTemporalMask` writes a per-frame
  mask onto the audio latent: `0` where you keep the seed, `1` where you
  regenerate. It maps your `start_time`/`end_time` seconds onto the audio
  latent's time axis using the clip's real duration (so the mapping is correct
  regardless of the audio VAE's frame rate). In pass 2 the audio latent is held
  frozen and the **video** latent is the one being regenerated.
- **The seed is real audio, not a synthesis prompt.** The kept window is the
  actual source waveform held as context. The model continues its timbre into
  the generated span — that's the "clone." It also means the seed seconds carry
  the original words verbatim, which is the central tradeoff.
- **Keyframes anchor identity without per-frame cost.** In pass 2 the look-frames
  are VAE-encoded once and *selected* per iteration by `LTXIterKeyframeSchedule`
  — they pin the person/scene so the regenerated video doesn't drift away from
  the original while it invents new mouth motion.
- **NAG only earns its keep when video is generated.** In pass 1 the video is
  frozen, so steering it away from a negative prompt is wasted work (and the
  distilled checkpoint is freeze-prone under aggressive NAG) — hence bypassed.
  In pass 2 it's back on.

## See also

- Design + the voice/video axes: [`../../example_workflows/working_docs/av_dialogue_replacement_design.md`](../../example_workflows/working_docs/av_dialogue_replacement_design.md)
- The research probe this is built on (can the model *infer* audio from video?): [`../../example_workflows/working_docs/av_inversion_test_examples.md`](../../example_workflows/working_docs/av_inversion_test_examples.md)
- Keyframe mechanics + the `target_iters` footgun: [`../../example_workflows/working_docs/keyframe_iter_anchor_design.md`](../../example_workflows/working_docs/keyframe_iter_anchor_design.md)
- Prompt-authoring rules + the verb/cross-attention behavior: [`prompt_creation_guide.md`](prompt_creation_guide.md)
- Dialogue prompt generator: the `/ltx-dialogue-prompt` skill
