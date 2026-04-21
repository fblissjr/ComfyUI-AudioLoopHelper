Last updated: 2026-04-20

# Music Video — "the vocal track" / Warrior + Kitten (v1)

First-pass prompt + schedule + widget settings for
`example_workflows/audio-loop-music-video_latent.json`, targeting the
`your-illustrated-init-image.png` init image (painterly fantasy illustration of a warrior
woman with a long silver-blonde braid and an orange tabby kitten in
aviator goggles on her shoulder) and the track
`your-vocal-track.mp3` (181s / 3:01, F Minor, 99.4 BPM, male vocals).

Built on the same lip-sync-safe bones validated in
`internal/prompt_comedy4.md` (v4 standup): byte-exact subject anchor,
`"Cut to ..."` hand-off language at iteration boundaries, no wide shots
or dolly-out until the outro, stride-aligned timestamps so every entry
runs on a single pure prompt.

## Why these choices

- **Init image is fantasy illustration**, not live-action. Per the
  LLM inference block in `_LLM_SYSTEM_PROMPT` (see
  `scripts/analyze_audio_features.py` 4/17 rework), the schedule
  should pull from the **animated / comic / graphic-novel** beat pool
  (impact frames, speed lines, supersaturation, silhouetted accents,
  motion blur lines) — NOT the live-action pool (lens flares, shallow
  DoF, practical lighting). Mixing pools reads as visual chaos.
- **User asked for cuts, not a static camera.** Matches tier `2a`
  (performance_dynamic, handheld / rock-video motion) — camera and
  body beats rotate every iteration. The `"Cut to ..."` lead-in on
  every entry after the first re-frames the iteration-boundary seam
  as an intentional edit (v3→v4 finding: iteration hand-offs are
  inherent visual discontinuities; naming them makes them feel
  deliberate).
- **No dolly-out, even on the outro.** CLAUDE.md's R7 permits dolly
  out on the final OUTRO entry, but in practice it shrinks the face
  over an 18s sampler pass and LTX's cross-attention loses the mouth
  signal → face morphs. Safer to hold on a close-up and let the audio
  fade do the outro work. This constraint is applied to every entry
  in this schedule (and in v2).
- **Chorus peaks are the close-ups.** Song analysis places CHORUS 1
  at 60-76s and CHORUS 2 at 152-164s. On the `stride=17.92s` grid
  those land inside iteration windows `0:53-1:11` and `2:23-2:41`
  respectively — those two iterations are the close-up / impact-frame
  payoffs. Quiet BRIDGE (0-20s) gets the wide opening; OUTRO (174-181)
  gets the only dolly-out.
- **Subject is two entities (woman + kitten).** The init image commits
  both. R3 says keep the subject description identical in every entry;
  the subject string below names both so the kitten doesn't morph or
  disappear mid-loop. Verb is `are singing together` (R1 multi-performer
  plural form) so LTX's audio-video joint cross-attention applies the
  singing mouth-animation signal to BOTH the woman and the kitten —
  the kitten actually tries to sing along rather than sitting passive.
- **Male vocals on a female visual subject is fine.** The audio track
  drives lip sync via cross-attention on mouth articulation, not
  timbre matching. The visual subject just needs to be singing on-beat;
  she will lip-sync whatever voice is in the track.

## Inputs

- **Audio**: `<your-vocal-track.mp3>` (181.03s,
  99.4 BPM, F Minor, male F0 median 77.4 Hz). Nine sections:
  BRIDGE 0-20 quiet, VERSE 20-60 medium, CHORUS 60-76 loud, VERSE 76-120
  medium, VERSE 120-152 medium, CHORUS 152-164 loud, BRIDGE 164-174
  quiet, OUTRO 174-181 quiet.
- **Image**: `<your-illustrated-init-image.png>` (painterly fantasy
  illustration, warrior woman + shoulder-kitten, stormy archway,
  lightning).
- **Subject string (byte-exact, identity-anchoring)**:

  `a warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder`

- **Grid**: `overlap_seconds=2.0`, `window_seconds=19.88`,
  `stride=17.92s`. Snap points:
  `0:00, 0:17, 0:35, 0:53, 1:11, 1:29, 1:47, 2:05, 2:23, 2:41`. With
  the 10s audio trim, the song is 171s in video-time (last iteration
  is `2:41+`, covering video 2:41-2:51).

## Schedule

Paste the `node_169_prompt` paragraph into Node 169 (`CLIPTextEncode`
immediately below `DualCLIPLoader`, the one currently holding
`"video of a woman passionately singing alone"`). Paste the `schedule`
block (everything after `schedule:` and before the closing fence) into
the `TimestampPromptSchedule` widget at Node 1558. First schedule line
is byte-exact to Node 169 — do not edit one without editing the other
(see `CLAUDE.md` "Node 169 prompt MUST match schedule's 0:00 entry").

```
node_169_prompt: Style: illustrated. In a medium shot, static camera, locked off shot, a warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together, voices already carrying the opening verse, mouths open and shaping the words, arms held high with quiet conviction, painterly rendering, illustrated film feel. Faint hum of distant thunder.

schedule:
0:00-0:17: Style: illustrated. In a medium shot, static camera, locked off shot, a warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together, voices already carrying the opening verse, mouths open and shaping the words, arms held high with quiet conviction, painterly rendering, illustrated film feel. Faint hum of distant thunder.
0:17-0:35: Cut to a medium close-up, static camera, locked off shot. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together with a steady voice, brow furrowing, jaw set, head turning a fraction toward the kitten, painterly rendering. Cool blue rim light beginning to gather. Low distant rumble.
0:35-0:53: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together with a low gravelly voice, eyes narrowing with conviction, mouth shaping each lyric, the kitten alert and twitching its ears, motion blur lines in the braid. Rim light sharpening. Faint thunder building.
0:53-1:11: Cut to a close-up, static camera, locked off shot. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together with full power, voice rising, eyes wide, mouth open, arms lifting higher, impact frame at the hit, supersaturation of color on the chorus, silhouetted accents at the edges of the frame. Bright, electric backlight. The kitten bracing wide-eyed.
1:11-1:29: Cut to a medium shot, jib up, camera rising up. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together with steady conviction, shoulders squaring, weight shifting forward, the kitten's goggles catching a stray highlight, motion blur lines trailing her braid. Warmer mid-tones settling in. Room tone of crackling air.
1:29-1:47: Cut to a medium close-up, focus shift, rack focus from the kitten to her face. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together with bright clear tone, chin lifting, eyes locked straight ahead, the kitten turning its head toward her cheek, painterly rendering. Steady rim light. Faint distant hum.
1:47-2:05: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together with rising intensity, brow tightening, lips pressing between lines, a single strand of hair lifting against the pull of the frame, speed lines faintly radiating at the edges. Highlights sharpening. Low thunder building.
2:05-2:23: Cut to a medium shot, dolly right, camera tracking right. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together with steady conviction, arms held firm, head rocking a fraction with the rhythm, the kitten's goggles reflecting a flash of light, painterly rendering. Punchier contrast. Thunder rolling closer.
2:23-2:41: Cut to a close-up, static camera, locked off shot. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together with full power, voice peaking, eyes wide, mouth fully open, arms at maximum extension, impact frame at the hit, supersaturation of color, silhouetted accents, motion blur lines tracing across the frame. Brightest backlight. The kitten wide-eyed and braced.
2:41+: Cut to a close-up, static camera, locked off shot. A warrior woman with a long silver-blonde braid and her small orange tabby kitten wearing tiny aviator goggles perched on her shoulder are singing together the final notes, voices softening and trailing off, eyes lowering, arms lowering a fraction, shoulders easing, the kitten relaxing and settling against her neck, painterly stillness, illustrated film feel. Fading, gentle light. The sound fades quietly. Room tone settles.
```

## Workflow widget values

Apply these to `example_workflows/audio-loop-music-video_latent.json`.
Only two values need to change from the committed defaults
(`AudioLoopController.overlap_seconds` and `LoadAudio.filename`).
Everything else listed below is the currently-shipped distilled
8-step configuration, documented here so you can verify nothing
drifted.

| Widget | Node | Value | Reason |
|---|---|---|---|
| `image` | 444 `LoadImage` | `your-illustrated-init-image.png` (copy to ComfyUI `input/`) | init image |
| `audio` (filename) | 565 `LoadAudio` | `your-vocal-track.mp3` (copy to ComfyUI `input/`) | song track |
| `overlap_seconds` | 1582 `AudioLoopController` | `2.0` (committed default post-2026-04-20) | target value; node internally quantizes to effective 1.96s / stride 17.92s to guarantee integer-latent lip-sync alignment |
| `window_seconds` | 688 `FloatConstant` (drives both `AudioLoopController` and `EmptyLTXVLatentVideo.length` via `TrimAudioDuration`) | `19.88` (unchanged) | distilled 8-step window |
| `length` | 526 `PrimitiveNode` → `EmptyLTXVLatentVideo` | `497` (unchanged; = 19.88 * 25) | iteration length in pixel frames |
| resolution | 445 `ImageResizeKJv2` | `832 x 480` (unchanged) | LTX 2.3 target |
| schedule text | 1558 `TimestampPromptSchedule` | paste `schedule:` block above | per-iteration prompt |
| `blend_seconds` | 1558 `TimestampPromptSchedule` | `0.0` (unchanged) | sub-stride blend is auto-clamped; hard cuts land cleanly on iteration boundaries |
| `snap_boundaries` | 1558 `TimestampPromptSchedule` | `true` (unchanged) | safety net if you tweak `overlap_seconds` later |
| Node 169 text | 169 `CLIPTextEncode` | paste `node_169_prompt` paragraph above | initial ~20s render; MUST be byte-exact to schedule's 0:00 entry |
| `sampler_name` | 154 `KSamplerSelect` | `euler` (unchanged) | ancestral re-noise causes subject drift on the plateaued schedule (see CLAUDE.md) |
| `steps / scheduler` | 1421 `BasicScheduler` | `linear_quadratic, 8, 1` (unchanged) | distilled 8-step schedule |
| `shift` | 1513 `ModelSamplingSD3` | `13` (unchanged) | distilled package default |
| `cfg` | 153 `CFGGuider` | `1.0` (unchanged) | distilled model; higher CFG over-denoises |
| `decoder type` | 1604, 1597 `LTXVTiledVAEDecode` | unchanged | post-DR1 decoder (no temporal-tile seams) |
| `start` (outer trim) | 567 `TrimAudioDuration` | **`10`** | skip the ~10s quiet intro before lyrics kick in. Shifts the song into video-time such that video-t=0 maps to song-t=10 |
| `start_seed` | 1527 `INTConstant` | `42` or whatever you prefer | deterministic; `AudioLoopController` derives per-iteration seeds from this |

After editing the widgets, run
`python3 -c "import json; json.load(open('example_workflows/audio-loop-music-video_latent.json'))"`
to confirm the JSON still parses.

## Negative prompt (unchanged from current)

Keep the shipped negative prompt on Node 507 `CLIPTextEncode`
("Set_audio_vae" color group):

```
still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone
```

No reason to add "scene cut, jump cut" here — the iteration hand-off
is already the cut, and we're explicitly leaning into it with
`"Cut to ..."`.

## Observations

- **Why the animated beat pool here, not cinematic**: the init is a
  digital painting. If I wrote "rack focus" or "anamorphic lens flare"
  into the schedule, LTX would try to reconcile film-lens texture with
  a painterly source, and the result is usually a muddy compromise.
  "Impact frame", "speed lines", "motion blur lines", "supersaturation",
  "silhouetted accents" all belong to the illustrated-film vocabulary
  the image is already committed to, so the text conditioning pulls
  the same direction as the image conditioning.
- **The kitten is a stress test.** It's small, peripheral, and easy
  for LTX to morph away over 10 iterations. Naming it and giving it
  small beats in every entry ("the kitten alert and twitching its
  ears", "goggles catching a stray highlight", "turning its head
  toward her cheek", "wide-eyed and braced", "relaxing its posture",
  "settling against her neck") gives LTX a concrete animation target
  per window. Same technique as the crowd-member beats in v4 standup.
- **Chorus alignment is deliberate.** CHORUS 1 at 60-76s falls 88%
  inside iteration `0:53-1:11`; CHORUS 2 at 152-164s falls 75% inside
  iteration `2:23-2:41`. Those two entries get identical
  "full power, eyes wide, impact frame, supersaturation, silhouetted
  accents" language so the visual payoff syncs with the audio payoff.
  Mid-verse iterations rotate through softer beats so the chorus hits
  read as peaks, not as "another frame in the sequence".
- **Lip-sync drift at higher overlap values is now a structural
  non-issue.** Post-2026-04-20, `AudioLoopController` derives stride
  from integer-latent counts rather than `window - overlap` seconds,
  so audio advance per iteration exactly matches video pixel advance
  regardless of overlap. Your widget `overlap_seconds=2` becomes
  effective 1.96s and stride 17.92s; `overlap=4` becomes effective
  3.88s / stride 16.0s. Both are drift-free. Change overlap freely
  for subject-continuity tuning without worrying about lip-sync.
- **Outro is a held close-up, not a pull-out.** Final iteration
  (`2:41+`) stays locked on the face. The audio fade and the kitten
  settling carry the close — LTX can animate both without the camera
  needing to move.
- **`Style: illustrated.` not `Style: cinematic.`** `Style: cinematic`
  is one of the strongest photoreal anchors in Gemma 3's embedding
  space (film-look, skin texture, practical lighting). On a painterly
  fantasy init image that's a direct tug-of-war between text and
  image, and over 10 iterations the cinematic pull compounds — the
  subject drifts toward live-action, which reads as "aging". The
  prompt guide explicitly says to omit `Style:` when the init
  establishes style strongly (which this init does). Switching to
  `Style: illustrated.` keeps the structural starts-with-Style prefix
  but actively counter-pulls toward the image's native vocabulary.
  Alternatives if this still drifts: `Style: painterly illustration.`
  or `Style: digital painting.` — more specific, slightly more tokens.
- **Why `overlap_seconds=2` was the right committed default.** At
  `overlap=1` the effective stride is 18.88s (grid points `0:00, 0:18,
  0:37, 0:56, 1:15 ...`). That grid puts the CHORUS 1 peak at 0:56 —
  almost exactly where the chorus starts (0:60) — but the window ends
  at 1:15, cutting the chorus off mid-peak. At `overlap=2` the
  effective stride is 17.92s (grid `0:00, 0:17, 0:35, 0:53, 1:11 ...`),
  so the chorus iteration is `0:53-1:11` — contains more of the chorus
  and lands the close-up framing on the sustained payoff rather than
  just the entry. Same argument for CHORUS 2.
- **This is an identity-safety schedule, not a storytelling schedule.**
  It's tier 2a (performance dynamic), not tier 4a (narrative). She
  stays in the archway with the kitten in every frame; camera and
  performance beats rotate, environment doesn't. If you want a
  narrative (e.g. she lowers her arms, the lightning fades, she walks
  into the archway), that's a different tier and probably wants the
  `_keyframe` workflow variant with multiple init images.
- **What I'd revisit if I had another pass**: (1) beat-snapping the
  grid to BPM-aligned phrase boundaries (the chorus starts at 0:60
  but the iteration window starts at 0:53; seven seconds of verse
  still has chorus framing applied). Per the 4/17 session log this
  is a ~30 LOC change with high perceived-quality return and is the
  single best improvement I'd prioritize next. (2) Whether the kitten
  should get a dedicated close-up entry as a b-roll break — can't do
  it safely in i2v single-image mode, but trivial in the `_keyframe`
  variant. (3) The male-vocal / female-visual combination may feel
  odd even with perfect lip sync; if it does, consider generating a
  different track or swapping the init image to a male warrior with
  the kitten.
