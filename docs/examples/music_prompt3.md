Last updated: 2026-04-25 (audio descriptors and init-committed redundancy stripped per `docs/guides/prompt_creation_guide.md`)

# Music Video — "the vocal track" / Warrior + Cat (v3, cinematic)

Cinematic-init variant of `music_prompt2.md`. Same structure
(simpler one-sentence-per-entry form, canonical LTX 2.3 camera
phrases, plural `are singing together`, no dolly-out, 10s audio
trim, stride-17.92 grid), but tuned for the cinematic-realism init
image (`your-cinematic-init-image.png`) instead of the painterly illustration
(`your-illustrated-init-image.png`).

## What changed from v2

1. **`Style: cinematic.` instead of `Style: illustrated.`** The init
   commits cinematic realism (realistic skin texture, photoreal
   lighting, rendered hair strands); `illustrated` would fight it
   the same way `cinematic` was fighting `your-illustrated-init-image.png`. Match the
   style prefix to the init's style family.
2. **Subject string: "orange tabby cat" instead of "kitten"**. The
   cinematic render shows a full-size cat on her shoulder, not a
   small kitten. Match the visible subject.
3. **Every other clause is identical to v2** — subject anchor,
   verb form, camera sequence, shot-size progression, chorus vs
   verse framing.

## Why this should work better for the cinematic init

LTX 2.3's audio-video cross-attention pathway was trained
predominantly on photoreal footage — the singing mouth prior is
photoreal. With an illustrated init, the prior and the image
disagree and the model drifts toward photoreal over iterations
(the "broadway musical" failure mode observed in
`your-illustrated-init-image.png` runs). With a cinematic-realism init, the prior and
the image AGREE. Expected consequences:

- **Less cross-iteration style drift.** Starting close to where
  the audio-video pathway wants to be, there's no gradient to run
  down.
- **Tighter lip-sync.** Mouth region is already photoreal-compatible;
  the model doesn't have to hallucinate a photoreal mouth onto an
  illustrated face.
- **Better temporal stability.** Photoreal-to-photoreal hand-off
  across iteration boundaries is what LTX was trained on. No
  modality mismatch during latent concat.

## Inputs

- **Audio**: `<your-vocal-track.mp3>`
- **Image**: `<your-cinematic-init-image.png>` (or whatever
  cinematic render you're feeding; swap filename in LoadImage node 444).
- **Subject string (byte-exact in every entry)**:

  `a warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder`

- **Grid**: `overlap_seconds=2.0`, `window_seconds=19.88`,
  `stride=17.92s`. Snap points:
  `0:00, 0:17, 0:35, 0:53, 1:11, 1:29, 1:47, 2:05, 2:23, 2:41`.
  With the 10s audio trim, video is 171s; last iteration is
  `2:41+` covering video 2:41-2:51.

## Schedule

Paste `node_169_prompt` into Node 169. Paste the `schedule:` block
into `TimestampPromptSchedule` (Node 1558). First schedule line is
byte-exact to Node 169.

```
node_169_prompt: Style: cinematic. In a medium shot, static camera, locked off shot, a warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder are singing together.

schedule:
0:00-0:17: Style: cinematic. In a medium shot, static camera, locked off shot, a warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder are singing together.
0:17-0:35: In a medium close-up, static camera, locked off shot. A warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder are singing together, brow furrowing.
0:35-0:53: In a close-up, dolly in, camera pushing forward. A warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder are singing together, eyes narrowing.
0:53-1:11: In a close-up, static camera, locked off shot. A warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder are singing together, eyes wide, mouth open.
1:11-1:29: In a medium shot, jib up, camera rising up. A warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder are singing together, shoulders squaring.
1:29-1:47: In a medium close-up, focus shift, rack focus. A warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder are singing together, chin lifting.
1:47-2:05: In a close-up, dolly in, camera pushing forward. A warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder are singing together, brow tightening.
2:05-2:23: In a medium shot, dolly left, camera tracking left. A warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder are singing together, head rocking a fraction.
2:23-2:41: In a close-up, static camera, locked off shot. A warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder are singing together, eyes wide, mouth fully open.
2:41+: In a medium close-up, jib down, camera lowering down. A warrior woman with a long silver-blonde braid and her orange tabby cat perched on her shoulder are singing together, eyes lowering.
```

## Workflow widget values (identical to v2)

Only Node 169 text + Node 1558 schedule text change between
illustrated (v2) and cinematic (v3) runs. All other widgets stay.

| Widget | Node | Value |
|---|---|---|
| `image` | 444 `LoadImage` | `your-cinematic-init-image.png` |
| `audio` filename | 565 `LoadAudio` | `your-vocal-track.mp3` |
| `start` (outer trim) | 567 `TrimAudioDuration` | `10` |
| `overlap_seconds` | 1582 `AudioLoopController` | `2.0` |
| `window_seconds` | 688 `FloatConstant` | `19.88` |
| `length` | 526 `PrimitiveNode` | `497` |
| resolution | 445 `ImageResizeKJv2` | `832 x 448` |
| Node 169 text | 169 `CLIPTextEncode` | paste `node_169_prompt` above |
| schedule text | 1558 `TimestampPromptSchedule` | paste `schedule:` block above |
| `blend_seconds` / `snap_boundaries` | 1558 | `0.0` / `true` |
| `sampler_name` | 154 `KSamplerSelect` | `euler` |
| scheduler | 1421 `BasicScheduler` | `linear_quadratic, 8, 1` |
| `shift` | 1513 `ModelSamplingSD3` | `13` |
| `cfg` | 153 `CFGGuider` | `1.0` |
| decoder | 1604, 1597 `LTXVTiledVAEDecode` | unchanged |

## Negative prompt (unchanged)

```
still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone
```

## Observations

- **A/B this against v2 at the same seed.** Identical schedule
  structure, only the style prefix + `kitten`→`cat` differ. Any
  quality delta is attributable to the init image + style match,
  not to schedule choices.
- **If this run looks materially better than illustrated runs**,
  the session's "photoreal drift" failure mode was entirely about
  init-style mismatch with LTX's training distribution. No
  workflow knob will fix illustrated→photoreal drift short of
  multi-image-guide per iteration (`LTXVAddGuideMulti`, still on
  punch list). In that case, use cinematic inits for this
  workflow and save the illustrated path for a future
  LoRA-stacked or fine-tune variant.
- **Kitten → cat subject change.** The cinematic render shows a
  full-size cat without goggles. The illustrated one had a
  kitten with aviator goggles. If your cinematic render includes
  the goggles, add them back to the subject string. Keep subject
  byte-exact across entries regardless.
- **Why the simpler v2 structure for the cinematic variant.**
  The richer v1 vocabulary (`impact frame`, `supersaturation`)
  was specifically designed to lean into the illustrated/animated
  beat pool. On a cinematic init those terms fight the realism
  just like `Style: illustrated.` did. v2's canonical-camera-only
  structure is closer to live-action cinematography vocabulary
  and pairs cleanly with this init.
