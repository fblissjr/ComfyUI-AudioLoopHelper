Last updated: 2026-04-20

# Action Sequence v2 — Single-Init Maximized

Companion to `action_prompt1.md`. Same audio
(`your-instrumental-track.mp3`), same init
(`your-cinematic-init-image.png`), same grid (9 iterations, 17.92s stride, trim=0).

v1's flaw: it tried to hit Gemini's 7-act multi-shot plan, which
assumes fundamentally different compositions LTX i2v can't produce
from a single init image. Also, Gemini's "peak chorus" at 1:55 was
actually 13s too early vs. the real song (CHORUS 2 starts at 2:08).

v2 flips the approach: **stay inside the single init's visual
vocabulary and extract maximum variety from what IS in the frame.**
She's not always climbing — she can swing, retrieve, balance, nearly
fall, recover, hold. The init already shows everything needed:
daggers, leather flight gear, wooden+iron vertical structure,
volumetric fog abyss below, stormy sky, lightning.

## What's different from v1

1. **Aligned to the actual song sections**, not Gemini's approximate
   timings. Iter 6 is the real BRIDGE quiet (not v1's mis-stated
   "full orchestral swell"); iter 7 aligns to the real CHORUS 2 peak
   start (song 2:08 = video 2:08).
2. **Action variety within one composition.** Instead of pretending
   we can cut to the comedian's ring or to a POV-downward chasm
   shot, every iteration uses a different *body-action + camera
   relationship* on the scene the init image already shows.
3. **Camera pans INTO the abyss.** `jib down, camera lowering down`
   in iter 1 (early scale-setting) and iter 6 (the near-fall) —
   these tilt the camera to show the volumetric fog + depth already
   present in the init. No new content needed; just reframe what's
   already there.
4. **Near-fall moment explicit in iter 6.** A hand slip, a caught
   recovery, then the BRIDGE's hush before the CHORUS 2 payoff in
   iter 7. Creates narrative tension without needing a new location.

## Inputs

- **Audio**: `<your-instrumental-track.mp3>`
  (150.47s, 123 BPM, C Minor, instrumental).
- **Image**: `<your-cinematic-init-image.png>`.
- **Subject string (byte-exact in every entry)**:

  `a warrior woman with a brown-and-silver braid in black leather flight gear, daggers wedged into a brutalist wooden and iron Gauntlet suspended above volumetric fog`

- **Grid**: `overlap_seconds=2.0`, `window_seconds=19.88`,
  `stride=17.92s`. Snap points: `0:00, 0:17, 0:35, 0:53, 1:11, 1:29,
  1:47, 2:05, 2:23`. No audio trim.

## Schedule

Paste `node_169_prompt` into Node 169. Paste the `schedule:` block
into `TimestampPromptSchedule` (Node 1558). First schedule line is
byte-exact to Node 169.

```
node_169_prompt: In a wide shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear, daggers wedged into a brutalist wooden and iron Gauntlet suspended above volumetric fog, held mid-traverse with one hand gripping an embedded blade and the other raised with a drawn dagger, the scale of the structure dwarfing her, a distant lightning flicker catching the metal.

schedule:
0:00-0:17: In a wide shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear, daggers wedged into a brutalist wooden and iron Gauntlet suspended above volumetric fog, held mid-traverse with one hand gripping an embedded blade and the other raised with a drawn dagger, the scale of the structure dwarfing her, a distant lightning flicker catching the metal.
0:17-0:35: Cut to a medium shot, jib down, camera lowering down. A warrior woman with a brown-and-silver braid in black leather flight gear, daggers wedged into a brutalist wooden and iron Gauntlet suspended above volumetric fog, tips her gaze toward the chasm as the camera tilts with her, revealing the mile-deep drop through swirling mist.
0:35-0:53: Cut to a close-up, focus shift, rack focus. A warrior woman with a brown-and-silver braid in black leather flight gear, daggers wedged into a brutalist wooden and iron Gauntlet suspended above volumetric fog, focus pulls from her weather-bitten face to the steel of the dagger blade biting into the dark grain of weathered wood, then back to her narrowed eyes.
0:53-1:11: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear, daggers wedged into a brutalist wooden and iron Gauntlet suspended above volumetric fog, swings outward in a pendulum arc, her full weight on a single embedded dagger, free arm extended for counterbalance, a massive blue lightning bolt striking the sky behind her.
1:11-1:29: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear, daggers wedged into a brutalist wooden and iron Gauntlet suspended above volumetric fog, yanks a dagger free from the beam with a hard backwards pull, splinters flying, the blade scraping out in a sharp metallic rasp.
1:29-1:47: Cut to a medium shot, dolly right, camera tracking right. A warrior woman with a brown-and-silver braid in black leather flight gear, daggers wedged into a brutalist wooden and iron Gauntlet suspended above volumetric fog, traverses horizontally along a swinging iron crossbeam, planting a dagger with each lateral step, her braid trailing in the wind off the chasm.
1:47-2:05: Cut to a close-up, jib down, camera lowering down. A warrior woman with a brown-and-silver braid in black leather flight gear, daggers wedged into a brutalist wooden and iron Gauntlet suspended above volumetric fog, her gloved hand slipping a full inch on a dagger hilt as the wood groans, she catches herself just before falling, the camera tilts downward with her near-plunge revealing the abyss beneath.
2:05-2:23: Cut to a medium shot, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear, daggers wedged into a brutalist wooden and iron Gauntlet suspended above volumetric fog, recovers and drives her primary dagger deep into an iron seam with a shower of sparks, pushing off for the final upward surge, the camera rising with her.
2:23+: Cut to a close-up, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear, daggers wedged into a brutalist wooden and iron Gauntlet suspended above volumetric fog, holds at the apex of her reach with the dagger raised, chest heaving, a long-held moment of breath as the last lightning fades and the wind comes through.
```

## Camera + action progression

Nine iterations, seven distinct canonical moves, **camera pans into the abyss twice** (iter 1 scale-setting, iter 6 near-fall):

| Iter | Time | Shot | Camera | Song section | Action beat |
|---|---|---|---|---|---|
| init | 0:00-0:17 | wide | `static camera, locked off shot` | BRIDGE quiet | held mid-traverse, scale setting |
| 1 | 0:17-0:35 | medium | `jib down, camera lowering down` | VERSE medium | **camera tilts into the abyss** as her gaze drops |
| 2 | 0:35-0:53 | close-up | `focus shift, rack focus` | VERSE medium | tactical concentration, blade focus pull |
| 3 | 0:53-1:11 | wide | `static camera, locked off shot` | **CHORUS 1** | **pendulum swing on single dagger + lightning** |
| 4 | 1:11-1:29 | close-up | `dolly in, camera pushing forward` | VERSE medium | dagger retrieval (pulling free, not jamming) |
| 5 | 1:29-1:47 | medium | `dolly right, camera tracking right` | VERSE + BRIDGE | horizontal traverse along a crossbeam |
| 6 | 1:47-2:05 | close-up | `jib down, camera lowering down` | BRIDGE quiet | **the near-fall: hand slip + abyss reveal** |
| 7 | 2:05-2:23 | medium | `jib up, camera rising up` | **CHORUS 2** | **recovery surge upward, sparks, peak brass** |
| 8 | 2:23+ | close-up | `static camera, locked off shot` | CHORUS 2 end | held apex, chord sustain |

Design notes:
- **Both chorus payoffs** use different camera techniques (iter 3
  static lightning-swing vs. iter 7 jib-up recovery-surge) to avoid
  the payoffs looking identical.
- **Both quiet BRIDGE moments** (init and iter 6) are the only two
  places the camera is most composed / slow — iter 6's near-fall is
  emphasized by the music's hush. The song itself supports this
  pacing choice.
- **Three different dagger actions**: jamming (in the init/first
  schedule line), retrieving/pulling-free (iter 4), driving deep
  with sparks (iter 7). Each iteration is one specific dagger
  moment, not a generic "climbing" description.
- **No duplicate camera moves on adjacent iterations.** The rotation
  goes static → jib-down → focus-shift → static → dolly-in →
  dolly-right → jib-down → jib-up → static. Only `static` and
  `jib-down` appear twice, and always with at least one other move
  between them.

## Workflow widget values

Identical to `action_prompt1.md`. Only Node 169 text and schedule
text change between v1 and v2. Use `audio-loop-music-video_latent.json`
(baseline) for the first test.

| Widget | Node | Value |
|---|---|---|
| `image` | 444 `LoadImage` | `your-cinematic-init-image.png` |
| `audio` filename | 565 `LoadAudio` | `your-instrumental-track.mp3` |
| `start` (outer trim) | 567 `TrimAudioDuration` | `0` |
| `overlap_seconds` | 1582 `AudioLoopController` | `2.0` |
| `window_seconds` | 688 `FloatConstant` | `19.88` |
| `length` | 526 `PrimitiveNode` | `497` |
| resolution | 445 `ImageResizeKJv2` | `832 x 448` |
| `img_compression` | 446 `LTXVPreprocess` | `18` |
| Node 169 text | 169 `CLIPTextEncode` | paste `node_169_prompt` above |
| schedule text | 1558 `TimestampPromptSchedule` | paste `schedule:` block above |
| `blend_seconds` / `snap_boundaries` | 1558 | `0.0` / `true` |
| `sampler_name` | 154 `KSamplerSelect` | `euler` |
| scheduler | 1421 `BasicScheduler` | `linear_quadratic, 8, 1` |
| `shift` | 1513 `ModelSamplingSD3` | `13` |
| `cfg` | 153 `CFGGuider` | `1.0` |
| decoder | 1604, 1597 `LTXVTiledVAEDecode` | unchanged |

## Negative prompt

Same as `action_prompt1.md`. Node 507 `CLIPTextEncode`:

```
still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone, blurry motion, slow motion blur, cartoonish proportions, unrealistic physics, floating pose, mid-air suspended without anchor, dagger floating free of hand
```

## Observations

- **If this works, the pattern is "use the init as a visual vocabulary,
  extract multiple beats from the same frame."** Every iteration is
  an answer to: "given this composition — her mid-traverse, two
  daggers, stormy sky, fog below — what's one specific thing she could
  be doing RIGHT NOW that differs from the last thing?" Swinging, not
  climbing. Retrieving, not jamming. Nearly falling, not pushing
  forward. Held breath, not action.
- **Subject string is now rich with environment-anchor**: "daggers
  wedged into a brutalist wooden and iron Gauntlet suspended above
  volumetric fog". This repeats the init's actual content in every
  entry so the model stays inside the composition. Longer than usual
  but the repetition is load-bearing — it keeps drift from moving her
  off the Gauntlet.
- **The near-fall in iter 6 is the riskiest beat.** A hand slipping
  exactly one inch, caught before falling, is a subtle animation
  that LTX may not cleanly produce. Fallback if it reads as a
  full-fall or reads as nothing: change to "a sharp intake of breath
  as the wind gusts and the dagger wavers in its seat, sweat beading
  at her temple, the silence before recovery."
- **v1 vs v2 choice**: v1 tried to replicate Gemini's multi-shot
  ambition and probably can't land it on a single init. v2 aims
  lower (single-location action variety) and probably lands it.
  If v2 works and you want more narrative scope later, port to the
  `_latent_keyframe.json` workflow with multiple renders of your subject
  at different Gauntlet moments. Not possible from one image alone.
- **Camera "pan-down to abyss"** appears twice deliberately. Iter 1
  uses it to establish scale when the audio is quiet (low strings
  building); iter 6 uses it to intensify the near-fall during the
  second BRIDGE. The two instances read as thematically related
  ("the drop is always there") rather than as a repeated gimmick.
