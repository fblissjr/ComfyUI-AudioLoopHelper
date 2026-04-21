Last updated: 2026-04-20

# Action Sequence v4 — Dragon Leap (instrumental action)

Different audio this time: `your-instrumental-track.mp3`
(154.28s / 2:34, 99.4 BPM, D Minor, orchestral). Same init
(`your-cinematic-init-image.png`). Grid unchanged (9 iterations, 17.92s stride,
trim=0).

User brief: "fast-paced, tons of cuts, intense speed, leap onto the
dragon at the end. The previous run was too slow-motion."

## What's different from v1-v3

1. **Speed language replaces contemplation everywhere.** No "held",
   no "caught", no "frozen moments". Every verb implies continuous
   rapid motion: `sprints`, `whips`, `rips`, `slams`, `rockets`,
   `accelerates`, `leaps`, `drives`.
2. **Dragon arrives narratively, not in the init.** Init shows
   her on the Gauntlet. Iter 6 introduces the dragon as a storm
   shadow descending; iter 7 is the LEAP onto it; iter 8 is the
   flight-away. This is genuinely out-of-distribution for LTX i2v
   — observations at the bottom flag what can go wrong.
3. **Subject identity anchor stays byte-exact**: `a warrior woman
   with a brown-and-silver braid in black leather flight gear`
   appears verbatim in every entry. Only the environment phrase
   changes in iters 6-8 as the scene transitions from Gauntlet to
   dragon flight.
4. **Audio section alignment checked directly** against the
   analyzer output, not assumed from a narrative plan. CHORUS 1
   peak lands in iter 3 (`0:53-1:11`); CHORUS 2 peak lands in
   iter 7 (`2:05-2:23`). The LEAP is aligned to the actual peak of
   the second chorus, not to a guessed timestamp.

## Audio structure (new track)

- INTRO 0:00-0:10 (quiet) — very short intro
- VERSE 0:10-0:48 (medium) — 38s
- CHORUS 1 0:48-1:12 (loud) — **24s — long payoff**
- VERSE 1:12-1:38 (medium) — 26s
- BRIDGE 1:38-2:00 (quiet) — 22s
- CHORUS 2 2:00-2:28 (loud) — **28s — longest payoff, dragon leap here**
- OUTRO 2:28-2:34 (quiet) — 6s

CHORUS 1 is longer here than in the instrumental track (24s vs 8s), so
the first payoff has more screen-time. CHORUS 2 at 2:00-2:28 +
6-second OUTRO at 2:28-2:34 is where the dragon sequence lives.

## Inputs

- **Audio**: `<your-instrumental-track.mp3>`
- **Image**: `<your-cinematic-init-image.png>`
- **Subject identity (byte-exact in every entry)**:

  `a warrior woman with a brown-and-silver braid in black leather flight gear`

- **Grid**: `overlap_seconds=2.0`, `window_seconds=19.88`,
  `stride=17.92s`, trim=0. Snap points: `0:00, 0:17, 0:35, 0:53,
  1:11, 1:29, 1:47, 2:05, 2:23`.

## Schedule

Paste `node_169_prompt` into Node 169. Paste the `schedule:` block
into `TimestampPromptSchedule` (Node 1558). First schedule line is
byte-exact to Node 169.

```
node_169_prompt: In a wide shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear sprints hard across the first beam of the Gauntlet with daggers already drawn in both hands, full speed from frame one, stormy sky churning behind her.

schedule:
0:00-0:17: In a wide shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear sprints hard across the first beam of the Gauntlet with daggers already drawn in both hands, full speed from frame one, stormy sky churning behind her.
0:17-0:35: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear whips a dagger into a narrow wooden slot at full speed, her other hand already pulling the next blade from its thigh sheath.
0:35-0:53: Cut to a medium shot, dolly right, camera tracking right. A warrior woman with a brown-and-silver braid in black leather flight gear rips across a horizontal iron beam with rapid-fire dagger plants, one blade after another at blazing speed.
0:53-1:11: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear slams both daggers home mid-leap across a massive gap in the Gauntlet, a blue lightning bolt splitting the sky behind her, hard impact frame.
1:11-1:29: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear rips a dagger free from weathered wood with splinters exploding outward, already reaching for the next slot before her arm finishes the pull, no pause between strikes.
1:29-1:47: Cut to a medium shot, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear rockets upward with hard push-offs from embedded daggers, her braid whipping in the wind, the Gauntlet streaking past, storm intensifying around her.
1:47-2:05: Cut to a wide shot, dolly left, camera tracking left. A warrior woman with a brown-and-silver braid in black leather flight gear reaches the summit of the Gauntlet as a massive storm dragon descends through the torn clouds above, wings blotting out the lightning, she accelerates toward the edge.
2:05-2:23: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear leaps from the summit with a dagger raised, suspended mid-air at the peak of her jump, the storm dragon passing beneath her with its back clear and open.
2:23+: Cut to a wide shot, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear drives her dagger into the scales of the storm dragon's neck as it beats its wings and rises through the parting clouds, lightning receding behind them, wind streaming across them.
```

## Camera + action progression

Nine iterations, five distinct canonical moves, relentless speed
except the split-second suspension at the leap:

| Iter | Time | Shot | Camera | Song section | Action |
|---|---|---|---|---|---|
| init | 0:00-0:17 | wide | `static camera, locked off shot` | INTRO+VERSE | **sprint from frame one** |
| 1 | 0:17-0:35 | close-up | `dolly in, camera pushing forward` | VERSE | whip + already-reaching |
| 2 | 0:35-0:53 | medium | `dolly right, camera tracking right` | VERSE end | rapid-fire dagger plants |
| 3 | 0:53-1:11 | wide | `static camera, locked off shot` | **CHORUS 1** | **double-dagger slam + lightning** |
| 4 | 1:11-1:29 | close-up | `dolly in, camera pushing forward` | VERSE 2 | rip free, already-reaching |
| 5 | 1:29-1:47 | medium | `jib up, camera rising up` | VERSE+BRIDGE | rocket-upward ascent |
| 6 | 1:47-2:05 | wide | `dolly left, camera tracking left` | BRIDGE → CHORUS 2 | **dragon descends + she accelerates** |
| 7 | 2:05-2:23 | wide | `static camera, locked off shot` | **CHORUS 2 PEAK** | **THE LEAP onto dragon's back** |
| 8 | 2:23+ | wide | `jib up, camera rising up` | CHORUS 2 end + OUTRO | dragon rising, final chord |

Design notes:
- **No "slowly", "carefully", "cautiously", "held", "suspended"
  motion language** anywhere except the single-moment leap
  suspension in iter 7 (physics — she has to peak mid-air to land
  on the dragon). "Suspended mid-air at the peak of her jump" is
  the one permitted pause because it's the single frame where she
  transitions from Gauntlet to dragon.
- **Rotation avoids adjacent repeats**: static → dolly-in →
  dolly-right → static → dolly-in → jib-up → dolly-left → static
  → jib-up. Static appears 3× (iter 0, 3, 7) always on anchor
  moments. Dolly-in appears 2× on tension builds.
- **Iter 6 is the hardest narrative beat to land.** Introducing a
  dragon that isn't in the init image during a BRIDGE-quiet audio
  moment. The dolly-left tracking camera motion carries the eye
  toward the descending dragon while the audio is hushed — more
  plausible than trying to do it during peak chorus.
- **Iter 7 is the money shot.** Wide + static = camera holds still
  on the hero moment, peak orchestral hit. The LEAP happens against
  the held camera, not with camera motion. This is the frame viewers
  will remember if the sequence works.
- **Iter 8's `jib up`** with both her and the dragon rising lets the
  camera follow them skyward while keeping them framed — avoids
  the banned dolly-out while still giving the ascent a sense of
  liftoff.

## Workflow widget values

Same as prior action_prompt variants.

| Widget | Node | Value |
|---|---|---|
| `image` | 444 `LoadImage` | `your-cinematic-init-image.png` |
| `audio` filename | 565 `LoadAudio` | **`your-instrumental-track.mp3`** |
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

## Negative prompt (dragon-aware)

Extended with creature-anatomy-suppression terms. Paste into Node 507:

```
still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone, slow motion, motion blur freeze, stopped action, held pose, cartoonish proportions, floating pose, dagger floating free of hand, deformed creature, multiple heads, disfigured wings, unnatural anatomy, incomplete creature body, misshapen dragon
```

New terms beyond action_prompt3's negative: `slow motion`,
`motion blur freeze`, `stopped action`, `held pose` (explicit
anti-slow-motion per the user brief), plus the dragon-anatomy
suppressors `deformed creature`, `multiple heads`, `disfigured
wings`, `unnatural anatomy`, `incomplete creature body`,
`misshapen dragon`.

## Observations

- **The dragon is OUT-OF-DISTRIBUTION for LTX i2v with this single
  init image.** The init shows a vertical Gauntlet, no dragon. The
  model has to hallucinate a large creature from its text-prior
  alone at CFG=1 (weak). Realistic outcomes ranked by likelihood:
  1. Dragon appears as a vague shadow/silhouette during iter 6 but
     doesn't fully render as a creature. She leaps in iter 7 toward
     a dark mass in the sky but it may not read as a dragon.
  2. Dragon renders partially — some wings visible, body shape off.
  3. Clean dragon render (lucky). Most likely with this level of
     text conditioning that won't happen.
  4. No dragon at all — model ignores the prompt and keeps
     rendering Gauntlet for the full 2:34.
- **If the dragon doesn't render cleanly, the run is still valid**
  for iters 0-5 as a fast-paced single-image action piece. You
  could crop the output to 0:00-1:47 and use that as a standalone
  sequence without the leap.
- **To land the dragon reliably**, the structural fix is the
  `_latent_keyframe.json` workflow with a second init image
  showing dragon + rider. Generate one still of your subject on a dragon
  (in your image generator), load both images, use
  `KeyframeImageSchedule` to swap to the dragon image at ~1:47
  (matching iter 6). That gets you a real dragon reveal. Not
  possible from one image alone.
- **"No slow-motion" is enforced in the negative prompt** with
  `slow motion, motion blur freeze, stopped action, held pose`.
  Combined with the action-verb-dense prompts in the positive,
  this should keep the model in "continuous rapid motion" mode.
  If you still see slow-mo moments in the output, the BRIDGE-quiet
  section (iter 5-6) is the suspect — quiet audio + "rocket
  upward" text can produce a model-side compromise. Adding
  `fast motion` or `continuous motion` to the positive of iter 5
  could sharpen it.
- **v3 vs v4 at same seed**: v3 uses a different song; v4 can't
  direct-A/B with v3. But v4 CAN A/B with a pure-Gauntlet version
  of this song (same track, but without the dragon leap — keep
  iters 6-8 on the Gauntlet summit). That isolates whether the
  dragon is helping or hurting.
