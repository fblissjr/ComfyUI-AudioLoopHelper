Last updated: 2026-04-25 (audio descriptors stripped per `docs/guides/prompt_creation_guide.md`)

# Action Sequence v3 — Fast-Paced / Cut-Heavy

Third variant. Same audio (`your-instrumental-track.mp3`), same
init (`your-cinematic-init-image.png`), same grid (9 iterations, 17.92s stride,
trim=0). v1 chased Gemini's multi-shot narrative; v2 stayed inside
the init's composition with varied pacing. v3 does everything in v2's
"one-composition" discipline but pushes the rhythm HARD — every entry
is a single punchy action beat, no near-fall, no contemplative
moments. The workflow's 9 iteration boundaries are the cuts; every
entry leans into them as intentional edits.

## What's different from v2

1. **No near-fall.** Iter 6 is no longer a "hand slips" void moment.
   Instead it's a tight-coiled BRIDGE-quiet beat — she snaps her
   gaze between two daggers deciding which to strike next. The music
   is externally quiet, but her INTERNAL rhythm stays fast. Tension
   without de-escalation.
2. **Every entry is one punchy action.** No "holds", no "catches",
   no "suspends". Verbs: `slashes`, `rips`, `slams`, `hurls`,
   `lunges`, `drives`. Present-progressive rapid motion, never
   contemplative.
3. **Music-beat callouts embedded in prompts.** `on the downbeat`,
   `on the brass hit`, `every anvil strike a match cut`. Reinforces
   that iteration boundaries ARE cuts and the cuts land on specific
   audio events.
4. **Compact subject anchor.** Shortened to "a warrior woman with a
   brown-and-silver braid in black leather flight gear on the
   Gauntlet" — shorter than v2's env-embedded version so the verb
   gets more token-weight per entry.
5. **Final frame is an impact, not a held breath.** Iter 8 ends on a
   dagger-into-wood impact frame timed to the final chord, not on
   her apex recovery. Matches the fast-paced aesthetic through to
   the last beat.

## Inputs

- **Audio**: `<your-instrumental-track.mp3>`
- **Image**: `<your-cinematic-init-image.png>`
- **Subject string (byte-exact in every entry)**:

  `a warrior woman with a brown-and-silver braid in black leather flight gear on the Gauntlet`

- **Grid**: `overlap_seconds=2.0`, `window_seconds=19.88`,
  `stride=17.92s`, trim=0. Snap points: `0:00, 0:17, 0:35, 0:53,
  1:11, 1:29, 1:47, 2:05, 2:23`.

## Schedule

Paste `node_169_prompt` into Node 169. Paste the `schedule:` block
into `TimestampPromptSchedule` (Node 1558). First schedule line is
byte-exact to Node 169.

```
node_169_prompt: In a wide shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear on the Gauntlet crouches poised with a dagger in each hand, eyes locked on the first obstacle, a coiled spring of muscle and intent, stormy sky churning behind her.

schedule:
0:00-0:17: In a wide shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear on the Gauntlet crouches poised with a dagger in each hand, eyes locked on the first obstacle, a coiled spring of muscle and intent, stormy sky churning behind her.
0:17-0:35: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear on the Gauntlet slashes a dagger across the frame and drives the blade deep into the first slot with a single hard impact.
0:35-0:53: Cut to a medium shot, dolly right, camera tracking right. A warrior woman with a brown-and-silver braid in black leather flight gear on the Gauntlet rips her second dagger from its thigh sheath mid-motion, blade flashing in the stormlight, already cocked for the next strike.
0:53-1:11: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear on the Gauntlet slams both daggers home mid-leap, a massive blue lightning bolt striking the sky behind her, impact frame.
1:11-1:29: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear on the Gauntlet rips a dagger free with a sharp backwards snap, splinters exploding out of weathered wood in fast succession.
1:29-1:47: Cut to a medium shot, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear on the Gauntlet hurls her body upward with a hard push off an embedded dagger, her braid whipping like a banner in the wind off the chasm.
1:47-2:05: Cut to an extreme close-up, focus shift, rack focus. A warrior woman with a brown-and-silver braid in black leather flight gear on the Gauntlet snaps her gaze between two daggers in her grip, choosing which to strike with next, thumb rolling across a hilt, wind coiling through the beams around her.
2:05-2:23: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear on the Gauntlet lunges airborne at the apex, driving her primary dagger deep into the highest iron seam, a blinding lightning flash behind her.
2:23+: Cut to an extreme close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear on the Gauntlet's dagger blade drives deep into wood, sparks trailing from iron on iron, the impact held as wind streams across her braid.
```

## Camera + action progression

Nine iterations, six distinct canonical moves, zero held moments:

| Iter | Time | Shot | Camera | Song section | Single action |
|---|---|---|---|---|---|
| init | 0:00-0:17 | wide | `static camera, locked off shot` | BRIDGE quiet | coiled poise pre-strike |
| 1 | 0:17-0:35 | close-up | `dolly in, camera pushing forward` | VERSE medium | slash + drive into first slot |
| 2 | 0:35-0:53 | medium | `dolly right, camera tracking right` | VERSE medium | rip second dagger from sheath |
| 3 | 0:53-1:11 | wide | `static camera, locked off shot` | **CHORUS 1** | **slam both + lightning on brass hit** |
| 4 | 1:11-1:29 | close-up | `dolly in, camera pushing forward` | VERSE medium | rip dagger free, splinters explode |
| 5 | 1:29-1:47 | medium | `jib up, camera rising up` | VERSE+BRIDGE | hurl upward, braid banner |
| 6 | 1:47-2:05 | extreme close-up | `focus shift, rack focus` | BRIDGE quiet | **snap gaze between daggers (internal rhythm)** |
| 7 | 2:05-2:23 | wide | `static camera, locked off shot` | **CHORUS 2** | **airborne lunge + peak brass + lightning** |
| 8 | 2:23+ | extreme close-up | `dolly in, camera pushing forward` | CHORUS 2 end | **dagger-into-wood final impact** |

Design notes:
- **Every entry names exactly one action** and one musical callout.
  No scene-setting language, no "as the camera shows", no
  contemplation. Screenplay-style cuts.
- **Two CHORUS payoffs are both wide+static** (iter 3 and iter 7).
  Different actions (double-slam vs. airborne lunge) + different
  context (first lightning vs. peak orchestral) keep them from
  reading as identical payoffs. Camera stays still on the chorus
  hits so the action does the visual work.
- **Iter 6 during BRIDGE quiet** is the riskiest balance — the music
  hushes, but her body stays fast. If LTX interprets the music too
  literally, she may freeze completely. The `focus shift, rack focus`
  camera move suggests continuous attention movement regardless of
  external tempo. "Thumb rolling across a hilt" is small but
  non-zero motion.
- **No camera move repeats on adjacent iterations.** Rotation:
  static → dolly-in → dolly-right → static → dolly-in → jib-up →
  focus-shift → static → dolly-in. Dolly-in appears 3× spaced out;
  static appears 3× (always on chorus/anchor moments).
- **Final iter 8's dolly-in on an impact frame** is the cinematic
  counterpart to how we closed the music-video runs: those held the
  face, these hold the blade. Same "let the audio do the fade" rule,
  different focal object.

## Workflow widget values

Identical to action_prompt1 and action_prompt2. Only Node 169 text
and schedule text change between variants. Use
`audio-loop-music-video_latent.json` (baseline).

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

## Negative prompt

Same as action_prompt1 and action_prompt2. Node 507:

```
still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone, blurry motion, slow motion blur, cartoonish proportions, unrealistic physics, floating pose, mid-air suspended without anchor, dagger floating free of hand
```

## Observations

- **The workflow cannot produce true sub-iteration cuts.** Each
  iteration is ~20s of one approximately-continuous framing. "Fast
  paced with tons of cuts" in our architecture is 8 actual cuts
  (iteration boundaries) labeled aggressively as edits. v3 leans
  into this by starting every non-init entry with literal "Cut to"
  language — which per v4-standup precedent makes the iteration-seam
  visible as intentional editing rather than a blending artifact.
- **Action-verb density matters more than prompt length.** v2
  averaged 40-50 words per entry with environment+action-description.
  v3 averages ~30 words but the verb-to-modifier ratio is higher.
  If the distilled model responds visibly to verb density, v3 will
  feel tighter; if it responds more to total token count, v2 will
  feel richer. A/B data point against v2 at same seed.
- **Risk: iter 6's "fast-internal-during-quiet-external" is the
  single hardest beat in the schedule.** LTX's audio-video cross-
  attention binds visual intensity to audio intensity. Asking for
  "fast eye movement + hilt rolling" during an audibly quiet moment
  is asking the model to ignore part of what it was trained on.
  Fallback: if iter 6 reads as too-still, change it to "wind gusts
  hard and her braid lashes across her face — she shakes it off
  and her gaze locks forward, jaw set". External (wind) motion
  aligns with BRIDGE without de-escalating her.
- **If this variant feels right but you want even MORE cuts,** the
  structural path is: halve window_seconds to ~9.88 (making stride
  ~9.0s), doubling the iteration count to 18. Rewrites the whole
  schedule but gives twice the cut density. Separate experiment.
- **v1/v2/v3 A/B**: all three at the same seed against the same
  init. v1 = narrative ambition (won't fully land with single init).
  v2 = varied pacing with near-fall. v3 = relentless impact. Pick
  the one whose rhythm matches the intended energy of the final
  edit.
