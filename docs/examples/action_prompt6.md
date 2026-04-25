Last updated: 2026-04-25 (no schedule changes; this file IS the canonical post-strip version. Framing updated: v5 has since been retroactively stripped to match v6, but the v5→v6 diff below is preserved as the original 2026-04-20 case study that established the strip rule.)

# Action Sequence v6 — No Audio Descriptors

Clean A/B against `action_prompt5.md`. **One variable changes: all
music/audio descriptors removed from every prompt.** Same init, same
audio track (the instrumental track), same grid (20 iterations, stride
8.0s), same subject anchor, same camera rotation, same widget values.
Only the prompt text differs.

## Why strip audio descriptors

The official LTX 2.3 i2v/t2v system prompt
(`docs/reference/ltx23_prompt_system_prompts.md`) directs the prompt engineer to weave audio
descriptions alongside actions. **That guidance assumes the model is
ALSO generating audio.** In our workflow, the audio is FROZEN via
`noise_mask=0` on the audio latent — the model is not generating
audio, it's being given audio and asked to produce video that
lip-syncs / action-syncs to it.

When we put "staccato strings entering" in the prompt while the audio
path is frozen, we're telling the text-conditioning branch to expect
staccato strings in the output — but there's no output audio path.
The cross-attention between text and audio latent may try to reconcile
"prompt says strings" with "audio has strings" with "video should
express strings" → possibly forcing visual intensity to match music
cues that are ALREADY in the frozen audio. Double-counting the music
signal.

**Cleaner trick**: let the frozen audio drive cross-attention alone.
Text conditioning focuses purely on VISUAL content. Audio-visual
binding happens through the audio latent, not through text
re-describing the audio.

## A/B hypothesis

| | v5 (with audio descriptors) | v6 (visual only) |
|---|---|---|
| Music-beat synchronization | Text + audio both signal brass hits → possibly over-emphasized | Audio alone signals → might be more natural |
| Visual punch | "Brass downbeat" amplifies impact frames | Action verbs do all the work |
| Risk | Double-signaling could over-crank chorus payoffs | Text conditioning fully on visuals — possibly tighter action rendering |

If v5's chorus peaks feel over-produced (too on-the-nose with music),
v6 will read as more natural. If v6's chorus peaks feel flat vs. v5,
the audio descriptors were earning their keep.

## What's stripped

Every `staccato strings`, `snare firing`, `brass downbeat`, `chorus
peaking`, `anvil strikes`, `orchestra rebuilding`, `peak orchestral`,
`final chord` — gone. In a few entries I added a visual descriptor
to compensate so the prompt isn't thinly shorter (e.g. `blade biting
into grain` replaces `snare firing`). Identity anchor and action
verbs are unchanged.

Wind and thunder references I LEFT IN where they describe scene-
diegetic sound (wind is present in the storm regardless of the track;
thunder is part of the visual storm). Those aren't redundant with the
frozen audio — they're ambient.

## Inputs (unchanged from v5)

- **Audio**: `<your-instrumental-track.mp3>`
- **Image**: `<your-cinematic-init-image.png>`
- **Subject identity (byte-exact in every entry)**:

  `a warrior woman with a brown-and-silver braid in black leather flight gear`

- **Grid**: `overlap_seconds=2.0`, `window_seconds=9.96`,
  `stride=8.0s`, trim=0. Snap points:
  `0:00, 0:08, 0:16, 0:24, 0:32, 0:40, 0:48, 0:56, 1:04, 1:12,
  1:20, 1:28, 1:36, 1:44, 1:52, 2:00, 2:08, 2:16, 2:24, 2:32`.

## Schedule

Paste `node_169_prompt` into Node 169. Paste the `schedule:` block
into `TimestampPromptSchedule` (Node 1558). First schedule line is
byte-exact to Node 169.

```
node_169_prompt: In a wide shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear sprints across the first beam of the Gauntlet at full speed, daggers drawn in both hands, stormy sky churning behind her.

schedule:
0:00-0:08: In a wide shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear sprints across the first beam of the Gauntlet at full speed, daggers drawn in both hands, stormy sky churning behind her.
0:08-0:16: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear whips a dagger into the first wooden slot at full-speed impact, the blade biting deep into weathered grain.
0:16-0:24: Cut to a medium shot, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear vaults off the embedded dagger and rises fast toward the next obstacle, body airborne mid-push.
0:24-0:32: Cut to a close-up, dolly left, camera tracking left. A warrior woman with a brown-and-silver braid in black leather flight gear slams a second dagger into a vertical beam with hard metallic impact, sparks off the iron.
0:32-0:40: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear rips a dagger free with a backward snap, splinters exploding outward from weathered wood.
0:40-0:48: Cut to a medium shot, dolly right, camera tracking right. A warrior woman with a brown-and-silver braid in black leather flight gear launches sideways across a swinging beam, blades flashing in the storm light.
0:48-0:56: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear slams both daggers home mid-leap across a massive gap as a blue lightning bolt splits the sky behind her.
0:56-1:04: Cut to a close-up, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear drives up the vertical beam kicking off embedded dagger hilts, rising fast.
1:04-1:12: Cut to a medium shot, dolly right, camera tracking right. A warrior woman with a brown-and-silver braid in black leather flight gear whips sideways across a swinging iron crossbeam, body parallel to the chasm at full speed.
1:12-1:20: Cut to a close-up, jib down, camera lowering down. A warrior woman with a brown-and-silver braid in black leather flight gear plunges down a steel rail with her dagger biting iron for control, sparks streaming in a trail behind her.
1:20-1:28: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear drives hard into the next vertical climb, daggers biting deep in rapid succession.
1:28-1:36: Cut to an extreme close-up, focus shift, rack focus. A warrior woman with a brown-and-silver braid in black leather flight gear switches daggers between hands mid-motion, blades flashing, thumbs rolling across the hilts.
1:36-1:44: Cut to a medium shot, dolly right, camera tracking right. A warrior woman with a brown-and-silver braid in black leather flight gear lays down a rapid sequence of dagger plants along a horizontal beam, steel biting wood in fast succession.
1:44-1:52: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear pauses on a pivotal beam with a dagger raised, wind roaring through the beams around her.
1:52-2:00: Cut to a low-angle shot, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear tilts her gaze skyward, storm clouds churning overhead, a massive dragon's shadow beginning to descend through them.
2:00-2:08: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear stands braced at the summit as a massive storm dragon descends through torn clouds, wings blotting out the lightning.
2:08-2:16: Cut to a medium shot, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear sprints hard across the final narrow beam toward the edge of the Gauntlet as the dragon soars beneath.
2:16-2:24: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear leaps from the edge with a dagger raised high, suspended mid-air at the apex of her jump, the dragon passing beneath her with its back clear.
2:24-2:32: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear drives her dagger deep into the storm dragon's neck scales and anchors hard, lightning flaring behind her.
2:32+: Cut to a wide shot, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear on the storm dragon's back as it beats its wings and rises through parting clouds, wind streaming across them.
```

## What changed in the original 2026-04-20 strip pass (preserved as discovery record)

> **Note (2026-04-25):** v5 has been retroactively stripped to match
> v6. The diff table below is the original 2026-04-20 case study that
> established the "strip music descriptors when audio is frozen" rule
> (now `docs/guides/prompt_creation_guide.md` §3.1). Comparing v5 and
> v6 schedules today shows them identical modulo minor wording; the
> table documents the *discovery*, not the current state.

| Iter | v5 had | v6 has |
|---|---|---|
| 0 | `...stormy sky churning, staccato strings entering.` | `...stormy sky churning behind her.` |
| 1 | `...full-speed impact, snare firing.` | `...full-speed impact, the blade biting deep into weathered grain.` |
| 2 | `...next obstacle, strings accelerating.` | `...next obstacle, body airborne mid-push.` |
| 3 | `...on the downbeat, hard metallic impact.` | `...hard metallic impact, sparks off the iron.` |
| 4 | `...splinters exploding outward.` | unchanged |
| 5 | `...strings building into the chorus.` | `...blades flashing in the storm light.` |
| 6 | `...exactly on the chorus brass downbeat.` | `...as a blue lightning bolt splits the sky behind her.` |
| 7 | `...chorus driving hard.` | `...rising fast.` |
| 8 | `...body parallel to the chasm, full speed.` | unchanged |
| 9 | `...sparks streaming, chorus peaking.` | `...sparks streaming in a trail behind her.` |
| 10 | `...rapid succession.` | unchanged |
| 11 | `...thumbs rolling across hilts.` | `...thumbs rolling across the hilts.` (minor) |
| 12 | `...staccato anvil strikes.` | `...steel biting wood in fast succession.` |
| 13 | `...wind roaring, music coiling tight before the surge.` | `...wind roaring through the beams around her.` |
| 14 | (unchanged structure) | unchanged |
| 15 | `...orchestra rebuilding.` | (dropped; ends on `blotting out the lightning`) |
| 16 | (unchanged) | unchanged |
| 17 | `...peak orchestral chorus.` | (dropped; ends on `with its back clear`) |
| 18 | `...lightning flaring behind her.` | `...and anchors hard, lightning flaring behind her.` |
| 19 | `...final chord sustaining.` | `...wind streaming across them.` |

All music-instrumentation references removed. Diegetic ambient
sound (`wind roaring`, `stormy sky churning`, `lightning splits`)
kept where it describes a visual scene element. Visual substitutes
added where prompts would have become too short.

## Workflow widget values (identical to v5)

| Widget | Node | Value |
|---|---|---|
| `image` | 444 `LoadImage` | `your-cinematic-init-image.png` |
| `audio` filename | 565 `LoadAudio` | `your-instrumental-track.mp3` |
| `start` (outer trim) | 567 `TrimAudioDuration` | `0` |
| `overlap_seconds` | 1582 `AudioLoopController` | `2.0` |
| `window_size_seconds` | 688 `FloatConstant` | **`9.96`** |
| `length` | 526 `PrimitiveNode` | **`249`** |
| resolution | 445 `ImageResizeKJv2` | `832 x 448` |
| `img_compression` | 446 `LTXVPreprocess` | `18` |
| Node 169 text | 169 `CLIPTextEncode` | paste `node_169_prompt` above |
| schedule text | 1558 `TimestampPromptSchedule` | paste `schedule:` block above |
| `blend_seconds` / `snap_boundaries` | 1558 | `0.0` / `true` |
| `sampler_name` | 154 `KSamplerSelect` | `euler` |
| scheduler | 1421 `BasicScheduler` | `linear_quadratic, 8, 1` |
| `shift` | 1513 `ModelSamplingSD3` | `13` |
| `cfg` | 153 `CFGGuider` | `1.0` |
| AdaIN factor | 2006 `LTXVAdainLatent` (inside subgraph) | `2.63` |

## Negative prompt (identical to v5)

```
still image with no motion, deformed facial features, extra limbs, disfigured hands, duplicate character, slow motion, frozen pose, floating pose, dagger floating free of hand, deformed creature
```

## Observations

- **The frozen-audio architectural insight is worth documenting
  elsewhere.** LTX's official prompting guide assumes audio is being
  generated; our workflow uses audio as FIXED context. Adding this
  nuance to `CLAUDE.md` or `docs/guides/debugging_guide.md` would help
  future prompt-writing.
- **If v6 reads as "flatter" than v5 on chorus peaks**, the audio
  descriptors were providing real reinforcement even with frozen
  audio. Fallback: restore music descriptors ONLY on chorus-peak
  iterations (6, 15-17) and leave mid-verse iters clean.
- **If v6 reads as tighter action** with cleaner visual cross-
  attention, the audio-in-prompt was actually double-signaling and
  this is the correct pattern going forward.
- **Present simple verbs kept from v5** (`sprints`, `slams`, `rips`)
  rather than switching to present progressive (`is sprinting`,
  `is slamming`). Progressive is off-spec but more continuous;
  simple is punchier. Single-variable A/B — don't want to change
  this at the same time as the audio strip. If v6 runs clean,
  progressive-verb rewrite becomes the next candidate change.
- **Same seed as v5 for clean comparison.** Diff chorus iterations
  (iters 6 and 17 especially). Those are where the two approaches
  will differ most visibly.
