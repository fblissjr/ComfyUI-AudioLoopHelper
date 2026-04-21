Last updated: 2026-04-20

# Action Sequence v5 — Rapid-Cut 21-Iteration Grid

**Architectural variant** of action_prompt4. Same audio
(`your-instrumental-track.mp3`), same init (`your-cinematic-init-image.png`), but
**halved window_seconds** so the workflow runs 21 iterations instead
of 9. Stride drops from 17.92s → 7.68s. Each iteration is a punchy
~7-second beat, MTV-edit-rhythm cut density.

## Why 20 iterations at stride=8.0s

action_prompt4 at stride=17.92 gave 9 cuts over 2:34. That's ~1 cut
per 17s — slower than actual action edit rhythm (1.5-4s per shot).

Halving `window_seconds` gets 2.5× the cut density. Specifically:
- `length=249` (valid — ComfyUI requires `(length-1) % 8 == 0`; 247
  rounds up to 249)
- `window_seconds = 249/25 = 9.96`
- `overlap_latents = 7` (from `overlap_seconds=2.0`)
- `window_latents = 32`, `new_latents = 25`
- **stride = 25 * 8 / 25 = 8.0s exactly**
- iterations to cover 154.28s audio: **20**

One cut per 8s is still not cinema-speed but it's **2.24× the cut
density** of v4 within the constraint of our one-sampler-pass-per-
iteration architecture. Stride=8.0 also gives round-number snap
points (`0:00, 0:08, 0:16, ...`) which is easier to reason about.

## Tradeoffs

1. **2.3× iteration seams** (21 vs 9). Drift-fix keeps audio
   aligned; each seam is a fresh sampler pass that inherits the
   previous iter's tail latent via overlap. Seams should read as
   intentional cuts rather than jitter, but more of them = more
   chances for a visible hand-off artifact.
2. **2.3× generation time.** Same model complexity per iteration,
   just more of them. If v4 took ~25 min, v5 takes ~55-60 min.
3. **Less per-iter buildup.** 7.68s windows give the model less
   time to establish continuous motion. For fast-paced action this
   ALIGNS with the goal — short windows force punchy beats. For
   anything slow/contemplative, this workflow variant is wrong.
4. **Overlap-as-fraction grows** from 10% → 22% (2s/9.88s). More
   compute-per-new-second vs v4. Could drop `overlap_seconds` to
   1.0 (stride = 8.32s, 20 iters) if render time matters more than
   continuity.

## Inputs

- **Audio**: `<your-instrumental-track.mp3>`
- **Image**: `<your-cinematic-init-image.png>`
- **Subject identity (byte-exact in every entry)**:

  `a warrior woman with a brown-and-silver braid in black leather flight gear`

- **Grid**: `overlap_seconds=2.0`, **`window_seconds=9.96`**,
  **`stride=8.0s`**, trim=0. Snap points:
  `0:00, 0:08, 0:16, 0:24, 0:32, 0:40, 0:48, 0:56, 1:04, 1:12,
  1:20, 1:28, 1:36, 1:44, 1:52, 2:00, 2:08, 2:16, 2:24, 2:32`.

## Narrative arc across 20 iterations (stride=8.0s)

| Iters | Audio section | Narrative act |
|---|---|---|
| 0-4 | INTRO + early VERSE | Sprint: first obstacles, dagger plants, ripping free |
| 5-8 | CHORUS 1 | Payoff 1: double-dagger slam + lightning, ascent, sideways whip |
| 9-11 | VERSE 2 | Second push: sparks, vertical drive, dagger exchange |
| 12-14 | BRIDGE | Storm rising: rapid plants → pause → gaze skyward |
| 15-17 | CHORUS 2 | Dragon descends → sprint to edge → **the leap** |
| 18-19 | CHORUS 2 end + OUTRO | Dagger into scales, dragon rises, final chord |

## Schedule

Paste `node_169_prompt` into Node 169. Paste the `schedule:` block
into `TimestampPromptSchedule` (Node 1558). First line byte-exact.

```
node_169_prompt: In a wide shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear sprints across the first beam of the Gauntlet at full speed, daggers drawn in both hands, stormy sky churning, staccato strings entering.

schedule:
0:00-0:08: In a wide shot, static camera, locked off shot, a warrior woman with a brown-and-silver braid in black leather flight gear sprints across the first beam of the Gauntlet at full speed, daggers drawn in both hands, stormy sky churning, staccato strings entering.
0:08-0:16: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear whips a dagger into the first wooden slot, full-speed impact, snare firing.
0:16-0:24: Cut to a medium shot, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear vaults off the embedded dagger, rising fast toward the next obstacle, strings accelerating.
0:24-0:32: Cut to a close-up, dolly left, camera tracking left. A warrior woman with a brown-and-silver braid in black leather flight gear slams a second dagger into a vertical beam on the downbeat, hard metallic impact.
0:32-0:40: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear rips a dagger free with a backward snap, splinters exploding outward.
0:40-0:48: Cut to a medium shot, dolly right, camera tracking right. A warrior woman with a brown-and-silver braid in black leather flight gear launches sideways across a swinging beam, blades flashing, strings building into the chorus.
0:48-0:56: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear slams both daggers home mid-leap across a massive gap, lightning striking exactly on the chorus brass downbeat.
0:56-1:04: Cut to a close-up, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear drives up the vertical beam kicking off embedded dagger hilts, chorus driving hard.
1:04-1:12: Cut to a medium shot, dolly right, camera tracking right. A warrior woman with a brown-and-silver braid in black leather flight gear whips sideways across a swinging iron crossbeam, body parallel to the chasm, full speed.
1:12-1:20: Cut to a close-up, jib down, camera lowering down. A warrior woman with a brown-and-silver braid in black leather flight gear plunges down a steel rail with her dagger biting iron for control, sparks streaming, chorus peaking.
1:20-1:28: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear drives hard into the next vertical climb, daggers biting deep in rapid succession.
1:28-1:36: Cut to an extreme close-up, focus shift, rack focus. A warrior woman with a brown-and-silver braid in black leather flight gear switches daggers between hands mid-motion, blades flashing, thumbs rolling across hilts.
1:36-1:44: Cut to a medium shot, dolly right, camera tracking right. A warrior woman with a brown-and-silver braid in black leather flight gear lays down a rapid sequence of dagger plants along a horizontal beam, staccato anvil strikes.
1:44-1:52: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear pauses on a pivotal beam with a dagger raised, wind roaring, music coiling tight before the surge.
1:52-2:00: Cut to a low-angle shot, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear tilts her gaze skyward, storm clouds churning, a massive dragon's shadow beginning to descend through them.
2:00-2:08: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear stands braced at the summit as a massive storm dragon descends through torn clouds, wings blotting out the lightning, orchestra rebuilding.
2:08-2:16: Cut to a medium shot, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear sprints hard across the final narrow beam toward the edge of the Gauntlet as the dragon soars beneath.
2:16-2:24: Cut to a wide shot, static camera, locked off shot. A warrior woman with a brown-and-silver braid in black leather flight gear leaps from the edge with a dagger raised high, suspended mid-air at the apex of her jump, the dragon passing beneath her with its back clear, peak orchestral chorus.
2:24-2:32: Cut to a close-up, dolly in, camera pushing forward. A warrior woman with a brown-and-silver braid in black leather flight gear drives her dagger deep into the storm dragon's neck scales, anchored, lightning flaring behind her.
2:32+: Cut to a wide shot, jib up, camera rising up. A warrior woman with a brown-and-silver braid in black leather flight gear on the storm dragon's back as it beats its wings and rises through parting clouds, final chord sustaining into wind.
```

## Camera rotation (21 iterations, 7 canonical moves)

| Iter | Time | Camera | Anchor moment? |
|---|---|---|---|
| 0 | 0:00 | static | init scale-setting |
| 1 | 0:07 | dolly in | |
| 2 | 0:15 | jib up | |
| 3 | 0:23 | dolly left | |
| 4 | 0:30 | dolly in | |
| 5 | 0:38 | dolly right | |
| 6 | 0:46 | **static** | **CHORUS 1 downbeat** |
| 7 | 0:53 | jib up | |
| 8 | 1:01 | dolly right | |
| 9 | 1:09 | jib down | |
| 10 | 1:16 | dolly in | |
| 11 | 1:24 | focus shift | |
| 12 | 1:32 | dolly right | |
| 13 | 1:39 | **static** | bridge pause |
| 14 | 1:47 | jib up | looking up |
| 15 | 1:55 | **static** | dragon arrives |
| 16 | 2:02 | dolly in | sprint to edge |
| 17 | 2:10 | **static** | **THE LEAP** |
| 18 | 2:18 | dolly in | anchor dagger |
| 19 | 2:25 | jib up | dragon rising |
| 20 | 2:33+ | static | final silhouette |

Static appears 6×, always on anchor moments (init, chorus 1 peak,
bridge pause, dragon arrival, the leap, final). No two adjacent
iterations share a camera move.

## Workflow widget values

**Two widgets change vs. v4**: `window_seconds` and `length`. Everything
else identical.

| Widget | Node | Value | Change? |
|---|---|---|---|
| `image` | 444 `LoadImage` | `your-cinematic-init-image.png` | same |
| `audio` filename | 565 `LoadAudio` | `your-instrumental-track.mp3` | same |
| `start` (outer trim) | 567 `TrimAudioDuration` | `0` | same |
| `overlap_seconds` | 1582 `AudioLoopController` | `2.0` | same |
| **`window_size_seconds`** | **688 `FloatConstant`** | **`9.88`** | **CHANGED from 19.88** |
| **`length`** | **526 `PrimitiveNode`** | **`247`** | **CHANGED from 497** |
| resolution | 445 `ImageResizeKJv2` | `832 x 448` | same |
| `img_compression` | 446 `LTXVPreprocess` | `18` | same |
| Node 169 text | 169 `CLIPTextEncode` | paste above | new text |
| schedule text | 1558 `TimestampPromptSchedule` | paste above | new, 21 entries |
| `blend_seconds` / `snap_boundaries` | 1558 | `0.0` / `true` | same |
| `sampler_name` | 154 `KSamplerSelect` | `euler` | same |
| scheduler | 1421 `BasicScheduler` | `linear_quadratic, 8, 1` | same |
| `shift` | 1513 `ModelSamplingSD3` | `13` | same |
| `cfg` | 153 `CFGGuider` | `1.0` | same |

After changing the widget values: **restart ComfyUI or refresh the
workflow** so `AudioLoopController` picks up the new window and
computes stride=7.68s. Verify by checking `AudioLoopPlanner` summary
via the PreviewAny (node 1563) — should show ~21 iterations.

## Negative prompt

Trimmed to 10 precise terms. At CFG=1.0 on the distilled model, each
negative term gets relatively little weight but they compete for the
uncond branch's attention — a long negative dilutes. 10 focused terms
means each one actually pushes away from something we care about.

Paste into Node 507 `CLIPTextEncode`:

```
still image with no motion, deformed facial features, extra limbs, disfigured hands, duplicate character, slow motion, frozen pose, floating pose, dagger floating free of hand, deformed creature
```

### What each term targets

| Term | Failure mode it suppresses |
|---|---|
| `still image with no motion` | Model freezing on action frames |
| `deformed facial features`, `extra limbs`, `disfigured hands` | Standard anatomy — required for close-ups |
| `duplicate character` | Ghosting / doubled body across iteration boundaries |
| `slow motion`, `frozen pose` | **Primary failure from the last run** — Hollywood-slow-mo on chorus peaks |
| `floating pose`, `dagger floating free of hand` | Climbing/leaping physics — keeps her anchored to whatever she's gripping |
| `deformed creature` | Covers dragon-anatomy problems in iters 15-19 generically |

### Dropped from v4's 21-term negative

- `subtitles` — doesn't happen, wastes a slot
- `twin`, `clone` — redundant with `duplicate character`
- `motion blur freeze`, `stopped action`, `held pose` — all overlap with `slow motion` / `frozen pose`
- `cartoonish proportions` — init image already anchors realism strongly
- `multiple heads`, `disfigured wings`, `unnatural anatomy`, `incomplete creature body`, `misshapen dragon` — all subsumed by the single term `deformed creature`

### If you hit specific failures, add back

- Dragon looks particularly bad → add `misshapen wings`
- Still getting slow-mo → add `bullet time`
- Hands morphing between iterations → the term `disfigured hands` is already there; real fix is AdaIN factor or overlap_seconds, not the negative

## Observations

- **This is the architectural variant most likely to feel like
  action cinema.** 21 cuts over 2:34 = ~7.3s per cut. Action films
  average 2-4s; we're still slower than that, but 2.3× closer than
  v4's 17s cuts.
- **Per-iter buildup ~7s is genuinely tight.** The model has
  ~7 seconds to render one coherent action beat before it cuts.
  This matches how rapid-fire action IS edited in practice: each
  shot shows one discrete thing, no time for the thing to evolve.
  Short prompt = short beat = tight cut.
- **Longest subject-anchor repetition in the repo.** The byte-exact
  identity phrase appears 21 times across the schedule. This is
  the most aggressive application of the R3 rule we've done.
  Should maximally anchor identity across the cut-heavy sequence.
- **Dragon remains OUT-OF-DISTRIBUTION** (same caveat as v4). With
  21 iterations + 7.68s stride, the dragon has to render in iters
  14-20 (6 consecutive iterations). If the model fails to form it
  cleanly, you get 6 iterations of "something vague in the sky"
  rather than a clean reveal. Structural fix remains
  `_latent_keyframe.json` + a dragon-anchor image.
- **Test against v4 (9-iter) to measure cut-density vs. quality.**
  v4 is slower-edit but each iteration gets 17.92s to develop; v5
  is faster-edit but each iteration only gets 7.68s. If v5's
  individual iterations read as "half-finished", the cut density
  cost more than it bought. If v5 reads as "proper action cinema",
  the architectural change is a permanent win for this use case.
- **For extreme cut density** (~15 iterations for this song), go
  to `window_seconds=4.94, length=123`, stride=4.68s, giving ~33
  iterations. That's closer to actual action-film edit rhythm, but
  at 4.68s per iteration the model has very little time to
  establish any continuous motion — likely output looks like
  repeated still-frames with minimal animation. Not recommended
  without empirical validation.
