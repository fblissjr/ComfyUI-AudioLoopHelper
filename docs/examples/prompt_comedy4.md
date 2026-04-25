Last updated: 2026-04-25 (vocal-delivery ornaments stripped, off-canon camera moves normalized to canonical phrasings, and final outro changed from wide+dolly-out to close-up+static per the face-driven shot-scale rule in `docs/guides/prompt_creation_guide.md` §6.1. The original v4 final entry used `wide shot, slow dolly out, camera pulling back` per the README rule-7 outro exception in effect at the time; that exception has since been retracted for face-driven content because the shrinking face costs lip-sync cross-attention signal across the ~18s outro window.)

Original date: 2026-04-17

# Standup Comedy — Example Prompt Schedule (v4)

Fourth iteration on the standup use case. v3 was deliberately
conservative (all static camera, all medium shot, byte-exact
everything) to eliminate the lip-sync + jitter failures from v1/v2.
With decoder tile seams structurally fixed by DR1 and the blend
invariant gone, we now have headroom to push visual variety without
risking the quality floor v3 established.

v4 goal: **feel like a comedy special edit**, not a single static
lecture camera. Real TV specials cut between framing, push in on
punchlines, linger in close on reactions, pull back to land an outro.
We can replicate much of that without breaking lip sync if we respect
what i2v can and can't do.

> **Prose-vs-schedule note (2026-04-25):** the prose below describes
> the original v4 camera-vocabulary thinking (`slow dolly in`, `slight
> handheld sway`, `slow jib up`). The schedule has been retroactively
> normalized to the canonical-only list per
> `docs/guides/prompt_creation_guide.md` §7 — those off-canon phrasings
> no longer appear in the schedule entries. The prose is preserved as
> the original v4 reasoning.

## What i2v lets us vary — and what it doesn't

**Safe (within LTX's capabilities)**:
- Shot size: medium shot → medium close-up → close-up (via slow
  dolly in). The init image anchors the comedian's position in frame;
  pushing in crops toward the face without moving the camera.
- Body language: gestures, lean-ins, head turns, weight shifts.
- Focal behavior: rack focus (shift focus between comedian and
  foreground crowd member), focus shift within the plane.
- Crowd state: vary what specific people in the image are doing
  (one wiping eyes, one leaning in, one slamming the table laughing).
- Subtle camera texture: slight handheld sway (doesn't rescale the
  face), static locked-off for dialogue beats.
- Cut language at iteration boundaries: since iteration hand-offs
  are inherent visual discontinuities, naming them as "cuts" makes
  them feel intentional rather than accidental.

**NOT safe (fights i2v's image anchor or breaks lip sync)**:
- Angle changes (front → side → low): the init image anchors the
  camera position. Prompts asking for over-the-shoulder or low-angle
  would fight what's in the image → artifacts.
- Wide shots that shrink the face: fewer mouth pixels → worse lip
  sync. Already learned in v2/v3.
- Dolly out mid-iteration: face shrinks over 18s of a single sampler
  pass; cross-attention loses signal. Only OK as the final outro.
- Insert cutaways to entirely different scenes (audience close-ups,
  stage-left wings): i2v generates from the init image; there's no
  other scene to cut to. Would morph the current frame into something
  that half-matches both, looks bad.

## Strategy for "interesting" within those bounds

For each iteration, pick:
1. One shot size (medium / medium close-up / close-up)
2. One camera move (static / slow dolly in / slight handheld / rack
   focus / slow jib up). Never dolly out except in the final outro.
3. One performance verb from the standup pool
4. One body beat (gesturing, leaning, shifting weight, pacing,
   pointing, etc.)
5. One crowd beat (what a specific crowd member is doing)
6. Optional: delivery qualifier (deadpan, rapid, drawn-out)

> **Note (2026-04-25):** the original v4 schedule led each non-first
> entry with `In a [shot size]...` to "hand off the iteration
> boundary as an intentional edit." That convention has been
> retracted per `docs/guides/prompt_creation_guide.md` §5.1
> (Lightricks's official LTX 2.3 system prompt explicitly trains
> the model to treat scene-cut language as a discontinuation
> directive — the opposite of what the loop architecture wants).
> Schedule entries below have been retroactively normalized to
> `In a [shot size]...` continuation framing.

## Inputs

- **Audio**: preprocessed WAV at 184s (untrimmed); adjust schedule
  timestamps if using trim=5.
- **Image**: LIVE STANDUP CHICAGO init image (or similar tight-framed
  standup shot).
- **Subject string (byte-exact, identity-anchoring)**:
  `a male standup comedian in a striped sweater at a stand-up comedy club`
- **Schedule timestamps**: grid-aligned for `overlap_seconds=2`
  (stride=17.92s). If you run at `overlap=3`, see v3 for the
  alternative grid.

## Schedule

```
node_169_prompt: Style: cinematic. In a medium shot, static camera, locked off shot, a male standup comedian in a striped sweater at a stand-up comedy club is pausing for the laugh, mic held close to his chest, slight smile. Warm stage wash. The crowd on the right mid-laugh.

schedule:
0:00-0:17: Style: cinematic. In a medium shot, static camera, locked off shot, a male standup comedian in a striped sweater at a stand-up comedy club is pausing for the laugh, mic held close to his chest, slight smile. Warm stage wash. The crowd on the right mid-laugh.
0:17-0:35: In a medium close-up, static camera, locked off shot. A male standup comedian in a striped sweater at a stand-up comedy club is delivering the setup, raising an eyebrow. The crowd quiet, leaning in.
0:35-0:53: In a close-up, dolly in, camera pushing forward. A male standup comedian in a striped sweater at a stand-up comedy club is telling a joke, eyes wide with conviction, shaking his head slightly.
0:53-1:11: In a close-up, static camera, locked off shot. A male standup comedian in a striped sweater at a stand-up comedy club is delivering the punchline, leaning into the mic, mouth open. One person in the crowd on the right slapping the table laughing.
1:11-1:29: In a medium shot, static camera, locked off shot. A male standup comedian in a striped sweater at a stand-up comedy club is mid-bit, gesturing with his free hand, shifting weight between feet. The crowd watching attentively.
1:29-1:47: In a medium close-up, focus shift, rack focus. A male standup comedian in a striped sweater at a stand-up comedy club is smiling wryly, looking out into the audience, head tilted. A couple of patrons on the right whispering.
1:47-2:05: In a medium shot, static camera, locked off shot. A male standup comedian in a striped sweater at a stand-up comedy club is telling a joke, gesturing sharply with his left hand to emphasize a point, mic slightly lowered. Crowd members shifting in their seats.
2:05-2:23: In a close-up, dolly in, camera pushing forward. A male standup comedian in a striped sweater at a stand-up comedy club is delivering the punchline, leaning back slightly, pointing at a crowd member, eyes narrowed playfully. Someone on the right wiping their eye from laughing.
2:23-2:41: In a medium close-up, static camera, locked off shot. A male standup comedian in a striped sweater at a stand-up comedy club is leaning into the mic, building the final premise. The crowd leaning in, highly attentive.
2:41-2:58: In a close-up, dolly in, camera pushing forward. A male standup comedian in a striped sweater at a stand-up comedy club is delivering the final punchline, smiling wide, mic lowered. The crowd fully laughing, shoulders shaking.
2:58+: In a close-up, static camera, locked off shot. A male standup comedian in a striped sweater at a stand-up comedy club is reacting to the crowd, waving his free hand, slight smile spreading. The crowd animated, some standing, some wiping their eyes.
```

## What makes v4 different from v3

| Aspect | v3 | v4 |
|---|---|---|
| Shot size variety | Medium shot everywhere, one medium close-up | Alternates medium / medium close-up / close-up |
| Camera motion | Static always (except outro) | Static, slow dolly in, slight handheld, rack focus |
| Cut language | Not used | Originally every non-first entry started `"Cut to..."`; **retracted 2026-04-25** per guide §5.1 — schedule above now uses `In a ...` continuation form |
| Body beats | Minimal variation | Each entry has a distinct body beat |
| Crowd beats | Repeated "crowd reacting" | Specific per-entry (table-slap, whispering, weight-shifting, eye-wiping, shoulders-shaking) |
| Per-entry length | ~25 words | ~35-45 words (still short, but richer) |

Same lip-sync-safe bones: subject byte-exact, no wide shots until
outro, no dolly out mid-iteration, no angle changes.

## Workflow widget values (unchanged from v3-post-DR1)

| Widget | Node | Value |
|---|---|---|
| decoder type | 1604, 1597 | `LTXVTiledVAEDecode` (post-DR1) |
| `overlap_seconds` | AudioLoopController | `2.0` |
| `blend_seconds` | 1558 TimestampPromptSchedule | `0.0` |
| `snap_boundaries` | 1558 TimestampPromptSchedule | `true` |
| `sampler_name` | 154 KSamplerSelect | `euler` |
| `start` (outer trim) | 567 TrimAudioDuration | `0` (or `5` if trimming the cold-open) |
| `shift` | 1513 ModelSamplingSD3 | `13` |
| Scheduler | 1421 BasicScheduler | `linear_quadratic, 8, 1` |
| CFG | 153 CFGGuider | `1.0` |

## Negative prompt (current shipped)

```
still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone
```

(Historical: the original v4 negative-prompt note recommended
adding `"scene cut, jump cut, blurry transition"` if `Cut to`
language produced literal hard splices. Now that `Cut to` is
retracted per guide §5.1, this is no longer needed.)

## Observations

- **Why we thought `"Cut to..."` worked for us (2026-04-15, RETRACTED 2026-04-25)**:
  the original argument: iteration hand-offs in the loop architecture
  are visual discontinuities; text saying "cut to" reframes the
  discontinuity as intentional editing rather than a technical seam.
  **What was missed at the time**: (a) the v3→v4 A/B that produced
  this finding also added per-iter camera variety, so the perceived
  improvement was confounded; (b) Lightricks's own LTX 2.3 system
  prompt
  (`docs/reference/ltx23_prompt_system_prompts.md:44, 56, 93`)
  explicitly trains the model to treat scene-cut language as a
  *discontinuation directive*, the opposite of what the latent-side
  continuity mechanisms (`LTXVAddLatentGuide latent_idx=-1` +
  `LatentContextExtract` 1s overlap) are doing. The text branch was
  fighting the latent branch. Retracted; see guide §5.1.
- **Crowd beats matter more than I expected.** The image has ~2-3
  visible crowd members. Naming specific small actions per entry
  ("table-slap", "whispering", "wiping eyes", "shoulders shaking")
  gives LTX concrete animation targets for those specific people.
  Results feel less "a room with generic laughing people" and more
  "a real club with specific people reacting to specific moments."
- **Close-ups on punchlines, medium-close-ups for setups, medium
  shots for mid-bits**: this is the natural comedy-special rhythm.
  Setup → zoom → punchline → hold → pull back. v4 follows it.
- **If this feels too busy**: pull back some of the "Cut to"
  aggression. Let some entries continue the prior shot without a cut
  language, matching the v3 conservative approach for specific beats.
  The template is adjustable per iteration.
- **Future refinement — explicit "master two-shot" or "crowd-side
  angle"**: we're not doing these because they'd fight the init
  image's single-camera front-of-stage composition. If you prep init
  images from multiple angles and use `KeyframeImageSchedule` to swap
  between them per section, the `_keyframe` workflow variant could
  do true multi-angle coverage. That's a setup-time lift, not a
  widget change.
