Last updated: 2026-04-25

# LTX 2.3 Audio-Loop Prompt Guide

Canonical guide for writing prompt schedules for the audio-conditioned
video loop workflow. Covers frozen-audio architecture, what to strip
from prompts, what must stay, camera-motion keywords, negative-prompt
templates, the schedule format, and the seven-step authoring process.

For full documented runs see `docs/examples/` (music, action, comedy
case studies).

## Contents

1. [When this guide applies](#1-when-this-guide-applies)
2. [Core principle: schedule entries are a delta layer](#2-core-principle-schedule-entries-are-a-delta-layer)
3. [What to strip](#3-what-to-strip)
4. [What MUST stay (carve-outs)](#4-what-must-stay-carve-outs)
5. [Hypotheses under review](#5-hypotheses-under-review) (`Cut to`, context-frames)
6. [Shot scale: when wide and dolly-out are unsafe](#6-shot-scale-when-wide-and-dolly-out-are-unsafe)
7. [Camera motion keywords (canonical list)](#7-camera-motion-keywords-canonical-list)
8. [Negative prompts](#8-negative-prompts)
9. [Schedule format](#9-schedule-format)
10. [Variation patterns](#10-variation-patterns)
11. [Multi-character scenes](#11-multi-character-scenes-preventing-duplicates)
12. [Worked examples (case study pointers)](#12-worked-examples-case-study-pointers)
13. [Process](#13-process)

## 1. When this guide applies

This guide is for our shipped workflow:

- **Frozen audio** — `noise_mask=0` on the audio latent (via
  `SolidMask` + `SetLatentNoiseMask`). The model is given audio as
  fixed context, not asked to generate it.
- **Image init** — single init image is encoded once via `LTXVImgToVideoInplaceKJ`
  and locked into frame 0. Each loop iteration re-uses the encoded init via
  `LTXVAddLatentGuide latent_idx=-1` (frame BEFORE the window).
- **Iteration anchor** — previous iteration's latent overlap is fed forward via
  `LatentContextExtract` so each iteration continues from prior latents.

If you are working with audio-generating LTX-2 variants, T2V without a
fixed track, or dialogue scenes where the audio is part of the
generation target, see `docs/analysis/audio_in_prompt_research.md` —
the rules in that doc are different.

## 2. Core principle: schedule entries are a delta layer

Three things commit content for you, before the schedule prompt enters
the picture:

1. **Init image** commits style, palette, setting, lighting baseline,
   subject identity.
2. **Frozen audio latent** commits rhythm, timbre, voice timing,
   instrumentation, on-track ambient sound.
3. **Negative prompt** commits what to suppress.

The schedule prompt is what's left over: per-iteration *changes* in
framing, body language, action, lighting *delta*, diegetic sound that
isn't on the audio track. **Re-describing what the init or audio
already commits makes the text-conditioning branch fight against the
image- and audio-conditioning branches.** With the distilled CFG=1
schedule that's not a fair fight — the text usually loses, but the
fight produces visible drift, jitter, or over-cranked beats.

Less descriptive is better. Strip aggressively.

## 3. What to strip

### 3.1 Music and instrumentation descriptors

The frozen audio latent carries the music. Re-describing it in text
double-signals the same beats and over-cranks visual intensity at peaks.

Strip phrases like:
- `staccato strings entering`, `snare firing`, `brass downbeat`,
  `chorus peaking`, `the bass drops`, `orchestra rebuilding`,
  `final chord sustaining`
- `Music swells`, `the beat kicks in`, `synths build`,
  `drums punch through`
- `Brisk rhythmic delivery`, `Voice carries through the room`,
  `Faint distant hum` (when "the hum" is room tone the track also has)
- `Faint thunder building` / `Low distant rumble` *if* the audio has
  thunder/storm content (e.g. a thunderstorm SFX track). If the audio
  doesn't have those sounds and the storm is purely visual, those
  references are diegetic and OK — see §4.4.
- `Room tone of crackling air`, `Room tone settles`, `Ambient hum
  blending with her singing`
- `The sound fades quietly` on outros — let the audio fade do this; don't
  describe it.

Source: `docs/examples/action_prompt6.md` documented the v5→v6 strip
diff line-by-line on 2026-04-20. That's the canonical evidence.
`docs/analysis/audio_in_prompt_research.md:10-30` explains the
mechanism.

### 3.2 Init-committed scene words

The init image commits style/palette/setting/lighting baseline.
Re-describing each iteration wastes tokens and invites the text branch
to re-anchor what the image already committed.

Strip phrases like:
- `painterly rendering`, `illustrated film feel`, `cinematic realism`
  (repeated every entry)
- `dimly lit basement workshop`, `inside a dimly lit basement` (when
  the basement is the init's setting and stays put across all entries)
- `Warm stage wash` repeated every entry
- `Christmas lights strung across exposed beams` repeated every entry
- Specific palette descriptors that don't change (`amber lamps glowing`,
  `concrete walls`, `red phone on the table`)

Keep the *first* entry / Node 169 prompt loaded with this scene-setting
language — the initial render generation has no prior context to lean
on. After that, only re-introduce a scene word if it's varying (e.g.
`light shifts warmer, golden tones from the TV` is a delta, fine).

### 3.3 Vocal-delivery ornaments

The frozen vocal performance commits the singer's delivery. Re-describing
it in text double-signals.

Strip:
- `voices already carrying the opening verse`, `voice peaking on the
  chorus`, `voices softening and trailing off`
- `singing with quiet conviction → with steady voice → with full power`
  if it tracks the audio's actual intensity arc (the audio already
  drives this)
- `breathing hard between lines`, `voice cracking with emotion`

Keep delivery descriptions that name **physical** facets of performance
the audio doesn't carry: `mouth open and shaping the words`, `eyes
wide`, `brow furrowing slightly`, `chin lifting`, `head turning a
fraction toward the kitten`. Those describe visuals, not audio.

## 4. What MUST stay (carve-outs)

These are non-negotiable. Stripping them breaks load-bearing
mechanisms.

### 4.1 The verb

- **Vocal tracks:** every entry must contain `is singing` or
  `are singing together` (multi-subject). LTX 2.3's audio-video
  cross-attention binds lip sync to the singing verb. Generic
  `performing` / `vocalizing` / `delivering` *kills* the lip-sync
  signal. The auto-generator (`scripts/analyze_audio_features.py`)
  enforces this structurally.
- **Instrumental / action tracks:** drop the singing verb requirement;
  use **concrete action verbs** in every entry (`slams`, `lunges`,
  `whips`, `drives`, `vaults`, `rips`, `plunges`). The same
  cross-attention that handles lip sync handles action-to-audio
  binding for orchestral / impact-event tracks.
- **Comedy / dialogue:** use `is delivering the punchline`,
  `is mid-bit`, `is leaning into the mic`, `is telling a joke`.
  Concrete dialogue verbs.

### 4.2 Subject byte-exact across every entry

Same string. No shortening. No paraphrasing. If your subject is
`a warrior woman with a long silver-blonde braid and her orange tabby
cat perched on her shoulder`, every single entry should contain that
exact string. Drift from byte-exact is the #1 cause of cross-iteration
identity instability.

### 4.3 Canonical camera-motion phrasings only

See §7 for the full list. Off-list phrasings (`slow dolly in`,
`slight handheld sway`, `anamorphic lens flare`, `whip pan`) don't
reliably translate to LTX's camera-conditioning pathway.

### 4.4 Diegetic ambient sound NOT on the audio track

If the audio is purely orchestral and the visual scene includes a
storm, `wind roaring through the beams` and `thunder rumble` are
**describing visual elements** — those phrases anchor what the
*image* should show, not what the *audio* carries. Keep them.

The test: would removing this phrase change what the image should
look like? If yes, keep it. If it only describes what the audio
already plays, strip it.

## 5. Hypotheses under review

These are conventions we currently follow but haven't validated with a
clean A/B. They may be wrong. Treat as defaults to use, then test.

### 5.1 `Cut to ...` at iteration boundaries

**Current convention** (per `docs/examples/prompt_comedy4.md` v4
finding, 2026-04-15): every entry after the first leads with
`Cut to a [shot size], [camera motion]. ...`. Rationale: iteration
boundaries are inherent visual discontinuities (independent sampler
passes), and naming them as cuts re-frames the seam as an intentional
edit rather than as a technical artifact.

**The concern:** the loop architecture continues prior latents
(`LatentContextExtract` → ~1s of latent overlap; `LTXVAddLatentGuide
latent_idx=-1` re-anchors the init). Semantically, each iteration IS a
continuation of the previous one, not a fresh video. Telling LTX
"Cut to" may be signaling "start a new video," which is the opposite
of what the wiring is set up for. The boundary marker may be
*creating* the discontinuity it claims to disguise.

**Status:** kept in current case studies because removing it without
running the A/B would erase the v4 finding. Pulled to a follow-up:
exp_2026-MM-DD_cut_to_ablation.md will run the same schedule with vs
without `Cut to ...` prefixes at fixed seed and compare iteration
coherence.

**If the A/B confirms `Cut to` hurts continuity**, the alternatives
to test are: omit the boundary marker entirely (just `In a [shot]...`),
or `Continue: ...` / `Holding on ...` for narrative continuity, or
`Same scene, [shot] ...` for explicit no-cut.

### 5.2 Context-frames width

Currently `overlap_latent_frames=4` (~1s at 25fps). Increasing to 8 or
12 would feed more prior context into each iteration's denoising.
Plausibly reduces the perceived "fresh start" feeling at boundaries
that we currently mask with `Cut to`. Pairs cleanly with the §5.1 A/B
— run as a 2×2 cell matrix.

## 6. Shot scale: when wide and dolly-out are unsafe

Two regimes; pick from the right one.

### 6.1 Face-driven content (music videos, comedy, dialogue)

**No wide shots. No dolly-out anywhere — including outros.**

Mechanism: face shrinks → fewer mouth pixels → audio-video
cross-attention loses the lip-sync signal. Mid-iteration dolly-out
shrinks the face over an ~18s sampler pass; identity destabilizes.
Wide shots that establish setting also shrink the face.

Held close-up + audio fade is the safer outro. The viewer reads "audio
fading on a sustained close-up" as the ending; you don't need camera
motion to signal it.

Source: `docs/examples/prompt_comedy4.md:39-46` ("Wide shots that
shrink the face: fewer mouth pixels → worse lip sync. Already learned
in v2/v3."), `docs/examples/music_prompt1.md:34-39` ("CLAUDE.md's R7
permits dolly out on the final OUTRO entry, but in practice it
shrinks the face over an 18s sampler pass and LTX's cross-attention
loses the mouth signal → face morphs.").

### 6.2 Action / instrumental / no-lip-sync

Wide shots and dolly-out are fine — even encouraged when serving the
action. Examples from `action_prompt6.md`:
- 0:48-0:56 wide shot for "slams both daggers home mid-leap"
- 2:00-2:08 wide shot for "stands braced at the summit"
- 2:16-2:24 wide shot for "leaps from the edge with a dagger raised"

Lip sync isn't the binding mechanism here; visual impact frames bind
to orchestral hits. Composition can prioritize storytelling.

## 7. Camera motion keywords (canonical list)

Use these phrasings exactly. Off-list variants don't translate
reliably to LTX's camera conditioning.

| Keyword | Description |
|---------|-------------|
| `static camera, locked off shot` | No camera movement |
| `dolly in, camera pushing forward` | Smooth forward movement |
| `dolly left, camera tracking left` | Lateral left movement |
| `dolly right, camera tracking right` | Lateral right movement |
| `jib up, camera rising up` | Upward crane movement |
| `jib down, camera lowering down` | Downward crane movement |
| `focus shift, rack focus` | Changing focal point |
| `dolly out, camera pulling back` | Backward movement. **Only safe for action / instrumental content (§6.2). Forbidden for music videos and comedy.** |

Don't use: `slow dolly in`, `slight handheld sway`, `anamorphic lens
flare`, `whip pan`, `crane shot` (without `up`/`down`),
`tracking shot` (without `left`/`right`).

## 8. Negative prompts

Focused, ~10 terms. At CFG=1.0 each negative term gets relatively
little weight, but they compete for the uncond branch's attention — a
long negative dilutes. Short = each term carries more.

**Don't stack the old 30-term negatives.** `blurry, out of focus,
overexposed, low contrast, washed out colors, excessive noise, grainy
texture, poor lighting, flickering, motion blur, distorted
proportions, unnatural skin tones, asymmetrical face, missing facial
features, inconsistent perspective, camera shake, incorrect depth of
field, face swap, merged faces, wrong number of people, third person
appearing, ...` — the marginal benefit at CFG=1 is near zero, and the
collective weight on terms that matter drops.

Target the specific failure modes you actually see in your outputs.

### 8.1 Music / vocal

```
still image with no motion, deformed facial features, extra limbs, disfigured hands, duplicate character, subtitles, twin, clone
```

### 8.2 Instrumental / action

```
still image with no motion, deformed facial features, extra limbs, disfigured hands, duplicate character, slow motion, frozen pose, floating pose
```

Add `dagger floating free of hand` (or analogue) when the action
involves a held weapon/tool that's slipping in your outputs.

Add `deformed creature` when the init has dragons or non-human
creatures the model isn't fully committing.

### 8.3 Comedy / dialogue

```
still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone
```

If your "Cut to" entries are producing literal hard splices where
unwanted: add `scene cut, jump cut, blurry transition`. Usually
unnecessary — the iteration hand-off already functions as the cut.

### 8.4 What NOT to add

- Pre-emptive coverage of failure modes you haven't seen
- More than ~12 terms
- Quality descriptors (`blurry`, `low quality`, `washed out`) — these
  almost never help at CFG=1
- Style descriptors (`cartoonish`, `unrealistic`) — fight what your
  init image committed

## 9. Schedule format

One prompt for the entire song (simplest):

```
0:00+: Style: cinematic. A woman in her 30s with dark hair is singing passionately alone in a dimly lit basement workshop. Strings of colorful, mismatched Christmas lights provide a warm glow against damp stone walls.
```

Three sections with shot variation (keep core subject consistent):

```
0:00-0:38: In a medium close-up, a woman in her 30s with dark hair is singing passionately, static camera, locked off shot. Christmas lights cast warm reflections on her face.
0:38-1:15: A woman in her 30s with dark hair is singing with building energy, dolly in, camera pushing forward.
1:15+: In an extreme close-up, a woman in her 30s with dark hair is singing softly, static camera, locked off shot. Focus on her face and hands.
```

**Format rules:**
- Timestamps: `M:SS`, `M:SS.ss`, or bare seconds (`38.5`)
- Ranges: `start-end` (inclusive) or `start+` (from here onward)
- Last match wins if ranges overlap
- Fallback: last entry used if nothing matches

**Critical:** Node 169 prompt MUST match the schedule's first entry,
byte-for-byte. Node 169 generates the initial ~20 seconds; the
schedule's 0:00 entry controls the first loop extension. If they
differ, there's a visual discontinuity at the 20s boundary where the
conditioning shifts. The auto-generator
(`scripts/analyze_audio_features.py`) guarantees byte-exact equality
via `_build_prompt_for_section` + shared `_prepare_sections`.

## 10. Variation patterns

### Pattern A: framing only (safest)

Every entry has identical subject and action. Only the shot type changes.
No audio descriptions — the model conditions on the actual audio waveform.

```
0:00-0:40: In a medium shot, [subjects] are singing together.
0:40-1:20: [subjects] are singing together, static camera, locked off shot.
1:20-2:00: In a close-up, [subjects] are singing together, focus shift.
2:00+: [subjects] are singing together, static camera.
```

### Pattern B: framing + body language (moderate)

Shot type changes plus performance posture described visually
(body language, not voice).

```
0:00-0:40: In a medium shot, [subjects] are singing together.
0:40-1:00: [subjects] are singing together, leaning forward, animated, static camera.
1:00-1:40: In a close-up, [subjects] are singing together, focus shift.
1:40-2:00: [subjects] are singing together, still.
2:00+: [subjects] are singing together, relaxing back, static camera.
```

### Pattern C: framing + body language + lighting delta (most variation)

Everything above plus lighting *changes* (deltas only — don't re-describe
the baseline lighting).

```
0:00-0:40: In a medium shot, [subjects] are singing together. Soft ambient lighting.
0:40-1:00: [subjects] are singing together, animated, static camera. Light shifts warmer, golden tones.
1:00-1:40: In a close-up, [subjects] are singing together, focus shift. Warm light on faces, deep shadows.
1:40-2:00: [subjects] are singing together. Light dims, only faint glow remains.
2:00+: [subjects] are singing together, static camera. Room grows dim.
```

Each lighting line names a *change*, not the baseline state. "Soft
ambient lighting" appears once in the opener; afterwards only deltas
("Light shifts warmer", "Light dims", "Room grows dim").

## 11. Multi-character scenes (preventing duplicates)

LTX 2.3 tends to duplicate characters, especially distinct ones (e.g., a
cartoon duck gets cloned). The model interprets vague nouns like
`characters` or `figures` as an invitation to add more.

**Rules:**
- Name each character by position and visual trait, not by count alone.
  - BAD: `three characters talking on a beach`
  - GOOD: `The duck on the left, the tall creature in the center, and the small blue figure on the right are talking together`
- Add `No other characters` or `No one else appears` as an explicit constraint.
- Use `static camera, locked off shot` to prevent panning that reveals new areas
  where the model might spawn extras.
- Never use generic group words: `crowd`, `group`, `people`, `others`.
- Position-anchor each character: `on the left`, `in the center`, `on the right`
  tells the model the composition is fixed.

**Negative prompt additions for multi-character:**

```
duplicate character, clone, twin, copy, mirror image, extra characters
```

(stay focused — don't stack the full 30-term version)

**Example (3 cartoon characters talking in water):**

```
The duck on the left, the tall creature in the center, and the small blue
figure on the right are talking together. No other characters. Mouth movements
and subtle gestures. Water ripples. Static camera, locked off shot.
```

For conversation audio (non-music): the model maps speech audio to
mouth movements via cross-attention but won't perfectly assign
voice-to-character. It just ensures mouths move when audio is active.
Vocal separation (MelBandRoFormer) is less useful for multi-speaker
dialogue than for singing.

## 12. Worked examples (case study pointers)

Full documented runs with widget tables, observations, and results
live in `docs/examples/`. Pick the closest match to your use case and
copy:

| Scenario | Canonical | Why pick this |
|---|---|---|
| Vocal music video, cinematic init | `music_prompt3.md` | Simple structure, canonical camera, cinematic-realism style match. The clean baseline. |
| Vocal music video, illustrated init | `music_prompt2.md` | Same structure as music_prompt3 with `Style: illustrated.` and animated-pool vocabulary. |
| Instrumental / action, normal pacing | `action_prompt1.md` | 9-iteration narrative arc; full-length window. |
| Instrumental / action, rapid cuts | `action_prompt6.md` | 20-iteration rapid-cut grid (halved window). The frozen-audio strip rule was established here (v6 vs v5 diff). |
| Standup comedy / dialogue | `prompt_comedy4.md` | The "Cut to" finding (under review per §5.1), specific crowd-member beats, no-wide-shot rule. |
| Unusual-character init (out-of-distribution) | `prompt_comedy5.md` | How to rewrite the subject anchor when the init is visually atypical. |

`docs/examples/README.md` has the full evolution arc (comedy 1→4,
music 1→3, action 1→6) — useful for understanding why current rules
exist.

## 13. Process

### Step 1: Analyze the audio

Recommended path — auto-generated prompt templates with the
music-aware analyzer:

```bash
uv run --group analysis python scripts/analyze_audio_features.py path/to/song.wav \
  --subject "a woman in her 30s with dark hair singing in a basement workshop" \
  --trim 10
```

Outputs `TimestampPromptSchedule` entries with section-appropriate
framing per the rules in this guide. Every entry contains `is singing`
(enforced at code + test level). Use `--scene-diversity <tier><sub>`
(default `2a`) and `--montage` to dial ambition. `--trim N` matches
your node 567 `start_index`.

Full reference (basic ffmpeg variant, all flags, scene-diversity tiers,
JSON export): see `audio_analysis_guide.md`.

### Step 2: Vocal separation

Vocal separation is done by MelBandRoFormer **inside the ComfyUI workflow**
(nodes 568/569), not by a CLI script. The workflow already separates vocals
from instruments before encoding audio to latents.

MelBandRoFormer separates vocals from instruments but does NOT distinguish
male from female vocals. For duets, use the `AudioPitchDetect` runtime node
which detects male (F0 < 160 Hz) vs female (F0 > 160 Hz) vocal ranges
per iteration using the separated vocals output.

### Step 3: Study the init image

Identify, then **don't re-describe in prompts** (see §3.2):
- Who is in the frame (number of people, positions)
- What they're wearing (use for minimal identification in the byte-exact
  subject anchor only)
- The setting (image handles it)
- Lighting baseline (vary as a delta in the schedule, don't re-state)

### Step 4: Write prompts following the rules

Apply this guide:
- §2 — schedule entries are a delta layer
- §3 — strip music descriptors, init-committed scene, vocal-delivery ornaments
- §4 — keep the verb, byte-exact subject, canonical camera, diegetic-not-on-track sounds
- §6 — choose your shot-scale rules based on content type
- §7 — canonical camera-motion only
- §10 — pick a variation pattern (A safest, C most variation)

**Critical:** Node 169 prompt MUST match the schedule's first entry,
byte-for-byte. The auto-generator guarantees this; if you write
manually, copy the line.

### Step 5: Set workflow values

**Starting values (adjust per results):**

| Setting | Start with | Range | Notes |
|---------|-----------|-------|-------|
| `overlap_seconds` | **2.0** | 1.0-3.0 | Start at 2.0. Increase to 3.0 if jitter between iterations. |
| `blend_seconds` | **0.0** | 0 or ≥ stride_seconds | Start at 0 (hard switch). For cross-fade, use **≥ stride_seconds** (typically ~17-19s). Values between 0 and stride_seconds are auto-clamped to stride_seconds with a one-time warning. |
| `snap_boundaries` | **True** | True/False | Leave on. Rounds schedule boundaries to the iteration grid so every iteration runs on one pure prompt. |
| node 567 `start_index` | **10** | 0-30 | Seconds to skip. Match to your song's instrumental intro length. |
| node 169 prompt | — | — | Must match the 0:00 schedule entry byte-for-byte. |

**Fixed values (don't change):**

| Setting | Value | Why |
|---------|-------|-----|
| `window_seconds` | 19.88 | LTX 2.3 default window |
| fps | 25 | LTX 2.3 default framerate |
| TensorLoopOpen iterations | 50 | Safety cap. Auto-stop handles actual termination. |
| MelBand 568/569 | per workflow | Default may be bypassed; re-enable via apply script if needed for lip sync. |
| `trim_to_audio` | true | Clip output to audio length |
| `CFG` (node 644 inside subgraph) | 1.0 | LTX 2.3 is distilled. NAG handles guidance. |

### Step 6: Use an LLM to generate prompts (optional)

Add `--image-desc "..." -j analysis.json` to Step 1's command and paste
the JSON into Claude/Gemini — the embedded system prompt contains all
the rules in this guide. Full walkthrough with examples:
`docs/guides/prompt_workflow_end_to_end.md`.

### Step 7: Run and iterate

- **First run:** no prompt schedule (static prompt, blend_seconds=0).
  Verify the base loop works and consistency holds.
- **Second run:** add prompt schedule with conservative changes
  (Pattern A — same subject, vary only framing). Keep blend_seconds=0.
- **Third run:** add body-language / lighting deltas (Pattern B/C). If
  you see a visible seam at prompt transitions, enable cross-fade by
  setting `blend_seconds` to `stride_seconds` or higher (typically
  ~18-20s). Values below stride are auto-clamped.

If results drift across iterations:
- Bump `overlap_seconds` (1.0 → 2.0 → 3.0). More context, smoother
  hand-offs.
- Verify your subject string is byte-exact across every entry.
- Verify your camera-motion phrases are from the canonical list (§7).
- For face-driven content: confirm no wide shots, no dolly-out (§6.1).

If iteration boundaries feel like cuts: that's the §5.1 "Cut to"
hypothesis at work. Try omitting the `Cut to ...` prefixes as an A/B
and compare.

## See also

- `docs/analysis/audio_in_prompt_research.md` — when audio is being
  *generated* (not frozen), prompt rules differ; that doc covers
  community practices for those workflows.
- `docs/guides/audio_analysis_guide.md` — full audio-analysis script
  reference (BPM, key, F0, structure, scene-diversity tiers).
- `docs/guides/prompt_workflow_end_to_end.md` — LLM-assisted schedule
  generation walkthrough.
- `docs/guides/debugging_guide.md` — symptom-first recipes for common
  failure modes.
- `docs/examples/` — full case studies (music, action, comedy).
- `docs/reference/ltx23_prompt_system_prompts.md` — Lightricks's
  official LTX 2.3 i2v/t2v system prompts these rules derive from.
