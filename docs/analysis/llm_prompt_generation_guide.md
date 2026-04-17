Last updated: 2026-04-17

# LLM Prompt Generation Guide

How to use the audio analysis JSON export with an LLM (Claude, Gemini, etc.)
to generate a complete TimestampPromptSchedule and node 169 initial prompt
for the AudioLoopHelper music video workflow.

## Our workflow

- **Image-to-video (i2v)**: an init_image provides the first frame
- **Audio conditioning**: a full audio track (song/dialogue) is frozen at noise=0
- **The model generates only video**: audio cross-attention drives lip sync and rhythm
- **Loop**: video is generated in ~20-second windows with overlapping iterations
- **Two prompt locations**: node 169 (initial ~20s) + TimestampPromptSchedule (loop iterations)

## Step 1: Run the analysis

```bash
uv run --group analysis python scripts/analyze_audio_features.py your_song.wav \
  --trim 10 \
  --subject "a woman in her 30s with dark hair singing in a basement workshop" \
  --image-desc "Woman sitting on wooden chair, dark hair pulled back, vintage dress, dimly lit basement, Christmas lights on exposed beams, concrete walls" \
  --scene-diversity 2a \
  --window 19.88 \
  --overlap 2.0 \
  -j analysis.json
```

`--scene-diversity <tier><sub>` picks the ambition level + flavor
(default `2a`; see `docs/audio_analysis_guide.md` for the full 6-tier
taxonomy). `--montage` is an orthogonal flag for Arcane-style pacing.

This produces a JSON file containing:
- Audio analysis (BPM, key, sections, vocal F0)
- Workflow timing context (trim, window, stride, what node 169 covers,
  `scene_diversity`, `scene_diversity_tier_name`,
  `scene_diversity_mood_bundle`, `montage`)
- A complete LLM system prompt with all prompt engineering rules,
  tier semantics, and an INFERENCE block describing what the init
  image encodes vs what the schedule should drive

## Step 2: Paste into your LLM

### System prompt

Copy the `llm_system_prompt` field from the JSON and use it as the LLM's
system prompt (or prepend it to your message). This contains all the rules
for writing LTX 2.3 i2v prompts with frozen audio conditioning.

### User prompt

Paste the rest of the JSON as context, then add your creative direction:

```
Here is the audio analysis for my music video:

{paste the full JSON here, minus the llm_system_prompt field}

Init image: {describe what's in your first frame, or paste VLM output}

Creative direction:
- Song mood: melancholic indie folk
- I want subtle variations between verse and chorus
- Use Pattern B (framing + energy) -- no lighting changes yet
- Keep camera mostly static, use focus shift for chorus close-ups
- The woman is the only person in the frame

Generate the node_169_prompt and schedule.
```

### What the LLM returns

The LLM should output:

```
node_169_prompt: Style: cinematic. A woman in her 30s with dark hair pulled back, wearing a vintage dress, is singing softly. She sits still, hands resting in her lap, chin slightly lowered. Faint hum of Christmas lights, soft room tone from the concrete walls.

schedule:
0:00-0:42: Style: cinematic. In a medium shot, a woman in her 30s with dark hair pulled back, wearing a vintage dress, is singing softly. Hands resting in lap, slight sway. Faint hum of Christmas lights, soft room tone.
0:42-1:20: Style: cinematic. A woman in her 30s with dark hair pulled back, wearing a vintage dress, is singing with steady energy, leaning slightly forward, static camera, locked off shot. Soft room tone.
1:20-2:00: Style: cinematic. In a close-up, a woman in her 30s with dark hair pulled back, wearing a vintage dress, is singing with full expression, eyes half-closed, focus shift. Faint ambient hum.
2:00-2:30: Style: cinematic. A woman in her 30s with dark hair pulled back, wearing a vintage dress, is singing quietly, shoulders relaxed, still. Soft room tone settling.
2:30+: Style: cinematic. In a wide shot, dolly out, camera pulling back, a woman in her 30s with dark hair pulled back, wearing a vintage dress, is singing the final notes, growing still. Room tone fades.
```

**Copy node_169_prompt into node 169. Copy the schedule block into TimestampPromptSchedule (node 1558).**

## Step 3: Use a VLM for init_image description

For best subject anchoring, pass your init_image through a vision model to
extract both `--subject` and `--image-desc`. See `prompt_workflow_end_to_end.md`
Step 2 for complete VLM prompts (single-person and multi-person variants).

Use the VLM output as the `--image-desc` and `--subject` CLI args and in
your LLM user prompt. This gives the LLM precise visual traits to repeat
in every schedule entry, preserving character identity across the full video.

## The system prompt structure

The embedded `llm_system_prompt` is organized into three parts the LLM
reads as one document:

### INFERENCE block (what the image already commits to)

The init image anchors **style family** (live-action / animated / comic /
graphic-novel / 3D-render / stop-motion), **color palette**, **setting**
(indoor/outdoor, urban/natural, wardrobe, era), and **subject appearance
and count**. The LLM is told: do **NOT** re-describe these. Re-describing
invites the text conditioning to fight what the image already commits to
(e.g. writing "comic-book style" on a photorealistic image forces a
tug-of-war).

What the schedule **should** drive: camera framing / motion, body
beats, lighting *shifts over time* (not palette restatement), scene
cuts, emotional arc. Schedule entries are a **delta layer** over the
visual baseline the image provides.

Style-appropriate beat pools: animated/comic → speed lines, panel
transitions, supersaturation, impact frames. Live-action → rack focus,
practical lighting shifts, handheld / dolly moves. Infer from the image
which pool applies.

### HARD RULES R1-R8

1. **R1 — Singing verb is mandatory.** Every entry contains "is
   singing..." (single) or "are singing together..." (multi). Drives LTX
   2.3's audio-video cross-attention for lip sync. No "performing",
   "vocalizing", generic verbs. For instrumental scenes use "is playing
   <instrument>".
2. **R2 — Node 169 = first schedule entry, byte-exact.** The LLM MUST
   copy the first schedule entry verbatim into `node_169_prompt`. Any
   drift causes a visible seam at the ~20s loop-entry boundary.
3. **R3 — Identical subject across all entries.** Only vary framing,
   camera, lighting, body language, performance beats. Never
   re-describe the environment (image sets it).
4. **R4 — Multi-person position-anchoring.** Describe each person by
   position + wardrobe inside the subject string ("the man on the left
   in the dark jacket..."). No bare "crowd", "group", "duo".
5. **R5 — No meta-language.** No "The scene opens with...", "Cut
   to...", "camera shows...". Every entry begins "Style: cinematic."
   and moves straight to subject + action.
6. **R6 — Audio direction.** Do NOT describe the song (voice surging,
   music swelling — the model hears it). DO describe ambient/diegetic
   sounds not in the audio track (room tone, fluorescent hum). Vocal
   delivery qualifiers ("in a low gravelly voice") are encouraged.
7. **R7 — Camera motion.** Default "static camera, locked off shot".
   Available: dolly in, dolly left/right, jib up/down, focus shift.
   AVOID dolly out (breaks limbs/faces) except for the final OUTRO.
8. **R8 — One paragraph, ~200 words max.** No markdown or bullets.
   Present progressive throughout.

### AMBITION TIERS + MONTAGE

`workflow_context.scene_diversity` (e.g. `"3b"`) tells the LLM which
ambition ceiling to target:

- Tier 1 performance_live → single-camera concert feel
- Tier 2 performance_dynamic → camera + body beats rotate (DEFAULT)
- Tier 3 cinematic → + environmental storytelling / scene shifts
- Tier 4 narrative → + physical-action arc / loose story
- Tier 5 stylized → + genre overlay (noir / surreal / retro)
- Tier 6 avant_garde → non-linear, abstract, performative

Sub-letters (1a/1b/1c, 3a-3d, etc.) add mood bundles — lighting
palette, location keywords, camera-style adjectives.

`workflow_context.montage = true` is orthogonal to tier. When set,
each entry must advance an emotional beat (not merely describe a
scene), use emotional-arc language ("the feeling building", "catharsis
arriving", "release easing into stillness"), and dwell ~12s instead of
~20s. Arcane-style music-drives-narrative pacing.

## Common mistakes

| Mistake | What happens | Fix |
|---------|-------------|-----|
| Re-describing setting in every entry | Model "restarts" the scene, composition jumps | Only describe subject + changes |
| Different subject descriptions per entry | Style drift, identity loss | Copy-paste identical subject block |
| Describing audio dynamics ("voice swells") | Conflicts with frozen audio conditioning | Let the audio latents handle it |
| Dolly out in middle sections | Limbs stretch, faces distort | Use static camera or dolly in |
| Node 169 differs from schedule 0:00 | Visual discontinuity at ~20 seconds | Copy first schedule entry into node 169 |
| Emotional language ("singing passionately") | Model can't interpret internal states | Use physical cues instead |
| Too many short schedule entries | Rapid conditioning switches = drift | Consolidate same-type sections |

## Variation patterns

### Pattern A: Framing only (safest, start here)
- Every entry identical except shot type (medium, close-up, wide)
- No energy or lighting variation
- Best for first test run

### Pattern B: Framing + energy (moderate)
- Shot type + body language cues (leaning forward, relaxed, still)
- Matches song structure energy levels
- Good balance of variation and stability

### Pattern C: Framing + energy + lighting (most variation)
- Everything above + lighting shifts (warm, bright, dim, shadows)
- Most visual interest but highest risk of drift
- Use blend_seconds >= 5.0

See `prompt_creation_guide.md` for full examples of each pattern (6 complete
scenarios across two songs with three variation levels each).

## Timing reference

```
Song: |--skip--|----initial render (node 169)----|--loop iteration 1--|--iteration 2--|...
      0      trim    trim + window_seconds        trim + window + stride
                      (default: ~20s)

Schedule timestamps are in TRIMMED audio space (--trim already subtracted).
Node 169 covers trimmed 0:00 to ~0:20.
Schedule 0:00 entry is a fallback -- loop starts at iteration 1 (~0:18 with overlap=2).
```
