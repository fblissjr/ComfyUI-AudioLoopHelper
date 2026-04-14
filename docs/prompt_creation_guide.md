Last updated: 2026-04-12

# Music Video Prompt Creation Guide

Step-by-step process for creating prompt schedules for the LTX 2.3
audio-conditioned video loop workflow.

## Process

### 1. Analyze the audio

Two analysis scripts are available:

**Basic (ffmpeg only, no Python deps):**
```bash
uv run scripts/analyze_audio.py path/to/song.wav --trim 10
```
Gives you: energy timeline, song structure, prompt schedule template.

**Music-aware (librosa, requires `uv sync --group analysis`):**
```bash
uv run --group analysis python scripts/analyze_audio_features.py path/to/song.wav --trim 10
```
Gives you: BPM, musical key, chromagram, vocal F0, structure, JSON for LLM.

**Auto-generated prompt templates (recommended):**
```bash
uv run --group analysis python scripts/analyze_audio_features.py path/to/song.wav \
  --subject "a woman in her 30s with dark hair singing in a basement workshop" \
  --trim 10
```
Generates copy-pasteable TimestampPromptSchedule entries with LTX 2.3
camera, lighting, and energy modifiers matched to each song section.
You provide the scene description once; the script wraps it with
section-appropriate framing (close-up for chorus, wide shot for bridge, etc.).

Use `--trim N` to offset timestamps by your node 567 start_index value.

### 2. Vocal separation

Vocal separation is done by MelBandRoFormer **inside the ComfyUI workflow**
(nodes 568/569), not by a CLI script. The workflow already separates vocals
from instruments before encoding audio to latents.

MelBandRoFormer separates vocals from instruments but does NOT distinguish
male from female vocals. For duets, use the AudioPitchDetect runtime node
which detects male (F0 < 160 Hz) vs female (F0 > 160 Hz) vocal ranges
per iteration using the separated vocals output.

### 3. Study the init_image

Identify:
- Who is in the frame (number of people, positions)
- What they're wearing (use for minimal identification in prompts)
- The setting (don't re-describe in prompts -- image handles it)
- Lighting and mood (baseline to vary from)

### 4. Write prompts following i2v rules

See `ltx23_prompt_system_prompts.md` for the official LTX 2.3 i2v and t2v
system prompts these rules derive from.

Key rules for the loop workflow:
- **Describe only changes from the image.** Don't re-describe the setting.
- **Keep the core subject identical in every schedule entry.**
- **Use present-progressive verbs:** "is singing," "are leaning forward."
- **Weave audio descriptions with actions,** not at the end.
- **No meta-language:** No "The scene opens with..."
- **Camera motion only when intended.** Avoid dolly out (breaks limbs/faces).
- **For two people: say "singing together" in every entry.** Let the audio
  conditioning handle who's actually singing. Don't try to direct male vs
  female -- the model figures it out from the audio.

**Critical: node 169 prompt MUST match the schedule's first entry.**
Node 169 generates the initial ~20 seconds. The schedule's 0:00 entry
controls the first loop extension. If they differ, there's a visual
discontinuity at ~20 seconds where the conditioning shifts. Copy the
0:00 schedule entry into node 169 exactly.

### 5. Set workflow values

**Starting values (adjust per results):**

| Setting | Start with | Range | Notes |
|---------|-----------|-------|-------|
| overlap_seconds | **2.0** | 1.0-3.0 | Start at 2.0. Increase to 3.0 if jitter between iterations. |
| blend_seconds | **5.0** | 0-10.0 | Start at 5.0. Increase to 10.0 if style drifts at prompt boundaries. Set 0 to disable blending. |
| node 567 start_index | **10** | 0-30 | Seconds to skip. Match to your song's instrumental intro length. |
| node 169 prompt | — | — | Must closely match the 0:00 schedule entry AND the init_image. |

**Fixed values (don't change):**

| Setting | Value | Why |
|---------|-------|-----|
| window_seconds | 19.88 | LTX 2.3 default window |
| fps | 25 | LTX 2.3 default framerate |
| TensorLoopOpen iterations | 50 | Safety cap. Auto-stop handles actual termination. |
| MelBand 568/569 | Enabled (mode 0) | Vocal separation for lip sync |
| trim_to_audio | true | Clip output to audio length |
| SageAttention 268 | Bypassed (mode 4) | Enable only after quality is stable |
| CFG (node 644 inside subgraph) | 1.0 | LTX 2.3 is distilled. NAG handles guidance. |

### 6. Use an LLM to generate prompts (optional)

Instead of writing prompts manually, export the analysis as JSON and paste
it into an LLM. The JSON includes a system prompt with all the rules above.

```bash
uv run --group analysis python scripts/analyze_audio_features.py path/to/song.wav \
  --trim 10 \
  --subject "a woman singing in a basement workshop" \
  --image-desc "Woman on chair, dark hair, vintage dress, dim basement, Christmas lights" \
  -j analysis.json
```

Then paste the JSON into Claude/Gemini with your creative direction. The LLM
generates both node_169_prompt and the schedule. See
`docs/analysis/llm_prompt_generation_guide.md` for the full workflow with examples.

### 7. Run and iterate

- First run: no prompt schedule (static prompt, blend_seconds=0).
  Verify the base loop works and consistency holds.
- Second run: add prompt schedule with conservative changes
  (same subject, vary only framing). blend_seconds=5.0.
- Third run: add lighting variations. Increase blend_seconds if drift appears.

---

## Prompt Variation Patterns

### Pattern A: Framing only (safest)

Every entry has identical subject and action. Only the shot type changes.
No audio descriptions -- the model conditions on the actual audio waveform.

```
0:00-0:40: In a medium shot, [subjects] are singing together.
0:40-1:20: [subjects] are singing together, static camera, locked off shot.
1:20-2:00: In a close-up, [subjects] are singing together, focus shift.
2:00+: [subjects] are singing together, static camera.
```

### Pattern B: Framing + energy (moderate)

Shot type changes plus energy level described visually (body language, not voice).

```
0:00-0:40: In a medium shot, [subjects] are singing together softly.
0:40-1:00: [subjects] are singing together, leaning forward, animated and expressive, static camera.
1:00-1:40: In a close-up, [subjects] are singing together with full energy, focus shift.
1:40-2:00: [subjects] are singing together quietly, still.
2:00+: [subjects] are singing together, relaxing back, static camera.
```

### Pattern C: Framing + energy + lighting (most variation)

Everything above plus lighting shifts to match mood.

```
0:00-0:40: In a medium shot, [subjects] are singing together. Soft ambient lighting.
0:40-1:00: [subjects] are singing together, animated and expressive, static camera. Light shifts warmer, golden tones.
1:00-1:40: In a close-up, [subjects] are singing together, focus shift. Warm light on faces, deep shadows.
1:40-2:00: [subjects] are singing together softly. Light dims, only faint glow remains.
2:00+: [subjects] are singing together, static camera. Room grows dim.
```

**Note:** Don't describe audio dynamics ("voices surging," "voices trailing off").
LTX 2.3 conditions on the actual audio waveform -- it hears the song directly.
Only describe ambient sounds not in the audio track (e.g., "muted city sounds").

### Multi-character scenes (preventing duplicates)

LTX 2.3 tends to duplicate characters, especially distinct ones (e.g., a cartoon
duck gets cloned). The model interprets vague nouns like "characters" or "figures"
as an invitation to add more.

**Rules:**
- Name each character by position and visual trait, not by count alone.
  BAD: "three characters talking on a beach"
  GOOD: "The duck on the left, the tall creature in the center, and the small
  blue figure on the right are talking together"
- Add "No other characters" or "No one else appears" as an explicit constraint.
- Use "static camera, locked off shot" to prevent panning that reveals new areas
  where the model might spawn extras.
- Never use generic group words: "crowd", "group", "people", "others".
- Position-anchor each character: "on the left", "in the center", "on the right"
  tells the model the composition is fixed.

**Negative prompt additions for multi-character:**
```
duplicate character, clone, twin, copy, mirror image, extra characters,
additional people, new figures appearing, fourth character, background characters,
wrong number of characters
```

**Example (3 cartoon characters talking in water):**
```
The duck on the left, the tall creature in the center, and the small blue
figure on the right are talking together. No other characters. Mouth movements
and subtle gestures. Water ripples. Static camera, locked off shot. Three
distinct voices in conversation.
```

**For conversation audio (non-music):** The model maps speech audio to mouth
movements via cross-attention, but it won't perfectly assign voice-to-character.
It just ensures mouths move when audio is active. Vocal separation (MelBandRoFormer)
is less useful for multi-speaker dialogue than for singing.

---

## Example: "The Body Like a Lamp" with test3.jpg

Song structure (10s trim):
```
0:02-0:42: VERSE (medium energy)
0:42-0:50: CHORUS 1 (loud)
0:50-1:40: VERSE (medium, long section)
1:40-1:52: CHORUS 2 (loud)
1:52-2:22: VERSE (medium)
2:22-2:34: CHORUS 3 (loud, peak energy)
2:34+: OUTRO (fadeout)
```

Image: Man in brocade robe standing, woman in black tank top on couch.
Concrete apartment, red phone, TV.

### Variation 1: Framing only (Pattern A)

**Node 169** (must match first schedule entry):
```
In a medium shot, the man and the woman are singing together.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together.
0:42-1:40: The man and the woman are singing together, static camera, locked off shot.
1:40-2:22: In a close-up, the man and the woman are singing together, focus shift.
2:22+: The man and the woman are singing together, static camera.
```

### Variation 2: Energy-matched (Pattern B)

**Node 169** (must match first schedule entry):
```
In a medium shot, the man and the woman are singing together softly.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together softly.
0:42-0:50: The man and the woman are singing together, leaning forward, animated, static camera.
0:50-1:40: The man and the woman are singing together steadily.
1:40-1:52: The man and the woman are singing together, both expressive and intense, static camera.
1:52-2:22: In a close-up, the man and the woman are singing together, focus shift.
2:22-2:34: The man and the woman are singing together with full energy, static camera.
2:34+: The man and the woman are singing together, relaxing back. The room is growing still.
```

### Variation 3: Lighting shifts (Pattern C)

**Node 169** (must match first schedule entry):
```
In a medium shot, the man and the woman are singing together. Soft ambient light from the windows.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together. Soft ambient light from the windows.
0:42-0:50: The man and the woman are singing together, animated, static camera. Light shifts warmer, golden tones from the TV.
0:50-1:40: The man and the woman are singing together. Warm even lighting across the room.
1:40-1:52: The man and the woman are singing together, expressive, static camera. Light grows brighter, shadows sharpen.
1:52-2:22: In a close-up, the man and the woman are singing together, focus shift. Soft warm light on faces, deep shadows.
2:22-2:34: The man and the woman are singing together, static camera. Bright light fills the room.
2:34+: The man and the woman are singing together, static camera. The room is growing dim, only faint TV glow remains.
```

---

## Example: "The Body Like a Lamp" with test4.jpg

Same song structure. Image: Woman lying on couch, man sitting at end.
Same apartment, closer composition.

### Variation 1: Framing only

**Node 169** (must match first schedule entry):
```
In a medium shot, the man and the woman are singing together on the couch.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together on the couch.
0:42-1:40: The man and the woman are singing together on the couch, static camera, locked off shot.
1:40-2:22: In a close-up, the man and the woman are singing together on the couch, focus shift.
2:22+: The man and the woman are singing together on the couch, static camera.
```

### Variation 2: Energy-matched

**Node 169** (must match first schedule entry):
```
In a medium shot, the man and the woman are singing together softly on the couch.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together softly on the couch.
0:42-0:50: The man and the woman are singing together on the couch, leaning forward, animated, static camera.
0:50-1:40: The man and the woman are singing together steadily on the couch.
1:40-1:52: The man and the woman are singing together on the couch, both expressive and intense, static camera.
1:52-2:22: In a close-up, the man and the woman are singing together on the couch, focus shift.
2:22-2:34: The man and the woman are singing together with full energy on the couch, static camera.
2:34+: The man and the woman are singing together on the couch, relaxing back. The room is growing still.
```

### Variation 3: Lighting shifts

**Node 169** (must match first schedule entry):
```
In a medium shot, the man and the woman are singing together on the couch. Soft ambient light from windows.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together on the couch. Soft ambient light from windows.
0:42-0:50: The man and the woman are singing together on the couch, animated, static camera. Light shifts warmer, golden TV glow.
0:50-1:40: The man and the woman are singing together on the couch. Warm even lighting.
1:40-1:52: The man and the woman are singing together on the couch, expressive, static camera. Light brightening, shadows sharpening.
1:52-2:22: In a close-up, the man and the woman are singing together on the couch, focus shift. Soft warm light on faces, deep shadows.
2:22-2:34: The man and the woman are singing together on the couch, static camera. Bright light fills room.
2:34+: The man and the woman are singing together on the couch, static camera. Room growing dim, faint TV glow.
```

---

## Negative prompt (use for all variations)

```
still image with no motion, subtitles, text, scene change, blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, inconsistent perspective, camera shake, incorrect depth of field, face swap, merged faces, wrong number of people, third person appearing
```

## Testing order

### Phase 1: Prompt variations (keep sampler defaults)

1. **Variation 1** (framing only) -- establish baseline consistency
2. If stable, try **Variation 2** (energy-matched) -- more dynamic
3. If still stable, try **Variation 3** (lighting) -- most visual variation
4. If any variation causes drift, increase `blend_seconds` or `overlap_seconds`

### Phase 2: Sampler tuning (after finding a good prompt)

Keep your best prompt variation and test these one at a time:

| Node | Setting | Default | Try | Expected effect |
|------|---------|---------|-----|-----------------|
| 154 (KSamplerSelect) | sampler | euler_ancestral | **euler** | More deterministic. Less noise per step = more consistent between loop iterations. Try this first. |
| 154 | sampler | euler_ancestral | **dpmpp_2m** | Faster convergence. May produce cleaner results in fewer steps. |
| 1513 (ModelSamplingSD3) | shift | 13 | **9** | Lower shift = smoother denoising schedule. May reduce artifacts at edges. |
| 1513 | shift | 13 | **7** | Even lower. More gradual denoising. Test if 9 helps. |
| 1421 (BasicScheduler) | steps | 8 | **10** | More steps = better quality, ~25% slower per iteration. |
| 1421 | steps | 8 | **12** | Diminishing returns above 10. Only if 10 shows clear improvement. |
| 1421 | scheduler | linear_quadratic | **normal** | Different noise schedule shape. Worth comparing. |

**Testing rules:**
- Change ONE value at a time. Compare output to previous best.
- Use the same seed, audio, and prompt for fair comparison.
- If `euler` helps consistency, keep it and move to shift testing.
- If quality is fine at 8 steps, don't increase -- each step costs ~7.5s of GPU time.

### Phase 3: Overlap and blend tuning (if transitions are rough)

| Setting | Default | Try | When |
|---------|---------|-----|------|
| overlap_seconds | 2.0 | **3.0** | Jitter between iterations (not at prompt boundaries) |
| blend_seconds | 5.0 | **10.0** | Style drift specifically at prompt transition timestamps |
| blend_seconds | 5.0 | **0** | Disable blending to isolate whether prompts or overlap cause issues |
| overlap_seconds | 2.0 | **1.0** | If results are good and you want faster coverage (fewer iterations) |
