---
name: prompt-schedule
description: Generate TimestampPromptSchedule variations from audio analysis and init_image. Follows i2v rules (present-progressive, describe changes only, singing together for multi-person, no dolly out). Outputs 3 variations (framing-only, energy-matched, lighting shifts).
disable-model-invocation: true
---

# Prompt Schedule Generator

Generate prompt schedule variations for the LTX 2.3 audio-looped music video workflow.

## Usage

`/prompt-schedule <audio_file> [--image <image_path>] [--trim <seconds>]`

## Steps

1. Run `scripts/analyze_audio.py` on the audio file with the specified trim offset
2. Read the init_image if provided -- identify subjects, clothing, setting
3. Generate 3 prompt schedule variations following the patterns in `docs/guides/prompt_creation_guide.md`:
   - **Variation 1 (Pattern A)**: Framing only -- safest, identical subject/action, only shot type changes
   - **Variation 2 (Pattern B)**: Framing + energy -- shot type plus energy level matches the music
   - **Variation 3 (Pattern C)**: Framing + energy + lighting -- most visual variation

## Prompt Rules (from docs/reference/ltx23_prompt_system_prompts.md)

- i2v style: describe only changes from the init_image, NOT the full scene
- Present-progressive verbs: "is singing," "are leaning forward"
- Audio descriptions woven with actions, not at the end
- No meta-language: no "The scene opens with..."
- **No dolly out** -- breaks limbs/faces. Use static camera + lighting shifts.
- Camera keywords: `static camera, locked off shot`, `focus shift, rack focus`, `dolly in, camera pushing forward`
- Two-person scenes: always say "singing together" -- let audio conditioning handle who's singing
- Keep core subject description identical in every entry
- Don't re-describe the setting -- the image handles it

## Output Format

For each variation, output:

1. **Node 169 prompt** (initial render -- must match first schedule entry and init_image)
2. **TimestampPromptSchedule text** (ready to paste into the schedule widget)
3. **Recommended negative prompt**
4. **Workflow settings**: overlap_seconds, blend_seconds, node 567 start_index

## Negative Prompt Template

```
still image with no motion, subtitles, text, scene change, blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, inconsistent perspective, camera shake, incorrect depth of field, face swap, merged faces, wrong number of people, third person appearing
```

## Workflow Settings

| Setting | Default | Notes |
|---------|---------|-------|
| overlap_seconds | 2.0 | Increase to 3.0 if jitter |
| blend_seconds | 5.0 | Increase to 10.0 if style drift at boundaries |
| node 567 start_index | from --trim arg | Skip instrumental intro |
