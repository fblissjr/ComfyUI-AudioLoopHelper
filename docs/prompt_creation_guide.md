Last updated: 2026-04-09

# Music Video Prompt Creation Guide

Step-by-step process for creating prompt schedules for the LTX 2.3
audio-conditioned video loop workflow.

## Process

### 1. Analyze the audio

```bash
uv run scripts/analyze_audio.py path/to/song.wav --trim 10
```

This gives you:
- Energy timeline showing loud/quiet sections
- Detected song structure (verse/chorus/bridge)
- Prompt schedule template with timestamps

Use `--trim N` to offset timestamps by your node 567 start_index value.

### 2. Separate vocals (optional)

If you want to see when vocals are present vs instrumental:

```bash
uv run scripts/separate_vocals.py path/to/song.wav --output-dir ./output/
uv run scripts/analyze_audio.py ./output/vocals.wav --trim 10
```

Note: MelBandRoFormer separates vocals from instruments but does NOT
distinguish male from female vocals. For two-person scenes, write
prompts assuming both are singing together and let the audio conditioning
handle who's actually vocalizing.

### 3. Study the init_image

Identify:
- Who is in the frame (number of people, positions)
- What they're wearing (use for minimal identification in prompts)
- The setting (don't re-describe in prompts -- image handles it)
- Lighting and mood (baseline to vary from)

### 4. Write prompts following i2v rules

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

### 6. Run and iterate

- First run: no prompt schedule (static prompt, blend_seconds=0).
  Verify the base loop works and consistency holds.
- Second run: add prompt schedule with conservative changes
  (same subject, vary only framing). blend_seconds=5.0.
- Third run: add lighting variations. Increase blend_seconds if drift appears.

---

## Prompt Variation Patterns

### Pattern A: Framing only (safest)

Every entry has identical subject and action. Only the shot type changes.

```
0:00-0:40: In a medium shot, [subjects] are singing together. [audio description].
0:40-1:20: [subjects] are singing together, static camera, locked off shot. [audio description].
1:20-2:00: In a close-up, [subjects] are singing together, focus shift. [audio description].
2:00+: [subjects] are singing together, static camera. [audio description].
```

### Pattern B: Framing + energy (moderate)

Shot type changes plus energy level matches the music.

```
0:00-0:40: In a medium shot, [subjects] are singing together softly. [quiet audio].
0:40-1:00: [subjects] are singing together with growing intensity. [building audio].
1:00-1:40: In a close-up, [subjects] are singing together with full energy. [powerful audio].
1:40-2:00: [subjects] are singing together quietly. [intimate audio].
2:00+: [subjects] are singing together as the energy fades. [fading audio].
```

### Pattern C: Framing + energy + lighting (most variation)

Everything above plus lighting shifts to match mood.

```
0:00-0:40: In a medium shot, [subjects] are singing together. Soft ambient lighting. [audio].
0:40-1:00: [subjects] are singing together with growing energy, static camera. Light shifts warmer, golden tones. [audio].
1:00-1:40: In a close-up, [subjects] are singing together with full power, focus shift. Warm light on faces, deep shadows. [audio].
1:40-2:00: [subjects] are singing together softly. Light dims, only faint glow remains. [audio].
2:00+: [subjects] are singing together as energy fades, static camera. Room grows dim. [audio].
```

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
In a medium shot, the man and the woman are singing together. Their voices are filling the room with a warm resonance. Muted city sounds through the windows.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together. Their voices are filling the room with a warm resonance. Muted city sounds through the windows.
0:42-1:40: The man and the woman are singing together with growing energy, static camera, locked off shot. Their combined voices are echoing off the concrete walls.
1:40-2:22: In a close-up, the man and the woman are singing together, focus shift. Their voices are intimate and blending quietly.
2:22+: The man and the woman are singing together, static camera. Their voices are trailing off gently. The room is growing quiet.
```

### Variation 2: Energy-matched (Pattern B)

**Node 169** (must match first schedule entry):
```
In a medium shot, the man and the woman are singing together softly. Their voices are filling the room gently. Quiet city ambience through the windows.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together softly. Their voices are filling the room gently. Quiet city ambience through the windows.
0:42-0:50: The man and the woman are singing together with sudden intensity, static camera. Their voices are surging powerfully, echoing off the walls.
0:50-1:40: The man and the woman are singing together at a steady pace. Their voices carry with warmth. Ambient room tone.
1:40-1:52: The man and the woman are singing together with building power, static camera. Their voices are rising, the room resonating.
1:52-2:22: In a close-up, the man and the woman are singing together, focus shift. Their voices are layering and growing.
2:22-2:34: The man and the woman are singing together at full intensity, static camera. Their voices are at their peak, filling every corner.
2:34+: The man and the woman are singing together as the energy fades. Their voices are trailing off. The room is settling into quiet.
```

### Variation 3: Lighting shifts (Pattern C)

**Node 169** (must match first schedule entry):
```
In a medium shot, the man and the woman are singing together. Their voices are filling the room. Soft ambient light from the windows, muted city sounds.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together. Their voices are filling the room. Soft ambient light from the windows, muted city sounds.
0:42-0:50: The man and the woman are singing together with sudden intensity, static camera. Light shifts warmer, golden tones from the TV washing over them. Their voices are surging.
0:50-1:40: The man and the woman are singing together steadily. Warm even lighting across the room. Their voices carry with resonance.
1:40-1:52: The man and the woman are singing together with rising power, static camera. Light grows brighter, shadows sharpen. Their voices are building.
1:52-2:22: In a close-up, the man and the woman are singing together, focus shift. Soft warm light on their faces, deep shadows around them. Their voices are intimate.
2:22-2:34: The man and the woman are singing together at full intensity, static camera. Bright light fills the room. Their voices peak.
2:34+: The man and the woman are singing together as energy fades, static camera. The room is growing dim, only faint TV glow remains. Their voices trail off.
```

---

## Example: "The Body Like a Lamp" with test4.jpg

Same song structure. Image: Woman lying on couch, man sitting at end.
Same apartment, closer composition.

### Variation 1: Framing only

**Node 169** (must match first schedule entry):
```
In a medium shot, the man and the woman are singing together on the couch. Their voices are filling the room. Muted city sounds through the windows.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together on the couch. Their voices are filling the room. Muted city sounds through the windows.
0:42-1:40: The man and the woman are singing together on the couch, static camera, locked off shot. Their voices are echoing off the concrete walls.
1:40-2:22: In a close-up, the man and the woman are singing together on the couch, focus shift. Their voices are intimate and warm.
2:22+: The man and the woman are singing together on the couch, static camera. Their voices are trailing off gently. The room is growing quiet.
```

### Variation 2: Energy-matched

**Node 169** (must match first schedule entry):
```
In a medium shot, the man and the woman are singing together softly on the couch. Their voices are gentle in the room. City ambience.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together softly on the couch. Their voices are gentle in the room. City ambience.
0:42-0:50: The man and the woman are singing together with sudden intensity on the couch, static camera. Their voices are surging powerfully.
0:50-1:40: The man and the woman are singing together steadily on the couch. Their voices carry with warmth. Ambient room tone.
1:40-1:52: The man and the woman are singing together with building power on the couch, static camera. Their voices are rising.
1:52-2:22: In a close-up, the man and the woman are singing together on the couch, focus shift. Voices layering and growing.
2:22-2:34: The man and the woman are singing together at full intensity on the couch, static camera. Voices at their peak.
2:34+: The man and the woman are singing together as energy fades on the couch. Voices trailing off. Room settling into quiet.
```

### Variation 3: Lighting shifts

**Node 169** (must match first schedule entry):
```
In a medium shot, the man and the woman are singing together on the couch. Soft ambient light from windows. Their voices fill the room gently. Muted city sounds.
```

**Schedule:**
```
0:00-0:42: In a medium shot, the man and the woman are singing together on the couch. Soft ambient light from windows. Their voices fill the room gently. Muted city sounds.
0:42-0:50: The man and the woman are singing together with intensity on the couch, static camera. Light shifts warmer, golden TV glow. Voices surging.
0:50-1:40: The man and the woman are singing together on the couch. Warm even lighting. Voices carrying steadily.
1:40-1:52: The man and the woman are singing together with rising power on the couch, static camera. Light brightening, shadows sharpening. Voices building.
1:52-2:22: In a close-up, the man and the woman are singing together on the couch, focus shift. Soft warm light on faces, deep shadows. Intimate vocals.
2:22-2:34: The man and the woman are singing together at full intensity on the couch, static camera. Bright light fills room. Peak vocals.
2:34+: The man and the woman are singing together as energy fades on the couch, static camera. Room growing dim, faint TV glow. Voices trailing off.
```

---

## Negative prompt (use for all variations)

```
still image with no motion, subtitles, text, scene change, blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, inconsistent perspective, camera shake, incorrect depth of field, face swap, merged faces, wrong number of people, third person appearing
```

## Testing order

1. **Variation 1** (framing only) first -- establish baseline consistency
2. If stable, try **Variation 2** (energy-matched) -- more dynamic
3. If still stable, try **Variation 3** (lighting) -- most visual variation
4. If any variation causes drift, increase `blend_seconds` or `overlap_seconds`
