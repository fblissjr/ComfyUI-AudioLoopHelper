Last updated: 2026-04-12

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
  --window 19.88 \
  --overlap 2.0 \
  -j analysis.json
```

This produces a JSON file containing:
- Audio analysis (BPM, key, sections, vocal F0)
- Workflow timing context (trim, window, stride, what node 169 covers)
- A complete LLM system prompt with all prompt engineering rules

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

## The 17 prompt rules (summary)

These are embedded in the system prompt. The LLM follows them automatically.
Reference for understanding why prompts look the way they do:

1. **Subject anchoring**: describe WHO (traits, clothing) not WHERE (setting)
2. **Present-progressive verbs**: "is singing," "is playing"
3. **Ambient sounds WITH actions**: weave chronologically, not at the end
4. **No song dynamics**: don't describe "voice surging" -- model hears the audio
5. **Describe non-track ambience**: "soft room tone," "faint hum of lights"
6. **No meta-language**: no "The scene opens with..."
7. **~200 words max per entry**: single paragraph, no markdown
8. **Camera motion only when intended**: default to static camera
9. **Avoid dolly out**: breaks limbs/faces. Exception: OUTRO
10. **Style prefix**: "Style: cinematic." unless image establishes it
11. **Identical subject in every entry**: only vary framing, camera, lighting, body language
12. **Multi-person: "singing together"**: position-anchor each person
13. **Node 169 = schedule 0:00**: must match to avoid discontinuity at ~20s
14. **Physical cues over emotions**: "chin trembles" not "singing sadly"
15. **Vocal delivery description**: "in a low gravelly voice," "brisk rhythmic delivery"
16. **Action before dialogue**: "The man leans forward and sings: 'lyrics'"
17. **Over-emoting prevention**: add to negative prompt: "exaggerated expressions, warped facial features, identity drift"

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
