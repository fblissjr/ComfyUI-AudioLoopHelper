Last updated: 2026-04-17

# Prompt Workflow: End to End

Complete walkthrough from "I have a song and an image" to "prompts are
pasted into the workflow." Covers init image preparation, VLM description
extraction, audio analysis, LLM schedule generation, and workflow insertion.

Related docs (detailed reference, not required reading):
- `prompt_creation_guide.md` -- prompt rules, variation patterns, examples
- `analysis/llm_prompt_generation_guide.md` -- LLM system prompt details, 17 rules
- `audio_analysis_guide.md` -- offline/runtime analysis, AudioPitchDetect wiring

## Prerequisites

```bash
cd custom_nodes/ComfyUI-AudioLoopHelper
uv sync --group analysis
```

You need:
- An audio file (MP3, WAV, FLAC, etc.)
- An init image (the first frame of your video)
- Access to a multimodal LLM (Gemini, Claude, GPT-4o)

## Overview

```
[init image] ──> VLM ──> subject + image-desc
                              │
[audio file] ──> analyze_audio_features.py ──> analysis.json
                              │
[init image] + [analysis.json] + [creative direction] ──> LLM ──> prompts
                              │
                    ┌─────────┴──────────┐
                    │                    │
              node 169 prompt    TimestampPromptSchedule
              (initial ~20s)     (loop iterations)
```

Five steps. Each depends on the previous.

---

## Step 1: Prepare the init image

LTX i2v expects a **single frame** -- the first frame of your video. The
model generates all subsequent frames from this starting point.

**Requirements:**
- Single scene, not a collage/grid/multi-panel image
- Resolution matching your workflow (default: 832x480)
- The person(s) visible in their starting position/pose

**If your source is multi-panel or a collage:** crop to the single panel
you want to animate. The model will try to generate video of whatever it
sees -- a 4-panel grid produces video of a grid layout.

**If generating the init image with AI:** generate a single scene directly.
Don't generate a storyboard and crop.

---

## Step 2: Extract subject and image-desc from a VLM

Upload the init image to a multimodal LLM (Gemini, Claude, GPT-4o) and
run two prompts. These produce two separate text outputs that serve
different purposes in the pipeline.

### What these are for

| Field | Purpose | Where it goes | Length |
|-------|---------|---------------|--------|
| `--subject` | Identity anchor repeated verbatim in every prompt entry | Every schedule line | ~15-30 words |
| `--image-desc` | Full visual context so the LLM knows what NOT to re-describe | JSON metadata only | 2-4 sentences |

### Prompt for `--image-desc`

Upload the init image and send:

```
Describe this image for use as a video generation reference frame.

Focus on:
- The person(s): clothing, hair color/style, skin tone, body position,
  pose, facial expression, distinguishing features
- Lighting: quality, direction, color temperature, shadows
- General setting type (e.g., "dimly lit alleyway" not a paragraph
  about every brick)

Do NOT:
- Narrate or tell a story ("a man who appears to be...")
- Speculate about emotions or backstory
- Describe background environment in exhaustive detail

Output a single paragraph, 2-4 sentences. Factual, visual-only.
```

**Example output:**
```
A young man with messy brown hair in a dirty olive-green jacket and pants,
slumped against a wet brick wall with knees drawn up. Dim blue-white light
from the right side illuminates his face. Dark, grimy alleyway at night,
wet pavement reflecting faint light.
```

### Prompt for `--subject` (1 person)

Same image, separate prompt:

```
Write a short phrase (under 20 words) describing ONLY the person in this
image. Include: gender, approximate age, hair, clothing, and what they
could plausibly be doing (e.g., "singing", "playing guitar").

Format: "a [description] [action] in a [setting-type]"

Example: "a woman in her 30s with dark hair in a vintage dress singing
in a basement workshop"
```

### Prompt for `--subject` (2-3 people)

```
Write a short phrase (under 40 words) describing the people in this image.
For EACH person, include: position (left/center/right), gender, approximate
age, hair, and one distinguishing clothing item.

End with "performing together in a [setting-type]".

Example: "a tall man on the left in a black leather jacket, a woman in the
center with short red hair in a white blouse, and a younger man on the right
in a denim vest, performing together in a dim bar"
```

**Why position-anchoring matters:** LTX 2.3 tends to duplicate characters
when descriptions are vague. "The man on the left in the dark jacket" tells
the model the composition is fixed. "A man" invites the model to spawn extras.

### Save both outputs

You'll use them in the next step as CLI arguments. Keep them as plain text.

---

## Step 3: Run audio analysis

```bash
uv run --group analysis python scripts/analyze_audio_features.py \
  path/to/song.mp3 \
  --trim 5 \
  --subject "a young disheveled man in olive-green clothing sitting in a dark alleyway" \
  --image-desc "A young man with messy brown hair in dirty olive-green jacket, slumped against wet brick wall. Dim blue-white light from right. Dark grimy alleyway at night." \
  --scene-diversity 3a \
  --window 19.88 \
  --overlap 2.0 \
  -j analysis.json
```

### CLI flags

| Flag | Required | Purpose |
|------|----------|---------|
| `--trim N` | Yes | Seconds of instrumental intro to skip (matches node 567 start_index) |
| `--subject "..."` | Yes | The VLM subject phrase from step 2 |
| `--image-desc "..."` | Yes | The VLM image description from step 2 |
| `-j analysis.json` | Yes | JSON output path (this is what you send to the LLM) |
| `--scene-diversity <code>` | No | Ambition tier + flavor. Default `2a` (performance-dynamic). See `audio_analysis_guide.md#scene-diversity-taxonomy` for all tiers 1-6 + sub-letters. |
| `--montage` | No | Arcane-style pacing: ~12s dwell, emotional-arc language. Works with any tier 2-6. |
| `--window 19.88` | No | Window seconds (default 19.88, rarely change) |
| `--overlap 2.0` | No | Overlap seconds (default 2.0) |
| `--vocal-track path` | No | Separated vocal track for cleaner F0 analysis |
| `--png-dir ./viz` | No | Save spectrogram/chromagram PNGs (human review only) |

### What the JSON contains

The output `analysis.json` has everything the LLM needs:

- **Audio analysis**: BPM, key, sections (VERSE/CHORUS/BRIDGE/etc.), vocal F0
- **Workflow timing**: trim offset, window/stride/overlap, what node 169 covers,
  `scene_diversity`, `scene_diversity_tier_name`,
  `scene_diversity_mood_bundle`, `montage` flag
- **`llm_system_prompt`**: All 17 prompt engineering rules for LTX 2.3 i2v
- **`init_image_description`**: Your VLM output, passed through for LLM context
- **`subject`**: Your subject phrase, passed through for LLM context

---

## Step 4: Generate prompts with an LLM

Open a new conversation in Gemini, Claude, or GPT-4o.

### What to send

**Attach:** The init image file (same image from step 2)

**Message:**

```
Here is the audio analysis for my music video:

<paste entire contents of analysis.json here>

Creative direction:
- Song mood: [e.g., melancholic indie folk, upbeat pop, dark electronic]
- Variation pattern: [A (framing only), B (framing + energy), or C (framing + energy + lighting)]
- Camera preference: [e.g., mostly static, use focus shift for choruses]
- Number of people in frame: [1, 2, or 3]
- Any specific requests: [e.g., "subtle body sway in verses", "eyes closed during bridge"]

Generate the node_169_prompt and schedule.
```

The `llm_system_prompt` field inside the JSON contains all the rules. The
LLM reads it inline and follows them. If your LLM supports a separate
system prompt field, you can extract `llm_system_prompt` and put it there
instead -- functionally the same.

### Why attach the image

The image is strictly better than the text description alone:
- The LLM sees exact visual traits to build consistent subject anchoring
- It can verify its prompts won't re-describe things already visible
- It catches details the text description might miss

The `--image-desc` in the JSON is a fallback for non-multimodal contexts.
When the LLM can see the image directly, it uses both.

### Do NOT attach the audio

The LLM does not need to hear the song. The analysis JSON already extracted
everything relevant (BPM, key, sections, energy levels, vocal F0). The
actual audio goes into LTX's frozen latent path at generation time -- the
model hears the real audio directly via cross-attention.

### What the LLM returns

Two blocks of text:

```
node_169_prompt: Style: cinematic. In a medium shot, a young disheveled
man in olive-green clothing is sitting against a wall, chin lowered,
breathing slowly. Faint drip of water on pavement, soft ambient hum
from distant traffic.

schedule:
0:00-0:57: Style: cinematic. In a medium shot, a young disheveled man
in olive-green clothing is sitting against a wall, swaying slightly,
static camera, locked off shot. Faint drip of water on pavement.
0:57-1:44: Style: cinematic. In a close-up, a young disheveled man in
olive-green clothing is singing with steady energy, leaning forward
slightly, focus shift. Soft echo off brick walls.
1:44-2:05: Style: cinematic. A young disheveled man in olive-green
clothing is singing quietly, shoulders relaxed, still. Ambient hum
settling.
2:05+: Style: cinematic. In a wide shot, dolly out, camera pulling back,
a young disheveled man in olive-green clothing is growing still, the
final notes trailing. Faint room tone.
```

### Quick quality check

Before pasting into the workflow, verify:

- [ ] Subject phrase is identical (or near-identical) in every entry
- [ ] `node_169_prompt` matches the first schedule entry (0:00)
- [ ] No audio dynamics described ("voice swelling", "music building")
- [ ] No meta-language ("The scene opens with", "Cut to")
- [ ] No setting re-description (just subject + changes)
- [ ] Dolly out only appears in the final/OUTRO entry (if at all)
- [ ] Present-progressive verbs ("is singing", not "sings")

If something's off, tell the LLM what to fix. The rules are already in
its context from the system prompt.

---

## Step 5: Paste into the workflow

Two locations in the ComfyUI workflow:

### Node 169 (CLIPTextEncode) -- initial render prompt

Covers trimmed 0:00 to ~0:20 (the first window). Copy the
`node_169_prompt` text and paste it into node 169's text field.

### Node 1558 (TimestampPromptSchedule) -- loop iterations

Covers everything after the first window. Copy the `schedule:` block
(just the timestamp lines, not the "schedule:" header) and paste into
node 1558's text field.

### Verify timing alignment

```
Song: |--skip--|----initial render (node 169)----|--loop iteration 1--|--iter 2--|...
      0      trim    trim + window (~20s)         trim + window + stride
```

- Node 169 prompt = schedule's 0:00 entry (must match to avoid discontinuity)
- Schedule timestamps are in TRIMMED space (--trim already subtracted)
- Loop iteration 1 fires at ~0:18 trimmed time (stride = window - overlap)

---

## Variation patterns (quick reference)

| Pattern | What varies | Risk | Start here? |
|---------|-------------|------|-------------|
| A: Framing only | Shot type (medium, close-up, wide) | Lowest | Yes |
| B: Framing + energy | Shot type + body language cues | Moderate | After A works |
| C: Framing + energy + lighting | Everything above + lighting shifts | Highest drift risk | After B works |

Use blend_seconds >= 5.0 for patterns B and C. See `prompt_creation_guide.md`
for full examples of each.

---

## Multi-person notes

See `prompt_creation_guide.md` Multi-character scenes section for comprehensive
rules, negative prompt additions, and worked examples (3 cartoon characters).

### In every schedule entry

- Always say "singing together" or "performing together"
- Position-anchor each person: "the man on the left", "the woman on the right"
- Never use "crowd", "group", "people", "others"

### Don't direct individual actions

"The man raises his hand while the woman turns" tends to produce artifacts.
Keep them as a unit: "performing together", "swaying in sync", "leaning forward."
The audio conditioning handles who's actually singing.

### Negative prompt additions

Add to your negative prompt for multi-person scenes:

```
duplicate character, clone, twin, extra characters, additional people,
new figures appearing, background characters, wrong number of characters
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Visual jump at ~20 seconds | node 169 prompt differs from schedule 0:00 | Make them match |
| Identity drift across iterations | Subject description varies between entries | Copy-paste identical subject block |
| Scene "restarts" at prompt transitions | Setting re-described in schedule entries | Remove environment descriptions |
| Limbs/faces distort | Dolly out used in non-OUTRO section | Switch to static camera |
| Style shifts despite same subject | Lighting or energy words too different | Use Pattern A first, add variation gradually |
| Prompt too long, output degraded | Over-describing (>200 words per entry) | Cut to essentials: subject + action + ambient |
