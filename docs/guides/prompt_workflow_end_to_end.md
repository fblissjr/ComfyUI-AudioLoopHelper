Last updated: 2026-04-23 (absorbed llm_prompt_generation_guide.md; dropped duplicated sections)

# Prompt Workflow: End to End

Complete walkthrough from "I have a song and an image" to "prompts are
pasted into the workflow." Covers init image preparation, VLM description
extraction, audio analysis, LLM schedule generation, and workflow insertion.

Related docs (detailed reference, not required reading):
- `docs/guides/prompt_creation_guide.md` -- prompt rules, variation patterns, examples
- `docs/guides/audio_analysis_guide.md` -- offline/runtime analysis, AudioPitchDetect wiring

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
- **`llm_system_prompt`**: All 9 prompt engineering rules (R1-R9) for LTX 2.3 i2v
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

### `TimestampPromptScheduleBatchEncode` -- loop iterations

Covers everything after the first window. Copy the `schedule:` block
(just the timestamp lines, not the "schedule:" header) and paste into
the `schedule` widget on `TimestampPromptScheduleBatchEncode`. In
workflow copies from before 2026-04-22 this text went on node 1558
(`TimestampPromptSchedule`); the 2026-04-22 batch-encode fix replaced
that node — migrate via `scripts/apply_batch_encode_fix.py`. See
`docs/analysis/nag_object_patches_offload_asymmetry.md` for why.

### Verify timing alignment

```
Song: |--skip--|----initial render (node 169)----|--loop iteration 1--|--iter 2--|...
      0      trim    trim + window (~20s)         trim + window + stride
```

- Node 169 prompt = schedule's 0:00 entry (must match to avoid discontinuity)
- Schedule timestamps are in TRIMMED space (--trim already subtracted)
- Loop iteration 1 fires at ~0:18 trimmed time (stride = integer-latent quantized from window + overlap; ~17.92s at default)

---

## System prompt reference

The `llm_system_prompt` field embedded in the analyzer JSON is
organized into three parts the LLM reads as one document. You don't
need to read this to use the pipeline — you only need it if you're
debugging LLM output or writing a variant system prompt (e.g. the
standup version in `docs/reference/standup_system_prompt.md`).

### INFERENCE block (what the init image already commits to)

The init image anchors **style family** (live-action / animated / comic
/ graphic-novel / 3D-render / stop-motion), **color palette**, **setting**
(indoor/outdoor, urban/natural, wardrobe, era), and **subject appearance
and count**. The LLM is told: do NOT re-describe these. Re-describing
invites the text conditioning to fight what the image already commits
to (e.g. writing "comic-book style" on a photorealistic image forces a
tug-of-war).

What the schedule **should** drive: camera framing / motion, body
beats, lighting *shifts over time* (not palette restatement), scene
cuts, emotional arc. Schedule entries are a delta layer over the
visual baseline the image provides.

Style-appropriate beat pools: animated/comic → speed lines, panel
transitions, supersaturation, impact frames. Live-action → rack focus,
practical lighting shifts, handheld / dolly moves. The LLM infers from
the image which pool applies.

### Hard rules R1-R8

1. **R1 — Singing verb is mandatory.** Every entry contains "is
   singing..." (single) or "are singing together..." (multi). Drives
   LTX 2.3's audio-video cross-attention for lip sync. No
   "performing", "vocalizing", generic verbs. For instrumental scenes
   use "is playing <instrument>".
2. **R2 — Node 169 = first schedule entry, byte-exact.** The LLM MUST
   copy the first schedule entry verbatim into `node_169_prompt`. Any
   drift causes a visible seam at the ~20s loop-entry boundary.
3. **R3 — Identical subject across all entries.** Only vary framing,
   camera, lighting, body language, performance beats. Never
   re-describe the environment (image sets it).
4. **R4 — Multi-person position-anchoring.** Describe each person by
   position + wardrobe inside the subject string ("the man on the
   left in the dark jacket..."). No bare "crowd", "group", "duo".
5. **R5 — No meta-language.** No "The scene opens with...", "Cut
   to...", "camera shows...". Every entry begins "Style: cinematic."
   (or omits Style: when the init image commits style strongly) and
   moves straight to subject + action.
6. **R6 — Audio direction.** Do NOT describe the song (voice surging,
   music swelling — the frozen audio latent already carries that).
   DO describe ambient/diegetic sounds not in the audio track (room
   tone, fluorescent hum, rain). Vocal delivery qualifiers ("in a
   low gravelly voice") are encouraged when relevant.
7. **R7 — Camera motion.** Default "static camera, locked off shot".
   Available: dolly in, dolly left/right, jib up/down, focus shift.
   AVOID dolly out (breaks limbs/faces) except for the final OUTRO.
8. **R8 — One paragraph, ~200 words max.** No markdown or bullets.
   Present progressive throughout.

9. **R9 — Snap to stride grid.** Schedule timestamps are snapped to
   the effective `stride_seconds` grid at runtime. The LLM doesn't
   have to hand-align perfectly, but getting close reduces the shift
   at snap time.

### Ambition tiers and montage

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
scene), use emotional-arc language ("the feeling building",
"catharsis arriving", "release easing into stillness"), and dwell
~12s instead of ~20s. Arcane-style music-drives-narrative pacing.

---

## Variation patterns, multi-person rules, troubleshooting

Moved to `docs/guides/prompt_creation_guide.md` to avoid duplication.
That doc covers:

- Full worked examples of Patterns A / B / C (framing-only →
  framing+energy → framing+energy+lighting) across two songs.
- Multi-character scenes: position-anchoring, "singing together",
  negative-prompt additions, worked examples (3 cartoon characters).
- Troubleshooting table covering identity drift, iteration-boundary
  seams, dolly-out distortions, length-over-200w degradation.
