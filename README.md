# ComfyUI-AudioLoopHelper

Last updated: 2026-04-12

**TLDR**: Custom ComfyUI nodes for generating full-length music videos with LTX 2.3.
Handles loop timing, auto-stopping at the audio boundary, per-iteration seed
variation, timestamp-based prompt scheduling, smooth conditioning blending,
and latent-space overlap conversion. No manual iteration counting or fragile constants.

Two workflow variants included:
- **Image workflow** (`audio-loop-music-video_image.json`) -- tested/working.
  Per-iteration VAE decode/encode. Proven lip sync.
- **Latent workflow** (`audio-loop-music-video_latent.json`) -- UNTESTED.
  Operates in latent space using LatentContextExtract and LatentOverlapTrim.
  No per-iteration VAE round-trip. These nodes handle noise_mask stripping
  internally (critical for sampler correctness). Use at your own risk until
  testing confirms lip sync parity.

Workflow adapted from [kijai's LTX 2.3 long loop extension test](https://github.com/kijai/ComfyUI-NativeLooping_testing/blob/main/ltx23_long_loop_extension_test.json).

Built for use alongside:
- [ComfyUI-NativeLooping](https://github.com/kijai/ComfyUI-NativeLooping_testing) -- TensorLoopOpen/Close loop mechanism
- [ComfyUI-LTXVideo](https://github.com/logtd/ComfyUI-LTXVideo) -- LTXVAddLatentGuide, LTXVCropGuides, LTXVPreprocess, spatial upscaler
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) -- video output and batching
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) -- Set/Get nodes, LTX2_NAG, LTXVImgToVideoInplaceKJ, ImageResizeKJv2
- [ComfyUI-MelBandRoFormer](https://github.com/DrJKL/ComfyUI-MelBandRoFormer) -- vocal separation for improved lip sync

## Quick start

1. Set TensorLoopOpen iterations to **50** (high safety cap)
2. Add **Audio Loop Controller** -- wire `current_iteration` from TensorLoopOpen,
   `window_seconds` from your window size constant, `audio` from your trimmed audio
3. Wire outputs: `start_index` -> extension component, `should_stop` -> TensorLoopClose stop,
   `iteration_seed` -> extension noise_seed, `stride_seconds` -> Timestamp Prompt Schedule
4. Add **Timestamp Prompt Schedule** -- write prompts for each section of your song
5. Add **Audio Loop Planner** -- see the iteration timeline to plan your prompts
6. Run. The loop auto-stops when it reaches the end of the audio.

## Nodes

### Audio Loop Controller

The core node. Computes loop timing, stop signal, seed, and stride from
just the audio tensor + window/overlap settings.

**Inputs:**

| Input | Type | Description |
|-------|------|-------------|
| current_iteration | INT | From TensorLoopOpen (1-based) |
| window_seconds | FLOAT | Video generation window duration (default 19.88) |
| overlap_seconds | FLOAT | Overlap between consecutive windows (default 1.0). Stride = window - overlap. Match to overlap_frames/fps in the extension component. |
| audio | AUDIO | The audio track |
| seed | INT | Base seed. Output = seed + current_iteration |
| fps | INT | Video frame rate (default 25). Used to compute overlap_frames output. |

**Outputs:**

| Output | Type | Wire to |
|--------|------|---------|
| start_index | FLOAT | Extension component's start_index input |
| should_stop | BOOLEAN | TensorLoopClose's stop input |
| audio_duration | FLOAT | Informational (total audio length) |
| iteration_seed | INT | Extension component's noise_seed input |
| stride_seconds | FLOAT | TimestampPromptSchedule and AudioLoopPlanner |
| overlap_frames | INT | Extension component's overlap_frames input (pixel space) |
| overlap_latent_frames | INT | LatentContextExtract / LatentOverlapTrim in latent-space subgraph |
| overlap_seconds | FLOAT | Extension subgraph's LTXVAudioVideoMask video_start_time |

Changing `overlap_seconds` automatically adjusts stride, stop timing,
start indices, iteration count, overlap_frames, overlap_latent_frames,
and the subgraph's video_start_time mask. One value, one place.

### Timestamp Prompt Schedule

Per-iteration prompt variation based on song timestamps. Write prompts
for verse, chorus, bridge -- the node picks the right one each iteration.
Supports gradual blending between prompts at transitions via `blend_seconds`.

**Inputs:**

| Input | Type | Description |
|-------|------|-------------|
| current_iteration | INT | From TensorLoopOpen |
| stride_seconds | FLOAT | From AudioLoopController |
| schedule | STRING | Timestamp-based prompt schedule (see format below) |
| blend_seconds | FLOAT | Transition duration (default 0 = hard switch). Set to e.g. 5.0 to blend over 5 seconds before each boundary. |

**Outputs:**

| Output | Type | Wire to |
|--------|------|---------|
| prompt | STRING | Text encode -> ConditioningBlend conditioning_a |
| next_prompt | STRING | Text encode -> ConditioningBlend conditioning_b |
| blend_factor | FLOAT | ConditioningBlend blend_factor |
| current_time | FLOAT | Informational (position in audio) |

When `blend_seconds = 0`, `next_prompt` equals `prompt` and `blend_factor` is
always 0.0. You can wire just `prompt` through a single text encoder directly
to the extension component -- no blending needed.

**Schedule format:**

One prompt for the entire song (simplest):
```
0:00+: Style: cinematic. A woman in her 30s with dark hair is singing passionately alone in a dimly lit basement workshop. Strings of colorful, mismatched Christmas lights provide a warm glow against damp stone walls. Her voice carries through the small space, resonating with raw emotion. Soft ambient hum of the lights blends with her singing.
```

Three sections with shot variation (keep core subject consistent):
```
0:00-0:38: Style: cinematic. In a medium close-up, a woman in her 30s with dark hair is singing passionately in a dimly lit basement workshop. Christmas lights cast warm colorful reflections on her face. Her voice fills the small space with raw emotion. Soft ambient hum from the lights.
0:38-1:15: Style: cinematic. A woman in her 30s with dark hair is singing with building energy in a dimly lit basement workshop, static camera, locked off shot. Wide shot reveals the full workshop space, tools on walls, Christmas lights strung across exposed beams. Her voice grows more powerful, echoing off stone walls. The lights buzz softly.
1:15+: Style: cinematic. In an extreme close-up, a woman in her 30s with dark hair is singing softly in a dimly lit basement. Focus on her face and hands, Christmas light bokeh in background. Her voice is gentle and emotional, almost a whisper. Quiet ambient room tone.
```

Format rules:
- Timestamps: `M:SS`, `M:SS.ss`, or bare seconds (`38.5`)
- Ranges: `start-end` (inclusive) or `start+` (from here onward)
- Last match wins if ranges overlap
- Fallback: last entry used if nothing matches

### Audio Loop Planner

Shows the iteration timeline so you know what timestamps to use in your
prompt schedule. Leave it in the workflow -- it auto-updates when you
change audio or overlap settings.

**Inputs:**

| Input | Type | Description |
|-------|------|-------------|
| audio | AUDIO | The audio track |
| stride_seconds | FLOAT | From AudioLoopController |
| window_seconds | FLOAT | Same as AudioLoopController |

**Outputs:**

| Output | Type | Description |
|--------|------|-------------|
| summary | STRING | Wire to PreviewAny/ShowText to see on canvas |
| total_iterations | INT | Estimated iteration count |

**Example summary output:**
```
Audio: 143.0s (2:23)
Stride: 18.88s | Window: 19.88s
Overlap: 1.00s
Estimated 7 iterations:

  Iter 1:  0:18 - 0:38  (18.9s - 38.8s)
  Iter 2:  0:37 - 0:57  (37.8s - 57.6s)
  Iter 3:  0:56 - 1:16  (56.6s - 76.5s)
  ...
```

### Conditioning Blend

Blends two conditionings with a factor. Works with LTX 2.3 Gemma 3
text encoder (no pooled_output required).

**Inputs:**

| Input | Type | Description |
|-------|------|-------------|
| conditioning_a | CONDITIONING | Current prompt conditioning |
| conditioning_b | CONDITIONING | Next prompt conditioning |
| blend_factor | FLOAT | 0.0 = all A, 1.0 = all B. Wire from TimestampPromptSchedule. |

**Outputs:**

| Output | Type | Wire to |
|--------|------|---------|
| conditioning | CONDITIONING | Extension component's positive input |

Handles sequence length alignment (zero-pads shorter tensor), attention mask
combining (OR of both masks), and pooled_output blending when present.

When blend_factor = 0.0, passes conditioning_a through unchanged (no computation).

**Note:** LTX 2.3 wraps its Gemma 3 text encoder in ComfyUI's
"CLIPTextEncode" node for compatibility. Despite the CLIP name, the
conditioning has no pooled_output. ConditioningBlend handles this correctly.

**Wiring for prompt blending:**
```
TimestampPromptSchedule
  |           |           |
  prompt   next_prompt  blend_factor
  |           |           |
  v           v           |
Text       Text          |
Encode     Encode        |
  |           |           |
  v           v           v
ConditioningBlend --------+
  |
  v
Extension #843 positive input
```

### Audio Duration

Simple utility. Returns duration, sample rate, and total samples from
an audio tensor.

**Inputs:** `audio` (AUDIO)
**Outputs:** `duration_seconds` (FLOAT), `sample_rate` (INT), `total_samples` (INT)

### Latent Context Extract

Extracts the last N latent frames as context for the next loop iteration.
Replaces LTXVSelectLatents + StripLatentNoiseMask in the latent-space
subgraph (latent workflow). Strips noise_mask so LTXVAudioVideoMask creates a
fresh mask (matching VAEEncode behavior from the IMAGE workflow).

**Inputs:**

| Input | Type | Description |
|-------|------|-------------|
| latent | LATENT | Previous iteration's video latent (from TensorLoopOpen) |
| overlap_latent_frames | INT | Number of tail latent frames to extract (default 4). Wire from AudioLoopController. |

**Outputs:**

| Output | Type | Wire to |
|--------|------|---------|
| context | LATENT | LTXVAudioVideoMask video_latent input |

### Latent Overlap Trim

Trims the first N latent frames (overlap region) from a sampler's output.
Keeps new content only, strips noise_mask. Used in the latent-space
subgraph (latent workflow) to avoid duplicating overlap frames across iterations.

**Inputs:**

| Input | Type | Description |
|-------|------|-------------|
| latent | LATENT | Sampler output video latent (after CropGuides) |
| overlap_latent_frames | INT | Number of leading latent frames to trim (default 4). Wire from AudioLoopController. |

**Outputs:**

| Output | Type | Wire to |
|--------|------|---------|
| trimmed | LATENT | Subgraph output / TensorLoopClose |

### Strip Latent Noise Mask

Low-level utility that removes noise_mask from a latent dict. Prefer
LatentContextExtract or LatentOverlapTrim which handle this automatically.

**Inputs:** `latent` (LATENT)
**Outputs:** `latent` (LATENT, noise_mask removed)

### Audio Pitch Detect

Per-iteration vocal pitch detection using torchaudio. Best results when
wired to MelBandRoFormer's separated vocals output (already in the workflow).

**Inputs:**

| Input | Type | Description |
|-------|------|-------------|
| audio | AUDIO | Audio track (wire to MelBandRoFormer vocals output for clean signal) |
| start_seconds | FLOAT | From AudioLoopController.start_index |
| window_seconds | FLOAT | Same value as AudioLoopController.window_seconds |
| freq_low | FLOAT | Min detection frequency (default 85 Hz, low male vocals) |
| freq_high | FLOAT | Max detection frequency (default 400 Hz, high female vocals) |

**Outputs:**

| Output | Type | Use |
|--------|------|-----|
| median_f0 | FLOAT | Median fundamental frequency in Hz (0.0 if unvoiced) |
| has_vocals | BOOLEAN | True if pitched content detected in window |
| is_male_range | BOOLEAN | True if median F0 < 160 Hz |
| is_female_range | BOOLEAN | True if median F0 > 160 Hz |
| vocal_fraction | FLOAT | Ratio of voiced frames (0.0-1.0) |

**Example wiring for vocal/instrumental prompt switching:**
```
MelBandRoFormerSampler (vocals output)
  └→ AudioPitchDetect.audio

AudioLoopController
  └→ start_index → AudioPitchDetect.start_seconds

AudioPitchDetect
  └→ has_vocals → Switch node → select vocal vs instrumental prompt
```

Note: requires a Switch/Mux node (from ComfyUI-KJNodes or similar) to
conditionally select between prompt paths based on the BOOLEAN output.

## Tuning guide

### overlap_seconds (AudioLoopController)

Controls how much context the model sees from the previous iteration.

| Value | Frames at 25fps | Effect |
|-------|-----------------|--------|
| 0.5 | 12-13 | Minimal context. Faster coverage but transitions may be jarring. |
| **1.0** | **25** | **Default. Good balance of coherence and speed.** |
| 2.0 | 50 | More context. Smoother transitions, better coherence, but each iteration covers less new ground (more total iterations needed). |
| 3.0+ | 75+ | Diminishing returns. Much slower progress per iteration. Only useful if coherence is very poor. |

When you change overlap, the stride auto-adjusts, the stop signal auto-adjusts,
and the planner output updates. Nothing else to touch.

The `overlap_frames` output from AudioLoopController feeds directly into
the extension subgraph -- no manual sync needed. Changing overlap_seconds
automatically propagates to the frame extraction and trimming logic.

### window_seconds

How many seconds of video each generation window produces. Tied to the
LTX 2.3 model's video_end_time parameter.

| Value | Effect |
|-------|--------|
| 10-15 | Shorter windows. Faster per-iteration but more iterations needed. May reduce quality. |
| **19.88** | **Default for LTX 2.3. Recommended.** |
| 25-30 | Longer windows. Fewer iterations but heavier VRAM and slower per-iteration. Quality may degrade at edges. |

Changing this also requires updating the `window_size_seconds` FloatConstant
(node 688) that feeds the extension component's video_end_time.

### Audio trim offset (Node 567 TrimAudioDuration)

The `start_index` on the global audio trim is song-dependent. It skips
instrumental intro that doesn't contribute to lip sync. Set to 0 for songs
that start with vocals, or skip a few seconds for instrumental intros.

### seed

Base seed for generation. Each iteration gets `seed + iteration_number`.

| Approach | Effect |
|----------|--------|
| Fixed seed (e.g., 42) | Reproducible results. Same seed + same audio = same video. |
| Random seed | Different results each run. Good for exploration. |

The per-iteration increment prevents the "infinite loop of nothingness"
where DiT models generate degenerate/repetitive content with identical
seeds across loop iterations.

### Prompt schedule timing

Match your prompt schedule timestamps to your song structure, not to
iteration boundaries. The node handles the mapping.

Tips:
- Use the **Audio Loop Planner** output to see exactly what time range
  each iteration covers, then align your prompts to the song.
- **Keep the core subject consistent** across all schedule entries. Only
  vary framing, lighting, and energy. Different subjects = different text
  embeddings = style drift even at CFG 1.0.
- Use **blend_seconds** (e.g., 5.0) to smooth transitions. This prevents
  the hard conditioning switch that causes style jumps at boundaries.
- The init_image anchors the first frame of the first pass via
  LTXVImgToVideoInplaceKJ. The extension subgraph also uses the
  init_image as a guide at frame -1 each loop iteration via
  LTXVAddLatentGuide for continuity.

### blend_seconds (TimestampPromptSchedule)

Controls how gradually prompt transitions happen.

| Value | Effect |
|-------|--------|
| **0** | **Hard switch (default).** Prompt changes instantly at timestamp boundaries. Can cause style drift. |
| 3-5 | Gradual blend over 3-5 seconds. Good starting point. |
| 10+ | Very slow transition. Useful when prompts differ significantly. |

The blend_factor ramps linearly from 0 to 1 over `blend_seconds` before
each timestamp boundary. At 0, only the current prompt's conditioning is
used. At 1, fully transitioned to the next prompt.

## Audio feature analysis (offline)

Analyze your audio track before building your prompt schedule. Extracts
musical features that help you write better, music-aware prompts.

**Setup** (one time):
```bash
cd custom_nodes/ComfyUI-AudioLoopHelper
uv sync --group analysis
```

**Usage:**
```bash
# Full analysis with markdown report
uv run --group analysis python scripts/analyze_audio_features.py your_song.wav

# Write report to file
uv run --group analysis python scripts/analyze_audio_features.py your_song.wav -o analysis.md

# Export JSON (for LLM-assisted schedule generation)
uv run --group analysis python scripts/analyze_audio_features.py your_song.wav -j analysis.json

# PNG visualizations (spectrogram, chromagram, onset envelope)
uv run --group analysis python scripts/analyze_audio_features.py your_song.wav --png-dir ./viz

# With trim offset and separated vocal track for F0 analysis
uv run --group analysis python scripts/analyze_audio_features.py your_song.wav \
  --trim 10 --vocal-track vocals_only.wav

# Generate full LTX 2.3 prompt templates (copy-paste ready)
uv run --group analysis python scripts/analyze_audio_features.py your_song.wav \
  --subject "a woman in her 30s with dark hair singing in a basement workshop" \
  --trim 10
```

**What it extracts:**

| Feature | Description | Use for scheduling |
|---------|-------------|-------------------|
| BPM | Tempo and beat grid timestamps | Align prompt transitions to beats |
| Key | Musical key (e.g., "G Major") | Annotate mood shifts |
| Chromagram | 12-pitch-class heatmap (PNG) | Visual harmonic review |
| Mel spectrogram | Frequency x time heatmap (PNG) | Visual BPM/vocal review |
| Vocal F0 | Fundamental frequency + male/female classification | Choose appropriate prompts |
| Structure | Labeled sections (intro, verse, chorus, bridge, outro) | Scaffold your TimestampPromptSchedule |

**Using with an LLM for schedule generation:**

The JSON output is designed to be pasted into an LLM prompt:
```
You are a music video director. Here is the analysis for the track:
{paste JSON here}

Write a TimestampPromptSchedule for a woman singing in a dimly lit
basement workshop with Christmas lights. Keep the core subject
identical across all entries. Vary only framing, camera, and energy.
```

The LLM reads the structured data (BPM, key, sections, vocal range) and
generates a complete schedule. PNG visualizations are for your own review
only -- don't feed them to the LLM.

## Prompt writing guide (LTX 2.3)

LTX 2.3 is a distilled model -- CFG is 1.0 by default (NAG handles guidance).
This means text prompts have less direct influence than in non-distilled models,
but different conditioning vectors still shift the generation space. Write
prompts carefully for best results.

### Rules for loop-friendly prompts

1. **Keep the core subject identical across all schedule entries.** Change
   framing, camera, lighting -- not the person or setting. "A woman in her
   30s with dark hair singing in a basement workshop" should appear in every
   entry. This keeps Gemma 3 embeddings close in vector space.

2. **Use active, present-progressive language.** "is singing," "is walking."
   If no action specified, describe natural movements.

3. **Describe only what changes from the previous section.** Don't re-describe
   established visual details. Redundant descriptions can cause the model to
   "restart" the scene.

4. **Include audio descriptions alongside visuals.** LTX 2.3 is audio-video
   joint. Describe the soundscape: "her voice echoes off stone walls," "soft
   ambient hum of Christmas lights." Align audio intensity with action tempo.

5. **Start with Style.** `Style: cinematic.` or `Style: realistic.` at the
   beginning. Omit if the init_image already establishes the style strongly.

6. **No timestamps, scene cuts, or meta-language.** Don't write "The scene
   opens with..." or "Cut to..." -- just describe what is happening.

7. **Camera motion only when intended.** Don't add camera movement unless you
   want it. Available motions:

### Camera motion keywords

Append these to your prompt to control camera:

| Keyword | Description |
|---------|-------------|
| `static camera, locked off shot` | No camera movement |
| `dolly in, camera pushing forward` | Smooth forward movement |
| `dolly out, camera pulling back` | Smooth backward movement |
| `dolly left, camera tracking left` | Lateral left movement |
| `dolly right, camera tracking right` | Lateral right movement |
| `jib up, camera rising up` | Upward crane movement |
| `jib down, camera lowering down` | Downward crane movement |
| `focus shift, rack focus` | Changing focal point |

### Negative prompt

Use this as the base negative prompt (node 169 / static negative conditioning):

```
blurry, out of focus, overexposed, underexposed, low contrast, washed out colors,
excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted
proportions, unnatural skin tones, deformed facial features, asymmetrical face,
missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts
around text, inconsistent perspective, camera shake, incorrect depth of field
```

### Example: music video schedule with consistent subject

```
0:00-0:35: Style: cinematic. In a medium close-up, a woman in her 30s with dark hair is singing passionately in a dimly lit basement workshop. Christmas lights cast warm colorful reflections on her face. Her voice fills the small space. Soft ambient hum from the lights.
0:35-1:10: Style: cinematic. A woman in her 30s with dark hair is singing with building energy in a dimly lit basement workshop, static camera, locked off shot. Wide shot reveals the full space, tools on walls, Christmas lights across exposed beams. Her voice grows powerful. Lights buzz softly.
1:10-1:40: Style: cinematic. In an extreme close-up, a woman in her 30s with dark hair is singing softly in a dimly lit basement. Focus on face and hands, Christmas light bokeh in background, focus shift. Her voice is gentle, almost a whisper. Quiet room tone.
1:40+: Style: cinematic. A woman in her 30s with dark hair is singing in a dimly lit basement workshop, dolly out, camera pulling back. Medium shot, gentle swaying, final verse. Her voice carries with quiet intensity. Christmas lights glow warmly. Soft ambient hum fades.
```

Note: every entry repeats "a woman in her 30s with dark hair" and "dimly lit
basement workshop." Only framing, camera, and energy change.