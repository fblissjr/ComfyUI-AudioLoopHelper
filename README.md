# ComfyUI-AudioLoopHelper

**TLDR**: Drop in these nodes to generate full-length music videos with LTX 2.3.
They handle loop timing, auto-stopping at the audio boundary, per-iteration
seed variation, timestamp-based prompt scheduling, and smooth conditioning
blending between prompt transitions. No manual iteration counting or fragile constants.

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
| overlap_frames | INT | Extension component's overlap_frames input |

Changing `overlap_seconds` automatically adjusts the stride, stop timing,
start indices, iteration count, and overlap_frames. One value, one place.

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
0:00+: a woman singing in a dimly lit basement with colorful Christmas lights
```

Three sections (verse / chorus / bridge):
```
0:00-0:38: a woman singing in a dimly lit basement, close-up, soft lighting
0:38-1:15: wide shot of full band playing, bright colorful stage lights, energetic
1:15+: extreme close-up of singer, emotional performance, warm bokeh
```

Detailed per-section with bare seconds:
```
0-20: opening shot, slow zoom in on singer, dark moody atmosphere
20-45: singer at microphone, medium shot, Christmas lights twinkling
45-75: wide shot showing full room, band visible, bright energetic lighting
75-100: close-up hands on guitar, intercut with singer face
100+: final verse, slow pull back, soft warm glow, singer silhouette
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

Blends two conditionings with a factor. Works with LTX 2.3's Gemma 3
text encoder (no pooled_output required) and standard CLIP conditioning.

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
  vary framing, lighting, and energy. Different subjects = different CLIP
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