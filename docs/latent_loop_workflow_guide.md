Last updated: 2026-04-09

# Latent-Space Loop Workflow (LTXVLoopingSampler)

Alternative to the TensorLoopOpen/Close workflow. Uses LTXVLoopingSampler
from ComfyUI-LTXVideo which stays entirely in latent space between
segments. No per-iteration VAE decode/encode round-trip.

## Why this approach

The TensorLoopOpen/Close workflow decodes to IMAGE every iteration and
re-encodes context frames to LATENT -- a lossy round-trip that compounds
over 8+ iterations. LTXVLoopingSampler avoids this entirely:

| | TensorLoop workflow | LTXVLoopingSampler workflow |
|--|---|---|
| VAE decode per iteration | Yes (lossy) | No |
| VAE encode per iteration | Yes (context frames) | No |
| Total VAE decodes | N iterations + 1 final | 1 final only |
| Quality degradation | Compounds over iterations | None |
| Per-segment prompts | TimestampPromptSchedule + CLIPTextEncode in loop | ScheduleToMultiPrompt + MultiPromptProvider (upfront) |
| Overlap handling | Manual subgraph (GetImageRangeFromBatch) | Built-in temporal_overlap |

## Pipeline

```
LoadImage → ImageResize
  ↓
LTXVImgToVideoInplaceKJ → initial latent (sized for full audio duration)
  ↓
Audio processing:
  LoadAudio → TrimAudioDuration → MelBandRoFormer (vocals) → LTXVAudioVAEEncode
  → LTXVConcatAVLatent (combine video + audio latent)
  ↓
Prompt scheduling:
  ScheduleToMultiPrompt (our node) → pipe-separated prompts
  → MultiPromptProvider (LTXVideo node) → conditioning list
  ↓
LTXVLoopingSampler
  model, vae, noise, sampler, sigmas, guider
  latents: combined AV latent
  temporal_tile_size: 497 (19.88s * 25fps)
  temporal_overlap: 25 (1s * 25fps)
  optional_cond_images: init image
  optional_positive_conditionings: from MultiPromptProvider
  ↓
LATENT output (full video, never left latent space)
  ↓
LTXVSeparateAVLatent → video_latent + audio_latent
  ↓
VAEDecodeTiled → IMAGE (decode ONCE)
  ↓
VHS_VideoCombine (with original audio) → final mp4
```

## Build steps in ComfyUI

### Step 1: Model loading (same as TensorLoop workflow)

- UNETLoader → model patches (ChunkFeedForward, AttentionTuner, NAG) → model
- DualCLIPLoader → Gemma 3 + text projection
- VAELoader → video VAE + audio VAE

### Step 2: Image and audio

- LoadImage → ImageResize (832x480)
- LoadAudio → TrimAudioDuration (start_index=10 or song-dependent)
- MelBandRoFormer → vocals (for lip sync conditioning)

### Step 3: Initial latent

- LTXVImgToVideoInplaceKJ: creates the initial latent sized for
  the full video duration. Set total frames based on audio duration.
  - Image guide at index 0, strength 1.0

### Step 4: Audio latent

- LTXVAudioVAEEncode: encode the (vocal-separated) audio
- LTXVConcatAVLatent: combine video latent + audio latent

### Step 5: Conditioning

- CLIPTextEncode: base positive prompt (for the guider)
- ConditioningZeroOut: negative (NAG handles actual negative)
- LTXVConditioning: add frame_rate=25

### Step 6: Prompt scheduling

- **ScheduleToMultiPrompt** (our node):
  - audio: trimmed audio (for duration)
  - window_seconds: 19.88
  - overlap_seconds: 1.0
  - schedule: your timestamp-based schedule
  - Output: pipe-separated prompts string

- **MultiPromptProvider** (LTXVideo node):
  - prompts: from ScheduleToMultiPrompt
  - clip: from DualCLIPLoader
  - Output: conditioning list (one per tile)

### Step 7: Sampler

- RandomNoise: seed
- KSamplerSelect: euler
- ManualSigmas: [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0] (8 steps)
- CFGGuider (or STGGuiderAdvanced): cfg=1.0

- **LTXVLoopingSampler**:
  - temporal_tile_size: 497 (pixel frames per window = 19.88s * 25fps)
  - temporal_overlap: 25 (pixel frames = 1s * 25fps)
  - temporal_overlap_cond_strength: 0.5
  - cond_image_strength: 1.0
  - horizontal_tiles: 1
  - vertical_tiles: 1
  - spatial_overlap: 1
  - optional_cond_images: init image
  - optional_positive_conditionings: from MultiPromptProvider

### Step 8: Decode and save

- LTXVSeparateAVLatent: split video/audio
- VAEDecodeTiled: video latent → pixels (decode ONCE)
- VHS_VideoCombine: save with original audio (from LoadAudio/TrimAudio)

## Key differences from TensorLoop workflow

### What you DON'T need
- TensorLoopOpen / TensorLoopClose
- Extension subgraph (node 843)
- AudioLoopController (stop signal, per-iteration seed)
- TimestampPromptSchedule (replaced by ScheduleToMultiPrompt)
- ConditioningBlend (LTXVLoopingSampler handles overlap natively)

### What you DO still need
- ScheduleToMultiPrompt (converts timestamps to tile prompts)
- AudioLoopPlanner (for planning timestamps -- still useful)
- AudioDuration (for computing latent size)
- MelBand vocal separation (for lip sync)
- All model loading nodes (same as before)

## Settings

| Setting | Value | Notes |
|---------|-------|-------|
| temporal_tile_size | 497 | Pixel frames per window (19.88s * 25fps). Equivalent to window_seconds. |
| temporal_overlap | 25 | Pixel frames of overlap (1s * 25fps). Equivalent to overlap_frames. |
| temporal_overlap_cond_strength | 0.5 | How strongly the overlap region conditions the next tile. Start here, increase if drift. |
| cond_image_strength | 1.0 | Init image conditioning strength. |
| adain_factor | 0.0 | Set > 0 if colors oversaturate across tiles. Try 0.1-0.3. |
| horizontal_tiles / vertical_tiles | 1 / 1 | No spatial tiling at 832x480. Increase for higher resolutions. |

## Prompt schedule format

Same timestamp format as before. ScheduleToMultiPrompt handles the conversion:

```
0:00-0:42: In a medium shot, the man and the woman are singing together.
0:42-1:40: The man and the woman are singing together, static camera, locked off shot.
1:40-2:22: In a close-up, the man and the woman are singing together, focus shift.
2:22+: The man and the woman are singing together, static camera.
```

The node computes how many tiles the audio needs, maps each tile's midpoint
to the matching schedule entry, and outputs a pipe-separated string for
MultiPromptProvider.

## Chaining with upscale

Since LTXVLoopingSampler outputs LATENT, you can chain LTXVLatentUpsampler
directly (no VAE encode needed):

```
LTXVLoopingSampler → LATENT
  → LTXVLatentUpsampler (2x)
  → SamplerCustomAdvanced (3 steps refinement)
  → VAEDecodeTiled → 1664x960 pixels
```

Zero VAE round-trips in the entire pipeline.
