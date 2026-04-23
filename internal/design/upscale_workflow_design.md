Last updated: 2026-04-09

# Upscale Workflow Guide

Separate workflow to upscale the loop output from 832x480 to 1664x960.
Based on RuneXX's 3-pass approach: upscale in latent space, refine with
3 diffusion steps, decode only at the end.

Reference: `coderef/RuneXX_LTX-2.3-Workflows/3-Pass-Experimental/`

## Why a separate workflow?

The loop workflow generates at 832x480. A previous attempt to add
upscaling inside the same workflow failed:
1. VAE re-encoding (IMAGE → LATENT) is lossy
2. Refinement sampler OOM'd on 24GB GPU

This separate workflow loads the finished video and upscales it.
One unavoidable VAE encode (video → latent), then everything stays
in latent space until the final decode.

## Pipeline

```
VHS_LoadVideo (loop output, 832x480, 25fps)
  ↓
VAEEncode (encode to latent, 104x60)
  ↓
LTXVLatentUpsampler (2x → 208x120 latent)
  ↓
LTXVImgToVideoConditionOnly (re-condition upscaled latent)
  ↓
SamplerCustomAdvanced (3 steps, refine the upscaled latent)
  ↓
LTXVSeparateAVLatent (split video/audio latent)
  ↓
VAEDecodeTiled (decode to 1664x960 pixels)
  ↓
VHS_VideoCombine (save with original audio)
```

## Build steps in ComfyUI

### Step 1: Load models

Add these nodes:
- **UNETLoader**: Load LTX 2.3 model (same as loop workflow)
- **DualCLIPLoader**: Load Gemma 3 + text projection (same as loop workflow)
- **VAELoader**: Load LTX VAE (same as loop workflow)
- **LTXVLatentUpsamplerModelLoader** (or however it loads):
  Model: `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`

### Step 2: Load the loop output video

- **VHS_LoadVideo**: Point to the loop output mp4
  - frame_rate: 25
  - force_rate: true

### Step 3: Encode to latent

- **VAEEncode**: video frames → latent
  - If OOM: use **VAEEncodeTiled** instead

### Step 4: Upscale in latent space

- **LTXVLatentUpsampler**: 2x spatial upscale
  - Input: latent from VAEEncode
  - Output: 2x larger latent (208x120)

### Step 5: Conditioning

- **CLIPTextEncode**: Encode a prompt (use the same prompt as your loop workflow)
- **ConditioningZeroOut**: Zero the negative (same as loop workflow)
- **LTXVConditioning**: Add frame_rate=25 metadata
- **LTXVImgToVideoConditionOnly**: Re-condition the upscaled latent
  (this tells the model what the content is, so refinement doesn't drift)

### Step 6: Memory optimization patches

Add these to the model chain (same as loop workflow):
- **LTXVChunkFeedForward**: chunk_size=2, threshold=4096
- **LTX2AttentionTunerPatch**: defaults (1,1,1,1, triton=true)
- Optional: **PatchSageAttentionKJ** if you need more VRAM savings

### Step 7: Refinement sampler

- **ManualSigmas**: `[0.85, 0.725, 0.4219, 0.0]` (3 steps)
- **KSamplerSelect**: `euler` (not euler_ancestral -- more deterministic for refinement)
- **CFGGuider**: cfg=1.0 (same as loop)
- **SamplerCustomAdvanced**: Wire guider, sampler, sigmas, upscaled latent

### Step 8: Decode and save

- **LTXVSeparateAVLatent**: Split video/audio from sampler output
- **VAEDecodeTiled**: Decode video latent to pixels (tiled for memory)
- **VHS_VideoCombine**: Save output
  - frame_rate: 25
  - format: video/h264-mp4
  - Wire audio from the loaded video (or from a separate audio file)

## Refinement step tuning

If 3 steps OOMs:

| Steps | Sigmas | Quality | VRAM |
|-------|--------|---------|------|
| 3 | [0.85, 0.725, 0.4219, 0.0] | Best | Highest |
| 2 | [0.85, 0.4219, 0.0] | Good | Less |
| 1 | [0.85, 0.0] | Acceptable | Minimal |

Start with 3. Drop to 2 if OOM. Drop to 1 as last resort.

## Important notes

- The upscaled video should match the loop output exactly in content --
  refinement only adds detail/sharpness, it doesn't change the scene.
- Use the SAME prompt as your loop workflow for conditioning. A different
  prompt during refinement would shift the content.
- Audio passes through unchanged. Wire it from the VHS_LoadVideo directly
  to VHS_VideoCombine.
- This workflow processes the ENTIRE video at once. For a 2:48 video at
  25fps = 4200 frames. If VAEEncode OOMs, switch to VAEEncodeTiled
  with a VHS_BatchManager to process in chunks.

## Expected results

| Metric | Loop output | After upscale |
|--------|-------------|---------------|
| Resolution | 832x480 | 1664x960 |
| Sharpness | Native | Enhanced |
| Detail | Good | Better (faces, text, textures) |
| File size | ~50-100MB | ~200-400MB |
| Processing time | 8-10 min (loop) | ~3-5 min (upscale) |
