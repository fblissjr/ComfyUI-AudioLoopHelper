Last updated: 2026-04-10

# Build Guide: LTXVLoopingSampler Workflow (VIDEO-ONLY)

**STATUS: VIDEO-ONLY. LTXVLoopingSampler does NOT support AV latents.
For music videos with audio conditioning, use the TensorLoop latent-space
workflow (`audio-loop-music-video_latent.json`) with `subgraph_latent_rework_guide.md` instead.**

Build this from scratch in ComfyUI. No subgraph, no TensorLoop.
LTXVLoopingSampler handles all looping internally in latent space.

## Architecture

```
[Model Loading] → [Model Patches] → [NAG] → model
[Image Loading] → [Resize] → [Preprocess] → [ImgToVideo] → initial latent
[Audio Loading] → [Trim] → [MelBand vocals] → [AudioVAEEncode] → audio latent
                                                     ↓
[Initial latent] + [Audio latent] → [ConcatAVLatent] → combined AV latent
                                                           ↓
[Text Encode] → [Conditioning] → [Guider]                 ↓
[ScheduleToMultiPrompt] → [MultiPromptProvider] → per-tile conditionings
                                                           ↓
                              LTXVLoopingSampler ←─────────┘
                                     ↓
                              LATENT (full video, never decoded between tiles)
                                     ↓
                              [SeparateAVLatent] → [VAEDecodeTiled] → IMAGE
                                     ↓
                              [VHS_VideoCombine] → final mp4
```

## Step-by-step (exact node types and values)

### Group 1: Model Loading

| # | Node Type | Widget Values | Notes |
|---|-----------|---------------|-------|
| 1 | UNETLoader | model: `ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors` | Main diffusion model |
| 2 | DualCLIPLoader | clip1: `gemma_3_12B_it_fpmixed.safetensors`, clip2: `ltx-2.3_text_projection_bf16.safetensors`, type: `ltxv` | Gemma 3 text encoder |
| 3 | VAELoaderKJ | model: `vae/LTX23_video_vae_bf16.safetensors`, device: main_device, dtype: bf16 | Video VAE |
| 4 | VAELoaderKJ | model: `vae/LTX23_audio_vae_bf16.safetensors`, device: main_device, dtype: bf16 | Audio VAE |
| 5 | MelBandRoFormerModelLoader | model: `MelBandRoformer_fp32.safetensors` | Vocal separator |

### Group 2: Model Patch Chain

Wire these in sequence: UNET → each patch → next patch → NAG → final model

| # | Node Type | Widget Values | Input | Output |
|---|-----------|---------------|-------|--------|
| 6 | PatchSageAttentionKJ | (defaults) | MODEL from #1 | MODEL |
| 7 | LTXVChunkFeedForward | chunk_size: 2, threshold: 4096 | MODEL from #6 | MODEL |
| 8 | LTX2AttentionTunerPatch | "", 1, 1, 1, 1, true | MODEL from #7 | MODEL |
| 9 | LTX2_NAG | nag_scale: 11, alpha: 0.25, tau: 2.5, inplace: true | MODEL from #8, nag_cond from negative CLIPTextEncode (#14) | MODEL (final) |

**Note**: Set #6 (SageAttention) to **bypassed (mode 4)** initially. Enable after quality is confirmed.

### Group 3: Image Loading

| # | Node Type | Widget Values | Notes |
|---|-----------|---------------|-------|
| 10 | LoadImage | your reference image | Init frame |
| 11 | ImageResizeKJv2 | width: 832, height: 480, method: lanczos, crop | Standardize resolution |
| 12 | LTXVPreprocess | brightness: 0 | LTXV format prep |

### Group 4: Audio Loading & Processing

| # | Node Type | Widget Values | Notes |
|---|-----------|---------------|-------|
| 13 | LoadAudio | your audio file | Raw audio |
| 14 | TrimAudioDuration | start_index: 10, duration: 300 | Trim intro (adjust per song) |
| 15 | MelBandRoFormerSampler | | Inputs: model from #5, audio from #14. Outputs: vocals, instruments |

### Group 5: Text Encoding & Conditioning

| # | Node Type | Widget Values | Notes |
|---|-----------|---------------|-------|
| 16 | CLIPTextEncode | Your positive prompt | Positive. Wire clip from #2 |
| 17 | CLIPTextEncode | Full negative prompt (see below) | For NAG. Wire clip from #2 |
| 18 | ConditioningZeroOut | | Input: positive from #16. Output: zeroed conditioning (used as CFG negative) |
| 19 | LTXVConditioning | frame_rate: 25 | Input: positive from #16, negative from #18. Adds frame_rate metadata |

**Negative prompt for #17:**
```
still image with no motion, subtitles, text, scene change, blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, inconsistent perspective, camera shake, incorrect depth of field, face swap, merged faces, wrong number of people, third person appearing
```

### Group 6: Initial Latent Setup

This is where the video's temporal size is defined. The latent needs to be
large enough for the FULL audio duration.

| # | Node Type | Widget Values | Notes |
|---|-----------|---------------|-------|
| 20 | EmptyLTXVLatentVideo | width: 832, height: 480, length: COMPUTE (see below), batch: 1 | Base video latent |
| 21 | LTXVImgToVideoInplaceKJ | num_images: "1", strength_1: 1.0, index_1: 0 | Embed init image at frame 0. Inputs: VAE from #3, latent from #20, image from #12 |
| 22 | LTXVAudioVAEEncode | | Input: vocals audio from #15, audio VAE from #4 |
| 23 | LTXVConcatAVLatent | | Input: video latent from #21, audio latent from #22. Output: combined AV latent |

**Computing latent length for #20:**
```
audio_duration_after_trim = total_audio_seconds - trim_start
num_pixel_frames = round(audio_duration_after_trim * 25)  # 25 fps
# Must satisfy (frames - 1) % 8 == 0
num_pixel_frames = ((num_pixel_frames - 1) // 8) * 8 + 1
```
Example: 168s audio after trim → 168 * 25 = 4200 → ((4200-1)//8)*8+1 = 4193 frames

### Group 7: Prompt Scheduling (our nodes)

| # | Node Type | Widget Values | Notes |
|---|-----------|---------------|-------|
| 24 | AudioDuration | | Input: trimmed audio from #14. Output: duration_seconds |
| 25 | ScheduleToMultiPrompt | stride: 18.88, window: 19.88, schedule: your timestamp prompts | Input: audio from #14. Output: pipe-separated prompts |
| 26 | MultiPromptProvider | | Input: prompts string from #25, clip from #2. Output: conditioning list |
| 27 | AudioLoopPlanner | stride: 18.88, window: 19.88 | Input: audio from #14. Output: summary (wire to PreviewAny) |

### Group 8: Sampler Setup

| # | Node Type | Widget Values | Notes |
|---|-----------|---------------|-------|
| 28 | ModelSamplingSD3 | shift: 13 | Input: MODEL from #9 |
| 29 | KSamplerSelect | sampler: euler_ancestral | Try euler if jitter |
| 30 | ManualSigmas | sigmas: `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0` | 8-step distilled schedule |
| 31 | RandomNoise | seed: your seed | |
| 32 | CFGGuider | cfg: 1.0 | Input: model from #28, positive from #19 output 0, negative from #19 output 1 |

### Group 9: LTXVLoopingSampler (THE CORE)

| # | Node Type | Widget Values |
|---|-----------|---------------|
| 33 | **LTXVLoopingSampler** | temporal_tile_size: **497**, temporal_overlap: **25**, temporal_overlap_cond_strength: **0.5**, cond_image_strength: **1.0**, horizontal_tiles: 1, vertical_tiles: 1, spatial_overlap: 1 |

**Inputs:**
- model: from #28 (ModelSamplingSD3 output)
- vae: from #3 (video VAE)
- noise: from #31 (RandomNoise)
- sampler: from #29 (KSamplerSelect)
- sigmas: from #30 (ManualSigmas)
- guider: from #32 (CFGGuider)
- latents: from #23 (ConcatAVLatent -- combined video+audio latent)
- optional_cond_images: from #12 (preprocessed init image)
- optional_positive_conditionings: from #26 (MultiPromptProvider)

**Output:** LATENT (full video, all tiles processed in latent space)

### Group 10: Decode & Output

| # | Node Type | Widget Values | Notes |
|---|-----------|---------------|-------|
| 34 | LTXVSeparateAVLatent | | Input: latent from #33. Output: video_latent, audio_latent |
| 35 | VAEDecodeTiled | (defaults or tile settings) | Input: video latent from #34, vae from #3. THE ONLY VAE DECODE. |
| 36 | VHS_VideoCombine | frame_rate: 25, format: video/h264-mp4, pix_fmt: yuv420p, crf: 19, trim_to_audio: true, save_output: true | Input: images from #35, audio from #14 (trimmed full mix, NOT vocals) |

**Important**: Wire the FULL MIX audio (from #14 TrimAudioDuration, not from MelBand vocals) to VHS_VideoCombine. The viewer hears the full song, not just vocals.

---

## Total Node Count: ~36

Compare to the TensorLoop workflow: ~50+ nodes including the subgraph internals.

## Key Differences from TensorLoop Workflow

| Aspect | TensorLoop (old) | LTXVLoopingSampler (new) |
|--------|-------------------|--------------------------|
| Loop mechanism | TensorLoopOpen/Close + subgraph | Built into LTXVLoopingSampler |
| VAE decodes per iteration | 1 (lossy) | 0 (all latent) |
| VAE encodes per iteration | 1 (context frames) | 0 (all latent) |
| Total VAE decodes | N+1 | 1 |
| Prompt scheduling | TimestampPromptSchedule per iteration | ScheduleToMultiPrompt upfront |
| Stop signal | AudioLoopController should_stop | Not needed (latent sized for full audio) |
| Overlap handling | Manual subgraph nodes (615, 1509) | Built-in temporal_overlap |
| Extension subgraph | Required (complex) | Not needed |
| Quality over iterations | Degrades (VAE round-trip) | Constant (all latent) |

## Settings Reference

See `ltxv_looping_sampler_settings.md` for detailed parameter explanations
and tuning guidance.

| Parameter | Value | Where |
|-----------|-------|-------|
| temporal_tile_size | 497 | #33 LTXVLoopingSampler. Pixel frames per window (19.88s * 25fps). |
| temporal_overlap | 25 | #33. Overlap frames (1s at 25fps). Increase to 50 for more coherence. |
| temporal_overlap_cond_strength | 0.5 | #33. How strongly previous tile conditions next. Increase if drift. |
| cond_image_strength | 1.0 | #33. Init image influence. |
| adain_factor | 0.0 | #33. Set 0.1-0.3 if colors oversaturate across tiles. |
| shift | 13 | #28 ModelSamplingSD3. Try 9 for smoother results. |
| sampler | euler_ancestral | #29. Try euler for more consistency. |
| sigmas | distilled 8-step | #30. Don't change unless experimenting. |
| cfg | 1.0 | #32. LTX 2.3 is distilled, NAG handles guidance. |

## Optional: Upscale Chain

After #34 (SeparateAVLatent), before #35 (VAEDecodeTiled):

| # | Node Type | Notes |
|---|-----------|-------|
| 34b | LTXVLatentUpsampler | Model: `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`. Input: video latent from #34. 2x upscale. |
| 34c | ManualSigmas | sigmas: `0.85, 0.725, 0.4219, 0.0` (3-step refinement) |
| 34d | SamplerCustomAdvanced | Refine the upscaled latent. 3 steps. |

Then wire #34d output to #35 instead of #34 directly.
