Last updated: 2026-04-14

# LTX 2.3 Model Reference

Technical reference for LTX 2.3 model behavior relevant to the audio loop
workflow. Extracted from CLAUDE.md for progressive disclosure -- read this
when working on model-facing code, not for every conversation.

## How image guides actually work

Guide strength does NOT control how much the image influences style.
It controls the denoise mask (noise addition), which is only one of three
layers. Text conditioning operates on a separate, unattenuated pathway:

```
1. Cross-attention (text -> all tokens)  <- ALWAYS FULL STRENGTH, no per-guide control
2. Self-attention (guide <-> generated)  <- controlled by attention_strength (default 1.0)
3. Denoise mask (noise addition)         <- controlled by guide strength (1.0 = no noise)
```

- strength=1.0 -> denoise_mask=0.0 -> guide frames spatially frozen
- BUT cross-attention still pulls style/appearance toward text description
- Guides are CONCATENATED to the latent sequence (extra frames at the end),
  not blended at the target index. keyframe_idxs tells RoPE their logical position.
- This is why changing text causes style drift even at guide strength 1.0:
  the guide anchors composition, but text controls style via cross-attention.
- The right fix for style consistency is keeping text aligned (consistent
  prompts + ConditioningBlend), not increasing guide strength.

Source: ComfyUI-LTXVideo/latents.py (LTXVAddLatentGuide),
comfy_extras/nodes_lt.py (append_keyframe), comfy/ldm/lightricks/model.py
(per-reference attention masking).

## Video/Audio VAE temporal conversion

- **Video VAE**: First pixel frame -> own latent frame, then 8 pixels per latent.
  Formula: `latent = (pixel - 1) // 8 + 1`. NOT `pixel // 8`.
  Pixel frames must follow 8n+1 (1, 9, 17, 25, 497...).
- **Audio VAE**: 25 latents/second, 1D, completely independent of video latent
  temporal dimension. They live in separate NestedTensor sub-tensors.
- Using `pixel // 8` instead of `(pixel - 1) // 8 + 1` caused the v0409
  sync bug: 25 pixels -> 3 latent frames (wrong) vs 4 (correct).

## Resolution and latent volume

- **Latent volume limit**: `(width/32) * (height/32) * ((frames-1)/8 + 1)` should
  stay below ~15,000-20,000. Exceeding causes artifacts, grid patterns, color loss.
- **832x480 at 497 frames**: 26*15*63 = 24,570 -- already at the edge. Don't increase
  resolution without reducing frame count per window.
- **Higher resolution improves motion/lip-sync/audio quality** but costs more VRAM
  and risks latent volume overflow. 720p+ with 48-50fps gives smoother motion.
- **Portrait (vertical) resolutions are unstable** -- keep height < 1600px.
  Landscape and square work best.
- **Two-stage approach is the recommended workaround**: generate at lower res (720p),
  then spatial latent upscale to 1080p+. This is what LTX-Desktop and native LTX-2
  both do. See `upscale_guide.md` and `analysis/ltx23_gaps_analysis.md`.
- For our loop workflow: each window is 497 frames at 832x480. Changing resolution
  requires adjusting window_seconds or temporal_tile_size to stay under the limit.

## Color drift prevention (AdaIN)

Loop iterations progressively darken because each iteration's latent statistics
drift from the initial render. The init_image guide anchors composition but not
color -- guide strength controls the denoise mask, not cross-attention style.

Two AdaIN approaches (can be used together or independently):

**Per-iteration AdaIN** (LTXVAdainLatent, inside subgraph):
- Location: after SeparateAVLatent (#596), before CropGuides (#655)
- Reference: initial render video latent from SeparateAV #245
- Factor: 0.2 default (gentle). Increase to 0.5 for stronger correction.
- per_frame=False (global statistics). Try True if per-frame flickering occurs.
- Present in all three workflows. Bypass (mode=4) to disable.

**Per-step AdaIN** (LTXVPerStepAdainPatcher, model chain):
- Location: after SamplingPreviewOverride, before Set_model
- Reference: node 531 (init image embed latent, available before sampling)
- Factors: per-denoising-step, e.g., "0.3,0.2,0.1,0.05,0.0,0.0,0.0,0.0"
  (stronger at early noisy steps, none at late detail steps)
- Only in `audio-loop-music-video_image_adain_perstep.json`
- More aggressive than per-iteration. Applied during sampling, not after.

**Testing order**: Start with per-iteration only (factor=0.2). If drift persists,
try the per-step workflow. Compare iteration 5+ brightness against initial render.

## LTX 2.3 prompt format

- LTX 2.3 is distilled -- CFG=1.0 by default (NAG handles guidance, not CFG).
- Prompts are i2v (image-to-video) style: describe changes from the init_image, not the full scene.
- Start with `Style: cinematic.` (or omit if init_image establishes style).
- Use present-progressive verbs: "is singing," "is walking."
- Include audio descriptions inline with visuals (LTX 2.3 is audio-video joint).
- No meta-language: no "The scene opens with...", no timestamps, no cuts.
- Camera motion only when intended. Keywords: `static camera`, `dolly in/out/left/right`, `jib up/down`, `focus shift`.
- **Avoid dolly out** -- breaks limbs and faces. Use static camera with lighting shifts for visual variation.
- **i2v rule: describe only changes from the init_image.** Re-describing the setting causes the model to "restart" the scene.
- **Two-person scenes: always "singing together."** Don't direct male vs female vocals -- audio conditioning handles it.
- **Subject anchoring, not setting re-description.** Describe WHO (traits,
  clothing, position) in every entry to anchor identity. Do NOT re-describe
  the environment -- that's in the init_image.
- **Node 169 covers trimmed 0:00 to window_seconds (~20s).** TimestampPromptSchedule
  does NOT run during the initial render. Node 169 prompt MUST match the
  schedule's 0:00 entry to avoid visual discontinuity at ~20s.

Full system prompts: `ltx23_prompt_system_prompts.md`
Prompt creation guide: `prompt_creation_guide.md`

## Conditioning path

- LTX 2.3 uses Gemma 3 text encoder (NOT CLIP). Conditioning format is
  `[tensor, {"attention_mask": mask}]` with no pooled_output. Standard
  ConditioningAverage won't work -- use our ConditioningBlend instead.
- Workflow uses DualCLIPLoader + CLIPTextEncode nodes. Despite the names,
  these are Gemma 3 encoders (loaded via gemma_3_12B + ltx-2.3_text_projection).
- Extension #843 positive/negative should come from Get_base_cond_pos/neg
  DIRECTLY, NOT through an extra LTXVConditioning node. Node 1587 is bypassed
  because it corrupted the initial render's audio-video cross-attention.
- **Blending wiring (fully connected in all 3 workflows)**:
  ```
  TimestampPromptSchedule (1558)
    |-> prompt -> CLIPTextEncode A (1559) -> ConditioningBlend.conditioning_a
    |-> next_prompt -> CLIPTextEncode B (1604*) -> ConditioningBlend.conditioning_b
    |-> blend_factor -> ConditioningBlend.blend_factor
                              -> Extension #843 input 6 (positive)
  ```
  blend_seconds=0 means hard switch. blend_seconds=5.0 smoothly transitions.

## Initial render path (critical for sync)

- TensorLoopOpen MUST receive the sampled initial render, NOT the raw
  image-embed latent from LTXVImgToVideoInplaceKJ.
- LTXVAddLatentGuide APPENDS guide frames to temporal dim (torch.cat dim=2).
  Sampler output latent has shape [B,C,63+N_guides,H,W], not [B,C,63,H,W].
- For LATENT workflows: initial render prepended via LatentConcat (dim=t)
  using CropGuides output (guide-stripped).
- Correct latent path: #531 -> #350 ConcatAV -> #161 Sampler -> #245 SeparateAV -> #1539 TensorLoopOpen

## noise_mask handling (critical for latent-space loop)

- **VAEEncode produces latent with NO noise_mask key.** LTXVAudioVideoMask
  then creates a fresh all-zeros mask. This is the correct behavior.
- **LTXVSelectLatents PRESERVES the existing noise_mask** from its input.
  Inherited stale masks corrupt the sampler's mask semantics and break sync.
- **StripLatentNoiseMask** (our node) removes noise_mask so downstream nodes
  create fresh masks. REQUIRED between LTXVSelectLatents and LTXVAudioVideoMask.
- Use LatentContextExtract and LatentOverlapTrim instead of raw LTXVSelectLatents
  in the latent-space subgraph -- they strip noise_mask automatically.

## Dual workflow support (IMAGE vs LATENT)

- **IMAGE workflow** (`audio-loop-music-video_image.json`): Subgraph uses
  GetImageRangeFromBatch + VAEEncode/Decode. ImageBatch prepends initial render.
- **LATENT workflow** (`audio-loop-music-video_latent.json`): Subgraph uses
  LatentContextExtract + LatentOverlapTrim. LatentConcat prepends initial render.
  No per-iteration VAE round-trip.
- AudioLoopController outputs work for both: overlap_frames (pixel) + overlap_latent_frames (latent).

## Extension subgraph (Node 843)

Per-iteration sampling pipeline. IMAGE and LATENT workflows differ in context
extraction and output trimming nodes. Shared internals:
- VAEEncode (1520) -- encodes init_image to latent (scene anchor guide)
- LTXVAddLatentGuide (1519) -- merges conditioning + both guides into latent
- LTXVConcatAVLatent (583) -- adds audio latent
- CFGGuider (644) -- packages for sampling (cfg=1.0, NAG does guidance)
- SamplerCustomAdvanced (573) -- generates new frames

Full traces: `pipeline_flow_image.md` and `pipeline_flow_latent.md`.

## LTX 2.3 audio-video alignment

- TrimAudioDuration (Node 567) start_index is song-dependent. It trims
  instrumental intro that doesn't contribute to lip sync.
- Audio and video durations must match: 497 frames / 25fps = 19.88s audio.
- LTXVAudioVideoMask (Node 606): audio_start_time and audio_end_time are
  BOTH wired to window_size_seconds (19.88). This creates an empty mask
  range (start=end), so audio stays fixed as the encoded song. DO NOT change.

## DynamicCombo widget format

LTXVImgToVideoInplaceKJ (and similar multi-input nodes) serialize
widgets as: `[num_items, strength_1, strength_2, ..., index_1, index_2, ...]`
Strengths come FIRST for all items, THEN indices. NOT interleaved.
Example: `['2', 1.0, 0.5, 0, -1]` = 2 images, strengths [1.0, 0.5], indices [0, -1].
Getting this wrong silently misconfigures the node.

## Upscaling

- Upscale is a SEPARATE workflow, not part of the loop workflow.
- Stay in latent space: Load video -> VAEEncode (once) -> LTXVLatentUpsampler (2x) ->
  3-step refinement sampler -> VAEDecodeTiled.
- Model: `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- Refinement sigmas: [0.85, 0.725, 0.4219, 0.0] (3 steps).
- Guide: `upscale_guide.md`
