Last updated: 2026-04-16

# LTX-2 Native: Conditioning Types and Pipeline Analysis

Analysis of LTX-2's native conditioning system (Lightricks' canonical reference
implementation), focused on what's available for multi-frame conditioning.

Source: `coderef/LTX-2/packages/ltx-core/src/ltx_core/` and
`coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/`

## Three Conditioning Types

### VideoConditionByLatentIndex
- Location: `ltx_core/conditioning/types/latent_cond.py:9`
- **Replaces tokens in-place** at a specific latent frame index.
- Sets `denoise_mask = 1.0 - strength` at those positions.
- This is the "overwrite" approach — guide tokens directly replace latent
  tokens at the target position.
- ComfyUI equivalent: `LTXVImgToVideoInplaceKJ` (KJNodes)

### VideoConditionByKeyframeIndex
- Location: `ltx_core/conditioning/types/keyframe_cond.py:10`
- **Appends keyframe tokens** to the latent sequence (not in-place).
- Uses pixel coordinates with `frame_idx` offset for RoPE positioning.
- Updates attention mask to include new tokens.
- This is the "concatenate" approach — the ComfyUI `LTXVAddGuide.append_keyframe`
  pattern. Multiple keyframes accumulate correctly.
- ComfyUI equivalent: `LTXVAddGuide`, `LTXVAddLatentGuide`, `LTXVAddGuideAdvanced`

### VideoConditionByReferenceLatent
- Location: `ltx_core/conditioning/types/reference_video_cond.py:12`
- **Appends an entire reference video** as conditioning tokens.
- Supports `downscale_factor` for lower-resolution references (e.g., 384px
  reference for 768px output). Spatial positions are scaled accordingly.
- Used for IC-LoRA inference — the reference video becomes the "control signal"
  the LoRA learned to attend to during training.
- ComfyUI equivalent: partial via IC-LoRA nodes, but no dedicated
  "reference video conditioning" node.

**Key insight**: All three types follow the same `ConditioningItem.apply_to()`
protocol. They can be combined freely in a single pipeline call via the
`conditionings` list in `ModalitySpec`.

## MultiModalGuiderFactory: Per-Step Guidance

- Location: `ltx_core/components/guiders.py:290`
- Creates `MultiModalGuider` instances parameterized by sigma (noise level).
- Each `MultiModalGuiderParams` controls:
  - `cfg_scale`: classifier-free guidance strength
  - `stg_scale`: spatio-temporal guidance strength
  - `stg_blocks`: which transformer blocks to perturb
  - `rescale_scale`: anti-oversaturation rescaling
  - `modality_scale`: per-modality weighting
  - `skip_step`: step skipping frequency

The factory maps sigma ranges to different parameter sets via
`_params_for_sigma_from_sorted_dict()`. This means guidance can be aggressive
at high noise (early steps) and gentle at low noise (late steps).

**ComfyUI gap**: No sigma-dependent guidance variation. The distilled model
(CFG=1.0 + NAG) mitigates this somewhat, but for non-distilled models, this
would improve quality at no speed cost.

## KeyframeInterpolationPipeline

- Location: `ltx_pipelines/keyframe_interpolation.py:42`
- Two-stage pipeline that interpolates between keyframe images.
- Stage 1: half-resolution generation with full model
- Stage 2: 2x spatial upsample + distilled refinement

Uses `image_conditionings_by_adding_guiding_latent()` helper (the append
approach, not in-place) to add multiple keyframe images at different frame
indices. The model then generates video that smoothly transitions between
the keyframes.

**Relevance**: This is the closest native equivalent to what we're building.
The pipeline takes multiple images at specified frames and generates video
between them. Our `KeyframeImageSchedule` + per-iteration guide switching
achieves a similar effect through temporal chunking rather than single-pass
generation.

## RetakePipeline

- Location: `ltx_pipelines/retake.py:153`
- Uses `TemporalRegionMask(start_time, end_time, fps)` to define the
  regeneration window within an existing video.
- Encodes the full source video/audio into latent space.
- Applies noise only in the masked temporal region.
- Three modes: replace video+audio, video only, audio only.

**Key mechanism**: `TemporalRegionMask` creates a denoise mask where
`mask=1.0` in the specified time window and `mask=0.0` outside. The sampler
denoises only where mask=1.0, preserving the rest.

**ComfyUI opportunity**: This could be built as a post-loop "fix" node:
1. Accumulate all iteration outputs into a full video latent
2. Define a time window to regenerate
3. Apply TemporalRegionMask-style noise masking
4. Run through the sampler

## A2VidPipelineTwoStage

- Location: `ltx_pipelines/a2vid_two_stage.py:40`
- Audio-to-video with two stages (half-res -> upsample -> refine).
- Audio is frozen via `ModalitySpec(frozen=True, noise_scale=0.0)`.
- Image conditionings applied as usual.

**Pattern we already use**: Our workflow achieves frozen audio via the
`audio_start_time == audio_end_time` mask trick in `LTXVAudioVideoMask`.
The native code uses an explicit `frozen` flag, which is cleaner.

## What We Can Leverage

### Already leveraged (via ComfyUI nodes)
1. KeyframeIndex conditioning (append approach) — `LTXVAddLatentGuide`
2. Multiple keyframes in sequence — chainable in ComfyUI
3. Frozen audio via denoise mask — `LTXVAudioVideoMask` trick

### Available but not yet used
1. **LatentIndex conditioning** (in-place approach) — `LTXVImgToVideoInplaceKJ`
   could be used for stronger spatial anchoring at specific frames
2. **Reference video conditioning** — IC-LoRA nodes exist but aren't wired
   for per-iteration reference switching
3. **Multiple images per generation** — `LTXVAddGuideMulti` (KJNodes)
   already supports up to 20 guides in a single node

### Not available in ComfyUI
1. **TemporalRegionMask** — time-windowed retake/regeneration
2. **Per-sigma guidance** — MultiModalGuiderFactory's sigma-dependent params
3. **Two-stage per-window** — half-res + upsample + refine per iteration
4. **Explicit frozen modality flag** — ModalitySpec.frozen
