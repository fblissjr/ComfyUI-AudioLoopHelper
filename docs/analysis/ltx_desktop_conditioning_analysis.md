Last updated: 2026-04-16

# LTX-Desktop: Conditioning Capabilities Analysis

Analysis of LTX-Desktop's conditioning system, focused on capabilities we can
borrow or adapt for the AudioLoopHelper ComfyUI workflow.

Source: `coderef/LTX-Desktop/backend/`

## Key Concepts We Can Borrow

### 1. ModalitySpec: Independent Audio/Video Control

LTX-Desktop uses `ModalitySpec` objects to independently control each modality:

```python
ModalitySpec(
    context=...,           # text conditioning context
    conditionings=[...],   # list of TemporalRegionMask, image conds, etc.
    initial_latent=...,    # starting latent (encoded video/audio)
    frozen=True/False,     # if True, latent passes through unmodified
    noise_scale=0.0-1.0,   # how much noise to add
)
```

**What this enables**: Audio can be frozen (`frozen=True, noise_scale=0.0`)
while video is denoised normally. Each modality has its own conditioning list.

**ComfyUI equivalent**: We achieve this manually via `LTXVAudioVideoMask`
with `audio_start_time == audio_end_time` (zero-width mask = all frozen).
The Desktop's explicit `frozen` flag is cleaner but functionally identical.

### 2. TemporalRegionMask: Time-Windowed Regeneration

The retake pipeline uses `TemporalRegionMask(start_time, end_time, fps)` to
define which temporal region gets noised/denoised:

```python
# Only regenerate video from 1:30 to 1:48
video_modality_spec = ModalitySpec(
    conditionings=[TemporalRegionMask(start_time=90.0, end_time=108.0, fps=25)],
    initial_latent=encoded_full_video,
    frozen=False,
)
```

Outside the mask, the initial latent is preserved (frozen).

**What this enables**: Fix one bad iteration without re-rendering the entire
video. Encode the full output, define a time window, regenerate just that
section with a different prompt/seed.

**ComfyUI gap**: No equivalent node exists. This is the #1 missing capability
for our workflow. A "retake" node would need to:
1. Accept the full accumulated video latent
2. Accept start_time/end_time
3. Create a noise mask that's 1.0 in the window, 0.0 outside
4. Feed through the standard sampler

This could be built as a ComfyUI node without modifying the model.

### 3. Multi-Image Guides at Arbitrary Frame Indices

Desktop's API type:
```python
ImageConditioningInput = NamedTuple('ImageConditioningInput', [
    ('path', str), ('frame_idx', int), ('strength', float)
])
```

Pipelines accept `images: list[ImageConditioningInput]` — multiple images
at different frame positions, each with independent strength.

**ComfyUI equivalent**: `LTXVAddGuideAdvanced` with `frame_idx` parameter.
Can be chained for multiple guides. Our workflow currently uses a single
guide via `LTXVAddLatentGuide`. The underlying capability exists; we just
don't wire it.

**Desktop limitation**: The standard generation API only sends a single image
at frame_idx=0. Only IC-LoRA exposes the full multi-image capability.
So even Lightricks doesn't fully use this in their standard pipeline.

### 4. Two-Stage A2V Pipeline

Desktop's distilled A2V pipeline runs two stages:
- **Stage 1**: Generate at half resolution (H/2 x W/2)
- **Stage 2**: Upsample 2x + refine with distilled sigma steps

Audio is frozen in both stages. The initial latent for Stage 2 is the
upsampled output of Stage 1.

**Relevance**: Our loop workflow generates at full resolution per iteration.
A two-stage approach could improve quality: generate at 416x240, upsample
to 832x480, refine. But this doubles per-iteration compute and requires
careful noise schedule management across stages.

**Not recommended for Phase 1** — the upscale workflow is already separate.

### 5. Gap Prompt Suggestion (Gemini Integration)

Desktop uses Gemini 2.0 Flash to suggest prompts for gaps between clips on
the timeline. It analyzes before/after frames and prompts to generate a
bridging prompt.

**Relevance**: Our `analyze_audio_features.py` already generates LLM-ready
JSON with a system prompt. We could feed keyframe images + audio features to
an LLM to generate both prompt AND keyframe image schedules together. This
is a workflow/tooling improvement, not a node change.

### 6. Retake Modes

Desktop supports three retake modes:
- `replace_audio_and_video` — regenerate both modalities in the time window
- `replace_video` — regenerate video only, keep original audio
- `replace_audio` — regenerate audio only, keep original video

**Relevance for our workflow**: After generating a full music video, users
often want to fix one bad section. Currently they must re-render from that
iteration onward. A retake node would allow fixing just that section.

## Actionable Items for ComfyUI

### High Priority (directly improves workflow)
1. **RetakeNode** — encode full video + define time window + regenerate.
   Uses the `TemporalRegionMask` concept but implemented as noise mask
   manipulation in ComfyUI's existing sampler infrastructure.

### Medium Priority (quality improvement)
2. **Multi-guide subgraph** — chain a second `LTXVAddLatentGuide` for
   start+end guides per iteration window.
3. **ModalitySpec-aware nodes** — expose `frozen` and `noise_scale` as
   explicit controls rather than relying on the mask trick.

### Low Priority (nice to have)
4. **Gap prompt suggestion** — LLM-assisted keyframe+prompt planning.
5. **Two-stage per-iteration** — half-res generate + upsample + refine.
