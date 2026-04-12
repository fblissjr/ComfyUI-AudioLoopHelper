Last updated: 2026-04-12

# End-to-End Workflow Pipeline Trace (HISTORICAL)

**SUPERSEDED by `docs/pipeline_flow_image.md` and `docs/pipeline_flow_latent.md`
which are auto-generated, more detailed, and reflect current workflow state
including the node 1587 bypass, AdaIN additions, and overlap_seconds auto-wiring.**

Original source: kijai's `ltx23_long_loop_extension_test.json`
Current version: `example_workflows/audio-loop-music-video_image.json`

---

## 1. Text Encoding Pipeline

### 1a. CLIP Model Loading (Node 416: DualCLIPLoader)

Loads two models that work together as the text encoder:
- `gemma_3_12B_it_fp8_scaled.safetensors` -- Gemma 3 12B language model (text understanding)
- `LTX23_text_projection_bf16.safetensors` -- LTX 2.3 text projection layer (maps Gemma embeddings to LTX's conditioning space)
- Projection type: `ltxv`

Output: CLIP model object -> feeds both Node 169 and Node 507

### 1b. Positive Prompt Encoding (Node 169: CLIPTextEncode)

- Text: `"video of a woman passionately singing in a studio alone"`
- Input: CLIP from Node 416
- Output: CONDITIONING -> Node 164 (positive input) AND Node 420

### 1c. Negative Conditioning (Node 420: ConditioningZeroOut)

- Takes positive conditioning from Node 169 and zeros it out
- This creates a "null" conditioning -- no text guidance for the negative path
- Output: CONDITIONING -> Node 164 (negative input)

**Why zero instead of a negative prompt?** Because NAG (Node 508) handles the negative guidance at the attention level. CFG negative is zeroed out so NAG does all the "what to avoid" work without interference.

### 1d. LTXVConditioning (Node 164)

- Inputs: positive from 169, negative from 420
- **frame_rate: 25 fps** -- critical metadata for the video model's temporal understanding
- Outputs both positive and negative conditioning with frame_rate baked in
- Stored via SetNodes:
  - Node 645: `"base_cond_pos"` -- positive conditioning with frame_rate
  - Node 646: `"base_cond_neg"` -- negative (zeroed) conditioning with frame_rate
- Also feeds directly to Node 153 (CFGGuider) and Node 381 (LTXVCropGuides)

### 1e. NAG Conditioning (Node 507: CLIPTextEncode)

- Text: `"still image with no motion, subtitles, text, scene change, instruments, violin"`
- Input: CLIP from Node 416
- Output: CONDITIONING -> Node 508 (nag_cond_video input)
- See `nag_technical_reference.md` for why this prompt matters

### 1f. NAG Model Patching (Node 508: LTX2_NAG)

- **nag_scale: 11** -- strong negative guidance
- **alpha: 0.25** -- 25% NAG, 75% original attention
- **tau: 2.5** -- clipping threshold
- **inplace: True** -- memory optimization
- Input model: from Node 1523 (LTX2AttentionTunerPatch, which itself received the model from earlier patches)
- Output: MODEL with NAG cross-attention patches applied -> Node 503

**What this does:** Patches every cross-attention layer in the transformer to run dual attention (positive + negative contexts) and blend them using the NAG formula. The model itself is modified -- every forward pass will now apply NAG guidance.

### 1g. Prompt Scheduling Path (inside loop body)

When TimestampPromptSchedule is enabled, the positive conditioning path
changes per iteration:

```
TensorLoopOpen (1539) current_iteration
  -> TimestampPromptSchedule (1558): selects prompt based on audio position
     -> prompt + next_prompt (STRING outputs)
  -> CLIPTextEncode (1559 "Loop Prompt Encode"): encodes current prompt
     [optional: second CLIPTextEncode for next_prompt -> ConditioningBlend]
  -> LTXVConditioning (1587, frame_rate=25): adds frame_rate metadata
  -> Extension subgraph 843 positive input
```

Key: LTXVConditioning (1587) is REQUIRED between text encode and the
subgraph. Without it, positive conditioning lacks frame_rate metadata
and mismatches the negative, causing the model to mishandle conditioning.

When ConditioningBlend is used (blend_seconds > 0):
```
  prompt -> CLIPTextEncode A -> conditioning_a
  next_prompt -> CLIPTextEncode B -> conditioning_b
  blend_factor -> ConditioningBlend -> LTXVConditioning (1587) -> subgraph
```

---

## 2. Model Pipeline

The model passes through several patches before reaching the sampler:

```
Node 414: UNETLoader (loads LTX 2.3 model)
  -> Node 268: PatchSageAttentionKJ (attention optimization)
  -> Node 504: LTXVChunkFeedForward (chunked feed-forward for memory)
  -> Node 1523: LTX2AttentionTunerPatch (attention tuning)
  -> Node 508: LTX2_NAG (normalized attention guidance) 
  -> Node 503: LTX2SamplingPreviewOverride
  -> Node 572: SetNode ("model") -- stored for retrieval
  -> Node 153: CFGGuider (receives model)
```

---

## 3. Image Processing Pipeline

### 3a. Image Loading (Node 444: LoadImage)

- File: `multitalk_single_example.png`
- Output: IMAGE + MASK -> Node 445

### 3b. Image Resizing (Node 445: ImageResizeKJv2)

- Target: **832 x 480** (16:9-ish aspect ratio)
- Method: `lanczos` (high-quality downscaling)
- Crop type: `crop` (center crop to exact dimensions)
- Oversample: 2 (for cleaner downscaling)
- Outputs:
  - IMAGE -> Node 446 (preprocessing) AND Node 650 (SetNode for later retrieval as guide)
  - Width (832) -> Node 344 (empty latent)
  - Height (480) -> Node 344 (empty latent)

**Why 832x480?** LTX 2.3 works best at these dimensions for real-time generation. Spatial upscaling can happen later, but generation quality is optimized at this resolution.

### 3c. LTXVPreprocess (Node 446)

- **compression: 10** (NOT 0)
- Applies H.264 video codec compression (CRF=10, which is near-lossless)
- Source: `ComfyUI/comfy_extras/nodes_lt.py` lines 590-614

**What this does:** Encodes the image as a single-frame MP4 at CRF 10 quality, then decodes it back. This simulates video codec artifacts that the LTX model was trained on. CRF 10 is very high quality (near-lossless), so the image looks almost identical but has the subtle characteristics of video-sourced content.

**CRF scale reference:**
- 0 = bypass (no compression)
- 10 = near-lossless (used here)
- 18-28 = typical streaming quality
- 35 = default, noticeable compression

### 3d. Empty Latent Video (Node 344: EmptyLTXVLatentVideo)

- Width: **832** (from Node 445)
- Height: **480** (from Node 445)
- Frames: **497** (from Node 526 PrimitiveNode)
- Batch: 1
- Output: LATENT (empty video latent of the right dimensions) -> Node 531

### 3e. LTXVImgToVideoInplaceKJ (Node 531)

- Source: `ComfyUI-KJNodes/nodes/ltxv_nodes.py` lines 1021-1132
- **num_images: 1**
- **strength: 1** (full conditioning -- the first frame IS the input image)
- **index_1: 0** (insert at frame 0 -- the very first frame)
- Inputs:
  - VAE from GetNode 413 (`"video_vae"`)
  - Latent from Node 344 (empty video)
  - Image from Node 446 (preprocessed)

**What this does:**
1. VAE-encodes the preprocessed image to latent space
2. REPLACES frame 0 of the empty latent with the encoded image
3. Sets `noise_mask[frame_0] = 1.0 - strength = 0.0` (zero noise at frame 0 = fully determined)
4. The rest of the latent remains empty with noise_mask = 1.0 (fully noised = to be generated)

**Result:** A video latent where frame 0 is the input image and all subsequent frames are noise. The model will generate video starting from this fixed first frame.

---

## 4. Audio Pipeline

### 4a. Audio Loading and Encoding

```
Node 565: LoadAudio -> audio file
Node 601: TrimAudioDuration (0-10 seconds)
Node 566: LTXVAudioVAEEncode (audio -> audio latent)
Node 571: SolidMask (creates uniform mask)
Node 570: SetLatentNoiseMask (applies mask to audio latent)
```

### 4b. Audio-Video Latent Concatenation (Node 350: LTXVConcatAVLatent)

- Source: `ComfyUI/comfy_extras/nodes_lt.py` lines 618-650 (native ComfyUI node)
- **video_latent** from Node 531 (image-at-frame-0 latent)
- **audio_latent** from Node 570 (encoded audio with noise mask)

**What this does:** Creates a `NestedTensor` containing both video and audio latents as a tuple, NOT a dimensional concatenation. The model can process both modalities jointly while keeping their different shapes separate.

```python
output["samples"] = NestedTensor((video_samples, audio_samples))
output["noise_mask"] = NestedTensor((video_mask, audio_mask))
```

---

## 5. Sampling Pipeline

### 5a. Sigma Schedule

```
Node 1513: ModelSamplingSD3 (timestep_type: 13) -- configures noise schedule
Node 1421: BasicScheduler
  - scheduler: linear_quadratic
  - num_steps: 8
  - denoise: 1.0
Node 1422: VisualizeSigmasKJ (pass-through + visualization)
Node 579: SetNode ("sigmas") -> Node 161
```

8 steps with linear_quadratic schedule at full denoise.

### 5b. CFGGuider (Node 153)

- **CFG scale: 1.0** -- effectively NO classifier-free guidance
- Model from Node 572 (the NAG-patched model)
- Positive conditioning from Node 164[0] (with frame_rate)
- Negative conditioning from Node 164[1] (zeroed, with frame_rate)

**Why CFG scale 1.0?** Because NAG is doing all the negative guidance work at the attention level. CFG would be redundant and could cause artifacts. CFG=1.0 means the guider just passes through the positive prediction unchanged.

### 5c. Noise (Node 1322: RandomNoise)

- Seed from GetNode 1530 (`"start_seed"`)
- Mode: `fixed`

### 5d. Sampler Selection (Node 154: KSamplerSelect)

- Sampler: `euler_ancestral`

### 5e. Main Sampler (Node 161: SamplerCustomAdvanced)

All inputs assembled:
- **noise** from Node 1322
- **guider** from Node 153 (CFG=1.0, NAG-patched model, positive + zeroed-negative conditioning with frame_rate)
- **sampler** from Node 154 (euler_ancestral)
- **sigmas** from Node 579 (8-step linear_quadratic)
- **latent_image** from Node 350 (audio+video latent, with image at frame 0)

**What happens here:** The sampler denoises the latent over 8 steps. Frame 0 has noise_mask=0 (fixed to input image). All other frames have noise_mask=1 (fully generated). The NAG-patched model guides generation away from static frames, text artifacts, scene changes, and face-occluding objects. The positive prompt steers toward "woman singing in a studio."

---

## 6. Post-Sampling (First Pass)

```
Node 161 output -> Node 245: LTXVSeparateAVLatent (split audio/video)
  Video -> Node 381: LTXVCropGuides (remove guide metadata)
    -> Node 1318: VAEDecode (latent -> pixels)
  Audio -> handled separately
```

---

## 7. Loop Extension (v0406 workflow additions)

In the v0406 workflow, the output of the first pass becomes the input to a TensorLoop that extends the video:

```
TensorLoopOpen (Node 1539)
  -> AudioLoopController (Node 1582) -- computes timing, seed, stop condition
  -> TimestampPromptSchedule (Node 1558) -- selects prompt for this timestamp
  -> CLIPTextEncode (Node 1559) -- encodes the scheduled prompt
  -> Subgraph "extension" (Node 843) -- contains:
      -> VAEEncode (1520) -- encodes init_image as guide latent
      -> LTXVAddLatentGuide (1519) -- combines conditioning + image guide
      -> LTXVConcatAVLatent (583) -- adds audio
      -> CFGGuider (644) -- packages for sampling
      -> SamplerCustomAdvanced (573) -- generates next window
  -> TensorLoopClose (Node 1540)
```

### What the Loop Subgraph is MISSING vs. the First Pass

| Step | First Pass (Nodes 164->153->161) | Loop Subgraph (843 internal) |
|------|----------------------------------|------------------------------|
| Text encoding | CLIPTextEncode (169) | CLIPTextEncode (1559) from scheduler |
| LTXVConditioning | Node 164 (frame_rate=25) | **MISSING** |
| Negative conditioning | ConditioningZeroOut (420) | GetNode 648 ("base_cond_neg") -- has frame_rate from first pass |
| CFGGuider | Node 153 (cfg=1.0) | Node 644 (inside subgraph) |
| NAG | Applied to model (508) | Same model enters subgraph -- NAG IS present |
| Image guide | LTXVImgToVideoInplaceKJ (531) at frame 0 | LTXVAddLatentGuide (1519) at frame -1 |

**Gaps FIXED in v0407:**
1. ~~Positive conditioning had NO frame_rate metadata~~ -- Added LTXVConditioning (Node 1587, frame_rate=25) between scheduler CLIPTextEncode (1559) and subgraph (843)
2. ~~Negative conditioning had frame_rate but positive didn't~~ -- Both now flow through the same LTXVConditioning node
3. Added GetNode 1588 ("base_cond_pos", bypassed) as a static-mode alternative -- user can activate and wire it to skip the scheduler entirely

**v0407 loop conditioning path:**
```
TimestampPromptSchedule (1558) -> CLIPTextEncode (1559) -+
                                                          |-> LTXVConditioning (1587, fps=25) -> Subgraph 843
GetNode "base_cond_neg" (648) -------------------------- +
```

---

## 8. Node Quick Reference

### Image Path
| Node | Type | Key Params | Purpose |
|------|------|-----------|---------|
| 444 | LoadImage | -- | Load input image |
| 445 | ImageResizeKJv2 | 832x480, lanczos | Resize for LTX |
| 446 | LTXVPreprocess | compression=10 | Video codec simulation |
| 344 | EmptyLTXVLatentVideo | 832x480, 497 frames | Create empty latent |
| 531 | LTXVImgToVideoInplaceKJ | 1 image, index=0, strength=1 | Bake image at frame 0 |

### Conditioning Path
| Node | Type | Key Params | Purpose |
|------|------|-----------|---------|
| 416 | DualCLIPLoader | gemma3 + ltx_proj | Load text encoders |
| 169 | CLIPTextEncode | positive prompt | Encode positive text |
| 420 | ConditioningZeroOut | -- | Create null negative |
| 164 | LTXVConditioning | frame_rate=25 | Add temporal metadata |
| 507 | CLIPTextEncode | NAG negative prompt | Encode what to avoid |
| 508 | LTX2_NAG | scale=11, alpha=0.25, tau=2.5 | Patch model attention |

### Sampling Path
| Node | Type | Key Params | Purpose |
|------|------|-----------|---------|
| 153 | CFGGuider | cfg=1.0 | Package conditioning (no CFG) |
| 1421 | BasicScheduler | linear_quad, 8 steps | Noise schedule |
| 154 | KSamplerSelect | euler_ancestral | Sampler algorithm |
| 161 | SamplerCustomAdvanced | -- | Run denoising |
| 350 | LTXVConcatAVLatent | -- | Merge audio+video |

### SetNode Storage
| Node | Variable Name | Contains |
|------|--------------|----------|
| 645 | `base_cond_pos` | Positive conditioning with frame_rate |
| 646 | `base_cond_neg` | Zeroed negative conditioning with frame_rate |
| 572 | `model` | NAG-patched model |
| 579 | `sigmas` | 8-step noise schedule |
| 650 | (init image) | Resized input image for guide retrieval |

---

## 9. Overlap and previous_images Mechanics

### How previous_images Flows Through the Extension Subgraph (Node 843)

Each loop iteration, TensorLoopOpen (1539) provides the previous iteration's
output frames. Inside the subgraph:

```
previous_images (TensorLoopOpen output 1)
  |
  v
Node 615: GetImageRangeFromBatch (start_index=-1, num_frames=overlap_frames)
  -- Extracts the LAST overlap_frames from previous output
  |
  v
Node 614: VAEEncode
  -- Encodes tail frames to latent space (continuity guide)
  |
  v
Node 1519: LTXVAddLatentGuide (latent_idx=-1, strength from input 12)
  -- Receives TWO guides:
  --   guiding_latent = init_image (from Node 1520, scene anchor)
  --   previous frames inform the base latent context
  -- Adds keyframe_idxs + guide_attention_entries to conditioning
  |
  v
Node 573: SamplerCustomAdvanced
  -- Generates ~497 frames guided by both previous tail + init image
  |
  v
Node 1509: GetImageRangeFromBatch (start_index=overlap_frames, num_frames=4096)
  -- DISCARDS first overlap_frames (they were context, not new content)
  -- Outputs only genuinely new frames
  |
  v
Subgraph output -> TensorLoopClose (accumulates)
```

overlap_frames (from AudioLoopController output 5) drives BOTH extraction
nodes -- it controls how many frames are pulled from previous output AND
how many frames are trimmed from new output.

### Overlap Seconds Impact

| overlap_seconds | overlap_frames (25fps) | Context | New frames/iter (~497 total) | Tradeoff |
|---|---|---|---|---|
| 1 (default) | 25 | 1s | ~472 | Minimal context, fastest throughput |
| 2 | 50 | 2s | ~447 | Better face/pose continuity |
| 4 | 100 | 4s | ~397 | Strong anchor, 20% of each gen discarded |

More overlap = model sees more prior frames as guide context = stronger
visual continuity at iteration boundaries. Cost: stride shrinks, so more
iterations needed for the same song duration.

### With vs Without Prompt Scheduling

The previous_images mechanism is identical either way. It always extracts
the last N overlap frames, VAE-encodes them, and uses them as guides. The
only difference is what text conditioning accompanies the guide:
- Without scheduler: same prompt every iteration
- With scheduler: different prompt per timestamp

---

## 10. Post-Loop Spatial Upscaler (v0408)

After the loop accumulates all frames at 832x480, a 2x spatial upscaler
chain refines them to ~1664x960 before final video output.

### Upscaler Pipeline

```
ImageBatch (1507) -- accumulated low-res frames
  |
  v
VAEEncode (1590) -- encode frames to latent
  |
  v
LTXVLatentUpsampler (1591) -- learned 2x spatial upscale in latent space
  |                           model: ltx-2.3-spatial-upscaler-x2-1.0.safetensors
  v
SamplerCustomAdvanced (1596) -- 3-step refinement pass
  |   guider: CFGGuider (1594, cfg=1.0, NAG model)
  |   sigmas: ManualSigmas (1592, "0.85, 0.7250, 0.4219, 0.0")
  |   sampler: euler_cfg_pp (deterministic for consistency)
  |   conditioning: base_cond_pos/neg with frame_rate=25
  v
VAEDecodeTiled (1597) -- decode to high-res frames (tile_size=512)
  |
  v
VHS_VideoCombine (617) -- final MP4
```

### Key Settings

| Node | Setting | Value | Why |
|------|---------|-------|-----|
| LTXVLatentUpsampler | model | spatial-upscaler-x2 | Learned upscaling, not bilinear |
| ManualSigmas | schedule | 0.85, 0.725, 0.422, 0.0 | Low denoise, 3 steps -- just refine detail |
| KSamplerSelect | sampler | euler_cfg_pp | Deterministic (no ancestral noise) |
| CFGGuider | cfg | 1.0 | NAG handles guidance, no CFG needed |
| VAEDecodeTiled | tile_size | 512 | Memory-safe decode for high-res |

### VRAM Note

This upscaler runs on the FULL accumulated video at once. For very long
videos (5+ minutes), the latent may not fit in VRAM. In that case, the
upscaler chain can be bypassed (set nodes to mode=4) and frames exported
at native 832x480 resolution.
