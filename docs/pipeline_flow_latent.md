Last updated: 2026-04-11

# Pipeline Flow: LATENT-Based Music Video Workflow

Source: `example_workflows/audio-loop-music-video_latent.json`

This document traces every node in the latent-space audio loop music video workflow, from model loading through final output. The workflow generates full-length music videos by iteratively extending an initial render in latent space (no per-iteration VAE round-trip).

---

## High-Level Data Flow

```
LoadAudio -> TrimAudio(skip intro) -> MelBandRoFormer(vocals) -> Set_actual_audio
                                   \-> Set_orig_audio

LoadImage -> Resize -> LTXVPreprocess -> Set_input_image

Models: UNETLoader -> SageAttn -> ChunkFF -> AttnTuner -> NAG -> PreviewOverride -> Set_model
        VAELoaderKJ(video) -> Set_video_vae
        VAELoaderKJ(audio) -> Set_audio_vae
        DualCLIPLoader -> text encoders

Initial Render:
  EmptyLTXVLatent -> LTXVImgToVideoInplace -> LTXVConcatAV -> SamplerCustomAdvanced
  -> LTXVSeparateAV -> video_latent -> LTXVCropGuides -> LatentConcat(prepend)
                    -> video_latent -> VAEDecode (preview)
                    -> video_latent -> TensorLoopOpen

Loop Body (per iteration):
  TensorLoopOpen.previous_value -> Extension #843 -> TensorLoopClose
  AudioLoopController -> start_index, should_stop, iteration_seed, overlap_latent_frames

Extension Subgraph #843 internals:
  LatentContextExtract -> LTXVAudioVideoMask -> LTXVAddLatentGuide -> LTXVConcatAV
  -> SamplerCustomAdvanced -> LTXVSeparateAV -> LTXVCropGuides -> LatentOverlapTrim
  -> output (extended_latent)

Output Assembly:
  LatentConcat(initial + loop_output) -> VAEDecodeTiled -> VHS_VideoCombine
```

---

## Stage 1: Model Loading

### Node 414 -- UNETLoader
- **Type**: `UNETLoader` (ComfyUI core: `nodes.py`)
- **What**: Loads the LTX 2.3 diffusion transformer model
- **Inputs**: None linked
  - `unet_name`: `diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors`
  - `weight_dtype`: `default`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | MODEL | MODEL | #268 PathchSageAttentionKJ (slot 0) |

### Node 268 -- PathchSageAttentionKJ (BYPASSED, mode=4)
- **Type**: `PathchSageAttentionKJ` (KJNodes: `nodes/model_optimization_nodes.py`)
- **What**: Patches model to use SageAttention for faster inference. Currently bypassed -- model passes through unchanged.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | model | MODEL | #414 UNETLoader (slot 0) |
- **Widgets**: `attention_backend`: `auto`, `force_fp32_output`: `false`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | MODEL | MODEL | #504 LTXVChunkFeedForward (slot 0) |

### Node 504 -- LTXVChunkFeedForward
- **Type**: `LTXVChunkFeedForward` (KJNodes: `nodes/ltxv_nodes.py`)
- **What**: Chunks the feed-forward layers to reduce peak VRAM usage during inference
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | model | MODEL | #268 PathchSageAttentionKJ (slot 0) |
- **Widgets**: `chunk_size`: `2`, `dim`: `4096`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | model | MODEL | #1523 LTX2AttentionTunerPatch (slot 0) |

### Node 1523 -- LTX2AttentionTunerPatch
- **Type**: `LTX2AttentionTunerPatch` (KJNodes: `nodes/ltxv_nodes.py`)
- **What**: Patches model attention weights for fine-grained control over self/cross attention scaling
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | model | MODEL | #504 LTXVChunkFeedForward (slot 0) |
- **Widgets**: `block_indices`: `""`, `self_attn_scale`: `1`, `cross_attn_text_scale`: `1`, `cross_attn_guide_scale`: `1`, `cross_attn_ref_scale`: `1`, `apply_to_all`: `true`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | model | MODEL | #508 LTX2_NAG (slot 0) |

### Node 508 -- LTX2_NAG
- **Type**: `LTX2_NAG` (KJNodes: `nodes/ltxv_nodes.py`)
- **What**: Applies Normalized Attention Guidance to the model. LTX 2.3 is distilled (CFG=1.0), so NAG provides the actual guidance signal.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | model | MODEL | #1523 LTX2AttentionTunerPatch (slot 0) |
  | nag_cond_video | CONDITIONING | #507 CLIPTextEncode (slot 0) -- negative prompt text |
  | nag_cond_audio | CONDITIONING | **unwired** (null) -- not used; audio guidance not needed for music video |
- **Widgets**: `nag_scale`: `11`, `nag_init_scale`: `0.25`, `nag_sigma_end`: `2.5`, `rescale_cfg`: `true`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | model | MODEL | #503 LTX2SamplingPreviewOverride (slot 0) |

### Node 503 -- LTX2SamplingPreviewOverride
- **Type**: `LTX2SamplingPreviewOverride` (KJNodes: `nodes/ltxv_nodes.py`)
- **What**: Configures sampling preview rate (how often intermediate frames are decoded for preview during sampling)
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | model | MODEL | #508 LTX2_NAG (slot 0) |
  | latent_upscale_model | LATENT_UPSCALE_MODEL | **unwired** (null) -- no upscale preview |
  | vae | VAE | **unwired** (null) -- uses model's built-in decoder |
- **Widgets**: `preview_interval`: `8`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | MODEL | MODEL | #572 Set_model (slot 0), #1513 ModelSamplingSD3 (slot 0) |

### Node 572 -- Set_model
- **Type**: `SetNode` (KJNodes)
- **What**: Stores model reference as `model` for Get nodes throughout the workflow
- **Inputs**: MODEL from #503
- **Outputs**: MODEL -> #153 CFGGuider (slot 0)

### Node 1537 -- VAELoaderKJ (Video VAE)
- **Type**: `VAELoaderKJ` (KJNodes: `nodes/ltxv_nodes.py`)
- **What**: Loads the LTX 2.3 video VAE
- **Inputs**: None linked
  - `vae_name`: `vae/LTX23_video_vae_bf16.safetensors`
  - `device`: `main_device`
  - `dtype`: `bf16`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | VAE | VAE | #228 Set_video_vae (slot 0) |

### Node 228 -- Set_video_vae
- **Type**: `SetNode` (KJNodes)
- **What**: Stores video VAE as `video_vae` for Get nodes

### Node 1538 -- VAELoaderKJ (Audio VAE)
- **Type**: `VAELoaderKJ` (KJNodes: `nodes/ltxv_nodes.py`)
- **What**: Loads the LTX 2.3 audio VAE
- **Inputs**: None linked
  - `vae_name`: `vae/LTX23_audio_vae_bf16.safetensors`
  - `device`: `main_device`
  - `dtype`: `bf16`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | VAE | VAE | #252 Set_audio_vae (slot 0) |

### Node 252 -- Set_audio_vae
- **Type**: `SetNode` (KJNodes)
- **What**: Stores audio VAE as `audio_vae` for Get nodes

### Node 416 -- DualCLIPLoader
- **Type**: `DualCLIPLoader` (ComfyUI core: `nodes.py`)
- **What**: Loads the Gemma 3 text encoder + LTX text projection. Despite the "CLIP" name, this produces Gemma 3 conditioning (no pooled_output).
- **Inputs**: None linked
  - `clip_name1`: `gemma_3_12B_it_fpmixed.safetensors`
  - `clip_name2`: `ltx-2.3_text_projection_bf16.safetensors`
  - `type`: `ltxv`
  - `device`: `default`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | CLIP | CLIP | #169 CLIPTextEncode pos (slot 0), #507 CLIPTextEncode neg (slot 0), #1559 Loop Prompt Encode (slot 0) |

---

## Stage 2: Audio Preparation

### Node 565 -- LoadAudio
- **Type**: `LoadAudio` (ComfyUI core)
- **What**: Loads the audio file from disk
- **Inputs**: None linked
  - `audio`: `example_audio.mp4`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | AUDIO | AUDIO | #567 TrimAudioDuration (slot 0) |

### Node 567 -- TrimAudioDuration (Intro Skip)
- **Type**: `TrimAudioDuration` (ComfyUI core)
- **What**: Trims the start of the audio to skip instrumental intro (not useful for lip sync). Also sets total audio duration cap.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | audio | AUDIO | #565 LoadAudio (slot 0) |
- **Widgets**: `start_index`: `5` (skip 5s of intro), `duration`: `300` (cap at 300s)
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | AUDIO | AUDIO | #569 MelBandRoFormerSampler (slot 1), #581 Set_orig_audio, #1560 AudioLoopPlanner (slot 0), #1582 AudioLoopController (slot 0) |

### Node 568 -- MelBandRoFormerModelLoader
- **Type**: `MelBandRoFormerModelLoader` (MelBandRoFormer: `nodes.py`)
- **What**: Loads the MelBandRoFormer vocal separation model
- **Inputs**: None linked
  - `model_name`: `MelBandRoformer_fp32.safetensors`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | model | MELROFORMERMODEL | #569 MelBandRoFormerSampler (slot 0) |

### Node 569 -- MelBandRoFormerSampler
- **Type**: `MelBandRoFormerSampler` (MelBandRoFormer: `nodes.py`)
- **What**: Separates audio into vocals and instruments using the MelBandRoFormer model
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | model | MELROFORMERMODEL | #568 MelBandRoFormerModelLoader (slot 0) |
  | audio | AUDIO | #567 TrimAudioDuration (slot 0) |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | vocals | AUDIO | #640 Set_actual_audio (slot 0) |
  | instruments | AUDIO | **unwired** -- instruments discarded; vocals only for lip sync |

### Node 640 -- Set_actual_audio
- **Type**: `SetNode` (KJNodes)
- **What**: Stores separated vocals as `actual_audio`. This is the audio fed to the extension subgraph for lip-sync conditioning.
- **Inputs**: AUDIO from #569 (vocals)
- **Outputs**: AUDIO -> #601 TrimAudioDuration (slot 0)

### Node 581 -- Set_orig_audio
- **Type**: `SetNode` (KJNodes)
- **What**: Stores the full (unseparated) audio as `orig_audio` for final video muxing

### Node 688 -- FloatConstant (window_size_seconds)
- **Type**: `FloatConstant` (KJNodes)
- **What**: Defines the duration of each generation window in seconds
- **Inputs**: None
- **Widgets**: `value`: `19.88` (497 frames / 25fps)
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | value | FLOAT | #601 TrimAudioDuration duration (slot 1), #689 Set_window_size_seconds (slot 0) |

### Node 689 -- Set_window_size_seconds
- **Type**: `SetNode` (KJNodes)
- **What**: Stores window size as `window_size_seconds`

### Node 601 -- TrimAudioDuration (Window Trim)
- **Type**: `TrimAudioDuration` (ComfyUI core)
- **What**: Trims the separated vocals to exactly one window duration (19.88s) for the initial render's audio conditioning
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | audio | AUDIO | #640 Set_actual_audio (slot 0) |
  | duration | FLOAT | #688 FloatConstant (slot 0) -- 19.88 |
- **Widgets**: `start_index`: `0`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | AUDIO | AUDIO | #566 LTXVAudioVAEEncode (slot 0) |

### Node 254 -- Get_audio_vae
- **Type**: `GetNode` (KJNodes)
- **What**: Retrieves stored `audio_vae`
- **Outputs**: VAE -> #566 LTXVAudioVAEEncode (slot 1)

### Node 566 -- LTXVAudioVAEEncode
- **Type**: `LTXVAudioVAEEncode` (ComfyUI core: `comfy_extras/nodes_lt.py`)
- **What**: Encodes audio waveform into latent space using the audio VAE
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | audio | AUDIO | #601 TrimAudioDuration (slot 0) |
  | audio_vae | VAE | #254 Get_audio_vae (slot 0) |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | Audio Latent | LATENT | #570 SetLatentNoiseMask (slot 0) |

### Node 571 -- SolidMask
- **Type**: `SolidMask` (ComfyUI core)
- **What**: Creates a solid black mask (value=0). mask=0 means "fixed/context" in LTX noise mask semantics -- audio stays as the real encoded song, not regenerated from noise.
- **Inputs**: None
- **Widgets**: `value`: `0`, `width`: `512`, `height`: `512`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | MASK | MASK | #570 SetLatentNoiseMask (slot 1) |

### Node 570 -- SetLatentNoiseMask
- **Type**: `SetLatentNoiseMask` (ComfyUI core)
- **What**: Attaches the all-zeros noise mask to the audio latent. This tells the sampler the audio is fixed context (do not regenerate).
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | samples | LATENT | #566 LTXVAudioVAEEncode (slot 0) |
  | mask | MASK | #571 SolidMask (slot 0) |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | LATENT | LATENT | #350 LTXVConcatAVLatent (slot 1) |

---

## Stage 3: Text Encoding / Conditioning

### Node 169 -- CLIPTextEncode (Positive Prompt)
- **Type**: `CLIPTextEncode` (ComfyUI core)
- **What**: Encodes the positive prompt using Gemma 3. Output is `[tensor, {"attention_mask": ...}]` with no pooled_output.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | clip | CLIP | #416 DualCLIPLoader (slot 0) |
- **Widgets**: `text`: `"video of a woman passionately singing alone"`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | CONDITIONING | CONDITIONING | #164 LTXVConditioning pos (slot 0), #420 ConditioningZeroOut (slot 0) |

### Node 507 -- CLIPTextEncode (Negative/NAG Prompt)
- **Type**: `CLIPTextEncode` (ComfyUI core)
- **What**: Encodes the negative prompt for NAG guidance. This text is used by LTX2_NAG's nag_cond_video input.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | clip | CLIP | #416 DualCLIPLoader (slot 0) |
- **Widgets**: `text`: `"still image with no motion, subtitles, text, scene change, instruments, violin, blurry, out of focus, overexposed, underexposed..."` (long negative)
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | CONDITIONING | CONDITIONING | #508 LTX2_NAG nag_cond_video (slot 1) |

### Node 420 -- ConditioningZeroOut
- **Type**: `ConditioningZeroOut` (ComfyUI core)
- **What**: Zeros out the conditioning tensor to create an unconditional (empty) negative. Used as the CFGGuider negative since NAG handles actual guidance.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | conditioning | CONDITIONING | #169 CLIPTextEncode pos (slot 0) |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | CONDITIONING | CONDITIONING | #164 LTXVConditioning neg (slot 1) |

### Node 164 -- LTXVConditioning (Initial Render)
- **Type**: `LTXVConditioning` (ComfyUI core: `comfy_extras/nodes_lt.py`)
- **What**: Adds `frame_rate` metadata to positive and negative conditioning. Without this, the sampler has no temporal rate info.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | positive | CONDITIONING | #169 CLIPTextEncode (slot 0) |
  | negative | CONDITIONING | #420 ConditioningZeroOut (slot 0) |
- **Widgets**: `frame_rate`: `25`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | positive | CONDITIONING | #381 LTXVCropGuides (slot 0), #153 CFGGuider (slot 1), #645 Set_base_cond_pos |
  | negative | CONDITIONING | #381 LTXVCropGuides (slot 1), #153 CFGGuider (slot 2), #646 Set_base_cond_neg |

### Node 645 -- Set_base_cond_pos
- **Type**: `SetNode` (KJNodes)
- **What**: Stores positive conditioning as `base_cond_pos` for loop body retrieval

### Node 646 -- Set_base_cond_neg
- **Type**: `SetNode` (KJNodes)
- **What**: Stores negative conditioning as `base_cond_neg` for loop body retrieval

---

## Stage 4: Image Loading & Initial Render

### Node 444 -- LoadImage
- **Type**: `LoadImage` (ComfyUI core)
- **What**: Loads the reference/init image for i2v generation
- **Inputs**: None
  - `image`: `reference_image.png`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | IMAGE | IMAGE | #445 ImageResizeKJv2 (slot 0) |
  | MASK | MASK | #445 ImageResizeKJv2 (slot 1) |

### Node 445 -- ImageResizeKJv2
- **Type**: `ImageResizeKJv2` (KJNodes)
- **What**: Resizes the reference image to the target generation resolution
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | image | IMAGE | #444 LoadImage (slot 0) |
  | mask | MASK | #444 LoadImage (slot 1) |
- **Widgets**: `width`: `832`, `height`: `480`, `interpolation`: `lanczos`, `method`: `crop`, `padding_color`: `0, 0, 0`, `align`: `top`, `multiple_of`: `2`, `device`: `cpu`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | IMAGE | IMAGE | #446 LTXVPreprocess (slot 0), #650 Set_input_image (slot 0) |
  | width | INT | #344 EmptyLTXVLatentVideo width (slot 0) |
  | height | INT | #344 EmptyLTXVLatentVideo height (slot 1) |
  | mask | MASK | unwired |

### Node 650 -- Set_input_image
- **Type**: `SetNode` (KJNodes)
- **What**: Stores resized image as `input_image` for the extension subgraph

### Node 446 -- LTXVPreprocess
- **Type**: `LTXVPreprocess` (ComfyUI core: `comfy_extras/nodes_lt.py`)
- **What**: Preprocesses the image for LTX i2v (pixel normalization). Widget `noise_aug_strength`: `0`.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | image | IMAGE | #445 ImageResizeKJv2 (slot 0) |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | output_image | IMAGE | #531 LTXVImgToVideoInplaceKJ image_1 (slot 2) |

### Node 526 -- PrimitiveNode (length)
- **Type**: `PrimitiveNode` (ComfyUI core)
- **What**: Provides the frame count constant
- **Widgets**: `value`: `497` (19.88s * 25fps = 497 frames, satisfies 8n+1 rule)
- **Outputs**: INT -> #344 EmptyLTXVLatentVideo length (slot 2)

### Node 344 -- EmptyLTXVLatentVideo
- **Type**: `EmptyLTXVLatentVideo` (ComfyUI core: `comfy_extras/nodes_lt.py`)
- **What**: Creates an empty video latent tensor at the specified resolution and frame count
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | width | INT | #445 ImageResizeKJv2 width (slot 1) -- 832 |
  | height | INT | #445 ImageResizeKJv2 height (slot 2) -- 480 |
  | length | INT | #526 PrimitiveNode (slot 0) -- 497 |
- **Widgets**: `batch_size`: `1`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | LATENT | LATENT | #531 LTXVImgToVideoInplaceKJ (slot 1) |

### Node 413 -- Get_video_vae
- **Type**: `GetNode` (KJNodes) -- retrieves `video_vae`
- **Outputs**: VAE -> #531 LTXVImgToVideoInplaceKJ (slot 0)

### Node 531 -- LTXVImgToVideoInplaceKJ
- **Type**: `LTXVImgToVideoInplaceKJ` (KJNodes: `nodes/ltxv_nodes.py`)
- **What**: VAE-encodes the preprocessed init image and injects it into frame index 0 of the empty latent with a denoise mask (strength=1.0 means no noise added to that frame)
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | vae | VAE | #413 Get_video_vae (slot 0) |
  | latent | LATENT | #344 EmptyLTXVLatentVideo (slot 0) |
  | image_1 | IMAGE | #446 LTXVPreprocess (slot 0) |
- **Widgets**: `num_images`: `1`, `strength_1`: `1`, `index_1`: `0`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | latent | LATENT | #350 LTXVConcatAVLatent video_latent (slot 0) |

### Node 350 -- LTXVConcatAVLatent (Initial)
- **Type**: `LTXVConcatAVLatent` (ComfyUI core: `comfy_extras/nodes_lt.py`)
- **What**: Combines video latent (with init image embedded) and audio latent into a single AV NestedTensor for sampling
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | video_latent | LATENT | #531 LTXVImgToVideoInplaceKJ (slot 0) |
  | audio_latent | LATENT | #570 SetLatentNoiseMask (slot 0) |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | latent | LATENT | #161 SamplerCustomAdvanced latent_image (slot 4) |

### Sampling Configuration (Initial Render)

#### Node 1513 -- ModelSamplingSD3
- **Type**: `ModelSamplingSD3` (ComfyUI core)
- **What**: Configures sigma shift for the sampling schedule
- **Inputs**: MODEL from #503
- **Widgets**: `shift`: `13`
- **Outputs**: MODEL -> #1421 BasicScheduler (slot 0)

#### Node 1421 -- BasicScheduler
- **Type**: `BasicScheduler` (ComfyUI core)
- **What**: Generates the sigma schedule for sampling
- **Inputs**: MODEL from #1513
- **Widgets**: `scheduler`: `linear_quadratic`, `steps`: `8`, `denoise`: `1`
- **Outputs**: SIGMAS -> #1422 VisualizeSigmasKJ (slot 0)

#### Node 1422 -- VisualizeSigmasKJ
- **Type**: `VisualizeSigmasKJ` (KJNodes)
- **What**: Passes sigmas through and generates a preview image of the sigma schedule
- **Inputs**: SIGMAS from #1421
- **Outputs**: sigmas_out -> #579 Set_sigmas, image -> #1423 PreviewImage

#### Node 579 -- Set_sigmas
- **Type**: `SetNode` (KJNodes)
- **What**: Stores sigmas as `sigmas` for loop body retrieval
- **Outputs**: SIGMAS -> #161 SamplerCustomAdvanced (slot 3)

#### Node 1527 -- INTConstant (start_seed)
- **Type**: `INTConstant` (KJNodes)
- **What**: Base seed for deterministic generation
- **Widgets**: `value`: `42`
- **Outputs**: INT -> #1528 Set_start_seed

#### Node 1528 -- Set_start_seed
- **Type**: `SetNode` (KJNodes)
- **What**: Stores base seed as `start_seed`

#### Node 1530 -- Get_start_seed
- **Type**: `GetNode` (KJNodes) -- retrieves `start_seed`
- **Outputs**: INT -> #1322 RandomNoise (slot 0)

#### Node 1322 -- RandomNoise (Initial)
- **Type**: `RandomNoise` (ComfyUI core)
- **What**: Creates noise tensor from seed
- **Inputs**: noise_seed from #1530 Get_start_seed
- **Widgets**: `control_after_generate`: `fixed`
- **Outputs**: NOISE -> #161 SamplerCustomAdvanced (slot 0)

#### Node 153 -- CFGGuider (Initial)
- **Type**: `CFGGuider` (ComfyUI core)
- **What**: Wraps model + conditioning into a guider object for the sampler. cfg=1.0 because NAG handles guidance.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | model | MODEL | #572 Set_model (slot 0) |
  | positive | CONDITIONING | #164 LTXVConditioning pos (slot 0) |
  | negative | CONDITIONING | #164 LTXVConditioning neg (slot 1) |
- **Widgets**: `cfg`: `1`
- **Outputs**: GUIDER -> #161 SamplerCustomAdvanced (slot 1), #575 Set_guider

#### Node 575 -- Set_guider
- **Type**: `SetNode` (KJNodes)
- **What**: Stores guider as `guider` (available but not used -- loop uses its own CFGGuider)

#### Node 576 -- Set_sampler
- **Type**: `SetNode` (KJNodes)
- **What**: Stores sampler as `sampler` for loop body

#### Node 154 -- KSamplerSelect
- **Type**: `KSamplerSelect` (ComfyUI core)
- **What**: Selects the sampling algorithm
- **Widgets**: `sampler_name`: `euler_ancestral`
- **Outputs**: SAMPLER -> #161 SamplerCustomAdvanced (slot 2), #576 Set_sampler

### Node 161 -- SamplerCustomAdvanced (Initial Render)
- **Type**: `SamplerCustomAdvanced` (ComfyUI core)
- **What**: Performs the initial video generation (first window of the full video)
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | noise | NOISE | #1322 RandomNoise (slot 0) |
  | guider | GUIDER | #153 CFGGuider (slot 0) |
  | sampler | SAMPLER | #154 KSamplerSelect (slot 0) |
  | sigmas | SIGMAS | #579 Set_sigmas (slot 0) |
  | latent_image | LATENT | #350 LTXVConcatAVLatent (slot 0) |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | output | LATENT | #245 LTXVSeparateAVLatent (slot 0) |
  | denoised_output | LATENT | unwired |

### Node 245 -- LTXVSeparateAVLatent (Post-Initial)
- **Type**: `LTXVSeparateAVLatent` (ComfyUI core: `comfy_extras/nodes_lt.py`)
- **What**: Splits the sampled AV NestedTensor back into separate video and audio latents. The video latent still contains appended guide frames.
- **Inputs**: av_latent from #161 SamplerCustomAdvanced
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | video_latent | LATENT | #381 LTXVCropGuides (slot 2), #1318 VAEDecode (slot 0), #1539 TensorLoopOpen initial_value (slot 0) |
  | audio_latent | LATENT | unwired -- discarded; loop re-encodes audio each iteration |

**Critical path split**: The video_latent from #245 goes to THREE places:
1. **#1539 TensorLoopOpen** -- full latent (WITH guides) as loop initial value
2. **#381 LTXVCropGuides** -- strips guides for prepending to final output
3. **#1318 VAEDecode** -- preview of initial render

### Node 381 -- LTXVCropGuides (Initial)
- **Type**: `LTXVCropGuides` (ComfyUI core: `comfy_extras/nodes_lt.py`)
- **What**: Removes appended guide frames from the video latent. The guide frames were added by LTXVImgToVideoInplaceKJ and are NOT part of the actual generated content.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | positive | CONDITIONING | #164 LTXVConditioning pos |
  | negative | CONDITIONING | #164 LTXVConditioning neg |
  | latent | LATENT | #245 LTXVSeparateAVLatent video_latent |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | positive | CONDITIONING | unwired |
  | negative | CONDITIONING | unwired |
  | latent | LATENT | #1605 LatentConcat samples1 (slot 0) -- initial render for prepend |

### Node 1318 -- VAEDecode (Preview)
- **Type**: `VAEDecode` (ComfyUI core)
- **What**: Decodes the initial render latent to pixel space for preview
- **Inputs**: samples from #245, vae from #236 Get_video_vae
- **Outputs**: IMAGE -> #560 VHS_VideoCombine (preview, mode=4 bypassed), #618 Reroute

### Node 560 -- VHS_VideoCombine (Initial Preview, BYPASSED mode=4)
- **Type**: `VHS_VideoCombine` (VideoHelperSuite)
- **What**: Would preview the initial render as video. Currently bypassed.

---

## Stage 5: Loop Setup

### Node 1269 -- FloatConstant (first_frame_guide_strength)
- **Type**: `FloatConstant` (KJNodes)
- **What**: Controls guide strength for the init image in the extension subgraph
- **Widgets**: `value`: `1` (full strength -- init image frame is spatially frozen)
- **Outputs**: FLOAT -> #1271 Set_first_frame_guide_strength

### Node 1271 -- Set_first_frame_guide_strength
- **Type**: `SetNode` (KJNodes) -- stores as `first_frame_guide_strength`

### Node 1539 -- TensorLoopOpen
- **Type**: `TensorLoopOpen` (NativeLooping: `nodes.py`)
- **What**: Opens the extension loop. Receives the initial render's full video latent (with guides) as `initial_value`. On iteration 1, `previous_value` is this initial latent. On subsequent iterations, `previous_value` is the extended_latent from the previous iteration's subgraph output.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | initial_value | LATENT | #245 LTXVSeparateAVLatent video_latent (slot 0) |
- **Widgets**: `mode`: `iterations`, `iterations`: `50`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | flow_control | FLOW_CONTROL | #1540 TensorLoopClose (slot 0) |
  | previous_value | LATENT | #843 Extension previous_latent (slot 4) |
  | accumulated_count | INT | unwired |
  | current_iteration | INT | #1582 AudioLoopController (slot 1), #1558 TimestampPromptSchedule (slot 0) |

### Node 1582 -- AudioLoopController
- **Type**: `AudioLoopController` (AudioLoopHelper: `nodes.py`)
- **What**: Computes per-iteration timing: start_index for audio trimming, should_stop signal, iteration_seed, stride, overlap frames. Reads audio duration from the tensor -- no manual constants needed.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | audio | AUDIO | #567 TrimAudioDuration (slot 0) |
  | current_iteration | INT | #1539 TensorLoopOpen (slot 3) |
  | window_seconds | FLOAT | #691 Get_window_size_seconds (slot 0) -- 19.88 |
  | seed | INT | #1529 Get_start_seed (slot 0) |
- **Widgets**: `overlap_seconds`: `1`, `fps`: `25`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | start_index | FLOAT | #843 Extension start_index (slot 11) |
  | should_stop | BOOLEAN | #1540 TensorLoopClose stop (slot 2) |
  | audio_duration | FLOAT | unwired -- informational |
  | iteration_seed | INT | #843 Extension noise_seed (slot 13) |
  | stride_seconds | FLOAT | #1558 TimestampPromptSchedule (slot 1), #1560 AudioLoopPlanner (slot 1) |
  | overlap_frames | INT | #1586 PreviewAny (display only) |
  | overlap_latent_frames | INT | #843 Extension num_frames/overlap (slot 14) |

### Node 1558 -- TimestampPromptSchedule (BYPASSED, mode=4)
- **Type**: `TimestampPromptSchedule` (AudioLoopHelper: `nodes.py`)
- **What**: Selects per-iteration prompt from a timestamp-based schedule. Currently bypassed -- static prompt used instead via Get_base_cond_pos.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | current_iteration | INT | #1539 TensorLoopOpen (slot 3) |
  | stride_seconds | FLOAT | #1582 AudioLoopController stride_seconds (slot 4) |
- **Widgets**: `schedule`: `"0:00+: video of a woman passionately singing alone"`, `blend_seconds`: not shown (default 0)
- **Outputs**: prompt STRING -> #1559 Loop Prompt Encode (slot 1)

### Node 1559 -- CLIPTextEncode (Loop Prompt Encode)
- **Type**: `CLIPTextEncode` (ComfyUI core)
- **What**: Encodes the per-iteration prompt for the loop body. When TimestampPromptSchedule is bypassed, this receives no text input.
- **Inputs**: clip from #416 DualCLIPLoader, text from #1558 TimestampPromptSchedule
- **Outputs**: CONDITIONING -> unwired (empty links array)

### Node 1588 -- Get_base_cond_pos (Static Mode)
- **Type**: `GetNode` (KJNodes) -- retrieves `base_cond_pos`
- **What**: In static mode (TimestampPromptSchedule bypassed), the base positive conditioning from the initial render is reused for every loop iteration.
- **Outputs**: CONDITIONING -> #843 Extension positive (slot 6) directly

### Node 648 -- Get_base_cond_neg
- **Type**: `GetNode` (KJNodes) -- retrieves `base_cond_neg`
- **Outputs**: CONDITIONING -> #843 Extension negative (slot 7) directly

### Node 1587 -- LTXVConditioning (Loop) -- BYPASSED (mode=4)
- **Type**: `LTXVConditioning` (ComfyUI core)
- **Why bypassed**: Was wrapping the Extension's conditioning with frame_rate=25.
  This caused ComfyUI's execution engine to evaluate the conditioning graph in a
  way that corrupted the initial render's audio-video cross-attention, destroying
  lip sync. Removed 2026-04-12. See `internal/postmortem_v0409_latent_rework.md` Issue 6.
- Conditioning now flows directly from #1588/#648 to #843.

### Node 1560 -- AudioLoopPlanner (BYPASSED, mode=4)
- **Type**: `AudioLoopPlanner` (AudioLoopHelper: `nodes.py`)
- **What**: Displays iteration timeline for prompt schedule planning. Purely informational.
- **Inputs**: audio from #567, stride_seconds from #1582, window_seconds from #691
- **Outputs**: summary STRING -> #1563 PreviewAny

### Node 1563 -- PreviewAny (Iteration Timestamps, BYPASSED)
- **Type**: `PreviewAny` (ComfyUI core)
- **What**: Displays the planner output text

---

## Stage 6: Extension Subgraph #843 -- FULL DETAIL

**Subgraph ID**: `b4973d68-09b9-4da5-9845-38ad62ae9aca`
**Display name**: `extension`
**Internal node IDs**: -10 (input distributor), -20 (output collector), 573, 574, 583, 596, 598, 600, 606, 644, 655, 1519, 1520, 2004, 2005

### Subgraph External Inputs

The subgraph receives 15 inputs from the outer workflow via internal node -10:

| Slot | Name (label) | Type | External Source |
|------|-------------|------|----------------|
| 0 | sampler | SAMPLER | #578 Get_sampler |
| 1 | sigmas | SIGMAS | #580 Get_sigmas |
| 2 | model | MODEL | #654 Get_model |
| 3 | vae | VAE | #619 Get_video_vae |
| 4 | previous_latent (previous_images) | LATENT | #1539 TensorLoopOpen previous_value (slot 1) |
| 5 | video_end_time (window_size_seconds) | FLOAT | #691 Get_window_size_seconds -- 19.88 |
| 6 | positive | CONDITIONING | #1588 Get_base_cond_pos directly |
| 7 | negative | CONDITIONING | #648 Get_base_cond_neg directly |
| 8 | num_guides.image_1 (init_image) | IMAGE | #651 Get_input_image |
| 9 | audio_vae (Audio VAE) | VAE | #599 Get_audio_vae |
| 10 | audio | AUDIO | #641 Get_actual_audio |
| 11 | start_index | FLOAT | #1582 AudioLoopController start_index (slot 0) |
| 12 | num_guides.strength_1 (first_frame_guide_strength) | FLOAT | #1273 Get_first_frame_guide_strength |
| 13 | noise_seed | INT | #1582 AudioLoopController iteration_seed (slot 3) |
| 14 | num_frames (overlap_latent_frames) | INT | #1582 AudioLoopController overlap_latent_frames (slot 6) |

### Subgraph External Output

| Slot | Name | Type | External Target |
|------|------|------|----------------|
| 0 | extended_latent | LATENT | #1540 TensorLoopClose processed (slot 1) |

### Internal Node Detail

#### Node 2004 -- LatentContextExtract ("Context Extract")
- **Type**: `LatentContextExtract` (AudioLoopHelper: `nodes.py`)
- **What**: Extracts the last N latent frames from the previous iteration's video latent as overlap context. **Critically, strips noise_mask** so LTXVAudioVideoMask creates a fresh mask downstream.
- **Source**: `nodes.py` lines 652-694
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | latent | LATENT | Subgraph input slot 4 (previous_latent) via link 2957 |
  | overlap_latent_frames | INT | Subgraph input slot 14 via link 2996 |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | context | LATENT | #606 LTXVAudioVideoMask video_latent (slot 0) via link 2958 |
- **Why noise_mask stripping is critical**: The previous iteration's latent carries a noise_mask from its own sampling pass. If this stale mask propagates to LTXVAudioVideoMask, the `existing_mask_mode: "add"` logic would merge the old mask with the new one, corrupting the sampler's understanding of which frames are context (mask=0) vs. to-be-generated (mask=1). By stripping noise_mask, we force LTXVAudioVideoMask to create a fresh mask from scratch, exactly matching the behavior of VAEEncode in the IMAGE workflow.

#### Node 600 -- TrimAudioDuration (Per-Iteration Audio)
- **Type**: `TrimAudioDuration` (ComfyUI core)
- **What**: Trims the vocals audio to the correct window for this iteration. `start_index` comes from AudioLoopController (advances each iteration), `duration` is window_size_seconds.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | audio | AUDIO | Subgraph input slot 10 (audio) via link 1686 |
  | start_index | FLOAT | Subgraph input slot 11 via link 1927 |
  | duration | FLOAT | Subgraph input slot 5 (window_size_seconds) via link 2084 |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | AUDIO | AUDIO | #598 LTXVAudioVAEEncode (slot 0) via link 1604 |

#### Node 598 -- LTXVAudioVAEEncode (Per-Iteration)
- **Type**: `LTXVAudioVAEEncode` (ComfyUI core)
- **What**: Encodes the trimmed per-iteration audio into latent space
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | audio | AUDIO | #600 TrimAudioDuration (slot 0) via link 1604 |
  | audio_vae | VAE | Subgraph input slot 9 via link 1602 |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | Audio Latent | LATENT | #606 LTXVAudioVideoMask audio_latent (slot 1) via link 2956 |

#### Node 606 -- LTXVAudioVideoMask
- **Type**: `LTXVAudioVideoMask` (KJNodes: `nodes/ltxv_nodes.py`)
- **What**: Creates noise masks for the video and audio latents. Sets up which frames are context (mask=0, kept fixed) and which are to be generated (mask=1). For audio: start_time=end_time=window_size_seconds creates an empty mask range, keeping audio fixed as the real encoded song.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | video_latent | LATENT | #2004 LatentContextExtract context (slot 0) via link 2958 |
  | audio_latent | LATENT | #598 LTXVAudioVAEEncode (slot 0) via link 2956 |
  | video_end_time | FLOAT | Subgraph input slot 5 (window_size_seconds=19.88) via link 1784 |
  | audio_start_time | FLOAT | Subgraph input slot 5 (window_size_seconds=19.88) via link 1782 |
  | audio_end_time | FLOAT | Subgraph input slot 5 (window_size_seconds=19.88) via link 1783 |
- **Widgets**: `video_fps`: `25`, `video_start_time`: `1` (proxy default), `max_length`: `pad`, `existing_mask_mode`: `add`
- **Key behavior**: audio_start_time == audio_end_time == 19.88 means the audio mask range is empty -- audio latent gets mask=0 everywhere (fixed context). Video latent gets mask=1 for new frames (to be generated) and mask=0 for the overlap context frames extracted by LatentContextExtract. The `max_length: pad` mode extends the video latent to the full window duration if needed.
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | video_latent | LATENT | #1519 LTXVAddLatentGuide latent (slot 3) via link 2828 |
  | audio_latent | LATENT | #583 LTXVConcatAVLatent audio_latent (slot 1) via link 2239 |

#### Node 1520 -- VAEEncode (Init Image Guide)
- **Type**: `VAEEncode` (ComfyUI core)
- **What**: Encodes the init image to latent space as a scene anchor guide
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | pixels | IMAGE | Subgraph input slot 8 (init_image) via link 2825 |
  | vae | VAE | Subgraph input slot 3 (video_vae) via link 2826 |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | LATENT | LATENT | #1519 LTXVAddLatentGuide guiding_latent (slot 4) via link 2827 |

#### Node 1519 -- LTXVAddLatentGuide
- **Type**: `LTXVAddLatentGuide` (ComfyUI-LTXVideo: `latents.py`)
- **What**: Adds the init image as a latent guide. Appends guide frames to the temporal dimension, sets up keyframe indices for RoPE positioning, and creates a denoise mask (strength=1.0 means guide frame gets zero noise). Also modifies positive and negative conditioning with guide attention entries.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | vae | VAE | Subgraph input slot 3 (video_vae) via link 2822 |
  | positive | CONDITIONING | Subgraph input slot 6 via link 2823 |
  | negative | CONDITIONING | Subgraph input slot 7 via link 2824 |
  | latent | LATENT | #606 LTXVAudioVideoMask video_latent (slot 0) via link 2828 |
  | guiding_latent | LATENT | #1520 VAEEncode (slot 0) via link 2827 |
  | strength | FLOAT | Subgraph input slot 12 (first_frame_guide_strength) via link 2839 |
- **Widgets**: `latent_idx`: `-1` (guide positioned before first frame), `strength`: widget overridden by input
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | positive | CONDITIONING | #644 CFGGuider pos (slot 1) via link 2830, #655 LTXVCropGuides pos (slot 0) via link 2832 |
  | negative | CONDITIONING | #644 CFGGuider neg (slot 2) via link 2831, #655 LTXVCropGuides neg (slot 1) via link 2833 |
  | latent | LATENT | #583 LTXVConcatAVLatent video_latent (slot 0) via link 2829 |

#### Node 583 -- LTXVConcatAVLatent (Per-Iteration)
- **Type**: `LTXVConcatAVLatent` (ComfyUI core: `comfy_extras/nodes_lt.py`)
- **What**: Combines video latent (with guide appended) and masked audio latent into AV NestedTensor for sampling
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | video_latent | LATENT | #1519 LTXVAddLatentGuide latent (slot 2) via link 2829 |
  | audio_latent | LATENT | #606 LTXVAudioVideoMask audio_latent (slot 1) via link 2239 |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | latent | LATENT | #573 SamplerCustomAdvanced latent_image (slot 4) via link 1598 |

#### Node 644 -- CFGGuider (Per-Iteration)
- **Type**: `CFGGuider` (ComfyUI core)
- **What**: Packages model + guide-modified conditioning for the sampler. cfg=1.0 (NAG handles guidance).
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | model | MODEL | Subgraph input slot 2 via link 1707 |
  | positive | CONDITIONING | #1519 LTXVAddLatentGuide pos (slot 0) via link 2830 |
  | negative | CONDITIONING | #1519 LTXVAddLatentGuide neg (slot 1) via link 2831 |
- **Widgets**: `cfg`: `1`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | GUIDER | GUIDER | #573 SamplerCustomAdvanced guider (slot 1) via link 1706 |

#### Node 574 -- RandomNoise (Per-Iteration)
- **Type**: `RandomNoise` (ComfyUI core)
- **What**: Creates noise from the per-iteration seed (base_seed + current_iteration)
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | noise_seed | INT | Subgraph input slot 13 (iteration_seed) via link 2840 |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | NOISE | NOISE | #573 SamplerCustomAdvanced noise (slot 0) via link 1573 |

#### Node 573 -- SamplerCustomAdvanced (Per-Iteration)
- **Type**: `SamplerCustomAdvanced` (ComfyUI core)
- **What**: Generates the next window of video+audio frames. The noise_mask on the latent tells the sampler which frames to keep (context overlap + audio) and which to generate fresh.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | noise | NOISE | #574 RandomNoise (slot 0) via link 1573 |
  | guider | GUIDER | #644 CFGGuider (slot 0) via link 1706 |
  | sampler | SAMPLER | Subgraph input slot 0 via link 1577 |
  | sigmas | SIGMAS | Subgraph input slot 1 via link 1579 |
  | latent_image | LATENT | #583 LTXVConcatAVLatent (slot 0) via link 1598 |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | output | LATENT | #596 LTXVSeparateAVLatent (slot 0) via link 1595 |
  | denoised_output | LATENT | unwired |

#### Node 596 -- LTXVSeparateAVLatent (Per-Iteration)
- **Type**: `LTXVSeparateAVLatent` (ComfyUI core: `comfy_extras/nodes_lt.py`)
- **What**: Splits AV NestedTensor back into video and audio. Audio is discarded (each iteration re-encodes from source).
- **Inputs**: av_latent from #573
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | video_latent | LATENT | #655 LTXVCropGuides latent (slot 2) via link 1713 |
  | audio_latent | LATENT | unwired -- discarded |

#### Node 655 -- LTXVCropGuides (Per-Iteration)
- **Type**: `LTXVCropGuides` (ComfyUI core: `comfy_extras/nodes_lt.py`)
- **What**: Removes appended guide frames from the sampler output. The guide was appended by LTXVAddLatentGuide and is not real generated content.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | positive | CONDITIONING | #1519 LTXVAddLatentGuide pos (slot 0) via link 2832 |
  | negative | CONDITIONING | #1519 LTXVAddLatentGuide neg (slot 1) via link 2833 |
  | latent | LATENT | #596 LTXVSeparateAVLatent video_latent (slot 0) via link 1713 |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | positive | CONDITIONING | unwired |
  | negative | CONDITIONING | unwired |
  | latent | LATENT | #2005 LatentOverlapTrim (slot 0) via link 2959 -- also provides full latent back to TensorLoopClose as previous_value |

#### Node 2005 -- LatentOverlapTrim ("Overlap Trim")
- **Type**: `LatentOverlapTrim` (AudioLoopHelper: `nodes.py`)
- **What**: Trims the first N latent frames (the overlap context region) from the sampler output, keeping only the newly generated content. **Strips noise_mask** for clean accumulation.
- **Source**: `nodes.py` lines 697-738
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | latent | LATENT | #655 LTXVCropGuides latent (slot 2) via link 2959 |
  | overlap_latent_frames | INT | Subgraph input slot 14 via link 2997 |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | trimmed | LATENT | Subgraph output slot 0 -> external #1540 TensorLoopClose processed via link 2960 |

### Subgraph Data Flow Summary

```
Input: previous_latent -----> LatentContextExtract(2004) ---> LTXVAudioVideoMask(606) -+
Input: audio ------------> TrimAudioDuration(600) -> AudioVAEEncode(598) -> Mask(606) --+
Input: init_image -------> VAEEncode(1520) -> LTXVAddLatentGuide(1519) <-- Mask(606) video_latent
Input: positive/negative -> LTXVAddLatentGuide(1519)                                   |
                                   |                                                   |
                    +--------------+--- latent -> ConcatAV(583) <-- Mask(606) audio_latent
                    |              |
                    |    pos/neg -> CFGGuider(644) -> SamplerCustomAdvanced(573) <-- ConcatAV(583)
                    |                                        |
                    |                        SeparateAV(596) <- output
                    |                                |
                    +--- pos/neg -> CropGuides(655) <- video_latent
                                         |
                                    LatentOverlapTrim(2005) -> OUTPUT
```

### noise_mask Flow Explanation

The noise_mask is the mechanism by which the sampler knows which latent frames to keep unchanged (context) and which to generate from noise.

1. **LatentContextExtract (#2004)** receives previous_latent which carries a stale noise_mask from the prior sampling pass. It **strips noise_mask** (`s.pop("noise_mask", None)`) and outputs only the tail frames.

2. **LTXVAudioVideoMask (#606)** receives the clean (no-mask) video latent. Since there is no existing mask, it creates a fresh mask from scratch:
   - Overlap context frames: mask=0 (fixed, do not denoise)
   - New frames to generate: mask=1 (add noise, denoise these)
   - Audio: mask=0 everywhere (audio_start==audio_end, so empty range -- keep real audio)

3. **LTXVAddLatentGuide (#1519)** appends guide frames to the latent and extends the noise_mask with guide-specific mask values (based on strength).

4. **SamplerCustomAdvanced (#573)** uses the noise_mask to selectively add noise only to mask=1 regions. Context and audio are preserved.

5. **LTXVCropGuides (#655)** crops the guide frames and their mask entries.

6. **LatentOverlapTrim (#2005)** trims overlap frames and **strips noise_mask** again, so the accumulated output is clean for downstream concat.

**What happens if noise_mask is preserved (not stripped):**
- The stale mask from iteration N gets fed into iteration N+1's LatentContextExtract
- LTXVAudioVideoMask receives a latent WITH an existing mask
- `existing_mask_mode: "add"` takes `max(existing, new)` -- stale mask values corrupt the new mask
- The sampler sees wrong mask semantics: context frames might get mask=1 (re-generated) or vice versa
- Result: sync loss, artifacts, visual corruption that compounds each iteration

---

## Stage 7: Loop Close & Output Assembly

### Node 1540 -- TensorLoopClose
- **Type**: `TensorLoopClose` (NativeLooping: `nodes.py`)
- **What**: Closes the loop. Accumulates the trimmed latent from each iteration. When `should_stop` is True, outputs the full accumulated latent batch.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | flow_control | FLOW_CONTROL | #1539 TensorLoopOpen (slot 0) |
  | processed | LATENT | #843 Extension extended_latent (slot 0) |
  | stop | BOOLEAN | #1582 AudioLoopController should_stop (slot 1) |
- **Widgets**: `accumulate`: `true`, `overlap`: `disabled`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | output | LATENT | #1605 LatentConcat samples2 (slot 1) |

### Node 1605 -- LatentConcat ("Prepend Initial Render")
- **Type**: `LatentConcat` (ComfyUI core: `comfy_extras/nodes_latent.py`)
- **What**: Concatenates the guide-cropped initial render latent (from #381) with the accumulated loop output (from #1540) along the temporal dimension. This prepends the first window's content so the final video starts from frame 0.
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | samples1 | LATENT | #381 LTXVCropGuides latent (slot 2) -- initial render (guide-stripped) |
  | samples2 | LATENT | #1540 TensorLoopClose output (slot 0) -- accumulated loop output |
  | dim | COMBO | `t` (temporal dimension) |
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | LATENT | LATENT | #1604 VAEDecodeTiled samples (slot 0) |

### Node 1598 -- Get_video_vae (Final Decode)
- **Type**: `GetNode` (KJNodes) -- retrieves `video_vae`
- **Outputs**: VAE -> #1604, #1590, #1591, #1597

### Node 1604 -- VAEDecodeTiled ("Final VAE Decode (once)")
- **Type**: `VAEDecodeTiled` (ComfyUI core)
- **What**: Decodes the full concatenated latent to pixel space. Uses tiled decoding to handle the potentially very long video without OOM. This is the ONLY VAE decode in the main pipeline (latent-space loop advantage).
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | samples | LATENT | #1605 LatentConcat (slot 0) |
  | vae | VAE | #1598 Get_video_vae (slot 0) |
- **Widgets**: `tile_size`: `320`, `overlap`: `240`, `temporal_size`: `32`, `temporal_overlap`: `16`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | IMAGE | IMAGE | #617 VHS_VideoCombine images (slot 0) |

### Node 604 -- Get_orig_audio
- **Type**: `GetNode` (KJNodes) -- retrieves `orig_audio` (full unseparated audio)
- **Outputs**: AUDIO -> #617 VHS_VideoCombine audio (slot 1)

### Node 617 -- VHS_VideoCombine (Final Output)
- **Type**: `VHS_VideoCombine` (VideoHelperSuite)
- **What**: Combines decoded frames with the original (unseparated) audio into the final MP4 video
- **Inputs**:
  | Input | Type | Source |
  |-------|------|--------|
  | images | IMAGE | #1604 VAEDecodeTiled (slot 0) |
  | audio | AUDIO | #604 Get_orig_audio (slot 0) |
  | meta_batch | VHS_BatchManager | unwired |
  | vae | VAE | unwired |
- **Widgets**: `frame_rate`: `25`, `loop_count`: `0`, `filename_prefix`: `LTX-2`, `format`: `video/h264-mp4`, `pix_fmt`: `yuv420p`, `crf`: `19`, `save_metadata`: `true`, `trim_to_audio`: `true`, `pingpong`: `false`, `save_output`: `true`
- **Outputs**:
  | Output | Type | Connected To |
  |--------|------|-------------|
  | Filenames | VHS_FILENAMES | unwired |

---

## Bypassed Upscale Chain (mode=4)

These nodes are present but bypassed. They implement a post-loop 2x spatial upscale pipeline.

| Node ID | Type | Purpose |
|---------|------|---------|
| 1589 | LatentUpscaleModelLoader | Load `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` |
| 1590 | VAEEncode | Encode decoded video back to latent for upscaling |
| 1591 | LTXVLatentUpsampler | 2x spatial upscale in latent space |
| 1597 | VAEDecodeTiled | Decode upscaled latent |

The upscale chain is bypassed because per-loop VAE round-trip quality loss and VRAM constraints make it impractical within this workflow. Upscaling should be done as a separate workflow.

---

## Utility / Display Nodes

| Node ID | Type | Purpose | Mode |
|---------|------|---------|------|
| 1423 | PreviewImage | Shows sigma schedule visualization | Active |
| 1586 | PreviewAny | Shows overlap_frames value | Active |
| 618 | Reroute | Routes initial render preview image | Active |
| 1585 | Note | Contains example prompt schedule text | Active |
| 1533 | Note | Documents MelBandRoFormer usage | Active |

---

## User-Configurable Values

### Core Timing

| Widget | Node | Default | What It Controls | Valid Range |
|--------|------|---------|-----------------|-------------|
| `overlap_seconds` | #1582 AudioLoopController | `1` | Seconds of overlap between consecutive windows. Stride = window - overlap. More overlap = smoother transitions but slower. | 0.0 -- window_seconds |
| `window_size_seconds` | #688 FloatConstant | `19.88` | Duration of each generation window. Must match `length / fps` (497/25=19.88). | Must satisfy 8n+1 frame rule |
| `iterations` | #1539 TensorLoopOpen | `50` | Max loop iterations. AudioLoopController's should_stop overrides this when audio runs out. | 0+ |
| `fps` | #1582 AudioLoopController | `25` | Video frame rate for overlap_frames calculation | 1+ |

### Sampling

| Widget | Node | Default | What It Controls | Valid Range |
|--------|------|---------|-----------------|-------------|
| `sampler_name` | #154 KSamplerSelect | `euler_ancestral` | Sampling algorithm | Any ComfyUI sampler |
| `scheduler` | #1421 BasicScheduler | `linear_quadratic` | Sigma schedule shape | Any ComfyUI scheduler |
| `steps` | #1421 BasicScheduler | `8` | Number of denoising steps per window | 1+ |
| `denoise` | #1421 BasicScheduler | `1` | Denoise strength (1.0 = full denoise) | 0.0 -- 1.0 |
| `shift` | #1513 ModelSamplingSD3 | `13` | Sigma shift for sampling distribution | 0+ |
| `cfg` | #153 CFGGuider (initial), #644 (subgraph) | `1` | CFG scale (1.0 because NAG handles guidance) | 0+ |
| `start_seed` (value) | #1527 INTConstant | `42` | Base seed. iteration_seed = seed + current_iteration | 0 -- 2^64 |

### NAG (Guidance)

| Widget | Node | Default | What It Controls | Valid Range |
|--------|------|---------|-----------------|-------------|
| `nag_scale` | #508 LTX2_NAG | `11` | NAG guidance strength (replaces CFG for distilled models) | 0+ |
| `nag_init_scale` | #508 LTX2_NAG | `0.25` | Initial NAG scale at high sigma | 0+ |
| `nag_sigma_end` | #508 LTX2_NAG | `2.5` | Sigma below which NAG is fully active | 0+ |
| `rescale_cfg` | #508 LTX2_NAG | `true` | Whether to rescale CFG output | true/false |

### Model Optimization

| Widget | Node | Default | What It Controls | Valid Range |
|--------|------|---------|-----------------|-------------|
| `chunk_size` | #504 LTXVChunkFeedForward | `2` | Feed-forward chunking (lower = less VRAM, slower) | 1+ |
| `dim` | #504 LTXVChunkFeedForward | `4096` | Feed-forward dimension | Model-dependent |
| `preview_interval` | #503 LTX2SamplingPreviewOverride | `8` | Preview decode every N steps | 1+ |
| `attention_backend` | #268 PathchSageAttentionKJ | `auto` | Attention implementation (bypassed) | auto, sage, etc. |

### Attention Tuning

| Widget | Node | Default | What It Controls | Valid Range |
|--------|------|---------|-----------------|-------------|
| `self_attn_scale` | #1523 LTX2AttentionTunerPatch | `1` | Self-attention weight scale | 0+ |
| `cross_attn_text_scale` | #1523 LTX2AttentionTunerPatch | `1` | Text cross-attention scale | 0+ |
| `cross_attn_guide_scale` | #1523 LTX2AttentionTunerPatch | `1` | Guide cross-attention scale | 0+ |
| `cross_attn_ref_scale` | #1523 LTX2AttentionTunerPatch | `1` | Reference cross-attention scale | 0+ |

### Prompts

| Widget | Node | Default | What It Controls | Valid Range |
|--------|------|---------|-----------------|-------------|
| `text` (positive) | #169 CLIPTextEncode | `"video of a woman passionately singing alone"` | Base positive prompt for initial render and loop (when schedule bypassed) | Free text |
| `text` (negative) | #507 CLIPTextEncode | Long negative prompt (see Stage 3) | NAG negative conditioning text | Free text |
| `schedule` | #1558 TimestampPromptSchedule | `"0:00+: video of a woman passionately singing alone"` | Timestamp-based per-iteration prompt schedule (currently bypassed) | Timestamp schedule format |
| `blend_seconds` | #1558 TimestampPromptSchedule | `0` | Transition blend duration between schedule entries | 0+ seconds |

### Audio

| Widget | Node | Default | What It Controls | Valid Range |
|--------|------|---------|-----------------|-------------|
| `audio` | #565 LoadAudio | `example_audio.mp4` | Source audio file | Any supported audio |
| `start_index` | #567 TrimAudioDuration | `5` | Seconds to skip at start (instrumental intro trim) | 0+ |
| `duration` | #567 TrimAudioDuration | `300` | Max audio duration cap in seconds | 0+ |

### Image / Resolution

| Widget | Node | Default | What It Controls | Valid Range |
|--------|------|---------|-----------------|-------------|
| `image` | #444 LoadImage | `reference_image.png` | Init/reference image | Any image |
| `width` | #445 ImageResizeKJv2 | `832` | Generation width | Must be divisible by 32 |
| `height` | #445 ImageResizeKJv2 | `480` | Generation height | Must be divisible by 32 |
| `length` | #526 PrimitiveNode | `497` | Frames per window (must be 8n+1) | 1, 9, 17, ..., 497 |

### Guide Strength

| Widget | Node | Default | What It Controls | Valid Range |
|--------|------|---------|-----------------|-------------|
| `first_frame_guide_strength` | #1269 FloatConstant | `1` | Init image guide strength in extension subgraph. 1.0 = frame spatially frozen (no noise added). | 0.0 -- 1.0 |

### Final Output

| Widget | Node | Default | What It Controls | Valid Range |
|--------|------|---------|-----------------|-------------|
| `frame_rate` | #617 VHS_VideoCombine | `25` | Output video frame rate | 1+ |
| `crf` | #617 VHS_VideoCombine | `19` | H.264 quality (lower = better quality, larger file) | 0-51 |
| `trim_to_audio` | #617 VHS_VideoCombine | `true` | Trim video to match audio duration | true/false |
| `tile_size` | #1604 VAEDecodeTiled | `320` | Spatial tile size for VAE decode | 64+ |
| `overlap` (tile) | #1604 VAEDecodeTiled | `240` | Spatial tile overlap | 0+ |
| `temporal_size` | #1604 VAEDecodeTiled | `32` | Temporal tile size for VAE decode | 1+ |
| `temporal_overlap` | #1604 VAEDecodeTiled | `16` | Temporal tile overlap | 0+ |

---

## Custom Node Source Locations

| Package | Source Path |
|---------|------------|
| AudioLoopHelper (our nodes) | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-AudioLoopHelper/nodes.py` |
| NativeLooping | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-NativeLooping_testing/nodes.py` |
| KJNodes (LTX) | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py` |
| KJNodes (model opt) | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/model_optimization_nodes.py` |
| ComfyUI-LTXVideo | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-LTXVideo/latents.py` |
| VideoHelperSuite | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/` |
| MelBandRoFormer | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-MelBandRoFormer/nodes.py` |
| ComfyUI core (LTX) | `/home/fbliss/ComfyUI/comfy_extras/nodes_lt.py` |
| ComfyUI core (latent) | `/home/fbliss/ComfyUI/comfy_extras/nodes_latent.py` |
| ComfyUI core (nodes) | `/home/fbliss/ComfyUI/nodes.py` |

---

## Complete Node Index (Execution Order)

| Order | Node ID | Type | Display Name / Title | Stage |
|-------|---------|------|---------------------|-------|
| 0 | 604 | GetNode | Get_orig_audio | Output |
| 1 | 154 | KSamplerSelect | KSamplerSelect | Sampling Config |
| 2 | 1269 | FloatConstant | FloatConstant (guide strength) | Loop Setup |
| 3 | 568 | MelBandRoFormerModelLoader | MelBandRoFormerModelLoader | Audio Prep |
| 4 | 254 | GetNode | Get_audio_vae | Audio Prep |
| 5 | 571 | SolidMask | SolidMask | Audio Prep |
| 6 | 413 | GetNode | Get_video_vae | Initial Render |
| 7 | 688 | FloatConstant | window_size_seconds | Audio Prep |
| 8 | 1533 | Note | (MelBand note) | -- |
| 9 | 414 | UNETLoader | UNETLoader | Model Loading |
| 10 | 1537 | VAELoaderKJ | VAELoaderKJ (video) | Model Loading |
| 11 | 1538 | VAELoaderKJ | VAELoaderKJ (audio) | Model Loading |
| 12 | 416 | DualCLIPLoader | DualCLIPLoader | Model Loading |
| 13 | 526 | PrimitiveNode | length (497) | Initial Render |
| 14-30 | various | GetNode | Various Get nodes | Loop Setup |
| 31 | 1585 | Note | (prompt schedule note) | -- |
| 32 | 444 | LoadImage | LoadImage | Image Loading |
| 33 | 576 | SetNode | Set_sampler | Sampling Config |
| 34 | 1271 | SetNode | Set_first_frame_guide_strength | Loop Setup |
| 35 | 689 | SetNode | Set_window_size_seconds | Audio Prep |
| 36 | 268 | PathchSageAttentionKJ | PathchSageAttentionKJ (bypassed) | Model Loading |
| 37 | 228 | SetNode | Set_video_vae | Model Loading |
| 38 | 252 | SetNode | Set_audio_vae | Model Loading |
| 39 | 169 | CLIPTextEncode | CLIPTextEncode (positive) | Text Encoding |
| 40 | 507 | CLIPTextEncode | CLIPTextEncode (negative/NAG) | Text Encoding |
| 41 | 1322 | RandomNoise | RandomNoise (initial) | Initial Render |
| 42 | 1528 | SetNode | Set_start_seed | Loop Setup |
| 43 | 567 | TrimAudioDuration | TrimAudioDuration (intro skip) | Audio Prep |
| 44 | 445 | ImageResizeKJv2 | ImageResizeKJv2 | Image Loading |
| 45 | 504 | LTXVChunkFeedForward | LTXVChunkFeedForward | Model Loading |
| 46 | 420 | ConditioningZeroOut | ConditioningZeroOut | Text Encoding |
| 47 | 569 | MelBandRoFormerSampler | MelBandRoFormerSampler | Audio Prep |
| 48 | 581 | SetNode | Set_orig_audio | Audio Prep |
| 49 | 446 | LTXVPreprocess | LTXVPreprocess | Image Loading |
| 50 | 650 | SetNode | Set_input_image | Image Loading |
| 51 | 344 | EmptyLTXVLatentVideo | EmptyLTXVLatentVideo | Initial Render |
| 52 | 1523 | LTX2AttentionTunerPatch | LTX2AttentionTunerPatch | Model Loading |
| 53 | 164 | LTXVConditioning | LTXVConditioning (initial) | Text Encoding |
| 54 | 640 | SetNode | Set_actual_audio | Audio Prep |
| 55 | 531 | LTXVImgToVideoInplaceKJ | LTXVImgToVideoInplaceKJ | Initial Render |
| 56 | 508 | LTX2_NAG | LTX2_NAG | Model Loading |
| 57 | 645 | SetNode | Set_base_cond_pos | Text Encoding |
| 58 | 646 | SetNode | Set_base_cond_neg | Text Encoding |
| 59 | 601 | TrimAudioDuration | TrimAudioDuration (window) | Audio Prep |
| 60 | 503 | LTX2SamplingPreviewOverride | LTX2SamplingPreviewOverride | Model Loading |
| 61 | 566 | LTXVAudioVAEEncode | LTXVAudioVAEEncode | Audio Prep |
| 62 | 572 | SetNode | Set_model | Model Loading |
| 63 | 1513 | ModelSamplingSD3 | ModelSamplingSD3 | Sampling Config |
| 64 | 570 | SetLatentNoiseMask | SetLatentNoiseMask | Audio Prep |
| 65 | 153 | CFGGuider | CFGGuider (initial) | Initial Render |
| 66 | 1421 | BasicScheduler | BasicScheduler | Sampling Config |
| 67 | 350 | LTXVConcatAVLatent | LTXVConcatAVLatent (initial) | Initial Render |
| 68 | 575 | SetNode | Set_guider | Sampling Config |
| 69 | 1422 | VisualizeSigmasKJ | VisualizeSigmasKJ | Sampling Config |
| 70 | 579 | SetNode | Set_sigmas | Sampling Config |
| 71 | 1423 | PreviewImage | PreviewImage (sigmas) | Display |
| 72 | 161 | SamplerCustomAdvanced | SamplerCustomAdvanced (initial) | Initial Render |
| 73 | 245 | LTXVSeparateAVLatent | LTXVSeparateAVLatent (post-initial) | Initial Render |
| 74 | 381 | LTXVCropGuides | LTXVCropGuides (initial) | Initial Render |
| 75 | 1318 | VAEDecode | VAEDecode (preview) | Initial Render |
| 76 | 560 | VHS_VideoCombine | VHS_VideoCombine (preview, bypassed) | Display |
| 77 | 618 | Reroute | Reroute (preview image) | Display |
| 78 | 1539 | TensorLoopOpen | TensorLoopOpen | Loop Setup |
| 79 | 1582 | AudioLoopController | AudioLoopController | Loop Body |
| 80 | 1558 | TimestampPromptSchedule | Prompt Schedule (bypassed) | Loop Body |
| 81 | 1560 | AudioLoopPlanner | AudioLoopPlanner (bypassed) | Loop Body |
| 82 | 1586 | PreviewAny | PreviewAny (overlap_frames) | Display |
| 83 | 1559 | CLIPTextEncode | Loop Prompt Encode | Loop Body |
| 84 | 1587 | LTXVConditioning | Loop LTXVConditioning | Loop Body |
| 84 | 1563 | PreviewAny | Iteration Timestamps (bypassed) | Display |
| 85 | 843 | extension subgraph | Extension Subgraph | Loop Body |
| 86 | 1540 | TensorLoopClose | TensorLoopClose | Loop Close |
| 88 | 617 | VHS_VideoCombine | VHS_VideoCombine (final) | Output |
| 95 | 1604 | VAEDecodeTiled | Final VAE Decode (once) | Output |
| 96 | 1605 | LatentConcat | Prepend Initial Render | Output |
