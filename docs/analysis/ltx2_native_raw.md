Perfect! Now I have a comprehensive understanding. Let me create a detailed summary of all the capabilities found in the LTX-2 repository.

## Comprehensive Analysis: LTX-2 Repository Capabilities

Based on my thorough analysis of the canonical LTX-2 reference implementation, here are ALL capabilities organized by category:

---

### **CORE ARCHITECTURES & CAPABILITIES**

#### **1. Asymmetric Dual-Stream Audio-Video Transformer**
- **What it does**: Joint audio-video generation model (not sequential T2V→V2A)
- **Architecture**:
  - 48 transformer blocks
  - Video stream: 14B parameters (3D RoPE for spatial-temporal)
  - Audio stream: 5B parameters (1D RoPE for temporal only)
  - Bidirectional cross-modal attention (A↔V) using 1D temporal RoPE for sub-frame synchronization
  - Cross-modality AdaLN (Adaptive Layer Normalization) for sync across different timesteps/resolutions
- **Unique aspect**: Separate modality-specific text embeddings (video context 4096-dim, audio context 2048-dim) from same prompt, enabling better sync and natural speech

#### **2. Modality-Specific Text Encoding (Gemma 3)**
- **What it does**: Multi-stage text encoding pipeline
- **Components**:
  - Gemma 3 (12B) backbone decoder-only LLM
  - Multi-layer feature extractor (aggregates features from ALL decoder layers, not just last)
  - Learnable mean-centering scaling
  - Separate bidirectional transformer connectors with learnable "registers" (thinking tokens) for:
    - AVGemmaTextEncoderModel: Audio-video generation (two separate connectors)
    - VideoGemmaTextEncoderModel: Video-only generation
- **Prompt Enhancement**:
  - `enhance_t2v()`: Text-to-video prompt enhancement
  - `enhance_i2v()`: Image-to-video prompt enhancement (considers image context)

#### **3. System Prompts**
Located in `/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/`:
- **`gemma_t2v_system_prompt.txt`**: Comprehensive text-to-video prompt engineering instructions
  - Emphasizes active present-progressive verbs, chronological flow
  - Integrated audio descriptions (soundscape, SFX, speech with exact quoted dialogue)
  - Visual style guidance (cinematic-realistic default)
  - No timestamps or scene cuts unless requested
  - Avoids invented dialogue, restraint in language
- **`gemma_i2v_system_prompt.txt`**: Image-to-video variant
  - Analyzes input image for subject, setting, style, mood
  - Describes only changes from image to avoid scene cuts
  - Generates contextual dialogue if general conversation mentioned

---

### **VAE & COMPRESSION SYSTEMS**

#### **4. Video VAE**
- Encoder: `[B, 3, F, H, W]` → `[B, 128, F', H/32, W/32]`
  - Spatial: 32× downsampling
  - Temporal: F' = 1 + (F-1)/8 (frame count must satisfy (F-1) % 8 == 0)
  - Example: 33 frames at 512×512 → 5 frames at 16×16 latent
- Decoder: Reverses with 8× temporal upsampling
- Supports tiling for high-resolution/long-duration generation

#### **5. Audio VAE**
- Encoder: `[B, mel_bins, T]` → `[B, 8, T/4, 16]` latents
  - 4× temporal downsampling
  - Input: 16 kHz mel-spectrograms
  - Output: 8 channels, 16 mel bins (fixed frequency compression)
  - ~1/25s per token, 128-dim feature vector
- Decoder: Reverses spatial compression, 4× temporal upsampling
- **Vocoder**: HiFi-GAN modified for stereo synthesis, 16 kHz mel → 24 kHz waveform

#### **6. Spatial Upsampler**
- 2x spatial upsampling (x2 or x1.5 variants available)
- Used in Stage 2 of two-stage pipelines
- Supports memory-efficient sequential upsampling

---

### **INFERENCE PIPELINES**

#### **7. Text/Image-to-Video (TI2VidTwoStagesPipeline)**
- **Stages**:
  - Stage 1: Low-resolution generation at H/2, W/2 with full CFG + STG guidance
  - Stage 2: 2x upsampling + refinement with distilled LoRA
- **Features**:
  - Multimodal guidance (separate video & audio guiders)
  - Image conditioning (replacing or guiding latents)
  - Prompt enhancement support
  - Two sampler variants: standard Euler, second-order res_2s

#### **8. High-Quality Variant (TI2VidTwoStagesHQPipeline)**
- Same two-stage as above but:
  - Uses **res_2s second-order sampler** instead of Euler
  - Fewer steps (15 vs 40) while maintaining quality
  - Optimized for LTX-2.3 models specifically

#### **9. Single-Stage (TI2VidOneStagePipeline)**
- Generates at target resolution in single stage
- Educational use; significantly lower quality than two-stage
- Faster but not recommended for production

#### **10. Distilled Pipeline (DistilledPipeline)**
- **Fastest inference**: 8 predefined sigmas (8 steps stage 1, 4 steps stage 2)
- Uses distilled model only
- No guidance required
- Significantly faster, decent quality trade-off
- Fixed sigma schedule: `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]`

#### **11. Audio-to-Video (A2VidPipelineTwoStage)**
- **Unique aspect**: Audio-driven video generation
- **Workflow**:
  - Encodes input audio → audio latent (frozen during generation)
  - Stage 1: Generates video at H/2, W/2 with audio conditioning (video-only denoising)
  - Stage 2: Upsamples & refines both modalities with distilled LoRA
- **Special handling**: Returns original input audio waveform (not VAE-decoded) to preserve fidelity
- Supports image conditioning + prompt enhancement

#### **12. Video-to-Video with IC-LoRA (ICLoraPipeline)**
- **Unique aspect**: Reference video conditioning for style transfer/transformation
- **Features**:
  - Condition on entire reference videos (not just images)
  - **Attention strength control**: `conditioning_attention_strength` [0.0-1.0] parameter
  - **Spatial attention mask**: Optional pixel-space mask for targeted control
  - Reference downscaling: Reads from LoRA metadata for training-conditioned resolution
  - Two-stage generation (distilled model)
  - Supports multiple IC-LoRA types (Union Control, Motion Track Control, Detailer, Pose Control)

#### **13. Keyframe Interpolation (KeyframeInterpolationPipeline)**
- **Unique aspect**: Smooth interpolation between keyframe images
- **Method**: Additive conditioning (guiding latents) instead of replacing for smoother transitions
- Two-stage with full model Stage 1 + distilled LoRA Stage 2
- Uses multimodal guidance

#### **14. Temporal Editing (RetakePipeline)**
- **Unique aspect**: Regenerate specific time regions of existing video
- **Features**:
  - `TemporalRegionMask` conditioning item sets denoise_mask outside [start_time, end_time]
  - Preserves content outside time window
  - Independent control: `regenerate_video` and `regenerate_audio` flags
  - Can use full model with CFG or distilled model with fixed sigma
  - Works in both pixel space and latent space with proper temporal bounds
  - Constraints: Source video must be 8k+1 frames (e.g., 97, 193)

---

### **GUIDANCE & CONTROL SYSTEMS**

#### **15. Multimodal Guidance Framework**
- **Components**:
  1. **CFG (Classifier-Free Guidance)**: Text prompt adherence (scale 1.0-5.0 typical)
  2. **STG (Spatio-Temporal Guidance)**: Perturbation-based temporal coherence (scale 0.5-1.5)
  3. **Modality CFG**: Steers away from unsynced audio-video (scale 1.0-3.0)
  4. **Rescaling**: Variance matching to prevent over-saturation (scale 0.0-0.7)

- **Perturbation System** (`src/ltx_core/guidance/perturbations.py`):
  - Skip perturbations for specific attention types:
    - `SKIP_A2V_CROSS_ATTN`: Disable audio-to-video attention
    - `SKIP_V2A_CROSS_ATTN`: Disable video-to-audio attention
    - `SKIP_VIDEO_SELF_ATTN`: Disable video self-attention
    - `SKIP_AUDIO_SELF_ATTN`: Disable audio self-attention
  - Block-specific: Can target specific transformer blocks (e.g., `[29]` for last block)

- **MultiModalGuiderFactory**: Sigma-dependent guidance parameters (different params at different noise levels)

#### **16. Advanced Guidance Variants**
- **CFGGuider**: Basic CFG: `delta = (scale - 1) * (cond - uncond)`
- **CFGStarRescalingGuider**: Rescaled unconditioned sample for minimal offset
- **STGGuider**: Perturbation delta: `delta = scale * (pos_denoised - perturbed_denoised)`
- **LtxAPGGuider**: Adaptive Projected Guidance - decomposes guidance into parallel & orthogonal components
- **LegacyStatefulAPGGuider**: APG with momentum accumulation for multi-step guidance

#### **17. Sampling & Denoising Strategies**
- **Euler Denoising Loop**: Standard diffusion denoising
- **Gradient Estimation Denoising** (`gradient_estimating_euler_denoising_loop`):
  - Velocity-based correction using previous steps
  - Enables fewer steps (20-30 vs 40) with comparable quality
  - `ge_gamma` parameter (default 2.0) controls correction strength
  - Citation: https://openreview.net/pdf?id=o2ND9v0CeK

#### **18. Scheduling & Noise Control**
- **LTX2Scheduler**: Adaptive noise schedule that respects token count
- **Sigma Schedules**:
  - Full model: Customizable via `num_inference_steps` (typically 40)
  - Distilled: Predefined 8-step schedule
  - Stage 2: Reduced 4-step schedule `[0.909375, 0.725, 0.421875, 0.0]`

---

### **CONDITIONING SYSTEMS**

#### **19. Image Conditioning Methods**
- **By Replacing Latent**: `VideoConditionByLatentIndex` - replaces latent at specific frame
  - Strong control over specific frames
  - Used by: TI2VidOneStagePipeline, TI2VidTwoStagesPipeline, DistilledPipeline
- **By Adding Guiding Latent**: `VideoConditionByKeyframeIndex` - additive conditioning
  - Better for smooth interpolation
  - Used by: KeyframeInterpolationPipeline
- **Reference Video**: `VideoConditionByReferenceLatent` - condition on full videos
  - Used by: ICLoraPipeline
  - Supports attention strength & spatial masking

#### **20. Video Conditioning**
- **VideoConditionByKeyframeIndex**: Keyframe at specific frame indices
- **VideoConditionByLatentIndex**: Replace entire latent representation
- **VideoConditionByReferenceLatent**: Reference video with attention control
- **ConditioningItemAttentionStrengthWrapper**: Wraps any conditioning with attention strength scaling

#### **21. Custom Conditioning (Temporal Region Mask)**
- `TemporalRegionMask`: Sets denoise_mask=0 outside [start_time, end_time]
- Works in patchified (token) space
- Handles both video (latent frame indices) and audio (seconds) coordinates

---

### **QUANTIZATION & OPTIMIZATION**

#### **22. FP8 Quantization**
- **FP8 Cast**: Simple downcast of weights to FP8, upcast during inference
  - No extra dependencies
  - Works on any GPU with FP8 support
- **FP8 Scaled MM** (TensorRT-LLM):
  - Per-tensor scaling factors
  - Dynamic or static (calibration file) input quantization
  - Best on Hopper GPUs
  - Requires `tensorrt_llm` package

#### **23. Memory Optimization**
- Automatic memory cleanup between stages
- LoRA loading on CPU by default (transfers to GPU during fusion)
- Optional: Load LoRAs directly to GPU if space permits
- Environment variable: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

#### **24. LoRA Support & Fusion**
- Multiple LoRA adapters with independent strength values
- Fuse during model loading (no runtime overhead)
- Weight types supported:
  - bfloat16
  - float8_e4m3fn (scaled and cast variants)
- Specialized fusion for FP8: dequantize → add delta → re-quantize

---

### **ADVANCED FEATURES**

#### **25. Rotary Positional Embeddings (RoPE) Variants**
- **3D RoPE (Video)**: Spatial (x,y) + temporal (t)
  - Positional embeddings: [20, 2048, 2048] (default)
  - Fractional positions scaled to [-1, 1]
- **1D RoPE (Audio)**: Temporal only
  - Positional embeddings: [20] (default)
- **Cross-Attention RoPE**: 1D temporal for audio-video synchronization
- **Rope Types**: Interleaved vs Split application modes

#### **26. Timestep Embedding**
- Diffusion timesteps converted to embeddings
- Scaled by `timestep_scale_multiplier` (default 1000)
- Used by AdaLN for conditioning

#### **27. Adaptive Layer Normalization (AdaLN)**
- Per-block timestep conditioning
- Cross-modality variants for A↔V attention
- Separate scale/shift parameters per modality

#### **28. Attention Patterns**
- Self-attention within each modality
- Text cross-attention (separate contexts for video/audio)
- Audio-visual cross-attention (bidirectional)
- Gated attention support (optional)
- Attention masking support

#### **29. Prompt Enhancement via Gemma**
- Text-to-video: `text_encoder.enhance_t2v(prompt, seed)`
- Image-to-video: `text_encoder.enhance_i2v(prompt, image, seed)`
- Uses learned Gemma representations to expand/refine prompts
- Seed-deterministic for reproducibility

---

### **TRAINING & FINE-TUNING (ltx-trainer package)**

#### **30. LoRA Training**
- Low-rank adaptation on transformer weights
- Configurable rank and scaling
- Supported training modes:
  - Standard text-to-video LoRA
  - IC-LoRA for video-to-video transformations
  - LoRA for specific control types (Pose, Motion Track, Detailer, Camera, Union Control)

#### **31. Training Strategies**
- **Text-to-Video Strategy**: Standard T2V fine-tuning
- **Video-to-Video Strategy**: IC-LoRA training on reference videos
- Base strategy framework for custom training

#### **32. Dataset Support**
- Video preparation with frame extraction
- Caption generation from videos
- Dataset splitting and validation
- Reference latent computation for IC-LoRA training

#### **33. Quantization Options for Training**
- INT8 quantization for low-VRAM setups (32GB GPUs)
- FP8 quantization variants
- Configuration presets for different GPU memory tiers

---

### **UTILITIES & HELPERS**

#### **34. Model Loading & Management**
- **SingleGPUModelBuilder**: Frozen dataclass for loading + fusing LoRAs
- **ModelLedger**: Manages multiple model components (transformer, VAEs, upsampler, text encoder)
- Weight remapping and registry system
- Device management (CPU staging, GPU loading)

#### **35. Media I/O**
- Video decoding/encoding (with frame rate control)
- Audio decoding/encoding (24 kHz waveform)
- Image loading with aspect-ratio preservation
- Tiling support for high-resolution VAE operations

#### **36. Type System & Constants**
- `VideoPixelShape`, `VideoLatentShape`, `AudioLatentShape`
- `LatentState`: Tuple of (latent, denoise_mask, clean_latent, positions)
- `Modality`: Transformer input format (enabled, latent, sigma, timesteps, positions, context)
- Scale factors: `SpatioTemporalScaleFactors`
- Default negative prompt library with comprehensive artifacts list

---

### **TRANSFORMER INTERNALS**

#### **37. TransformerConfig & Preprocessing**
- `MultiModalTransformerArgsPreprocessor`: Handles video/audio token preparation
- Per-modality patchification with separate dimension handling
- Modality input format: `Modality` dataclass with enabled flag, latent, sigma, timesteps

#### **38. Model Types**
- `LTXModelType.AudioVideo`: Full dual-stream (default)
- `LTXModelType.VideoOnly`: Video stream only
- `LTXModelType.AudioOnly`: Audio stream only (rare)

---

### **DEFAULT HYPERPARAMETERS & SETTINGS**

#### **39. Pipeline Parameter Presets**
- **LTX-2.0 defaults**: 40 steps, STG blocks [29]
- **LTX-2.3 defaults**: 30 steps, STG blocks [28]
- **LTX-2.3 HQ defaults**: 15 steps (res_2s), higher resolution
- **Video guider defaults**: cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7, modality_scale=3.0
- **Audio guider defaults**: cfg_scale=7.0 (stronger text adherence), same STG/rescale/modality

#### **40. Default Negative Prompt**
Comprehensive list covering:
- Video artifacts (blur, noise, distortion, artifacts)
- Audio issues (muted, distorted, echo, off-sync)
- Animation problems (jitter, unnatural transitions, tilted camera)
- Face/body issues (deformed features, wrong proportions, extra limbs)

---

### **DISTINGUISHING FEATURES NOT IN COMFYUI-LTXVIDEO**

1. **Prompt Enhancement**: Integrated enhance_t2v() and enhance_i2v() methods via Gemma
2. **System Prompts**: Detailed prompt engineering instructions built into text encoder
3. **Audio-to-Video**: Dedicated pipeline with audio conditioning (frozen during video denoising)
4. **IC-LoRA with Attention Masks**: Pixel-space spatial control masks for conditioning
5. **Temporal Editing**: RetakePipeline for selective regeneration of time regions
6. **Gradient Estimation Sampling**: Velocity-based denoising for fewer steps
7. **Advanced Guidance**: APG (Adaptive Projected Guidance) with momentum variants
8. **Sigma-Dependent Guidance**: MultiModalGuiderFactory for step-wise param variation
9. **Perturbation-Based Control**: Fine-grained attention skipping (STG with specific blocks/types)
10. **Multi-Layer Text Features**: Aggregates from ALL Gemma layers, not just final
11. **Cross-Modality AdaLN**: Timestep-conditioned scaling between audio-video streams
12. **Modality-Specific Text Embeddings**: Separate video (4096-dim) and audio (2048-dim) contexts from single prompt
13. **Reference Downscaling Metadata**: IC-LoRA models encode their training resolution for automatic inference adaptation
14. **Causal Attention Handling**: Special temporal coordinate handling in patchifiers
15. **Video-Only Denoising Mode**: A2VidPipeline freezes audio latents while denoising video
16. **LoRA Metadata Reading**: Dynamic detection of model parameters from checkpoint config

---

### **KEY FILES & ENTRY POINTS**

- **Pipelines**: `/packages/ltx-pipelines/src/ltx_pipelines/*.py`
- **Core Models**: `/packages/ltx-core/src/ltx_core/model/`
- **Guidance**: `/packages/ltx-core/src/ltx_core/components/guiders.py`
- **Conditioning**: `/packages/ltx-core/src/ltx_core/conditioning/`
- **Text Encoding**: `/packages/ltx-core/src/ltx_core/text_encoders/gemma/`
- **Training**: `/packages/ltx-trainer/src/ltx_trainer/`
- **Sampling**: `/packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py`

This is the complete, production-grade reference implementation of LTX-2 audio-video generation.