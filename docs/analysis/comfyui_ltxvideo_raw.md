Perfect! Now let me compile the comprehensive inventory. I have enough information to create a complete capabilities inventory. Let me organize and present all the findings:

## ComfyUI-LTXVideo Complete Capabilities Inventory

Based on comprehensive analysis of the ComfyUI-LTXVideo codebase, here is a complete structured inventory of all node types and capabilities:

---

### CORE SAMPLING NODES

**1. LTXVBaseSampler**
- Category: `sampling`
- Function: Basic text-to-video generation with optional image conditioning on first frame
- Key Parameters: width, height, num_frames, guider, sampler, sigmas, noise, optional_cond_images, strength
- Unique to ComfyUI: Yes - convenience wrapper around core LTX-2 diffusion model

**2. LTXVExtendSampler**
- Category: `sampling`
- Function: Extends an existing video by conditioning on previous frames
- Key Parameters: similar to BaseSampler with latent conditioning options
- Unique to ComfyUI: Yes - enables video continuation

**3. LTXVInContextSampler**
- Category: `sampling`
- Function: In-context video generation that respects previously denoised latents
- Key Parameters: guiding_strength, optional_cond_images, strength
- Unique to ComfyUI: Yes - for seamless video editing

**4. LTXVLoopingSampler**
- Category: `sampling`
- Function: Generates very long videos using temporal tiling with overlapping regions
- Key Parameters: temporal_tile_size, temporal_overlap, temporal_overlap_cond_strength, horizontal_tiles, vertical_tiles, spatial_overlap, guiding_strength, adain_factor
- Unique to ComfyUI: Yes - handles videos beyond model's native frame limit

**5. LTXVTiledSampler**
- Category: `sampling`
- Function: Spatial tiling sampler for generating higher resolution videos with tile blending
- Key Parameters: horizontal_tiles, vertical_tiles, overlap, latents_cond_strength, boost_latent_similarity
- Unique to ComfyUI: Yes - memory-efficient high-resolution generation

**6. LTXVNormalizingSampler**
- Category: `sampling`
- Function: Sampler that normalizes output latents during inference
- Key Parameters: normalization parameters
- Unique to ComfyUI: Yes - statistical consistency

---

### GUIDER & GUIDANCE NODES

**7. STGGuiderNode**
- Category: `guiding`
- Function: Basic Spatio-Temporal Guidance (STG) guider for diffusion sampling
- Key Parameters: CFG scale, STG scale, rescale factor
- Unique to ComfyUI: Yes - advanced guidance mechanism

**8. STGGuiderAdvancedNode**
- Category: `guiding`
- Function: Advanced STG with per-modality control (video/audio)
- Key Parameters: video_cfg, video_stg, audio_cfg, audio_stg, advanced parameters
- Unique to ComfyUI: Yes - multi-modal guidance

**9. STGAdvancedPresetsNode**
- Category: `guiding`
- Function: Pre-configured STG parameter sets for common use cases
- Key Parameters: preset selection (loads from stg_advanced_presets.json)
- Unique to ComfyUI: Yes

**10. MultimodalGuiderNode**
- Category: `guiding`
- Function: Combines video and audio modality guidance in single node
- Key Parameters: separate CFG/STG for video and audio
- Unique to ComfyUI: Yes - audio-video coordination

**11. GuiderParametersNode**
- Category: `guiding`
- Function: Builds parameter dictionaries for multi-modal guidance
- Key Parameters: modality (VIDEO/AUDIO), cfg, stg, perturb_attn, rescale, modality_scale, skip_step, cross_attn
- Unique to ComfyUI: Yes - guidance parameter builder

**12. LTXVApplySTG**
- Category: `lightricks/LTXV`
- Function: Applies STG technique to specific transformer blocks
- Key Parameters: model, block_indices (comma-separated)
- Unique to ComfyUI: Yes - selective block-level guidance

---

### GUIDE & KEYFRAME NODES

**13. LTXVAddGuideAdvanced**
- Category: `conditioning/video_models`
- Function: Adds keyframe/video conditioning at specific frame index with preprocessing
- Key Parameters: frame_idx, strength, crf, blur_radius, interpolation, crop
- Unique to ComfyUI: Yes - flexible keyframe insertion

**14. LTXVAddGuideAdvancedAttention**
- Category: `conditioning/video_models`
- Function: Guide node with per-frame attention strength control
- Key Parameters: attention_strength, latent_shape, attention_mask
- Unique to ComfyUI: Yes - attention-weighted guidance

**15. LTXVAddLatentGuide**
- Category: `conditioning/video_models`
- Function: Adds guidance using latent space (not image space)
- Key Parameters: guiding_latent, latent_idx, strength
- Unique to ComfyUI: Yes - latent-space guidance

**16. LTXAddVideoICLoRAGuide**
- Category: `Lightricks/IC-LoRA`
- Function: Adds IC-LoRA conditioning guides from images
- Key Parameters: frame_idx, strength, latent_downscale_factor, crop, use_tiled_encode, tile_size
- Unique to ComfyUI: Yes - IC-LoRA-specific guide

**17. LTXAddVideoICLoRAGuideAdvanced**
- Category: `Lightricks/IC-LoRA`
- Function: Advanced IC-LoRA guide with additional control options
- Key Parameters: multi-frame support, downscaling, tiled encoding
- Unique to ComfyUI: Yes

---

### LATENT MANIPULATION NODES

**18. LTXVSelectLatents**
- Category: `latent/video`
- Function: Selects frame range from video latents with positive/negative indexing
- Key Parameters: start_index, end_index (supports -1 for last frame)
- Unique to ComfyUI: Yes - flexible latent slicing

**19. LTXVAddLatents**
- Category: `latent/video`
- Function: Concatenates two video latents along frame dimension
- Key Parameters: latents1, latents2
- Unique to ComfyUI: Yes - latent composition

**20. LTXVSetVideoLatentNoiseMasks**
- Category: `latent/video`
- Function: Applies noise masks to video latents for inpainting
- Key Parameters: samples, masks (2D/3D/4D tensor support)
- Unique to ComfyUI: Yes - mask-based latent control

**21. LTXVImgToVideoConditionOnly**
- Category: `latent/video`
- Function: Encodes image to latent space for conditioning without full VAE decode
- Key Parameters: image, vae
- Unique to ComfyUI: Yes - lightweight image encoding

---

### LATENT NORMALIZATION NODES

**22. LTXVAdainLatent**
- Category: `Lightricks/latents`
- Function: Applies AdaIN (Adaptive Instance Normalization) to normalize latent statistics
- Key Parameters: latents, reference, factor, per_frame
- Unique to ComfyUI: Yes - style normalization

**23. LTXVStatNormLatent**
- Category: `Lightricks/latents`
- Function: Statistical normalization of latents (mean/std adjustment)
- Key Parameters: latents, target_mean, target_std
- Unique to ComfyUI: Yes - distribution matching

**24. LTXVPerStepAdainPatcher**
- Category: `Lightricks/latents`
- Function: Model patcher that applies per-step AdaIN during sampling
- Key Parameters: model, reference_latents, factor
- Unique to ComfyUI: Yes - dynamic per-step normalization

**25. LTXVPerStepStatNormPatcher**
- Category: `Lightricks/latents`
- Function: Model patcher for per-step statistical normalization
- Key Parameters: model, normalization parameters
- Unique to ComfyUI: Yes

---

### DYNAMIC CONDITIONING NODES

**26. DynamicConditioning**
- Category: `lightricks/LTXV`
- Function: Applies power-based denoising mask modulation for first frame emphasis
- Key Parameters: power (1.0-2.0), only_first_frame (boolean)
- Unique to ComfyUI: Yes - per-step mask adjustment

---

### IMAGE PREPROCESSING NODES

**27. LTXVDilateVideoMask**
- Category: `Lightricks/mask_operations`
- Function: Spatial and temporal mask dilation using separable max-pooling
- Key Parameters: spatial_radius, temporal_radius, mask (or image_as_mask)
- Unique to ComfyUI: Yes - video mask morphology

**28. LTXVInpaintPreprocess**
- Category: `Lightricks/image_processing`
- Function: Composites images onto green background (#66FF00) for inpainting
- Key Parameters: images, mask (broadcasts single-frame to video length)
- Unique to ComfyUI: Yes - inpainting preparation

**29. LTXVPreprocessMasks**
- Category: `Lightricks/mask_operations`
- Function: Temporal mask processing with pooling, morphology, and clamping
- Key Parameters: masks, pooling_method (max/mean/min), grow_mask, tapered_corners, clamp_min/max, ignore_first_mask, invert_input_masks
- Unique to ComfyUI: Yes - comprehensive mask preprocessing

---

### VAE & DECODER NODES

**30. LTXVTiledVAEDecode**
- Category: `latent`
- Function: Memory-efficient VAE decoding using spatial tiling
- Key Parameters: horizontal_tiles, vertical_tiles, overlap, last_frame_fix, working_device, working_dtype
- Unique to ComfyUI: Yes - large-scale decoding

**31. Set VAE Decoder Noise (DecoderNoise)**
- Category: `lightricks/LTXV`
- Function: Adds stochastic noise to VAE decoder for improved quality
- Key Parameters: timestep (0.0-1.0), scale (0.0-1.0), seed
- Unique to ComfyUI: Yes - decoder regularization

**32. LTXVPatcherVAE**
- Category: `lightricks/LTXV`
- Function: Applies Q8 kernel optimizations to VAE for faster decoding
- Key Parameters: vae
- Unique to ComfyUI: Yes (requires q8_kernels)

---

### PROMPT & TEXT ENCODING NODES

**33. LTXVPromptEnhancerLoader**
- Category: `lightricks/LTXV`
- Function: Loads Llama-3.2 and Florence-2 models for prompt enhancement
- Key Parameters: llm_name, image_captioner_name
- Unique to ComfyUI: Yes - local prompt enhancement

**34. LTXVPromptEnhancer**
- Category: `lightricks/LTXV`
- Function: Enhances text prompts using LLMs and optional image reference
- Key Parameters: prompt, prompt_enhancer, max_resulting_tokens, optional image_prompt
- Unique to ComfyUI: Yes - cinematic prompt generation

**35. LTXVGemmaCLIPModelLoader**
- Category: `lightricks/LTXV`
- Function: Loads Gemma-3 multimodal text encoder
- Key Parameters: model_name, device
- Unique to ComfyUI: Yes - advanced text encoding

**36. LTXVGemmaEnhancePrompt**
- Category: `lightricks/LTXV`
- Function: Enhances prompts using local Gemma-3 with optional image context
- Key Parameters: prompt, gemma_model, optional_image
- Unique to ComfyUI: Yes

**37. GemmaAPITextEncode**
- Category: `api node/text/Lightricks`
- Function: Remote API-based text encoding using Lightricks servers
- Key Parameters: api_key, prompt, ckpt_name, enhance_prompt
- Unique to ComfyUI: Yes - cloud-based encoding

---

### CONDITIONING PERSISTENCE NODES

**38. LTXVLoadConditioning**
- Category: `lightricks/LTXV`
- Function: Loads pre-computed conditioning embeddings from SafeTensors files
- Key Parameters: file_name (from embeddings folder), device (cpu/gpu)
- Unique to ComfyUI: Yes - caching mechanism

**39. LTXVSaveConditioning**
- Category: `lightricks/LTXV`
- Function: Saves conditioning embeddings to SafeTensors for reuse
- Key Parameters: conditioning, filename, dtype (bfloat16/float16)
- Unique to ComfyUI: Yes - conditioning caching

---

### IC-LORA NODES

**40. LTXICLoRALoaderModelOnly**
- Category: `lightricks/IC-LoRA`
- Function: Loads IC-LoRA adapter weights without image encoding
- Key Parameters: lora_name, strength
- Unique to ComfyUI: Yes - LoRA-specific loading

---

### MOTION TRACKING NODES

**41. LTXVSparseTrackEditor**
- Category: `Lightricks/motion_tracking`
- Function: Interactive spline editor for drawing motion tracks on reference images
- Key Parameters: reference_image, num_frames (interpolates spline to output points)
- Unique to ComfyUI: Yes - visual track creation

**42. LTXVDrawTracks**
- Category: `Lightricks/motion_tracking`
- Function: Renders sparse motion tracks as image overlays
- Key Parameters: image, tracks, frame_idx, color, width, opacity
- Unique to ComfyUI: Yes - track visualization

---

### AUDIO-VIDEO NODES

**43. MultiPromptProvider**
- Category: `sampling`
- Function: Supplies changing prompts per temporal tile (for looping sampler)
- Key Parameters: prompts list, per-tile conditioning
- Unique to ComfyUI: Yes - temporal prompt variation

---

### QUANTIZATION & OPTIMIZATION NODES

**44. LTXVQ8LoraModelLoader**
- Category: `lightricks/LTXV`
- Function: Loads quantized LoRA adapters with Q8 kernels
- Key Parameters: lora_name, strength
- Unique to ComfyUI: Yes (requires q8_kernels) - memory optimization

**45. LTXQ8Patch (LTXVQ8Patch)**
- Category: `lightricks/LTXV`
- Function: Applies int8 quantization to model with selective layer control
- Key Parameters: model, use_fp8_attention, quantization_preset (0.9.8/ltxv2/full_bf16/custom), quantize_self_attn, quantize_cross_attn, quantize_ffn
- Unique to ComfyUI: Yes (requires q8_kernels) - memory reduction

---

### MODEL LOADING NODES

**46. LowVRAMCheckpointLoader**
- Category: `LTXV/loaders`
- Function: Loads checkpoint with dependency input for sequential loading
- Key Parameters: ckpt_name, optional dependencies (for execution order control)
- Unique to ComfyUI: Yes - VRAM optimization

**47. LowVRAMAudioVAELoader**
- Category: `LTXV/loaders`
- Function: Loads audio VAE with sequential loading support
- Key Parameters: ckpt_name, optional dependencies
- Unique to ComfyUI: Yes - audio-specific VAE

**48. LowVRAMLatentUpscaleModelLoader**
- Category: `LTXV/loaders`
- Function: Loads latent upscale model with dependency chaining
- Key Parameters: model_name, optional dependencies
- Unique to ComfyUI: Yes

---

### UTILITY NODES

**49. ImageToCPU**
- Category: `utility`
- Function: Moves image tensor from GPU to CPU memory
- Key Parameters: image
- Unique to ComfyUI: Yes - device management

**50. LinearOverlapLatentTransition**
- Category: `latent`
- Function: Linearly blends overlapping latent regions for seamless transitions
- Key Parameters: latents1, latents2, transition_frames, blend_factor
- Unique to ComfyUI: Yes - latent interpolation

---

### EXPERIMENTAL TRICKS NODES (tricks/ subdirectory)

**51. ModifyLTXModel**
- Category: `Lightricks`
- Function: General model modification hook system
- Unique to ComfyUI: Yes

**52. AddLatentGuide**
- Category: `Lightricks`
- Function: Alternative latent guide implementation
- Unique to ComfyUI: Yes

**53. LTXForwardModelSamplingPred**
- Category: `Lightricks`
- Function: Forward pass model sampling prediction
- Unique to ComfyUI: Yes (experimental)

**54. LTXReverseModelSamplingPred**
- Category: `Lightricks`
- Function: Reverse pass model sampling prediction
- Unique to ComfyUI: Yes (experimental)

**55. LTXRFForwardODESampler**
- Category: `Lightricks`
- Function: Rectified Flow ODE sampler (forward direction)
- Unique to ComfyUI: Yes (experimental)

**56. LTXRFReverseODESampler**
- Category: `Lightricks`
- Function: Rectified Flow ODE sampler (reverse direction)
- Unique to ComfyUI: Yes (experimental)

**57. LTXAttentionBank**
- Category: `Lightricks`
- Function: Caches attention patterns for reuse
- Unique to ComfyUI: Yes (experimental)

**58. LTXPrepareAttnInjections**
- Category: `Lightricks`
- Function: Prepares attention injection configurations
- Unique to ComfyUI: Yes (experimental)

**59. LTXAttentioOverride**
- Category: `Lightricks`
- Function: Overrides attention blocks with custom implementations
- Unique to ComfyUI: Yes (experimental)

**60. LTXPerturbedAttention**
- Category: `Lightricks`
- Function: Applies perturbations to attention for exploration
- Unique to ComfyUI: Yes (experimental)

**61. LTXAttnOverride**
- Category: `Lightricks`
- Function: Alternative attention override mechanism
- Unique to ComfyUI: Yes (experimental)

**62. LTXFlowEditCFGGuider**
- Category: `Lightricks`
- Function: CFG guider specialized for flow-based editing
- Unique to ComfyUI: Yes (experimental)

**63. LTXFlowEditSampler**
- Category: `Lightricks`
- Function: Sampler optimized for flow-based video editing
- Unique to ComfyUI: Yes (experimental)

**64. LTXFetaEnhance**
- Category: `Lightricks`
- Function: FETA (Frame-Enhanced Temporal Attention) enhancement
- Unique to ComfyUI: Yes (experimental)

---

### SUMMARY STATISTICS

- **Total Node Types**: 64 nodes
- **Core Sampling Nodes**: 6
- **Guidance Nodes**: 6
- **Guide/Keyframe Nodes**: 5
- **Latent Manipulation**: 4
- **Latent Normalization**: 4
- **Conditioning**: 4
- **VAE/Decoding**: 3
- **Text Encoding**: 5
- **IC-LoRA**: 2
- **Motion Tracking**: 2
- **Quantization/Optimization**: 2
- **Model Loading**: 3
- **Utility**: 2
- **Experimental Tricks**: 14

**Key Unique Features** (not in native LTX-2):
- Temporal and spatial tiling for extended/high-res generation
- Multimodal (audio+video) guidance
- IC-LoRA integration
- Prompt enhancement with local LLMs
- Motion tracking with spline editors
- Conditioning caching system
- Q8 quantization support
- Flexible latent manipulation tools
- Advanced STG guidance variants
- Per-step normalization and patchering

All nodes are registered through the `nodes_registry.py` system and use the `@comfy_node` decorator for consistency.