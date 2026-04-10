Last updated: 2026-04-10

# LTX 2.3 Gaps Analysis: ComfyUI-LTXVideo vs LTX-2 Native vs LTX-Desktop

Cross-reference of capabilities across all three codebases. Identifies what
exists where, what's missing, and what could be brought into ComfyUI workflows.

## Sources Analyzed

- **ComfyUI-LTXVideo**: 64 nodes, community-built ComfyUI integration
- **LTX-2 (native)**: Canonical PyTorch reference, 14 pipelines, Lightricks official
- **LTX-Desktop**: Electron app with Python backend, Lightricks official product

---

## Capability Matrix

### Generation Pipelines

| Capability | ComfyUI-LTXVideo | LTX-2 Native | LTX-Desktop | Notes |
|---|---|---|---|---|
| Text-to-Video (T2V) | LTXVBaseSampler | TI2VidTwoStagesPipeline | Fast pipeline | All have it |
| Image-to-Video (I2V) | LTXVBaseSampler + image guide | TI2VidTwoStagesPipeline + image cond | Fast pipeline + image | All have it |
| Audio-to-Video (A2V) | Via LTXVConcatAVLatent (manual) | A2VidPipelineTwoStage (dedicated) | Distilled A2V pipeline (2-stage) | **ComfyUI has no dedicated A2V pipeline** |
| Two-stage generation | Manual chain (sampler → upscale → refine) | TI2VidTwoStagesPipeline (built-in) | A2V pipeline (automatic) | **ComfyUI requires manual setup** |
| Distilled fast mode | Via distilled sigmas manually | DistilledPipeline (dedicated) | Fast pipeline (default) | ComfyUI works but no dedicated node |
| Long video (looping) | LTXVLoopingSampler (ComfyUI invention) | NOT IN NATIVE | NOT IN DESKTOP | **Unique to ComfyUI** |
| Temporal editing (retake) | Not available | RetakePipeline | Retake handler | **MISSING from ComfyUI** |
| Keyframe interpolation | Via LTXVAddGuideAdvanced | KeyframeInterpolationPipeline | Not in Desktop | Native has dedicated pipeline |
| IC-LoRA | LTXICLoRALoaderModelOnly + guides | ICLoraPipeline (full) | IC-LoRA handler | All have it, native most complete |
| Image generation | Not available | Not available | ZIT pipeline | Desktop only (for reference images) |

### Audio Handling

| Capability | ComfyUI-LTXVideo | LTX-2 Native | LTX-Desktop | Notes |
|---|---|---|---|---|
| Audio VAE encode | LTXVAudioVAEEncode (core ComfyUI) | AudioConditioner block | Audio VAE in A2V pipeline | All have it |
| Audio-video concat | LTXVConcatAVLatent | Built into A2VidPipeline | Built into A2V pipeline | ComfyUI is manual |
| Frozen audio during video denoising | Manual via noise mask=0 | Explicit frozen=True flag | Explicit frozen=True | **ComfyUI relies on mask trick, others have native flag** |
| Audio regeneration control | Not available | regenerate_audio flag in Retake | regenerate_audio in Retake | **MISSING from ComfyUI** |
| Vocal separation | Via MelBandRoFormer (external) | Not in repo | Not in Desktop | Our addition via AudioLoopHelper |
| Audio trim/offset | Via TrimAudioDuration (core) | decode_audio_from_file(start_time) | audio_start_time param | All have it |

### Guidance & Sampling

| Capability | ComfyUI-LTXVideo | LTX-2 Native | LTX-Desktop | Notes |
|---|---|---|---|---|
| CFG guidance | CFGGuider | CFGGuider, CFGStarRescalingGuider | GuidedDenoiser (retake only) | Native has rescaling variant |
| STG (Spatio-Temporal Guidance) | LTXVApplySTG, STGGuiderNode | STGGuider | Not used (distilled) | ComfyUI + native |
| NAG (Normalized Attention Guidance) | Via KJNodes LTX2_NAG | Not in native | Not in Desktop | **KJNodes addition** |
| APG (Adaptive Projected Guidance) | Not available | LtxAPGGuider, LegacyStatefulAPGGuider | Not in Desktop | **MISSING from ComfyUI** |
| Multimodal guidance (video+audio CFG) | MultimodalGuiderNode | MultiModalGuiderFactory | Not used (distilled) | Both have it |
| Sigma-dependent guidance | Not available | MultiModalGuiderFactory per-step | Not available | **MISSING: different guidance params at different noise levels** |
| Gradient estimation denoising | Not available | gradient_estimating_euler_denoising_loop | Not available | **MISSING: fewer steps with comparable quality** |
| Rescaling (anti-oversaturation) | Via STGGuiderAdvanced rescale_factor | Explicit rescale_scale param | Not used | Both have it |
| Perturbation control (skip specific attention) | Via STG block_indices | Detailed SKIP_* perturbation types | Not used | Native more granular |

### Conditioning & Prompting

| Capability | ComfyUI-LTXVideo | LTX-2 Native | LTX-Desktop | Notes |
|---|---|---|---|---|
| Gemma 3 text encoding | LTXVGemmaCLIPModelLoader | Multi-layer feature extractor | Local or API encoding | All have it |
| Prompt enhancement (local LLM) | LTXVPromptEnhancer (Llama-3.2 + Florence-2) | enhance_t2v() / enhance_i2v() via Gemma | API-only (server-side) | ComfyUI has local, Desktop uses API |
| Prompt enhancement (Gemma-native) | LTXVGemmaEnhancePrompt | Built into text encoder | Via API | |
| System prompts (i2v/t2v) | In docs (manual use) | Built into Gemma encoder pipeline | Built into API | **ComfyUI: manual. Others: automatic** |
| Per-segment prompts | MultiPromptProvider | Not in native | Not in Desktop | **Unique to ComfyUI** |
| Gap prompt suggestion (AI) | Not available | Not in native | Gemini 2.0 Flash integration | **Desktop uses Gemini to suggest prompts for gaps between clips** |
| Camera motion keywords | In docs (manual append) | Not in native | Config-driven auto-append | **Desktop auto-appends to prompt** |
| Conditioning caching | LTXVSaveConditioning/LoadConditioning | Not in native | Not in Desktop | **Unique to ComfyUI** |
| Modality-specific embeddings | Not explicit | Separate video (4096-dim) + audio (2048-dim) contexts | Via text encoder | Native most explicit |

### Image/Video Conditioning

| Capability | ComfyUI-LTXVideo | LTX-2 Native | LTX-Desktop | Notes |
|---|---|---|---|---|
| Image guide at frame index | LTXVAddGuideAdvanced | VideoConditionByLatentIndex | images[{path, frame_idx, strength}] | All have it |
| Additive keyframe guide | LTXVAddGuideAdvanced | VideoConditionByKeyframeIndex | Not available | ComfyUI + native |
| Reference video conditioning | Via IC-LoRA | VideoConditionByReferenceLatent | Via IC-LoRA | Native has dedicated type |
| Attention strength per guide | LTXVAddGuideAdvancedAttention | ConditioningItemAttentionStrengthWrapper | Not available | ComfyUI + native |
| Spatial attention mask per guide | LTXVAddGuideAdvancedAttention | Built into conditioning system | Not available | ComfyUI + native |
| CRF preprocessing for guides | LTXVAddGuideAdvanced (crf param) | Not in native | Not in Desktop | **Unique to ComfyUI** |
| Motion tracking / sparse tracks | LTXVSparseTrackEditor | Not in native | Not in Desktop | **Unique to ComfyUI** |

### Latent Operations

| Capability | ComfyUI-LTXVideo | LTX-2 Native | LTX-Desktop | Notes |
|---|---|---|---|---|
| Latent frame selection | LTXVSelectLatents | Via tensor slicing | Not exposed | ComfyUI has node |
| Latent concatenation | LTXVAddLatents | Via tensor ops | Not exposed | ComfyUI has node |
| Latent upscale (2x) | LTXVLatentUpsampler (uses official model) | Spatial upsampler in 2-stage | A2V stage 2 auto-upscale | All use same model |
| Latent dilation | LTXVDilateLatent | Not in native | Not in Desktop | **Unique to ComfyUI** |
| AdaIn normalization | LTXVAdainLatent, LTXVPerStepAdainPatcher | Not in native | Not in Desktop | **Unique to ComfyUI** |
| Statistical normalization | LTXVStatNormLatent, LTXVPerStepStatNormPatcher | Not in native | Not in Desktop | **Unique to ComfyUI** |
| Noise mask control | LTXVSetVideoLatentNoiseMasks | Via TemporalRegionMask | Via TemporalRegionMask | Different mechanisms |

### Memory & Optimization

| Capability | ComfyUI-LTXVideo | LTX-2 Native | LTX-Desktop | Notes |
|---|---|---|---|---|
| Q8 quantization | LTXQ8Patch, LTXVQ8Patch | FP8 cast + FP8 Scaled MM | Auto-detect FP8 | Different approaches |
| Tiled VAE decode | LTXVTiledVAEDecode | tiled_decode() in VAE | TilingConfig | All have it |
| Tiled sampling | LTXVTiledSampler, LTXVLoopingSampler | VAE tiling only (not sampling) | Not available | **ComfyUI tiles during SAMPLING, not just decode** |
| Low-VRAM loaders | LowVRAMCheckpointLoader | Not in native | Not in Desktop | **Unique to ComfyUI** |
| Decoder noise injection | DecoderNoise | Not in native | Not in Desktop | **Unique to ComfyUI** |
| Safetensors mmap fix | Not needed (ComfyUI handles) | Not needed | Critical patch (3 fixes) | Desktop has specific memory patches |

### Experimental / Advanced

| Capability | ComfyUI-LTXVideo | LTX-2 Native | LTX-Desktop | Notes |
|---|---|---|---|---|
| Flow editing | LTXFlowEditCFGGuider, LTXFlowEditSampler | Not in native | Not in Desktop | **Unique to ComfyUI** |
| FETA (Frame-Enhanced Temporal Attention) | LTXFetaEnhance | Not in native | Not in Desktop | **Unique to ComfyUI** |
| Attention banking/injection | LTXAttentionBank, LTXPrepareAttnInjections | Not in native | Not in Desktop | **Unique to ComfyUI** |
| Perturbed attention | LTXPerturbedAttention | Not in native | Not in Desktop | **Unique to ComfyUI** |
| Forward/Reverse ODE samplers | LTXRFForwardODESampler, LTXRFReverseODESampler | Not in native | Not in Desktop | **Unique to ComfyUI** |
| Vanish nodes | vanish_nodes.py | Not in native | Not in Desktop | **Unique to ComfyUI** |

---

## Top Gaps: What's MISSING from ComfyUI

### HIGH PRIORITY (would directly improve our music video workflow)

1. **Dedicated A2V Pipeline Node**
   - Native and Desktop have purpose-built audio-to-video pipelines with frozen audio
   - ComfyUI requires manual LTXVConcatAVLatent + noise mask tricks
   - A dedicated node would simplify workflow and ensure correct audio handling

2. **Two-Stage Generation (Half-res → Upscale → Refine)**
   - Native's TI2VidTwoStagesPipeline generates at half resolution then upscales
   - Desktop's A2V pipeline does this automatically (Stage 1: H/2, W/2 → Stage 2: full)
   - Prevents mode collapse, better quality, more stable
   - ComfyUI can approximate this manually but it's fragile

3. **Temporal Region Editing (Retake)**
   - Regenerate a specific time window while keeping everything else frozen
   - Native has RetakePipeline, Desktop has Retake handler
   - Could fix bad sections without re-rendering entire video
   - **Huge time saver for our loop workflow** (fix one bad iteration)

4. **Gradient Estimation Denoising**
   - Native's velocity-based correction enables fewer steps (20-30 vs 40) with same quality
   - Would speed up each loop iteration by ~30-50%
   - gamma parameter controls correction strength

### MEDIUM PRIORITY (would improve quality or workflow)

5. **APG (Adaptive Projected Guidance)**
   - Decomposes guidance into parallel & orthogonal components
   - Better text adherence without oversaturation
   - Native has it, ComfyUI doesn't

6. **Sigma-Dependent Guidance Parameters**
   - Different CFG/STG values at different noise levels
   - MultiModalGuiderFactory in native allows per-step variation
   - Would improve quality at no speed cost

7. **Gap Prompt Suggestion (Gemini/LLM)**
   - Desktop uses Gemini 2.0 Flash to suggest prompts for gaps between clips
   - Could be adapted for our timestamp schedule (suggest prompts per section)
   - We partially have this with our `/prompt-schedule` skill

8. **System Prompts Auto-Apply**
   - Native builds i2v/t2v system prompts into the Gemma encoding pipeline
   - Desktop auto-appends camera motion keywords to prompts
   - ComfyUI requires manual prompt engineering (our docs help but it's manual)

### LOW PRIORITY (nice to have)

9. **CFGStar Rescaling Guider** -- prevents offset in unconditioned samples
10. **Per-modality text embeddings** -- separate 4096-dim video + 2048-dim audio contexts from same prompt
11. **LoRA metadata auto-detection** -- reads training resolution from checkpoint
12. **Reference video conditioning type** -- dedicated conditioning type for video references (not just IC-LoRA)

---

## Top Strengths: What ComfyUI has that OTHERS DON'T

1. **LTXVLoopingSampler** -- Long video via temporal chunking (neither native nor Desktop has this)
2. **Per-segment prompts** -- MultiPromptProvider (unique to ComfyUI)
3. **Latent-space tiling during sampling** -- LTXVTiledSampler (others only tile VAE decode)
4. **AdaIn / StatNorm per-step** -- Prevents drift in long sequences
5. **Motion tracking / sparse tracks** -- LTXVSparseTrackEditor
6. **Flow editing** -- Forward/reverse ODE samplers, flow edit guiders
7. **Attention banking/injection** -- Cache and reuse attention patterns
8. **FETA** -- Frame-enhanced temporal attention
9. **Conditioning caching** -- Save/load CONDITIONING to disk
10. **CRF preprocessing for guides** -- Quality-aware guide preprocessing
11. **NAG (via KJNodes)** -- Normalized Attention Guidance (not in native or Desktop)
12. **Decoder noise injection** -- DecoderNoise node for VAE quality

---

## Architectural Differences

### Audio-to-Video Philosophy

| Approach | Implementation | Who |
|---|---|---|
| **Frozen audio latent** | Audio encoded, noise_mask=0, never denoised | ComfyUI (manual), Native (flag), Desktop (automatic) |
| **Audio as conditioning context** | Audio cross-attends to video during denoising | All three |
| **Audio regeneration** | Can regenerate audio in a time window | Native (Retake), Desktop (Retake). NOT ComfyUI. |

### Sampling Philosophy

| Approach | Implementation | Who |
|---|---|---|
| **Single-pass full video** | Generate all frames at once | Native (default) |
| **Autoregressive temporal chunking** | Generate chunks, overlap-blend | ComfyUI (LTXVLoopingSampler) |
| **Two-stage (half-res → upscale)** | Stage 1 cheap, Stage 2 refine | Native, Desktop |
| **Distilled (fixed sigmas)** | No guidance needed, fast | All three |
| **Full CFG + scheduler** | 40 steps with guidance | Native (Retake), not others |

### Text Encoding Philosophy

| Approach | Implementation | Who |
|---|---|---|
| **Local Gemma 3** | On-device encoding | ComfyUI, Desktop (fallback) |
| **API encoding + enhancement** | Server-side with Gemini | Desktop (primary) |
| **System prompts built-in** | Auto-prepended to encoding | Native |
| **Manual prompt engineering** | User writes i2v-style prompts | ComfyUI (with our docs) |

---

## Recommendations for Our Workflow

### Immediate (could implement now)

1. **Use LTXVLoopingSampler workflow** (already planned) -- eliminates VAE round-trip
2. **Auto-apply system prompts** -- build into ScheduleToMultiPrompt or a wrapper node
   that prepends the i2v system prompt context to each prompt before encoding
3. **Auto-append camera motion keywords** -- add camera_motion input to
   TimestampPromptSchedule that appends keywords per entry

### Next Phase

4. **Build a RetakeSection node** -- wraps the temporal region mask concept for
   fixing specific loop iterations without re-rendering everything
5. **Implement two-stage generation inside the loop** -- generate at half-res in
   LTXVLoopingSampler, then upscale+refine per chunk (like Desktop A2V)
6. **Gradient estimation denoising** -- port from native to reduce steps per iteration

### Future

7. **APG guider** -- better text adherence
8. **LLM-based prompt suggestion** -- use local Gemma or API to suggest prompts per section
9. **Per-modality guidance tuning** -- separate video/audio CFG during generation

---

## File References

### LTX-Desktop
- `coderef/LTX-Desktop/backend/services/a2v_pipeline/` -- A2V pipeline
- `coderef/LTX-Desktop/backend/services/retake_pipeline/` -- Retake
- `coderef/LTX-Desktop/backend/services/fast_video_pipeline/` -- Fast T2V/I2V
- `coderef/LTX-Desktop/backend/handlers/suggest_gap_prompt_handler.py` -- Gap prompts

### LTX-2 Native
- `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/` -- All pipelines
- `coderef/LTX-2/packages/ltx-core/src/ltx_core/components/guiders.py` -- Guidance
- `coderef/LTX-2/packages/ltx-core/src/ltx_core/conditioning/` -- Conditioning
- `coderef/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/` -- Gemma 3

### ComfyUI-LTXVideo
- `coderef/ComfyUI-LTXVideo/looping_sampler.py` -- Long video
- `coderef/ComfyUI-LTXVideo/easy_samplers.py` -- Base/Extend/InContext samplers
- `coderef/ComfyUI-LTXVideo/latents.py` -- Latent ops
- `coderef/ComfyUI-LTXVideo/guide.py` -- Guide/keyframe nodes
- `coderef/ComfyUI-LTXVideo/tricks/` -- Experimental nodes
