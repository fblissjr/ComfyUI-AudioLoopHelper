Perfect! Now I have comprehensive knowledge of the system. Let me provide a thorough written analysis:

## COMPREHENSIVE ANALYSIS: LTX-Desktop Backend Architecture & Capabilities

Based on thorough code review of `LTX-Desktop/backend/`, here is the complete breakdown of all generation capabilities, pipelines, and techniques in the official Lightricks LTX Desktop application.

---

## I. ARCHITECTURE OVERVIEW

**Request Flow:**
```
HTTP Request → _routes/* (thin endpoints)
           → AppHandler (composition root)
           → handlers/* (domain logic + state)
           → services/* (side effects boundary)
           → state/* (AppState mutations under RLock)
```

**Key Principles:**
- Single AppHandler owns all sub-handlers, AppState, and a shared RLock
- Thread-safe state mutations only via with_state_lock decorator
- Services are the boundary for GPU/IO/network side effects
- Tests inject fake services via ServiceBundle
- Exception handling: boundary-owned traceback logging (app_factory.py owns it)

---

## II. GENERATION CAPABILITIES (ALL PIPELINES)

### A. FAST VIDEO PIPELINE (Text-to-Video & Image-to-Video)

**File:** `/backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`

**What it does:**
- Single-stage distilled LTX 2.3 "fast" model
- Generates video from text prompt (T2V) or text + image (I2V)
- Fast inference with fixed distilled sigmas (no scheduler needed)

**Key Parameters:**
```python
generate(
    prompt: str,                          # Required text description
    seed: int,                            # Random seed (reproducibility)
    height: int, width: int,              # Resolution (rounded to 64x64)
    num_frames: int,                      # Total frames to generate
    frame_rate: float,                    # FPS (8, 24, 25 typical)
    images: list[ImageConditioningInput], # Optional image guides [{"path": str, "frame_idx": 0, "strength": 1.0}]
    output_path: str                      # Where to save MP4
)
```

**Audio Conditioning:** NOT SUPPORTED in fast pipeline (audio-only is in A2V)

**Sampling Strategy:**
- Uses `DistilledPipeline` from ltx_core
- Fixed sigma values from `DISTILLED_SIGMA_VALUES` constant
- Uses `SimpleDenoiser` (no classifier-free guidance in distilled mode)
- Single-pass generation (no upsampling)

**Unique Techniques:**
- Quantization support: Auto-detects FP8 capability via `device_supports_fp8()`
- Streaming prefetch_count=2 for memory efficiency
- Lazy video decoding (Iterator[Tensor]) for large outputs
- Warmup phase: generates 9-frame test at 256x384@8fps then deletes output

**Long Video Handling:** Not explicitly supported. Single generation up to configured frame limits.

**Video Output Encoding:**
- Uses `encode_video_output()` which calls ltx_pipelines media_io
- Calculates video chunks dynamically based on frame count + tiling config
- Returns MP4 with muxed audio (if any)

---

### B. A2V PIPELINE (Audio-to-Video / Audio-Conditioned)

**File:** `/backend/services/a2v_pipeline/distilled_a2v_pipeline.py` + `ltx_a2v_pipeline.py`

**What it does:**
- **TWO-STAGE distilled pipeline** specifically designed for audio-conditioned video
- Stage 1: Half-resolution generation with frozen audio encoding
- Stage 2: 2x spatial upsampling + full-resolution refinement
- Critical for lip-sync and synchronization with audio tracks

**Key Parameters:**
```python
generate(
    prompt: str,                          # Text description
    negative_prompt: str,                 # Negative guidance (for distilled mode, may be ignored)
    seed: int,                            # Random seed
    height: int, width: int,              # Resolution (auto-halved for stage 1)
    num_frames: int,                      # Total frames
    frame_rate: float,                    # FPS
    num_inference_steps: int,             # Total steps (distilled: ~8 stage1 + ~3 stage2)
    images: list[ImageConditioningInput], # Optional image guides
    audio_path: str,                      # REQUIRED: path to audio file
    audio_start_time: float,              # Trim audio from this timestamp
    audio_max_duration: float | None,     # Limit audio length (None = full duration)
    output_path: str                      # MP4 output path
)
```

**Audio Conditioning (THE CORE FEATURE):**
1. **Audio Encoding:** 
   - Uses `AudioConditioner` block with audio VAE
   - Decodes audio file via `decode_audio_from_file(audio_path, device, audio_start_time, audio_max_duration)`
   - Audio encoding shape: AudioLatentShape(batch=1, duration=num_frames/fps, channels=8, mel_bins=16)
   - Pads or crops encoded audio to match target frame count

2. **Audio as Frozen Context:**
   - Stage 1: `ModalitySpec(audio=..., frozen=True, noise_scale=0.0, initial_latent=encoded_audio_latent)`
   - Stage 2: Same frozen audio applied to upsampled video
   - **Frozen=True** means audio is NOT regenerated, only used for conditioning

3. **Video Denoising with Audio:**
   - `SimpleDenoiser(video_context, audio_context)` runs diffusion with both modalities
   - Audio context guides video generation via cross-attention
   - No audio regeneration (preserves original audio fidelity)

**Sampling Strategy:**
- Stage 1 sigmas: `DISTILLED_SIGMA_VALUES` (predefined distilled steps)
- Stage 2 sigmas: `STAGE_2_DISTILLED_SIGMA_VALUES` (refinement steps, typically fewer)
- Both stages use `GaussianNoiser` with same seed-derived generator
- No classifier-free guidance (distilled mode only)

**Unique Techniques:**
- **Two-stage approach prevents mode collapse:** Generate at half-res first (cheaper, more stable), then upscale and refine
- **Audio latent padding:** If audio shorter than video, zero-pad. If longer, crop to match.
- **Image conditioning per stage:** Recompute image conditionings at stage 1 (half-res) and stage 2 (full-res)
- Returns **original audio** (not VAE-decoded) for fidelity: `trimmed_waveform = decoded_audio.waveform.squeeze(0)[..., :max_samples]`

**Long Video Handling:**
- Tiling configs default to `TilingConfig.default()` (splits large frames into tiles to avoid OOM)
- Both video encoder and upsampler support tiling

---

### C. RETAKE PIPELINE (In-Painting / Region Re-generation)

**File:** `/backend/services/retake_pipeline/ltx_retake_pipeline.py`

**What it does:**
- "Retake" = selective re-generation of a time-bounded region in an existing video
- Can regenerate video ONLY, audio ONLY, or BOTH in the specified time window
- Keeps surrounding frames frozen as guidance context

**Key Parameters:**
```python
generate(
    video_path: str,              # Source video to retake from
    prompt: str,                  # How to re-generate the region
    start_time: float,            # Region start in seconds
    end_time: float,              # Region end in seconds
    seed: int,                    # Random seed
    output_path: str,             # Output video path
    negative_prompt: str = "",    # Negative guidance
    num_inference_steps: int = 40, # For non-distilled mode
    video_guider_params: MultiModalGuiderParams | None = None,  # CFG for video
    audio_guider_params: MultiModalGuiderParams | None = None,  # CFG for audio
    regenerate_video: bool = True,   # Whether to regenerate video in region
    regenerate_audio: bool = True,   # Whether to regenerate audio in region
    enhance_prompt: bool = False,    # Enable prompt enhancement via API
    distilled: bool = True           # Use distilled sigmas (fast) or full scheduler
)
```

**How Retake Works:**
1. **Load entire video:** Encode both video and audio via VAE from source file
2. **Create region mask:** `TemporalRegionMask(start_time, end_time, fps)` specifies which frames to regenerate
3. **Build ModalitySpec:** 
   - If regenerate_video=True: add TemporalRegionMask to video conditionings
   - If regenerate_video=False: frozen=True on video (use as fixed context)
   - Same for audio
4. **Denoiser selection:**
   - If distilled=True: `SimpleDenoiser(v_context, a_context)` with fixed sigmas
   - If distilled=False: `GuidedDenoiser` with CFG guiders + LTX2Scheduler (40 steps typical)
5. **Run single diffusion pass:** Entire video is input; region is regenerated, rest frozen
6. **Output:** Only the specified region is regenerated; context outside region is preserved from source

**Audio Handling in Retake:**
- Can preserve original audio while regenerating video (or vice versa)
- Audio regeneration uses same TemporalRegionMask mechanism
- Output audio duration trimmed to match video frames: `max_samples = round(num_frames / fps * sample_rate)`

**Sampling Strategy:**
- **Distilled mode (default):** `DISTILLED_SIGMA_VALUES` (fast, 8-11 steps)
- **Full mode:** `LTX2Scheduler().execute(steps=40)` produces 40-step sigma schedule with full CFG
- Both use `GaussianNoiser(generator)` seeded for reproducibility

**Unique Techniques:**
- **Lazy video decoding:** Returns Iterator[Tensor] for memory efficiency on large videos
- **Tiled encoding:** Encodes source video with smaller tiles (256px spatial, 24 frames temporal) to avoid OOM
- **Region masking:** Uses ltx_core TemporalRegionMask conditioning type (not tied to spatial masks)
- **@torch.no_grad() instead of @torch.inference_mode():** Accommodates custom autograd in transformer

**Use Cases:**
- Fix a bad lip-sync section while keeping the rest
- Change dialogue emotion in a specific time window
- Re-generate background in one region while keeping foreground
- Regenerate audio only (change narration) while video stays the same

---

### D. IC-LoRA PIPELINE (Image Conditioning via Low-Rank Adaptation)

**File:** `/backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`

**What it does:**
- Uses a LoRA ("Low-Rank Adaptation") fine-tune to condition video generation on visual features from one or more reference images
- Can generate video conditioned on both text AND image features
- Particularly useful for style consistency across long sequences

**Key Parameters:**
```python
generate(
    prompt: str,                          # Text description
    seed: int,                            # Random seed
    height: int, width: int,              # Resolution
    num_frames: int,                      # Frame count
    frame_rate: float,                    # FPS
    images: list[ImageConditioningInput], # Standard image guides [{"path", "frame_idx", "strength"}]
    video_conditioning: list[tuple[str, float]], # IC-LoRA specific: [(image_path, "canny"|"depth" feature), ...]
    output_path: str                      # Output MP4
)
```

**IC-LoRA Architecture:**
1. **Loads LoRA weights** from checkpoint: `ICLoraPipeline(loras=[lora_entry])`
2. **LoRA entry:** `LoraPathStrengthAndSDOps(path=lora_path, strength=1.0, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)`
3. **Passes video_conditioning to pipeline:** List of (image_path, feature_type) tuples
4. **Feature extraction options:**
   - Canny edge detection for structural guidance
   - Depth maps for spatial composition
   - Raw images for direct visual conditioning

**Sampling Strategy:**
- Single-stage distilled pipeline (like Fast)
- Uses `DISTILLED_SIGMA_VALUES` fixed sigmas
- LoRA weights active throughout diffusion (no layer-wise dropout)

**Unique Techniques:**
- **LoRA integration:** Wraps `ICLoraPipeline` from ltx_pipelines, which loads LoRA weights with strength=1.0
- **Reference downscale factor:** Reads from LoRA metadata (`_read_lora_reference_downscale_factor`) to determine preprocessing
- **Streaming prefetch:** prefetch_count=2 for efficient memory usage

---

### E. IMAGE GENERATION PIPELINE (Image-Only via Z-Image-Turbo)

**File:** `/backend/services/image_generation_pipeline/zit_image_generation_pipeline.py`

**What it does:**
- Generates static images using Z-Image-Turbo (ZIT) model
- NOT for video, but used for initial frames or reference images
- Part of the larger ecosystem (e.g., generate reference image → use as video guide)

**Key Parameters:**
```python
generate(
    prompt: str,                  # Text description
    height: int, width: int,      # Image resolution
    guidance_scale: float,        # Classifier-free guidance (Note: ZIT ignores this)
    num_inference_steps: int,     # Denoising steps
    seed: int                     # Random seed
) -> ImagePipelineOutputLike    # Returns PIL images
```

**Sampling Strategy:**
- Uses Hugging Face diffusers `ZImagePipeline`
- guidance_scale parameter accepted but **explicitly ignored** (guidance_scale=0.0 hardcoded)
- Generates via standard diffusers scheduler

**Unique Techniques:**
- **CPU offload support:** Enables `model_cpu_offload()` for CUDA/MPS to save VRAM
- **BFloat16 dtype:** Pipeline initialized with torch.bfloat16 for efficiency
- **Output validation:** Ensures images are PIL.Image instances before returning

---

## III. SHARED UTILITIES & TECHNIQUES

### A. Text Encoding (Gemma 3 Integration)

**File:** `/backend/services/text_encoder/ltx_text_encoder.py`

**How Text Encoding Works:**

1. **Local Encoding (Default):**
   - Uses Gemma 3 text encoder (bundled with LTX 2.3)
   - Lives in PromptEncoder block within each pipeline
   - Produces two outputs: video_context (4096-dim) and audio_context (remaining dims)
   - Gemma 3 outputs embeddings that are split: first 4096 dims = video, rest = audio

2. **API Encoding (Remote):**
   - When API key configured: offloads text encoding to LTX API service
   - Endpoint: `{ltx_api_base_url}/v1/prompt-embedding`
   - Sends: `{"prompt": str, "model_id": str, "enhance_prompt": bool}`
   - Returns pickled conditioning (video_context + audio_context)
   - `enhance_prompt=True` triggers prompt enhancement on server

3. **Monkey-Patching Strategy:**
   - `install_patches(state_getter)` installs three patches:
     a. **PromptEncoder.__init__** patch: Allows None gemma_root for API-only mode
     b. **PromptEncoder.__call__** patch: Intercepts encoding calls, returns API embeddings if available
     c. **cleanup_memory** patch: Moves cached encoder to CPU before memory cleanup

**Prompt Enhancement:**
- Enabled via `enhance_prompt_enabled_i2v` / `enhance_prompt_enabled_t2v` settings
- Only used when API encoding is active
- Server-side Gemini-based enhancement (details in gap prompt logic)

---

### B. Patches (Memory & Stability Fixes)

**1. safetensors_metadata_fix.py:**
- **Problem:** `safe_open` reserves mmap commit charge equal to full file size (22GB for checkpoint = 22GB reserved)
- **Solution:** Direct file header reading without mmap:
  - Read first 8 bytes (little-endian header size)
  - Read header JSON
  - Parse metadata without torch.UntypedStorage
- **Applies to:**
  - SafetensorsModelStateDictLoader.metadata()
  - ltx_pipelines.ic_lora._read_lora_reference_downscale_factor()
  - ltx_pipelines.utils.constants.detect_params()
  - LTXTextEncoder.get_model_id_from_checkpoint()

**2. record_stream_fix.py:**
- **Problem:** `tensor.record_stream()` corrupts CUDA allocator on RTX 5090 + CUDA 12.8
- **Solution:** Replace with explicit Python reference holding + CUDA events:
  - Store tensor data in gpu_refs dict by layer index
  - Record CUDA events at layer boundaries
  - Drain completed refs when events complete
  - Prevents allocator reuse while compute still reading

**3. safetensors_loader_fix.py:**
- **Problem:** Windows mmap page faults cause STATUS_ACCESS_VIOLATION under memory pressure
- **Solution:** Custom mmap-based loader with Python error handling:
  - Load safetensors header + tensor metadata manually
  - Use Python mmap.mmap() instead of torch.UntypedStorage.from_file()
  - torch.frombuffer() to create read-only tensor views
  - Store mmap reference in tensor storage to keep mapping alive

---

### C. Video Processing (cv2-based)

**File:** `/backend/services/video_processor/video_processor_impl.py`

**Capabilities:**
1. **open_video(path)** → cv2.VideoCapture
2. **get_video_info(cap)** → {fps, frame_count, width, height}
3. **read_frame(cap, frame_idx)** → numpy array (BGR)
4. **apply_canny(frame)** → edge map (HW, repeated to HWC3 for IC-LoRA)
5. **apply_depth(frame, depth_pipeline)** → depth map (delegates to depth processor)
6. **apply_pose(frame, pose_pipeline)** → pose skeleton (delegates to pose processor)
7. **encode_frame_jpeg(frame, quality)** → JPEG bytes (for API uploads)
8. **create_writer(path, fourcc, fps, size)** → cv2.VideoWriter
9. **release(cap_or_writer)** → cleanup with exception tolerance

**IC-LoRA Conditioning Preprocessing:**
- **Canny:** 64-pixel padding to nearest multiple, edge detection, then unpad + convert to 3-channel
- **Depth:** Delegates to depth processor pipeline (loaded separately)
- Used to generate video_conditioning features for IC-LoRA mode

---

## IV. HANDLER ORCHESTRATION

### A. VideoGenerationHandler

**Main Entry Point:** `handlers/video_generation_handler.py`

**Routes:**
```
POST /generate-video → VideoGenerationHandler.generate()
```

**Flow:**
1. **Check if API-forced:** `should_video_generate_with_ltx_api(force_api_generations, settings)`
2. **If forced API:** Delegate to `_generate_forced_api()` (uploads to LTX cloud)
3. **If local:** Check audio file → choose A2V or Fast pipeline
4. **Fast pipeline (T2V/I2V):**
   - Load fast model via `PipelinesHandler.load_gpu_pipeline("fast")`
   - Prepare text encoding (API or local)
   - Optionally prepare image guide
   - Run inference via `pipeline.generate()`
5. **A2V pipeline (audio-to-video):**
   - Load A2V model via `PipelinesHandler.load_a2v_pipeline()`
   - Validate audio file
   - Prepare text encoding
   - Optionally prepare image guide
   - Run inference with audio_path, audio_start_time=0.0, audio_max_duration=None

**Resolution Support (Local):**
- 16:9 aspect: 960x544 (540p), 1280x704 (720p), 1920x1088 (1080p)
- 9:16 aspect: swapped dimensions
- Rounds to nearest 64x64 before inference

**Frame Count Computation:**
```python
num_frames = ((duration * fps) // 8) * 8 + 1
num_frames = max(num_frames, 9)  # Minimum 9 frames
```

**Camera Motion Integration:**
- Prompts augmented with camera motion keywords from config
- `config.camera_motion_prompts[camera_motion]` appended to user prompt
- Supports: none, dolly_in, dolly_out, dolly_left, dolly_right, jib_up, jib_down, static, focus_shift

**Forced API Flow:**
- Validates API key, model, resolution, FPS, duration
- Handles three modes: A2V (audio + optional image), I2V (image only), T2V (text only)
- Uploads files via `LTXAPIClient.upload_file(api_key, file_path)`
- Calls appropriate API endpoint (generate_audio_to_video, generate_image_to_video, generate_text_to_video)
- Polls progress and downloads final MP4

---

### B. RetakeHandler

**Main Entry Point:** `handlers/retake_handler.py`

**Routes:**
```
POST /retake-video → RetakeHandler.run()
```

**Flow:**
1. **Validate inputs:** video_path exists, duration >= 2 seconds
2. **Check if API-forced:** Route to `_run_api_retake()` or `_run_local_retake()`
3. **Local Retake:**
   - Load full LTX model (not distilled by default, but supports distilled=True option)
   - Encode source video & audio via tiled VAE
   - Create TemporalRegionMask for [start_time, end_time)
   - Run diffusion with frozen context outside mask
   - Optional: enhance prompt via API
   - Output only the regenerated region + surrounding context

**Retake Modes:**
- audio_video (both)
- video_only
- audio_only

---

### C. IcLoraHandler

**Main Entry Point:** `handlers/ic_lora_handler.py`

**Routes:**
```
POST /ic-lora/extract → IcLoraHandler.extract_conditioning()
POST /ic-lora/generate → IcLoraHandler.generate()
```

**Extract Conditioning:**
- Read frame at specified time from video
- Build conditioning frame (Canny or Depth)
- Encode to JPEG
- Cache in conditioning store for reuse

**Generate:**
- Load IC-LoRA model
- Prepare text encoding
- Prepare video conditioning (list of images + feature types)
- Run generation with both image guides and IC-LoRA conditioning
- Output MP4

---

### D. SuggestGapPromptHandler

**Main Entry Point:** `handlers/suggest_gap_prompt_handler.py`

**What is "Gap Prompt"?**
- Gap = a time window between two video clips in a timeline
- User provides: before_frame, after_frame, before_prompt, after_prompt, gap_duration
- Handler uses **Gemini 2.0 Flash** (via Google API) to suggest a text prompt
- Suggested prompt is used to generate a video/image to fill the gap

**Gemini Integration:**
1. **Build system prompt:** "You are a video production assistant..."
2. **Build context:** Before shot info + after shot info + gap duration
3. **Collect images:** Load and base64-encode before_frame, after_frame, input_image (if I2V mode)
4. **Gemini request:**
   - Endpoint: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent`
   - Headers: x-goog-api-key + Content-Type
   - Body: {contents, systemInstruction, generationConfig}
   - Temperature: 0.7, maxOutputTokens: 512
5. **Return suggested_prompt** from Gemini response

**Modes:**
- text-to-image: Gap is an image
- image-to-video: Gap is a video starting from input_image
- text-to-video: Gap is a video (full generative)

---

## V. LONG VIDEO GENERATION & TEMPORAL HANDLING

### A. Audio-Aligned Looping (ComfyUI Integration)

The LTX Desktop **itself** is NOT designed for looping. However, it integrates with:
- **ComfyUI-AudioLoopHelper** (in coderef/) for full-length music video looping
- Uses LTX Desktop A2V pipeline within ComfyUI loop nodes
- Handles temporal continuity via:
  - Overlap frames (configurable overlap_seconds)
  - Stride computation (window_seconds - overlap_seconds)
  - Start index clamping to avoid MEL crash on short audio tail

### B. Tiling for Large Frames (VAE Encoding/Decoding)

**Default Tiling Config:**
```python
TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=128),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=16, tile_overlap_in_frames=8)
)
```

**Retake Encoding (More Conservative):**
```python
TilingConfig(
    spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
    temporal_config=TemporalTilingConfig(tile_size_in_frames=24, tile_overlap_in_frames=16)
)
```

**Applies to:**
- VAE video encoder (DistilledA2VPipeline, LTXRetakePipeline)
- VAE video decoder (all pipelines when frame count large)
- ltx_core.model.video_vae.get_video_chunks_number() calculates iteration count

---

## VI. KEY GENERATION PARAMETERS & SEMANTICS

### Seeds & Randomness
- **Seed < 0:** Auto-generates via `torch.randint(0, 2**31, (1,))`
- **Seed locked in settings:** Uses config locked_seed instead of time-based
- **Dev mode:** Always uses seed=1000 for reproducibility

### Frame Rate Support
- **Fast pipeline:** 8, 24, 25 typical
- **A2V pipeline:** 24, 25 typical
- **API-forced:** 24, 25, 48, 50 allowed

### Aspect Ratios
- **Local:** 16:9 or 9:16 only
- **API-forced:** 16:9 or 9:16
- Resolution labels: 540p, 720p, 1080p, 1440p, 2160p

### Duration Constraints (API-Forced)
```python
_get_allowed_durations(model_id, resolution_label, fps):
  if model_id == "ltx-2-3-fast" and resolution == "1080p" and fps in {24, 25}:
    return {6, 8, 10, 12, 14, 16, 18, 20}
  else:
    return {6, 8, 10}
```

---

## VII. QUALITY & TECHNIQUES

### Prompt Enhancement
- **Enabled:** `settings.prompt_enhancer_enabled_i2v` / `_t2v`
- **Only with API encoding** (local Gemma 3 doesn't enhance)
- Server-side enhancement (upstream LTX API)

### Camera Motion Integration
```python
config.camera_motion_prompts = {
    "dolly_in": "...",
    "dolly_out": "...",
    "dolly_left": "...",
    # etc.
}
enhanced_prompt = req.prompt + camera_motion_prompts.get(camera_motion, "")
```

### Image Guide Strength & Frame Index
- **strength:** Ignored in diffusion (document notes guidance doesn't work as expected)
- **frame_idx:** Tells model where in sequence to apply guide
- All guides concatenated to latent end (not blended at target index)

### No CFG in Distilled Mode
- Distilled pipelines use **SimpleDenoiser** (no classifier-free guidance)
- Full-mode retake can use **GuidedDenoiser** with CFG scale ~3.0

---

## VIII. API INTEGRATION

### LTX API Client
- **Cloud generation** via Lightricks' hosted service
- **Upload files:** Audio, images (returns URIs)
- **Generate endpoints:**
  - generate_text_to_video(api_key, prompt, model, resolution, duration, fps, generate_audio, camera_motion)
  - generate_image_to_video(api_key, prompt, image_uri, model, resolution, duration, fps, generate_audio, camera_motion)
  - generate_audio_to_video(api_key, prompt, audio_uri, image_uri, model, resolution)
  - retake(api_key, video_path, start_time, duration, prompt, mode)

### Text Encoding API
- Endpoint: `{ltx_api_base_url}/v1/prompt-embedding`
- Returns pickled EmbeddingsProcessorOutput with video_context + audio_context
- Supports prompt enhancement: enhance_prompt_seed for deterministic variation

---

## IX. STATE MANAGEMENT

**State Machine for Generation:**
```python
GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled

GpuGeneration {state: GenerationState}   # For local generations
ApiGeneration {state: GenerationState}   # For forced API generations
```

**Progress Tracking:**
```python
GenerationProgress {
    phase: str,              # "loading_model", "encoding_text", "inference", "complete"
    progress: int,           # 0-100%
    current_step: int | None,
    total_steps: int | None
}
```

**Cancellation:**
- `update_progress()` called frequently
- Handler checks `is_generation_cancelled()` at key points
- If cancelled: delete output file, raise RuntimeError("Generation was cancelled")

---

## X. CRITICAL FINDINGS & QUIRKS

1. **Audio-Only Conditioning:** A2V pipeline treats audio as frozen context (noise_scale=0.0), never regenerates. Original audio returned for fidelity.

2. **Two-Stage A2V Prevents Mode Collapse:** Stage 1 at half-res is more stable; Stage 2 refines without degradation.

3. **Retake Region Masking:** Uses TemporalRegionMask(start_time, end_time, fps) — NOT spatial. Entire latent is processed; region loss is applied via masking in denoiser.

4. **Prompt Enhancement Only in API Mode:** Local Gemma 3 encoder doesn't enhance; must use API encoding to enable enhancement.

5. **Memory Patches Essential:** The three patches (safetensors, record_stream, loader_fix) are critical for stability on high-VRAM GPUs and Windows.

6. **IC-LoRA Strength Fixed at 1.0:** Cannot control LoRA blend strength per-request; always 1.0.

7. **No CFG in Distilled Mode:** SimpleDenoiser ignores negative prompts; only full-mode retake supports CFG.

8. **Video Output Chunking:** Dynamic video_chunks calculation based on frame count to avoid OOM during encoding.

---

## SUMMARY TABLE

| Pipeline | Type | Audio | Sampling | Stages | Key Use Case |
|----------|------|-------|----------|--------|---|
| **Fast** | T2V/I2V | No | Distilled (fixed σ) | 1 | Quick video generation |
| **A2V** | T2V+Audio/I2V+Audio | Frozen (essential) | Distilled (stages 1→2) | 2 | Lip-sync, music videos |
| **Retake** | Region regen | Optional regen | Distilled OR full CFG | 1 | Fix/re-do specific sections |
| **IC-LoRA** | T2V+images | No | Distilled (fixed σ) | 1 | Consistent style across shots |
| **ZIT** | Image only | N/A | Diffusers scheduler | 1 | Reference image generation |

---

## FILE PATHS REFERENCED

**Critical pipeline files:**
- `/backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`
- `/backend/services/a2v_pipeline/distilled_a2v_pipeline.py`
- `/backend/services/a2v_pipeline/ltx_a2v_pipeline.py`
- `/backend/services/retake_pipeline/ltx_retake_pipeline.py`
- `/backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `/backend/services/image_generation_pipeline/zit_image_generation_pipeline.py`

**Shared utilities:**
- `/backend/services/ltx_pipeline_common.py` (DistilledNativePipeline, default_tiling_config, video_chunks_number)
- `/backend/services/text_encoder/ltx_text_encoder.py` (Gemma 3 + API encoding + patches)
- `/backend/services/patches/` (safetensors_metadata_fix, record_stream_fix, safetensors_loader_fix)
- `/backend/services/video_processor/video_processor_impl.py` (cv2-based preprocessing)

**Handlers:**
- `/backend/handlers/video_generation_handler.py` (Main T2V/I2V/A2V orchestration)
- `/backend/handlers/retake_handler.py` (Region regeneration)
- `/backend/handlers/ic_lora_handler.py` (IC-LoRA + conditioning extraction)
- `/backend/handlers/suggest_gap_prompt_handler.py` (Gemini gap prompt)
- `/backend/handlers/generation_handler.py` (Generation lifecycle + progress)

**Architecture & Configuration:**
- `/backend/architecture.md` (System design document)
- `/backend/api_types.py` (Request/response schemas)

---

This analysis captures all generation modes, sampling strategies, audio conditioning mechanics, and unique techniques in the official Lightricks LTX Desktop backend.
