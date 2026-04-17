Last updated: 2026-04-11

# Pipeline Flow: IMAGE-based Music Video Workflow

Complete start-to-end pipeline document for `example_workflows/audio-loop-music-video_image.json`.
Every node is documented in execution order, grouped by stage.

---

## High-level data flow

```
LoadAudio ──> TrimAudio(intro) ──> MelBandRoFormer ──> Set_actual_audio
                                       │
                                       ├──> TrimAudio(window) ──> AudioVAEEncode ──> SetLatentNoiseMask ──┐
                                       │                                                                   │
LoadImage ──> Resize ──> LTXVPreprocess ──> ImgToVideoInplace ──────────────────────┐                      │
                 │                                                                   │                      │
                 ├──> EmptyLTXVLatent ──────────────────────────────────────────> ImgToVideoInplace          │
                 │                                                                   │                      │
                 │                                                              LTXVConcatAVLatent <────────┘
                 │                                                                   │
DualCLIPLoader ──> CLIPTextEncode(pos) ──> LTXVConditioning ──> CFGGuider           │
                │                              │                    │                │
                └──> CLIPTextEncode(neg) ──> ZeroOut ──────────────┘                │
                                                                                     │
UNETLoader ──> SageAttn ──> ChunkFF ──> AttnTuner ──> NAG ──> PreviewOverride ──> Model
                                                                                     │
RandomNoise ──────────────────────────────────> SamplerCustomAdvanced <──────────────┘
                                                        │
                                                   SeparateAV ──> CropGuides
                                                        │              │
                                                   VAEDecode      (unused)
                                                        │
                                                    Reroute ──┬──> ImageBatch ──> VHS_VideoCombine
                                                              │
                                                         TensorLoopOpen
                                                              │
                      ┌───────────────────────────────────────┘
                      │
            AudioLoopController ──> start_index, should_stop, iteration_seed,
                      │               stride_seconds, overlap_frames
                      │
            TimestampPromptSchedule ──> prompt ──> CLIPTextEncode ──> LTXVConditioning
                      │
                Extension Subgraph #843
                      │
                 TensorLoopClose ──> ImageBatch ──> VHS_VideoCombine
```

---

## Stage 1: Model Loading

### Node 414 -- UNETLoader

| Field | Value |
|-------|-------|
| **Type** | `UNETLoader` |
| **Source** | ComfyUI core (`/home/fbliss/ComfyUI/nodes.py`) |
| **What it does** | Loads the LTX 2.3 diffusion transformer model from disk. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| unet_name | STRING (widget) | -- | `diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors` |
| weight_dtype | COMBO (widget) | -- | `default` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| MODEL | MODEL | Node 268 (PathchSageAttentionKJ) input 0 |

---

### Node 1537 -- VAELoaderKJ (Video VAE)

| Field | Value |
|-------|-------|
| **Type** | `VAELoaderKJ` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py`) |
| **What it does** | Loads the LTX 2.3 video VAE with explicit device and dtype control. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| vae_name | STRING (widget) | -- | `vae/LTX23_video_vae_bf16.safetensors` |
| device | COMBO (widget) | -- | `main_device` |
| dtype | COMBO (widget) | -- | `bf16` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| VAE | VAE | Node 228 (Set_video_vae) |

---

### Node 1538 -- VAELoaderKJ (Audio VAE)

| Field | Value |
|-------|-------|
| **Type** | `VAELoaderKJ` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py`) |
| **What it does** | Loads the LTX 2.3 audio VAE. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| vae_name | STRING (widget) | -- | `vae/LTX23_audio_vae_bf16.safetensors` |
| device | COMBO (widget) | -- | `main_device` |
| dtype | COMBO (widget) | -- | `bf16` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| VAE | VAE | Node 252 (Set_audio_vae) |

---

### Node 416 -- DualCLIPLoader

| Field | Value |
|-------|-------|
| **Type** | `DualCLIPLoader` |
| **Source** | ComfyUI core (`/home/fbliss/ComfyUI/nodes.py`) |
| **What it does** | Loads the Gemma 3 text encoder + LTX text projection. Despite the name "CLIP", this loads the Gemma 3 12B encoder used by LTX 2.3. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| clip_name1 | STRING (widget) | -- | `gemma_3_12B_it_fpmixed.safetensors` |
| clip_name2 | STRING (widget) | -- | `ltx-2.3_text_projection_bf16.safetensors` |
| type | COMBO (widget) | -- | `ltxv` |
| device | COMBO (widget) | -- | `default` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| CLIP | CLIP | Node 169 (pos text encode), Node 507 (neg text encode), Node 1559 (loop prompt encode) |

---

### Node 228 -- SetNode (Set_video_vae)

| Field | Value |
|-------|-------|
| **Type** | `SetNode` |
| **Source** | KJNodes |
| **What it does** | Stores video VAE under name `video_vae` for retrieval via GetNode anywhere in the graph. |

**Inputs:** VAE from Node 1537 (link 2863).
**Outputs:** Available via `Get_video_vae` nodes.

---

### Node 252 -- SetNode (Set_audio_vae)

Stores audio VAE under name `audio_vae`. Input: VAE from Node 1538 (link 2864).

---

### Node 444 -- LoadImage

| Field | Value |
|-------|-------|
| **Type** | `LoadImage` |
| **Source** | ComfyUI core |
| **What it does** | Loads the reference/init image from disk. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| image | STRING (widget) | -- | `reference_image.png` |
| upload | COMBO (widget) | -- | `image` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| IMAGE | IMAGE | Node 445 (ImageResizeKJv2) input 0 |
| MASK | MASK | Node 445 (ImageResizeKJv2) input 1 |

---

## Stage 2: Audio Preparation

### Node 565 -- LoadAudio

| Field | Value |
|-------|-------|
| **Type** | `LoadAudio` |
| **Source** | ComfyUI core |
| **What it does** | Loads the audio file from the input folder. Returns `{"waveform": Tensor, "sample_rate": int}`. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| audio | STRING (widget) | -- | `example_audio.mp4` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| AUDIO | AUDIO | Node 567 (TrimAudioDuration) input 0 |

---

### Node 567 -- TrimAudioDuration (Intro Trim)

| Field | Value |
|-------|-------|
| **Type** | `TrimAudioDuration` |
| **Source** | ComfyUI core |
| **What it does** | Trims the start of the audio to skip instrumental intro. `start_index` (seconds) sets where to begin; `duration` caps total length. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| audio | AUDIO | Node 565 (LoadAudio) | -- |
| start_index | FLOAT (widget) | -- | `5` (skip first 5 seconds) |
| duration | FLOAT (widget) | -- | `300` (max 300 seconds) |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| AUDIO | AUDIO | Node 569 (MelBandRoFormer), Node 581 (Set_orig_audio), Node 1560 (AudioLoopPlanner), Node 1582 (AudioLoopController) |

---

### Node 568 -- MelBandRoFormerModelLoader

| Field | Value |
|-------|-------|
| **Type** | `MelBandRoFormerModelLoader` |
| **Source** | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-MelBandRoFormer/nodes.py` |
| **What it does** | Loads the MelBandRoFormer vocal separation model. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| model_name | STRING (widget) | -- | `MelBandRoformer_fp32.safetensors` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| MELROFORMERMODEL | MELROFORMERMODEL | Node 569 (MelBandRoFormerSampler) input 0 |

---

### Node 569 -- MelBandRoFormerSampler

| Field | Value |
|-------|-------|
| **Type** | `MelBandRoFormerSampler` |
| **Source** | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-MelBandRoFormer/nodes.py` |
| **What it does** | Separates audio into vocals and instruments using mel-band roformer. Only vocals output is used -- instruments output is unwired. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| model | MELROFORMERMODEL | Node 568 | -- |
| audio | AUDIO | Node 567 (trimmed audio) | -- |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| vocals | AUDIO | Node 640 (Set_actual_audio) |
| instruments | AUDIO | **unwired** -- not needed for lip sync workflow |

---

### Node 640 -- SetNode (Set_actual_audio)

Stores the vocals-only audio under name `actual_audio`. This is the audio that feeds into the loop body for per-iteration encoding.

**Input:** AUDIO from Node 569 vocals output (link 2893).

---

### Node 581 -- SetNode (Set_orig_audio)

Stores the full (non-separated) trimmed audio under name `orig_audio`. This is used for the final video mux (VHS_VideoCombine) so the viewer hears the complete song.

**Input:** AUDIO from Node 567 (link 1580).

---

### Node 688 -- FloatConstant (window_size_seconds)

| Field | Value |
|-------|-------|
| **Type** | `FloatConstant` |
| **Source** | KJNodes |
| **What it does** | Holds the window duration constant: `19.88` seconds. This equals 497 frames / 25fps. |

**Inputs:** Widget value `19.88`.

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| value | FLOAT | Node 601 (TrimAudioDuration duration), Node 689 (Set_window_size_seconds) |

---

### Node 689 -- SetNode (Set_window_size_seconds)

Stores `19.88` under name `window_size_seconds` for retrieval in the loop section.

---

### Node 601 -- TrimAudioDuration (Window Trim)

| Field | Value |
|-------|-------|
| **Type** | `TrimAudioDuration` |
| **Source** | ComfyUI core |
| **What it does** | Trims the actual_audio to exactly `window_size_seconds` (19.88s) for the initial render's audio encoding. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| audio | AUDIO | Node 640 (actual_audio) via link 2821 | -- |
| duration | FLOAT | Node 688 (window_size_seconds) via link 1779 | `19.88` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| AUDIO | AUDIO | Node 566 (LTXVAudioVAEEncode) |

---

### Node 254 -- GetNode (Get_audio_vae)

Retrieves `audio_vae` for the initial audio encoding path.

---

### Node 566 -- LTXVAudioVAEEncode

| Field | Value |
|-------|-------|
| **Type** | `LTXVAudioVAEEncode` |
| **Source** | ComfyUI core (`/home/fbliss/ComfyUI/comfy_extras/nodes_lt.py`) |
| **What it does** | Encodes the trimmed audio waveform into audio latent space using the LTX audio VAE. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| audio | AUDIO | Node 601 (trimmed window audio) | -- |
| audio_vae | VAE | Node 254 (Get_audio_vae) | -- |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| Audio Latent | LATENT | Node 570 (SetLatentNoiseMask) |

---

### Node 571 -- SolidMask

| Field | Value |
|-------|-------|
| **Type** | `SolidMask` |
| **Source** | ComfyUI core |
| **What it does** | Creates an all-zeros mask (512x512). Value `0` means "fixed context" -- the audio latent is kept as-is (real encoded song, not regenerated). |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| value | FLOAT (widget) | -- | `0` |
| width | INT (widget) | -- | `512` |
| height | INT (widget) | -- | `512` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| MASK | MASK | Node 570 (SetLatentNoiseMask) input 1 |

---

### Node 570 -- SetLatentNoiseMask

| Field | Value |
|-------|-------|
| **Type** | `SetLatentNoiseMask` |
| **Source** | ComfyUI core |
| **What it does** | Applies the all-zeros mask to the audio latent. This tells the sampler to treat the audio latent as fixed context (mask=0 = no denoising). |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| samples | LATENT | Node 566 (audio latent) | -- |
| mask | MASK | Node 571 (SolidMask, value=0) | -- |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| LATENT | LATENT | Node 350 (LTXVConcatAVLatent) audio_latent input |

---

## Stage 3: Image Preparation and Latent Construction

### Node 445 -- ImageResizeKJv2

| Field | Value |
|-------|-------|
| **Type** | `ImageResizeKJv2` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/image_nodes.py`) |
| **What it does** | Resizes the input image to the target generation resolution using lanczos interpolation with center crop. Also outputs the final width/height for the empty latent. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| image | IMAGE | Node 444 (LoadImage) | -- |
| mask | MASK | Node 444 (LoadImage) mask output | -- |
| width | INT (widget) | -- | `832` |
| height | INT (widget) | -- | `480` |
| interpolation | COMBO (widget) | -- | `lanczos` |
| method | COMBO (widget) | -- | `crop` |
| divisor | INT (widget) | -- | `2` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| IMAGE | IMAGE | Node 446 (LTXVPreprocess), Node 650 (Set_input_image) |
| width | INT | Node 344 (EmptyLTXVLatentVideo) width |
| height | INT | Node 344 (EmptyLTXVLatentVideo) height |
| mask | MASK | **unwired** -- not needed downstream |

---

### Node 650 -- SetNode (Set_input_image)

Stores the resized image under name `input_image` for use in the extension subgraph as the scene anchor guide.

---

### Node 446 -- LTXVPreprocess

| Field | Value |
|-------|-------|
| **Type** | `LTXVPreprocess` |
| **Source** | ComfyUI core (`/home/fbliss/ComfyUI/comfy_extras/nodes_lt.py`) |
| **What it does** | Applies LTX-specific JPEG-like compression preprocessing to the image (simulates training distribution). `img_compression=0` means no compression. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| image | IMAGE | Node 445 (resized image) | -- |
| img_compression | INT (widget) | -- | `0` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| output_image | IMAGE | Node 531 (LTXVImgToVideoInplaceKJ) image_1 |

---

### Node 526 -- PrimitiveNode (length)

| Field | Value |
|-------|-------|
| **Type** | `PrimitiveNode` |
| **What it does** | Supplies the video frame count constant. 497 = 8*62 + 1, the maximum LTX 2.3 temporal window. |

**Value:** `497`
**Output:** INT -> Node 344 (EmptyLTXVLatentVideo) length input.

---

### Node 344 -- EmptyLTXVLatentVideo

| Field | Value |
|-------|-------|
| **Type** | `EmptyLTXVLatentVideo` |
| **Source** | ComfyUI core (`/home/fbliss/ComfyUI/comfy_extras/nodes_lt.py`) |
| **What it does** | Creates an empty video latent tensor of specified dimensions. The temporal dimension follows the LTX formula: `(497-1)//8 + 1 = 63` latent frames. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| width | INT | Node 445 output 1 | `832` |
| height | INT | Node 445 output 2 | `480` |
| length | INT | Node 526 (PrimitiveNode) | `497` |
| batch_size | INT (widget) | -- | `1` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| LATENT | LATENT | Node 531 (LTXVImgToVideoInplaceKJ) latent input |

---

### Node 413 -- GetNode (Get_video_vae)

Retrieves `video_vae` for image-to-video latent encoding.

---

### Node 531 -- LTXVImgToVideoInplaceKJ

| Field | Value |
|-------|-------|
| **Type** | `LTXVImgToVideoInplaceKJ` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py`) |
| **What it does** | VAE-encodes the preprocessed image and replaces the first frame of the empty latent with the encoded image. Sets noise_mask=0 at frame index 0 (fixed context) and mask=1 everywhere else (to be denoised). This creates the i2v (image-to-video) conditioning. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| vae | VAE | Node 413 (Get_video_vae) | -- |
| latent | LATENT | Node 344 (empty latent) | -- |
| num_images | DynamicCombo (widget) | -- | `1` |
| image_1 | IMAGE | Node 446 (preprocessed image) | -- |
| strength_1 | FLOAT (widget) | -- | `1.0` |
| index_1 | INT (widget) | -- | `0` (first frame) |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| latent | LATENT | Node 350 (LTXVConcatAVLatent) video_latent input |

---

### Node 350 -- LTXVConcatAVLatent

| Field | Value |
|-------|-------|
| **Type** | `LTXVConcatAVLatent` |
| **Source** | ComfyUI core (`/home/fbliss/ComfyUI/comfy_extras/nodes_lt.py`) |
| **What it does** | Combines the video latent (with init image embedded) and the masked audio latent into a single audio-visual NestedTensor latent. This is the input to the sampler. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| video_latent | LATENT | Node 531 (img-embedded latent) | -- |
| audio_latent | LATENT | Node 570 (masked audio latent) | -- |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| latent | LATENT | Node 161 (SamplerCustomAdvanced) latent_image input |

---

## Stage 4: Text Encoding / Conditioning

### Node 169 -- CLIPTextEncode (Positive Prompt)

| Field | Value |
|-------|-------|
| **Type** | `CLIPTextEncode` |
| **Source** | ComfyUI core |
| **What it does** | Encodes the positive prompt text into conditioning tensors via the Gemma 3 encoder. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| clip | CLIP | Node 416 (DualCLIPLoader) | -- |
| text | STRING (widget) | -- | `"video of a woman passionately singing alone"` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| CONDITIONING | CONDITIONING | Node 164 (LTXVConditioning) positive, Node 420 (ConditioningZeroOut) |

---

### Node 507 -- CLIPTextEncode (Negative Prompt)

| Field | Value |
|-------|-------|
| **Type** | `CLIPTextEncode` |
| **Source** | ComfyUI core |
| **What it does** | Encodes the negative prompt. This is used as the NAG video conditioning reference, NOT as the CFG negative (that uses zeroed-out conditioning). |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| clip | CLIP | Node 416 (DualCLIPLoader) | -- |
| text | STRING (widget) | -- | `"still image with no motion, subtitles, text, scene change, instruments, violin, blurry, out of focus..."` (full negative prompt) |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| CONDITIONING | CONDITIONING | Node 508 (LTX2_NAG) nag_cond_video |

---

### Node 420 -- ConditioningZeroOut

| Field | Value |
|-------|-------|
| **Type** | `ConditioningZeroOut` |
| **Source** | ComfyUI core |
| **What it does** | Zeros out all conditioning tensor values. LTX 2.3 is distilled (CFG=1.0), so the actual negative guidance is handled by NAG. This zeroed conditioning becomes the CFGGuider's "negative" input. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| conditioning | CONDITIONING | Node 169 (positive encode) | -- |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| CONDITIONING | CONDITIONING | Node 164 (LTXVConditioning) negative |

---

### Node 164 -- LTXVConditioning (Initial Render)

| Field | Value |
|-------|-------|
| **Type** | `LTXVConditioning` |
| **Source** | ComfyUI core (`/home/fbliss/ComfyUI/comfy_extras/nodes_lt.py`) |
| **What it does** | Wraps conditioning with LTX frame_rate metadata (25fps). Required so the sampler knows the temporal resolution. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| positive | CONDITIONING | Node 169 (positive encode) | -- |
| negative | CONDITIONING | Node 420 (zeroed conditioning) | -- |
| frame_rate | INT (widget) | -- | `25` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| positive | CONDITIONING | Node 153 (CFGGuider), Node 381 (CropGuides), Node 645 (Set_base_cond_pos) |
| negative | CONDITIONING | Node 153 (CFGGuider), Node 381 (CropGuides), Node 646 (Set_base_cond_neg) |

---

### Node 645 -- SetNode (Set_base_cond_pos)

Stores positive conditioning under `base_cond_pos` for use in the loop conditioning path.

### Node 646 -- SetNode (Set_base_cond_neg)

Stores negative conditioning under `base_cond_neg`.

---

## Stage 5: Model Pipeline (Optimization + NAG)

### Node 268 -- PathchSageAttentionKJ (BYPASSED, mode=4)

| Field | Value |
|-------|-------|
| **Type** | `PathchSageAttentionKJ` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/model_optimization_nodes.py`) |
| **What it does** | Patches attention to use SageAttention for faster inference. Currently BYPASSED (mode=4) -- passes model through unchanged. |

**Inputs:** MODEL from Node 414.
**Outputs:** MODEL to Node 504.

---

### Node 504 -- LTXVChunkFeedForward

| Field | Value |
|-------|-------|
| **Type** | `LTXVChunkFeedForward` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py`) |
| **What it does** | Chunks feedforward activations to reduce peak VRAM. Splits activations above `dim_threshold` into `chunks` pieces. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| model | MODEL | Node 268 (passthrough) | -- |
| chunks | INT (widget) | -- | `2` |
| dim_threshold | INT (widget) | -- | `4096` |

**Outputs:** MODEL to Node 1523.

---

### Node 1523 -- LTX2AttentionTunerPatch

| Field | Value |
|-------|-------|
| **Type** | `LTX2AttentionTunerPatch` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py`) |
| **What it does** | Patches the LTX2 transformer forward pass with per-modality attention scaling. Also reduces peak VRAM. All scales at 1.0 = no change to attention weights. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| model | MODEL | Node 504 | -- |
| blocks | STRING (widget) | -- | `""` (all blocks) |
| video_scale | FLOAT (widget) | -- | `1.0` |
| audio_scale | FLOAT (widget) | -- | `1.0` |
| audio_to_video_scale | FLOAT (widget) | -- | `1.0` |
| video_to_audio_scale | FLOAT (widget) | -- | `1.0` |
| triton_kernels | BOOLEAN (widget) | -- | `true` |

**Outputs:** MODEL to Node 508.

---

### Node 508 -- LTX2_NAG

| Field | Value |
|-------|-------|
| **Type** | `LTX2_NAG` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py`) |
| **What it does** | Applies Normalized Attention Guidance (NAG) to the model. NAG is the primary guidance mechanism for distilled LTX 2.3 (not CFG). The `nag_cond_video` input provides the negative prompt conditioning that NAG uses to steer away from undesired outputs. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| model | MODEL | Node 1523 | -- |
| nag_scale | FLOAT (widget) | -- | `11.0` |
| nag_alpha | FLOAT (widget) | -- | `0.25` |
| nag_tau | FLOAT (widget) | -- | `2.5` |
| nag_cond_video | CONDITIONING | Node 507 (negative text encode) | -- |
| nag_cond_audio | CONDITIONING | -- | **unwired** (no separate audio NAG conditioning) |
| inplace | BOOLEAN (widget) | -- | `true` |

**Outputs:** MODEL to Node 503.

---

### Node 503 -- LTX2SamplingPreviewOverride

| Field | Value |
|-------|-------|
| **Type** | `LTX2SamplingPreviewOverride` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py`) |
| **What it does** | Overrides the model's preview callback to show intermediate sampling frames during generation. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| model | MODEL | Node 508 (NAG-patched) | -- |
| preview_rate | INT (widget) | -- | `8` |
| latent_upscale_model | LATENT_UPSCALE_MODEL | -- | **unwired** |
| vae | VAE | -- | **unwired** |

**Outputs:** MODEL to Node 572 (Set_model), Node 1513 (ModelSamplingSD3).

---

### Node 572 -- SetNode (Set_model)

Stores the fully-patched model under `model`. Both the initial render CFGGuider and the extension subgraph retrieve it.

---

### Node 1513 -- ModelSamplingSD3

| Field | Value |
|-------|-------|
| **Type** | `ModelSamplingSD3` |
| **Source** | ComfyUI core |
| **What it does** | Sets the model's noise schedule shift parameter. Value `13` controls the sigma schedule shift for LTX 2.3. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| model | MODEL | Node 503 | -- |
| shift | FLOAT (widget) | -- | `13` |

**Outputs:** MODEL to Node 1421 (BasicScheduler).

---

## Stage 6: Initial Render (First Pass Sampling)

### Node 1527 -- INTConstant (start_seed)

| Field | Value |
|-------|-------|
| **Type** | `INTConstant` |
| **Source** | KJNodes |
| **What it does** | Holds the base seed value. |

**Value:** `42`
**Output:** INT -> Node 1528 (Set_start_seed).

---

### Node 1528 -- SetNode (Set_start_seed)

Stores `42` under `start_seed`.

---

### Node 1530 -- GetNode (Get_start_seed)

Retrieves `start_seed` for the initial render's noise.

---

### Node 1322 -- RandomNoise

| Field | Value |
|-------|-------|
| **Type** | `RandomNoise` |
| **Source** | ComfyUI core |
| **What it does** | Generates random noise from the given seed for the initial render. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| noise_seed | INT | Node 1530 (Get_start_seed) | `42` |

**Outputs:** NOISE to Node 161 (SamplerCustomAdvanced).

---

### Node 153 -- CFGGuider (Initial Render)

| Field | Value |
|-------|-------|
| **Type** | `CFGGuider` |
| **Source** | ComfyUI core |
| **What it does** | Packages model + conditioning for the sampler. `cfg=1.0` because LTX 2.3 is distilled -- NAG handles guidance, not CFG. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| model | MODEL | Node 572 (Set_model) | -- |
| positive | CONDITIONING | Node 164 (LTXVConditioning) positive | -- |
| negative | CONDITIONING | Node 164 (LTXVConditioning) negative | -- |
| cfg | FLOAT (widget) | -- | `1.0` |

**Outputs:** GUIDER to Node 161 (sampler), Node 575 (Set_guider).

---

### Node 575 -- SetNode (Set_guider)

Stores the guider. (Not consumed by the loop -- the loop builds its own CFGGuider inside the subgraph.)

### Node 576 -- SetNode (Set_sampler)

Stores the sampler from Node 154.

---

### Node 154 -- KSamplerSelect

| Field | Value |
|-------|-------|
| **Type** | `KSamplerSelect` |
| **Source** | ComfyUI core |
| **What it does** | Selects the sampling algorithm. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| sampler_name | COMBO (widget) | -- | `euler` |

**Outputs:** SAMPLER to Node 161, Node 576 (Set_sampler).

---

### Node 1421 -- BasicScheduler

| Field | Value |
|-------|-------|
| **Type** | `BasicScheduler` |
| **Source** | ComfyUI core |
| **What it does** | Generates sigma schedule for the sampler using the model's noise schedule. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| model | MODEL | Node 1513 (ModelSamplingSD3) | -- |
| scheduler | COMBO (widget) | -- | `linear_quadratic` |
| steps | INT (widget) | -- | `8` |
| denoise | FLOAT (widget) | -- | `1.0` |

**Outputs:** SIGMAS to Node 1422 (VisualizeSigmasKJ).

---

### Node 1422 -- VisualizeSigmasKJ

| Field | Value |
|-------|-------|
| **Type** | `VisualizeSigmasKJ` |
| **Source** | KJNodes |
| **What it does** | Passes sigmas through unchanged while generating a preview image of the sigma schedule. |

**Outputs:** SIGMAS to Node 579 (Set_sigmas), preview IMAGE to Node 1423 (PreviewImage).

---

### Node 579 -- SetNode (Set_sigmas)

Stores sigmas under `sigmas` for the extension subgraph.

---

### Node 1269 -- FloatConstant (first_frame_guide_strength)

**Value:** `1.0`
**Output:** FLOAT to Node 1271 (Set_first_frame_guide_strength).

### Node 1271 -- SetNode (Set_first_frame_guide_strength)

Stores `1.0` under `first_frame_guide_strength`. This controls the init image guide strength in the extension subgraph.

---

### Node 161 -- SamplerCustomAdvanced (Initial Render)

| Field | Value |
|-------|-------|
| **Type** | `SamplerCustomAdvanced` |
| **Source** | ComfyUI core |
| **What it does** | Runs the full denoising pass for the initial 497-frame video from the image + audio latent. This is the first window of the music video. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| noise | NOISE | Node 1322 (RandomNoise) | -- |
| guider | GUIDER | Node 153 (CFGGuider) | -- |
| sampler | SAMPLER | Node 154 (KSamplerSelect) | -- |
| sigmas | SIGMAS | Node 579 (Set_sigmas) | -- |
| latent_image | LATENT | Node 350 (AV concat latent) | -- |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| output | LATENT | Node 245 (LTXVSeparateAVLatent) |
| denoised_output | LATENT | **unwired** |

---

### Node 245 -- LTXVSeparateAVLatent

| Field | Value |
|-------|-------|
| **Type** | `LTXVSeparateAVLatent` |
| **Source** | ComfyUI core |
| **What it does** | Splits the NestedTensor AV latent back into separate video and audio latents. Only the video latent is used downstream. |

**Inputs:** LATENT from Node 161.

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| video_latent | LATENT | Node 381 (CropGuides), Node 1318 (VAEDecode) |
| audio_latent | LATENT | **unwired** |

---

### Node 381 -- LTXVCropGuides (Initial Render)

| Field | Value |
|-------|-------|
| **Type** | `LTXVCropGuides` |
| **Source** | ComfyUI core |
| **What it does** | Removes appended guide frames from the latent so only the generated content remains. Guide frames were appended by LTXVImgToVideoInplaceKJ. All outputs are unwired here -- this node is present for completeness but not consumed in the IMAGE workflow initial path. |

**Inputs:** positive/negative from Node 164, latent from Node 245.
**Outputs:** All unwired (latent, positive, negative).

---

### Node 236 -- GetNode (Get_video_vae)

Retrieves `video_vae` for decoding.

---

### Node 1318 -- VAEDecode (Initial Render)

| Field | Value |
|-------|-------|
| **Type** | `VAEDecode` |
| **Source** | ComfyUI core |
| **What it does** | Decodes the video latent back to pixel-space IMAGE tensor (497 frames). |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| samples | LATENT | Node 245 (video_latent) | -- |
| vae | VAE | Node 236 (Get_video_vae) | -- |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| IMAGE | IMAGE | Node 560 (preview VHS_VideoCombine, BYPASSED), Node 618 (Reroute) |

---

### Node 560 -- VHS_VideoCombine (Initial Render Preview, BYPASSED mode=4)

Preview-only video combiner for the initial render. Bypassed by default.

---

### Node 618 -- Reroute

Passes the decoded initial render images to both the ImageBatch and TensorLoopOpen.

**Input:** IMAGE from Node 1318.
**Outputs:** IMAGE to Node 1507 (ImageBatch image1), Node 1539 (TensorLoopOpen initial_value).

---

## Stage 7: Loop Setup

### Node 1539 -- TensorLoopOpen

| Field | Value |
|-------|-------|
| **Type** | `TensorLoopOpen` |
| **Source** | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-NativeLooping_testing/nodes.py` |
| **What it does** | Opens the loop. On first iteration, `previous_value` = the decoded initial render images. On subsequent iterations, it receives the previous loop body output. The `current_iteration` output (1-based) drives AudioLoopController and TimestampPromptSchedule. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| mode | DynamicCombo (widget) | -- | `iterations` |
| iterations | INT (widget) | -- | `50` (max, but AudioLoopController stops early) |
| initial_value | IMAGE | Node 618 (Reroute, initial render) | -- |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| flow_control | FLOW_CONTROL | Node 1540 (TensorLoopClose) |
| previous_value | IMAGE | Node 843 (extension subgraph) `previous_images` input |
| accumulated_count | INT | **unwired** |
| current_iteration | INT | Node 1582 (AudioLoopController), Node 1558 (TimestampPromptSchedule) |

---

### Node 1582 -- AudioLoopController

| Field | Value |
|-------|-------|
| **Type** | `AudioLoopController` |
| **Source** | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-AudioLoopHelper/nodes.py` |
| **What it does** | Core loop timing node. Computes `start_index = iteration * stride`, checks if next iteration would overshoot audio, outputs per-iteration seed, stride, and overlap frame counts. Clamps start_index to prevent mel spectrogram crash on final iteration. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| current_iteration | INT | Node 1539 (TensorLoopOpen) output 3 | -- |
| window_seconds | FLOAT | Node 691 (Get_window_size_seconds) | `19.88` |
| overlap_seconds | FLOAT (widget) | -- | `1.0` |
| audio | AUDIO | Node 567 (trimmed full audio) | -- |
| seed | INT | Node 1529 (Get_start_seed) | `42` |
| fps | INT (widget) | -- | `25` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| start_index | FLOAT | Node 843 (extension subgraph) `start_index` input |
| should_stop | BOOLEAN | Node 1540 (TensorLoopClose) `stop` input |
| audio_duration | FLOAT | **unwired** |
| iteration_seed | INT | Node 843 (extension subgraph) `noise_seed` input |
| stride_seconds | FLOAT | Node 1558 (TimestampPromptSchedule), Node 1560 (AudioLoopPlanner) |
| overlap_frames | INT | Node 843 (extension subgraph) `overlap` input, Node 1586 (PreviewAny) |
| overlap_latent_frames | INT | **unwired** (only used in LATENT workflow) |

**Computed values at defaults (overlap_seconds=1.0, window_seconds=19.88):**
- stride = 19.88 - 1.0 = 18.88 seconds
- overlap_frames = round(1.0 * 25) = 25 frames
- overlap_latent_frames = (25-1)//8 + 1 = 4

---

### Node 1558 -- TimestampPromptSchedule (BYPASSED, mode=4)

| Field | Value |
|-------|-------|
| **Type** | `TimestampPromptSchedule` |
| **Source** | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-AudioLoopHelper/nodes.py` |
| **What it does** | Selects a prompt based on the current audio position. BYPASSED by default -- uses static prompt mode instead. When enabled, enables per-section prompt scheduling (verse, chorus, bridge, etc.). |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| current_iteration | INT | Node 1539 output 3 | -- |
| stride_seconds | FLOAT | Node 1582 output 4 | -- |
| schedule | STRING (widget) | -- | `"0:00+: video of a woman passionately singing alone"` |
| blend_seconds | FLOAT (widget) | -- | `0.0` (not shown, default) |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| prompt | STRING | Node 1559 (CLIPTextEncode) text input |
| next_prompt | STRING | **unwired** (no blending active) |
| blend_factor | FLOAT | **unwired** |
| current_time | FLOAT | **unwired** |

---

### Node 1559 -- CLIPTextEncode (Loop Prompt Encode, BYPASSED via schedule)

| Field | Value |
|-------|-------|
| **Type** | `CLIPTextEncode` |
| **Source** | ComfyUI core |
| **What it does** | Encodes the per-iteration prompt from TimestampPromptSchedule. Currently the prompt schedule is bypassed, so this node's output is not consumed. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| clip | CLIP | Node 416 (DualCLIPLoader) | -- |
| text | STRING | Node 1558 (prompt output) | -- |

**Outputs:** CONDITIONING -- links array is empty (not connected to subgraph in static mode).

---

### Node 1588 -- GetNode (Get_base_cond_pos, Static Mode)

Retrieves `base_cond_pos` -- the static positive conditioning for loop iterations.

**Output:** CONDITIONING directly to Node 843 (extension subgraph) `positive` input.

---

### Node 648 -- GetNode (Get_base_cond_neg)

Retrieves `base_cond_neg`.

**Output:** CONDITIONING directly to Node 843 (extension subgraph) `negative` input.

---

### Node 1587 -- LTXVConditioning (Loop) -- BYPASSED

| Field | Value |
|-------|-------|
| **Type** | `LTXVConditioning` |
| **Mode** | **4 (BYPASSED)** |
| **Source** | ComfyUI core |
| **Why bypassed** | Was wrapping the Extension's conditioning with frame_rate=25. This caused ComfyUI's execution engine to evaluate the conditioning graph in a way that corrupted the initial render's audio-video cross-attention, destroying lip sync. Removed 2026-04-12. See `internal/postmortem_v0409_latent_rework.md` Issue 6. |

Conditioning now flows directly from Get_base_cond_pos (#1588) and
Get_base_cond_neg (#648) to Extension #843 positive/negative inputs.

---

### Node 1560 -- AudioLoopPlanner (BYPASSED, mode=4)

| Field | Value |
|-------|-------|
| **Type** | `AudioLoopPlanner` |
| **Source** | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-AudioLoopHelper/nodes.py` |
| **What it does** | Displays the iteration timeline for prompt schedule planning. Shows what time range each iteration covers. Runs once (outside loop). Currently bypassed. |

**Inputs:** audio from Node 567, stride_seconds from Node 1582, window_seconds from Node 691.
**Outputs:** summary STRING to Node 1563 (PreviewAny).

---

### Node 1586 -- PreviewAny (overlap_frames preview)

Displays the computed overlap_frames value. Useful for verifying the AudioLoopController's output.

---

## Stage 8: Extension Subgraph #843

Node 843 is a **group node (subgraph)** with ID `b4973d68-09b9-4da5-9845-38ad62ae9aca`. It encapsulates the per-iteration video extension pipeline. The subgraph has an input distributor node (-10) and output collector (-20).

### Subgraph External Inputs

| Slot | Label | Type | External Source |
|------|-------|------|-----------------|
| 0 | sampler | SAMPLER | Node 578 (Get_sampler) |
| 1 | sigmas | SIGMAS | Node 580 (Get_sigmas) |
| 2 | model | MODEL | Node 654 (Get_model) |
| 3 | vae | VAE | Node 619 (Get_video_vae) |
| 4 | previous_images | IMAGE | Node 1539 (TensorLoopOpen) previous_value |
| 5 | window_size_seconds | FLOAT | Node 691 (Get_window_size_seconds) |
| 6 | positive | CONDITIONING | Node 1588 (Get_base_cond_pos) directly |
| 7 | negative | CONDITIONING | Node 648 (Get_base_cond_neg) directly |
| 8 | init_image | IMAGE | Node 651 (Get_input_image) |
| 9 | Audio VAE | VAE | Node 599 (Get_audio_vae) |
| 10 | audio | AUDIO | Node 641 (Get_actual_audio) |
| 11 | start_index | FLOAT | Node 1582 (AudioLoopController) start_index |
| 12 | first_frame_guide_strength | FLOAT | Node 1273 (Get_first_frame_guide_strength) |
| 13 | noise_seed | INT | Node 1582 (AudioLoopController) iteration_seed |
| 14 | overlap | INT | Node 1582 (AudioLoopController) overlap_frames |

### Subgraph External Outputs

| Slot | Label | Type | External Target |
|------|-------|------|-----------------|
| 0 | extended_images | IMAGE | Node 1540 (TensorLoopClose) processed input |

---

### Internal Node 615 -- GetImageRangeFromBatch (Extract Overlap Tail)

| Field | Value |
|-------|-------|
| **Type** | `GetImageRangeFromBatch` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/image_nodes.py`) |
| **What it does** | Extracts the last `overlap` (25) frames from `previous_images`. These tail frames provide visual continuity for the next generation window. `start_index=-1` means "start from the end minus num_frames". |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| images | IMAGE | Subgraph input slot 4 (previous_images) | -- |
| masks | MASK | -- | **unwired** |
| start_index | INT (widget) | -- | `-1` (from end) |
| num_frames | INT | Subgraph input slot 14 (overlap) | `25` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| IMAGE | IMAGE | Internal Node 614 (VAEEncode) |
| MASK | MASK | **unwired** |

---

### Internal Node 614 -- VAEEncode (Encode Overlap Tail)

| Field | Value |
|-------|-------|
| **Type** | `VAEEncode` |
| **Source** | ComfyUI core |
| **What it does** | Encodes the overlap tail frames into video latent space. This becomes the video latent input for the mask/guide pipeline. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| pixels | IMAGE | Internal Node 615 (overlap tail frames) | -- |
| vae | VAE | Subgraph input slot 3 (video_vae) | -- |

**Outputs:** LATENT to Internal Node 606 (LTXVAudioVideoMask) video_latent.

---

### Internal Node 600 -- TrimAudioDuration (Per-Iteration Audio)

| Field | Value |
|-------|-------|
| **Type** | `TrimAudioDuration` |
| **Source** | ComfyUI core |
| **What it does** | Trims audio for this iteration's window: starts at `start_index` seconds, duration = `window_size_seconds`. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| audio | AUDIO | Subgraph input slot 10 (actual_audio) | -- |
| start_index | FLOAT | Subgraph input slot 11 (start_index from AudioLoopController) | -- |
| duration | FLOAT | Subgraph input slot 5 (window_size_seconds) | `19.88` |

**Outputs:** AUDIO to Internal Node 598 (LTXVAudioVAEEncode).

---

### Internal Node 598 -- LTXVAudioVAEEncode (Per-Iteration Audio)

| Field | Value |
|-------|-------|
| **Type** | `LTXVAudioVAEEncode` |
| **Source** | ComfyUI core |
| **What it does** | Encodes this iteration's audio window into latent space. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| audio | AUDIO | Internal Node 600 (trimmed audio) | -- |
| audio_vae | VAE | Subgraph input slot 9 (audio_vae) | -- |

**Outputs:** LATENT to Internal Node 606 (LTXVAudioVideoMask) audio_latent.

---

### Internal Node 606 -- LTXVAudioVideoMask

| Field | Value |
|-------|-------|
| **Type** | `LTXVAudioVideoMask` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py`) |
| **What it does** | Creates noise masks for both video and audio latents. The video gets a mask indicating which temporal range to generate (from start to `video_end_time`). The audio mask has `audio_start_time = audio_end_time = window_size_seconds`, creating an empty mask range so audio stays fixed as the encoded song (mask=0). |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| video_latent | LATENT | Internal Node 614 (encoded overlap tail) | -- |
| audio_latent | LATENT | Internal Node 598 (encoded audio) | -- |
| video_fps | FLOAT (widget) | -- | `25` |
| video_start_time | FLOAT (widget) | -- | `1` |
| video_end_time | FLOAT | Subgraph input slot 5 (window_size_seconds) | `19.88` |
| audio_start_time | FLOAT | Subgraph input slot 5 (window_size_seconds) | `19.88` |
| audio_end_time | FLOAT | Subgraph input slot 5 (window_size_seconds) | `19.88` |
| max_length | COMBO (widget) | -- | `pad` |
| existing_mask_mode | COMBO (widget) | -- | `add` |

**Critical detail:** Both `audio_start_time` and `audio_end_time` are wired to the same `window_size_seconds` value (19.88). This creates a zero-width audio mask range (start=end), meaning NO audio frames are masked for regeneration. The audio latent stays as-is (real encoded song).

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| video_latent | LATENT | Internal Node 1519 (LTXVAddLatentGuide) latent |
| audio_latent | LATENT | Internal Node 583 (LTXVConcatAVLatent) audio_latent |

---

### Internal Node 1520 -- VAEEncode (Encode Init Image for Guide)

| Field | Value |
|-------|-------|
| **Type** | `VAEEncode` |
| **Source** | ComfyUI core |
| **What it does** | Encodes the init image (scene anchor) into latent space to serve as a guide frame. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| pixels | IMAGE | Subgraph input slot 8 (init_image) | -- |
| vae | VAE | Subgraph input slot 3 (video_vae) | -- |

**Outputs:** LATENT to Internal Node 1519 (LTXVAddLatentGuide) guiding_latent.

---

### Internal Node 1519 -- LTXVAddLatentGuide

| Field | Value |
|-------|-------|
| **Type** | `LTXVAddLatentGuide` |
| **Source** | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-LTXVideo/latents.py` |
| **What it does** | Appends the init image as a guide frame at `latent_idx=-1` (before the first frame). Guide strength controls the denoise mask. This is the scene anchor that maintains visual consistency across loop iterations. Also appends guide metadata to conditioning. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| vae | VAE | Subgraph input slot 3 (video_vae) | -- |
| positive | CONDITIONING | Subgraph input slot 6 (positive) | -- |
| negative | CONDITIONING | Subgraph input slot 7 (negative) | -- |
| latent | LATENT | Internal Node 606 (masked video latent) | -- |
| guiding_latent | LATENT | Internal Node 1520 (encoded init image) | -- |
| latent_idx | INT (widget) | -- | `-1` (before first frame) |
| strength | FLOAT | Subgraph input slot 12 (first_frame_guide_strength) | `1.0` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| positive | CONDITIONING | Internal Node 644 (CFGGuider), Internal Node 655 (CropGuides) |
| negative | CONDITIONING | Internal Node 644 (CFGGuider), Internal Node 655 (CropGuides) |
| latent | LATENT | Internal Node 583 (LTXVConcatAVLatent) video_latent |

---

### Internal Node 583 -- LTXVConcatAVLatent (Loop)

| Field | Value |
|-------|-------|
| **Type** | `LTXVConcatAVLatent` |
| **What it does** | Combines the guide-embedded video latent with the masked audio latent into a single AV latent for sampling. |

**Inputs:** video_latent from Node 1519 latent, audio_latent from Node 606 audio output.
**Output:** LATENT to Internal Node 573 (SamplerCustomAdvanced).

---

### Internal Node 644 -- CFGGuider (Loop)

| Field | Value |
|-------|-------|
| **Type** | `CFGGuider` |
| **Source** | ComfyUI core |
| **What it does** | Packages model + guide-modified conditioning for the loop sampler. cfg=1.0 (NAG handles guidance). |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| model | MODEL | Subgraph input slot 2 (model) | -- |
| positive | CONDITIONING | Internal Node 1519 output 0 | -- |
| negative | CONDITIONING | Internal Node 1519 output 1 | -- |
| cfg | FLOAT (widget) | -- | `1.0` |

**Output:** GUIDER to Internal Node 573.

---

### Internal Node 574 -- RandomNoise (Loop)

| Field | Value |
|-------|-------|
| **Type** | `RandomNoise` |
| **Source** | ComfyUI core |
| **What it does** | Generates per-iteration noise from `iteration_seed` (base_seed + current_iteration). |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| noise_seed | INT | Subgraph input slot 13 (noise_seed from AudioLoopController) | -- |

**Output:** NOISE to Internal Node 573.

---

### Internal Node 573 -- SamplerCustomAdvanced (Loop)

| Field | Value |
|-------|-------|
| **Type** | `SamplerCustomAdvanced` |
| **Source** | ComfyUI core |
| **What it does** | Runs the denoising pass for this loop iteration. Generates 497 new frames conditioned on the overlap tail, init image guide, and audio. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| noise | NOISE | Internal Node 574 | -- |
| guider | GUIDER | Internal Node 644 | -- |
| sampler | SAMPLER | Subgraph input slot 0 | -- |
| sigmas | SIGMAS | Subgraph input slot 1 | -- |
| latent_image | LATENT | Internal Node 583 (AV latent) | -- |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| output | LATENT | Internal Node 596 (SeparateAV) |
| denoised_output | LATENT | **unwired** |

---

### Internal Node 596 -- LTXVSeparateAVLatent (Loop)

Splits AV latent. Video latent to Node 655 (CropGuides).

---

### Internal Node 655 -- LTXVCropGuides (Loop)

| Field | Value |
|-------|-------|
| **Type** | `LTXVCropGuides` |
| **Source** | ComfyUI core |
| **What it does** | Removes appended guide frames from the sampled latent, leaving only the 63-frame generated video latent. |

**Inputs:** positive/negative from Node 1519, latent from Node 596 video_latent.
**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| positive | CONDITIONING | **unwired** |
| negative | CONDITIONING | **unwired** |
| latent | LATENT | Internal Node 1521 (VAEDecode) |

---

### Internal Node 1521 -- VAEDecode (Loop)

| Field | Value |
|-------|-------|
| **Type** | `VAEDecode` |
| **Source** | ComfyUI core |
| **What it does** | Decodes the guide-cropped video latent back to pixel space (497 frames). |

**Inputs:** LATENT from Node 655, VAE from subgraph input slot 3.
**Output:** IMAGE to Internal Node 1504.

---

### Internal Node 1504 -- GetImageSizeAndCount

| Field | Value |
|-------|-------|
| **Type** | `GetImageSizeAndCount` |
| **Source** | KJNodes (`/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/image_nodes.py`) |
| **What it does** | Passes the image through while displaying its dimensions. Debugging/verification node. |

**Output:** IMAGE to Internal Node 1509.

---

### Internal Node 1509 -- GetImageRangeFromBatch (Trim Overlap from Output)

| Field | Value |
|-------|-------|
| **Type** | `GetImageRangeFromBatch` |
| **Source** | KJNodes |
| **What it does** | Trims the first `overlap` (25) frames from the generated output. These frames overlap with the previous iteration's tail and would cause duplication if kept. `start_index` = overlap value, `num_frames` = 4096 (effectively "rest of batch"). |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| images | IMAGE | Internal Node 1504 | -- |
| start_index | INT | Subgraph input slot 14 (overlap) | `25` |
| num_frames | INT (widget) | -- | `4096` (all remaining) |

**Output:** IMAGE to subgraph output slot 0 (extended_images) -> Node 1540 (TensorLoopClose).

---

### Internal Node 1584 -- GetImageRangeFromBatch (Unused)

| Field | Value |
|-------|-------|
| **Type** | `GetImageRangeFromBatch` |
| **What it does** | Present but has no image input connected (link null). Receives `overlap` as num_frames. This is a vestigial node from an earlier workflow version. |

---

## Stage 9: Loop Close

### Node 1540 -- TensorLoopClose

| Field | Value |
|-------|-------|
| **Type** | `TensorLoopClose` |
| **Source** | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-NativeLooping_testing/nodes.py` |
| **What it does** | Receives each iteration's output and accumulates it. When `should_stop` is True, the loop terminates after the current iteration completes. The `accumulate=true` setting concatenates all iteration outputs into a single IMAGE batch. Overlap is set to `disabled` because the subgraph already trims overlap frames internally. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| flow_control | FLOW_CONTROL | Node 1539 (TensorLoopOpen) | -- |
| processed | IMAGE | Node 843 (extension subgraph) extended_images | -- |
| accumulate | BOOLEAN (widget) | -- | `true` |
| overlap | DynamicCombo (widget) | -- | `disabled` |
| stop | BOOLEAN | Node 1582 (AudioLoopController) should_stop | -- |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| output | IMAGE | Node 1507 (ImageBatch) image2 |

---

## Stage 10: Output Assembly

### Node 1507 -- ImageBatch

| Field | Value |
|-------|-------|
| **Type** | `ImageBatch` |
| **Source** | ComfyUI core |
| **What it does** | Concatenates the initial render (image1) with all loop iterations (image2) into a single continuous image batch. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| image1 | IMAGE | Node 618 (Reroute, initial render) | -- |
| image2 | IMAGE | Node 1540 (TensorLoopClose) output | -- |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| IMAGE | IMAGE | Node 617 (VHS_VideoCombine), Node 1590 (VAEEncode for upscale, BYPASSED) |

---

### Node 604 -- GetNode (Get_orig_audio)

Retrieves `orig_audio` (the full, non-separated trimmed audio) for the final video mux.

---

### Node 617 -- VHS_VideoCombine (Final Output)

| Field | Value |
|-------|-------|
| **Type** | `VHS_VideoCombine` |
| **Source** | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/` |
| **What it does** | Encodes the full image batch + original audio into an H.264 MP4 video file. `trim_to_audio=true` trims the video to match audio duration. |

**Inputs:**

| Input | Type | Source | Value |
|-------|------|--------|-------|
| images | IMAGE | Node 1507 (ImageBatch) | -- |
| audio | AUDIO | Node 604 (Get_orig_audio) | -- |
| meta_batch | VHS_BatchManager | -- | **unwired** |
| vae | VAE | -- | **unwired** |

**Widget values:**

| Widget | Value |
|--------|-------|
| frame_rate | `25` |
| loop_count | `0` |
| filename_prefix | `LTX-2` |
| format | `video/h264-mp4` |
| pix_fmt | `yuv420p` |
| crf | `19` |
| save_metadata | `true` |
| trim_to_audio | `true` |
| save_output | `true` |

**Outputs:**

| Output | Type | Connects to |
|--------|------|-------------|
| Filenames | VHS_FILENAMES | **unwired** |

---

## Stage 11: Post-Loop Upscale Chain (ALL BYPASSED, mode=4)

These nodes form a spatial upscale pipeline that is present but disabled by default. They would encode the combined output, upscale 2x in latent space, and decode tiled.

### Node 1589 -- LatentUpscaleModelLoader (BYPASSED)

Loads `ltx-2.3-spatial-upscaler-x2-1.0.safetensors`.

### Node 1598 -- GetNode (Get_video_vae) (BYPASSED)

Retrieves video VAE for encoding.

### Node 1590 -- VAEEncode (BYPASSED)

Encodes combined images to latent. Input: Node 1507 (ImageBatch).

### Node 1591 -- LTXVLatentUpsampler (BYPASSED)

Spatial 2x upscale in latent space. Input: Node 1590 latent + Node 1589 upscale model + video_vae.

### Node 1597 -- VAEDecodeTiled (BYPASSED)

Tiled decode of upscaled latent. tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8.
Output would feed into Node 617 instead of Node 1507 if enabled.

---

## User-Configurable Values

### Primary Settings (most commonly changed)

| Widget | Node | Default | What it controls | Valid range |
|--------|------|---------|------------------|-------------|
| `audio` (file) | 565 LoadAudio | `example_audio.mp4` | Source audio file | Any audio file in input/ |
| `image` (file) | 444 LoadImage | `reference_image.png` | Init/reference image | Any image in input/ |
| `start_index` | 567 TrimAudioDuration | `5` | Seconds to skip at song start (instrumental intro) | 0 - audio length |
| `duration` | 567 TrimAudioDuration | `300` | Max audio duration in seconds | 0.01+ |
| `text` (positive) | 169 CLIPTextEncode | `"video of a woman passionately singing alone"` | Positive prompt for generation | Free text |
| `text` (negative) | 507 CLIPTextEncode | `"still image with no motion, subtitles..."` | Negative prompt for NAG guidance | Free text |
| `overlap_seconds` | 1582 AudioLoopController | `1.0` | Overlap between consecutive windows in seconds; controls continuity | 0.0 - window_seconds |
| `seed` | 1527 INTConstant | `42` | Base seed for reproducibility | 0 - 2^64 |
| `iterations` | 1539 TensorLoopOpen | `50` | Max loop iterations (AudioLoopController stops early) | 0 - 10000 |

### Video/Sampling Settings

| Widget | Node | Default | What it controls | Valid range |
|--------|------|---------|------------------|-------------|
| `width` | 445 ImageResizeKJv2 | `832` | Output video width in pixels | Multiple of 32 |
| `height` | 445 ImageResizeKJv2 | `480` | Output video height in pixels | Multiple of 32 |
| `length` | 526 PrimitiveNode | `497` | Frames per window (497 = 8*62+1) | 8n+1 values |
| `window_size_seconds` | 688 FloatConstant | `19.88` | Duration of each generation window | Must match length/fps |
| `sampler_name` | 154 KSamplerSelect | `euler` | Sampling algorithm | euler, euler_ancestral, dpm_2, etc. |
| `scheduler` | 1421 BasicScheduler | `linear_quadratic` | Sigma schedule type | linear, linear_quadratic, etc. |
| `steps` | 1421 BasicScheduler | `8` | Denoising steps per window | 1 - 100 |
| `shift` | 1513 ModelSamplingSD3 | `13` | Noise schedule shift | 0 - 100 |
| `first_frame_guide_strength` | 1269 FloatConstant | `1.0` | Init image guide denoise strength (1.0 = no noise on guide) | 0.0 - 1.0 |
| `img_compression` | 446 LTXVPreprocess | `0` | JPEG-like preprocessing on init image | 0 - 100 |

### NAG (Normalized Attention Guidance) Settings

| Widget | Node | Default | What it controls | Valid range |
|--------|------|---------|------------------|-------------|
| `nag_scale` | 508 LTX2_NAG | `11.0` | Strength of negative attention guidance | 0.0 - 100.0 |
| `nag_alpha` | 508 LTX2_NAG | `0.25` | Balance between normalized guided and original representation | 0.0 - 1.0 |
| `nag_tau` | 508 LTX2_NAG | `2.5` | Clipping threshold for attention deviation | 0.0 - 10.0 |

### Model Optimization Settings

| Widget | Node | Default | What it controls | Valid range |
|--------|------|---------|------------------|-------------|
| `chunks` | 504 LTXVChunkFeedForward | `2` | FF activation chunks for VRAM reduction | 1 - 100 |
| `dim_threshold` | 504 LTXVChunkFeedForward | `4096` | Apply chunking above this dimension | 0 - 16384 |
| `preview_rate` | 503 LTX2SamplingPreviewOverride | `8` | Preview frame rate during sampling | 1 - 60 |

### Prompt Scheduling (when TimestampPromptSchedule is enabled)

| Widget | Node | Default | What it controls | Valid range |
|--------|------|---------|------------------|-------------|
| `schedule` | 1558 TimestampPromptSchedule | `"0:00+: ..."` | Timestamp-to-prompt mapping for song sections | See format below |
| `blend_seconds` | 1558 TimestampPromptSchedule | `0.0` | Transition blend duration between prompts | 0.0+ |

Schedule format:
```
0:00-0:38: prompt for verse
0:38-1:15: prompt for chorus
1:15+: prompt from here onward
```

### Output Settings

| Widget | Node | Default | What it controls | Valid range |
|--------|------|---------|------------------|-------------|
| `frame_rate` | 617 VHS_VideoCombine | `25` | Output video framerate | 1 - 120 |
| `crf` | 617 VHS_VideoCombine | `19` | H.264 quality (lower = higher quality) | 0 - 51 |
| `filename_prefix` | 617 VHS_VideoCombine | `LTX-2` | Output filename prefix | Any string |
| `trim_to_audio` | 617 VHS_VideoCombine | `true` | Trim video to match audio length | true/false |

---

## Custom Node Source Locations

| Node Package | Source Path |
|--------------|-------------|
| AudioLoopHelper (AudioLoopController, TimestampPromptSchedule, ConditioningBlend, AudioLoopPlanner, AudioDuration) | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-AudioLoopHelper/nodes.py` |
| NativeLooping (TensorLoopOpen, TensorLoopClose) | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-NativeLooping_testing/nodes.py` |
| KJNodes: LTX2_NAG, LTXVChunkFeedForward, LTX2SamplingPreviewOverride, LTX2AttentionTunerPatch, LTXVImgToVideoInplaceKJ, LTXVAudioVideoMask | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py` |
| KJNodes: GetImageRangeFromBatch, GetImageSizeAndCount | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/image_nodes.py` |
| KJNodes: PathchSageAttentionKJ | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/model_optimization_nodes.py` |
| KJNodes: SetNode, GetNode, FloatConstant, INTConstant, ImageResizeKJv2, VisualizeSigmasKJ, VAELoaderKJ | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-KJNodes/` (various files) |
| LTXVideo: LTXVAddLatentGuide | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-LTXVideo/latents.py` |
| VideoHelperSuite: VHS_VideoCombine | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/` |
| MelBandRoFormer: MelBandRoFormerModelLoader, MelBandRoFormerSampler | `/home/fbliss/ComfyUI/custom_nodes/ComfyUI-MelBandRoFormer/nodes.py` |
| ComfyUI core: UNETLoader, DualCLIPLoader, CLIPTextEncode, LTXVConditioning, LTXVPreprocess, LTXVConcatAVLatent, LTXVSeparateAVLatent, LTXVCropGuides, EmptyLTXVLatentVideo, LTXVAudioVAEEncode, TrimAudioDuration, LoadAudio, LoadImage, VAEEncode, VAEDecode, CFGGuider, SamplerCustomAdvanced, KSamplerSelect, BasicScheduler, ModelSamplingSD3, RandomNoise, ConditioningZeroOut, SetLatentNoiseMask, SolidMask, ImageBatch, PrimitiveNode, Reroute, PreviewImage, PreviewAny, Note | `/home/fbliss/ComfyUI/comfy_extras/nodes_lt.py` and `/home/fbliss/ComfyUI/nodes.py` |

---

## Complete Node Index

| Node ID | Type | Display Name / Title | Stage |
|---------|------|---------------------|-------|
| 414 | UNETLoader | -- | Model Loading |
| 1537 | VAELoaderKJ | -- | Model Loading |
| 1538 | VAELoaderKJ | -- | Model Loading |
| 416 | DualCLIPLoader | -- | Model Loading |
| 228 | SetNode | Set_video_vae | Model Loading |
| 252 | SetNode | Set_audio_vae | Model Loading |
| 444 | LoadImage | -- | Image Prep |
| 445 | ImageResizeKJv2 | -- | Image Prep |
| 446 | LTXVPreprocess | -- | Image Prep |
| 650 | SetNode | Set_input_image | Image Prep |
| 526 | PrimitiveNode | length | Latent Construction |
| 344 | EmptyLTXVLatentVideo | -- | Latent Construction |
| 413 | GetNode | Get_video_vae | Latent Construction |
| 531 | LTXVImgToVideoInplaceKJ | -- | Latent Construction |
| 565 | LoadAudio | -- | Audio Prep |
| 567 | TrimAudioDuration | -- (intro trim) | Audio Prep |
| 568 | MelBandRoFormerModelLoader | -- | Audio Prep |
| 569 | MelBandRoFormerSampler | -- | Audio Prep |
| 640 | SetNode | Set_actual_audio | Audio Prep |
| 581 | SetNode | Set_orig_audio | Audio Prep |
| 688 | FloatConstant | window_size_seconds | Audio Prep |
| 689 | SetNode | Set_window_size_seconds | Audio Prep |
| 601 | TrimAudioDuration | -- (window trim) | Audio Prep |
| 254 | GetNode | Get_audio_vae | Audio Prep |
| 566 | LTXVAudioVAEEncode | -- | Audio Prep |
| 571 | SolidMask | -- | Audio Prep |
| 570 | SetLatentNoiseMask | -- | Audio Prep |
| 350 | LTXVConcatAVLatent | -- | Latent Construction |
| 169 | CLIPTextEncode | -- (positive) | Text Encoding |
| 507 | CLIPTextEncode | -- (negative) | Text Encoding |
| 420 | ConditioningZeroOut | -- | Text Encoding |
| 164 | LTXVConditioning | -- (initial render) | Text Encoding |
| 645 | SetNode | Set_base_cond_pos | Text Encoding |
| 646 | SetNode | Set_base_cond_neg | Text Encoding |
| 268 | PathchSageAttentionKJ | -- (BYPASSED) | Model Pipeline |
| 504 | LTXVChunkFeedForward | -- | Model Pipeline |
| 1523 | LTX2AttentionTunerPatch | -- | Model Pipeline |
| 508 | LTX2_NAG | -- | Model Pipeline |
| 503 | LTX2SamplingPreviewOverride | -- | Model Pipeline |
| 572 | SetNode | Set_model | Model Pipeline |
| 1513 | ModelSamplingSD3 | -- | Model Pipeline |
| 1527 | INTConstant | start_seed | Initial Render |
| 1528 | SetNode | Set_start_seed | Initial Render |
| 1530 | GetNode | Get_start_seed | Initial Render |
| 1322 | RandomNoise | -- | Initial Render |
| 153 | CFGGuider | -- (initial) | Initial Render |
| 154 | KSamplerSelect | -- | Initial Render |
| 575 | SetNode | Set_guider | Initial Render |
| 576 | SetNode | Set_sampler | Initial Render |
| 1421 | BasicScheduler | -- | Initial Render |
| 1422 | VisualizeSigmasKJ | -- | Initial Render |
| 579 | SetNode | Set_sigmas | Initial Render |
| 1269 | FloatConstant | -- (guide strength) | Initial Render |
| 1271 | SetNode | Set_first_frame_guide_strength | Initial Render |
| 161 | SamplerCustomAdvanced | -- (initial) | Initial Render |
| 245 | LTXVSeparateAVLatent | -- | Initial Render |
| 381 | LTXVCropGuides | -- | Initial Render |
| 236 | GetNode | Get_video_vae | Initial Render |
| 1318 | VAEDecode | -- | Initial Render |
| 560 | VHS_VideoCombine | -- (BYPASSED preview) | Initial Render |
| 582 | GetNode | Get_orig_audio | Initial Render |
| 618 | Reroute | -- | Initial Render |
| 1423 | PreviewImage | -- (sigma preview) | Initial Render |
| 1539 | TensorLoopOpen | -- | Loop Setup |
| 1582 | AudioLoopController | -- | Loop Body |
| 1558 | TimestampPromptSchedule | -- (BYPASSED) | Loop Body |
| 1559 | CLIPTextEncode | Loop Prompt Encode | Loop Body |
| 1588 | GetNode | Get_base_cond_pos (Static) | Loop Body |
| 648 | GetNode | Get_base_cond_neg | Loop Body |
| 1587 | LTXVConditioning | Loop LTXVConditioning | Loop Body |
| 1560 | AudioLoopPlanner | -- (BYPASSED) | Loop Planner |
| 1563 | PreviewAny | Iteration Timestamps | Loop Planner |
| 1586 | PreviewAny | -- (overlap preview) | Loop Body |
| 578 | GetNode | Get_sampler | Loop Body |
| 580 | GetNode | Get_sigmas | Loop Body |
| 654 | GetNode | Get_model | Loop Body |
| 619 | GetNode | Get_video_vae | Loop Body |
| 651 | GetNode | Get_input_image | Loop Body |
| 599 | GetNode | Get_audio_vae | Loop Body |
| 641 | GetNode | Get_actual_audio | Loop Body |
| 1529 | GetNode | Get_start_seed | Loop Body |
| 691 | GetNode | Get_window_size_seconds | Loop Body |
| 1273 | GetNode | Get_first_frame_guide_strength | Loop Body |
| 647 | GetNode | Get_base_cond_pos | Loop Body (unused) |
| 843 | Subgraph (extension) | extension | Loop Body |
| 1540 | TensorLoopClose | -- | Loop Close |
| 1507 | ImageBatch | -- | Output Assembly |
| 604 | GetNode | Get_orig_audio | Output Assembly |
| 617 | VHS_VideoCombine | -- (final output) | Output Assembly |
| 1585 | Note | -- (prompt examples) | Documentation |
| 1533 | Note | -- (MelBand note) | Documentation |
| 1589 | LatentUpscaleModelLoader | -- (BYPASSED) | Upscale Chain |
| 1598 | GetNode | Get_video_vae (BYPASSED) | Upscale Chain |
| 1590 | VAEEncode | Encode for Upscale (BYPASSED) | Upscale Chain |
| 1591 | LTXVLatentUpsampler | Spatial Upscale 2x (BYPASSED) | Upscale Chain |
| 1597 | VAEDecodeTiled | Decode Upscaled (BYPASSED) | Upscale Chain |

### Extension Subgraph #843 Internal Nodes

| Internal ID | Type | Purpose |
|-------------|------|---------|
| 615 | GetImageRangeFromBatch | Extract overlap tail from previous_images |
| 614 | VAEEncode | Encode overlap tail to latent |
| 600 | TrimAudioDuration | Trim audio for this iteration's window |
| 598 | LTXVAudioVAEEncode | Encode iteration audio to latent |
| 606 | LTXVAudioVideoMask | Create video/audio noise masks |
| 1520 | VAEEncode | Encode init image for guide |
| 1519 | LTXVAddLatentGuide | Merge init image guide into latent + conditioning |
| 583 | LTXVConcatAVLatent | Combine video + audio latents |
| 644 | CFGGuider | Package model + conditioning for sampling |
| 574 | RandomNoise | Per-iteration noise from iteration_seed |
| 573 | SamplerCustomAdvanced | Run denoising pass |
| 596 | LTXVSeparateAVLatent | Split AV latent |
| 655 | LTXVCropGuides | Remove guide frames from output |
| 1521 | VAEDecode | Decode video latent to pixels |
| 1504 | GetImageSizeAndCount | Debug: show output dimensions |
| 1509 | GetImageRangeFromBatch | Trim overlap from output start |
| 1584 | GetImageRangeFromBatch | Vestigial, no input connected |
