Last updated: 2026-04-12

# LTX2_NAG -- Normalized Attention Guidance Technical Reference

## What is NAG?

NAG (Normalized Attention Guidance) is a technique that enables effective negative prompting on diffusion transformer models. It modifies **cross-attention layers** during inference to guide generation away from undesired content while maintaining coherence. Think of it as CFG but operating at the attention level rather than the noise prediction level.

Reference: https://github.com/ChenDarYen/Normalized-Attention-Guidance

## Source Files

- **Node implementation:** `ComfyUI-KJNodes/nodes/ltxv_nodes.py` (lines 336-521)
- **Generic NAG (non-LTX):** `ComfyUI/comfy_extras/nodes_nag.py`

## Parameters

### nag_scale (default: 11.0, range: 0-100)

Strength of the negative guidance effect. The core guidance formula is:

```
guided = x_positive * nag_scale - x_negative * (nag_scale - 1)
```

With `nag_scale = 11`: `guided = 11 * x_pos - 10 * x_neg`

Higher values push output further from the negative conditioning. Setting to 0 disables NAG entirely.

### nag_alpha (default: 0.25, range: 0-1)

Mixing coefficient between the NAG-guided result and the original positive attention:

```
final = guided_normalized * alpha + x_positive * (1 - alpha)
```

- 0.0 = No NAG effect (pure positive attention)
- 0.25 = 25% NAG, 75% original (typical)
- 1.0 = Full NAG guidance

### nag_tau (default: 2.5, range: 0-10)

Clipping threshold that prevents guidance from growing unbounded. Regularizes the guided signal by comparing its norm to the positive signal's norm:

```
scale = norm(guided) / norm(positive)
if scale > tau:
    guided *= (norm_positive * tau) / (norm_guided + epsilon)
```

Prevents extreme artifacts while allowing meaningful guidance.

### inplace (default: True)

Memory optimization flag. When True, uses in-place tensor operations (lower VRAM, slightly different floating-point rounding). When False, standard out-of-place operations.

### nag_cond_video (CONDITIONING)

The negative conditioning for video. This is the encoded text describing what to AVOID. Example from the workflow:

```
"still image with no motion, subtitles, text, scene change, instruments, violin"
```

### nag_cond_audio (CONDITIONING, optional)

Same concept but for audio modality. Only used if the model has audio capability.

## How it Works

### Step 1: Dual Attention
For each cross-attention layer in the transformer, compute attention with both positive and negative conditioning contexts (same query, different key/value from different prompts).

### Step 2: Guidance Calculation
```
guided = x_positive * nag_scale - x_negative * (nag_scale - 1)
```

### Step 3: Norm Regularization (tau clipping)
```
norm_pos = L1_norm(x_positive)
norm_guided = L1_norm(guided)
if norm_guided/norm_pos > tau:
    guided *= (norm_pos * tau) / norm_guided
```

### Step 4: Alpha Blending
```
final = guided * alpha + x_positive * (1 - alpha)
```

## How it Patches the Model

NAG works by patching the model's cross-attention (`attn2`) layers. It does NOT patch self-attention (`attn1`). For each transformer block:

```python
model.add_object_patch(
    f"diffusion_model.transformer_blocks.{idx}.attn2.forward",
    patched_forward_function
)
```

The patched forward function runs attention twice (with positive and negative contexts) and applies the guidance formula above.

For audio-capable models, it also patches `audio_attn2` layers separately.

## Why the NAG Prompt Matters for Video Coherence

The negative NAG prompt in our workflow:
```
"still image with no motion, subtitles, text, scene change, instruments, violin"
```

Each term addresses a specific failure mode:
- **"still image with no motion"** -- prevents static/frozen frames
- **"subtitles, text"** -- avoids text artifact generation
- **"scene change"** -- prevents abrupt visual transitions that break continuity
- **"instruments, violin"** -- prevents objects from appearing that would occlude the subject's face (once something blocks the face, the model reconstructs a different face on the next frame, cascading into decoherence)

The goal is: maintain realistic motion and expressions, but prevent anything that would cause the model to lose track of the subject's identity across frames.

## LTX2_NAG vs Generic NAGuidance

| Aspect | LTX2_NAG | NAGuidance (generic) |
|--------|----------|---------------------|
| Target | Cross-attention (attn2) | Self-attention output (attn1) |
| Models | LTX Video 2 only | Flux, Schnell, etc. |
| Audio | Yes (separate audio_attn2) | No |
| Default scale | 11.0 | 5.0 |
| Default alpha | 0.25 | 0.5 |
| Default tau | 2.5 | 1.5 |

## Context Preprocessing

Before patching, the node preprocesses the negative conditioning through the model's own projection layers:

1. Extract conditioning tensor, move to model device/dtype
2. Split video/audio dimensions if combined
3. Run through `caption_projection` (if model has it)
4. Run through `video_embeddings_connector` (model-specific)
5. Reshape to model's expected attention dimension

This ensures the negative conditioning is in the same representational space as the positive conditioning the model will use during inference.
