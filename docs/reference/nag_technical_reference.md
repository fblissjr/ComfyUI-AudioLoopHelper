Last updated: 2026-05-25

# LTX2_NAG -- Normalized Attention Guidance Technical Reference

## What is NAG?

NAG (Normalized Attention Guidance) is a technique that enables effective negative prompting on diffusion transformer models. It modifies **cross-attention layers** during inference to guide generation away from undesired content while maintaining coherence. Think of it as CFG but operating at the attention level rather than the noise prediction level.

Reference: https://github.com/ChenDarYen/Normalized-Attention-Guidance

## Source Files

- **Node implementation:** `ComfyUI-KJNodes/nodes/ltxv_nodes.py` (LTX2_NAG class at line 442, LTXVCrossAttentionPatch at 422, kernels at 336-419)
- **Generic NAG (non-LTX):** `ComfyUI/comfy_extras/nodes_nag.py` (patches `attn1_output`; different target)
- **ComfyUI patch plumbing:** `ComfyUI/comfy/model_patcher.py` (object_patches lifecycle — see "How it Patches the Model")

## Parameters

### nag_scale

Strength of the negative guidance effect. The core guidance formula (`ltxv_nodes.py:349-353`):

```
guided = x_positive * nag_scale - x_negative * (nag_scale - 1)
```

With `nag_scale = 11`: `guided = 11 * x_pos - 10 * x_neg`. Higher values push output further from the negative conditioning. Setting to 0 disables NAG entirely and returns the unpatched model (`ltxv_nodes.py:468-469`).

**KJNodes default:** 11.0. Range 0-100. This default is tuned for the full 22B LTX 2.3 model.

**For the merged distilled-1.1 checkpoint (our default):** scale=11 is aggressive. Combined with a negative like `still image with no motion, deformed ..., duplicate character`, the attention-level extrapolation can push activations out-of-distribution enough that the sampler produces zero-update steps — i.e. the initial latent never denoises and you see a frozen first render. **Start at nag_scale=5** and A/B test 3-7. If you hit a stall at sigma ~= 0.99+ or the output freezes, lower the scale before adjusting anything else.

### nag_alpha

Mixing coefficient between the NAG-guided result and the original positive attention (`ltxv_nodes.py:371-374`):

```
final = guided_normalized * alpha + x_positive * (1 - alpha)
```

- 0.0 = no NAG effect (pure positive attention)
- 0.25 = 25% NAG, 75% original (KJNodes default, typical working value)
- 1.0 = full NAG guidance

Default 0.25. Range 0-1.

### nag_tau

Clipping threshold that prevents guidance from growing unbounded. Compares the L1 norm of the guided signal to the positive signal's norm (`ltxv_nodes.py:357-369`):

```
norm_pos     = L1_norm(x_positive, dim=-1)
norm_guided  = L1_norm(nag_guidance, dim=-1)
scale        = norm_guided / norm_pos
mask         = scale > tau
adjustment   = (norm_pos * tau) / (norm_guided + 1e-7)
nag_guidance *= where(mask, adjustment, 1.0)
```

Only token positions where the guided norm exceeds `tau * norm_pos` get clipped back. Prevents extreme artifacts while allowing meaningful guidance. Default 2.5. Range 0-10.

### inplace

Memory optimization flag. When True, uses in-place tensor ops (`mul_`, `neg_`, `add_`) in the guidance math and the alpha blend (`ltxv_nodes.py:350-354, 371-374`). Lower VRAM, slightly different floating-point rounding. When False, standard out-of-place operations. Default True.

### nag_cond_video / nag_cond_audio

The negative CONDITIONING for each modality. These are CLIP-encoded embeddings describing what to AVOID — NOT raw strings. Example of the source text encoded into `nag_cond_video`:

```
"still image with no motion, subtitles, deformed facial features,
 extra limbs, disfigured hands, duplicate character, twin, clone"
```

`nag_cond_video` patches the video cross-attention (`attn2`); `nag_cond_audio` independently patches the audio cross-attention (`audio_attn2`) if the model has audio capability. See "Audio_attn2 for AV-capable models" below.

### Operational constraint: CLIP must not enter the loop body

**Rule.** When LTX2_NAG is active inside a TensorLoop, pre-encode the prompt schedule OUTSIDE the loop with `TimestampPromptScheduleBatchEncode` and index per-iteration via `ConditioningSelectByIteration` (both in `nodes.py`). Do NOT place `CLIPTextEncode` or the legacy `CachedTextEncode` inside the loop body.

**Why.** NAG stores its negative conditioning tensor inside a Python closure registered on ComfyUI's `ModelPatcher.object_patches`. ComfyUI's model-offload path migrates `transformer_options["patches"]` tensors to the right device but does **not** migrate `object_patches` closures. If CLIP encoding runs per-iteration, it can trigger `load_models_gpu -> free_memory -> model.detach(unpatch_weights=True)` on the DiT; when the DiT reloads, NAG's captured `nag_cond_video` tensor points at a device layout that no longer matches, and NAG silently disengages from iteration 2 onward. Every class the negative was suppressing (microphones, "still image", deformed hands, duplicate characters) leaks back simultaneously.

Full forensic root-cause with exact line references: `docs/analysis/nag_object_patches_offload_asymmetry.md`.

## How NAG Works (Conceptual)

### Step 1: Dual Attention
For each cross-attention layer, compute attention twice with the same query but different key/value contexts (one from the positive prompt, one from the captured negative prompt). Both results have shape `[batch, seq_len, dim]`.

### Step 2: Guidance Calculation
```
guided = x_positive * nag_scale - x_negative * (nag_scale - 1)
```

### Step 3: Norm Regularization (tau clipping)
```
norm_pos    = L1_norm(x_positive)
norm_guided = L1_norm(guided)
if norm_guided/norm_pos > tau:
    guided *= (norm_pos * tau) / norm_guided
```

### Step 4: Alpha Blending
```
final = guided * alpha + x_positive * (1 - alpha)
```

The final tensor is what the cross-attention block returns to the transformer.

## NAG's Cross-Attention Mechanism

Transformer blocks in LTX 2.3 have two attention layers:

- **attn1 (self-attention):** query, key, and value all come from the latent tokens themselves. It learns spatial/temporal coherence across the latent. Text conditioning never appears here.
- **attn2 (cross-attention):** query comes from the latent tokens; key and value come from projected text conditioning. This is where text exerts its guiding pull on the image features.

NAG patches `attn2` because that is the only layer where text conditioning is actually wired in. Patching `attn1` would do nothing to text guidance — the layer doesn't see text at all. Generic `NAGuidance` in `comfy_extras/nodes_nag.py:84` takes a different tack and uses `set_model_attn1_output_patch` because it's designed for models where post-processing of the self-attention output matters; that path does not apply to LTX's dual-attention structure.

The patched cross-attention forward is `ltxv_crossattn_forward_nag` (`ltxv_nodes.py:380-419`). Per call it:

1. Computes `q_pos = q_norm(to_q(x_positive))` once (line 391).
2. Calls `nag_attention(...)` which computes `optimized_attention(q, K_pos, V_pos)` and `optimized_attention(q, K_neg, V_neg)` — same query, two different contexts (lines 344-347).
3. Runs the 4-step guidance math (`normalized_attention_guidance`, lines 349-377).
4. If the batch includes a separate unconditional pass (for CFG > 1), runs that branch unmodified so NAG and CFG compose cleanly (lines 400-409).

## How NAG Captures Negative Conditioning

The capture pattern is a Python descriptor-based closure. It runs ONCE at node execution and holds for the entire generation.

Step-by-step (`ltxv_nodes.py:467-521`):

1. **Move negative embedding to GPU.** `context_video = nag_cond_video[0][0].to(device, dtype)` (line 486). `device` is `mm.get_torch_device()` (target GPU); `dtype` is `model.model.manual_cast_dtype` falling back to `diffusion_model.dtype`.
2. **Preprocess through model-specific projections.** Split the combined video/audio dim if needed (lines 487-489), run through `caption_projection` when the model has a first-linear caption projector (490-493), run through `video_embeddings_connector` (494-497). The projection modules are moved to GPU just for this forward, then back to offload device. Final shape: `[1, seq_len, inner_dim]` (line 498).
3. **Wrap in descriptor.** `LTXVCrossAttentionPatch(context_video, nag_scale, nag_alpha, nag_tau, inplace=inplace)` stores the tensor and params in `self.nag_context` etc. (`ltxv_nodes.py:422-428`).
4. **Bind to each transformer block.** `.__get__(block.attn2, block.__class__)` returns a `types.MethodType` closure (`ltxv_nodes.py:430-440`). The closure body is `wrapped_attention`: it sets `self_module.nag_context = self.nag_context` (and the scale/alpha/tau/inplace params) on the attention module, then calls `ltxv_crossattn_forward_nag(self_module, *args, **kwargs)`.
5. **Register as object_patch.** `model_clone.add_object_patch(f"diffusion_model.transformer_blocks.{idx}.attn2.forward", patched_attn2)` (line 501). One patch per transformer block, all sharing references to the same captured tensor (the clone's object_patches dict is a shallow copy — `model_patcher.py:339`).

The same tensor reference is reused across every sigma step and every transformer block. NAG never re-encodes text, never rebuilds the descriptor, never moves the tensor. That is its efficiency but also the source of the device-migration hazard — once the captured tensor exists, the only code that would move it to a different device is `model_patches_to()`, which doesn't walk `object_patches` (`model_patcher.py:561-580`).

### Tensor shape, dtype, device, lifetime

| Property | Value | Source |
|---|---|---|
| Shape | `[1, seq_len, inner_dim]` | `ltxv_nodes.py:498` (`context_video.view(1, -1, img_dim)`) |
| dtype | `model.model.manual_cast_dtype` (fallback `diffusion_model.dtype`) | `ltxv_nodes.py:473-475` |
| Device | GPU (`mm.get_torch_device()`) | `ltxv_nodes.py:471, 486` |
| Created | Once, at LTX2_NAG node execution | `execute()` at 467 |
| Reused | Every sigma step x every transformer block x every sampler run until the patch is removed | Closure reference shared via `add_object_patch` |
| Moved | **Never** by ComfyUI's offload path; `model_patches_to` ignores `object_patches` | `model_patcher.py:561-580` |

### Audio_attn2 for AV-capable models

LTX 2.3 is AV-capable (`diffusion_model.audio_caption_projection is not None`). When `nag_cond_audio` is supplied, an independent capture-and-patch pass runs for audio cross-attention (`ltxv_nodes.py:503-519`):

- Separate `context_audio` tensor, optionally split off from a combined embedding at `vid_split:` (line 507).
- Separate projection path: `audio_caption_projection` -> `audio_embeddings_connector` (lines 508-515).
- Final reshape: `context_audio.view(1, -1, audio_dim)` (line 516).
- Registered as `object_patch` on `diffusion_model.transformer_blocks.{idx}.audio_attn2.forward` for every block (line 519).

The two modalities run the same guidance math via the same `LTXVCrossAttentionPatch` class — audio just uses a different captured tensor and targets a different attention layer. You can enable one, both, or neither.

## NAG and CFG Composition

NAG and CFG operate at **different layers** of the denoising pipeline.

- **CFG:** applied at the noise-prediction layer AFTER the DiT forward pass. Blends: `noise_pred = uncond + CFG * (cond - uncond)`. Requires running the DiT twice per step (once with positive conditioning, once with negative).
- **NAG:** applied INSIDE each transformer block's cross-attention, BEFORE the noise prediction is computed. Modifies the intermediate features the DiT produces. Does not require a separate DiT pass — the negative context is wired directly into the attention op.

Both can be active simultaneously. NAG modulates intermediate cross-attention features; CFG modulates final noise residuals. They stack multiplicatively in effect but not in compute: with NAG + CFG=2, you still run the DiT twice (for CFG's uncond/cond split), and every `attn2` call on both passes runs the NAG math.

**For distilled LTX 2.3 the default is CFG=1** (trivial — uncond branch is skipped entirely). This is deliberate: the distilled checkpoint was trained without CFG guidance, and stacking aggressive NAG on top of CFG > 1 compounds non-linearly. **If the output freezes or the sampler stalls, REDUCE nag_scale before raising CFG.** NAG-at-scale-5-to-7 with CFG=1 is the working combination on the merged distilled-1.1 checkpoint.

The STG-hybrid workflow (`example_workflows/archive/audio-loop-music-video_latent_stg.json`, archived) is the one exception where CFG is non-trivial, and there it's CFG=2 to work around an unbound-variable bug in `MultimodalGuider` when `cfg=1.0` — not because CFG=2 is better for distilled. See the STG workflow docs in CLAUDE.md.

## How it Patches the Model

NAG uses ComfyUI's `ModelPatcher.add_object_patch` mechanism (`model_patcher.py:527`). For each transformer block:

```python
model_clone.add_object_patch(
    f"diffusion_model.transformer_blocks.{idx}.attn2.forward",
    patched_forward_function
)
```

When `patch_model()` runs at sampler start, it walks `self.object_patches` and for each key calls `comfy.utils.set_attr(self.model, k, v)` to replace the attribute on the live module (`model_patcher.py:900-905`). The original attribute is stashed in `object_patches_backup` so `unpatch_model()` can restore it on cleanup (lines 952-956).

For AV models it also patches `audio_attn2.forward` on every block the same way (`ltxv_nodes.py:519`).

**Device-consistency callout.** `ModelPatcher.clone()` does a shallow `object_patches.copy()` (`model_patcher.py:339`). Every clone shares the same closure references, which means the captured negative tensor is identical across all clones downstream. `model_patches_to(device)` walks `transformer_options["patches"]` and moves its tensors to the target device (`model_patcher.py:561-576`), but it does NOT iterate `object_patches`. If the DiT is offloaded and reloaded, the captured NAG tensor stays wherever it was — so it must live on the GPU the reloaded DiT will use. That holds as long as CLIP (or any other model) doesn't evict the DiT mid-run, which is exactly the "CLIP in the loop body" hazard.

## Why the NAG Prompt Matters for Video Coherence

The negative NAG prompt wired on node 507 in our shipped `_latent.json` workflow:
```
"still image with no motion, subtitles, deformed facial features, extra limbs, disfigured hands, duplicate character, twin, clone"
```

Each term addresses a specific failure mode:
- **"still image with no motion"** — prevents static/frozen frames (a notorious LTX 2.3 failure mode on pristine init images).
- **"subtitles"** — avoids text-artifact generation.
- **"deformed facial features, extra limbs, disfigured hands"** — targets the distilled model's anatomy failure modes. These classes are what silently reappear when NAG disengages mid-loop (the 2026-04-22 regression).
- **"duplicate character, twin, clone"** — prevents the model from spawning extras in multi-person or solo scenes; combines with R4 position-anchoring in the positive prompt.

The goal is: maintain realistic motion and expressions, but prevent anything that would cause the model to lose character identity or anatomy coherence across frames. Because audio is frozen in this workflow, **music/instrument terms are deliberately absent from the negative** — negating them would fight the audio latent rather than support it.

## LTX2_NAG vs Generic NAGuidance

| Aspect | LTX2_NAG | NAGuidance (generic) |
|--------|----------|---------------------|
| Target | Cross-attention (attn2) | Self-attention output (attn1_output patch) |
| Models | LTX Video 2 only | Flux, Schnell, etc. |
| Audio | Yes (separate audio_attn2) | No |
| Default scale | 11.0 | 5.0 |
| Default alpha | 0.25 | 0.5 |
| Default tau | 2.5 | 1.5 |
| Patch mechanism | `add_object_patch` on `attn2.forward` (per-block) | `set_model_attn1_output_patch` (transformer_options) |
| Survives DiT offload? | Captured tensor does NOT migrate | Patches dict tensor DOES migrate via `model_patches_to` |

The last row is the one that bites in loop architectures: because generic `NAGuidance` lives in `transformer_options["patches"]`, ComfyUI's `model_patches_to` moves its tensors with the model. LTX2_NAG's `object_patches` closures do not get this treatment. For single-shot generation neither matters; for iterative loops where something else can evict the DiT mid-run, the asymmetry is the root cause of silent NAG failures.

## Recommended Workflow Pattern (Loop case)

For TensorLoop workflows with LTX2_NAG active (our music-video architecture):

1. **Build the schedule once, outside the loop.** `TimestampPromptScheduleBatchEncode` (nodes.py) takes `clip`, `schedule`, `stride_seconds`, `audio_duration`, `snap_boundaries` and emits a pre-encoded `conditioning_list` plus `iteration_count`. CLIP runs once for the whole schedule (deduped on unique prompt text).
2. **Select per-iteration, inside the loop.** `ConditioningSelectByIteration` indexes the list by `current_iteration`. No CLIP reference inside the loop body.
3. **Wire LTX2_NAG downstream of the selector**, before the sampler. Its `nag_cond_video` comes from a single `CLIPTextEncode` of the negative prompt — also outside the loop — and is captured once for the whole run.

This architecture guarantees CLIP loads exactly `num_unique_prompts` times per generation, the DiT never gets evicted mid-run, and NAG's captured tensor stays device-consistent. Cross-reference: `docs/analysis/nag_object_patches_offload_asymmetry.md` for the forensic trace and `scripts/apply_batch_encode_fix.py` for the migration helper.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Initial render freezes / sampler stalls at sigma ~= 0.99+ / first-step output = latent unchanged | `nag_scale` too high for the distilled checkpoint: attention-level extrapolation pushes activations OOD, producing zero-update sampler steps | Dial `nag_scale` to 3-7 (start at 5) and re-run. Verify motion returns before tuning anything else. |
| Suppressed classes (microphones, "still image with no motion", deformed hands, duplicate characters, photoreal creep from illustrated init) reappear starting at iteration 2+ | CLIP encoding inside the loop body evicted the DiT; on reload, NAG's captured `nag_cond_video` tensor's device mapping went stale (`object_patches` is not migrated by `model_patches_to`). NAG silently disengages. | Move CLIP out of the loop: `TimestampPromptScheduleBatchEncode` outside, `ConditioningSelectByIteration` inside. Full root-cause in `docs/analysis/nag_object_patches_offload_asymmetry.md`. |
