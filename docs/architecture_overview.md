Last updated: 2026-07-05

# Architecture overview — ComfyUI-AudioLoopHelper

Single-entry-point reference for understanding the full stack:
this project's workflow, the ComfyUI core it runs on, the
ComfyUI-LTXVideo and ComfyUI-KJNodes layers we depend on, the
native LTX-2 reference implementation, and the known bugs with
their current status.

Every code-behavior claim cites `path:line`. Paths starting with
`comfy/` or `comfy_extras/` refer to ComfyUI core; paths starting
with `nodes/` or similar refer to files inside the named custom-
node package.

Upstream repos referenced:

- ComfyUI — https://github.com/comfyanonymous/ComfyUI
- ComfyUI-LTXVideo — https://github.com/Lightricks/ComfyUI-LTXVideo
- ComfyUI-KJNodes — https://github.com/kijai/ComfyUI-KJNodes
- ComfyUI-NativeLooping_testing — https://github.com/kijai/ComfyUI-NativeLooping_testing
- LTX-2 (native pure-torch) — https://github.com/Lightricks/LTX-2
- LTX-Desktop (Electron app) — https://github.com/Lightricks/LTX-Desktop

## 0. TL;DR + reading order

This project generates full-length music videos with LTX 2.3 distilled-
1.1 in ComfyUI. Pipeline: init image + audio track → outer
`SamplerCustomAdvanced` render (first ~20 s) → `TensorLoop`
iterations that extend the video in latent space → final VAE decode
→ MP4.

The stack:

- **ComfyUI core** — graph executor, samplers, ModelPatcher, CLIP
  wrapper.
- **ComfyUI-LTXVideo** (Lightricks official) — guide / crop /
  preprocessing / tiled VAE decode / MultimodalGuider / STG / APG.
- **ComfyUI-KJNodes** (community) — `LTX2_NAG`, attention tuner,
  chunk-feedforward, `LTXVImgToVideoInplaceKJ`, Set/Get, sage-attn.
  **NAG lives here, not in Lightricks's code.**
- **ComfyUI-NativeLooping_testing** — `TensorLoopOpen`/`Close`.
- **ComfyUI-AudioLoopHelper** (this repo) — loop timing, batch-
  encode schedule handling, per-iteration selector, latent context/
  overlap trim, profiler nodes.

Key open bugs (§9):

1. First ~20 s of output is a frozen still image. Leading
   hypothesis: `LTX2_NAG` at `nag_scale=11` extrapolating
   out-of-distribution on a model distilled without guidance.
2. "Microphones" (and other NAG-suppressed classes) can return
   iteration 2+. Partly addressed by the 2026-04-22 batch-encode
   migration.
3. Image identity drift across schedule boundaries.

**Reading order if you only have a few minutes:**

| Need | Read |
|------|------|
| Why is the first 20 s frozen? | §4 Sampler/mask routing + §7 KJNodes `LTX2_NAG` + §9 bugs |
| Are we encoding prompts correctly? | §3 CLIP path (short answer: yes, bit-identical to `CLIPTextEncode`) |
| Can I port a feature from native LTX-2 to ComfyUI? | §8 portability table |
| How do I add a new debug tool? | §10 Extension playbook |
| What runs when I hit Queue? | §1 node-by-node + §2 ComfyUI execution model |

## 1. Our workflow, node-by-node

### Outer initial render (runs once, before the loop)

```
┌─ Loaders ─────────────────────────────────────────┐
│ UNETLoader(414)           → LTX 2.3 DiT           │
│ DualCLIPLoader(416)       → Gemma 3 + projection  │
│ VAELoaderKJ(1537,1538)    → video + audio VAEs    │
│ MelBandRoFormerModelLoader(568, mode=4 bypassed)  │
└───────────────────────────────────────────────────┘
         │
         ▼
┌─ Model patch chain ────────────────────────────────┐
│ UNETLoader → SageAttn(268, active, mode=auto) →    │
│ ChunkFeedForward(504) → AttentionTuner(1523) →     │
│ LTX2_NAG(508) → SamplingPreviewOverride(503) →     │
│ Set_model(572)                                     │
└────────────────────────────────────────────────────┘
         │
         ▼
┌─ Input paths ──────────────────────────────────────┐
│ LoadImage → ImageResizeKJv2(445, 832×448) →       │
│   LTXVPreprocess(446, img_compression=18) →       │
│   LTXVImgToVideoInplaceKJ(531, strength=1, frame 0)│
│                                                    │
│ LoadAudio → TrimAudioDuration(567, skip 5s) →     │
│   Set_actual_audio(640) [direct; MelBand bypassed] │
│                                                    │
│ TrimAudioDuration(567) → LTXVAudioVAEEncode(566) → │
│   SetLatentNoiseMask(570) + SolidMask(571, val=0)  │
│   [audio latent gets noise_mask=0 = frozen context]│
└────────────────────────────────────────────────────┘
         │
         ▼
┌─ AV concat → outer sampler ─────────────────────────┐
│ EmptyLTXVLatentVideo(344, 832×448, len=497) →      │
│ LTXVImgToVideoInplaceKJ → video_latent             │
│                                                    │
│ LTXVConcatAVLatent(350): video + audio →           │
│   NestedTensor((video_5d, audio_4d))               │
│                                                    │
│ SamplerCustomAdvanced(161):                        │
│   noise:     RandomNoise(1322, seed=0 fixed)       │
│   guider:    CFGGuider(153, cfg=1, model=post-NAG) │
│   sampler:   KSamplerSelect(154, "euler")          │
│   sigmas:    ManualSigmas(1421, 8 fixed distilled  │
│              sigmas, NO ModelSamplingSD3 shift) →  │
│              VisualizeSigmasKJ(1422) → Set_sigmas  │
│   latent:    LTXVConcatAVLatent(350)               │
│                                                    │
│ Output → LTXVSeparateAVLatent(245) →               │
│   LTXVCropGuides(381) → LatentConcat(1605) samples1│
└────────────────────────────────────────────────────┘
```

### Loop body (TensorLoop subgraph, runs N times)

Subgraph instance 843. Inside the subgraph definition
(`definitions.subgraphs[0]`), ~15 nodes perform one iteration:

```
previous_latent (from TensorLoopOpen)
  → LatentContextExtract(2004)   [tail N latent frames, strip mask]
  → LTXVAudioVideoMask(606)      [combine with audio]
  → LTXVAddLatentGuide(1519)     [inject init image at frame -1]
  → LTXVConcatAVLatent(583)      [concat with audio]
  → SamplerCustomAdvanced(573)   [inner sampler, cfg=1]
  → LTXVSeparateAVLatent(596)    [split AV]
  → LTXVAdainLatent(2006)        [AdaIN color match vs initial render]
  → LTXVCropGuides(655)          [strip guide frames]
  → LatentOverlapTrim(2005)      [trim leading overlap]
  → IterationCleanup(2007)       [gc + empty_cache]
  → extended_latent (subgraph output)

Per-iteration inputs from outside the subgraph:
  sampler, sigmas:          Get_sampler, Get_sigmas (shared w/ outer)
  model:                    Get_model (post-NAG, post-all-patches)
  positive CONDITIONING:    ConditioningSelectByIteration(1616)
                              ← TimestampPromptScheduleBatchEncode(1615)
  negative CONDITIONING:    Get_base_cond_neg (static zeroed, inert at CFG=1)
  audio:                    Get_actual_audio
  current_iteration:        TensorLoopOpen(1539)
  start_index, noise_seed,
  overlap params:           AudioLoopController(1582)
```

Loop collects outputs via `LatentConcat(1605)`: outer-render slice
prepended, then all loop-body slices appended. Final
`LTXVSpatioTemporalTiledVAEDecode(1604, [1,1,63,7,True,'cpu','float16'])`
(stride-aligned temporal chunks — 56 latents = 17.92s per chunk advance,
matching the iteration stride; designed to bound decode RAM at any song
length, >=4-min validation pending) +
audio → `VHS_VideoCombine(617)`.

**Deeper:** `docs/reference/pipeline_flow_latent.md` for the long-form
node-by-node trace. Live inspection via
`scripts/analyze_workflow_dag.py <workflow> --save-run`.

## 2. ComfyUI core execution model

**Entry point.** "Queue" submits a prompt dict to
`PromptExecutor.execute_async` in ComfyUI core's `execution.py`,
which drives a topological DAG walk — each node executes in
dependency order.

**Caching.** Each node's result is cached by a hash of its inputs.
If the next run's inputs are bit-identical, the cached output is
reused. The `IS_CHANGED` classmethod lets a node override the hash
(used by `TimestampPromptScheduleBatchEncode` to hold cache across
iterations; see §4 of this doc and `nodes.py` in this repo).

**`TensorLoop` is an `enable_expand=True` node.** `TensorLoopOpen`
and `TensorLoopClose` rebuild the subgraph body dynamically on
every iteration via `GraphBuilder` + `io.NodeOutput(..., expand=graph.finalize())`.
Two consequences:

1. Nodes INSIDE the expanded subgraph body are fresh instances every
   iteration. Their framework-cache slot is new.
2. Any node OUTSIDE the subgraph but whose inputs transitively
   depend on `TensorLoopOpen.current_iteration` also gets
   invalidated per-iteration. `AudioLoopController` depends on
   `current_iteration`, so anything downstream of its outputs
   (stride, duration, overlap) is invalidated transitively.

**Framework-cache transitivity bit us.** The batch-encode node's
inputs include `AudioLoopController.stride_seconds` and
`audio_duration` — VALUE-stable across iterations but transitively
`current_iteration`-dependent. Without internal memoization the
batch encoder re-executed every iteration and re-encoded every
unique prompt. Fix: module-level LRU keyed on `(id(clip), schedule,
stride_seconds, audio_duration, snap_boundaries)` plus `IS_CHANGED`
returning the same tuple stringified.

**ModelPatcher clone asymmetry** (in ComfyUI core's
`comfy/model_patcher.py`):

- L339: `.clone()` does a SHALLOW copy of `object_patches`. Closures
  are shared references between clone and original.
- L561-580: `model_patches_to(device)` migrates tensors inside
  `transformer_options["patches"]` to the target device on offload.
  It does **NOT** touch `object_patches`.
- L900-915: `patch_model()` re-injects closures on reload but makes
  no device-state guarantees about tensors captured inside.
- L1078-1085: `detach(unpatch_all=True)` offloads weights; leaves
  `object_patches` closures pointing at whatever device the captured
  tensors were on when the closure was built.

`LTX2_NAG` captures `nag_cond_video[0][0].to(device, dtype)` inside a
`LTXVCrossAttentionPatch` closure (see §7) and registers it via
`add_object_patch`. When the DiT is offloaded (e.g. because per-iter
CLIP wants VRAM, pre-batch-encode-fix), the closure's captured
tensor goes stale and NAG effectively disengages from iteration 2.
This is THE root cause of the "microphones return iter 2+" symptom
pre-fix.

**Deeper:** `docs/analysis/nag_object_patches_offload_asymmetry.md`.

## 3. The CLIP encoding path

**Short answer: our batch encoder produces bit-identical CONDITIONING
to ComfyUI core's `CLIPTextEncode`.** No divergence, no concern.

Our path (`nodes.py` — both `CachedTextEncode.execute` and inside
`TimestampPromptScheduleBatchEncode.execute`):

```python
tokens = clip.tokenize(text)
cond = clip.encode_from_tokens_scheduled(tokens)
```

ComfyUI core `CLIPTextEncode.encode` (`nodes.py:76-80` in ComfyUI
core):

```python
def encode(self, clip, text):
    if clip is None:
        raise RuntimeError("ERROR: clip input is invalid: None...")
    tokens = clip.tokenize(text)
    return (clip.encode_from_tokens_scheduled(tokens), )
```

Identical two-call sequence.

**What does `encode_from_tokens_scheduled` do?** (ComfyUI core
`comfy/sd.py:307-364`):

- Reads `self.patcher.forced_hooks` and `self.use_clip_schedule`.
- If no hooks OR `use_clip_schedule == False` → falls through to
  plain `encode_from_tokens(tokens)` and returns a single
  CONDITIONING entry.
- `use_clip_schedule` defaults to `False` in `CLIP.__init__`. No
  workflow in this repo sets it.

So our call takes the no-schedule fast path — mathematically
equivalent to calling `encode_from_tokens` directly.

**`CLIPType.LTXV` dispatch** (`comfy/sd.py:1392-1394, 1543-1546`):
for LTXV, ComfyUI loads a Gemma 3 text encoder under the same `CLIP`
wrapper. No branching in `encode_from_tokens_scheduled` is
LTXV-specific — same code path as SD/SDXL/SD3, different underlying
model.

**What about ComfyUI-LTXVideo's own Gemma encoder?**
[`gemma_encoder.py`](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/main/gemma_encoder.py)
defines its own tokenizer + encoder that BYPASS ComfyUI's CLIP
wrapper entirely — uses HuggingFace `AutoTokenizer` and calls
`self.model()` directly. **We do NOT use that path.** Our workflow
loads CLIP via `DualCLIPLoader` (ComfyUI core) and feeds its
standard CLIP object to both our batch encoder and any residual
`CLIPTextEncode` nodes. Both paths go through the same ComfyUI
`CLIP` wrapper.

The `encode_from_tokens_scheduled` name is misleading (suggests
"scheduler-aware"); with our workflow config it's equivalent to
`encode_from_tokens`.

## 4. The Sampler + Mask routing

**User question:** given that our outer render has `mask[frame 0]=0`
(locked to init image) and `mask[frames 1..62]=1` (free to denoise),
why do frames 1..62 come out identical to frame 0?

**Routing chain** for our outer `SamplerCustomAdvanced(161)`:

1. **`SamplerCustomAdvanced.execute`** (ComfyUI core
   `comfy_extras/nodes_custom_sampler.py:948-978`): calls
   `guider.sample(noise.generate_noise(latent_image), latent_image,
   sampler, sigmas, denoise_mask=noise_mask, ...)` at L963.

2. **`CFGGuider.sample`** (ComfyUI core `comfy/samplers.py:1004-1063`):
   handles our NestedTensor AV latent correctly. L1008-1019 unpacks
   the NestedTensor into per-frame `denoise_masks` and
   `latent_shapes`. L1024-1025 calls `prepare_mask(denoise_masks[i],
   latent_shapes[i], device)` per frame. L1028-1030 re-packs if
   multiple frames. Dispatches to `outer_sample` at L1052.

3. **`outer_sample` → `inner_sample`** (ComfyUI core `comfy/samplers.py:984-1002,
   966-982`): loads models, casts tensors, calls
   `process_conds(...)` to expand conditioning, then invokes the
   sampler.

4. **`KSAMPLER.sample`** (ComfyUI core `comfy/samplers.py:734-753`):
   - L735: `extra_args["denoise_mask"] = denoise_mask`.
   - L736: wraps `self.inner_model` in `KSamplerX0Inpaint(model_wrap,
     sigmas)`.
   - L751: calls `sampler_function(model_k, noise, sigmas,
     extra_args=extra_args, ...)`. For `"euler"` sampler this is
     `sample_euler`.

5. **`sample_euler`** (ComfyUI core `comfy/k_diffusion/sampling.py:190-212`):
   classic Karras Alg 2 Euler loop.

   ```python
   for i in trange(len(sigmas) - 1, disable=disable):
       ...
       denoised = model(x, sigma_hat * s_in, **extra_args)
       d = to_d(x, sigma_hat, denoised)
       dt = sigmas[i + 1] - sigma_hat
       x = x + d * dt       # Euler step on the FULL batch x
   ```

   Per step: `model(x)` → `denoised`. Gradient
   `d = (x - denoised) / sigma_hat`. Step `x += d * dt`. **No
   per-frame branching; the whole latent batch updates uniformly.**

6. **`KSamplerX0Inpaint.__call__`** (ComfyUI core
   `comfy/samplers.py:394-403`) — THIS IS WHERE THE MASK ACTUALLY
   APPLIES:

   ```python
   def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None):
       if denoise_mask is not None:
           ...
           latent_mask = 1. - denoise_mask
           x = x * denoise_mask + self.inner_model.inner_model.scale_latent_inpaint(
                   x=x, sigma=sigma, noise=self.noise,
                   latent_image=self.latent_image
               ) * latent_mask
       out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
       if denoise_mask is not None:
           out = out * denoise_mask + self.latent_image * latent_mask
       return out
   ```

   Final mask application is **post-model-forward**:
   - Frame 0 (mask=0): `out[0] = model(x)[0] * 0 + latent_image[0] * 1
     = latent_image[0]` — forced back to the init image.
   - Frames 1..N (mask=1): `out[i] = model(x)[i] * 1 + latent_image[i]
     * 0 = model(x)[i]` — whatever the model output.

7. **CFG=1 short-circuit** (ComfyUI core `comfy/samplers.py:370-371`):

   ```python
   if math.isclose(cond_scale, 1.0) and
      model_options.get("disable_cfg1_optimization", False) == False:
       uncond_ = None
   ```

   At CFG=1 the uncond branch is skipped entirely. The model only
   runs with positive conditioning. Our `CFGGuider(cfg=1)` is
   effectively guidance-free at the guider level — all guidance in
   our workflow comes from `LTX2_NAG` patching cross-attention, not
   from CFG.

**Implication for the frozen-first-20s bug.** The mask machinery
works correctly. When frames 1..62 of the outer render look
identical to frame 0 (the init image), it is NOT because the mask
is locking them to `latent_image`. Mask=1 means they come straight
from `model(x)[i]`. So the MODEL is producing output that, for
frames 1..62, matches the init image.

At `nag_scale=11`, `LTX2_NAG` modifies every transformer block's
`attn2.forward` to compute `x_pos * 11 - x_neg * 10` in attention-
output space. With a "still image" negative prompt, this
extrapolates aggressively away from the negative direction. On a
distilled model trained without any guidance at all, the
extrapolation can land far outside the training distribution, and
the model's denoising prediction collapses toward a trivial fixed
point (per-step update magnitude → zero). Across 8 Euler steps the
latent drifts only marginally from its init state.

## 5. Model patches (object_patches vs transformer_options)

Two distinct patch systems in `ModelPatcher`:

- **`transformer_options["patches"]`**: dict of named tensors
  registered via `set_model_attn2_patch()` and friends. Migrated by
  `model_patches_to()` on offload/reload.
- **`object_patches`**: method overrides registered via
  `add_object_patch(key, bound_method)`. `.clone()` shallow-copies
  them; `model_patches_to()` does NOT touch them; `patch_model()`
  re-injects them on reload but doesn't track captured state.

`LTX2_NAG` uses `add_object_patch` for each transformer block's
`attn2.forward`. The bound method captures the encoded
`nag_cond_video` tensor. When ComfyUI offloads the DiT (e.g. because
per-iter CLIP wants VRAM), the closure survives in `object_patches`
but the tensor it holds can drift out of the expected device layout.

This is THE root cause of "microphones return iter 2+" pre-batch-
encode-fix. The 2026-04-22 migration keeps CLIP out of the per-iter
path → no offload pressure → closure stays consistent → NAG stays
active. The fix reduces the failure surface but doesn't fix
ComfyUI's underlying asymmetry.

**Deeper:** `docs/analysis/nag_object_patches_offload_asymmetry.md`.

## 6. ComfyUI-LTXVideo (Lightricks's Comfy wrapper)

What it ships that we use:

| Node | Purpose | Our workflow |
|------|---------|---------------|
| `LTXVAddLatentGuide` | Inject image as guide frame | inside subgraph (id 1519) |
| `LTXVCropGuides` | Strip guide frames | 381 (outer), 655 (inside) |
| `LTXVPreprocess` | H.264-compression noise on init | 446 |
| `LTXVSpatioTemporalTiledVAEDecode` | Spatio-temporal tiled VAE decode (stride-aligned chunks) | 1604 |
| `MultimodalGuider` + `GuiderParameters` | Joint AV guidance | not wired in `_latent.json`; available in the archived `_latent_stg.json` A/B variant (`example_workflows/archive/`) |
| `EmptyLTXVLatentVideo` | Blank latent | 344 |

What they ship that we DON'T use:

- [`gemma_encoder.py`](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/main/gemma_encoder.py)
  — bypasses ComfyUI's CLIP wrapper. See §3.
- `LTXVLoopingSampler` — incompatible with AV latents (2 root
  blockers + 3 type-system cascades per
  `docs/analysis/ltx23_gaps_analysis.md`).
- `STG` / `APG` — Lightricks's in-house guidance mechanisms. Not
  wired in baseline. APG is what Lightricks uses instead of NAG.

**Key preset from Lightricks's own repo**
([`presets/stg_advanced_presets.json`](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/main/presets/stg_advanced_presets.json),
`13b Distilled`):

```json
{
  "sigmas": [1.0],
  "cfg_values": [1],
  "stg_scale_values": [0],
  "stg_rescale_values": [1],
  "stg_layers_indices": [[25]]
}
```

Effectively a single-step denoising with no CFG, no STG. Matches
what native LTX-2's `SimpleDenoiser` does. **No NAG anywhere in
ComfyUI-LTXVideo.**

## 7. ComfyUI-KJNodes (the community layer)

What we depend on:

| Node | Purpose | Critical widgets |
|------|---------|------------------|
| `LTX2_NAG` | Cross-attention negative guidance | `[nag_scale, nag_alpha, nag_tau, inplace]` — default `[11, 0.25, 2.5, True]` |
| `LTX2AttentionTunerPatch` | Per-block attention scale | default scales all `1.0` = identity |
| `LTXVChunkFeedForward` | Memory-saving FFN chunking | `[chunks=2, dim_threshold=4096]` |
| `LTX2SamplingPreviewOverride` | Preview callback rate | only affects preview |
| `LTXVImgToVideoInplaceKJ` | Inject image at latent frame 0 | `[num_images, strength, frame_idx]` |
| `LTXVAddGuideMulti` | Multi-image guide (up to 20) | not in baseline |
| `PathchSageAttentionKJ` | Sage attention kernel patch (KJNodes) | superseded in baseline by `AudioLoopHelperSageAttention` (mode `auto`); see `docs/reference/sage_attention.md` |
| `VAELoaderKJ` | VAE loader with dtype + device control | |
| `Set*` / `Get*` | Variable-like wiring | used liberally |

**`LTX2_NAG` deep dive**
([`nodes/ltxv_nodes.py`](https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/ltxv_nodes.py),
class at line 442):

Schema (L452-459):
```python
io.Float.Input("nag_scale", default=11.0, min=0.0, max=100.0, ...)
io.Float.Input("nag_alpha", default=0.25, min=0.0, max=1.0, ...)
io.Float.Input("nag_tau", default=2.5, min=0.0, max=10.0, ...)
io.Conditioning.Input("nag_cond_video", optional=True)
io.Conditioning.Input("nag_cond_audio", optional=True)
io.Boolean.Input("inplace", default=True, optional=True)
```

**There is no `skip_blocks` parameter.** Verified against source.

Attention math (L351-353):
```python
nag_guidance = x_negative.mul_(self.nag_scale - 1).neg_().add_(
    x_positive, alpha=self.nag_scale,
)
# mathematically: x_positive * nag_scale - x_negative * (nag_scale - 1)
```

At `nag_scale=11`: `positive * 11 - negative * 10`. Aggressive
extrapolation. `nag_tau=2.5` is a norm clipping threshold; at high
scale even the tau clip is loose.

Registration (L500-501):
```python
patched_attn2 = LTXVCrossAttentionPatch(
    context_video, nag_scale, nag_alpha, nag_tau, inplace=inplace,
).__get__(block.attn2, block.__class__)
model_clone.add_object_patch(
    f"diffusion_model.transformer_blocks.{idx}.attn2.forward",
    patched_attn2,
)
```

Uses `add_object_patch` — hence the closure-capture + offload
asymmetry issue in §5.

**Deeper:** `docs/reference/nag_technical_reference.md` for the full NAG
treatment.

## 8. Native LTX-2 → ComfyUI portability

Classification for each native primitive:

| Native primitive (in [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2)) | ComfyUI equivalent | Class | Notes |
|---|---|---|---|
| `VideoConditionByLatentIndex` (`packages/ltx-core/src/ltx_core/conditioning/types/latent_cond.py:9-44`) | `LTXVImgToVideoInplaceKJ` | MINOR TWEAK | KJNodes' node is effectively a port. Same semantics (`strength → mask = 1 − strength` at a latent index). |
| `TemporalRegionMask` (`packages/ltx-core/src/ltx_core/conditioning/types/noise_mask_cond.py:10-45`) | `SolidMask + SetLatentNoiseMask + LTXVConcatAVLatent` | MAJOR PORT | Our noise_mask shape is 4-D `(1,1,H,W)`, native is 5-D per-frame. Semantics match; shape plumbing is less clean. |
| `ModalitySpec(frozen=True)` (`a2vid_two_stage.py:184-188`) | no direct analog | MAJOR PORT | ComfyUI doesn't have a first-class "frozen modality" concept. Implicit via `denoise_mask=0`. Native custom node would preserve the API. |
| `SimpleDenoiser` (`packages/ltx-pipelines/src/ltx_pipelines/distilled.py:129`) | `CFGGuider(cfg=1)` | ~AS-IS | At CFG=1 the uncond branch is skipped (§4). Mathematically equivalent for distilled. |
| `euler_denoising_loop` (`packages/ltx-pipelines/src/ltx_pipelines/samplers.py:34-74`) | `sample_euler` (ComfyUI core `comfy/k_diffusion/sampling.py:190-212`) | AS-IS | Same Karras Alg 2 step formula. No material divergence. |
| `MultiModalGuider` | `ComfyUI-LTXVideo/guiders/multimodal_guider.py` | AS-IS | Ported by Lightricks. Our baseline doesn't wire it, but the archived A/B `_latent_stg.json` variant (`example_workflows/archive/`) does. |
| `APG` (orthogonal projection guidance) | `ComfyUI-LTXVideo/stg.py:55-71` | AS-IS | Ported by Lightricks. Could replace NAG if we wanted Lightricks-style guidance. |
| A2V two-stage pipeline | — | INFEASIBLE | Two imperative stages with frozen-modality flip between them. ComfyUI DAG can't express this directly; needs a custom node encapsulating both stages with explicit handoff wiring. |
| `DISTILLED_SIGMA_VALUES` (`distilled.py:16-22`) = `[1.0, 0.994, 0.988, 0.981, 0.975, 0.909, 0.725, 0.422, 0.0]` | `ManualSigmas` with the 8 fixed distilled values (NO `ModelSamplingSD3` shift node) | AS-IS by literal values | Canonical distilled path feeds the values directly; verified bit-exact via `VisualizeSigmasKJ`. See `docs/reference/sampler_reference.md`. |

**Ranking.** 6 of 9 primitives port AS-IS or with minor tweaks. The
two MAJOR-PORT items are both about modality-frozen semantics —
ComfyUI expresses "this stays" via `noise_mask` rather than via a
first-class frozen-flag. The INFEASIBLE item (two-stage A2V) needs
a custom node if we ever want it.

**Key finding from cross-reading all three Lightricks codebases:
LTX 2.3 distilled-1.1 is a guidance-FREE pipeline at inference.**
Native LTX-2's `SimpleDenoiser`, Lightricks's ComfyUI `13b Distilled`
preset (cfg_values=[1], stg_scale=0), and LTX-Desktop's production
distilled path all agree: single forward pass per step, no CFG, no
STG, **no NAG**. Our workflow stacks `LTX2_NAG` at `nag_scale=11`
on top of a pipeline Lightricks designed without guidance — that's
the suspect configuration for the frozen-render bug.

## 9. Known bugs + current status

### 9.1 Frozen first 20 s of output (OPEN)

**Symptom.** Outer initial render produces a still image of the
init image for its full ~20 s duration, regardless of prompt text.

**Leading hypothesis.** `LTX2_NAG.nag_scale=11` extrapolates
attention output out-of-distribution on a distilled model trained
without guidance. Across 8 Euler steps the per-step update
magnitude approaches zero.

**Evidence for.** §4 shows the mask system works correctly (frames
1..62 come from the model output, not the mask-to-latent_image
blend). §7 shows the attention math at `nag_scale=11` is aggressive
extrapolation. §8 shows native LTX-2 uses no guidance at all for
distilled.

**Evidence against.** Community reference workflows also use
`nag_scale=11` — but nobody has verified those workflows actually
produce motion-rich output vs. ship the same symptom unnoticed.

**Empirical A/B pending.** The test: set `LTX2_NAG(508).nag_scale=0`
and re-run. If motion returns, confirmed.

### 9.2 Microphones returning iteration 2+ (PARTIALLY ADDRESSED)

**Symptom.** Items explicitly negated in the NAG `nag_cond_video`
are suppressed on iteration 1 but reappear from iteration 2 onward.

**Fix landed 2026-04-22.** Batch-encode migration + internal cache.
CLIP no longer loads per-iteration, so the DiT no longer gets
evicted, so NAG's `object_patches` closure doesn't get exposed to
an offload/reload round-trip.

**Residual uncertainty.** Could also be the NAG-at-scale-11 blowout
(same root cause as 9.1) rather than offload drift. Could be
content-specific (the user's init image + prompt happen to make
microphones easy for the model to regenerate).

**Deeper:** §5; `docs/analysis/nag_object_patches_offload_asymmetry.md`.

### 9.3 Image identity drift across schedule boundaries (OPEN)

**Symptom.** Subject's face / clothing / gear shifts between
schedule entries.

**Known contributors:**
- LTX 2.3 audio-video cross-attention is photoreal-trained.
  Illustrated / painterly inits drift toward photoreal even with
  `Style: illustrated.` in every schedule entry.
- Per-iteration prompt changes cause embedding drift even at CFG=1.

**Mitigations in place.**
- `snap_boundaries=True` in batch encoder → every iteration runs on
  one pure prompt (no mid-iteration mixed conditioning).
- `LTXVAddLatentGuide(strength=1, frame_index=-1)` inside subgraph
  re-anchors style at the end of each iteration.

**Structural fix not yet built.** Multi-image guide per iteration
via `LTXVAddGuideMulti` (KJNodes) to re-anchor subject mid-iteration.

## 10. Extension playbook

For "I want to...":

| Goal | Start here |
|------|------------|
| Add per-iteration NEGATIVE prompts | §7 `LTX2_NAG` inside the loop (re-patch model per-iter), OR bump inner `CFGGuider` to CFG>1 and wire a scheduled negative (doubles sampling cost) |
| Trim i2v opening-filler frames from the saved clip | `LTXHeadTrim` between the decoded `IMAGE` output and `VHS_VideoCombine`; drops the first `trim_latent_frames * 8` pixel frames + matching audio span. Default 0 = no-op. Post-decode trim, not pre-sample — doesn't fight the model's temporal prior, just hides the window where it dominates |
| Multi-image guide per iteration (re-anchor style) | §6 + §8 — `LTXVAddGuideMulti` (KJNodes, up to 20 images) or chain `LTXVAddLatentGuide` |
| A2V two-stage pipeline | §8 — INFEASIBLE as straight port; build a custom node that runs both stages with an explicit frozen-modality handoff |
| Replace NAG with Lightricks-native guidance | §6 — swap to `MultimodalGuider` + `GuiderParameters` + optional `STG` / `APG`. See `example_workflows/archive/audio-loop-music-video_latent_stg.json` for a working A/B variant (archived) |
| Add a new debug tool | Use `scripts/workflow_utils.py::timestamped_run_path()` to land output under the gitignored runs dir. Follow the pattern of `exec_logger.py` or `scripts/analyze_workflow_dag.py` |
| Verify any node's widget schema | `uv run --group dev python scripts/trace_node_source.py <workflow> <node_id> --include-inputs` — prints the authoritative schema from source. Run this BEFORE trusting any widget annotation in docs |
| See execution order + DAG | `uv run --group dev python scripts/analyze_workflow_dag.py <workflow> --save-run --format ascii` (or `mermaid` / `dot` / `json`) |
| Runtime per-node timing + shapes | `COMFYUI_EXEC_LOG=auto python main.py` (from ComfyUI root) then run the workflow. JSONL lands in the gitignored runs dir |
| Torch.profiler trace | Apply `scripts/apply_profiling_nodes.py <workflow>` to insert `ProfileBegin/IterStep/End`. Trace lands in `runs/profiler/<timestamp>/` |
| Migrate a workflow to the batch-encode path | `uv run --group dev python scripts/apply_batch_encode_fix.py <workflow.json>` — idempotent, includes type-based orphan sweep on re-runs |
| Bypass MelBand + rewire audio | `uv run --group dev python scripts/apply_melband_default_off.py <workflow.json>` — sets mode=4 on MelBand nodes AND rewires `Set_actual_audio` direct from `TrimAudioDuration` |
| Add a new node class | Copy the pattern of `TimestampPromptScheduleBatchEncode` in `nodes.py`: `define_schema` + `execute` + optional `IS_CHANGED` + module-level LRU if it needs to cache across loop iterations (see §2 on TensorLoop cache invalidation) |

## See also

- `docs/reference/pipeline_flow_latent.md` — full node-by-node trace
  of the LATENT workflow (the primary baseline)
- `docs/reference/sampler_reference.md` — sampler walkthrough with
  ComfyUI + MultimodalGuider source references
- `docs/reference/nag_technical_reference.md` — NAG deep dive
- `docs/analysis/nag_object_patches_offload_asymmetry.md` —
  ModelPatcher offload asymmetry explainer
- `docs/analysis/ltx23_gaps_analysis.md` — capability gaps
- `docs/guides/debugging_guide.md` — symptom → first-check lookup
- `CLAUDE.md` — authoritative project memory / coding conventions
