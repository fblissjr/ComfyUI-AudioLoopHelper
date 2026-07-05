Last updated: 2026-07-05

# Spatial-inpaint retake — design (experimental)

Post-loop workflow: paint a spatial mask over a region of a finished
music-video render, regenerate just that region via the official
Lightricks in-outpainting IC-LoRA, keep the song frozen. Complements the
existing temporal-only `LatentTemporalMask` retake (which regenerates a
whole `[start,end]` time span). Companion to PLAN §3 "spatial retake via
official inpaint stack".

## Official reference topology (extracted 2026-07-05)

Source: `ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_ICLoRA_Inpaint_Two_Stage_Distilled.json`, STAGE 1 only (2x refine stage dropped). Node IDs are the official file's.

**Model chain**: `CheckpointLoaderSimple#3940 → LoraLoaderModelOnly#4922 (distilled-lora-384-1.1 @0.5) → LTXICLoRALoaderModelOnly#5011 (in-outpainting-0.9 @1.0)` → `CFGGuider#4828.model`.

**Mask branch**: `LoadVideo#5375 (mask mp4) → GetVideoComponents#5376.image → ImageToMask#5377 ('red') → ResizeImageMaskNode#5381 (match size) → LTXVDilateVideoMask#5382 (spatial_radius=32, temporal=0)`.

**Source branch**: `LoadVideo#5368 (source mp4) → GetVideoComponents#5168.image → ResizeImageMaskNode#5399 (scale shorter 1024) → #5371 (scale to multiple 64) → #5384 (scale by 0.5) → LTXVInpaintPreprocess#5378.images`. Dilated mask → `#5378.mask`. Output = green-composited (#66FF00 where masked) video.

**Base latent (conditioning)**: `LoadImage#2004 (still init) → ResizeImageMaskNode#5188 (scale longer 1024) → LTXVPreprocess#5388 (img_compression=18) → LTXVImgToVideoConditionOnly#3159 (strength=0.7)`, whose `latent` in = `EmptyLTXVLatentVideo#3059` (sized via `GetImageSize#5054` of the source). So the base is an EMPTY video latent conditioned on a still first frame — NOT an encoded source video.

**IC-LoRA guide**: `LTXAddVideoICLoRAGuideAdvanced#5114` — `positive/negative <- LTXVConditioning#1241` (frame_rate=24), `vae <- checkpoint`, `latent <- ImgToVideoConditionOnly#3159`, `image <- LTXVInpaintPreprocess#5378` (green composite). Widgets `[frame_idx=0, strength=1, latent_downscale=1, crop=disabled, tiled=False, tile_size=256, tile_overlap=64, attention_strength=1]`. Outputs `positive/negative/latent`.

**Audio**: `VAEEncodeAudio#5389 (source audio) → LTXVSetAudioRefTokens#5390 (appends to guide pos/neg) → LTXVConcatAVLatent#5391 (guide latent + audio ref)` → sampler. **Audio is REGENERATED** (reference-token conditioned), not frozen.

**Sampler**: `SamplerCustomAdvanced#5093` — `euler_ancestral_cfg_pp`, canonical 8-sigma, CFG 1, `latent_image <- ConcatAV#5391` → `LTXVSeparateAVLatent#5394 → LTXVCropGuides#5013`.

## Adaptation decisions for our frozen-audio music-video case

**Corrected 2026-07-05 after full wiring extraction** — two earlier assumptions were wrong: (1) the official does NOT regenerate audio, it FREEZES it (`LTXVSetAudioRefTokens.frozen_audio`, `noise_mask=0`); (2) unmasked-region preservation is NOT a base-latent property — the base latent is empty (the i2v seed path is `bypass=True`, inert). Preservation is a final pixel-space `LTXVLaplacianPyramidBlend` that composites generated pixels only inside the mask onto the clean source. This dissolves the old D2/D6 open questions.

| # | Decision | Our choice | Why |
|---|---|---|---|
| D1 | Audio | **Bit-identical passthrough** (source audio → VHS; sample video-only, reuse retake Option A) | Official freezes audio via VAE round-trip; our passthrough is strictly cleaner (no round-trip loss). Drops `VAEEncodeAudio/SetAudioRefTokens/ConcatAV`. Retake precedent proves video-only sampling works. |
| D2 | Base latent | **Empty latent** (mirror official; i2v path omitted since it's `bypass=True`/inert there) | The IC-LoRA reconstructs from the green-composite guide; the Laplacian blend restores unmasked pixels. No init image needed. RESOLVED. |
| D3 | Sampler | **`euler`** (not official `euler_ancestral_cfg_pp`) | Project convention D-EA-3: distilled + ancestral re-noise unvalidated. Ancestral = an A/B variable. |
| D4 | Model | **fp8 distilled UNETLoader + LTXICLoRALoaderModelOnly**, NO distilled-lora-384 @0.5 | Our fp8 checkpoint bakes distillation. IC-LoRA loader AFTER UNETLoader, BEFORE the sage/chunkff/attntuner chain (canonical mutation order). |
| D5 | Preservation | **`LTXVLaplacianPyramidBlend`** (generated frames × mask, over clean source) | The official mechanism that keeps unmasked regions exact. Include it — without it the whole frame is model-reconstructed and drifts. |
| D6 | fps / decode | **25 / `LTXVTiledVAEDecode [1,1,1,cpu,float16]`** | Standard conventions; single-pass. |

## v1 scope: A — faithful spatial-only

Mirror official stage-1 with D1-D6. User supplies a B/W spatial mask video (white = region to regenerate). Follow-up **B (combined spatial+temporal** — keep `LatentTemporalMask` time-gate AND the spatial IC-LoRA, "edit this region only during 0:42-1:12") deferred: stacks two mechanisms, build after A renders clean.

## Build plan (once scope locked)

Fork from `example_workflows/audio-loop-music-video_retake.json` (already has: video ingress, video-only sampler, audio passthrough, fp8 model + sage/nag chain, canonical decode + trim/RunId chain). Deltas: insert IC-LoRA loader in model chain; add mask branch (LoadVideo+GetVideoComponents+ImageToMask+DilateVideoMask); add InpaintPreprocess; add EmptyLTXVLatentVideo + img-cond base; interpose LTXAddVideoICLoRAGuideAdvanced between conditioning and sampler; drop LatentTemporalMask (v1-A). Output → `example_workflows/experimental/audio-loop-music-video_spatial_inpaint.json`. Apply: `scripts/apply_spatial_inpaint.py` (staged variant; filename avoids the `retake` substring so `_is_retake` audit checks don't misfire). Audit: WARN-level presence check.

## Render gate

Behavioral unknowns that only a render settles: (1) does the in-outpainting IC-LoRA reconstruct non-masked regions faithfully at distilled CFG=1 with frozen-audio video-only sampling; (2) D2 (empty vs encoded-source base) if A looks wrong; (3) D3 euler vs ancestral if motion in the masked region stalls. Multi-seed per the ±20 BPM variance rule.
