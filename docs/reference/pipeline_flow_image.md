Last updated: 2026-04-23 (trimmed summary; full node-by-node archived to internal/archive/)

# Pipeline Flow: IMAGE-based Music Video Workflow (summary)

> **Reference only.** The LATENT loop
> (`docs/reference/pipeline_flow_latent.md` →
> `audio-loop-music-video_latent.json`) is the primary working
> baseline per CLAUDE.md. This summary covers the IMAGE path for
> anyone running the legacy `_image.json` workflow.
>
> **Full node-by-node trace** (1923 lines, privacy-unscrubbed with
> absolute paths) lives at `internal/archive/pipeline_flow_image_full.md`
> (gitignored). Retrieve it when you need widget values per node.

Source: `example_workflows/audio-loop-music-video_image.json`.

## What makes this the IMAGE path (vs LATENT)

The two workflows differ in **where the loop decodes**:

| Path | Loop body decodes to… | VAE round-trip per iteration? |
|---|---|---|
| IMAGE (`_image.json`) | Pixel images; loop re-encodes per iteration | **Yes** — costs ~2-3s/iter extra |
| LATENT (`_latent.json`, primary) | Latents; decodes once at end | **No** |

Everything else is close to identical. The LATENT rework landed
2026-04-09 and became primary because the per-iteration VAE
round-trip was the largest single time cost in the IMAGE path (no
quality upside over LATENT).

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

## Where this diverges from LATENT

Three notable structural differences with the LATENT path
(`docs/reference/pipeline_flow_latent.md`):

1. **Loop body extension subgraph (#843) decodes + re-encodes.**
   In the LATENT variant, the subgraph keeps everything in latent
   space (`LatentContextExtract` / `LatentOverlapTrim` trim the
   tail latent frames directly; `StripLatentNoiseMask` ensures the
   tail has no stale mask). In the IMAGE variant, the subgraph
   decodes the tail to pixels, feeds the pixels back through
   `LTXVImgToVideoInplaceKJ` for the next iteration.
2. **CLIP enters the loop body.** This is fine for the IMAGE path
   because re-encoding is already the cost driver; the
   CLIP-in-loop object_patches asymmetry (see
   `docs/analysis/nag_object_patches_offload_asymmetry.md`) is less
   visible when the VAE round-trip dominates. The LATENT path
   cannot tolerate it — see
   `docs/reference/nag_technical_reference.md` "Operational
   constraint" callout.
3. **AdaIN is applied per-iteration at pixel level.** The LATENT
   path applies AdaIN on the latent tensor (faster, no decode
   needed). Not a qualitative difference, just a performance one.

## When to keep running this workflow

- You have a saved `_image.json` that predates the LATENT rework
  and don't want to migrate.
- You're debugging a difference between the two variants and need
  the pixel-level AdaIN pipeline active for A/B comparison.
- You're patching something that specifically needs the per-iter
  decode loop (rare — most needs are better served by extending
  the LATENT path).

Otherwise, migrate to `audio-loop-music-video_latent.json`.

## See also

- Full node-by-node trace: `internal/archive/pipeline_flow_image_full.md`
- Primary working baseline: `docs/reference/pipeline_flow_latent.md`
- Architecture-level walkthrough of the loop: `docs/architecture_overview.md`
