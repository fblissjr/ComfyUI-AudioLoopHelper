Last updated: 2026-04-11

# Subgraph Latent Rework: Eliminate Per-Iteration VAE Round-Trip

Modify the extension subgraph (#843) to pass LATENT between loop iterations
instead of IMAGE. Eliminates lossy VAE decode/encode each iteration.

## Frame Count Conversion

LTX temporal scale factor = 8. Pixel frames to latent frames:
```
latent_frames = (pixel_frames - 1) / 8 + 1
```

| Pixel frames | Latent frames | Seconds at 25fps |
|---|---|---|
| 25 (overlap) | 4 | 1.0s |
| 50 (2s overlap) | 7 | 2.0s |
| 497 (window) | 63 | 19.88s |

## Before (current subgraph)

```
INPUT: previous_images (IMAGE) from TensorLoopOpen
  ↓
GetImageRangeFromBatch #615: last 25 IMAGE frames → context
  ↓
VAEEncode #614: context IMAGE → LATENT                    ← REMOVE
  ↓
VAEEncode #1520: init_image IMAGE → LATENT (guide)        ← KEEP
  ↓
LTXVAddLatentGuide #1519: merge both guides
  ↓
LTXVConcatAVLatent #583: add audio latent
  ↓
SamplerCustomAdvanced #573: generate
  ↓
LTXVSeparateAVLatent #596: split video/audio
  ↓
LTXVCropGuides #655: trim guide padding
  ↓
VAEDecode #1521: LATENT → IMAGE                           ← REMOVE
  ↓
GetImageSizeAndCount #1504: (info only)                   ← REMOVE
  ↓
GetImageRangeFromBatch #1509: trim first 25 IMAGE frames
  ↓
OUTPUT: extended_images (IMAGE) to TensorLoopClose
```

## After (latent-space subgraph)

```
INPUT: previous_latent (LATENT) from TensorLoopOpen
  ↓
LatentContextExtract #2004: last 4 LATENT frames → context (strips noise_mask)
  ↓
VAEEncode #1520: init_image IMAGE → LATENT (guide)        ← KEEP
  ↓
LTXVAddLatentGuide #1519: merge both guides
  ↓
LTXVConcatAVLatent #583: add audio latent
  ↓
SamplerCustomAdvanced #573: generate
  ↓
LTXVSeparateAVLatent #596: split video/audio
  ↓
LTXVCropGuides #655: trim guide padding
  ↓
LatentOverlapTrim #2005: trim first 4 LATENT frames (strips noise_mask)
  ↓
OUTPUT: extended_latent (LATENT) to TensorLoopClose
```

NOTE: LatentContextExtract and LatentOverlapTrim replace LTXVSelectLatents +
StripLatentNoiseMask. They handle noise_mask stripping internally, which is
critical -- stale noise_mask from previous iterations corrupts the sampler.
See `internal/postmortem_v0409_latent_rework.md` Issue 5 for the full analysis.

## Step-by-step in ComfyUI

### Step 1: Change TensorLoopOpen initial_value type

Currently TensorLoopOpen (#1539) receives IMAGE from Reroute #618
(which comes from VAEDecode #1318 of the initial render).

Change: wire TensorLoopOpen initial_value from the initial LATENT instead.
This is the video-only latent from LTXVImgToVideoInplaceKJ (#531),
BEFORE ConcatAVLatent (#350).

The loop now passes LATENT between iterations instead of IMAGE.

### Step 2: Open extension subgraph #843

Double-click #843 to enter the subgraph.

### Step 3: Replace context extraction (node 615)

**Delete** node 615 (GetImageRangeFromBatch).
**Delete** node 614 (VAEEncode for context frames).

**Add** LatentContextExtract node (#2004). Wire:
- latent ← subgraph input (now receives LATENT from TensorLoopOpen)
- overlap_latent_frames ← AudioLoopController overlap_latent_frames output (4 for 1s overlap)

Wire context output → LTXVAudioVideoMask video_latent input
(where VAEEncode #614 output used to go).

LatentContextExtract extracts the tail frames AND strips noise_mask
internally. No need for a separate StripLatentNoiseMask node.

### Step 4: Replace output trimming (node 1509)

**Delete** node 1509 (GetImageRangeFromBatch).
**Delete** node 1521 (VAEDecode).
**Delete** node 1504 (GetImageSizeAndCount).

**Add** LatentOverlapTrim node (#2005). Wire:
- latent ← LTXVCropGuides #655 latent output
- overlap_latent_frames ← AudioLoopController overlap_latent_frames output (4 for 1s overlap)

Wire trimmed output → subgraph output (extended_latent).

LatentOverlapTrim trims the overlap AND strips noise_mask internally.
Clamps to prevent empty tensors if overlap >= total frames.

### Step 5: Update subgraph input type

The subgraph input "images" (slot 4) currently accepts IMAGE.
It now receives LATENT from TensorLoopOpen.

In ComfyUI, the subgraph input type may auto-detect from the wired connection.
If not, you may need to recreate the input as LATENT type.

### Step 6: Update subgraph output type

The subgraph output currently produces IMAGE.
It now produces LATENT.

Same as Step 5 -- may auto-detect or need recreation.

### Step 7: Outside the subgraph -- add final VAEDecode

After TensorLoopClose (#1540), the output is now LATENT (accumulated
across all iterations). Add:

- **VAEDecodeTiled** (or VAEDecode): LATENT from TensorLoopClose → IMAGE
- Wire IMAGE → VHS_VideoCombine #617 images input

Remove the old ImageBatch #1507 if it still exists (was combining
initial IMAGE + loop IMAGE, no longer needed).

### Step 8: Adjust overlap_frames

AudioLoopController currently outputs overlap_frames in PIXEL frames (25).
LTXVSelectLatents needs LATENT frames (4).

Options:
a. Add a conversion in AudioLoopController: output `overlap_latent_frames`
   computed as `round((overlap_frames - 1) / 8) + 1`
b. Hardcode the LTXVSelectLatents values (simpler, less flexible)
c. Wire a math node to convert pixel → latent frames

Option (a) is cleanest. Add a new output to AudioLoopController.

### Step 9: Test

1. Run with same audio + image as v0408
2. Compare quality (should be equal or better)
3. Verify all iterations complete
4. Verify lip sync maintained
5. Check final video duration matches audio

## Important Notes

- LTXVSelectLatents indices are in LATENT frame space, not pixel frame space
- The temporal scale factor is 8 (hardcoded in LTX 2.3 architecture)
- TensorLoopClose LATENT accumulation concatenates along dim 2 (temporal)
  for 5D tensors -- this is correct for video latents
- The init_image VAEEncode (#1520) stays because it converts IMAGE → LATENT
  for the frame guide. This is the same image every iteration so the cost
  is constant (not compounding).
- Audio conditioning (ConcatAVLatent) is unaffected -- it happens after
  the context extraction, inside the sampler
