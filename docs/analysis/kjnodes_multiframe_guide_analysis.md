Last updated: 2026-04-16

# KJNodes: Multi-Frame Guide Capabilities Analysis

Analysis of KJNodes' LTX-specific nodes for multi-frame conditioning,
focused on capabilities we can leverage in the AudioLoopHelper workflow.

Source: `~/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py`

## Key Discovery: LTXVAddGuideMulti

KJNodes already provides a node that adds up to 20 guide images at
different frame indices in a single node call:

```
LTXVAddGuideMulti (KJNodes)
  Inputs:
    - positive, negative (CONDITIONING)
    - vae (VAE)
    - latent (LATENT)
    - num_guides (DynamicCombo: 1-20)
      Per guide: image_N (IMAGE), frame_idx_N (INT), strength_N (FLOAT)
  Outputs:
    - positive, negative, latent (with all guides appended)
```

Location: `ltxv_nodes.py:16`

**How it works**: Iterates through each guide, VAE-encodes the image, calls
`append_keyframe()` for each one. All guides accumulate in the conditioning
metadata. `LTXVCropGuides` strips them all correctly.

**Implication for our workflow**: If we modify the subgraph to use
`LTXVAddGuideMulti` instead of `LTXVAddLatentGuide`, we could inject
multiple guide images at different frame positions within each iteration
window — e.g., a "start of window" guide and an "end of window" guide.

### Integration with KeyframeImageSchedule

Our `KeyframeImageSchedule` already outputs `image` and `next_image` with
a `blend_factor`. Instead of blending in pixel space via `ImageBlend`, we
could feed both images directly to `LTXVAddGuideMulti`:
- `image_1` = current keyframe at `frame_idx=0` (start of window)
- `image_2` = next keyframe at `frame_idx=-1` (end of window)

This gives the model a visual trajectory: "start here, end there." Much
stronger than a single guide + text conditioning change.

**Trade-off**: Two guides = ~2x VRAM for guide processing. For our
window size (63 latent frames + 2 guide frames), the overhead is ~3%.

## LTXVAddGuidesFromBatch

Location: `ltxv_nodes.py:101`

Takes an IMAGE batch and automatically uses each non-black image as a
guide at its corresponding batch index as frame_idx.

```
LTXVAddGuidesFromBatch (KJNodes)
  Inputs:
    - positive, negative, vae, latent
    - images (IMAGE batch)
    - strength (FLOAT, shared across all guides)
  Logic: for each image in batch, if not black, add as guide at frame_idx=i
```

**Use case**: Load a reference video as an image batch, pass it to this node.
Every frame becomes a guide at its natural position. Combined with our
`VideoFrameExtract` for selecting the right temporal segment per iteration.

**Limitation**: Strength is shared across all guides (no per-guide control).
For our use case, `LTXVAddGuideMulti` with per-guide strength is more
flexible.

## LTXVAudioVideoMask

Location: `ltxv_nodes.py:163`

Already used in our subgraph (node 606). Creates noise masks for video and
audio latents with independent temporal control.

Key inputs:
- `video_start_time` — wired from `AudioLoopController.overlap_seconds`
- `audio_start_time` / `audio_end_time` — both wired to `window_size_seconds`
  (creates zero-width range = audio frozen)

**No changes needed** — this node is correctly wired for frozen audio.

## LTXVImgToVideoInplaceKJ

Location: `ltxv_nodes.py:1021`

In-place guide injection (overwrites latent tokens at target position).
Currently used only for the initial render (node 531, frame_idx=0).

Supports multiple images via DynamicCombo with per-image strength and index.

**Difference from append approach**: In-place is stronger (latent tokens are
replaced, not just concatenated as side-channel conditioning) but can only
target positions within the existing latent — cannot extend beyond the
latent's temporal range.

## Recommendation: Subgraph Upgrade Path

### Current state (Phase 1, done)
Subgraph uses `LTXVAddLatentGuide` (node 1519) with single guide image.
`KeyframeImageSchedule` switches the guide per iteration.

### Phase 2: Multi-guide per window
Replace `LTXVAddLatentGuide` with `LTXVAddGuideMulti` in the subgraph.
Wire `KeyframeImageSchedule.image` as `image_1` at `frame_idx=0` and
`KeyframeImageSchedule.next_image` as `image_2` at `frame_idx=-1`.

This requires:
1. Adding a second IMAGE input to subgraph #843
2. Replacing node 1519 (LTXVAddLatentGuide) with LTXVAddGuideMulti
3. Wiring both KeyframeImageSchedule outputs through the subgraph
4. `LTXVCropGuides` (node 655) already handles multiple guides correctly

### Phase 3: Reference video conditioning
Use `LTXVAddGuidesFromBatch` to inject multiple frames from a reference
video as guides throughout each iteration window. This is the full
video-to-video approach.
