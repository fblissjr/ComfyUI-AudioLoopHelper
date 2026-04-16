Last updated: 2026-04-16

# ComfyUI-LTXVideo: Multi-Frame Guide Conditioning Analysis

Analysis of multi-frame conditioning capabilities in ComfyUI-LTXVideo nodes,
focused on what can be leveraged for per-iteration and intra-window visual
grounding in the AudioLoopHelper workflow.

## Key Finding: Guide Chaining Works

**Multiple `LTXVAddLatentGuide` calls can be chained.** Each call appends
guide frames to the latent tensor and accumulates `keyframe_idxs` in the
conditioning metadata. `LTXVCropGuides` correctly strips ALL accumulated
guide frames by counting `num_keyframes = torch.unique(keyframe_idxs)`.

This means a subgraph could add a guide at frame 0 (start of window) AND
a guide at frame 62 (end of window) by chaining two `LTXVAddLatentGuide`
nodes. `LTXVCropGuides` would strip both. No special handling needed.

Source: `comfy_extras/nodes_lt.py:395-411` (LTXVCropGuides.execute)

## Guide Node Hierarchy

### LTXVAddGuide (core ComfyUI)
- Location: `comfy_extras/nodes_lt.py:197`
- The primitive. Takes IMAGE, VAE-encodes it, calls `append_keyframe()`.
- `append_keyframe()` concatenates guide latent to temporal dim, updates
  `keyframe_idxs` and `noise_mask` in conditioning metadata.

### LTXVAddLatentGuide (ComfyUI-LTXVideo)
- Location: `ComfyUI-LTXVideo/latents.py:400`
- Takes pre-encoded LATENT (not IMAGE). Calls `append_keyframe()` internally.
- Adds `guide_attention_entries` for per-reference attention control.
- Used in our subgraph #843 (node 1519). Currently adds ONE guide at `latent_idx`.
- **Chainable**: returns (positive, negative, latent) — output can feed
  another LTXVAddLatentGuide to add a second guide at a different index.

### LTXVAddGuideAdvanced (ComfyUI-LTXVideo)
- Location: `ComfyUI-LTXVideo/guide.py:21`
- Wrapper around core `LTXVAddGuide`. Adds CRF preprocessing, blur, interpolation.
- Takes IMAGE (not LATENT), processes it, then calls `LTXVAddGuide.execute()`.
- Supports `frame_idx` from -9999 to 9999 (negative = from end of video).
- Single image per call — chain for multiple guides.

### LTXVAddGuideAdvancedAttention (ComfyUI-LTXVideo)
- Location: `ComfyUI-LTXVideo/guide.py:258`
- Same as Advanced but adds `attention_strength` (0.0-1.0) and optional
  `attention_mask` (MASK type) per guide.
- Allows per-guide control of how strongly the guide influences self-attention.
- Spatial masks can restrict a guide's influence to specific image regions.

## How append_keyframe Works

From `comfy_extras/nodes_lt.py:256-370`:

1. Computes `frame_idx` (pixel space) and `latent_idx` from the guide position
2. Encodes guide image via VAE if needed
3. Concatenates guide latent to the END of the latent temporal dimension:
   `latent = torch.cat([latent, guide], dim=2)`
4. Extends `noise_mask` with guide mask (strength controls mask value)
5. Creates `keyframe_idxs` that tell RoPE where the guide logically belongs
6. Stores everything in conditioning metadata

The guide is NOT inserted at its target position — it's appended and RoPE
handles the positional encoding. This is why guides don't disrupt the
contiguous latent structure.

## Implications for Multi-Guide-Per-Iteration

### What works now (no subgraph changes)
- Per-iteration guide switching via `KeyframeImageSchedule` (our new node)
- Single guide per iteration through existing `LTXVAddLatentGuide` in subgraph

### What's possible with subgraph modification
- **Start + end guide per window**: Chain a second `LTXVAddLatentGuide` in
  the subgraph. First guide at `latent_idx=-1` (before frame 0), second at
  `latent_idx=62` (end of window). Creates a visual trajectory within each
  iteration — the model sees where to start AND where to end up.
- **Cost**: Each guide adds ~1 latent frame of VRAM. Two guides = negligible.
- **CropGuides**: Already handles multiple guides correctly.
- **Attention control**: Could use `LTXVAddGuideAdvancedAttention` to give
  the end-of-window guide lower attention_strength (0.3-0.5) so it suggests
  direction without fighting the start guide.

### LTXVLoopingSampler per-tile guides
- Location: `ComfyUI-LTXVideo/looping_sampler.py:67`
- Accepts `optional_guiding_latents` — a full video latent that gets sliced
  per temporal tile. Each tile gets the corresponding chunk as a guide.
- Also accepts `optional_cond_images` with per-tile frame indices.
- **Cannot use for our workflow** (rejects AV latents), but the approach of
  slicing a reference video into per-tile guides is exactly what
  `VideoFrameExtract` does at the image level.

### MultiPromptProvider
- Location: `ComfyUI-LTXVideo/looping_sampler.py:943`
- Assigns one prompt per temporal tile for LTXVLoopingSampler.
- Our `TimestampPromptSchedule` serves the same role but for TensorLoop.

### LTXVSparseTrackEditor
- Location: `ComfyUI-LTXVideo/sparse_tracks.py:65`
- Defines motion trajectories as sparse point tracks (x, y coordinates
  at specific frames). The model follows these trajectories during generation.
- Could theoretically be used to enforce character positions across iterations,
  but the tracks apply within a single sampling pass, not across loop iterations.
- Not directly useful for our multi-frame conditioning goal.

## Recommendation

**Phase 1** (done): `KeyframeImageSchedule` for per-iteration guide switching.
**Phase 2** (future): Chain a second `LTXVAddLatentGuide` in the subgraph for
start+end guides per window. Use `KeyframeImageSchedule` with `blend_seconds > 0`
to provide both current and next keyframe images as the two guides.
