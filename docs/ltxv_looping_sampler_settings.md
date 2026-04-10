Last updated: 2026-04-10

# LTXVLoopingSampler Settings Reference

**NOTE: VIDEO-ONLY. Does not support AV latents for music video workflows.**

Settings for the LTXVLoopingSampler node. Maps each parameter to its
TensorLoop equivalent. Kept as reference for video-only use cases.

## Settings Table

| Setting | Value | Unit | Old workflow equivalent | Notes |
|---------|-------|------|----------------------|-------|
| temporal_tile_size | **497** | pixel frames | window_seconds=19.88 (497/25fps) | Size of each generation window |
| temporal_overlap | **25** | pixel frames | overlap_seconds=1.0 (25/25fps) | Context frames shared between tiles. Increase to 50 for more coherence. |
| guiding_strength | **1.0** | 0.0-1.0 | init_image guide strength=1.0 | How strongly guiding latents (IC-LoRA) condition. |
| temporal_overlap_cond_strength | **0.5** | 0.0-1.0 | No equivalent (new) | How strongly previous tile's tail frames condition the next tile. Key coherence dial. |
| cond_image_strength | **1.0** | 0.0-1.0 | first_frame_guide_strength=1.0 | Init image conditioning strength. |
| horizontal_tiles | **1** | count | Same | No spatial tiling at 832x480. Increase for higher res. |
| vertical_tiles | **1** | count | Same | No spatial tiling at 832x480. |
| spatial_overlap | **1** | tiles | N/A | Overlap between spatial tiles. Only matters if h/v tiles > 1. |
| adain_factor | **0.0** | 0.0-1.0 | No equivalent (new) | Adaptive Instance Normalization per tile. Prevents color/style drift. Try 0.1-0.3 if colors oversaturate. |
| guiding_start_step | **0** | step | No equivalent | Step at which guide latents activate. 0 = from start. |
| guiding_end_step | **1000** | step | No equivalent | Step at which guide latents deactivate. 1000 = effectively always on. |
| optional_cond_image_indices | **"0"** | comma-sep | init_image at index 0 | Which frames get image conditioning. "0" = first frame only. |

## Tuning Guide

### temporal_overlap_cond_strength (most impactful new parameter)

Controls how strongly the overlap region from the previous tile influences
the next tile's generation. This is the primary coherence dial.

| Value | Effect |
|-------|--------|
| 0.0 | No conditioning from previous tile. Each tile generates independently. Maximum drift. |
| 0.3 | Light conditioning. More creative freedom per tile, some continuity. |
| **0.5** | **Default. Balanced coherence and variation.** |
| 0.8 | Strong conditioning. Very smooth transitions, less variation. |
| 1.0 | Maximum conditioning. Previous tile heavily constrains next. May feel repetitive. |

Start at 0.5. Increase if you see discontinuities between tiles.
Decrease if the video feels too "locked" and lacks variation.

### adain_factor (color/style drift prevention)

AdaIN normalizes each new tile's statistics to match the first tile.
Without it, colors and contrast can gradually shift over many tiles.

| Value | Effect |
|-------|--------|
| **0.0** | **Default. No normalization. Fine for short videos (< 1 min).** |
| 0.1 | Light normalization. Subtle correction. |
| 0.2-0.3 | Moderate. Good for 2+ minute videos. |
| 0.5+ | Strong. May flatten dynamic range. Use sparingly. |

### temporal_overlap (context frames)

| Value | Frames | Seconds at 25fps | Effect |
|-------|--------|-------------------|--------|
| 16 | 16 | 0.64s | Minimum. Fast coverage but jarring transitions. |
| **25** | **25** | **1.0s** | **Default. Good balance.** |
| 50 | 50 | 2.0s | More context. Smoother but each tile covers less new ground. |
| 80 | 80 | 3.2s | Maximum. Very smooth but slow progress. |

### temporal_tile_size

| Value | Frames | Seconds at 25fps | Notes |
|-------|--------|-------------------|-------|
| 257 | 257 | 10.3s | Shorter windows. Faster per-tile, more tiles needed. |
| **497** | **497** | **19.88s** | **Default. Matches LTX 2.3 optimal window.** |
| 745 | 745 | 29.8s | Longer windows. Fewer tiles but heavier per-tile. May degrade at edges. |

## Comparison: LTXVLoopingSampler vs TensorLoop

| Aspect | TensorLoop (old) | LTXVLoopingSampler (new) |
|--------|-------------------|--------------------------|
| VAE decodes per iteration | 1 (lossy) | 0 (all latent) |
| VAE encodes per iteration | 1 (context frames) | 0 (all latent) |
| Total VAE decodes | N+1 | 1 |
| Quality over iterations | Degrades (round-trip) | Constant |
| Overlap handling | Manual subgraph (nodes 615, 1509) | Built-in temporal_overlap |
| Color drift prevention | None | adain_factor |
| Overlap strength control | None (fixed) | temporal_overlap_cond_strength |
| Per-tile prompts | TimestampPromptSchedule + CLIPTextEncode in loop | ScheduleToMultiPrompt + MultiPromptProvider (upfront) |
| Stop signal | AudioLoopController should_stop | Not needed (latent pre-sized for full audio) |
