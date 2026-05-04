# noise_mask semantics

Last updated: 2026-05-04

## Role

`noise_mask` is the per-frame flag that tells the LTX 2.3 sampler which latent positions to **regenerate** and which to **preserve as fixed context**. `noise_mask=0` means "this frame is locked, sample around it"; `noise_mask=1` means "regenerate this from noise." The audio-loop pipeline relies on this asymmetry to keep audio frozen across iterations while video is generated fresh — the same latent tensor carries both modalities through one diffusion pass without the sampler corrupting the audio half.

A wrong or stale `noise_mask` is the failure class that produces audio drift, video discontinuities at iteration seams, and the heatmap-frame bug when audio visualizations leak into the video latent. Most loop-body bugs in this codebase trace back to a `noise_mask` not being where it should be.

## Key facts

- **`noise_mask=0` = fixed context.** Audio frames are always 0 across the entire loop — that's why "audio is FROZEN."
- **`noise_mask=1` = regenerate.** New video frames in each iteration carry 1.
- **Overlap context frames carry 0** even though they're video — they were generated in the previous iteration and need to stay byte-stable for continuity.
- **The mask is a tensor on the LATENT dict**, accessed as `latent["noise_mask"]`. Shape matches the latent's frame axis.
- **Stale masks corrupt downstream operations.** `LTXVAudioVideoMask` uses `existing_mask_mode: "add"` — if a previous mask is still attached, the merger produces a corrupted region map.

## How it works

The sampler sees a latent of shape `[B, C, T, H, W]` and a `noise_mask` of compatible shape. For each position where mask=1, it adds noise and runs the denoising chain; for mask=0, it skips noise injection and treats the existing value as ground truth that the attention layers can condition on.

In our pipeline, the typical iteration looks like:

```
LatentContextExtract → LTXVAudioVideoMask → LTXVAddLatentGuide → SamplerCustomAdvanced → LatentOverlapTrim
        ↓                      ↓                                          ↓                      ↓
  strips noise_mask     creates fresh mask                    sampler reads mask          strips noise_mask
                        (audio=0, video=1,                                                  (clean for concat)
                         overlap=0)
```

Each iteration: prior latent's stale mask is **stripped** at the boundary, a fresh mask is **created** by `LTXVAudioVideoMask` for the new window, the sampler **reads** it, then the output's mask is **stripped again** before accumulation. Stripping is mandatory at both seams.

## Setters and strippers

| Node | Behavior | Use it for |
|---|---|---|
| `LTXVAudioVideoMask` (Node 606) | **Sets** mask: audio frames = 0, video frames = 1, overlap frames = 0 | Per-iteration mask creation |
| `LTXVImgToVideoInplaceKJ` | **Sets** mask: encoded init frame at position 0 = 0, rest = 1 | Initial render only |
| `LatentContextExtract` | **Strips** mask after slicing the tail of the prior latent | Loop-body input boundary |
| `LatentOverlapTrim` | **Strips** mask after trimming the overlap region | Loop-body output boundary |
| `LatentTemporalMask` | **Sets** mask for retake: regenerate only inside `[start, end]` window | Retake workflow |
| `StripLatentNoiseMask` | (Removed 2026-04-27) — `LatentContextExtract` and `LatentOverlapTrim` auto-strip | — |

`LatentContextExtract` and `LatentOverlapTrim` exist specifically to wrap raw latent slicing AND strip the mask in one operation. **Don't use raw `LTXVSelectLatents`** — it preserves the mask, which then mis-merges in the next `LTXVAudioVideoMask` call.

## Gotchas

- **`LTXVPreprocess img_compression=0` skips preprocessing AND leaves frozen-frame mask state.** Use `img_compression=18` (Lightricks default) or `35` (core).
- **The audio path is sacred.** Never feed audio visualizations (spectrograms, energy curves) into the video latent with `noise_mask=1` — the sampler will then "denoise" them, producing heatmap-style frames in the output. The `LTXVConcatAVLatent` boundary is non-negotiable: encoded audio in, never an image.
- **`LTXVAudioVideoMask` Node 606's `audio_start_time = audio_end_time = window_size` wiring is intentional** — empty range keeps audio fixed. Don't change it expecting per-iteration audio regeneration.
- **The retake workflow flips the convention** for one specific window: inside `LatentTemporalMask`'s `[start, end]`, mask=1; outside, mask=0. This is the *only* place in the pipeline where some video frames intentionally carry mask=0 mid-loop. Don't generalize the retake pattern.
- **A latent with no `noise_mask` key** is treated as "regenerate everything" by the sampler — equivalent to mask=1 everywhere. This is why stripping the mask at iteration boundaries works: downstream nodes that need a mask create one fresh from the surrounding logic; downstream nodes that don't need one default to whole-tensor regeneration safely.
- **Sage attention's `auto_mask_aware` mode is defensive only on current LTX workflows.** LTX 2.3 cross-attention passes `attention_mask=None` (verified at `comfy/ldm/lightricks/model.py:482`); the mask routing never fires in production. This is `noise_mask`-unrelated despite the naming overlap — `attention_mask` (sage) ≠ `noise_mask` (sampler).

## When this matters most

- **Designing a new loop-body node**: if it consumes or produces a latent, decide explicitly whether to strip, preserve, or set `noise_mask`. Default to strip on input boundary, strip on output boundary; only set if you're producing a new region map.
- **Debugging iter-over-iter drift**: trace the `noise_mask` value through one full iteration. The most common bug is a stale mask escaping a boundary it should have been stripped at.
- **Adding retake-style features**: `LatentTemporalMask` is the reference implementation for "regenerate a temporal subset." Don't reinvent — extend the pattern.

## References

- `nodes.py` — `LatentContextExtract` (~line 1741), `LatentOverlapTrim` (~line 1787), `LatentTemporalMask` (~line 1840)
- `docs/reference/pipeline_flow_latent.md` — full pipeline trace; the "noise_mask Flow Explanation" section walks through one iteration end-to-end
- `docs/reference/frame_planner_reference.md` — same "set once, propagate everywhere" pattern applied to dimension config
- `docs/analysis/audio_in_prompt_research.md` — why audio is frozen in this workflow (the rationale `noise_mask=0` enforces)
- `docs/analysis/nag_object_patches_offload_asymmetry.md` — adjacent constraint on what can/cannot enter the loop body
- `docs/reference/_atomic_note_template.md` — this doc follows the concept-note variant
