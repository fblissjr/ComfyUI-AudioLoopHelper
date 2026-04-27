Last updated: 2026-04-23 (trimmed + moved from docs/guides/latent_loop_build_guide.md — this workflow is video-only; we don't recommend building it for music video)

# LTXVLoopingSampler reference (video-only; not recommended for music video)

> **Do not build this for music video.** LTXVLoopingSampler cannot
> consume AV latents (frozen-audio + video NestedTensor). For
> music-video workflows, use the TensorLoop architecture in
> `audio-loop-music-video_latent.json` — start at
> `docs/architecture_overview.md` and drill into
> `docs/reference/pipeline_flow_latent.md` for the node-by-node.
> The AV incompatibility is architectural; the reasoning is at
> `docs/analysis/ltx23_gaps_analysis.md` (2 root blockers, 3
> type-system cascades).

This doc describes what an LTXVLoopingSampler-based workflow looks
like, and when it's the right tool, for readers who are exploring
the sampler for **video-only** generation (no frozen audio track).

## When LTXVLoopingSampler is appropriate

- Long video generation (> single-window length) from a text prompt
  or init image only.
- No audio track is being conditioned on (pure T2V or I2V extended
  to long form).
- Spatial tiling is wanted for higher output resolution within a
  single VRAM budget.

## High-level architecture

```
[Model loading + patches] → [NAG] → patched model
[Init image] → [Resize] → [LTXVPreprocess] → [ImgToVideoInplace] → initial latent
[Text encode + conditioning] → [Guider]
[ScheduleToMultiPrompt — REMOVED 2026-04-27] → [MultiPromptProvider] → per-tile conditioning
(Class deleted; use TimestampPromptScheduleBatchEncode + ConditioningSelectByIteration instead.)
                                                        ↓
                       LTXVLoopingSampler ←─────────────┘
                              ↓
                         full-length latent (never decoded between tiles)
                              ↓
                       [VAEDecodeTiled] → images → [VHS_VideoCombine]
```

The distinctive piece is `LTXVLoopingSampler` itself — it handles
temporal chunking, spatial tiling, and overlap blending internally
in latent space. The sub-samplers it uses (`LTXVBaseSampler`,
`LTXVExtendSampler`) are the ComfyUI-LTXVideo building blocks.

## Why this architecture doesn't work for music video

The `LTXVLoopingSampler.execute()` check at `looping_sampler.py`
line 722 explicitly rejects NestedTensor AV latents with the
message "LoopingSampler currently does not support Audio Visual
latents."

The architectural obstacles are covered in
`docs/analysis/ltx23_gaps_analysis.md`, but in summary: (1) the
temporal-chunking loop iterates on video frame count, which is
~8× smaller than audio frame count in latent space — there's no
trivially-correct way to align tile boundaries across the two; (2)
the LTX 2.3 model's cross-attention was trained on joint AV
tokens, not tiled AV, so even if boundaries aligned the quality
story is unvalidated. Neither is a small fix.

## How this differs from the music-video architecture

| Dimension | LTXVLoopingSampler (video-only) | Music-video (TensorLoop + LTXVConcatAVLatent) |
|---|---|---|
| Long-form mechanism | Internal temporal chunking + spatial tiling | External loop (TensorLoop) around a whole-window sampler |
| AV handling | Not supported | `LTXVConcatAVLatent` bundles audio + video per iteration |
| Per-iteration inputs | Prompt schedule via `MultiPromptProvider` | Pre-encoded schedule via `TimestampPromptScheduleBatchEncode` + `ConditioningSelectByIteration` |
| VRAM profile | Fits large outputs via spatial tiling | One window at a time; spatial tiling not required |

## Pointers for a video-only build

If you're building a video-only LTXVLoopingSampler workflow:

- Upstream reference: the ComfyUI-LTXVideo repo's README and
  `looping_sampler.py`. Their docs are more up to date than this
  doc.
- Model patches we use that still apply: `PatchSageAttentionKJ`,
  `LTXVChunkFeedForward`, `LTX2AttentionTunerPatch`. These are
  generic LTX 2.3 performance patches — nothing audio-specific.
- Distilled sigma chain: if using the merged distilled-1.1 22B
  checkpoint, the 8-step linear-quadratic sigma schedule
  (`DISTILLED_SIGMAS`) still applies. See
  `docs/reference/sampler_reference.md` for the walkthrough.
- `LTX2_NAG` still applies but scale=11 default is too aggressive
  for distilled — dial to 3-7 per
  `docs/reference/nag_technical_reference.md`.

