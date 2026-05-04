# noise_mask semantics

Last updated: 2026-05-04

## Role

Per-frame flag on the LATENT dict that tells the LTX 2.3 sampler which positions to **regenerate** vs preserve as **fixed context**. `noise_mask=0` = locked; `noise_mask=1` = regenerate. Audio frames are always 0; video frames are 1; overlap context frames are 0. The asymmetry is what lets a single latent carry both modalities through one diffusion pass.

## Disambiguation

- `noise_mask` (sampler input) ≠ `attention_mask` (transformer input) ≠ `LTXVAudioVideoMask` (region-map node)
- mask absent (no `noise_mask` key) ≡ mask=1 everywhere (regenerate-all default)
- "audio is FROZEN" = audio frames carry mask=0; not the same as "audio is bypassed"

## Key facts

- Tensor lives at `latent["noise_mask"]`. Shape compatible with the latent's frame axis.
- Set per-iteration by `LTXVAudioVideoMask` (Node 606); stripped at iteration boundaries by `LatentContextExtract` and `LatentOverlapTrim` (`nodes.py::LatentContextExtract.execute`, `nodes.py::LatentOverlapTrim.execute`).
- `LTXVAudioVideoMask` uses `existing_mask_mode: "add"` — stale masks merge incorrectly. Stripping at boundaries is mandatory, not optional.
- Retake workflow inverts the convention for one window only: `LatentTemporalMask` writes mask=1 inside `[start, end]`, mask=0 outside.

## Setters and strippers

| Node | Mask behavior |
|---|---|
| `LTXVAudioVideoMask` (Node 606) | Sets `{audio:0, video:1, overlap:0}` per iteration |
| `LTXVImgToVideoInplaceKJ` | Sets frame-0 = 0, rest = 1 (initial render only) |
| `LatentContextExtract` | Strips mask after slicing tail of prior latent |
| `LatentOverlapTrim` | Strips mask after trimming overlap region |
| `LatentTemporalMask` | Sets `{inside [start,end]:1, outside:0}` (retake workflow) |
| `LTXVSelectLatents` (raw) | Preserves mask — **don't use directly**; wraps via `LatentContextExtract` / `LatentOverlapTrim` |
| `StripLatentNoiseMask` | Removed 2026-04-27 (zero workflow uses; auto-strip in wrappers) |

## Per-iteration flow

```
prior latent → LatentContextExtract → LTXVAudioVideoMask → LTXVAddLatentGuide → SamplerCustomAdvanced → LatentOverlapTrim → next iter
   (stale mask)        (strips)            (sets fresh)         (extends)             (reads)              (strips)
```

Boundary discipline (mandatory):
- in: strip prior latent's mask
- set: create fresh `{audio:0, video:1, overlap:0}` map
- out: strip before accumulation

## Decision table — writing a new loop-body node

| Node consumes latent? | Node produces latent? | Required action |
|---|---|---|
| yes | no  | strip on read (already done if upstream is `LatentContextExtract`/`LatentOverlapTrim`) |
| no  | yes | strip before emitting OR set explicitly |
| yes | yes | strip on read; strip or set on emit (never propagate stale) |

## Failure modes

| Symptom | Likely cause |
|---|---|
| Audio drift across iterations | Stale mask escaped a boundary strip; `LTXVAudioVideoMask` merged old + new |
| Heatmap-style frames in video | Audio visualization (spectrogram, energy curve) fed to video latent with mask=1 — sampler "denoised" it |
| Mis-merged region map | Prior mask present at `LTXVAudioVideoMask` input (`existing_mask_mode: "add"` collision) |
| Iter-seam discontinuity | Overlap context frames carry mask=1 instead of 0 |
| Final iter crashes | Unrelated — see `audio_loop_controller.md` `start_index` clamping |

Edge cases:
- `LTXVPreprocess img_compression=0` skips preprocessing AND leaves frozen-frame mask state. Use `img_compression=18` (Lightricks) or `35` (core).
- `LTXVAudioVideoMask` Node 606 wiring `audio_start_time = audio_end_time = window_size` is intentional — empty range keeps audio fixed. Don't change.
- `LTXVConcatAVLatent` is the audio-path boundary: encoded audio in, never an image. Feeding visualizations here is the heatmap-frame root cause.
- `attention_mask` is sage-attention's mask routing; LTX 2.3 cross-attn passes `attention_mask=None` (`comfy/ldm/lightricks/model.py::BasicTransformerBlock.attn2`). Sage `auto_mask_aware` defensive only — unrelated to `noise_mask` despite naming overlap.

## Audit + tests

- Audit: **none currently.** Stale-mask bugs caught only via runtime symptoms (above failure modes).
- Indirect coverage: `LatentContextExtract` and `LatentOverlapTrim` strip behavior tested in their respective unit tests.

## References

- `nodes.py::LatentContextExtract`, `nodes.py::LatentOverlapTrim`, `nodes.py::LatentTemporalMask` — strip / set implementations
- `docs/reference/pipeline_flow_latent.md` — full pipeline trace; "noise_mask Flow Explanation" section walks through one iteration
- `docs/reference/frame_planner_reference.md` — same "set once, propagate everywhere" pattern for dimensions
- `docs/reference/audio_loop_controller.md` — emits `overlap_latent_frames` count consumed by `LatentContextExtract`/`LatentOverlapTrim`
- `docs/analysis/audio_in_prompt_research.md` — why audio is frozen (rationale `noise_mask=0` enforces)
- `docs/analysis/nag_object_patches_offload_asymmetry.md` — adjacent constraint on what enters the loop body
- `docs/reference/timestamp_prompt_schedule_batch_encode.md` — solves the same class of "what must stay outside the loop body" question for CONDITIONING
- `docs/reference/_atomic_note_template.md` — concept-note variant template
