Last updated: 2026-05-10

# Upscale guide

Take a loop-rendered music video at 832×480 and upscale to 1664×960
via LTX 2.3's official spatial upscaler model, then refine with a
3-step low-σ pass to sharpen detail without hallucinating. Audio is
preserved unchanged.

Workflow: `internal/workflows/upscale_loop_output.draft.json` (gated
on first-render validation before promotion to `example_workflows/`).
Built by: `scripts/build_upscale_workflow.py`. Re-running the build
script re-emits the draft from constants; chain
`scripts/apply_trim_image_batch_to_audio.py` afterward to splice the
loop-audio-overshoot fix back in.

## When to use this

- Loop output is content-correct but needs sharpness for screen
  playback or compression headroom (1080p target).
- The visible region of interest (faces, text, textures) is rate-limited
  on 832×480; upscaling in latent space preserves style without the
  photoreal-sharpening artifacts you'd get from a pixel-domain upscaler.

If you need to regenerate content (different prompts, different cuts),
this is not the tool — use retake (`docs/guides/retake_guide.md`) or
re-render the loop.

## Why the input is a `.latent` file, not your loop's mp4

Loading the rendered mp4 and re-encoding via `VAEEncode` materializes
the entire image batch in pixel space first — ~16 GB for a 4-minute
song at 832×480 fp32. That OOMs on 24 GB before the upsampler runs.
The upscale workflow reads the loop's assembled video latent directly
via `LoadLatent` (~855 MB for the same content; 20× reduction). Bonus:
no decode → re-encode VAE round-trip, so no detail loss in the hand-off.

## The 4-step chain

### 1. Apply the run-id layout (one-time, per workflow you use)

Adds a per-render output folder + a bypassed `SaveLatent` toggle to
every loop workflow.

```bash
uv run --group dev python scripts/apply_run_id_layout.py
```

Re-running is a no-op (idempotent). Use `--revert` to remove. After
this, your loop workflow has a node titled **"Save assembled latent
(toggle)"** that's currently bypassed.

### 2. Enable the SaveLatent toggle, then render your loop

In the ComfyUI UI:

- Find the **"Save assembled latent (toggle)"** node (greyed-out by
  default).
- Right-click → **Set Mode → Always** (or `Ctrl+M` with the node
  selected). The greyed look goes away.
- Queue your normal loop render.

The render writes the usual mp4 outputs PLUS the assembled video latent
under:

```
<output>/audio-loop-music-video_latent/<timestamp>/latents/segment_00001_.latent
```

Toggle the node back to **Bypass** (`Ctrl+M` again) once you have the
`.latent` — you don't need to keep saving it on every subsequent loop run.

### 3. Move the `.latent` into ComfyUI's input directory

`LoadLatent` reads from ComfyUI's **input** dir, not the output dir
where your loop wrote.

Use the helper script — it finds the most recent `.latent` for the
named workflow and copies under a deterministic filename so the
upscale workflow's `LoadLatent` widget always picks up the same name:

```bash
# Set both dirs once per shell (or pass --output-dir / --input-dir)
export COMFYUI_OUTPUT_DIR=/path/to/comfy/output
export COMFYUI_INPUT_DIR=/path/to/comfy/input

uv run --group dev python scripts/promote_latent_for_upscale.py audio-loop-music-video_latent
# → copies <output>/audio-loop-music-video_latent/<latest_timestamp>/latents/segment_00001_.latent
#   to <input>/assembled_latent.latent
```

Use `--dry-run` to preview without copying, `--dest-name foo.latent`
to override the destination filename. Manual `cp` works fine too if
you'd rather:

```bash
cp <output>/audio-loop-music-video_latent/<timestamp>/latents/segment_00001_.latent \
   <comfyui_input_dir>/assembled_latent.latent
```

### 4. Run the upscale workflow

Open `internal/workflows/upscale_loop_output.draft.json` in ComfyUI.
Set the widgets:

- **LoadLatent** (`#22`, "Load assembled video latent"): pick the
  `.latent` you just moved.
- **LoadAudio** (`#23`, "Source audio (same as loop)"): pick the same
  source mp3 you used to drive the loop.
- **CLIPTextEncode** (`#7`, "Positive prompt (match loop)"): paste the
  same prompt you used for the loop. (For the partial-refine pass at
  σ=0.85 the prompt has minimal influence; an empty string also works
  but matching the loop is safer.)
- **RunIdPrefix** (`#26`): leave the defaults (`workflow_name` is
  set to `upscale_loop_output` so outputs cluster under their own
  per-run folder).

Queue. Watch VRAM — should peak around 16-18 GB on a 24 GB card during
the refine sampler; tiled VAE decode is single-tile by default. If it
OOMs you can drop `LTXVTiledVAEDecode #25`'s `tile_size` widgets from
`[1, 1, 1]` to `[2, 2, 1]` (3× slower cold pass but works on ≤16 GB).

Output lands at:

```
<output>/upscale_loop_output/<timestamp>_00001-audio.mp4
```

## Verifying the output

After the render completes:

```bash
ffprobe -v error -show_entries stream=codec_type,duration -of default=noprint_wrappers=1:nokey=0 <upscaled>.mp4
```

Expected: `container.duration == video.duration == audio.duration`
exactly. If video exceeds audio, the `TrimImageBatchToAudio` node
(F14) isn't wired — re-run `apply_trim_image_batch_to_audio.py`.

## Topology summary

```
LoadLatent (assembled video latent from loop's toggled SaveLatent)
  ↓
LTXVLatentUpsampler (×2 spatial: 104×60 → 208×120 latent)
  ↓
LTXVConcatAVLatent (re-attach empty audio latent, sized from
                    LatentFrameCount(load_latent).pixel_frames)
  ↓
SamplerCustomAdvanced (3-step partial refine, σ-tail
                       [0.85, 0.7250, 0.4219, 0.0], euler, CFG=1.0)
  ↓
LTXVSeparateAVLatent → LTXVCropGuides → LTXVTiledVAEDecode (1664×960)
  ↓
TrimImageBatchToAudio (clip to floor(audio.duration * fps))
  ↓
VHS_VideoCombine (mp4 + audio from LoadAudio)
```

Model chain through CFGGuider mirrors the canonical loop's perf/VRAM
patches: `UNETLoader → AudioLoopHelperSageAttention → LTXVChunkFeedForward → LTX2AttentionTunerPatch → CFGGuider`.
Widget values are byte-equal to the loop. Without them the refine
OOMs on 24 GB at 1664×960 × ~526 latent frames.

## Refinement step tuning

The sigma profile `[0.85, 0.7250, 0.4219, 0.0]` is corroborated by 3
independent third-party authors (RuneXX 3-pass, edit-anything, 10Eros
TripleSample). 3 steps = 4 sigma values; partial refine from σ=0.85
means only ~15% of the latent is denoised, so the upscaled content is
preserved.

If 3 steps OOM:

| Steps | Sigmas | Quality | VRAM |
|---|---|---|---|
| 3 | `[0.85, 0.7250, 0.4219, 0.0]` | Best | Highest |
| 2 | `[0.85, 0.4219, 0.0]` | Good | Less |
| 1 | `[0.85, 0.0]` | Acceptable | Minimal |

Drop step count by editing `ManualSigmas #19`'s widget.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `LoadLatent` widget shows no files | `.latent` not in ComfyUI's input dir | `cp` it there; see Step 3 |
| OOM at sampler step 1 | Model-chain patches missing | Re-run `build_upscale_workflow.py` to regenerate the draft |
| OOM at VAE decode | Tiled decode set to single-tile on a small card | Edit `LTXVTiledVAEDecode #25` widgets `[1,1,1]` → `[2,2,1]` |
| Saved mp4 has silence at end | `TrimImageBatchToAudio` not wired | `apply_trim_image_batch_to_audio.py` |
| Output looks bilinear-resized, not refined | `LTXVImgToVideoConditionOnly` somehow back in the topology | Re-run `build_upscale_workflow.py`; see `internal/analysis/i2v_v5_workflow_assessment.md` Issue A |
| Upscaled video has trailing seconds of stalled visuals | Loop-side overshoot (audio latent was truncated for those frames) | Image-batch trim clips them at the output; if you see them in the saved mp4, the F14 trim isn't applied (see row 4) |

## References

- `internal/design/upscale_workflow_design.md` — design rationale + sigma profile sourcing
- `internal/analysis/loop_audio_overshoot_analysis.md` — silence-at-end postmortem (the F14 trim that backstops this guide)
- `internal/analysis/i2v_v5_workflow_assessment.md` — `LTXVImgToVideoConditionOnly` trap (don't re-introduce)
- `scripts/build_upscale_workflow.py` — re-generates the draft from constants
- `scripts/apply_run_id_layout.py` — installs the per-render folder layout + SaveLatent toggle
- `scripts/apply_trim_image_batch_to_audio.py` — F14 audit pair
