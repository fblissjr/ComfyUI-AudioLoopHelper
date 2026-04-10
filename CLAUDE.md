# ComfyUI-AudioLoopHelper

## TLDR

4 ComfyUI nodes that automate loop timing for full-length music video
generation with LTX 2.3. The main node (AudioLoopController) reads audio
duration from the tensor, computes stride from window + overlap, outputs
start_index / should_stop / audio_duration / iteration_seed / stride_seconds /
overlap_frames / overlap_latent_frames. No manual constants to keep in sync.

## Architecture

Single file: `nodes.py`. Uses ComfyUI's extension API (`ComfyExtension`,
`io.ComfyNode`). Entry point: `comfy_entrypoint()`.

5 nodes:
- `AudioLoopController` -- core: start_index, should_stop, audio_duration, iteration_seed, stride_seconds, overlap_frames, overlap_latent_frames
- `TimestampPromptSchedule` -- per-iteration prompt from timestamp ranges, with blend support
- `ConditioningBlend` -- lerps two conditioning tensors for smooth prompt transitions (works with LTX Gemma 3 and CLIP)
- `AudioLoopPlanner` -- displays iteration timeline for planning
- `AudioDuration` -- extracts duration/sample_rate from audio tensor

Helper functions:
- `_audio_duration(audio)` -- shared duration extraction (used by 3 nodes)
- `_parse_timestamp(ts)` -- "M:SS" or bare seconds to float
- `_format_timestamp(seconds)` -- float to "M:SS" (preserves sub-second)
- `_parse_schedule(schedule)` -- multiline schedule text to entries list
- `_match_schedule(entries, time)` -- find matching prompt for a timestamp
- `_match_schedule_with_next(entries, time, blend_seconds)` -- returns (prompt, next_prompt, blend_factor)

## Key patterns

- AUDIO type: `{"waveform": Tensor, "sample_rate": int}`.
  Duration = `waveform.shape[-1] / sample_rate`.
  Note: the comfy_api TypedDict says `sampler_rate` but actual code uses `sample_rate`.
- Stride is computed internally: `window_seconds - overlap_seconds`.
  overlap_frames is computed as `round(overlap_seconds * fps)`.
  User sets overlap once; stride and overlap_frames propagate via outputs.
- start_index is clamped so at least 0.5s of audio always remains after
  trimming. This prevents the mel spectrogram crash on the final loop
  iteration (TensorLoopClose checks should_stop AFTER the body executes).
- Timestamp parsing regex `_LINE_RE` handles colons in M:SS timestamps
  vs the colon separator between range and prompt.
- **Prompt changes cause style drift even at CFG 1.0.** Different text = different conditioning vectors = shifted generation space. Use ConditioningBlend with blend_seconds > 0 for gradual transitions, and keep the core subject description consistent across all schedule entries.
- LTX 2.3 uses Gemma 3 text encoder (NOT CLIP). Conditioning format is
  `[tensor, {"attention_mask": mask}]` with no pooled_output. Standard
  ConditioningAverage won't work -- use our ConditioningBlend instead.
- ConditioningBlend handles: sequence length alignment (zero-pads shorter),
  attention mask combining (OR), optional pooled_output blending (for CLIP compat).
- Workflow uses DualCLIPLoader + CLIPTextEncode nodes. Despite the names,
  these are Gemma 3 encoders (loaded via gemma_3_12B + ltx-2.3_text_projection).
  They produce `[tensor, {"attention_mask": ...}]` with no pooled_output.
- **Conditioning path**: Text encode output MUST go through LTXVConditioning
  (node 1587, frame_rate=25) before reaching Extension #843 positive input.
  Without frame_rate metadata, positive conditioning mismatches negative.
  Path: Text Encode → (ConditioningBlend if blending) → LTXVConditioning → #843.
- **Blending wiring**: TimestampPromptSchedule outputs STRING, not CONDITIONING.
  Blending requires two text encode nodes (both from same DualCLIPLoader) inside
  the loop body: prompt → encode A → conditioning_a, next_prompt → encode B →
  conditioning_b, blend_factor → ConditioningBlend. Without blending, one encode suffices.
- Nodes that need per-iteration evaluation (TimestampPromptSchedule,
  ConditioningBlend, text encoders) must be inside the loop body (between
  TensorLoopOpen and TensorLoopClose in the dependency graph) to be cloned
  each iteration.
- AudioLoopPlanner runs once (outside the loop). It uses a closed-form
  formula matching AudioLoopController's stop condition.
- overlap_frames feeds into the extension subgraph (Node 843) where it
  controls how many tail frames are extracted from previous_images as
  guide context AND how many leading frames are trimmed from new output
  to avoid duplication.

## ComfyUI gotchas learned the hard way

- **Workflow JSON has two link representations**: node body `"link"` fields AND the `"links"` array. Both must be updated when editing JSON or wires break on reload.
- Link array format: `[link_id, source_node, source_output, target_node, target_input, type]`
- Node `"mode": 0` = active, `"mode": 4` = bypassed.
- **PrimitiveNode cannot feed DynamicCombo sub-inputs** (e.g., `mode.iterations` on TensorLoopOpen). Set values directly on the widget instead.
- **ComfyMathExpression rejects boolean results** (raises ValueError). Use `int(expr)` wrapper, or use KJNodes `SimpleCalculatorKJ` which outputs BOOLEAN natively.
- **Graph expansion preview limitation**: preview/display nodes connected to TensorLoopOpen outputs only show first-pass values. Cloned iterations produce correct values but those go to cloned preview nodes you can't see.
- **TensorLoopClose checks should_stop AFTER the loop body executes.**
  The final iteration's body runs even when should_stop=True. Any node
  in the loop body must handle edge-case inputs (e.g., near-empty audio)
  gracefully. AudioLoopController clamps start_index to prevent this.
- **mask=0 means "fixed context" in LTX noise masks.** Audio latent with
  mask=0 keeps the real encoded song. Setting mask=1 tells the sampler to
  regenerate audio from noise, destroying lip sync. Verify mask semantics
  before changing any LTXVAudioVideoMask wiring.
- **Deleting PrimitiveNodes breaks subgraph wiring.** PrimitiveNodes override widget values via a special mechanism. Delete them first, then rewire the freed inputs from the component input panel.
- **After changing a node's define_schema(), users must delete and re-add the node in the UI.** JSON slot indices are baked at save time. Editing JSON slot numbers manually is fragile -- ComfyUI routes by slot index, not name.
- **Removing a subgraph component input shifts all higher slot indices.** Internal links referencing `origin_slot` must be decremented. Miss one and wires silently disconnect.
- **Always validate workflow JSON after programmatic edits:** `python3 -c "import json; json.load(open('file.json'))"`
- **Scrub workflows before open-sourcing:** filenames, absolute paths, UUIDs, image previews, videopreview fullpath/filename, creative prompts, clipspace references.
- Pyright `reportIncompatibleMethodOverride` on `execute()` methods is a false positive -- standard ComfyUI node API pattern.

## LTXVLoopingSampler limitation

LTXVLoopingSampler (ComfyUI-LTXVideo) does NOT support Audio-Visual latents.
It throws `ValueError: LoopingSampler currently does not support Audio Visual latents`.
For music video workflows that need audio conditioning for lip sync, use
TensorLoopOpen/Close with the extension subgraph instead.
ScheduleToMultiPrompt node is kept for future use if AV support is added.

## Video/Audio VAE temporal conversion

- **Video VAE**: First pixel frame → own latent frame, then 8 pixels per latent.
  Formula: `latent = (pixel - 1) // 8 + 1`. NOT `pixel // 8`.
  Pixel frames must follow 8n+1 (1, 9, 17, 25, 497...).
- **Audio VAE**: 25 latents/second, 1D, completely independent of video latent
  temporal dimension. They live in separate NestedTensor sub-tensors.
- Using `pixel // 8` instead of `(pixel - 1) // 8 + 1` caused the v0409
  sync bug: 25 pixels → 3 latent frames (wrong) vs 4 (correct).

## Resolution and latent volume

- **Latent volume limit**: `(width/32) * (height/32) * ((frames-1)/8 + 1)` should
  stay below ~15,000-20,000. Exceeding causes artifacts, grid patterns, color loss.
- **832x480 at 497 frames**: 26*15*63 = 24,570 -- already at the edge. Don't increase
  resolution without reducing frame count per window.
- **Higher resolution improves motion/lip-sync/audio quality** but costs more VRAM
  and risks latent volume overflow. 720p+ with 48-50fps gives smoother motion.
- **Portrait (vertical) resolutions are unstable** -- keep height < 1600px.
  Landscape and square work best.
- **Two-stage approach is the recommended workaround**: generate at lower res (720p),
  then spatial latent upscale to 1080p+. This is what LTX-Desktop and native LTX-2
  both do. See `docs/upscale_guide.md` and `internal/analysis/ltx23_gaps_analysis.md`.
- For our loop workflow: each window is 497 frames at 832x480. Changing resolution
  requires adjusting window_seconds or temporal_tile_size to stay under the limit.

## Subgraph editing

- ALWAYS use WorkflowEditor from `internal/scripts/workflow_utils.py` for
  subgraph modifications. Manual JSON surgery on subgraphs has failed
  repeatedly (stale slots, broken links, shifted indices).
- Use the `/workflow-edit` skill which documents the full API.
- LTXVSelectLatents operates in LATENT frame space (pixel_frames // 8).
  25 pixel overlap = 3 latent frames. Use AudioLoopController `overlap_latent_frames` output.

## How image guides actually work in LTX 2.3

Guide strength does NOT control how much the image influences style.
It controls the denoise mask (noise addition), which is only one of three
layers. Text conditioning operates on a separate, unattenuated pathway:

```
1. Cross-attention (text → all tokens)  ← ALWAYS FULL STRENGTH, no per-guide control
2. Self-attention (guide ↔ generated)   ← controlled by attention_strength (default 1.0)
3. Denoise mask (noise addition)        ← controlled by guide strength (1.0 = no noise)
```

- strength=1.0 → denoise_mask=0.0 → guide frames spatially frozen
- BUT cross-attention still pulls style/appearance toward text description
- Guides are CONCATENATED to the latent sequence (extra frames at the end),
  not blended at the target index. keyframe_idxs tells RoPE their logical position.
- This is why changing text causes style drift even at guide strength 1.0:
  the guide anchors composition, but text controls style via cross-attention.
- The right fix for style consistency is keeping text aligned (consistent
  prompts + ConditioningBlend), not increasing guide strength.

Source: ComfyUI-LTXVideo/latents.py (LTXVAddLatentGuide),
comfy_extras/nodes_lt.py (append_keyframe), comfy/ldm/lightricks/model.py
(per-reference attention masking).

## Extension subgraph (Node 843)

The "extension" group node inside the loop contains the per-iteration
sampling pipeline. Key internals:

- GetImageRangeFromBatch (615) -- extracts last overlap_frames from previous_images
- VAEEncode (614) -- encodes those tail frames to latent (continuity guide)
- VAEEncode (1520) -- encodes init_image to latent (scene anchor guide)
- LTXVAddLatentGuide (1519) -- merges conditioning + both guides into latent
- LTXVConcatAVLatent (583) -- adds audio latent
- CFGGuider (644) -- packages for sampling (cfg=1.0, NAG does guidance)
- SamplerCustomAdvanced (573) -- generates new frames
- GetImageRangeFromBatch (1509) -- trims first overlap_frames, keeps only new content

Fixes applied across workflow versions:
- v0407: added LTXVConditioning (Node 1587, frame_rate=25) between
  conditioning source and the subgraph input so positive conditioning
  gets frame_rate metadata matching the negative.
- v0408: added post-loop 2x spatial upscaler chain (bypassed by default,
  nodes 1589-1591/1597 mode=4). VAE round-trip causes blurriness; needs
  a different approach (per-iteration latent upscale or external post-processing).
- v0408: AudioLoopController start_index clamp to prevent mel crash on short audio.

## LTX 2.3 prompt format

- LTX 2.3 is distilled -- CFG=1.0 by default (NAG handles guidance, not CFG).
- Prompts are i2v (image-to-video) style: describe changes from the init_image, not the full scene.
- Start with `Style: cinematic.` (or omit if init_image establishes style).
- Use present-progressive verbs: "is singing," "is walking."
- Include audio descriptions inline with visuals (LTX 2.3 is audio-video joint).
- No meta-language: no "The scene opens with...", no timestamps, no cuts.
- Camera motion only when intended. Keywords: `static camera`, `dolly in/out/left/right`, `jib up/down`, `focus shift`.
- **Avoid dolly out** -- breaks limbs and faces. Use static camera with lighting shifts for visual variation.
- **i2v rule: describe only changes from the init_image.** Re-describing the setting causes the model to "restart" the scene.
- **Two-person scenes: always "singing together."** Don't direct male vs female vocals -- audio conditioning handles it.
- Full system prompts for i2v and t2v: `docs/ltx23_prompt_system_prompts.md`
- Prompt creation guide with variation patterns: `docs/prompt_creation_guide.md`
- For prompt scheduling: keep core subject identical across all entries, vary only framing/camera/lighting.

## Dependencies

Companion custom nodes (not imported, just used alongside in workflows):
- ComfyUI-NativeLooping_testing -- TensorLoopOpen/Close. Don't fork; graph expansion is deeply coupled to ComfyUI execution engine. Likely headed for core inclusion.
- ComfyUI-LTXVideo -- LTXVAddLatentGuide, LTXVCropGuides, LTXVPreprocess, looping sampler
- ComfyUI-KJNodes -- Set/Get nodes, FloatConstant, LTX2_NAG, LTXVImgToVideoInplaceKJ, ImageResizeKJv2
- ComfyUI-VideoHelperSuite -- VHS_VideoCombine
- ComfyUI-MelBandRoFormer -- vocal separation

## Audio analysis

- Use `scripts/analyze_audio.py` for energy timeline and structure detection.
- Use ffmpeg astats for RMS levels, NOT Python wave module (produces compressed/misleading values).
- `--trim N` offsets timestamps for node 567 start_index.
- MelBandRoFormer separates vocals/instruments only. No male/female distinction.

## Testing

No comfy_api outside ComfyUI runtime. Test parsing logic inline:
```bash
uv run python -c "
import re
# paste _parse_timestamp, _parse_schedule, _match_schedule
# run assertions
print('pass')
"
```

## Editing workflow JSON (subgraphs)

Top-level links use array format: `[id, src_node, src_slot, tgt_node, tgt_slot, type]`
Subgraph internal links use dict format: `{id, origin_id, origin_slot, target_id, target_slot, type}`
Subgraph definitions live at `wf['definitions']['subgraphs'][0]` with keys:
`nodes`, `links`, `inputs` (with `linkIds` arrays), `outputs`, `widgets`.
Subgraph input distributor node ID is -10. Output collector is -20.
All three representations (top-level links, node link fields, subgraph linkIds)
must stay in sync or wires break on reload.
Use `internal/scripts/workflow_utils.py` for programmatic edits.

## LTX 2.3 audio-video alignment

- TrimAudioDuration (Node 567) start_index is song-dependent. It trims
  instrumental intro that doesn't contribute to lip sync. Set to 0 for
  songs that start with vocals, or skip seconds for instrumental intros.
- Audio and video durations must match: 497 frames / 25fps = 19.88s audio.
  These are manually synced via FloatConstant (688) and PrimitiveNode (526).
- LTXVAudioVideoMask (Node 606): audio_start_time and audio_end_time are
  BOTH wired to window_size_seconds (19.88). This is intentional -- it creates
  an empty mask range (start=end), so audio stays fixed as the encoded song.
  The sampler generates video guided by real audio, not regenerated audio.
  DO NOT change this wiring.

## DynamicCombo widget format

LTXVImgToVideoInplaceKJ (and similar multi-input nodes) serialize
widgets as: `[num_items, strength_1, strength_2, ..., index_1, index_2, ...]`
Strengths come FIRST for all items, THEN indices. NOT interleaved.
Example: `['2', 1.0, 0.5, 0, -1]` = 2 images, strengths [1.0, 0.5], indices [0, -1].
Getting this wrong silently misconfigures the node (wrong strength/index mapping).

## Debugging workflow regressions

Compare against a known-working workflow JSON (keep copies in internal/scratch/).
When multiple settings differ, change ONE at a time and test. Do not batch changes.
Run `internal/scripts/test_workflow_integrity.py` after every programmatic edit.

## Upscaling

- Upscale is a SEPARATE workflow, not part of the loop workflow.
- Previous attempt to integrate upscaling into the loop failed: VAE round-trip
  quality loss + refinement sampler OOM at 24GB. See internal/postmortem_v0408_session.md Issue 4.
- Correct approach (from RuneXX 3-pass): stay in latent space. Load video → VAEEncode (once) →
  LTXVLatentUpsampler (2x) → 3-step refinement sampler → VAEDecodeTiled. Never leave latent
  space between upscale and refinement.
- Model: `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- Refinement sigmas: [0.85, 0.725, 0.4219, 0.0] (3 steps). Drop to 2 or 1 if OOM.
- Guide: `docs/upscale_guide.md`

## Workflow docs

- `example_workflows/native-audio-looping-music-video_v0409_latent.json` -- UNTESTED latent-space loop (no VAE round-trip, likely has issues)
- `example_workflows/native-audio-looping-music-video_v0408.json` -- legacy TensorLoop (per-iteration VAE round-trip)
- `example_workflows/upscale-loop-output.json` -- separate upscale workflow (when built)
- `coderef/origiltx23_long_loop_extension_test.json` -- original pre-scheduler workflow
- `coderef/RuneXX_LTX-2.3-Workflows/` -- reference LTX 2.3 workflows (3-pass upscale pattern)
- `docs/workflow_pipeline_trace.md` -- end-to-end pipeline trace
- `docs/nag_technical_reference.md` -- LTX2_NAG technical documentation
- `docs/prompt_creation_guide.md` -- prompt writing guide with variation patterns
- `docs/ltx23_prompt_system_prompts.md` -- official i2v/t2v system prompts
- `docs/upscale_guide.md` -- upscale workflow build guide
- `docs/ltxv_looping_sampler_settings.md` -- LTXVLoopingSampler reference (VIDEO-ONLY, no AV latent support)
- `docs/latent_loop_build_guide.md` -- build guide for LTXVLoopingSampler (video-only, not for music video)
- `docs/subgraph_latent_rework_guide.md` -- how the latent rework was done
- `internal/analysis/ltx23_gaps_analysis.md` -- capability gaps across LTX-Desktop, LTX-2, ComfyUI-LTXVideo
- `internal/postmortem_v0408_session.md` -- debugging history (6 issues with fixes)
