# ComfyUI-AudioLoopHelper

## TLDR

ComfyUI nodes that automate loop timing and audio analysis for full-length
music video generation with LTX 2.3. The main node (AudioLoopController)
reads audio duration from the tensor, computes stride from window + overlap,
outputs start_index / should_stop / audio_duration / iteration_seed /
stride_seconds / overlap_frames / overlap_latent_frames.

## Architecture

Two files: `nodes.py` (core loop nodes) and `nodes_analysis.py` (runtime
audio analysis). Uses ComfyUI's extension API (`ComfyExtension`,
`io.ComfyNode`). Single entry point: `comfy_entrypoint()` in nodes.py
imports analysis nodes from nodes_analysis.py.

Core nodes (nodes.py):
- `AudioLoopController` -- core: 8 outputs (start_index, should_stop, audio_duration, iteration_seed, stride_seconds, overlap_frames, overlap_latent_frames, overlap_seconds). overlap_seconds (slot 7) auto-wires to LTXVAudioVideoMask video_start_time.
- `TimestampPromptSchedule` -- per-iteration prompt from timestamp ranges, with blend support
- `ConditioningBlend` -- lerps two conditioning tensors for smooth prompt transitions (works with LTX Gemma 3 and CLIP)
- `AudioLoopPlanner` -- displays iteration timeline for planning
- `AudioDuration` -- extracts duration/sample_rate from audio tensor

Analysis nodes (nodes_analysis.py, torchaudio only):
- `AudioPitchDetect` -- per-iteration F0 detection, vocal presence, male/female classification. Wire to MelBandRoFormer separated vocals for clean signal. Outputs FLOAT/BOOLEAN only.

Helper functions:
- `_audio_duration(audio)` -- shared duration extraction (used by 3 nodes)
- `_parse_timestamp(ts)` -- "M:SS" or bare seconds to float
- `_format_timestamp(seconds)` -- float to "M:SS" or "M:SS.ss" (preserves sub-second).
  NOT the same as `_fmt_ts()` in analyze_audio_features.py which truncates to
  whole seconds for schedule timestamps. Different semantics, intentionally separate.
- `_parse_schedule(schedule)` -- multiline schedule text to entries list
- `_match_schedule(entries, time)` -- find matching prompt for a timestamp
- `_match_schedule_with_next(entries, time, blend_seconds)` -- returns (prompt, next_prompt, blend_factor)
- **TimestampPromptSchedule only runs in loop iterations, NOT the initial render.**
  Initial render (0 to window_seconds) uses the static prompt on node 169.
  AudioLoopPlanner output shows this explicitly with an "Initial:" line.

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
- **Conditioning path**: Extension #843 positive/negative should come from
  Get_base_cond_pos/neg DIRECTLY, NOT through an extra LTXVConditioning node.
  Node 1587 (LTXVConditioning) was previously in this path but caused the
  initial render to lose lip sync -- ComfyUI's execution engine evaluates the
  conditioning graph and the extra LTXVConditioning corrupted the initial
  render's audio-video cross-attention. Node 1587 is now bypassed.
  The initial render's conditioning goes through node 164 (LTXVConditioning,
  frame_rate=25) which is sufficient.
  Path: Get_base_cond_pos → #843 positive (static mode).
  Path: Text Encode → (ConditioningBlend if blending) → #843 positive (scheduling mode).
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
- **Workflow JSON output slots must match nodes.py define_schema().** If you add
  an output to a node's schema, the workflow JSON won't have it until the user
  deletes and re-adds the node. Fix programmatically via WorkflowEditor.
- **After changing a node's define_schema(), users must delete and re-add the node in the UI.** JSON slot indices are baked at save time. Editing JSON slot numbers manually is fragile -- ComfyUI routes by slot index, not name.
- **Removing a subgraph component input shifts all higher slot indices.** Internal links referencing `origin_slot` must be decremented. Miss one and wires silently disconnect.
- **Always validate workflow JSON after programmatic edits:** `python3 -c "import json; json.load(open('file.json'))"`
- **Scrub workflows before open-sourcing:** filenames, absolute paths, UUIDs, image previews, videopreview fullpath/filename, creative prompts, clipspace references.
- Pyright `reportIncompatibleMethodOverride` on `execute()` methods is a false positive -- standard ComfyUI node API pattern.
- **Module constant ordering**: constants (`_SECTION_MODIFIERS`, `_LLM_SYSTEM_PROMPT`,
  etc.) must be defined BEFORE the functions that reference them. Python's late
  binding makes it work either way, but declare-before-use is the project convention.
- **ComfyUI execution engine evaluates downstream conditioning graphs before
  upstream sampling.** Adding nodes to the Extension's conditioning path (e.g.,
  extra LTXVConditioning) can corrupt the initial render even though the Extension
  runs AFTER the initial render. Always compare against a known-working workflow
  snapshot when debugging lip sync regressions.
- **Node 560 (VHS_VideoCombine, bypassed by default)**: Enable to preview the
  initial render in isolation. Decodes from VAEDecode #1318 (same path as the
  working v0408 output). Useful for debugging lip sync without loop iterations.
- **Keep known-working workflow snapshots in `internal/scratch/`.** When lip sync
  regresses, diff against the working snapshot to find structural changes.
  LTX-2_00032.json and LTX-2_00040.json are confirmed working (April 9, 2026).
- **torchaudio detect_pitch_frequency on silence gives false positives.** Always
  gate with an RMS energy check (< 0.005 threshold) before calling. AudioPitchDetect
  does this automatically. If adding new pitch-based nodes, replicate the gate.

## LTXVLoopingSampler AV incompatibility (settled)

LTXVLoopingSampler CANNOT support AV latents. 5 blocking architectural issues:
spatial tiling on 5D fails on 4D audio, temporal tiling uses video frame count
not audio 25Hz, weighted accumulation can't index-assign NestedTensor,
sub-samplers expect video-only, model forward requires both modalities
simultaneously for cross-attention. This is not a TODO -- it's fundamental.
TensorLoop is the correct approach for AV. Do not attempt to build an AV
looping sampler -- use two-stage upscale for resolution instead.
Full analysis: `docs/analysis/ltx23_gaps_analysis.md`
ScheduleToMultiPrompt node kept for video-only LTXVLoopingSampler use cases.

## Video/Audio VAE temporal conversion

- **Video VAE**: First pixel frame → own latent frame, then 8 pixels per latent.
  Formula: `latent = (pixel - 1) // 8 + 1`. NOT `pixel // 8`.
  Pixel frames must follow 8n+1 (1, 9, 17, 25, 497...).
- **Audio VAE**: 25 latents/second, 1D, completely independent of video latent
  temporal dimension. They live in separate NestedTensor sub-tensors.
- Using `pixel // 8` instead of `(pixel - 1) // 8 + 1` caused the v0409
  sync bug: 25 pixels → 3 latent frames (wrong) vs 4 (correct).

## Initial render path (critical for sync)

- TensorLoopOpen MUST receive the sampled initial render, NOT the raw
  image-embed latent from LTXVImgToVideoInplaceKJ.
- LTXVAddLatentGuide APPENDS guide frames to temporal dim (torch.cat dim=2).
  Sampler output latent has shape [B,C,63+N_guides,H,W], not [B,C,63,H,W].
- For LATENT workflows: initial render prepended via LatentConcat (dim=t)
  using CropGuides output (guide-stripped). TensorLoopOpen receives from
  SeparateAV directly (with guides, matching v0408 behavior).
- Correct latent path: #531 → #350 ConcatAV → #161 Sampler → #245 SeparateAV → #1539 TensorLoopOpen
  Plus: #245 → #381 CropGuides → #1605 LatentConcat (prepend to loop output)

## noise_mask handling (critical for latent-space loop)

- **VAEEncode produces latent with NO noise_mask key.** LTXVAudioVideoMask
  then creates a fresh all-zeros mask. This is the correct behavior.
- **LTXVSelectLatents PRESERVES the existing noise_mask** from its input.
  Inherited stale masks corrupt the sampler's mask semantics and break sync.
- **StripLatentNoiseMask** (our node) removes noise_mask so downstream nodes
  create fresh masks. REQUIRED between LTXVSelectLatents and LTXVAudioVideoMask
  in the latent-space subgraph.
- **TensorLoopClose passes previous_value as-is** — no noise_mask transformation.
  The loop does not strip masks anywhere. LATENT workflows must handle this.
- **_AccumulationToImageBatch** strips noise_mask first, reconstructs only if
  ALL items have it. VAEDecode ignores noise_mask, so this is safe.

## Dual workflow support (IMAGE vs LATENT)

Our nodes support both workflow types:
- **IMAGE workflow** (`audio-loop-music-video_image.json`): Subgraph uses
  GetImageRangeFromBatch + VAEEncode/Decode. No StripLatentNoiseMask needed.
  ImageBatch prepends initial render.
- **LATENT workflow** (`audio-loop-music-video_latent.json`): Subgraph uses
  LatentContextExtract + LatentOverlapTrim. LatentConcat prepends initial render.
  No per-iteration VAE round-trip.
- AudioLoopController outputs work for both: overlap_frames (pixel) + overlap_latent_frames (latent).
- **LatentContextExtract**: Replaces LTXVSelectLatents + StripLatentNoiseMask.
  Extracts tail frames AND strips noise_mask in one node. Wire overlap_latent_frames.
- **LatentOverlapTrim**: Replaces LTXVSelectLatents for output trimming.
  Trims overlap AND strips noise_mask. Wire overlap_latent_frames.
- Always use these instead of raw LTXVSelectLatents in the latent-space loop
  subgraph. They hide noise_mask complexity from the user.
- StripLatentNoiseMask kept as low-level utility if needed separately.

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

## Color drift prevention (AdaIN)

Loop iterations progressively darken because each iteration's latent statistics
drift from the initial render. The init_image guide anchors composition but not
color -- guide strength controls the denoise mask, not cross-attention style.

Two AdaIN approaches (can be used together or independently):

**Per-iteration AdaIN** (LTXVAdainLatent, inside subgraph):
- Location: after SeparateAVLatent (#596), before CropGuides (#655)
- Reference: initial render video latent from SeparateAV #245
- Factor: 0.2 default (gentle). Increase to 0.5 for stronger correction.
- per_frame=False (global statistics). Try True if per-frame flickering occurs.
- Present in all three workflows. Bypass (mode=4) to disable.

**Per-step AdaIN** (LTXVPerStepAdainPatcher, model chain):
- Location: after SamplingPreviewOverride, before Set_model
- Reference: node 531 (init image embed latent, available before sampling)
- Factors: per-denoising-step, e.g., "0.3,0.2,0.1,0.05,0.0,0.0,0.0,0.0"
  (stronger at early noisy steps, none at late detail steps)
- Only in `audio-loop-music-video_image_adain_perstep.json`
- More aggressive than per-iteration. Applied during sampling, not after.

**Testing order**: Start with per-iteration only (factor=0.2). If drift persists,
try the per-step workflow. Compare iteration 5+ brightness against initial render.

## Subgraph editing

- ALWAYS use WorkflowEditor from `scripts/workflow_utils.py` for
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
sampling pipeline. IMAGE and LATENT workflows differ in context extraction
and output trimming nodes. Shared internals:

- VAEEncode (1520) -- encodes init_image to latent (scene anchor guide)
- LTXVAddLatentGuide (1519) -- merges conditioning + both guides into latent
- LTXVConcatAVLatent (583) -- adds audio latent
- CFGGuider (644) -- packages for sampling (cfg=1.0, NAG does guidance)
- SamplerCustomAdvanced (573) -- generates new frames

IMAGE workflow only:
- GetImageRangeFromBatch (615) -- extracts last overlap_frames from previous_images
- VAEEncode (614) -- encodes those tail frames to latent (continuity guide)
- GetImageRangeFromBatch (1509) -- trims first overlap_frames, keeps only new content

LATENT workflow only:
- LatentContextExtract (2004) -- extracts last overlap_latent_frames, strips noise_mask
- LatentOverlapTrim (2005) -- trims first overlap_latent_frames, strips noise_mask

Full traces: `docs/pipeline_flow_image.md` and `docs/pipeline_flow_latent.md`.

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
- **Subject anchoring, not setting re-description.** Describe WHO (traits,
  clothing, position) in every entry to anchor identity. Do NOT re-describe
  the environment -- that's in the init_image. This resolves the tension
  between "describe the image perfectly" and "don't re-describe."
- **Node 169 covers trimmed 0:00 to window_seconds (~20s).** TimestampPromptSchedule
  does NOT run during the initial render. Schedule fires at iteration 1:
  `current_time = 1 * stride_seconds` (~18s with overlap=2). Node 169 prompt MUST
  match the schedule's 0:00 entry to avoid visual discontinuity at ~20s.

## Dependencies

Companion custom nodes (not imported, just used alongside in workflows):
- ComfyUI-NativeLooping_testing -- TensorLoopOpen/Close. Don't fork; graph expansion is deeply coupled to ComfyUI execution engine. Likely headed for core inclusion.
- ComfyUI-LTXVideo -- LTXVAddLatentGuide, LTXVCropGuides, LTXVPreprocess, looping sampler
- ComfyUI-KJNodes -- Set/Get nodes, FloatConstant, LTX2_NAG, LTXVImgToVideoInplaceKJ, ImageResizeKJv2
- ComfyUI-VideoHelperSuite -- VHS_VideoCombine
- ComfyUI-MelBandRoFormer -- vocal separation
- **MelBandRoFormer loader has hardcoded architecture** (`dim=384, depth=6,
  num_stems=1`). Only `MelBandRoformer_fp16.safetensors` loads. "Big" models
  (dim=512) and 4-stem models (num_stems=4) require code changes to the loader.
  No HF model exists for male/female voice separation -- AudioPitchDetect fills
  this gap via F0 classification on the separated vocals output.

## Audio analysis

Two analysis scripts with different dependency boundaries:

### `scripts/analyze_audio.py` (ffmpeg only, no Python deps)
- Energy timeline and structure detection via ffmpeg astats.
- Use ffmpeg astats for RMS levels, NOT Python wave module (produces compressed/misleading values).
- `--trim N` offsets timestamps for node 567 start_index.
- MelBandRoFormer separates vocals/instruments only. No male/female distinction.

### `scripts/analyze_audio_features.py` (librosa, optional)
- Music-aware feature extraction: BPM, key, chromagram, mel spectrogram,
  vocal F0, structure segmentation.
- Requires: `uv sync --group analysis` (installs librosa + deps into project venv).
- Run: `uv run --group analysis python scripts/analyze_audio_features.py audio.wav`
- Outputs: JSON (for LLM prompt generation), markdown report, PNG visualizations.
- JSON output is the primary format -- paste into LLM prompt for schedule generation.
  PNG visualizations are for human review only, NOT for LLM consumption.
- **Design principle**: LTX-2.3 audio path is sacred. Audio enters the model
  via `LTXVAudioVAEEncode -> LTXVConcatAVLatent` where cross-attention translates
  mel features into visual motion. Never feed audio visualizations (spectrograms,
  chromagrams) into the video latent stream via `LTXVAddLatentGuide` -- the DiT
  would generate frames that look like heatmaps.
- `--subject` flag generates copy-pasteable LTX 2.3 prompt templates with
  section-appropriate camera/lighting/energy modifiers.
- **LLM prompt generation**: `-j` exports JSON with `workflow_context` and
  `llm_system_prompt`. Paste into Claude/Gemini with creative direction to
  generate node_169_prompt + schedule. 17 rules embedded in system prompt.
  CLI: `--window`, `--overlap`, `--image-desc` add timing context to JSON.
  Guide: `docs/analysis/llm_prompt_generation_guide.md`
- Full guide: `docs/audio_analysis_guide.md`

### Dependency boundary
- **Offline scripts** (scripts/): librosa allowed via optional dep group.
- **Runtime ComfyUI nodes** (nodes.py, nodes_analysis.py): torchaudio only, zero extra deps.
  All analysis outputs must be FLOAT or INT. No IMAGE outputs for audio features.
- **AudioPitchDetect.vocal_fraction wires directly to ConditioningBlend.blend_factor.**
  It's a 0-1 FLOAT representing how much of the window has vocals. This enables
  audio-reactive prompt blending without TimestampPromptSchedule's time-based blend.
  Disconnect TPS blend_factor, wire vocal_fraction instead. Sections with heavy
  vocals emphasize the vocal-focused prompt; instrumental sections emphasize the other.
  See `docs/audio_analysis_guide.md` wiring pattern 2.

## Testing

Tests run via the project's own venv with pytest:
```bash
uv run --group dev --group analysis python -m pytest tests/ -v --rootdir=.
```
- `__init__.py` has a try/except guard so pytest can import the package
  without comfy_api (only available inside ComfyUI runtime).
- `tests/conftest.py` adds `scripts/` to sys.path for import.
- `tests/test_audio_features.py` -- 24 tests for librosa extraction (synthetic audio).
- `tests/test_audio_analysis_nodes.py` -- 9 tests for runtime AudioPitchDetect node.
- `tests/test_workflows.py` -- workflow JSON structural validation.

## Editing workflow JSON (subgraphs)

Top-level links use array format: `[id, src_node, src_slot, tgt_node, tgt_slot, type]`
Subgraph internal links use dict format: `{id, origin_id, origin_slot, target_id, target_slot, type}`
Subgraph definitions live at `wf['definitions']['subgraphs'][0]` with keys:
`nodes`, `links`, `inputs` (with `linkIds` arrays), `outputs`, `widgets`.
Subgraph input distributor node ID is -10. Output collector is -20.
All three representations (top-level links, node link fields, subgraph linkIds)
must stay in sync or wires break on reload.
Use `scripts/workflow_utils.py` for programmatic edits.

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
Run `scripts/test_workflow_integrity.py` after every programmatic edit.

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

- `example_workflows/audio-loop-music-video_image.json` -- image-space loop (per-iteration VAE round-trip, tested/working). Has per-iteration AdaIN (factor=0.2, bypassable).
- `example_workflows/audio-loop-music-video_latent.json` -- latent-space loop (no per-iteration VAE round-trip, UNTESTED). Has per-iteration AdaIN (factor=0.2, bypassable).
- `example_workflows/audio-loop-music-video_image_adain_perstep.json` -- image-space loop + per-step AdaIN model patcher. Has BOTH per-iteration AdaIN inside subgraph AND LTXVPerStepAdainPatcher on model chain. Reference: node 531 (init image embed). Experimental.
- `example_workflows/upscale-loop-output.json` -- separate upscale workflow (when built)
- `coderef/origiltx23_long_loop_extension_test.json` -- original pre-scheduler workflow
- `coderef/RuneXX_LTX-2.3-Workflows/` -- reference LTX 2.3 workflows (3-pass upscale pattern)
- `docs/workflow_pipeline_trace.md` -- end-to-end pipeline trace
- `docs/nag_technical_reference.md` -- LTX2_NAG technical documentation
- `docs/prompt_creation_guide.md` -- prompt writing guide with variation patterns
- `docs/audio_analysis_guide.md` -- offline/runtime analysis, AudioPitchDetect wiring patterns, vocal_fraction as blend_factor
- `docs/analysis/llm_prompt_generation_guide.md` -- LLM-assisted prompt generation: system prompt, user template, 17 rules, examples
- `docs/analysis/audio_in_prompt_analysis.md` -- community research on LTX 2.3 lip sync prompting (transcription, delivery, volume trick)
- `docs/analysis/audio_in_prompt_guide_notebooklm.md` -- additional i2v + audio prompting research (subject anchoring, frozen video fix, over-emoting)
- `docs/ltx23_prompt_system_prompts.md` -- official i2v/t2v system prompts
- `docs/upscale_guide.md` -- upscale workflow build guide
- `docs/ltxv_looping_sampler_settings.md` -- LTXVLoopingSampler reference (VIDEO-ONLY, no AV latent support)
- `docs/latent_loop_build_guide.md` -- build guide for LTXVLoopingSampler (video-only, not for music video)
- `docs/pipeline_flow_image.md` -- full pipeline trace for image workflow (every node, wire, widget)
- `docs/pipeline_flow_latent.md` -- full pipeline trace for latent workflow (includes noise_mask flow)
- `docs/subgraph_latent_rework_guide.md` -- how the latent rework was done
- `docs/analysis/ltx23_gaps_analysis.md` -- capability gaps + LTXVLoopingSampler AV incompatibility analysis
- `internal/postmortem_v0408_session.md` -- debugging history (6 issues with fixes)
- `internal/postmortem_v0409_latent_rework.md` -- latent-space loop rework (5 issues with fixes, noise_mask root cause)
