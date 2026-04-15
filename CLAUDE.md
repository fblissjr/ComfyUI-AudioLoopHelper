# ComfyUI-AudioLoopHelper

## TLDR

ComfyUI nodes that automate loop timing and audio analysis for full-length
music video generation with LTX 2.3. The main node (AudioLoopController)
reads audio duration from the tensor, computes stride from window + overlap,
outputs start_index / should_stop / audio_duration / iteration_seed /
stride_seconds / overlap_frames / overlap_latent_frames / overlap_seconds.

## Architecture

Two files: `nodes.py` (core loop nodes) and `nodes_analysis.py` (runtime
audio analysis). Uses ComfyUI's extension API (`ComfyExtension`,
`io.ComfyNode`). Single entry point: `comfy_entrypoint()` in nodes.py
imports analysis nodes from nodes_analysis.py.

Core nodes (nodes.py):
- `AudioLoopController` -- core: 8 outputs. overlap_seconds (slot 7) auto-wires to LTXVAudioVideoMask video_start_time.
- `TimestampPromptSchedule` -- per-iteration prompt from timestamp ranges, with blend support
- `ConditioningBlend` -- lerps two conditioning tensors (works with LTX Gemma 3 and CLIP)
- `AudioLoopPlanner` -- displays iteration timeline for planning
- `AudioDuration` -- extracts duration/sample_rate from audio tensor
- `LatentContextExtract` -- extracts tail latent frames + strips noise_mask
- `LatentOverlapTrim` -- trims overlap latent frames + strips noise_mask
- `StripLatentNoiseMask` -- low-level noise_mask removal utility
- `KeyframeImageSchedule` -- per-iteration keyframe image selection from timestamp schedule (like TimestampPromptSchedule but for images). Outputs image/next_image/blend_factor.
- `VideoFrameExtract` -- extracts frame from reference video at current iteration's timestamp for video-to-video conditioning
- `ImageBlend` -- pixel-space lerp of two images by a factor. Pairs with KeyframeImageSchedule for smooth keyframe transitions.

Analysis nodes (nodes_analysis.py, torchaudio only):
- `AudioPitchDetect` -- per-iteration F0 detection, vocal presence, male/female classification. Outputs FLOAT/BOOLEAN only.

Key helper functions: `_audio_duration`, `_parse_timestamp` ("M:SS" or bare seconds),
`_format_timestamp` (preserves sub-second; NOT same as `_fmt_ts()` in analyze_audio_features.py
which truncates), `_parse_schedule`, `_match_schedule`, `_match_schedule_with_next`,
`_parse_image_schedule` (like `_parse_schedule` but int image indices),
`_match_image_schedule`, `_match_image_schedule_with_next`.

## Key patterns

- AUDIO type: `{"waveform": Tensor, "sample_rate": int}`. Duration = `waveform.shape[-1] / sample_rate`.
- Stride = `window_seconds - overlap_seconds`. overlap_frames = `round(overlap_seconds * fps)`.
- start_index is clamped so at least 0.5s of audio always remains (prevents mel crash on final iteration).
- TimestampPromptSchedule only runs in loop iterations, NOT the initial render. Node 169 handles initial ~20s.
- **Prompt changes cause style drift even at CFG 1.0.** Use ConditioningBlend with blend_seconds > 0.
- LTX 2.3 uses Gemma 3 text encoder (NOT CLIP). Format: `[tensor, {"attention_mask": mask}]`, no pooled_output. Standard ConditioningAverage won't work.
- **Audio path is sacred.** Audio enters via `LTXVAudioVAEEncode -> LTXVConcatAVLatent`. Never feed visualizations into the video latent stream.
- Video VAE formula: `latent = (pixel - 1) // 8 + 1`. NOT `pixel // 8`.
- **mask=0 means "fixed context"** in LTX noise masks. Audio latent with mask=0 keeps real song. mask=1 regenerates from noise, destroying lip sync.
- LTXVLoopingSampler CANNOT support AV latents (5 blocking architectural issues). TensorLoop is the correct approach. See `docs/analysis/ltx23_gaps_analysis.md`.

## Critical constraints

- **Never feed audio visualizations into video latent stream** -- DiT generates heatmap-looking frames.
- **Never change LTXVAudioVideoMask (Node 606) wiring** -- audio_start_time = audio_end_time = window_size is intentional (empty mask range keeps audio fixed).
- **Use LatentContextExtract/LatentOverlapTrim** instead of raw LTXVSelectLatents in latent-space subgraph -- they strip noise_mask automatically.
- **Node 169 prompt MUST match schedule's 0:00 entry** to avoid visual discontinuity at ~20s.
- **Always use WorkflowEditor** from `scripts/workflow_utils.py` for subgraph edits. Manual JSON surgery breaks links.

## ComfyUI gotchas

- **Workflow JSON has two link representations**: node body `"link"` fields AND the `"links"` array. Both must stay in sync.
- Link array: `[link_id, source_node, source_output, target_node, target_input, type]`
- Node `"mode": 0` = active, `"mode": 4` = bypassed.
- **PrimitiveNode cannot feed DynamicCombo sub-inputs.** Set values directly on the widget.
- **TensorLoopClose checks should_stop AFTER the loop body executes.** Handle edge-case inputs gracefully.
- **After changing define_schema(), users must delete and re-add the node in UI.** JSON slot indices are baked at save time.
- **Removing a subgraph component input shifts all higher slot indices.** Decrement `origin_slot` references.
- **ComfyUI execution engine evaluates downstream conditioning graphs before upstream sampling.** Extra nodes in conditioning path can corrupt initial render.
- **torchaudio detect_pitch_frequency on silence gives false positives.** Gate with RMS energy check (< 0.005).
- Pyright `reportIncompatibleMethodOverride` on `execute()` is a false positive.
- Module constants must be defined BEFORE functions that reference them (project convention).
- **Scrub workflows before open-sourcing:** filenames, paths, UUIDs, image previews, creative prompts.
- Validate workflow JSON after edits: `python3 -c "import json; json.load(open('file.json'))"`

## Subgraph editing

- ALWAYS use WorkflowEditor from `scripts/workflow_utils.py`.
- Top-level links: array format `[id, src_node, src_slot, tgt_node, tgt_slot, type]`
- Subgraph internal links: dict format `{id, origin_id, origin_slot, target_id, target_slot, type}`
- Subgraph defs at `wf['definitions']['subgraphs'][0]` with keys: `nodes`, `links`, `inputs`, `outputs`, `widgets`.
- Distributor node ID = -10. Output collector = -20.
- DynamicCombo widgets: `[num_items, strength_1, strength_2, ..., index_1, index_2, ...]` -- strengths FIRST, then indices. NOT interleaved.

## Testing

```bash
uv run --group dev --group analysis python -m pytest tests/ -v --rootdir=.
```
- `__init__.py` guards ComfyUI-only import with try/except for pytest.
- `nodes.py` has try/except for `comfy_api` with `_IOStub`/`_Passthrough` fallback for test imports.
- `tests/conftest.py` adds `scripts/` to sys.path.
- `tests/test_audio_features.py` -- 33 tests (offline analysis)
- `tests/test_audio_analysis_nodes.py` -- 9 tests (runtime AudioPitchDetect)
- `tests/test_keyframe_nodes.py` -- 28 tests (KeyframeImageSchedule, VideoFrameExtract, ImageBlend)
- `tests/test_workflows.py` -- workflow JSON structural validation

## Dependencies

Companion custom nodes (not imported, used alongside in workflows):
- ComfyUI-NativeLooping_testing -- TensorLoopOpen/Close
- ComfyUI-LTXVideo -- LTXVAddLatentGuide, LTXVCropGuides, LTXVPreprocess
- ComfyUI-KJNodes -- Set/Get nodes, FloatConstant, LTX2_NAG, LTXVImgToVideoInplaceKJ
- ComfyUI-VideoHelperSuite -- VHS_VideoCombine
- ComfyUI-MelBandRoFormer -- vocal separation (hardcoded `dim=384, depth=6, num_stems=1`)

### Dependency boundary
- **Offline scripts** (scripts/): librosa allowed via optional `analysis` dep group.
- **Runtime nodes** (nodes.py, nodes_analysis.py): torchaudio only, zero extra deps. All outputs FLOAT or INT.
- AudioPitchDetect.vocal_fraction wires directly to ConditioningBlend.blend_factor for audio-reactive blending.

## Audio analysis scripts

- `scripts/analyze_audio.py` -- ffmpeg-only energy/structure detection (no Python deps)
- `scripts/analyze_audio_features.py` -- librosa: BPM, key, vocal F0, structure, JSON for LLM prompt generation
- JSON export (`-j`) includes `llm_system_prompt` with all prompt rules. Paste into Claude/Gemini.
- Full guide: `docs/audio_analysis_guide.md`

## Debugging workflow regressions

Compare against known-working workflow JSON (keep copies in `internal/scratch/`).
Change ONE setting at a time. Run `scripts/test_workflow_integrity.py` after every edit.
LTX-2_00032.json and LTX-2_00040.json are confirmed working (April 9, 2026).

## Documentation index

### User-facing guides
- `docs/prompt_workflow_end_to_end.md` -- complete pipeline: init image -> VLM -> audio analysis -> LLM -> workflow
- `docs/prompt_creation_guide.md` -- prompt rules, variation patterns (A/B/C), sampler tuning, examples
- `docs/audio_analysis_guide.md` -- offline/runtime analysis, AudioPitchDetect wiring patterns
- `docs/analysis/llm_prompt_generation_guide.md` -- LLM-assisted schedule generation, 17 rules, examples
- `docs/ltx23_prompt_system_prompts.md` -- official i2v/t2v system prompts

### Technical reference (read on-demand)
- `docs/ltx23_model_reference.md` -- image guides, latent volume, VAE conversion, AdaIN, conditioning path, noise_mask, dual workflow, extension subgraph, upscaling
- `docs/nag_technical_reference.md` -- NAG (Normalized Attention Guidance)
- `docs/pipeline_flow_image.md` -- IMAGE workflow node-by-node trace
- `docs/pipeline_flow_latent.md` -- LATENT workflow node-by-node trace
- `docs/subgraph_latent_rework_guide.md` -- how the latent rework was done
- `docs/upscale_guide.md` -- separate upscale workflow build guide

### Video-only (LTXVLoopingSampler, not for music video)
- `docs/latent_loop_build_guide.md` -- build guide
- `docs/ltxv_looping_sampler_settings.md` -- parameter reference

### Analysis and research
- `docs/analysis/ltx23_gaps_analysis.md` -- capability gaps, LTXVLoopingSampler AV incompatibility
- `docs/analysis/audio_in_prompt_analysis.md` -- community lip sync prompting research
- `docs/analysis/audio_in_prompt_guide_notebooklm.md` -- i2v + audio prompting research

### Example workflows
- `example_workflows/audio-loop-music-video_image.json` -- IMAGE loop (tested/working, per-iteration AdaIN)
- `example_workflows/audio-loop-music-video_latent.json` -- LATENT loop (UNTESTED, per-iteration AdaIN)
- `example_workflows/audio-loop-music-video_image_adain_perstep.json` -- IMAGE + per-step AdaIN (experimental)
- `example_workflows/upscale-loop-output.json` -- separate upscale workflow (when built)

### Internal (gitignored)
- `internal/postmortem_v0408_session.md` -- debugging history (6 issues)
- `internal/postmortem_v0409_latent_rework.md` -- latent rework (5 issues, noise_mask root cause)
- `internal/workflow_pipeline_trace.md` -- superseded pipeline trace (historical)
