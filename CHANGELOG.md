# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).
This project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `docs/prompt_workflow_end_to_end.md`: complete end-to-end walkthrough from
  init image preparation through VLM description extraction, audio analysis,
  LLM schedule generation, and workflow insertion. Includes exact VLM prompts
  for single and multi-person scenes.
- `docs/ltx23_model_reference.md`: extracted LTX 2.3 model behavior reference
  (image guides, latent volume, VAE conversion, AdaIN, conditioning path,
  noise_mask, dual workflow support, extension subgraph, upscaling)

### Changed
- CLAUDE.md reorganized for progressive disclosure (481 -> 154 lines).
  Deep implementation details moved to `docs/ltx23_model_reference.md`.
  CLAUDE.md now focuses on architecture, key patterns, critical constraints,
  gotchas, and a categorized documentation index.
- Removed `docs/latent_loop_workflow_guide.md` (redundant with
  `docs/latent_loop_build_guide.md`, had confusing supersession header)
- Moved raw analysis artifacts to `internal/analysis/` (comfyui_ltxvideo_raw,
  ltx2_native_raw, ltx_desktop_raw)
- Moved superseded `workflow_pipeline_trace.md` to `internal/`
- Added cross-references between docs: VLM extraction prompts linked from
  audio_analysis_guide and llm_prompt_generation_guide; variation patterns
  linked to full examples in prompt_creation_guide; multi-person rules
  linked from end-to-end guide to prompt_creation_guide
- Enhanced JSON export (`-j`): includes `workflow_context` (trim, window, stride,
  subject, image description) and `llm_system_prompt` with all 17 prompt engineering
  rules for the i2v + frozen audio loop workflow. Paste directly into Claude/Gemini.
- New CLI args: `--window`, `--overlap`, `--image-desc` for workflow timing context
- `get_node_169_prompt()`: script now outputs a separate "Node 169" section showing
  exactly what to paste into the initial CLIPTextEncode (matches first schedule entry)
- TimestampPromptSchedule + ConditioningBlend fully wired in all 3 example workflows:
  prompt -> CLIPTextEncode A -> ConditioningBlend.a, next_prompt -> CLIPTextEncode B ->
  ConditioningBlend.b, blend_factor -> ConditioningBlend. Extension subgraph input 6
  rewired from static GetNode to ConditioningBlend output.
- `scripts/patch_scheduling_wiring.py`: one-shot patch script for wiring scheduling
- `docs/analysis/llm_prompt_generation_guide.md`: complete guide for LLM-assisted
  prompt schedule generation with system prompt, user template, and examples
- Per-iteration AdaIN color correction (LTXVAdainLatent) inside Extension subgraph
  for all workflows. Normalizes each iteration's latent statistics against the
  initial render. factor=0.2 default, bypassable. Prevents progressive darkening.
- Per-step AdaIN workflow variant (`audio-loop-music-video_image_adain_perstep.json`).
  Adds LTXVPerStepAdainPatcher to model chain for denoising-time correction.
- `overlap_seconds` output on AudioLoopController (slot 7). Automatically wires
  to LTXVAudioVideoMask video_start_time inside the Extension subgraph. No more
  manual sync when changing overlap.
- Multi-character prompting guide in docs/prompt_creation_guide.md
- AudioLoopPlanner now shows initial render time range with "[uses static prompt,
  not schedule]" annotation, making it clear the schedule only applies to loop iterations
- AudioPitchDetect node: per-iteration vocal pitch detection using torchaudio
  (median F0, has_vocals, is_male_range, is_female_range, vocal_fraction).
  Wire to MelBandRoFormer separated vocals for clean signal.
- `nodes_analysis.py`: separate file for audio analysis runtime nodes
- `_slice_audio_window()` shared helper for extracting iteration audio windows
- `scripts/analyze_audio_features.py`: librosa-based music feature extraction
  (BPM, key detection, chromagram, mel spectrogram, vocal F0, structure segmentation).
  Outputs JSON (for LLM prompt generation), markdown report, and PNG visualizations.
- `--subject` flag on analyze_audio_features.py: generates full LTX 2.3 prompt
  templates with section-appropriate camera, lighting, and energy modifiers.
  Copy-pasteable into TimestampPromptSchedule.
- `pyproject.toml` with `analysis` and `dev` dependency groups
- `tests/test_audio_features.py`: 24 tests for offline feature extraction
- `tests/test_audio_analysis_nodes.py`: 9 tests for runtime AudioPitchDetect
- `tests/conftest.py`: pytest path configuration for scripts/ imports
- `conftest.py` (root): prevents pytest from importing ComfyUI-only `__init__.py`
- LatentContextExtract node: extracts tail latent frames + strips noise_mask
- LatentOverlapTrim node: trims overlap latent frames + strips noise_mask
- StripLatentNoiseMask node: low-level utility for noise_mask removal
- ScheduleToMultiPrompt node: converts TimestampPromptSchedule to MultiPromptProvider format
- overlap_latent_frames output on AudioLoopController
- Latent workflow variant (`audio-loop-music-video_latent.json`) -- UNTESTED
- Pipeline flow documentation for both workflow variants (`docs/pipeline_flow_*.md`)
- Workflow validator agent supporting both image and latent workflows
- `scripts/workflow_utils.py` and `scripts/test_workflow_integrity.py` (moved from
  internal/scripts/ for open-source distribution)
- CHANGELOG.md

### Changed
- `__init__.py` now guards ComfyUI-only import with try/except (allows pytest to run)
- Renamed workflows: removed date-based versioning (v0408/v0409), now
  `audio-loop-music-video_image.json` and `audio-loop-music-video_latent.json`
- AudioLoopController now outputs 7 values (added overlap_latent_frames)
- overlap_latent_frames uses correct formula: `(pixel-1)//8+1` not `pixel//8`
- LatentOverlapTrim clamps overlap to prevent empty tensor edge case

### Fixed
- Latent workflow: overlap_latent_frames now dynamically wired from
  AudioLoopController through subgraph to LatentContextExtract and
  LatentOverlapTrim (was hardcoded to 4)
- Latent workflow: subgraph input 14 now receives latent-space frames
  (from slot 6) instead of pixel-space frames (from slot 5)
- AudioLoopController overlap_latent_frames tooltip: references correct
  downstream nodes instead of stale LTXVSelectLatents
