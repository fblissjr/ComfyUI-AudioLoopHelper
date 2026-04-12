# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).
This project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
