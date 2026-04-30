---
name: audio-analyze
description: Analyze audio file for music video prompt scheduling. Picks the right analyzer (ffmpeg-only for quick energy/structure, librosa for BPM/key/F0/JSON-export-for-LLM), and outputs a prompt schedule template with timestamps. Also works on generated audio for diagnostic comparison.
disable-model-invocation: true
---

Last updated: 2026-04-30

# Audio Analyzer

Analyze an audio file (input track OR generated output) and surface what's
needed for prompt scheduling, or for diagnosing differences between source
and generated audio.

## Usage

`/audio-analyze <audio_file> [--trim <seconds>] [--rich]`

## Which analyzer to use

| Mode | Tool | When |
|---|---|---|
| Quick energy + structure (no librosa) | `scripts/analyze_audio.py` | Triage; works on a fresh clone with zero Python deps |
| Rich features (BPM, key, F0, JSON for LLM) | `scripts/analyze_audio_features.py` | Authoritative analysis for prompt-schedule generation; default when `--rich` is passed or when the user asks for BPM/key/structure-with-confidence |

If the user gave an audio file with no further context, run the quick analyzer
first; if the output suggests vocal+music but no prompt schedule will be built,
stop there. If they're going to author prompts (or diagnose generated audio),
run the rich analyzer.

## Steps — quick mode

1. ```bash
   uv run scripts/analyze_audio.py <audio_file> --trim <seconds>
   ```
2. Present results: energy timeline, detected structure, key transitions.
3. Output a prompt-schedule template stub with timestamps adjusted for trim.
4. Suggest a trim value if not provided (first energy jump = vocal entry).

## Steps — rich mode (`--rich`)

1. ```bash
   uv run --group analysis scripts/analyze_audio_features.py <audio_file> \
     --scene-diversity --montage --style \
     --json /tmp/audio_features.json
   ```
   Add `--trim <seconds>` if the user gave one. The flags pull style hints +
   montage cuts + section diversity into the JSON for LLM consumption.
2. Read `/tmp/audio_features.json`, surface BPM, key, F0 trajectory, structure.
3. Generate prompt-schedule template using the structure boundaries as
   schedule timestamps.
4. If the user is diagnosing GENERATED audio (the audio is from one of our
   renders, not the source track), do an A/B: run rich mode on both source
   and generated, diff BPM / key / F0 / structure. Spectrogram via
   `scripts/spectrogram_to_reference.py --audio <wav>` makes the audio
   reviewable in the chat.

## Output

- Basic info (duration, sample rate, loudness)
- Song structure table (section, timestamps, energy level)
- Energy timeline visualization
- BPM, key, F0 (rich mode only)
- Ready-to-fill prompt schedule template
- For generated-audio comparisons: side-by-side feature deltas

## Notes

- Quick analyzer uses ffmpeg for RMS (not Python `wave` — that gives
  misleading values).
- Rich analyzer uses librosa; requires `--group analysis` on the uv invocation.
- Structure detection uses percentile-based thresholds with short-section
  merging.
- For vocal timing, run MelBandRoFormer separation then re-analyze the
  vocals.wav separately.
- MelBandRoFormer separates vocals/instruments only — no male/female
  distinction.
- Working with generated audio: extract via `ffmpeg -i <mp4> -vn -acodec
  pcm_s16le <wav>` first.

## Reference

- Full analysis guide: `docs/guides/audio_analysis_guide.md`
- End-to-end prompt workflow: `docs/guides/prompt_workflow_end_to_end.md`
- Spectrogram visualization (diagnostic): `scripts/spectrogram_to_reference.py`
