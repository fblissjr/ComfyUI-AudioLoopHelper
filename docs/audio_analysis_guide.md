Last updated: 2026-04-12

# Audio Analysis Guide

Two analysis approaches: **offline** (librosa, before workflow run) and
**runtime** (torchaudio, per-iteration inside the loop).

## When to use which

| Approach | Use when | Tool |
|----------|----------|------|
| Offline analysis | Planning prompt schedules, understanding song structure | `scripts/analyze_audio_features.py` |
| Offline analysis (no deps) | Quick energy/loudness check, no librosa available | `scripts/analyze_audio.py` |
| Runtime AudioPitchDetect | Vocal/instrumental switching per iteration, duet male/female detection | AudioPitchDetect node |

## Offline analysis

### Setup

```bash
cd custom_nodes/ComfyUI-AudioLoopHelper
uv sync --group analysis
```

### Basic usage

```bash
# Full analysis
uv run --group analysis python scripts/analyze_audio_features.py song.wav

# With trim offset (matches node 567 start_index)
uv run --group analysis python scripts/analyze_audio_features.py song.wav --trim 10

# Auto-generate prompt templates
uv run --group analysis python scripts/analyze_audio_features.py song.wav \
  --subject "a woman in her 30s with dark hair singing in a basement workshop"

# Export JSON for LLM-assisted schedule generation
uv run --group analysis python scripts/analyze_audio_features.py song.wav -j analysis.json

# PNG visualizations (spectrogram, chromagram, onset envelope)
uv run --group analysis python scripts/analyze_audio_features.py song.wav --png-dir ./viz

# Vocal F0 analysis on separated track
uv run --group analysis python scripts/analyze_audio_features.py song.wav \
  --vocal-track vocals_only.wav
```

### The --subject flag

Without `--subject`, the script outputs placeholder prompts:
```
0:02-0:45: [VERSE - medium] describe action and audio here
```

With `--subject`, it generates full LTX 2.3 prompt templates:
```
0:02-0:45: Style: cinematic. In a medium shot, a woman singing in a workshop is performing. Warm lighting, steady energy. The voice fills the space. Soft ambient hum.
```

The subject is wrapped with section-appropriate modifiers. Copy-paste into
TimestampPromptSchedule and edit the details.

### Section modifier mapping

| Section | Framing | Lighting | Audio description |
|---------|---------|----------|-------------------|
| INTRO | Wide establishing shot, static camera | Soft, gentle | Quiet ambient tone |
| VERSE | Medium shot | Warm, steady | Voice fills space |
| CHORUS | Close-up | Bright, dynamic | Powerful and resonant |
| BRIDGE | Wide shot | Moody, low contrast | Subdued, reflective |
| OUTRO | Wide shot, dolly out | Fading, gentle | Sound fades, room tone |
| BREAK | Medium shot, static | Dim, still | Instrumental moment |

These follow LTX 2.3 i2v conventions from the prompt creation guide.
"Dolly out" is avoided except for OUTRO (it can break limbs/faces).

## Runtime analysis: AudioPitchDetect

### What it does

Per-iteration vocal pitch detection using torchaudio autocorrelation.
Runs inside the loop on the current ~20s audio window. Outputs scalars only.

### Outputs

| Output | Type | Range | Meaning |
|--------|------|-------|---------|
| median_f0 | FLOAT | 0-400 Hz | Median fundamental frequency (0 if unvoiced) |
| has_vocals | BOOLEAN | T/F | Pitched content detected |
| is_male_range | BOOLEAN | T/F | median_f0 < 160 Hz |
| is_female_range | BOOLEAN | T/F | median_f0 >= 160 Hz |
| vocal_fraction | FLOAT | 0.0-1.0 | Ratio of voiced frames in window |

### Best practices

- **Wire to separated vocals** (MelBandRoFormer node 569 output), not raw audio.
  Raw audio has drums/bass that confuse pitch detection.
- **Energy gate**: the node automatically skips analysis when RMS < 0.005
  (near-silence), preventing false positives from noise floor.
- **Autocorrelation limitations**: less robust than probabilistic YIN (librosa's
  pyin). Works well on clean separated vocals, may give false positives if
  MelBandRoFormer bleeds through instruments.

### Wiring pattern 1: Vocal/instrumental prompt switching

Use has_vocals with a Switch node to select between prompts:

```
MelBandRoFormerSampler (vocals)
  └→ AudioPitchDetect.audio

AudioLoopController
  └→ start_index → AudioPitchDetect.start_seconds

AudioPitchDetect
  └→ has_vocals → Switch/Mux node
       ├→ True:  "close-up, singing with emotion" → TextEncode
       └→ False: "wide shot, swaying to instrumental" → TextEncode
```

### Wiring pattern 2: vocal_fraction as blend_factor

**Key insight**: `vocal_fraction` is already a 0.0-1.0 FLOAT. It can wire
directly to `ConditioningBlend.blend_factor`, bypassing TimestampPromptSchedule's
time-based blend:

```
TimestampPromptSchedule
  ├→ prompt → TextEncode A → conditioning_a ─┐
  └→ next_prompt → TextEncode B → conditioning_b ─┐
                                                    │
AudioPitchDetect                                    │
  └→ vocal_fraction ──→ ConditioningBlend.blend_factor
                                    │
                        blended conditioning → sampler
```

This creates **audio-reactive blending**: sections with heavy vocals (fraction
near 1.0) emphasize the vocal-focused prompt, instrumental sections (fraction
near 0.0) emphasize the instrumental prompt. The timestamp schedule still
picks WHICH prompts to use (verse/chorus), but HOW MUCH of each is driven
by actual vocal content rather than fixed time boundaries.

To use this pattern: disconnect TimestampPromptSchedule's blend_factor output
from ConditioningBlend and wire AudioPitchDetect.vocal_fraction instead.

### Wiring pattern 3: Male/female duet switching

For songs with alternating male and female vocals:

```
AudioPitchDetect
  ├→ is_female_range → Switch → "close-up of the woman singing"
  └→ is_male_range   → Switch → "medium shot of the man singing"
```

Note: MelBandRoFormer cannot separate male from female voices -- it only
separates vocals from instruments. AudioPitchDetect fills this gap by
classifying the F0 of whatever vocal content is present. Both voices must
already be in the separated vocals output.

## MelBandRoFormer model limitations

The ComfyUI-MelBandRoFormer extension has a hardcoded architecture config
(`dim=384, depth=6, num_stems=1`). This means:

- Only model weights matching this architecture load (e.g., `MelBandRoformer_fp16.safetensors`)
- "Big" models (`pcunwa/Mel-Band-Roformer-big`, `dim=512`) won't load -- architecture mismatch
- 4-stem models (vocals + drums + bass + other) need `num_stems=4` -- code change required
- No HF model exists for male/female voice separation

AudioPitchDetect works around the gender limitation by analyzing pitch on
the separated output. For instrument isolation (drums, bass), the extension
would need modification or replacement.

## Design principle

LTX-2.3 audio path is sacred. Audio enters the model via
`LTXVAudioVAEEncode -> LTXVConcatAVLatent` where cross-attention translates
mel features into visual motion (lip sync, rhythm). Analysis nodes extract
scalar features FROM audio to modulate text conditioning parameters. They
never feed visual representations (spectrograms, chromagrams) into the video
latent stream -- the DiT would generate frames that look like heatmaps.
