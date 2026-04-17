Last updated: 2026-04-17

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

# Auto-generate prompt templates (see prompt_workflow_end_to_end.md Step 2 for
# how to extract --subject and --image-desc from a VLM)
uv run --group analysis python scripts/analyze_audio_features.py song.wav \
  --subject "a woman in her 30s with dark hair singing in a basement workshop"

# Pick an ambition tier (default 2a). See "Scene-diversity taxonomy" below.
uv run --group analysis python scripts/analyze_audio_features.py song.wav \
  --subject "a woman singing in a workshop" --scene-diversity 3b

# Montage mode: shorter dwell (~12s), emotional-arc language,
# Arcane-style music-drives-narrative pacing.
uv run --group analysis python scripts/analyze_audio_features.py song.wav \
  --subject "a young woman walking through a snowy alley" \
  --scene-diversity 4a --montage

# Export JSON for LLM-assisted schedule generation (includes system prompt + workflow context)
uv run --group analysis python scripts/analyze_audio_features.py song.wav -j analysis.json \
  --image-desc "description of your init image" --window 19.88 --overlap 2.0

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

With `--subject`, the output has two clearly labeled sections. Every line
contains `is singing` (singular) or `are singing together` (if the subject
names 2+ people) — the singing verb keeps LTX 2.3's audio-video
cross-attention signal for lip sync.

**Node 169 (initial render prompt):**
```
Style: cinematic. In a wide establishing shot, static camera, locked off shot, a woman singing in a workshop is singing softly, easing into the song, static camera, mouth opening softly, handheld energy, rock-video motion. Soft lighting, gentle. Quiet ambient tone, gentle room presence.
```

**TimestampPromptSchedule (node 1558):**
```
0:00-0:22: Style: cinematic. In a wide establishing shot, static camera, locked off shot, a woman singing in a workshop is singing softly, easing into the song, static camera, mouth opening softly, handheld energy, rock-video motion. Soft lighting, gentle. Quiet ambient tone, gentle room presence.
0:22-0:45: Style: cinematic. In a wide establishing shot, static camera, locked off shot, a woman singing in a workshop is singing softly, easing into the song, slow dolly in, head tilted slightly, handheld energy, rock-video motion. Soft lighting, gentle. Quiet ambient tone, gentle room presence.
0:45-1:01: Style: cinematic. In a medium shot, a woman singing in a workshop is singing with a steady voice, static camera, head bobbing slightly, handheld energy, rock-video motion. Warm lighting, steady energy. The voice fills the space. Soft ambient hum.
...
1:39+: Style: cinematic. In a close-up, a woman singing in a workshop is singing with full power, voice rising, slow jib up, arms slightly raised, handheld energy, rock-video motion. Bright, dynamic lighting. The voice is powerful and resonant.
```

Paste the node 169 prompt into node 169 (CLIPTextEncode). Paste the schedule
into node 1558. The node 169 prompt matches the first schedule entry
byte-for-byte by construction (enforced in test suite).

Long sections are automatically subdivided into ~20s chunks (~12s with
`--montage`) so every iteration window gets its own prompt. A 3-minute
song produces 7+ entries instead of 4-5.

### Section modifier mapping

| Section | Framing | Lighting | Action | Audio description |
|---------|---------|----------|--------|-------------------|
| INTRO   | Wide establishing shot, static camera | Soft, gentle | is singing softly, easing into the song | Quiet ambient tone |
| VERSE   | Medium shot | Warm, steady | is singing with a steady voice | Voice fills space |
| CHORUS  | Close-up | Bright, dynamic | is singing with full power, voice rising | Powerful and resonant |
| BRIDGE  | Wide shot | Moody, low contrast | is singing with quiet emotion | Subdued, reflective |
| OUTRO   | Wide shot, dolly out | Fading, gentle | is singing the final notes, voice trailing off | Sound fades, room tone |
| BREAK   | Medium shot, static | Dim, still | is singing softly, pausing in place | Instrumental moment |

These follow LTX 2.3 i2v conventions from the prompt creation guide.
"Dolly out" is avoided except for OUTRO (it can break limbs/faces).
Multi-subject detection rewrites "is singing" to "are singing together".

### Scene-diversity taxonomy

`--scene-diversity <tier><sub>` controls ambition + flavor. Default `2a`.

| Tier | Name | Maps to | Sub-variants |
|------|------|---------|--------------|
| 1 | performance_live      | `internal/prompt.md`   | 1a close-up concert, 1b wide stage, 1c studio-live |
| 2 | performance_dynamic   | `internal/prompt2.md`  | 2a handheld energetic (DEFAULT), 2b steady-cam polished |
| 3 | cinematic             | `internal/prompt3.md`  | 3a urban night, 3b natural outdoor, 3c interior character, 3d perf + b-roll |
| 4 | narrative             | `internal/prompt4.md`  | 4a linear story, 4b flashback / dream |
| 5 | stylized              | —                      | 5a noir monochrome, 5b surreal, 5c retro / period |
| 6 | avant_garde           | —                      | (abstract / non-linear; no sub-letters) |

Main tier selects which beat pools activate (camera, body, scene-shift,
narrative, style, avant). Sub-letter adds a single mood-bundle phrase
per prompt (palette / location / camera-style adjectives).

### Montage mode

`--montage` is orthogonal to the tier. When set:
- Subdivision target drops from 20s → 12s (more cuts).
- Each entry gets an emotional-arc beat ("the feeling building",
  "catharsis arriving", "release easing into stillness").
- The LLM system prompt adds the Arcane-style instruction: *"music drives
  art drives narrative."*

Works with any tier 2-6. Use when the video should advance an emotional
beat per cut rather than dwell on a single shot.

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
