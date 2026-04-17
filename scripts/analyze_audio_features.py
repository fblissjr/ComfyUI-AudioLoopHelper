"""Music-aware audio feature extraction for prompt scheduling.

Extracts BPM, key, chromagram, mel spectrogram, vocal F0, and song
structure using librosa. Outputs JSON (for LLM consumption), markdown
report, and PNG visualizations (for human review).

Requires the 'analysis' dependency group:
    uv sync --group analysis
    uv run --group analysis python scripts/analyze_audio_features.py audio.wav

This script is OFFLINE ONLY -- not imported by ComfyUI nodes.
Runtime nodes use torchaudio exclusively (see nodes.py).
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import librosa
    import numpy as np
except ImportError:
    print(
        "Error: librosa is required. Install with:\n"
        "  uv sync --group analysis",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import orjson
except ImportError:
    orjson = None

# Krumhansl-Schmuckler key profiles for major and minor keys
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MAJOR_PROFILE_NORM = _MAJOR_PROFILE / np.linalg.norm(_MAJOR_PROFILE)
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_MINOR_PROFILE_NORM = _MINOR_PROFILE / np.linalg.norm(_MINOR_PROFILE)
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def detect_bpm(audio: np.ndarray, sr: int) -> dict:
    """Detect BPM and beat timestamps.

    Returns:
        {"bpm": float, "beat_times": list[float]}
    """
    if np.max(np.abs(audio)) < 1e-6:
        return {"bpm": 0.0, "beat_times": []}

    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # tempo can be ndarray or scalar depending on librosa version
    bpm = float(np.atleast_1d(tempo)[0])

    return {
        "bpm": round(bpm, 1),
        "beat_times": [round(float(t), 3) for t in beat_times],
    }


def detect_key(audio: np.ndarray, sr: int, chroma: np.ndarray | None = None) -> dict:
    """Detect musical key using Krumhansl-Schmuckler algorithm.

    Returns:
        {"key": str, "confidence": float}
    """
    if chroma is None:
        chroma = compute_chromagram(audio, sr)
    mean_chroma = chroma.mean(axis=1)

    # Normalize
    norm = np.linalg.norm(mean_chroma)
    if norm < 1e-10:
        return {"key": "Unknown", "confidence": 0.0}
    mean_chroma = mean_chroma / norm

    best_corr = -2.0
    best_key = "C Major"

    for shift in range(12):
        rotated = np.roll(mean_chroma, -shift)

        # Major
        corr_major = float(np.corrcoef(rotated, _MAJOR_PROFILE_NORM)[0, 1])
        if corr_major > best_corr:
            best_corr = corr_major
            best_key = f"{_PITCH_CLASSES[shift]} Major"

        # Minor
        corr_minor = float(np.corrcoef(rotated, _MINOR_PROFILE_NORM)[0, 1])
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = f"{_PITCH_CLASSES[shift]} Minor"

    # Map correlation [-1, 1] to confidence [0, 1]
    confidence = max(0.0, min(1.0, (best_corr + 1.0) / 2.0))

    return {"key": best_key, "confidence": round(confidence, 3)}


def compute_chromagram(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute CQT chromagram (12 pitch classes x time frames).

    Returns:
        np.ndarray of shape (12, T)
    """
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    return chroma


def compute_mel_spectrogram(
    audio: np.ndarray, sr: int, n_mels: int = 128
) -> np.ndarray:
    """Compute mel spectrogram in dB scale.

    Returns:
        np.ndarray of shape (n_mels, T) in dB
    """
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db


def estimate_vocal_f0(audio: np.ndarray, sr: int) -> dict:
    """Estimate fundamental frequency and classify male/female.

    Male: 85-155 Hz, Female: 165-255 Hz.

    Returns:
        {"median_f0": float, "mean_f0": float, "classification": str,
         "f0_timeline": list[float|None]}
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, fmin=60, fmax=400, sr=sr
    )

    voiced_f0 = f0[~np.isnan(f0)]
    if len(voiced_f0) == 0:
        return {
            "median_f0": 0.0,
            "mean_f0": 0.0,
            "classification": "unknown",
            "f0_timeline": [],
        }

    median_f0 = float(np.median(voiced_f0))
    mean_f0 = float(np.mean(voiced_f0))

    # Classification based on median F0
    if median_f0 < 160:
        classification = "male"
    elif median_f0 > 160:
        classification = "female"
    else:
        classification = "ambiguous"

    return {
        "median_f0": round(median_f0, 1),
        "mean_f0": round(mean_f0, 1),
        "classification": classification,
        "f0_timeline": [round(float(v), 1) if not np.isnan(v) else None for v in f0],
    }


def detect_structure_librosa(audio: np.ndarray, sr: int, window_s: float = 2.0) -> list[dict]:
    """Detect song structure using onset strength + RMS energy.

    Returns list of sections with start, end, label, level.
    """
    hop_length = 512
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    if len(rms) == 0:
        return []

    # Aggregate to windows
    window_frames = max(1, int(window_s * sr / hop_length))
    windowed = []
    for i in range(0, len(rms), window_frames):
        chunk = rms[i : i + window_frames]
        t = times[min(i, len(times) - 1)]
        windowed.append((t, float(np.mean(chunk))))

    if not windowed:
        return []

    rms_values = sorted([v for _, v in windowed])
    p25 = rms_values[len(rms_values) // 4]
    p75 = rms_values[3 * len(rms_values) // 4]

    levels = []
    for t, v in windowed:
        if v < p25:
            levels.append((t, "quiet"))
        elif v < p75:
            levels.append((t, "medium"))
        else:
            levels.append((t, "loud"))

    # Group consecutive same-level windows
    raw_sections = []
    current_level = levels[0][1]
    section_start = levels[0][0]

    for i in range(1, len(levels)):
        if levels[i][1] != current_level:
            raw_sections.append({
                "start": round(section_start, 2),
                "end": round(levels[i][0], 2),
                "level": current_level,
            })
            current_level = levels[i][1]
            section_start = levels[i][0]

    # Final section
    duration = len(audio) / sr
    raw_sections.append({
        "start": round(section_start, 2),
        "end": round(duration, 2),
        "level": current_level,
    })

    # Merge short sections (< 6s)
    sections = [raw_sections[0]]
    for s in raw_sections[1:]:
        prev = sections[-1]
        dur = s["end"] - s["start"]
        prev_dur = prev["end"] - prev["start"]
        rank = {"quiet": 0, "medium": 1, "loud": 2}

        if dur < 6:
            if prev_dur < 6:
                prev["end"] = s["end"]
                if rank.get(s["level"], 0) > rank.get(prev["level"], 0):
                    prev["level"] = s["level"]
            else:
                prev["end"] = s["end"]
        elif prev_dur < 6:
            s["start"] = prev["start"]
            if rank.get(prev["level"], 0) > rank.get(s["level"], 0):
                s["level"] = prev["level"]
            sections[-1] = s
        else:
            sections.append(s)

    # Label sections
    for i, s in enumerate(sections):
        dur = s["end"] - s["start"]
        if i == 0 and s["level"] == "quiet" and dur < 15:
            s["label"] = "INTRO"
        elif i == len(sections) - 1 and s["level"] == "quiet":
            s["label"] = "OUTRO"
        elif s["level"] == "loud":
            s["label"] = "CHORUS"
        elif s["level"] == "quiet" and dur < 6:
            s["label"] = "BREAK"
        elif s["level"] == "quiet":
            s["label"] = "BRIDGE"
        elif s["level"] == "medium":
            s["label"] = "VERSE"
        else:
            s["label"] = s["level"].upper()

    return sections


# Section-to-prompt modifier mapping for LTX 2.3 i2v conventions.
#
# The action phrases use "singing" explicitly because LTX 2.3's audio-video
# joint attention drives lip sync off the text's action verb. Generic "is
# performing" loses the cross-attention signal. _build_action_phrase() below
# rewrites these to group form ("are singing together...") when the subject
# is detected as multi-character.
#
# Camera motions from CLAUDE.md prompt guide. Avoid dolly out (breaks limbs
# and faces) except for OUTRO where it's the expected visual pattern.
_SECTION_MODIFIERS = {
    "INTRO": {
        "framing": "In a wide establishing shot, static camera, locked off shot,",
        "lighting": "Soft lighting, gentle.",
        "action": "is singing softly, easing into the song",
        "audio_desc": "Quiet ambient tone, gentle room presence.",
    },
    "VERSE": {
        "framing": "In a medium shot,",
        "lighting": "Warm lighting, steady energy.",
        "action": "is singing with a steady voice",
        "audio_desc": "The voice fills the space. Soft ambient hum.",
    },
    "CHORUS": {
        "framing": "In a close-up,",
        "lighting": "Bright, dynamic lighting.",
        "action": "is singing with full power, voice rising",
        "audio_desc": "The voice is powerful and resonant.",
    },
    "BRIDGE": {
        "framing": "In a wide shot,",
        "lighting": "Moody, low contrast lighting.",
        "action": "is singing with quiet emotion",
        "audio_desc": "Subdued melody, reflective atmosphere.",
    },
    "OUTRO": {
        "framing": "In a wide shot, dolly out, camera pulling back,",
        "lighting": "Fading, gentle lighting.",
        "action": "is singing the final notes, voice trailing off",
        "audio_desc": "The sound fades quietly. Room tone settles.",
    },
    "BREAK": {
        "framing": "In a medium shot, static camera,",
        "lighting": "Dim lighting, still.",
        "action": "is singing softly, pausing in place",
        "audio_desc": "Brief instrumental moment, ambient quiet.",
    },
}

# Fallback for unknown section labels
_DEFAULT_MODIFIER = {
    "framing": "In a medium shot,",
    "lighting": "Natural lighting.",
    "action": "is singing with steady delivery",
    "audio_desc": "Music continues.",
}


def _is_multi_subject(subject: str) -> bool:
    """Heuristic: does the subject description name multiple people?

    True when the subject contains words that imply 2+ people. Used to pick
    the group-verb form ("are singing together") instead of the singular
    ("is singing"). Conservative -- false negatives just produce the
    singular verb, which is still valid output.
    """
    lower = subject.lower()
    # Explicit plural markers
    for marker in (" two ", " three ", " four ", " both ", " and ", " duo ", " pair "):
        if marker in f" {lower} ":
            # "and" without a person/noun after it is a common false trigger,
            # but since prompts almost always put "and" between subjects
            # ("a man and a woman"), the heuristic is accurate enough in practice.
            return True
    # "people" / "men" / "women" (plural nouns) at the start also indicate group
    for word in lower.split():
        if word in {"men", "women", "people", "singers", "performers", "couple"}:
            return True
    return False


def _build_action_phrase(base_action: str, multi: bool) -> str:
    """Rewrite a singular action phrase ("is singing ...") to group form
    when multi-character. For the group form, also tack on "together" so
    the cross-attention signal stays clear.
    """
    if not multi:
        return base_action
    # Replace the leading "is singing" with "are singing together"
    if base_action.startswith("is singing"):
        rest = base_action[len("is singing") :]
        return "are singing together" + rest
    # Fallback for any non-singing action phrase
    return base_action.replace("is ", "are ", 1) + " together"


# Scene-diversity taxonomy: tier (1-6) + sub-letter flavor.
#
# The tier is the AMBITION CEILING -- which beat pools layer onto the base
# section modifier. The sub-letter is a MOOD BUNDLE that biases lighting /
# location / style flavor (applied as a single extra phrase per prompt, so
# adjacent entries don't get mud-mouthed with it).
#
# Mapping to internal/ prompt docs:
#   1  performance_live        ~ internal/prompt.md   (static, safe)
#   2  performance_dynamic     ~ internal/prompt2.md  (camera + body beats)
#   3  cinematic               ~ internal/prompt3.md  (+ scene shifts)
#   4  narrative               ~ internal/prompt4.md  (+ physical-action arc)
#   5  stylized                (beyond prompt4: genre overlay)
#   6  avant_garde             (beyond prompt4: abstract / non-linear)
#
# Montage is ORTHOGONAL to the tier (see `montage` flag): a structural
# property that shortens dwell and layers emotional-arc language on top of
# whichever tier is active.
_DIVERSITY_TIERS: dict[int, str] = {
    1: "performance_live",
    2: "performance_dynamic",
    3: "cinematic",
    4: "narrative",
    5: "stylized",
    6: "avant_garde",
}

_DEFAULT_DIVERSITY = "2a"

# Which beat pools each tier activates. Later tiers inherit earlier pools.
# "camera"/"body" are gentle motion variation; "scene" is atmospheric;
# "narrative" is physical-action arc; "style" is genre overlay (tier 5+);
# "avant" is abstract/non-linear framing (tier 6).
_TIER_POOLS: dict[int, tuple[str, ...]] = {
    1: ("camera",),
    2: ("camera", "body"),
    3: ("camera", "body", "scene"),
    4: ("camera", "body", "scene", "narrative"),
    5: ("camera", "body", "scene", "narrative", "style"),
    6: ("camera", "body", "scene", "narrative", "style", "avant"),
}

# Variant-indexed beat pools: cycled via variant % len(beats). Each list
# supplies short phrases that ADD detail; the base modifier stays the
# primary descriptor.
_DYNAMIC_CAMERA_BEATS = {
    "INTRO": ["static camera", "slow dolly in", "static camera, locked off shot"],
    "VERSE": ["static camera", "slow dolly in", "slight focus shift"],
    "CHORUS": ["static camera", "slow jib up", "slow dolly in"],
    "BRIDGE": ["static camera", "slow focus shift", "static camera, locked off shot"],
    "OUTRO": ["dolly out, camera pulling back", "dolly out slowly", "camera pulling back"],
    "BREAK": ["static camera", "slight focus shift", "static camera"],
}

_DYNAMIC_BODY_BEATS = {
    "INTRO": ["mouth opening softly", "head tilted slightly", "eyes half-closed"],
    "VERSE": ["head bobbing slightly", "leaning forward", "slight sway at the shoulders"],
    "CHORUS": ["eyes wide, mouth open", "arms slightly raised", "chest forward, head tilted back"],
    "BRIDGE": ["eyes closed briefly", "head lowered", "hands at the sides"],
    "OUTRO": ["shoulders easing", "gaze dropping", "breath settling"],
    "BREAK": ["still pose", "gentle sway", "steady posture"],
}

_SCENE_SHIFT_BEATS = {
    "INTRO": ["the atmosphere quiet and still", "soft ambient glow", "cool muted tones"],
    "VERSE": ["warm steady ambience", "subtle reflections catching the light", "light shifting gently"],
    "CHORUS": ["colors intensifying", "bright accents across the scene", "energetic highlights in the frame"],
    "BRIDGE": ["shadows lengthening", "moody low-key tones", "atmosphere tightening"],
    "OUTRO": ["colors fading toward stillness", "light softening", "warmth draining from the scene"],
    "BREAK": ["brief pocket of quiet", "ambience settling", "stillness in the frame"],
}

_NARRATIVE_BEATS = {
    "INTRO": ["standing in place", "turning slowly toward the camera", "settling into position"],
    "VERSE": ["shifting weight between feet", "taking a half-step forward", "steadying the stance"],
    "CHORUS": ["leaning into the performance", "stepping forward decisively", "raising the head higher"],
    "BRIDGE": ["pulling back, looking inward", "pausing mid-motion", "gathering for what comes next"],
    "OUTRO": ["easing back, gaze softening", "settling into stillness", "slowly letting go"],
    "BREAK": ["holding position", "a brief pause", "holding the moment"],
}

# Tier 5 stylized overlay beats -- genre / treatment flavor that layers ON
# TOP of tier 4. Picked once per prompt (not per-section), so the overlay
# rotates across variants keeping tone consistent within a section.
_STYLE_BEATS = {
    "INTRO": ["treated with a stylized overlay", "with a graphic-design feel", "with stylized color treatment"],
    "VERSE": ["with stylized color treatment", "with a graphic-design feel", "treated with a stylized overlay"],
    "CHORUS": ["with a heightened stylized overlay", "with bold graphic accents", "with a treated color bath"],
    "BRIDGE": ["with a low-saturation stylized look", "with a noir-inflected treatment", "with a muted palette overlay"],
    "OUTRO": ["with color draining to a treated palette", "with a stylized wash", "with stylized fade treatment"],
    "BREAK": ["with a still stylized overlay", "with muted treatment", "with a subdued stylized flourish"],
}

# Tier 6 avant-garde beats -- abstract / non-linear framing that further
# departs from literal depiction. Still anchored on the singing subject so
# lip sync cross-attention survives.
_AVANT_BEATS = {
    "INTRO": ["composition edging toward abstraction", "frame feeling unmoored", "composition leaning graphic and flat"],
    "VERSE": ["composition breaking from literal space", "frame tilting off-axis", "composition drifting toward the abstract"],
    "CHORUS": ["composition fragmenting with rhythm", "frame pulsing with form", "composition collapsing toward pure shape"],
    "BRIDGE": ["composition dissolving inward", "frame softening into suggestion", "composition quieting to essentials"],
    "OUTRO": ["composition resolving toward pure form", "frame collapsing to a gesture", "composition settling into stillness"],
    "BREAK": ["composition holding a single abstract beat", "frame pausing on pure form", "composition steady and reduced"],
}

# Dispatch table: tier-pool name -> beats-by-label dict. _build_prompt_for_section
# iterates the tier's active pools in `_TIER_POOLS` order and looks up beats here.
_POOL_BEATS: dict[str, dict] = {
    "camera": _DYNAMIC_CAMERA_BEATS,
    "body": _DYNAMIC_BODY_BEATS,
    "scene": _SCENE_SHIFT_BEATS,
    "narrative": _NARRATIVE_BEATS,
    "style": _STYLE_BEATS,
    "avant": _AVANT_BEATS,
}

# Montage emotional-arc beats. Applied ONLY when `montage=True`. These carry
# the "music drives art drives narrative" feeling, biased by section.
_MONTAGE_ARC_BEATS = {
    "INTRO": ["the feeling gathering", "a quiet build beginning", "stillness about to break"],
    "VERSE": ["the feeling building", "tension collecting beat by beat", "momentum gathering"],
    "CHORUS": ["the feeling releasing", "the emotional peak landing", "catharsis arriving"],
    "BRIDGE": ["the feeling turning inward", "a moment of reckoning", "the emotional pivot"],
    "OUTRO": ["the feeling settling", "release easing into stillness", "the final emotional note"],
    "BREAK": ["a held breath between beats", "stillness in the emotional arc", "a pause before the next swell"],
}

# Sub-letter mood bundles. Appended as a SINGLE phrase to the extras, so the
# prompt doesn't get mud-mouthed. Keyed by f"{tier}{letter}" for clarity.
# Missing sub-letters fall back to "" (no flavor note, still valid).
_MOOD_BUNDLES: dict[str, str] = {
    "1a": "tight performance framing, sweat visible",
    "1b": "wide stage framing, warm stage wash",
    "1c": "controlled studio lighting, subtle film grain",
    "2a": "handheld energy, rock-video motion",
    "2b": "smooth dolly motion, pop-video polish",
    "3a": "urban night palette, neon reflections",
    "3b": "natural-light palette, open outdoor feel",
    "3c": "quiet interior palette, introspective framing",
    "3d": "classic performance + b-roll intercut feel",
    "4a": "linear story beat progression",
    "4b": "dreamlike flashback tint, memory-tinged overlay",
    "5a": "noir monochrome, high contrast",
    "5b": "surreal saturated palette",
    "5c": "retro period grain, period wardrobe feel",
    "6a": "abstract non-literal framing",
}


def _parse_diversity(value: str | None) -> tuple[int, str | None]:
    """Parse a diversity spec like '3b' or '4' into (tier, sub_letter).

    Falls back to _DEFAULT_DIVERSITY on malformed input rather than raising,
    so CLI callers get lenient behavior.
    """
    v = (value or _DEFAULT_DIVERSITY).strip().lower()
    try:
        tier = int(v[0])
    except (ValueError, IndexError):
        tier = int(_DEFAULT_DIVERSITY[0])
    if tier not in _DIVERSITY_TIERS:
        tier = int(_DEFAULT_DIVERSITY[0])
    sub = v[1:] if len(v) > 1 else None
    return tier, sub


def _pick_beat(beats_by_label: dict, label: str, variant: int) -> str:
    """Return the variant-th beat for this label, with cycling fallback."""
    beats = beats_by_label.get(label)
    if not beats:
        return ""
    return beats[variant % len(beats)]


def _build_prompt_for_section(
    section: dict,
    subject: str,
    diversity: str = _DEFAULT_DIVERSITY,
    montage: bool = False,
) -> str:
    """Build the prompt string for one section.

    Single source of truth shared by `get_node_169_prompt` and
    `_generate_subject_schedule` — ensures Node 169 equals the first
    schedule entry byte-for-byte (CLAUDE.md constraint).
    """
    mods = _SECTION_MODIFIERS.get(section["label"], _DEFAULT_MODIFIER)
    multi = _is_multi_subject(subject)
    action = _build_action_phrase(mods["action"], multi)
    label = section["label"]
    variant = section.get("variant", 0)
    tier, sub = _parse_diversity(diversity)
    pools = _TIER_POOLS.get(tier, _TIER_POOLS[2])

    extras: list[str] = []
    for pool in pools:
        beat = _pick_beat(_POOL_BEATS[pool], label, variant)
        if beat:
            extras.append(beat)

    # Mood bundle — one phrase per prompt, keyed by full "tier+sub" code.
    if sub:
        mood = _MOOD_BUNDLES.get(f"{tier}{sub}")
        if mood:
            extras.append(mood)

    # Montage arc — one emotional-arc phrase per prompt, orthogonal to tier.
    if montage:
        beat = _pick_beat(_MONTAGE_ARC_BEATS, label, variant)
        if beat:
            extras.append(beat)

    extras_text = f", {', '.join(extras)}" if extras else ""

    return (
        f"Style: cinematic. {mods['framing']} {subject} {action}{extras_text}. "
        f"{mods['lighting']} {mods['audio_desc']}"
    )


# Each schedule entry should cover roughly one iteration window so the
# prompt changes keep pace with the visual loop. Our loop stride is
# ~18.88s; target a schedule entry every ~20s and split anything over
# ~30s. Without this, raw section detection can produce single entries
# spanning 1-2 minutes which feels static.
_SCHEDULE_TARGET_SECONDS = 20.0
_SCHEDULE_SPLIT_THRESHOLD_SECONDS = 30.0

# Montage mode uses faster cuts — subdivide anything over ~18s and target
# ~12s per entry so the emotional arc progresses more briskly.
_MONTAGE_TARGET_SECONDS = 12.0
_MONTAGE_SPLIT_THRESHOLD_SECONDS = 18.0


def _subdivide_long_sections(
    sections: list[dict],
    target: float = _SCHEDULE_TARGET_SECONDS,
    split_above: float = _SCHEDULE_SPLIT_THRESHOLD_SECONDS,
) -> list[dict]:
    """Split sections longer than `split_above` into ~target-sized chunks.

    Returns a new list. Sections already short enough pass through with
    `variant=0`. Longer sections become N chunks where N is
    round(duration / target), each getting the same label/level and an
    incrementing variant index.
    """
    result: list[dict] = []
    for s in sections:
        dur = s["end"] - s["start"]
        if dur <= split_above:
            result.append({**s, "variant": 0})
            continue
        n = max(2, int(round(dur / target)))
        chunk_dur = dur / n
        for i in range(n):
            result.append({
                "start": s["start"] + i * chunk_dur,
                "end": s["start"] + (i + 1) * chunk_dur,
                "label": s["label"],
                "level": s["level"],
                "variant": i,
            })
    return result


def _prepare_sections(sections: list[dict], montage: bool) -> list[dict]:
    """Apply subdivision with mode-appropriate dwell target.

    Shared between the schedule builder and get_node_169_prompt so the
    first chunk (Node 169) sees the SAME subdivision the schedule sees.
    That's the invariant documented in CLAUDE.md: Node 169 MUST equal the
    first schedule entry byte-for-byte.
    """
    if montage:
        return _subdivide_long_sections(
            sections,
            target=_MONTAGE_TARGET_SECONDS,
            split_above=_MONTAGE_SPLIT_THRESHOLD_SECONDS,
        )
    return _subdivide_long_sections(sections)


def generate_schedule_suggestion(
    sections: list[dict],
    subject: str = "",
    trim_offset: float = 0.0,
    diversity: str = _DEFAULT_DIVERSITY,
    montage: bool = False,
) -> str:
    """Generate a TimestampPromptSchedule text block from sections.

    Without subject: produces placeholder prompts with section labels.
    With subject: produces full LTX 2.3 i2v prompts using the subject
    description wrapped with section-appropriate camera, lighting, and
    energy modifiers. Copy-pasteable into TimestampPromptSchedule.

    `diversity` and `montage` control the ambition / pacing of the output
    — see `_build_prompt_for_section`.

    Args:
        sections: list of dicts with start, end, label, level keys.
        subject: scene description (e.g., "a woman singing in a workshop").
            If empty, falls back to placeholder output.
        trim_offset: seconds to subtract from timestamps.
        diversity: tier+sub-letter code (e.g., "3b"). Default "2a".
        montage: when True, cuts faster and adds emotional-arc language.
    """
    prepared = _prepare_sections(sections, montage)
    if not subject:
        return _generate_placeholder_schedule(prepared, trim_offset)
    return _generate_subject_schedule(
        prepared, subject, trim_offset, diversity, montage
    )


def get_node_169_prompt(
    sections: list[dict],
    subject: str = "",
    trim_offset: float = 0.0,
    diversity: str = _DEFAULT_DIVERSITY,
    montage: bool = False,
) -> str:
    """Extract the node 169 initial render prompt.

    Returns the prompt text for the first section (matching the 0:00
    schedule entry). Node 169 covers trimmed 0:00 to ~20s and must
    match the schedule's first entry to avoid visual discontinuity.

    Without subject: returns a placeholder.
    """
    if not sections:
        return ""

    prepared = _prepare_sections(sections, montage)

    # Find the first prepared chunk that survives the trim offset.
    first_section = None
    for s in prepared:
        if s["end"] - trim_offset > 0:
            first_section = s
            break
    if not first_section:
        first_section = prepared[0]

    if not subject:
        return f"[{first_section['label']} - {first_section['level']}] describe initial scene"

    return _build_prompt_for_section(
        first_section, subject, diversity=diversity, montage=montage
    )


def _build_schedule(
    sections: list[dict],
    trim_offset: float,
    build_prompt,
) -> str:
    """Shared loop for schedule generation. build_prompt(section) -> str."""
    lines = []
    for i, s in enumerate(sections):
        start = max(0, s["start"] - trim_offset)
        end = max(0, s["end"] - trim_offset)
        if end <= 0:
            continue

        prompt = build_prompt(s)
        ts_start = _fmt_ts(start)

        if i == len(sections) - 1 or s["label"] == "OUTRO":
            lines.append(f"{ts_start}+: {prompt}")
        else:
            lines.append(f"{ts_start}-{_fmt_ts(end)}: {prompt}")

    return "\n".join(lines)


def _generate_placeholder_schedule(
    sections: list[dict], trim_offset: float
) -> str:
    """Placeholder output with section labels for manual editing."""
    def build(s):
        return f"[{s['label']} - {s['level']}] describe action and audio here"
    return _build_schedule(sections, trim_offset, build)


def _generate_subject_schedule(
    sections: list[dict],
    subject: str,
    trim_offset: float,
    diversity: str,
    montage: bool,
) -> str:
    """Full prompt schedule with subject wrapped in section modifiers.

    Uses the SAME builder as `get_node_169_prompt` so the first schedule
    line is bit-exact equal to Node 169. See `_build_prompt_for_section`.
    """
    return _build_schedule(
        sections,
        trim_offset,
        lambda s: _build_prompt_for_section(
            s, subject, diversity=diversity, montage=montage
        ),
    )


def _fmt_ts(seconds: float) -> str:
    """Format seconds as M:SS for schedule timestamps."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


_LLM_SYSTEM_PROMPT = """\
You are a video prompt engineer for LTX 2.3, an audio-visual video generation model.

WORKFLOW CONTEXT
- Image-to-video (i2v): an init_image provides the first frame.
- A full audio track is FROZEN as conditioning (noise=0).
- The model generates ONLY video; audio-video cross-attention drives lip sync.
- Video is generated in ~20-second windows with overlapping loop iterations.
- The init_image anchors spatial composition throughout.

INPUTS YOU RECEIVE
1. This analysis JSON (sections, BPM, key, vocal F0, workflow context).
2. The init_image (first frame) — pasted alongside this system prompt.
3. A subject string describing the singer(s).

YOU PRODUCE TWO OUTPUTS
1. node_169_prompt: one paragraph for the initial ~20s.
2. schedule: TimestampPromptSchedule, one entry per song section (further
   subdivided so entries dwell ~20s each — or ~12s in montage mode).

==========================================================================
INFERENCE: WHAT THE INIT IMAGE ENCODES (DO NOT RE-DESCRIBE)
==========================================================================
The init image already commits the model to:
  - Style family (live-action, animated, comic / graphic-novel, 3D-render,
    stop-motion, painterly, etc.)
  - Color palette and overall mood
  - Setting (indoor/outdoor, urban/natural, wardrobe, era)
  - Subject appearance (face, body, clothing, number of people)

Do NOT re-describe these in the schedule. Re-describing them wastes tokens
and invites the text-conditioning to fight what the image already commits
to (e.g. writing "photorealistic" on a comic-style init forces a tug-of-war).

WHAT THE SCHEDULE DRIVES (DO DESCRIBE)
  - Camera framing and motion over time.
  - Body language / performance beats.
  - Lighting SHIFTS (accents brightening, shadows lengthening), not a
    literal restatement of the palette.
  - Scene cuts, location shifts (textual — init anchors are still primary).
  - Emotional arc.

STYLE-APPROPRIATE BEAT POOLS (pick the pool matching the image):
  - Animated / comic / graphic-novel: speed lines, panel transitions,
    supersaturation, impact frames, silhouetted accents, motion blur lines.
  - Live-action cinematic: rack focus, practical lighting, handheld/dolly,
    lens flares, shallow depth of field.
  - 3D-render / stylized: stylized color treatment, stylized shadows,
    treated palette overlay.
Infer from the image which pool applies; mixing pools across entries reads
as visual chaos.

==========================================================================
HARD RULES (non-negotiable)
==========================================================================

R1. The singing verb drives lip sync. EVERY entry MUST contain an explicit
    form of "singing":
    - Single performer: "is singing ..." (present progressive).
    - Multiple performers: "are singing together ..." (group verb, always
      with "together"). Use this form whenever the subject names 2+
      people (e.g. "two men", "a man and a woman", "the duo").
    Do NOT substitute "is performing", "is vocalizing", or any generic
    verb. If the scene is explicitly instrumental, use "is playing
    <instrument>" and skip "singing".

R2. node_169_prompt MUST be IDENTICAL, character-for-character, to the
    first schedule entry's prompt text (everything after the timestamp
    and colon). Copy it verbatim. Any drift causes a visible seam at
    the ~20s boundary when the loop starts. Do not summarize, shorten,
    or rephrase.

R3. Keep the SUBJECT description identical across every entry. Only vary:
    framing, camera, lighting, body language, performance beats. Do NOT
    re-describe the environment (the init_image sets it).

R4. For multi-person scenes: position-anchor each person explicitly
    ("the man on the left in the dark jacket, the woman on the right
    with short hair") inside the subject string. Do NOT use "crowd",
    "group", or undescribed collectives.

R5. No meta-language. No "The scene opens with...", "Cut to...",
    "camera shows...". Begin each prompt with "Style: cinematic." and
    move straight to subject + action.

R6. Audio direction:
    - Do NOT describe the song itself ("voice surging", "music
      swelling") — the model already hears the audio.
    - DO describe ambient / diegetic sounds that are NOT in the audio
      track ("soft room tone", "faint hum of fluorescent lights").
    - Vocal delivery qualifiers are encouraged: "in a low gravelly
      voice", "with bright clear tone", "brisk rhythmic delivery".

R7. Camera motion:
    - Default: "static camera, locked off shot".
    - Available motions: dolly in, dolly left, dolly right, jib up,
      jib down, focus shift.
    - AVOID dolly out — it breaks limbs and faces. Exception: the
      final OUTRO entry may use it for fade-out.

R8. One paragraph per entry, no markdown or bullets, ~200 words max.
    Use "is singing" in the present progressive tense — not past tense
    ("sang") and not generic nouns ("singer").

R9. Schedule timestamps MUST fall on integer multiples of
    `workflow_context.stride_seconds` — the loop advances in fixed
    stride-sized steps (typically ~17-19s). Boundaries that fall
    mid-stride cause one iteration to run on a mixed conditioning that
    looks discontinuous on video.
    - Truncate to integer seconds, then snap to the nearest multiple of
      stride_seconds (in seconds, then format as M:SS).
    - Example: stride_seconds = 17.88. Natural boundaries at 0, 18, 36,
      54, 71, 89, 107, 125, 143, 161, 179, ... (i * 17.88, truncated
      to integer seconds → "0:00", "0:17", "0:35", "0:53", "1:11",
      "1:29", "1:47", "2:05", "2:23", "2:41", "2:59", ...).
    - If a natural scene transition doesn't land on a stride multiple
      (e.g. a punchline at 0:55 when the grid is at 0:53), round DOWN
      to the nearest grid point — do NOT split a stride window.
    - The runtime auto-snaps as a safety net, so mismatched timestamps
      won't crash. But emitting pre-snapped schedules means the user's
      workflow widget shows exactly what will run (no silent runtime
      drift), which is strictly better.

==========================================================================
AMBITION TIERS (scene_diversity)
==========================================================================
The caller passes a scene_diversity code in workflow_context. Match your
output's ambition to that tier. Tiers layer ON TOP of each other — tier 3
includes tier 2 includes tier 1, etc. Sub-letters are mood bundles.

  1 performance_live      single-camera concert feel
     1a close-up concert    (tight framing, sweat/mouth detail)
     1b wide stage          (band-room feel, warm stage wash)
     1c studio-live         (controlled lighting, subtle film grain)
  2 performance_dynamic   camera + body beats rotate (DEFAULT)
     2a handheld energetic  (shaky, MTV-rock)
     2b steady-cam polished (smooth dolly, pop-video polish)
  3 cinematic             + environmental storytelling / scene shifts
     3a urban night         (neon, rain, street)
     3b natural outdoor     (golden hour, landscape)
     3c interior character  (domestic, emotional, introspective)
     3d performance + b-roll (classic radio-single format)
  4 narrative             + physical-action arc / loose story
     4a linear story
     4b flashback / dream structure
  5 stylized              + genre overlay (noir, sci-fi, surreal)
     5a noir / monochrome
     5b surreal / dreamlike
     5c retro / period
  6 avant_garde           non-linear, abstract, performative

MONTAGE FLAG (orthogonal)
If montage=true, each entry must:
  - Advance an emotional BEAT (not merely describe a scene).
  - Use emotional-arc language: "the feeling building", "tension
    releasing", "a held breath", "catharsis arriving".
  - Dwell ~12s instead of ~20s (more entries for the same duration).
Montage is structural pacing, not a tier — it layers over whichever tier
is active. Think Arcane (Netflix) music-driven montages: music drives
art drives narrative drives emotion.

==========================================================================
WORKED EXAMPLE — single character, tier 2a (default)
==========================================================================
Input subject: "a woman in her 30s with dark hair"
Analysis: INTRO 0:00-0:20, VERSE 0:20-1:00, CHORUS 1:00-2:00, OUTRO 2:00+
scene_diversity: 2a

Output:

node_169_prompt: Style: cinematic. In a wide establishing shot, static camera, locked off shot, a woman in her 30s with dark hair is singing softly, easing into the song, static camera, mouth opening softly, handheld energy, rock-video motion. Soft lighting, gentle. Quiet ambient tone, gentle room presence.

schedule:
0:00-0:20: Style: cinematic. In a wide establishing shot, static camera, locked off shot, a woman in her 30s with dark hair is singing softly, easing into the song, static camera, mouth opening softly, handheld energy, rock-video motion. Soft lighting, gentle. Quiet ambient tone, gentle room presence.
0:20-1:00: Style: cinematic. In a medium shot, a woman in her 30s with dark hair is singing with a steady voice, static camera, head bobbing slightly, handheld energy, rock-video motion. Warm lighting, steady energy. The voice fills the space. Soft ambient hum.
1:00-2:00: Style: cinematic. In a close-up, a woman in her 30s with dark hair is singing with full power, voice rising, static camera, eyes wide, mouth open, handheld energy, rock-video motion. Bright, dynamic lighting. The voice is powerful and resonant.
2:00+: Style: cinematic. In a wide shot, dolly out, camera pulling back, a woman in her 30s with dark hair is singing the final notes, voice trailing off, dolly out, camera pulling back, shoulders easing, handheld energy, rock-video motion. Fading, gentle lighting. The sound fades quietly. Room tone settles.

(Note: first schedule line is byte-exact to node_169_prompt — that is R2.)

==========================================================================
WORKED EXAMPLE — multi-character, tier 3b (cinematic, natural outdoor)
==========================================================================
Input subject: "two men on a rooftop, the man on the left in a green jacket, the man on the right in a black shirt"
Analysis: INTRO 0:00-0:28, VERSE 0:28-1:15, CHORUS 1:15-2:05, OUTRO 2:05+
scene_diversity: 3b

Output:

node_169_prompt: Style: cinematic. In a wide establishing shot, static camera, locked off shot, two men on a rooftop, the man on the left in a green jacket, the man on the right in a black shirt are singing together softly, easing into the song, static camera, mouth opening softly, the atmosphere quiet and still, natural-light palette, open outdoor feel. Soft lighting, gentle. Quiet ambient tone, gentle room presence.

schedule:
0:00-0:28: Style: cinematic. In a wide establishing shot, static camera, locked off shot, two men on a rooftop, the man on the left in a green jacket, the man on the right in a black shirt are singing together softly, easing into the song, static camera, mouth opening softly, the atmosphere quiet and still, natural-light palette, open outdoor feel. Soft lighting, gentle. Quiet ambient tone, gentle room presence.
0:28-1:15: Style: cinematic. In a medium shot, two men on a rooftop, the man on the left in a green jacket, the man on the right in a black shirt are singing together with a steady voice, static camera, head bobbing slightly, warm steady ambience, natural-light palette, open outdoor feel. Warm lighting, steady energy. The voice fills the space. Soft ambient hum.
1:15-2:05: Style: cinematic. In a close-up, two men on a rooftop, the man on the left in a green jacket, the man on the right in a black shirt are singing together with full power, voice rising, static camera, eyes wide, mouth open, colors intensifying, natural-light palette, open outdoor feel. Bright, dynamic lighting. The voice is powerful and resonant.
2:05+: Style: cinematic. In a wide shot, dolly out, camera pulling back, two men on a rooftop, the man on the left in a green jacket, the man on the right in a black shirt are singing together the final notes, voice trailing off, dolly out, camera pulling back, shoulders easing, colors fading toward stillness, natural-light palette, open outdoor feel. Fading, gentle lighting. The sound fades quietly. Room tone settles.

(Note: "are singing together" in every entry. First line byte-exact to
node_169_prompt. The subject is identical across all entries.)

==========================================================================
WORKED EXAMPLE — montage, tier 4a (narrative + montage pacing)
==========================================================================
Input subject: "a young woman walking through a snowy alley at dusk"
Analysis: INTRO 0:00-0:12, VERSE 0:12-1:00, CHORUS 1:00-2:00, OUTRO 2:00+
scene_diversity: 4a
montage: true

Each entry must advance an emotional beat, not just describe a scene.
Dwell times shrink to ~12s so more entries cover the same runtime.

node_169_prompt: Style: cinematic. In a wide establishing shot, static camera, locked off shot, a young woman walking through a snowy alley at dusk is singing softly, easing into the song, static camera, mouth opening softly, the atmosphere quiet and still, standing in place, linear story beat progression, the feeling gathering. Soft lighting, gentle. Quiet ambient tone, gentle room presence.

schedule:
0:00-0:12: <byte-exact copy of node_169_prompt above>
0:12-0:24: Style: cinematic. In a medium shot, a young woman walking through a snowy alley at dusk is singing with a steady voice, slow dolly in, leaning forward, subtle reflections catching the light, taking a half-step forward, linear story beat progression, tension collecting beat by beat. Warm lighting, steady energy. The voice fills the space. Soft ambient hum.
... (more short entries as the song progresses) ...
2:00+: Style: cinematic. In a wide shot, dolly out, camera pulling back, a young woman walking through a snowy alley at dusk is singing the final notes, voice trailing off, dolly out, camera pulling back, shoulders easing, colors fading toward stillness, easing back, gaze softening, linear story beat progression, release easing into stillness. Fading, gentle lighting. The sound fades quietly. Room tone settles.

(Note: montage entries layer emotional-arc language — "the feeling
gathering", "tension collecting", "release easing into stillness" —
ON TOP of the tier-4 narrative beats. That combination is what gives
Arcane-style music-driven sequences their emotional density.)

==========================================================================
OUTPUT FORMAT
==========================================================================

node_169_prompt: <single paragraph, MUST equal the first schedule prompt verbatim>

schedule:
<M:SS-M:SS: prompt>
<M:SS-M:SS: prompt>
...
<M:SS+: prompt>   # last entry is open-ended

Use the analysis's section labels and energy levels to choose framing
(quiet → wider / static; loud → close-up / dynamic). Subdivide long
sections so each entry dwells roughly the iteration window (~20s
default, ~12s in montage mode)."""


def format_json_report(
    bpm_result: dict,
    key_result: dict,
    sections: list[dict],
    f0_result: dict | None = None,
    duration: float = 0.0,
    trim_offset: float = 0.0,
    window_seconds: float = 19.88,
    overlap_seconds: float = 2.0,
    subject: str = "",
    init_image_description: str = "",
    diversity: str = _DEFAULT_DIVERSITY,
    montage: bool = False,
) -> dict:
    """Build structured JSON report for LLM consumption.

    Includes audio analysis, workflow timing context, and an LLM system
    prompt that embeds all prompt engineering rules for the i2v + frozen
    audio loop workflow. The user pastes this JSON into an LLM alongside
    their creative direction to generate a complete TimestampPromptSchedule.

    Returns a dict (caller serializes with orjson or stdlib json).
    """
    stride = window_seconds - overlap_seconds

    report = {
        "duration": round(duration, 2),
        "bpm": bpm_result.get("bpm", 0.0),
        "key": key_result.get("key", "Unknown"),
        "key_confidence": key_result.get("confidence", 0.0),
        "sections": sections,
    }

    if f0_result:
        report["vocal_f0"] = {
            "median_hz": f0_result.get("median_f0", 0.0),
            "mean_hz": f0_result.get("mean_f0", 0.0),
            "classification": f0_result.get("classification", "unknown"),
        }

    tier, sub = _parse_diversity(diversity)
    tier_name = _DIVERSITY_TIERS.get(tier, "performance_dynamic")
    mood_bundle = _MOOD_BUNDLES.get(f"{tier}{sub}") if sub else None

    report["workflow_context"] = {
        "trim_offset": trim_offset,
        "window_seconds": window_seconds,
        "overlap_seconds": overlap_seconds,
        "stride_seconds": round(stride, 2),
        "initial_render_covers": (
            f"trimmed 0:00 to {_fmt_ts(window_seconds)} "
            f"(song {_fmt_ts(trim_offset)} to {_fmt_ts(trim_offset + window_seconds)})"
        ),
        "schedule_starts_at": f"trimmed {_fmt_ts(stride)} (iteration 1)",
        "subject": subject,
        "init_image_description": init_image_description,
        "scene_diversity": f"{tier}{sub or ''}",
        "scene_diversity_tier_name": tier_name,
        "scene_diversity_mood_bundle": mood_bundle,
        "montage": montage,
    }

    report["llm_system_prompt"] = _LLM_SYSTEM_PROMPT

    return report


def format_markdown_report(
    audio_path: str,
    duration: float,
    bpm_result: dict,
    key_result: dict,
    sections: list[dict],
    f0_result: dict | None = None,
    trim_offset: float = 0.0,
    subject: str = "",
    window_seconds: float = 19.88,
    overlap_seconds: float = 2.0,
    init_image_description: str = "",
    diversity: str = _DEFAULT_DIVERSITY,
    montage: bool = False,
) -> str:
    """Format a human-readable markdown report."""
    lines = []
    lines.append(f"# Audio Feature Analysis: {os.path.basename(audio_path)}")
    lines.append(f"Last updated: {_today()}")
    lines.append("")

    # Summary
    lines.append(f"**Duration:** {duration:.1f}s ({_fmt_ts(duration)})")
    lines.append(f"**BPM:** {bpm_result['bpm']}")
    lines.append(f"**Key:** {key_result['key']} (confidence: {key_result['confidence']:.2f})")
    lines.append(f"**Beats detected:** {len(bpm_result.get('beat_times', []))}")
    if f0_result and f0_result.get("classification") != "unknown":
        lines.append(
            f"**Vocal F0:** {f0_result['median_f0']} Hz ({f0_result['classification']})"
        )
    if trim_offset > 0:
        lines.append(f"**Trim offset:** {trim_offset}s")
    lines.append("")

    # Structure table
    lines.append("## Song Structure")
    lines.append("")
    lines.append("| Section | Time | Duration | Energy |")
    lines.append("|---------|------|----------|--------|")
    for s in sections:
        t_range = f"{_fmt_ts(s['start'])}-{_fmt_ts(s['end'])}"
        dur = s["end"] - s["start"]
        lines.append(f"| {s['label']:8s} | {t_range:11s} | {dur:5.0f}s   | {s['level']:6s} |")
    lines.append("")

    # Node 169 prompt (initial render)
    lines.append("## Node 169 (initial render prompt)")
    lines.append("")
    lines.append("Paste this into node 169 (CLIPTextEncode). Covers the first ~20 seconds.")
    lines.append("")
    lines.append("```")
    lines.append(get_node_169_prompt(
        sections, subject=subject, trim_offset=trim_offset,
        diversity=diversity, montage=montage,
    ))
    lines.append("```")
    lines.append("")

    # Prompt schedule suggestion
    lines.append("## TimestampPromptSchedule (node 1558)")
    lines.append("")
    lines.append("Paste this into the schedule text box:")
    lines.append("")
    lines.append("```")
    lines.append(generate_schedule_suggestion(
        sections, subject=subject, trim_offset=trim_offset,
        diversity=diversity, montage=montage,
    ))
    lines.append("```")
    lines.append("")

    # JSON block for LLM
    lines.append("## Structured Data (for LLM prompt)")
    lines.append("")
    lines.append("Paste this into your LLM prompt for schedule generation:")
    lines.append("")
    lines.append("```json")
    json_report = format_json_report(
        bpm_result, key_result, sections, f0_result, duration,
        trim_offset=trim_offset, window_seconds=window_seconds,
        overlap_seconds=overlap_seconds, subject=subject,
        init_image_description=init_image_description,
        diversity=diversity, montage=montage,
    )
    if orjson:
        lines.append(orjson.dumps(json_report, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode())
    else:
        import json
        lines.append(json.dumps(json_report, indent=2))
    lines.append("```")

    return "\n".join(lines)


def _today() -> str:
    from datetime import date
    return date.today().isoformat()


def save_png_visualizations(
    audio: np.ndarray,
    sr: int,
    output_dir: str,
    basename: str,
    precomputed_chroma: np.ndarray | None = None,
) -> list[str]:
    """Save spectrogram, chromagram, and onset envelope as PNGs.

    Returns list of saved file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping PNG output", file=sys.stderr)
        return []

    saved = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mel spectrogram
    fig, ax = plt.subplots(figsize=(12, 4))
    S_db = compute_mel_spectrogram(audio, sr)
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(f"Mel Spectrogram: {basename}")
    fig.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    path = output_dir / f"{basename}_mel_spectrogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path))

    # Chromagram (use precomputed if available to avoid redundant CQT)
    fig, ax = plt.subplots(figsize=(12, 3))
    chroma = precomputed_chroma if precomputed_chroma is not None else compute_chromagram(audio, sr)
    librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=ax)
    ax.set_title(f"Chromagram: {basename}")
    fig.tight_layout()
    path = output_dir / f"{basename}_chromagram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path))

    # Onset strength envelope
    fig, ax = plt.subplots(figsize=(12, 3))
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    ax.plot(times, onset_env, linewidth=0.5)
    ax.set_title(f"Onset Strength: {basename}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strength")
    fig.tight_layout()
    path = output_dir / f"{basename}_onset_envelope.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path))

    return saved


def analyze_file(
    audio_path: str,
    sr: int = 22050,
    trim_offset: float = 0.0,
    vocal_track: str | None = None,
) -> dict:
    """Full analysis pipeline. Returns all results as a dict."""
    audio, file_sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = len(audio) / sr

    chroma = compute_chromagram(audio, sr)
    bpm_result = detect_bpm(audio, sr)
    key_result = detect_key(audio, sr, chroma=chroma)
    sections = detect_structure_librosa(audio, sr)

    f0_result = None
    if vocal_track and os.path.exists(vocal_track):
        vocal_audio, _ = librosa.load(vocal_track, sr=sr, mono=True)
        f0_result = estimate_vocal_f0(vocal_audio, sr)
    elif vocal_track is None:
        # Attempt F0 on main audio (noisy but sometimes useful)
        f0_result = estimate_vocal_f0(audio, sr)

    return {
        "audio": audio,
        "sr": sr,
        "duration": duration,
        "bpm": bpm_result,
        "key": key_result,
        "sections": sections,
        "f0": f0_result,
        "chroma": chroma,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Music-aware audio feature extraction for prompt scheduling"
    )
    parser.add_argument("audio_path", help="Path to audio file (WAV, MP3, etc.)")
    parser.add_argument("--output", "-o", help="Write markdown report to file")
    parser.add_argument("--json", "-j", help="Write JSON report to file")
    parser.add_argument("--png-dir", help="Directory for PNG visualizations")
    parser.add_argument("--trim", "-t", type=float, default=0.0,
                        help="Trim offset in seconds for schedule timestamps")
    parser.add_argument("--vocal-track", help="Separated vocal track for F0 analysis")
    parser.add_argument("--subject", "-s",
                        help="Scene description for prompt templates (e.g., 'a woman singing in a workshop')")
    parser.add_argument("--image-desc",
                        help="Init image description for LLM context (e.g., 'Man with guitar, dim room, brick wall')")
    parser.add_argument("--window", type=float, default=19.88,
                        help="Window seconds (default: 19.88) -- for timing context in JSON")
    parser.add_argument("--overlap", type=float, default=2.0,
                        help="Overlap seconds (default: 2.0) -- for stride calculation in JSON")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument(
        "--scene-diversity",
        default=_DEFAULT_DIVERSITY,
        help=(
            "Ambition tier + optional sub-letter flavor. "
            "Tiers: 1=performance-live, 2=performance-dynamic (default), "
            "3=cinematic, 4=narrative, 5=stylized, 6=avant-garde. "
            "Sub-letters add mood bundles (e.g. 3a urban-night, 3b natural, "
            "3c interior, 3d perf+b-roll; 4a linear, 4b flashback; "
            "5a noir, 5b surreal, 5c retro)."
        ),
    )
    parser.add_argument(
        "--montage",
        action="store_true",
        help=(
            "Orthogonal flag: music-video-montage pacing. Cuts faster "
            "(~12s per entry vs ~20s) and adds emotional-arc language "
            "('building', 'release', 'stillness'). Works with any tier."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {args.audio_path}...", file=sys.stderr)

    results = analyze_file(
        args.audio_path,
        sr=args.sr,
        trim_offset=args.trim,
        vocal_track=args.vocal_track,
    )

    # Markdown report
    md_report = format_markdown_report(
        audio_path=args.audio_path,
        duration=results["duration"],
        bpm_result=results["bpm"],
        key_result=results["key"],
        sections=results["sections"],
        f0_result=results["f0"],
        trim_offset=args.trim,
        subject=args.subject or "",
        window_seconds=args.window,
        overlap_seconds=args.overlap,
        init_image_description=args.image_desc or "",
        diversity=args.scene_diversity,
        montage=args.montage,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(md_report)
        print(f"Markdown report written to {args.output}", file=sys.stderr)
    else:
        print(md_report)

    # JSON report
    if args.json:
        json_report = format_json_report(
            bpm_result=results["bpm"],
            key_result=results["key"],
            sections=results["sections"],
            f0_result=results["f0"],
            duration=results["duration"],
            trim_offset=args.trim,
            window_seconds=args.window,
            overlap_seconds=args.overlap,
            subject=args.subject or "",
            init_image_description=args.image_desc or "",
            diversity=args.scene_diversity,
            montage=args.montage,
        )
        if orjson:
            data = orjson.dumps(json_report, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY)
        else:
            import json
            data = json.dumps(json_report, indent=2).encode()

        with open(args.json, "wb") as f:
            f.write(data)
        print(f"JSON report written to {args.json}", file=sys.stderr)

    # PNG visualizations
    if args.png_dir:
        basename = Path(args.audio_path).stem
        saved = save_png_visualizations(
            results["audio"], results["sr"], args.png_dir, basename,
            precomputed_chroma=results.get("chroma"),
        )
        for p in saved:
            print(f"Saved: {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
