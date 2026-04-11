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
# Camera motions from CLAUDE.md prompt guide. Avoid dolly out (breaks limbs/faces)
# except for OUTRO where it's the expected visual pattern.
_SECTION_MODIFIERS = {
    "INTRO": {
        "framing": "In a wide establishing shot, static camera, locked off shot,",
        "lighting": "Soft lighting, gentle.",
        "energy_verb": "is beginning to perform softly",
        "audio_desc": "Quiet ambient tone, gentle room presence.",
    },
    "VERSE": {
        "framing": "In a medium shot,",
        "lighting": "Warm lighting, steady energy.",
        "energy_verb": "is performing",
        "audio_desc": "The voice fills the space. Soft ambient hum.",
    },
    "CHORUS": {
        "framing": "In a close-up,",
        "lighting": "Bright, dynamic lighting.",
        "energy_verb": "is performing with intensity",
        "audio_desc": "The voice is powerful and resonant.",
    },
    "BRIDGE": {
        "framing": "In a wide shot,",
        "lighting": "Moody, low contrast lighting.",
        "energy_verb": "is performing with quiet emotion",
        "audio_desc": "Subdued melody, reflective atmosphere.",
    },
    "OUTRO": {
        "framing": "In a wide shot, dolly out, camera pulling back,",
        "lighting": "Fading, gentle lighting.",
        "energy_verb": "is performing softly, trailing off",
        "audio_desc": "The sound fades quietly. Room tone settles.",
    },
    "BREAK": {
        "framing": "In a medium shot, static camera,",
        "lighting": "Dim lighting, still.",
        "energy_verb": "pauses, swaying gently",
        "audio_desc": "Brief instrumental moment, ambient quiet.",
    },
}

# Fallback for unknown section labels
_DEFAULT_MODIFIER = {
    "framing": "In a medium shot,",
    "lighting": "Natural lighting.",
    "energy_verb": "is performing",
    "audio_desc": "Music continues.",
}


def generate_schedule_suggestion(
    sections: list[dict],
    subject: str = "",
    trim_offset: float = 0.0,
) -> str:
    """Generate a TimestampPromptSchedule text block from sections.

    Without subject: produces placeholder prompts with section labels.
    With subject: produces full LTX 2.3 i2v prompts using the subject
    description wrapped with section-appropriate camera, lighting, and
    energy modifiers. Copy-pasteable into TimestampPromptSchedule.

    Args:
        sections: list of dicts with start, end, label, level keys.
        subject: scene description (e.g., "a woman singing in a workshop").
            If empty, falls back to placeholder output.
        trim_offset: seconds to subtract from timestamps.
    """
    if not subject:
        return _generate_placeholder_schedule(sections, trim_offset)

    return _generate_subject_schedule(sections, subject, trim_offset)


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
    sections: list[dict], subject: str, trim_offset: float
) -> str:
    """Full prompt schedule with subject wrapped in section modifiers."""
    def build(s):
        mods = _SECTION_MODIFIERS.get(s["label"], _DEFAULT_MODIFIER)
        return (
            f"Style: cinematic. {mods['framing']} {subject} {mods['energy_verb']}. "
            f"{mods['lighting']} {mods['audio_desc']}"
        )
    return _build_schedule(sections, trim_offset, build)


def _fmt_ts(seconds: float) -> str:
    """Format seconds as M:SS for schedule timestamps."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def format_json_report(
    bpm_result: dict,
    key_result: dict,
    sections: list[dict],
    f0_result: dict | None = None,
    duration: float = 0.0,
) -> dict:
    """Build structured JSON report for LLM consumption.

    Returns a dict (caller serializes with orjson or stdlib json).
    """
    report = {
        "duration": round(duration, 2),
        "bpm": bpm_result.get("bpm", 0.0),
        "beat_times": bpm_result.get("beat_times", []),
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

    # Prompt schedule suggestion
    lines.append("## Suggested TimestampPromptSchedule")
    lines.append("")
    lines.append("Copy and adapt:")
    lines.append("")
    lines.append("```")
    lines.append(generate_schedule_suggestion(sections, subject=subject, trim_offset=trim_offset))
    lines.append("```")
    lines.append("")

    # JSON block for LLM
    lines.append("## Structured Data (for LLM prompt)")
    lines.append("")
    lines.append("Paste this into your LLM prompt for schedule generation:")
    lines.append("")
    lines.append("```json")
    json_report = format_json_report(bpm_result, key_result, sections, f0_result, duration)
    if orjson:
        lines.append(orjson.dumps(json_report, option=orjson.OPT_INDENT_2).decode())
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
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
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
        )
        if orjson:
            data = orjson.dumps(json_report, option=orjson.OPT_INDENT_2)
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
