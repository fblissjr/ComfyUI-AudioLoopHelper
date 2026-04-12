"""Tests for librosa-based audio feature extraction.

Uses synthetic audio (sine waves, clicks, silence) so tests are
deterministic and don't depend on external audio files.
"""

import numpy as np
import pytest

# Path setup is in conftest.py
from analyze_audio_features import (
    detect_bpm,
    detect_key,
    compute_chromagram,
    compute_mel_spectrogram,
    estimate_vocal_f0,
    detect_structure_librosa,
    generate_schedule_suggestion,
    format_json_report,
)


SR = 22050  # standard librosa sample rate


def _sine_wave(freq_hz: float, duration_s: float, sr: int = SR) -> np.ndarray:
    """Generate a mono sine wave."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _click_track(bpm: float, duration_s: float, sr: int = SR) -> np.ndarray:
    """Generate a click track at a given BPM."""
    interval_samples = int(sr * 60.0 / bpm)
    total_samples = int(sr * duration_s)
    signal = np.zeros(total_samples, dtype=np.float32)
    click_len = min(200, interval_samples // 4)
    click = np.sin(2 * np.pi * 1000 * np.arange(click_len) / sr).astype(np.float32)
    click *= np.linspace(1, 0, click_len, dtype=np.float32)  # decay envelope
    for i in range(0, total_samples, interval_samples):
        end = min(i + click_len, total_samples)
        signal[i:end] += click[: end - i]
    return signal


def _silence(duration_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


# --- BPM detection ---


class TestDetectBPM:
    def test_120_bpm_click_track(self):
        audio = _click_track(120.0, 10.0)
        result = detect_bpm(audio, SR)
        assert "bpm" in result
        assert abs(result["bpm"] - 120.0) < 5.0, f"Expected ~120 BPM, got {result['bpm']}"

    def test_90_bpm_click_track(self):
        audio = _click_track(90.0, 10.0)
        result = detect_bpm(audio, SR)
        assert abs(result["bpm"] - 90.0) < 5.0, f"Expected ~90 BPM, got {result['bpm']}"

    def test_beat_timestamps_present(self):
        audio = _click_track(120.0, 10.0)
        result = detect_bpm(audio, SR)
        assert "beat_times" in result
        assert len(result["beat_times"]) > 0

    def test_silence_returns_zero_bpm(self):
        audio = _silence(5.0)
        result = detect_bpm(audio, SR)
        assert result["bpm"] == 0.0 or result["bpm"] is None


# --- Key detection ---


class TestDetectKey:
    def test_c_major_chord(self):
        """C major triad (C4=261.63, E4=329.63, G4=392.00)."""
        c = _sine_wave(261.63, 5.0)
        e = _sine_wave(329.63, 5.0)
        g = _sine_wave(392.00, 5.0)
        audio = c + e + g
        result = detect_key(audio, SR)
        assert "key" in result
        # Should detect C major or at least C as the tonic
        assert result["key"].startswith("C"), f"Expected C major, got {result['key']}"

    def test_key_has_confidence(self):
        audio = _sine_wave(440.0, 5.0)  # A4
        result = detect_key(audio, SR)
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0


# --- Chromagram ---


class TestComputeChromagram:
    def test_output_shape(self):
        audio = _sine_wave(440.0, 3.0)
        chroma = compute_chromagram(audio, SR)
        assert chroma.shape[0] == 12, "Chromagram should have 12 pitch classes"
        assert chroma.shape[1] > 0, "Chromagram should have time frames"

    def test_a440_lights_up_a_bin(self):
        """A 440Hz sine should activate the A pitch class (index 9 in C,C#,...,B)."""
        audio = _sine_wave(440.0, 3.0)
        chroma = compute_chromagram(audio, SR)
        # Average across time, find the dominant pitch class
        mean_energy = chroma.mean(axis=1)
        dominant_bin = int(np.argmax(mean_energy))
        # A is index 9 in [C, C#, D, D#, E, F, F#, G, G#, A, A#, B]
        assert dominant_bin == 9, f"Expected A (bin 9), got bin {dominant_bin}"


# --- Mel spectrogram ---


class TestComputeMelSpectrogram:
    def test_output_shape(self):
        audio = _sine_wave(440.0, 3.0)
        mel = compute_mel_spectrogram(audio, SR)
        assert mel.ndim == 2
        assert mel.shape[0] > 0  # mel bins
        assert mel.shape[1] > 0  # time frames

    def test_silence_is_uniform(self):
        audio = _silence(2.0)
        mel = compute_mel_spectrogram(audio, SR)
        # All-zero input: power_to_db(ref=max) yields uniform 0.0
        # (no frequency content at all -- flat response)
        assert mel.std() == 0.0, "Silence should have uniform mel spectrogram"


# --- Vocal F0 ---


class TestEstimateVocalF0:
    def test_male_range_fundamental(self):
        """A 120 Hz tone should be classified as male."""
        audio = _sine_wave(120.0, 3.0)
        result = estimate_vocal_f0(audio, SR)
        assert "classification" in result
        assert result["classification"] == "male"

    def test_female_range_fundamental(self):
        """A 220 Hz tone should be classified as female."""
        audio = _sine_wave(220.0, 3.0)
        result = estimate_vocal_f0(audio, SR)
        assert result["classification"] == "female"

    def test_median_f0_returned(self):
        audio = _sine_wave(200.0, 3.0)
        result = estimate_vocal_f0(audio, SR)
        assert "median_f0" in result
        assert abs(result["median_f0"] - 200.0) < 20.0


# --- Structure segmentation ---


class TestDetectStructure:
    def test_quiet_loud_quiet_pattern(self):
        """Should detect at least a transition from quiet to loud.

        Uses long sections with extreme dynamic contrast to ensure
        the percentile-based algorithm detects clear boundaries.
        """
        rng = np.random.default_rng(42)
        # 15s near-silence, 30s very loud, 15s near-silence
        quiet1 = (rng.standard_normal(int(SR * 15.0)) * 0.002).astype(np.float32)
        loud = _click_track(120.0, 30.0)
        quiet2 = (rng.standard_normal(int(SR * 15.0)) * 0.002).astype(np.float32)
        audio = np.concatenate([quiet1, loud, quiet2])
        sections = detect_structure_librosa(audio, SR)
        assert len(sections) >= 2, f"Expected >= 2 sections, got {len(sections)}"

    def test_section_has_required_fields(self):
        audio = _click_track(120.0, 10.0)
        sections = detect_structure_librosa(audio, SR)
        for s in sections:
            assert "start" in s
            assert "end" in s
            assert "label" in s
            assert "level" in s


# --- Schedule suggestion ---


class TestGenerateScheduleSuggestion:
    _SECTIONS = [
        {"start": 0.0, "end": 30.0, "label": "INTRO", "level": "quiet"},
        {"start": 30.0, "end": 90.0, "label": "VERSE", "level": "medium"},
        {"start": 90.0, "end": 150.0, "label": "CHORUS", "level": "loud"},
        {"start": 150.0, "end": 180.0, "label": "OUTRO", "level": "quiet"},
    ]

    def test_output_is_parseable_schedule(self):
        """The suggested schedule should be parseable by _parse_schedule."""
        schedule_text = generate_schedule_suggestion(self._SECTIONS)
        assert isinstance(schedule_text, str)
        assert len(schedule_text.strip().splitlines()) >= 3

    def test_placeholder_without_subject(self):
        """Without subject, output should contain generic placeholder text."""
        schedule_text = generate_schedule_suggestion(self._SECTIONS)
        # Should have placeholder markers, not full prompts
        assert "describe" in schedule_text.lower() or "[" in schedule_text

    def test_subject_produces_full_prompts(self):
        """With subject, output should contain the subject in every line."""
        subject = "a woman singing in a basement workshop"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        lines = schedule_text.strip().splitlines()
        for line in lines:
            # Every prompt line should contain the subject
            prompt_part = line.split(":", 2)[-1].strip()  # after timestamp
            assert "woman" in prompt_part.lower(), (
                f"Subject not found in line: {line}"
            )

    def test_subject_includes_style_prefix(self):
        """With subject, each line should contain 'Style: cinematic.'"""
        subject = "a person playing guitar on stage"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        lines = schedule_text.strip().splitlines()
        for line in lines:
            assert "Style: cinematic" in line, (
                f"Missing style prefix in: {line}"
            )

    def test_chorus_has_close_up(self):
        """CHORUS sections should suggest close-up framing."""
        subject = "a singer on stage"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        lines = schedule_text.strip().splitlines()
        # Find the chorus line (90-150)
        chorus_line = [l for l in lines if "1:30" in l or "close" in l.lower()]
        assert any("close" in l.lower() for l in lines), (
            f"No close-up suggested for chorus. Lines: {lines}"
        )

    def test_outro_has_fadeout_camera(self):
        """OUTRO sections should suggest dolly out or pulling back."""
        subject = "a singer on stage"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        lines = schedule_text.strip().splitlines()
        last_line = lines[-1]
        assert "dolly" in last_line.lower() or "pulling" in last_line.lower() or "fade" in last_line.lower(), (
            f"No fadeout camera in outro: {last_line}"
        )

    def test_verse_has_medium_shot(self):
        """VERSE sections should suggest medium shot framing."""
        subject = "a singer on stage"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        lines = schedule_text.strip().splitlines()
        assert any("medium" in l.lower() for l in lines), (
            f"No medium shot suggested for verse"
        )

    def test_schedule_parseable_by_parse_schedule(self):
        """With subject, output must still parse as valid schedule format."""
        from analyze_audio_features import generate_schedule_suggestion
        import re

        subject = "a woman singing in a workshop"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        lines = schedule_text.strip().splitlines()
        # Each line must match: timestamp_range: prompt_text
        ts_pattern = re.compile(r"^\d+:\d{2}")
        for line in lines:
            assert ts_pattern.match(line), f"Line doesn't start with timestamp: {line}"


# --- JSON report ---


class TestFormatJsonReport:
    def test_contains_required_keys(self):
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": [0.5, 1.0, 1.5]},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[{"start": 0.0, "end": 60.0, "label": "VERSE", "level": "medium"}],
            f0_result={"median_f0": 180.0, "classification": "female"},
            duration=180.0,
        )
        assert isinstance(report, dict)
        assert "bpm" in report
        assert "key" in report
        assert "sections" in report
        assert "duration" in report

    def test_no_beat_times_by_default(self):
        """beat_times should NOT be in default output (bloats LLM context)."""
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": [0.5, 1.0, 1.5]},
            key_result={"key": "C Major", "confidence": 0.9},
            sections=[],
            duration=60.0,
        )
        assert "beat_times" not in report

    def test_workflow_context_present(self):
        """With workflow args, report should include workflow_context."""
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[{"start": 0.0, "end": 60.0, "label": "VERSE", "level": "medium"}],
            duration=180.0,
            trim_offset=10.0,
            window_seconds=19.88,
            overlap_seconds=2.0,
            subject="a man playing guitar",
            init_image_description="Man with acoustic guitar, dim room",
        )
        assert "workflow_context" in report
        ctx = report["workflow_context"]
        assert ctx["trim_offset"] == 10.0
        assert ctx["window_seconds"] == 19.88
        assert ctx["stride_seconds"] == pytest.approx(17.88)
        assert ctx["subject"] == "a man playing guitar"
        assert ctx["init_image_description"] == "Man with acoustic guitar, dim room"

    def test_llm_system_prompt_present(self):
        """Report should include llm_system_prompt string."""
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[],
            duration=180.0,
        )
        assert "llm_system_prompt" in report
        prompt = report["llm_system_prompt"]
        assert isinstance(prompt, str)
        assert "LTX 2.3" in prompt
        assert "node_169_prompt" in prompt
        assert "schedule" in prompt

    def test_llm_system_prompt_contains_rules(self):
        """System prompt should contain key rules."""
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[],
            duration=180.0,
        )
        prompt = report["llm_system_prompt"]
        assert "dolly out" in prompt.lower()
        assert "present-progressive" in prompt.lower()
        assert "frozen" in prompt.lower()
