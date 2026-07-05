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
    get_node_169_prompt,
    format_json_report,
    _subdivide_long_sections,
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
        """With subject, each line should contain 'Style: cinematic.' by default."""
        subject = "a person playing guitar on stage"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        lines = schedule_text.strip().splitlines()
        for line in lines:
            assert "Style: cinematic" in line, (
                f"Missing style prefix in: {line}"
            )

    def test_style_flag_applies_illustrated_prefix(self):
        """`style='illustrated'` should produce `Style: illustrated.` in every entry."""
        subject = "a warrior and her cat"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject, style="illustrated"
        )
        lines = schedule_text.strip().splitlines()
        for line in lines:
            assert "Style: illustrated" in line
            assert "Style: cinematic" not in line

    def test_style_none_omits_prefix_entirely(self):
        """`style='none'` should produce entries with no `Style:` prefix at all."""
        subject = "a warrior and her cat"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject, style="none"
        )
        lines = schedule_text.strip().splitlines()
        for line in lines:
            prompt_part = line.split(": ", 1)[1] if ": " in line else line
            assert not prompt_part.startswith("Style:"), (
                f"Expected no Style prefix, got: {prompt_part[:40]}"
            )

    def test_style_painterly_uses_full_phrase(self):
        """`style='painterly'` uses the multi-word `painterly illustration` phrase."""
        subject = "a warrior"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject, style="painterly"
        )
        assert "Style: painterly illustration" in schedule_text

    def test_style_unknown_falls_back_to_default(self):
        """An unknown style value should not crash; it falls back to default."""
        subject = "a warrior"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject, style="wat-is-this"
        )
        # Default is cinematic.
        assert "Style: cinematic" in schedule_text

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

    def test_outro_uses_held_close_up(self):
        """OUTRO should be a held close-up, not dolly-out.

        Dolly-out shrinks the face over an 18s sampler pass and loses
        lip-sync cross-attention signal. OUTRO framing is close-up +
        static. Fade is handled by the audio closing, not the camera.
        """
        subject = "a singer on stage"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        lines = schedule_text.strip().splitlines()
        last_line = lines[-1].lower()
        assert "close-up" in last_line and "static camera" in last_line, (
            f"OUTRO should be held close-up + static: {last_line}"
        )
        assert "dolly out" not in last_line and "pulling back" not in last_line, (
            f"OUTRO must not use dolly-out: {last_line}"
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

    def test_prompts_always_include_singing_verb_with_subject(self):
        """Every generated schedule line must contain the word 'singing'.

        LTX 2.3's audio-video joint attention relies on the singing verb to
        drive lip sync cross-attention. Generic 'is performing' loses the
        signal.
        """
        subject = "a woman in a workshop"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        for line in schedule_text.strip().splitlines():
            assert "singing" in line.lower(), (
                f"Line missing 'singing': {line!r}"
            )

    def test_single_character_subject_uses_is_singing(self):
        """Single-subject prompts use 'is singing' (singular)."""
        subject = "a woman in a workshop"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        for line in schedule_text.strip().splitlines():
            assert "is singing" in line.lower(), (
                f"Line missing 'is singing' (singular): {line!r}"
            )

    def test_multi_character_subject_triggers_singing_together(self):
        """Multi-subject prompts use 'are singing together' (group verb)."""
        subject = "two men in an alleyway"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        for line in schedule_text.strip().splitlines():
            assert "are singing together" in line.lower(), (
                f"Line missing 'are singing together': {line!r}"
            )

    def test_multi_character_and_conjunction(self):
        """Subject with 'and' connector is detected as multi-character."""
        subject = "a man and a woman on a rooftop"
        schedule_text = generate_schedule_suggestion(
            self._SECTIONS, subject=subject
        )
        for line in schedule_text.strip().splitlines():
            assert "are singing together" in line.lower(), (
                f"'and' conjunction not detected as multi-subject: {line!r}"
            )

    def test_three_minute_song_produces_many_prompts(self):
        """A typical 3-min song (180s) should produce ~8 schedule entries,
        not 4. Long sections get subdivided into ~25s chunks so each
        iteration-sized window gets its own prompt.
        """
        sections = [
            {"start": 0.0, "end": 20.0, "label": "INTRO", "level": "quiet"},
            {"start": 20.0, "end": 75.0, "label": "VERSE", "level": "medium"},
            {"start": 75.0, "end": 115.0, "label": "CHORUS", "level": "loud"},
            {"start": 115.0, "end": 150.0, "label": "VERSE", "level": "medium"},
            {"start": 150.0, "end": 180.0, "label": "OUTRO", "level": "quiet"},
        ]
        schedule = generate_schedule_suggestion(
            sections, subject="a singer in a studio"
        )
        lines = schedule.strip().splitlines()
        assert len(lines) >= 7, (
            f"3-min song should produce 7+ entries after subdivision, got "
            f"{len(lines)}: {lines}"
        )

    def test_long_section_subdivided(self):
        """A 60s VERSE gets split into multiple entries, each ~25s."""
        sections = [
            {"start": 0.0, "end": 60.0, "label": "VERSE", "level": "medium"},
        ]
        schedule = generate_schedule_suggestion(
            sections, subject="a singer in a studio"
        )
        lines = schedule.strip().splitlines()
        # 60s / ~25s = 2-3 subdivisions
        assert len(lines) >= 2, (
            f"60s VERSE should subdivide into 2+ entries, got {len(lines)}"
        )

    def test_short_section_not_subdivided(self):
        """Sections already shorter than ~30s stay as single entries."""
        sections = [
            {"start": 0.0, "end": 20.0, "label": "INTRO", "level": "quiet"},
            {"start": 20.0, "end": 45.0, "label": "VERSE", "level": "medium"},
        ]
        schedule = generate_schedule_suggestion(
            sections, subject="a singer in a studio"
        )
        lines = schedule.strip().splitlines()
        assert len(lines) == 2, (
            f"Short sections should not subdivide, got {len(lines)}: {lines}"
        )

    def test_diversity_tier_affects_beats(self):
        """A higher-ambition tier (4 narrative) produces longer prompts
        than tier 1 performance_live on the same section.
        """
        sections = [{"start": 0.0, "end": 20.0, "label": "VERSE", "level": "medium"}]
        subject = "a singer in a studio"
        tier_1 = generate_schedule_suggestion(sections, subject=subject, diversity="1a")
        tier_4 = generate_schedule_suggestion(sections, subject=subject, diversity="4a")
        # Tier 4 adds more beat pools (scene + narrative), so its prompt is
        # noticeably longer.
        assert len(tier_4) > len(tier_1), (
            f"Tier 4 should be richer than tier 1.\n  1a: {tier_1!r}\n  4a: {tier_4!r}"
        )
        # Tier 1 is still a valid singing prompt
        assert "is singing" in tier_1
        assert "is singing" in tier_4

    def test_sub_letter_adds_mood_bundle(self):
        """Sub-letter selects a mood bundle that appears in the prompt.
        3a (urban night) vs 3b (natural outdoor) differ in their mood phrase.
        """
        sections = [{"start": 0.0, "end": 20.0, "label": "VERSE", "level": "medium"}]
        subject = "a singer"
        schedule_3a = generate_schedule_suggestion(sections, subject=subject, diversity="3a")
        schedule_3b = generate_schedule_suggestion(sections, subject=subject, diversity="3b")
        # 3a and 3b both tier-3 so beat pools match; only mood bundle
        # differs. Therefore schedules must differ in the mood phrase.
        assert schedule_3a != schedule_3b
        # 3a mentions "neon" or "urban"; 3b mentions "outdoor" or "natural"
        assert any(t in schedule_3a.lower() for t in ("neon", "urban"))
        assert any(t in schedule_3b.lower() for t in ("outdoor", "natural"))

    def test_companion_animal_triggers_plural_verb(self):
        """`a woman with her cat` should emit `are singing together`, not
        `is singing`, so LTX's audio-video cross-attention animates the
        animal's mouth too. Extends the multi-subject heuristic beyond
        humans-only.
        """
        sections = [{"start": 0.0, "end": 20.0, "label": "VERSE", "level": "medium"}]
        subject = "a warrior woman with her orange tabby cat"
        schedule = generate_schedule_suggestion(sections, subject=subject)
        assert "are singing together" in schedule
        assert " is singing " not in schedule

    def test_single_human_stays_singular(self):
        """Baseline: subject without companion or plural markers stays singular."""
        sections = [{"start": 0.0, "end": 20.0, "label": "VERSE", "level": "medium"}]
        schedule = generate_schedule_suggestion(sections, subject="a woman in her 30s with dark hair")
        # "with dark hair" should NOT trigger multi-subject (hair isn't an animal)
        assert "is singing" in schedule
        assert "are singing together" not in schedule

    def test_json_report_omits_vocal_f0_classification(self):
        """As of 2026-04-20, `classification` (male/female) is no longer
        exported. Invites the LLM to second-guess the init image; median_hz
        and mean_hz are sufficient for any downstream pitch-aware logic.
        """
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[],
            f0_result={"median_f0": 180.0, "mean_f0": 200.0, "classification": "female"},
            duration=180.0,
        )
        assert "vocal_f0" in report
        # Exact-set assertion: future additions to vocal_f0 must be
        # intentional (update this test).
        assert set(report["vocal_f0"].keys()) == {"median_hz", "mean_hz"}

    def test_montage_flag_shortens_dwell(self):
        """Enabling montage on the same song produces more schedule entries
        (shorter dwell).
        """
        sections = [
            {"start": 0.0, "end": 30.0, "label": "INTRO", "level": "quiet"},
            {"start": 30.0, "end": 120.0, "label": "VERSE", "level": "medium"},
            {"start": 120.0, "end": 180.0, "label": "CHORUS", "level": "loud"},
            {"start": 180.0, "end": 210.0, "label": "OUTRO", "level": "quiet"},
        ]
        subject = "a singer"
        normal = generate_schedule_suggestion(sections, subject=subject)
        montage = generate_schedule_suggestion(sections, subject=subject, montage=True)
        n_normal = len(normal.strip().splitlines())
        n_montage = len(montage.strip().splitlines())
        assert n_montage > n_normal, (
            f"Montage should produce more entries than normal, "
            f"got normal={n_normal}, montage={n_montage}"
        )

    def test_montage_adds_emotional_arc_language(self):
        """Montage mode must inject emotional-arc beats per entry."""
        sections = [
            {"start": 0.0, "end": 60.0, "label": "VERSE", "level": "medium"},
        ]
        schedule = generate_schedule_suggestion(
            sections, subject="a singer", montage=True
        )
        lower = schedule.lower()
        assert any(
            t in lower
            for t in ("building", "collecting", "gathering", "tension", "release", "feeling")
        ), f"Montage missing emotional-arc language: {schedule!r}"

    def test_diversity_default_is_2a_when_none_passed(self):
        """Omitting diversity defaults to '2a' (tier 2, handheld energetic)."""
        sections = [{"start": 0.0, "end": 20.0, "label": "VERSE", "level": "medium"}]
        schedule = generate_schedule_suggestion(sections, subject="a singer")
        # 2a mood bundle mentions "handheld" or "rock-video"
        assert any(t in schedule.lower() for t in ("handheld", "rock-video"))


# --- Node 169 prompt ---


class TestGetNode169Prompt:
    _SECTIONS = [
        {"start": 0.0, "end": 30.0, "label": "INTRO", "level": "quiet"},
        {"start": 30.0, "end": 90.0, "label": "VERSE", "level": "medium"},
        {"start": 90.0, "end": 150.0, "label": "CHORUS", "level": "loud"},
    ]

    def test_with_subject_contains_subject(self):
        prompt = get_node_169_prompt(self._SECTIONS, subject="a singer on stage")
        assert "singer" in prompt.lower()

    def test_with_subject_has_style_prefix(self):
        prompt = get_node_169_prompt(self._SECTIONS, subject="a singer on stage")
        assert "Style: cinematic" in prompt

    def test_without_subject_is_placeholder(self):
        prompt = get_node_169_prompt(self._SECTIONS)
        assert "INTRO" in prompt or "describe" in prompt.lower()

    def test_matches_first_schedule_entry(self):
        """Node 169 prompt should match the first schedule entry's prompt text."""
        subject = "a singer on stage"
        prompt_169 = get_node_169_prompt(self._SECTIONS, subject=subject)
        schedule = generate_schedule_suggestion(self._SECTIONS, subject=subject)
        first_line = schedule.strip().splitlines()[0]
        # Extract prompt text after the timestamp
        first_prompt = first_line.split(": ", 1)[1] if ": " in first_line else first_line
        assert prompt_169 == first_prompt

    def test_node_169_bit_exact_match_across_subjects(self):
        """Bit-exact equality required across varied subjects (single +
        multi-character) so the ~20s seam hand-off is seamless.
        """
        for subject in [
            "a woman in a workshop",
            "a man playing guitar on stage",
            "two men in an alleyway",
            "a man and a woman on a rooftop",
        ]:
            prompt_169 = get_node_169_prompt(self._SECTIONS, subject=subject)
            schedule = generate_schedule_suggestion(
                self._SECTIONS, subject=subject
            )
            first_line = schedule.strip().splitlines()[0]
            first_prompt = first_line.split(": ", 1)[1]
            assert prompt_169 == first_prompt, (
                f"Mismatch for subject {subject!r}:\n"
                f"  node_169: {prompt_169!r}\n"
                f"  first_schedule: {first_prompt!r}"
            )

    def test_empty_sections(self):
        assert get_node_169_prompt([]) == ""


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
        # Effective stride after integer-latent quantization: window=497 pixels,
        # overlap=50 pixels → 7 latents trimmed from 63 → 56 latents remain →
        # 56 * 8 / 25 = 17.92s (not 17.88 naive window - overlap). Matches
        # what AudioLoopController.execute actually uses.
        assert ctx["stride_seconds"] == pytest.approx(17.92)
        assert ctx["overlap_seconds_target"] == 2.0
        assert ctx["overlap_seconds_effective"] == pytest.approx(1.96)
        assert ctx["subject"] == "a man playing guitar"
        assert ctx["init_image_description"] == "Man with acoustic guitar, dim room"
        assert ctx["style"] == "cinematic"  # default

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
        assert "present progressive" in prompt.lower()
        assert "frozen" in prompt.lower()

    def test_llm_system_prompt_enforces_singing_verb(self):
        """System prompt must explicitly tell the LLM to use 'is singing' or
        'are singing together' so outputs don't drift to generic 'performs'.
        """
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[],
            duration=180.0,
        )
        prompt = report["llm_system_prompt"]
        assert "is singing" in prompt.lower()
        assert "are singing together" in prompt.lower()

    def test_llm_system_prompt_enforces_node_169_identity(self):
        """System prompt must state node_169_prompt = first schedule entry
        verbatim, not just 'should match'.
        """
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[],
            duration=180.0,
        )
        prompt = report["llm_system_prompt"]
        lower = prompt.lower()
        assert "identical" in lower or "exactly" in lower or "verbatim" in lower

    def test_llm_system_prompt_contains_few_shot_examples(self):
        """System prompt must include at least one worked example (single +
        multi-character) so LLMs have a concrete target format.
        """
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[],
            duration=180.0,
        )
        prompt = report["llm_system_prompt"]
        lower = prompt.lower()
        assert "example" in lower
        # At least one example should show the "are singing together" pattern
        assert "are singing together" in lower

    def test_llm_system_prompt_contains_inference_block(self):
        """System prompt must tell the LLM what is inferable from the init
        image (style, palette, subject appearance) vs what the schedule
        drives (camera, body, lighting shifts, cuts, arc). This keeps the
        schedule as a temporal delta layer instead of fighting the image.
        """
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[],
            duration=180.0,
        )
        prompt = report["llm_system_prompt"]
        lower = prompt.lower()
        # The inference block must say what the image already commits
        assert "init image" in lower
        assert "do not re-describe" in lower or "do not re describe" in lower
        # And what the schedule drives instead
        assert "camera" in lower and "lighting" in lower
        # Style-family examples (at least one)
        assert any(
            t in lower
            for t in ("comic", "graphic-novel", "graphic novel", "animated", "live-action")
        )

    def test_llm_system_prompt_contains_ambition_tiers(self):
        """System prompt must document the 6-tier scene_diversity taxonomy
        so the LLM can match output ambition to the user's choice.
        """
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[],
            duration=180.0,
        )
        prompt = report["llm_system_prompt"]
        lower = prompt.lower()
        for term in (
            "performance_live",
            "performance_dynamic",
            "cinematic",
            "narrative",
            "stylized",
            "avant",
        ):
            assert term in lower, f"Missing tier {term!r} in system prompt"

    def test_llm_system_prompt_contains_montage_semantics(self):
        """System prompt must describe montage as an orthogonal pacing
        flag — emotional arc, shorter dwell, music-drives-narrative feel.
        """
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[],
            duration=180.0,
        )
        prompt = report["llm_system_prompt"]
        lower = prompt.lower()
        assert "montage" in lower
        # Emotional-arc language markers
        assert "emotional" in lower
        # Should call out shorter dwell / faster cuts
        assert "12s" in lower or "12 s" in lower or "shorter" in lower or "faster" in lower

    def test_workflow_context_surfaces_diversity_and_montage(self):
        """workflow_context carries scene_diversity + montage so the LLM
        knows which tier/sub-letter to target.
        """
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[],
            duration=180.0,
            diversity="3b",
            montage=True,
        )
        ctx = report["workflow_context"]
        assert ctx["scene_diversity"] == "3b"
        assert ctx["scene_diversity_tier_name"] == "cinematic"
        assert ctx["montage"] is True
        # Mood bundle label for 3b should surface something outdoor-ish
        assert ctx["scene_diversity_mood_bundle"] is not None
        assert "outdoor" in ctx["scene_diversity_mood_bundle"].lower()

    def test_workflow_context_defaults_to_2a_without_diversity(self):
        """If diversity is unset, default '2a' (performance_dynamic) shows."""
        report = format_json_report(
            bpm_result={"bpm": 120.0, "beat_times": []},
            key_result={"key": "G Major", "confidence": 0.85},
            sections=[],
            duration=180.0,
        )
        ctx = report["workflow_context"]
        assert ctx["scene_diversity"] == "2a"
        assert ctx["scene_diversity_tier_name"] == "performance_dynamic"
        assert ctx["montage"] is False


class TestBeatSnapping:
    """Snapping subdivision boundaries to phrase-aligned beats.

    `_subdivide_long_sections` splits long sections into ~target chunks.
    When `beat_times` is passed, internal chunk boundaries snap to the
    nearest phrase boundary (every `phrase_beats` beats) so prompt changes
    land on musical phrase edges, not mid-phrase. Default (no beats) keeps
    the exact uniform behavior.
    """

    _LONG = [{"start": 0.0, "end": 60.0, "label": "VERSE", "level": "medium"}]

    # 120 BPM = 0.5s/beat, offset 0.3s so phrase boundaries (every 8 beats =
    # 4.0s) land at 0.3, 4.3, 8.3, ... 20.3, ... 40.3 — deliberately off the
    # uniform 20.0/40.0 split so a snap is observable.
    _BEATS = [round(0.3 + 0.5 * i, 3) for i in range(130)]  # 0.3 .. 64.8s

    def test_default_none_is_exact_uniform(self):
        """No beats -> boundaries stay exactly uniform (byte-identical)."""
        out = _subdivide_long_sections(self._LONG, target=20.0, split_above=30.0)
        # 60 / 20 = 3 chunks: [0,20],[20,40],[40,60]
        starts = [round(c["start"], 3) for c in out]
        ends = [round(c["end"], 3) for c in out]
        assert starts == [0.0, 20.0, 40.0]
        assert ends == [20.0, 40.0, 60.0]

    def test_internal_boundaries_snap_to_phrase(self):
        """Internal boundaries snap to the nearest 8-beat phrase edge."""
        out = _subdivide_long_sections(
            self._LONG, target=20.0, split_above=30.0,
            beat_times=self._BEATS, phrase_beats=8,
        )
        starts = [round(c["start"], 3) for c in out]
        ends = [round(c["end"], 3) for c in out]
        # 20.0 -> 20.3, 40.0 -> 40.3 (nearest phrase edges); section edges fixed.
        assert starts == [0.0, 20.3, 40.3]
        assert ends == [20.3, 40.3, 60.0]

    def test_section_edges_never_move(self):
        """The section's own start/end are fixed; only internal cuts snap."""
        out = _subdivide_long_sections(
            self._LONG, target=20.0, split_above=30.0,
            beat_times=self._BEATS, phrase_beats=8,
        )
        assert out[0]["start"] == 0.0
        assert out[-1]["end"] == 60.0

    def test_chunks_stay_monotonic_and_nonempty(self):
        """Every chunk keeps start < end after snapping."""
        out = _subdivide_long_sections(
            self._LONG, target=20.0, split_above=30.0,
            beat_times=self._BEATS, phrase_beats=8,
        )
        for c in out:
            assert c["end"] > c["start"]

    def test_sparse_beats_fall_back_to_uniform(self):
        """Too few beats to form a phrase edge near a boundary -> uniform."""
        out = _subdivide_long_sections(
            self._LONG, target=20.0, split_above=30.0,
            beat_times=[0.3, 59.5], phrase_beats=8,
        )
        starts = [round(c["start"], 3) for c in out]
        # No phrase edge near 20 or 40 -> boundaries unchanged.
        assert starts == [0.0, 20.0, 40.0]

    def test_short_section_unaffected_by_beats(self):
        """A section below split_above passes through regardless of beats."""
        short = [{"start": 0.0, "end": 20.0, "label": "INTRO", "level": "quiet"}]
        out = _subdivide_long_sections(
            short, target=20.0, split_above=30.0, beat_times=self._BEATS,
        )
        assert len(out) == 1
        assert out[0]["start"] == 0.0 and out[0]["end"] == 20.0

    def test_node_169_matches_schedule_first_entry_with_beats(self):
        """The Node-169 == schedule[0] byte-exact invariant survives snapping."""
        sections = [
            {"start": 0.0, "end": 60.0, "label": "VERSE", "level": "medium"},
        ]
        subject = "a singer in a workshop"
        schedule = generate_schedule_suggestion(
            sections, subject=subject, beat_times=self._BEATS,
        )
        first_entry = schedule.strip().splitlines()[0]
        node_169 = get_node_169_prompt(
            sections, subject=subject, beat_times=self._BEATS,
        )
        # The schedule line is "MM:SS - <prompt>"; Node 169 is the bare prompt.
        assert node_169 in first_entry, (
            f"Node 169 must be the first schedule entry's prompt.\n"
            f"  node_169: {node_169!r}\n  first:    {first_entry!r}"
        )
