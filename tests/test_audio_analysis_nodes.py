"""Tests for runtime audio analysis nodes (torchaudio only).

Uses synthetic torch tensors to test AudioPitchDetect and _slice_audio_window.
No librosa dependency -- these tests verify the torchaudio-based runtime nodes.
"""

import torch
import pytest


def _make_audio(waveform: torch.Tensor, sample_rate: int = 44100) -> dict:
    """Create a ComfyUI AUDIO dict from a waveform tensor."""
    return {"waveform": waveform, "sample_rate": sample_rate}


def _sine_tensor(freq_hz: float, duration_s: float, sr: int = 44100) -> torch.Tensor:
    """Generate a mono sine wave as a torch tensor [1, samples]."""
    t = torch.linspace(0, duration_s, int(sr * duration_s), dtype=torch.float32)
    return (0.5 * torch.sin(2 * torch.pi * freq_hz * t)).unsqueeze(0)


def _silence_tensor(duration_s: float, sr: int = 44100) -> torch.Tensor:
    """Generate silence as a torch tensor [1, samples]."""
    return torch.zeros(1, int(sr * duration_s))


# --- _slice_audio_window ---


class TestSliceAudioWindow:
    def test_basic_slice(self):
        from nodes_analysis import _slice_audio_window

        audio = _make_audio(_sine_tensor(440.0, 10.0))
        waveform, sr = _slice_audio_window(audio, start_seconds=2.0, window_seconds=3.0)
        expected_samples = int(44100 * 3.0)
        assert waveform.shape[-1] == expected_samples
        assert sr == 44100

    def test_clamps_to_end(self):
        from nodes_analysis import _slice_audio_window

        audio = _make_audio(_sine_tensor(440.0, 5.0))
        # Request beyond audio end
        waveform, sr = _slice_audio_window(audio, start_seconds=4.0, window_seconds=3.0)
        # Should get remaining 1s, not crash
        assert waveform.shape[-1] > 0
        assert waveform.shape[-1] <= int(44100 * 1.5)  # some tolerance

    def test_minimum_samples(self):
        from nodes_analysis import _slice_audio_window

        audio = _make_audio(_sine_tensor(440.0, 0.1))  # very short
        waveform, sr = _slice_audio_window(audio, start_seconds=0.0, window_seconds=0.01)
        assert waveform.shape[-1] >= 1024, "Should return at least 1024 samples"

    def test_start_at_zero(self):
        from nodes_analysis import _slice_audio_window

        audio = _make_audio(_sine_tensor(440.0, 5.0))
        waveform, sr = _slice_audio_window(audio, start_seconds=0.0, window_seconds=2.0)
        expected = int(44100 * 2.0)
        assert waveform.shape[-1] == expected


# --- AudioPitchDetect ---


class TestAudioPitchDetect:
    def test_male_range_sine(self):
        """120 Hz sine should be classified as male vocal range."""
        from nodes_analysis import AudioPitchDetect

        audio = _make_audio(_sine_tensor(120.0, 3.0))
        result = AudioPitchDetect.execute(
            audio=audio, start_seconds=0.0, window_seconds=3.0,
            freq_low=85.0, freq_high=400.0,
        )
        median_f0, has_vocals, is_male, is_female, vocal_fraction = result
        assert has_vocals is True
        assert is_male is True
        assert is_female is False
        assert median_f0 > 0

    def test_female_range_sine(self):
        """220 Hz sine should be classified as female vocal range."""
        from nodes_analysis import AudioPitchDetect

        audio = _make_audio(_sine_tensor(220.0, 3.0))
        result = AudioPitchDetect.execute(
            audio=audio, start_seconds=0.0, window_seconds=3.0,
            freq_low=85.0, freq_high=400.0,
        )
        median_f0, has_vocals, is_male, is_female, vocal_fraction = result
        assert has_vocals is True
        assert is_female is True
        assert is_male is False

    def test_silence_no_vocals(self):
        """Silence should have no vocals detected."""
        from nodes_analysis import AudioPitchDetect

        audio = _make_audio(_silence_tensor(3.0))
        result = AudioPitchDetect.execute(
            audio=audio, start_seconds=0.0, window_seconds=3.0,
            freq_low=85.0, freq_high=400.0,
        )
        median_f0, has_vocals, is_male, is_female, vocal_fraction = result
        assert has_vocals is False
        assert median_f0 == 0.0
        assert vocal_fraction == 0.0

    def test_vocal_fraction_range(self):
        """vocal_fraction should be between 0 and 1."""
        from nodes_analysis import AudioPitchDetect

        audio = _make_audio(_sine_tensor(200.0, 3.0))
        result = AudioPitchDetect.execute(
            audio=audio, start_seconds=0.0, window_seconds=3.0,
            freq_low=85.0, freq_high=400.0,
        )
        _, _, _, _, vocal_fraction = result
        assert 0.0 <= vocal_fraction <= 1.0

    def test_windowed_slice(self):
        """Should analyze only the requested window, not full audio."""
        from nodes_analysis import AudioPitchDetect

        # 5s silence + 5s of 200Hz tone
        silence = _silence_tensor(5.0)
        tone = _sine_tensor(200.0, 5.0)
        waveform = torch.cat([silence, tone], dim=-1)
        audio = _make_audio(waveform)

        # Analyze first 5s (silence) -- should have no vocals
        result_silent = AudioPitchDetect.execute(
            audio=audio, start_seconds=0.0, window_seconds=4.0,
            freq_low=85.0, freq_high=400.0,
        )
        _, has_vocals_silent, _, _, _ = result_silent
        assert has_vocals_silent is False

        # Analyze last 5s (tone) -- should have vocals
        result_tone = AudioPitchDetect.execute(
            audio=audio, start_seconds=5.0, window_seconds=4.0,
            freq_low=85.0, freq_high=400.0,
        )
        _, has_vocals_tone, _, _, _ = result_tone
        assert has_vocals_tone is True
