"""Runtime audio analysis nodes using torchaudio only.

These nodes run per-iteration inside the loop and output FLOAT/INT/BOOLEAN
scalars for modulating text conditioning and sampling parameters.
No IMAGE outputs. No librosa dependency.

Design principle: LTX-2.3 audio path is sacred. Audio enters the model
via LTXVAudioVAEEncode -> LTXVConcatAVLatent where cross-attention
translates mel features into visual motion. These nodes extract scalar
features FROM audio to modulate text conditioning parameters. They never
touch the audio or video latent streams directly.
"""

import torch
import torchaudio

try:
    from comfy_api.latest import io
except ImportError:
    io = None  # Outside ComfyUI runtime (e.g., pytest)

MIN_SAMPLES = 1024


def _slice_audio_window(
    audio: dict, start_seconds: float, window_seconds: float
) -> tuple[torch.Tensor, int]:
    """Extract a waveform slice for the current iteration window.

    Args:
        audio: ComfyUI AUDIO dict with "waveform" and "sample_rate".
        start_seconds: Start time in seconds.
        window_seconds: Duration of the window in seconds.

    Returns:
        (waveform_slice, sample_rate). Waveform is at least MIN_SAMPLES long.
        Clamps to audio boundaries.
    """
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    total_samples = waveform.shape[-1]

    start_sample = int(start_seconds * sr)
    end_sample = int((start_seconds + window_seconds) * sr)

    # Clamp to audio boundaries
    start_sample = max(0, min(start_sample, total_samples - MIN_SAMPLES))
    end_sample = min(end_sample, total_samples)

    # Ensure minimum length
    if end_sample - start_sample < MIN_SAMPLES:
        end_sample = min(start_sample + MIN_SAMPLES, total_samples)
        if end_sample - start_sample < MIN_SAMPLES:
            start_sample = max(0, end_sample - MIN_SAMPLES)

    return waveform[..., start_sample:end_sample], sr


class AudioPitchDetect:
    """Per-iteration vocal pitch detection on audio windows.

    Best results when wired to MelBandRoFormer's separated vocals output.
    Uses torchaudio.functional.detect_pitch_frequency (autocorrelation, GPU).

    Outputs FLOAT/BOOLEAN scalars only -- does not touch audio or video latents.
    """

    MALE_CEILING = 160.0  # Hz -- above this is female range

    @classmethod
    def define_schema(cls):
        if io is None:
            return None  # Outside ComfyUI
        return io.Schema(
            node_id="AudioPitchDetect",
            display_name="Audio Pitch Detect",
            category="audio/analysis",
            description=(
                "Per-iteration vocal pitch detection. Outputs median F0, "
                "vocal presence, and male/female classification. "
                "Wire to MelBandRoFormer vocals output for clean signal."
            ),
            inputs=[
                io.Audio.Input("audio", tooltip="Audio track (best with separated vocals)."),
                io.Float.Input(
                    "start_seconds",
                    default=0.0,
                    min=0.0,
                    step=0.01,
                    tooltip="Start of analysis window in seconds. Wire from AudioLoopController.start_index.",
                ),
                io.Float.Input(
                    "window_seconds",
                    default=19.88,
                    min=0.1,
                    step=0.01,
                    tooltip="Duration of analysis window. Match to AudioLoopController.window_seconds.",
                ),
                io.Float.Input(
                    "freq_low",
                    default=85.0,
                    min=20.0,
                    max=500.0,
                    step=1.0,
                    tooltip="Minimum frequency for pitch detection (Hz). 85 Hz covers low male vocals.",
                ),
                io.Float.Input(
                    "freq_high",
                    default=400.0,
                    min=100.0,
                    max=2000.0,
                    step=1.0,
                    tooltip="Maximum frequency for pitch detection (Hz). 400 Hz covers high female vocals.",
                ),
            ],
            outputs=[
                io.Float.Output("median_f0", tooltip="Median fundamental frequency in Hz (0.0 if unvoiced)."),
                io.Boolean.Output("has_vocals", tooltip="True if pitched content detected in window."),
                io.Boolean.Output("is_male_range", tooltip="True if median F0 < 160 Hz."),
                io.Boolean.Output("is_female_range", tooltip="True if median F0 > 160 Hz."),
                io.Float.Output("vocal_fraction", tooltip="Ratio of voiced frames (0.0-1.0)."),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio: dict,
        start_seconds: float = 0.0,
        window_seconds: float = 19.88,
        freq_low: float = 85.0,
        freq_high: float = 400.0,
    ):
        waveform, sr = _slice_audio_window(audio, start_seconds, window_seconds)

        # Convert to mono if stereo
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Energy gate: skip pitch detection on near-silence
        rms = torch.sqrt(torch.mean(waveform ** 2)).item()
        if rms < 0.005:
            if io is not None:
                return io.NodeOutput(0.0, False, False, False, 0.0)
            return (0.0, False, False, False, 0.0)

        # Detect pitch using autocorrelation
        pitch = torchaudio.functional.detect_pitch_frequency(
            waveform, sample_rate=sr, freq_low=int(freq_low), freq_high=int(freq_high)
        )

        # Filter out unvoiced frames (pitch == 0)
        voiced_mask = pitch > 0
        voiced_count = voiced_mask.sum().item()
        total_frames = pitch.numel()

        if voiced_count == 0:
            if io is not None:
                return io.NodeOutput(0.0, False, False, False, 0.0)
            return (0.0, False, False, False, 0.0)

        voiced_pitches = pitch[voiced_mask]
        median_f0 = float(voiced_pitches.median().item())
        vocal_fraction = voiced_count / max(total_frames, 1)

        has_vocals = True
        is_male = median_f0 < cls.MALE_CEILING
        is_female = median_f0 >= cls.MALE_CEILING

        if io is not None:
            return io.NodeOutput(median_f0, has_vocals, is_male, is_female, vocal_fraction)
        return (median_f0, has_vocals, is_male, is_female, vocal_fraction)
