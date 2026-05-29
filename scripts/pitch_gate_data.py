"""Pitch-gate data generator (audio-loop side) — DSP core + manifest.

Design: internal/audio_iclora_training/audio_only_iclora_pitch_gate.md (private clone only)
Goal: build (voiced-tone reference @ F0=P, speech-target pitch-shifted to P) pairs
where the reference's F0 is the ONLY thing co-varying with the target pitch.

This module is the CPU/DSP + manifest core (no GPU, no VAE). The VAE-encode +
256 re-render step is a separate, GPU-gated stage that consumes the manifest.

Contract locked by test_pitch_gate_data.py.
"""

from __future__ import annotations

import numpy as np

# Pitch-free constant caption (the seesaw: pitch must come from the reference, not text).
# Gender-neutral ("a person" — gender↔pitch is the strongest natural leak) + no pitch words;
# "speaking" nudges audible speech (need voiced audio to carry F0) without claiming framing
# ("to camera" was false for the corpus's profile/off-camera clips). Reviewed by the
# captioning specialist: leak-safe, KEEP. [audio_only_iclora_pitch_gate.md §5.2]
NEUTRAL_CAPTION = "a person speaking"
PITCH_BAN = ("pitch", "high", "low", "helium", "deep", "squeak", "tone", "hz", "semitone")

# Output location (data/audio_iclora/pitch_ref_gate_v1/, flat clips/+references/+manifest.jsonl
# like the synth_e1_* sets; split is a per-row manifest field, not a directory) is owned by the
# GPU encode stage that does the I/O — not this pure DSP+manifest core.


# ---------------------------------------------------------------- DSP: voiced tone

def synth_voiced_tone(
    f0: float,
    duration: float = 1.0,
    sr: int = 16_000,
    n_harmonics: int = 24,
    timbre: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Glottal-pulse-like voiced tone at F0: a harmonic stack (NOT a pure sine,
    which is OOD for a speech-trained audio VAE). `timbre` tilts the harmonic
    rolloff (changes spectrum/centroid) WITHOUT moving F0 — so timbre can be a
    decorrelated nuisance axis. Content-free, formant-free -> identity-free."""
    n = int(round(duration * sr))
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    rolloff = 1.0 + 1.5 * float(timbre)  # steeper rolloff -> darker timbre
    kmax = min(n_harmonics, max(1, int((sr / 2 - 1) / f0)))  # harmonics strictly below Nyquist
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, kmax + 1):
        amp = 1.0 / (k ** rolloff)
        phase = rng.uniform(0, 2 * np.pi)
        sig += amp * np.sin(2 * np.pi * k * f0 * t + phase)
    peak = np.max(np.abs(sig)) + 1e-9
    return (sig / peak).astype(np.float32)


def dominant_f0(audio: np.ndarray, sr: int = 16_000, fmin: float = 50.0) -> float:
    """FFT-peak fundamental (robust for synthetic harmonic tones — the 1/k^rolloff
    stack puts the strongest peak at the fundamental). Same FFT-peak pattern as the
    LTX-2 trainer's synthetic_av.measure_pulse_rate (sibling gate; not importable here)."""
    a = np.asarray(audio, dtype=np.float64)
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    freqs = np.fft.rfftfreq(len(a), 1 / sr)
    spec[freqs < fmin] = 0.0
    return float(freqs[int(np.argmax(spec))])


def spectral_centroid(audio: np.ndarray, sr: int = 16_000) -> float:
    """Mean spectral centroid (Hz) via librosa (already a dependency)."""
    import librosa

    a = np.asarray(audio, dtype=np.float32)
    return float(librosa.feature.spectral_centroid(y=a, sr=sr).mean())


# ---------------------------------------------------------------- DSP: pitch shift

def pitch_shift_semitones(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Pitch-shift preserving duration (phase vocoder) — timing must stay so the
    target video's lip-sync remains valid."""
    import librosa

    y = np.asarray(audio, dtype=np.float32)
    out = np.asarray(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=float(n_steps)), dtype=np.float32)
    # librosa preserves length; clamp to exact (guards ±sample phase-vocoder rounding)
    if len(out) > len(y):
        return out[: len(y)]
    if len(out) < len(y):
        return np.pad(out, (0, len(y) - len(out)))
    return out


def semitones_to_target(natural_f0: float, target_f0: float) -> float:
    """Interval (semitones) to move natural_f0 -> target_f0."""
    return 12.0 * np.log2(float(target_f0) / float(natural_f0))


# ---------------------------------------------------------------- manifest

def build_manifest(
    natural_f0: dict[str, float],
    seed: int = 0,
    max_semitones: float = 7.0,
    heldout_frac: float = 0.2,
    n_timbres: int = 4,
) -> list[dict]:
    """Option B (bounded shift). Assign each clip a BOUNDED pitch shift (±max_semitones)
    from its own natural F0 -> target_f0 = natural * 2^(shift/12). Bounded shifts stay in
    the artifact-free range (the smoke confirmed >±8 st degrades). Leak-free because there
    is NO init frame and a constant caption, so the source clip's natural pitch is not an
    input at all -> the reference tone is the sole F0-bearing channel; target tracking
    natural cannot leak. Timbre is a decorrelated nuisance axis; held-out is random clips
    (continuous F0 -> generalization is tested across the range, not discrete levels)."""
    rng = np.random.default_rng(seed)
    clips = sorted(natural_f0)
    nclip = len(clips)

    shifts = rng.uniform(-max_semitones, max_semitones, size=nclip)
    timbres = rng.integers(0, n_timbres, size=nclip)
    held_mask = rng.random(nclip) < heldout_frac  # random clip holdout (continuous F0)

    rows = []
    for i, cid in enumerate(clips):
        nat = float(natural_f0[cid])
        rows.append(
            {
                "clip_id": cid,
                "natural_f0": nat,
                "shift_semitones": float(shifts[i]),
                "target_f0": nat * (2.0 ** (shifts[i] / 12.0)),
                "timbre": int(timbres[i]),
                "caption": NEUTRAL_CAPTION,
                "split": "heldout" if held_mask[i] else "train",
            }
        )
    return rows
