"""Contract tests for the pitch-gate F0-tracking metric (the eval gate).

The gate (design doc §7): feed held-out reference tones at F0=P, decode the generated
audio, measure its F0. A trained LoRA should make output F0 TRACK P (slope≈1, high R2);
the no-LoRA base should NOT (slope≈0). gate = LoRA tracks AND beats base.

These tests use synthetic signals + synthetic (tone,output) pairs — no GPU, no model,
no checkpoint. They lock the metric math before it scores real renders.

Run: uv run --group dev --group analysis python -m pytest tests/test_pitch_gate_eval.py -v --rootdir=.
"""

from __future__ import annotations

import numpy as np

import pitch_gate_data as D  # synth_voiced_tone, for measure_output_f0 ground truth
import pitch_gate_eval as E


SR = 16_000


# ---- measure_output_f0: recovers the fundamental of voiced audio ----

def test_measure_output_f0_recovers_tone():
    for f0 in (110.0, 175.0, 260.0):
        a = D.synth_voiced_tone(f0, duration=2.0, sr=SR)
        got = E.measure_output_f0(a, SR)
        assert abs(got - f0) < 12.0, f"F0 {f0}: measured {got:.1f}"


def test_measure_output_f0_handles_silence():
    """All-silence (unvoiced) → NaN, not a bogus number that pollutes the slope."""
    got = E.measure_output_f0(np.zeros(SR, dtype=np.float32), SR)
    assert np.isnan(got), f"silence should be NaN, got {got}"


# ---- tracking_slope: least-squares slope + R^2 over (tone_f0, output_f0) ----

def test_tracking_slope_perfect():
    tone = np.array([110.0, 150.0, 190.0, 240.0, 290.0])
    out = tone.copy()  # perfect tracking
    s = E.tracking_slope(tone, out)
    assert abs(s.slope - 1.0) < 1e-6
    assert s.r2 > 0.999


def test_tracking_slope_flat_is_zero():
    """Base arm that ignores the tone → output ~constant → slope ~0, R^2 ~0."""
    tone = np.array([110.0, 150.0, 190.0, 240.0, 290.0])
    out = np.full_like(tone, 165.0)
    s = E.tracking_slope(tone, out)
    assert abs(s.slope) < 0.05
    assert s.r2 < 0.1


def test_tracking_slope_ignores_nan_pairs():
    tone = np.array([110.0, 150.0, np.nan, 240.0, 290.0])
    out = np.array([108.0, 152.0, 200.0, np.nan, 291.0])
    s = E.tracking_slope(tone, out)  # only 3 finite pairs (110,150,290) used
    assert s.n == 3
    assert s.slope > 0.8


# ---- gate_verdict: LoRA tracks AND beats base ----

def test_gate_passes_when_lora_tracks_and_base_flat():
    tone = np.array([110.0, 150.0, 190.0, 240.0, 290.0])
    lora = tone * 0.95 + 5  # strong tracking
    base = np.full_like(tone, 170.0)  # flat
    v = E.gate_verdict(tone, lora, tone, base)
    assert v.passed
    assert v.lora.slope > 0.7 and v.base.slope < 0.2


def test_gate_fails_when_lora_also_flat():
    tone = np.array([110.0, 150.0, 190.0, 240.0, 290.0])
    flat = np.full_like(tone, 170.0)
    v = E.gate_verdict(tone, flat, tone, flat)
    assert not v.passed


def test_gate_fails_when_lora_no_better_than_base():
    """Both track equally → LoRA earns nothing over base → fail (earn-its-keep)."""
    tone = np.array([110.0, 150.0, 190.0, 240.0, 290.0])
    both = tone.copy()
    v = E.gate_verdict(tone, both, tone, both)
    assert not v.passed, "LoRA must BEAT base, not merely match a base that already tracks"
