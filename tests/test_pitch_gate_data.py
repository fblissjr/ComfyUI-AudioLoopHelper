"""Red-first contract tests for the pitch-gate data generator.

Locks the leak discipline (design doc in internal/, private clone only) BEFORE
generating GPU-encoded latents: the reference voiced-tone's F0 must be the ONLY
thing that co-varies with the target pitch. Everything else decorrelated.

Run: uv run --group dev --group analysis python -m pytest tests/test_pitch_gate_data.py -v --rootdir=.
"""

from __future__ import annotations

import numpy as np

import pitch_gate_data as P


SR = 16_000
LEVELS = [90.0, 120.0, 150.0, 190.0, 240.0]


# ---- voiced tone: carries F0, harmonic (not sine), timbre ⊥ F0 ----

def test_voiced_tone_hits_requested_f0():
    for f0 in (90.0, 150.0, 240.0):
        a = P.synth_voiced_tone(f0, duration=1.0, sr=SR)
        d = P.dominant_f0(a, SR)
        assert abs(d - f0) < 5.0, f"F0 off for {f0}: got {d:.1f}"


def test_voiced_tone_is_harmonic_not_sine():
    """A pure sine has energy only at f0; a voiced tone has a harmonic stack.
    Assert real energy at 2*f0 and 3*f0 (the OOD-avoidance LTX-2 flagged)."""
    f0 = 150.0
    a = P.synth_voiced_tone(f0, duration=1.0, sr=SR)
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    freqs = np.fft.rfftfreq(len(a), 1 / SR)

    def energy_at(f):
        return spec[np.argmin(np.abs(freqs - f))]

    e1, e2, e3 = energy_at(f0), energy_at(2 * f0), energy_at(3 * f0)
    assert e2 > 0.05 * e1 and e3 > 0.02 * e1, "no harmonic stack -> too sine-like / OOD"


def test_timbre_varies_spectrum_not_pitch():
    f0 = 150.0
    a0 = P.synth_voiced_tone(f0, timbre=0.0, sr=SR)
    a1 = P.synth_voiced_tone(f0, timbre=1.0, sr=SR)
    # same F0
    assert abs(P.dominant_f0(a0, SR) - P.dominant_f0(a1, SR)) < 5.0
    # different timbre -> different spectral centroid
    assert abs(P.spectral_centroid(a0, SR) - P.spectral_centroid(a1, SR)) > 50.0


# ---- pitch shift: moves F0 by the requested interval, preserves duration ----

def test_pitch_shift_doubles_f0_at_octave():
    a = P.synth_voiced_tone(150.0, duration=1.0, sr=SR)
    up = P.pitch_shift_semitones(a, SR, +12)
    d = P.dominant_f0(up, SR)
    assert abs(d - 300.0) < 20.0, f"octave up wrong: {d:.1f}"
    assert len(up) == len(a), "duration not preserved (timing must stay for lip-sync)"


# ---- manifest: the decorrelation matrix ----

def _natural(n=200, seed=1):
    rng = np.random.default_rng(seed)
    return {f"clip_{i:04d}": float(rng.uniform(90, 240)) for i in range(n)}


def test_manifest_balanced_across_levels():
    rows = P.build_manifest(_natural(), LEVELS, seed=0)
    counts = {lv: sum(r["target_f0"] == lv for r in rows) for lv in LEVELS}
    lo, hi = min(counts.values()), max(counts.values())
    assert hi - lo <= 1, f"levels not balanced: {counts}"


def test_target_pitch_decorrelated_from_natural_pitch():
    """The killer leak: if target F0 tracks the clip's natural F0, the init
    frame (face->gender->F0) leaks pitch. Must be ~0 correlation."""
    rows = P.build_manifest(_natural(n=400), LEVELS, seed=0)
    nat = np.array([r["natural_f0"] for r in rows])
    tgt = np.array([r["target_f0"] for r in rows])
    r = np.corrcoef(nat, tgt)[0, 1]
    assert abs(r) < 0.15, f"target F0 correlates with natural F0 (r={r:.2f}) -> face leaks pitch"


def test_timbre_decorrelated_from_target_pitch():
    rows = P.build_manifest(_natural(n=400), LEVELS, seed=0)
    tim = np.array([r["timbre"] for r in rows], dtype=float)
    tgt = np.array([r["target_f0"] for r in rows])
    r = np.corrcoef(tim, tgt)[0, 1]
    assert abs(r) < 0.15, f"timbre correlates with F0 (r={r:.2f}) -> learns waveform not F0"
    assert len(set(tim)) > 1, "timbre must actually vary (else lookup-table risk)"


def test_caption_is_pitch_free():
    rows = P.build_manifest(_natural(), [120.0, 150.0, 190.0], seed=0)
    caps = {r["caption"].lower() for r in rows}
    assert len(caps) == 1, "caption must be constant"
    cap = caps.pop()
    words = set(cap.split())
    for banned in P.PITCH_BAN:
        assert banned not in words, f"caption leaks pitch word '{banned}'"


def test_heldout_split_disjoint_and_has_middle_level():
    rows = P.build_manifest(_natural(n=400), LEVELS, seed=0, heldout_frac=0.2)
    train = {r["clip_id"] for r in rows if r["split"] == "train"}
    held = {r["clip_id"] for r in rows if r["split"] == "heldout"}
    assert train and held and not (train & held), "splits overlap or empty"
    held_levels = {r["target_f0"] for r in rows if r["split"] == "heldout"}
    assert 150.0 in held_levels, "held-out must include a middle level (interpolation eval)"
