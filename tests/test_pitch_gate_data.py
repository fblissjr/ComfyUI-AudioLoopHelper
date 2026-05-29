"""Red-first contract tests for the pitch-gate data generator.

Locks the contract for the pitch-gate generator (design doc in internal/, private
clone only) BEFORE generating GPU-encoded latents. Option B (bounded shift): the
reference voiced-tone's F0 is the sole F0-bearing input; shifts stay in the
artifact-free range; timbre is a decorrelated nuisance axis; the caption is constant.

Run: uv run --group dev --group analysis python -m pytest tests/test_pitch_gate_data.py -v --rootdir=.
"""

from __future__ import annotations

import numpy as np

import pitch_gate_data as P


SR = 16_000


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


# ---- manifest (Option B: bounded shift from natural; reference is the sole carrier) ----
# NOTE: the old target⊥natural decorrelation test is intentionally GONE. Its only purpose
# was to stop an init FACE from leaking pitch (face->gender->F0). R-init resolved to DROP
# the init frame, so there is no face input; with a constant caption, nothing but the
# reference tone carries F0. Option B therefore lets target track natural (bounded shift)
# leak-free, which buys artifact-free shifts (the smoke confirmed >±8 st degrades).

def _natural(n=200, seed=1):
    rng = np.random.default_rng(seed)
    return {f"clip_{i:04d}": float(rng.uniform(80, 260)) for i in range(n)}


def test_shifts_are_bounded():
    rows = P.build_manifest(_natural(n=400), seed=0, max_semitones=7.0)
    assert all(abs(r["shift_semitones"]) <= 7.0 + 1e-9 for r in rows), "shift exceeded the artifact-free bound"


def test_target_is_natural_times_shift():
    rows = P.build_manifest(_natural(n=50), seed=0, max_semitones=7.0)
    for r in rows:
        expected = r["natural_f0"] * 2.0 ** (r["shift_semitones"] / 12.0)
        assert abs(r["target_f0"] - expected) < 1e-6


def test_target_pitch_spread_is_wide():
    """LTX-2's ask: actual_f0 must span a real range so the eval tests F0-tracking
    across pitch. Bounded shift over a diverse natural set delivers it."""
    rows = P.build_manifest(_natural(n=400), seed=0, max_semitones=7.0)
    tgt = np.array([r["target_f0"] for r in rows])
    assert tgt.max() - tgt.min() > 150.0, f"target spread too narrow: {tgt.min():.0f}-{tgt.max():.0f}"


def test_timbre_decorrelated_from_target_pitch():
    rows = P.build_manifest(_natural(n=400), seed=0)
    tim = np.array([r["timbre"] for r in rows], dtype=float)
    tgt = np.array([r["target_f0"] for r in rows])
    r = np.corrcoef(tim, tgt)[0, 1]
    assert abs(r) < 0.15, f"timbre correlates with F0 (r={r:.2f}) -> learns waveform not F0"
    assert len(set(tim)) > 1, "timbre must actually vary (else lookup-table risk)"


def test_caption_is_pitch_free():
    rows = P.build_manifest(_natural(), seed=0)
    caps = {r["caption"].lower() for r in rows}
    assert len(caps) == 1, "caption must be constant"
    cap = caps.pop()
    words = set(cap.split())
    for banned in P.PITCH_BAN:
        assert banned not in words, f"caption leaks pitch word '{banned}'"


def test_heldout_split_disjoint_and_spans_range():
    rows = P.build_manifest(_natural(n=400), seed=0, heldout_frac=0.2)
    train = {r["clip_id"] for r in rows if r["split"] == "train"}
    held = [r for r in rows if r["split"] == "heldout"]
    held_ids = {r["clip_id"] for r in held}
    assert train and held_ids and not (train & held_ids), "splits overlap or empty"
    htgt = np.array([r["target_f0"] for r in held])
    assert htgt.max() - htgt.min() > 100.0, "held-out F0 too clustered for an interpolation test"
