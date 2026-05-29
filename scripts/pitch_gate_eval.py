"""Pitch-gate F0-tracking metric (the eval gate).

Design: internal/audio_iclora_training/audio_only_iclora_pitch_gate.md §7 (private clone only).

The gate: feed held-out reference tones at F0=P, decode the generated audio, measure its
F0. A trained audio-reference LoRA should make output F0 TRACK P (slope≈1, high R2); the
no-LoRA base should NOT (slope≈0). Pass iff the LoRA tracks AND beats base by a margin
(earn-its-keep — pitch is base-absent, so a real win is unambiguous).

Pure measurement + stats; no GPU, no model. Feeds on decoded-audio wavs (LoRA arm + base
arm) produced by the ComfyUI eval graph, plus the tone F0s that drove them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Gate thresholds (design §7). LoRA must track meaningfully AND beat base by a margin.
MIN_SLOPE = 0.5     # output F0 moves at least half a Hz per Hz of tone (ideal 1.0)
MIN_R2 = 0.5        # the tracking is consistent, not noise
MIN_SLOPE_DELTA = 0.3   # LoRA − base slope: the earn-its-keep margin over the base prior


@dataclass
class Slope:
    slope: float
    intercept: float
    r2: float
    n: int


@dataclass
class Verdict:
    passed: bool
    lora: Slope
    base: Slope
    reason: str


def measure_output_f0(audio: np.ndarray, sr: int = 16_000,
                      fmin: float = 70.0, fmax: float = 400.0) -> float:
    """Median voiced F0 (Hz) of generated speech via librosa.pyin. Returns NaN if no
    voiced frames (silence/unvoiced) so it can't pollute the tracking slope."""
    import librosa

    y = np.asarray(audio, dtype=np.float32)
    f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
    vals = f0[np.isfinite(f0) & voiced] if voiced is not None else f0[np.isfinite(f0)]
    return float(np.median(vals)) if vals.size else float("nan")


def tracking_slope(tone_f0: np.ndarray, output_f0: np.ndarray) -> Slope:
    """Least-squares slope + R^2 of output_f0 vs tone_f0 over finite pairs."""
    x = np.asarray(tone_f0, dtype=float)
    y = np.asarray(output_f0, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(x.size)
    if n < 2 or np.allclose(x, x[0]):
        return Slope(slope=float("nan"), intercept=float("nan"), r2=float("nan"), n=n)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return Slope(slope=float(slope), intercept=float(intercept), r2=float(r2), n=n)


def gate_verdict(tone_lora: np.ndarray, out_lora: np.ndarray,
                 tone_base: np.ndarray, out_base: np.ndarray,
                 min_slope: float = MIN_SLOPE, min_r2: float = MIN_R2,
                 min_delta: float = MIN_SLOPE_DELTA) -> Verdict:
    """PASS iff the LoRA arm tracks (slope≥min_slope, R2≥min_r2) AND beats the base arm's
    slope by ≥min_delta. The margin is the earn-its-keep check: a base that already tracks
    leaves nothing for the LoRA to prove."""
    lora = tracking_slope(tone_lora, out_lora)
    base = tracking_slope(tone_base, out_base)

    base_slope = 0.0 if not np.isfinite(base.slope) else base.slope
    tracks = np.isfinite(lora.slope) and lora.slope >= min_slope and lora.r2 >= min_r2
    beats = np.isfinite(lora.slope) and (lora.slope - base_slope) >= min_delta

    if tracks and beats:
        reason = f"LoRA tracks (slope={lora.slope:.2f}, R2={lora.r2:.2f}) and beats base (slope={base_slope:.2f})"
    elif not tracks:
        reason = f"LoRA does not track (slope={lora.slope:.2f}, R2={lora.r2:.2f}; need slope≥{min_slope}, R2≥{min_r2})"
    else:
        reason = f"LoRA slope {lora.slope:.2f} does not beat base {base_slope:.2f} by ≥{min_delta} (earn-its-keep fail)"
    return Verdict(passed=bool(tracks and beats), lora=lora, base=base, reason=reason)
