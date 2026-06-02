"""Tests for the audio reference shaper: pick which window of a reference clip to use, and
shape what within it the model weighs most. Pure torch math (no ComfyUI / GPU), per the
design at internal/audio_iclora_training/audio_reference_shaper_design.md.

Run: uv run --group dev --group analysis python -m pytest tests/test_audio_reference_shaper.py -v --rootdir=.
"""

from __future__ import annotations

import pytest
import torch

import audio_reference_shaping as S


SR = 16000


def _const(amp: float, seconds: float, sr: int = SR) -> torch.Tensor:
    """A [1, 1, n] constant-amplitude mono clip."""
    n = int(round(seconds * sr))
    return torch.full((1, 1, n), float(amp))


def _concat(*clips: torch.Tensor) -> torch.Tensor:
    return torch.cat(clips, dim=-1)


# ---- frame_rms ----------------------------------------------------------------

def test_frame_rms_tracks_amplitude():
    """A quiet head then a loud tail -> low frame-RMS then high."""
    wav = _concat(_const(0.0, 0.5), _const(1.0, 0.5))
    rms = S.frame_rms(wav, SR)
    assert rms.ndim == 1
    assert rms[0].item() < 0.1          # quiet head
    assert rms[-1].item() > 0.9          # loud tail


# ---- window selection: sustained, not peak ------------------------------------

def test_select_window_prefers_sustained_over_transient():
    """A single loud frame (transient) early, then a sustained moderate block. A ~1 s
    window should land on the SUSTAINED block, not be dragged onto the spike."""
    wav = _concat(
        _const(0.0, 0.2),
        _const(10.0, 0.04),   # one ~frame-length transient spike
        _const(0.0, 0.8),
        _const(1.0, 1.6),     # sustained block, longer than the window
        _const(0.0, 0.2),
    )
    start, end = S.select_reference_window(wav, SR, window_sec=1.0)
    win_start_sec = start / SR
    # the sustained block starts at 0.2+0.04+0.8 = 1.04 s; the window should sit inside it
    assert win_start_sec >= 1.0
    assert (end - start) == int(round(1.0 * SR))


def test_select_window_whole_clip_when_shorter_than_window():
    wav = _const(0.5, 2.0)
    start, end = S.select_reference_window(wav, SR, window_sec=3.5)
    assert (start, end) == (0, wav.shape[-1])


def test_select_window_whole_when_window_sec_zero():
    wav = _const(0.5, 5.0)
    start, end = S.select_reference_window(wav, SR, window_sec=0.0)
    assert (start, end) == (0, wav.shape[-1])


# ---- emphasis envelope --------------------------------------------------------

def test_envelope_flat_when_emphasis_zero():
    """Every mode collapses to all-ones at emphasis=0 (== today's behavior)."""
    rms = torch.tensor([0.0, 0.2, 1.0, 0.1, 0.9])
    for mode in ("flat", "energy", "gate", "spotlight"):
        env = S.build_emphasis_envelope(rms, mode=mode, emphasis=0.0, silence_floor=0.1)
        assert torch.allclose(env, torch.ones_like(env)), mode


def test_envelope_gate_drops_quiet_to_floor():
    """gate: frames below the gate threshold -> silence_floor; loud frames -> 1.0."""
    rms = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
    env = S.build_emphasis_envelope(rms, mode="gate", emphasis=1.0, silence_floor=0.1)
    assert torch.allclose(env[[0, 1, 4]], torch.full((3,), 0.1), atol=1e-6)
    assert torch.allclose(env[[2, 3]], torch.ones(2), atol=1e-6)


def test_envelope_energy_tracks_rms_and_respects_floor():
    rms = torch.tensor([0.0, 0.5, 1.0])
    env = S.build_emphasis_envelope(rms, mode="energy", emphasis=1.0, silence_floor=0.2)
    assert abs(env[0].item() - 0.2) < 1e-6           # quietest -> floor
    assert abs(env[2].item() - 1.0) < 1e-6           # loudest -> 1
    assert env[0] < env[1] < env[2]                  # monotonic with energy
    assert env.min() >= 0.2 - 1e-6


def test_envelope_spotlight_peaks_at_focus_center():
    rms = torch.zeros(100)                            # spotlight ignores energy
    env = S.build_emphasis_envelope(
        rms, mode="spotlight", emphasis=1.0, silence_floor=0.0,
        focus_center=0.2, focus_width=0.3,
    )
    assert env.argmax().item() == int(0.2 * 100)      # peak at 20%
    # a wider spotlight raises the weight far from the peak
    wide = S.build_emphasis_envelope(
        rms, mode="spotlight", emphasis=1.0, silence_floor=0.0,
        focus_center=0.2, focus_width=0.8,
    )
    assert wide[80].item() > env[80].item()


# ---- end-to-end shaping -------------------------------------------------------

def test_shape_dialogue_short_voiced_clip_is_near_flat():
    """A short, fully-voiced dialogue clip should come back ~unchanged (parity): we don't
    accidentally reshape a clip that is already a good reference."""
    wav = _const(0.7, 3.0)                            # uniform energy, shorter than 3.5 s
    out = S.shape_reference_waveform(
        wav, SR, input_type="dialogue", window_sec=3.5, window_select="auto",
        window_offset_sec=0.0, silence_floor=0.1, emphasis=1.0,
        focus_center=0.5, focus_width=0.5,
    )
    assert out.shape == wav.shape
    assert torch.allclose(out, wav, atol=1e-5)        # nothing gated (all voiced), no window cut


def test_shape_song_selects_hook_and_returns_window_length():
    """A full 'song' with a quiet intro/outro and a loud hook: output is the hook window,
    length == window_sec, and not the quiet intro."""
    wav = _concat(_const(0.02, 4.0), _const(1.0, 3.5), _const(0.02, 4.0))  # quiet, hook, quiet
    out = S.shape_reference_waveform(
        wav, SR, input_type="song", window_sec=3.5, window_select="auto",
        window_offset_sec=0.0, silence_floor=0.15, emphasis=1.0,
        focus_center=0.5, focus_width=0.5,
    )
    assert out.shape[-1] == int(round(3.5 * SR))      # window length
    assert out.abs().mean().item() > 0.5              # landed on the loud hook, not the quiet intro


def test_shape_manual_offset_takes_requested_slice():
    wav = _concat(_const(0.1, 2.0), _const(0.9, 2.0))
    out = S.shape_reference_waveform(
        wav, SR, input_type="manual", window_sec=1.0, window_select="manual",
        window_offset_sec=2.5, silence_floor=0.0, emphasis=0.0,   # emphasis 0 => pure slice
        focus_center=0.5, focus_width=0.5,
    )
    assert out.shape[-1] == int(round(1.0 * SR))
    assert torch.allclose(out, torch.full_like(out, 0.9), atol=1e-6)  # slice from the loud half


# ---- select_window_bounds (the single windowing primitive) --------------------

def test_window_bounds_head_takes_first_n():
    wav = _const(0.5, 5.0)
    assert S.select_window_bounds(wav, SR, window_sec=3.5, mode="head") == (0, int(round(3.5 * SR)))


def test_window_bounds_manual_takes_offset_slice():
    wav = _const(0.5, 5.0)
    start, end = S.select_window_bounds(wav, SR, window_sec=1.0, mode="manual", offset_sec=2.0)
    assert (start, end) == (int(round(2.0 * SR)), int(round(3.0 * SR)))


def test_window_bounds_whole_ignores_window_sec():
    wav = _const(0.5, 5.0)
    assert S.select_window_bounds(wav, SR, window_sec=3.5, mode="whole") == (0, wav.shape[-1])


def test_window_bounds_window_ge_clip_returns_whole():
    wav = _const(0.5, 2.0)
    for mode in ("head", "manual", "auto"):
        assert S.select_window_bounds(wav, SR, window_sec=3.5, mode=mode) == (0, wav.shape[-1])


def test_window_bounds_nonpositive_window_returns_whole():
    wav = _const(0.5, 5.0)
    assert S.select_window_bounds(wav, SR, window_sec=0.0, mode="head") == (0, wav.shape[-1])


def test_window_bounds_auto_matches_select_reference_window():
    wav = _concat(_const(0.02, 4.0), _const(1.0, 3.5), _const(0.02, 4.0))
    assert S.select_window_bounds(wav, SR, window_sec=3.5, mode="auto") == S.select_reference_window(wav, SR, 3.5)


def test_window_bounds_unknown_mode_raises():
    wav = _const(0.5, 5.0)
    with pytest.raises(ValueError):
        S.select_window_bounds(wav, SR, window_sec=3.5, mode="bogus")


# ---- compose_reference (multi-slice composer) --------------------------------

def test_compose_empty_returns_whole_clip():
    wav = _const(0.5, 5.0)
    out = S.compose_reference(wav, SR, [])
    assert out.shape == wav.shape


def test_compose_single_slice_is_that_window():
    wav = _const(0.5, 5.0)
    out = S.compose_reference(wav, SR, [{"start_sec": 1.0, "end_sec": 2.0, "gain": 1.0}])
    assert out.shape[-1] == int(round(1.0 * SR))


def test_compose_two_slices_concatenate_in_order():
    wav = _concat(_const(0.2, 2.0), _const(0.9, 2.0))   # quiet [0,2), loud [2,4)
    # request the LOUD slice first, then the QUIET one — order must be preserved
    out = S.compose_reference(wav, SR, [
        {"start_sec": 2.5, "end_sec": 3.5, "gain": 1.0},
        {"start_sec": 0.5, "end_sec": 1.5, "gain": 1.0},
    ], fade_sec=0.0)
    assert out.shape[-1] == int(round(2.0 * SR))
    half = int(round(1.0 * SR))
    assert abs(out[..., half // 2].item() - 0.9) < 1e-4    # first slice = loud
    assert abs(out[..., half + half // 2].item() - 0.2) < 1e-4  # second slice = quiet


def test_compose_per_slice_gain():
    wav = _const(1.0, 5.0)
    out = S.compose_reference(wav, SR, [{"start_sec": 1.0, "end_sec": 2.0, "gain": 0.5}], fade_sec=0.0)
    assert abs(out[..., out.shape[-1] // 2].item() - 0.5) < 1e-5


def test_compose_edge_fade_avoids_click():
    wav = _const(1.0, 5.0)
    out = S.compose_reference(wav, SR, [{"start_sec": 1.0, "end_sec": 2.0, "gain": 1.0}], fade_sec=0.05)
    assert out[..., 0].item() < 0.2                         # faded in from ~0
    assert abs(out[..., out.shape[-1] // 2].item() - 1.0) < 1e-5  # full in the middle


def test_compose_invalid_segments_fall_back_to_whole():
    wav = _const(0.5, 5.0)
    out = S.compose_reference(wav, SR, [{"start_sec": 2.0, "end_sec": 1.0, "gain": 1.0}])  # end<=start
    assert out.shape == wav.shape


# ---- auto_window_segment (the "auto-find hook" button engine) ----------------

def test_auto_window_segment_returns_single_hook_slice():
    """A quiet/loud/quiet clip: auto-find returns ONE segment sitting on the loud hook,
    window_sec long, with unity gain — i.e. a single compose slice."""
    wav = _concat(_const(0.02, 4.0), _const(1.0, 3.5), _const(0.02, 4.0))  # quiet, hook, quiet
    seg = S.auto_window_segment(wav, SR, window_sec=3.5)
    assert set(seg) == {"start_sec", "end_sec", "gain"}
    assert seg["gain"] == 1.0
    assert abs((seg["end_sec"] - seg["start_sec"]) - 3.5) < 1e-3   # window length
    assert seg["start_sec"] >= 4.0 - 0.05                          # landed on/after the hook start


def test_auto_window_segment_short_clip_returns_whole():
    """A clip shorter than window_sec comes back as the whole clip (start 0, end == duration)."""
    wav = _const(0.5, 2.0)
    seg = S.auto_window_segment(wav, SR, window_sec=3.5)
    assert seg["start_sec"] == 0.0
    assert abs(seg["end_sec"] - 2.0) < 1e-3


def test_auto_window_segment_matches_select_window_bounds():
    """Parity lock: the segment bounds equal select_window_bounds(mode='auto') / sr, so the
    button can never drift from the engine the head-trim + training selection use."""
    wav = _concat(_const(0.02, 4.0), _const(1.0, 3.5), _const(0.02, 4.0))
    start, end = S.select_window_bounds(wav, SR, window_sec=3.5, mode="auto")
    seg = S.auto_window_segment(wav, SR, window_sec=3.5)
    assert abs(seg["start_sec"] - start / SR) < 1e-9
    assert abs(seg["end_sec"] - end / SR) < 1e-9


# ---- reference_envelope (for the visual editor) ------------------------------

def test_reference_envelope_shape_norm_and_duration():
    wav = _concat(_const(0.1, 1.0), _const(1.0, 1.0))   # 2s: quiet half then loud half
    env = S.reference_envelope(wav, SR, buckets=100)
    assert abs(env["duration"] - 2.0) < 1e-3
    assert env["sr"] == SR
    assert 0 < len(env["peaks"]) <= 105
    assert max(env["peaks"]) == 1.0                      # normalized to the loudest bucket
    assert env["peaks"][0] < env["peaks"][-1]            # quiet first half < loud second half


def test_reference_envelope_empty_clip():
    env = S.reference_envelope(torch.zeros(1, 1, 0), SR)
    assert env["peaks"] == [] and env["duration"] == 0.0
