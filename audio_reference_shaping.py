"""Pure (torch-only, no ComfyUI / GPU) reference-shaping math for the audio IC-LoRA.

Two jobs, both on a raw waveform handed to the guide node:
  1. SELECT which ~window_sec slice of a reference clip to use (sustained energy, not peak).
  2. SHAPE what within it the model weighs most (a per-sample gain envelope).

It is waveform-domain on purpose: the VAE then always sees plausible audio (in-distribution),
unlike a latent-magnitude envelope. The gain is an honest *proxy* for attention weight
(louder content -> bigger latents -> more pull), not a true per-token attention scalar (that
would be a model-side change). See internal/audio_iclora_training/audio_reference_shaper_design.md.

Inputs are ``[..., n]`` waveforms (last dim = samples); energy analysis mono-mixes the leading
dims, and the resulting per-sample gain broadcasts back over them.
"""

from __future__ import annotations

import torch

# Energy is analysed on short non-overlapping frames. 25 ms is the usual speech/music frame.
_FRAME_SEC = 0.025
# gate (dialogue): a frame counts as "content" if its RMS is at least this fraction of the
# window's loudest frame. Breaths / room tone fall below it.
_GATE_FRAC = 0.15
# spotlight (manual): Gaussian width = focus_width * this. A Gaussian (vs a hard window) means
# a wider focus_width lifts the tails everywhere, which is the intuitive "broaden the emphasis".
_SPOTLIGHT_SIGMA_SCALE = 0.5
# dialogue auto-window: if the clip is no longer than window_sec * this, use the whole clip
# (a short voiced reference is already ~one window; don't go hunting).
_DIALOGUE_SHORT_MULT = 1.5

# input_type preset -> envelope mode.
_PRESET_MODE = {"dialogue": "gate", "song": "energy", "manual": "spotlight"}


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Collapse ``[..., n]`` to a 1-D ``[n]`` mono signal for energy analysis."""
    n = waveform.shape[-1]
    return waveform.reshape(-1, n).float().mean(dim=0)


def _frame_len(sample_rate: int, frame_sec: float = _FRAME_SEC) -> int:
    """Samples per analysis frame (at least 1)."""
    return max(1, int(round(frame_sec * sample_rate)))


def _clamp_window(start: int, window_samples: int, n: int) -> tuple[int, int]:
    """Clamp a ``window_samples``-long window starting near ``start`` into ``[0, n]``. Returns
    ``(start, end)`` exactly ``window_samples`` long when it fits, else bounded to the clip."""
    start = max(0, start)
    if start + window_samples > n:
        start = max(0, n - window_samples)
    return start, min(start + window_samples, n)


def frame_rms(waveform: torch.Tensor, sample_rate: int, frame_sec: float = _FRAME_SEC) -> torch.Tensor:
    """Per-frame RMS of the mono mix, on non-overlapping ``frame_sec`` frames. Returns ``[num_frames]``."""
    mono = _to_mono(waveform)
    n = mono.shape[-1]
    flen = _frame_len(sample_rate, frame_sec)
    nf = n // flen
    if nf == 0:
        return torch.sqrt((mono**2).mean().reshape(1) + 1e-12)
    frames = mono[: nf * flen].reshape(nf, flen)
    return torch.sqrt((frames**2).mean(dim=1) + 1e-12)


def select_reference_window(
    waveform: torch.Tensor, sample_rate: int, window_sec: float, frame_sec: float = _FRAME_SEC
) -> tuple[int, int]:
    """Pick the ``window_sec`` slice with the highest **sustained** energy (max mean frame-RMS
    over the window — NOT peak, which would chase a transient). Returns ``(start, end)`` sample
    indices, exactly ``window_sec`` long. A clip shorter than the window (or ``window_sec <= 0``)
    returns the whole clip."""
    n = waveform.shape[-1]
    window_samples = int(round(window_sec * sample_rate))
    if window_samples <= 0 or window_samples >= n:
        return (0, n)
    flen = _frame_len(sample_rate, frame_sec)
    rms = frame_rms(waveform, sample_rate, frame_sec)
    nf = rms.shape[0]
    wf = max(1, window_samples // flen)
    if wf >= nf:
        start = 0
    else:
        csum = torch.cat([torch.zeros(1, dtype=rms.dtype), rms.cumsum(0)])
        window_sums = csum[wf:] - csum[:-wf]   # mean over the window ∝ this sum (constant wf)
        start = int(torch.argmax(window_sums).item()) * flen
    return _clamp_window(start, window_samples, n)


def build_emphasis_envelope(
    rms: torch.Tensor,
    *,
    mode: str,
    emphasis: float,
    silence_floor: float,
    focus_center: float = 0.5,
    focus_width: float = 0.5,
) -> torch.Tensor:
    """Per-frame weight in ``[silence_floor, 1]``, blended toward flat by ``(1 - emphasis)`` so
    ``emphasis == 0`` is always all-ones (today's behavior). Modes:

    - ``flat``     — ones.
    - ``energy``   — weight ∝ loudness relative to the loudest frame (uniform-loud stays ~1).
    - ``gate``     — frames below ``_GATE_FRAC`` of the loudest -> floor; the rest -> 1 (dialogue).
    - ``spotlight``— Gaussian bump peaking at ``focus_center`` with width ``focus_width`` (manual).
    """
    num = rms.shape[0]
    floor = float(silence_floor)

    # Each mode produces a [0, 1] per-frame `signal`; the affine to [floor, 1] and the
    # emphasis blend toward flat are shared (so emphasis == 0 is all-ones for every mode).
    if mode == "flat":
        signal = torch.ones(num, dtype=torch.float32)
    elif mode == "energy":
        signal = rms.float() / (float(rms.max()) + 1e-12)   # relative to loudest, NOT min-max
    elif mode == "gate":
        signal = (rms >= _GATE_FRAC * float(rms.max())).float()
    elif mode == "spotlight":
        pos = torch.linspace(0.0, 1.0, num)
        sigma = max(1e-3, float(focus_width)) * _SPOTLIGHT_SIGMA_SCALE
        signal = torch.exp(-0.5 * ((pos - float(focus_center)) / sigma) ** 2)
    else:
        raise ValueError(f"unknown emphasis mode: {mode!r}")

    shaped = floor + (1.0 - floor) * signal
    e = float(emphasis)
    return (1.0 - e) + e * shaped


def _expand_frames_to_samples(env_frames: torch.Tensor, frame_len: int, total_len: int) -> torch.Tensor:
    """Frame weights -> per-sample gain (hold each frame's value across its samples; pad the
    trailing partial frame with the last value)."""
    gain = env_frames.repeat_interleave(frame_len)
    if gain.shape[0] < total_len:
        pad = env_frames[-1].expand(total_len - gain.shape[0])
        gain = torch.cat([gain, pad])
    return gain[:total_len]


def shape_reference_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    input_type: str,
    window_sec: float,
    window_select: str,
    window_offset_sec: float,
    silence_floor: float,
    emphasis: float,
    focus_center: float,
    focus_width: float,
) -> torch.Tensor:
    """Select a window of the reference and apply the emphasis envelope. Returns the shaped
    (and windowed) waveform with the same leading dims. ``input_type`` (dialogue/song/manual)
    chooses the envelope mode and the auto-window policy; ``window_select`` (auto/manual/whole)
    chooses where the window comes from."""
    if input_type not in _PRESET_MODE:
        raise ValueError(f"unknown input_type: {input_type!r}")
    n = waveform.shape[-1]
    window_samples = int(round(window_sec * sample_rate)) if window_sec > 0 else 0

    if window_select == "whole" or window_samples <= 0 or window_samples >= n:
        start, end = 0, n
    elif window_select == "manual":
        start, end = _clamp_window(int(round(window_offset_sec * sample_rate)), window_samples, n)
    else:  # auto
        if input_type == "dialogue" and n <= int(round(window_sec * _DIALOGUE_SHORT_MULT * sample_rate)):
            start, end = 0, n
        else:
            start, end = select_reference_window(waveform, sample_rate, window_sec)

    windowed = waveform[..., start:end]
    flen = _frame_len(sample_rate)
    rms = frame_rms(windowed, sample_rate)
    env = build_emphasis_envelope(
        rms,
        mode=_PRESET_MODE[input_type],
        emphasis=emphasis,
        silence_floor=silence_floor,
        focus_center=focus_center,
        focus_width=focus_width,
    )
    gain = _expand_frames_to_samples(env, flen, windowed.shape[-1]).to(windowed.dtype)
    return windowed * gain
