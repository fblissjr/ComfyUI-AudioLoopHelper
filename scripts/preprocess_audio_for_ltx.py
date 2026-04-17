#!/usr/bin/env python3
"""Preprocess audio for LTX 2.3 audio-video generation.

Applies a spectral-rebalance + loudness-normalize chain targeted at LTX-2's
audio VAE characteristics (16 kHz internal, n_fft=1024, mel_hop_length=160
-- see coderef/LTX-2/packages/ltx-core/src/ltx_core/model/audio_vae/).

The defaults correct for two recurring source issues that hurt lip sync:

1. Bass-heavy voice (boomy mic / proximity effect / untreated room) that
   masks consonants in the 1.5-4 kHz intelligibility band.

2. Rolled-off sibilance (4-8 kHz) that leaves fricative phonemes
   (/s/, /sh/, /t/, /k/) with no signal for LTX's audio-video
   cross-attention to attend to when deciding mouth shape.

Emits WAV to avoid MP3 re-encoding overshoot above the true-peak ceiling.

Usage:
    uv run --group analysis python scripts/preprocess_audio_for_ltx.py \\
        input.mp3 output.wav \\
        [--trim-start 0] [--trim-end 184] \\
        [--target-lufs -16] [--tp-ceiling -2.0]

Requires: ffmpeg on PATH, librosa installed (dependency group: analysis).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np

# EQ stages, tuned for LTX-2 audio VAE (16 kHz band-limited, mel-hop 10ms).
# Each entry: (type, freq, Q/width, gain_dB).
# Types: "hp" = highpass, "eq" = peaking EQ bell.
_EQ_CHAIN = [
    ("hp", 80, None, None),            # kill subsonic rumble, mic handling
    ("eq", 200, 1.0, -3.0),            # low-shelf-ish cut: de-boom chest resonance
    ("eq", 400, 1.2, -2.0),            # box cut: untreated-room artifact
    ("eq", 3000, 1.0, +4.0),           # presence boost: F2/F3 formants
    ("eq", 6500, 0.7, +3.0),           # sibilance shelf: recover fricatives
]


def _build_filter_chain(target_lufs: float, tp_ceiling: float, lra: float) -> str:
    """Build the ffmpeg -af filter chain string."""
    stages = []
    for kind, f, q, g in _EQ_CHAIN:
        if kind == "hp":
            stages.append(f"highpass=f={f}")
        elif kind == "eq":
            stages.append(f"equalizer=f={f}:t=q:w={q}:g={g}")
    stages.append(f"loudnorm=I={target_lufs}:LRA={lra}:TP={tp_ceiling}")
    return ",".join(stages)


def _analyze(path: Path) -> dict:
    """Spectral + level + SNR analysis at LTX's internal 16 kHz rate."""
    y, _ = librosa.load(str(path), sr=16000, mono=True)
    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    frames = librosa.util.frame(y, frame_length=2048, hop_length=1024)
    frame_rms = np.sqrt((frames**2).mean(axis=0))
    nf = float(np.percentile(frame_rms, 5))
    p90 = float(np.percentile(frame_rms, 90))
    snr = 20.0 * np.log10(p90 / (nf + 1e-12))

    S_power = np.abs(librosa.stft(y, n_fft=1024, hop_length=160)) ** 2
    freqs = librosa.fft_frequencies(sr=16000, n_fft=1024)
    bands = [(60, 300), (300, 800), (800, 1500), (1500, 4000), (4000, 8000)]
    band_e = {}
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_e[(lo, hi)] = float(S_power[mask].sum() / (hi - lo))
    ref = max(band_e.values())
    shape = {b: 10.0 * np.log10(e / ref + 1e-12) for b, e in band_e.items()}

    return {
        "rms_db": 20.0 * np.log10(rms + 1e-12),
        "peak_db": 20.0 * np.log10(peak + 1e-12),
        "snr_db": snr,
        "shape": shape,
    }


def _print_comparison(before: dict, after: dict) -> None:
    """Print side-by-side before/after table."""
    print()
    print(f"{'metric':32s}  {'before':>10s}  {'after':>10s}  {'delta':>10s}")
    print("-" * 68)
    for key, label in [("rms_db", "RMS (dBFS)"),
                       ("peak_db", "Peak (dBFS)"),
                       ("snr_db", "SNR voice/noise (dB)")]:
        b, a = before[key], after[key]
        print(f"{label:32s}  {b:>+10.1f}  {a:>+10.1f}  {a - b:>+10.1f}")
    print()
    print(f"{'spectral band (rel. to loudest)':32s}  "
          f"{'before':>10s}  {'after':>10s}  {'delta':>10s}")
    print("-" * 68)
    for lo, hi in [(60, 300), (300, 800), (800, 1500), (1500, 4000), (4000, 8000)]:
        label = f"{lo}-{hi} Hz"
        b = before["shape"][(lo, hi)]
        a = after["shape"][(lo, hi)]
        print(f"{label:32s}  {b:>+10.1f}  {a:>+10.1f}  {a - b:>+10.1f}")
    print()


def _run_ffmpeg(
    src: Path,
    dst: Path,
    chain: str,
    trim_start: float | None,
    trim_end: float | None,
    sample_rate: int,
) -> None:
    """Invoke ffmpeg with the filter chain."""
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-y", "-i", str(src)]
    if trim_start is not None:
        cmd += ["-ss", str(trim_start)]
    if trim_end is not None:
        cmd += ["-to", str(trim_end)]
    cmd += [
        "-af", chain,
        "-ac", "1",                 # mono — audio VAE is single-channel
        "-ar", str(sample_rate),    # source-rate, LTX resamples to 16k internally
        "-c:a", "pcm_s16le",        # WAV 16-bit; avoids MP3 inter-sample-peak overshoot
        str(dst),
    ]
    print(f"Running: ffmpeg ... -af \"{chain}\" {dst.name}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"ffmpeg failed: exit {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess audio for LTX 2.3 lip-sync quality. Rebalances "
            "spectrum (de-boom, lift presence + sibilance) and normalizes "
            "loudness. Outputs mono WAV."
        )
    )
    parser.add_argument("input", type=Path, help="Input audio (any format ffmpeg reads)")
    parser.add_argument("output", type=Path, help="Output WAV path")
    parser.add_argument("--trim-start", type=float, default=None,
                        help="Trim start time in seconds (default: no trim)")
    parser.add_argument("--trim-end", type=float, default=None,
                        help="Trim end time in seconds (default: no trim)")
    parser.add_argument("--target-lufs", type=float, default=-16.0,
                        help="Integrated loudness target in LUFS (default -16, streaming-safe)")
    parser.add_argument("--tp-ceiling", type=float, default=-2.0,
                        help="True-peak ceiling in dBTP (default -2.0)")
    parser.add_argument("--lra", type=float, default=11.0,
                        help="Loudness range target in LU (default 11)")
    parser.add_argument("--sample-rate", type=int, default=44100,
                        help="Output sample rate in Hz (default 44100). LTX resamples to 16k internally regardless.")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the before/after spectral analysis")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    chain = _build_filter_chain(args.target_lufs, args.tp_ceiling, args.lra)

    if args.no_verify:
        _run_ffmpeg(args.input, args.output, chain,
                    args.trim_start, args.trim_end, args.sample_rate)
        print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
        return

    before = _analyze(args.input)
    _run_ffmpeg(args.input, args.output, chain,
                args.trim_start, args.trim_end, args.sample_rate)
    after = _analyze(args.output)
    _print_comparison(before, after)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
