"""Build the pitch-gate dataset (CPU media-gen stage; no GPU/VAE).

Design: internal/audio_iclora_training/audio_only_iclora_pitch_gate.md (private clone only)

Per source clip, emits a (voiced-tone reference @ F0, AV target whose speech is
pitch-shifted to that same F0) pair where the reference's F0 is the ONLY thing that
co-varies with the target pitch. No init frame (R-init resolved -> drop). The GPU
VAE-encode to latents is a SEPARATE coordinated step (encode the AV target + the tone
reference through the video/audio VAEs); this script only produces the media + manifest.

Output (flat, matches the synth_e1_* sets) under data/audio_iclora/<name>/:
  clips/clip_NNNN.mp4       AV target: source video scaled to 256, pitched audio muxed
  references/ref_NNNN.wav   voiced-tone reference @ the target's actual F0 (16 kHz mono)
  manifest.jsonl            one row per pair (target_f0/actual_f0/natural_f0/timbre/split)
  captions.json            [{video, reference, caption}] (constant, pitch-free caption)

SMOKE FIRST:
    uv run --group analysis python scripts/build_pitch_gate_dataset.py --smoke 10
Full:
    uv run --group analysis python scripts/build_pitch_gate_dataset.py --n 300
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import pitch_gate_data as P

SR = 16_000           # audio VAE convention
RES = 256             # re-render resolution (fits 4090, dodges the 512-token wall)
FPS = 25
LEVELS = [90.0, 120.0, 150.0, 190.0, 240.0]
DEFAULT_SOURCE = "data/noakraicer_ID-LoRA-CelebVHQ/train"


def _voiced_f0(audio: np.ndarray) -> float:
    """Median voiced F0 (Hz) via librosa.yin — for real speech (not the FFT-peak
    estimator, which is for clean synthetic tones)."""
    import librosa

    f0 = librosa.yin(audio, fmin=70, fmax=400, sr=SR)
    return float(np.median(f0))


def _load_audio(mp4: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(mp4), sr=SR, mono=True)
    return y.astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    import soundfile as sf

    sf.write(str(path), audio, sr)


def _render_target(src_mp4: Path, pitched_audio: np.ndarray, out_mp4: Path) -> None:
    """Scale source video to RES, replace its audio with the pitched track, mux."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_wav = Path(tf.name)
    try:
        _write_wav(tmp_wav, pitched_audio)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src_mp4), "-i", str(tmp_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", f"scale={RES}:{RES}", "-r", str(FPS),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(out_mp4),
        ]
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg render failed: {p.stderr.decode()[-300:]}")
    finally:
        tmp_wav.unlink(missing_ok=True)


def _process(src_mp4: Path, natural_f0: float, target_f0: float, timbre: int,
             dur_s: float, out_clip: Path, out_ref: Path) -> float:
    """Pitch-shift the clip's speech to target_f0, synth a voiced-tone reference at the
    shifted audio's ACTUAL F0 (so reference F0 == target's real pitch), render both.
    Returns the actual F0."""
    audio = _load_audio(src_mp4)
    shifted = P.pitch_shift_semitones(audio, SR, P.semitones_to_target(natural_f0, target_f0))
    actual_f0 = _voiced_f0(shifted)
    tone = P.synth_voiced_tone(actual_f0, duration=dur_s, sr=SR, timbre=float(timbre))
    _write_wav(out_ref, tone)
    _render_target(src_mp4, shifted, out_clip)
    return actual_f0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="dir of source mp4s")
    ap.add_argument("--out", default="data/audio_iclora/pitch_ref_gate_v1")
    ap.add_argument("--n", type=int, default=300, help="number of source clips to use")
    ap.add_argument("--ref-seconds", type=float, default=2.0, help="reference tone duration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", type=int, default=0, help="process N clips + validate, then stop")
    args = ap.parse_args()

    n = args.smoke or args.n
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "references").mkdir(parents=True, exist_ok=True)

    srcs = sorted(Path(args.source).glob("*.mp4"))[:n]
    if not srcs:
        sys.exit(f"no source mp4s under {args.source}")

    # Pass 1: probe natural F0 (so target pitch can be assigned decorrelated from it).
    print(f"probing natural F0 for {len(srcs)} clips...")
    natural = {}
    for src in srcs:
        try:
            natural[src.stem] = _voiced_f0(_load_audio(src))
        except Exception as e:  # noqa: BLE001 - skip unreadable clips, keep going
            print(f"  skip {src.name}: {e}")
    rows_meta = P.build_manifest(natural, LEVELS, seed=args.seed)
    by_id = {src.stem: src for src in srcs}

    # Pass 2: produce media.
    manifest, captions = [], []
    for i, m in enumerate(rows_meta):
        src = by_id[m["clip_id"]]
        clip_rel = f"clips/clip_{i:04d}.mp4"
        ref_rel = f"references/ref_{i:04d}.wav"
        try:
            actual = _process(src, m["natural_f0"], m["target_f0"], m["timbre"],
                              args.ref_seconds, out / clip_rel, out / ref_rel)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] skip {src.name}: {e}")
            continue
        row = {"video": clip_rel, "reference": ref_rel, "caption": P.NEUTRAL_CAPTION,
               "target_f0": m["target_f0"], "actual_f0": round(actual, 1),
               "natural_f0": round(m["natural_f0"], 1), "timbre": m["timbre"],
               "split": m["split"]}
        manifest.append(row)
        captions.append({"video": clip_rel, "reference": ref_rel, "caption": P.NEUTRAL_CAPTION})
        print(f"  [{i}] {src.name[:24]:24} natural={m['natural_f0']:.0f} "
              f"target={m['target_f0']:.0f} actual={actual:.0f}Hz {m['split']}")

    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (out / "captions.json").write_text(json.dumps(captions, indent=2))
    print(f"\nwrote {len(manifest)} pairs to {out}/")

    if args.smoke:
        _validate(manifest, out)


def _validate(manifest: list[dict], out: Path) -> None:
    """Smoke validation: reference tone F0 lands on actual_f0; durations sane; the
    leak invariants hold on what was actually written."""
    import librosa
    import soundfile as sf

    print("\n=== smoke validation ===")
    fails = 0
    for r in manifest[: min(5, len(manifest))]:
        ref, _ = sf.read(str(out / r["reference"]))
        ref_f0 = P.dominant_f0(np.asarray(ref, dtype=np.float64), SR)
        ok = abs(ref_f0 - r["actual_f0"]) < 15
        fails += not ok
        print(f"  {r['reference']}: tone F0={ref_f0:.0f} vs actual={r['actual_f0']:.0f} "
              f"{'OK' if ok else 'OFF'}")
    # leak invariants on the written manifest
    tgt = np.array([r["target_f0"] for r in manifest])
    nat = np.array([r["natural_f0"] for r in manifest])
    if len(manifest) > 2:
        rcorr = float(np.corrcoef(nat, tgt)[0, 1])
        print(f"  corr(natural, target) = {rcorr:+.2f} {'OK' if abs(rcorr) < 0.4 else 'LEAK'}")
    caps = {r["caption"] for r in manifest}
    print(f"  captions constant: {'OK' if len(caps) == 1 else 'FAIL'} ({len(caps)} unique)")
    print("SMOKE OK" if not fails else f"SMOKE: {fails} F0 mismatches — inspect")


if __name__ == "__main__":
    main()
