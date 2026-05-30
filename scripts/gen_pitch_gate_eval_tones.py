"""Generate held-out voiced-tone reference wavs for the pitch-gate eval.

The eval feeds these tones (one F0 per render) to the audio-reference IC-LoRA and measures
whether the generated speech's F0 tracks the tone (slope output_F0 vs tone_F0, LoRA vs base).
These are the REFERENCE inputs for the LoadAudio node in the eval workflow.

Format matches the training references exactly (so eval-encode == train-encode):
  - voiced tone (harmonic stack, NOT a pure sine) via pitch_gate_data.synth_voiced_tone
  - 2.0 s, 16 kHz mono float32 wav (the trainer's reference format + the "2s is enough" precedent)
  - neutral timbre (timbre=0.0) — eval tones are clean probes, no nuisance variation
F0s span the achieved training band (~110-300 Hz, dense 150-280) — NOT the 76/353 extremes
(too few training neighbors → extrapolation). Default sweep gives well-separated slope points.

    # into the repo (record):
    uv run --group analysis python scripts/gen_pitch_gate_eval_tones.py
    # into ComfyUI's input dir so LoadAudio finds them (pass your input dir):
    uv run --group analysis python scripts/gen_pitch_gate_eval_tones.py --out <comfyui_input_dir>

Also writes the template default `pitch_gate_tone.wav` (= the middle F0) so the eval graph
loads out of the box, plus tones.json mapping filename -> ground-truth F0 for the scorer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pitch_gate_data as P

SR = 16_000
DURATION = 2.0
DEFAULT_F0S = [120.0, 150.0, 185.0, 220.0, 260.0]  # in-distribution, evenly spaced for a slope fit
DEFAULT_OUT = "data/audio_iclora/pitch_ref_gate_v1/eval_tones"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir (pass your ComfyUI input dir to drop them where LoadAudio looks)")
    ap.add_argument("--f0s", type=float, nargs="+", default=DEFAULT_F0S, help="tone F0s in Hz (stay within ~110-300)")
    ap.add_argument("--duration", type=float, default=DURATION)
    args = ap.parse_args()

    import soundfile as sf

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for f0 in args.f0s:
        tone = P.synth_voiced_tone(f0, duration=args.duration, sr=SR, timbre=0.0)
        # independent confirm the tone lands on F0 before we ship it as the ground truth
        measured = P.dominant_f0(tone, SR)
        name = f"pitch_gate_tone_{int(round(f0))}hz.wav"
        sf.write(str(out / name), tone, SR)
        manifest.append({"file": name, "target_f0": f0, "measured_f0": round(measured, 1)})
        flag = "OK" if abs(measured - f0) < 8 else "OFF"
        print(f"  {name}: target {f0:.0f}Hz, measured {measured:.0f}Hz  {flag}")

    # template default = the middle tone, so the eval graph loads out of the box
    mid = sorted(args.f0s)[len(args.f0s) // 2]
    default_tone = P.synth_voiced_tone(mid, duration=args.duration, sr=SR, timbre=0.0)
    sf.write(str(out / "pitch_gate_tone.wav"), default_tone, SR)
    print(f"  pitch_gate_tone.wav (template default = {mid:.0f}Hz)")

    (out / "tones.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} tones + default + tones.json to {out}/")


if __name__ == "__main__":
    main()
