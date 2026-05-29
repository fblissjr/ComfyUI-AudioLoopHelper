"""F0-measurability filter for the pitch-gate dataset (the one real quality gate).

A pitch-shifted clip is a VALID training pair only if its F0 is cleanly measurable —
that's the target the model learns to match. Clips where the phase vocoder broke the
pitch track (octave flip, breathy garbage) feed a WRONG target, so drop them from the
TRAINING manifest (keep them on disk; eval ignores them too).

Drop rule: remeasured median F0 disagrees with manifest actual_f0 by >30 Hz (the pitch
shift broke → wrong target). Voiced-fraction is REPORTED but NOT a drop gate: real speech
has pauses/consonants/silence, so a perfectly-pitched clip routinely scores low
voiced-frac — a hard voiced-frac<0.4 gate over-rejects ~58 fine pairs whose F0 is accurate
to <15 Hz. F0-accuracy is the principled "is the target measurable" test; voicing density
is not. (--min-voiced-frac defaults to 0.0 = off; raise it only to inspect.)

    uv run --group analysis python scripts/filter_pitch_gate_dataset.py \
        --dataset data/audio_iclora/pitch_ref_gate_v1

Writes manifest_train.jsonl (kept rows) + dropped.json (drop-list for LTX-2) next to
the input manifest. Idempotent; re-running recomputes from manifest.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SR = 16_000


def measure(audio: np.ndarray) -> tuple[float, float]:
    """Return (median voiced F0, voiced-fraction) — fraction of frames whose pitch
    sits within 15% of the median (a stable tone tracks tightly; garbage scatters)."""
    import librosa

    f0 = librosa.yin(audio, fmin=70, fmax=500, sr=SR)
    med = float(np.median(f0))
    frac = float(np.mean(np.abs(f0 - med) < 0.15 * med)) if med > 0 else 0.0
    return med, frac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/audio_iclora/pitch_ref_gate_v1")
    ap.add_argument("--max-f0-error", type=float, default=30.0, help="drop if |remeasured - actual_f0| exceeds this (Hz)")
    ap.add_argument("--min-voiced-frac", type=float, default=0.0,
                    help="drop if voiced-fraction below this (default 0.0 = OFF; speech is naturally sparse-voiced, "
                         "so this over-rejects fine pairs — F0-accuracy is the real gate)")
    args = ap.parse_args()

    import librosa

    root = Path(args.dataset)
    rows = [json.loads(l) for l in (root / "manifest.jsonl").read_text().splitlines()]

    kept, dropped = [], []
    for r in rows:
        y, _ = librosa.load(str(root / r["video"]), sr=SR, mono=True)
        med, frac = measure(y)
        err = abs(med - r["actual_f0"])
        bad_f0 = err > args.max_f0_error
        bad_voiced = frac < args.min_voiced_frac  # default OFF (0.0); reported, not gated
        if bad_f0 or bad_voiced:
            reason = ("f0_drift" if bad_f0 else "") + ("+unvoiced" if bad_voiced else "")
            dropped.append({"video": r["video"], "actual_f0": r["actual_f0"],
                            "remeasured_f0": round(med, 1), "voiced_frac": round(frac, 2),
                            "reason": reason.strip("+")})
        else:
            kept.append(r)

    (root / "manifest_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept))
    (root / "dropped.json").write_text(json.dumps(dropped, indent=2))

    print(f"kept {len(kept)} / {len(rows)}  (dropped {len(dropped)})")
    for d in dropped:
        print(f"  DROP {d['video']}: actual={d['actual_f0']:.0f} remeasured={d['remeasured_f0']:.0f} "
              f"voiced={d['voiced_frac']:.2f} [{d['reason']}]")
    ksplit = {s: sum(r["split"] == s for r in kept) for s in ("train", "heldout")}
    print(f"kept split: {ksplit}")
    print(f"-> {root/'manifest_train.jsonl'} (training)  +  {root/'dropped.json'} (drop-list for LTX-2)")


if __name__ == "__main__":
    main()
