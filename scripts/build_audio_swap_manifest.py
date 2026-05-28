#!/usr/bin/env python3
"""Build the audio-swap eval manifest from a directory of rendered A/B videos.

Convention — matches what RunIdPrefix produces from our generated workflows
(`<output>/lora_eval/<arm>/<BPM>bpm/<timestamp>/<file>.mp4`):

    renders_dir/
      lora/
        50bpm/
          20260528_153022/<file>.mp4
        70bpm/
          20260528_153105/<file>.mp4
          20260528_154210/<file>.mp4   # re-render — script picks the NEWEST
        ...
      baseline/
        50bpm/
          20260528_153044/<file>.mp4
        ...
      (optional) neutral/
        silent_audio/
          20260528_153300/<file>.mp4    # base-preservation, LoRA arm only

For each `<bpm>bpm/` subdir that exists in BOTH lora/ and baseline/, the
NEWEST timestamped render in each is used. So re-rendering one cell with
a tweaked config just drops a new timestamped folder; old runs are ignored
but kept on disk for review.

Usage:

    uv run --group dev python scripts/build_audio_swap_manifest.py <renders_dir>
        [--output <manifest.json>]   # defaults to <renders_dir>/manifest.json

Output schema is what `coderef/LTX-2/packages/ltx-trainer/scripts/run_audio_coupling_eval.py`
expects:

    {"cases": [{"expected": <int>, "lora_video": "...", "baseline_video": "..."}],
     "neutral_cases": [{"lora_video": "..."}]}

Paths are stored RELATIVE to the manifest's parent dir (the eval script
resolves them that way).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


BPM_RE = re.compile(r"^(\d+)bpm\.mp4$")


def _bpm_from_name(p: Path) -> int | None:
    m = BPM_RE.match(p.name)
    return int(m.group(1)) if m else None


def build(renders_dir: Path) -> dict:
    lora_dir = renders_dir / "lora"
    base_dir = renders_dir / "baseline"
    if not lora_dir.is_dir() or not base_dir.is_dir():
        raise SystemExit(
            f"Expected <renders_dir>/lora/ and <renders_dir>/baseline/ subdirs;"
            f" got dir={renders_dir} (lora exists={lora_dir.is_dir()}, baseline={base_dir.is_dir()})"
        )
    def _index(d: Path) -> dict[int, Path]:
        out: dict[int, Path] = {}
        for p in d.glob("*.mp4"):
            bpm = _bpm_from_name(p)
            if bpm is not None:
                out[bpm] = p
        return out
    lora_files = _index(lora_dir)
    base_files = _index(base_dir)
    paired_bpms = sorted(set(lora_files) & set(base_files))
    cases = [
        {
            "expected": bpm,
            "lora_video": str(lora_files[bpm].relative_to(renders_dir)),
            "baseline_video": str(base_files[bpm].relative_to(renders_dir)),
        }
        for bpm in paired_bpms
    ]
    # Optional neutral cases (LoRA arm only — base-preservation check)
    neutral_dir = renders_dir / "neutral"
    neutral_cases = []
    if neutral_dir.is_dir():
        for p in sorted(neutral_dir.glob("*.mp4")):
            neutral_cases.append({"lora_video": str(p.relative_to(renders_dir))})
    return {"cases": cases, "neutral_cases": neutral_cases}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("renders_dir", type=Path, help="dir containing lora/ + baseline/ subdirs of *.mp4")
    ap.add_argument("--output", type=Path, default=None, help="manifest path (default: <renders_dir>/manifest.json)")
    args = ap.parse_args()

    manifest = build(args.renders_dir)
    out = args.output or (args.renders_dir / "manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))

    n_cases = len(manifest["cases"])
    n_neutral = len(manifest["neutral_cases"])
    print(f"wrote {out}: {n_cases} A/B cases, {n_neutral} neutral cases")
    if n_cases == 0:
        print("WARN: no paired BPMs found — check that lora/ and baseline/ have matching <bpm>bpm.mp4 files",
              file=sys.stderr)
        sys.exit(1)
    print()
    print("Run eval with:")
    print(f"  uv run --group dev python coderef/LTX-2/packages/ltx-trainer/scripts/run_audio_coupling_eval.py "
          f"{out} --coupling beat_pulse --res 256x256x25")


if __name__ == "__main__":
    main()
