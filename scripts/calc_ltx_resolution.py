"""calc_ltx_resolution — offline companion to the LTXResolutionFromAspect node.

Last updated: 2026-06-03

Resolves a target aspect ratio + long edge to LTX 2.3-valid (W, H) and
reports the resulting latent volume with an informational VRAM advisory
(there is no hard latent-volume ceiling — see
`docs/reference/frame_planner_reference.md` §"Latent-volume
classification"). Use this when picking dimensions for
`EmptyLTXVLatentVideo` widgets without opening ComfyUI.

Exit code is 0 for any LTX-valid resolution (incl. HIGH_VRAM, which is
informational, not an error); a `(frames-1)%8` violation raises.

Examples:
    # 16:9-ish at 832 long edge for 497-frame window (cinema 1.85:1 = 832x448)
    uv run python scripts/calc_ltx_resolution.py --aspect 1.778 --long 832

    # Square 1:1 at 704 long edge
    uv run python scripts/calc_ltx_resolution.py --aspect 1.0 --long 704 --orientation square

    # Higher long edge — reports HIGH_VRAM advisory (still valid, exits 0)
    uv run python scripts/calc_ltx_resolution.py --aspect 1.778 --long 1216

The shared math lives in `nodes._compute_ltx_resolution`; this script
imports it so node + CLI never drift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `from nodes import ...` when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nodes import _compute_ltx_resolution


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--aspect", type=float, default=16 / 9,
        help="Width/height ratio (16:9=1.778, cinema 1.85:1=1.857, 4:3=1.333). "
             "Default: %(default).3f",
    )
    ap.add_argument(
        "--long", dest="long_edge", type=int, default=832,
        help="Target long-edge in pixels. Snapped UP to nearest div-32 boundary. "
             "Default: %(default)s",
    )
    ap.add_argument(
        "--frames", type=int, default=497,
        help="Total frames. Must satisfy (frames-1)%%8==0. Default: %(default)s",
    )
    ap.add_argument(
        "--orientation", choices=["landscape", "portrait", "square"],
        default="landscape",
    )
    args = ap.parse_args()

    width, height, volume, status = _compute_ltx_resolution(
        args.aspect, args.long_edge, args.frames, args.orientation
    )

    print(f"width:         {width}")
    print(f"height:        {height}")
    print(f"frames:        {args.frames}")
    print(f"latent_volume: {volume}")
    print(f"status:        {status}")
    # Any LTX-valid resolution exits 0 — HIGH_VRAM is an informational advisory,
    # not an error (a (frames-1)%8 violation raises before we get here).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
