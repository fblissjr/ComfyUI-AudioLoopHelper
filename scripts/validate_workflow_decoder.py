"""Check decoder configuration in each example workflow.

Validates that each workflow is in one of two acceptable states:

  1. LTXVTiledVAEDecode (preferred — no stride-alignment invariant to
     maintain, no temporal tiling possible). Emits OK.

  2. VAEDecodeTiled with widgets aligned to the iteration stride
     derived from AudioLoopController.overlap_seconds. Specifically:

         (temporal_size − temporal_overlap) / fps
             ≈ window_seconds − overlap_seconds

     within 0.1 s. Emits OK.

Anything else emits a WARNING with the expected widget values, so the
user can either run `apply_ltx_decoder.py` to get the structural fix or
manually tune widgets to realign the stride.

Check-only — never writes to any file.

Usage:
    uv run python scripts/validate_workflow_decoder.py
    uv run python scripts/validate_workflow_decoder.py --workflow path/to/workflow.json

Exits non-zero if any workflow is misaligned, so this can be wired into
CI.
"""

import argparse
import sys
from pathlib import Path

from workflow_utils import WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOWS = sorted((REPO_ROOT / "example_workflows").glob(
    "audio-loop-music-video_*.json"
))

# AudioLoopController node ID in our example workflows.
_LOOP_CONTROLLER_ID = 1582

# Conventional fps for LTX 2.3 at our window=19.88 settings; matches
# the VHS_VideoCombine output frame_rate across all example workflows.
_FPS = 25.0

# How close tile stride and iteration stride need to be (in seconds)
# for the alignment to be considered good. 0.04s (one frame at 25fps)
# is the tightest you can get with integer widgets; 0.1s is the
# practical threshold where drift doesn't re-introduce seams over a
# 3-min video.
_ALIGNMENT_TOLERANCE_S = 0.1


_DECODER_TYPES = ("VAEDecodeTiled", "LTXVTiledVAEDecode", "LTXVSpatioTemporalTiledVAEDecode")


def _get_window_and_overlap(ed: WorkflowEditor) -> tuple[float, float] | None:
    """Pull window_seconds and overlap_seconds from AudioLoopController.

    Widget order per nodes.py: [current_iteration, window_seconds,
    overlap_seconds, seed, fps]. current_iteration is a widget-input
    (linked from TensorLoopOpen) and may or may not appear in
    widgets_values depending on ComfyUI's serialization, so we try both
    4- and 5-entry layouts and gate on a sanity range.
    """
    controllers = ed.find_nodes_by_type("AudioLoopController")
    if not controllers:
        return None
    node = next((n for n in controllers if n.get("id") == _LOOP_CONTROLLER_ID), controllers[0])
    widgets = node.get("widgets_values")
    if not widgets:
        return None
    for w_start in (1, 0):
        if len(widgets) > w_start + 1:
            try:
                w = float(widgets[w_start])
                o = float(widgets[w_start + 1])
                if 5.0 <= w <= 60.0 and 0.0 <= o <= w - 1.0:
                    return w, o
            except (TypeError, ValueError):
                continue
    return None


def _expected_stride_widgets(iter_stride_s: float) -> tuple[int, int]:
    """Return (temporal_size, temporal_overlap) producing a tile stride
    that matches `iter_stride_s` at 25 fps.

    Picks temporal_overlap = temporal_size // 8 (within the ≤ temporal_size/4
    ComfyUI constraint) for a reasonable blend region.
    """
    # Target: (ts - to) / 25 == iter_stride_s  →  ts - to == round(iter_stride_s * 25)
    target_delta = max(16, round(iter_stride_s * _FPS))
    # Pick temporal_overlap = target_delta // 7 (gives ~1/8 blend region), rounded to 4
    temporal_overlap = max(8, (target_delta // 7) & ~3)
    temporal_size = target_delta + temporal_overlap
    return temporal_size, temporal_overlap


def _validate_node(node: dict, iter_stride_s: float) -> tuple[bool, str]:
    """Return (ok, message) for one decoder node."""
    node_type = node.get("type")
    node_id = node.get("id")
    title = node.get("title") or f"node {node_id}"

    if node_type == "LTXVTiledVAEDecode":
        return True, f"  OK {title} ({node_type}) — spatial-only tiling, no stride concern"

    if node_type == "LTXVSpatioTemporalTiledVAEDecode":
        return True, f"  OK {title} ({node_type}) — LTX-tuned temporal blending"

    if node_type == "VAEDecodeTiled":
        widgets = node.get("widgets_values") or []
        if len(widgets) < 4:
            return False, (
                f"  ⚠ {title} ({node_type}) — has {len(widgets)} widget values, "
                f"expected 4 [tile_size, overlap, temporal_size, temporal_overlap]"
            )
        _, _, t_size, t_overlap = widgets[:4]
        tile_stride_s = (int(t_size) - int(t_overlap)) / _FPS
        delta = abs(tile_stride_s - iter_stride_s)
        if delta <= _ALIGNMENT_TOLERANCE_S:
            return True, (
                f"  OK {title} ({node_type}) — tile stride {tile_stride_s:.2f}s "
                f"aligned with iter stride {iter_stride_s:.2f}s "
                f"(Δ {delta:.3f}s ≤ {_ALIGNMENT_TOLERANCE_S}s)"
            )
        expected_ts, expected_to = _expected_stride_widgets(iter_stride_s)
        return False, (
            f"  ⚠ {title} ({node_type}) — tile stride {tile_stride_s:.2f}s "
            f"DRIFT from iter stride {iter_stride_s:.2f}s (Δ {delta:.3f}s). "
            f"Either:\n"
            f"      - Run `uv run python scripts/apply_ltx_decoder.py` to swap "
            f"to LTXVTiledVAEDecode (recommended), OR\n"
            f"      - Set widgets to [512, 64, {expected_ts}, {expected_to}] "
            f"(tile stride {(expected_ts - expected_to)/_FPS:.2f}s)"
        )

    return False, f"  ⚠ {title} — unknown decoder type {node_type!r}"


def validate_workflow(path: Path) -> bool:
    """Returns True if the workflow's decoder configuration is OK."""
    ed = WorkflowEditor(path)
    print(f"=== {path.name} ===")

    params = _get_window_and_overlap(ed)
    if params is None:
        print(f"  ⚠ could not find AudioLoopController widget values; skipping stride check")
        return False
    window_s, overlap_s = params
    iter_stride_s = window_s - overlap_s
    print(
        f"  AudioLoopController: window={window_s}s, overlap={overlap_s}s, "
        f"iter_stride={iter_stride_s:.2f}s"
    )

    decoders = [n for t in _DECODER_TYPES for n in ed.find_nodes_by_type(t)]
    if not decoders:
        print(f"  ⚠ no VAE decoder nodes found")
        return False

    all_ok = True
    for node in decoders:
        ok, msg = _validate_node(node, iter_stride_s)
        print(msg)
        if not ok:
            all_ok = False
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--workflow", type=Path, action="append",
        help="Specific workflow to check (defaults to all audio-loop-music-video_*.json)",
    )
    args = parser.parse_args()

    workflows = args.workflow if args.workflow else DEFAULT_WORKFLOWS
    if not workflows:
        print(f"No workflows found under {REPO_ROOT / 'example_workflows'}")
        return 1

    results = [validate_workflow(p) for p in workflows]
    print()
    if all(results):
        print(f"All {len(results)} workflow(s) pass decoder validation.")
        return 0
    failing = sum(1 for ok in results if not ok)
    print(f"{failing}/{len(results)} workflow(s) have misaligned decoder configuration.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
