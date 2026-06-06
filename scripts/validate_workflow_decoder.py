"""Check decoder configuration in each loop-family workflow.

Validates that each workflow is in one of three acceptable states:

  1. LTXVSpatioTemporalTiledVAEDecode (preferred — temporal chunking
     bounds decode RAM; removes the song-length ceiling) with chunk
     stride EXACTLY equal to the loop iteration stride in latent frames:

         temporal_tile_length − temporal_overlap == new_latent_frames

     so decode-chunk seams land on iteration boundaries. Also expects
     the cpu/float16 accumulator widgets (the "auto"/"auto" default is
     a full-video fp32 buffer — the decode-buffer-stack OOM class; see
     docs/reference/benchmarking_memory_pressure.md). Emits OK.

  2. LTXVTiledVAEDecode (spatial-only — no stride invariant, but
     MONOLITHIC along time: decode RAM scales with song length and
     >=4-min songs at 960x544 kernel-OOM a 128GB box). Emits OK with a
     length-ceiling note.

  3. VAEDecodeTiled with widgets aligned to the iteration stride:

         (temporal_size − temporal_overlap) / fps ≈ stride_seconds

     within 0.1 s. Emits OK.

Anything else emits a WARNING with the expected widget values, so the
user can either run `apply_ltx_decoder.py --spatiotemporal` to get the
structural fix or manually tune widgets to realign the stride.

The iteration stride is derived by TRACING the AudioLoopController's
window_seconds/overlap_seconds input links to their sources
(LTXFramePlanner.actual_seconds / FloatConstant) — the controller's own
widgets_values are stale placeholders in every shipped workflow and
read 19.88/2 regardless of variant. Stride math goes through
nodes._compute_loop_geometry so it is bit-exact with the runtime.

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

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow_utils import WorkflowEditor, has_active_tensor_loop
from loop_geometry import LoopGeometry, _compute_loop_geometry, _snap_frames  # noqa: E402

# AudioLoopController node ID in our example workflows.
_LOOP_CONTROLLER_ID = 1582

# Conventional fps for LTX 2.3; matches the VHS_VideoCombine output
# frame_rate across all example workflows.
_FPS = 25

# How close tile stride and iteration stride need to be (in seconds)
# for the VAEDecodeTiled fallback to be considered aligned. 0.04s (one
# frame at 25fps) is the tightest you can get with integer widgets;
# 0.1s is the practical threshold where drift doesn't re-introduce
# seams over a 3-min video.
_ALIGNMENT_TOLERANCE_S = 0.1


_DECODER_TYPES = ("VAEDecodeTiled", "LTXVTiledVAEDecode", "LTXVSpatioTemporalTiledVAEDecode")


def _loop_family() -> list[Path]:
    """All workflows with an active TensorLoopOpen — the decode-tail OOM
    exposure class. Shared with apply_ltx_decoder.py."""
    out = []
    for pattern in ("example_workflows/*.json", "example_workflows/experimental/*.json"):
        for path in sorted(REPO_ROOT.glob(pattern)):
            try:
                ed = WorkflowEditor(path)
            except Exception:
                continue
            if has_active_tensor_loop(ed):
                out.append(path)
    return out


def _traced_input_value(ed: WorkflowEditor, node: dict, input_name: str) -> float | None:
    """Resolve the runtime value feeding a linked widget-input.

    Follows the input's link to its source node and returns the float
    that source emits. Returns None when the input is unlinked (caller
    falls back to the local widget) or the source type is unknown.
    """
    try:
        slot = WorkflowEditor.find_input_slot(node, input_name)
    except ValueError:
        return None
    link = ed.find_link_to_slot(node["id"], slot)
    if link is None:
        return None
    src = ed.find_node(link[1])
    src_slot = link[2]
    widgets = src.get("widgets_values") or []
    try:
        if src.get("type") == "LTXFramePlanner" and src_slot == 3:
            # Output slot 3 is actual_seconds — the SNAPPED window, not
            # the raw target widget (target 15s -> 369 frames = 14.76s).
            # Widgets: [target_width, target_height, target_seconds, fps].
            _, actual = _snap_frames(float(widgets[2]), int(widgets[3]))
            return actual
        if src.get("type") in ("FloatConstant", "INTConstant"):
            return float(widgets[0])
    except (TypeError, ValueError, IndexError):
        return None
    return None


def _widget_window_and_overlap(node: dict) -> tuple[float, float] | None:
    """Fallback: read window/overlap from the controller's own widgets.

    Widget order per nodes.py: [current_iteration, window_seconds,
    overlap_seconds, base_seed, fps]. current_iteration is a widget-input
    (linked from TensorLoopOpen) and may or may not appear in
    widgets_values depending on ComfyUI's serialization, so we try both
    4- and 5-entry layouts and gate on a sanity range.
    """
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


def _get_window_and_overlap(ed: WorkflowEditor) -> tuple[float, float] | None:
    """Pull the runtime window_seconds and overlap_seconds for the loop.

    Traces the AudioLoopController's input LINKS first — in every shipped
    workflow window_seconds comes from LTXFramePlanner.actual_seconds and
    overlap_seconds from a FloatConstant, while the controller's own
    widgets are stale placeholders. Falls back to local widgets per-value
    when an input is unlinked (e.g. the fml2v variant's window).
    """
    controllers = ed.find_nodes_by_type("AudioLoopController")
    if not controllers:
        return None
    node = next((n for n in controllers if n.get("id") == _LOOP_CONTROLLER_ID), controllers[0])

    window = _traced_input_value(ed, node, "window_seconds")
    overlap = _traced_input_value(ed, node, "overlap_seconds")
    if window is None or overlap is None:
        fallback = _widget_window_and_overlap(node)
        if fallback is not None:
            window = fallback[0] if window is None else window
            overlap = fallback[1] if overlap is None else overlap
    if window is None or overlap is None:
        return None
    if not (0.0 <= overlap < window):
        return None
    return window, overlap


def _expected_stride_widgets(iter_stride_s: float) -> tuple[int, int]:
    """Return (temporal_size, temporal_overlap) producing a VAEDecodeTiled
    tile stride that matches `iter_stride_s` at 25 fps.

    Picks temporal_overlap = temporal_size // 8 (within the ≤ temporal_size/4
    ComfyUI constraint) for a reasonable blend region.
    """
    # Target: (ts - to) / 25 == iter_stride_s  →  ts - to == round(iter_stride_s * 25)
    target_delta = max(16, round(iter_stride_s * _FPS))
    # Pick temporal_overlap = target_delta // 7 (gives ~1/8 blend region), rounded to 4
    temporal_overlap = max(8, (target_delta // 7) & ~3)
    temporal_size = target_delta + temporal_overlap
    return temporal_size, temporal_overlap


def _validate_node(node: dict, g: LoopGeometry) -> tuple[bool, str]:
    """Return (ok, message) for one decoder node against the loop geometry."""
    node_type = node.get("type")
    node_id = node.get("id")
    title = node.get("title") or f"node {node_id}"

    if node_type == "LTXVTiledVAEDecode":
        return True, (
            f"  OK {title} ({node_type}) — spatial-only tiling, no stride "
            f"concern; NOTE monolithic temporal decode (>=4-min songs at "
            f"960x544 OOM a 128GB box — run apply_ltx_decoder.py --spatiotemporal)"
        )

    if node_type == "LTXVSpatioTemporalTiledVAEDecode":
        widgets = node.get("widgets_values") or []
        if len(widgets) < 7:
            return False, (
                f"  ⚠ {title} ({node_type}) — has {len(widgets)} widget values, "
                f"expected 7 [spatial_tiles, spatial_overlap, temporal_tile_length, "
                f"temporal_overlap, last_frame_fix, working_device, working_dtype]"
            )
        t_len, t_overlap = int(widgets[2]), int(widgets[3])
        device, dtype = widgets[5], widgets[6]
        problems = []
        if t_len - t_overlap != g.new_latent_frames:
            expected_to = min(8, max(1, g.overlap_latent_frames))
            problems.append(
                f"chunk stride {t_len - t_overlap} latents != iteration stride "
                f"{g.new_latent_frames} latents — set [temporal_tile_length, "
                f"temporal_overlap] = [{g.new_latent_frames + expected_to}, {expected_to}]"
            )
        if (device, dtype) != ("cpu", "float16"):
            problems.append(
                f"accumulator [{device!r}, {dtype!r}] — use ['cpu', 'float16'] "
                f"(auto/auto pre-allocates a full-video fp32 buffer; the "
                f"decode-buffer-stack OOM class)"
            )
        if problems:
            return False, f"  ⚠ {title} ({node_type}) — " + "; ".join(problems)
        return True, (
            f"  OK {title} ({node_type}) — chunk stride {t_len - t_overlap} latents "
            f"== iteration stride ({g.stride_seconds:.2f}s), cpu/float16 accumulator"
        )

    if node_type == "VAEDecodeTiled":
        widgets = node.get("widgets_values") or []
        if len(widgets) < 4:
            return False, (
                f"  ⚠ {title} ({node_type}) — has {len(widgets)} widget values, "
                f"expected 4 [tile_size, overlap, temporal_size, temporal_overlap]"
            )
        _, _, t_size, t_overlap = widgets[:4]
        tile_stride_s = (int(t_size) - int(t_overlap)) / _FPS
        delta = abs(tile_stride_s - g.stride_seconds)
        if delta <= _ALIGNMENT_TOLERANCE_S:
            return True, (
                f"  OK {title} ({node_type}) — tile stride {tile_stride_s:.2f}s "
                f"aligned with iter stride {g.stride_seconds:.2f}s "
                f"(Δ {delta:.3f}s ≤ {_ALIGNMENT_TOLERANCE_S}s)"
            )
        expected_ts, expected_to = _expected_stride_widgets(g.stride_seconds)
        return False, (
            f"  ⚠ {title} ({node_type}) — tile stride {tile_stride_s:.2f}s "
            f"DRIFT from iter stride {g.stride_seconds:.2f}s (Δ {delta:.3f}s). "
            f"Either:\n"
            f"      - Run `uv run python scripts/apply_ltx_decoder.py "
            f"--spatiotemporal` (recommended), OR\n"
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
        print(f"  ⚠ could not resolve AudioLoopController window/overlap; skipping stride check")
        return False
    window_s, overlap_s = params
    g = _compute_loop_geometry(window_s, overlap_s, _FPS)
    print(
        f"  AudioLoopController: window={window_s}s, overlap={overlap_s}s, "
        f"iter_stride={g.stride_seconds:.2f}s ({g.new_latent_frames} latents)"
    )

    decoders = [n for t in _DECODER_TYPES for n in ed.find_nodes_by_type(t)]
    if not decoders:
        print(f"  ⚠ no VAE decoder nodes found")
        return False

    all_ok = True
    for node in decoders:
        ok, msg = _validate_node(node, g)
        print(msg)
        if not ok:
            all_ok = False
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--workflow", type=Path, action="append",
        help="Specific workflow to check (defaults to the loop family — "
             "every workflow with an active TensorLoopOpen)",
    )
    args = parser.parse_args()

    workflows = args.workflow if args.workflow else _loop_family()
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
