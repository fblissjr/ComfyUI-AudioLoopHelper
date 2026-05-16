"""apply_fps_24_default.

Last updated: 2026-05-16

TODO: name is misleading. LTX 2.3's canonical inference fps is 25, not 24.
The 2026-05-15 sweep that flipped widgets to 24 was based on a misread of
the `PipelineParams.frame_rate=24.0` Python-library placeholder in
`coderef/LTX-2/.../constants.py`; that's NOT the canonical inference
value. Lightricks's own ComfyUI-LTXVideo example workflows
(`LTX-2_{I2V,T2V}_{Distilled,Full}_wLora.json`) all set
`LTXVConditioning.frame_rate=25`. V2V is the exception (preserves
source-video fps). Postmortem: `internal/analysis/fps_24_partial_reading_postmortem.md`.

The TARGET_FPS now applies = 25 (canonical inference value). The script
name and module name are kept for git-history continuity; a renamed
`apply_fps_25_default.py` (with WorkflowEditor + history-preserving move)
is a followup. See companion-agent memo in `internal/` for the rename plan.

Symptom this script fixes: drift between fps/frame_rate widgets across a
workflow. `LTXVConditioning.frame_rate` scales the model's temporal
positional embedding at `comfy/ldm/lightricks/av_model.py:866`
(`v_pixel_coords[:, 0] = v_pixel_coords[:, 0] * (1.0 / frame_rate)`).
Mixed fps across nodes in a single workflow produces inconsistent
temporal scaling between initial render and loop iterations.

Fix: flip every fps/frame_rate widget that drives model-side time
coordinates or output playback to 25 (canonical inference fps), across
every shipped workflow under `example_workflows/`. The nodes touched:

  - `LTXVConditioning.frame_rate`                          (load-bearing
        — pos-embed time-axis)
  - `VHS_VideoCombine.frame_rate` + `videopreview...frame_rate`
        (output mux; playback must match generation)
  - `LTXFramePlanner.fps`                                  (dimension SSoT;
        downstream consumers receive this)
  - `AudioLoopController.fps`                              (stride math)
  - `AudioLoopPlanner.fps`                                 (stride math)
  - `TimestampPromptScheduleBatchEncode.frame_rate`        (stamped into
        conditioning metadata; must equal LTXVConditioning's value)
  - `TrimVideoLatentToAudio.frame_rate`                    (F14 latent
        boundary snap; matches output fps)
  - `TrimImageBatchToAudio.frame_rate`                     (F14 image
        residue trim; matches output fps)
  - `LatentTemporalMask.frame_rate`                        (retake mask
        time axis)
  - `LoopConfigValidator.fps`                              (validator's
        fps assertion; must agree with the model's frame_rate to avoid
        spurious WARN reports about length/window mismatch)
  - `LTXVEmptyLatentAudio.frame_rate`                      (audio-latent
        sizing: `num_of_latents_from_frames = ceil((frames / frame_rate)
        * latents_per_second)` at `audio_vae.py:188-189`. If video runs
        at 24fps but this node says 25, the computed audio latent is
        undersized by ~4% relative to the actual video duration. For
        math self-consistency we flip this to 24 too — supersedes the
        prior "leave at 25 per Lightricks's convention" call which was
        an unverified inference.)

Explicitly NOT touched:

  - `LTX2_NAG` — no fps widget (only 4 widgets: scale, alpha, tau, inplace).
  - `audio_vae.py`'s internal `latents_per_second = 25` — this is the
        audio VAE's own emission rate (16kHz/160/4), independent of
        video fps. Don't touch.

Compatibility with other apply scripts:
  - Orthogonal to every other apply script. Touches only fps/frame_rate
    widget values; never changes link topology, node count, or schema.
  - The widget-index for fps differs per node type; the script looks up
    the slot by introspecting widget order against the cached map below
    rather than hand-coding indices.

Usage:
    uv run --group dev python scripts/apply_fps_24_default.py
    uv run --group dev python scripts/apply_fps_24_default.py --revert
    uv run --group dev python scripts/apply_fps_24_default.py --dry-run

Idempotent. Already-25 workflows report "no change (already 25)".
`--revert` restores fps=24 (the intermediate 2026-05-15 sweep value,
retained only for emergency rollback to the just-prior state).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_utils import LTX23_FPS, WorkflowEditor

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "example_workflows"

# Canonical inference fps. LTX 2.3 Lightricks example workflows ship with
# LTXVConditioning.frame_rate=25 across the I2V/T2V Distilled+Full set;
# the 8n+1 video-latent boundary aligns cleanly at 25fps.
TARGET_FPS = LTX23_FPS  # 25
# The just-prior intermediate value (the misread 2026-05-15 sweep). Kept
# as `--revert` target so a rollback returns to the immediately prior
# state, not all the way back to legacy. If you need pre-2026-05-15
# state, run with `--revert` once to land here, then re-author the
# widget manually — there's no canonical "two-steps-back" automation.
LEGACY_FPS = 24

# Per-node fps widget index. Verified against each class's
# `define_schema()` in nodes.py (widget order = inputs list order,
# minus non-widget input slots like AUDIO/CONDITIONING/IMAGE/LATENT).
# Tuples are (node_type, widget_index, current_value_must_be_numeric).
LIST_WIDGET_NODES: dict[str, int] = {
    # LTXVConditioning: only one widget, frame_rate.
    "LTXVConditioning": 0,
    # AudioLoopController inputs:
    #   current_iteration, window_seconds, overlap_seconds, audio*, base_seed, fps
    # (`audio` is a non-widget slot.) Widget order is the 5 scalars in
    # define-order: [current_iteration, window_seconds, overlap_seconds,
    # base_seed, fps]. fps is widget[4].
    "AudioLoopController": 4,
    # AudioLoopPlanner: [window_seconds, overlap_seconds, fps, ...]
    "AudioLoopPlanner": 2,
    # LTXFramePlanner: [width, height, target_seconds, fps]
    "LTXFramePlanner": 3,
    # TimestampPromptScheduleBatchEncode: [schedule_text, default_window,
    #   default_duration, snap_boundaries, frame_rate]
    "TimestampPromptScheduleBatchEncode": 4,
    # TrimVideoLatentToAudio: [frame_rate]
    "TrimVideoLatentToAudio": 0,
    # TrimImageBatchToAudio: [frame_rate]
    "TrimImageBatchToAudio": 0,
    # LatentTemporalMask: [mask_start_seconds, mask_end_seconds, frame_rate, ...]
    "LatentTemporalMask": 2,
    # LTXVEmptyLatentAudio: [frames_number, frame_rate, batch_size]
    "LTXVEmptyLatentAudio": 1,
    # LoopConfigValidator: [window_seconds, overlap_seconds, fps, length, ...]
    "LoopConfigValidator": 2,
    # LTXHeadTrim: widgets are [trim_latent_frames, fps]
    # (images, audio are sockets and don't count toward widgets_values).
    "LTXHeadTrim": 1,
    # LatentSeamZoneMask: widgets are [iteration_count, window_latents,
    # overlap_latents, seam_band_seconds, edge_taper_seconds, fps]
    # (latent is a socket).
    "LatentSeamZoneMask": 5,
    # LTXVAudioVideoMask (KJNodes upstream): widgets are [video_fps,
    # video_start_time, video_end_time, audio_start_time, audio_end_time,
    # max_length, existing_mask_mode] — LATENT inputs (video_latent,
    # audio_latent) are sockets and don't count toward widgets_values.
    # Source: ComfyUI-KJNodes/nodes/ltxv_nodes.py LTXVAudioVideoMask schema.
    # Lives INSIDE the loop subgraph. Builds per-iter noise_mask boundaries
    # in pixel-frame space via `start_time * fps` / `end_time * fps`; an fps
    # mismatch vs the rest of the pipeline slips the boundary by
    # ~(fps_mismatch / actual_fps) per iter, drifting the audio-frozen /
    # video-new boundary at every iter transition.
    "LTXVAudioVideoMask": 0,

    # EXCLUDED — looks like an fps widget by value but isn't:
    # GetImageRangeFromBatch.widget[1] is `num_frames`, not a rate. The
    # audio-loop's ref-video slicer ships with widget[1] = 25 meaning
    # "25 frames per iter," coincidentally matching fps. Do NOT add this
    # to LIST_WIDGET_NODES — the apply script would flip 25 → 24 and
    # silently shorten the ref-video window by one frame per iter.
}

# VHS_VideoCombine stores its widgets as a dict, not a list. Two slots
# carry fps: the top-level `frame_rate` and the nested
# `videopreview.params.frame_rate` used for the in-UI preview.
VHS_NODE_TYPE = "VHS_VideoCombine"


def _flip_list_widget(
    node: dict, widget_idx: int, target: int, legacy: int
) -> tuple[int, int] | None:
    """Return (old, new) if a change is needed, None if already at target.

    Raises ValueError if the widget value isn't numeric or isn't one of
    {target, legacy} — protects against silent corruption of unfamiliar
    workflow state.
    """
    wv = node.get("widgets_values") or []
    if widget_idx >= len(wv):
        raise ValueError(
            f"node #{node.get('id')} type={node.get('type')} has only "
            f"{len(wv)} widgets, expected index {widget_idx}"
        )
    cur = wv[widget_idx]
    if not isinstance(cur, (int, float)):
        raise ValueError(
            f"node #{node.get('id')} type={node.get('type')} widget[{widget_idx}] "
            f"is not numeric: {cur!r}"
        )
    if isinstance(cur, float) and not cur.is_integer():
        raise ValueError(
            f"node #{node.get('id')} type={node.get('type')} widget[{widget_idx}] "
            f"= {cur!r} is non-integer; refuse to silently truncate"
        )
    cur_int = int(cur)
    if cur_int == target:
        return None
    if cur_int != legacy:
        raise ValueError(
            f"node #{node.get('id')} type={node.get('type')} widget[{widget_idx}] "
            f"= {cur!r}, expected {legacy} or {target}"
        )
    return (cur_int, target)


def _flip_vhs(
    node: dict, target: int, legacy: int
) -> list[tuple[str, int, int]]:
    """Return list of (path_description, old, new) for VHS_VideoCombine.

    Touches both `widgets_values.frame_rate` and
    `widgets_values.videopreview.params.frame_rate`. Returns empty list
    if both slots already at target.
    """
    wv = node.get("widgets_values")
    if not isinstance(wv, dict):
        raise ValueError(
            f"node #{node.get('id')} VHS_VideoCombine widgets_values "
            f"is not a dict: {type(wv).__name__}"
        )
    changes: list[tuple[str, int, int]] = []

    cur = wv.get("frame_rate")
    if isinstance(cur, (int, float)):
        cur_int = int(cur)
        if cur_int == legacy:
            changes.append(("frame_rate", cur_int, target))
        elif cur_int != target:
            raise ValueError(
                f"node #{node.get('id')} VHS_VideoCombine frame_rate "
                f"= {cur!r}, expected {legacy} or {target}"
            )

    preview = wv.get("videopreview")
    if isinstance(preview, dict):
        params = preview.get("params")
        if isinstance(params, dict):
            cur = params.get("frame_rate")
            if isinstance(cur, (int, float)):
                cur_int = int(cur)
                if cur_int == legacy:
                    changes.append(
                        ("videopreview.params.frame_rate", cur_int, target)
                    )
                elif cur_int != target:
                    # Benchmark workflows ship preview.params.frame_rate=24
                    # already (preserved as-is). Not an error.
                    pass

    return changes


def _iter_all_nodes(ed: WorkflowEditor):
    """Yield (node_dict, source_descr) for top-level + subgraph nodes."""
    for n in ed.wf.get("nodes", []):
        yield n
    for sg in ed.wf.get("definitions", {}).get("subgraphs", []):
        for n in sg.get("nodes", []):
            yield n


def _apply_one(wf_path: Path, revert: bool, dry_run: bool) -> str:
    try:
        ed = WorkflowEditor(wf_path)
    except Exception as e:  # noqa: BLE001
        return f"load error: {e}"

    target = LEGACY_FPS if revert else TARGET_FPS
    legacy = TARGET_FPS if revert else LEGACY_FPS

    edit_count = 0
    type_touch_count: dict[str, int] = {}
    any_matching = False

    try:
        for n in _iter_all_nodes(ed):
            t = n.get("type", "")
            if t in LIST_WIDGET_NODES:
                any_matching = True
                widget_idx = LIST_WIDGET_NODES[t]
                result = _flip_list_widget(n, widget_idx, target, legacy)
                if result is not None:
                    _, new = result
                    n["widgets_values"][widget_idx] = new
                    edit_count += 1
                    type_touch_count[t] = type_touch_count.get(t, 0) + 1
            elif t == VHS_NODE_TYPE:
                any_matching = True
                for path, _old, new in _flip_vhs(n, target, legacy):
                    wv = n["widgets_values"]
                    if path == "frame_rate":
                        wv["frame_rate"] = new
                    elif path == "videopreview.params.frame_rate":
                        wv["videopreview"]["params"]["frame_rate"] = new
                    edit_count += 1
                    type_touch_count[t] = type_touch_count.get(t, 0) + 1
    except ValueError as e:
        return f"skip ({e})"

    if not any_matching:
        return "skip (no matching nodes)"
    if edit_count == 0:
        return f"no change (already {target})"

    type_summary = ", ".join(
        f"{n} {t}" for t, n in sorted(type_touch_count.items())
    )
    if dry_run:
        verb = "would revert" if revert else "would update"
        return f"{verb} (#{edit_count} edits: {type_summary})"

    ed.save(wf_path)
    verb = "reverted" if revert else "updated"
    return f"{verb} (#{edit_count} edits rate={legacy}->{target}: {type_summary})"


def apply(revert: bool, dry_run: bool) -> int:
    action = (
        f"Would {'revert' if revert else 'apply'}"
        if dry_run
        else ("Reverting" if revert else "Applying")
    )
    print(f"{action} fps_24_default across example_workflows/...")
    fail = 0
    for wf_path in sorted(WORKFLOWS_DIR.rglob("*.json")):
        status = _apply_one(wf_path, revert, dry_run)
        rel = wf_path.relative_to(WORKFLOWS_DIR)
        print(f"  {rel}: {status}")
        if status.startswith("load error") or status.startswith("internal error"):
            fail += 1
    return 1 if fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--revert", action="store_true",
        help="Undo the change applied by this script.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Report what WOULD change without writing files.",
    )
    args = ap.parse_args()
    return apply(revert=args.revert, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
