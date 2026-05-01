"""Align a driving reference video to the audio-loop IC-LoRA workflow.

Probes the audio source + the ref video, computes the required ref length
from workflow parameters (output fps, output resolution, slicer window,
mode), and emits an ffmpeg command (or runs it) to cut + re-encode the
ref so VHS_LoadVideo consumes it cleanly with no resize / fps coercion
inside the graph.

What it computes:

  STATIC mode (default; matches today's shipped iclora workflow):
    Required ref duration = window_frames / fps
    (e.g., 25 frames / 25 fps = 1.0 s)
    Audio length doesn't constrain ref length — every iter sees the
    same window of ref frames.

  SLIDING mode (Phase 2; not yet shipped):
    Effective stride per iter = window_seconds - overlap_seconds
    Iteration count from audio = ceil(audio_duration / effective_stride)
    Required ref duration = iters * effective_stride + window_seconds
    (Audio length DOES constrain ref length.)

What it produces:

  - An ffmpeg command that:
      * Seeks to --source-start (default 0)
      * Takes the required duration (with optional --loop for sliding-
        mode + short refs)
      * Scales to output width x height (default 832x448)
      * Coerces to --fps (default 25)
      * Drops audio (-an; ref doesn't need it)
      * Re-encodes at CRF 18 H.264 yuv420p (LTX VAE-friendly)
  - A matching VHS_LoadVideo widget config to paste in the workflow:
      * force_rate, frame_load_cap, skip_first_frames, select_every_nth

Usage:
    # Probe + print plan, don't run ffmpeg:
    uv run --group dev python scripts/align_ref_video.py \\
        --audio /path/to/song.mp3 \\
        --ref /path/to/source_ref.mp4 \\
        --out /path/to/aligned_ref.mp4

    # Actually run the ffmpeg cut:
    uv run --group dev python scripts/align_ref_video.py ... --execute

    # Sliding-mode pre-cut for a long song with a short source ref:
    uv run --group dev python scripts/align_ref_video.py \\
        --audio long_song.mp3 --ref short_ref.mp4 --out aligned.mp4 \\
        --mode sliding --loop --execute

    # Pull the good 5-second segment starting at 1m30s into a long source:
    uv run --group dev python scripts/align_ref_video.py \\
        --audio song.mp3 --ref source_long.mp4 --out clip.mp4 \\
        --source-start 90 --execute

Defaults match `example_workflows/audio-loop-music-video_latent_iclora.json`.
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import orjson


@dataclass
class MediaInfo:
    path: Path
    duration_s: float
    fps: float | None
    width: int | None
    height: int | None
    codec: str | None


def _ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    return orjson.loads(out)


def _parse_fps(rate: str) -> float | None:
    """Parse '25/1' or '24000/1001' -> float fps."""
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        a, b = rate.split("/", 1)
        try:
            denom = float(b)
            return float(a) / denom if denom != 0 else None
        except ValueError:
            return None
    try:
        return float(rate)
    except ValueError:
        return None


def probe(path: Path) -> MediaInfo:
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    data = _ffprobe_json(path)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    fps = width = height = codec = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and fps is None:
            fps = _parse_fps(stream.get("r_frame_rate") or stream.get("avg_frame_rate") or "")
            width = stream.get("width")
            height = stream.get("height")
            codec = stream.get("codec_name")
            stream_duration = stream.get("duration")
            if duration == 0 and stream_duration:
                try:
                    duration = float(stream_duration)
                except ValueError:
                    pass
        elif stream.get("codec_type") == "audio" and duration == 0:
            stream_duration = stream.get("duration")
            if stream_duration:
                try:
                    duration = float(stream_duration)
                except ValueError:
                    pass
    return MediaInfo(path=path, duration_s=duration, fps=fps,
                     width=width, height=height, codec=codec)


@dataclass
class CutPlan:
    mode: str
    source_start_s: float
    cut_duration_s: float
    cut_frames: int
    needs_loop: bool
    iters_estimate: int
    notes: list[str]


def compute_cut_plan(
    audio: MediaInfo, ref: MediaInfo,
    *, mode: str, fps: float, window_frames: int,
    window_seconds: float, overlap_seconds: float,
    source_start_s: float, allow_loop: bool,
) -> CutPlan:
    notes: list[str] = []
    if mode == "static":
        cut_duration = window_frames / fps
        iters = 1  # not meaningful; static reuse
    elif mode == "sliding":
        effective_stride = window_seconds - overlap_seconds
        if effective_stride <= 0:
            raise SystemExit(
                f"sliding mode needs window_seconds > overlap_seconds, "
                f"got {window_seconds} / {overlap_seconds}"
            )
        iters = max(1, math.ceil(audio.duration_s / effective_stride))
        cut_duration = iters * effective_stride + window_seconds
        notes.append(
            f"sliding: {iters} iters at {effective_stride:.2f}s stride "
            f"+ {window_seconds:.2f}s window = {cut_duration:.2f}s ref needed"
        )
    else:
        raise SystemExit(f"unknown mode {mode!r}; use 'static' or 'sliding'")

    available = max(0.0, ref.duration_s - source_start_s)
    needs_loop = False
    if cut_duration > available:
        if allow_loop and mode == "sliding":
            needs_loop = True
            notes.append(
                f"ref too short ({available:.2f}s available after "
                f"source-start={source_start_s:.2f}s) — looping with "
                f"-stream_loop to fill {cut_duration:.2f}s"
            )
        elif mode == "static":
            notes.append(
                f"WARN: ref too short ({available:.2f}s available, need "
                f"{cut_duration:.2f}s) — output will be truncated; consider "
                f"a longer source or smaller --source-start"
            )
        else:
            raise SystemExit(
                f"sliding mode needs {cut_duration:.2f}s of ref content but "
                f"only {available:.2f}s available after source-start. "
                f"Pass --loop to repeat the ref, or pick a longer source."
            )

    cut_frames = int(round(cut_duration * fps))
    return CutPlan(
        mode=mode,
        source_start_s=source_start_s,
        cut_duration_s=cut_duration,
        cut_frames=cut_frames,
        needs_loop=needs_loop,
        iters_estimate=iters,
        notes=notes,
    )


def build_ffmpeg_cmd(
    ref: MediaInfo, plan: CutPlan, out_path: Path,
    *, fps: float, width: int, height: int, crf: int,
) -> list[str]:
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-y"]

    if plan.needs_loop:
        cmd += ["-stream_loop", "-1"]

    # -ss before -i is fast (keyframe seek) but less frame-accurate.
    # For 25fps + GOP-friendly source this is fine; for frame-precision
    # users should pre-cut with -ss after -i.
    if plan.source_start_s > 0:
        cmd += ["-ss", f"{plan.source_start_s:.3f}"]

    cmd += ["-i", str(ref.path)]
    cmd += ["-t", f"{plan.cut_duration_s:.3f}"]

    vf_parts = []
    if (ref.width, ref.height) != (width, height):
        vf_parts.append(f"scale={width}:{height}")
    # Always coerce fps; cheap if already matches.
    cmd += ["-r", str(fps)]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    cmd += [
        "-an",                                # drop audio (ref doesn't need it)
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",                # LTX VAE-friendly
        "-crf", str(crf),
        "-preset", "slow",
        "-movflags", "+faststart",
        str(out_path),
    ]
    return cmd


def vhs_widget_recommendations(fps: float) -> dict:
    """After this script runs, the output is exactly the right shape.
    Tell the user the matching VHS_LoadVideo widget config."""
    return {
        "force_rate": int(round(fps)),       # match output exactly
        "frame_load_cap": 0,                 # already-cut file; load all
        "skip_first_frames": 0,              # we baked the offset in
        "select_every_nth": 1,
        "format": "LTXV",
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--audio", required=True, type=Path,
                    help="Audio source (mp3/wav/flac).")
    ap.add_argument("--ref", required=True, type=Path,
                    help="Source reference video.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output path for the aligned ref clip.")
    ap.add_argument("--mode", choices=("static", "sliding"), default="static",
                    help="static (today's shipped wiring) or sliding (Phase 2).")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="Output fps (default 25, matches LTXFramePlanner).")
    ap.add_argument("--width", type=int, default=832,
                    help="Output width (default 832).")
    ap.add_argument("--height", type=int, default=448,
                    help="Output height (default 448).")
    ap.add_argument("--window-frames", type=int, default=25,
                    help="Slicer window in frames (static mode; default 25).")
    ap.add_argument("--window-seconds", type=float, default=5.0,
                    help="Loop window in seconds (sliding mode; default 5.0).")
    ap.add_argument("--overlap-seconds", type=float, default=1.0,
                    help="Loop overlap in seconds (sliding mode; default 1.0).")
    ap.add_argument("--source-start", type=float, default=0.0,
                    help="Timestamp in source where the good content starts (s).")
    ap.add_argument("--loop", action="store_true",
                    help="Loop the ref via -stream_loop -1 if too short (sliding only).")
    ap.add_argument("--crf", type=int, default=18,
                    help="x264 CRF (default 18, near-lossless; lower = larger file).")
    ap.add_argument("--execute", action="store_true",
                    help="Run ffmpeg (default: print command only).")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress probe summary.")
    args = ap.parse_args()

    audio = probe(args.audio)
    ref = probe(args.ref)

    if not args.quiet:
        print(f"audio: {audio.path}  duration={audio.duration_s:.2f}s")
        print(f"ref:   {ref.path}  duration={ref.duration_s:.2f}s "
              f"fps={ref.fps}  res={ref.width}x{ref.height}  codec={ref.codec}")

    plan = compute_cut_plan(
        audio, ref,
        mode=args.mode, fps=args.fps,
        window_frames=args.window_frames,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
        source_start_s=args.source_start,
        allow_loop=args.loop,
    )

    print(f"\nplan ({plan.mode}):")
    print(f"  source-start: {plan.source_start_s:.2f}s")
    print(f"  cut duration: {plan.cut_duration_s:.2f}s ({plan.cut_frames} frames at {args.fps:.0f}fps)")
    if plan.mode == "sliding":
        print(f"  iters from audio: ~{plan.iters_estimate}")
    if plan.needs_loop:
        print(f"  looping ref: yes (-stream_loop -1)")
    for note in plan.notes:
        print(f"  note: {note}")

    cmd = build_ffmpeg_cmd(
        ref, plan, args.out,
        fps=args.fps, width=args.width, height=args.height, crf=args.crf,
    )
    print("\nffmpeg command:")
    print("  " + " ".join(shlex.quote(c) for c in cmd))

    widgets = vhs_widget_recommendations(args.fps)
    print("\nVHS_LoadVideo widget config (after this clip is in place):")
    for k, v in widgets.items():
        print(f"  {k}: {v}")
    print(f"  video: {args.out.name}  (drop into <comfyui>/input/)")

    if args.execute:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print("\nrunning...")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"ffmpeg failed (rc={rc})", file=sys.stderr)
            return rc
        print(f"wrote {args.out}")
    else:
        print("\n(dry-run; pass --execute to actually run ffmpeg)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
