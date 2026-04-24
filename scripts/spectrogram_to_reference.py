"""Render a Mel spectrogram of an audio file as a PNG frame sequence.

Phase 2.0 PoC for IC-LoRA spectrogram-as-reference. Design doc:
`internal/design/spectrogram_reference_design.md`.

The output PNG sequence is intended to feed `LTXAddVideoICLoRAGuide.image`
in a Phase 0b-shaped IC-LoRA workflow (via `LoadImage` + `ImageBatch` in
ComfyUI). Hypothesis: a spectrogram-derived structural reference gives
the LTX 2.3 video beat-locked visual rhythm that a frozen-audio workflow
currently can't achieve through conditioning alone.

Pure-function core (`compute_mel_log`, `prepare_mel_for_render`,
`render_frame`, `render_spectrogram_frame`, `frame_count_for`,
`time_bin_for_frame`) is tested by `tests/test_spectrogram_lib.py`.

Usage:
    uv run --group analysis python scripts/spectrogram_to_reference.py \\
        --audio /path/to/song.wav

    # Sweep render modes (Phase 2.1):
    uv run --group analysis python scripts/spectrogram_to_reference.py \\
        --audio /path/to/song.wav --mode blurred --blur-sigma 3.0
    uv run --group analysis python scripts/spectrogram_to_reference.py \\
        --audio /path/to/song.wav --mode edge_detected
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Literal

import librosa
import numpy as np
import orjson
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflow_utils import timestamped_run_dir


RenderMode = Literal["raw", "normalized", "blurred", "edge_detected"]
VALID_MODES: tuple[RenderMode, ...] = ("raw", "normalized", "blurred", "edge_detected")

LTX_TEMPORAL_SCALE = 8  # matches nodes.py; offline script intentionally duplicates to avoid importing ComfyUI
LTX_RESOLUTION_DIVISOR = 32  # single-stage constraint; matches scripts/validate_workflow_resolution.py DIV_PERMISSIVE


# ---------------------------------------------------------------------------
# Pure-function core (tested)
# ---------------------------------------------------------------------------


def compute_mel_log(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 128,
    hop_length: int = 512,
    fmin: float = 40.0,
    fmax: float | None = None,
    log_scale: bool = True,
) -> np.ndarray:
    """Compute a Mel spectrogram. Returns shape (n_mels, time_bins).

    Distinct from `scripts/analyze_audio_features.py::compute_mel_spectrogram`
    (which is fixed-log, fewer knobs); this variant exposes hop_length /
    fmin / fmax / log_scale for Phase 2.1 render-mode sweeps.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length,
        fmin=fmin, fmax=fmax if fmax is not None else sr / 2,
    )
    if log_scale:
        mel = librosa.power_to_db(mel, ref=np.max)
    return mel


def _sobel_magnitude(arr: np.ndarray) -> np.ndarray:
    gx = sobel(arr, axis=1)
    gy = sobel(arr, axis=0)
    return np.hypot(gx, gy)


# Per-mode preprocessing. Runs once on the full mel before the frame loop.
_PREPROCESS: dict[str, Callable[..., np.ndarray]] = {
    "raw":           lambda mel, **_: mel.astype(np.float32),
    "normalized":    lambda mel, **_: mel.astype(np.float32),
    "blurred":       lambda mel, blur_sigma=1.5, **_: gaussian_filter(mel.astype(np.float32), sigma=blur_sigma),
    "edge_detected": lambda mel, **_: _sobel_magnitude(mel.astype(np.float32)),
}

# Per-mode target brightness range. natural-image-centered [0.2, 0.8] reduces
# OOD gap to the IC-LoRA's VAE (trained on photos); raw/edge intentionally use
# full [0, 1] so high-gradient structure stays prominent for A/B comparison.
_TARGET_RANGE: dict[str, tuple[float, float]] = {
    "raw": (0.0, 1.0),
    "normalized": (0.2, 0.8),
    "blurred": (0.2, 0.8),
    "edge_detected": (0.0, 1.0),
}


def normalize_to_range(arr: np.ndarray, target_range: tuple[float, float]) -> np.ndarray:
    """Min-max rescale arr to target_range. Silent-input guard returns
    midpoint when range is degenerate."""
    low, high = target_range
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max - arr_min < 1e-9:
        return np.full_like(arr, (low + high) / 2.0, dtype=np.float32)
    unit = (arr.astype(np.float32) - arr_min) / (arr_max - arr_min)
    return unit * (high - low) + low


def prepare_mel_for_render(
    mel: np.ndarray,
    mode: RenderMode = "normalized",
    blur_sigma: float = 1.5,
) -> np.ndarray:
    """Preprocess + GLOBALLY normalize the full mel once. Returns a
    uint8 array flipped so low frequencies are at the bottom.

    Global normalization (vs. per-frame) is the correctness choice:
    quiet passages stay dim, loud beats stay bright. Per-frame
    normalization would wash out the beat-amplitude signal — the exact
    variable Phase 2 is testing.
    """
    preprocess = _PREPROCESS[mode]  # KeyError on unknown mode is the natural error
    preprocessed = preprocess(mel, blur_sigma=blur_sigma)
    normalized = normalize_to_range(preprocessed, target_range=_TARGET_RANGE[mode])
    return (np.flipud(normalized) * 255.0).astype(np.uint8)


def render_frame(
    prepared: np.ndarray,
    time_idx: int,
    window_bins: int,
    resolution: tuple[int, int],
) -> np.ndarray:
    """Slice a prepared mel (preprocessed + globally normalized, uint8)
    and resize to `resolution` as (H, W, 3) RGB.

    Sliding window: frame at time_idx shows mel bins
    [time_idx - window_bins, time_idx]. Left-pad with dark when early.
    """
    start = max(0, time_idx - window_bins)
    slice_ = prepared[:, start:time_idx + 1]
    if slice_.shape[1] == 0:
        slice_ = prepared[:, :1]
    if slice_.shape[1] < window_bins + 1:
        pad_width = window_bins + 1 - slice_.shape[1]
        slice_ = np.pad(slice_, ((0, 0), (pad_width, 0)), mode="constant", constant_values=0)

    h, w = resolution
    pil = Image.fromarray(slice_).resize((w, h), resample=Image.Resampling.BILINEAR)
    gray = np.asarray(pil, dtype=np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def render_spectrogram_frame(
    mel: np.ndarray,
    time_idx: int,
    window_bins: int,
    resolution: tuple[int, int],
    mode: RenderMode = "normalized",
    blur_sigma: float = 1.5,
) -> np.ndarray:
    """Test-facing convenience: prepare + render in one call. Production
    code (the CLI render loop) calls `prepare_mel_for_render` once and
    `render_frame` per frame to avoid re-preprocessing the full mel."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown render mode {mode!r}. Valid: {VALID_MODES}")
    prepared = prepare_mel_for_render(mel, mode=mode, blur_sigma=blur_sigma)
    return render_frame(prepared, time_idx, window_bins, resolution)


def frame_count_for(
    duration_seconds: float,
    fps: float,
    align_ltx_latent: bool = False,
) -> int:
    """Number of video frames for a given duration at a given fps.

    `align_ltx_latent=True` rounds UP to the nearest valid LTX 2.3
    pixel-frame count satisfying `(n-1) % 8 == 0`, so the VAE encoder
    produces a latent length that `LTXAddVideoICLoRAGuide` can consume
    without tripping the `latent_idx + guide_latent.shape[2] <= latent_length`
    assertion.
    """
    naive = int(duration_seconds * fps)
    if not align_ltx_latent:
        return naive
    remainder = (naive - 1) % LTX_TEMPORAL_SCALE
    if remainder == 0:
        return naive
    return naive + (LTX_TEMPORAL_SCALE - remainder)


def time_bin_for_frame(frame_idx: int, fps: float, sr: int, hop_length: int) -> int:
    """Map a video frame index to the corresponding mel time-bin index."""
    return int(round(frame_idx / fps * sr / hop_length))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _render_run(cfg: argparse.Namespace) -> Path:
    audio_path = Path(cfg.audio).expanduser().resolve()
    print(f"Loading audio: {audio_path}")
    # librosa.load offset+duration decodes only the requested segment --
    # skips the intro without loading the full file into memory.
    audio, sr_out = librosa.load(
        str(audio_path), sr=cfg.sr, mono=True,
        offset=cfg.start, duration=cfg.duration,
    )
    sr = int(sr_out)
    duration = len(audio) / sr
    print(f"  start={cfg.start}s, duration={duration:.2f}s, sr={sr}")

    print(f"Computing Mel spectrogram: n_mels={cfg.n_mels}, hop={cfg.hop_length}, log={cfg.log_scale}")
    mel = compute_mel_log(
        audio, sr=sr, n_mels=cfg.n_mels, hop_length=cfg.hop_length,
        fmin=cfg.fmin, fmax=cfg.fmax, log_scale=cfg.log_scale,
    )
    prepared = prepare_mel_for_render(mel, mode=cfg.mode, blur_sigma=cfg.blur_sigma)
    print(f"  mel shape: {mel.shape}; prepared min/max: {prepared.min()}/{prepared.max()}")

    total_frames = frame_count_for(
        duration_seconds=duration, fps=cfg.fps, align_ltx_latent=cfg.align_ltx_latent,
    )
    window_bins = int(round(cfg.window_seconds * sr / cfg.hop_length))
    print(f"  total frames: {total_frames} (fps={cfg.fps}, align_ltx={cfg.align_ltx_latent})")
    print(f"  sliding window: {cfg.window_seconds}s = {window_bins} mel bins")

    run_dir = timestamped_run_dir(Path(cfg.output_dir))
    print(f"\nRendering frames -> {run_dir}")

    for frame_idx in range(total_frames):
        time_idx = min(time_bin_for_frame(frame_idx, fps=cfg.fps, sr=sr, hop_length=cfg.hop_length), mel.shape[1] - 1)
        frame = render_frame(prepared, time_idx=time_idx, window_bins=window_bins,
                             resolution=(cfg.resolution_h, cfg.resolution_w))
        Image.fromarray(frame).save(run_dir / f"frame_{frame_idx:05d}.png")
        if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
            print(f"  frame {frame_idx + 1}/{total_frames}")

    video_path: Path | None = None
    if cfg.emit_video:
        video_path = _emit_video(run_dir, fps=cfg.fps)
        print(f"  emitted video: {video_path}")

    metadata = {
        **{k: v for k, v in vars(cfg).items() if k != "audio"},
        "audio_path": str(audio_path),
        "output_dir": str(run_dir),
        "duration_seconds": duration,
        "total_frames": total_frames,
        "sr_effective": sr,
        "video_path": str(video_path) if video_path else None,
    }
    (run_dir / "metadata.json").write_bytes(
        orjson.dumps(metadata, option=orjson.OPT_INDENT_2),
    )
    (run_dir / "README.txt").write_text(_wiring_instructions(metadata))
    print(f"\nDone. Next steps in {run_dir / 'README.txt'}")
    return run_dir


def _emit_video(frames_dir: Path, fps: float) -> Path:
    """Stitch frame_*.png sequence into a single mp4 via ffmpeg. Lets the
    user feed a single file into ComfyUI's LoadVideo instead of wiring
    N LoadImage nodes. Near-lossless (crf=18) since we want the reference
    to survive VAE encoding intact."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH; install it or re-run without --emit-video.")
    out_path = frames_dir / "spectrogram.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _wiring_instructions(meta: dict) -> str:
    return f"""Spectrogram PNG sequence -- Phase 2.0 PoC
=========================================

Source audio: {meta['audio_path']}
Render mode:  {meta['mode']}
Total frames: {meta['total_frames']} at {meta['fps']} fps
Resolution:   {meta['resolution_h']} x {meta['resolution_w']}

To A/B this against a canonical render:

1. In ComfyUI, open your IC-LoRA workflow (Phase 0b or later).

2. Replace the `LTXAddVideoICLoRAGuide.image` input source with a
   LoadImage + ImageBatch chain loading these PNGs:
     - Use the VHS LoadImages node (sorted by filename) OR chain N
       LoadImage nodes into a single ImageBatch via
       `KJNodes.ImageBatchMulti`.
     - Source directory: {meta['output_dir']}

3. Run the render with identical seed + prompts to your canonical
   baseline.

4. Compare beat-sync qualitatively (does the visual rhythm track the
   song's beats?) and quantitatively via:
     uv run --group analysis python scripts/measure_beat_sync.py \\
         --audio "{meta['audio_path']}" --video /path/to/output.mp4

Full design + iteration plan:
  internal/design/spectrogram_reference_design.md
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--audio", required=True, help="Path to input audio file.")
    ap.add_argument(
        "--output-dir",
        default="internal/scratch/spectrogram_runs",
        help="Parent directory for timestamped run output (default: %(default)s).",
    )
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--sr", type=int, default=22050, help="Resample to this rate.")
    ap.add_argument("--n-mels", type=int, default=128)
    ap.add_argument("--hop-length", type=int, default=512)
    ap.add_argument("--fmin", type=float, default=40.0)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--log-scale", action="store_true", default=True)
    ap.add_argument("--no-log-scale", dest="log_scale", action="store_false")
    ap.add_argument("--resolution-h", type=int, default=448,
                    help=f"Output height. Must be div by {LTX_RESOLUTION_DIVISOR} for LTX 2.3.")
    ap.add_argument("--resolution-w", type=int, default=832,
                    help=f"Output width. Must be div by {LTX_RESOLUTION_DIVISOR} for LTX 2.3.")
    ap.add_argument("--mode", choices=VALID_MODES, default="blurred",
                    help="Render mode (default: blurred -- tamer edge stats).")
    ap.add_argument("--blur-sigma", type=float, default=1.5,
                    help="Gaussian blur sigma for `blurred` mode.")
    ap.add_argument("--window-seconds", type=float, default=2.0,
                    help="Sliding-window width in seconds.")
    ap.add_argument("--align-ltx-latent", action="store_true", default=True,
                    help=f"Round frame count to (n-1) %% {LTX_TEMPORAL_SCALE} == 0 for LTX latent alignment.")
    ap.add_argument("--no-align-ltx-latent", dest="align_ltx_latent", action="store_false")
    ap.add_argument("--start", type=float, default=0.0,
                    help="Skip this many seconds from the start of the audio (e.g. to skip an intro).")
    ap.add_argument("--duration", type=float, default=None,
                    help="Length in seconds to render after --start (default: rest of file).")
    ap.add_argument("--emit-video", action="store_true", default=False,
                    help="After PNG emission, stitch frames into spectrogram.mp4 via ffmpeg.")

    cfg = ap.parse_args()

    for name, val in (("resolution-h", cfg.resolution_h), ("resolution-w", cfg.resolution_w)):
        if val % LTX_RESOLUTION_DIVISOR != 0:
            raise SystemExit(f"--{name}={val} not divisible by {LTX_RESOLUTION_DIVISOR} (LTX 2.3 constraint).")

    _render_run(cfg)


if __name__ == "__main__":
    main()
