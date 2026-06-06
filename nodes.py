"""Audio-aware loop helper nodes for ComfyUI.

Provides nodes to automatically manage loop iteration timing against an
audio track, eliminating manual iteration count calculation and preventing
crashes from overshooting audio boundaries.

Built for use alongside ComfyUI-NativeLooping (TensorLoopOpen/Close),
ComfyUI-VideoHelperSuite, ComfyUI-KJNodes, and ComfyUI-MelBandRoFormer
for generating full-length music videos with LTX 2.3.
"""

import gc
import logging
import math
import os
import re
import sys
import warnings
from collections import OrderedDict
from contextlib import nullcontext
from typing import NamedTuple

import torch
from typing_extensions import override

try:
    from comfy_api.latest import ComfyExtension, io
except ImportError:
    # Outside ComfyUI runtime (e.g., pytest). Provide minimal stubs so
    # helper functions and execute() methods remain testable.
    # __getattr__ handles io.Schema, io.Int.Input, etc. used in annotations
    # and define_schema() without enumerating every attribute.
    class _Passthrough:
        """Returns itself for any attribute access or call."""
        def __getattr__(self, _name):
            return _Passthrough()
        def __call__(self, *args, **kwargs):
            return _Passthrough()

    class _IOStub(_Passthrough):
        class ComfyNode:
            pass

        @staticmethod
        def NodeOutput(*args):
            return args

    ComfyExtension = type("ComfyExtension", (), {})
    io = _IOStub()


LTX_TEMPORAL_SCALE = 8  # LTX 2.3 VAE temporal compression factor (pixel_frames // 8 = latent_frames)


# Per-stage maximum reduction ratio kept below lanczos kernel's clean range
# (kernel radius 3 -> ~6 input samples per output pixel). Anything above 2x
# linear reduction in a single pass leaves visible aliasing on faces, fine
# textures, text — content the cross-attention reads as "explorable detail"
# and which manifests downstream as spurious camera motion in i2v renders.
_LANCZOS_MAX_PER_STAGE_RATIO = 2.0


def _compute_resize_stages(
    src_w: int, src_h: int, tgt_w: int, tgt_h: int,
) -> list[tuple[int, int]]:
    """Plan adaptive multi-stage downscaling — kernel-agnostic stage layout.

    For each stage, output dims roughly halve the previous dims (down to
    target). Each stage stays at <= 2x linear reduction. The 2x cap is
    set by the final-stage lanczos kernel's clean anti-alias range
    (kernel radius 3, ~6 samples per output pixel); intermediate stages
    use bicubic+antialias so the bound applies uniformly.

    Returns a list of (width, height) per stage, ending at exactly
    (tgt_w, tgt_h). Empty list = no work needed (source matches target).
    Single-element list = direct one-pass resize (ratio <= 2x or upscale).

    Pure function — easy to unit-test without torch.
    """
    if src_w == tgt_w and src_h == tgt_h:
        return []

    ratio_w = src_w / max(tgt_w, 1)
    ratio_h = src_h / max(tgt_h, 1)
    ratio = max(ratio_w, ratio_h)

    if ratio <= _LANCZOS_MAX_PER_STAGE_RATIO:
        # Either upscale (ratio < 1) or single-pass downscale within the
        # kernel's clean range — no staging benefit.
        return [(tgt_w, tgt_h)]

    # ceil(log2(ratio)) gives the smallest N such that 2^N >= ratio,
    # i.e. N stages of <=2x reduction can reach the target.
    n_stages = max(1, math.ceil(math.log2(ratio)))

    stages: list[tuple[int, int]] = []
    cur_w, cur_h = src_w, src_h
    for i in range(n_stages):
        if i == n_stages - 1:
            stages.append((tgt_w, tgt_h))
        else:
            # Geometric interpolation: each stage shrinks by the same
            # factor so the last stage also lands within 2x of target.
            remaining_stages = n_stages - i
            step_ratio_w = (cur_w / tgt_w) ** (1.0 / remaining_stages)
            step_ratio_h = (cur_h / tgt_h) ** (1.0 / remaining_stages)
            next_w = max(tgt_w, int(round(cur_w / step_ratio_w)))
            next_h = max(tgt_h, int(round(cur_h / step_ratio_h)))
            stages.append((next_w, next_h))
            cur_w, cur_h = next_w, next_h
    return stages


def _crop_to_aspect(
    image: "torch.Tensor",
    target_w: int,
    target_h: int,
    crop_position: str,
) -> "torch.Tensor":
    """Center/edge-crop a [B, H, W, C] IMAGE tensor to match target aspect.

    Mirrors `ImageResizeKJv2(keep_proportion="crop")` semantics: if the
    source aspect ratio differs from the target, crop one axis to match
    so the subsequent resize doesn't distort. No-op if aspects already
    match.
    """
    src_h, src_w = int(image.shape[1]), int(image.shape[2])
    if src_w <= 0 or src_h <= 0 or target_w <= 0 or target_h <= 0:
        return image

    src_aspect = src_w / src_h
    tgt_aspect = target_w / target_h
    if abs(src_aspect - tgt_aspect) < 1e-6:
        return image

    if src_aspect > tgt_aspect:
        # Source is too wide — crop width.
        crop_w = int(round(src_h * tgt_aspect))
        crop_h = src_h
    else:
        # Source is too tall — crop height.
        crop_w = src_w
        crop_h = int(round(src_w / tgt_aspect))

    if crop_position == "center":
        x = (src_w - crop_w) // 2
        y = (src_h - crop_h) // 2
    elif crop_position == "top":
        x = (src_w - crop_w) // 2
        y = 0
    elif crop_position == "bottom":
        x = (src_w - crop_w) // 2
        y = src_h - crop_h
    elif crop_position == "left":
        x = 0
        y = (src_h - crop_h) // 2
    elif crop_position == "right":
        x = src_w - crop_w
        y = (src_h - crop_h) // 2
    else:
        # Defensive: unknown position falls back to center.
        x = (src_w - crop_w) // 2
        y = (src_h - crop_h) // 2

    # View is fine — downstream resize materializes its own contiguous output.
    return image[:, y : y + crop_h, x : x + crop_w, :]


def _resize_bchw(image_bchw: "torch.Tensor", width: int, height: int, *, final_stage: bool) -> "torch.Tensor":
    """Resize [B, C, H, W] image. Avoids float→uint8→float quantization
    loss on intermediate multi-stage passes.

    `comfy.utils.lanczos` is implemented as PIL.Image.LANCZOS — it
    converts float32 → uint8, resizes, converts back to float32. That's
    fine for a single pass (one round of 8-bit quantization), but
    multi-stage stacks the loss: stage 1 quantizes, stage 2 quantizes
    the already-quantized intermediate. The accumulated banding noise
    on real photographs (faces, textures) is exactly the kind of
    high-frequency content LTX 2.3's i2v cross-attention reads as
    "explorable detail" and turns into spurious motion cues.

    Strategy:
      - intermediate stages: `F.interpolate(mode='bicubic', antialias=True)`.
        Stays float32 throughout. Bicubic+antialias is the standard
        downsampling kernel in torchvision/Pillow's modern ANTIALIAS path.
      - final stage: PIL lanczos via `comfy.utils.common_upscale` so
        the final output has the same kernel character as a single-pass
        canonical render.

    `final_stage=True` is also the path for upscale (ratio < 1, single
    stage) — no quantization concern there. Same-size passthrough never
    reaches this function (planner returns an empty stage list).
    """
    if final_stage:
        try:
            from comfy.utils import common_upscale
            return common_upscale(image_bchw, width, height, "lanczos", crop="disabled")
        except ImportError:
            pass
    return torch.nn.functional.interpolate(
        image_bchw, size=(height, width), mode="bicubic",
        align_corners=False, antialias=True,
    )



def _make_cosine_taper_pair(
    taper_latents: int, samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    """Return (ramp_up_b, ramp_down_b) broadcast to `samples.shape` along all
    non-temporal dims, with `taper_latents` along the temporal axis.

    Cosine ease 0 -> 1 (rising) and 1 -> 0 (falling). Endpoints excluded so
    the taper region itself is strictly partial — the literal 0 and 1 sit
    just outside the returned slice. Returns (None, None) when taper_latents
    is non-positive so callers can treat this as a no-op signal.

    Used by `LatentTemporalMask` (single retake band) and `LatentSeamZoneMask`
    (multi-band centered on iteration boundaries) to write soft mask edges.
    Centralized so the broadcast invariant — shape `[B, C, taper_latents, H, W]`
    suitable for direct slice assignment — has one home.
    """
    if taper_latents <= 0:
        return None, None
    ramp_up = 0.5 * (1.0 - torch.cos(
        torch.linspace(
            0.0, math.pi, taper_latents + 2,
            device=samples.device, dtype=samples.dtype,
        )[1:-1]
    ))
    ramp_down = ramp_up.flip(0)
    shape = (1, 1, taper_latents, 1, 1)
    expand_dims = (
        samples.shape[0], samples.shape[1], -1,
        samples.shape[3], samples.shape[4],
    )
    return ramp_up.view(shape).expand(*expand_dims), ramp_down.view(shape).expand(*expand_dims)


class LoopGeometry(NamedTuple):
    """Integer-latent loop geometry derived from user widget values."""
    window_pixel_frames: int
    overlap_pixel_frames: int
    window_latent_frames: int
    overlap_latent_frames: int
    new_latent_frames: int
    stride_pixel_frames: int
    stride_seconds: float
    effective_overlap_pixel_frames: int
    effective_overlap_seconds: float
    overlap_clamped: bool


def _compute_loop_geometry(
    window_seconds: float, overlap_seconds: float, fps: int
) -> LoopGeometry:
    """Derive stride from integer-latent counts, not seconds.

    Per-iter video pixel advance must match audio advance exactly, else
    lip-sync drifts. `overlap_clamped` is True when the requested overlap
    was reduced to `window_latents-1` to guarantee at least one new latent
    per iteration. See CLAUDE.md "Stride is derived from integer-latent
    counts" and tests/test_audio_loop_controller.py.
    """
    window_px = max(1, round(window_seconds * fps))
    overlap_px = max(0, round(overlap_seconds * fps))
    window_latents = (window_px - 1) // LTX_TEMPORAL_SCALE + 1
    overlap_latents = (
        (overlap_px - 1) // LTX_TEMPORAL_SCALE + 1 if overlap_px > 0 else 0
    )
    clamped = False
    if overlap_latents >= window_latents:
        overlap_latents = window_latents - 1
        clamped = True
    new_latents = window_latents - overlap_latents
    stride_px = new_latents * LTX_TEMPORAL_SCALE
    return LoopGeometry(
        window_pixel_frames=window_px,
        overlap_pixel_frames=overlap_px,
        window_latent_frames=window_latents,
        overlap_latent_frames=overlap_latents,
        new_latent_frames=new_latents,
        stride_pixel_frames=stride_px,
        stride_seconds=stride_px / fps,
        effective_overlap_pixel_frames=window_px - stride_px,
        effective_overlap_seconds=(window_px - stride_px) / fps,
        overlap_clamped=clamped,
    )


def _compute_tile_count(audio_duration: float, stride: float) -> int:
    """Number of valid loop iterations. Matches AudioLoopController stop condition.

    Uses ``floor(audio_duration / stride)`` so the last iter's START is
    within audio bounds. **Does NOT bound the last iter's WINDOW END** —
    the window extends ``window_seconds`` past its start, so the
    assembled video can overshoot audio length by up to
    ``window − stride`` seconds. ``scripts/apply_trim_image_batch_to_audio.py``
    (F14) clips that overshoot off before muxing; without it the saved
    mp4 ends with audible silence.

    Why not bound the WINDOW END here instead? Bounding by
    ``floor((audio − window) / stride)`` would lose up to ``window − stride``
    seconds of audio coverage at the END of the song — strictly worse
    user experience than the trim-at-output fix (silence is gone either
    way; the trim version preserves full audio coverage). The trade-off
    is ~3-5% wasted sampler compute per render on the overshoot frames,
    which is much cheaper than truncating the song. Postmortem:
    ``internal/analysis/loop_audio_overshoot_analysis.md`` (private).

    Caps at 200 for display/planning purposes. AudioLoopController itself
    has no cap — it runs until ``should_stop`` fires (see
    ``AudioLoopController.execute``).
    """
    if stride <= 0:
        return 1
    return max(1, min(int(audio_duration // stride), 200))


def _audio_duration(audio: dict) -> float:
    """Extract duration in seconds from a ComfyUI AUDIO dict."""
    return audio["waveform"].shape[-1] / audio["sample_rate"]


def _parse_timestamp(ts: str) -> float:
    """Parse a timestamp string into seconds.

    Supports formats:
      - "1:23"     -> 83.0
      - "1:23.5"   -> 83.5
      - "0:05"     -> 5.0
      - "83"       -> 83.0
      - "83.5"     -> 83.5
    """
    ts = ts.strip()
    if ":" in ts:
        parts = ts.split(":")
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    return float(ts)


def _format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or M:SS.ss if fractional."""
    m = int(seconds) // 60
    s = seconds - m * 60
    if s == int(s):
        return f"{m}:{int(s):02d}"
    return f"{m}:{s:05.2f}"


_TS_PATTERN = r"\d+(?::\d{1,2})?(?:\.\d+)?"
_LINE_RE = re.compile(
    rf"^({_TS_PATTERN}(?:\s*-\s*{_TS_PATTERN})?\+?)\s*:\s*(.+)$"
)


from typing import Callable, Literal, TypeVar

_T = TypeVar("_T")

BlendShape = Literal["raised_cosine", "spike"]


def _parse_schedule_generic(
    schedule: str,
    convert_value: Callable[[str], _T | None],
) -> list[tuple[float, float | None, _T]]:
    """Parse a timestamp-based schedule with a pluggable value converter.

    Each line: `timestamp_range: value`
    Range formats:
      - "0:00-0:38: value"   (start-end, inclusive)
      - "1:15+: value"       (from here onward)
      - "38-75: value"       (bare seconds)

    convert_value receives the raw string after the colon. Return None to
    skip the line (e.g. invalid integer).
    """
    entries: list[tuple[float, float | None, _T]] = []
    for line in schedule.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if not match:
            continue
        range_part = match.group(1).strip()
        value = convert_value(match.group(2).strip())
        if value is None:
            continue

        if range_part.endswith("+"):
            start = _parse_timestamp(range_part[:-1])
            entries.append((start, None, value))
        elif "-" in range_part:
            parts = range_part.split("-", 1)
            start = _parse_timestamp(parts[0])
            end = _parse_timestamp(parts[1])
            entries.append((start, end, value))
        else:
            t = _parse_timestamp(range_part)
            entries.append((t, t, value))
    return entries


def _snap_schedule_to_iterations(
    entries: list[tuple[float, float | None, _T]],
    stride_seconds: float,
) -> list[tuple[float, float | None, _T]]:
    """Round each entry boundary to the nearest integer multiple of stride.

    Prevents the "mid-iteration mixed conditioning" failure mode — the loop
    advances in fixed-stride steps, so any schedule boundary that doesn't
    land on a stride multiple produces an iteration whose window straddles
    the boundary. That iteration runs on one conditioning for its entire
    window, effectively applying whichever prompt was nearest to the
    iteration start-point; the further the boundary is from the stride
    grid, the more timing drift accumulates.

    Pure function over entries — no loop-state dependency.

    - `end=None` (open last entry) is preserved.
    - Zero-length entries after snapping are dropped.
    - Entries whose snapped starts collide: later one wins (consistent with
      `_match_schedule_generic`'s "last match wins" rule).
    - `stride_seconds <= 0` is a no-op (returns input unchanged) to avoid
      divide-by-zero.
    """
    if stride_seconds <= 0 or not entries:
        return entries

    def _snap(t: float) -> float:
        return round(t / stride_seconds) * stride_seconds

    snapped: list[tuple[float, float | None, _T]] = []
    for start, end, value in entries:
        s = _snap(start)
        e = None if end is None else _snap(end)
        if e is not None and e <= s:
            continue
        snapped.append((s, e, value))

    # Collapse duplicates at the same snapped start — later entry wins.
    by_start: dict[float, tuple[float, float | None, _T]] = {}
    for entry in snapped:
        by_start[entry[0]] = entry
    return sorted(by_start.values(), key=lambda e: e[0])


def _match_schedule_generic(
    entries: list[tuple[float, float | None, _T]],
    current_time: float,
    default: _T,
) -> _T:
    """Find the matching value for the given time. Last match wins."""
    result: _T | None = None
    for start, end, value in entries:
        if end is None:
            if current_time >= start:
                result = value
        else:
            if start <= current_time <= end:
                result = value
    if result is None and entries:
        result = entries[-1][2]
    return result if result is not None else default


def _match_schedule_with_next_generic(
    entries: list[tuple[float, float | None, _T]],
    current_time: float,
    blend_seconds: float,
    default: _T,
    blend_shape: BlendShape = "raised_cosine",
) -> tuple[_T, _T, float]:
    """Find current value, next value, and blend factor.

    Returns (current_value, next_value, blend_factor).

    `blend_shape` controls how blend_factor evolves near a boundary:

    - `"raised_cosine"` (default): ramp centered on each boundary, spanning
      `±blend_seconds/2`. At boundary−half_window: blend_factor=0,
      current=pre-boundary. At boundary itself: blend_factor=0.5. At
      boundary+half_window: blend_factor=1, values still describe the
      pre→post transition so downstream ConditioningBlend can lerp
      smoothly across the entire window.
    - `"spike"`: legacy behavior — blend_factor only moves during the
      blend_seconds window BEFORE the boundary, then snaps to the new
      entry. Kept behind `snap_boundaries=False` for backcompat; this
      shape is the root cause of the jitter bug when the blend window
      is smaller than the loop stride.

    Formula (raised-cosine): `blend_factor = 0.5 * (1 - cos(π * dt))`
    where `dt = clip((current_time - boundary + half_window) /
    blend_seconds, 0, 1)`.
    """
    if blend_seconds <= 0 or not entries:
        current = _match_schedule_generic(entries, current_time, default)
        return current, current, 0.0

    if blend_shape == "spike":
        return _match_spike(entries, current_time, blend_seconds, default)
    if blend_shape == "raised_cosine":
        return _match_raised_cosine(entries, current_time, blend_seconds, default)
    raise ValueError(f"Unknown blend_shape: {blend_shape!r}")


def _match_spike(
    entries: list[tuple[float, float | None, _T]],
    current_time: float,
    blend_seconds: float,
    default: _T,
) -> tuple[_T, _T, float]:
    """Legacy per-iteration spike blend — kept for `snap_boundaries=False`."""
    current_value = _match_schedule_generic(entries, current_time, default)

    next_boundary: float | None = None
    next_value = current_value
    for start, _end, value in entries:
        if start > current_time:
            if next_boundary is None or start < next_boundary:
                next_boundary = start
                next_value = value

    if next_boundary is None:
        return current_value, current_value, 0.0

    time_to_boundary = next_boundary - current_time
    if time_to_boundary < blend_seconds:
        blend_factor = 1.0 - (time_to_boundary / blend_seconds)
        return current_value, next_value, blend_factor

    return current_value, current_value, 0.0


def _match_raised_cosine(
    entries: list[tuple[float, float | None, _T]],
    current_time: float,
    blend_seconds: float,
    default: _T,
) -> tuple[_T, _T, float]:
    """Raised-cosine blend centered on each boundary.

    Within the blend window, returns (pre_boundary_value, post_boundary_value,
    ramp) so downstream consumers can smoothly lerp across the transition.
    Outside the window, returns the pure current value (blend_factor=0).
    """
    sorted_entries = sorted(entries, key=lambda e: e[0])
    boundaries = [e[0] for e in sorted_entries[1:]]  # transitions between entries
    if not boundaries:
        current = _match_schedule_generic(entries, current_time, default)
        return current, current, 0.0

    half_window = blend_seconds / 2.0
    # Find the nearest boundary to current_time
    nearest = min(boundaries, key=lambda b: abs(current_time - b))
    if abs(current_time - nearest) > half_window:
        current = _match_schedule_generic(entries, current_time, default)
        return current, current, 0.0

    # Sample pre- and post-boundary values. Use a small epsilon so
    # _match_schedule_generic's "last match wins" rule picks the right side.
    eps = 1e-6
    pre_value = _match_schedule_generic(entries, nearest - eps, default)
    post_value = _match_schedule_generic(entries, nearest + eps, default)

    dt = (current_time - nearest + half_window) / blend_seconds
    dt = max(0.0, min(1.0, dt))
    blend_factor = 0.5 * (1.0 - math.cos(math.pi * dt))
    return pre_value, post_value, blend_factor


# --- Prompt schedule (str values) ---


def _parse_schedule(schedule: str) -> list[tuple[float, float | None, str]]:
    """Parse a timestamp-based prompt schedule."""
    return _parse_schedule_generic(schedule, str.strip)


def _match_schedule(entries: list[tuple[float, float | None, str]], current_time: float) -> str:
    """Find the matching prompt for the given time. Last match wins."""
    return _match_schedule_generic(entries, current_time, "")


def _match_schedule_with_next(
    entries: list[tuple[float, float | None, str]],
    current_time: float,
    blend_seconds: float,
) -> tuple[str, str, float]:
    """Find current prompt, next prompt, and blend factor."""
    return _match_schedule_with_next_generic(entries, current_time, blend_seconds, "")


# --- Image schedule (int values) ---


def _safe_int(s: str) -> int | None:
    """Convert string to int, returning None on failure (skips the entry)."""
    try:
        return int(s)
    except ValueError:
        return None


def _parse_image_schedule(schedule: str) -> list[tuple[float, float | None, int]]:
    """Parse a timestamp-based image schedule (values are integer indices)."""
    return _parse_schedule_generic(schedule, _safe_int)


def _match_image_schedule(
    entries: list[tuple[float, float | None, int]], current_time: float
) -> int:
    """Find the matching image index for the given time. Last match wins."""
    return _match_schedule_generic(entries, current_time, 0)


def _match_image_schedule_with_next(
    entries: list[tuple[float, float | None, int]],
    current_time: float,
    blend_seconds: float,
) -> tuple[int, int, float]:
    """Find current image index, next image index, and blend factor.

    Uses the legacy `spike` blend shape for backcompat — `KeyframeImageSchedule`
    does not yet expose a `snap_boundaries` widget. Logged as a Phase 1
    finding to address in a follow-up.
    """
    return _match_schedule_with_next_generic(
        entries, current_time, blend_seconds, 0, blend_shape="spike",
    )


class AudioLoopController(io.ComfyNode):
    """Computes start_index, stop signal, and iteration seed for audio-conditioned
    video extension loops.

    Wire current_iteration from TensorLoopOpen, connect the audio track, and
    this node outputs the correct start_index for TrimAudioDuration, a
    should_stop boolean for TensorLoopClose, and a per-iteration seed.
    No manual constants needed -- audio duration is read directly from the tensor.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioLoopController",
            display_name="Audio Loop Controller",
            category="looping/audio",
            description=(
                "Computes start_index, auto-stop signal, and per-iteration seed "
                "for audio-conditioned video extension loops. Reads audio duration "
                "directly from the tensor so no manual constants are needed."
            ),
            inputs=[
                io.Int.Input(
                    "current_iteration",
                    default=1,
                    min=0,
                    tooltip="Current loop iteration (1-based) from TensorLoopOpen.",
                ),
                io.Float.Input(
                    "window_seconds",
                    default=19.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Duration of each video generation window in seconds.",
                ),
                io.Float.Input(
                    "overlap_seconds",
                    default=2.0,
                    min=0.0,
                    step=0.01,
                    tooltip=(
                        "Target overlap between consecutive windows in seconds. "
                        "Internally quantized to the nearest integer latent frame "
                        "(LTX video VAE: 8 pixel frames per latent frame). "
                        "Outputs reflect the EFFECTIVE quantized values so that "
                        "audio stride exactly matches what the video decoder "
                        "emits per iteration — prevents lip-sync drift that "
                        "would otherwise accumulate from integer-latent rounding."
                    ),
                ),
                io.Audio.Input("audio", tooltip="The audio track being used for generation."),
                io.Int.Input(
                    "base_seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Base seed. Output iteration_seed = base_seed + "
                        "current_iteration. Renamed from 'seed' on 2026-04-26 "
                        "to suppress ComfyUI's auto-attached "
                        "control_after_generate dropdown — see "
                        "internal/analysis/id_lora_ablation_and_seed_widget_audit.md."
                    ),
                ),
                io.Int.Input(
                    "fps",
                    default=25,
                    min=1,
                    tooltip="Video frame rate. Used to compute overlap_frames output.",
                ),
            ],
            outputs=[
                io.Float.Output(
                    "start_index",
                    tooltip="Start time in seconds for this iteration's audio window.",
                ),
                io.Boolean.Output(
                    "should_stop",
                    tooltip=(
                        "True when the next iteration would overshoot the audio. "
                        "Wire to TensorLoopClose's stop input."
                    ),
                ),
                io.Float.Output(
                    "audio_duration",
                    tooltip="Total duration of the input audio in seconds.",
                ),
                io.Int.Output(
                    "iteration_seed",
                    tooltip="base_seed + current_iteration. Wire to extension's noise_seed.",
                ),
                io.Float.Output(
                    "stride_seconds",
                    tooltip=(
                        "Effective stride per iteration in seconds. Computed as "
                        "(new_latent_frames * 8) / fps where new_latent_frames = "
                        "window_latents - overlap_latents. Matches exactly what "
                        "the video decoder emits per iteration, so audio advances "
                        "by the same number of real frames. Wire to "
                        "TimestampPromptSchedule and AudioLoopPlanner."
                    ),
                ),
                io.Int.Output(
                    "overlap_frames",
                    tooltip=(
                        "Effective overlap in pixel frames (window_frames - "
                        "stride_frames). Wire to extension component's "
                        "overlap_frames input."
                    ),
                ),
                io.Int.Output(
                    "overlap_latent_frames",
                    tooltip=(
                        "Number of leading latents to trim each iteration "
                        "((overlap_frames - 1)//8 + 1). Wire to "
                        "LatentContextExtract / LatentOverlapTrim."
                    ),
                ),
                io.Float.Output(
                    "overlap_seconds",
                    tooltip=(
                        "Effective overlap in seconds (after latent quantization). "
                        "May differ slightly from the input widget because we snap "
                        "to integer latent boundaries to guarantee lip-sync "
                        "stays aligned. Wire to Extension subgraph's "
                        "video_start_time on LTXVAudioVideoMask."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        current_iteration: int,
        window_seconds: float,
        overlap_seconds: float,
        audio: dict,
        base_seed: int,
        fps: int,
    ) -> io.NodeOutput:
        audio_duration = _audio_duration(audio)
        g = _compute_loop_geometry(window_seconds, overlap_seconds, fps)

        start_index = current_iteration * g.stride_seconds

        # Clamp start_index so TrimAudioDuration always has enough audio
        # for the mel spectrogram (needs >1024 samples). Without this,
        # the loop body crashes on the final iteration because
        # TensorLoopClose checks should_stop AFTER the body executes.
        min_audio_seconds = 0.5
        max_start = max(0.0, audio_duration - min_audio_seconds)
        start_index = min(start_index, max_start)

        next_start = (current_iteration + 1) * g.stride_seconds
        should_stop = next_start >= audio_duration

        return io.NodeOutput(
            start_index,
            should_stop,
            float(audio_duration),
            base_seed + current_iteration,
            g.stride_seconds,
            g.effective_overlap_pixel_frames,
            g.overlap_latent_frames,
            g.effective_overlap_seconds,
        )


class TimestampPromptSchedule(io.ComfyNode):
    """Selects a prompt based on the current audio position using a timestamp schedule.

    Write prompts for different sections of your song using timestamps you
    already know (verse, chorus, bridge). The node computes the current
    audio position from the iteration number and stride, then returns the
    matching prompt.

    When blend_seconds > 0, also outputs the next_prompt and a blend_factor
    for smooth transitions. Wire both prompts through text encoders into
    ConditioningBlend for gradual prompt transitions.

    `snap_boundaries` (default True) rounds every schedule boundary to the
    nearest iteration multiple, so every iteration runs on exactly one
    prompt. Prevents the "mid-iteration mixed conditioning" jitter failure
    mode that affected pre-fix behavior at any `blend_seconds < stride`.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TimestampPromptSchedule",
            display_name="Timestamp Prompt Schedule",
            category="looping/audio",
            description=(
                "Selects a prompt based on the current audio position. "
                "Write timestamp-based schedules matching your song structure. "
                "Supports gradual blending between prompts at transitions."
            ),
            inputs=[
                io.Int.Input(
                    "current_iteration",
                    default=1,
                    min=0,
                    tooltip="Current loop iteration (1-based) from TensorLoopOpen.",
                ),
                io.Float.Input(
                    "stride_seconds",
                    default=18.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Audio stride per iteration (same as AudioLoopController).",
                ),
                io.String.Input(
                    "schedule",
                    default="0:00+: default prompt",
                    multiline=True,
                    tooltip=(
                        "Timestamp-based prompt schedule. One entry per line.\n"
                        "Formats:\n"
                        "  0:00-0:38: prompt for this range\n"
                        "  0:38-1:15: prompt for chorus\n"
                        "  1:15+: prompt from here onward\n"
                        "Timestamps: M:SS, M:SS.ss, or bare seconds."
                    ),
                ),
                io.Float.Input(
                    "blend_seconds",
                    default=0.0,
                    min=0.0,
                    step=0.5,
                    tooltip=(
                        "Transition duration in seconds. "
                        "0 (default) = hard switch at each boundary — clean when "
                        "subject is identical across entries. "
                        "Values 0 < blend_seconds < stride_seconds are auto-clamped "
                        "to stride_seconds and emit one warning (smaller values "
                        "cannot produce smooth ramps at iteration resolution). "
                        "Values >= stride_seconds produce a raised-cosine ramp "
                        "across multiple iterations. "
                        "Wire next_prompt and blend_factor to ConditioningBlend."
                    ),
                ),
                io.Boolean.Input(
                    "snap_boundaries",
                    default=True,
                    tooltip=(
                        "Snap schedule boundaries to the iteration grid "
                        "(default on). Prevents mid-iteration prompt mixing that "
                        "causes jitter. Turn off only if you need sub-stride "
                        "timing precision and accept the jitter risk (uses the "
                        "legacy spike-blend behavior)."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("prompt", tooltip="The prompt for this iteration's audio position."),
                io.String.Output("next_prompt", tooltip="The upcoming prompt at the next boundary. Same as prompt when not near a transition."),
                io.Float.Output("blend_factor", tooltip="0.0 = fully current prompt, ramps to 1.0 at the boundary. Wire to ConditioningBlend."),
                io.Float.Output("current_time", tooltip="Current position in seconds."),
            ],
        )

    @classmethod
    def execute(
        cls,
        current_iteration: int,
        stride_seconds: float,
        schedule: str,
        blend_seconds: float,
        snap_boundaries: bool = True,
    ) -> io.NodeOutput:
        _warn_legacy_use(
            "TimestampPromptSchedule",
            "TimestampPromptScheduleBatchEncode + ConditioningSelectByIteration "
            "(pre-encodes all prompts once outside the loop, avoids per-iter "
            "CLIP eviction; F5 invariant in CLAUDE.md)",
        )
        current_time = current_iteration * stride_seconds
        entries = _parse_schedule(schedule)

        if snap_boundaries:
            entries = _snap_schedule_to_iterations(entries, stride_seconds)
            blend_shape = "raised_cosine"
            # Auto-clamp sub-stride blend_seconds — values below stride can't
            # produce a smooth ramp at iteration resolution.
            if 0 < blend_seconds < stride_seconds:
                _log_once(
                    "blend_seconds_clamped",
                    (
                        f"TimestampPromptSchedule: blend_seconds="
                        f"{blend_seconds:.2f} is below stride_seconds="
                        f"{stride_seconds:.2f}; clamping to stride. "
                        "Sub-stride values can't produce smooth ramps at "
                        "iteration resolution — see docs/guides/prompt_creation_guide.md."
                    ),
                )
                blend_seconds = stride_seconds
        else:
            blend_shape = "spike"

        prompt, next_prompt, blend_factor = _match_schedule_with_next_generic(
            entries, current_time, blend_seconds, "", blend_shape=blend_shape,
        )
        return io.NodeOutput(prompt, next_prompt, blend_factor, current_time)


# Module-level LRU for batch-encoded schedules. Survives framework-level
# cache churn when upstream AudioLoopController re-executes per iteration
# (its `current_iteration` input changes every loop pass even though its
# stride_seconds / audio_duration OUTPUT values are constant). Without
# this cache, the batch encoder re-ran N Gemma forwards per iteration on
# an N-entry schedule, defeating the whole point of moving CLIP out of
# the loop.
_BATCH_ENCODE_CACHE: OrderedDict = OrderedDict()
_BATCH_ENCODE_CACHE_MAX = 4  # typically 1 live schedule; 4 covers A/B runs

# Rounding tolerances absorb float noise from upstream AudioLoopController
# quantization. Shared between cache-key construction and IS_CHANGED so
# both representations stay in lockstep if either ever changes.
_STRIDE_SECONDS_PRECISION = 4
_AUDIO_DURATION_PRECISION = 2

# Floor on stride for `audio_duration / stride` when computing iteration
# count. Prevents divide-by-zero / inf when widget defaults to 0.
_SAFE_STRIDE_EPSILON = 1e-6


def _batch_encode_cache_key(
    clip, schedule: str, stride_seconds: float,
    audio_duration: float, snap_boundaries: bool,
    frame_rate: float,
) -> tuple:
    # (id(clip), type(clip).__name__) as identity token. CLIP models are
    # large (15+ GB) and stay resident today, so address recycling isn't
    # currently a hazard -- the type discriminator is cheap insurance for
    # the future: multi-GPU splits, offload under pressure, or a ComfyUI
    # update that changes eviction policy could all produce id() collisions
    # between a freed CLIP and a different object reloaded at the same
    # address. With the type tag, a Gemma->T5 swap can't produce a ghost
    # hit. Same discriminator used in _COND_CACHE.
    return (
        id(clip),
        type(clip).__name__,
        schedule,
        round(stride_seconds, _STRIDE_SECONDS_PRECISION),
        round(audio_duration, _AUDIO_DURATION_PRECISION),
        bool(snap_boundaries),
        round(float(frame_rate), 3),
    )


# Keyframe-latent batch-encode cache. Mirrors _BATCH_ENCODE_CACHE: same
# eviction policy (LRU, small cap), same id+typename identity tagging
# pattern as cheap insurance against future address recycling.
_KEYFRAME_LATENT_CACHE: OrderedDict = OrderedDict()
_KEYFRAME_LATENT_CACHE_MAX = 4


def _keyframe_latent_cache_key(
    vae, images, schedule: str, stride_seconds: float,
    audio_duration: float, snap_boundaries: bool,
) -> tuple:
    return (
        id(vae),
        type(vae).__name__,
        id(images),
        type(images).__name__,
        schedule,
        round(stride_seconds, _STRIDE_SECONDS_PRECISION),
        round(audio_duration, _AUDIO_DURATION_PRECISION),
        bool(snap_boundaries),
    )


def _parse_iter_targets(s: str) -> set[int]:
    """Parse a comma-separated iter-index string to a set of ints.

    Empty / whitespace-only input returns an empty set (no-op default).
    Invalid integer tokens raise ValueError -- fail loudly rather than
    silently swallow typos.
    """
    if not s or not s.strip():
        return set()
    return {int(p.strip()) for p in s.split(",") if p.strip()}


class TimestampPromptScheduleBatchEncode(io.ComfyNode):
    """Pre-encodes every per-iteration prompt up front, OUTSIDE the loop.

    Pair with ConditioningSelectByIteration inside the loop. CLIP loads
    exactly once per generation; DiT + model-level patches (NAG,
    AttentionTuner, ChunkFeedForward) stay resident for the full run.

    Replaces the CachedTextEncode-inside-the-loop pattern that forced
    CLIP/DiT offload thrash and silenced NAG on iteration 2+
    (microphones/anatomy-regressions/style-drift returning after iter 1).
    The root cause is a ComfyUI ModelPatcher asymmetry: object_patches
    closures are never device-migrated on offload, so NAG's captured
    nag_cond_video tensor goes stale across a CLIP-load-triggered
    eviction. Keeping CLIP out of the loop eliminates the failure mode.

    Behavior matches TimestampPromptSchedule for every iteration index
    (snap_boundaries parity). Dedup ensures identical prompt strings are
    encoded once regardless of how many iterations they span.

    Caching: output is memoized on `(id(clip), schedule, stride_seconds,
    audio_duration, snap_boundaries)`. ComfyUI's framework cache
    invalidates this node each iteration (because upstream
    AudioLoopController re-executes), so we carry our own LRU — same
    pattern `CachedTextEncode` uses.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TimestampPromptScheduleBatchEncode",
            display_name="Timestamp Prompt Schedule (Batch Encode)",
            category="looping/audio",
            description=(
                "Pre-encodes every per-iteration prompt outside the loop. "
                "Pair with ConditioningSelectByIteration inside the loop. "
                "Eliminates CLIP offload thrash that silences NAG iter 2+."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP model (Gemma 3 for LTX 2.3)."),
                io.String.Input(
                    "schedule",
                    default="0:00+: default prompt",
                    multiline=True,
                    tooltip=(
                        "Timestamp-based prompt schedule (same format as "
                        "TimestampPromptSchedule).\n"
                        "  0:00-0:38: prompt for this range\n"
                        "  0:38-1:15: chorus prompt\n"
                        "  1:15+: prompt from here onward"
                    ),
                ),
                io.Float.Input(
                    "stride_seconds",
                    default=17.92,
                    min=0.01,
                    step=0.01,
                    tooltip=(
                        "Audio stride per iteration. Wire from "
                        "AudioLoopController.stride_seconds (effective, "
                        "latent-quantized value)."
                    ),
                ),
                io.Float.Input(
                    "audio_duration",
                    default=180.0,
                    min=0.01,
                    step=0.1,
                    tooltip=(
                        "Total audio duration in seconds. Wire from "
                        "AudioLoopController.audio_duration."
                    ),
                ),
                io.Boolean.Input(
                    "snap_boundaries",
                    default=True,
                    tooltip=(
                        "Snap schedule boundaries to the iteration grid. "
                        "Default on -- matches TimestampPromptSchedule."
                    ),
                ),
                io.Float.Input(
                    "frame_rate",
                    default=25.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    tooltip=(
                        "Stamped onto every emitted CONDITIONING as "
                        "`{'frame_rate': ...}` metadata — same thing "
                        "`LTXVConditioning` does on the initial-render path. "
                        "Default 25.0 matches LTX 2.3's canonical inference "
                        "fps (Lightricks's shipped example workflows). Keep "
                        "identical to the `frame_rate` set on the initial "
                        "render's `LTXVConditioning` node, otherwise the "
                        "model's temporal scaling differs between the "
                        "initial window and subsequent loop iterations, "
                        "producing identity drift + hallucinated objects "
                        "(e.g. microphones) escalating iter-over-iter."
                    ),
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    "conditioning_list",
                    tooltip=(
                        "List of pre-encoded CONDITIONING, one per iteration. "
                        "Wire to ConditioningSelectByIteration inside the loop."
                    ),
                ),
                io.Int.Output(
                    "iteration_count",
                    tooltip=(
                        "Number of entries in conditioning_list. Includes "
                        "+1 headroom beyond the expected loop length so the "
                        "selector's clamp absorbs overshoot."
                    ),
                ),
            ],
        )

    @classmethod
    def IS_CHANGED(
        cls,
        clip,
        schedule: str,
        stride_seconds: float,
        audio_duration: float,
        snap_boundaries: bool = True,
        frame_rate: float = 25.0,
    ) -> str:
        # Returned string tells ComfyUI's scheduler "inputs are
        # value-stable, reuse my cached output." Uses the same key as
        # the internal cache so the two can't drift.
        return repr(_batch_encode_cache_key(
            clip, schedule, stride_seconds, audio_duration,
            snap_boundaries, frame_rate,
        ))

    @classmethod
    def execute(
        cls,
        clip,
        schedule: str,
        stride_seconds: float,
        audio_duration: float,
        snap_boundaries: bool = True,
        frame_rate: float = 25.0,
    ) -> io.NodeOutput:
        cache_key = _batch_encode_cache_key(
            clip, schedule, stride_seconds, audio_duration,
            snap_boundaries, frame_rate,
        )
        cached = _BATCH_ENCODE_CACHE.get(cache_key)
        if cached is not None:
            _BATCH_ENCODE_CACHE.move_to_end(cache_key)
            return io.NodeOutput(*cached)

        entries = _parse_schedule(schedule)
        if snap_boundaries and entries:
            entries = _snap_schedule_to_iterations(entries, stride_seconds)

        # +1 headroom: if the loop runs one more iteration than the audio
        # length strictly allows, the selector's clamp returns the last
        # encoded prompt rather than crashing.
        safe_stride = max(stride_seconds, _SAFE_STRIDE_EPSILON)
        iteration_count = max(1, math.ceil(audio_duration / safe_stride) + 1)

        prompts_per_iter = [
            _match_schedule_generic(entries, i * stride_seconds, "")
            for i in range(iteration_count)
        ]

        # Dedup preserves insertion order -- same unique prompt always
        # produces the same CONDITIONING object, so the selector's output
        # is identity-stable per unique prompt.
        unique: list[str] = []
        seen: set[str] = set()
        for prompt in prompts_per_iter:
            if prompt not in seen:
                seen.add(prompt)
                unique.append(prompt)

        # Stamp frame_rate onto every encoded conditioning's metadata dict,
        # matching what LTXVConditioning does on the initial-render path.
        # Without this, positive conditioning in the loop body has no
        # frame_rate while negative (sourced from the initial-render's
        # LTXVConditioning) does — the asymmetry drives the model's
        # temporal scaling inconsistent across iterations.
        encoded: dict[str, list] = {}
        for prompt in unique:
            tokens = clip.tokenize(prompt)
            cond = clip.encode_from_tokens_scheduled(tokens)
            encoded[prompt] = [
                [t[0], {**t[1], "frame_rate": float(frame_rate)}]
                for t in cond
            ]

        conditioning_list = [encoded[p] for p in prompts_per_iter]

        _BATCH_ENCODE_CACHE[cache_key] = (conditioning_list, iteration_count)
        if len(_BATCH_ENCODE_CACHE) > _BATCH_ENCODE_CACHE_MAX:
            _BATCH_ENCODE_CACHE.popitem(last=False)
        return io.NodeOutput(conditioning_list, iteration_count)


class ConditioningSelectByIteration(io.ComfyNode):
    """Selects a pre-encoded CONDITIONING by iteration index.

    Runs INSIDE the loop. No CLIP dependency -> no CLIP load -> no DiT
    eviction. Pair with TimestampPromptScheduleBatchEncode outside the
    loop.

    Clamp behavior:
      - current_iteration >= len(conditioning_list) -> returns last entry
        (absorbs the batch encoder's +1 headroom).
      - current_iteration < 0                       -> returns first entry
        (defensive; real workflows wire current_iteration from
        TensorLoopOpen which starts at 1).
      - empty conditioning_list                     -> raises ValueError
        (wiring bug; fail loudly).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ConditioningSelectByIteration",
            display_name="Conditioning Select (by Iteration)",
            category="looping/audio",
            description=(
                "Inside-loop selector for pre-encoded conditioning. "
                "Pair with TimestampPromptScheduleBatchEncode."
            ),
            inputs=[
                io.AnyType.Input(
                    "conditioning_list",
                    tooltip=(
                        "List of CONDITIONING from "
                        "TimestampPromptScheduleBatchEncode."
                    ),
                ),
                io.Int.Input(
                    "current_iteration",
                    default=0,
                    min=0,
                    tooltip="Iteration index from TensorLoopOpen.",
                ),
            ],
            outputs=[
                io.Conditioning.Output("conditioning"),
            ],
        )

    @classmethod
    def execute(cls, conditioning_list, current_iteration: int) -> io.NodeOutput:
        if not conditioning_list:
            raise ValueError(
                "ConditioningSelectByIteration: conditioning_list is empty. "
                "Wire the output of TimestampPromptScheduleBatchEncode."
            )
        idx = max(0, min(current_iteration, len(conditioning_list) - 1))
        return io.NodeOutput(conditioning_list[idx])


class AudioLoopPlanner(io.ComfyNode):
    """Shows the iteration timeline for planning prompt schedules.

    Takes the same primitives as AudioLoopController (window_seconds,
    overlap_seconds, fps) and applies the same `_compute_loop_geometry`
    formula to derive stride. Both nodes therefore agree on stride without
    needing a wire between them — which is what previously closed a
    dependency cycle:
        AudioLoopController -> AudioLoopPlanner -> TensorLoopOpen
            -> AudioLoopController
    once `AudioLoopPlanner.total_iterations -> TensorLoopOpen.iterations_in`
    was auto-wired (2026-04-26).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioLoopPlanner",
            display_name="Audio Loop Planner",
            category="looping/audio",
            description=(
                "Shows iteration timeline with timestamps. "
                "Helps you write prompt schedules by showing what time each iteration covers. "
                "Computes stride internally from window/overlap/fps — no stride wire needed."
            ),
            inputs=[
                io.Audio.Input("audio", tooltip="The audio track."),
                io.Float.Input(
                    "window_seconds",
                    default=19.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Video generation window per iteration.",
                ),
                io.Float.Input(
                    "overlap_seconds",
                    default=2.0,
                    min=0.0,
                    step=0.01,
                    tooltip=(
                        "Target overlap between consecutive windows in seconds. "
                        "Quantized to integer latents — same formula as "
                        "AudioLoopController so total_iterations matches the loop."
                    ),
                ),
                io.Int.Input(
                    "fps",
                    default=25,
                    min=1,
                    tooltip="Video frame rate. Same value as AudioLoopController.fps.",
                ),
                io.Int.Input(
                    "max_iterations",
                    default=0,
                    min=0,
                    max=999,
                    tooltip=(
                        "Cap the iteration count for debug / short-test runs. "
                        "0 = auto (compute from audio_duration / stride; full-song "
                        "render). >0 = run that many iterations regardless of song "
                        "length. Useful for quickly testing prompt schedules without "
                        "burning a full song's render. The cap NEVER inflates iter "
                        "count above what the audio supports — short audio runs as "
                        "many iters as it can fit, ignoring an oversized cap."
                    ),
                ),
                io.String.Input(
                    "schedule",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Optional. Same TimestampPromptScheduleBatchEncode schedule "
                        "text. When provided, the summary shows snap-boundary "
                        "diagnostics: which entries snap to which iter, drift from "
                        "your timestamps, and any collisions where two entries "
                        "snap to the same iter (last-wins; one is silently dropped "
                        "at runtime). Leave empty to skip the snap report."
                    ),
                ),
            ],
            outputs=[
                io.String.Output("summary", tooltip="Iteration timeline text + snap-boundary report."),
                io.Int.Output("total_iterations", tooltip="Iteration count (after max_iterations cap)."),
                io.Float.Output(
                    "stride_seconds",
                    tooltip=(
                        "Quantized stride per iteration (seconds). Same value "
                        "AudioLoopController emits — both nodes share the "
                        "_compute_loop_geometry formula. Wire to nodes that "
                        "need stride without pulling current_iteration into "
                        "their input closure (e.g. TimestampPromptScheduleBatchEncode "
                        "feeding initial-render conditioning)."
                    ),
                ),
                io.Float.Output(
                    "audio_duration",
                    tooltip=(
                        "Total audio duration in seconds. Read from the AUDIO "
                        "input. Cycle-free alternative to AudioLoopController."
                        "audio_duration when downstream nodes feed back into "
                        "the initial-render path."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio: dict,
        window_seconds: float,
        overlap_seconds: float,
        fps: int,
        max_iterations: int = 0,
        schedule: str = "",
    ) -> io.NodeOutput:
        audio_duration = _audio_duration(audio)
        geometry = _compute_loop_geometry(window_seconds, overlap_seconds, fps)
        stride_seconds = geometry.stride_seconds

        auto_iterations = _compute_tile_count(audio_duration, stride_seconds)
        if max_iterations > 0 and max_iterations < auto_iterations:
            iterations = max_iterations
            cap_note = (
                f" [CAPPED by max_iterations={max_iterations}; "
                f"auto would have been {auto_iterations}]"
            )
        else:
            iterations = auto_iterations
            cap_note = ""

        lines = [
            f"Audio: {audio_duration:.1f}s ({_format_timestamp(audio_duration)})",
            f"Stride: {stride_seconds:.2f}s | Window: {window_seconds:.2f}s",
            f"Overlap: {geometry.effective_overlap_seconds:.2f}s "
            f"(target {overlap_seconds:.2f}s, quantized to "
            f"{geometry.overlap_latent_frames} latents)",
            f"Estimated {iterations} iterations{cap_note}:",
            "",
            f"  Initial:  {_format_timestamp(0)} - {_format_timestamp(window_seconds)}"
            f"  (0.0s - {window_seconds:.1f}s)  [uses static prompt, not schedule]",
        ]
        for i in range(1, iterations + 1):
            start = i * stride_seconds
            end = start + window_seconds
            lines.append(
                f"  Iter {i:2d}:  {_format_timestamp(start)} - {_format_timestamp(end)}"
                f"  ({start:.1f}s - {end:.1f}s)"
            )

        # Snap-boundary diagnostics — surface the snap_boundaries footgun
        # (drift up to 1/2 stride, collisions silently last-wins) so the user
        # sees what their schedule will actually become at runtime.
        if schedule.strip():
            lines.append("")
            lines.append("Schedule snap report (snap_boundaries=True effects):")
            try:
                entries = _parse_schedule(schedule)
                snapped = _snap_schedule_to_iterations(entries, stride_seconds)
                # Build a lookup from snapped-start -> last-winning entry.
                snapped_by_start: dict[float, tuple[float, float | None, str]] = {}
                for s, e, v in snapped:
                    snapped_by_start[s] = (s, e, v)
                # Walk original entries; for each, find what it snapped to and
                # whether it's the survivor at that snapped boundary.
                for orig_start, _orig_end, orig_value in entries:
                    snapped_start = round(orig_start / stride_seconds) * stride_seconds if stride_seconds > 0 else orig_start
                    drift = snapped_start - orig_start
                    survivor = snapped_by_start.get(snapped_start)
                    label = (orig_value if isinstance(orig_value, str) else str(orig_value))[:60]
                    if survivor is None:
                        # Entry was filtered out (e.g., zero-length after snap)
                        lines.append(
                            f"  {_format_timestamp(orig_start)} \"{label}\" "
                            f"-> DROPPED (zero-length after snap)"
                        )
                    elif survivor[2] != orig_value:
                        # This entry collided and lost the last-wins race
                        winner_label = (survivor[2] if isinstance(survivor[2], str) else str(survivor[2]))[:60]
                        lines.append(
                            f"  {_format_timestamp(orig_start)} \"{label}\" "
                            f"-> {_format_timestamp(snapped_start)} (drift "
                            f"{drift:+.2f}s) DROPPED — collides with "
                            f"\"{winner_label}\""
                        )
                    else:
                        marker = " (no change)" if abs(drift) < 1e-3 else f" (drift {drift:+.2f}s)"
                        lines.append(
                            f"  {_format_timestamp(orig_start)} \"{label}\" "
                            f"-> {_format_timestamp(snapped_start)}{marker}"
                        )
            except Exception as e:  # noqa: BLE001
                lines.append(f"  (schedule parse error: {e})")

        return io.NodeOutput("\n".join(lines), iterations, float(stride_seconds), float(audio_duration))


# `ScheduleToMultiPrompt` removed 2026-04-27 — zero workflow + only-doc-mention
# external usage. Targeted upstream `LTXVLoopingSampler.MultiPromptProvider`
# which we don't ship workflows for. Our canonical multi-prompt path is
# TimestampPromptScheduleBatchEncode + ConditioningSelectByIteration.


# `AudioDuration` removed 2026-04-27 — zero workflow + zero external usage.
# AudioLoopController already exposes `audio_duration` as an output;
# AudioLoopPlanner already prints duration in its summary. The standalone
# 5-line getter was dead weight. If you need duration outside ComfyUI
# in script form: `audio["waveform"].shape[-1] / audio["sample_rate"]`.


def _warn_legacy_use(class_name: str, replacement: str) -> None:
    """Once-per-process deprecation print for legacy node classes. Reuses
    the project's `_log_once` mechanism (defined later in this file)."""
    _log_once(
        f"deprecated_{class_name}",
        f"DEPRECATED: {class_name} is legacy. Migrate to {replacement}. "
        f"Class will be removed in a future release.",
    )


# Latent-volume advisory anchor. There is NO hard model-side latent-volume
# ceiling (RoPE max_pos are normalizers, not caps; the distilled path is
# token-count-independent — see docs/reference/frame_planner_reference.md
# §"Latent-volume classification"). The only structural limits are grid
# alignment (div-32 spatial, (frames-1)%8==0 temporal) + VRAM. We anchor an
# *informational* VRAM advisory on LTX-2's own HQ production default:
# 960x544 @ 497 = 32,130 latent tokens (coderef/LTX-2/.../utils/constants.py
# LTX_2_3_HQ_PARAMS). At/under = OK; above = HIGH_VRAM (informational — more
# VRAM, NOT a quality cliff; we deliberately don't impose an OOM limit because
# we don't know the user's hardware).
_LTX_HQ_PRODUCTION_VOLUME = 32_130

_LTXOrientation = Literal["landscape", "portrait", "square"]


def _snap_dimensions(target_width: int, target_height: int) -> tuple[int, int]:
    """Snap each of (target_width, target_height) DOWN to the nearest div-32
    boundary, with a floor of 32. LTX 2.3 requires div-by-32 dimensions
    (single-stage); both axes are independently snapped.

    Snap DOWN (not nearest, not up) keeps dims at/under the requested size and
    biases toward lower VRAM. (The only hard constraint is div-by-32 grid
    alignment, not a latent-volume ceiling — there is none.)"""
    w = max(32, (target_width // 32) * 32)
    h = max(32, (target_height // 32) * 32)
    return w, h


def _snap_frames(target_seconds: float, fps: int) -> tuple[int, float]:
    """Convert (target_seconds, fps) -> (frames, actual_seconds) where
    frames satisfies the LTX video VAE temporal constraint (frames - 1) % 8 == 0.

    Snap DOWN to the nearest valid frame count (smaller chunks = safer for
    VRAM and gives more frequent re-anchoring). Returns the actual_seconds
    that the snapped frame count represents (= frames / fps), which may be
    slightly less than target_seconds. Minimum result is frames=1, the
    only-9-or-greater rule is relaxed to 1 to allow degenerate test cases.
    """
    pixel_frames = max(1, round(target_seconds * fps))
    # ((pixel_frames - 1) // 8) * 8 + 1 is the snap-down to (L-1)%8==0
    snapped = ((pixel_frames - 1) // 8) * 8 + 1
    snapped = max(1, snapped)
    actual_seconds = snapped / fps
    return snapped, actual_seconds


def _compute_ltx_resolution(
    aspect_ratio: float,
    target_long_edge: int,
    frames: int,
    orientation: _LTXOrientation = "landscape",
) -> tuple[int, int, int, str]:
    """Snap an aspect ratio + target long edge to LTX 2.3-valid (W, H), and
    return the resulting latent volume + an informational VRAM advisory.

    Returns (width, height, latent_volume, status_string).

    width/height are guaranteed div-by-32. frames must satisfy (frames-1)%8==0
    (the only hard model-side constraints). Status is one of OK / HIGH_VRAM (a
    VRAM advisory, not a quality cliff — see `_classify_latent_volume`); the
    volume value is included in the string so callers can parse it.
    """
    assert (frames - 1) % 8 == 0, (
        f"frames {frames} violates LTX video VAE temporal constraint "
        "(frames - 1) % 8 == 0; valid: 9, 17, 25, ..., 489, 497, ..."
    )

    long_edge = ((target_long_edge + 31) // 32) * 32

    if orientation == "square":
        width = height = long_edge
    elif orientation == "portrait":
        height = long_edge
        raw_width = long_edge / aspect_ratio
        width = max(32, (int(raw_width) // 32) * 32)
    else:  # landscape (default)
        width = long_edge
        raw_height = long_edge / aspect_ratio
        height = max(32, (int(raw_height) // 32) * 32)
    # Short-edge snap is DOWN (not up) — keeps the resolved dims at or under
    # the requested long edge and biases toward lower VRAM. (Not about an
    # artifact ceiling — there is none; see _classify_latent_volume.)

    latent_volume, status = _classify_latent_volume(width, height, frames)
    return width, height, latent_volume, status


def _classify_latent_volume(width: int, height: int, frames: int) -> tuple[int, str]:
    """Compute LTX 2.3 latent volume = (W/32) * (H/32) * ((L-1)/8 + 1) and
    return an *informational VRAM advisory* (NOT a quality cliff).

    There is no hard latent-volume ceiling; the anchor is LTX-2's own HQ
    production default (`_LTX_HQ_PRODUCTION_VOLUME` = 32,130). At/under = OK;
    above = HIGH_VRAM (more VRAM on this rig — not a limit we impose, since the
    safe ceiling is hardware-dependent). Returns (latent_volume, status_string).
    Shared between `_compute_ltx_resolution` and `LTXFramePlanner.execute` so
    the rule lives in one place.
    """
    latent_volume = (width // 32) * (height // 32) * ((frames - 1) // 8 + 1)
    if latent_volume <= _LTX_HQ_PRODUCTION_VOLUME:
        category = "OK"
    else:
        category = "HIGH_VRAM"
    status = (
        f"{category}: latent_volume={latent_volume} "
        f"(informational VRAM advisory; LTX-2 HQ production default = "
        f"{_LTX_HQ_PRODUCTION_VOLUME}, no hard limit)"
    )
    return latent_volume, status


def _ltx_clear_keyframe_idxs(positive, negative):
    """Clear `keyframe_idxs` from both CONDITIONING lists if `positive` has any.
    Mirrors the CONDITIONING-side behavior of upstream `LTXVCropGuides` without
    touching a LATENT. Returns (positive, negative). Imports comfy modules
    lazily so the module is testable outside the ComfyUI runtime."""
    try:
        from comfy_extras.nodes_lt import get_keyframe_idxs
        from comfy import node_helpers
    except ImportError:
        # Outside ComfyUI runtime — fall back to a minimal in-process equivalent
        # that lets our unit tests exercise the keyframe-clearing logic.
        def get_keyframe_idxs(cond):
            kf = cond[0][1].get("keyframe_idxs") if cond and len(cond[0]) > 1 else None
            num = 0 if kf is None or len(kf) == 0 else len(kf)
            return kf, num

        class _NH:
            @staticmethod
            def conditioning_set_values(cond, values):
                return [(t, {**meta, **values}) for (t, meta) in cond]

        node_helpers = _NH

    _, num_keyframes = get_keyframe_idxs(positive)
    if num_keyframes == 0:
        return positive, negative
    # Must be None, not []. KJNodes' OuterSampleCallbackWrapper
    # (ltxv_nodes.py:867) gates `if keyframe_idxs is not None:` then indexes as
    # a 4D tensor; [] slips through and raises TypeError on tuple-indexing.
    # Upstream LTXVCropGuides (comfy_extras/nodes_lt.py:404) sets None — match.
    positive = node_helpers.conditioning_set_values(positive, {"keyframe_idxs": None})
    negative = node_helpers.conditioning_set_values(negative, {"keyframe_idxs": None})
    return positive, negative


class LTXVCropGuidesNoLatent(io.ComfyNode):
    """CONDITIONING-only equivalent of `LTXVCropGuides`. Strips
    `keyframe_idxs` from the positive/negative CONDITIONING; takes no
    LATENT input and produces no LATENT output.

    Used in the loop subgraph's CONDITIONING path (replaces the upstream
    `LTXVCropGuides(655)` for the F3 wiring) so the post-sampling
    keyframe-padding crop can run on a sibling LATENT-only `LTXVCropGuides`
    instance without creating a dependency cycle. Removes the wasted
    `latent["samples"].clone()` upstream `LTXVCropGuides` does on the
    CONDITIONING-only role.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXVCropGuidesNoLatent",
            display_name="LTX Crop Guides (CONDITIONING only)",
            category="AudioLoopHelper/utility",
            description=(
                "Strips keyframe_idxs from positive/negative CONDITIONING. "
                "CONDITIONING-only variant of LTXVCropGuides — no LATENT in/out, "
                "so it can sit on the F3 path without creating a sample-output "
                "dependency cycle in loop subgraphs."
            ),
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative) -> io.NodeOutput:
        new_pos, new_neg = _ltx_clear_keyframe_idxs(positive, negative)
        return io.NodeOutput(new_pos, new_neg)


class LTXResolutionFromAspect(io.ComfyNode):
    """Resolve a target aspect ratio to LTX 2.3-valid (width, height) + report
    the latent volume with an informational VRAM advisory.

    Wire `width` / `height` into `EmptyLTXVLatentVideo`. `latent_volume` +
    `status` are informational: there is no hard latent-volume ceiling, only
    grid-alignment (div-32 / 8k+1) + VRAM. `HIGH_VRAM` means "above LTX-2's HQ
    production default (32,130 tokens) — watch memory on your card", not
    "expect artifacts" (see `docs/reference/frame_planner_reference.md`).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXResolutionFromAspect",
            display_name="LTX Resolution From Aspect",
            category="AudioLoopHelper/utility",
            description=(
                "Snap an aspect ratio + target long edge to an LTX 2.3-valid "
                "(W, H) pair (both div-by-32). Computes latent volume + an "
                "informational VRAM advisory (no hard ceiling)."
            ),
            inputs=[
                io.Float.Input(
                    "aspect_ratio", default=16 / 9, min=0.25, max=4.0, step=0.001,
                    tooltip="Width/height ratio. 16:9 = 1.778, 4:3 = 1.333, 1:1 = 1.0.",
                ),
                io.Int.Input(
                    "target_long_edge", default=832, min=128, max=1920, step=32,
                    tooltip="Long edge in pixels. Snapped UP to nearest div-32 boundary.",
                ),
                io.Int.Input(
                    "frames", default=497, min=9, max=4097, step=8,
                    tooltip="Total frames. Must satisfy (frames-1)%8==0.",
                ),
                io.Combo.Input(
                    "orientation",
                    options=["landscape", "portrait", "square"],
                    default="landscape",
                ),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="latent_volume"),
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(
        cls,
        aspect_ratio: float,
        target_long_edge: int,
        frames: int,
        orientation: str,
    ) -> io.NodeOutput:
        w, h, vol, status = _compute_ltx_resolution(
            aspect_ratio, target_long_edge, frames, orientation
        )
        return io.NodeOutput(w, h, vol, status)


class LTXFramePlanner(io.ComfyNode):
    """Single source of truth for LTX 2.3 dimension config: width, height,
    frames-per-iteration, fps. Auto-snaps to LTX-architectural constraints
    so the user types human-readable values (832, 448, 20.0, 25) and the
    node emits the snapped LTX-valid versions everywhere downstream.

    Replaces scattered widget values across EmptyLTXVLatentVideo,
    AudioLoopController, AudioLoopPlanner, LTXVConditioning, and
    ImageResizeKJv2. Wire its outputs to those nodes' inputs and the
    user only edits ONE node for "what shape and length is this render?"

    Rules enforced (so the user never has to remember them):
      - width and height are each div-by-32 (LTX 2.3 single-stage rule)
      - frames satisfies (frames - 1) % 8 == 0 (video VAE temporal rule)
      - actual_seconds = frames / fps (always self-consistent)
      - latent_volume reported with an informational VRAM advisory status
        (OK / HIGH_VRAM vs LTX-2's HQ production default 32,130 — NOT a hard
        ceiling; see docs/reference/frame_planner_reference.md)

    Wiring map (apply via scripts/apply_frame_planner_consolidation.py):
      LTXFramePlanner outputs:
        width        -> EmptyLTXVLatentVideo.width, ImageResizeKJv2.width
        height       -> EmptyLTXVLatentVideo.height, ImageResizeKJv2.height
        frames       -> EmptyLTXVLatentVideo.length
        actual_seconds -> AudioLoopController.window_seconds,
                          AudioLoopPlanner.window_seconds,
                          subgraph video_end_time slot
        fps_int      -> AudioLoopController.fps, AudioLoopPlanner.fps
        fps_float    -> LTXVConditioning.frame_rate
        latent_volume, status, summary -> PreviewAny for visibility
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXFramePlanner",
            display_name="LTX Frame Planner",
            category="AudioLoopHelper/utility",
            description=(
                "Single source of truth for LTX 2.3 dimension config. "
                "Type human-readable target values; node snaps to LTX-valid "
                "(div-32 width/height, (L-1)%8 frames) and emits everything "
                "downstream consumers need. Wire outputs to EmptyLTXVLatentVideo, "
                "AudioLoopController, AudioLoopPlanner, LTXVConditioning, "
                "and ImageResizeKJv2."
            ),
            inputs=[
                io.Int.Input(
                    "target_width", default=832, min=32, max=4096, step=32,
                    tooltip=(
                        "Desired output width in pixels. Snapped DOWN to the "
                        "nearest div-32 boundary. Common: 832 (16:9 cinema), "
                        "960 (16:9 wider), 448 (9:16 portrait), 512 (1:1 square)."
                    ),
                ),
                io.Int.Input(
                    "target_height", default=448, min=32, max=4096, step=32,
                    tooltip=(
                        "Desired output height in pixels. Snapped DOWN to the "
                        "nearest div-32 boundary. 448 pairs with 832 width for "
                        "1.86:1 cinema aspect (volume 22,932). The shipped "
                        "960x544 = 32,130 = LTX-2's HQ production default."
                    ),
                ),
                io.Float.Input(
                    "target_seconds", default=19.88, min=0.04, max=120.0, step=0.01,
                    tooltip=(
                        "Per-iteration window duration in seconds. NOT total "
                        "video length (that's determined by your audio). "
                        "Snapped DOWN to (L-1)%8 frame count. Lower values = "
                        "more iters = more re-anchoring (better identity, "
                        "higher resolution headroom, more boundary seams). "
                        "Default 19.88 = 497 frames at 25fps = ~9 iters per 3-min song."
                    ),
                ),
                io.Int.Input(
                    "fps", default=25, min=1, max=120,
                    tooltip=(
                        "Frames per second. 25 is LTX 2.3's canonical "
                        "inference fps (matches Lightricks's shipped "
                        "example workflows + the 8n+1 latent boundary); "
                        "LTXVConditioning.frame_rate scales the "
                        "model's temporal positional embedding from this value."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="frames"),
                io.Float.Output(display_name="actual_seconds"),
                io.Int.Output(display_name="fps_int"),
                io.Float.Output(display_name="fps_float"),
                io.Int.Output(display_name="latent_volume"),
                io.String.Output(display_name="status"),
                io.String.Output(display_name="summary"),
            ],
        )

    @classmethod
    def execute(
        cls,
        target_width: int,
        target_height: int,
        target_seconds: float,
        fps: int,
    ) -> io.NodeOutput:
        width, height = _snap_dimensions(target_width, target_height)
        frames, actual_seconds = _snap_frames(target_seconds, fps)
        latent_volume, status = _classify_latent_volume(width, height, frames)
        category = status.split(":", 1)[0]

        snap_notes = []
        if (target_width, target_height) != (width, height):
            snap_notes.append(f"size {target_width}x{target_height} -> {width}x{height}")
        if abs(actual_seconds - target_seconds) > 1e-3:
            snap_notes.append(f"window {target_seconds:.2f}s -> {actual_seconds:.2f}s ({frames} frames)")
        snap_str = " (snapped: " + "; ".join(snap_notes) + ")" if snap_notes else ""

        summary = (
            f"{width}x{height}, {actual_seconds:.2f}s @ {fps}fps, "
            f"{frames} frames, latent volume {latent_volume} ({category})"
            + snap_str
        )

        return io.NodeOutput(
            width, height, frames, actual_seconds,
            fps, float(fps),
            latent_volume, status, summary,
        )


class ConditioningBlend(io.ComfyNode):
    """Blends two conditionings with a factor. Works with any text encoder
    including LTX 2.3 Gemma 3 (no pooled_output required).

    When blend_factor = 0.0, passes conditioning_a through unchanged.
    When blend_factor = 1.0, passes conditioning_b through unchanged.
    Values between lerp the conditioning tensors.

    Wire TimestampPromptSchedule's blend_factor here for smooth transitions.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ConditioningBlend",
            display_name="Conditioning Blend",
            category="looping/audio",
            description=(
                "Blends two conditionings with a factor. Works with LTX Gemma 3 "
                "Also compatible with CLIP conditioning. Use with TimestampPromptSchedule "
                "for smooth prompt transitions."
            ),
            inputs=[
                io.Conditioning.Input("conditioning_a", tooltip="Current prompt conditioning."),
                io.Conditioning.Input("conditioning_b", tooltip="Next prompt conditioning."),
                io.Float.Input(
                    "blend_factor",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="0.0 = all A, 1.0 = all B. Wire from TimestampPromptSchedule.",
                ),
            ],
            outputs=[
                io.Conditioning.Output("conditioning"),
            ],
        )

    @classmethod
    def execute(
        cls,
        conditioning_a: list,
        conditioning_b: list,
        blend_factor: float,
    ) -> io.NodeOutput:
        # Passthrough when no blending needed
        if blend_factor <= 0.0:
            return io.NodeOutput(conditioning_a)
        if blend_factor >= 1.0:
            return io.NodeOutput(conditioning_b)

        out = []
        # Uses only conditioning_b[0] -- LTX Gemma 3 produces single-element conditioning.
        # For multi-element CLIP conditioning, this would need zip/min indexing.
        cond_b = conditioning_b[0][0]

        for i in range(len(conditioning_a)):
            t_a = conditioning_a[i][0]
            t_b = cond_b

            # Align sequence lengths by zero-padding the shorter one
            if t_b.shape[1] < t_a.shape[1]:
                t_b = torch.cat([t_b, torch.zeros((1, t_a.shape[1] - t_b.shape[1], t_b.shape[2]), device=t_b.device)], dim=1)
            elif t_a.shape[1] < t_b.shape[1]:
                t_a = torch.cat([t_a, torch.zeros((1, t_b.shape[1] - t_a.shape[1], t_a.shape[2]), device=t_a.device)], dim=1)

            # Lerp the conditioning tensors
            blended = t_a * (1.0 - blend_factor) + t_b * blend_factor

            # Copy metadata from conditioning_a, blend pooled_output if present
            opts = conditioning_a[i][1].copy()
            pooled_a = conditioning_a[i][1].get("pooled_output", None)
            pooled_b = conditioning_b[0][1].get("pooled_output", None)
            if pooled_a is not None and pooled_b is not None:
                opts["pooled_output"] = pooled_a * (1.0 - blend_factor) + pooled_b * blend_factor

            # Combine attention masks (OR -- valid if either is valid)
            mask_a = conditioning_a[i][1].get("attention_mask", None)
            mask_b = conditioning_b[0][1].get("attention_mask", None)
            if mask_a is not None and mask_b is not None:
                # Pad masks to same length
                max_len = max(mask_a.shape[-1], mask_b.shape[-1])
                if mask_a.shape[-1] < max_len:
                    mask_a = torch.cat([mask_a, torch.zeros((*mask_a.shape[:-1], max_len - mask_a.shape[-1]), device=mask_a.device)], dim=-1)
                if mask_b.shape[-1] < max_len:
                    mask_b = torch.cat([mask_b, torch.zeros((*mask_b.shape[:-1], max_len - mask_b.shape[-1]), device=mask_b.device)], dim=-1)
                opts["attention_mask"] = torch.clamp(mask_a + mask_b, 0, 1)

            out.append([blended, opts])

        return io.NodeOutput(out)


# Singleton nullcontext avoids per-call allocation on the hot path.
_NULL_CTX = nullcontext()


def _profile_span(name: str):
    # _PROFILER_STATE is a module-level alias to the dict attached to `torch`
    # (bound later in this module). Using it here skips the getattr lookup
    # that _get_profiler_state does on every hot-path call.
    if _PROFILER_STATE.get("profiler") is None:
        return _NULL_CTX
    return torch.profiler.record_function(name)


# Env-gated diagnostic: trace per-iter LATENT shapes on the loop-body
# guide path. Off by default (no-op on the hot path). Set
# AUDIOLOOPHELPER_KF_DEBUG=1 to log to stderr (lands in the ComfyUI
# console log). Used to localize the final-iteration LTXVAddLatentGuide
# spatial-ratio crash on full-length renders — see which loop input
# (video latent vs guide latent) diverges from the steady-state shape
# and on which iteration.
_KF_DEBUG = os.environ.get("AUDIOLOOPHELPER_KF_DEBUG", "") not in ("", "0")


def _kf_debug(msg: str) -> None:
    if _KF_DEBUG:
        print(f"[KFDEBUG] {msg}", file=sys.stderr, flush=True)


class LatentContextExtract(io.ComfyNode):
    """Extracts the last N latent frames as context for the next loop iteration.

    Replaces LTXVSelectLatents + StripLatentNoiseMask in the latent-space loop.
    Takes the tail frames and strips noise_mask so LTXVAudioVideoMask creates
    a fresh mask (matching VAEEncode behavior from the IMAGE workflow).

    Wire: TensorLoopOpen previous_value → this → LTXVAudioVideoMask video_latent
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LatentContextExtract",
            display_name="Latent Context Extract",
            category="looping/audio",
            description=(
                "Extracts last N latent frames as context for the next loop iteration. "
                "Strips noise_mask for clean sampler behavior."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Previous iteration's video latent."),
                io.Int.Input("overlap_latent_frames", default=4, min=1,
                             tooltip="Number of tail latent frames to extract. Wire from AudioLoopController."),
            ],
            outputs=[
                io.Latent.Output("context", tooltip="Clean context latent (no noise_mask). Wire to LTXVAudioVideoMask."),
            ],
        )

    @classmethod
    def execute(cls, latent: dict, overlap_latent_frames: int) -> io.NodeOutput:
        with _profile_span("LatentContextExtract"):
            s = latent.copy()
            video = s["samples"]
            frames = video.shape[2]

            start = max(0, frames - overlap_latent_frames)
            s["samples"] = video[:, :, start:]

            # Strip noise_mask so downstream creates fresh (matches VAEEncode behavior)
            s.pop("noise_mask", None)

        if _KF_DEBUG:
            _kf_debug(
                f"ContextExtract: in_shape={tuple(video.shape)} "
                f"overlap={overlap_latent_frames} out_shape={tuple(s['samples'].shape)}"
            )
        return io.NodeOutput(s)


class LatentOverlapTrim(io.ComfyNode):
    """Trims the first N latent frames (overlap region) from a sampler's output.

    Replaces LTXVSelectLatents for output trimming in the latent-space loop.
    Keeps everything after the overlap region, strips noise_mask.

    Wire: LTXVCropGuides latent → this → subgraph output
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LatentOverlapTrim",
            display_name="Latent Overlap Trim",
            category="looping/audio",
            description=(
                "Trims first N latent frames (overlap) from sampler output. "
                "Keeps new content only."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Sampler output video latent (after CropGuides)."),
                io.Int.Input("overlap_latent_frames", default=4, min=0,
                             tooltip="Number of leading latent frames to trim. Wire from AudioLoopController."),
            ],
            outputs=[
                io.Latent.Output("trimmed", tooltip="New content only (overlap removed)."),
            ],
        )

    @classmethod
    def execute(cls, latent: dict, overlap_latent_frames: int) -> io.NodeOutput:
        with _profile_span("LatentOverlapTrim"):
            s = latent.copy()
            video = s["samples"]

            # Clamp to avoid empty tensor if overlap >= total frames
            trim = min(overlap_latent_frames, video.shape[2] - 1)
            s["samples"] = video[:, :, trim:]

            # Strip noise_mask for clean accumulation
            s.pop("noise_mask", None)

        return io.NodeOutput(s)


# `StripLatentNoiseMask` removed 2026-04-27 — zero workflow + only-doc-mention
# external usage. Its own docstring redirected users to LatentContextExtract /
# LatentOverlapTrim which auto-strip noise_mask. The standalone 4-line helper
# was dead weight. If you genuinely need the bare strip:
#   out = latent.copy(); out.pop("noise_mask", None)


class RunIdPrefix(io.ComfyNode):
    """Emits a per-render unique filename prefix shared across every save
    node in a workflow.

    Wire ``video_prefix`` into ``VHS_VideoCombine.filename_prefix`` and
    ``SaveImage.filename_prefix``; wire ``latent_prefix`` into
    ``SaveLatent.filename_prefix``. All artifacts of one render land
    under ``<output>/<workflow_name>/<timestamp>/`` — the run is now a
    folder, not a counter on the global namespace.

    ``fingerprint_inputs`` returns NaN so ComfyUI re-evaluates the node
    on every queue submission. Without that the cached timestamp would
    propagate across all renders and the unification contract would
    break.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RunIdPrefix",
            display_name="Run ID Prefix",
            category="utility",
            description=(
                "Emits a per-render filename prefix (`<workflow_name>/"
                "<timestamp>`) shared across every save node in the "
                "workflow. Forces re-evaluation each queue submission so "
                "the timestamp is fresh per render."
            ),
            inputs=[
                io.String.Input(
                    "workflow_name",
                    default="render",
                    tooltip="Output sub-folder (typically the workflow's filename base).",
                ),
                io.String.Input(
                    "timestamp_format",
                    default="%Y%m%d_%H%M%S",
                    tooltip="strftime format for the per-run timestamp segment.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "video_prefix",
                    tooltip="<workflow_name>/<timestamp> — wire to VHS_VideoCombine and SaveImage.",
                ),
                io.String.Output(
                    "latent_prefix",
                    tooltip="<workflow_name>/<timestamp>/latents/segment — wire to SaveLatent.",
                ),
            ],
        )

    @classmethod
    def execute(cls, workflow_name: str, timestamp_format: str) -> io.NodeOutput:
        import datetime
        ts = datetime.datetime.now().strftime(timestamp_format)
        base = f"{workflow_name}/{ts}"
        return io.NodeOutput(base, f"{base}/latents/segment")

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        return float("NaN")


class LatentFrameCount(io.ComfyNode):
    """Emits pixel-frame and latent-frame counts from a video LATENT.

    Used by the latent-load-based upscale + seam workflows to size
    ``LTXVEmptyLatentAudio.frames_number`` directly from the loaded
    video latent's temporal extent — no AUDIO source required for that
    sizing step. The LTX video VAE convention is
    ``pixel_frames = (latent_frames - 1) * 8 + 1`` (LTX_TEMPORAL_SCALE
    = 8). Mirrors ``_snap_frames`` in reverse.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LatentFrameCount",
            display_name="Latent Frame Count",
            category="latent/utility",
            description=(
                "Counts frames in a video LATENT. Returns (pixel_frames, "
                "latent_frames) where pixel_frames = (latent_frames - 1)*8 + 1 "
                "per the LTX video VAE temporal scale. Useful for sizing "
                "LTXVEmptyLatentAudio.frames_number when loading a "
                "pre-saved latent instead of decoding a video."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Video latent (e.g. from LoadLatent)."),
            ],
            outputs=[
                io.Int.Output("pixel_frames", tooltip="pixel_frames = (latent_frames - 1)*8 + 1"),
                io.Int.Output("latent_frames", tooltip="Raw temporal dim of the latent."),
            ],
        )

    @classmethod
    def execute(cls, latent: dict) -> io.NodeOutput:
        latent_frames = int(latent["samples"].shape[2])
        pixel_frames = (latent_frames - 1) * LTX_TEMPORAL_SCALE + 1
        return io.NodeOutput(pixel_frames, latent_frames)


def _purge_stale_loaded_models() -> None:
    """Prune stale entries from ``comfy.model_management.current_loaded_models``
    (wrappers whose underlying ``.model`` was GC'd to None) and force a
    cleanup pass.

    Defensive workaround for a ComfyUI weakref-finalize race where
    ``free_memory()`` and ``cleanup_models()`` crash with
    ``AttributeError: 'NoneType' object has no attribute 'model_size'``
    when walking entries whose model was finalized but the wrapper
    survived. Surfaces on workflows where a large model (e.g. LTX 2.3
    22B at 24 GB) gets swapped out and another model needs to load
    afterwards.

    No-op when ``comfy.model_management`` isn't importable (tests,
    headless harness). All sub-steps wrapped in try/except so this
    function never raises into the workflow.
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return  # expected when running under pytest/headless harness
    try:
        mm.current_loaded_models[:] = [
            e for e in mm.current_loaded_models
            if getattr(e, "model", None) is not None
        ]
    except Exception as e:
        warnings.warn(f"PurgeVRAM: stale-prune failed: {e!r}", stacklevel=2)
    try:
        mm.cleanup_models()
    except Exception as e:
        warnings.warn(f"PurgeVRAM: cleanup_models failed: {e!r}", stacklevel=2)
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception as e:
        warnings.warn(f"PurgeVRAM: empty_cache failed: {e!r}", stacklevel=2)
    try:
        gc.collect()
    except Exception as e:
        warnings.warn(f"PurgeVRAM: gc.collect failed: {e!r}", stacklevel=2)


class PurgeVRAM(io.ComfyNode):
    """LATENT pass-through that prunes stale entries from ComfyUI's
    ``current_loaded_models`` and forces a cleanup pass as a side effect.

    Wire between ``SamplerCustomAdvanced`` output and the next
    model-using node when you hit the ``AttributeError: 'NoneType'
    object has no attribute 'model_size'`` crash during model swap.
    The crash is a ComfyUI core bug (weakref finalize leaves dead
    entries in ``current_loaded_models``); this node prunes them
    before the next ``free_memory()`` walk hits the stale entry.

    Not in the canonical workflow by default — splice manually when
    needed. If the bug becomes load-bearing, we'll wire it via an
    apply script.

    See ``docs/guides/debugging_guide.md`` "Model swap crash" for
    symptom + when to use this.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PurgeVRAM",
            display_name="Purge VRAM (defensive)",
            category="utility",
            description=(
                "Pass-through LATENT that prunes stale entries from "
                "comfy.model_management.current_loaded_models and "
                "calls cleanup. Workaround for the model-swap crash "
                "(AttributeError on .model_size of a None model)."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="LATENT pass-through. Cleanup runs as a side effect before returning.",
                ),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="Same LATENT, unchanged."),
            ],
        )

    @classmethod
    def execute(cls, latent: dict) -> io.NodeOutput:
        _purge_stale_loaded_models()
        return io.NodeOutput(latent)


class TrimVideoLatentToAudio(io.ComfyNode):
    """Latent-space companion to ``TrimImageBatchToAudio`` (F14).

    Clips a video LATENT's temporal dim so that — after LTX VAE
    decode — the resulting image batch is at most
    ``int(audio_duration * fps)`` pixel frames. Saves VAE decode work
    on overshoot frames (~3-5% on typical loop renders, ~17% on
    short-audio cases). Pair with F14 downstream as a safety net for
    any off-by-one in the latent → pixel arithmetic.

    LTX video VAE convention: ``pixel_frames = (latent_frames - 1) * 8 + 1``.
    For a target pixel count ``P`` we snap DOWN to the largest valid
    ``P' = ((P - 1) // 8) * 8 + 1`` and emit ``L = (P' - 1) // 8 + 1``
    latent frames. Snap-down guarantees decoded output never exceeds
    audio duration.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrimVideoLatentToAudio",
            display_name="Trim Video Latent to Audio",
            category="looping/audio",
            description=(
                "Clip a video latent's temporal dim so its VAE-decoded "
                "pixel-frame count is ≤ floor(audio_duration * fps). "
                "Latent-space companion to TrimImageBatchToAudio — "
                "saves VAE decode work on overshoot frames."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="Assembled video latent (typically LatentConcat output before final VAE decode).",
                ),
                io.Audio.Input(
                    "audio",
                    tooltip="Reference audio. Wire from the same source feeding VHS_VideoCombine.audio.",
                ),
                io.Int.Input(
                    "fps",
                    default=25,
                    min=1,
                    tooltip="Output frame rate. Wire from LTXFramePlanner.fps_int.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    "latent",
                    tooltip="Latent trimmed to a count whose decoded pixel-frame count ≤ audio_duration * fps.",
                ),
            ],
        )

    @classmethod
    def execute(cls, latent: dict, audio: dict, fps: int) -> io.NodeOutput:
        # Snap UP to the smallest valid latent count whose decoded pixel
        # count >= target_pixel. Snapping DOWN would clip up to 7 pixel
        # frames (0.28s at 25fps) of audio at the END because ffmpeg
        # -shortest clips audio when video < audio. Snapping UP keeps
        # video >= audio; downstream F14 (`TrimImageBatchToAudio`) clips
        # the small overshoot at exact pixel precision.
        audio_duration = _audio_duration(audio)
        target_pixel = max(1, int(audio_duration * fps))
        # pixel = (latent - 1) * 8 + 1; solve latent >= (pixel - 1) / 8 + 1, round up.
        target_latent = max(1, math.ceil((target_pixel - 1) / LTX_TEMPORAL_SCALE) + 1)

        samples = latent["samples"]
        keep = min(samples.shape[2], target_latent)
        out: dict = {**latent, "samples": samples[:, :, :keep]}
        return io.NodeOutput(out)


class TrimImageBatchToAudio(io.ComfyNode):
    """Clips an IMAGE batch to ``floor(audio_duration * fps)`` frames.

    Wire between the loop's assembled IMAGE output (typically
    ``LTXVTiledVAEDecode.image``) and ``VHS_VideoCombine.images``.
    Eliminates the silence-at-end seen in saved mp4s, which arises
    because per-iter video generation uses fixed-stride math
    (``total = 245 + N * 448 px`` for canonical defaults) and can
    overshoot the audio by up to ``window_seconds - stride_seconds``
    per loop run. ``-shortest`` in ffmpeg can't truncate ``-c:v copy``
    streams reliably, so the saved container ends up the longer of
    audio/video. This node trims the image batch directly so the mp4
    matches audio length exactly.

    Empirical verification (2026-05-10, 20 random renders, 3 distinct
    audio sources): observed video length matches ``245 + N * 448``
    exactly; observed audio matches the trimmed source. Postmortem:
    ``internal/analysis/loop_audio_overshoot_analysis.md``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrimImageBatchToAudio",
            display_name="Trim Image Batch to Audio",
            category="looping/audio",
            description=(
                "Clips an IMAGE batch to floor(audio_duration * fps) frames. "
                "Place between the loop output and VHS_VideoCombine.images "
                "to prevent silence-at-end caused by fixed-stride iteration "
                "overshooting the audio length."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Assembled video frames (typically LTXVTiledVAEDecode output).",
                ),
                io.Audio.Input(
                    "audio",
                    tooltip=(
                        "Reference audio used to determine target frame count. "
                        "Wire from the same source feeding VHS_VideoCombine.audio."
                    ),
                ),
                io.Int.Input(
                    "fps",
                    default=25,
                    min=1,
                    tooltip=(
                        "Output frame rate. Wire from LTXFramePlanner.fps_int "
                        "for the canonical loop."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    "images",
                    tooltip=(
                        "Image batch trimmed to floor(audio_duration * fps) "
                        "frames. Pass-through when video is already shorter "
                        "than audio."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, audio: dict, fps: int) -> io.NodeOutput:
        audio_duration = _audio_duration(audio)
        target_frames = max(1, int(audio_duration * fps))
        keep = min(images.shape[0], target_frames)
        return io.NodeOutput(images[:keep])


class LTXHeadTrim(io.ComfyNode):
    """Drops the first N latent-frames' worth of pixel frames + the
    matching audio span. Composite IMAGE + AUDIO so they stay in lockstep.

    Use case: LTX 2.3 i2v "filler" frames. The model spends 0.5-2 s
    easing out of the init image before motion develops; this node
    discards that window after sampling so the saved mp4 starts where
    the action does. The trim is post-VAE-decode (image-level), not
    pre-decode: simpler to compose, single-node A/B, no NestedTensor
    plumbing, and the VAE work on the dropped frames is the only cost.

    Place between the decoded `IMAGE` feed and `VHS_VideoCombine`:

        LTXVTiledVAEDecode.image -> LTXHeadTrim.images -> VHS_VideoCombine.images
        Set_orig_audio          -> LTXHeadTrim.audio  -> VHS_VideoCombine.audio

    Default `trim_latent_frames=0` is a no-op pass-through; opt in via
    widget. Pixel-frame trim is `trim_latent_frames * LTX_TEMPORAL_SCALE`
    (= 8). Audio waveform trims by the same duration in seconds
    (pixel_trim / fps) so video + audio stay aligned.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTXHeadTrim",
            display_name="LTX Head Trim (Image+Audio)",
            category="looping/audio",
            description=(
                "Drop the first N latent-frames' worth of pixel frames "
                "and the matching audio span. Composite IMAGE + AUDIO. "
                "Default 0 = no-op. Use to clip i2v filler frames at "
                "clip start."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Decoded video frames (typically LTXVTiledVAEDecode output).",
                ),
                io.Audio.Input(
                    "audio",
                    tooltip="Audio waveform to trim in sync with images.",
                ),
                io.Int.Input(
                    "trim_latent_frames",
                    default=0,
                    min=0,
                    max=4096,
                    tooltip=(
                        "How many LATENT frames' worth of pixel frames "
                        "to drop from the start. 0 = no-op. 1 latent "
                        "frame = 8 pixel frames = ~0.32 s at 25 fps. "
                        "Typical i2v filler is 2-6 latent frames."
                    ),
                ),
                io.Int.Input(
                    "fps",
                    default=25,
                    min=1,
                    tooltip="Output frame rate. Used only to convert pixel-frame trim to audio-seconds trim.",
                ),
            ],
            outputs=[
                io.Image.Output("images", tooltip="Trimmed image batch."),
                io.Audio.Output("audio", tooltip="Audio with matching head span dropped."),
            ],
        )

    @classmethod
    def execute(cls, images, audio: dict, trim_latent_frames: int, fps: int) -> io.NodeOutput:
        if trim_latent_frames <= 0 or fps <= 0:
            return io.NodeOutput(images, audio)

        pixel_trim = trim_latent_frames * LTX_TEMPORAL_SCALE
        # Floor at images.shape[0] - 1 so VHS_VideoCombine never receives an
        # empty image batch (it errors). max(0, ...) handles the 0-frame edge.
        pixel_trim = min(pixel_trim, max(0, images.shape[0] - 1))
        trimmed_images = images[pixel_trim:]

        seconds_trim = pixel_trim / fps
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        sample_trim = min(int(round(seconds_trim * sample_rate)), waveform.shape[-1])
        trimmed_waveform = waveform[..., sample_trim:]
        trimmed_audio = {"waveform": trimmed_waveform, "sample_rate": sample_rate}

        return io.NodeOutput(trimmed_images, trimmed_audio)


class LatentTemporalMask(io.ComfyNode):
    """Writes a retake noise_mask to a video latent: regenerate only
    `[start_time, end_time]`, hold the rest fixed as context.

    Reversed or zero-width ranges yield an all-zero mask (no-op) rather
    than raising — safer for UI widget drift.

    `edge_taper_seconds > 0` ramps the mask 0->1 at the leading boundary
    and 1->0 at the trailing boundary using a cosine ease, so a downstream
    inpainting sampler blends the regenerated region into surrounding
    context instead of hitting a hard step. Default 0.0 preserves the
    historical hard-mask output bit-identically.

    Port of `TemporalRegionMask.apply_to` from
    `coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/retake.py`.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LatentTemporalMask",
            display_name="Latent Temporal Mask (Retake)",
            category="looping/audio",
            description=(
                "Writes a noise_mask to a video latent so only [start_time, end_time] "
                "regenerates. Rest stays fixed as context. Use for retake / section-regen."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Accumulated video latent to retake a section of."),
                io.Float.Input(
                    "start_time", default=0.0, min=0.0, max=10_000.0, step=0.01,
                    tooltip="Start of the retake window in seconds. Clamped to 0 if negative.",
                ),
                io.Float.Input(
                    "end_time", default=10.0, min=0.0, max=10_000.0, step=0.01,
                    tooltip="End of the retake window in seconds. Clamped to video duration.",
                ),
                io.Float.Input(
                    "fps", default=25.0, min=1.0, max=120.0, step=0.01,
                    tooltip="Video frame rate. LTX 2.3 canonical inference fps is 25.",
                ),
                io.Float.Input(
                    "edge_taper_seconds", default=0.0, min=0.0, max=2.0, step=0.01,
                    tooltip=(
                        "Cosine taper width in seconds at each end of the retake range. "
                        "0.0 (default) = hard mask. >0 ramps 0->1 over taper window at "
                        "start, 1->0 at end. Reduces seam artifacts at section boundaries."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(tooltip="Latent with noise_mask set: 1.0 inside [start,end], 0.0 outside; cosine ramps at boundaries when taper > 0."),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent: dict,
        start_time: float,
        end_time: float,
        fps: float,
        edge_taper_seconds: float = 0.0,
    ) -> io.NodeOutput:
        with _profile_span("LatentTemporalMask"):
            out = latent.copy()
            samples = out["samples"]
            total_frames = samples.shape[2]

            mask = torch.zeros_like(samples)
            if end_time > start_time:
                start_latent = max(0, int(start_time * fps / LTX_TEMPORAL_SCALE))
                end_latent = min(
                    total_frames,
                    int(end_time * fps / LTX_TEMPORAL_SCALE) + 1,
                )
                if end_latent > start_latent:
                    mask[:, :, start_latent:end_latent] = 1.0
                    if edge_taper_seconds > 0.0:
                        range_latents = end_latent - start_latent
                        taper_latents = max(1, int(edge_taper_seconds * fps / LTX_TEMPORAL_SCALE))
                        taper_latents = min(taper_latents, range_latents // 2)
                        ramp_up_b, ramp_down_b = _make_cosine_taper_pair(taper_latents, samples)
                        if ramp_up_b is not None:
                            mask[:, :, start_latent:start_latent + taper_latents] = ramp_up_b
                            mask[:, :, end_latent - taper_latents:end_latent] = ramp_down_b
            out["noise_mask"] = mask

        return io.NodeOutput(out)


class LatentSeamZoneMask(io.ComfyNode):
    """Writes a multi-band noise_mask centered on iteration boundaries.

    Companion to `LatentTemporalMask`: where the temporal mask retakes a
    user-specified `[start_time, end_time]` section, this one targets
    every internal iteration boundary in an assembled loop output. The
    boundaries are derived from the same integer-latent counts that
    `AudioLoopController` runs the loop with — `stride = window - overlap`,
    seams at `[stride, 2*stride, ..., (N-1)*stride]`.

    `edge_taper_seconds > 0` cosine-ramps the outer edges of each band so
    the corrective sampler blends seam-zone regenerations into frozen
    context. Default 0.0 = hard band edges.

    Use case: after a loop renders, run this node + a low-σ corrective
    sampler to refine just the seam zones if the diagnostic shows
    boundary-aligned artifacts above the noise floor.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LatentSeamZoneMask",
            display_name="Latent Seam-Zone Mask",
            category="looping/audio",
            description=(
                "Writes a multi-band noise_mask: 1.0 in bands centered on each internal "
                "iteration boundary, 0.0 elsewhere. Optional cosine taper at band edges."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Assembled loop output latent."),
                io.Int.Input(
                    "iteration_count", default=1, min=1, max=10_000,
                    tooltip="Total iterations stitched (from AudioLoopController.total_iterations).",
                ),
                io.Int.Input(
                    "window_latents", default=8, min=1, max=10_000,
                    tooltip="Latents per window (from AudioLoopController.window_latents).",
                ),
                io.Int.Input(
                    "overlap_latents", default=2, min=0, max=10_000,
                    tooltip="Overlap latents per window (from AudioLoopController.overlap_latents). Must be < window_latents.",
                ),
                io.Float.Input(
                    "seam_band_seconds", default=0.96, min=0.0, max=10.0, step=0.01,
                    tooltip="Full width of the band centered on each seam, in seconds.",
                ),
                io.Float.Input(
                    "edge_taper_seconds", default=0.0, min=0.0, max=2.0, step=0.01,
                    tooltip=(
                        "Cosine taper width at each end of each band. "
                        "0.0 (default) = hard band edges. >0 ramps 0->1 at the leading "
                        "edge of each band and 1->0 at the trailing edge."
                    ),
                ),
                io.Float.Input(
                    "fps", default=25.0, min=1.0, max=120.0, step=0.01,
                    tooltip="Video frame rate. LTX 2.3 canonical inference fps is 25.",
                ),
            ],
            outputs=[
                io.Latent.Output(tooltip="Latent with multi-band noise_mask set."),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent: dict,
        iteration_count: int,
        window_latents: int,
        overlap_latents: int,
        seam_band_seconds: float,
        fps: float,
        edge_taper_seconds: float = 0.0,
    ) -> io.NodeOutput:
        with _profile_span("LatentSeamZoneMask"):
            stride = window_latents - overlap_latents
            if stride <= 0:
                raise ValueError(
                    f"window_latents ({window_latents}) must exceed overlap_latents "
                    f"({overlap_latents}); stride={stride} would yield no valid seams."
                )

            out = latent.copy()
            samples = out["samples"]
            total_frames = samples.shape[2]
            mask = torch.zeros_like(samples)

            half_band = max(1, int(seam_band_seconds * fps / LTX_TEMPORAL_SCALE / 2))
            taper_latents = (
                max(1, int(edge_taper_seconds * fps / LTX_TEMPORAL_SCALE))
                if edge_taper_seconds > 0.0 else 0
            )
            taper_latents = min(taper_latents, half_band)
            # Loop-invariant: taper tensors depend only on taper_latents +
            # samples.shape, so build once and reuse for every seam.
            ramp_up_b, ramp_down_b = _make_cosine_taper_pair(taper_latents, samples)

            for i in range(1, iteration_count):
                seam = stride * i
                lo = max(0, seam - half_band)
                hi = min(total_frames, seam + half_band)
                if hi <= lo:
                    continue
                mask[:, :, lo:hi] = 1.0
                if ramp_up_b is not None:
                    lead_lo = seam - half_band
                    lead_hi = lead_lo + taper_latents
                    trail_hi = seam + half_band
                    trail_lo = trail_hi - taper_latents
                    # Skip ramp at edges where it doesn't fit; the hard band
                    # write above leaves those frames at 1.0, which is a
                    # reasonable default at a latent boundary (no frozen
                    # context on the other side to blend with).
                    if lead_lo >= 0 and lead_hi <= total_frames:
                        mask[:, :, lead_lo:lead_hi] = ramp_up_b
                    if trail_lo >= 0 and trail_hi <= total_frames:
                        mask[:, :, trail_lo:trail_hi] = ramp_down_b

            out["noise_mask"] = mask

        return io.NodeOutput(out)


class AudioTemporalMask(io.ComfyNode):
    """Writes a retake/extension noise_mask to an AUDIO latent: regenerate only
    `[start_time, end_time]` seconds, hold the rest fixed as context.

    Audio analog of `LatentTemporalMask`. The video node maps seconds to latent
    frames via the fixed video VAE scale (`fps / 8`); the audio VAE's latent rate
    is NOT a clean constant (mel hop_length / autoencoder downscale factor), so
    this node derives it empirically from the latent's own temporal dim and the
    known source duration — same approach as `AudioLatentSlice._infer_latent_rate`:

        rate (audio-latent frames / sec) = T / audio_duration_seconds

    Audio latents are `[B, C, T, F]` (rank 4: batch, channels, time, mel_bins),
    distinct from video `[B, C, F, H, W]` (rank 5). The mask is built as a 1-D
    temporal profile and broadcast over the remaining dims, so it is rank-agnostic
    (works whatever the audio latent's exact rank, as long as dim 2 is time).

    Primary use: AV temporal-extension probe — freeze the first N seconds of
    audio (context), regenerate the tail (`start_time=N, end_time=audio_duration`).
    Pair the video stream with `LatentTemporalMask(start_time=N, end_time=N+window,
    fps=25)` using the SAME seconds so both streams' clean prefixes align in time.

    Reversed / zero-width ranges, out-of-range starts, and non-positive durations
    yield an all-zero mask (no-op) rather than raising — safer for UI widget drift.

    `edge_taper_seconds > 0` cosine-ramps the mask 0->1 at the leading boundary and
    1->0 at the trailing boundary. Default 0.0 = hard mask.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioTemporalMask",
            display_name="Audio Temporal Mask (Retake)",
            category="looping/audio",
            description=(
                "Writes a noise_mask to an AUDIO latent so only [start_time, end_time] "
                "regenerates; the rest stays fixed as context. Audio analog of "
                "LatentTemporalMask. Use for audio retake / AV temporal-extension."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Audio latent [B, C, T, F] from LTXVAudioVAEEncode."),
                io.Float.Input(
                    "start_time", default=0.0, min=0.0, max=10_000.0, step=0.01,
                    tooltip="Start of the regenerate window in seconds. Clamped to 0 if negative.",
                ),
                io.Float.Input(
                    "end_time", default=10.0, min=0.0, max=10_000.0, step=0.01,
                    tooltip="End of the regenerate window in seconds. Clamped to audio duration.",
                ),
                io.Float.Input(
                    "audio_duration_seconds", default=10.0, min=0.0, max=10_000.0, step=0.01,
                    tooltip=(
                        "Real-world duration (seconds) of the audio encoded into THIS latent "
                        "(e.g. the TrimAudioDuration output length, or AudioDuration of the "
                        "same clip). Used to map seconds to audio-latent frames: rate = T / "
                        "duration. <= 0 yields an all-zero no-op mask."
                    ),
                ),
                io.Float.Input(
                    "edge_taper_seconds", default=0.0, min=0.0, max=2.0, step=0.01,
                    tooltip=(
                        "Cosine taper width in seconds at each end of the regenerate range. "
                        "0.0 (default) = hard mask."
                    ),
                ),
                io.Boolean.Input(
                    "invert", default=False,
                    tooltip=(
                        "False (default): [start_time, end_time] REGENERATES, the rest is kept "
                        "(prefix-seed / extension). True: [start_time, end_time] is the KEPT "
                        "SEED window and everything else regenerates — use to pick an arbitrary "
                        "slice of audio (e.g. the cleanest 2s of voice) as the clone seed."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(tooltip="Audio latent with noise_mask set: 1.0 = regenerate, 0.0 = keep (invert flips which range is which); cosine ramps at boundaries when taper > 0."),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent: dict,
        start_time: float,
        end_time: float,
        audio_duration_seconds: float,
        edge_taper_seconds: float = 0.0,
        invert: bool = False,
    ) -> io.NodeOutput:
        with _profile_span("AudioTemporalMask"):
            out = latent.copy()
            samples = out["samples"]
            total_frames = samples.shape[2]

            # 1-D temporal profile, broadcast to the latent's full shape below.
            # Built in float32 (not samples.dtype) so the cosine taper keeps strict
            # monotonicity on bf16/fp16 latents; expand() preserves the float32 mask.
            profile = torch.zeros(
                total_frames, device=samples.device, dtype=torch.float32,
            )
            window_set = False  # only flip on a VALID window; degenerate stays a no-op
            if end_time > start_time and audio_duration_seconds > 0:
                rate = total_frames / audio_duration_seconds  # audio-latent frames / sec
                start_latent = max(0, int(start_time * rate))
                end_latent = min(total_frames, int(end_time * rate) + 1)
                if end_latent > start_latent:
                    profile[start_latent:end_latent] = 1.0
                    window_set = True
                    if edge_taper_seconds > 0.0:
                        range_latents = end_latent - start_latent
                        taper_latents = max(1, int(edge_taper_seconds * rate))
                        taper_latents = min(taper_latents, range_latents // 2)
                        if taper_latents > 0:
                            ramp_up = 0.5 * (1.0 - torch.cos(
                                torch.linspace(
                                    0.0, math.pi, taper_latents + 2,
                                    device=samples.device, dtype=torch.float32,
                                )[1:-1]
                            ))
                            profile[start_latent:start_latent + taper_latents] = ramp_up
                            profile[end_latent - taper_latents:end_latent] = ramp_up.flip(0)

            # invert: [start,end] becomes the KEPT seed (0) and everything else
            # regenerates (1). Complementing the whole profile also flips the
            # tapered edges cleanly (ramp 0->1 becomes 1->0). Only flip a VALID
            # window — a degenerate window stays an all-zero no-op (a fat-fingered
            # seed range must not silently regenerate the entire track).
            if invert and window_set:
                profile = 1.0 - profile

            # Broadcast the temporal profile over all non-temporal dims (rank-agnostic).
            view_shape = [1, 1, total_frames] + [1] * (samples.ndim - 3)
            out["noise_mask"] = profile.view(*view_shape).expand_as(samples).contiguous()

        return io.NodeOutput(out)


def _keyframe_guide_placements(
    *,
    batch_size: int,
    n_latent_frames: int,
    output_fps: float,
    seconds_per_keyframe: float,
    temporal_scale: int,
) -> list[tuple[int, int]]:
    """Map a keyframe IMAGE batch to (image_index, target_frame_idx) pairs.

    Keyframe `i` targets output time `i * seconds_per_keyframe`, i.e. pixel
    frame `round(i * seconds_per_keyframe * output_fps)`. Single-frame guides
    are NOT snapped to the 8-frame latent grid (core `LTXVAddGuide.get_latent_index`
    only snaps multi-frame guides), so a keyframe lands at its EXACT pixel frame
    — `1s @ 25fps -> frame 25`, dead on. `target_frame_idx` is forced strictly
    increasing so dense keyframes never collide on the same pixel frame.

    A keyframe is dropped once its latent index — `ceil(frame_idx / temporal_scale)`,
    matching core's `(frame_idx + t - 1) // t` — reaches `n_latent_frames`
    (core then rejects it via `latent_idx + 1 <= latent_length` for a 1-frame
    guide). Dropping here avoids a wasted VAE encode of a keyframe that won't fit.

    Pure (no torch / no ComfyUI) so the placement math is unit-testable. The
    node wraps each emitted frame_idx with core `LTXVAddGuide.encode` (resize +
    VAE) -> `get_latent_index` -> `append_keyframe`.
    """
    placements: list[tuple[int, int]] = []
    prev_frame = -1
    for i in range(max(0, int(batch_size))):
        frame_idx = round(i * seconds_per_keyframe * output_fps)
        if frame_idx <= prev_frame:
            frame_idx = prev_frame + 1
        latent_idx = (frame_idx + temporal_scale - 1) // temporal_scale  # ceil; matches core
        if latent_idx >= n_latent_frames:
            break
        placements.append((i, frame_idx))
        prev_frame = frame_idx
    return placements


def _required_pixel_length(
    *,
    batch_size: int,
    output_fps: float,
    seconds_per_keyframe: float,
    tail_seconds: float,
) -> int:
    """Smallest valid PIXEL length for an `EmptyLTXVLatentVideo` that holds a
    time-spaced keyframe batch with no keyframe dropped.

    Asks `_keyframe_guide_placements` (the canonical placement seam) for the
    last keyframe's pixel frame — with an unbounded latent so nothing drops —
    which includes the strictly-increasing collision bump for dense keyframes
    (`seconds_per_keyframe * output_fps < 1`). Required length = last frame + 1
    (1-based count) + `round(tail_seconds * output_fps)` extra room, then
    snapped UP to satisfy the video VAE temporal grid `(length - 1) % 8 == 0`.

    Snap is always UP, never down: rounding down would re-introduce the very
    keyframe drop this node exists to prevent. `batch_size <= 0` is degenerate
    (no keyframes) — returns the minimal valid length 9 rather than crashing.

    Pure (no torch / no ComfyUI) so the sizing math is unit-testable.
    """
    placements = _keyframe_guide_placements(
        batch_size=batch_size,
        n_latent_frames=sys.maxsize,  # unbounded: sizing wants the no-drop last frame
        output_fps=output_fps,
        seconds_per_keyframe=seconds_per_keyframe,
        temporal_scale=LTX_TEMPORAL_SCALE,
    )
    last_target_px = placements[-1][1] if placements else 0
    raw = last_target_px + 1 + round(tail_seconds * output_fps)
    # Snap UP: smallest L >= raw with (L - 1) % LTX_TEMPORAL_SCALE == 0.
    scale = LTX_TEMPORAL_SCALE
    snapped = ((raw - 1 + scale - 1) // scale) * scale + 1
    return max(9, snapped)


def _conditioning_frame_rate(cond) -> float | None:
    """Best-effort read of the `frame_rate` LTXVConditioning stamps into the
    conditioning dict (`[[tensor, {"frame_rate": fps, ...}], ...]`).

    Returns None when absent or the structure isn't the expected shape, so
    callers can skip the consistency check without a false alarm.
    """
    try:
        return cond[0][1].get("frame_rate")
    except (IndexError, TypeError, AttributeError):
        return None


# Mean-abs-pixel-diff (0-1 image scale) below which two consecutive keyframes
# are flagged as near-duplicates. Adjacent frames of slow footage sit well
# under this; a real composition change sits well above.
_KEYFRAME_SIMILARITY_WARN_MAD = 0.01


class EvenlySpacedKeyframes(io.ComfyNode):
    """Pick `count` frames spread evenly across an IMAGE batch — auto keyframe sampling.

    Replaces hand-loading keyframe images: feed the loaded video frames (e.g.
    `VHS_LoadVideo`) and get `count` frames sampled evenly across the clip
    (`count=3` -> first/middle/last; `count=5` -> 0/25/50/75/100%). Endpoints are
    always included for `count >= 2`. Feeds the keyframe encode chain.

    KJNodes' `GetImagesFromBatchIndexed` selects by explicit indices (the manual
    override path); this computes the evenly-spaced indices from the batch length.

    `count` is clamped to `[1, batch_size]`: `count=1` -> first frame; `count >`
    frame count -> all frames.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EvenlySpacedKeyframes",
            display_name="Evenly-Spaced Keyframes (from video)",
            category="looping/keyframes",
            description=(
                "Pick N frames spread evenly across an IMAGE batch (count=3 -> "
                "first/middle/last). Auto keyframe sampling from a loaded video; feeds "
                "the keyframe encode chain. Endpoints always included for count >= 2."
            ),
            inputs=[
                io.Image.Input("images", tooltip="IMAGE batch to sample from (e.g. VHS_LoadVideo frames)."),
                io.Int.Input(
                    "count", default=3, min=1, max=10_000,
                    tooltip=(
                        "Number of evenly-spaced frames to pick. Clamped to the batch "
                        "size; count=1 = first frame. Wire from AudioLoopPlanner."
                        "total_iterations (+1) to track the song length."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The `count` selected frames, in clip order."),
                io.String.Output(
                    display_name="placement_info",
                    tooltip=(
                        "Human-readable 'selected N/M frames' summary — names count "
                        "clamping and near-identical consecutive picks (the frozen-"
                        "window footgun). Wire to a preview, or read the WARN in the "
                        "console."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, count: int) -> io.NodeOutput:
        log = logging.getLogger(__name__)
        with _profile_span("EvenlySpacedKeyframes"):
            total = int(images.shape[0])
            if total == 0:
                # Empty batch (bad video path) — don't IndexError here.
                return io.NodeOutput(images, "selected 0 frames (empty input batch)")
            n = max(1, min(int(count), total))
            info = f"selected {n}/{total} frames"
            if int(count) > total:
                info += f" (count={int(count)} clamped to the batch size)"
                log.warning(
                    "[EvenlySpacedKeyframes] count=%d clamped to %d — the batch has "
                    "only %d frames, so the picks are closer together in the source.",
                    int(count), n, total,
                )
            if n == 1:
                idx = torch.tensor([0], device=images.device)
            else:
                idx = torch.linspace(0, total - 1, n, device=images.device).round().long()
            selected = images[idx]
            if n >= 2:
                # Near-duplicate guard (advisory): compare on a spatially
                # subsampled view — full resolution buys nothing at this
                # threshold and would allocate ~the whole batch again.
                sub = selected[:, ::8, ::8, :]
                pair_mad = (sub[1:] - sub[:-1]).abs().mean(dim=(1, 2, 3))
                n_dup = int((pair_mad < _KEYFRAME_SIMILARITY_WARN_MAD).sum().item())
                if n_dup:
                    info += f"; {n_dup}/{n - 1} consecutive pairs nearly identical"
                    log.warning(
                        "[EvenlySpacedKeyframes] %d/%d consecutive keyframe pairs are "
                        "nearly identical (mean abs pixel diff < %.3f) — loop windows "
                        "anchored between them can render frozen. Use a more varied "
                        "source, fewer keyframes, or a lower END anchor strength.",
                        n_dup, n - 1, _KEYFRAME_SIMILARITY_WARN_MAD,
                    )
        return io.NodeOutput(selected, info)


class KeyframeGuidesTimeSpaced(io.ComfyNode):
    """Auto-expand a DENSE keyframe IMAGE batch into time-spaced LTX latent
    guides — one node, single pass, no loop.

    Replaces the hand-wired N-guide chain (`Index -> Get Image from Batch ->
    Math Expression -> LTXVAddGuideMulti.image_N/frame_idx_N`, repeated per
    keyframe): feed the keyframe batch (e.g. a 1fps `VHS_LoadVideo`) plus the
    target `output_fps` and `seconds_per_keyframe`, and every keyframe is
    resized, VAE-encoded, and placed as an `LTXVAddGuide` keyframe at its
    computed frame index. Output `(positive, negative, latent)` goes straight
    to the sampler; LTX fills the gaps ("sparse keyframes -> fill in the middle").

    vs KJNodes `LTXVAddGuidesFromBatch` (the existing batch-guide node):
      - KJNodes places keyframe `i` at frame `i` (CONSECUTIVE), so spacing
        requires a FULL-LENGTH batch with black gap-frames (it skips black
        images). Feed it a dense 10-frame batch and they bunch at frames 0-9.
      - This node computes the frame index from `i * seconds_per_keyframe *
        output_fps`, so you feed only the keyframes (dense) and they land at
        the right times — no black padding, no big sparse tensor in RAM.
      - Both reuse the same core guide machinery; this one is the time-spacing
        + dense-input companion. Use KJNodes' node when your batch already IS
        the full-length sparse track.

    Resolution: keyframes are resized to the latent's pixel size via core
    `LTXVAddGuide.encode` (bilinear center-crop), so they need NOT be
    pre-matched to the generation resolution.

    Video-only: `latent` must be a video latent (e.g. `EmptyLTXVLatentVideo`),
    NOT a combined AV latent — core `append_keyframe` rejects an AV latent with
    "Adding guide to a combined AV latent is not supported", which is the
    desired guard (this path leaves audio free to generate, it does not freeze
    a fed-in audio track).

    Composes core `comfy_extras.nodes_lt.LTXVAddGuide` (stable ComfyUI) rather
    than the ComfyUI-LTXVideo custom node, so it does not couple to that
    package's module layout.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KeyframeGuidesTimeSpaced",
            display_name="Keyframe Guides (Time-Spaced)",
            category="looping/keyframes",
            description=(
                "Auto-expand a dense keyframe IMAGE batch into time-spaced LTX "
                "guides in a single pass — no loop, no per-keyframe Index/Math/"
                "Multi wiring, no black-frame padding. Feed keyframes + output_fps "
                "+ seconds_per_keyframe; outputs (positive, negative, latent) ready "
                "for the sampler. Video-only latent (no frozen audio)."
            ),
            inputs=[
                io.Vae.Input("vae", tooltip="Video VAE (LTX 2.3). Resizes + encodes each keyframe to a single-frame guide latent."),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "Empty VIDEO latent to fill (e.g. EmptyLTXVLatentVideo), sized "
                        "for the clip duration. NOT a combined AV latent."
                    ),
                ),
                io.Image.Input(
                    "images",
                    tooltip=(
                        "Dense keyframe IMAGE batch in clip order (e.g. a 1fps "
                        "VHS_LoadVideo). Index 0 = first frame. Resized to the "
                        "generation resolution automatically."
                    ),
                ),
                io.Float.Input(
                    "output_fps",
                    default=25.0, min=1.0, max=120.0, step=0.1,
                    tooltip=(
                        "Frame rate of the video you're generating (matches "
                        "LTXVConditioning.frame_rate, canonical 25). Used only to convert "
                        "seconds -> frames for keyframe placement."
                    ),
                ),
                io.Float.Input(
                    "seconds_per_keyframe",
                    default=1.0, min=0.01, max=60.0, step=0.01,
                    tooltip=(
                        "Real-time gap between consecutive keyframes. 1.0 = one keyframe "
                        "per second (a 1fps source). Single-frame guides land at the exact "
                        "pixel frame (1s @ 25fps -> frame 25)."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "Guide strength per keyframe (LTXVAddGuide). 1.0 = hard anchor; "
                        "lower frees the model to interpolate more loosely."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.String.Output(
                    display_name="placement_info",
                    tooltip=(
                        "Human-readable 'placed N/M keyframes' summary (names any "
                        "dropped past the latent end). Wire to a preview, or read the "
                        "WARN in the console."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, vae, positive, negative, latent, images,
        output_fps: float, seconds_per_keyframe: float, strength: float,
    ) -> io.NodeOutput:
        import comfy_extras.nodes_lt as nodes_lt

        add_guide = nodes_lt.LTXVAddGuide
        log = logging.getLogger(__name__)

        with _profile_span("KeyframeGuidesTimeSpaced"):
            scale_factors = vae.downscale_index_formula
            temporal = int(scale_factors[0])

            # Placement is driven by output_fps; if the conditioning carries a
            # different stamped frame_rate, every keyframe lands at the wrong
            # time with no other signal — warn loudly (validation only, no
            # behavior change; output_fps stays authoritative).
            cond_fps = _conditioning_frame_rate(positive)
            if cond_fps is not None and abs(float(cond_fps) - float(output_fps)) > 0.01:
                log.warning(
                    "[KeyframeGuidesTimeSpaced] output_fps=%.3f disagrees with the "
                    "conditioning's frame_rate=%.3f — keyframes will land at the wrong "
                    "times. Set output_fps to match LTXVConditioning.frame_rate.",
                    float(output_fps), float(cond_fps),
                )

            samples = latent["samples"]
            noise_mask = nodes_lt.get_noise_mask(latent)
            # latent_length/H/W are read ONCE pre-loop; append_keyframe grows
            # samples along the frame axis but placement bounds use the original
            # length (matches core/KJNodes batch-guide convention).
            _, _, latent_length, latent_height, latent_width = samples.shape
            batch_size = int(images.shape[0])

            placements = _keyframe_guide_placements(
                batch_size=batch_size,
                n_latent_frames=latent_length,
                output_fps=output_fps,
                seconds_per_keyframe=seconds_per_keyframe,
                temporal_scale=temporal,
            )

            for image_index, target_frame in placements:
                img = images[image_index : image_index + 1]
                # core encode: resize to latent resolution + VAE -> single-frame guide
                _pixels, guide = add_guide.encode(
                    vae, latent_width, latent_height, img, scale_factors,
                )
                # Delegate frame canonicalization to core (a passthrough for the
                # single-frame guides this node produces, but inherits any future
                # core change). The placement helper already dropped keyframes
                # whose latent index falls past latent_length, so no per-iter
                # bounds recheck is needed (it would be dead for positive frames).
                frame_idx, _latent_idx = add_guide.get_latent_index(
                    positive, latent_length, guide.shape[2], target_frame, scale_factors,
                )
                positive, negative, samples, noise_mask = add_guide.append_keyframe(
                    positive, negative, frame_idx, samples, noise_mask,
                    guide, strength, scale_factors,
                )

            dropped = batch_size - len(placements)
            placement_info = f"placed {len(placements)}/{batch_size} keyframes"
            if dropped > 0:
                # NOT silent: a too-short latent drops user keyframes; say so loudly.
                placement_info += (
                    f"; {dropped} dropped past the latent end "
                    f"(holds {latent_length} latent-frames). Increase "
                    f"EmptyLTXVLatentVideo length or seconds_per_keyframe."
                )
            log.log(
                logging.WARNING if dropped else logging.INFO,
                "[KeyframeGuidesTimeSpaced] %s (output_fps=%.1f, seconds_per_keyframe=%.2f)",
                placement_info, output_fps, seconds_per_keyframe,
            )

        return io.NodeOutput(
            positive, negative, {"samples": samples, "noise_mask": noise_mask}, placement_info,
        )


class KeyframeFillLength(io.ComfyNode):
    """Compute the `EmptyLTXVLatentVideo.length` (PIXEL frames) a time-spaced
    keyframe batch needs — so the latent can never under-size and drop keyframes.

    Companion to `KeyframeGuidesTimeSpaced`: that node only WARNS (and drops the
    keyframe) once a keyframe's frame index falls past the latent end. Wire this
    node's `length` output into `EmptyLTXVLatentVideo.length` and the drop can't
    happen — the latent is always sized to hold the full keyframe batch.

    The last keyframe lands at pixel frame `round((batch_size - 1) *
    seconds_per_keyframe * output_fps)` (same placement math as
    `KeyframeGuidesTimeSpaced`); `length` = that + 1 + `round(tail_seconds *
    output_fps)`, snapped UP to the video VAE temporal grid `(length - 1) % 8
    == 0`. Snap is always UP — rounding down would re-introduce the drop.

    Feed the SAME `images` / `output_fps` / `seconds_per_keyframe` you feed
    `KeyframeGuidesTimeSpaced`. Sizing only — no VAE, no encode; pure int math.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KeyframeFillLength",
            display_name="Keyframe Fill Length",
            category="looping/keyframes",
            description=(
                "Compute the EmptyLTXVLatentVideo length (PIXEL frames) needed to "
                "hold a time-spaced keyframe batch, snapped UP to the 8n+1 grid. "
                "Wire length -> EmptyLTXVLatentVideo.length so KeyframeGuidesTimeSpaced "
                "never under-sizes and drops keyframes. Pure sizing — no VAE."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The SAME dense keyframe IMAGE batch you feed "
                        "KeyframeGuidesTimeSpaced. batch_size = images.shape[0]."
                    ),
                ),
                io.Float.Input(
                    "output_fps",
                    default=25.0, min=1.0, max=120.0, step=0.1,
                    tooltip=(
                        "Frame rate of the video you're generating (match "
                        "LTXVConditioning.frame_rate, canonical 25 — same value you give "
                        "KeyframeGuidesTimeSpaced). Converts seconds -> PIXEL frames."
                    ),
                ),
                io.Float.Input(
                    "seconds_per_keyframe",
                    default=1.0, min=0.01, max=60.0, step=0.01,
                    tooltip=(
                        "Real-time gap between consecutive keyframes (same value you give "
                        "KeyframeGuidesTimeSpaced). The last keyframe lands at PIXEL frame "
                        "round((batch_size - 1) * seconds_per_keyframe * output_fps)."
                    ),
                ),
                io.Float.Input(
                    "tail_seconds",
                    default=0.0, min=0.0, max=60.0, step=0.01,
                    tooltip=(
                        "Extra room (real seconds) after the LAST keyframe, in PIXEL "
                        "frames. Default 0.0 = no tail (the latent ends one frame past the "
                        "last keyframe, pre-snap). Raise to give the model room to settle "
                        "after the final anchor."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="length",
                    tooltip=(
                        "EmptyLTXVLatentVideo length in PIXEL frames, snapped UP to satisfy "
                        "(length - 1) % 8 == 0. Wire to EmptyLTXVLatentVideo.length."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, output_fps: float, seconds_per_keyframe: float, tail_seconds: float,
    ) -> io.NodeOutput:
        with _profile_span("KeyframeFillLength"):
            batch_size = int(images.shape[0])
            length = _required_pixel_length(
                batch_size=batch_size,
                output_fps=output_fps,
                seconds_per_keyframe=seconds_per_keyframe,
                tail_seconds=tail_seconds,
            )
        return io.NodeOutput(length)


class KeyframeLatentScheduleBatchEncode(io.ComfyNode):
    """Pre-encodes every per-iteration keyframe LATENT up front, OUTSIDE the loop.

    Pair with LatentSelectByIteration inside the loop. VAE encodes each
    UNIQUE keyframe image exactly once per generation regardless of how
    many iterations share it. Replaces the per-iteration
    `KeyframeImageSchedule + ImageBlend + VAEEncode` chain — same
    architectural shape as the conditioning-side
    `TimestampPromptScheduleBatchEncode + ConditioningSelectByIteration`
    pair shipped 2026-04-22.

    Caching: output is memoized on `(id(vae), id(images), schedule,
    stride_seconds, audio_duration, snap_boundaries)`. ComfyUI's
    framework cache invalidates this node each iteration (because
    upstream AudioLoopController re-executes), so we carry our own LRU
    — same pattern `TimestampPromptScheduleBatchEncode` and
    `CachedTextEncode` use.

    Out-of-bounds image indices (schedule references idx >= batch_size,
    or negative) are clamped at runtime to `[0, batch_size-1]`. Mirrors
    legacy `KeyframeImageSchedule` clamp; LoopConfigValidator catches
    the bug pre-render with WARN.

    No `frame_rate` parameter (unlike `TimestampPromptScheduleBatchEncode`):
    VAE encoding is frame-rate-agnostic; LATENT carries no temporal
    metadata that would need stamping. The conditioning side stamps
    `frame_rate` because LTX 2.3's CONDITIONING dict format does carry
    temporal scaling; LATENT does not.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KeyframeLatentScheduleBatchEncode",
            display_name="Keyframe Latent Schedule (Batch Encode)",
            category="looping/audio",
            description=(
                "Pre-encodes every per-iteration keyframe LATENT outside "
                "the loop. Pair with LatentSelectByIteration inside the "
                "loop. Replaces KeyframeImageSchedule + ImageBlend + "
                "per-iter VAEEncode."
            ),
            inputs=[
                io.Vae.Input("vae", tooltip="Video VAE (LTX 2.3)."),
                io.Image.Input(
                    "images",
                    tooltip=(
                        "Batch of keyframe images. Index 0 = first image. "
                        "Schedule values reference these indices."
                    ),
                ),
                io.String.Input(
                    "schedule",
                    default="0:00+: 0",
                    multiline=True,
                    tooltip=(
                        "Timestamp-to-image-index schedule (same format as "
                        "KeyframeImageSchedule).\n"
                        "  0:00-0:38: 0\n"
                        "  0:38-1:15: 1\n"
                        "  1:15+: 2\n"
                        "Values are 0-based image indices into the batch.\n"
                        "Or the single word `auto`: stride-aligned identity "
                        "mapping (window i anchors keyframe i) — no per-song "
                        "text; pair with EvenlySpacedKeyframes.count wired "
                        "from AudioLoopPlanner.total_iterations + 1."
                    ),
                ),
                io.Float.Input(
                    "stride_seconds",
                    default=17.92,
                    min=0.01,
                    step=0.01,
                    tooltip=(
                        "Audio stride per iteration. Wire from "
                        "AudioLoopController.stride_seconds."
                    ),
                ),
                io.Float.Input(
                    "audio_duration",
                    default=180.0,
                    min=0.01,
                    step=0.1,
                    tooltip=(
                        "Total audio duration. Wire from "
                        "AudioLoopController.audio_duration."
                    ),
                ),
                io.Boolean.Input(
                    "snap_boundaries",
                    default=True,
                    tooltip=(
                        "Snap schedule boundaries to the iteration grid. "
                        "Default on -- matches TimestampPromptScheduleBatchEncode."
                    ),
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    "latent_list",
                    tooltip=(
                        "List of pre-encoded LATENT, one per iteration. "
                        "Wire to LatentSelectByIteration inside the loop."
                    ),
                ),
                io.Int.Output(
                    "iteration_count",
                    tooltip=(
                        "Number of entries in latent_list. Includes +1 "
                        "headroom beyond the expected loop length so the "
                        "selector's clamp absorbs overshoot."
                    ),
                ),
            ],
        )

    @classmethod
    def IS_CHANGED(
        cls,
        vae,
        images,
        schedule: str,
        stride_seconds: float,
        audio_duration: float,
        snap_boundaries: bool = True,
    ) -> str:
        return repr(_keyframe_latent_cache_key(
            vae, images, schedule, stride_seconds,
            audio_duration, snap_boundaries,
        ))

    @classmethod
    def execute(
        cls,
        vae,
        images,
        schedule: str,
        stride_seconds: float,
        audio_duration: float,
        snap_boundaries: bool = True,
    ) -> io.NodeOutput:
        cache_key = _keyframe_latent_cache_key(
            vae, images, schedule, stride_seconds,
            audio_duration, snap_boundaries,
        )
        cached = _KEYFRAME_LATENT_CACHE.get(cache_key)
        if cached is not None:
            _KEYFRAME_LATENT_CACHE.move_to_end(cache_key)
            return io.NodeOutput(*cached)

        safe_stride = max(stride_seconds, _SAFE_STRIDE_EPSILON)
        iteration_count = max(1, math.ceil(audio_duration / safe_stride) + 1)

        batch_size = int(images.shape[0])
        if schedule.strip().lower() == "auto":
            # Stride-aligned identity mapping: iteration i covers
            # [i*stride, (i+1)*stride), so window i anchors keyframe i. No
            # schedule text to hand-author per song — pairs with the
            # planner-driven EvenlySpacedKeyframes count (= iterations + 1).
            raw_indices = list(range(iteration_count))
        else:
            entries = _parse_image_schedule(schedule)
            if snap_boundaries and entries:
                entries = _snap_schedule_to_iterations(entries, stride_seconds)
            # Schedule emits per-iteration image INDICES; clamp to batch range.
            raw_indices = [
                _match_schedule_generic(entries, i * stride_seconds, 0)
                for i in range(iteration_count)
            ]
        per_iter_indices = [max(0, min(r, batch_size - 1)) for r in raw_indices]
        oob = sorted({r for r in raw_indices if not (0 <= r < batch_size)})
        if oob:
            logging.getLogger(__name__).warning(
                "[KeyframeLatentScheduleBatchEncode] schedule references image "
                "indices %s but the batch has %d images — clamped into range. "
                "Iterations sharing a clamped index anchor to the SAME keyframe "
                "(start == end anchor -> frozen-window risk). Align the keyframe "
                "count with the schedule.",
                oob, batch_size,
            )

        # Dedup preserves insertion order. Encode each unique index once;
        # all iterations sharing that index reference the SAME LATENT
        # dict object so the selector returns identity-stable refs.
        unique_indices: list[int] = []
        seen: set[int] = set()
        for idx in per_iter_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        encoded: dict[int, dict] = {}
        for idx in unique_indices:
            single = images[idx : idx + 1]
            encoded[idx] = {"samples": vae.encode(single)}

        latent_list = [encoded[i] for i in per_iter_indices]

        _KEYFRAME_LATENT_CACHE[cache_key] = (latent_list, iteration_count)
        if len(_KEYFRAME_LATENT_CACHE) > _KEYFRAME_LATENT_CACHE_MAX:
            _KEYFRAME_LATENT_CACHE.popitem(last=False)
        return io.NodeOutput(latent_list, iteration_count)


class LatentSelectByIteration(io.ComfyNode):
    """Selects a pre-encoded LATENT by iteration index.

    Runs INSIDE the loop. No VAE dependency. Pair with
    KeyframeLatentScheduleBatchEncode outside the loop. Mirrors
    `ConditioningSelectByIteration`.

    Clamp behavior:
      - current_iteration >= len(latent_list) -> returns last entry
        (absorbs the batch encoder's +1 headroom).
      - current_iteration < 0                 -> returns first entry
        (defensive; real workflows wire current_iteration from
        TensorLoopOpen which starts at 1).
      - empty latent_list                     -> raises ValueError.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LatentSelectByIteration",
            display_name="Latent Select (by Iteration)",
            category="looping/audio",
            description=(
                "Inside-loop selector for pre-encoded keyframe LATENT. "
                "Pair with KeyframeLatentScheduleBatchEncode."
            ),
            inputs=[
                io.AnyType.Input(
                    "latent_list",
                    tooltip=(
                        "List of LATENT from KeyframeLatentScheduleBatchEncode."
                    ),
                ),
                io.Int.Input(
                    "current_iteration",
                    default=0,
                    min=0,
                    tooltip="Iteration index from TensorLoopOpen.",
                ),
            ],
            outputs=[
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(cls, latent_list, current_iteration: int) -> io.NodeOutput:
        if not latent_list:
            raise ValueError(
                "LatentSelectByIteration: latent_list is empty. "
                "Wire the output of KeyframeLatentScheduleBatchEncode."
            )
        idx = max(0, min(current_iteration, len(latent_list) - 1))
        return io.NodeOutput(latent_list[idx])


def _kf_spatial_dims(latent) -> tuple[int, int] | None:
    """(H, W) of a latent dict's samples, or None when there is no usable
    tensor shape (unit-test stand-ins, exotic latent types) — callers skip."""
    try:
        shape = latent["samples"].shape
        if len(shape) < 2:
            return None
        return int(shape[-2]), int(shape[-1])
    except (KeyError, TypeError, AttributeError, IndexError):
        return None


def _kf_validate_dims(rows, fallback_latent) -> tuple[list[str], list[str]]:
    """Cross-check keyframe latent spatial dims against the fallback's.

    The fallback is the reference: canonical workflows wire it from the
    planner-sized init-image encode, so its H/W equal the window latent's.
    A keyframe whose dims don't divide the reference fails core
    LTXVAddGuide's integer-ratio assertion MID-RENDER at the first iteration
    its target_iters fires — typically a hand-copied keyframe branch whose
    resize node lost its planner width/height wires (pasted copies drop
    incoming links). Surfacing it here turns a 5-iterations-in core
    AssertionError into an immediate error naming the slot.

    Returns (errors, warnings): errors for rows that can fire (non-empty
    targets) and would crash; warnings for integer-ratio mismatches (core
    accepts a low-res guide) and for mis-sized rows that cannot fire (empty
    targets — still a landmine for the next target re-spread).

    A mis-wired fallback (not planner-sized) defeats the check — no better
    reference is visible to this node; `ref is None` fails open.
    """
    ref = _kf_spatial_dims(fallback_latent)
    if ref is None:
        return [], []
    errors: list[str] = []
    warnings: list[str] = []
    for label, iters, lat in rows:
        dims = _kf_spatial_dims(lat)
        if dims is None or dims == ref or 0 in dims:
            continue
        desc = (
            f"keyframe_latent_{label} is {dims[1]}x{dims[0]} (latent WxH) but the "
            f"fallback/init latent is {ref[1]}x{ref[0]}"
        )
        if ref[0] % dims[0] == 0 and ref[1] % dims[1] == 0:
            warnings.append(
                desc + " — integer ratio, core accepts a low-res guide, but mixed "
                "keyframe resolutions usually mean a lost resize wire."
            )
        elif iters:
            errors.append(
                desc + f"; core LTXVAddGuide will assert mid-render at iters "
                f"{sorted(iters)}. Likely cause: the resize feeding this keyframe "
                "lost its planner width/height wires (pasted node copies drop "
                "incoming links) — rewire width + height from LTXFramePlanner."
            )
        else:
            warnings.append(
                desc + " (target_iters empty so it can never fire — fix the "
                "resize before re-spreading targets)."
            )
    return errors, warnings


def _kf_select(rows, fallback_latent, current_iteration: int):
    """Pick the keyframe latent for the current iter and describe the choice.

    `rows` is a list of `(row_label, iters_set, latent)` in selection order
    (lowest-index row wins on overlap). Returns `(chosen_latent, message,
    matched_label_or_None)`. The message is what the node reports to the console
    so a render makes plain which keyframe — or the init fallback — it used, and
    when EVERY row's target_iters is empty, that the keyframes are effectively
    disabled (the silent-no-op footgun, surfaced at runtime).
    """
    all_empty = all(not iters for _, iters, _ in rows) if rows else True
    for label, iters, lat in rows:
        if current_iteration in iters:
            return lat, (
                f"iter {current_iteration} -> keyframe {label} "
                f"(target_iters {sorted(iters)} matched)"
            ), label
    if all_empty:
        msg = (
            f"iter {current_iteration} -> init fallback (all target_iters empty "
            "-- keyframes disabled, every iter uses the init image)"
        )
    else:
        msg = f"iter {current_iteration} -> init fallback (no keyframe row targets this iter)"
    return fallback_latent, msg, None


class LTXIterKeyframeSchedule(io.ComfyNode):
    """Per-iter keyframe SELECTOR for long-form video loops.

    Picks which pre-encoded keyframe latent anchors the current
    iteration, by per-row `target_iters` lists. Runs OUTSIDE the loop
    body; its output feeds the loop's existing `guide_latent` input (in
    the canonical audio-loop subgraph: sg.input `guide_latent` →
    `LTXVAddLatentGuide`). When no row matches the current iteration, it
    passes `fallback_latent` through (typically the static init-image
    guide), so behavior is identical to the no-keyframe canonical on
    un-targeted iterations.

    Mitigates DiT drift on long renders by re-referencing a fresh image
    at chosen iterations without per-iter VAE work — keyframes are
    encoded ONCE outside the loop (VAEEncode / KeyframeLatentScheduleBatchEncode)
    and only SELECTED by index here. Anchoring strength + frame index are
    governed downstream by the existing `LTXVAddLatentGuide` (soft anchor).

    Each row: a pre-encoded `keyframe_latent` + a comma-separated
    `target_iters` string ('10, 25, 40'). Empty `target_iters` = that
    row never matches. Lowest-index matching row wins if an iteration
    appears in multiple rows (one latent feeds one guide_latent).

    Prints one line per iter to the ComfyUI console BY DEFAULT (no env flag)
    saying what it used: which keyframe row anchored the iter, or that it fell
    back to the init latent — and if EVERY row's `target_iters` is empty, that
    the keyframes are disabled. `AUDIOLOOPHELPER_KF_DEBUG=1` adds latent-shape
    detail on top. Decision logic lives in `_kf_select` (pure / unit-tested).
    """

    @classmethod
    def define_schema(cls):
        options = []
        for num in range(1, 21):
            slot_inputs: list = []
            for i in range(1, num + 1):
                slot_inputs.extend([
                    io.Latent.Input(f"keyframe_latent_{i}", tooltip=f"Pre-encoded keyframe latent {i}."),
                    io.String.Input(
                        f"target_iters_{i}",
                        default="",
                        tooltip=(
                            f"Comma-separated iterations where keyframe {i} "
                            f"anchors (e.g. '10, 25, 40'). Empty = never matches."
                        ),
                    ),
                ])
            options.append(io.DynamicCombo.Option(key=str(num), inputs=slot_inputs))

        return io.Schema(
            node_id="LTXIterKeyframeSchedule",
            display_name="LTX Iter Keyframe Schedule",
            category="looping/audio",
            description=(
                "Per-iter keyframe selector. Picks which pre-encoded keyframe "
                "latent anchors the current iteration; passes a fallback "
                "through on un-targeted iters. Feeds the loop's guide_latent."
            ),
            inputs=[
                io.Latent.Input(
                    "fallback_latent",
                    tooltip=(
                        "Latent used when no row matches current_iteration "
                        "(typically the static init-image guide latent)."
                    ),
                ),
                io.Int.Input(
                    "current_iteration",
                    default=0,
                    min=0,
                    tooltip=(
                        "Current iteration index from TensorLoopOpen / AudioLoopController. "
                        "TensorLoopOpen.current_iteration is 1-based (emits 1,2,3,…), so "
                        "target_iters lists should use 1-based indices."
                    ),
                ),
                io.DynamicCombo.Input(
                    "num_keyframes",
                    options=options,
                    display_name="Number of Keyframes",
                    tooltip="How many keyframe rows to schedule.",
                ),
            ],
            outputs=[
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(cls, fallback_latent, current_iteration: int, num_keyframes) -> io.NodeOutput:
        latent_keys = sorted(
            (k for k in num_keyframes.keys() if k.startswith("keyframe_latent_")),
            key=lambda k: int(k.rsplit("_", 1)[1]),
        )
        rows = [
            (
                lat_key.rsplit("_", 1)[1],
                _parse_iter_targets(num_keyframes.get(f"target_iters_{lat_key.rsplit('_', 1)[1]}", "")),
                num_keyframes[lat_key],
            )
            for lat_key in latent_keys
        ]
        # Fail FAST on mis-sized keyframe latents (the lost-resize-wire
        # footgun) instead of letting core assert cryptically mid-render.
        errors, dim_warnings = _kf_validate_dims(rows, fallback_latent)
        log = logging.getLogger(__name__)
        warned = _get_warned_keys()  # this node re-executes EVERY iteration;
        for w in dim_warnings:       # warn once per distinct message, not 10-30x
            if w not in warned:
                warned.add(w)
                log.warning("[LTXIterKeyframeSchedule] %s", w)
        if errors:
            raise ValueError(
                "[LTXIterKeyframeSchedule] mis-sized keyframe latent(s):\n"
                + "\n".join(errors)
            )
        chosen, msg, _matched = _kf_select(rows, fallback_latent, current_iteration)
        # Report what we actually used, by default — a render should make plain
        # which keyframe (or the init fallback) anchored each iter, and flag the
        # empty-target_iters footgun at runtime, not just at audit time.
        print(f"[AudioLoopHelper] Keyframe selector: {msg}", flush=True)
        if _KF_DEBUG:
            _kf_debug(f"Selector: {msg} guide_shape={tuple(chosen['samples'].shape)}")
        return io.NodeOutput(chosen)


class KeyframeImageSchedule(io.ComfyNode):
    """Selects a keyframe image based on the current audio position using a
    timestamp schedule, analogous to how TimestampPromptSchedule selects prompts.

    Write a schedule mapping time ranges to image indices (0-based into the
    input IMAGE batch). The node picks the right keyframe each iteration so
    different song sections can use different reference images.

    When blend_seconds > 0, outputs next_image and blend_factor for smooth
    visual transitions via ImageBlend.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KeyframeImageSchedule",
            display_name="Keyframe Image Schedule",
            category="looping/audio",
            description=(
                "Selects a keyframe image based on the current audio position. "
                "Maps timestamp ranges to image indices for per-iteration visual grounding. "
                "Supports gradual blending between keyframes at transitions."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Batch of keyframe images. Index 0 = first image in batch.",
                ),
                io.Int.Input(
                    "current_iteration",
                    default=1,
                    min=0,
                    tooltip="Current loop iteration from TensorLoopOpen (0 = initial render).",
                ),
                io.Float.Input(
                    "stride_seconds",
                    default=18.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Audio stride per iteration (same as AudioLoopController).",
                ),
                io.String.Input(
                    "schedule",
                    default="0:00+: 0",
                    multiline=True,
                    tooltip=(
                        "Timestamp-to-image-index schedule. One entry per line.\n"
                        "Formats:\n"
                        "  0:00-0:38: 0\n"
                        "  0:38-1:15: 1\n"
                        "  1:15+: 2\n"
                        "Values are 0-based image indices into the batch."
                    ),
                ),
                io.Float.Input(
                    "blend_seconds",
                    default=0.0,
                    min=0.0,
                    step=0.5,
                    tooltip=(
                        "Transition duration in seconds. 0 = hard switch (default). "
                        "Set to e.g. 5.0 to blend over ~5 seconds before each boundary. "
                        "Wire next_image and blend_factor to ImageBlend."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output("image", tooltip="Keyframe image for this iteration."),
                io.Image.Output("next_image", tooltip="Upcoming keyframe at next boundary. Same as image when not near a transition."),
                io.Float.Output("blend_factor", tooltip="0.0 = fully current image, ramps to 1.0 at the boundary. Wire to ImageBlend."),
                io.Float.Output("current_time", tooltip="Current position in seconds."),
                io.Int.Output("image_index", tooltip="Which image index was selected."),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        current_iteration: int,
        stride_seconds: float,
        schedule: str,
        blend_seconds: float,
    ) -> io.NodeOutput:
        _warn_legacy_use(
            "KeyframeImageSchedule",
            "KeyframeLatentScheduleBatchEncode + LatentSelectByIteration "
            "(VAE-encodes once outside the loop instead of per-iter)",
        )
        current_time = current_iteration * stride_seconds
        entries = _parse_image_schedule(schedule)
        batch_size = images.shape[0]

        current_idx, next_idx, blend_factor = _match_image_schedule_with_next(
            entries, current_time, blend_seconds
        )

        # Clamp indices to valid range
        current_idx = max(0, min(current_idx, batch_size - 1))
        next_idx = max(0, min(next_idx, batch_size - 1))

        image = images[current_idx : current_idx + 1]
        next_image = images[next_idx : next_idx + 1]

        return io.NodeOutput(image, next_image, blend_factor, current_time, current_idx)


class VideoFrameExtract(io.ComfyNode):
    """Extracts the frame from a reference video/image batch at the current
    iteration's timestamp. Enables video-to-video style transfer by using
    reference video frames as per-iteration guides.

    Wire the output image to the subgraph's init_image input to ground
    each iteration in the corresponding reference frame.

    Status (2026-04-25): tested via `tests/test_keyframe_nodes.py`; not
    wired in any shipped workflow. Retained as a discoverable primitive
    for future V2V workflows (e.g. a reference-video retake variant) —
    zero runtime cost when unused. Removing it would force re-writing
    if the V2V use case materializes.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VideoFrameExtract",
            display_name="Video Frame Extract",
            category="looping/audio",
            description=(
                "Extracts a frame from a reference video at the current iteration's timestamp. "
                "Enables video-to-video style transfer across full songs."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Reference video as an image batch.",
                ),
                io.Int.Input(
                    "current_iteration",
                    default=1,
                    min=0,
                    tooltip="Current loop iteration from TensorLoopOpen (0 = initial render).",
                ),
                io.Float.Input(
                    "stride_seconds",
                    default=18.88,
                    min=0.01,
                    step=0.01,
                    tooltip="Audio stride per iteration (same as AudioLoopController).",
                ),
                io.Float.Input(
                    "source_fps",
                    default=25.0,
                    min=0.01,
                    step=0.01,
                    tooltip="Frame rate of the source video batch. Override if the source isn't LTX-generated (LTX 2.3 canonical inference = 25).",
                ),
            ],
            outputs=[
                io.Image.Output("image", tooltip="Single frame at the matching timestamp."),
                io.Int.Output("frame_index", tooltip="Which frame index was extracted."),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        current_iteration: int,
        stride_seconds: float,
        source_fps: float,
    ) -> io.NodeOutput:
        current_time = current_iteration * stride_seconds
        frame_index = round(current_time * source_fps)
        batch_size = images.shape[0]

        # Clamp to valid range
        frame_index = max(0, min(frame_index, batch_size - 1))

        image = images[frame_index : frame_index + 1]
        return io.NodeOutput(image, frame_index)


class LTXSmartImageResize(io.ComfyNode):
    """Adaptive multi-stage resize for i2v init images.

    Drop-in replacement for `ImageResizeKJv2 (lanczos, single-pass)` that
    picks the number of stages based on the source/target ratio AND the
    per-stage kernel based on stage position to avoid stacking 8-bit
    quantization noise.

    Why staging matters at large reductions: the lanczos kernel (radius
    3) integrates ~6 input samples per output pixel. At reduction ratios
    above ~2x linear, the kernel sees too few samples and leaves
    residual aliasing on faces / text / fine textures. LTX 2.3's
    cross-attention reads aliasing as "high-frequency content to
    explore" and pushes the camera in the first window — manifesting as
    spurious zoom/dolly motion in i2v renders even when the prompt asks
    for static framing.

    Why per-stage kernel choice matters: ComfyUI's `comfy.utils.lanczos`
    is implemented as PIL.Image.LANCZOS — converts float32 → uint8 →
    resize → float32. One round of 8-bit quantization PER CALL. Single-
    pass is fine (matches what KJv2 always did). Naive multi-stage
    stacks rounds of quantization, accumulating banding noise that
    LTX 2.3 reads identically to aliasing — i.e. as motion cues. So
    intermediate stages stay in float32 throughout
    (`F.interpolate(bicubic, antialias=True)`); only the final stage
    uses PIL lanczos for kernel-character continuity with single-pass
    behavior. Postmortem:
    `internal/analysis/smart_resize_quantization_postmortem.md`.

    Behavior:
      - source <= target: single-pass upscale.
      - source / target <= 2x: single-pass PIL lanczos (1× quantization,
        identical character to KJv2).
      - source / target > 2x: ceil(log2(ratio)) progressive 2x-ish
        downscales — bicubic+antialias for intermediates (no PIL
        roundtrip), PIL lanczos for the final stage.

    Unlike the workflow-only "two-stage lanczos preprocess"
    (`scripts/apply_lanczos_init_preprocess.py`), this node reads
    source dims at runtime and picks stage count adaptively. One node
    handles 1024x576 init images and 4K+ init images correctly.

    Aspect handling: when `keep_proportion=True` (default), the source
    is center/top/bottom-cropped to the target aspect ratio BEFORE the
    multi-stage resize, mirroring `ImageResizeKJv2(keep_proportion="crop")`.
    Set `keep_proportion=False` to stretch (matches `crop="disabled"`).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXSmartImageResize",
            display_name="LTX Smart Image Resize",
            category="image/upscaling",
            description=(
                "Adaptive multi-stage lanczos. Picks stage count based "
                "on source/target ratio so each pass stays in lanczos's "
                "clean anti-alias range."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Source IMAGE."),
                io.Int.Input(
                    "width",
                    default=832,
                    min=8,
                    max=8192,
                    step=1,
                    tooltip="Target width in pixels.",
                ),
                io.Int.Input(
                    "height",
                    default=448,
                    min=8,
                    max=8192,
                    step=1,
                    tooltip="Target height in pixels.",
                ),
                io.Boolean.Input(
                    "keep_proportion",
                    default=True,
                    tooltip=(
                        "Crop source to target aspect ratio before resize "
                        "(matches ImageResizeKJv2 keep_proportion='crop'). "
                        "Off = stretch."
                    ),
                ),
                io.Combo.Input(
                    "crop_position",
                    options=["center", "top", "bottom", "left", "right"],
                    default="top",
                    tooltip="When keep_proportion=True, which edge to crop from.",
                ),
            ],
            outputs=[
                io.Image.Output("image", tooltip="Resized IMAGE [B, height, width, C]."),
                io.Int.Output("width", tooltip="Output width (passthrough of input)."),
                io.Int.Output("height", tooltip="Output height (passthrough of input)."),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        width: int,
        height: int,
        keep_proportion: bool = True,
        crop_position: str = "top",
    ):
        # ComfyUI IMAGE shape is [B, H, W, C].
        if keep_proportion:
            image = _crop_to_aspect(image, int(width), int(height), crop_position)
        src_h, src_w = int(image.shape[1]), int(image.shape[2])
        stages = _compute_resize_stages(src_w, src_h, int(width), int(height))
        if not stages:
            return io.NodeOutput(image, int(width), int(height))

        # Operate in BCHW for the inner loop so multi-stage paths skip
        # the per-stage HWC/contiguous round-trip. Intermediate stages
        # use float32 bicubic+antialias to avoid stacking 8-bit
        # quantization rounds; final stage uses PIL lanczos to match
        # canonical single-pass kernel character.
        bchw = image.movedim(-1, 1)
        last_idx = len(stages) - 1
        for i, (stage_w, stage_h) in enumerate(stages):
            bchw = _resize_bchw(bchw, stage_w, stage_h, final_stage=(i == last_idx))
        out = bchw.movedim(1, -1).contiguous()
        return io.NodeOutput(out, int(width), int(height))


class ImageBlend(io.ComfyNode):
    """Blends two images with a factor. Pairs with KeyframeImageSchedule
    for smooth visual transitions between keyframes.

    When blend_factor = 0.0, passes image_a through unchanged.
    When blend_factor = 1.0, passes image_b through unchanged.
    Values between lerp the pixel values.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ImageBlend_AudioLoop",
            display_name="Image Blend",
            category="looping/audio",
            description=(
                "Blends two images with a factor. Use with KeyframeImageSchedule "
                "for smooth transitions between keyframes."
            ),
            inputs=[
                io.Image.Input("image_a", tooltip="Current keyframe image."),
                io.Image.Input("image_b", tooltip="Next keyframe image."),
                io.Float.Input(
                    "blend_factor",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="0.0 = all A, 1.0 = all B. Wire from KeyframeImageSchedule.",
                ),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        blend_factor: float,
    ) -> io.NodeOutput:
        _warn_legacy_use(
            "ImageBlend (node_id=ImageBlend_AudioLoop)",
            "KeyframeLatentScheduleBatchEncode + LatentSelectByIteration "
            "(latent-space keyframe path, no per-iter VAE)",
        )
        if blend_factor <= 0.0:
            return io.NodeOutput(image_a)
        if blend_factor >= 1.0:
            return io.NodeOutput(image_b)

        blended = image_a * (1.0 - blend_factor) + image_b * blend_factor
        return io.NodeOutput(blended)


# Module-level LRU cache for CachedTextEncode.
# Persists across loop iterations (our goal) and across workflow runs.
# Keyed on (id(clip), type(clip).__name__, text). Bounded so long-running
# sessions don't grow unbounded VRAM from cached CONDITIONING tensors.
#
# Hazard: id(clip) can be recycled by CPython if the original CLIP is freed
# and a new object lands at the same memory address. In practice CLIP models
# are large (>10GB) and stay resident across iterations, so this is a latent
# risk rather than an observed bug. The type tag is cheap insurance: a
# swap from Gemma->T5 (or vice versa) can't produce a ghost hit even if the
# address is recycled. If actual hits degrade anyway, switch to weakref keying.
_COND_CACHE: OrderedDict = OrderedDict()
_COND_CACHE_MAX = 20


class CachedTextEncode(io.ComfyNode):
    """Drop-in replacement for CLIPTextEncode that caches conditioning by
    (clip, text). On cache hit, skips tokenize + encode entirely.

    Speedup is significant for LTX 2.3 Gemma 3 12B: TimestampPromptSchedule
    emits the same prompt string across multiple iterations when a schedule
    range covers more than one iteration (e.g. "0:00-0:38: ..." at stride 19s
    covers iterations 0-2). Without caching, Gemma re-encodes the identical
    text each time.

    The cache is module-level and bounded (LRU, max 20 entries). Each entry
    holds a CONDITIONING tensor on GPU; 20 entries at ~16MB is ~320MB --
    negligible next to the 22B DiT.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CachedTextEncode_AudioLoop",
            display_name="Cached Text Encode",
            category="looping/audio",
            description=(
                "CLIPTextEncode with an LRU cache keyed on (clip, text). "
                "Skips re-encoding when the same prompt is used across "
                "multiple loop iterations. Drop-in replacement for CLIPTextEncode."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP model (Gemma 3 for LTX 2.3)."),
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="Prompt text to encode. Identical text + same CLIP hits the cache.",
                ),
            ],
            outputs=[
                io.Conditioning.Output("conditioning"),
            ],
        )

    @classmethod
    def execute(cls, clip, text: str) -> io.NodeOutput:
        _warn_legacy_use(
            "CachedTextEncode (node_id=CachedTextEncode_AudioLoop)",
            "TimestampPromptScheduleBatchEncode (pre-encodes ALL schedule "
            "prompts once outside the loop; F5 invariant — keeps CLIP out "
            "of loop body)",
        )
        key = (id(clip), type(clip).__name__, text)
        cached = _COND_CACHE.get(key)
        if cached is not None:
            _COND_CACHE.move_to_end(key)
            return io.NodeOutput(cached)

        # Only the miss path hits GPU (Gemma encode) -- that's the only
        # branch worth a named span in the profile trace.
        with _profile_span("CachedTextEncode.miss"):
            tokens = clip.tokenize(text)
            cond = clip.encode_from_tokens_scheduled(tokens)
            _COND_CACHE[key] = cond
            if len(_COND_CACHE) > _COND_CACHE_MAX:
                _COND_CACHE.popitem(last=False)
        return io.NodeOutput(cond)


class IterationCleanup(io.ComfyNode):
    """LATENT passthrough that runs PyTorch allocator hygiene as a side
    effect. Place in the subgraph output path so every iteration ends with
    a clean allocator state.

    comfy-aimdo's README recommends flushing the caching allocator between
    model runs to prevent fragmentation. This node is the idiomatic way to
    do that inside a TensorLoop iteration.

    Modes:
      - always:   gc.collect() + torch.cuda.empty_cache() (default)
      - gpu_only: only torch.cuda.empty_cache() (skips Python gc pass)
      - never:    passthrough only, no side effects
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IterationCleanup",
            display_name="Iteration Cleanup",
            category="looping/audio",
            description=(
                "LATENT passthrough that flushes the PyTorch caching allocator "
                "and runs Python gc. Reduces fragmentation across loop iterations."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Latent to pass through unchanged."),
                io.Combo.Input(
                    "mode",
                    options=["always", "gpu_only", "never"],
                    default="always",
                    tooltip=(
                        "always: gc + empty_cache. "
                        "gpu_only: empty_cache only. "
                        "never: passthrough (disables the cleanup)."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(cls, latent, mode: str) -> io.NodeOutput:
        with _profile_span("IterationCleanup"):
            if mode == "always":
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif mode == "gpu_only":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return io.NodeOutput(latent)


_FREE_ALL_PINS_BYTES = 1 << 62  # byte budget large enough to release every pin


def _unload_models_for_decode() -> None:
    """Free pinned staging + unload all models + flush allocators, never
    raising into the workflow — this runs at the LAST step of a long render,
    and an error here must not kill it (same defensive shape as
    ``_purge_stale_loaded_models``).

    Call order is load-bearing: ``free_pins`` walks comfy's
    ``current_loaded_models``, which ``unload_all_models`` EMPTIES — pins
    must be released first or the call silently frees nothing.
    ``evict_active=True`` because the just-used diffusion model's staging
    pins count as active. No-op when ``comfy.model_management`` isn't
    importable (tests, headless harness).
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return  # expected under pytest/headless harness
    log = logging.getLogger(__name__)
    # Log-visible by design: three identical decode-stage kernel kills were
    # undiagnosable because nothing recorded whether the cleanup ran or
    # what it freed. Report pins freed + loaded-model delta + free-RAM delta.
    try:
        import psutil
        ram_before = psutil.virtual_memory().available
    except Exception:
        ram_before = None
    models_before = len(getattr(mm, "current_loaded_models", []) or [])
    pins_freed = 0
    try:
        pins_freed = mm.free_pins(_FREE_ALL_PINS_BYTES, evict_active=True) or 0
    except Exception as e:
        warnings.warn(f"PreDecodeCleanup: free_pins failed: {e!r}", stacklevel=2)
    try:
        mm.unload_all_models()
    except Exception as e:
        warnings.warn(f"PreDecodeCleanup: unload_all_models failed: {e!r}", stacklevel=2)
    gc.collect()
    try:
        mm.soft_empty_cache()
    except Exception as e:
        warnings.warn(f"PreDecodeCleanup: soft_empty_cache failed: {e!r}", stacklevel=2)
    models_after = len(getattr(mm, "current_loaded_models", []) or [])
    if ram_before is not None:
        try:
            import psutil
            ram_delta = (psutil.virtual_memory().available - ram_before) / (1024 ** 3)
            ram_note = f", free RAM {ram_delta:+.1f}GB"
        except Exception:
            ram_note = ""
    else:
        ram_note = ""
    log.info(
        "[PreDecodeCleanup] freed %.1fGB of pinned staging; loaded models %d -> %d%s",
        pins_freed / (1024 ** 3), models_before, models_after, ram_note,
    )


class PreDecodeCleanup(io.ComfyNode):
    """LATENT passthrough that frees pinned staging and unloads ALL models —
    wire immediately before the full-song final VAE decode. Sampling no
    longer needs the models by then; freeing them reclaims ~40-50GB of
    RAM/VRAM pressure. NOTE: this is HYGIENE, not the decode-OOM fix — the
    decode buffer-stack OOM reproduced with this node proven in-graph; the
    actual bound is a temporal-chunked decode (mechanism + allocation map:
    docs/reference/benchmarking_memory_pressure.md).

    Cost: the next prompt cold-reloads (~1 min). Memoized re-runs skip the
    side effect — fine, the decode they protect is skipped too. Sibling of
    `IterationCleanup` (in-loop allocator hygiene); this is the
    end-of-render teardown. NOT for single-pass/short-clip workflows: no
    spike to dodge, and back-to-back battery renders would pay the reload
    on every prompt.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PreDecodeCleanup",
            display_name="Pre-Decode Cleanup (unload models)",
            category="looping/audio",
            description=(
                "LATENT passthrough that unloads all models and frees pinned "
                "staging. Wire right before the full-song final VAE decode so "
                "the decode's RAM spike fits — sampling is done by then. Next "
                "prompt cold-reloads (~1 min)."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="Latent to pass through unchanged."),
                io.Combo.Input(
                    "mode",
                    options=["always", "never"],
                    default="always",
                    tooltip=(
                        "always: free pinned staging + unload all models + "
                        "empty caches. never: passthrough (disables the cleanup)."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(cls, latent, mode: str) -> io.NodeOutput:
        with _profile_span("PreDecodeCleanup"):
            if mode == "always":
                _unload_models_for_decode()
        return io.NodeOutput(latent)


class LoopIterationStamp(io.ComfyNode):
    """MODEL passthrough that stamps `transformer_options["iteration"]`.

    Wire inside the loop body between the sampler's MODEL source (top-level
    patch chain -- sage, NAG, tuner, etc.) and the sampler itself, with
    `current_iteration` taken from `TensorLoopOpen`. Downstream consumers
    that read `transformer_options["iteration"]` can then attribute work
    to a specific loop pass. Canonical consumer: the sage tracer at
    `nodes_sage.py::_iter_from_kwargs`, which groups JSONL trace rows
    by iteration for offload-asymmetry forensics.

    Additive: does not overwrite other `transformer_options` keys (the
    sage `optimized_attention_override` in particular survives).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoopIterationStamp",
            display_name="Loop Iteration Stamp",
            category="AudioLoopHelper",
            description=(
                "Stamp the current loop iteration onto the model's "
                "transformer_options so per-iteration tracers (sage, "
                "profiler) can attribute calls to a loop pass."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "current_iteration",
                    default=0, min=0,
                    tooltip="Wire from TensorLoopOpen.current_iteration.",
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, current_iteration) -> io.NodeOutput:
        (stamped,) = cls._stamp_impl(model, current_iteration=current_iteration)
        return io.NodeOutput(stamped)

    @classmethod
    def _stamp_impl(cls, model, *, current_iteration):
        """Testable seam. Returns (stamped_model,) without io.NodeOutput
        wrapping so tests can run without the v3 runtime in scope."""
        clone = model.clone()
        transformer_options = clone.model_options.setdefault("transformer_options", {})
        transformer_options["iteration"] = int(current_iteration)
        return (clone,)


# Module-level call counters keyed by label. ComfyUI's v3 _io API locks
# class attributes on the executor's clone of the ComfyNode subclass, so
# `cls._call_counter += 1` fails with AttributeError at runtime even
# though the pytest fake (_IOStub) tolerates it. Keying by label here
# also keeps two inspector instances in the same workflow from sharing
# a counter when they're given different labels.
_INSPECTOR_CALL_COUNTERS: dict[str, int] = {}


class IterPatchInspector(io.ComfyNode):
    """Diagnostic pass-through that logs model patch state per call.

    Insert in the model chain (typically inside the loop subgraph or just
    before the subgraph invoker) to confirm whether patches like NAG /
    sage / AttentionTuner / ChunkedFFN survive comfy-aimdo's dynamic VRAM
    reload between iterations. Pure pass-through: never mutates the model.

    Each call logs:
      - call counter (per-label, module-level) and user-supplied label
      - len(model.patches) and len(model.object_patches) if present
      - top-level keys of transformer_options
      - whether `optimized_attention_override` is set (sage/tuner sentinel)

    With `verbose=True`, also dumps the full keys of `model.patches`.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="IterPatchInspector",
            display_name="Iteration Patch Inspector (diagnostic)",
            category="AudioLoopHelper/debug",
            is_experimental=True,
            description=(
                "Pass-through diagnostic that logs the model's attached "
                "patch state on every execute() call. Use inside the loop "
                "body to verify whether NAG / sage / AttentionTuner / "
                "ChunkedFFN patches survive comfy-aimdo's dynamic VRAM "
                "reload between iterations."
            ),
            inputs=[
                io.Model.Input("model"),
                io.String.Input(
                    "label",
                    default="patch_inspect",
                    tooltip=(
                        "Prefix for log lines so multiple inspectors are "
                        "distinguishable in console output."
                    ),
                ),
                io.Boolean.Input(
                    "verbose",
                    default=False,
                    tooltip="Print full patch keys, not just counts.",
                ),
            ],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model, label, verbose) -> io.NodeOutput:
        (out,) = cls._inspect_impl(model, label=label, verbose=verbose)
        return io.NodeOutput(out)

    @classmethod
    def _inspect_impl(cls, model, *, label: str, verbose: bool):
        """Testable seam. Returns (model,) unchanged after emitting a
        log line summarizing patch state. No mutation, no GPU work."""
        call_n = _INSPECTOR_CALL_COUNTERS.get(label, 0) + 1
        _INSPECTOR_CALL_COUNTERS[label] = call_n

        # `getattr` with default so absent surfaces don't AttributeError on
        # minimal fakes (and tolerates future ModelPatcher API drift).
        patches = getattr(model, "patches", None)
        patches_n = len(patches) if patches is not None else 0

        object_patches = getattr(model, "object_patches", None)
        obj_n = len(object_patches) if object_patches is not None else 0

        model_options = getattr(model, "model_options", {}) or {}
        transformer_options = model_options.get("transformer_options", {}) or {}
        to_keys = sorted(transformer_options.keys())
        attn_override = "optimized_attention_override" in transformer_options

        logger = logging.getLogger(__name__)
        msg = (
            f"[{label}] call={call_n} "
            f"patches={patches_n} object_patches={obj_n} "
            f"attention_override={attn_override} "
            f"transformer_options_keys={to_keys}"
        )
        logger.info(msg)

        if verbose and patches:
            logger.info(
                f"[{label}] call={call_n} patch_keys="
                f"{sorted(patches.keys())}"
            )

        return (model,)


# --- Profiling nodes ---
#
# Three coordinated nodes capture end-to-end profile data for the audio loop:
#   ProfileBegin    -> placed before the loop, starts torch.profiler
#   ProfileIterStep -> placed inside the loop body, marks iteration boundaries
#   ProfileEnd      -> placed after the loop, finalizes and writes outputs
#
# All settings live on ProfileBegin. ProfileIterStep and ProfileEnd have zero
# widgets -- they read shared state from _PROFILER_STATE. Toggle off via the
# `enabled` widget on ProfileBegin (master switch), or via ComfyUI's native
# node bypass (mode=4 on any of the three).
# Profiler state MUST survive ComfyUI-HotReloadHack reimports of this module.
# Module-level globals here would be reset mid-workflow if any file in our
# package changes (file mtime, git pull, IDE autosave). Attaching to `torch`
# (which never hot-reloads) keeps the state reachable even after our module
# is reimported -- ProfileBegin / ProfileIterStep / ProfileEnd then coordinate
# through a single live dict instead of three stale module copies.
_STATE_ATTR = "_audioloophelper_profiler_state"
_WARNED_ATTR = "_audioloophelper_warned_keys"


def _get_profiler_state() -> dict:
    state = getattr(torch, _STATE_ATTR, None)
    if state is None:
        state = {}
        setattr(torch, _STATE_ATTR, state)
    return state


def _get_warned_keys() -> set:
    warned = getattr(torch, _WARNED_ATTR, None)
    if warned is None:
        warned = set()
        setattr(torch, _WARNED_ATTR, warned)
    return warned


# Backward-compat names for tests / imports. These point to the same live
# objects attached to torch, so `.clear()` in tests works correctly.
_PROFILER_STATE = _get_profiler_state()
_WARNED_KEYS = _get_warned_keys()


def _log_once(key: str, message: str) -> None:
    """Emit a warning message once per key per Python process."""
    warned = _get_warned_keys()
    if key in warned:
        return
    warned.add(key)
    print(f"[AudioLoopHelper] {message}")


class ProfileBegin(io.ComfyNode):
    """Start torch.profiler before the audio loop.

    Place this node between the audio/model loaders and TensorLoopOpen.
    The `trigger` input is any value you want to pass through -- it exists
    only to force this node into the execution order before the loop.

    All profile settings live on this node. ProfileIterStep and ProfileEnd
    read shared state, so you only change settings here.

    Toggle off in three ways:
      1. Set `enabled=False` (zero overhead)
      2. Right-click bypass this node (mode=4)
      3. Remove all three profile nodes from the workflow
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ProfileBegin_AudioLoop",
            display_name="Profile Begin",
            category="looping/audio/profile",
            description=(
                "Starts torch.profiler for end-to-end audio-loop profiling. "
                "Use with ProfileIterStep (inside loop) and ProfileEnd (after loop)."
            ),
            inputs=[
                io.AnyType.Input("trigger", tooltip="Any value to sequence this node before the loop (passthrough)."),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip="Master on/off. False = all three profile nodes are passthroughs with zero overhead.",
                ),
                io.String.Input(
                    "output_dir",
                    default="internal/analysis/runs/profiler",
                    tooltip=(
                        "Root dir for profile outputs. Relative paths resolve "
                        "against the ComfyUI-AudioLoopHelper plugin folder "
                        "(gitignored under internal/). Use an absolute path to "
                        "write elsewhere. A timestamped subdir is created per run. "
                        "Overridden to data/runs/${RUN_ID}/profiler when RUN_ID is "
                        "set (e.g. via start_experiment.sh)."
                    ),
                ),
                io.Int.Input(
                    "warmup_iterations",
                    default=1,
                    min=0,
                    max=10,
                    tooltip="Skip this many iterations before recording (iteration 1 has compilation noise).",
                ),
                io.Int.Input(
                    "active_iterations",
                    default=3,
                    min=1,
                    max=20,
                    tooltip="Record this many iterations after warmup. More = better variance data, larger files.",
                ),
                io.Boolean.Input(
                    "include_cpu",
                    default=True,
                    tooltip="Profile CPU activities too (Python overhead, dispatcher cost). Adds ~10% overhead.",
                ),
                io.Boolean.Input(
                    "include_memory",
                    default=True,
                    tooltip="Record VRAM allocation timeline. Adds ~3% overhead.",
                ),
                io.Boolean.Input(
                    "include_shapes",
                    default=True,
                    tooltip="Record tensor shapes per op. Helps identify which layer is slow. Adds ~5% overhead.",
                ),
                io.Boolean.Input(
                    "include_flops",
                    default=False,
                    tooltip="Count FLOPS per op. Expensive; enable only for deeper analysis.",
                ),
            ],
            outputs=[
                io.AnyType.Output("trigger", tooltip="Passthrough of input trigger."),
            ],
        )

    @classmethod
    def execute(
        cls,
        trigger,
        enabled: bool,
        output_dir: str,
        warmup_iterations: int,
        active_iterations: int,
        include_cpu: bool,
        include_memory: bool,
        include_shapes: bool,
        include_flops: bool,
    ) -> io.NodeOutput:
        state = _get_profiler_state()

        # Stop any prior profiler that was left running (user cancelled a run
        # before ProfileEnd fired, ComfyUI workflow re-queued, etc.) to avoid
        # orphaning an active torch.profiler that keeps collecting invisibly.
        prior = state.get("profiler")
        if prior is not None:
            try:
                prior.stop()
            except Exception:  # noqa: BLE001 -- torch.profiler errors are unhelpful
                pass
        state.clear()

        if not enabled:
            return io.NodeOutput(trigger)

        # Torch profiler only meaningful with CUDA; guard gracefully.
        if not torch.cuda.is_available():
            _log_once("no_cuda", "ProfileBegin: CUDA not available, profiling disabled.")
            return io.NodeOutput(trigger)

        import datetime
        import sys
        from pathlib import Path

        # When RUN_ID is set the profiler artifacts join the rest of this
        # render's telemetry under data/runs/${RUN_ID}/profiler/, sharing
        # the correlation key with exec_log + sage. Otherwise fall back to
        # the legacy `output_dir` widget value with a fresh timestamped
        # subdir per run. Single source of truth for the RUN_ID read lives
        # in scripts/workflow_utils.py::_current_run_id.
        plugin_dir = Path(__file__).resolve().parent
        scripts_dir = plugin_dir / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from workflow_utils import _current_run_id  # noqa: E402

        if _current_run_id() is not None:
            from workflow_utils import run_artifact_dir  # noqa: E402
            run_dir = run_artifact_dir("profiler")
        else:
            # Resolve relative output_dir against the plugin folder so profile
            # data lands alongside our code (and is covered by our .gitignore)
            # rather than wherever ComfyUI happened to be launched from.
            out_root = Path(output_dir)
            if not out_root.is_absolute():
                out_root = plugin_dir / out_root

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = out_root / ts
            run_dir.mkdir(parents=True, exist_ok=True)

        activities = [torch.profiler.ProfilerActivity.CUDA]
        if include_cpu:
            activities.append(torch.profiler.ProfilerActivity.CPU)

        schedule = torch.profiler.schedule(
            wait=0,
            warmup=warmup_iterations,
            active=active_iterations,
            repeat=1,
        )

        profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=include_shapes,
            profile_memory=include_memory,
            with_flops=include_flops,
            with_stack=False,
            acc_events=True,  # retain events across cycle transitions
        )
        profiler.start()

        state["profiler"] = profiler
        state["run_dir"] = run_dir
        state["settings"] = {
            "warmup_iterations": warmup_iterations,
            "active_iterations": active_iterations,
            "include_cpu": include_cpu,
            "include_memory": include_memory,
            "include_shapes": include_shapes,
            "include_flops": include_flops,
        }
        print(f"[AudioLoopHelper] ProfileBegin: recording to {run_dir}")
        return io.NodeOutput(trigger)


class ProfileIterStep(io.ComfyNode):
    """Mark an iteration boundary for torch.profiler.

    Place inside the TensorLoop body (typically after LatentOverlapTrim or
    IterationCleanup). Calls profiler.step() to advance its schedule.

    No widgets -- settings are shared from ProfileBegin. Passthrough when
    ProfileBegin isn't active or was set to disabled.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ProfileIterStep_AudioLoop",
            display_name="Profile Iter Step",
            category="looping/audio/profile",
            description="Calls torch.profiler.step() at iteration boundary. Passthrough LATENT.",
            inputs=[
                io.Latent.Input("latent", tooltip="Latent passed through unchanged."),
            ],
            outputs=[
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        profiler = _get_profiler_state().get("profiler")
        if profiler is None:
            _log_once(
                "step_uninit",
                "ProfileIterStep called without an active ProfileBegin -- passthrough. "
                "Wire a ProfileBegin node before the loop to enable profiling.",
            )
            return io.NodeOutput(latent)
        profiler.step()
        return io.NodeOutput(latent)


class ProfileEnd(io.ComfyNode):
    """Stop torch.profiler and write outputs.

    Place AFTER the TensorLoop completes. The `trigger` input exists only
    to sequence this node after the loop (pass any downstream value, e.g.,
    the TensorLoopClose output).

    Emits (in the timestamped dir from ProfileBegin):
      - trace.json        : chrome trace (open at perfetto.dev or chrome://tracing)
      - summary.txt       : top kernels by cumulative time, categorized
      - memory_timeline.html : VRAM timeline (if include_memory was True)

    Passthrough when ProfileBegin isn't active.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ProfileEnd_AudioLoop",
            display_name="Profile End",
            category="looping/audio/profile",
            description="Stops torch.profiler and writes chrome trace + summary to disk.",
            inputs=[
                io.AnyType.Input("trigger", tooltip="Any value to sequence this node after the loop (passthrough)."),
            ],
            outputs=[
                io.AnyType.Output("trigger"),
            ],
        )

    @classmethod
    def execute(cls, trigger) -> io.NodeOutput:
        state = _get_profiler_state()
        profiler = state.get("profiler")
        if profiler is None:
            _log_once(
                "end_uninit",
                "ProfileEnd called without an active ProfileBegin -- passthrough.",
            )
            return io.NodeOutput(trigger)

        run_dir = state["run_dir"]
        settings = state["settings"]

        try:
            profiler.stop()

            # Write trace atomically: .tmp then rename, so a partial write
            # on disk-full / permission error doesn't leave a corrupt file.
            trace_path = run_dir / "trace.json"
            tmp_path = run_dir / "trace.json.tmp"
            try:
                profiler.export_chrome_trace(str(tmp_path))
                tmp_path.replace(trace_path)
            except (RuntimeError, OSError, ValueError) as e:
                _log_once("trace_export", f"ProfileEnd: trace export failed: {e}")
                if tmp_path.exists():
                    tmp_path.unlink()

            try:
                summary = profiler.key_averages().table(
                    sort_by="cuda_time_total",
                    row_limit=50,
                )
            except (RuntimeError, ValueError) as e:
                summary = f"Summary generation failed: {e}"
            (run_dir / "summary.txt").write_text(str(summary))

            if settings.get("include_memory"):
                try:
                    profiler.export_memory_timeline(
                        str(run_dir / "memory_timeline.html"),
                        device="cuda:0",
                    )
                except (RuntimeError, OSError, ValueError) as e:
                    _log_once("mem_timeline", f"ProfileEnd: memory_timeline export failed: {e}")

            print(f"[AudioLoopHelper] ProfileEnd: wrote profile to {run_dir}")
        finally:
            state.clear()
        return io.NodeOutput(trigger)


class AudioLoopHelperExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        from .nodes_analysis import AudioPitchDetect
        from .nodes_audio_iclora import (
            LTXAddAudioICLoRAGuide,
            LTXAddAudioICLoRAGuideAdvanced,
            LTXAudioICLoRALoader,
            LTXAudioICLoRALoaderPerStream,
            LTXLoadComposeReferenceAudio,
            LTXAudioSetRefTokens,
        )
        from .nodes_audio_latent_slice import AudioLatentSlice
        from .nodes_easycache import LTXVideoEasyCache
        from .nodes_regional_compile import LTXVideoRegionalCompile
        from .nodes_ffn import AudioLoopHelperSageFFN
        from .nodes_sage import AudioLoopHelperSageAttention
        from .nodes_validation import LoopConfigValidator

        return [
            AudioLoopController,
            TimestampPromptSchedule,
            TimestampPromptScheduleBatchEncode,
            ConditioningSelectByIteration,
            ConditioningBlend,
            AudioLoopPlanner,
            LoopConfigValidator,
            LatentContextExtract,
            LatentOverlapTrim,
            LatentTemporalMask,
            LatentSeamZoneMask,
            AudioTemporalMask,
            EvenlySpacedKeyframes,
            KeyframeGuidesTimeSpaced,
            KeyframeFillLength,
            RunIdPrefix,
            LatentFrameCount,
            TrimVideoLatentToAudio,
            TrimImageBatchToAudio,
            LTXHeadTrim,
            PurgeVRAM,
            AudioPitchDetect,
            LTXAddAudioICLoRAGuide,
            LTXAddAudioICLoRAGuideAdvanced,
            LTXAudioICLoRALoader,
            LTXAudioICLoRALoaderPerStream,
            LTXLoadComposeReferenceAudio,
            LTXAudioSetRefTokens,
            LTXResolutionFromAspect,
            LTXFramePlanner,
            LTXVCropGuidesNoLatent,
            KeyframeImageSchedule,
            KeyframeLatentScheduleBatchEncode,
            LatentSelectByIteration,
            LTXIterKeyframeSchedule,
            VideoFrameExtract,
            ImageBlend,
            LTXSmartImageResize,
            CachedTextEncode,
            IterationCleanup,
            PreDecodeCleanup,
            LoopIterationStamp,
            IterPatchInspector,
            AudioLoopHelperSageAttention,
            AudioLoopHelperSageFFN,
            AudioLatentSlice,
            LTXVideoEasyCache,
            LTXVideoRegionalCompile,
            ProfileBegin,
            ProfileIterStep,
            ProfileEnd,
        ]


def comfy_entrypoint() -> AudioLoopHelperExtension:
    return AudioLoopHelperExtension()
