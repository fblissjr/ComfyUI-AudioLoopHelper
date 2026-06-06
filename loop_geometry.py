"""Integer-latent loop geometry — the stride/window/snap math SSoT.

Last updated: 2026-06-06

Stdlib-only on purpose: this is the shared math behind AudioLoopController /
AudioLoopPlanner / LTXFramePlanner (runtime, via ``nodes.py`` re-exports)
AND the offline tooling (``scripts/validate_workflow_decoder.py``,
``scripts/apply_ltx_decoder.py``), which must not pay ``import nodes``'s
torch cost (~3s) for ~60 lines of arithmetic. Two scripts used to carry
private copies of ``LTX_TEMPORAL_SCALE`` for exactly that reason.

Import via ``nodes`` (re-exported, back-compat) or directly from here.
"""

from __future__ import annotations

from typing import NamedTuple

LTX_TEMPORAL_SCALE = 8  # LTX 2.3 VAE temporal compression factor (pixel_frames // 8 = latent_frames)


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
